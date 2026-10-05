# NoC Ports, Port Controllers and Store-and-Forward Hubs

**Date:** 2026-10-05
**Status:** Design (Q1-Q6 decided on review, 2026-10-05)
**Related:** #286 (tile-flow debugger), `docs/plans/dram-bank-model.md` (step 3, DMA window),
`docs/plans/cycle-accurate-dma-noc.md` (the older mesh/wormhole plan this supersedes for the
checkerboard), `kpu-architecture.md` §5.2

## 1. Problem Statement

A fold-end NoC port is a resource with binary occupancy. Today nothing models it as one: on the
T4 reference run (512³ in 64×64 tiles), all 8 DMA engines attach to `t4/noc/port[2]`, and that
port carries **up to 8 transfers at once**, which is physically impossible.

### 1.1 Provenance: how the port came to be concurrent

The port was only ever specified as a **place**. A resource with occupancy needs a capacity and a
sequencing rule, and neither was ever written down. The only concurrency model in the stack is
the L-T1 executor's DMA lane pool, and every layer below it inherited that model.

| Layer | Where | What it says about a port |
|---|---|---|
| Architecture | `kpu-architecture.md:807` | "where traffic enters or leaves the ring. DMA channels connect there." Location, not capacity |
| Debugger plan | `tile-flow-debugger.md` Q7, §3.1 | same: fold ends are ports, DMAs attach |
| Layout | `array_layout.hpp:47-49, 106-108` (#313) | `NocPort`: index, loop, end, two fold hubs. No capacity, no direction, no queue |
| Spec | `deployment_spec.hpp` | no port fields at all, so there is no config to find |
| Executor (L-T1) | `tile_transaction_executor.hpp:494-498` | DMA = `dma.engines` independent lanes at `dma_bytes_per_cycle` each (T4: 8 × 64 B/cycle). No port, and no DRAM ceiling either |
| Floorplan | `floorplan.cpp:378-384` (#318) | every engine of a controller attaches round-robin to its N/S ports: "eight engines over two ports is four per port, which is what lets eight engines contend for a multi-banked DRAM at all". **The error:** engine count (outstanding DRAM requests) was read as concurrency through the port |
| Binding | `bind.py:33-36, 321` (#318) | "every hub and port on the path is busy for the transfer's whole interval". Overlaps stack; nothing is exclusive. Overload counts transfers against engines, not ports |
| Viewer | `index.html:362` (#318) | port row "capacity = engines attached". 8 concurrent transfers draw as a legal 100% |
| Checker | `tflow_check.py` TF3 | forbids two transits on one *lane*; nothing forbids two through one port |
| DRAM note | `dram-bank-model.md` §1.2, §2 | "push to L3 via the NoC", with no port arbitration |

The floorplan, binding and viewer rows were written in #286 step 4. The port was never designed;
the lane pool's concurrency was drawn on it.

### 1.2 The machine, as the architect describes it (2026-10-05)

- **Every transaction is a push, a block write.** The KPU is a push architecture.
  - A **push** is a DMA engine writing a block into an L3. This is how computation starts.
  - A **pull** is not a DMA read. It is a **BlockMover writing a block from an L3 into a DMA
    engine's buffer**, which the DMA engine writes to DRAM later.
- **Full duplex is two buses**, each with its own initiator:
  - **injection**: a DMA engine pushing a block into the machine;
  - **ejection**: a BlockMover pushing a block out of the machine into a DMA engine buffer.
- **The transaction unit** is a tile at L-T1, and a burst at L-T2 and L-CA. (The architect
  wrote "bursts in L-T1/L-CA"; this plan reads that as L-T2/L-CA, since L-T1 is defined as
  tile-granular. Confirmed on review, Q1.)
- **L-T1 tests tile sequencing.** Port occupancy is not what L-T1 is for. **L3 occupancy** is,
  because that is what tests whether a CSP program delivers its operator.
- **The port arbitrates the ring.** Blocks circulating on the torus ring pass the fold link the
  port sits on, and the port arbitrates between them and new blocks being injected.
- **Hubs store and forward.** Each NoC hub has four inputs and four outputs, and stores at
  least **four blocks**: one per input, enough for full concurrency on its links. That buffering
  cuts the livelock risk enough to make **port arbitration greedy and stateless**.
  **This may change** as we learn more about how the schedules behave and perform.
- **Port queues:**
  - **one input queue per attached DMA engine**, from which the arbiter picks the next block to
    inject;
  - **one output queue per attached DMA engine**, staging the BlockMover writes that come off the
    NoC, deep enough to hide the DMA engine's DRAM write latency, so ejection never stalls a NoC
    transfer out of the checkerboard.

## 2. Architecture

```
            memory controller (DRAM side)
     DMA e0        DMA e1   ...   DMA e7          one engine = one descriptor stream
       |  ^          |  ^           |  ^           (DRAM step 3: burst window W)
  push |  | drain    |  |           |  |
       v  |          v  |           v  |
   +---------------------------------------------------------+
   | PORT CONTROLLER  noc/port[k]                            |
   |                                                         |
   |  input queues (one per engine)    output queues (one per engine)
   |  [e0][e1]...[e7]  -- credit -->   [e0][e1]...[e7]  <-- credit --
   |        \  |  /                          ^  ^  ^          |
   |      ARBITER (greedy, stateless)        |  |  |          |
   |   ring-through first, then the oldest   |  |  |          |
   |   head across input queues              |  |  |          |
   |          |                              |  |  |          |
   |   injection bus (1 block)       ejection bus (1 block)   |
   +----------|------------------------------^-----------------+
              v                              |
   ===== fold link ===== hub_a <--------> hub_b ===== ring =====
                          |                  |
             +------------+------+   +-------+-----------+
             | HUB (L3 tile)     |   | HUB (L3 tile)     |   store-and-forward:
             | 4 in, 4 out       |   | 4 in, 4 out       |   >= 4 block buffers,
             | >= 4 block bufs   |   | >= 4 block bufs   |   credit per link
             +---------+---------+   +---------+---------+
                       |                       |
                  L3 slots (credit)       L3 slots (credit)    <- what L-T1 models
```

Credit flow is unchanged in kind:
- an engine pushes into its input queue only with a queue credit;
- a hub forwards only into a downstream buffer it holds a credit for;
- a block lands in L3 only with an L3 credit;
- a BlockMover ejects only with an output-queue credit.

There is no request/response anywhere. Every edge is push-with-credit.

## 3. Design

### 3.1 Spec

Optional fields, so an existing spec keeps its bytes:

```json
"noc": {
  "hub_buffer_blocks": 4,
  "port": {
    "input_queue_blocks": 2,
    "output_queue_blocks": 0,
    "arbitration": "ring_first_oldest"
  }
}
```

- `hub_buffer_blocks`: validated as at least the hub's input count (4).
- `input_queue_blocks`: per attached engine.
- `output_queue_blocks`: per attached engine; 0 means derived (§3.4).
- `arbitration`: the only policy for now.

### 3.2 The port is two buses, each with binary occupancy

```cpp
// WRONG: a port's capacity is the number of engines attached to it
row("port", n, label, attached.get(n) || 0, ...);            // index.html:362

// WRONG: every transfer on the path marks the port busy, and overlaps stack
for (node in path) busy[node].push(interval);                // bind.py

// CORRECT: two buses, each holding at most one block in transit
struct PortBus { std::optional<BlockInTransit> holder; };
struct PortController {
    PortBus injection, ejection;                 // full duplex = two buses
    std::vector<BlockQueue> in_q, out_q;         // one of each per attached engine
};
```

### 3.3 The arbiter is greedy and stateless

Each cycle the injection bus is free, the arbiter makes one choice:
1. A **ring block** crossing the fold link goes first. It already holds a hub buffer, and
   holding it would back the ring up.
2. Otherwise, the **oldest head** across the input queues is injected, if the downstream hub has
   a free buffer (credit).

"Stateless" means no round-robin pointer and no history; age is a property of the block. The
ejection bus has no competing traffic, so it moves the head block to its engine's output queue
when that queue has a credit.

**Known risk: ring-first can starve injection.** If a ring block is ready at every arbitration,
no queued head is ever injected. Nothing here bounds that wait: TF-HUB-2 bounds hub-buffer
residence, not input-queue wait. The plan therefore:
- **measures it:** TF-PORT-3 records the longest input-queue wait, and a test drives continuous
  ring traffic past a port;
- **leaves the fix to the architect:** an age bound that lets a queued head whose wait has
  reached a limit go before ring traffic, when the downstream hub has credit. It stays
  stateless, because age is a property of the block. It is not adopted here, because it changes
  the ring-first decision (Q2); it is the first candidate if TF-PORT-3 shows starvation on real
  schedules.

### 3.4 Output-queue sizing (ejection never stalls the NoC)

```
depth_blocks >= ceil( L_dma_write * r_eject / S ) + 1
   L_dma_write  the DMA engine's DRAM write latency for one block (from the hosted
                controller at L-CA, the calibrated table at L-T2)
   r_eject      the ejection bus rate (bytes/cycle)
   S            block bytes
```

Depth hides **latency** only. It cannot make up for **rate**: if the DMA engines cannot write
blocks to DRAM as fast as the ejection bus delivers them, any finite queue eventually fills. So
the sizing rests on a rate condition:

```
mu_dma_write >= r_eject / S        (blocks per cycle, per port, over the engines draining it)
```

`mu_dma_write` is the sustained rate at which the attached engines retire blocks to DRAM. At L-CA
it comes from the hosted controller, so the DRAM's own write bandwidth caps it. When the
condition holds, the derived depth makes ejection stall-free and TF-PORT-2 applies. When it does
not, the bottleneck is DRAM write bandwidth, not the queue. Ejection then back-pressures through
the output-queue credit, as push-with-credit requires, and the stalls are counted and attributed
to the rate, not reported as a sizing violation.

A declared depth below the derived one is accepted but reported. Every ejection stall is
counted (the TF-PORT-2 invariant, §6).

### 3.5 Liveness

Three things together bound the livelock risk:
- store-and-forward hubs with at least as many block buffers as inputs;
- ring-first arbitration at the ports;
- ejection that never refuses.

It is not a proof. Greedy injection can still fill every buffer on a ring, so the plan adds a
**liveness watchdog**: no block sits in a hub buffer longer than a bound. That is tested on
saturating-injection runs. If the watchdog ever fires, the fallback is a **bubble rule**:
inject only while the downstream hub keeps one buffer free. **Recorded as provisional; the
architect expects to revise it** as the schedules' dynamics are understood.

## 4. Implementation Steps

Each step is one PR.

1. **Stop misrepresenting the port at L-T1** (small, now).
   - The viewer's port rows show occupancy against **1 per direction**, labelled "derived path,
     not modelled at L-T1", and mark over-subscription instead of normalizing it away.
   - `bind.py` reports port over-subscription by count. It is not a violation, since L-T1 does
     not model ports.
   - The `floorplan.cpp` attachment comment drops the engine-concurrency rationale: an
     attachment is queue membership, not bandwidth.
   - L-T1 executor timing is unchanged, deliberately (§1.2).
2. **Spec** (`deployment_spec.hpp`, `deployment_json.cpp`): the `noc` section, validation, and
   an `unmodelled_fields` entry until step 4 lands.
3. **Push-only hop vocabulary.** Writeback becomes *BlockMover pushes the block to a DMA engine
   buffer* (an ejection), then *DMA engine writes DRAM*. It replaces `DmaL3ToDram` (a "DMA
   read" the machine does not have) in the L-T1 executor, the record (`Hop`), the checker and
   the viewer. Bumps `.tflow` to version 3.
4. **CSP processes** (`include/sw/kpu/timing/`): `NocHubProcess` and `NocPortProcess`.
   - **`NocHubProcess`:** store-and-forward, `hub_buffer_blocks` buffers, 4 in and 4 out, a
     credit per link.
   - **`NocPortProcess`:** per-engine input and output queues, the injection and ejection
     buses, the §3.3 arbiter.
   - **Routing:** the shortest path on the folded torus from `ArrayLayout`, deterministic on
     ties.
   - **Wiring:** DMA engines inject through their port; BlockMovers eject through it. This
     sits beside DRAM step 3 (the burst window), which supplies the engine side.
5. **Record and checker:**
   - port-bus and hub-buffer occupancy columns at L-T2/L-CA;
   - new invariants (§6) in `tflow_check.py`;
   - the liveness watchdog as an executor diagnostic.
6. **Viewer:** per-port injection and ejection rows, per-engine queue-depth rows, and hub
   buffer occupancy.

## 5. Verification

```bash
ctest --test-dir build -R 'noc|port|hub|tflow|tileflow|floorplan' --output-on-failure
python3 tools/trace/test_tflow_check.py <bundle>
```

| Test | What it must show | Mutation that must fail it |
|---|---|---|
| Binary occupancy | two engines on one port, same cycle: their injections serialize; the second waits a full block time | let the bus hold two blocks |
| Ring first | a ring block at the fold link and a queued injection, same cycle: the ring block crosses first | swap the priority |
| Oldest first, stateless | three engines with heads of ages 3, 1, 2: injection order is by age, whatever the engine index | pick the lowest index |
| No ejection stall | T4 512³ writeback at the derived output depth, rate condition met: zero NoC cycles stalled on a full output queue | depth one block below derived: stalls appear and are counted |
| Rate, not depth | DMA write rate set below `r_eject / S`: stalls appear at any depth and are attributed to rate | attribute them to depth |
| Injection wait | continuous ring traffic past a port with queued injections: TF-PORT-3 reports the wait (unbounded under ring-first, as §3.3 says) | none yet; it becomes an invariant if the age bound is adopted |
| Liveness | saturating injection from every port of the T64, every engine always ready: the watchdog never fires, and every injected block arrives | reduce hub buffers to 1: the watchdog fires (or the run is refused at validation) |
| Values never move | T4 and T64 matmul outputs bit-identical with and without the NoC model | none; this is ADR 0002 |
| L3 is still the authority | peak L3 occupancy never exceeds capacity; TF1 and TF9 unchanged | none |
| L-T1 honesty (step 1) | the T4 reference port row shows peak 8 against capacity 1, flagged | restore "capacity = engines attached" |

## 6. Key Invariants

- **TF-PORT-1:** a port's injection bus and its ejection bus each carry at most one block at any
  time.
- **TF-PORT-2:** while the §3.4 rate condition holds, a NoC transfer never waits on a full output
  queue at the derived depth. Any such wait is counted, and attributed to depth or to rate.
- **TF-PORT-3:** the longest input-queue wait is recorded per port. Ring-first has no bound on
  it (§3.3); this is a measurement, not yet an invariant.
- **TF-HUB-1:** a hub never holds more than `hub_buffer_blocks` blocks.
- **TF-HUB-2 (liveness):** no block stays in a hub buffer longer than the watchdog bound.
- **Credit:** every edge is push-with-credit. Nothing pulls.
- **Values:** the NoC model changes when, never what.

## 7. Decided on Review (2026-10-05)

All six went with the recommendation.

| # | Decision |
|---|---|
| Q1 | Tiles move through the port at L-T1; bursts at L-T2 and L-CA |
| Q2 | Ring-through traffic goes before injection |
| Q3 | Greedy injection into any free downstream buffer, with the liveness watchdog; the bubble rule is the documented fallback |
| Q4 | A port injects toward the fold hub on the shorter path to the destination, with a deterministic tie-break |
| Q5 | Output-queue depth is derived (§3.4) with an override; a too-small depth is reported and its stalls are counted |
| Q6 | The push-only hop vocabulary lands now, as step 3, bumping `.tflow` to version 3 |

## 8. Open Questions (as posed)

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Transaction unit above L-T1 | burst at L-T2 and L-CA (as read here); something else | **burst at L-T2/L-CA** |
| Q2 | Ring-through vs injection priority | ring first (§3.3); oldest across both | **ring first**: it keeps hub buffers draining, which is what the liveness argument leans on |
| Q3 | Injection rule | greedy, any free downstream buffer; bubble, keep one free | **greedy** as decided, with the watchdog, and bubble as the documented fallback |
| Q4 | Which way a port injects | toward the hub on the shorter path to the destination; fixed direction | **shorter path**, deterministic tie-break |
| Q5 | Output-queue depth | derived (§3.4) with an override; declared only | **derived with an override** |
| Q6 | Writeback hop rename (step 3) | do it now, bumping `.tflow` to v3; defer until the CSP NoC lands | **now**: the record should not name a "DMA read" the machine does not have |
