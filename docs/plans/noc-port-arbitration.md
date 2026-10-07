# NoC Ports, Port Controllers and Store-and-Forward Hubs

**Date:** 2026-10-05
**Status:** Steps 1-3 and 4a done. Q7 was decided 2026-10-06, after step 4a's liveness test
deadlocked the design as first written (§3.5). Step 4b is designed in §4.1 (Q10-Q12 decided
2026-10-07).
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
  (It did: four shared buffers deadlock a saturated T64. Hubs now hold two blocks per input,
  and blocks enter a ring under the bubble rule; §3.5, Q7.)
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
             | 2+ blocks/input   |   | 2+ blocks/input   |   credit per link
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
  "hub_buffer_blocks": 8,
  "port": {
    "input_queue_blocks": 2,
    "output_queue_blocks": 0,
    "arbitration": "ring_first_oldest"
  }
}
```

- `hub_buffer_blocks`: split evenly over the hub's 4 ring inputs; validated as a multiple of 4
  and at least 8, two per input: one for the block, one for the bubble (§3.5, Q7). Step 2
  shipped "at least 4"; step 4a raised it.
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
2. Otherwise, the **oldest head** across the input queues is injected, if the hub it lands in
   has a free slot (credit) in this port's **entry queue** (§3.5). It enters a ring from there.

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

**As first written, the design deadlocked.** It bounded the risk with three things:
store-and-forward hubs with at least as many block buffers as inputs, ring-first arbitration at
the ports, and ejection that never refuses. It added a watchdog, and the bubble rule as the
fallback if the watchdog fired.

Step 4a built that design and ran §5's liveness row. Every port of the T64 had two engines, and
every engine was always ready with a block for a pseudo-random hub. The four buffers per hub
were shared by all inputs, and routing was the shortest path over both loops.

| Hub buffers | Greedy injection | Bubble on injection and turns |
|---|---|---|
| 4 (shared) | deadlock by cycle ~1,000, every seed | deadlock by cycle ~1,100, every seed |
| 6 (shared) | deadlock by cycle 13k-62k | deadlock by cycle 6k-14k |
| 8+ (shared) | survived 100k cycles, 3 seeds | survived 100k cycles, 3 seeds |

Sixteen hubs, two whole row loops' worth, filled completely, each block waiting on the next
hub. The bubble did not help, because a shared pool spans both loops at a hub, and the bubble
argument assumes a queue per ring. Eight buffers only survived; nothing showed they could not
wedge.

**Adopted on review (Q7, 2026-10-06): dimension-ordered routing with per-input queues and the
bubble rule.** This is the standard deadlock-free scheme for a torus (bubble flow control).
1. **Dimension-ordered routing.** A block travels its row loop to a hub on the destination's
   column loop, turns, then travels that column loop. It never turns back, so no column ring
   ever waits on a row ring. Within that order it takes the shortest path: the turn hub with the
   fewer total hops (lower index on a tie), and the shorter way round each loop (forward on a
   tie). This amends Q4: "shortest" now means shortest dimension-ordered.
2. **Per-input ring queues.** Each of a hub's four ring inputs has its own queue of
   `hub_buffer_blocks / 4` blocks. Each direction of a loop is a unidirectional ring, and a
   block waits in the queue of the channel it arrived on.
3. **Entry queues, outside every ring.** A block that enters the NoC at a hub waits in an
   entry queue: one for the L3 (BlockMover pushes) and one for each port whose fold link ends
   there (injections). A column-port injection that landed in a column ring queue and then
   needed its row leg would break the dimension order.
4. **The bubble rule.** A block that enters a ring may take a slot in the next hub's queue only
   if a second slot stays free there. That covers a block out of an entry queue, and a block
   turning onto a column loop. A block continuing along its ring needs one slot. So every ring
   always keeps a free slot, and a block on it can always move. This is why a queue needs two
   slots, and why the spec minimum is 8.

Ejection and L3 delivery leave the ring into sinks that always drain: the L3 here, and DMA
engines that retire their output queues to DRAM. So no waiting cycle can close through them.

**Measured on the adopted design** (`tests/timing/test_noc_fabric.cpp`):
- The saturated T64 drains with every block delivered once, to the right hub, tag intact. So
  does a mix of injections, L3 -> L3 moves and ejections.
- **The bubble is what keeps a full ring moving.** Every hub of every row loop sends 2, 3 or 4
  hops ahead on its own loop. At one slot per input with greedy entry, the ring wedges at every
  distance. With two slots and the bubble, it is live.
- **What it costs.** At two slots per input, the bubble lowers saturation throughput by about a
  third (uniform traffic: 247k vs 361k blocks in 100k cycles; same-ring traffic: about 35%). At
  four slots per input (16 per hub) the cost is gone. The spec keeps 8 as the minimum; whether
  the T4 and T64 should declare 16 is open.
- **Entry starvation.** The bubble guarantees the rings move, not that every entering block is
  admitted. Under continuous same-ring traffic, an entering block can wait a long time. So the
  watchdog (TF-HUB-2) watches ring queues only, and the entry wait is measured (TF-HUB-3), the
  same way TF-PORT-3 measures the injection wait.
- Without the bubble, greedy entry at two slots per input did not wedge on any pattern tried.
  That is not a proof, and the bubble stays on. `NocFabric::Config::bubble = false` exists only
  for the mutation test.

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
   (Done.)
   - The hops are explicit values: 5 is `BlockMoverL3ToDmaBuffer`, the ejection; 6 is
     `DmaBufferToDram`; L3->L3 moved to 7.
   - The record gains a pooled, unmodelled `dmabuf` station.
   - Unlike step 1, **this changes L-T1 timing.** The ejection is a BlockMover push, so the
     BlockMover pool now carries three legs (L3->L2, L2->L3, L3->DMA buffer), and a writeback
     holds its L3 slot until it has been ejected. That is the push model's cost, not an
     artifact. On a device with one lane per mover, a small GEMM becomes BlockMover-bound, so
     extra compute tiles stop shortening it.
4. **CSP processes** (`include/sw/kpu/timing/`): `NocHubProcess` and `NocPortProcess`.
   Split in two on review (2026-10-06), because the wiring needs things the executor does not
   have yet.
   - **4a, the fabric on its own.** (Done.)
     - `noc_topology.hpp`: the channels and fold links from `ArrayLayout`, and
       dimension-ordered routing (§3.5).
     - `noc_hub_process.hpp`: per-input ring queues, entry queues, the bubble rule, the L3
       side, and the watchdog.
     - `noc_port_process.hpp`: per-engine input and output queues, the injection and ejection
       buses, the §3.3 arbiter, and the §3.4 depth and stall attribution.
     - `noc_fabric.hpp`: owns them all and fixes the tick order. Its entry points are the three
       pushes: inject, eject, and L3 -> L3 transfer. `Config::from` reads the spec.
     - `SpecField::Noc` stays unmodelled at every level until 4b.
   - **4b, wiring.** DMA engines inject through their port; BlockMovers eject through it.
     Designed in §4.1 (2026-10-07); five PRs, 4b.1-4b.5.
5. **Record and checker:**
   - port-bus and hub-buffer occupancy columns at L-T2/L-CA;
   - new invariants (§6) in `tflow_check.py`;
   - the liveness watchdog as an executor diagnostic.
6. **Viewer:** per-port injection and ejection rows, per-engine queue-depth rows, and hub
   buffer occupancy.

### 4.1 Step 4b design (2026-10-07; Q10-Q12 decided the same day)

**What the CSP tier has today** (surveyed at `a532f34`):
- **L3 is one pool.** `ConcurrentTimingExecutor` holds a single `l3_credits_` and `l3_tag_cam_`
  that every process shares (`concurrent_timing_executor.hpp:657-677`). `l3_tile_rows/cols` only
  name the BlockMovers. Nothing gives a block a home L3 tile.
- **Writeback is still a DMA read.** Step 3's push vocabulary landed only in the L-T1 executor.
  In `timing/`, a STORE waits on the L3 TagCAM, the MC reads, and the L3 slot frees on DMA
  completion (`dma_engine_process.hpp:438-461`, `:311-328`). BlockMoverProcess has no eject leg.
- **No spec reaches the executor.** Every `ConcurrentTimingExecutor::Config` is built by hand.
  DRAM step 2 added only the opt-in `dram` field, set only by tests. The driver does not run
  L-CA at all (`level_implemented`, `execution_level.hpp:88`).
- **The engine-to-port attachment is buried.** It sits inside `generate_floorplan`
  (`floorplan.cpp:389-435`) and is emitted only as drawing links.
  - The floorplan numbers engines per controller (`mc[m]/dma[e]`), while the executor gives
    engine `i` to controller `i % mcs`.
  - All 8 T4 engines attach to one port (`col0.N`); the T16 and T64 attach 4 per N/S port.
- **No oracle for "values never move" on a spec device.** The CSP value tests run on hand-built
  default configs.
- **Two clocks.** The fabric keeps its own `now_`. Wiring must tick it inside the executor's step
  so one cycle means one cycle.

**Steps.** Each is one PR and ends green.

1. **4b.1, spec to executor, and the oracle.** (Done.)
   - `csp_config_from(DeviceSpecification)` builds `ConcurrentTimingExecutor::Config` from the
     spec: MCs, engines, the DRAM hosting, L3 capacity and BlockMover count.
   - `dma_port_attachment(spec)` is lifted out of `generate_floorplan`, which then uses it.
     It returns, for each engine `(mc, e)`, its port. The executor adopts the floorplan's
     engine numbering.
   - A CSP matmul on T4, T16 and T64 spec configs with payloads, checked against the L0
     reference. This is the oracle every later step must keep.
   - No timing change.
   - As built:
     - `timing/csp_config_from_spec.hpp` lists in `unmapped` what the executor has no
       counterpart for, and keeps its defaults for those: the spec's rates, `macs_per_cycle`,
       and `l2.banks_per_tile`.
     - `Config::dma_engine_controller` is the explicit engine-to-controller map. Empty keeps
       round-robin.
     - `dma_port_attachment()` reads the floorplan's Attach links back, so its rule stays in
       `generate_floorplan`.
     - `program/value_tolerance.hpp` is the §7 answer 5 comparator.
     - The oracle passes the ADR bar on all three devices, and is bit-identical too: `fill()`'s
       inputs keep every product and partial sum exact, so accumulation order cannot round.
2. **4b.2, push-only writeback in the CSP tier** (step 3's vocabulary, without the NoC).
   - A BlockMover ejects a written-back tile from L3 into its DMA engine's buffer, then frees
     the L3 slot.
   - The DMA engine writes that buffer to DRAM, so a STORE no longer reads L3.
   - New events `BM_EJECT_START/COMPLETE`, and the L3->DRAM value copy moves to the ejection.
   - **This changes CSP timing**, as step 3 did at L-T1. The oracle and the regression
     baselines are re-pinned in the same PR.
3. **4b.3, L3 placement.** Every block gets a **home L3 tile** (Q10).
   - The L3 credit pool and TagCAM split per L3 tile (`capacity_tiles / l3.tiles` each). A load
     needs its home tile's credit.
   - A BlockMover serves only its own L3 tile's blocks: movers are the layout's sites, one per
     abutting edge, replacing the hash pool.
   - Still no NoC: a load lands in its home tile instantly, as today.
4. **4b.4, wire the fabric.** Opt-in, like `dram`: `Config::noc`.
   - **Load:** MC completion is no longer arrival. The DMA injects the block at its port, bound
     for the home hub. The hub's L3 delivery reserves nothing new, since the home credit was
     taken at load issue (§2). `TILE_ARRIVED_L3` and the DRAM->L3 copy fire on delivery.
   - **Writeback:** the 4b.2 ejection goes over the NoC, from the home hub to the engine's port
     and output queue. The DMA's DRAM write starts when the block reaches the output queue.
     That replaces the fabric's own write model (`dma_write_latency`), so the MC is the only
     DRAM writer.
   - The fabric ticks in `step()`: after the DMA engines, before the BlockMovers.
     `is_complete()` and `reset()` include it.
   - `level_models(L-CA, SpecField::Noc)` becomes true when `Config::noc` is set.
   - Tests:
     - the oracle stays bit-identical on T4, T16 and T64 with the NoC on;
     - TF-PORT-1/2/3 and TF-HUB-1/2 hold on a real matmul;
     - the T4 port row shows 8 engines through one bus, now serialized.
5. **4b.5, burst granularity** (after DRAM step 3). The NoC block becomes the burst (Q11), and
   the DMA window supplies the injection stream.

**Decided on review (2026-10-07): all three as recommended.**

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q10 | How a block gets its home L3 tile | (a) the schedule op carries it (`l3_tile`), with a deterministic default when absent; (b) a hash of the tile id, like today's engine selection; (c) derived from the consumer: the L3 tile beside the compute tile that uses it | **(a) with (b) as the default.** Placement is the compiler's decision, so the schedule should be able to state it. A hash default keeps every existing schedule runnable. (c) needs compute placement, which does not exist yet |
| Q11 | The NoC block at the CSP tier before DRAM step 3 | (a) a whole tile, until 4b.5; (b) wait for DRAM step 3 and wire bursts directly | **(a).** Q1 says bursts at L-CA, but tile-granular wiring exercises every credit edge now, and 4b.5 only changes the block size |
| Q12 | Order of 4b.2 (push writeback) and 4b.3 (placement) | as listed; or placement first | **as listed.** 4b.2 is the vocabulary change that re-pins baselines; doing it before placement keeps the two timing changes separately attributable |

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
| Injection wait | continuous ring traffic past a port with queued injections: TF-PORT-3 reports the wait (unbounded under ring-first, as §3.3 says). Step 4a drives it with an ejection stream across the fold channel: the bubble limits a single L3 -> L3 source to one block every two block times, which leaves gaps | none yet; it becomes an invariant if the age bound is adopted |
| Liveness | saturating injection from every port of the T64, every engine always ready: the watchdog never fires, and every injected block arrives (also with ejection and L3 -> L3 traffic mixed in) | one slot per input without the bubble, under same-ring traffic: the watchdog fires; one slot with the bubble admits nothing, and the spec refuses it |
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
- **TF-HUB-1:** a ring queue never holds more than `hub_buffer_blocks / 4` blocks, so a hub's
  ring queues never hold more than `hub_buffer_blocks` together.
- **TF-HUB-2 (liveness):** no block stays in a **ring** queue longer than the watchdog bound.
- **TF-HUB-3:** the longest entry-queue wait is recorded per hub. The bubble rule does not
  bound it (§3.5); this is a measurement, like TF-PORT-3.
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

### Decided on review (2026-10-06), after step 4a

| # | Decision |
|---|---|
| Q7 | Liveness: dimension-ordered routing (row loop, then column loop), per-input ring queues of `hub_buffer_blocks / 4`, entry queues outside the rings, and the bubble rule on ring entry and on the turn (§3.5). Amends Q3, because the greedy shared-buffer design deadlocked and the bubble fallback did not fix it. Amends Q4: "shortest" is now within dimension order. Raises the spec minimum to 8 |

### Decided on review (2026-10-06), after Q7

| # | Decision |
|---|---|
| Q8 | The T4 and T64 declare `hub_buffer_blocks: 16` (four per input). The spec minimum stays 8 |
| Q9 | Entry-queue depth becomes a spec field, `noc.entry_queue_blocks`, per entry source. Today it is fixed at 2 in `NocFabric::Config` |

Not done yet; both land with the next NoC PR.

**To investigate: why the bubble at 8 costs a third of saturation throughput.** At two slots per
input, the bubble lowers saturated throughput by about a third (uniform: 247k vs 361k blocks in
100k cycles; same-ring: about 35%). At four slots per input the cost is gone (§3.5). Q8 sidesteps
this by moving the fixtures to 16. It does not explain it, and the explanation decides whether
8 is a usable minimum or only a legal one. Candidates to check:
- **Admission.** An entering block needs two free slots in a two-slot queue, which means an
  empty one. Under load, a queue may rarely be empty, so entries wait while links idle.
- **Store-and-forward slot hold.** A slot is held from the start of the incoming transfer until
  the outgoing one completes, about two block times per hop. At two slots, that alone may leave
  the queue rarely empty.
- **Turns.** A turn pays the bubble too, so column rings may be starved of turning traffic.

Measure link utilization and entry-queue wait per hub at 8, 12 and 16, with the bubble on and
off, to separate admission loss from hold time.

## 8. Open Questions (as posed)

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Transaction unit above L-T1 | burst at L-T2 and L-CA (as read here); something else | **burst at L-T2/L-CA** |
| Q2 | Ring-through vs injection priority | ring first (§3.3); oldest across both | **ring first**: it keeps hub buffers draining, which is what the liveness argument leans on |
| Q3 | Injection rule | greedy, any free downstream buffer; bubble, keep one free | **greedy** as decided, with the watchdog, and bubble as the documented fallback |
| Q4 | Which way a port injects | toward the hub on the shorter path to the destination; fixed direction | **shorter path**, deterministic tie-break |
| Q5 | Output-queue depth | derived (§3.4) with an override; declared only | **derived with an override** |
| Q6 | Writeback hop rename (step 3) | do it now, bumping `.tflow` to v3; defer until the CSP NoC lands | **now**: the record should not name a "DMA read" the machine does not have |
