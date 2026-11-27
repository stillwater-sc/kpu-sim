# A bottom-up, cycle-based KPU prototype: the cycle core, the memory subsystem, the compute subsystem

**Status:** planned 2026-11-24. The direction (§2) and the decisions (§8) are the user's. Phase 0 is next.
**Replaces, as the route to a working prototype:** #401's slices 3 and later (`docs/plans/systolic-tile-at-lca.md`).
That plan's RTL analysis (§1, §3.2) stays the reference for what the compute subsystem must do.
**Parked:** branch `feat/issue-458-systolic-array-lca` (an OS gemm timed by the RTL's formulas inside the
transaction executor). It is not merged.

## 1. Why

L-CA today (`ConcurrentTimingExecutor`) steps a clock, but most of its processes are **transaction
schedulers**. At a transfer's start, a process computes its completion cycle from a rate:
- a Feed takes `startup + bytes / rate` (`streamer_process.hpp`);
- a compute takes `fill + MACs / rate`;
- a BlockMove takes its bytes at the mover's rate.

Nothing below the transfer exists: no PE array, no L1, no Streamer datapath, no Combiner datapath.
Cycle counts are inputs, not results. So L-CA can't be the reference for the other levels, and every
step that changed its numbers forced re-baselining across L-T1, the records and the viewers. That is
where the time went.

**What is already cycle-level and is kept** (ported onto the core, §3.4, not rewritten):

| Piece | Where | State |
|---|---|---|
| LPDDR5 controller: banks, bank groups, tRRD/tCCD/tFAW, refresh, FR-FCFS, per-channel data bus | `models/temporal/memory/controllers/lpddr5_controller.hpp`, hosted by `timing/dram_bridge.hpp` | cycle-accurate, invariant-checked by `patterns/memory/lpddr5` |
| NoC hubs and ports: per-input queues, the bubble rule, credit-reserved slots, a stated tick order | `timing/noc_hub_process.hpp`, `noc_port_process.hpp`, `noc_fabric.hpp` | cycle-stepped; not wired into the executor (its step 4b) |
| DMA engine burst window and store buffer with tickets | `timing/dma_engine_process.hpp` | burst-level against the hosted controller |
| The memory side on its own, NoC ports stubbed | `timing/memory_side_harness.hpp` (`docs/plans/memory-side-debugger.md`) | a harness, with records and a viewer |
| Event counters and operational analysis (X, U, E; A, C, B) | `include/sw/xue/` (`docs/xue-observation-architecture.md`) | exists; used by the older `models/` memory controller and compute fabric, not by any process in `timing/` |

## 2. Direction (the user's, 2026-11-24)

1. **Freeze L-T1 and the comparisons between fidelities.** This covers L-T1 features, the L-T1/L-CA
   tolerance and calibration, and `.tflow`/`.dflow` work for L-T1. The existing L-CA executor is frozen
   too: no new features, so nothing re-baselines while the new core is built.
2. **Build the cycle-based simulator core first, and test it on a test structure** (§3).
3. **Model two subsystems bottom-up at L-CA, each verified on its own:**
   - **memory (§4):** DRAM + MC + DMA + NoC + L3. The DMAs run CSP programs that fill the L3s,
     and a hypothesized result C goes from the L3s back to DRAM. This tests the credit flows and
     that data lands in the right L3 slots.
   - **compute (§5):** L3 → BM → L2 → STR → L1 → array → L1 → CMB → L2 → BM → L3, with the CSP
     programs that orchestrate it.
4. **DRAM is included from the start.**
5. **Each subsystem gets visualization, debug and performance measurement** (operational-analysis
   event counters) as part of its definition of done, not afterwards.

## 3. Phase 0: the cycle core

A new layer, `include/sw/kpu/cycle/` (namespace `sw::kpu::cycle`), alongside `timing/`. Nothing existing
changes, so no test re-baselines.

### 3.1 Rules every component follows

- **A component is a clocked state machine.** On each edge of its clock it reads its inputs as
  registered in the previous cycle, computes its next state, and commits. **No component computes a
  future completion time.** Latency and throughput emerge from state transitions cycle by cycle.
- **Two phases, evaluate then commit.** Every component in a domain evaluates, then all commit. The
  result doesn't depend on the order components are listed, and two runs are identical.
- **Components meet only through links.** There is no shared state and no reaching into another
  component.
- **Credits up, data down.** A sender pushes only with a credit, and the receiver returns the credit
  when it frees the slot. Credits travel on the link's return path with their own latency.
- **Payloads are real.** A beat carries its address and its bytes, so values are produced by the
  datapath and are checked, never assumed.

### 3.2 The pieces

| Piece | What it is |
|---|---|
| `Clock`, `Domain` | Global time in picoseconds. Each domain has a period: for example, the array at 1–2 GHz, the L2 banks at 250 MHz, the NoC, and the DRAM controller clock. The scheduler ticks each domain at its edges, in time order, with a fixed order between domains at the same instant. |
| `Component` | `evaluate()`, `commit()`, a name, a domain, its counters, and its invariants. |
| `Link<Beat>` | A point-to-point channel: latency, capacity (the credits), and a credit-return latency. `can_push()`, `push()`, `can_pop()`, `pop()` and `return_credit()`, each legal only in `evaluate()`. A link between domains is an async FIFO with a synchronizer delay. |
| `Beat` | An address, a byte count, the bytes, a tag (tile, slot, command), and first/last markers. |
| `Simulator` | Owns the domains, components and links. `run_until(t | predicate)`, `step()`, and determinism checks. |

### 3.3 Observability, part of the core

- **Counters (XUE).** Every link and component registers operational-analysis counters:
  - arrivals A and completions C;
  - busy cycles B, plus stall cycles by cause (no credit, no data, arbitration lost, a bank busy);
  - the occupancy integral, for Little's law.

  From these come X = C/T, U = B/T, S = B/C and N = X·R. The counters are exact, not sampled.
- **Trace.** Per-cycle transaction events (push, pop, credit, state change) go to a Perfetto/Chrome
  trace (one track per component and per link) and a counters file. That gives a viewer from day one
  without writing one (Q2).
- **Debug.**
  - `run_until` a cycle or an event predicate.
  - A state dump of every component and link, and per-cycle invariant hooks that fail loudly.
  - A **deadlock detector**: no progress for N cycles. It reports every blocked component, what it
    waits on, and each link's credits and occupancy.

### 3.4 Porting what exists

The LPDDR5 controller gets a `Component` adapter in the DRAM domain, as `DramBridge` does for the
executor. The NoC hub and port logic is re-expressed on `Link`s. Both keep their own invariant checks
(the LPDDR5 trace validator, and the bubble rule).

### 3.5 The test structure and Phase 0's acceptance

The test structure is a synthetic pipeline: a producer, N stages, and a consumer. Each stage's latency,
service rate, buffer depth and clock domain are configurable. Phase 0 is done when tests show:

- **Throughput.** It equals the slowest stage's rate. Latency equals the sum of the stage latencies
  plus the queueing, exactly as computed by hand.
- **The operational laws hold exactly on the counters.** For every component, U = X·S. Little's law
  holds on every link: mean occupancy = X × mean residence.
- **Backpressure.** A stalled consumer stops the producer with no beat lost or duplicated. Credits are
  conserved every cycle, and the stalled cycles are counted under the right cause.
- **Clock domains.** A crossing in each direction delivers every beat in order. With a 4:1 clock
  ratio (array : L2 banks), the crossing's throughput is the slower side's.
- **Determinism.** Two runs produce identical traces, and reordering the component list changes
  nothing.
- **Deadlock.** The detector fires on a deliberately built credit cycle and names it.

## 4. Phase 1: the memory subsystem (DRAM + MC + DMA + NoC + L3)

### 4.1 Components

| Component | Role |
|---|---|
| **DRAM + controller** | The LPDDR5 controller on the core (§3.4), in its own domain, one per channel as the deployment declares. |
| **DMA engine** | Executes its **CSP program**: a queue of Load / Store commands. It splits each tile into bursts, keeps a burst window in flight, and pushes beats toward an L3 only with that L3 slot's credit. A Store reads an L3 slot and writes DRAM through its store buffer. |
| **NoC** | Hubs and ports on links (§3.4), from the deployment's layout. A deployment with one L3 tile gets a direct link instead. |
| **L3 tile** | Addressed SRAM: slots × bytes, banks, and read/write ports. Each slot carries a credit and a tile-ready token. It holds the bytes it was written. |

### 4.2 Where the programs come from

From a CSP program's own L3 actions, as the walker emits them: each Load into an L3 slot, each
Release, each Store. For the hypothesized C, the L3 slots that the program's Stores read are
**preloaded with the C tiles L-B computes**. The Stores then carry C to DRAM as if the array had
produced it. L-B is used here only as the **data oracle** ("these bytes belong in this slot"). It is
not a timing comparison.

### 4.3 Ladder (the debug-ladder rule: the simplest case first)

1. One tile, one DMA, one L3, no NoC. Load, then Store it back.
2. Several tiles into one L3: slot credits, the ring of slots, a Release freeing a slot for the next
   Load.
3. Two DMAs and two L3 tiles over the NoC.
4. The full L3 traffic of a real program, with C preloaded and stored: the RTL's 8×8×16 int8 matmul
   (`kpu_soc_2x2`), then an int8 64³ (`kpu_s1`).

### 4.4 Done when

- **Data.** After each Load, every L3 slot holds exactly the bytes the program places there. After the
  run, DRAM holds C bit for bit.
- **Invariants, every cycle.**
  - Credit conservation on every link and every L3 slot.
  - No Load into a slot holding a live tile, and no Store from a slot whose tile hasn't arrived.
  - The LPDDR5 trace validator passes.
  - The NoC's bubble rule holds.
- **Counters.**
  - DRAM: data-bus utilization, page hit/empty/conflict, refresh.
  - DMA: busy cycles, stalls by cause (no L3 credit, controller queue full, NoC refused), and bursts
    in flight.
  - NoC: per-link utilization and queue occupancy.
  - L3: slot occupancy over time.
  - The operational laws are checked on each.
- **Viewer and debug.** A Perfetto trace with one track per DMA, channel, bank group, hub, port and L3
  slot, and a counters summary that `kpu-run` prints. The deadlock detector is wired in.

## 5. Phase 2: the compute subsystem (L3 → … → array → … → L3)

### 5.1 Components

The behaviour of each comes from `docs/plans/systolic-tile-at-lca.md` §1 and §3, the RTL's design at
embodied-ai-soc-generator `b57c881`. Here every rule is **built as state**, not computed by formula.

| Component | Role |
|---|---|
| **BlockMover** | L3 → L2 by rows, at its bus width per beat, strided, under the L2 slot's credit. Writeback runs L2 → L3. |
| **L2** | Banks × rows, with read and write ports per bank, a read latency, and its own clock (250 MHz). Slots are in the FREE, OWNED, READY and READING states (#387's `SlotPool`, reused). |
| **Streamer** | A command queue. An OS COMPUTE issues one A row and one B row per clock to L2, and loads the returning rows into the L1 lane buffers. It signals `completed` when its last read lands. The rows return through a read pipeline D clocks deep, the RTL's fixed read delay. The L2's latency is not fixed, so a row back early waits, and a row not back by its load is an `L1Fault`. The least D a configuration keeps is measured (step 2b: 26 clocks from 1 GHz to the 250 MHz L2). |
| **L1 lane buffers** | One per array lane, holding one block, with one credit (`left ≤ 1`). They stream one element per clock with the `first` and `last` markers. A load into a buffer without its credit is a fault. |
| **PE array** | R × C PEs, each a registered MAC. A moves east, B moves south, and on `last` the result launches north. A north collision is a fault. |
| **North result FIFOs** | One per column. A row leaves when every column has its element. |
| **Combiner** | Writes each result row into the result's L2 slot (an epilogue, when one is declared), and signals `result` at the last row. |

### 5.2 Where the programs come from

From the program's Move, Feed, Call, Drain and Writeback actions, mapped onto the BlockMover's and the
Streamer's command queues. The L3s start with A and B in the slots the program places them in. That
state is Phase 1's verified result, or the same state loaded directly.

As built in step 2e (`cycle/compute/csp_compute_program.hpp`):
- **The L3 is loaded directly.** At first (2e) it was in the L2's lane layout. Since step 3a (#548)
  it is Phase 1's dense, row-major packing. The BlockMover transforms each tile on its write into
  L2: A by rows, B by columns (the transpose, which waits for the whole tile), and C's rows back to
  the dense tile.
- **Each L3 slot holds one residency over the run.** A program that reuses a slot needs its Loads
  run in time, which is also the join's work. So the 64³ runs with `l3 = 96`.
- **A chain ends at its last call,** since the RTL's `last` launches its results north at once. A
  deferred Drain (`store … late`) is the same L2 Result, in program order.
- **A call needs A and B in two L2 slots.** The Streamer reads both in the same clocks, while the CSP
  feeds each tile into L1 whole. So `l2 = 2`, and `l2 = 3` with late stores, are refused.

### 5.3 Ladder

1. **One PE:** one MAC per clock, and `first`/`last` handled.
2. **A 4×4 array, one command.** Check PE(i,j) multiplying element k at `s+1+i+j+k`, and row i
   leaving at `s+K+C+2i`. These are **measured from the trace**, as emergent properties, never coded
   as formulas.
3. A K chain over several commands, with only the last producing results.
4. Back-to-back commands with no bubble. Then the north-drain spacing: 2R−1 never collides, and a
   forced 2R−2 does.
5. Double buffering in L2 (one slot and two), and early L2 release (`completed` before `result`).
6. The RTL's 8×8×16 matmul, then an int8 64³ matmul on `kpu_s1`'s 128×64 array, a tile on its corner.

### 5.4 Done when

- **Values.** C in L3 equals L-B's per tile, int32 exact.
- **The RTL's properties hold as behaviour:** the L1 credit, no bubbles, the 2R−1 drain, early
  release, a chained reduction writing only at its end.
- **Invariants, every cycle:**
  - L1 credit conservation and no north collision;
  - L2 slot states (#387's checks);
  - no bank serving more reads than it has ports.
- **Counters.**
  - Array utilization and PE-cycles idle by cause (waiting on L1, draining).
  - L2 bank utilization and **requests in flight per bank**. This is #495's question, answered by
    measurement.
  - Streamer stalls by cause, BlockMover busy and stalled cycles, and Combiner occupancy.
- **Viewer and debug.** A Perfetto trace with tracks per lane buffer, PE row, FIFO, L2 bank and slot,
  and an array heatmap from the counters.

As built in step 2f (`cycle/compute/compute_subsystem.hpp`, `kpu-run --compute-subsystem`):
- **PE-clocks are counted by cause** in the array itself: useful (A and B), padded (A with no B, a
  column past the tile), draining (a result passing north) and waiting on L1. Each PE's causes add
  to the array's clocks, and that is checked as one of the laws.
- **The heatmap** is each PE's counts, written as `heatmap.json` with an `heatmap.svg` beside it.
- **The int8 64³ on `kpu_s1`** runs 5632 clocks at 0.57% array utilization: one 16×16 tile on the
  128×64 array, mover-bound (the BlockMover is busy 80% of the run). Each L2 bank averages 1.64 reads
  in flight, 8 at most.

## 6. Phase 3: join, evaluate, plan forward

Both subsystems on one clock: DRAM → L3 → array → L3 → DRAM for one tile's program, with the same data,
invariant and counter checks end to end. Then **evaluate**:
- where the cycles go, from the counters;
- which components bound the run;
- what the RTL's 907/744 comparison says now.

Then plan the next step from what is measured. Only after this phase does the question come back of
what L-T1 is calibrated against, and whether the new core replaces `ConcurrentTimingExecutor` as L-CA
(Q5).

## 7. Steps (one PR each: one component and its tests)

| Step | PR | Done when |
|---|---|---|
| 0a | `Clock`/`Domain`, `Component`, `Link`, `Simulator`; the test structure | §3.5's throughput, latency, backpressure and determinism tests |
| 0b | XUE counters on components and links | the operational laws exact on the test structure |
| 0c | Trace (Perfetto), state dump, invariant hooks, deadlock detector | the detector names a built credit cycle; a trace opens in Perfetto |
| 0d | Clock-domain crossing | the crossing tests at 4:1 |
| 1a | The LPDDR5 controller as a component | its invariant suite and trace validator pass on the core |
| 1b | L3 tile and DMA engine, direct link (rungs 1–2) | data in the right slots; credit invariants |
| 1c | NoC on the core (rung 3) | the bubble rule; two DMAs and two L3s |
| 1d | CSP-driven DMA programs, hypothesized C (rung 4) | §4.4 on the 8×8×16 and 64³ programs |
| 1e | The memory subsystem's counters summary and trace in `kpu-run` | §4.4's viewer and counters |
| 2a | L2 banks and slots on the core; BlockMover L3 ↔ L2 | rows in the right banks; slot states |
| 2b | Streamer and L1 lane buffers | the L1 credit; one A row and one B row per clock; `completed` |
| 2c | PE array (rungs 1–2) | the wavefront measured from the trace |
| 2d | North FIFOs and Combiner into L2; `result` | rungs 3–4 |
| 2e | CSP orchestration of the whole path (rungs 5–6) | §5.4 |
| 2f | The compute subsystem's counters, trace and array heatmap | §5.4's viewer and counters |
| 3 | Join, and the evaluation write-up | §6 |

Each PR adds its component's own tests and touches nothing in `timing/`, `program/csp/transactional.hpp`
or the records. So no existing number moves.

## 8. Decisions (the user's, 2026-11-24)

1. **Time base:** picoseconds, with a period per clock domain.
2. **Visualization:** Perfetto/Chrome traces and a counters summary first. Custom panels, such as the
   array heatmap, are added per subsystem.
3. **Phase 1's programs:** the walker's L3 actions from matmul programs, with C preloaded from L-B as
   the data oracle.
4. **Reuse:** the existing cycle-accurate LPDDR5 controller and NoC are ported onto the new core.
5. **Afterwards:** whether the new core replaces `ConcurrentTimingExecutor` as L-CA is decided at
   Phase 3's evaluation.
6. **Issues:** one epic in v0.10.0, with one issue per step in §7. The old issues (#458's PE-grid item,
   #389, #385, #495) are re-scoped under it.
