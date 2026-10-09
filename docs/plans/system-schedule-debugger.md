# System-Schedule Debugger: The Operator's Schedule, From Compute Down to DRAM Commands

**Date:** 2026-10-08
**Status:** Q1-Q5 decided 2026-10-08 (all as recommended; §7); steps 1-2 done; step 3 next
**Related:**
- `docs/plans/memory-side-debugger.md`: the `.mflow` record, `mflow_check.py` and the memflow viewer. This
  plan reuses all three for its memory half.
- `docs/plans/tile-flow-debugger.md` (#286): the L-T1 `.tflow` debugger. It shows tiles, but with L3
  and the movers pooled and no DRAM.
- `docs/plans/noc-port-arbitration.md`: the NoC that the multi-tile machines add (phase 2 here).
- `docs/01-architecture/kpu-execution-model.md`: credits up, data down.

## 1. Problem Statement

The memory-side debugger shows the T4's eight DMA engines served one after another. Engine k's
bursts start when engine k-1's end (dma[0] at cycles 0-1,727, dma[1] at 1,797-3,464, ...,
dma[7] at 10,119-13,254). Every load holds its L3 credit from about cycle 0 while waiting for
the memory controller. We cannot tell from any view whether a schedule is good, because no
view shows a schedule. The causes, in the code as it stands at `539b9aa`:

| What we saw | Cause | Where |
|---|---|---|
| Engines served in index order | No arbitration. Each engine calls `submit_burst` inside its own tick, the controller refuses when its queue is full, and engines tick in index order. Engine 0 gets every freed slot first. | `dma_engine_process.hpp:519-541`, `memory_controller_process.hpp:202` |
| Requests early, data late, credits held idle | Every operation is enqueued before cycle 0 (`ScheduleExecutor::execute`). A load takes its L3 credit as soon as a credit exists, so a credit can sit empty for thousands of cycles waiting for DRAM. | `schedule_executor.hpp:178`, `dma_engine_process.hpp:547-642` |
| No view of the schedule | The memory-side debugger stubs the compute by design. `.tflow` comes only from L-T1, which has no DRAM. Nothing records the cycle-level executor. | `kpu_run.cpp:705`, `execution_level.hpp:87` |
| No compute-tile timing | A compute is a list entry, not a process. Any number run at once, and latency is a fixed `32 + 32·(k-1)`. `macs_per_cycle` and `compute_tiles` are not mapped from the spec. Operations carry no compute-tile index. | `concurrent_timing_executor.hpp:1196-1282`, `csp_config_from_spec.hpp:127-133` |
| Reuse is accidental | The generator emits a LOAD for every use and relies on the L3 tag CAM to deduplicate it at run time. Reuse depends on whether the tile happens to still be resident, not on a decision in the schedule. | `matmul_schedule_generator.hpp:291-298`, `dma_engine_process.hpp:556-585` |

**Why reuse is the question.** A rough balance on the machine this plan emulates (§2):
- **Machine:** one compute fabric of 8,192 MACs/cycle, and DRAM at 34.1 B/cycle.
- **One k-slice of a 32³ fp32 tile:** 32,768 MACs, so **4 cycles** to compute.
- **Its A and B tiles:** 8 KB, so **about 240 cycles** to load.

A tile must therefore be reused about **60 times** from L3 before compute, not DRAM, sets the
pace. At 64³ the figures are 32 cycles against about 960, so about **30 times**. The schedule
that achieves that reuse has the lowest DRAM traffic per MAC, and the lowest energy. The
debugger must make reuse, and the compute time lost waiting for DRAM, visible and measurable.

## 2. Architecture

**Phase 1: one large L3 and one large compute fabric** (`kpu_s1`). This is the T4's resources
folded into one site: the T4's L3 capacity (two tiles' worth) in one L3, the T4's two compute
tiles' MACs in one fabric, and the same DRAM. There is no NoC: a DMA lands directly in the L3.
It isolates three things: the DMA traffic needed to start compute, the loading of later tiles
while compute runs, and reuse. Running the same schedule on the T4 (phase 2) then measures
exactly what partitioning and the NoC cost.

```
                              CREDITS (upstream)
                                     ↑
┌──────────────────────────────────────────────────────────────────────────────┐
│                     ConcurrentTimingExecutor  (kpu_s1)                       │
│                                                                              │
│  ┌────────────────────────┐ release when op ≤ cursor + P   ┌──────────────┐  │
│  │  Schedule dispatcher   │───────────────────────────────►│  DMA engines │  │
│  │  ScheduleResult + P    │  (prefetch depth P, §3.3)      │  0 .. E-1    │  │
│  │  cursor = oldest       │                                │  window W    │  │
│  │  unfinished compute    │                                └──────┬───────┘  │
│  └───────────▲────────────┘                                       │ burst    │
│              │ compute done                                       │ requests │
│              │                                                    ▼          │
│  ┌───────────┴────────────┐                     ┌────────────────────────────┐│
│  │  Compute fabric (CF)   │                     │  Memory controller         ││
│  │  one tile, M MACs/cyc  │                     │  ┌──────────────────────┐  ││
│  │  one compute at a time │                     │  │ Round-robin arbiter  │  ││
│  │  latency = MACs / M    │                     │  │ (§3.1) grants queue  │  ││
│  └───────────▲────────────┘                     │  │ slots = credits      │  ││
│              │ feed (L2/L1 streamers)           │  └──────────┬───────────┘  ││
│  ┌───────────┴────────────┐   tile arrives      │  hosted LPDDR5 (DramBridge)││
│  │  L3 (one, large)       │◄────────────────────│  every burst + command     ││
│  │  credits + tag CAM     │                     └────────────────────────────┘│
│  │  residency per tile    │                                                   │
│  └────────────────────────┘                                                   │
│                                                                              │
│  Recorder (§3.4) ──► .sflow: schedule tables + the .mflow memory tables     │
└──────────────────────────────────────────────────────────────────────────────┘
                                     ↓
                              DATA/TILES (downstream)
```

**Phase 2** runs the same record, checker and viewer on the T4 and T16: several L3 tiles,
several compute tiles, the NoC wired (NoC step 4b.4). The record carries an L3 tile and a
compute-tile index on every row from the start, so phase 2 adds no format change.

## 3. Design

### 3.1 Round-robin arbitration at the memory controller

The controller's request queue holds `request_queue_depth` bursts, and its free slots are the
controller's credits. Today the first engine to tick takes a free slot. With round-robin, the
controller grants free slots to waiting engines in turn, starting after the last engine it
granted.

WRONG (today): an engine submits from its own tick, so tick order is the priority.
```cpp
// DMAEngineProcess::issue_bursts -- first caller wins every freed slot
while (req.sent < req.bursts && bursts_in_flight_ < config_.window)
    if (!mc_.submit_burst(req.tile, req.sent, req.is_load, config_.engine_id)) return;
```

CORRECT: an engine posts its next burst as a request. The controller grants from its own tick,
round-robin over the engines with a request posted.
```cpp
// The engine's side: at most one burst request posted per engine (its head-of-line burst).
struct BurstRequest { TileDescriptor tile; std::uint64_t index; bool is_load; Cycle posted; };

class MemoryControllerProcess {
public:
    // The engine offers its next burst. False = it already has one posted.
    bool post_burst(std::uint32_t engine, const BurstRequest& r);
    // Granted bursts, for the engine to count against its window (the grant is the credit).
    std::optional<BurstRequest> take_grant(std::uint32_t engine);
private:
    // In tick(), before the controller advances: grant free queue slots to posted requests,
    // starting at rr_next_, one per engine per pass, until the queue is full.
    void arbitrate(Cycle now);
    std::vector<std::optional<BurstRequest>> posted_;   // by engine
    std::uint32_t rr_next_ = 0;
};
```

**Timing.** The executor ticks the controllers before the DMAs. A burst posted in cycle t is
granted at the earliest in cycle t+1. That is one cycle of request latency per burst, which the
window of W bursts in flight hides. The `posted` cycle is recorded, so the wait for a grant is
visible.

**Scope.**
- The window path (`issue_bursts`) and the whole-tile hosted path (`submit_hosted`) both go
  through the arbiter.
- The memory-side harness uses the same classes, so `kpu-memsim` gets round-robin with no
  change of its own.
- `Config::arbitration` takes `round_robin` (the new default) or `fixed`, today's behaviour,
  kept so the difference can be measured.
- **As built:** the legacy, unhosted fixed-latency model stays first-come. It has no DRAM, is
  not on the system-schedule path, and many suites pin its timings. Every hosted controller is
  arbitrated.

**Which burst on an engine's turn (as built).** One burst per engine per turn, as decided (Q5).
Of the engine's posted bursts, the arbiter grants the one whose channel has the fewest bursts
outstanding, the oldest on a tie.
- **Why:** a tile's bursts are independent (the tile completes when all are home). Strict
  post order makes engines that walk their tiles in step convoy onto one channel at a time.
- **Measured** on the T4, whose map interleaves channels every 2 KB:
  - strict-order round-robin kept one channel busy 89% of the time and reached 55% of the
    ceiling;
  - the channel-balancing pick reaches 96%; fixed priority reaches 93%.
- **Optional quantum:** `grant_quantum` (default 1, a model option) grants up to Q bursts per
  turn, for characterization.

### 3.2 The compute fabric

A compute tile becomes a resource. It runs one compute at a time, and its latency comes from
the spec, not from fixed constants.

WRONG (today): unlimited concurrent computes, with latency `32 + 32·(k-1)`.

CORRECT:
```cpp
struct ComputeTileModel {
    double macs_per_cycle;        // spec.macs_per_cycle (one compute tile's throughput)
    Cycle fill;                   // pipeline fill and drain, cycles (Config; default 2·tile edge)
};
// Latency of a compute over Ti x Tj x Tk x k_slices:
//   fill + ceil(Ti·Tj·Tk·k_slices / macs_per_cycle)
// One PendingCompute runs per compute tile at a time; the rest wait, and the wait is recorded
// with its reason (inputs not yet fed, or the compute tile busy).
```

- **Spec mapping:** `csp_config_from` maps `compute_tiles` and `macs_per_cycle`, which are
  unmapped today.
- **Assignment:** `ScheduleOperation` gets `int32_t cf_tile = -1`. A schedule names the tile;
  -1 means the executor picks the least-loaded tile.
- **Values:** `schedule_matmul_compute` still computes real values at completion. Only when it
  completes changes, so values stay bit-identical (CLAUDE.md, ADR 0002).

### 3.3 Schedule-paced DMA: a dispatcher with prefetch depth P

WRONG (today): `ScheduleExecutor` enqueues every operation before cycle 0, and loads take
credits greedily.

CORRECT: a dispatcher releases operations in schedule order against a **cursor**, the oldest
compute not yet complete:
- An operation is released when it belongs to a compute at most **P** computes ahead of the
  cursor.
- P = 1 is classic double buffering: load the next tile while computing this one.
- P = infinity reproduces today's behaviour, so the difference can be measured.

```cpp
// A SCHEDULE parameter, not a machine parameter (decided, Q2): it never appears in a deployment
// spec, it is chosen per run (--prefetch), and records file it under "schedule", beside the
// strategy, never under "machine".
struct DispatchConfig {
    std::uint32_t prefetch_depth = 1;        // P: computes ahead of the cursor whose inputs may load
};
// Each op is tagged with the compute it serves (its consumer, from make_compute_dependencies);
// a STORE/WRITEBACK with the compute that produced it. The dispatcher releases op i when
//   consumer_index(i) <= cursor + P
// A released LOAD still waits for its L3 credit (push with credit): pacing decides WHEN a load
// may ask, credits decide WHETHER there is room.
```

**What it measures.** For every tile:
- **Credit idle:** from credit acquired to arrived in L3. Credit held, slot empty.
- **Resident:** from arrival to the last consumer's release.
- **Uses:** how many computes consumed it.

For every compute:
- **Waiting on data:** from released to all inputs fed.
- **Waiting on the compute tile:** from inputs fed to start.
- **Running:** start to end.

**Reuse, made explicit.** The run also reports:
- **Loads per distinct tile:** 1.0 is perfect reuse.
- **DRAM bytes per MAC.**
- **DRAM bytes against the minimum:** each operand tile loaded once, C stored once.

This plan does not write a reuse-optimal generator (§7, Q4). It makes the generator's reuse a
number that a better schedule must beat.

### 3.4 The record: `.sflow`

**Format:** "kpu-sflow" v1, in the shared columnar container (`record/columnar.hpp`). It holds
the `.mflow` tables unchanged (bursts, commands, requests, buffers, ports), so `mflow_check.py`
and the memflow lanes work on its memory half. It adds the schedule tables:

| Table | One row per | Columns |
|---|---|---|
| `ops` | schedule operation | index, type, tile (matrix, ti, tj, tk), consumer compute, cf_tile, l3_tile, released, started, done |
| `computes` | compute | op, cf_tile, released, inputs_fed, started, done, macs, input range (into `compute_inputs`) |
| `compute_inputs` | (compute, input tile) | compute, tile, arrived_l3, fed |
| `residency` | tile resident in an L3 slot | tile, l3_tile, slot, credit, arrived, released, uses |
| `requests` (extended) | DMA request | the `.mflow` columns, plus op, tile, and the compute it serves |
| `bursts` (extended) | DRAM burst | the `.mflow` columns, plus `t_posted` (the grant wait, §3.1) |

**Manifest:** three groups, kept apart so that a schedule is never mistaken for a machine
property:
- **`machine`:** the device spec's values (L3 tiles and capacity, compute tiles and MACs/cycle,
  DMA engines and window, the DRAM and its timing as in `.mflow` v2), and the controller's
  arbitration.
- **`schedule`:** the algorithm and problem size, the strategy, and the prefetch depth P.
- **`metrics`:** makespan, compute utilization, DRAM bytes against the minimum, loads per tile,
  and credit-idle tile-cycles.

**Recorder.** The executor emits `TimingEvent`s that carry neither an L3 tile nor a compute
tile. They gain `cf_tile` and `l3_tile`, and the executor gains `record()`, which assembles the
tables from its events, its homes map and the controllers' recorded bursts and commands.
Recording is off unless asked for, as for the controllers today.

### 3.5 The checker: `tools/trace/sflow_check.py`

It has the same exit-code contract as the others. It runs `mflow_check`'s TF10, TF11 and M1-M5 on
the memory tables, and adds:

| Check | Invariant |
|---|---|
| S1 | a compute starts only after every input arrived in L3 and was fed (no compute on data not yet pushed) |
| S2 | a compute tile runs one compute at a time |
| S3 | L3 occupancy (resident, plus credits held while waiting) never exceeds capacity, per L3 tile |
| S4 | a load is never released before the dispatcher allows it (consumer ≤ cursor + P at release) |
| S5 | round-robin: while two engines have bursts posted, neither is granted twice before the other once |
| S6 | a tile's residency is filled by a DMA arrival or a compute write (the TF9 analogue), and released only after its last consumer is done |
| S7 | the result's values match the reference (the tool compares them, and the manifest records the verdict) |

### 3.6 The viewer: `tools/visualization/sysflow/`

The memflow page's code (loader, lanes, transport, heat map, inspector), with a schedule half
above its memory lanes:
- **Compute lanes:** one per compute tile. Each compute is a bar labelled C[i,j] (k-slices).
  Before each bar, a hatched lead-in shows the time it waited, coloured by reason: data or a
  busy compute tile.
- **L3 lanes:** occupancy against capacity, split into resident tiles and credits held for
  tiles not yet arrived (the idle reservation the T4 run showed). A tile filter highlights one
  tile's residencies.
- **DMA requests** labelled with their tile and coloured by operand (A, B, C).
- **Critical path:** clicking a compute shows the input that arrived last. That leads to its
  request (released, credit, first burst, landed), its bursts and the DRAM commands behind
  them, with the wait at each step attributed to the dispatcher, a credit, the DMA window, the
  arbiter or DRAM.
- **Reuse panel:** loads per tile as a histogram, DRAM bytes against the minimum, and DRAM
  bytes per MAC.
- **Summary chips:** compute utilization, time to first compute, and the share of compute time
  spent waiting on data.

### 3.7 The tool: `kpu-sysim`

```
kpu-sysim --deploy tests/program/deploy/kpu_s1.json --algo matmul --size 512 --tile 32
          [--strategy interleaved|output_stationary|prefetch_next|blocked]
          [--prefetch P] [--arbitration round_robin|fixed] --out run.sflow
```

`--strategy` and `--prefetch` are schedule parameters, and `--arbitration` is a model option of
the controller. None of them is read from, or written to, the deployment spec.

- It runs the cycle-level executor with values, as `test_csp_spec_oracle.cpp`'s `run_matmul`
  does: payloads seeded, `schedule_matmul_compute`, C read back from DRAM. That routine becomes
  a library function both use.
- It compares C with the L0 reference (bit-exact for matmul, per ADR 0001 D5).
- It prints the summary metrics and writes `.sflow`.
- Exit codes: 0 ok, 1 values differ or did not finish, 2 bad input.

## 4. Implementation Steps

Each step is one PR and ends green.

1. **Round-robin arbitration** (§3.1).
   - Files: `memory_controller_process.hpp`, `dma_engine_process.hpp`, `concurrent_timing_executor.hpp`
     (Config::arbitration), `memory_side_harness.hpp`; `.mflow` gains bursts `t_posted` (v3);
     `mflow_check.py` gains M6 (S5's rule).
   - Tests:
     - eight engines loaded at once overlap in time, not one after another;
     - the grant order is round-robin, including when engines have unequal amounts of work;
     - `fixed` reproduces today's order;
     - DRAM bandwidth on the T4 mixed scenario stays within 2% of today's;
     - values are unchanged in `test_csp_spec_oracle`.
   - (Done.)
   - **As built:**
     - **Engines post.** A window burst goes into a per-engine queue (`post_burst`), and a
       whole tile goes into one posted slot per engine. A second tile is refused, so the
       engine releases its L3 credit and retries, as it did on a full queue.
     - **The controller grants** at the top of its tick (§3.1). The record carries each
       burst's `t_posted` (`.mflow` v3), plus `arbitration` and `grant_quantum` in the manifest.
     - **Settings:** `kpu-memsim` scenarios take `arbitration` and `grant_quantum`. Executor
       Config: `mc_arbitration`, `mc_grant_quantum`.
     - **M6** (`mflow_check.py`): while an engine waits with a burst posted, no other engine
       of its controller is granted more than two turns. It reads waiting from the earliest
       post among the engine's later grants, because grants may leave post order.
     - The viewer shows the arbitration and each burst's wait for a grant.
   - **Measured** on the T4 (`kpu-memsim`):

     | Scenario | fixed | round-robin |
     |---|---|---|
     | mixed (`t4_mixed.json`) | 90% of the ceiling, engines in turn | 91%, all eight engines served throughout |
     | infinite sink, 8 engines | 93% | 96% |
   - **Found along the way:**
     - **Channel convoy:** described under §3.1.
     - **A tFAW bug in the LPDDR5 controller.** Its four-activate window used 0 as "empty",
       so an ACT at controller cycle 0 was invisible. The arbiter now grants on the
       controller's first tick, and M5 caught five ACTs in 24 ticks against a tFAW of 32.
       - Fixed: `kNever` marks empty entries.
       - The existing "tFAW constraint" test now measures the ACT spacing. It fails on the
         old code (21 < 24) and passes now.
       - The GDDR6/7, DDR5 and HBM2/3 controllers share the 0-sentinel pattern. **Follow-up,
         not changed here.**
       - The controller's own INV-TIME-4 check tests "more than 4" in a 4-entry window, so it
         can never fire. Also a follow-up.
     - **Over-refresh, observed, not changed.** AUTOMATIC refresh refreshes a bank whenever
       `tREFIpb` has passed since that bank's last refresh. That is 16 times the rate its own
       deadline logic assumes (16 x `tREFIpb` per bank).
       - Under strict-order round-robin it issued 1,138 REF against 148.
       - Pacing it to one bank per `tREFIpb` cut the commands but not the bandwidth gap, so
         it was not the cause here. Left for a DRAM-model follow-up.
     - **Address map (a machine question for the architect).** With channel bits lowest
       (`ch:co:…`, bursts alternating channels), both arbitrations reach 94-97% on the T4.
       The current map's 2 KB channel interleave is why channel balancing matters.
     - **Port lockstep at Q = 1.** On `t4_mixed`, the port stub refuses 425 landings against
       27 at Q = 4, with the same bandwidth: per-burst fairness finishes the engines' tiles
       together.
     - The LPDDR5 trace validator passes 18 of 19 regenerated pattern traces. The 19th,
       `complex/random_trace.json` (INV-001), fails identically on the trace committed in
       January.
2. **The compute fabric and `kpu_s1`** (§3.2).
   - Files: the `ComputeTileModel` in `concurrent_timing_executor.hpp`; `cf_tile` on
     `ScheduleOperation`; `csp_config_from` maps `compute_tiles` and `macs_per_cycle`;
     `ArrayLayout` and `csp_config_from` accept `single` with a DRAM section;
     `tests/program/deploy/kpu_s1.json`.
   - Tests:
     - one compute at a time per tile;
     - latency follows the MACs model;
     - values are bit-identical to the L0 reference on `kpu_s1`, T4 and T16.
   - (Done.)
   - **As built:**
     - **Executor Config:** `num_compute_tiles` (0 = the legacy unbounded model, which every
       hand-built config keeps), `macs_per_cycle` (0 = the legacy latency) and
       `compute_fill_per_edge` (2.0).
     - **Latency:** `ceil(2 x the result's longer edge) + ceil(rows x cols x K / macs_per_cycle)`.
       K is summed over the compute's A inputs, each at its width as fed.
     - **The matmul generator gives each matrix its own tile shape:** A is `Ti x Tk`, B is
       `Tk x Tj`, C is `Ti x Tj` (`Config::tile_bytes(matrix)`, used for sizes and DRAM
       addresses). It used to size every tile `Ti x Tj`, which made K wrong for non-square
       tiles (found in review). Square tiles are unchanged.
     - **Assignment:** `TileDescriptor::cf_tile` names the compute tile, beside `l3_tile`; the
       plan's `ScheduleOperation::cf_tile` would not reach the executor's compute overloads,
       which take a descriptor. Unnamed computes take the first free tile. A tile whose compute
       completes in a cycle may start the next one in that cycle.
     - **Events and accessors:** `COMPUTE_START`/`COMPUTE_COMPLETE` carry the compute tile as
       `component_id`. New accessors: `compute_tiles()`, `compute_tile_busy_cycles(t)`.
     - **Spec mapping:** `csp_config_from` maps `compute_tiles` and `macs_per_cycle`, so
       every spec-driven run uses the fabric.
     - **`kpu_s1.json`:** the T4's DMA, L3 capacity (128 tiles), DRAM and streamers. One L3
       tile and one compute tile of 8,192 MACs/cycle. The `single` layout derives **one**
       BlockMover (one L3-compute edge), where the T4 has four.
   - **Tests** (`test_csp_spec_oracle`):
     - values are bit-identical to the L0 reference on S1, T4, T16 and T64;
     - on S1, the 16 computes never overlap, each takes exactly 64 + 16 cycles, and the busy
       counter matches;
     - on T4, both tiles are used, neither is double-booked, and each compute takes 64 + 32;
     - a named compute tile is honoured, and one the fabric lacks is refused by name.
   - **First numbers,** matmul 128³ with 32³ tiles:

     | Machine | Cycles |
     |---|---|
     | S1 | 9,836 |
     | T4 | 9,549 |
     | T16 | 5,084 |
     | T64 | 3,796 |

     S1 has twice the MACs per compute tile and is no faster than the T4. That is consistent
     with §1's balance (memory-bound) and with S1's single BlockMover. Attributing it is the
     step-4 record's job.

3. **The dispatcher** (§3.3).
   - Files: `schedule_executor.hpp` (the cursor and P), consumer tagging in
     `make_compute_dependencies`.
   - Tests:
     - P = infinity reproduces today's event order;
     - at P = 1 no load is released early;
     - credit-idle time falls as P falls;
     - values are unchanged.
4. **The record and the tool** (§3.4, §3.7).
   - Files: `record/system_flow_record.hpp`/`.cpp` (sharing `.mflow`'s table writers);
     `TimingEvent` gains `cf_tile` and `l3_tile`; `ConcurrentTimingExecutor::record()`;
     `tools/kpu-sysim/`.
   - Tests:
     - a round trip;
     - every compute input names a residency;
     - ctests on `kpu_s1` at P = 1 and P = infinity.
5. **The checker** (§3.5).
   - `sflow_check.py` and a self-test that breaks each of S1-S7; ctests on the step-4 bundles.
6. **The viewer** (§3.6).
   - `tools/visualization/sysflow/`: pack, smoke test, and a headless render checked by eye.
   - The how-to (`docs/tools/debuggers-how-to.md`) gains a third debugger.
7. **Phase 2.** T4 and T16 with the NoC wired.
   - The same tool, record, checker and viewer, plus NoC transit rows in the request chain.
   - Compare against `kpu_s1` on the same problem: what partitioning and the NoC cost.

## 5. Verification

- **Step 1.** On the T4 mixed scenario (`kpu-memsim`), each engine's bursts span most of the
  run rather than one-eighth of it. The share of grants per engine over any 512-cycle window
  stays within one grant of fair while all engines have work. `mflow_check` M6 passes, and its
  self-test fails a fixed-priority bundle.
- **Steps 2-4.** On `kpu_s1`, matmul 512³ with 32³ tiles:
  - values are bit-exact with the reference;
  - the first compute starts once its first A and B tiles have landed, and the record gives
    that startup time;
  - at P = 1, later loads overlap running computes;
  - compute utilization, DRAM bytes against the minimum, and loads per tile are reported for
    each strategy and for P in {1, 2, 4, infinity}.

  The expected shape follows §1's balance: memory-bound, with utilization well under 50% for
  every strategy whose loads per tile are much above 1. These numbers are the baseline a
  reuse-aware schedule must beat.
- **Step 5.** `sflow_check` passes every bundle, and its self-test fails each broken case.
- **Step 6.** A headless render: the compute lanes show wait lead-ins, and a click on a stalled
  compute reaches the DRAM commands behind its last input.
- **Throughout:** the full suite stays green, and `-Wconversion` is clean on changed files.

## 6. Key Invariants

- Credits up, data down. A load is released by the schedule (P) and then pushes only with an L3
  credit. Nothing is fetched on demand, and pacing never bypasses a credit.
- Every hosted memory-controller queue slot is granted by the arbiter. No engine reaches the
  controller except through `post_burst`.
- A compute tile runs one compute at a time, and a compute starts only on fed inputs.
- Values never move: arbitration, pacing and the compute model change when things happen,
  never what is computed (ADR 0002).
- DRAM timing in every record is checked in controller ticks (`.mflow` v2).
- The record never invents a place. Every row's L3 tile and compute tile come from the
  executor's own assignment, not from a derived binding.

## 7. Decided on Review (2026-10-08)

| Q | Decision |
|---|---|
| Q1 | One `.sflow` bundle that embeds the `.mflow` tables unchanged |
| Q2 | P is a run option (`--prefetch`, `DispatchConfig`), marked as a schedule parameter, not a machine parameter: the manifest files it under `schedule`, and no deployment spec carries it |
| Q3 | `kpu_s1`: one L3 of 128 tiles, one compute tile of 8,192 MACs/cycle, the T4's DRAM |
| Q4 | The reuse-aware schedule is deferred to the next plan |
| Q5 | The arbiter grants one burst at a time, round-robin |

## 8. Questions (as posed)

- **Q1. One record or two?** Recommended: `.sflow` holds the `.mflow` tables unchanged, so the
  memory checker and lanes are reused as is. The alternative is two bundles from one run,
  joined by IDs, which is simpler to write but harder to keep consistent.
- **Q2. Where does P live?** Recommended: a run option, not a spec field (`--prefetch`,
  `DispatchConfig`). Prefetch depth is a property of the schedule, not of the machine.
  Alternatively it could live in the ScheduleResult's metadata, chosen by the generator.
- **Q3. `kpu_s1` sizing.** Recommended: the T4's resources folded together:
  - one L3 of 128 tiles (the T4's total `capacity_tiles`);
  - one compute tile of 8,192 MACs/cycle;
  - the T4's DRAM.

  This makes it a direct comparison with the T4. The alternative is a larger L3 (say 1,024
  tiles) to show what reuse buys when capacity is not the limit. Running both is cheap once
  the fixture exists.
- **Q4. A reuse-aware schedule in this plan, or the next?** Recommended: the next. This plan
  makes reuse measurable and visible, and gives a generator a target to beat. The next plan
  writes the generator: L3-resident panels sized to capacity, so each tile loads once.
- **Q5. The arbiter's granularity.** Recommended: a burst at a time, round-robin. The
  alternative grants a whole tile's bursts at once, which keeps page locality per engine but
  lets one large tile block the rest. With round-robin per burst, the page outcomes show
  whether interleaving engines costs locality; the T4's sequential streams will say.
