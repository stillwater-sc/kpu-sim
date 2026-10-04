# DRAM Bank Concurrency, the DMA Abstraction, and Bank-Aware Allocation

**Date:** 2026-10-04
**Status:** Design (Q1-Q6 decided on review, 2026-10-04)
**Related:** #283 (L-T2 resource-transactional level), #286 (tile-flow debugger), ADR 0001 D5 / §7.5,
`docs/plans/memory_controller_transactional.md`, `docs/plans/dma_csp.md`

## 1. Problem Statement

The T64 deployment has 4 memory controllers and 32 DMA engines (8 per controller). The question
behind this note:

1. How much concurrency does an LPDDR5X channel actually offer?
2. Should the DMA engine count follow bank count?
3. Is the cost of bank conflicts something we can model before a run, or only measure during
   one?

These are the answers, followed by what the simulator needs in order to model them.

### 1.1 LPDDR5X geometry

| Quantity | Value | Consequence |
|---|---|---|
| Channel width | x16 | the unit of independent command/data buses |
| Banks per x16 channel | 16 (4 bank groups x 4, BG mode above 3200 MT/s) | 16 rows can be open at once |
| 32-bit "channel" | two x16 channels | 2 command buses, 32 banks per rank |
| Data rate | 8533 MT/s in this note (the T64 default; faster parts exist, e.g. 10.7 Gbps) | 17.1 GB/s per x16 channel, 34.1 GB/s per 32-bit, at that rate |
| Burst | BL16 = 32 B, BL32 = 64 B on x16 | the minimum transfer, not a tile |
| ACT limits | tRRD_S / tRRD_L, tFAW (4 ACT per window) | at most ~4 row openings per tFAW, whatever the bank count |
| Refresh | per-bank (tREFIpb, tRFCpb) or all-bank | a bank disappears for tRFC, on a clock the schedule does not see |

**Banks hide latency; they do not add bandwidth.** The banks of a channel share one data bus.
Sixteen open banks let the controller overlap ACT/PRE of one bank with data transfer from
another, so the bus can be kept busy. The bus is still the ceiling, at 17.1 GB/s per x16.

### 1.2 Little's law sizes the window, not the engine count

To keep one x16 channel busy, the bytes in flight must cover bandwidth x loaded latency. The
figures below are for 8533 MT/s; the window scales with the declared `data_rate_mtps`:

```
17.1 GB/s x ~80 ns  ~= 1.4 KB  ~= 21 x 64 B bursts in flight per x16 channel
                                ~= 42 per 32-bit channel
```

This is a count of **outstanding bursts**, not of engines. One engine that pipelines its bursts
(a 4 KiB tile is 64 bursts) can cover it alone. Thirty-two engines that each wait for one burst
before issuing the next cannot.

So: **no, the DMA count should not track the bank count.** Concurrency in this note is split
three ways, and each part has its own sizing rule:

| Part | Owns | Sized by |
|---|---|---|
| DMA engine | one descriptor stream: address generation, tile -> bursts, push to L3 under credit | the number of tensor streams the schedule keeps live at once |
| DMA outstanding window `W` | bursts in flight per engine | Little's law: `N_engines x W >= bursts needed in flight per controller` |
| Memory controller | per-bank queues, row state, ACT/CAS arbitration, FR-FCFS reordering, refresh | the DRAM part (banks, bank groups, timing) |

Bank concurrency is the **controller's** job. If engines are made bank-addressed, the
controller's scheduling decision moves into the compiler, and the compiler cannot see refresh or
other masters. The engine count of 8 per controller is a reasonable **stream** count. It was
never a bank count.

### 1.3 A priori or runtime? Both, split along one line

- **The structure of conflicts is static.** It is fixed before the run by three things, all of
  which the compiler or loader knows:
  - the operator's schedule (affine, so which tiles stream concurrently is known);
  - tensor allocation (`TensorRef::device_address` in the loadable);
  - the address map (address -> channel, bank group, bank, row).

  Together they determine which streams share a bank, and whether they share it with different
  rows. That is a conflict graph, computable without running anything.
- **The cost of conflicts is dynamic.** Four things the static view cannot see:
  - refresh, which steals a bank on its own clock;
  - other masters: the CPU, other devices, and `foreign_slots` traffic;
  - runtime allocation that differs from the compiler's, for example under multi-tenancy or a
    loader relocation;
  - feedback: a conflict delays a stream, which shifts its overlap with every other stream,
    which changes the next window's conflicts.

So we model it a priori as a **bound and an expectation**, and we measure the actual cost at
runtime. The difference between the two is itself a diagnostic, an unmodelled residual (§3.5).
Once the deterministic costs the bound omits are added back (startup and CAS latency, ACT-rate
limits, turnaround), what remains is the part caused by refresh, interference and feedback.

### 1.4 What the simulator cannot do today

There are two controller models. They are not at the same fidelity:

| Model | Where | Banks | Bank groups / tRRD / tFAW / refresh | Burst decomposition | Data-bus contention |
|---|---|---|---|---|---|
| `LPDDR5MemoryController` | `include/sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp` | 16 per channel, 1-2 channels | **yes**, FR-FCFS, validated by `patterns/memory/lpddr5/` | per request | yes |
| `MemoryControllerProcess` | `include/sw/kpu/timing/memory_controller_process.hpp` (the CSP executor's) | 16 | no | **no**: a tile is one request | **no** |

The CSP-tier `MemoryControllerProcess` is the one on the program path, and it has three defects
that make it blind to everything in §1.1–1.3:

1. **A tile is one request.** The bank and row come from the tile's first address
   (`submit_request`, line 172). Latency is `startup + tRP + tRCD + tCL + t_burst` regardless of
   `tile.size_bytes` (`compute_latency`, line 405ff). A 64 KiB tile costs the same as a
   64 B burst.
2. **The data bus is not a shared resource.** `data_bus_ready_` is written at line 503 and never
   read. Sixteen banks can each finish a tile in the same window, so bandwidth scales with
   bank count. That is the opposite of §1.1.
3. **No address map in the spec.** The map is hard-coded (`bank = (addr >> col_bits) & 0xF`),
   with no channel bits and no hash. `DeploymentSpec::memory` declares only `controllers`, and
   no DRAM size or geometry, so `bind.py` has to invent "top of memory".

The overload figure in the tile-flow binding (115 DMA transfers past 8 engines per controller at
512³ on T64) comes from the same single-outstanding assumption. It overstates contention, and it
should be re-expressed in outstanding bursts once `W` exists.

Per CLAUDE.md, timing answers to L-CA. Until defects 1–3 are fixed, **L-CA is not an authority
for DRAM timing.** The temporal LPDDR5 controller is the only bank-faithful model in the repo.

## 2. Architecture

```
  DeploymentSpec.memory.dram ------------------------------------------------+
  (channels, width, banks, BGs, page, burst, rate, capacity, address map)    |
                                                                             v
  Loadable                                                       +----------------------+
  TensorRef.device_address --+                                   |  DramAddressMap      |
                             |                                   |  decode(addr) ->     |
  Schedule (affine) ---------+---> [static] DramConflictModel -->|  {mc, ch, rk, bg, ba,|
                                    per-window bank load,        |   row, col}          |
                                    predicted conflicts          +----------+-----------+
                                           |                                |
                                           v                                v (shared, one implementation)
                                     tile-flow viewer          +-------------------------------------+
                                     (predicted vs measured)   |            run time                 |
                                           ^                   |                                     |
                                           |                   |  DMA engine (CSP process)           |
                                           |                   |   L3 credit first (reserve kept)    |
                                           |                   |   descriptor -> bursts, window W    |
                                           |                   |        |  submit burst              |
                                           |                   |        v                            |
                                           |                   |  Memory controller                  |
                                           |                   |   per-bank queues, FR-FCFS,         |
                                           |                   |   tRRD/tFAW/refresh, bus per channel|
                                           |                   |        |  burst complete            |
                                           |                   |        v                            |
                                           |                   |  DMA: all bursts home; the tile is  |
                                           |                   |       in its reserved L3 slot       |
                                           +-------------------+  .tflow: per-bank occupancy         |
                                                               +-------------------------------------+

  Levels:  L-CA  per-burst, per-command (calibration source)
           L-T2  per-burst read/write on bank resources B (below), service times from L-CA (#283)
           L-T1  per-tile, DRAM as an aggregate rate (unchanged)
```

Credit flow is unchanged. DRAM is upstream of the DMA engine. The engine reads bursts from the
controller, which applies back-pressure when its request queue is full: `submit_request` returns
false.

**The L3 credit is taken before the first burst is submitted, not after the tile is assembled.**
That is today's order in `DmaEngineProcess::process_pending_loads`: a load acquires its
partition credit, refusing to dip into `l3_credit_reserve`, and only then submits; if the
controller refuses, it releases the credit. The burst decomposition keeps that order. Taking the
credit after assembly would need a staging buffer of assembled tiles waiting for credit, with a
bound nobody has stated. A greedy engine could also fill that buffer and then race writebacks
for every freed credit, which is the livelock the reserve exists to prevent. Holding the credit
across the bursts costs slot-time and nothing else, and the slot is the one the bursts fill. Nothing
in this note introduces demand fetch or cache terms. "Page hit / page conflict" is DRAM row-buffer
classification, which is a legitimate use.

## 3. Design

### 3.1 DRAM geometry and address map in the spec

These fields are optional, like `array`, `memory.controllers` and `cpu.harts`. If they are
absent, the current behavior is kept, and the binding says it assumed a geometry.

```cpp
struct Memory {
    std::optional<Dim> controllers;
    struct Dram {
        std::string technology = "lpddr5x";  // lpddr5x | hbm3 | ...
        Dim channels = 2;                    // per controller: x16 channels (32-bit = 2)
        Dim channel_width_bits = 16;
        Dim ranks = 1;
        Dim bank_groups = 4;
        Dim banks_per_group = 4;
        Dim page_bytes = 2048;               // row-buffer size per channel
        Dim burst_bytes = 64;
        Dim data_rate_mtps = 8533;
        std::uint64_t capacity_bytes = 0;    // top of memory; REQUIRED, a power of two
        // Bit order low -> high above the burst offset, plus optional XOR folds.
        std::string map = "co:ch:rk:bg:ba:mc:ro";
        std::vector<XorFold> xor_folds;
    };
    std::optional<Dram> dram;
} memory;
```

**Fields.** `co` is the burst within the page (log2(page_bytes / burst_bytes) bits), `ch`
the channel within its controller, `rk` the rank, `bg` the bank group, `ba` the bank within its
group, `mc` the controller (log2 of `memory.controllers`), and `ro` the row. `ro` takes whatever
`capacity_bytes` leaves, so the map covers the capacity exactly. That is why the capacity is
required: without it there is no top of memory and no row field.

**The fold contract.** A fold `{into, from, bits, from_lsb}` XORs bits
`[from_lsb, from_lsb + bits)` of field `from` into bits `[0, bits)` of field `into`, bit i onto
bit i. `bits` may not exceed `into`'s width or run past `from`'s. A fold never reads its own
destination (`into` differs from `from`), and a field that one fold reads is never written by
another, so decode and encode stay inverses.

**The T64 row->bank fold** is 4 row bits onto the 4 bank-identity bits. `ba` and `bg` are 2 bits
each, so the fold is written as two folds:

```json
"xor_folds": [ { "into": "ba", "from": "ro", "bits": 2 },
               { "into": "bg", "from": "ro", "bits": 2, "from_lsb": 2 } ]
```

That is `ba[0,2) ^= ro[0,2)` and `bg[0,2) ^= ro[2,4)`. Sixteen consecutive rows that a linear
map stacks on one bank land in sixteen banks. The C++ decoder (`DramAddressMap`, step 1) is the
definition. Any mirror, such as the viewer's `bind.py`, is checked against a fixture the C++
writes.

### 3.2 One address map, used everywhere

```cpp
// WRONG: each consumer decodes addresses its own way
uint32_t bank = (addr >> config_.col_bits) & 0xF;           // MemoryControllerProcess
bank = (addr // 4096) % 16                                  # bind.py, a guess

// CORRECT: one decode, derived from the spec, shared by the MC, the static model and the
// record; bind.py mirrors it and is checked against a C++-generated fixture
struct DramCoord { Dim mc, channel, rank, bank_group, bank; std::uint64_t row, col; };
class DramAddressMap {
public:
    static DramAddressMap of(const DeploymentSpec&);
    DramCoord decode(std::uint64_t addr) const;
    std::uint64_t encode(const DramCoord&) const;   // decode(encode(c)) == c, tested
};
```

### 3.3 The DMA engine issues bursts within a window

```cpp
// WRONG: the engine is single-outstanding and the tile is the request
mc_.submit_request(req.tile, true, engine_id);   // one request, size-independent latency

// WRONG: one engine per bank -- moves the controller's scheduling decision into the compiler,
// which cannot see refresh or other masters

// CORRECT: the L3 credit is taken first, exactly as today (partition, reserve and release on
// refusal unchanged); then the tile is decomposed into bursts; up to W are in flight; the MC
// reorders by bank
if (!tile_has_credit_ && !acquire_l3_credit(tile)) return;        // reserve honoured
tile_has_credit_ = true;
while (in_flight_ < config_.window && next_burst_ < tile_bursts_) {
    if (!mc_.submit_burst({tile.id, burst_addr(next_burst_), is_load, engine_id})) break;  // MC full
    ++in_flight_; ++next_burst_;
}
// ... on each burst completion: --in_flight_; when all bursts are home, the tile is complete in
// the slot its credit reserved, and the engine pushes it (no second wait, no staging buffer)
```

### 3.4 Each channel has one data bus

```cpp
// WRONG: data_bus_ready_ written, never read -- bandwidth scales with bank count
data_bus_ready_ = current_cycle + latency;

// WRONG: one data_bus_ready_ for a two-channel controller -- serializes independent channels
// and caps a 32-bit controller at one x16's bandwidth

// CORRECT: CAS issue requires the slot on the burst's OWN channel's data bus
Cycle& bus = data_bus_ready_[req.coord.channel];   // one per decoded channel
if (cas_data_start < bus) continue;                 // try another ready bank (FR-FCFS)
bus = cas_data_start + t_burst;
```

### 3.5 The static conflict model

The input is the schedule, the loadable and the spec. The output is a predicted occupancy per
bank and per time window, plus a conflict list. The model reasons in windows of the schedule, not
in cycles:

```
for each schedule window w:
    S_w = tile transfers the schedule keeps concurrently live in w
    B = (mc, channel, rank, bank_group, bank)        -- a physical bank's full identity
    for each transfer s in S_w: rows(s) = { (B, row) of its bursts }
    conflict(w) = pairs (s, t) with a shared B and different rows
    load(w, B) = bursts addressed to bank B in w
predicted cost(w) >= max over banks of row-switch count x (tRP + tRCD), and >= max over channels of
                     bursts x t_burst  (the data-bus floor)
```

These are lower bounds plus counts. They are not a timing prediction. Their job is to rank
allocations and to explain measured stalls. The measured-minus-bound residual is therefore an
**unmodelled residual**, not the dynamic part. It includes deterministic costs the bound leaves
out: startup and CAS latency, ACT-rate limits (tRRD, tFAW), and read/write turnaround. Only what
remains after those are added back can be attributed to refresh, interference and feedback.
Step 5 adds them as the model matures, and reports the two separately.

### 3.6 Bank-aware allocation (compiler / loader)

This is the lever. Today `device_address` comes from program-order packing. The allocator should
place tensors that are concurrently streamed so they land in different banks or bank groups, and
in different channels where the map allows. It can do this by choosing base offsets modulo the
map's bank stride, or by tiling so that a tile's bursts stripe across banks. The allocator uses
§3.5 as its cost function. The allocation is recorded in the loadable, so the decision is
inspectable and reproducible. A loader that relocates must preserve bank offsets modulo the bank
stride, or the static analysis no longer holds.

## 4. Implementation Steps

Each step is one PR and ends green.

1. **Spec and map.** (Done, PR #317; the rank field `rk` and the required capacity came from it.)
   - `deployment_spec.hpp` gains `memory.dram`, with validation (powers of two, map fields name
     every coordinate once, and `capacity_bytes` is covered by the map) and JSON in
     `src/program/deployment_json.cpp`.
   - New `include/sw/kpu/program/platform/dram_address_map.hpp` plus `src/program/dram_address_map.cpp`.
   - `tests/program/deploy/kpu_t64.json` declares the T64 DRAM.
   - Tests: encode/decode round-trip, xor folds, and every coordinate reachable.
2. **Fix the CSP-tier controller's three defects (§1.4)** by hosting the temporal
   `LPDDR5MemoryController` behind the `MemoryControllerProcess` interface (Q1, decided).
   - It brings bank groups, tRRD/tFAW, refresh and FR-FCFS.
   - The CSP side adds burst submission, per-submitter completion, `DramAddressMap` decode and a
     data bus per channel (§3.4).
   - The `patterns/memory/lpddr5/` suites remain the oracle.
3. **DMA window.** `dma_engine_process.hpp`: tile -> bursts, window `W` from the spec
   (`dma.window`), completion per burst. The L3 credit is still acquired before the first burst
   is submitted (§2), with the writeback reserve unchanged.
   Tests: one engine with `W = 32` saturates a channel, and 32 engines with `W = 1` do not.
4. **Record.** `.tflow` gains per-bank columns: busy intervals and page-conflict counts per
   bank `B = (mc, channel, rank, bank_group, bank)`. These are stations of kind `dram_bank`, which the LOD pyramid picks up
   without change. `tflow_check.py` gains TF10 (a bank never has two open rows) and TF11 (the
   data bus never carries two bursts at once).
5. **Static model.** `tools/trace/dram_conflicts.py` (or C++ under `src/program/analysis/`)
   computes §3.5 from the loadable, the schedule and the spec, and emits predicted per-bank
   load into the binding.
6. **Viewer.** The DRAM panel colors addresses by bank or channel (toggle), and each controller
   expands into per-bank swimlanes. Predicted occupancy is drawn as an outline over measured.
   `bind.py` replaces its single-outstanding overload with outstanding-burst pressure against
   `N x W`.
7. **L-T2 (#283).** The `read`/`write` vocabulary acts on bank resources
   `B = (mc, channel, rank, bank_group, bank)`, the same identity as §3.5, at burst
   granularity. Service times come from a table calibrated against step 2's controller:
   page hit, empty and conflict, by bank-group relation.
8. **Allocator.** Bank-aware placement of `device_address` in the loadable producer, behind a
   flag. A/B against program-order packing on the M2/M3 workloads. Filed as its own issue.

## 5. Verification

```bash
cmake --build --preset release
cd build && ctest -L timing --output-on-failure          # controller + DMA changes
ctest -R 'deployment|dram_address_map' --output-on-failure
ctest -R tflow --output-on-failure                        # TF10/TF11 on recorded runs
python3 ../tools/trace/test_tflow_check.py
```

Expected outcomes, each with a mutation that must make it fail:

- **Values never move.** The same program under two address maps, and under bank-aware vs
  packed allocation, produces bit-identical outputs at every level (ADR 0002: decomposition
  changes *when*, never *what*).
  - Mutation: none possible. This is the ADR's bar.
- **The data bus is a ceiling.** Sixteen streams to sixteen banks of one channel take at least
  `bytes / (channel bandwidth)`.
  - Mutation: stop reading `data_bus_ready_`. The test must fail.
- **Size matters.** A 64 KiB tile takes about 1024x the bus time of a 64 B burst.
  - Mutation: revert to size-independent latency. The test must fail.
- **Banks matter.** Two streams with different rows in one bank are measurably slower than the
  same streams in two banks of different bank groups.
  - Mutation: map every address to bank 0. The two runs then collapse to the same time, and the
    test must fail.
- **Window, not engine count.** One engine with `W >= 21` reaches more than 90% of one x16
  channel; 32 engines with `W = 1` do not.
- **L-T2 within tolerance of L-CA**, using the ADR 0001 §7.5 comparator, on the
  `patterns/memory/lpddr5` workloads and on the T64 512³ matmul.
- **Static model sound as a bound.** The predicted per-window lower bound is at most the measured
  value on every recorded run.
  - Mutation: drop the row-switch term; the bound still holds. Then mutate the address map in
    the model only; the unmodelled residual must jump.

## 6. Key Invariants

1. Values are independent of the address map, the allocation, `W`, and bank timing.
2. One address map: the controller, the record, the static model and the viewer agree on
   `decode(addr)`, checked by a shared fixture.
3. A bank has at most one open row. A channel's data bus carries at most one burst at a time.
4. The ACT rate respects tRRD_S/L and tFAW. Refresh takes the bank for tRFC.
5. A DMA engine has at most `W` bursts outstanding. It submits a tile's first burst only while
   holding that tile's L3 credit, never one from the writeback reserve, and it pushes the tile
   only once all its bursts are home.
6. Every burst belongs to exactly one tile transfer. A tile transfer completes only when all its
   bursts have.
7. The static model never claims a tighter cost than was measured.
8. No cache terms: a page hit is a row-buffer state, not a lookup that missed.

## 7. Decided on Review (2026-10-04)

All six went with the recommendation.

| # | Decision |
|---|---|
| Q1 | Host the temporal `LPDDR5MemoryController` behind the `MemoryControllerProcess` interface: one bank model, validated by `patterns/memory/lpddr5/`. Step 2 takes option (b) |
| Q2 | T64 default map: linear `co:ch:bg:ba:mc:ro` plus a 4-bit row->bank XOR fold. Both remain selectable in the spec |
| Q3 | L-T2 (#283) models DRAM per burst on bank resources `(mc, channel, rank, bank_group, bank)` |
| Q4 | The compiler owns `device_address` and records it in the loadable. The loader may relocate only modulo the bank stride |
| Q5 | HBM uses the same abstraction: pseudo-channels are channels, and `technology` selects the timing table |
| Q6 | `W` is spec-wide (`dma.window`) |

## 8. Open Questions (as posed)

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | How to fix the CSP-tier controller | (a) grow `MemoryControllerProcess`; (b) host the temporal `LPDDR5MemoryController` behind its interface | **(b)**: it already has bank groups, tFAW, refresh and FR-FCFS, and it is validated. Two bank models will drift |
| Q2 | Default address map for T64 | linear `co:ch:bg:ba:mc:ro`; with an XOR bank hash; per-vendor | linear plus a 4-bit row->bank XOR fold, which is the common LPDDR5X controller choice. Both are kept selectable, so the allocator can be studied against each |
| Q3 | L-T2 granularity | per burst; per tile with an analytic service time | per burst: #283's vocabulary is per resource, and per-tile would hide exactly the conflicts we want |
| Q4 | Who owns allocation | compiler (in the loadable); loader (at run time) | compiler decides; loader may relocate only modulo the bank stride (§3.6) |
| Q5 | HBM | same abstraction with pseudo-channels as channels; separate | same abstraction. The spec's `technology` selects the timing table |
| Q6 | `W` per engine | spec-wide; per engine | spec-wide `dma.window` first |
