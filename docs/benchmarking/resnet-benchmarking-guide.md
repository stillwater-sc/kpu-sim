# Benchmarking ResNet on the KPU — Research Guide

**Purpose:** enable performance research on ResNet-18 running end-to-end on the KPU
CSP timing simulator — how to run it, what the numbers mean, and how to extract
**utilization**. This is a working document for the benchmarking effort, not a
milestone writeup. For the milestone framing see
[`docs/milestones/M2_resnet.md`](../milestones/M2_resnet.md); for the design see
[`docs/plans/m2_resnet_dfg.md`](../plans/m2_resnet_dfg.md).

**TL;DR of the current state:** ResNet-18 runs end-to-end, oracle-validated, and
reports **cycles / ops / stall-cycles** plus **movement-fabric utilization**
(DMA/BlockMover/Streamer busy%, tiles, effective DRAM bandwidth). Utilization is a
**directly measured** active-cycle count (each component counts the cycles a
transfer occupied it) — not the earlier `cycles − stalls/N` heuristic — so idle
cycles are excluded and the numbers are a true activity fraction. It also reports
**compute FLOP efficiency** (GEMM MACs vs a 16×16 PE-array peak, arithmetic
intensity, and roofline position — §6), which independently confirms the workload
is memory-bound. The demo prints a "utilization" table and a "compute" table.

---

## 1. What you can measure today vs. what needs a small change

| Metric | Available now? | Source |
|--------|----------------|--------|
| Total cycles (whole network) | ✅ yes | `RunStats.total_cycles` |
| CSP ops (post-fusion) | ✅ yes | `RunStats.ops` |
| Cycles/op | ✅ yes | derived |
| DMA / BlockMover / Streamer stall cycles | ✅ yes | `RunStats.{dma,bm,str}_stalls` |
| Output correctness (max err vs oracle) | ✅ yes | demo validate mode |
| **DMA/BM/Streamer utilization (busy/total)** | ✅ yes | `RunStats.{dma,bm,str}_utilization()` |
| **Tiles moved/fed/loaded/stored** | ✅ yes | `RunStats.tiles_*` |
| **Effective DRAM bandwidth (GB/s)** | ✅ yes | `RunStats.effective_{load,store}_bandwidth(ghz)` |
| Compute-fabric utilization / FLOP efficiency | ❌ not in this path | see §6 caveat |

The utilization/tiles/bandwidth fields are aggregated from
`ConcurrentTimingExecutor::get_statistics()` into `RunStats` by
`detail::accumulate` (`csp_op_runners.hpp`) and printed by the demo's utilization
table. See §5 for a sample and §6 for how the metric is derived and its limits.

---

## 2. Assets — what exists and where

### Model builder — `include/sw/kpu/timing/graph/resnet18.hpp`

Header-only. Namespace `sw::kpu::timing::graph`.

```cpp
[[nodiscard]] ResNet18 build_resnet18(KernelGraph& g, const ResNet18Spec& sp);
```

Populates the caller's `KernelGraph` with the operator DAG (stem conv→BN→ReLU;
four stages of BasicBlocks with stride-2 downsample + 1×1 projection skip on the
first block of stages 2–4; global-average-pool → FC head) and returns
weights + synthetic input + a host oracle.

`ResNet18Spec` (all fields have defaults):

| Field | Default | Meaning |
|-------|---------|---------|
| `batch` | 16 | batch N (**must be a multiple of `tile`** — it is the FC GEMM's M axis) |
| `in_channels` | 16 | stem input channels |
| `height`, `width` | 4, 4 | stem input spatial extent |
| `stage_channels` | `{16,16,16,16}` | per-stage channel count (each a `tile` multiple) |
| `blocks_per_stage` | 1 | block depth; real ResNet-18 is `[2,2,2,2]` |
| `num_classes` | 16 | FC output width |
| `tile` | 16 | GEMM tile; batch/channels/classes must be multiples |
| `eps` | 1e-3 | BatchNorm epsilon |
| `seed` | 1000 | LCG seed for synthetic weights |

Returns `ResNet18`: `node_data` (per-node weights/BN/FC params), `input` (NCHW),
`oracle` (`[batch, num_classes]`), `output_node`, `num_nodes`. Validates dims and
throws `std::invalid_argument` on non-tile-aligned specs.

### Demo — `examples/milestones/m2_resnet.cpp` (target `m2_resnet`)

Runs three modes every invocation: **demonstrate** (end-to-end on the CSP
executor; `--dot FILE` emits Graphviz), **validate** (elementwise vs oracle, PASS
if `max_err < 5e-3`), **benchmark** (a 3-row spec sweep). Flags: `--dot FILE`,
`--help`. Registered as ctest `m2_resnet` (labels `timing;resnet;milestone;v0.9`).

### Bridge executor — `include/sw/kpu/timing/graph/graph_csp_executor.hpp`

```cpp
GraphCspExecutor exec;
Result r = exec.run(g, net.input, net.node_data, /*T=*/16);
// r.output : sink tensor (the classification logits)
// r.stats  : RunStats { total_cycles, dma_stalls, bm_stalls, str_stalls, ops }
```

Walks the graph in topological order, applies conv+BN fold and conv+ReLU-epilogue
fusion during lowering, and runs each node on the value path. `T` is the GEMM tile
size and must divide every conv/matmul node's M/N/K.

### Timing counters — `include/sw/kpu/timing/concurrent_timing_executor.hpp`

`get_statistics()` returns a `Statistics` struct (lines ~170–222) with everything
needed for a utilization study: `total_cycles`; `{dma,bm,str}_busy_cycles`; the
stall breakdown; `tiles_{loaded,stored,moved,writeback,fed,drained}`;
`bytes_{loaded,stored}`; and the derived helpers `dma_utilization()`,
`bm_utilization()`, `str_utilization()`, `effective_{load,store}_bandwidth(ghz)`.

### Occupancy observer — `include/sw/kpu/timing/tile_tracker.hpp`

`TileTracker` renders horizontal L3 | L2 | L1/array occupancy bands over time via
`ConcurrentTimingExecutor::tiles_at(MemoryLevel)`. Use it to see *where* tiles pile
up during a forward pass (which buffer level is the bottleneck), complementing the
scalar utilization numbers.

---

## 3. Assumptions & scaling model — read before interpreting any number

The demo runs a **structurally faithful but dimensionally scaled** ResNet-18. The
CSP executor models every tile movement cycle-by-cycle, so a full-resolution
224×224 network would be minutes-to-hours per pass. What is preserved vs. scaled:

- **Preserved:** the full operator graph and topology — stem, four residual
  stages, downsample + 1×1 projection skips, GAP→FC; all fusions (conv+BN,
  conv+ReLU); im2col conv lowering; fp32 values flowing DRAM→L3→L2→compute with an
  exact oracle.
- **Scaled:** small spatial extents (default 4×4), uniform 16-channel stages, and
  `blocks_per_stage=1` by default. The real `[2,2,2,2]` depth is exercised by
  `test_m2_resnet18_tower` and by the demo's second sweep row.

Consequences for benchmarking:

1. **Absolute cycle counts are not silicon-representative.** They are self-
   consistent within the simulator's cost model and valid for *relative*
   comparisons (config A vs. B, before/after a scheduler change), not for
   projecting wall-clock throughput of a real 224×224 ResNet.
2. **`batch = 16` is a hard constraint of the scaled demo** (FC GEMM M axis must be
   tile-aligned). Batch sweeps must stay tile-multiples (16, 32, …).
3. **Nodes execute sequentially in topological order.** Measured concurrency is the
   tile-level pipeline overlap *within* each op — not operator-branch overlap. So
   utilization here answers "how busy is each mover *while an op runs*," not "how
   well are independent branches overlapped." A mover reading 80%+ busy therefore
   means it is well-fed within ops; it does **not** account for cross-branch overlap
   the sequential walk leaves on the table. Concurrent branch scheduling is a named
   follow-on (§7).
4. **Weights are deterministic synthetic**, oracle from the same weights → exact
   validation. Trained-weight loading (ONNX/PyTorch) is an E15 follow-on; it does
   not change timing, only the values.

---

## 4. How to build and run

```bash
# Build the demo
cmake --build --preset release --target m2_resnet

# Run: prints the benchmark table + validation status
./build/examples/milestones/m2_resnet

# Emit the KernelGraph for inspection
./build/examples/milestones/m2_resnet --dot resnet18.dot
dot -Tpng resnet18.dot -o resnet18.png     # optional, needs graphviz

# CI smoke test
ctest -R m2_resnet
```

To sweep configurations beyond the three built-in rows, add rows to the sweep
table in `m2_resnet.cpp` (each row is a `ResNet18Spec`), or write a small driver
that constructs specs and calls `build_resnet18` + `GraphCspExecutor::run`
directly. Any new spec must keep batch/channels/classes as `tile` multiples.

---

## 5. Current baseline results (scaled demo)

```text
  configuration           nodes   ops     cycles    cyc/op   dmaStl    bmStl   strStl     maxErr  check
  resnet18 (base)            39    22      39881      1813     7016    99797    26511    6.0e-08   PASS
  resnet18 [2,2,2,2]         67    38      51469      1354     9984   108557    29951    6.0e-08   PASS
  resnet18 (batch 32)        39    22      72169      3280    16430   196470    49551    6.0e-08   PASS

  utilization              dmaU%    bmU%   strU%  tilesLd  tilesMv  tilesFd    ldGB/s    stGB/s
  resnet18 (base)           84.6    14.1    10.7     1805     1438     1438      18.5       2.9
  resnet18 [2,2,2,2]        77.9    18.4    13.9     2773     2318     2318      25.4       4.0
  resnet18 (batch 32)       92.3    14.9    11.8     3610     2876     2876      19.3       3.2

  compute                    MFLOP   GFLOP/s  peakEff%   AI(F/B)  roofEff%   bound
  resnet18 (base)             4.48     112.4      21.9      5.25      33.4     mem
  resnet18 [2,2,2,2]          7.73     150.1      29.3      5.11      45.9     mem
  resnet18 (batch 32)         8.96     124.2      24.3      5.54      35.1     mem
```

Reading it:
- **Fusion payoff:** `[2,2,2,2]` runs 67 graph nodes as **38 CSP ops** (every BN
  folded, every block-internal ReLU fused). Base `[1,1,1,1]` = 39 nodes → 22 ops.
- **DMA (DRAM→L3) is the near-saturated bottleneck** — 78–92% active, and **batch
  32 drives it to 92%** (the most DRAM-bound config). At this scale the network is
  bandwidth-bound on the DRAM load path.
- **The on-chip movers starve behind it:** BlockMover 14–18%, Streamer 11–14%
  active. They are not the throughput limiter — they spend most cycles waiting for
  the DMA to deliver tiles (note `bmStl`/`strStl` are large: much *waiting*, little
  *transferring*). Widening DRAM bandwidth / adding DMA engines is the lever, not
  more on-chip buffering.
- **Deeper (`[2,2,2,2]`)** lifts on-chip utilization slightly (more work amortizes
  each load) and eases DMA to 78%.
- **`ldGB/s`/`stGB/s`** are at an assumed 1.0 GHz clock (so GB/s == bytes/cycle);
  the unit is a knob, the *ratio* between configs is the signal.
- **cyc/op is not efficiency** — it is total cycles / op count, inflated by stalls.
- **Compute confirms it from the other side:** arithmetic intensity ≈ 5.2 FLOP/byte
  is below the 8 FLOP/byte ridge, so every config is **memory-bound** (`bound=mem`),
  reaching only ~22–29% of the 16×16 array's peak. Full detail in §6.

One accounting note so the columns are not misread:
- **Stall columns are summed across all parallel components and all ops**, so a
  value can exceed `cycles` (e.g. `bmStl` 99797 > 39881 with ~4 BlockMovers). They
  are a workload stall total, not a per-component fraction — high `bmStl` with low
  `bmU%` means the BlockMovers *wait* a lot but rarely *transfer*, consistent with
  starving behind the DMA.

---

## 6. Measuring utilization — how it works and what it means

**Status: implemented.** `detail::accumulate` (`csp_op_runners.hpp`) sums the
executor's per-op `get_statistics()` — `{dma,bm,str}_busy_cycles`, `tiles_*`,
`bytes_*` — into `RunStats`, which exposes:

```cpp
r.stats.dma_utilization();   // 0.0–1.0 = Σ dma_busy / Σ total_cycles (L3 push)
r.stats.bm_utilization();    //           L3→L2 BlockMover
r.stats.str_utilization();   //           L2→L1 Streamer
r.stats.tiles_loaded; r.stats.tiles_moved; r.stats.tiles_fed;
r.stats.effective_load_bandwidth(clock_ghz);   // GB/s
r.stats.effective_store_bandwidth(clock_ghz);
```

The demo prints these as the "utilization" table. Because nodes run sequentially on
a fresh `ConcurrentTimingExecutor` per op, the per-op `Statistics` are additive and
the whole-network ratio is `Σ busy / Σ total_cycles`.

### How `busy` is measured — directly, not derived

`busy` is a **directly measured active-cycle count**. Every cycle, in each
component's `tick()`, the component increments an `active_cycles_` counter iff a
transfer actually occupied it that cycle:

- **DMA** — a request is `SUBMITTED` (the memory controller is transferring on its
  behalf).
- **BlockMover / Streamer** — an `InFlightTransfer` occupies the cycle
  (`in_flight_.has_value()`). Zero-latency dedup moves and the drain compute-wait do
  **not** count.

`collect_statistics` sums these across the N parallel components and divides by N
(in floating point, so the per-component mean is exact — not integer-floored), so
`busy` is the mean active cycles per component and `utilization = busy /
total_cycles ∈ [0,1]` is the mean fraction of time a component of that type was
transferring. Only WORKING cycles count — the former `cycles − stalls/N` heuristic
counted IDLE as busy, inflating and mis-ranking the movers; a directly-measured
counter was follow-on 1b, now done.

For the single-transfer **BlockMover/Streamer** the per-cycle outcome is exactly one
of WORKING / STALLED / IDLE (`active` and `stall` cannot both fire in a tick). The
**DMA holds multiple requests**, so a tick can *both* count active (one request
`SUBMITTED`) *and* record a stall (another request waiting on an L3 credit) — for
the DMA, `active` and `stall` may overlap in a cycle. `active` still never exceeds
`total_cycles` (one increment per tick), so utilization stays in `[0,1]`.

**Stall columns are still un-normalized** (summed over all N and all ops), which is
why `bmStl` (99797, ~4 movers) exceeds `cycles` (39881). Utilization normalizes;
the stall column does not. High `bmStl` **with** low `bmU%` is the signature of a
stage that waits a lot but rarely transfers — i.e. starves behind an upstream
bottleneck (here the DMA).

**Regression check:** `tests/timing/test_resnet_utilization.cpp` asserts, per mover,
`0 < busy ≤ total_cycles` (util in `(0,1]`), that not every mover is pinned at 100%
(idle is genuinely excluded), and accessor consistency `util == busy/total`. A
component whose counter was never wired into its `tick()` would read a flat `busy=0`
and fail.

Practical guidance:
- Utilization is now a **measured activity fraction**, usable as an absolute (with
  the caveat that it is a per-component mean and, under the sequential-node walk,
  excludes cross-branch overlap — see §3).
- The signal is the **DMA near-saturation (78–92%) vs. starved on-chip movers
  (11–18%)**: the workload is DRAM-bandwidth-bound at this scale.

### Compute FLOP efficiency (roofline position)

`RunStats` now also carries the **compute** side. During the graph walk the GEMM
ops accumulate `total_macs` (`conv.gemm_M·gemm_N·gemm_K`, which handles `groups`,
plus `fc_M·fc_N·fc_K`); fused BatchNorm, ReLU epilogues, residual adds, and pooling
carry no GEMM and contribute nothing. From that:

```cpp
r.stats.total_flops();                       // 2 * total_macs
r.stats.achieved_gflops(clock_ghz);          // FLOP/cycle * clock
r.stats.arithmetic_intensity();              // FLOPs per DRAM byte
r.stats.compute_efficiency(512.0);           // achieved / peak FLOP/cycle
r.stats.roofline_efficiency(512.0, 64.0, 1.0); // achieved / min(AI*bw, peak)
```

Conventions match the matmul `kpu-benchmark` harness (`sw::benchmark::HardwareSpec`):
a **16×16 PE array, 2 FLOP/MAC → 512 GFLOP/s peak at 1 GHz**, **64 GB/s** external
DRAM, ridge point `512/64 = 8` FLOP/byte. The demo prints a compute table:

```text
  compute                    MFLOP   GFLOP/s  peakEff%   AI(F/B)  roofEff%   bound
  resnet18 (base)             4.48     112.4      21.9      5.25      33.4     mem
  resnet18 [2,2,2,2]          7.73     150.1      29.3      5.11      45.9     mem
  resnet18 (batch 32)         8.96     124.2      24.3      5.54      35.1     mem
```

Reading it, and how it corroborates the movement story:
- **Arithmetic intensity ≈ 5.2 FLOP/byte < the 8 ridge → memory-bound.** This is the
  *same* conclusion the movement fabric gave (DMA at 78–92%), now from the compute
  side: the scaled network does too little arithmetic per DRAM byte to saturate the
  PE array.
- **peakEff 22–29%** — only about a quarter of the 16×16 array's peak is reached,
  because compute is starved behind DRAM.
- **roofEff 33–46%** — even against the *memory* ceiling (AI·64) there is headroom:
  movement stalls and the sequential-node walk keep it below the attainable limit.
- **Deeper `[2,2,2,2]` is best** on both (29% / 46%): more arithmetic amortizes each
  load, nudging toward the ridge.
- These are **relative** numbers on a scaled topology (§3) — the *shape* (memory-
  bound, quarter-peak) is the signal, not the absolute GFLOP/s.

Follow-ons: a full-resolution ResNet (higher AI, likely still memory-bound at this
buffer sizing) and a per-layer AI breakdown to find which convs are compute- vs
memory-bound.

---

## 7. Research roadmap

Ordered by dependency:

1. ~~**Surface movement utilization**~~ — **done** (§6): the demo prints per-mover
   `busy/total`, tiles, and effective bandwidth.
1b. ~~**Refine `busy` to a directly-measured active-cycle count**~~ — **done**:
   each component now counts, in its `tick()`, the cycles a transfer occupied it, so
   utilization excludes idle cycles and is a measured activity fraction (§6).
2. ~~**Compute FLOP efficiency / roofline position**~~ — **done** (§6): the demo
   prints MFLOP, GFLOP/s, peak efficiency, arithmetic intensity, and roofline
   position; the network is memory-bound (AI ≈ 5.2 < 8), corroborating §5.
3. **Occupancy timeline** — wire `TileTracker` into the demo (behind a flag) to see
   which buffer level saturates during the forward pass. **Now the top open task.**
4. **Full-scale ResNet-18** — larger spatial dims + `[2,2,2,2]` + channel growth as
   a benchmark spec (slow; run offline, not in CI). Needed before any number is
   quoted as representative.
5. **Concurrent branch scheduling** — the sequential-node assumption caps
   achievable utilization; overlapping execution-level-independent branches is the
   architectural lever the utilization study will eventually motivate.
6. **JSON output + regression baseline** — emit the same schema as
   `tests/benchmarks/baselines/baseline.json` and add ResNet to the
   `benchmark-regression` workflow so utilization is tracked over time.

---

## 8. Pointers

| Asset | Path |
|-------|------|
| Model builder | `include/sw/kpu/timing/graph/resnet18.hpp` |
| Demo | `examples/milestones/m2_resnet.cpp` |
| Bridge executor | `include/sw/kpu/timing/graph/graph_csp_executor.hpp` |
| Value-path runners + `RunStats` | `include/sw/kpu/timing/graph/csp_op_runners.hpp` |
| Timing counters (`Statistics`) | `include/sw/kpu/timing/concurrent_timing_executor.hpp` |
| Occupancy observer | `include/sw/kpu/timing/tile_tracker.hpp` |
| Milestone writeup | `docs/milestones/M2_resnet.md` |
| Design plan | `docs/plans/m2_resnet_dfg.md` |
| Validation tests | `tests/timing/test_m2_resnet*.cpp` |
| Matmul benchmark harness (utilization reference) | `tools/benchmark/kpu-benchmark/` |
