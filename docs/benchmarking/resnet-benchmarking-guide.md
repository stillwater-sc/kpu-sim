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
is memory-bound. The demo prints a "utilization" table and a "compute" table,
`--occupancy` renders a `TileTracker` L3|L2|L1/array buffer-occupancy timeline, and
`--full` runs a channel-growing `[2,2,2,2]` config offline to confirm the findings
survive realistic scale (§6). A "concurrency" table reports the idealized
branch-overlap critical path — for ResNet the headroom is ≤ 2%, so the
sequential-node execution model costs almost nothing (§6). `--json` exports the
sweep, and a committed baseline + the `resnet_regression` ctest track every metric
against drift in CI (§6).

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

# Buffer-occupancy timeline (L3|L2|L1/array bands + peak occupancy per level)
./build/examples/milestones/m2_resnet --occupancy

# Representative-scale offline run: channel growth + [2,2,2,2] depth (~50 s)
./build/examples/milestones/m2_resnet --full

# Machine-readable sweep (for the regression baseline + external tooling)
./build/examples/milestones/m2_resnet --json

# CI smoke test + regression check (default sweep; --full/--occupancy are not in CI)
ctest -R "m2_resnet|resnet_regression"

# Regenerate the committed baseline after an intentional timing change
python3 scripts/resnet_regression_check.py generate \
  --binary ./build/examples/milestones/m2_resnet \
  --baseline tests/benchmarks/resnet_baseline.json
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

### Occupancy timeline — which buffer level saturates

`m2_resnet --occupancy` runs a representative layer (a 1×1 projection conv) one
executor cycle at a time and renders, via `TileTracker`, the horizontal
**L3 | L2 | L1/array** occupancy bands — one row per occupancy *transition* (idle
cycles are elided), `*` marking a tile in the compute array. You watch A/B tiles
arrive at L3, move to L2, feed L1/array, the computes produce C tiles, and those
drain back out to DRAM — all in credit-managed **buffers** (arrive / resident /
credit-returned, never hit/miss/evict). It closes with a **peak simultaneous
occupancy** line per level against that level's capacity:

```text
  peak simultaneous occupancy:  L3 4/32   L2 2/64   L1 6   array 6
```

The headline: **no buffer level saturates** — peak L3/L2 occupancy sits far below
the credit counts. So buffers are *not* the binding resource at this scale; the
constraint is DMA throughput, exactly the §5 diagnosis from the other two angles. A
peak *at* capacity would flag that level as the binding buffer (the first place to
add credits) — a check to re-run once the network is scaled up (task 4).

### Representative-scale offline run (`--occupancy`'s bigger sibling, `--full`)

The §5 tables use a deliberately tiny topology (4×4, uniform 16 channels) so the
demo is CI-fast. Before quoting any number as representative you must check the
findings survive **channel growth and real depth**. `m2_resnet --full` runs one
offline config — batch 16, stem 16ch 8×8, stages **{16, 32, 64, 128} ×2** (the true
`[2,2,2,2]` depth with stride-2 downsampling + 1×1 projections) — in ≈ 50 s:

```text
  configuration       nodes  ops   cycles    cyc/op   dmaStl    bmStl   strStl    maxErr  check
  resnet18 (full)        67   38   393927    10366   161274  1019258   244014   1.1e-06  PASS
  utilization          dmaU%  bmU%  strU%  ...  ldGB/s  stGB/s
  resnet18 (full)       84.5  15.7  16.1   ...   21.8    2.8
  compute              MFLOP  GFLOP/s  peakEff%  AI(F/B)  roofEff%  bound
  resnet18 (full)      73.99   187.8    36.7     7.62     38.5     mem
```

What scaling up changes — and what it doesn't:
- **Arithmetic intensity rises 5.2 → 7.6 FLOP/byte**, right up against the 8 ridge:
  more channels means more FLOPs reused per DRAM byte. **Compute efficiency climbs
  too, 22–29% → 37%.**
- **But it is still `mem`-bound and still DMA-saturated (84.5%)** — the memory-bound
  diagnosis *holds* at realistic proportions; scale moves it toward the ridge, not
  across it. That is the answer task 4 exists to give.
- Not full 224×224 (intractable cycle-by-cycle: a stage-4 3×3 conv on 512 channels
  is K = 4608, hundreds of K-slices per output tile). `--full` trades absolute size
  for the *structure* that matters — growth + depth + downsampling.

### Concurrency headroom — how much would branch overlap buy?

Nodes execute strictly sequentially today (§3), so a fair question is how much the
sequential-node assumption itself costs. The demo now answers it with an idealized
**critical-path** model: give each executed node its measured cycle cost, then
compute `finish[n] = cost[n] + max over predecessors of finish[p]` — so
execution-level-independent branches (a residual's 1×1 projection skip vs. its main
`conv→conv` path) *overlap* (a `max`, not a sum), with **unbounded resources**. The
critical path is the resource-free **upper bound** on what concurrent branch
scheduling could achieve:

```text
  concurrency          seqCyc   critCyc   ovlp x
  resnet18 (base)       39881    39119     1.02
  resnet18 [2,2,2,2]    51469    50707     1.02
  resnet18 (batch 32)   72169    71333     1.01
```

The headline: **≤ 2%.** ResNet's DAG is essentially a chain — its only parallelism
is the short 1×1 projection skips in the three downsampling blocks, and those convs
are so cheap they hide almost entirely behind the main path even when *serial*. So
the sequential-node assumption is **not** leaving meaningful performance on the
table for this workload, and true concurrent multi-op execution (a large executor
rewrite) is not worth building for ResNet — the DMA-bandwidth lever from §5 remains
the only one that matters. (The model is validated in
`test_resnet_utilization.cpp`: a single-node graph is a chain, `critCyc == seqCyc`;
the ResNet graph has branches, `critCyc < seqCyc`.)

### JSON export + CI regression tracking

`m2_resnet --json` emits the sweep as machine-readable JSON — per config: `config`,
`timing` (cycles, critical path, ops, nodes), `stalls`, `throughput` (tiles/bytes),
`utilization`, `compute` (macs, GFLOP/s, efficiencies, AI), `validation`. The sweep
is **deterministic** (fixed-seed synthetic weights), so integer metrics reproduce
exactly run-to-run and across platforms (integer schedule logic); the derived floats
are IEEE-deterministic to well within `1e-6`.

That determinism makes a tight regression check possible.
`tests/benchmarks/resnet_baseline.json` is the committed snapshot;
`scripts/resnet_regression_check.py check --binary <m2_resnet> --baseline <file>`
diffs a fresh run against it — **integer metrics exact, floats within `1e-6`** — and
exits non-zero on any drift. It is wired as the **`resnet_regression` ctest**, so it
runs in the standard multi-platform CI on every PR (a ctest rather than a graft into
the matmul-specific `benchmark-regression` workflow: it gets all-platform coverage
on every change and avoids that workflow's artifact-baseline machinery). `max_err`
is excluded from the diff — it is the one non-deterministic field (fp reduction
order / FMA differ across compilers); correctness is enforced separately by the
`m2_resnet` PASS check. If a change intentionally moves the numbers, regenerate:
`resnet_regression_check.py generate ...`.

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
3. ~~**Occupancy timeline**~~ — **done** (§6): `m2_resnet --occupancy` renders the
   `TileTracker` L3|L2|L1/array bands for a representative conv plus peak occupancy
   per level. Peaks sit far below capacity (L3 4/32, L2 2/64) — buffers are **not**
   the bottleneck, corroborating the DMA-bandwidth-bound finding.
4. ~~**Representative-scale ResNet-18**~~ — **done** (§6): `m2_resnet --full` runs a
   `{16,32,64,128}×2` channel-growing `[2,2,2,2]` config offline (~50 s, not in CI).
   Finding: AI rises 5.2 → 7.6 (toward the ridge) and peak efficiency 22–29% → 37%,
   but it stays memory-bound and DMA-saturated — the diagnosis holds at scale.
   Remaining: a true full-resolution run (needs a native grouped/large-K schedule to
   be tractable) and a per-layer AI breakdown.
5. ~~**Concurrent branch scheduling (headroom analysis)**~~ — **done** (§6): the
   demo prints the idealized branch-overlap critical path (`seqCyc`/`critCyc`/`ovlp`).
   Finding: **≤ 2%** — ResNet's DAG is essentially a chain, so the sequential-node
   assumption costs almost nothing and true concurrent multi-op execution (a large
   executor rewrite) is **not worth building** for this workload. The DMA-bandwidth
   lever remains the only one that matters.
6. ~~**JSON output + regression baseline**~~ — **done** (§6): `m2_resnet --json`
   emits the sweep as structured JSON; a committed baseline
   (`tests/benchmarks/resnet_baseline.json`) + `scripts/resnet_regression_check.py`
   are wired as the `resnet_regression` ctest, so every metric is diffed against the
   baseline on every CI run (all platforms). Deterministic integer metrics must match
   exactly; floats within `1e-6`.

**All roadmap items (1–6) are complete.** Open follow-ons carried in the items above:
a true full-resolution run (needs a native large-K/grouped schedule), a per-layer AI
breakdown, and — only if a non-ResNet workload ever motivates it — real concurrent
multi-op execution. For ResNet the study's conclusion is settled and triangulated
from five angles: **the workload is DRAM-bandwidth-bound; nothing else binds.**

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
| Metric + concurrency tests | `tests/timing/test_resnet_utilization.cpp` |
| Regression baseline | `tests/benchmarks/resnet_baseline.json` |
| Regression check script | `scripts/resnet_regression_check.py` |
| Matmul benchmark harness (utilization reference) | `tools/benchmark/kpu-benchmark/` |
