# Benchmarking ResNet on the KPU — Research Guide

**Purpose:** enable performance research on ResNet-18 running end-to-end on the KPU
CSP timing simulator — how to run it, what the numbers mean, and how to extract
**utilization**. This is a working document for the benchmarking effort, not a
milestone writeup. For the milestone framing see
[`docs/milestones/M2_resnet.md`](../milestones/M2_resnet.md); for the design see
[`docs/plans/m2_resnet_dfg.md`](../plans/m2_resnet_dfg.md).

**TL;DR of the current state:** ResNet-18 runs end-to-end, oracle-validated, and
reports **cycles / ops / stall-cycles** plus **movement-fabric utilization**
(DMA/BlockMover/Streamer busy%, tiles, effective DRAM bandwidth). The utilization
plumbing described in §6 is **implemented** — the demo prints a second
"utilization" table. What is *not* yet available is compute-fabric FLOP efficiency
(§6 caveat) — that is the next research task.

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
  resnet18 (base)           82.4    37.5    83.4     1805     1438     1438      18.5       2.9
  resnet18 [2,2,2,2]        80.6    47.3    85.5     2773     2318     2318      25.4       4.0
  resnet18 (batch 32)       77.2    32.0    82.9     3610     2876     2876      19.3       3.2
```

Reading it:
- **Fusion payoff:** `[2,2,2,2]` runs 67 graph nodes as **38 CSP ops** (every BN
  folded, every block-internal ReLU fused). Base `[1,1,1,1]` = 39 nodes → 22 ops.
- **BlockMover (L3→L2) is the bottleneck stage.** It carries by far the most stall
  cycles (`bmStl` ≫ `strStl` > `dmaStl`) *and* the lowest utilization (32–47% vs.
  77–85% for DMA and Streamer) — the two views agree: the L3→L2 move is where the
  pipeline waits. It is the first place to add buffering / rebalance credits.
- **DMA and Streamer run 77–85% busy;** deeper (`[2,2,2,2]`) lifts BlockMover
  utilization (47%) as more work amortizes the moves, while batch 32 pushes it
  *down* (32%) — more per-move DRAM traffic, more waiting.
- **`ldGB/s`/`stGB/s`** are at an assumed 1.0 GHz clock (so GB/s == bytes/cycle);
  the unit is a knob, the *ratio* between configs is the signal.
- **cyc/op is not efficiency** — it is total cycles / op count, inflated by stalls.

Two accounting notes so the columns are not misread:
- **Stall columns are summed across all parallel components and all ops**, so a
  value can exceed `cycles` (e.g. `bmStl` 99797 > 39881 with ~4 BlockMovers). They
  are a workload stall total, not a per-component fraction.
- **Utilization normalizes per component** (`busy = cycles − stalls/N`), so it is
  *not* `1 − stall/cycles` on the raw columns. The two are consistent once you
  divide the stall column by the component count (see §6).

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

### How `busy` is derived — and why utilization ≠ 1 − stall/total

`busy` is **not** `cycles − stall column`. The executor derives it inside
`collect_statistics` (`concurrent_timing_executor.hpp`): per component type it
averages stalls across the **N parallel components**, subtracts that from the op's
`total_cycles`, and clamps to zero if the averaged stall exceeds the op span. So
`busy = cycles − stalls/N`, and the printed stall column is the *un-normalized*
`stalls` (summed over all N and all ops) — which is why `bmStl` (99797, ~4
movers) dwarfs `cycles` (39881) while `bmU` is a sane 37.5%. Divide the stall
column by the component count before comparing it to a utilization.

**Consistency invariant (regression-checked):** because `busy ≥ cycles − stalls`
per op, the aggregate satisfies `busy + stalls ≥ cycles` for every mover. This is
exactly what the earlier, buggy version violated — some ops added `cycles` to the
denominator without adding their `busy`/`stalls`, understating utilization to
≈12–39%. `tests/timing/test_resnet_utilization.cpp` asserts the invariant so that
regression cannot return.

Practical guidance:
- Treat utilization as the **executor's own relative metric**, good for A/B
  comparison across configs and before/after a scheduler change — **not** a
  validated absolute "fraction of peak." Refining `busy` to a directly-measured
  active-cycle count (instead of `cycles − stalls/N`) is follow-on §7 item 1b.
- The signal in the current numbers is the **spread between movers** (BlockMover
  32–47% vs. DMA/Streamer 77–85%), which localizes the L3→L2 bottleneck — not the
  absolute level.

### Caveat: compute-fabric utilization is not in this path

`Statistics` covers the **movement** fabric (DMA / BlockMover / Streamer) and DRAM
bandwidth. It does **not** carry a compute-fabric FLOP-efficiency number the way
the standalone `kpu-benchmark` matmul harness does (`tools/benchmark/`, whose
`baseline.json` has `utilization.compute` and `gflops`/`efficiency`). If the study
needs "what fraction of peak MACs are we hitting on ResNet's convs," that is a
separate metric: compute the network's total MACs (known from the graph) and divide
by `total_cycles × peak_MACs_per_cycle`. That is research task #2, and it belongs
either in the demo or in a ResNet row added to the `kpu-benchmark` harness.

---

## 7. Research roadmap

Ordered by dependency:

1. ~~**Surface movement utilization**~~ — **done** (§6): the demo prints per-mover
   `busy/total`, tiles, and effective bandwidth.
1b. **Validate/refine the `busy` derivation** — the current busy is
   `total − avg_stall` with clamping (§6), which is not a measured active-cycle
   count. Add a direct active-cycle counter per component so utilization becomes a
   defensible absolute, not just a relative metric.
2. **Compute utilization / FLOP efficiency** — MACs / (`total_cycles` × peak). Lets
   you say "roofline position" for each config.
3. **Occupancy timeline** — wire `TileTracker` into the demo (behind a flag) to see
   which buffer level saturates during the forward pass.
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
