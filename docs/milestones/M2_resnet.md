# DNN Milestone M2: ResNet-18 on the CSP Executor

**Status:** achieved (capability + demo). Issue #130.
**Demo:** `examples/milestones/m2_resnet.cpp` (`ctest -R m2_resnet`).
**Design:** `docs/plans/m2_resnet_dfg.md`.

---

## What M2 delivers

The first recognizable CNN — **ResNet-18** — expressed as a **dataflow graph
(DFG)** and executed end-to-end on the credit-based CSP concurrent executor, with
real fp32 values flowing DRAM → L3 → L2 → compute → … The whole network runs
through one `KernelGraph`, and its classification output is validated elementwise
against a composed host oracle.

In one artifact this exercises: the operators (conv, folded BatchNorm, ReLU,
residual add, global-average-pool, FC), the **fusions** (conv+BN fold,
conv+bias+ReLU epilogue), and the **layout/graph transformations** (im2col,
topological scheduling, residual-branch structure + fusible-pair analysis).

## Architecture: a KernelGraph DFG + a graph→CSP bridge

ResNet-18 is built as a **`KernelGraph`** (the operator DAG — `add_kernel(
Kernel::create_conv2d/batchnorm/elementwise/global_avg_pool2d/matmul)`, tensor
edges, topological order, `find_fusible_pairs`, `to_dot`). It is executed by
**`GraphCspExecutor`** (`include/sw/kpu/timing/graph/`), which walks the graph in
topological order and runs each node on the schedule-generator value path
(`csp_op_runners.hpp`), threading activations between nodes and applying the
operator fusions as it lowers:

- **conv+BN fold** — a conv's sole-consumer BatchNorm folds its scale/shift into
  the conv's weights+bias (E9); the BN node does not execute.
- **conv+ReLU fused epilogue** — a sole-consumer ReLU (directly or through the
  folded BN) applies in-compute (E10), so `conv→BN→ReLU` is a single GEMM.
- **im2col layout** — conv lowers to a GEMM over the unfolded `A_col` (E6).
- **residual add / GAP / FC** — the elementwise ADD joins the (identity or 1×1
  projected) skip; global-average-pool and FC run on the pooling/matmul value paths.

The graph gives structure, fusion detection, and visualization; the CSP path
gives oracle-validated numbers and credit/stall benchmarks. Nodes execute
**sequentially in topological order** — the measured concurrency is the tile-level
pipeline overlap *within* each op, not operator-branch overlap (a named follow-on).

## Three-tier definition of done

- **Demonstrate.** The full network runs end-to-end on the CSP executor;
  `m2_resnet --dot FILE` emits the `KernelGraph` (Graphviz).
- **Validate.** The classification output is compared elementwise against a
  composed whole-network host oracle (`conv2d_reference` + `batchnorm_reference` +
  ReLU + add + `global_avg_pool_reference` + matmul). Max error is ~3e-8 —
  fp-exact at this scale. Per-block and per-network checks:
  `test_m2_resnet_block`, `test_m2_resnet18_tower` (the `[2,2,2,2]` residual
  tower), `test_m2_resnet_head` (GAP→FC), `test_m2_resnet18` (full network).
- **Benchmark.** Cycles, CSP ops (post-fusion), cyc/op, and DMA/BlockMover/
  Streamer stall breakdown across a small spec sweep (`m2_resnet`).

### Example benchmark output (scaled demo)

```text
configuration           nodes   ops     cycles    cyc/op   dmaStl    bmStl   strStl
resnet18 (base)            39    22      39881      1813     4322     7696     2939
resnet18 [2,2,2,2]         67    38      51469      1354     7290    13548     5043
resnet18 (batch 32)        39    22      72169      3280    10226     8061     2171
```

**Fusion payoff:** the `[2,2,2,2]` network's 67 graph nodes execute as **38 CSP
ops** — every BatchNorm folded into its conv, every block-internal ReLU fused as
an epilogue. The base `[1,1,1,1]` scale runs 39 nodes as 22 ops.

## Scope & scaling notes

- **Dims are scaled for a fast CSP simulation.** The CSP executor models every
  tile movement cycle-by-cycle, so the demo uses small spatial extents and
  tile-aligned channels. **batch N=16** is required because the FC's GEMM `M` axis
  is the batch, and it must be a multiple of the tile size; the same batch keeps
  every conv GEMM's `M = N·Hout·Wout` aligned. The default demo uses uniform
  16-channel stages (the global-average-pool does `N·C` per-channel reductions, so
  its cost scales with the final channel count) at `[1,1,1,1]` depth; the full
  `[2,2,2,2]` residual depth with channel growth is validated by
  `test_m2_resnet18_tower`.
- **Weights are deterministic synthetic** (fixed-seed LCG), and the oracle is
  computed from the same weights, so validation is exact-to-fp32. Loading trained
  ResNet weights (ONNX/PyTorch) is an E15 runtime follow-on.
- **What M2 tests vs. the DFX compiler.** M2 exercises the *operator-level*
  fusions (conv+BN, conv+ReLU) + im2col + the `KernelGraph` structural passes.
  The full DFX tiling/dataflow-strategy compiler is matmul-only today (conv2d is a
  `TODO`); extending it to a whole CNN is a named follow-on, off M2's critical path.
- **Follow-ons:** ResNet-50 (bottleneck block), trained-weight loading, concurrent
  multi-node scheduling of execution-level-independent branches, a unified Chrome
  trace across the network, and the DFX conv compiler.

## Running it

```bash
cmake --build --preset release --target m2_resnet
./build/examples/milestones/m2_resnet            # benchmark table + validation
./build/examples/milestones/m2_resnet --dot resnet18.dot   # + KernelGraph
ctest -R m2_resnet                               # CI smoke test
```
