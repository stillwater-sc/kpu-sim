# DNN Milestone M3: MobileNetV2 on the CSP Executor

**Status:** MobileNetV2 achieved (capability + demo). Issue #131.
**Demo:** `examples/milestones/m3_mobilenet.cpp` (`ctest -R m3_mobilenet`).
**Design:** `docs/plans/m3_mobilenet_dfg.md`. **Template:** M2 (`docs/milestones/M2_resnet.md`).

---

## What M3 delivers

**MobileNetV2** — the inverted-residual, depthwise-separable CNN — expressed as a
**dataflow graph (DFG)** and executed end-to-end on the credit-based CSP
concurrent executor, with real fp32 values flowing DRAM → L3 → L2 → compute → …
The whole network runs through one `KernelGraph`, and its classification output
is validated elementwise against a composed host oracle.

Beyond M2's operator set, M3 adds the two things that define MobileNetV2:

- **Depthwise convolution** (`groups = Cin`) — each output channel is the 2D conv
  of a *single* input channel with its own `Kh×Kw` filter. It lowers not through
  im2col+GEMM (which assumes a full cross-channel reduction) but through the E7
  **pooling-window unfold + a per-channel filter dot-product reduce**
  (`run_depthwise_conv`), the M3 design's key decision — a Vector-Engine op, no
  new systolic kernel.
- **ReLU6** (`min(max(x, 0), 6)`) — MobileNetV2's activation, a first-class typed
  elementwise op (`ElementwiseOp::RELU6`) on the CSP path.

Both were delivered in M3-T2/T3; M3-T4 (this milestone) assembles them into the
**full network** and its demonstrate / validate / benchmark artifact.

## Architecture: a KernelGraph DFG + the M2/M3 bridge

MobileNetV2 is built by `build_mobilenetv2` (`include/sw/kpu/timing/graph/
mobilenetv2.hpp`) as a **`KernelGraph`** and executed by **`GraphCspExecutor`**,
reusing the M2 bridge unchanged and the M3 depthwise/ReLU6 dispatch:

- **stem** — `3×3 s1 conv → BN → ReLU6`.
- **inverted-residual bottleneck stack** — each block is `1×1 expand → BN → ReLU6
  → 3×3 depthwise (stride s) → BN → ReLU6 → 1×1 project → BN`, with an **identity
  residual** (an explicit graph edge, so blocks thread correctly through the
  stack) when `stride == 1 && Cin == Cout`. As in real MobileNetV2, the `t == 1`
  bottleneck omits the expansion conv and feeds the input straight into the
  depthwise. The project is a *linear* bottleneck
  (no activation), and there is **no activation after the residual add** — the
  MobileNetV2 distinction from ResNet.
- **head** — `1×1 conv → BN → ReLU6 → global-average-pool → FC`.

As it lowers, the bridge folds each BatchNorm into its conv, dispatches pointwise
`1×1` convs to the im2col→GEMM path and depthwise convs (`groups == Cin`) to the
pooling-window path, runs ReLU6 as a standalone Vector-Engine clamp, and joins the
identity residuals with an elementwise ADD.

## Definition of done

- **Demonstrate** — the whole network runs end-to-end on the CSP executor;
  `--dot FILE` emits the `KernelGraph` for visualization.
- **Validate** — the classification output is compared elementwise against a
  composed whole-network host oracle (pointwise conv / depthwise conv / BN /
  ReLU6 / add / GAP / FC references), `max_err < 5e-3` (observed ~1e-7).
- **Benchmark** — cycles, CSP ops (post-fusion), cyc/op, and DMA/BM/STR stall
  breakdown across a small spec sweep (base / an extra bottleneck stage / batch
  32).

## Scale and honest scope

Dimensions are scaled for a fast CSP simulation (as M2), preserving the
tile-alignment discipline: batch `N` (the FC/conv `M` axis) and all channel counts
are multiples of the tile size. The default topology is a compact but complete
MobileNetV2 — expansion, depthwise, a stride-2 downsample, identity residuals, the
`1×1` head, GAP and FC — kept small so the tile-by-tile CSP simulation stays fast
(the global-average-pool's `N·C` per-channel reductions dominate wall-clock at low
spatial, so the head width is kept modest; channel/depth scaling is exercised by
the sweep). The DFX tiling/dataflow compiler remains matmul-only (unchanged from
M2) — M3 tests operator-level fusions + the depthwise pooling-window reuse, not a
full CNN compiler.

**EfficientNet-B0** (MBConv + squeeze-and-excitation) is the remaining M3 block
and is tracked separately: its SE gate needs a sigmoid runner and a per-channel
broadcast-multiply on the CSP path (SiLU approximated by ReLU6 for the subset).

## Files

- `include/sw/kpu/timing/graph/mobilenetv2.hpp` — reusable `build_mobilenetv2`
  builder + composed host oracle + `MobileNetV2Spec`.
- `examples/milestones/m3_mobilenet.cpp` — demonstrate / validate / benchmark
  driver (`--dot`, spec sweep).
- `tests/timing/test_m3_mobilenet.cpp` — full network vs oracle + spec guard.
- `tests/timing/test_m3_mobilenet_block.cpp` — the inverted-residual block (M3-T3).
