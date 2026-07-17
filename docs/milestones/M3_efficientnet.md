# DNN Milestone M3: EfficientNet-B0 on the CSP Executor

**Status:** EfficientNet-B0 achieved (capability + demo). Issue #131.
**Demo:** `examples/milestones/m3_efficientnet.cpp` (`ctest -R m3_efficientnet`).
**Design:** `docs/plans/m3_mobilenet_dfg.md`. **Template:** M3 MobileNetV2
(`docs/milestones/M3_mobilenet.md`).

---

## What this delivers

**EfficientNet-B0** — the MBConv + squeeze-and-excitation CNN — expressed as a
**dataflow graph (DFG)** and executed end-to-end on the credit-based CSP
concurrent executor, with real fp32 values flowing DRAM → L3 → L2 → compute → …
The whole network runs through one `KernelGraph`, and its classification output
is validated elementwise against a composed host oracle.

Beyond the MobileNetV2 milestone, EfficientNet-B0 adds:

- **Squeeze-and-excitation** — a per-channel gate between the depthwise and the
  project: `GAP → FC_reduce → ReLU → FC_expand → sigmoid → channel-broadcast
  multiply` scales the block activation by its per-channel gate. This reuses GAP +
  matmul (with a ReLU epilogue) + two new CSP runners: **`run_sigmoid`** (the
  Vector Engine has no sigmoid, so `1/(1+e^-x)` is composed from four unary/scalar
  VE ops) and **`run_channel_broadcast_mul`** (the SE scale, a per-channel `[N,C]`
  gate × `[N,C,H,W]` activation).
- **SiLU/swish activation** (`x·sigmoid(x)`) — the real EfficientNet activation,
  on the CSP value path via `run_silu` (`run_sigmoid` then a binary multiply);
  the bridge dispatches `ElementwiseOp::SILU`.
- **Per-stage depthwise kernel sizes** (3×3 and 5×5) — `run_depthwise_conv`
  handles arbitrary `Kh×Kw`; the livelock-detector fix (counting backward-path +
  compute progress) lets wider/deeper depthwise run.

## Architecture: a KernelGraph DFG + the M2/M3 bridge

EfficientNet-B0 is built by `build_efficientnet_b0`
(`include/sw/kpu/timing/graph/efficientnet.hpp`) as a **`KernelGraph`** and
executed by **`GraphCspExecutor`**, reusing the M2/M3 bridge:

- **stem** — `3×3 s1 conv → BN → SiLU`.
- **MBConv+SE stack** — each block is `1×1 expand → BN → SiLU → k×k depthwise
  (stride s) → BN → SiLU → SE gate → 1×1 project → BN`, with an **identity
  residual** (an explicit graph edge) when `stride == 1 && Cin == Cout`. As in the
  real model, the `t == 1` stage omits the expansion, and the project is a linear
  bottleneck (no activation, none after the residual add).
- **head** — `1×1 conv → BN → SiLU → global-average-pool → FC`.

The bridge folds each BatchNorm into its conv, dispatches pointwise/expand/project
`1×1` convs to im2col→GEMM and depthwise convs (`groups == Cin`) to the
pooling-window path, runs the SE `sigmoid` and channel-broadcast `MUL` as VE ops,
and joins the identity residuals. Because the SE `sigmoid` alone is four VE ops
and each `SiLU` is five (sigmoid + multiply), the CSP op count can exceed the
graph node count.

## Definition of done

- **Demonstrate** — the whole network runs end-to-end on the CSP executor;
  `--dot FILE` emits the `KernelGraph`.
- **Validate** — output compared elementwise against a composed whole-network host
  oracle (pointwise / depthwise conv, BN, SiLU, the SE gate, add, GAP, FC),
  `max_err < 5e-3` (observed ~6e-8).
- **Benchmark** — cycles, CSP ops, cyc/op, and DMA/BM/STR stalls across a small
  sweep (compact base + a 5×5 MBConv variant).

## Honest scope

Dimensions are scaled for a fast CSP simulation (as the other milestones), with
the tile-alignment discipline: batch and all channel counts — **including the SE
reduce dim** — are multiples of the tile size, so the SE reduction uses a
tile-aligned floor (16) rather than the real `Cin/4` at these small widths.
Activations are the real **SiLU/swish** (`x·sigmoid(x)`), not the earlier ReLU6
approximation. The DFX tiling/dataflow compiler remains
matmul-only — M3 tests operator-level fusions + the SE composition, not a full CNN
compiler.

## Files

- `include/sw/kpu/timing/graph/efficientnet.hpp` — reusable `build_efficientnet_b0`
  builder + composed host oracle + `EfficientNetB0Spec`.
- `examples/milestones/m3_efficientnet.cpp` — demonstrate / validate / benchmark
  driver (`--dot`, spec sweep).
- `tests/timing/test_m3_efficientnet.cpp` — full network vs oracle + spec guard.
- `tests/timing/test_m3_efficientnet_block.cpp` — the MBConv+SE block.
