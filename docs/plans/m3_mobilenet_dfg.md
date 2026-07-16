# M3: MobileNetV2 + EfficientNet-B0 on the CSP Executor — DFG design

**Status:** DRAFT (M3-T1 design) — awaiting review
**Milestone:** #131 (DNN Milestone M3: MobileNetV2 + EfficientNet-B0)
**Builds on:** M2 (#130, done) — the `KernelGraph` DFG + `GraphCspExecutor` bridge
(`include/sw/kpu/timing/graph/`), plus E6 conv, E7 pooling, E9 BN fold, E10 fused
epilogue, E2 broadcast/elementwise. **Template:** M2 (`docs/plans/m2_resnet_dfg.md`,
`docs/milestones/M2_resnet.md`).

---

## 1. Goal

Two more recognizable models — **MobileNetV2** (inverted-residual bottlenecks)
and **EfficientNet-B0** (MBConv + squeeze-and-excitation) — expressed as
`KernelGraph` DFGs and executed end-to-end on the CSP executor, reusing the M2
bridge. This extends the same epic subset (depthwise stresses E6; SE is an
E7+E5+E2 composition) and delivers M2's three-tier DoD (demonstrate / validate /
benchmark) for these models.

## 2. What is new beyond M2

MobileNet/EfficientNet reuse M2's conv/BN/ReLU/add/GAP/FC, and add four things:

1. **Depthwise convolution** (`groups = Cin`) — each output channel is the 2D
   conv of a *single* input channel with its own `Kh×Kw` filter. This is the
   defining MobileNet op and the one E6 explicitly deferred.
2. **Pointwise convolution** (`1×1`) — already a standard conv (a GEMM), done.
3. **ReLU6** (`min(max(x, 0), 6)`) — MobileNetV2's activation.
4. **Squeeze-and-excitation** (EfficientNet) — channel-wise gating.

## 3. The key decision: depthwise conv = pooling-window + per-channel filter reduce

**Deliver depthwise conv by reusing the E7 pooling per-channel window unfold plus
a learned-filter dot-product reduce — not a new grouped-GEMM path.** A depthwise
conv is, per channel, exactly the pooling movement (`pool_window_channel` gives
`[N·Hout·Wout, Kh·Kw]` window rows, 0-filled at padding via the AVG mode) followed
by a reduce that is a **dot-product of each window row with that channel's
`Kh·Kw` filter** — a weighted sum — instead of pooling's max/mean:

```text
y[n, c, ho, wo] = sum_{kh, kw} x[n, c, ho*s+kh-p, wo*s+kw-p] * w[c, kh, kw]
               = dot( window_row(n,c,ho,wo), filter[c] )
```

Rationale (mirrors conv2d's "reuse what's correct"):

- **Both halves already exist.** The per-channel window patchify is E7's
  `pool_window_channel`; the reduce is a `FunctionalComputeSpec` dot-product (the
  functional value plane, as pooling/softmax/norms use). Depthwise is a
  Vector-Engine op, **no systolic-array GEMM, no new kernel** — just a per-channel
  window-times-filter reduce.
- **It sidesteps the grouped-GEMM.** The im2col+GEMM lowering (E6) assumes a full
  `K = Cin·Kh·Kw` cross-channel reduction; depthwise has *no* cross-channel term
  (`K = Kh·Kw` per channel). Forcing it into a GEMM means `C` tiny `[M, Kh·Kw] @
  [Kh·Kw, 1]` matmuls — wasteful. The pooling-window form is the natural fit and
  reuses proven movement.
- **Pointwise stays a GEMM.** The `1×1` expand/project convs are standard convs
  (`K = Cin`, done via E6); depthwise-separable = depthwise (this) + pointwise.

**BN fold + fused epilogue extend to depthwise:** the per-channel BN folds into
the depthwise filter+bias exactly as it folds into a conv's weights (E9), and
ReLU6 applies as the reduce's epilogue (Section 4).

## 4. ReLU6 and the sigmoid/gating ops

- **ReLU6** = `min(max(x, 0), 6)`. The executor's `FunctionalActivation` is
  `{NONE, RELU}`; T2 either adds a `RELU6` activation to the epilogue or applies a
  post-op elementwise clamp (`min(relu(x), 6)`) on the CSP path. The fused-epilogue
  form (activation in-compute) is preferred where the producer is a conv/depthwise.
- **Squeeze-and-excitation** (EfficientNet MBConv): `GAP → FC_reduce → ReLU →
  FC_expand → sigmoid → broadcast-multiply` the per-channel gate against the block
  activation. Reuses GAP (E7), FC (matmul), **sigmoid** (a new elementwise unary),
  and **channel-broadcast multiply** (E2 broadcast) — a composition, no new kernel.

## 5. The blocks

- **MobileNetV2 inverted residual:** `1×1 expand (→ t·Cin) → BN → ReLU6 → 3×3
  depthwise (stride s) → BN → ReLU6 → 1×1 project (→ Cout) → BN`, with a residual
  add when `s == 1` and `Cin == Cout`. All nodes are conv (pointwise)/depthwise/
  BN/ReLU6/add — the M2 bridge extended with depthwise + ReLU6.
- **EfficientNet-B0 MBConv:** the inverted residual **plus** an SE block between
  the depthwise and the project conv, and swish/SiLU is approximated by ReLU6 for
  the M3 subset (documented; SiLU is a follow-on activation). SE = Section 4.

## 6. Reuse map & bridge extensions

| Need | Reused from | New in M3 |
|---|---|---|
| Depthwise window movement | E7 `pool_window_channel` (0-fill) | filter dot-product reduce (`run_depthwise_conv`) |
| Pointwise conv, BN fold, ReLU epilogue, residual add, GAP, FC | M2 bridge (`csp_op_runners` / `GraphCspExecutor`) | — |
| ReLU6 | fused-epilogue activation / elementwise | clamp / `RELU6` activation |
| SE gating | E7 GAP + matmul FC + E2 broadcast | sigmoid unary + channel-broadcast multiply |
| Block/model graphs | M2 `KernelGraph` + oracle pattern | MobileNet/EfficientNet builders |

`GraphCspExecutor` dispatch gains: **depthwise-conv** nodes (a CONV2D with
`groups == Cin`, detected from `conv2d_config().groups`), **ReLU6/sigmoid**
elementwise, and **broadcast-multiply** (the SE gate). The conv+BN fold and
ReLU/ReLU6 fusion apply to depthwise as they do to standard conv.

## 7. Honest scope

- Depthwise is delivered via the **pooling-window + filter reduce**, not a
  dedicated grouped-conv schedule generator; a native grouped/depthwise generator
  (and dilation) remain E6 follow-ons.
- The DFX tiling/dataflow compiler is still matmul-only (unchanged from M2) — M3
  tests operator-level fusions + the pooling/broadcast reuse, not the full CNN
  compiler.
- **SiLU/swish** (EfficientNet's real activation) is approximated by ReLU6 for the
  M3 subset; a SiLU activation is a named follow-on. Demo dims are scaled for the
  CSP simulation (as M2), with the batch/tile-alignment discipline.

## 8. Staging

| Stage | Content |
|---|---|
| **T1 (this)** | Design |
| **T2** | `run_depthwise_conv` (pooling-window + per-channel filter reduce, BN fold + ReLU6 epilogue) + ReLU6/sigmoid/broadcast-multiply runners; bridge dispatch; unit tests vs host oracles |
| **T3** | MobileNetV2 inverted-residual block + EfficientNet MBConv (with SE) block DFGs, end-to-end oracle-validated |
| **T4** | Full MobileNetV2 + EfficientNet-B0 builders, demo (`m3_mobilenet`), benchmark table, writeup `docs/milestones/M3_mobilenet.md` |

## 9. Projected effect

On completion, MobileNetV2 and EfficientNet-B0 run end-to-end as DFGs on the CSP
executor, validated vs host oracles — adding depthwise conv, ReLU6, and SE gating
to the operator set, all as reuse/compositions over the M2 bridge. This also
advances the `conv2d` row's deferred depthwise/grouped work and exercises the
E2 gating path (per #131). Coverage: a `depthwise_conv` operator row is a natural
addition to `pattern_coverage.json`.
