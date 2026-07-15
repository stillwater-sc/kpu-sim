# E6: Conv2D CSP Movement Pattern

**Status:** DRAFT (conv2d-T1, issue #119) — awaiting review
**Epic:** #75 (pattern classes P1 dense GEMM, P2 sliding-window, P7 layout)
**Depends on:** E2 (broadcast/elementwise, done), the value-producing matmul
path (done, #66) — both landed.
**Will unlock (once T2–T5 land):** `conv2d.functional` (the largest M2
ResNet gate cell), and — via conv+BN folding — the ResNet residual block.

---

## 1. The operator

For NCHW-style conv (the generator uses NHWC addressing internally):

```text
y[n, co, ho, wo] = bias[co] +
    sum_{ci, kh, kw} x[n, ci, ho*s + kh - p, wo*s + kw - p] * w[co, ci, kh, kw]
```

Two standard lowerings to the KPU's systolic fabric:

1. **im2col + GEMM.** Unfold each output position's receptive field into a
   column, forming a matrix `Xcol` of shape `[Cin*Kh*Kw, N*Hout*Wout]`, then a
   single GEMM `Y = W_mat @ Xcol` with `W_mat = [Cout, Cin*Kh*Kw]` produces
   `[Cout, N*Hout*Wout]`, reshaped to the output tensor. The convolution
   *becomes* a matmul.
2. **Direct sliding-window (P2).** Keep an input tile resident and slide the
   `Kh x Kw` window across it, reusing the overlapping halo, accumulating into
   the output — no `Xcol` materialization.

## 2. The decision: im2col + GEMM for M2

**E6 delivers `conv2d.functional` via im2col + GEMM, reusing the
value-producing matmul path; the direct sliding-window (P2) path is a
memory/perf follow-on, deferred.** Rationale:

- **It reuses what is already correct.** The value-producing tiled matmul
  (`schedule_matmul_compute`, verified end-to-end in the matmul simulator) *is*
  the conv compute. Conv2D as im2col+GEMM needs no new compute kernel — the
  K-accumulation, the A/B streaming, the bias+activation epilogue
  (`MatMulComputeSpec::bias` / `activation`) are all in place. M2's bar is
  **correct values**, and this is the shortest correct path to them.
- **The generator already exists.** `Conv2dScheduleGenerator`
  (`IM2COL_INTERLEAVED`) already emits the im2col+GEMM movement; its one defect
  is that its COMPUTE is not executable (it emits DRAIN without a COMPUTE
  carrying the GEMM dependency set — the conv2d half of #139). T3 hardens it
  exactly as the elementwise/reduction/softmax generators were hardened:
  emit COMPUTE with its full A/B K-slice dependency set.
- **Direct conv is a perf optimization, not a correctness one.** Its win is
  avoiding im2col's `Kh*Kw` memory blowup and re-reads; but it needs the P2
  sliding-window movement (halo overlap, multicast credits) which is genuinely
  new. Building it now would gate M2 on new movement machinery for no
  functional gain. It is scheduled as an E6 follow-on (or a perf milestone),
  and P2 is shared with pooling (E7), so it is built once when perf demands it.

### Conv2D-as-GEMM dimensions

| GEMM axis | conv quantity |
|---|---|
| M (rows of `W_mat`, C rows) | `Cout` |
| N (cols, C cols) | `N * Hout * Wout` |
| K (reduction) | `Cin * Kh * Kw` |

where `Hout = (H_in + 2p - Kh)/s + 1` (and similarly `Wout`). Weights
`W_mat[co, (ci,kh,kw)]` are a static reshape of the filter (a `BM_RESHAPE`, or
laid out in DRAM once at load). The output GEMM tile `[Cout, Hout*Wout]`
reshapes to `y[n, co, ho, wo]`.

## 3. The im2col patchify — and the E4 relationship

The only conv-specific movement is producing the `Xcol` A-operand: each A tile
gathers, for a block of output positions, the input elements of their receptive
fields (with stride/padding; padded positions contribute 0). Two realizations,
and T1 recommends the first for M2 velocity:

1. **Materialized `Xcol` (M2 route).** The patchify is a deterministic layout of
   the input tensor; the generator's A-tile payloads are seeded with the im2col
   columns (host-side patchify for the functional tier, or a one-shot DMA that
   lays `Xcol` out in DRAM). The CSP schedule then runs the **exact** GEMM value
   path. Correct values immediately, zero new executor machinery. The cost —
   `Kh*Kw` memory blowup for `Xcol` — is acceptable for a functional milestone.
2. **Gather-addressed loads (CSP-native refinement).** The A-tile LOADs carry
   im2col addressing and the value plane gathers input elements into the column
   tile — no `Xcol` materialization. This is a *specific instance* of E4's
   patchify (P7); it does **not** require the full E4 layout epic, only an
   im2col gather-load. It is the faithful CSP form and the natural T-follow-on
   once (1) is functional.

Choosing (1) for M2 keeps conv2d's functional milestone off the E4 critical
path: E4's general layout machinery is not a prerequisite for correct conv
values, only for the eventual gather-load refinement.

## 4. Conv + BatchNorm folding (the ResNet block)

At inference, BN is an affine per output channel using precomputed running
stats, and it **folds into the preceding conv**:

```text
w'[co,...] = w[co,...] * gamma[co]/sqrt(var[co]+eps)
b'[co]     = beta[co] - mean[co]*gamma[co]/sqrt(var[co]+eps)
```

so `conv -> BN -> ReLU` becomes a single GEMM with folded weights, `bias = b'`,
and `activation = RELU` — all already expressed by `MatMulComputeSpec`. This is
the ResNet residual-block route and needs no new compute. (Standalone
`batchnorm.functional`, the separate M2 gate cell, is the E9 broadcast-affine
op; the fold is the conv-side composition that makes the block efficient.)

## 5. Envelope

im2col+GEMM inherits the matmul envelope discipline (#67): per-matrix burst
share bounds A/B residency, the existing generator carries `l3/l2` and validates
against `is_livelock_safe`. `Xcol`'s `Kh*Kw` blowup is a DRAM/bandwidth cost,
not a working-set one — the streaming working set stays one A/B pair plus the
pending output, as for any GEMM. Padding produces partial/zero receptive fields
at the borders; the patchify writes explicit zeros so no special compute path is
needed.

## 6. Deliverables mapped to the epic tasks

| Task | Content |
|---|---|
| T2 #120 (ISA/executor closure) | Confirm the GEMM value path + `MatMulComputeSpec` bias/activation cover conv (they do); add the im2col patchify helper (host-side `Xcol` materialization for the functional tier) + conv geometry (Hout/Wout, padding zeros); no new executor kernel |
| T3 #121 (generator) | Harden `Conv2dScheduleGenerator`: emit **executable** COMPUTE with the GEMM A/B K-slice dependency set (resolving #139 for conv2d), reshape weights to `W_mat`, wire bias/activation; envelope stamping |
| T4 #122 (functional + oracle) | Value-producing conv2d on the CSP executor vs a host conv2d oracle (incl. stride, padding, 1x1 pointwise, and a conv+BN+ReLU folded case); a small `conv2d_simulator` for the tile-state log is a natural artifact |
| T5 #123 (regression) | shape (N,C,H,W,K,stride,pad) x envelope matrix, credit/stall invariants, im2col-vs-direct DRAM-traffic note; coverage row `conv2d` design/isa/generator/functional/regression -> done |

Depthwise, grouped, and the direct sliding-window (P2) path are **E6 follow-ons
beyond the M2 gate** (M3 MobileNet stresses depthwise); T1 scopes them out of
the M2 critical path.

## 7. Risks

- **im2col memory blowup** (`Kh*Kw` redundancy in `Xcol`): acceptable for the
  functional milestone; the gather-load refinement (Section 3.2) removes it.
  T5 records the DRAM-traffic cost so the perf follow-on has a baseline.
- **Padding / non-unit stride edge cases**: the patchify (Section 3) is where
  they live; the oracle must cover `stride>1`, `padding>0`, and non-square
  `Kh!=Kw`, plus the `Hout/Wout` rounding. Pinned in T4.
- **Reusing the matmul path** means conv inherits its correctness — the matmul
  value path is verified (max-error-0 in the matmul simulator), so the risk is
  in the *lowering* (im2col addressing, weight reshape), not the compute. T4's
  oracle targets the lowering.
- **BN fold numerics**: `gamma/sqrt(var+eps)` reuses the E3/E8 clamped-variance
  discipline; the fold is done in fp32 at weight-prep time.

## 8. Projected coverage-matrix effect (conditional on T2–T5)

Nothing is claimed done here (the #93 contract). WHEN T2–T5 land, the `conv2d`
row goes to done across all five stages, **`conv2d.functional` — the largest M2
gate cell — is satisfied**, and with conv+BN folding the ResNet residual block
executes. Remaining M2 gate after E6: `pooling.functional` (E7),
`batchnorm.functional` (E9), `epilogue_fused.regression` (E10-T5).
