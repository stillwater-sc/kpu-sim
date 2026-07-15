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

where

```text
Hout = floor((H_in + 2p - Kh)/s) + 1
Wout = floor((W_in + 2p - Kw)/s) + 1
```

(the floor is explicit — no divisibility of the padded extent by the stride is
assumed). Weights `W_mat[co, (ci,kh,kw)]` are a static reshape of the filter (a
`BM_RESHAPE`, or laid out in DRAM once at load). The batch dimension `N` lives on
the GEMM N axis, so the output GEMM tile is `[Cout, N*Hout*Wout]` (not just
`[Cout, Hout*Wout]` — for `N > 1` the batch is not dropped); it reshapes to
`y[n, co, ho, wo]`. `Conv2dScheduleGenerator` and the DSL `conv2d_im2col`
schedule use the transposed convention (`M = N*Hout*Wout`, `N-axis = Cout`); the
axis roles are symmetric, only the labels swap.

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
   im2col gather-load. It is the faithful CSP form.

   **This form is not greenfield: the movement/ISA half already exists.** The
   DSL `conv2d_im2col` schedule (`src/schedules/conv2d_schedule.cpp`) already
   emits `load_gather(MatrixID::A, im2col_params)`; `Im2ColParams` and the
   `LOAD_GATHER` scope live in `include/sw/kpu/dsl/schedule.hpp`; the
   `DMA_LOAD_GATHER` opcode is defined in the data-movement ISA and interpreted
   by the behavioral program executor. What is **absent** is the gather addressing
   on the *value-producing* `ConcurrentTimingExecutor` path — that executor's
   LOAD carries no im2col descriptor and its value plane does not gather. So the
   realization choice for M2 is not "materialize vs. build gather from scratch";
   it is "seed the existing GEMM value path with a materialized `Xcol` (no
   executor change) vs. teach the value executor the gather addressing the ISA
   already names."

Choosing (1) for M2 keeps conv2d's functional milestone off both the E4 critical
path and the value-executor gather work: E4's general layout machinery and the
value-plane gather are prerequisites only for the eventual refinement, not for
correct conv values. T2–T4 therefore target the materialized-`Xcol` value path;
wiring `DMA_LOAD_GATHER` through to the value executor is the named E6 follow-on
(Section 6), reusing the DSL/ISA gather form already in the tree.

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

## 5. Envelope (issue #119)

### 5.1 im2col + GEMM envelope

im2col+GEMM inherits the matmul envelope discipline (#67): per-matrix burst
share bounds A/B residency, the existing generator carries `l3/l2` and validates
against `is_livelock_safe`. The characterization is the GEMM one applied to the
lowered `(M=Cout, N=N·Hout·Wout, K=Cin·Kh·Kw)` shape:

- **Working set (streaming).** One A-tile `[Ti×Tk]` + one B-tile `[Tk×Tj]` in
  flight plus the pending output tile `[Ti×Tj]` — `Ti·Tk + Tk·Tj + Ti·Tj`
  elements, independent of `Kh·Kw`. `Xcol`'s `Kh·Kw` inflation is a **DRAM
  footprint / bandwidth** cost (each input element is re-read up to `Kh·Kw`
  times), **not** a working-set one; the streaming residency is a single A/B
  pair as for any GEMM. This is precisely why the direct/halo form (Section 3.2)
  is a *bandwidth* optimization, not a working-set one.
- **Reuse.** B (weights) is reused across all `N·Hout·Wout` output positions —
  weight-stationary reuse factor `= N·Hout·Wout / Tj`. A (`Xcol` columns) is
  reused across the `Cout / Ti` weight-row tiles. Reuse per element is the
  standard GEMM `min(M,N,K)`-order; the conv specialization is that A's reuse is
  degraded by im2col redundancy (the same input element appears in up to `Kh·Kw`
  distinct `Xcol` columns, each re-streamed independently).
- **Block sizes from the envelope.** Given L3/L2 burst shares `b_l3, b_l2`, the
  canonical bound is `Tk ≤ per_matrix_burst_share(min(l3,l2))` (the same
  `min(l3,l2)/4` share used by `is_livelock_safe`, shared with matmul). With the
  full receptive field taken as a single `Tk = K = Cin·Kh·Kw` (the DSL
  `conv2d_im2col` choice), this bound becomes a constraint on how large a
  receptive field may be streamed in one K-slice before the schedule must split
  `K`; T3 stamps the generation envelope so T5 can assert the split point.
- **Padding.** Border positions have partial/zero receptive fields; the patchify
  writes explicit zeros into `Xcol`, so no special compute or credit path is
  needed — padded columns are ordinary zero-valued A elements.

### 5.2 Scope vs. issue #119 — partially satisfied

Issue #119 asks for envelope characterization across the conv **family**
(direct/halo, depthwise, grouped), including SURE/domain-flow derivation. This
T1 delivers the **im2col+GEMM** characterization above in full, because that is
the only lowering E6 builds for M2. The remaining families are **explicitly out
of E6-T1** and #119 is therefore **only partially satisfied**:

- **Direct / halo (P2).** Its distinguishing envelope property — halo reuse
  turning the `Kh·Kw` DRAM re-reads into on-chip overlap, so DRAM traffic drops
  from `Kh·Kw·|X|` toward `|X|` — is stated qualitatively here (Section 3.2,
  Risks) but its quantitative SURE/domain-flow derivation and halo working-set
  formula are deferred with the direct-conv movement work.
- **Depthwise / grouped.** `K = Kh·Kw` per group (no `Cin` reduction across
  groups) changes the reuse and working-set balance materially; characterized
  when depthwise is built (M3 MobileNet stresses it).

**Follow-up deliverable:** the direct/depthwise/grouped envelope derivation is
tracked as the P2 sliding-window envelope note, landing with the direct-conv
movement follow-on (Section 6, "E6 follow-ons"); it is not a silent omission but
a scoped-out deliverable of #119. T5's DRAM-traffic note (im2col vs. direct)
establishes the baseline that derivation will quantify against.

## 6. Deliverables mapped to the epic tasks

| Task | Content |
|---|---|
| T2 #120 (ISA/executor closure) | Confirm the GEMM value path + `MatMulComputeSpec` bias/activation cover conv (they do); add the im2col patchify helper (host-side `Xcol` materialization for the functional tier) + conv geometry (Hout/Wout, padding zeros); no new executor kernel |
| T3 #121 (generator) | Harden `Conv2dScheduleGenerator`: emit **executable** COMPUTE with the GEMM A/B K-slice dependency set (resolving #139 for conv2d), reshape weights to `W_mat`, wire bias/activation; envelope stamping |
| T4 #122 (functional + oracle) | Value-producing conv2d on the CSP executor vs a host conv2d oracle (incl. stride, padding, 1x1 pointwise, and a conv+BN+ReLU folded case); a small `conv2d_simulator` for the tile-state log is a natural artifact |
| T5 #123 (regression) | shape (N,C,H,W,K,stride,pad) x envelope matrix, credit/stall invariants, im2col-vs-direct DRAM-traffic note; coverage row `conv2d` design/isa/generator/functional/regression -> done |

**E6 follow-ons beyond the M2 gate** (T1 scopes them out of the M2 critical
path):

- **Value-plane gather-load.** Wire the existing `DMA_LOAD_GATHER` /
  `Im2ColParams` / `load_gather` form (already in the DSL/ISA/behavioral stack,
  Section 3.2) through to the value-producing `ConcurrentTimingExecutor`,
  removing `Xcol` materialization. Reuses the movement form already in the tree.
- **Direct / halo (P2), depthwise, grouped.** New movement machinery and the
  #119 envelope derivation deferred in Section 5.2 (M3 MobileNet stresses
  depthwise); P2 is shared with pooling (E7), so the sliding-window movement is
  built once.

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
