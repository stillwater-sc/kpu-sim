# E9: BatchNorm (inference) CSP Movement Pattern

**Status:** DRAFT (batchnorm-T1) — awaiting review
**Epic:** #78 (E9 norm family — LayerNorm/RMSNorm/BatchNorm; pattern classes P3
reduction + P5 broadcast)
**Depends on:** E2 (broadcast/elementwise, done), E3 (online reductions, done —
needed only for the *training* follow-on) — both landed.
**Will unlock (once T2–T5 land):** `batchnorm.functional` — an M2 ResNet gate
cell.

> **Sub-issue note.** E9 is a Wave-2 epic; its T1–T5 sub-issues were not filed
> when the wave was still closed. This design is the batchnorm slice of E9-T1;
> the LayerNorm/RMSNorm slices of E9 are tracked separately (LayerNorm movement
> already has a simulator + online-reduction generator from E3/E8 work). On
> approval, the batchnorm T2–T5 sub-issues can be filed to mirror the conv2d
> ladder (#120–#123).

---

## 1. The operator

BatchNorm at **inference** normalizes each channel with precomputed running
statistics (NCHW):

```text
y[n, c, h, w] = gamma[c] * (x[n, c, h, w] - mean[c]) / sqrt(var[c] + eps) + beta[c]
```

`gamma, beta, mean, var` are per-channel `[C]` vectors fixed at inference. There
is **no reduction** at inference — every output element is a per-channel affine
of its input element. (Training, which *computes* batch mean/var, is a separate
reduction pass — Section 6.)

## 2. The decision: fold to a per-channel affine, reuse the E2 broadcast value path

**E9 delivers `batchnorm.functional` (inference) by folding the four per-channel
parameters into a single scale/shift pair and executing the result as a
per-channel broadcast-affine on the existing value-producing functional path —
no new compute kernel.** The fold is algebraic and exact:

```text
scale[c] = gamma[c] / sqrt(var[c] + eps)
shift[c] = beta[c] - mean[c] * scale[c]
=>  y[n, c, h, w] = x[n, c, h, w] * scale[c] + shift[c]
```

Rationale:

- **It is the same fold conv2d T4 already validated.** The conv+BN fold
  (`docs/plans/e6_conv2d_pattern.md` §4) collapses BN into exactly this
  `scale/shift` per output channel. Standalone `batchnorm.functional` is that
  same affine when BN is *not* preceded by a foldable conv (or is kept as a
  distinct op). Reusing the fold keeps one numerically-audited BN definition.
- **BN inference is a broadcast, not a reduction or a matmul.** `y = x*scale +
  shift` with a per-channel scalar is precisely the E2 broadcast-affine pattern
  (P5). The value plane is a `FunctionalComputeSpec` (operation lambda over the
  streamed input tile plus the channel's resident scalars), the same machinery
  that produces elementwise/broadcast values today — **no `MatMulComputeSpec`,
  no new executor kernel.**
- **Folding halves resident parameter residency.** The current generator
  preloads *four* per-channel params (`4*C + 1` working set). Folding to
  `scale/shift` preloads *two* (`2*C + 1`), a strictly smaller working set for
  the same all-channel-resident envelope discipline (#90).

### Where the fold happens

The `scale/shift` fold is a host-side / param-prep step (fp32, once), exactly as
the conv+BN weight-prep is. The generator then loads `scale[c]`, `shift[c]`
(two `[C]` vectors) instead of the four raw params. This is the recommended M2
route; the four-param in-VE form (compute the full `gamma*(x-mean)*rsqrt(var+eps)
+beta` expression per element) is retained as an option (Section 5) but is not
the default — it carries 2× the resident params for no functional gain at
inference.

## 3. The #139 defect and the value path

`BatchNormScheduleGenerator::generate_inference_mode` today emits, per spatial
tile, `LOAD → MOVE → FEED → DRAIN → WRITEBACK → STORE` with a comment *"VE
performs (x-mean)*rsqrt(var+eps)*gamma+beta"* — **but no COMPUTE is emitted**, so
the DRAIN has no producer. This is the batchnorm instance of #139 (the same
defect class fixed for conv2d in T3, #121). T3 fixes it by emitting a COMPUTE
that binds:

- the **streamed input tile** for `(n, c, spatial-tile)` (a fed dependency), and
- the channel's **resident** `scale[c]` / `shift[c]` scalars (a resident
  dependency — the `resident_tiles` mechanism, #155, so the per-channel params
  stay in the fabric across all of channel `c`'s spatial tiles without a reload).

The COMPUTE's `FunctionalComputeSpec` operation is `tile -> tile * scale[c] +
shift[c]` (broadcast scalar over the tile), producing values; the existing
DRAIN/writeback/store then move the result out.

## 4. Reuse map

| Need | Reused from | New in E9 |
|---|---|---|
| Per-element affine value plane | `FunctionalComputeSpec` lambda (#66), E2 broadcast | the BN `x*scale+shift` binder |
| Per-channel scalar broadcast to a tile | E2 `FunctionalElementwiseExecutor` broadcast (#102) | channel-affine binding |
| Resident per-channel params across spatial tiles | `resident_tiles` / `schedule_compute(resident)` (#155) | scale/shift residency |
| All-channel preload envelope refusal | generator working-set check (#90) | fold `4C+1 -> 2C+1` |
| Executable COMPUTE before DRAIN | conv2d T3 (#121) COMPUTE-emission pattern | batchnorm generator emission |

## 5. Alternative realizations (not the M2 default)

1. **Four-param in-VE expression.** Feed `gamma/beta/mean/var` and compute
   `gamma*(x-mean)*rsqrt(var+eps)+beta` per element. Matches the current
   generator's four-load structure but keeps `4*C+1` resident params and
   recomputes `rsqrt(var+eps)` per element instead of once at prep. Kept as an
   option for callers that need the raw params live (e.g. a fused
   BN+something that consumes them), but not the inference default.
2. **Conv+BN fold (E6).** When BN follows a conv, it is absorbed into the conv
   GEMM (`docs/plans/e6_conv2d_pattern.md` §4) and there is no standalone BN op
   at all — the ResNet-efficient path. Standalone `batchnorm.functional` (this
   design) is the form for BN that is *not* fused.

## 6. Training mode (deferred beyond M2)

Training BN computes batch `mean[c]`, `var[c]` across the `N*H*W` elements of
each channel — a **P3 reduction** — then normalizes. The generator already
sketches the three-pass structure (mean, var, normalize) but, like inference,
emits no COMPUTE. Training reuses the E3 online-reduction machinery (mean/var
moment accumulation, #72) for the two stat passes, then the same affine apply as
inference. **M2 ResNet is inference-only**, so training is an E9 follow-on and is
out of the batchnorm M2 critical path; T1 scopes it out.

## 7. Envelope

BN inference streams one input tile at a time; the per-channel params are
resident. With the fold, the working set is `2*C + 1` (scale/shift for every
channel preloaded, plus the one streaming input tile), down from `4*C + 1`. This
is the all-channel-preload residency the generator already envelope-checks
against `per_matrix_burst_share` (#90); the fold strictly relaxes it. Large `C`
(e.g. ResNet's 512/2048-channel stages) is the residency stress — the design
records that folding is what keeps `2C+1` within a realistic L3/L2 envelope, and
T3 stamps the generation envelope so T5 asserts the refusal boundary.

## 8. Deliverables mapped to the epic tasks (batchnorm slice)

| Task | Content |
|---|---|
| T2 (ISA/executor closure) | Confirm the `FunctionalComputeSpec` broadcast-affine path covers BN (it does); add the host-side `scale/shift` fold helper (`gamma/beta/mean/var -> scale/shift`) + a `batchnorm_reference` oracle; no new executor kernel |
| T3 (generator) | Harden `BatchNormScheduleGenerator` inference: load `scale/shift` (two `[C]` vectors) instead of four params; emit **executable** COMPUTE binding the streamed input tile + resident `scale[c]/shift[c]` (resolving #139 for batchnorm); fold `4C+1 -> 2C+1` working set; envelope stamping |
| T4 (functional + oracle) | Value-producing BN inference on the CSP executor vs a host `batchnorm_reference` across shapes/channel counts/eps; a `batchnorm_simulator` tile-log artifact is a natural companion |
| T5 (regression) | shape × envelope matrix (incl. large-`C` residency and the `2C+1` refusal boundary), credit/stall invariants, characterization |

Training mode (P3 reduction) is an **E9 follow-on beyond the M2 gate** (T1 scopes
it out of the M2 critical path).

## 9. Risks

- **Large-`C` residency.** `2*C+1` params resident can still exceed a tight
  envelope for 2048-channel stages; the generator refuses a priori (envelope
  check) rather than wedging. T5 pins the boundary. (Folding is what makes this
  tractable — the four-param form would be `4C+1`.)
- **`rsqrt(var+eps)` numerics.** Computed once at fold time in fp32; reuses the
  E3/E8 clamped-variance discipline (`var+eps > 0`), so no per-element
  divide-by-zero. `eps` is carried in the fold, not the VE.
- **Reusing the broadcast path** means BN inherits E2's correctness — the
  broadcast-affine value plane is verified (E2 functional tests), so the risk is
  in the *fold* (param prep) and the *binding* (resident scalar per channel),
  not the compute. T4's oracle targets the fold + binding.

## 10. Projected coverage-matrix effect (conditional on T2–T5)

Nothing is claimed done here (the #93 contract). WHEN T2–T5 land, the
`batchnorm` row advances design/isa/generator/functional (regression at T5), and
**`batchnorm.functional` — an M2 ResNet gate cell — is satisfied**. Remaining M2
gate cells after E9-batchnorm: `pooling.functional` (E7 #76) and
`epilogue_fused.regression` (E10-T5 #79).
