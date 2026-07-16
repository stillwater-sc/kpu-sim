# E7: Pooling CSP Movement Pattern

**Status:** DRAFT (pooling-T1) — awaiting review
**Epic:** #76 (E7 pooling; pattern classes P2 sliding-window + P3 reduction)
**Depends on:** E3 (online reductions, done), E6 (conv2d im2col, done). Both landed.
**Will unlock (once T2–T5 land):** `pooling.functional` — the **last** M2 ResNet
gate cell (after conv2d, batchnorm, elementwise, epilogue_fused).

> E7 is a Wave-1 epic whose T1–T5 sub-issues were not filed when the wave was
> open. This is pooling-T1; on approval the T2–T5 sub-issues can be filed
> mirroring the conv2d ladder (#120–#123).

---

## 1. The operator

Pooling reduces each channel over a spatial window — **no cross-channel mixing**
(unlike conv). Three forms cover M2 ResNet:

```text
max-pool:  y[n,c,ho,wo] = max_{kh,kw} x[n, c, ho*s + kh - p, wo*s + kw - p]
avg-pool:  y[n,c,ho,wo] = mean_{kh,kw} x[n, c, ho*s + kh - p, wo*s + kw - p]
gap:       y[n,c,0,0]   = mean_{h,w}   x[n, c, h, w]          (global average)
```

ResNet uses **max-pool** (3×3, stride 2, after the stem conv) and **global
average pool** (before the final FC). `Hout/Wout` use the conv floor:
`Hout = floor((H + 2p - Kh)/s) + 1`.

## 2. The decision: im2col-window + per-channel reduce (reusing E6 + E3)

**E7 delivers `pooling.functional` by unfolding each channel's pooling window
into a row and reducing it with the existing Vector-Engine reduction — no new
compute kernel and no new movement machinery for M2.** Concretely:

- **Windowed pooling (max/avg)** = a **per-channel im2col** producing
  `Xwin[c] = [N*Hout*Wout, Kh*Kw]`, reduced along the `Kh*Kw` axis by
  `VE_REDUCE` (`MAX` for max-pool, `MEAN` for avg-pool). This is the E6 im2col
  patchify (Section 3) specialized to a **single channel** (`K = Kh*Kw`, not
  `Cin*Kh*Kw`) followed by an E3 reduce, instead of a GEMM.
- **Global average pool** = a **full per-channel spatial reduction**
  `mean_{h,w}` — exactly the E3 online reduction (`FULL_REDUCE` with `MEAN`) run
  once per channel. No window im2col needed.

Rationale (mirrors the conv2d im2col decision):

- **Both primitives already exist.** The `VE_REDUCE` op set is `{MAX, MIN, SUM,
  MEAN, VAR}` (E3), so max-pool = `MAX`, avg-pool = `MEAN`, gap = `MEAN` over the
  whole plane — **no new reduction kernel**. The im2col patchify
  (`conv2d_im2col.hpp`) already materializes window rows.
- **It reuses what is correct.** The E3 `FunctionalReductionExecutor` /
  online-reduction value path is verified; pooling is that reduce applied to
  window rows. M2's bar is correct values, and this is the shortest correct path.
- **Direct sliding-window (P2) is a perf optimization, not a correctness one.**
  Its halo reuse (shared with conv's direct path) avoids the `Kh*Kw` im2col
  re-reads, but needs the P2 movement that conv also defers. Building it now
  gates M2 on new machinery for no functional gain. Deferred (Section 6).

## 3. The reduce, padding, and count semantics

- **max-pool** ignores padded positions (padding contributes `-inf`, so it never
  wins the max). The window im2col writes `-inf` (not `0`) for padded taps in the
  max case — a pooling-specific patchify tweak over the conv `0`-fill.
- **avg-pool** divides by the window count. The reference (`ref_pool2d`) supports
  count-includes-padding vs count-excludes-padding; T1 targets the ResNet default
  (count-excludes-padding for partial border windows) and the oracle pins it.
- **gap** is `MEAN` over exactly `H*W` valid elements — no padding.

The reduce itself is `VE_REDUCE`; the pooling-specific parts are the **window
patchify** (per-channel, `-inf` vs `0` fill) and the **avg count**.

## 4. Reuse map

| Need | Reused from | New in E7 |
|---|---|---|
| Window row materialization | E6 `conv2d_im2col.hpp` im2col patchify | per-channel `K=Kh*Kw`, `-inf` fill for max |
| Max / mean reduce value plane | E3 `VE_REDUCE {MAX, MEAN}` + `FunctionalReductionExecutor` | pooling reduce binding |
| Global average pool | E3 `FULL_REDUCE` (MEAN) per channel | per-channel driver |
| Executable reduce COMPUTE before DRAIN | conv2d-T3 / batchnorm-T3 COMPUTE-emission | pooling generator emission (avoids #139 from the start) |
| Host oracle | `ref_pool2d` (compute_harness.hpp) MAX/AVG/ADAPTIVE_AVG | — |

## 5. Value path — no new kernel

The pooling COMPUTE is a `FunctionalComputeSpec` (or the E3 `VE_REDUCE` reduce
spec) whose operation reduces the window-row input tile to one output element per
output position (`MAX`/`MEAN`). Like batchnorm (E9), pooling is a **Vector-Engine
op, not a systolic-array (matmul) op** — it uses the functional/reduction value
path, not `MatMulComputeSpec`. The generator (T3) emits an executable reduce
COMPUTE before each drain, so — unlike the pre-existing norm/conv generators —
pooling never has the #139 DRAIN-without-COMPUTE defect.

## 6. Scope for M2; deferred

- **In:** max-pool, avg-pool (windowed, strided, padded), global average pool —
  via im2col-window + `VE_REDUCE` and E3 full-reduce.
- **Deferred (E7 follow-on, off the M2 path):** the **direct sliding-window /
  halo (P2)** realization (perf; the `Kh*Kw` im2col re-reads become on-chip halo
  overlap), shared with conv's direct path — built once when perf demands it.
  Adaptive/fractional pooling beyond ResNet's needs is also out.

## 7. Envelope

Windowed pooling streams one channel's window rows and reduces; the streaming
working set is one window-row tile plus the pending output — independent of
channel count (channels are independent, processed in sequence or streamed).
The `Kh*Kw` im2col inflation is a DRAM/bandwidth cost (the same as conv's), not a
working-set one — the direct/halo follow-on removes it. Global average pool is a
single `K`-scaled reduce per channel (E3), `K = H*W`; T3 stamps the generation
envelope and T5 asserts the refusal boundary.

## 8. Deliverables mapped to the epic tasks

| Task | Content |
|---|---|
| T2 (ISA/executor closure) | Confirm `VE_REDUCE {MAX, MEAN}` + the functional/reduction value path cover pooling (they do); add the per-channel window-patchify helper (`-inf` fill for max, count for avg) + a `pool2d_reference` oracle wrapper; no new executor kernel |
| T3 (generator) | A `PoolingScheduleGenerator`: per-channel window im2col + executable `VE_REDUCE` COMPUTE (max/avg) before each drain; a global-average-pool mode (E3 full-reduce); envelope stamping. Executable from the start (no #139) |
| T4 (functional + oracle) | Value-producing max/avg/gap on the CSP executor vs `ref_pool2d` across window/stride/pad/global; a `pooling_simulator` tile-log artifact |
| T5 (regression) | shape × (pool-type) × envelope matrix, credit/stall invariants, refusal boundary, characterization |

## 9. Risks

- **`-inf` fill for max-pool.** The window patchify must fill padded taps with
  `-inf` (not `0`) so padding never wins the max; the oracle covers a padded
  border where `0 > x` would otherwise corrupt the result. Pinned in T4.
- **avg count at the border.** Count-excludes-padding vs includes-padding changes
  border values; T1 fixes the ResNet default and the oracle matches it.
- **Reusing E3/E6** means pooling inherits their correctness; the risk is in the
  *window patchify* (per-channel addressing, `-inf`/count) and the reduce
  *binding*, not the reduce compute. T4's oracle targets those.

## 10. Projected coverage-matrix effect (conditional on T2–T5)

Nothing is claimed done here (the #93 contract). WHEN T2–T5 land, the `pooling`
row advances design/isa/generator/functional/regression, and
**`pooling.functional` — the last M2 ResNet gate cell — is satisfied**, which
(with conv2d, batchnorm, elementwise, epilogue_fused already done) **completes
the M2 gate** and unblocks the M2 ResNet demo (#130). The direct/halo P2 path is
an E7 follow-on.
