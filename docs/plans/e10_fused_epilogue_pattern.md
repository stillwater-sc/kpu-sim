# E10: Fused Epilogue CSP Pattern

**Status:** DRAFT (fused-epilogue-T1) — awaiting review
**Epic:** #79 (E10 fused epilogue; pattern classes P1 GEMM + P5 broadcast +
P6 two-operand). Absorbs #47 (matmul-epilogue lowering).
**Depends on:** E2 (broadcast, done), E6 (conv2d, done). Both landed.
**Will unlock (for M2):** `epilogue_fused.regression` — the last non-pooling M2
ResNet gate cell.

> **Consolidation design.** Unlike conv2d (E6) and batchnorm (E9), the fused
> epilogue is **already functional** — `MatMulComputeSpec` applies bias +
> activation in the compute, validated numerically in milestone M1 (#137,
> matmul+bias+ReLU) and E6-T4 (conv+bias+ReLU and conv+BN+ReLU). This T1 does
> not introduce a new value path; it **formalizes the existing fusion**, decides
> the M2 scope (the regression cell), and scopes the epic remainder out of the
> M2 critical path. E10 is a Wave-2 epic whose T1–T5 sub-issues are not yet
> filed; on approval the M2-relevant task (T5 regression) can be filed.

---

## 1. The pattern

An epilogue is the pointwise tail of a GEMM/conv: a per-output-channel bias add
followed by an activation, applied to the accumulator **before it leaves the
compute fabric** — the intermediate GEMM/conv result never round-trips to DRAM.

```text
Z = A @ B                      (the GEMM/conv accumulator)
Y = act(Z + bias[col])         (the fused epilogue)
```

For the KPU this is a Vector Engine tail on the systolic-array output: bias is a
per-output-column (channel) broadcast operand (P5), activation is elementwise.

## 2. What already exists (the fusion is functional)

- **`MatMulComputeSpec { bias, activation }`** (`concurrent_timing_executor.hpp`)
  carries a per-output-column bias vector and a `FunctionalActivation`.
- **`execute_matmul`** applies the epilogue in-place on the accumulator
  (`value += bias[j]; if RELU && value < 0 value = 0`) and writes the result to
  the compute fabric — so the pre-bias `Z` is never materialized to DRAM.
- **Validated numerically:** M1 (#137) checks matmul+bias+ReLU against a host
  oracle end-to-end; E6-T4 (#122) checks conv+bias+ReLU and — via the conv+BN
  fold — conv+BN+ReLU. Both are the fused epilogue exercised through real
  schedules.

So `epilogue_fused` is `isa_closure = done`, `functional = done` today. The
missing M2 piece is a **dedicated regression** that locks the fused epilogue in
across a bias × activation × shape matrix as its own coverage row (rather than
implicitly, inside the matmul/conv tests).

## 3. The decision: an M2 regression over the existing fused path; defer the rest

**E10 for M2 delivers `epilogue_fused.regression` — a regression matrix over the
already-functional `MatMulComputeSpec` epilogue — and defers the epic's
generator/activation-breadth work beyond the M2 gate.** Rationale:

- **M2 needs the guarantee, not new machinery.** ResNet's blocks are
  conv→(BN fold)→ReLU and the final FC+bias; all are covered by the existing
  fused epilogue. The M2 risk is *regression* (a future change silently breaking
  the fuse or the bias/activation numerics), which a dedicated matrix pins.
- **ReLU is sufficient for ResNet.** The executor's `FunctionalActivation` is
  `{NONE, RELU}`; ResNet uses ReLU. Richer activations (GELU/SiLU/sigmoid/…,
  needed by transformers/MLPs) are an **E10 follow-on**, not an M2 blocker.
- **Dedicated tiled-epilogue generators (#47) are an optimization.** A schedule
  generator that emits the epilogue as an explicit fused DRAIN-path VE pass
  (vs. folding it into the compute) is the epic's generator task; it changes
  *how* the fuse is expressed, not *whether* M2's values are correct. Deferred.

## 4. The M2 regression (T5) — what it locks in

A matrix over the fused epilogue exercised through the matmul value path
(`schedule_matmul_compute` with `bias` / `activation`), each cell checked
elementwise against a host oracle `Y = act(A@B + bias)`:

- **bias:** none / per-column bias.
- **activation:** NONE / RELU (RELU cases include negative pre-activations so the
  clamp is exercised — the E6-T4 fp-associativity lesson: bounded operands keep
  the reduction exact so the epilogue diff is not masked).
- **shape × envelope:** a few GEMM shapes × {default, constrained-min,
  partitioned}, with the per-stage tile accounting / credit-conservation /
  stall invariants (the conv2d-T5 / batchnorm-T5 template).
- **the fusion invariant:** the pre-bias accumulator `Z` is never stored — the
  schedule has exactly the GEMM's STORE ops (one per output tile), no extra
  round-trip for the epilogue.
- **conv+BN+ReLU composition:** one cell drives the E6 conv path with a folded
  BN (E9 `bn_fold`) + ReLU, confirming the ResNet residual-block epilogue.

## 5. Deliverables mapped to the epic tasks

| Task | Content | M2? |
|---|---|---|
| T1 (this) | Formalize the fused-epilogue pattern; decide M2 scope | design |
| T2 (ISA/executor) | **Already done** — `MatMulComputeSpec` bias+activation | done |
| T3 (generator) | Dedicated tiled fused-epilogue / DRAIN-path VE generator (#47) | **follow-on** |
| T4 (functional) | **Already done** — M1 (#137) + E6-T4 (#122) oracle checks | done |
| T5 (regression) | The bias × activation × shape/envelope matrix of §4 | **M2 gate** |

Only **T5** is on the M2 critical path. T3 (explicit epilogue generators) and
activation breadth beyond ReLU are E10 follow-ons.

## 6. Risks

- **Overclaiming the row.** `design`/`generator` stay `partial` after M2 (the
  dedicated generator is deferred); only `regression` advances to `done`. The
  #93 milestone gate requires exactly the cells M2 lists (`epilogue_fused.
  regression`), so a partial generator does not block M2 — but the coverage
  notes must state the deferral so the row is not read as fully complete.
- **fp associativity in the RELU cells.** As in E6-T4, adding bias after the
  K-reduction (executor) vs. seeding the oracle differently diverges by ULPs at
  large magnitudes; bounded operands keep it exact and the clamp meaningful.

## 7. Projected coverage-matrix effect (conditional on T5)

Nothing is claimed done here. WHEN T5 lands, `epilogue_fused.regression` →
`done` and **the M2 gate cell is satisfied**; `design` advances to `done`
(this doc), `generator` stays `partial` (dedicated generator deferred, noted).
After E10-for-M2, the only remaining M2 gate cell is `pooling.functional`
(E7 #76).
