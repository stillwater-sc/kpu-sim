# CSP Data-Movement Pattern Coverage Roadmap

**Status:** APPROVED — filed 2026-07-11 (umbrella #88; epics #69–#87; Wave 0/1
sub-issues #89–#128; filing manifest/script in `tools/project/`)
**Date:** 2026-07-11
**Scope:** Build the complete set of CSP data-movement patterns needed to
execute the operators of computer-vision, LLM, and JEPA models on the
credit-based dataflow simulator — one epic per core operator (including the
important fused operators), each decomposed into medium-sized (estimate 5)
sub-issues.

---

## 1. Method: patterns first, operators second

Operators cluster onto a much smaller set of *data-movement pattern
classes*. Staging by pattern class (foundations first) means each operator
epic composes existing movement primitives instead of reinventing them, and
"coverage" becomes measurable: every operator in the target models maps to
pattern classes that exist, generate, execute functionally, and validate.

### Pattern classes and the operators that need them

| # | Pattern class | Movement signature | Consumed by |
|---|---|---|---|
| P1 | Dense GEMM streaming | A/B tile streams, K accumulation, C drain | every matmul, conv-as-GEMM, attention GEMMs, FFN, logits |
| P2 | Sliding-window reuse | input tile feeds multiple outputs (halo overlap), multicast credits | direct conv2d, pooling, depthwise |
| P3 | Row-streaming reduction | row resident while running stats accumulate (online max/sum/mean/var), then normalize in-place | softmax, layernorm, RMSNorm, global pool |
| P4 | Gather / scatter (indirect) | index-driven DMA, non-contiguous tiles | embedding lookup, KV-cache read, JEPA token masking |
| P5 | Broadcast | one small tile to many consumers | bias, RoPE cos/sin tables, BN scale/shift |
| P6 | Two-operand aligned streaming | two streams arriving rate-matched at compute | residual add, SwiGLU gating, elementwise binary |
| P7 | Layout transform | transpose / reshape / space-to-depth in flight | patchify, head split/merge, im2col, transposed conv |
| P8 | Chained-GEMM fusion | intermediate stays on-chip between GEMMs (resident tile + online reduction) | flash attention, fused FFN |
| P9 | Append / growing extent | streaming write to a growing region + windowed re-read | KV-cache append, autoregressive decode |

Status today: P1 exists (matmul generators, envelope-aware per #67, with
value-producing execution per #66). P3 exists in multi-pass form (softmax /
layernorm / batchnorm generators). P4, P5, P6 have ISA opcodes that are
**annotation-only** (`DMA_LOAD_GATHER/STORE_SCATTER`, `STR_BROADCAST_*`,
`VE_ELEMENTWISE/REDUCE` — no-ops in the executors); P7 has opcodes
(`BM_TRANSPOSE/RESHAPE`) exercised only for B-matrix transpose. P2, P8, P9
do not exist.

### Operator inventory by model family

- **CV (ResNet/MobileNet/UNet + ViT):** conv2d (im2col + direct), depthwise
  + pointwise + grouped conv, pooling (max/avg/global), BN (inference
  fold), activations, residual add, upsample/interp + transposed conv,
  patchify, attention/MLP blocks (shared with LLM).
- **LLM (Llama-class decoder):** embedding gather, RMSNorm, QKV/O
  projections, RoPE, attention (QKᵀ → softmax → PV) with KV cache, SwiGLU
  FFN (gate/up GEMMs + SiLU⊙ + down GEMM), logits GEMM.
- **JEPA (I-JEPA / V-JEPA):** patchify, context/target token masking
  (gather/scatter by mask indices), ViT encoder blocks, narrow predictor,
  embedding-space distance reductions. (Training-side EMA/optimizer is out
  of scope for the pattern catalog; masking is the JEPA-distinctive
  movement.)
- **Key fused operators:** matmul+bias+activation (epilogue — existing #47),
  conv+BN+relu, fused FFN block (SwiGLU / GELU-MLP), flash attention,
  norm+GEMM prologue, residual+norm.

---

## 2. Epic catalog

One epic per operator (or tight operator family), each built from the
pattern classes above. Foundations (Wave 0) are cross-cutting pattern
epics; operator epics then compose them.

### Wave 0 — Pattern foundations (enablers, mostly ISA-gap closure)

| Epic | Title | Pattern | Unlocks |
|---|---|---|---|
| E0 | Pattern infrastructure hardening: PartitionedCreditPool wiring, envelope for all existing generators, envelope-mismatch warning in ScheduleExecutor (absorbs #67 follow-ups, #18) | — | everything |
| E1 | Gather/scatter movement (indirect addressing, functional `DMA_LOAD_GATHER`/`STORE_SCATTER`) | P4 | embedding, KV-cache, JEPA masking |
| E2 | Broadcast + two-operand aligned streaming (functional `STR_BROADCAST_*`, `VE_ELEMENTWISE`) | P5, P6 | bias, residual, gating, RoPE |
| E3 | Streaming/online reductions (functional `VE_REDUCE`, running max/sum/mean/var in compute) | P3 | online softmax, norms, flash attention |
| E4 | Layout transforms in flight (functional `BM_TRANSPOSE`/`BM_RESHAPE` beyond B-transpose; patchify; head split/merge) | P7 | ViT, attention, im2col |

### Wave 1 — Dense compute operators

| Epic | Title | Patterns |
|---|---|---|
| E5 | GEMM family completion: batched, rectangular aspect ratios, envelope-aware weight/input-stationary strategies | P1 |
| E6 | Conv2D family: im2col path hardened + direct sliding-window with halo reuse; depthwise; pointwise; grouped | P1, P2, P7 |
| E7 | Pooling & spatial reductions (max/avg/global) | P2, P3 |

### Wave 2 — Normalization, activation, epilogue fusions

| Epic | Title | Patterns |
|---|---|---|
| E8 | Softmax: multi-pass → online single-pass generator | P3 |
| E9 | LayerNorm / RMSNorm (+ BN inference fold) | P3, P5 |
| E10 | Fused epilogue: matmul+bias+activation and conv+BN+relu (absorbs #47; coordinates with #45/#48-50 sweep suite) | P1+P5+P6 |
| E11 | Fused FFN block: SwiGLU / GELU-MLP (gate/up GEMMs + gating ⊙ + down GEMM, intermediates on-chip) | P8, P6 |

### Wave 3 — Attention & LLM-specific movement

| Epic | Title | Patterns |
|---|---|---|
| E12 | Attention: chained QKᵀ/softmax/PV, then flash-style fusion (resident Q tile, streamed K/V, online softmax) | P8, P3, P1 |
| E13 | KV-cache movement: append, windowed/paged gather, decode-step pattern | P9, P4 |
| E14 | RoPE & positional patterns (broadcast tables + pairwise rotate elementwise) | P5, P6 |

### Wave 4 — Model-boundary operators

| Epic | Title | Patterns |
|---|---|---|
| E15 | Embedding & logits (index gather + tall-skinny GEMM) | P4, P1 |
| E16 | ViT/CNN spatial boundary: patchify, upsample/interpolation, transposed conv | P7, P2 |
| E17 | JEPA masking: context/target token gather-scatter, predictor-block movement | P4, P7 |

### Wave 5 — Model-level integration

| Epic | Title |
|---|---|
| E18 | Model-block validation: ResNet block, ViT block, Llama decoder block, I-JEPA block executed end-to-end through the CSP executor against host oracles; pattern coverage matrix asserted in CI |

Dependencies: Wave 0 → everything; E6/E7 need E4/E2; E8/E9 need E3;
E10 needs E2; E11 needs E10+E2; E12 needs E3+E4(+E8 experience);
E13 needs E1; E14 needs E2; E15/E17 need E1; E18 needs its blocks' epics.
Waves 1–2 can proceed in parallel once Wave 0 lands; Wave 3 is the
critical path for LLM coverage.

---

## 3. Standard sub-issue decomposition (five medium tasks per epic)

Every operator epic decomposes into the same five sub-issues, each sized
**estimate 5** (medium). This mirrors the lifecycle that worked for matmul
(#61 → #63/#64 → #67) and makes progress comparable across epics:

1. **T1 — Pattern design & envelope analysis** (est. 5): data-movement
   characterization (reuse, residency, working set), SURE/domain-flow
   derivation where applicable, envelope-derived block-size formula (the
   #67 discipline), design doc in `docs/plans/`.
2. **T2 — ISA / executor capability closure** (est. 5): implement the
   annotation-only opcodes or executor capabilities the operator needs
   (functional semantics in the value-producing path from #66).
3. **T3 — Envelope-aware schedule generator** (est. 5): generator +
   strategies, constructive livelock-safety, `ScheduleValidator` rules.
4. **T4 — Functional integration & oracle** (est. 5): value-producing
   execution via tile payloads / `FunctionalComputeSpec`, verified
   elementwise against a host reference.
5. **T5 — Regression & characterization** (est. 5): execution regression in
   the `test_multi_tile_execution` style (strategy × size × envelope
   matrix), credit/stall invariants, performance characterization
   (cycles, utilization, stall breakdown) recorded in the epic.

Wave-0 pattern epics use the same template minus T3 or with T3 reduced
(they deliver primitives, not operator schedules). A few epics carry one
extra named task (E12: online-softmax integration; E13: paged-layout
design) — flagged in the epic body.

Scale: 19 epics × ~5 sub-issues ≈ **95 issues**, ~475 points total.
At a sustained pace of one epic per 1–2 weeks with waves 1–2 parallelized,
this is roughly a 2–3 quarter arc to full coverage (E18 green).

---

## 4. GitHub mechanics

- **Epics** are issues titled `Epic: <operator> CSP movement pattern`,
  labeled `epic` + `csp-patterns` + `wave-N` (new labels), containing the
  pattern-class mapping, dependency list, and definition of done.
- **Sub-issues** use GitHub's native sub-issue linking (GraphQL
  `addSubIssue`), labeled `csp-patterns` + `estimate:5`, titled
  `<epic-shortname>: <task>` (e.g. `attn-flash: envelope-aware generator`).
- **Milestones** one per wave (`csp-patterns-wave-0` … `-wave-5`) so the
  burn-down is visible per stage.
- **A single umbrella issue** (`Epic: CSP pattern coverage for CV/LLM/JEPA`)
  links all epics and carries the coverage matrix as a checklist.
- **Filing is scripted**: a JSON manifest (single source of truth for
  epics/sub-issues/labels/milestones) + a small script under
  `tools/project/` drives `gh` so the tree is reproducible and amendable.
- **Existing issues absorbed, not duplicated**: #47 becomes T3/T4 of E10;
  #45/#48/#49/#50 are referenced as the validation-sweep companion of E10/E11;
  #18 and the #67 follow-ups fold into E0.

### Filing strategy (recommendation)

File **all epics + the umbrella now** (they are stable and give the full
map), but file **sub-issues per wave as the wave opens** (Wave 0 + Wave 1
immediately, later waves when their predecessors near completion). This
keeps the tracker honest — sub-issues filed months ahead of their wave go
stale as the T2 capability landscape shifts underneath them.

---

## 5. Definition of done (per epic and overall)

Per epic: T1–T5 closed; operator executes functionally on the CSP executor
with oracle-verified values under at least two envelope configurations
(default + constrained); regression in CI; characterization numbers in the
epic.

Overall (umbrella): every operator row in the Section 1 inventory maps to a
closed epic, and E18's four model blocks run end-to-end green in CI.

---

## 6. DNN milestone ladder (outreach checkpoints)

Each milestone is an industry-recognizable model that becomes executable
once a specific epic subset closes. Milestones are ordered by subset
inclusion, so each rung adds only a few epics to the previous one — and
each is a public-facing artifact: researchers and customers recognize
"ResNet on the KPU," not "epic E6 closed."

Every milestone carries the same three-tier definition of done:

- **Demonstrate** — the model (or its defining block) runs end-to-end
  through the CSP executor with Chrome-trace visualization showing the
  credit-dataflow in motion (the educational artifact).
- **Validate** — full functional equivalence against a PyTorch/ONNX oracle
  (elementwise, per-layer), under default and constrained envelopes.
- **Benchmark** — cycles, utilization, stall breakdown, roofline position,
  and envelope sweeps (reusing the #48–#50 sweep machinery), packaged as a
  reproducible demo + short writeup under `docs/milestones/`.

| Milestone | Model | Epic set required (cumulative) | Ready after |
|---|---|---|---|
| **M1** | MLP (XOR / MNIST) | E5 (partial), E10-lite — `unified_xor_mlp` already runs functionally (#66) | **now** — formalize as the baseline demo/benchmark |
| **M2** | ResNet-18 → ResNet-50 | E0, E2, E4, E5, E6, E7, E9 (BN fold), E10 | Wave 1 + the CNN half of Wave 2 |
| **M3** | MobileNetV2 + EfficientNet-B0 | same set as M2 — depthwise stresses E6; squeeze-and-excitation is an E7+E5+E2 composition (global pool → 2 small FCs → sigmoid gating) | immediately after M2 |
| **M4** | Attention head (single MHA, chained → flash-fused) | + E3, E8, E12 | early Wave 3 — the flagship educational demo: same hardware, chained vs. flash, watch the DRAM traffic collapse in the trace |
| **M5** | Vision Transformer (DeiT-Tiny → ViT-B/16 encoder) | + E9 (LN), E11, E4/E16 (patchify) | mid Wave 3 (needs no KV-cache/RoPE) |
| **M6** | YOLO (v8n detection) | M2 set + E16 (upsample, concat); NMS stays host-side (standard practice) | parallel to Wave 3 — only E16 from Wave 4 is needed, and E16's true dependencies (E4, E6) close in Wave 1 |
| **M7** | LLM decoder (GPT-2 small / TinyLlama, token-by-token decode) | + E13, E14, E15 | end of Wave 3 + Wave 4 entry — "the KPU speaks" |
| **M8** | I-JEPA | M5 set + E1, E17 | Wave 4 — the research-frontier differentiator: no one benchmarks JEPA on novel dataflow hardware |

Notes:

- The ladder validates the wave ordering: Waves 1–2 unlock the **CNN
  family** (three model milestones from one epic subset), Wave 3 unlocks
  the **transformer family**, Wave 4 completes **LLM decode and JEPA**.
- **E16 should be pulled forward** if YOLO matters commercially: its real
  dependencies (E4, E6) close in Wave 1, so it can run parallel to Wave 3
  rather than waiting for Wave 4.
- E18 (model-block validation) is the CI-hardened, permanently-regressed
  form of M2/M5/M7/M8 — the milestones are the outreach-facing events, E18
  is what keeps them true afterwards.
- Tracking: one `DNN Milestone:` issue per rung (label `dnn-milestone`),
  listing its epic dependency set as a checklist and the three-tier DoD,
  linked from umbrella #88.
