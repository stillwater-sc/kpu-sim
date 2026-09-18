# What's missing for the KPU simulator to execute programs

**Date:** 2026-09-18
**Scope:** all 74 open issues, prioritized against one goal — a working functional
TRANSACTIONAL simulator for the KPU that can execute programs: BLAS first, then MLP,
then progressively more complicated DNNs.
**Method:** five parallel read-only code investigations (execution engines and fidelity
tiers; the program load/execute pipeline; BLAS and dense linear algebra; the MLP → CNN →
transformer path; open-issue staleness), plus direct spot checks of the load-bearing
claims. Repo state: branch `docs/trim-claude-md-lazy-skills` at `eba0fde`; built binaries
under `build/` date from 2026-07-16..21, and no ISA, loader, runtime or bindings source has
changed since, so those binaries still match the code.

---

## Bottom line

**A functional transactional simulator that executes programs doesn't exist yet, but it's
closer than the issue list suggests.** A matmul program can already go from a file to
correct numbers at transactional fidelity. What's missing is not operators. It's one
agreed program format, one executor that takes it, and honest timing in that executor.

M1, M2 and M3 computed real values, but only as hand-built C++ graphs with synthetic
weights at toy sizes (ResNet is 16 channels x 4x4 by default; the demo says 224x224 is
"intractable cycle-by-cycle"). They ran on the cycle-by-cycle CSP engine, not on a
transactional tier.

The first decision is architectural. After that, the BLAS → MLP → CNN → transformer ladder
mostly re-routes work that already exists.

---

## Where things actually stand

| Capability | State | Evidence |
|---|---|---|
| Load a program file and execute it at transactional fidelity | **Works for matmul only** | `.kpuasm` → `kpu-assembler` → `.kpubin` → `kpu-loader --fidelity transactional`; a random 16x16x16 run matched the reference to float32 precision (`tools/runtime/kpu-loader/main.cpp:164-296`) |
| Transactional timing on that path | **Broken** | `ResourceTimeline::schedule_compute` is never called, so runs report "Compute cycles: 0" (`src/software/isa/transactional_program_executor.cpp:57,160`); timing is a post-hoc overlay with no credits or buffer capacity |
| Compute in program executors | **Matmul only** | hardwired triple loop (`src/software/isa/behavioral_program_executor.cpp:700-745`); bias and activation are serialized but no executor reads `ve_activation` |
| Programs produced by a compiler | **None can be loaded** | 4 incompatible formats (`.kpubin` DMProgram, `.kpukernel`, DFX `.kpu`, unsaveable L0/L1); `kpu-kernel-compiler` fails on its own test graph ("No matrix operations found in graph"); #144 throws on operand types 7-15 (`program_serializer.cpp:414`); committed `kernels/bin/*.kpubin` no longer load (opcodes renumbered with no version bump) |
| Execution engines | **8 overlapping engines** | the July plan counted 4; see the inventory below |
| Fidelity selection | **Decorative except one path** | honored only in `create_program_executor`; CYCLE_ACCURATE returns `nullptr` there (`program_executor_interface.cpp:140-143`); `create_compute_fabric` and `create_l3_tile` have no callers (only their interface declarations); zero references to fidelity anywhere in `include/sw/kpu/timing/` |
| Richest value path | **CSP `ConcurrentTimingExecutor`** | runs all M1-M3 operators with real credit, tag-CAM and queue contention; but no program front end, steps every cycle (`:978-980`), unbounded compute concurrency (`:1020-1036`), and compute latency comes from the K-slice count while ignoring tile size (`:1030-1033`) |
| L0 `TileProgram` (#230) | **In-memory reference only** | matmul plus LU with pivoting confined to the diagonal tile (`include/sw/kpu/program/derive/lu_tile_program.hpp:8,18`); not serializable, and nothing lowers it to any executor |
| Real model ingestion | **Absent in C++** | no safetensors/npy/onnx/state_dict readers in `include/` or `src/`; the Python `torch.compile(backend="kpu_transactional")` path recomputes every output in NumPy (`python/kpu/fx_converter.py:1987-1990`) and falls back to torch for unknown ops |
| Numeric types | **float32 only** | `DataType` has no float64 (`include/sw/kpu/data_types.hpp:31-49`); matters for LU/Cholesky/QR residuals |
| Operator coverage matrix | **45 done / 15 partial / 45 missing** of 105 cells | `tests/coverage/pattern_coverage.json`; done: matmul, conv2d, pooling, softmax, batchnorm, elementwise, broadcast, online_reduction; nothing done on the transformer side except softmax |

### Engine inventory

| Engine | Input | Values | Timing | Status |
|---|---|---|---|---|
| `BehavioralProgramExecutor` | DMProgram | yes (matmul + VE ops) | none | works; reachable via `kpu-loader` |
| `TransactionalProgramExecutor` | DMProgram | yes (delegates to behavioral) | post-hoc greedy timeline, no credits, compute = 0 cycles | partial; the only end-to-end transactional path |
| `isa::ConcurrentExecutor` | DMProgram | **no** | analytical | timing only; this is what `KPURuntime::launch` and the C API use |
| `isa::ProgramExecutor` (legacy) | DMProgram | yes | per-component FSMs | legacy |
| `KPUSimulator` (temporal) | imperative API | yes | per-component latency, no backpressure | components only, no program path |
| behavioral orchestrator / MLP executor | ad hoc | yes | none (instant) | isolated |
| OFG flow executors | OperandFlowGraph | **no** (`execute_operation` is a no-op) | credit/token | wired-not |
| **CSP `ConcurrentTimingExecutor`** + wrappers (`ScheduleExecutor`, `GraphCspExecutor`, Functional{MLP,DomainFlow,Elementwise,Reduction,Softmax}) | schedules / KernelGraph | yes, float32 payloads | cycle-stepped with credits, tag CAM, queues | partial; closest to the goal |
| Python `NativeKPURuntime` | DFX JSON | stats only; values recomputed in NumPy | transactional components | partial, bypasses the tiled machine |
| L0 `TileProgramReference` | TileProgram | yes | none by design | works as a reference |
| characterization harness | TileProgram + `DeviceDescriptor` | checks against the reference | analytical list schedule on finite tiles/lanes | works for design-of-experiments |

### Places where the simulator reports results it didn't compute

Anyone relying on these should know:

- **C API / `KPURuntime::launch`** validates argument addresses, then runs a timing-only
  executor; it never computes values (`src/software/runtime/runtime.cpp:138-165`).
  `examples/mlp/xor_classifier.cpp:15-21` says so outright.
- **Python `kpu_transactional` backend** computes values in NumPy; unknown ops run in torch.
- **Transactional executor** reports 0 compute cycles.
- **Fidelity settings** are ignored by the harnesses and `SimulatorConfig`, and
  CYCLE_ACCURATE silently falls back to TRANSACTIONAL in the compute-fabric and L3-tile
  factories.
- **Transactional memory controller** seeds its RNG from `std::random_device`, so its
  timing isn't reproducible.

### What M1/M2/M3 actually proved

| | Input | Weights | Oracle | Engine |
|---|---|---|---|---|
| M1 MLP | layers added by hand (`FunctionalMLPExecutor::add_layer`) | XOR by hand; MNIST-shape from a fixed-seed LCG | exact XOR outputs; host forward pass, `max_abs_err < 1e-4` | CSP, **single untiled tile per matrix** |
| M2 ResNet-18 | `build_resnet18` KernelGraph, hand-built | synthetic, seed 1000 | composed host conv2d + batchnorm reference, tol 5e-3 | CSP, T=16; default 16ch 4x4, `--full` only 8x8 |
| M3 MobileNetV2 / EfficientNet-B0 | KernelGraph, hand-built | synthetic | composed host reference, tol 5e-3 | CSP |

All four built demos pass (max errors 0.0 to 6e-8). Bridge limits that block real models:
every GEMM dimension must be divisible by T; only GLOBAL_AVG pooling exists, so ResNet's
stem maxpool can't run; the bridge accepts only CONV2D, ELEMENTWISE, POOL2D and MATMUL and
throws on anything else (including SOFTMAX, LAYERNORM, ATTENTION); nodes run sequentially
on a fresh executor each, so cycle totals are an upper bound.

---

## P0: the decision that blocks everything

**Which program format is the executable one, and which engine is the transactional
executor?**

| Option | For | Against |
|---|---|---|
| **A. DMProgram `.kpubin` + `TransactionalProgramExecutor`** | already runs end to end; serializer and assembler exist | `docs/plans/kpu-program-model.md` (D6) makes DMProgram the device-specific output of the driver JIT, not the portable program; timing is post-hoc with no credits; compute is hardwired matmul |
| **B. L0 `TileProgram` + a transactional executor built from `TileProgramReference` (values) and the characterization harness's tile-DAG scheduler with `DeviceDescriptor` (timing)** | matches the direction #229/#230 already committed to; tile-algorithm-native, which suits BLAS/PLASMA; event-driven, so it scales to sizes cycle-stepping can't reach; finite compute tiles and movement lanes are already modeled | no serializer yet; only matmul and LU tile kinds; DNN operators must be ported from the CSP runners; needs buffer/credit capacity and calibration |
| **C. CSP executor with a program front end** | richest value path; real credit contention | steps every cycle, so it is the cycle-accurate tier, not the transactional one |

**Recommendation: B for the transactional tier.** Keep C as the cycle-accurate reference
that the same L0 program lowers to later for calibration (#230 increment 2). Freeze A as a
stopgap regression path.

**The decision should also retire engines, not add one.** Retire or freeze the OFG flow
executors (no-op compute), the legacy `ProgramExecutor`, and the timing-only
`ConcurrentExecutor` behind the runtime (or point the runtime at the new executor). Retire
the unused `models/transactional` component classes, and label the NumPy Python path.

Note on layering: for transactional fidelity, executing L0 against a device descriptor is
enough. The driver JIT and DMProgram are what "hardware-identical execution" (#231) needs.

---

## Priorities by rung

### P0 — program contract (do first)

- **New issue:** decision record choosing the program format and the transactional engine,
  including the retirement list above.
- **#230:** split increments 3 (driver JIT) and 4 (versioned serialization) into their own
  issues. Fix the definition of done: it requires "neighbor-pivot LU", but the code only
  pivots within the diagonal tile.
- **#144:** the serializer drops operand types 7-15. Still on the path under option B,
  because DMProgram remains the JIT output.
- **New issue:** format versioning for opcodes, regenerating the broken `kernels/bin`
  corpus, and a compile → file → load → execute → check-values acceptance test.

### P1 — BLAS executes as a program

Definition of done: a GEMM program with random non-square inputs and ragged multi-tile
layouts runs from a file at transactional fidelity, matches a reference, and reports
non-zero compute time. Then TRSM, LU, Cholesky.

1. **GEMM:** #231 (bindings described by the program instead of hardcoded A/B/C with
   4/8/2 buffers), #74 with #114-#118 re-scoped to the program path. Add alpha/beta and
   trailing tiles that don't divide the dimensions (today T must divide M, K and N).
2. **New issue:** the transactional executor must model compute time, buffer/credit
   capacity, and deterministic seeds.
3. **New issue:** BLAS L1 (axpy, scal, dot, asum, nrm2) as named compositions of existing
   elementwise and reduce ops. Cheap, and it widens the test surface.
4. **TRSM:** only 2 of 8 variants exist, inside LU. **New issue** for standalone TRSM plus
   the triangular-substitution capability (P2 in the PLASMA plan). #244 covers multiple
   compute tiles and inter-tile reuse moves.
5. **Cholesky:** re-scope #242 — `DIV` and `SQRT` already exist as elementwise ops
   (`data_movement_isa.hpp:402-410`); what's missing is using them inside the
   factorization recurrence, plus the probe and capability gate. **New issue** for
   POTRF/SYRK tile kernels.
6. **Pairwise-pivot LU:** #243 (argmax with index, data-selected row swap). **New issue**
   for a float64 value path. QR later (no issue); SVD has no plan and no issue.

### P2 — MLP as a program with real weights

- **New issue:** execute bias, activation and elementwise/reduce ops read from the program.
- **#235:** weights and numerical validation. Put the trained-weight MNIST MLP here,
  checked against PyTorch logits plus a top-1 accuracy check. Use the existing Python FX
  converter as the interim front end until #233.
- **#79:** narrow to GELU/SiLU inside the fused bias+activation stage (ReLU already works).
- **New issue:** make the Python backend and C API return simulator-computed values, or
  label them timing-only.
- **Replace** the untiled single-tile MLP path with tiled GEMM plus ragged edge tiles.
- **Fold into the characterization harness or close:** #45, #48, #49, #50.

### P3 — CNN from a real model

- **Ingestion and hardware-style execution:** #233 (ONNX front end), #231 (weights in
  HOST_MEMORY).
- **Operators:** port conv2d, pooling (including max pooling, which the bridge lacks),
  batchnorm and softmax to the program format.
- **#75:** re-scope. Halo, grouped and dilated conv are follow-ons; ResNet and MobileNet
  don't need them.
- **#87:** keep for block-level validation; the ResNet block already runs.
- **#232** (move the backend to domain_flow): start only after the program contract is
  stable, or you'll be tracking churn in two repos.

### P4 — one transformer block (M4 → M5)

- **#78:** LayerNorm/RMSNorm value path, composed from REDUCE MEAN/VAR + SQRT/DIV +
  broadcast. Today only a kernel program and a COMPUTE-less generator exist.
- **#73, #110-#113:** reshape, head split, transpose. CSP has only a transpose flag for
  B-matrix moves; `BM_RESHAPE_TILE` is handled only by the ISA executors.
- **#81:** attention as chained GEMMs (QK^T with transpose, scale, additive -inf mask,
  existing softmax, then xV). `Kernel::create_attention` is a matmul-shaped placeholder.
- **#80:** FFN, plus a GELU op — VEOp has no ERF or TANH, and `FunctionalActivation` is
  NONE/RELU only.
- **#85:** patchify, for ViT.
- **Milestones:** #132 (M4), #133 (M5).
- **Housekeeping:** add M3 and M5 gates to `pattern_coverage.json` (neither exists);
  update #88's stale checklist; fix the stale "SiLU approximated by ReLU6" banner in
  `m3_efficientnet.cpp:18` (the graph uses real SILU).

### P5 — LLM token decode (M7)

- **#70, #94-#98:** gather. `DMA_LOAD_GATHER`/`DMA_STORE_SCATTER` opcodes exist but do
  nothing (`behavioral_program_executor.cpp:251-255`).
- **#84:** embedding and logits, including a strongly rectangular logits GEMM.
- **#82:** KV-cache append into a growing extent (new ISA plus executor support).
- **#83:** RoPE — needs sin/cos ops, which don't exist.
- **#243:** argmax, which also blocks sampling.
- **Milestone:** #135 (M7).
- **New issue:** host-driven decode loop that keeps state between tokens.

### Parked (not on the path)

- Later milestones: #134 (YOLOv8n), #136 (I-JEPA), #86 (JEPA masking).
- #234 program cache (optimization).
- #43 NoC wiring (revisit if #244 needs inter-tile moves), #44 L2 interconnect.
- Tech debt: #42, #22.

---

## Close now (21 issues)

The PRs used "Relates to #N" instead of "Closes #N", so GitHub never closed these. The
coverage matrix `tests/coverage/pattern_coverage.json` is the authoritative record.

| Issue | Why | Evidence |
|---|---|---|
| #76 | pooling epic, all 5 stages done | PRs #190, #196-#199 |
| #120 | conv2d-T2 | PR #174, `conv2d_im2col.hpp` |
| #121 | conv2d-T3 | PR #175 |
| #122 | conv2d-T4 | PR #176, `test_functional_conv2d.cpp` |
| #123 | conv2d-T5 | PR #177, `test_conv2d_regression.cpp` |
| #124 | pool-T1 design | PR #190, `docs/plans/e7_pooling_pattern.md` |
| #125-#128 | duplicates of #192-#195 | same work, done in PRs #196-#199 |
| #129 | M1 MLP; coverage gate `achieved: true` | PR #137, `docs/milestones/M1_mlp_baseline.md`; trained weights belong to #235 |
| #179-#182 | batchnorm T2-T5 | PRs #183-#186 |
| #188 | epilogue-fused-T5 | PR #189, `test_epilogue_fused_regression.cpp` |
| #192-#195 | pooling T2-T5 | PRs #196-#199 |
| #47 | superseded | fusion works via `MatMulComputeSpec` bias+activation; D6 makes DMProgram the JIT layer; the E10 plan says it absorbs #47 |

Audit totals across all 74 open issues: 16 done-not-closed, 4 duplicates, 1 superseded,
13 partial, 40 genuinely open.

Process fix: use "Closes #N" in sub-task PRs so this doesn't build up again.

---

## New issues to file (no issue covers these today)

1. Decision record: the executable program format and the transactional engine, plus the
   engine retirement list.
2. Transactional executor: model compute time, buffer/credit capacity, deterministic seeds.
3. Program format versioning + golden corpus regeneration + round-trip acceptance test.
4. Execute bias/activation and VE ops carried in the program.
5. Host API honesty: C API and Python return simulator values, or are labeled timing-only.
6. BLAS L1 named compositions (axpy, scal, dot, asum, nrm2).
7. Standalone TRSM (8 variants) + triangular-substitution capability (PLASMA P2).
8. POTRF/SYRK tile kernels.
9. float64 value path.
10. Host-driven LLM decode loop with state carried between tokens.

---

## Verification notes

Spot-checked directly in the source rather than taken from the sub-reports:
`program_executor_interface.cpp:140-143` (CYCLE_ACCURATE returns `nullptr`);
`schedule_compute` defined but never called; no `ve_activation` reader in either executor;
`create_compute_fabric`/`create_l3_tile` have only interface declarations, no callers;
zero fidelity references under `include/sw/kpu/timing/`; `lu_tile_program.hpp:8,18`
(confined, not pairwise, pivoting); no float64 in `DataType`; `fx_converter.py:1987-1990`
(NumPy recomputation); `graph_csp_executor.hpp:259-260` (throws on other op types); no
weight loaders in `include/` or `src/`; PRs #174 and #196-#199 say "Relates to";
coverage counts 45/15/45.

Two sub-reports disagreed on two points, resolved as follows: the 16x16x16 loader run
matched the reference to float32 rounding (0.0 against one reference, 6.4e-7 against
another), and #230's "neighbor-pivot LU" definition of done is **not** met — the code
implements confined pivoting only.

Unverified: whether domain_flow (#232/#233) has matching work, checked only through a
`gh pr list` search on `branes-ai/domain_flow` with no local clone.
