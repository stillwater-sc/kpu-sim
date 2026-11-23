# `examples/` inventory and assessment

**Date:** 2026-11-23 · **Main at:** `58fa5c1` · **Scope:** every tracked file under `examples/`
(167 files: 60 C/C++ sources, 23 Python scripts, 2 shell scripts, the rest data, logs,
images and build files).

## How this was assessed

Each example was judged on four things, all checked in this session, not inferred from names:

1. **Engine.** Which simulator path it drives, from its `#include`s, compared against the
   engine dispositions in ADR 0001 D6 (`docs/architecture/adr/0001-program-contract-and-transactional-engine.md`)
   and the retirement tracker #266.
2. **Values.** Whether it computes values and checks them. CLAUDE.md treats a level that returns
   timing without values as unvalidated.
3. **Runs.** Every built C/C++ binary in `build/examples` (built 2026-11-16) was run with a
   60 s timeout. Every Python script except `python/generate.py` and
   `analysis/visualize_tiling.py` was run with a 120 s timeout, using `python/` and the
   `stillwater_kpu` binding on the path. `dfg/matmul.sh` was run. Not run, and graded from
   their source and build files alone: the six C/C++ sources with no build target, those two
   Python scripts, and `pipelines/matmul_pipeline.sh`. Each table's Run column says which.
4. **Vocabulary.** A grep for the forbidden cache terms (`cache hit/miss`, `evict`, `LRU`,
   `refetch`, `overfetch`).

Git dates are shown as *created → last touched*. Many of the "last touched" dates in June and
July 2026 come from repo-wide mechanical commits (include restructuring, `-Werror`, the
Combiner rename), so they don't mean the example was maintained.

### Verdicts

| Verdict | Meaning |
|---|---|
| **Current** | On a kept engine (CSP `ConcurrentTimingExecutor`, the program stack, domain-flow SURE). Computes and checks values. Runs clean. |
| **Current (component)** | Exercises the `KPUSimulator` component library. ADR D6 keeps that library for fidelity elevation, and `docs/ONBOARDING.md` §"About examples/" scopes these examples to *learning a component's interface*, not to showing how programs run. |
| **Review** | Works, but it is a judgment call: an analytical model, a frozen Python path, or no values checked. |
| **Outdated** | One or more of the reason codes below apply. |

| Code | Reason |
|---|---|
| **F** | Uses an engine that ADR 0001 D6 and #266 freeze and schedule for retirement. |
| **T** | Timing only: it reports cycles but computes or checks no values. |
| **B** | Broken: it crashes, times out, gives wrong values, or calls an API that no longer exists. |
| **C** | Uses cache semantics (overfetch/refetch/evict), which CLAUDE.md forbids. |
| **O** | Orphan: not built, a duplicate, or a committed generated artifact. |

## Summary

| Verdict | Source files (85 total) |
|---|---|
| Current | 13 |
| Current (component) | 10 |
| Review | 20 |
| Outdated | 42 |

The program stack is barely represented. **Only `characterize/tile_characterize.cpp` drives the
CSP program path** (`csp/gen`, `csp/lang/parse`, `driver/csp_run`, the virtual platform,
`tile_check`), which is the path ONBOARDING §4 teaches. No example runs a `.csp` program at
L-B, L-T1 or L-CA, or shows the L-B oracle with the per-tile `atol + rtol·|o|` check.

The findings that matter most are **silent failures**: examples that exit 0 while their own
output is wrong.

- `c/hello_kpu.c` prints `Error at C[0]: expected 32.0, got 0.0` (five errors) and exits 0.
- `mlp/mnist_classifier.cpp` predicts class 0 for every digit (10 % accuracy) and exits 0.
- `mlp/sine_approximation.cpp` outputs a column of `0.0000` and exits 0.
- `runtime/runtime_demo.cpp` logs 11 `KPURuntime::free: failed to deallocate` and exits 0.
- `basic/data_movement_pipeline.cpp` aborts (`std::out_of_range: Memory write out of bounds`).
  The top-level `README.md:310` links to it as *the* data-movement example.

## C/C++ examples

### `basic/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `hello_kpu.cpp` | 2025-10 → 2026-07 | `KPUSimulator` | rc 0 | **Current (component)** | Minimal component tour; still runs. |
| `memory_management.cpp` | 2025-10 → 2026-07 | `KPUSimulator` | rc 0 | **Current (component)** | Memory-component API demo. |
| `matrix_multiply.cpp` | 2025-10 → 2026-07 | `KPUSimulator` | rc 0, prints C | **Current (component)** | `README.md:306` links to it. |
| `host_to_l3_matmul.cpp` | 2025-12 → 2026-07 | `KPUSimulator` | rc 0, `Result: PASS` | **Current (component)** | Checks its values. |
| `resource_api_demo.cpp` | 2025-12 → 2026-07 | `resource_api` | rc 0 | **Current (component)** | Resource API demo. |
| `system_config_demo.cpp` | 2025-10 → 2025-12 | `system/config_loader` | rc 0 | **Current (component)** | Config loading is still how machines are described. |
| `config_formatter_demo.cpp` | 2025-10 | `system/config_formatter` | rc 0 | **Current (component)** | Same. |
| `tile_layout_test.cpp` | 2025-12 → 2026-06 | `isa/tile_layout` | rc 0 | **Current (component)** | Layout-policy comparison. Named `_test` but not registered as a test. |
| `data_movement_pipeline.cpp` | 2025-10 → 2026-07 | `KPUSimulator` | **rc 134, aborts** | **Outdated — B** | `Memory write out of bounds`. `README.md:310` links to it. |
| `data_movement_isa_matmul.cpp` | 2025-12 → 2026-06 | legacy `isa::ProgramExecutor` | rc 0 | **Outdated — F** | ADR D6 names this file as retiring together with `ProgramExecutor`. |
| `concurrent_execution_debug.cpp` | 2025-12 → 2026-06 | `isa::ConcurrentExecutor` | rc 0 | **Outdated — F, T** | Frozen, timing-only executor. |
| `kernel_graph_demo.cpp` | 2025-12 → 2026-06 | `KernelGraph` + `isa::ConcurrentExecutor` | rc 0 | **Outdated — F, T** | Frozen executor. ADR D7 rules out KernelGraph-only paths. |
| `kernel_demo.cpp` | 2025-12 | — | not built | **Outdated — O** | `basic/CMakeLists.txt:138` says it moved to `kernel/matmul_kernel.cpp`. A dead duplicate. |
| `data_movement_schedule.txt` | 2025-12 | — | — | **Outdated — O** | Committed output of `data_movement_isa_matmul`. |

### `behavioral/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `matmul_behavioral.cpp` | 2026-01 | `models/behavioral` orchestrator | rc 0, PASS; ctest | **Outdated — F** | ADR D6 names it as the orchestrator's only external dependent. Retires with it. |
| `xor_behavioral.cpp` | 2026-01 | `models/behavioral/mlp_executor` | rc 0, PASS; ctest | **Outdated — F** | Same frozen family. |
| `ofg_trace_demo.cpp` | 2026-01 | `models/dataflow` OFG flow executors | rc 0 | **Outdated — F, T** | `execute_operation` is a no-op. ADR D6 and #266 name this file. CLAUDE.md lists the path under AVOID. |
| `tiled_matmul_trace.cpp` | 2026-01 → 2026-06 | `kpu/behavioral/tiled_matmul_program` | rc 0 | **Outdated — T** | Cycles only. Reports `Pipelined cycles == Sequential cycles` (speedup 1.00x), so its pipeline model shows no overlap. Its header carries "legacy" fields. |

### `blas/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `big_matmul.cpp` | 2025-12 → 2026-07 | `KPUSimulator` + `trace/*` | rc 0, `Result: PASS`, 2.8 s | **Current (component)** | Checks its values. |
| `block_systolic_matmul.cpp` | 2025-12 → 2026-06 | `models/temporal/datamovement`, `kpu/dataflow/block_mover_compiler`, wormhole NoC | rc 0, "systolic order check PASSED" | **Review** | Not frozen, but it moves tile identity only and checks no values. The BlockMover-ISA compiler predates CSP. |
| `trace_infrastructure.cpp` | 2025-12 | `trace/*` | **not built** | **Outdated — O** | No CMake target (`blas/CMakeLists.txt` builds only two targets). |
| `big_matmul_fsim_bug.txt` | 2025-12 ("WIP: naive big matmul") | — | — | **Outdated — O** | A bug note kept as an example file. Belongs in an issue or `docs/`, if it is still open (not checked). |

### `c/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `hello_kpu.c` | 2026-01 → 2026-06 | C API → `isa::ConcurrentExecutor` | **rc 0 but wrong values** | **Outdated — B, F, T** | Prints `expected 32.0, got 0.0` and still exits 0. The C API sits on the frozen executor (ADR D6, #257). |
| `runtime_demo.c` | 2026-01 → 2026-06 | C runtime API | rc 0, cycles only | **Outdated — F, T** | Same backend. Reports cycles but no values. |

### `characterize/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `tile_characterize.cpp` + `README.md` | 2026-07 → 2026-10 | program stack: CSP gen/parse, `csp_run`, virtual platform, `tile_check` | rc 0; 4 ctests | **Current** | The one example on the program path. Validates values and takes `--deploy` machine specs. The model to copy. |

### `compiler/`

All created Nov 2025, before the CSP program stack. They are analytical planners, so values do
not apply, and nothing in the current program path consumes them.

| File | Run | Verdict | Why |
|---|---|---|---|
| `tile_optimizer_demo.cpp` | rc 0 | **Review** | Pre-CSP tile-size planner. Probably superseded by the characterization harness. |
| `schedule_generator_demo.cpp` | rc 0 | **Review** | `compiler/schedule_generator`. ADR D7 says operators don't become new schedule generators. |
| `schedule_characterizer_demo.cpp` | rc 0 | **Review** | Pareto sweep. Overlaps `characterize/`. |
| `hardware_design_explorer.cpp` | rc 0 | **Review** | Design-space sweep. Overlaps `characterize/`. |
| `l2_tile_scheduler_demo.cpp` | rc 0 | **Outdated — C** | 6 cache-term hits. |
| `dataflow_strategy.cpp` | rc 0 | **Outdated — C** | 3 cache-term hits. Prints only 6 lines. |
| `l3_overfetch_analyzer.cpp` | rc 0 | **Outdated — C** | 39 hits. "Overfetch" (refetch) is the analysis itself. |
| `l3_comprehensive_analysis.cpp` | **timeout (>60 s, no output)** | **Outdated — B, C** | 36 cache-term hits. |
| `l3_focused_analysis.cpp` | **timeout (>60 s, no output)** | **Outdated — B, C** | 14 cache-term hits. |
| `l3_transformer_analysis.cpp` | **timeout (>60 s)** | **Outdated — B, C** | 17 cache-term hits. Never finishes the LLaMA-2 case. |
| `hardware_design_space.csv`, `hardware_design_space_visualization.png` | — | **Outdated — O** | Committed generated output. |

### `dataflow/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `fused_mlp_sure_demo.cpp` | 2026-06 | `kpu_dataflow` FusedMlpSure (domain flow) | rc 0, `RESULT: PASS`; ctest | **Current** | Self-validating domain-flow SURE. |

### `kernel/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `matmul_kernel.cpp` | 2025-12 → 2026-06 | `KernelCompiler` + `isa::ConcurrentExecutor` | rc 0, cycles only | **Outdated — F, T** | Frozen, timing-only executor. |
| `mlp_kernel.cpp` | 2025-12 → 2026-11 (Combiner rename) | same + SFU | rc 0, cycles only | **Outdated — F, T** | Same. Its last commit was the rename, not maintenance. |

### `milestones/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `m1_mlp_baseline.cpp` | 2026-07 | `timing/functional_mlp_executor` | rc 0, ALL PASS vs host ref; ctest | **Current** | Values checked (tolerance 1e-4). Caveat: the oracle is a host reference, not L-B. |
| `m2_resnet.cpp` | 2026-07 | `KernelGraph` + `timing/graph/resnet18` | rc 0, ALL PASS; ctest + regression | **Current** | Same caveat. Also a KernelGraph front end (ADR D7). |
| `m3_mobilenet.cpp` | 2026-07 | `timing/graph/mobilenetv2` | rc 0, ALL PASS; ctest | **Current** | Same caveats. |
| `m3_efficientnet.cpp` | 2026-07 | `timing/graph/efficientnet` | rc 0, ALL PASS; ctest | **Current** | Same caveats. |

### `mlp/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `xor_classifier.cpp` | 2026-01 → 2026-07 | `runtime/graph` → `isa::ConcurrentExecutor` | rc 0 | **Outdated — F, T** | Prints "TIMING MODEL only". Points to `docs/FUNCTIONAL_SIMULATION_GAP_ANALYSIS.md`, which doesn't exist. |
| `sine_approximation.cpp` | same | same | **rc 0, output column all `0.0000`** | **Outdated — B, F, T** | Computes nothing. |
| `mnist_classifier.cpp` | same | same | **rc 0, predicts 0 for every class (10 %)** | **Outdated — B, F, T** | Computes nothing, yet prints a PASS/FAIL table. |

### `runtime/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `runtime_demo.cpp` | 2025-12 → 2026-07 | `KPURuntime` → `isa::ConcurrentExecutor` | **rc 0, 11 "failed to deallocate"** | **Outdated — B, F, T** | Frozen backend. Its frees fail silently. |

### `schedule/`

| File | Dates | Engine / API | Run | Verdict | Why |
|---|---|---|---|---|---|
| `csp_pipeline_demo.cpp` | 2026-02 → 2026-06 | `ConcurrentTimingExecutor`, `process_interface` | rc 0 | **Review** | Kept engine, but no values: it teaches credits and the TagCAM. A good onboarding companion. |
| `csp_schedule_demo.cpp` | 2026-02 → 2026-07 | `timing/schedule` generators + validator | rc 0, "Validation: PASSED" | **Review** | Kept engine, but no values. "Validation" is `validate_schedule`, a structural check of each generated schedule, and the demo exits 0 even when it prints FAILED. |
| `run_matmul.cpp` | 2026-02 → 2026-07 | `ConcurrentTimingExecutor` + schedule executor | rc 0 | **Review** | Kept engine, but timing only: it reports cycles and a status, with no value check. |
| `unified_xor_mlp.cpp` | 2026-07 | `timing/functional_mlp_executor` | rc 0, PASS; ctest | **Current** | |
| `matmul_simulator.cpp`, `softmax_simulator.cpp`, `layernorm_simulator.cpp` | 2026-07 | `ConcurrentTimingExecutor` + tile tracker + functional oracle | rc 0, `… OK`; ctests | **Current** | |
| `conv2d_simulator.cpp`, `batchnorm_simulator.cpp`, `pooling_simulator.cpp` | 2026-07 | same | rc 0, `… OK`; ctests | **Current** | Caveat: these are per-operator schedule generators, the pattern ADR D7 stops adding to. They are kept as cycle-accurate implementations. |
| `run_matmul_DEPRECATED.cpp` | 2026-02 → 2026-07 | `Behavioral`/`TransactionalProgramExecutor`, `kpu/dsl` | rc 0 | **Outdated — F** | Deprecated by design ("what NOT to do"). Uses two engines that D6 retires, yet is still built as a target. |

### `test/`

| File | Dates | Run | Verdict | Why |
|---|---|---|---|---|
| `debug_blockmover_delivery.cpp`, `debug_deadlock.cpp`, `debug_noc.cpp`, `simple_array.cpp` | 2025-12 → 2026-01 | **not built** | **Outdated — O** | `examples/CMakeLists.txt` never adds `test/`. These are one-off debugging programs. |
| `debug_blockmover_delivery`, `debug_deadlock`, `debug_noc` | 2025-12 | — | **Outdated — O** | **Committed ELF executables.** They should not be in git. |

## Analysis, scripts and Python

### `analysis/`

| File | Run | Verdict | Why |
|---|---|---|---|
| `matmul_schedule_analysis.cpp` | rc 0 | **Review** | Header-only analytical tiled-matmul model, with no simulation. Useful for intuition. Not tied to the program stack. |
| `visualize_tiling.py`, `tiling_configurations.csv`, `plots/` (12 svg/png) | not run | **Review** | The plotting companion, plus committed generated plots. |

### `dfg/`, `pipelines/`

| File | Run | Verdict | Why |
|---|---|---|---|
| `dfg/matmul.sh` | rc 0, using `build/tools/dfg` | **Review** | `kpu-dfg-*` toolchain: DFG → schedule → BlockMover programs. A parallel front end to the CSP path. Timing only. |
| `pipelines/matmul_pipeline.sh` | not run | **Review** | Same toolchain. Needs `scripts/kpu-env.sh` and installed tools. |

### `python/` (native `stillwater_kpu` binding)

| File | Run | Verdict | Why |
|---|---|---|---|
| `address_based_dma_example.py` | rc 0 | **Current (component)** | Uses the binding's `KPUSimulator`/DMA API, which still exists. |
| `baseline/basic_sim_pattern.py`, `core_features_demo.py`, `show_np_config.py` | **rc 1** | **Outdated — B** | `AttributeError: … has no attribute 'core'`. Written 2025-08 against an older binding. |
| `baseline/simple_kpu.py`, `educational/matrix_chain_demo.py`, `neural_network_demo.py`, `performance_scaling.py` | **rc 1** | **Outdated — B** | `… has no attribute 'Simulator'` (now `KPUSimulator`). |
| `kernel_graph_example.py`, `test_graph_validation.py`, `test_kernel_validation.py`, `test_compiler_regression.py` | **rc 1** | **Outdated — B, F** | `Kernel`, `KernelGraph` and `KernelCompiler` are no longer bound. These targeted the frozen kernel path. |
| `generate.py` | not run | **Outdated — O** | 32-line torchvision/transformers export snippet. No KPU use. |
| `README.md` | — | **Outdated** | Describes the broken scripts above. `README.md:358` points readers here. |

### `fusion/`, `torch/`, `applications/` (the `kpu` package in `python/kpu`, v0.8.0)

| File | Run | Verdict | Why |
|---|---|---|---|
| `fusion/ffn_fusion.py`, `fusion/fusion_showcase.py` | rc 0 | **Review** | They work. `kpu` runs on the Python native path, which ADR D6 freezes pending #257 ("Host API must return simulator-computed values", open). |
| `torch/resnet18_inference.py`, `mobilenetv2_inference.py`, `vit_inference.py`, `fasterrcnn_components.py` | rc 0 | **Review** | `torch.compile` KPU backend. Same #257 dependency. |
| `torch/model_compatibility.py` | **rc 1** | **Outdated — B** | Exits 1 right after "Testing 35 models…" with no error text. Not diagnosed. |
| `applications/image_classification/classify_image.py`, `applications/hybrid_pipeline/numpy_torch_pipeline.py` | rc 0 | **Review** | Same #257 dependency. |
| `*.log` (3, one empty), `.cache/` (41 jpg + class lists across `torch/` and `applications/`) | — | **Outdated — O** | Committed run output and downloaded images. They belong in `.gitignore`. |
| `torch/README.md`, `applications/README.md` | — | **Review** | Not checked line by line. `python/examples/` also exists, which may duplicate these. |

## Recommendations, in priority order

1. **Fix or remove the silently wrong examples**: `c/hello_kpu.c`, `mlp/*`,
   `runtime/runtime_demo.cpp`. An example that exits 0 on wrong values reads as passing. At
   minimum, they should exit non-zero on a mismatch.
2. **Fix `basic/data_movement_pipeline.cpp`, or repoint `README.md:310`.** It is the
   top-level README's data-movement example, and it aborts.
3. **Delete the orphans and the committed artifacts**: the three ELF binaries in `test/`, the
   unbuilt `test/*.cpp`, `basic/kernel_demo.cpp`, `blas/trace_infrastructure.cpp`, the `*.log`
   files, the `.cache/` images, and the generated CSV/PNG/TXT outputs. Add `.gitignore` rules
   so they don't return. This is a one-concern PR with no model changes.
4. **Retire the F-coded examples together with their engines under #266**, not ahead of them.
   ADR D6 already names `data_movement_isa_matmul`, `ofg_trace_demo` and `matmul_behavioral`.
   Add `xor_behavioral`, `kernel/*`, `concurrent_execution_debug`, `kernel_graph_demo`,
   `run_matmul_DEPRECATED`, `c/*`, `mlp/*` and `runtime/*` to #266's dependents so they leave
   with their engines.
5. **Retire or rewrite the `compiler/l3_*` analyzers.** Three of them
   (`l3_comprehensive_analysis`, `l3_focused_analysis`, `l3_transformer_analysis`) time out.
   All four, `l3_overfetch_analyzer` included, model "overfetch", which is cache semantics.
   `characterize/` already covers the design-space role.
6. **Decide what to do about the Python `baseline/`, `educational/` and `test_*` scripts**:
   rewrite them against the current binding, or delete them. They have failed since the
   binding's API changed.
7. **Gap: add program-stack examples.** At least one example should run a `.csp` program at
   L-B, L-T1 and L-CA, compare per tile against L-B, and show L3 placement. Today
   `tile_characterize` is the only one on that path, and the ONBOARDING note warns readers off
   the rest of `examples/`.

## Not verified

- `build/` dates from 2026-11-16. The sources the binaries were built from may differ slightly
  from main at `58fa5c1`.
- I didn't diagnose why `torch/model_compatibility.py` fails, or why the `l3_*` analyzers hang
  (a slow sweep or a real hang). I didn't check whether `big_matmul_fsim_bug.txt` is resolved.
- `pipelines/matmul_pipeline.sh` and `analysis/visualize_tiling.py` were not run.
