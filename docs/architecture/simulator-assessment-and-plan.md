# KPU Simulator: State Assessment and Implementation Plan

**Scope:** This document assesses the current state of the KPU simulator — its
capabilities, gaps, and shortcomings — against the target of a *functional,
transaction-ordered simulator* with resources and instruction set exposed as a
target for (a) a runtime executing KPU assembly programs and (b) a compiler
lowering high-level graphs (full DNNs, dense linear algebra factorizations,
signal processing operators, and Kalman-filter constraint solvers). It then
lays out an implementation plan from current state to that goal.

**Baseline:** v0.8.5 released (2026-02-04); v0.9.0 "Concurrent Timing Model
(CSP Architecture)" in active development. This document does not restate the
execution semantics or fidelity definitions — the authoritative references
remain [`docs/01-architecture/kpu-execution-model.md`](../01-architecture/kpu-execution-model.md)
(credit-based dataflow, buffers-not-caches) and
[`docs/02-simulation/fidelity-framework.md`](../02-simulation/fidelity-framework.md)
(BEHAVIORAL / TRANSACTIONAL / CYCLE_ACCURATE). Structural diagrams are in
[`class-diagram.md`](class-diagram.md); the detailed per-subsystem evidence
behind this assessment (file-level inventories with citations) is in
[`subsystem-assessment-details.md`](subsystem-assessment-details.md).
The canonical release plan is
[`docs/ROADMAP.md`](../ROADMAP.md); this document proposes how the remaining
work should be organized against the stated goal.

---

## 1. The Goal, Restated Precisely

KPU behavior is dependent on spatial and temporal sequencing: which tile is in
which buffer, in what order credits return, when a distributed wavefront fires. A
purely functional (instant, program-order) behavioral simulator cannot serve
as the ground truth for software bring-up, because programs that are
functionally "correct" under instant semantics can deadlock, livelock, or
compute against stale buffers under real credit-ordered semantics.

Therefore the target is a **transactional-functional simulator**: one engine
that simultaneously

1. **computes actual values** (tiles carry real data; compute performs real
   arithmetic with correct K-accumulation), and
2. **respects credit-based transaction ordering** (push-with-credit, tag-CAM
   matching, backpressure — the model defined in `kpu-execution-model.md`),

with two stable consumer-facing surfaces:

3. a **runtime target**: load a `.kpuasm`/`.kpubin` program through the C API,
   execute it, read back numerically correct results plus timing; and
4. a **compiler target**: a documented IR/ISA contract such that graph
   lowering (DNNs, Cholesky/QR/LU/SVD, convolutions/filters, EKF/UKF/multiscale
   Kalman) produces programs that run on that engine.

---

## 2. Current State: The Execution Engine Inventory

The single most important finding: the repository does not contain one
simulator with a fidelity knob. It contains **four parallel execution
engines**, split cleanly along the values/ordering axis. The engines that
compute real values do not model credit ordering; the engines that model
credit ordering do not carry data.

| Engine | Location | Computes values? | Models credit ordering? | Consumers |
|---|---|---|---|---|
| Behavioral tier (`BehavioralOrchestrator`, `BehavioralProgramExecutor`, `BehavioralComputeFabric`) | `include/sw/kpu/models/behavioral/`, `include/sw/kpu/isa/behavioral_program_executor.hpp` | **Yes** | **No** — instant, static program order ("no timing, no queues, no credit tracking", `behavioral_program_executor.hpp:41-43`) | Python bindings, kpu-loader, examples, tests |
| Main `KPUSimulator` (temporal tier) | `include/sw/kpu/kpu_simulator.hpp`, `src/system/simulator/kpu_simulator.cpp` | **Yes** (real MAC in `src/models/temporal/compute/compute_fabric.cpp:205`) | Partial — per-component latency FSMs, **no credit backpressure** between stages | C API legacy path, examples |
| OFG dataflow executors | `include/sw/kpu/models/dataflow/` | **No** (tokens/events; `execute_operation` is a no-op in the base, `flow_graph_executor.hpp:342`) | **Yes** — credits, buffer tokens, deadlock detection | `tests/dataflow/` only — never used from `src/` |
| CSP timing tier (v0.9) | `include/sw/kpu/timing/`, `ConcurrentTimingExecutor` | **No** — moves `TileDescriptor` metadata only; no payload field exists (`tile_descriptor.hpp:115-154`) | **Yes** — CreditPool + TagCAM + work-conserving queues, livelock detection | `examples/schedule/`, timing tests |

Two further disconnections compound this:

- **The runtime wall.** A complete assembler → `.kpubin` → functional-executor
  chain exists in C++, and a CUDA-like C runtime API exists, but they are not
  connected. The C API cannot load an assembled program, and its execution
  engine (`ConcurrentExecutor`) is timing-only.
- **The compiler wall.** The intended hardware-agnostic compiler IR
  (tile-level DFX, "the PTX-equivalent layer") is implemented on the front end
  (`.dfg` → DFX → `.kpu` object) but has no lowering to an executable
  `DMProgram`. The only full network-to-execution path is the Python
  torch.fx/ONNX route, which executes at op granularity on the behavioral
  fabric — bypassing the tiled, credit-ordered machine model entirely.

The rest of Section 3 assesses each subsystem; Section 4 consolidates the
gaps; Section 5 is the plan.

---

## 3. Subsystem Assessments

### 3.1 Instruction Set and Assembler — strong foundation, minor drift

**Exists and works:**

- A substantial data-movement ISA (`include/sw/kpu/isa/data_movement_isa.hpp`):
  ~50 opcodes across DMA (External↔L3), BlockMover (L3↔L2), Streamer
  (L2↔L1/systolic feed), sync (BARRIER/WAIT/SIGNAL), configuration register
  setup (SET_BASE/SET_STRIDE/SET_TILE_DIM/SET_MATRIX_DIM), hardware loops with
  `IndexRole` (TI/TJ/TK) for AUTO address generation, Vector Engine
  (VE_ELEMENTWISE/VE_REDUCE), and L2 scratch access. Consistent with the
  Domain Flow Architecture: the program IS the data-movement schedule; compute
  fires reactively on data arrival.
- A real assembler (`src/software/isa/assembler.cpp`, ~1150 lines) covering
  every specified opcode/directive, with CLI tool `kpu-assembler`.
- A binary/JSON serializer with round-trip `.kpuasm → .kpubin → DMProgram`
  (`src/software/isa/program_serializer.cpp`).
- `OutputStationaryProgramBuilder` for programmatic matmul schedule emission —
  in practice the primary program-generation path.

**Shortcomings:**

- **Spec drift.** `docs/kpuasm-specification.md` (v1.0) lags the
  implementation: AUTO addressing, `IndexRole` on LOOP_BEGIN, and the enhanced
  SET_* configuration opcodes are implemented but undocumented; the spec's
  `SET_STRIDE` form differs from the implemented one.
- **Encoding is a serialization format, not an ISA word format.** Instructions
  are C++ structs with a `std::variant` of operand payloads. Fine for a
  simulator target; a hardware-faithful bit encoding is future work and should
  be an explicit non-goal until the semantics stabilize.
- **Annotation-only opcodes.** `DMA_LOAD_GATHER`/`DMA_STORE_SCATTER`,
  `STR_BROADCAST_*`, `VE_ELEMENTWISE`/`VE_REDUCE`, and `L2_SCRATCH_*` are
  deliberate no-ops in the behavioral executor
  (`behavioral_program_executor.cpp:221-235`). These are precisely the opcodes
  the advanced operator classes (factorizations, filters, Kalman) will need.

### 3.2 Runtime C API — implemented shell, disconnected engine

**Exists and works:** the full CUDA-like surface
(`kpu_c_runtime.h`: malloc/free, memcpy h2d/d2h/d2d, launch, streams, events;
`kpu_c_executor.h`: graph executor with set_input/get_output;
`kpu_c_kernel.h`: matmul/MLP kernel factories), implemented in
`src/tools/bindings/c/`, sitting on a functional `ResourceManager`
(`resource_api.hpp`, `resource_handle.hpp`) that exposes every resource class
the goal requires: host/external memory, L3 tiles, L2 banks, L1 buffers, page
buffers, compute tiles, DMA engines, block movers, streamers — with
allocation, read/write, status, and statistics.

**Critical gaps:**

1. **No program loading.** There is no C API to load a `.kpuasm`/`.kpubin` or
   a serialized `DMProgram`/`Kernel`. Kernels are only constructible via the
   built-in `create_matmul`/`create_mlp` factories. The assembler output is
   unreachable from the runtime.
2. **The runtime never computes.** `kpu_runtime_launch` routes to
   `ConcurrentExecutor::execute` (`src/software/runtime/runtime.cpp:164`),
   which is a resource-occupancy timing scheduler with no reference to memory
   contents. `kpu_executor_get_output` returns whatever bytes happen to be in
   device memory. The value-producing executors
   (`BehavioralProgramExecutor`, `TransactionalProgramExecutor`) are never
   referenced from `src/software/runtime/` or the bindings.
3. **Fidelity factory incomplete.** `create_program_executor(fidelity, hw)`
   dispatches BEHAVIORAL and TRANSACTIONAL but returns `nullptr` for
   CYCLE_ACCURATE (`src/software/isa/program_executor_interface.cpp:140-143`),
   and is itself not called by the runtime.
4. **Type-system split.** `ConcurrentExecutor` uses its own internal
   `ResourceType` model, distinct from `resource_handle.hpp` — the timing
   model and the resource API do not share a resource vocabulary.

### 3.3 Behavioral Tier — correct values, wrong ordering model

The behavioral tier genuinely computes: `BehavioralComputeFabric` dispatches
real typed kernels (matmul, conv2d, softmax, layernorm, elementwise, etc.);
`BehavioralOrchestrator`/`BehavioralMLPExecutor` run full forward passes;
`BehavioralProgramExecutor` executes DMPrograms against real L1/L2/L3/external
memory with verified numerical correctness (`tests/isa/test_behavioral_program_executor.cpp`).

But all of it is instant and program-ordered. There is no credit flow, no
backpressure, no tag-CAM dynamics. `TransactionalProgramExecutor` adds an
*analytical timing overlay* (per-resource makespan, `cycles = startup +
bytes/bus_width`) on top of the behavioral core — it yields values plus cycle
estimates, but the ordering is still not credit-throttled; it cannot expose
sequencing hazards, and the v0.9 plan attributes 4–8× timing overestimation
to exactly this sequential-overlay approach.

Also noteworthy:

- The fidelity framework (`include/sw/kpu/fidelity/`) is largely scaffolding:
  `SimulatorConfig` with per-component fidelity is parsed from JSON but never
  consumed by `KPUSimulator`, whose own `Config` has no fidelity field and
  hard-instantiates temporal components. The only end-to-end fidelity switch
  is the program-executor factory (two of three tiers).
- `include/sw/kpu/behavioral/l3_cache_model.hpp` remains in-tree with
  hit/miss/evict semantics — deprecated and contradictory to the execution
  model; it should be removed or quarantined.
- The interface + factory seam (`models/interfaces/`,
  `*_factory.cpp`) exists and is the right dispatch point, but the
  transactional tier behind it is thin (4 components).

### 3.4 CSP Timing Tier (v0.9) — the right skeleton, tokens only

This is the most architecturally faithful implementation of the execution
model, and the natural backbone for the target simulator.

**Exists and works (all 12 timing test suites pass):**

- CSP primitives: `CreditPool`/`PartitionedCreditPool`, ref-counted `TagCAM`,
  work-conserving `WorkQueue`, `LivelockDetector`.
- Component processes: `MemoryControllerProcess` (per-bank FSM, row
  hit/miss/empty, shared across DMAs via `submitter_id`), `DMAEngineProcess`,
  `BlockMoverProcess`, `StreamerProcess` — all credit/tag-correct.
- `ConcurrentTimingExecutor` (~1000 lines): grid construction, cycle-ordered
  ticking (MC → DMA → BM → Streamers), Chrome-trace/CSV export, statistics.
- Automated per-operator schedule generators (matmul with 4 strategies,
  conv2d, softmax, layernorm, batchnorm) plus `ScheduleValidator` and
  `ScheduleExecutor`.
- The 1×1×1 pipeline demo (`examples/schedule/csp_pipeline_demo.cpp`)
  completes the full DRAM→L3→L2→Compute→L2→L3→DRAM round trip correctly.

**Shortcomings:**

1. **Multi-tile livelock — RESOLVED (issue #61, PR #62, 2026-07-10).**
   `run_matmul` livelocked at its default 64×64×64 / 16³-tile configuration
   and every larger size. Root causes: silent request drop in the DMA
   staging queue beyond `queue_depth`; an L2 credit leak on tile reuse
   (duplicate MOVEs each acquired a credit while the ref-counted TagCAM
   entry releases only one — fixed by implementing the schedule generators'
   documented "execution layer deduplicates" contract in the BlockMover);
   duplicate in-flight loads double-acquiring L3 credits (fixed by in-flight
   dedup plus tile-affine work assignment); and shared `static` slot
   counters. All strategies now complete for 32³–256³. Details:
   `docs/sessions/2026-07-09_v0.9_multi_tile_livelock_fix.md`.
2. **No data.** `TileDescriptor` carries addresses and metadata only. Nothing
   in `include/sw/kpu/timing/` moves a byte of payload; the tier has zero
   dependency on the behavioral/functional models.
3. **Compute is a placeholder.** There is no compute process; compute is a
   fixed-latency delay inline in the executor, K-independent, with the
   COMPUTE dependency keyed only to the *last B feed*
   (`matmul_schedule_generator.hpp:250-255`) rather than all K-slice A+B
   feeds. Accumulation over K is modeled neither as data nor as time.
4. **Simplified memory controller.** The CSP MC omits tRRD/tFAW/turnaround/
   refresh. Meanwhile a mature, invariant-validated LPDDR5-6400 controller
   exists (`models/temporal/memory/controllers/lpddr5_controller.hpp`, plus
   the `patterns/memory/lpddr5/` suite with 15 enforced invariants) — but it
   is not wired into the CSP executor.
5. conv2d generator partially implemented (im2col addressing simplified, one
   "Not implemented yet" branch).

### 3.5 Compiler Toolchains — four stacks, one missing middle

There are four partially overlapping graph toolchains:

1. **Python front-end (works end-to-end, behavioral only).**
   `torch.compile(backend="kpu")` / ONNX / `@kpu.compile` → `OpGraph` →
   fusion compiler → op-level DFX JSON → `kpu_native` → C++
   `BehavioralComputeFabric` + numpy, with optional transactional timing
   overlay and numpy fallback. Real pretrained models (ResNet-18,
   MobileNetV2, ViT) run and validate against PyTorch. This is op-granular
   execution — it never touches the tiled DMProgram machine model.
2. **C++ Kernel/KernelGraph path (works, hand-authored).** `Kernel::create_*`
   factories (matmul, MLP, conv2d, attention, norms, pool, softmax) and
   `KernelGraph` (topological sort, fusion analysis, `compile()` to a single
   `DMProgram` with barriers) → behavioral/transactional executors. Mature,
   but graphs are constructed by hand in C++; nothing lowers a model into it.
3. **Tile-level DFX compiler (front half only).** `.dfg` → `DFXGenerator`
   (with a real `TileOptimizer`: analytical/bounded-search/heuristic tiling)
   → `compiler::dfx::Program` → `.kpu` object file. **Dead-ends at
   execution:** the only consumer, `ScheduleBinder`, produces round-robin
   resource estimates and is referenced nowhere else; `kpu-loader` loads
   `.kpubin` DMPrograms and ignores `.kpu` objects. There is **no
   `compiler::dfx::Program → isa::DMProgram` lowering.** The front end also
   handles MATMUL only.
4. **`tools/dfg/` toolchain.** Separate DFG generate/schedule/compile/
   visualize/analyze CLI chain — analysis-oriented, disjoint from the above.

Additionally the DSL (`dsl::Schedule` + `schedule_compiler.cpp`) compiles
fluent tiled-loop schedules into DMPrograms (compute ops intentionally emit
NOP — compute is implicit in streaming), and the CSP schedule generators
(§3.4) emit ISA-independent `ScheduleOperation`s for the timing executor.
That is: there are **two schedule vocabularies** (`isa::DMInstruction` vs.
`timing::ScheduleOperation`) that do not share a compiler.

### 3.6 Operator and Datatype Coverage

**Covered (DNN inference):** matmul/GEMM (all tiers), conv2d via im2col,
depthwise conv, pooling, softmax, layernorm/batchnorm, elementwise +
activations (relu/gelu/silu/...), attention as matmul+softmax composition.
Verification taxonomy classes 0–6 (`verification/kernels/TAXONOMY.md`) with
behavioral references; DNN-level verification (VGG-16, SqueezeNet) and the
torch.compile examples.

**Datatypes:** FP32/FP16/BF16/FP8(E4M3/E5M2/E3M4/E2M5)/FP4/INT32-4 via the
Stillwater Universal library (`cfloat`, `bfloat16`) with a full quantization
stack (packing, quant params, type dispatch, templated kernels). Posits are
not used anywhere (one comment aside). The quantized paths run behaviorally.

**Absent entirely (the stated goals):**

| Target class | Status in code |
|---|---|
| Cholesky, QR, LU, SVD, eigensolvers | **No code at any fidelity.** Design prose only (`docs/design/tiling_discussion.md`, methodology docs). |
| FFT | **No code.** DFX/SURE aspiration in docs only. |
| FIR/IIR filter kernels (beyond DNN conv2d) | **No code.** |
| Kalman filters (EKF/UKF/multiscale) | **No code, no design doc.** |

The building blocks these need — tiled matmul, systolic array, triangular
data movement, elementwise/reduction vector ops, scalar sqrt/reciprocal —
exist partially (matmul yes; VE ops are no-op annotations; no triangular
tile schedules; no data-dependent scheduling).

Also: `benchmarks/` is an empty shell (its CMakeLists references directories
that do not exist).

### 3.7 Housekeeping Findings

- The three "pre-existing" timing test failures recorded in project memory
  (tag_cam duplicate insert, BM/STR process names) have been reconciled and
  **now pass** — the current suite is green.
- Duplicate doc-tree prefix: both `docs/07-fidelity-elevation/` and
  `docs/07-runtime/` exist.
- Several superseded roadmap docs remain findable and contradict
  `docs/ROADMAP.md` (they are marked superseded in ROADMAP; consider archive
  moves).
- Root-level status .txt files (`v0.8-status.txt`, `v0.9-status.txt`,
  `fidelity_status.txt`, ...) duplicate CHANGELOG/session-log content and
  drift; `fidelity_status.txt` still carries the key unchecked TODO
  ("Wire OFG executors with behavioral callbacks") that this plan promotes
  from "optional" to the central deliverable.

---

## 4. Consolidated Gap Analysis

Ordered by how directly each blocks the stated goal.

**G1 — Values and ordering are never simultaneous (the core gap).** No engine
produces numerically correct results under credit-accurate transaction
ordering. The behavioral/transactional executors compute without credit
dynamics; the CSP executor orders without data. This is the repo-level
formalization of `docs/07-fidelity-elevation/gap-assessment.md`, carried to
the v0.9 CSP tier.

**G2 — CSP multi-tile liveness bug — RESOLVED (#61/#62).** Token-only
multi-tile matmul livelocked at default configuration; fixed 2026-07-10
(see §3.4.1). What remains from this gap is the CI regression coverage
(Phase 0) — the livelock was invisible because no test executed a generated
multi-tile schedule end-to-end.

**G3 — Compute is not modeled in the CSP tier.** No compute process, no
K-accumulation semantics, constant latency, wrong dependency (last-B-feed
only). Both a functional and a timing-fidelity defect.

**G4 — The runtime cannot load or functionally execute programs.** No C API
program loading; `launch` routes to a timing-only engine; the
value-producing executors are unreachable from the C API/bindings.

**G5 — The compiler middle is missing.** No tile-level-DFX → DMProgram
lowering; the C++ graph compiler handles matmul only; the Python network
path bypasses the machine model; two schedule vocabularies
(DMInstruction vs. ScheduleOperation) with no shared lowering.

**G6 — ISA gaps for the target operator classes.** Gather/scatter, VE
elementwise/reduce, broadcast, and L2 scratch are annotation-only.
Factorizations additionally need: scalar/diagonal ops (sqrt, reciprocal),
triangular tile iteration spaces, and schedules with data-dependent /
wavefront dependencies (panel → trailing update). None are expressible today.

**G7 — Operator coverage.** Cholesky/QR/LU/SVD, FFT/filters, and all Kalman
variants are absent at every level (no kernels, no schedules, no
verification classes, no design docs for Kalman).

**G8 — Fidelity configuration is not wired.** `SimulatorConfig` fidelity is
parsed but never consumed by `KPUSimulator`; CYCLE_ACCURATE executor returns
nullptr; component factories are partially populated.

**G9 — Fragmentation and drift.** Four execution engines, four compiler
toolchains, two DFX namespaces, deprecated cache-semantics model in-tree,
spec/implementation drift in kpuasm, superseded docs and status files.

---

## 5. Implementation Plan

The strategy: **make the CSP tier the single execution authority** — it is
the only engine whose ordering semantics match the architecture — and elevate
it from token-only to transactional-functional by grafting the (already
correct) behavioral value computation onto it. Then expose that engine
through the runtime and give the compiler one sanctioned lowering path onto
it. Operator expansion comes last, because every new operator class rides on
those rails.

Phases are ordered by dependency; each has an exit criterion that is a
runnable, verifiable artifact.

### Phase 0 — Stabilize the CSP tier (completes v0.9)

Fix G2 and the latent defects that would otherwise contaminate every later
phase.

- ~~Fix the multi-tile livelock~~ **DONE (#61/#62, 2026-07-10):** DMA staging
  queue no longer drops requests; BlockMover deduplicates moves for
  L2-resident tiles (fixing the credit leak on reuse); DMA defers duplicate
  in-flight loads; work assignment is tile-affine; `static` slot counters are
  instance members. `run_matmul` completes 32³–256³ across all strategies.
- Correct the COMPUTE dependency to require **all** K-slice A and B feeds for
  a C tile, and make compute latency a function of tile dimensions and K
  (still token-only in this phase, but the dependency graph must be right
  before data flows over it).
- Add a timing-invariant regression: multi-tile matmul configurations
  (2×2×2 … 8×8×8, all four strategies) must complete without livelock, with
  stall-cycle budgets asserted.

**Exit criterion:** token-only multi-tile matmul completes across the
strategy × size matrix in CI; v0.9 validation plan
(`docs/plans/v0.9-validation.md`) targets met.

### Phase 1 — Transactional-functional unification (closes G1, G3)

The centerpiece. Give the CSP engine real data and real compute.

- **Data plane:** attach payloads to tiles — either an optional buffer handle
  on `TileDescriptor` or (preferred, keeps descriptors light) a side
  `TileDataStore` keyed by `TileID`, owned by the executor. DMA, BlockMover,
  and Streamer processes move bytes exactly when they move tokens; buffer
  capacities in the data store mirror the credit pools so a data-plane
  overflow is by construction a credit-accounting bug.
- **Compute process:** promote compute from inline executor stub to a proper
  `ComputeProcess` implementing `IProcess`, which on firing reads the fed A/B
  tile payloads, performs the tile matmul (reuse
  `models/temporal/compute/systolic_array.hpp` or the typed kernels in
  `quantization/kernels.hpp` for dtype coverage), **accumulates over K** into
  a C-tile accumulator whose lifetime spans the K loop, and publishes the
  result tile into `compute_result_tag_cam` only when the K loop completes.
- **Oracle verification:** every CSP run of an operator is checked against
  the behavioral tier result (which is itself checked against host
  references). New test family: `tests/timing/test_functional_*` comparing
  CSP-executed multi-tile matmul/conv2d/softmax outputs elementwise against
  `BehavioralProgramExecutor` on the same program.
- Wire the fidelity factory's CYCLE_ACCURATE / transaction-ordered slot
  (`create_program_executor`) to this engine — no more `nullptr` tier.

**Exit criterion:** a multi-tile matmul (and one non-matmul op, e.g.
softmax) executes through `ConcurrentTimingExecutor` producing bitwise/
tolerance-correct results *and* credit-accurate timing, validated in CI.
This is the "behavioral simulator as transactional simulator" the goal
statement demands.

### Phase 2 — One ISA, one lowering onto the engine (closes half of G5, G9)

Unify the two schedule vocabularies so that programs, not ad-hoc schedule
lists, drive the engine.

- Implement `DMProgram → ScheduleOperation` translation (or teach
  `ConcurrentTimingExecutor` to consume DMPrograms directly): the ISA already
  encodes engine assignment, tile geometry, loops, and sync; the CSP
  generators already prove the target vocabulary is sufficient for
  matmul/conv/softmax/norms.
- Route the existing CSP schedule generators through DMProgram emission
  (they become ISA-level program builders, like
  `OutputStationaryProgramBuilder`), eliminating the parallel vocabulary.
- Implement the annotation-only opcodes in the unified engine:
  VE_ELEMENTWISE/VE_REDUCE (there is already a behavioral vector engine to
  borrow semantics from), STR_BROADCAST_*, L2_SCRATCH_*, gather/scatter.
  These unblock Phase 5.
- Update `docs/kpuasm-specification.md` to v1.1: document AUTO addressing,
  IndexRole, enhanced SET_*; fix SET_STRIDE drift. The spec becomes the
  compiler/runtime contract.
- Retire or quarantine the redundant engines: mark the OFG executors and the
  older `ProgramExecutor` as superseded; delete
  `behavioral/l3_cache_model.hpp`; keep `BehavioralProgramExecutor` as the
  fast functional oracle (it is the reference, not a redundancy).

**Exit criterion:** the same `.kpubin` file executes on (a) the behavioral
oracle and (b) the transactional-functional engine, producing identical
values and a timing report; the CSP demo/examples run from DMPrograms.

### Phase 3 — Runtime completeness (closes G4)

Make the C API a real target for a runtime executing assembly programs.

- Add program loading to the C API: `kpu_program_load(path)` /
  `kpu_program_from_buffer(...)` → handle; `kpu_kernel_from_program(handle)`
  bridging into the existing Kernel/executor machinery
  (`ProgramSerializer::load` already exists — this is plumbing plus
  argument-binding metadata: map kernel launch args to the program's
  `MemoryMap` A/B/C base addresses).
- Route `kpu_runtime_launch` through a `HardwareContext`-backed executor
  selected by fidelity (behavioral for speed, transactional-functional for
  order-accurate runs) so `get_output` returns computed results alongside
  `KPULaunchResult` cycles. Retire `ConcurrentExecutor` as the launch engine
  (keep it, if at all, as a fast analytic estimator behind an explicit
  "estimate" API).
- Unify the resource vocabulary: the runtime, the executor, and
  `resource_handle.hpp` share one `ResourceType`.
- Wire `SimulatorConfig` per-component fidelity into simulator construction
  via the existing interface/factory seam (closes G8), so a runtime can
  request e.g. behavioral memory + transactional compute.
- Extend Python bindings to expose program load + launch, so the torch path
  can *optionally* execute lowered programs rather than op-level calls.

**Exit criterion:** a CLI/test that does exactly the goal sentence —
assemble `matmul.kpuasm`, load the `.kpubin` through the C API, launch, read
back a correct result matrix with cycle counts — plus the same from Python.

### Phase 4 — Compiler middle-end (closes the rest of G5)

One sanctioned path from a multi-op graph to an executable program.

- Implement the missing lowering `compiler::dfx::Program → isa::DMProgram`:
  DFX DataMoveOps/ComputeOps/Barriers map naturally onto DMA/BM/STR
  instructions + loops; `TileOptimizer` already supplies tiling. This turns
  the `.kpu` object format from an analysis dead-end into the compiler's
  portable artifact, per the `docs/06-compiler/` design.
- Extend the graph front end beyond single-MATMUL: consume the
  Kernel/KernelGraph factories (conv2d, norms, softmax, attention,
  elementwise, pool already exist) so a multi-op graph compiles to one
  DMProgram with barriers/fusion — the `KernelGraph::compile()` machinery is
  already there.
- Bridge the Python front end: `OpGraph` → KernelGraph (or → op-level DFX →
  tile-level DFX), so torch.fx/ONNX models can be lowered to DMPrograms and
  executed on the transactional-functional engine, not just the behavioral
  fabric. Keep the current behavioral op-level path as the fast mode.
- Decide the fate of the `tools/dfg/` toolchain (fold its
  scheduling/analysis passes into the main path or mark analysis-only).

**Exit criterion:** an MLP (XOR or MNIST) lowered from torch.fx/ONNX runs
end-to-end on the transactional-functional engine with results matching
PyTorch and a credit-accurate timeline; the kpu-loader consumes compiler
output directly.

### Phase 5 — Operator expansion (closes G6/G7)

Now, and only now, the target operator classes — each lands as: numerical
kernel + tile schedule/program builder + verification class + (where needed)
ISA extension. Recommended order, by dependency:

1. **Blocked LU and Cholesky** (first, because they define the new
   machinery): panel factorization + triangular solve (TRSM) + trailing
   update (GEMM/SYRK). GEMM dominates the FLOPs and already works; what is
   new is (a) triangular tile iteration spaces, (b) diagonal/scalar ops
   (sqrt, reciprocal) on the vector engine, (c) **wavefront dependencies**
   (step k+1's panel depends on step k's trailing update) — expressible as
   DMProgram dependency chains, but the schedule builders must generate them.
   Deliverable: right-looking blocked Cholesky + LU with partial pivoting
   (pivoting stresses gather/scatter from Phase 2).
2. **QR** (Householder, blocked WY): adds reductions (norms) and rank-k
   updates; reuses the TRSM/GEMM machinery.
3. **Kalman filters — EKF first**: predict/update are compositions of GEMM,
   Cholesky (covariance factorization), and triangular solves — i.e., a
   *graph-level* deliverable exercising Phase 4 on the Phase 5.1 kernels.
   Then UKF (adds sigma-point generation = Cholesky + broadcast) and
   multiscale variants. A design doc is required first (none exists).
4. **Signal processing**: 1-D/2-D FIR as strided/streamed matmul (the
   streamer model fits naturally); IIR (sequential dependence — a good
   stress of the dataflow model); FFT last (butterfly data movement will
   likely motivate dedicated BM permutation patterns — treat as its own
   design exercise).
5. **SVD/eigensolvers** (last; iterative, data-dependent convergence): start
   with one-sided Jacobi SVD or QR-iteration built from the QR kernel;
   requires host-in-the-loop convergence control — a runtime pattern
   (device kernels + host loop) that Phase 3's API must support (it does,
   via repeated launches).
- In parallel: extend the verification taxonomy with new classes
  (class7_factorization, class8_transform, class9_estimation) and populate
  `benchmarks/` (currently empty) with representative sizes.

**Exit criterion per operator:** behavioral-oracle match + transactional-
functional execution + verification class in CI. Milestone exit: EKF
tracking example running end-to-end from a graph description.

### Phase 6 — Fidelity elevation and calibration (closes G8 remainder)

- Wire the mature LPDDR5 controller (and siblings: DDR5/GDDR6/HBM) into the
  CSP executor behind `IMemoryController`, replacing the simplified MC for
  CYCLE_ACCURATE runs; carry the `patterns/memory/lpddr5/` invariant
  validation into pipeline-level traces.
- Calibrate the transactional tier's latency/bandwidth parameters from
  cycle-accurate runs (the multi-fidelity philosophy in CLAUDE.md), and
  validate the ±10% timing target from the v0.9 roadmap on the model suite
  (MNIST, SqueezeNet, MobileNetV2 per ROADMAP v1.0 criteria).

**Exit criterion:** ROADMAP v1.0 gate — CYCLE_ACCURATE functional, timing
within target vs. calibrated references, model suite passing.

### Sequencing and roadmap alignment

- Phases 0–1 complete and extend **v0.9** (Phase 1 is the promotion of
  `fidelity_status.txt`'s unchecked "wire OFG/behavioral" item to core
  scope, realized on the CSP tier).
- Phases 2–4 constitute a **v0.10 "Unified Execution & Runtime"** release —
  this is new scope relative to `docs/ROADMAP.md` and should be inserted
  before the current v1.0 gate, because v1.0's "API stability" promise is
  hollow while the C API cannot execute programs.
- Phase 5 is the operator track; 5.1–5.3 (factorizations + EKF) are strong
  v1.1 candidates (the current v1.1 "auto-tuning" can shift). Phase 6
  merges into the existing v1.0 CYCLE_ACCURATE criteria.

Parallelization: Phase 5's numerical-kernel and design-doc work (blocked
factorization algorithms, Kalman design doc, verification harnesses against
host references) has no dependency on Phases 2–4 and can start immediately
at the behavioral tier; only the *scheduled/tiled* versions wait for the
rails.

### Risks

- **Livelock class of bugs (Phase 0/1).** Credit protocols fail globally
  from local mistakes. Mitigation: the invariant-validation culture from the
  LPDDR5 patterns should be extended to pipeline traces early (credit
  conservation, tag-CAM ref-count conservation, single-writer-per-buffer) —
  add these as INV-3xx-style checks on CSP Chrome traces.
- **Two-vocabulary consolidation (Phase 2)** risks churn in green tests.
  Mitigation: keep `ScheduleOperation` as the executor-internal form and
  make DMProgram→ScheduleOperation a pure translation layer first; migrate
  generators second.
- **Factorization numerics (Phase 5)** at low precision (FP8/FP4) will not
  pass naive tolerance checks. Define per-dtype error models up front in the
  verification classes; validate FP32/BF16 first.
- **Scope magnetism.** The repo's history shows parallel engines accreting.
  The plan's guardrail: after Phase 2, any new execution capability must
  land in the unified engine or the behavioral oracle — nowhere else.

---

## 6. Summary

The project has already built, separately, every hard piece the goal needs:
a correct functional tier, a credit-faithful CSP ordering tier, a complete
ISA + assembler + serializer, a resource-complete runtime shell, automated
tiling, and a working Python model front end. What it does not have is the
connective tissue: data in the ordered engine, programs in the runtime, and
a lowering from the compiler IR to the ISA. The plan above is deliberately
integration-first (Phases 0–4) and expansion-second (Phase 5), because the
stated operator goals — factorizations, filters, Kalman — are compositions
of GEMM plus a small set of new primitives, and they become cheap exactly
when the unified transactional-functional target exists.
