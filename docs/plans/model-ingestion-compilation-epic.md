# Epic: Real-DNN Ingestion & Compilation (ONNX/PyTorch → domain_flow → KPU)

**Status:** Approved direction (decisions D1–D4 resolved 2026-07-19) — implementation not started.
**Author:** planning session 2026-07-19.
**Scope:** two repos — `kpu-sim` and `branes-ai/domain_flow` (consumed via FetchContent).

**Resolved decisions (2026-07-19):**
- **D1 — direct `ONNX → dfg` reader** (no LLVM/MLIR) for the front-end.
- **D2 — topology + timing first**: the first milestone needs weight *shapes*, not
  *values*; numerical trained-weight fidelity is a later milestone.
- **D3 — canonical IR = domain_flow's `DomainFlowGraph`** (not kpu-sim's
  `KernelGraph`). domain_flow is the compiler; its IR is the single source of truth.
  `ComputationalGraph` retires; `KernelGraph` is demoted to a generated lowering
  artifact (and a candidate for eventual removal). See §4a.
- **D4 — file as a formal epic + per-phase sub-issues, run alongside the M4
  milestone** (attention → flash attention).

---

## 1. Goal

Replace the hand-written DNN facsimiles (`resnet18.hpp`, `mobilenetv2.hpp`,
`efficientnet.hpp`, `m2_resnet.cpp`) with a real pipeline: **load a trained DNN from
disk (ONNX / PyTorch), compile it through domain_flow into cached KPU programs, and
execute it on the simulator with real weights.**

Target end-to-end flow:

```
  ONNX / PyTorch (.onnx / .pt)
     │  [A] domain_flow front-end: model file → domain flow graph
     ▼
  .dfg  (+ weight blob)                         ◄── the cross-repo interchange contract
     │  [B] kpu-sim import: .dfg → KernelGraph;  weights → HOST_MEMORY
     ▼
  KernelGraph + weights resident in host memory (ResourceManager)
     │  [C] compiler pass: operators → KPU "domain flow programs" (schedules)
     ▼
  compiled program (dfx / .kpu)  →  [D] program cache
     │  [E] loader: bind program → fabric resources  =  "personalize the data path"
     ▼
  execute on GraphCspExecutor / ConcurrentTimingExecutor  (real weights, timing model)
```

The five stages **[A]–[E]** are the user's description, mapped to concrete seams
below.

---

## 2. Current state (grounded in both repos)

### 2.1 domain_flow (`build/_deps/domain_flow-src`, header-only, `sw::dfa`)

| Capability | State | Evidence |
|---|---|---|
| dfg IR (`DomainFlowGraph` / `Node` / `Edge`, ~40 TOSA/DNN ops) | ✅ mature | `include/dfa/domain_flow_graph.hpp`, `domain_flow_operator.hpp:10-54` |
| `.dfg` **text** serialization (`save`/`load`, `<<`/`>>`) | ✅ works, round-trips | `domain_flow_graph.hpp:265-451`; samples `data/dfg/mobilenet_v2.dfg` |
| TOSA-MLIR-text → dfg importer | ✅ real, but **MLIR-gated** | `tools/import/dfa_import_tosa.cpp`, `include/dfa/mlir/dialect/tosa.hpp` |
| **ONNX importer** | ❌ none | repo-wide grep: zero onnx/protobuf hits |
| **PyTorch/torch importer** | ❌ stub | `dfa_import_torch.cpp` is a copy of the TOSA one; `torch.hpp` dialect empty |
| Polyhedral machinery (domain-of-computation, constraint sets, index spaces, simplex LP, schedules) | ⚠️ real primitives, **thin op coverage** | `domain_of_computation.hpp:116-390` implements only MATMUL / elementwise / CONSTANT |
| Whole-graph scheduling / fabric generation | ❌ TODO stubs | `domain_flow_graph.hpp:158-170,232` |
| **KPU-target lowering / backend / resource model** | ❌ none | hardware-agnostic; `sim/` is an empty placeholder |

**Front-end reality:** the intended path is `ONNX/PyTorch → TOSA MLIR → dfg`, where
the `ONNX → TOSA MLIR` step happens **upstream** (torch-mlir / iree-import) and is
**not in domain_flow**. The in-repo importer starts at TOSA MLIR text and needs a
full LLVM/MLIR 20.x build (`DOMAINFLOW_MLIR_TOOLS=ON`, default OFF).

### 2.2 kpu-sim (`7a992ad`)

| Concept the user named | State | Evidence |
|---|---|---|
| Executed graph = `KernelGraph` (12 op types) | ✅ | `include/sw/kpu/kernel_graph.hpp`, `kernel.hpp:19` |
| **Weights live OUTSIDE the graph** (`NodeData` map of `std::vector<float>`) | ⚠️ impedance mismatch | `graph_csp_executor.hpp:41-52`, synthetic `rn_synth` `resnet18.hpp:71` |
| `.dfg` reader (`DomainFlowGraphLoader`) | ⚠️ partial, **disconnected** | reads into `ComputationalGraph`, tensor metadata + topo-sort **stubbed**, not wired to the executor: `src/software/compiler/graph_loader.cpp:210,359` |
| Second graph type `ComputationalGraph` (compiler-side) | ⚠️ parallel to `KernelGraph` | `include/sw/compiler/graph_loader.hpp:91` |
| "DFX" = **two** layers: `sw::kpu::compiler::dfx` (`.kpu` object file, "PTX-for-KPU", serializable) + `sw::kpu::dfx` (DSL) | ⚠️ neither wired to `GraphCspExecutor` | `include/sw/compiler/dfx/dfx.hpp:1-24`, `dfx_object_file.hpp` |
| Prototype `.dfg → .kpu → BoundSchedule` toolchain | ⚠️ exists in `tools/`, **not integrated** | `tools/compiler/kpu-kernel-compiler/main.cpp`, `tools/runtime/kpu-loader/schedule_binder.hpp:82` |
| **ResourceManager** (HOST_MEMORY, L3/L2/L1, DMA/BM/STR alloc + r/w) | ✅ real allocator | `include/sw/kpu/resource_api.hpp:53`, `resource_manager.cpp` |
| Data-path "personalization" = `ConcurrentTimingExecutor::Config` (fabric topology) | ✅ but rebuilt per-op | `concurrent_timing_executor.hpp:72`; `csp_op_runners.hpp:264` |
| Host/DRAM memory model (`ExternalMemory`, HOST_MEMORY) | ✅ exists, **off the exec path** | `include/sw/memory/external_memory.hpp:32` |
| **Program cache** (compiled-graph persistence) | ❌ none | only single-`Kernel` `KernelSerializer` + tools-only `.kpu` |
| `KernelGraph`-level serialization | ❌ none | — |
| Schedule generators (matmul/conv2d/…) → `ScheduleResult` on CSP executor | ✅ the de-facto "KPU program", but **generated per-op at runtime, never serialized** | `include/sw/kpu/timing/schedule/*` |

**Execution seam (the integration point):**
`GraphCspExecutor::run(const KernelGraph&, const std::vector<float>& input, const
std::unordered_map<size_t,NodeData>& node_data, Size T)`
(`graph_csp_executor.hpp:70`). A loaded model must materialize as
`(KernelGraph, weights, input)`.

---

## 3. Gap analysis

| # | Gap | Where | Severity |
|---|-----|-------|----------|
| G1 | **ONNX/PyTorch → dfg front-end** | domain_flow | 🔴 largest — nothing exists |
| G2 | **Weights transport**: model file weights → resident in HOST_MEMORY; kernels reference by handle, not in-process `vector<float>` | kpu-sim (+ contract) | 🔴 cross-cutting refactor of the exec path |
| G3 | **`.dfg` → executed `KernelGraph`** (not the dead-end `ComputationalGraph`); real tensor metadata + topo sort; full op-config mapping | kpu-sim | 🟠 |
| G4 | **`.dfg` contract formalization** — versioned op set + shapes/dtypes + weight references shared by both repos | both | 🟠 the API between the repos |
| G5 | **Operator-set alignment** — domain_flow `DomainFlowOperator` (~40, TOSA; no softmax/layernorm/attention) vs kpu-sim `KernelOpType` (12; has them) | both | 🟠 |
| G6 | **Compile → serialized KPU program** (persist per-op schedules + tiling/dataflow strategy) | kpu-sim | 🟠 partly in `.kpu`/tools |
| G7 | **Program cache** (keyed compiled-program store, load-from-cache fast path) | kpu-sim | 🟠 net-new |
| G8 | **Op→resource binding / data-path personalization** wired to the real executor | kpu-sim | 🟠 prototype in `tools/schedule_binder.hpp`, unconnected |
| G9 | **Convergence risk** — two graph types + two DFX layers + a tools/ pipeline; must connect, not add a fourth path | kpu-sim | 🟡 architectural discipline |
| G10 | **Real-weight validation** — need an external reference (onnxruntime) since today's oracle is self-consistent synthetic | kpu-sim | 🟡 |

---

## 4. Decisions needed (before/early in the work)

These change the shape of the plan; recommendations given, but they are yours.

- **D1 — Front-end path (G1).**
  (a) *MLIR route*: `ONNX → TOSA MLIR` (torch-mlir/iree upstream) → domain_flow's
  existing TOSA importer. Pro: reuses real machinery, general. Con: heavy LLVM/MLIR
  20.x dependency; torch dialect still unimplemented.
  (b) *Direct route*: new `ONNX-protobuf → dfg` reader (no MLIR), curated to the op
  set we already execute. Pro: light, no LLVM; fast to a working slice. Con:
  reimplements op coverage; ONNX-opset churn.
  **Recommendation:** start **(b)** for the CNN op set to get a working vertical
  slice, keep **(a)** as the long-term general front-end. Revisit once transformer
  models (which need the broader op set) are in scope.

- **D2 — Weight transport format (G2).** ONNX initializers / PyTorch `state_dict`
  → what on-disk blob does the sim load? **Recommendation:** a sidecar
  `safetensors`-style flat blob (name → dtype/shape/offset) referenced by the
  `.dfg` node attributes; deposited into `ExternalMemory`/HOST_MEMORY via
  `ResourceManager`. Avoids embedding large tensors in the `.dfg` text.

- **D3 — Canonical IR = domain_flow's `DomainFlowGraph`.** *(Resolved: standardize on
  the compiler's IR — see §4a.)* domain_flow is the compiler; ONNX imports into
  `DomainFlowGraph`, the compiler passes rewrite it, and it serializes as `.dfg`.
  kpu-sim consumes it and lowers it to an executable KPU program. `ComputationalGraph`
  retires; `KernelGraph` is demoted to a *generated* lowering artifact (never
  hand-authored), on a path to removal.

- **D4 — Compiled-program artifact (G6).** Reuse the existing **`sw::kpu::compiler::dfx`
  `.kpu` object file** (already "serializable, PTX-for-KPU") as the compiled-program
  format rather than inventing one; make the schedule-generators emit into it. This
  is the *lowered* form — distinct from the `DomainFlowGraph` source (§4a).

### 4a. Canonical IR & dependency direction (the D3 architecture)

Two representations, cleanly separated — this is the crux of the design:

| Level | Representation | Owner | Carries |
|---|---|---|---|
| **Source / compute IR** | `sw::dfa::DomainFlowGraph` (`.dfg`) | **domain_flow (canonical)** | operators, shapes/dtypes, op attributes (conv stride/pad, matmul dims), weight *references* |
| **Lowered executable form** | `.kpu` object / CSP schedules | kpu-sim (derived) | tiling `T`, dataflow strategy, per-op schedules, resource binding |

`KernelGraph` today conflates both roles and is *hand-authored*; under D3 it stops
being a source of truth. The migration (§6) makes `DomainFlowGraph` the only graph
that is imported, rewritten, and serialized; kpu-sim **lowers** it to execution.
This dissolves gaps **G3, G5, G9** (no dual graph, no op-mapping impedance, no path
proliferation) at the cost of one refactor (executor consumes/derives-from
`DomainFlowGraph`).

**Dependency direction.** Making `DomainFlowGraph` canonical makes domain_flow a
**hard dependency of kpu-sim's execution path** (today it is *optional*,
`KPU_HAS_DOMAIN_FLOW`). This is acceptable and light **because** the relevant header
`<dfa/dfg.hpp>` is **header-only and MLIR-free** (the MLIR bridge `dfa_mlir.hpp` is
separate and stays out). **Prerequisite:** domain_flow's optional tools
(`DOMAINFLOW_MATPLOT_TOOLS`→Matplot++, `DOMAINFLOW_VISUALIZATION`→CGAL/Qt6,
`DOMAINFLOW_MLIR_TOOLS`→LLVM) must be forced **OFF** in
`cmake/DomainFlowIntegration.cmake` so the hard dependency never drags in heavy
packages — this also fixes the current Windows configure breakage.

---

## 5. Target architecture (seam-by-seam)

| Stage | Owner | Input → Output | Reuse / build |
|---|---|---|---|
| **[A]** front-end | domain_flow | `.onnx`/`.pt` → `.dfg` + weight blob | build (D1); weights per D2 |
| **[B]** import | kpu-sim | `.dfg` + weights → `KernelGraph` + HOST_MEMORY-resident weights | extend `DomainFlowGraphLoader`; new weight loader |
| **[C]** compile | kpu-sim | `KernelGraph` → compiled program (`.kpu`) | reuse schedule generators + `dfx` object file |
| **[D]** cache | kpu-sim | program ↔ cache keyed by (op sig, shapes, dtype, fabric cfg) | net-new (small) |
| **[E]** bind/exec | kpu-sim | program → bound to `ResourceManager` fabric → run | productionize `ScheduleBinder`; wire to `GraphCspExecutor` |

---

## 6. Phased implementation strategy

Each phase ends with a demonstrable milestone (DoD). The spine
(`DomainFlowGraph → lower → execute`) is stood up **before** the ONNX front-end
(Phase 2) by generating a golden `.dfg` from an existing hand-built graph — this
de-risks the canonical-IR migration and isolates the heaviest external gap (G1).
Per D2, the early phases carry weight **shapes**, not values (placeholder/synthetic
values); real weight values + numerical validation are Phase 5.

### Phase 0 — Foundation: build hygiene + contract + lowering spine  *(kpu-sim)*
- **Build hygiene** (prereq for the hard dep, §4a): force domain_flow's optional
  tools OFF in `DomainFlowIntegration.cmake` (unblocks Windows; keeps the dep
  header-only + MLIR-free).
- Formalize the **`.dfg` contract v1**: operator mapping (`DomainFlowOperator` ↔
  kpu-sim runners), tensor shape/dtype encoding, weight-*reference* convention.
- Stand up the **lowering spine** `DomainFlowGraph → KernelGraph` (KernelGraph now a
  *generated* execution artifact, D3/§4a) feeding the existing `GraphCspExecutor`.
  Add a one-time `KernelGraph → DomainFlowGraph` exporter to mint golden `.dfg`s.
- **DoD:** `build_resnet18`'s graph → export `DomainFlowGraph` → lower → execute →
  **identical topology + cycle counts** to the direct build. Round-trips the contract.

### Phase 1 — `.dfg` on disk → execution; retire the dual graph  *(kpu-sim)*
- Productionize `DomainFlowGraphLoader` to load `.dfg` → `DomainFlowGraph` (real
  tensor metadata, real topo sort). **Retire `ComputationalGraph`** — the loader now
  yields the canonical IR, lowered by Phase 0's spine.
- **DoD:** `m2_resnet --from-dfg model.dfg` runs ResNet-18 from a disk `.dfg`
  (shapes real, values synthetic); timing/utilization match the direct build.

### Phase 2 — Direct `ONNX → dfg` front-end  *(domain_flow; the big gap G1)*
- Per D1: implement a **direct ONNX-protobuf → `DomainFlowGraph`** reader (no MLIR)
  for the CNN op set — map ONNX ops → `DomainFlowOperator`, read tensor **shapes**
  from initializers/value-info (values optional at this stage, D2).
- Fill any `DomainFlowOperator` domain coverage the reader needs (G5).
- **DoD:** a stock `resnet18.onnx` (torchvision) → `.dfg` → runs on the sim with
  **correct topology + timing** (numerical output *not* yet validated — Phase 5).

### Phase 3 — Compile to a serialized KPU program + program cache  *(kpu-sim; G6, G7)*
- Make the schedule generators emit a **serialized** compiled program (reuse the
  `dfx` `.kpu` object file, D4): per-op schedules + tiling + dataflow strategy — the
  *lowered* form (§4a).
- Build the program cache: key = (op signature, shapes, dtype, fabric config) →
  compiled program; load-from-cache on repeat.
- **DoD:** second run of the same model loads compiled kernels from cache (no
  recompile); cache hit/miss reported.

### Phase 4 — Resource binding / data-path personalization  *(kpu-sim; G8)*
- Productionize `ScheduleBinder`: bind compiled-program ops → concrete fabric
  resources (DMA/BM/STR ids, L3/L2/L1 allocations) via `ResourceManager` =
  "personalize the data path"; wire the bound program into `GraphCspExecutor`.
- **DoD:** a cached, bound program executes on the personalized fabric; timing
  (utilization/cycles) matches the direct path.

### Phase 5 — Weight values, numerical validation & model zoo  *(both)*
- **Real weight values** (the deferred half of D2): load weight blob into
  `ExternalMemory`/HOST_MEMORY via `ResourceManager`; kernels reference by handle.
- Add an **onnxruntime reference** path; validate real-weight numerical output (G10).
- Extend to transformer ops (softmax/layernorm/attention — kpu-sim has them,
  `DomainFlowOperator` must be extended), and the milestone-ladder models; retire the
  synthetic builders as they are superseded.
- **DoD:** ≥2 real models beyond ResNet run from disk with reference-validated output.

---

## 7. Recommended first increment (thin vertical slice)

**Phase 0 for ResNet-18: `DomainFlowGraph` exported from `build_resnet18`, lowered,
executed.** Proves the canonical-IR spine — build hygiene → `.dfg` contract →
`DomainFlowGraph` → lowering → `GraphCspExecutor` → identical topology + cycles —
**without** the ONNX front-end (G1) and **without** the weight-values refactor
(topology + timing only, D2). Guard it with the existing `resnet_regression` harness
(identical cycles ⇒ the round-trip is lossless). Phase 1 then moves the source to a
disk `.dfg` and retires `ComputationalGraph`; Phase 2's ONNX reader plugs in ahead of
the proven spine with low risk.

## 8. Risks & mitigations

- **LLVM/MLIR weight (D1).** The "real" front-end needs LLVM/MLIR 20.x; also the
  cause of the current Windows `Matplot++`/`CGAL` configure breakage (domain_flow's
  optional tools). *Mitigation:* direct-ONNX route for the first slice; force
  domain_flow's optional tool flags OFF in `DomainFlowIntegration.cmake` so the
  library integration never drags in MLIR/CGAL/Matplot++.
- **Path proliferation (G9).** Two graph types, two DFX layers, a disconnected
  tools/ pipeline. *Mitigation:* D3 — canonical `DomainFlowGraph` source IR + the
  `dfx` `.kpu` lowered artifact; retire `ComputationalGraph`, demote `KernelGraph` to
  generated, connect the tools/ prototype rather than fork it.
- **Hard-dependency + build hygiene (§4a).** Canonical `DomainFlowGraph` makes
  domain_flow a hard dep of the exec path. *Mitigation:* it is header-only + MLIR-free;
  force domain_flow's optional tools OFF in `DomainFlowIntegration.cmake` (Phase 0
  prereq — also fixes the Windows Matplot++/CGAL breakage).
- **Weight-values refactor blast radius (G2b).** Deferred to Phase 5 per D2; the early
  milestones use shapes only, so the exec-seam refactor is not on the critical path.
- **Cross-repo coordination.** Front-end lands in domain_flow, the rest in kpu-sim,
  coupled only by the versioned `.dfg` contract (G4). *Mitigation:* freeze the contract
  in Phase 0 before parallelizing.
- **Executor migration risk.** Making `DomainFlowGraph` canonical means the tested
  `GraphCspExecutor` path changes. *Mitigation:* the Phase 0 lowering keeps the
  executor unchanged (fed by a *generated* `KernelGraph`); `resnet_regression` guards
  identical cycles.

## 9. Open questions (remaining, D1–D4 resolved)

1. `.dfg` contract versioning: extend domain_flow's existing text format in place, or
   introduce a versioned schema alongside it? (Affects Phase 0.)
2. Epic sub-issue granularity — one per phase, or per phase-DoD?
3. Fine sequencing vs. M4 (agreed: run alongside) — which phases interleave with M4
   vs. run after it?
