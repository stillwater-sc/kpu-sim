# Epic: Real-DNN Ingestion & Compilation (ONNX/PyTorch → domain_flow → KPU)

**Status:** Approved direction (decisions D1–D5 resolved 2026-07-19) — implementation not started.
**Author:** planning session 2026-07-19.
**Scope:** two repos — `kpu-sim` and `branes-ai/domain_flow` (consumed via FetchContent).

**Resolved decisions (2026-07-19):**
- **D1 — direct `ONNX → dfg` reader** (no LLVM/MLIR) for the front-end.
- **D2 — topology + timing first**: the first milestone needs weight *shapes*, not
  *values*; numerical trained-weight fidelity is a later milestone.
- **D3 — canonical *source* IR = domain_flow's `DomainFlowGraph`** (the compiler's
  IR). **The compiler lowers it to a binary KPU program; kpu-sim does not compile.**
  `ComputationalGraph` retires; kpu-sim's schedule-generation is compiler-backend work
  that relocates to domain_flow. See §4a.
- **D4 — file as a formal epic + per-phase sub-issues, run alongside the M4
  milestone** (attention → flash attention).
- **D5 — kpu-sim is the KPU *hardware simulator*, not a compiler.** It owns/defines
  the **KPU binary functional spec** (the binary program format + execution
  semantics); the domain_flow compiler *targets* it; kpu-sim *reads and executes* the
  binary with hardware-identical APIs. This is the crux — see §1 and §4a.

---

## 1. Goal

Replace the hand-written DNN facsimiles (`resnet18.hpp`, `mobilenetv2.hpp`,
`efficientnet.hpp`, `m2_resnet.cpp`) with a real pipeline: **load a trained DNN from
disk, compile it through the domain_flow compiler into a cached binary KPU program,
and execute that program on the simulator.**

**Input scope:** the ingestion path for this epic is **ONNX only** (D1: direct
`ONNX → DomainFlowGraph`). **PyTorch is supported by exporting to ONNX upstream**
(`torch.onnx` / `torch.export → ONNX`), *not* by a native `.pt`/TorchScript reader —
domain_flow's `torch` importer is a stub (§2.1), and a native path is a later,
separate milestone. `.pt` in the diagram below denotes that upstream export step.

**Separation of concerns (the load-bearing principle).** domain_flow is the
*compiler*; kpu-sim is the *hardware simulator*. **All compilation/lowering happens
in domain_flow**, which emits a **binary KPU program** ("domain flow program")
conforming to the **KPU binary functional spec**. **kpu-sim does not compile** — it
*reads and executes* that binary with **exactly the APIs a real KPU exposes**
(the `ResourceManager` sets up resources and signals "ready/start"; the fabric
executes). The binary program is the contract: **kpu-sim owns/defines the spec, the
compiler targets it.**

```
  ONNX / PyTorch (.onnx / .pt)
   ┌───────────────────────────────────────────────── domain_flow  (COMPILER) ──┐
   │  [A] direct ONNX → DomainFlowGraph   (canonical source IR, D1/D3)           │
   │  [B] compiler passes: rewrite / tile / schedule / KPU-backend lowering      │
   │  [C] emit BINARY KPU PROGRAM  (the "domain flow program", per the spec)     │
   └───────────────────────────────┬─────────────────────────────────────────────┘
                                    │  binary program (+ model weights/data)
                                    ▼           ◄── KPU binary functional spec = the contract
   ┌───────────────────────────────────────────────── kpu-sim  (HARDWARE SIM) ──┐
   │  [D] program cache: fast-load precompiled kernels                           │
   │  [E] ResourceManager sets up resources / "KPU ready → start";               │
   │      read the binary program; deposit DNN data in HOST_MEMORY;              │
   │      execute on the transactional/functional timing model —                 │
   │      SAME APIs as the real KPU. kpu-sim does NOT lower/compile.             │
   └─────────────────────────────────────────────────────────────────────────────┘
```

kpu-sim's runtime keeps only the **executor** (today `ScheduleExecutor` +
`ConcurrentTimingExecutor`), now fed by a *deserialized* binary program instead of
one generated inline. The scheduling/tiling logic currently in kpu-sim's schedule
generators is *compiler-backend* work that relocates to domain_flow (or a shared
KPU-backend library) — see §5.

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
| Schedule generators (matmul/conv2d/…) → `ScheduleResult` on the **CSP cycle-accurate** executor (`ConcurrentTimingExecutor`) | ✅ the benchmarked timing path, but **generated per-op at runtime, never serialized** | `include/sw/kpu/timing/schedule/*` |
| **ISA binary program path**: `DMProgram` (DMInstruction stream) → **`.kpubin`** (versioned: `DMPROGRAM_MAGIC` + `DMPROGRAM_VERSION`, magic-checked) / `.kpujson`, via `ProgramSerializer`; loaded by **`kpu-loader`**; executed by `IProgramExecutor` (**Behavioral / Transactional** wrappers) | ✅ a real, **serialized, versioned** binary + reader + loader + executors — but on behavioral/transactional fidelity, **not** the CSP cycle-accurate model | `include/sw/kpu/isa/program_serializer.hpp`, `behavioral_program_executor.hpp`, `program_executor_interface.cpp`, `tools/runtime/kpu-loader/` |

**Execution seam.** Today: `GraphCspExecutor::run(KernelGraph, input, node_data, T)`
(`graph_csp_executor.hpp:70`) — but note this seam *generates* the schedule inline,
which is the compiler-backend work that relocates out (§4a). The **target** seam is
one level down: `ScheduleExecutor` running a *deserialized* `ScheduleResult` (the
binary program) on `ConcurrentTimingExecutor`. The integration point for a loaded
model is therefore the **binary program reader** (P0) plus `ResourceManager` setup
(P1) — not `GraphCspExecutor`.

---

## 3. Gap analysis

| # | Gap | Where | Severity |
|---|-----|-------|----------|
| G1 | **ONNX/PyTorch → dfg front-end** | domain_flow | 🔴 largest — nothing exists |
| G2 | **Weights transport**: model file weights → resident in HOST_MEMORY; kernels reference by handle, not in-process `vector<float>` | kpu-sim (+ contract) | 🔴 cross-cutting refactor of the exec path |
| G3 | **`.dfg` → executed `KernelGraph`** (not the dead-end `ComputationalGraph`); real tensor metadata + topo sort; full op-config mapping | kpu-sim | 🟠 |
| G4 | **`.dfg` contract formalization** — versioned op set + shapes/dtypes + weight references shared by both repos | both | 🟠 the API between the repos |
| G5 | **Operator-set alignment** — domain_flow `DomainFlowOperator` (~40, TOSA; no softmax/layernorm/attention) vs kpu-sim `KernelOpType` (12; has them) | both | 🟠 |
| G6 | **Compile → serialized binary KPU program** (schedules + tiling/dataflow strategy) — the *compiler backend* | domain_flow (relocated) + `.kpubin` format owned by kpu-sim | 🟠 backend logic exists in kpu-sim's generators; must relocate + serialize |
| G6b | **Binary program reader/executor + KPU functional-spec conformance** | kpu-sim | 🟠 net-new; `ScheduleExecutor` executes, but nothing deserializes a program |
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

- **D3 — Canonical *source* IR = domain_flow's `DomainFlowGraph`.** *(§4a.)*
  domain_flow is the compiler; ONNX imports into `DomainFlowGraph`, the compiler
  passes rewrite it, and it serializes as `.dfg`. **The compiler lowers it to the
  binary KPU program (D5); kpu-sim does not.** `DomainFlowGraph` and its lowering are
  compiler-internal. `ComputationalGraph` retires; kpu-sim's `KernelGraph` +
  graph-walking schedule-generation are *compiler-backend* concerns (§5) — the
  simulator runtime keeps only the executor.

- **D4 — Canonical binary artifact = the ISA `.kpubin` (`DMProgram`).** The repo
  already has a **serialized, versioned** ISA binary: `DMProgram` →
  `ProgramSerializer` → `.kpubin` (magic + `DMPROGRAM_VERSION`) / `.kpujson`, with a
  `kpu-loader` and `IProgramExecutor`s. **This is the canonical binary KPU program**
  (D5) — Phase 0 builds on it, not the greenfield `dfx` `.kpu` object. *Two things to
  reconcile (Phase 0 decisions, not yet settled):* (1) the higher-level `dfx` `.kpu`
  object file — does the compiler emit it as an intermediate that lowers to `.kpubin`,
  or is it dropped? (2) `.kpubin`/`DMProgram` executes today on **behavioral/
  transactional** models, while the **cycle-accurate CSP** timing model (the
  benchmarked one) runs a non-serialized `ScheduleResult` — so either the CSP executor
  gains a `DMProgram`/`.kpubin` front-end, or the binary format carries a
  schedule/tiling section the CSP path consumes, or the two paths unify. Resolving
  this is the heart of Phase 0/2.

- **D5 — The KPU binary functional spec is the contract; kpu-sim owns it, the
  compiler targets it.** The spec is embodied by the existing ISA binary
  (`DMProgram`/`.kpubin`, D4): the instruction stream of DMA/BlockMover/Streamer/
  COMPUTE/DRAIN/WRITEBACK/STORE ops with tile + resource operands + memory map, with a
  versioned header. domain_flow emits binaries conforming to it. **kpu-sim reads and
  executes; it never compiles.** *(Recommended: promote the existing `.kpubin` header
  to the full versioned spec per `dfg-kpu-versioning.md`, with a golden-binary
  conformance suite so the two repos stay in lockstep.)*

### 4a. The compiler/hardware boundary & the two artifacts (D3 + D5)

Two artifacts, two owners, one hard boundary — this is the crux of the design:

| Artifact | Representation | Produced by | Consumed by | Carries |
|---|---|---|---|---|
| **Source / compute IR** | `sw::dfa::DomainFlowGraph` (`.dfg`) | domain_flow front-end | domain_flow passes | operators, shapes/dtypes, op attrs, weight *references* |
| **Binary KPU program** (the contract) | `.kpubin` (`DMProgram`, per the KPU binary functional spec) | **domain_flow compiler** (lowering) | **kpu-sim** (execution) | the op/tile/resource *schedule* — the "ISA" the KPU runs |
| Model weights/data | weight blob → HOST_MEMORY | (model file) | kpu-sim `ResourceManager` | tensor values (Phase 5) / shapes (early) |

The boundary is the **binary KPU program**. Everything *left* of it (ONNX import,
graph rewrite, tiling, scheduling, KPU-backend lowering) is **compiler** work in
domain_flow. Everything *right* of it (load program + data, set up resources,
execute with hardware-identical APIs) is **hardware-simulator** work in kpu-sim.
kpu-sim **never lowers**. This dissolves gaps **G3, G5, G9** (no dual graph, no
op-mapping impedance, no path proliferation) *and* correctly places the compiler
where it belongs.

**Where does kpu-sim's current schedule-generation go?** The schedule generators
(`matmul_schedule_generator`, `conv2d_im2col`, …) and the `GraphCspExecutor`
graph-walk are, architecturally, the **KPU compiler backend** — they turn ops into
the tile/resource schedule. Under this boundary they **relocate to domain_flow** (or
a shared `kpu-backend` library the compiler links). kpu-sim keeps only the
**executor** (`ScheduleExecutor` + `ConcurrentTimingExecutor`) that *interprets* a
deserialized program, plus the `ResourceManager` and memory model.

**Dependency direction.** kpu-sim's runtime depends on domain_flow **only** through
the binary-program format (and, for reading `.dfg`/weights into host memory, the
header-only, MLIR-free `<dfa/dfg.hpp>`). It does **not** depend on domain_flow's
compiler passes. **Prerequisite either way:** domain_flow's optional tools
(`DOMAINFLOW_MATPLOT_TOOLS`→Matplot++, `DOMAINFLOW_VISUALIZATION`→CGAL/Qt6,
`DOMAINFLOW_MLIR_TOOLS`→LLVM) must be forced **OFF** in
`cmake/DomainFlowIntegration.cmake` so the coupling never drags in heavy packages —
this also fixes the current Windows configure breakage.

---

## 5. Target architecture (seam-by-seam)

The hard boundary is between **[C]** and **[D]**: the binary KPU program.

| Stage | Owner | Input → Output | Reuse / build |
|---|---|---|---|
| **[A]** front-end | **domain_flow** | `.onnx` → `DomainFlowGraph` | new direct ONNX reader (D1). `.pt` is out of scope — export to ONNX upstream (§ Goal note) |
| **[B]** passes / KPU backend | **domain_flow** | `DomainFlowGraph` → tiled/scheduled lowering | relocate kpu-sim's schedule generators here (shared `kpu-backend`) |
| **[C]** emit binary program | **domain_flow** | lowering → **`.kpubin`** (`DMProgram`, per the spec, D4/D5) | reuse `ProgramSerializer`'s versioned format |
| **[D]** program cache | kpu-sim | `.kpubin` ↔ cache keyed by *(op sig, shapes, dtype, fabric cfg, **ISA/ABI version, compiler version, opset+type-sys version, target profile**)* | net-new (small) |
| **[E]** load + execute | **kpu-sim** | read `.kpubin` (via `ProgramSerializer`/`kpu-loader`) + weights; `ResourceManager` sets up fabric; run | reader exists; reconcile ISA-executor vs CSP timing path (D4); productionize `ScheduleBinder` |

`.dfg` (the `DomainFlowGraph` source IR) is **compiler-internal**; kpu-sim's runtime
**execution** input is `.kpubin`, not `.dfg`. kpu-sim reads model *data/weights* into
`HOST_MEMORY` (Phase 5) but does not consume `.dfg` to execute.

**Spec doc (D5):** the `.kpubin` binary format + execution semantics are written up as
the *KPU binary functional spec* (versioned per `dfg-kpu-versioning.md`), with a
golden-binary conformance suite run by kpu-sim — the single artifact keeping compiler
and simulator in lockstep.

---

## 6. Phased implementation strategy

Each phase ends with a demonstrable milestone (DoD). Ordering principle: **prove the
simulator can consume a binary program first (P0–P1), then build the compiler that
produces it (P2–P3), then optimize/validate (P4–P5).** The binary KPU program (D5)
is the pivot. Per D2, early phases carry weight **shapes**, not values.

Bootstrap note: in P0 the golden binary is produced by kpu-sim's *existing* inline
schedule generators, used **temporarily** as the KPU-backend *reference producer* —
they relocate to the compiler in P2. This lets the simulator side (ISA consumer) and
the compiler side (ISA producer) proceed in parallel against the binary contract.

### Phase 0 — KPU binary program spec + simulator reader/executor  *(kpu-sim)*
- **Build hygiene** (§4a prereq): force domain_flow's optional tools OFF in
  `DomainFlowIntegration.cmake` (unblocks Windows; keeps the dep header-only + MLIR-free).
- Promote the **existing `.kpubin` (`DMProgram`) format (D4)** to the versioned **KPU
  binary functional spec (D5)**: extend `ProgramSerializer`'s header per
  `dfg-kpu-versioning.md` (R1–R9 + `.kpu`/`.kpubin` §3 — version stamp + `min_consumer`
  gate + profile/capability dimension) and write the spec doc + golden-binary
  conformance corpus.
- **Reconcile the two execution paths (the crux, D4):** the `.kpubin`/`DMProgram`
  reader + `IProgramExecutor` exist but on behavioral/transactional models; the
  cycle-accurate CSP path (`ConcurrentTimingExecutor`) runs a non-serialized
  `ScheduleResult`. Decide + implement: give the CSP path a `.kpubin` front-end, or
  carry the schedule in the binary, or unify — so kpu-sim executes a *deserialized*
  program with **no inline generation**.
- **DoD:** the ResNet program → serialize `.kpubin` → read back → execute on the CSP
  timing model → **identical cycles** to the inline path (guarded by
  `resnet_regression`). Proves kpu-sim consumes a binary program.

### Phase 1 — Hardware-identical execution: ResourceManager setup + host memory  *(kpu-sim)*
- Execute the binary program through the **hardware flow**: `ResourceManager`
  allocates/sets up fabric resources per the program, deposits DNN data (shapes) in
  `HOST_MEMORY`, signals "ready → start", executes — the same APIs a real KPU exposes.
- **DoD:** `m2_resnet --run-program model.kpubin` executes a serialized program end-to-end
  through the resource-manager APIs; timing matches the direct path.

### Phase 2 — Relocate the KPU compiler backend to domain_flow  *(domain_flow / shared lib)*
- Move kpu-sim's schedule-generation (tiling/scheduling/dataflow strategy — the
  matmul/conv2d/… generators + graph-walk) into domain_flow (or a shared
  `kpu-backend` library the compiler links). domain_flow: `DomainFlowGraph` → `.kpubin`
  binary conforming to the spec (D5). Retire kpu-sim's `ComputationalGraph`.
- **DoD:** domain_flow emits a `.kpubin` from a `DomainFlowGraph` that kpu-sim executes
  with **identical results to P0's golden** (conformance suite passes).

### Phase 3 — Direct `ONNX → DomainFlowGraph` front-end  *(domain_flow; gap G1)*
- Per D1: direct ONNX-protobuf → `DomainFlowGraph` reader (no MLIR) for the CNN op
  set — map ONNX ops → `DomainFlowOperator`, read tensor **shapes** (values optional,
  D2). Then P2's backend compiles it to `.kpubin`.
- **DoD:** torchvision `resnet18.onnx` → `DomainFlowGraph` → `.kpubin` → runs on kpu-sim
  with **correct topology + timing** (numerical output not yet validated — P5).

### Phase 4 — Program cache + data-path personalization  *(kpu-sim; G7, G8)*
- **Program cache**: keyed compiled `.kpubin` programs (op sig, shapes, dtype, fabric
  cfg) → fast-load on repeat. Productionize `ScheduleBinder` to bind program ops →
  concrete fabric resources (DMA/BM/STR ids, L3/L2/L1) = "personalize the data path".
- **DoD:** second run fast-loads from cache (no recompile); the bound program runs on
  the personalized fabric.

### Phase 5 — Weight values, numerical validation & model zoo  *(both)*
- **Real weight values** (deferred half of D2): weight blob → `HOST_MEMORY`; ops
  reference by handle. Add an **onnxruntime reference** and validate numerically (G10).
- Extend to transformer ops (`DomainFlowOperator` gains softmax/layernorm/attention),
  the milestone-ladder models; retire superseded synthetic builders.
- **DoD:** ≥2 real models beyond ResNet run from disk with reference-validated output.

---

## 7. Recommended first increment (thin vertical slice)

**Phase 0 for ResNet-18: serialize the ResNet schedule to a `.kpubin` binary, read it
back, and execute it.** Proves the pivot — build hygiene → binary program format
(D5) → kpu-sim reader → `ScheduleExecutor` → **identical cycles** to the inline path
— **without** the compiler backend (P2), the ONNX front-end (P3), or weight values
(D2). Guard it with the existing `resnet_regression` harness (identical cycles ⇒ the
serialize/execute round-trip is lossless). This establishes kpu-sim as a pure
consumer of the binary contract; the compiler (P2–P3) is then built to *produce* that
same binary, validated against P0's golden.

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

## 9. Open questions (remaining, D1–D5 resolved)

1. ~~`.dfg` contract versioning~~ — **analyzed** in `dfg-kpu-versioning.md`: version
   `.dfg` **in place** (3 axes + `min_consumer` gate + profile dimension + golden
   corpus), and version the `.kpubin` binary from day one. Feeds Phase 0.
2. Epic sub-issue granularity — one per phase, or per phase-DoD?
3. Fine sequencing vs. M4 (agreed: run alongside) — which phases interleave with M4
   vs. run after it?
