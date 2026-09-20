# ADR 0001 — The executable program format and the transactional engine

| | |
|---|---|
| **Status** | **Accepted** (2026-09-19) — see §7 for the answers recorded on acceptance |
| **Date** | proposed 2026-09-18, accepted 2026-09-19 |
| **Issue** | #253 |
| **Context docs** | `docs/architecture/program-execution-assessment.md` (#252), `docs/plans/kpu-program-model.md` (D6), `docs/plans/model-ingestion-compilation-epic.md` (#229) |
| **Affects** | #229, #230, #231, #254, #255, #256, #257 |

---

## 1. Context

The goal is a **working functional transactional simulator that executes programs** —
BLAS first, then MLP, then progressively larger DNNs. The assessment in #252 found that
the obstacle is not missing operators:

- **Four program representations** exist (`.kpubin` DMProgram, `.kpukernel`, DFX `.kpu`,
  and the in-memory L0 `TileProgram` / L1 `StreamProgram`). No compiler output can be
  loaded and executed.
- **Eight execution engines** exist. None of them does all three things together: execute
  a program artifact, compute values, and produce transactional-grade timing.
- The only end-to-end transactional path (`kpu-loader` → `TransactionalProgramExecutor`)
  computes correct matmul values but reports **0 compute cycles** — `schedule_compute` is
  never called — and has no credits or buffer capacity.
- Fidelity selection is honored only by `create_program_executor`, where
  `CYCLE_ACCURATE` returns `nullptr`.

The ingestion epic already names this risk: **G9, "must connect, not add a fourth
path."** This ADR is the connection. Its job is to pick one program and one executor per
tier, and to retire the rest. It must not become a ninth engine.

### What already exists that the decision builds on

- **D6** (`kpu-program-model.md`) already establishes the three roles. The compiler
  produces a portable program (the PTX analog). The driver JIT lowers it to a device
  data-path config, `.kpubin` (the SASS analog). kpu-sim is the hardware. D6 also maps
  fidelities: *"L0 tile-sequences → BEHAVIORAL/functional validation; L0+L1 →
  TRANSACTIONAL/CYCLE_ACCURATE timing."*
- **`TileProgramReference`** executes L0 op by op (`exec_matmul_accum`,
  `exec_lu_diag_factor`, …) and is the functional oracle.
- **The characterization harness** (`program/characterize/`) already recovers the
  tile-dependency DAG from declared tile I/O (`tile_dag.hpp`). It models finite compute
  tiles and movement lanes on a `DeviceDescriptor`, uses L1 wavefront latencies when
  present, and checks L3 capacity feasibility (`characterization.hpp:143`).

## 2. Decision drivers

1. **One path.** Every rung of the BLAS → MLP → DNN ladder should land in the same
   program format and the same executor.
2. **Functional first.** The transactional tier must return the same numbers as the
   functional oracle. Timing without trustworthy values is not a simulator.
3. **Scales to real sizes.** ResNet at 224×224 is "intractable cycle-by-cycle"
   (`m2_resnet.cpp:261`). The transactional tier must be event-driven.
4. **Consistent with D6.** Don't reopen the program-model design. Implement it.
5. **Calibrated, not invented.** Transactional timing coefficients should come from the
   cycle-accurate tier, following the multi-fidelity philosophy in `CLAUDE.md`
   (characterize statistics from cycle-accurate, then build transactional models).

## 3. Options considered

### A. DMProgram `.kpubin` + `TransactionalProgramExecutor` (the `kpu-loader` path)

- **For:** it runs end to end today, and a serializer, assembler and loader exist.
- **Against:**
  - D6 §5 defines DMProgram as the driver-JIT output: device-bound and over-sequenced.
    Making it the portable program would reverse D6.
  - Its timing is a post-hoc overlay computed after the values, with no credits.
  - Compute is a hardwired matmul triple loop
    (`behavioral_program_executor.cpp:700-745`).
  - Bias and activation are serialized but never executed.

### B. L0 `TileProgram` as the portable program + a new tile-transaction executor

The executor is built from the reference kernels (for values) and the harness's
DAG/resource model (for timing).

- **For:**
  - It *is* D6. It is tile-algorithm-native, which suits BLAS/PLASMA.
  - It is event-driven.
  - Most of the parts exist: the per-op kernels, DAG recovery, device descriptor,
    L1 latencies and capacity feasibility.
- **Against:**
  - There is no L0 serializer yet.
  - Only matmul/LU tile kinds exist.
  - DNN ops must be added as tile kinds.
  - The harness's resource model is coarse: one aggregate "move lane" pool, and L3
    capacity checked statically rather than enforced dynamically.
  - Coefficients are uncalibrated.

### C. CSP `ConcurrentTimingExecutor` + a program front end

- **For:** it has the richest value path (all M1–M3 operators) and real credit, tag-CAM
  and queue contention.
- **Against:** it steps every cycle, so it is the cycle-accurate tier by construction.
  Using it as the transactional tier gives up driver 3.

## 4. Decision (proposed)

### D1 — The portable program is the L0 `TileProgram`, per D6

The portable, serializable, versioned KPU program is **L0 (tile sequences with explicit
tile I/O), optionally annotated with L1 (stream signatures) and the compute recurrence**.
This adopts D6 as written.

`DMProgram` / `.kpubin` is **only** the driver-JIT output (the data-path config). It is
never a compiler target, and never the file a user hands the simulator to "run a
program."

### D2 — One program, three tiers, one factory

| Tier | Consumes | Executor | Role |
|---|---|---|---|
| BEHAVIORAL | L0 | `TileProgramReference` | functional oracle |
| **TRANSACTIONAL** | L0 + placement + `DeviceDescriptor` (+ L1 when present) | **new tile-transaction executor** (D3) | values + queue/credit timing at tile granularity |
| CYCLE_ACCURATE | L0 → driver JIT → data-path config | CSP `ConcurrentTimingExecutor` | timing authority; calibration source |

A single factory selects the executor from `SimulationFidelity`. It is the **only**
place fidelity is read for program execution. `kpu-loader`, `KPURuntime`, the C API and
the Python backend all route through it (#257).

#### D2.1 — The load contract (how host code supplies these)

The existing interface cannot express D2's inputs, and that gap has to be named rather
than left to the implementation. Today `create_program_executor(SimulationFidelity,
HardwareContext&)` returns an `IProgramExecutor` whose `load_program(const DMProgram&,
a_base, b_base, c_base)` takes a DMProgram and three base addresses
(`include/sw/kpu/isa/program_executor_interface.hpp:49,114`). `HardwareContext` carries
memory components — no placement, no `DeviceDescriptor`.

The decision:

1. **A second, additive interface** for the portable program, rather than overloading the
   DMProgram one:

   ```cpp
   struct TileExecutionRequest {              // everything the tiers need, one struct
       TileProgram&              program;     // mutated in place: values live in operands
       const Placement&          placement;   // default single-topology until #230 incr. 3
       const DeviceDescriptor&   device;
       const stream::StreamProgram* streams = nullptr;   // optional (§7.4)
       std::uint64_t             seed = 0;
   };

   class ITileProgramExecutor {               // BEHAVIORAL + TRANSACTIONAL
       virtual RunResult run(const TileExecutionRequest&) = 0;
   };

   std::unique_ptr<ITileProgramExecutor>
   create_tile_program_executor(SimulationFidelity, const TileExecutionRequest&);
   ```

2. **Tensor binding replaces base addresses.** Values live in `TensorOperand::values`
   inside the program, named by operand, so the A/B/C base-address triple disappears
   rather than being generalized. `kpu-loader`'s hardcoded A/B/C and fixed buffer counts
   (`tools/runtime/kpu-loader/main.cpp:185-213`) are replaced by operand-named bindings.
   Host memory residency for real weights stays #231's job on the cycle-accurate path.
3. **Ownership.** The caller owns the program and the device descriptor. The **placement
   is produced by the driver JIT** and owned by the request; until #230 increment 3 lands,
   a `Placement::single(device)` factory supplies the default. Placement is an explicit
   object from the start, so the JIT later replaces a value rather than introducing a
   parameter.
4. **`SimulationFidelity::CYCLE_ACCURATE` on this interface is a distinct path**, not a
   third implementation of `ITileProgramExecutor`: it lowers L0 → data-path config via the
   JIT and runs the CSP engine. Until that lowering exists (#230 increments 2–3) the
   factory **throws with a message naming the missing capability** instead of returning
   `nullptr`, which is the current failure mode
   (`program_executor_interface.cpp:140-143`).
5. **The legacy DMProgram interface is frozen, not migrated.** `IProgramExecutor` and its
   two implementations keep their present contract until they retire (§7.2), so no adapter
   is written between the two interfaces — the deliberate alternative to a migration
   shim. #257's factory routing is complete when every host surface reaches **either** the
   new interface or the CSP path, and nothing reaches a timing-only executor by default.

The transactional tier needs the JIT's **placement** (which L3 tiles and which compute
tiles an op uses; D6 §4a). It does **not** need the per-engine DMA/BlockMover/Streamer
instruction streams. Until the placement pass lands (#230 increment 3), the executor
uses a default single-topology placement, which is what the harness assumes implicitly
today.

### D3 — The transactional executor's contract

These are requirements. The implementation design follows in its own plan.

1. **Values are bit-exact to the oracle.** Each op executes with the same per-op kernel
   `TileProgramReference` uses. Ops declare their full tile I/O (D6 §3a), and WAW
   ordering serializes accumulation into a tile, so running ops in dataflow order gives
   **bit-identical** results to running them in program order. That is the acceptance
   test.
2. **It reacts; it does not schedule.** An op fires when three things hold: its declared
   input tiles are resident at its assigned compute tile, the resource is free, and
   downstream credit is available for its output. This is the credit-based dataflow
   execution model (`CLAUDE.md`, "Credits UP, Data DOWN") at tile granularity. The
   harness's greedy critical-path list scheduler is an *optimizer*, so it belongs to the
   placement pass, not the executor. It stays in the harness as the analytical lower
   bound.
3. **Buffer capacity is enforced dynamically**, as tile credits per buffer level. An
   over-committed program stalls or is refused, rather than reporting an impossibly
   good number (#254).
4. **Movement is split by hop, at minimum DRAM→L3 and L3→compute-tile.** Each hop has
   its own lanes and bandwidth, not one aggregate pool.
5. **Durations come from calibrated coefficients.** They are fitted against the CSP tier
   and asserted within a documented error band on a sweep. They are never zero for
   compute (#254).
6. **Event-driven and deterministic.** It never steps idle cycles. Any stochastic
   latency is seeded from config, and the seed is recorded in the results.
7. **Unknown ops are refused loudly.** An op the executor cannot perform aborts with the
   op name, never gets skipped (#256).

### D4 — The transactional tier computes exact values (a change to the tier contract)

The fidelity table in `CLAUDE.md` lists TRANSACTIONAL as computing values
"Statistically", and `docs/02-simulation/fidelity-framework.md` describes Level 1 by its
timing only. **This ADR changes that.** The transactional tier computes **exact** values,
identical to BEHAVIORAL, and differs only in adding timing.

The reason is the goal: a *functional* transactional simulator that validates programs.
It costs almost nothing, because the kernels are shared. On acceptance, `CLAUDE.md` and
the fidelity framework doc get updated.

### D5 — Authorities

- **Values:** the L0 `TileProgramReference` is authoritative. The transactional tier
  must match it bit-exactly. The cycle-accurate tier must match it within a stated
  floating-point tolerance, because accumulation order may differ.
- **Timing:** the CSP cycle-accurate tier is authoritative. The transactional tier is
  calibrated against it and reports its error band.

### D6 — Engine disposition

Each retirement gets its own issue after acceptance, per #253's definition of done.

| Engine | Disposition | Condition |
|---|---|---|
| `TileProgramReference` (L0) | **Keep**: BEHAVIORAL tier and value oracle | — |
| Characterization harness (`TileDag`, `DeviceDescriptor`) | **Evolve**: its DAG recovery and resource model seed the transactional executor; the list scheduler stays as the design-of-experiments and lower-bound tool | — |
| CSP `ConcurrentTimingExecutor` + wrappers | **Keep**: CYCLE_ACCURATE tier; gains a front end from the JIT output (#230 increments 2–3) | — |
| `TransactionalProgramExecutor` (DMProgram) | **Freeze** now, then retire | when the L0 transactional executor reaches matmul parity and `kpu-loader` is repointed |
| `BehavioralProgramExecutor` (DMProgram) | **Freeze**, then retire | when nothing needs to execute `.kpubin` outside the CSP tier |
| `isa::ConcurrentExecutor` (timing-only, behind `KPURuntime` and the C API) | **Freeze**, then retire | when the runtime routes through the factory (#257); frozen first per §7.2 |
| Legacy `isa::ProgramExecutor` | **Freeze**, then retire | when the new path covers its tests' intent; its dependents are `tests/isa/test_data_movement_isa.cpp` and `examples/basic/data_movement_isa_matmul.cpp`, which retire with it |
| OFG flow executors (`models/dataflow/`) | **Freeze**, then retire (`execute_operation` is a no-op) | when the new path covers its tests' intent; its dependents are three tests in `tests/dataflow/` and `examples/behavioral/ofg_trace_demo.cpp`. **`CLAUDE.md` currently lists these under "USE THESE"** as the correct dataflow reference — that section is corrected **now**, not on deletion, because it points readers at a no-op compute path |
| `models/transactional` component classes | **Freeze**, then retire from the program path (no C++ callers) | with the Python rework (#257), their only consumer; frozen first per §7.2 |
| `models/behavioral` orchestrator and executors | **Freeze**, then retire | when the new path covers its intent; its only external dependent is `examples/behavioral/matmul_behavioral.cpp` |
| `KPUSimulator` temporal components | **Keep as a component library** for fidelity elevation (e.g. LPDDR5); no program-execution path | — |

### D7 — How DNN operators enter

DNN operators become **new `TileOpKind`s with reference kernels**: elementwise, the
bias+activation epilogue, reduce, softmax and norms. Conv2D lowers to im2col + GEMM, as
it does on the CSP path today. They do **not** become new executors, schedule
generators, or KernelGraph-only paths.

The CSP runners that implement these ops today (`csp_op_runners.hpp`) stay the
cycle-accurate implementations. They are reached through JIT output, not through a
parallel graph bridge.

## 5. Consequences

### Positive

- One program flows through every tier. A BLAS routine or an MLP layer is written once,
  as tile ops, and runs at all three fidelities.
- The transactional acceptance test is sharp: bit-exact against the oracle, plus timing
  within a band of cycle-accurate.
- Real model sizes become tractable at transactional fidelity.
- The engine count goes down, which answers G9.

### Negative and costs

- The M1–M3 demos stay on the CSP/KernelGraph path until their ops exist as tile kinds.
  The DNN rung is re-routed, not redone, but it is real work.
- `kpu-loader` and `.kpubin` stop being the "run a program" story. Existing users of that
  path need the parity condition in D6 before it retires.
- The harness's resource model must grow: per-hop movement and dynamic capacity. It is
  a starting point, not a finished executor.

### Changes to issues already filed

- **#230:** reorder the increments for the transactional rung. **L0 serialization
  (increment 4) moves ahead of the full driver JIT (increment 3)**, because the
  transactional tier needs only a default placement. Split increments 3 and 4 into their
  own issues, as #253 already requires.
- **#254:** retarget from `TransactionalProgramExecutor` to the new executor. Fixing
  timing in a frozen engine is wasted work.
- **#255:** versioning applies to the L0 format first. `.kpubin` versioning still matters
  as the JIT-output format.
- **#256:** becomes "add tile kinds for bias/activation/elementwise/reduce", not
  "teach the DMProgram executors to run the epilogue."
- **#257:** routes every host surface through the D2 factory.

### Risks

- **Calibration drift.** Transactional timing is only as good as its calibration. The
  error band must be a CI assertion, not a one-time report.
- **Placement coupling.** Timing depends on placement. A poor default placement can
  make the transactional tier look slower than the hardware would be. Report which
  placement was used with every result.
- **Scope creep toward cycle-accurate.** The executor must stay at tile granularity.
  Element-level detail belongs to the CSP tier.

## 6. First milestone: how we'll know it works

1. A GEMM `TileProgram`, with random non-square inputs and ragged trailing tiles, is
   serialized to a file, loaded, and executed through the factory at TRANSACTIONAL.
2. Its values are bit-exact against `TileProgramReference`.
3. Compute time is non-zero and scales with M, N, K and tile size.
4. Runs are deterministic.
5. Over a size sweep, its makespan is within the documented band of the CSP tier.
6. The same file, with the same checks, then runs tile LU (confined pivoting).

## 7. Answers recorded on acceptance

1. **D4 accepted.** The transactional tier computes exact values. `CLAUDE.md`'s fidelity
   table and the Level 1 section of `docs/02-simulation/fidelity-framework.md` **still say
   otherwise and are pending** — tracked in #267, which waits on #251 because that PR
   rewrites the same sections. Until #267 lands, those two documents state the old
   statistical-value contract and the ADR overrides them.
2. **Retirement timing: freeze now, delete at parity.** All seven D6 targets are frozen
   immediately — no new features, no new callers, no new tests — and each is deleted only
   on its **own** condition. They are not interchangeable:

   | # | Target | Deletion condition |
   |---|---|---|
   | 1 | `TransactionalProgramExecutor` (DMProgram) | `TileTransactionExecutor` reaches matmul parity **and** `kpu-loader` is repointed |
   | 2 | `BehavioralProgramExecutor` (DMProgram) | no component needs to execute `.kpubin` outside the cycle-accurate tier — **not** shared with row 1, since `.kpubin` consumers may outlive the transactional one |
   | 3 | `isa::ConcurrentExecutor` (timing-only) | the runtime and C API route through the D2 factory (#257) |
   | 4 | Legacy `isa::ProgramExecutor` | the new path covers the intent of `tests/isa/test_data_movement_isa.cpp` and `examples/basic/data_movement_isa_matmul.cpp` |
   | 5 | OFG flow executors (`models/dataflow/`) | the new path covers the intent of the three `tests/dataflow/` tests and `examples/behavioral/ofg_trace_demo.cpp` |
   | 6 | `models/transactional` component classes | the Python native path stops consuming them (#257), their only consumer |
   | 7 | `models/behavioral` orchestrator and executors | the new path covers the intent of `examples/behavioral/matmul_behavioral.cpp` |

   Every existing test and example stays green until its row's condition is met, so no
   coverage is dropped ahead of a replacement. Rows 3 and 6 are marked **Retire** in D6's
   table because nothing legitimate depends on them today; they are still frozen first, so
   the sequencing here governs.

   The `CLAUDE.md` "Implementation Reference" rewrite is **not** deferred with row 5: that
   section points readers at a no-op compute path today, so it is corrected as part of
   #267 rather than at deletion time.
3. **Statistical variance: deterministic first.** Calibrated means only, with a recorded
   seed. Variance is added once there is CSP calibration data to fit it to, and is
   tracked as a follow-on, not part of the first executor.
4. **L1 at TRANSACTIONAL: optional.** When an L1 `StreamProgram` is present, compute ops
   take their systolic wavefront latency and drains are stretched by the C-stream bubble.
   Without it, the first-order lumped model applies. This is what the harness already
   does (`tile_dag.hpp:163-190`), so dataflow-sensitivity is opt-in rather than a
   prerequisite.
5. **Value tolerances (D5).**
   - Transactional versus the L0 reference: **bit-exact**, no tolerance. Both run the same
     kernels in a dependency-respecting order, and WAW ordering fixes accumulation order.
     Any difference is a bug, not rounding.
   - Cycle-accurate versus the L0 reference: **mixed absolute/relative**, since
     accumulation order legitimately differs. A pure relative error is undefined at a zero
     reference and unstable near zero — common in these tensors, where ReLU zeroes
     activations and triangular factors carry structural zeros. The comparator is:

     ```
     pass  ⟺  |actual − reference| ≤ atol + rtol · |reference|     (elementwise)
     ```

     with **`atol = 1e-6`** for float32, and `rtol` taken from the tolerances already in
     use on the CSP path: **1e-4** for the MLP oracle, **5e-3** for the composed CNN
     references. Report max absolute *and* max relative error, excluding from the relative
     figure any element whose reference magnitude is below `atol`. The LU and softmax
     `rtol` bars are set from measurement when those first run at cycle-accurate fidelity,
     rather than guessed here — but they use this same comparator.
6. **Name: `TileTransactionExecutor`**, in namespace `sw::kpu::program`. It names what it
   executes (tile transactions) and where it sits (the L0 program layer), and does not
   collide with the existing `TransactionalProgramExecutor` it eventually replaces.

## 8. Follow-up work

Tracked separately; this ADR does not carry the implementation.

| Issue | Work |
|---|---|
| **#264** | `TileTransactionExecutor` — the TRANSACTIONAL tier executor (D3's seven requirements), design note first |
| **#265** | L0 portable-program serializer, versioned; split out of #230 and sequenced **ahead** of the driver JIT |
| **#266** | Freeze and retire the superseded engines (D6), with each deletion condition from §7.2 |
| **#267** | Align `CLAUDE.md` (fidelity table, "Implementation Reference", authorities) and the fidelity framework's Level 1; waits on #251 |
| **#268** | Statistical variance for transactional timing, after calibration (§7.3) |

Done as issue hygiene rather than new work:

- [x] Re-pointed #230, #231, #254, #255, #256 and #257 per §5 (comments on each,
      2026-09-19). One open recommendation: **#254 should close as superseded by #264**,
      since every requirement in it is now part of that issue.
