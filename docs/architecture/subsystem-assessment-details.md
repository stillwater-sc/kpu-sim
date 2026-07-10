# KPU Simulator: Subsystem Assessment Details

**Scope:** This is the evidence appendix behind
[`simulator-assessment-and-plan.md`](simulator-assessment-and-plan.md) — the
detailed per-subsystem findings, with file-level citations, from the
repository survey conducted 2026-07-09 (six parallel deep investigations:
ISA/runtime API, behavioral tier, CSP timing tier, compiler/DSL, operator
coverage, and existing plans/roadmaps). Read the main document first; this
one exists so the assessment's claims are checkable and so follow-up work
doesn't have to re-derive the inventory.

**Snapshot caveat:** citations reference the tree as of 2026-07-09 (main at
PR #60, v0.8.5 released, v0.9 in progress). The CSP multi-tile livelock
described in §3 was fixed the following day (issue #61, PR #62); that
section has been annotated accordingly. Other line numbers will drift.

---

## 1. ISA and Runtime-Facing API

### 1.1 The instruction set — `include/sw/kpu/isa/data_movement_isa.hpp`

A substantial *data-movement* ISA (Domain Flow Architecture): the program IS
the data-movement schedule; compute fires reactively when data tokens arrive;
there is no explicit compute opcode.

- `DMOpcode` enum (`data_movement_isa.hpp:50-120`), ~50 opcodes:
  - **DMA** (External↔L3): `DMA_LOAD_TILE`, `DMA_STORE_TILE`,
    `DMA_PREFETCH_TILE`, `_AUTO` variants, `DMA_LOAD_GATHER` /
    `DMA_STORE_SCATTER`.
  - **BlockMover** (L3↔L2): `BM_MOVE_TILE`, `BM_TRANSPOSE_TILE`,
    `BM_WRITEBACK_TILE`, `BM_RESHAPE_TILE`, `_AUTO` variants.
  - **Streamer** (L2↔L1 / array feed): `STR_FEED_ROWS/COLS`,
    `STR_DRAIN_OUTPUT`, `STR_BROADCAST_ROW/COL`, `_AUTO` variants (drain can
    inline a Vector Engine activation/bias).
  - **Sync:** `BARRIER`, `WAIT_DMA/BM/STR`, `SIGNAL`.
  - **Config registers:** `SET_BASE`, `SET_L3_BASE`, `SET_L2_BASE`,
    `SET_STRIDE`, `SET_TILE_DIM`, `SET_MATRIX_DIM` (plus deprecated
    `SET_TILE_SIZE`/`SET_BUFFER`).
  - **Hardware loops:** `LOOP_BEGIN`/`LOOP_END` with `IndexRole` (TI/TJ/TK)
    binding for AUTO address generation.
  - **Vector Engine / scratch:** `VE_ELEMENTWISE`, `VE_REDUCE`,
    `L2_SCRATCH_READ/WRITE`; `NOP`, `HALT`.
- **Encoding:** not a bit-packed hardware word format. Instructions are C++
  structs (`DMInstruction`, `data_movement_isa.hpp:398`) with a
  `std::variant` of per-category operand structs, timing hints
  (`earliest_cycle`/`deadline_cycle`), dependency lists, and a debug label.
  The `.kpubin` "encoding" is a serialization format (§1.4).
- Factory constructors (`dma_load`, `bm_move`, `str_feed_rows`, `set_base`,
  `loop_begin`, `*_auto`, …) at `:439-504`.
- `DMProgram` (`:518`) carries metadata (name, M/N/K, tiling, dataflow
  strategy: OUTPUT/WEIGHT/INPUT_STATIONARY), the instruction stream, a
  `MemoryMap` (L3/L2 allocations), and SURE-analysis `Estimates`.
- `OutputStationaryProgramBuilder` (`:597`) programmatically emits matmul
  schedules with tile-cache tracking — in practice the primary
  program-generation path, more so than the assembler.

### 1.2 Assembly language — `docs/kpuasm-specification.md`

v1.0, dated 2026-02-03. Defines `.kpuasm` text → `.kpubin` binary, comments
(`;`/`#`), labels, directives (`.name`, `.version`, `.dimensions`, `.tiling`,
`.l1_ki`, `.dataflow`, `.a_base/.b_base/.c_base`), per-opcode operand syntax,
two worked matmul examples, and the CLI `kpu-assembler input.kpuasm -o
output.kpubin`.

**Spec/implementation drift:** every specified opcode/directive has a parser
branch in `src/software/isa/assembler.cpp:269-324`, but the implementation is
ahead of the spec: AUTO addressing, `IndexRole` on `LOOP_BEGIN`, and the
enhanced `SET_BASE`/`SET_TILE_DIM`/`SET_MATRIX_DIM` opcodes are implemented
but undocumented; the spec's `SET_STRIDE stride_m,stride_n,stride_k` differs
from the implemented `SET_STRIDE matrix,...` form.

### 1.3 C API surface — `include/sw/kpu/kpu_c_*.h`

All five headers implemented in `src/tools/bindings/c/` (`kpu_c_api.cpp`,
`kpu_c_runtime.cpp`, `kpu_c_executor.cpp`):

- `kpu_c_api.h` — legacy simulator API: `kpu_create/destroy`, raw memory
  read/write, `kpu_dma_transfer_sync`, and legacy functional compute
  `kpu_matmul_f32` / `kpu_matmul_accumulate_f32` (these DO produce numbers,
  via the low-level simulator, not the ISA).
- `kpu_c_runtime.h` — CUDA-like runtime: create/destroy, `malloc/free`,
  `memcpy_h2d/d2h/d2d`, `memset`, `kpu_runtime_launch` (kernel + arg
  addresses → `KPULaunchResult` with cycle count), async launch, streams,
  events, device-info/stats.
- `kpu_c_kernel.h` — kernel factories only: `kpu_kernel_create_matmul`,
  `kpu_kernel_create_mlp`, metadata/statistics accessors.
- `kpu_c_executor.h` — `GraphExecutor`: auto tensor alloc,
  `set_input`/`get_output`, `execute`.
- `kpu_c_types.h` — handles, enums, result structs.

**Critical gaps:**

1. **No program loading.** No C API function loads a `.kpuasm`/`.kpubin` or
   serialized `DMProgram`. Kernels are only constructible via
   `create_matmul`/`create_mlp` (which internally run `KernelCompiler`,
   `kernel_compiler.cpp:277,401`). Assembler output is unreachable from the
   C API.
2. **No numeric execution through the runtime.** `kpu_runtime_launch` →
   `KPURuntime::launch` → `ConcurrentExecutor::execute` (`runtime.cpp:164`).
   `ConcurrentExecutor` is a timing/scheduling model constructed from a
   `ResourceConfig` with no reference to memory buffers
   (`concurrent_executor.cpp:192-360`); it never reads/writes tensor data.
   `kpu_executor_get_output` returns whatever bytes are already in device
   memory. The driver kernel tests (`tests/driver/test_kernel.cpp`) assert
   only metadata/shape/flops, never numeric correctness — consistent.
3. **Type-system split:** `ConcurrentExecutor` uses its own internal
   `ResourceType` (e.g. `COMPUTE_FABRIC`), distinct from
   `resource_handle.hpp`'s `COMPUTE_TILE` enum.

### 1.4 Assembler / serializer / executors — `src/software/`

- **Assembler:** `src/software/isa/assembler.cpp` (~1150 lines) + CLI tool
  `tools/development/kpu-assembler/` (wired in
  `tools/development/CMakeLists.txt:11`). Produces `DMProgram`.
- **Serializer:** `src/software/isa/program_serializer.cpp` (~980 lines):
  binary `.kpubin` (magic + version header, instructions, memory map,
  estimates) with `save()`/`load()`, plus JSON via nlohmann. Round-trips.
  **But no `.kpubin → Kernel` loader exists anywhere.**
- **Four executor engines:**

| Engine | File (src/software/isa/) | Numerics | Timing | Wired into C API? |
|---|---|---|---|---|
| `BehavioralProgramExecutor` | `behavioral_program_executor.cpp` (859 ln) | **Yes** (triple-loop matmul, real L1/L2/L3/ext memory) | No | **No** — tests/examples only |
| `TransactionalProgramExecutor` | `transactional_program_executor.cpp` (873 ln) | Yes (behavioral core) | Yes (analytical overlay + Chrome trace) | **No** — tests/examples only |
| `ConcurrentExecutor` | `concurrent_executor.cpp` (866 ln) | **No** | Yes (multi-resource scheduling) | **Yes** — this is what the runtime uses |
| `ProgramExecutor` (older) | `program_executor.cpp` (479 ln) | partial | partial | No; "Not yet implemented" at `:156,162` |

- Behavioral executor: several opcodes are deliberate no-ops —
  `STR_BROADCAST_*`, `DMA_LOAD_GATHER`/`STORE_SCATTER`,
  `VE_ELEMENTWISE`/`VE_REDUCE`, `L2_SCRATCH_*`
  (`behavioral_program_executor.cpp:221-235`).
- Fidelity factory `create_program_executor(fidelity, hw)`
  (`program_executor_interface.cpp:129-148`): BEHAVIORAL →
  `BehavioralExecutorWrapper`, TRANSACTIONAL → `TransactionalExecutorWrapper`,
  **CYCLE_ACCURATE → `nullptr`** ("Not yet integrated", `:141-143`). Not
  called by the C API/runtime; requires a `HardwareContext` of real memory
  components (`program_executor_interface.hpp:87-92`).
- Verified numerics:
  `tests/isa/test_behavioral_program_executor.cpp:79,139-152,195-219` checks
  `C[i,j]` against `reference_matmul` within tolerance.

### 1.5 Resource API — `resource_api.hpp` / `resource_handle.hpp`

Implemented (`src/system/simulator/resource_manager.cpp`). `ResourceType`
(`resource_handle.hpp:19-38`): `HOST_MEMORY`, `EXTERNAL_MEMORY`, `L3_TILE`,
`L2_BANK`, `L1_BUFFER`, `PAGE_BUFFER`, `COMPUTE_TILE`, `DMA_ENGINE`,
`BLOCK_MOVER`, `STREAMER`. `ResourceManager` (`resource_api.hpp:53`):
discovery, bump-allocator allocate/deallocate, read/write/copy/memset,
busy/ready + `wait_ready`, address→resource lookup, full statistics.
Functional; the C-API memory ops sit on it.

**Bottom line:** `.kpuasm` assembles, serializes, and executes with correct
numerics in C++ — but not end-to-end through the runtime-facing API. The
assembler and the C API are on opposite sides of a wall; the C API's engine
computes cycles, not values.

---

## 2. Behavioral Tier and Fidelity Framework

### 2.1 Four execution worlds

| Engine | Location | Real values? | Sequencing/credits? |
|---|---|---|---|
| Main `KPUSimulator` | `src/system/simulator/kpu_simulator.cpp` | **Yes** (temporal compute fabric) | Partial — per-component latency FSMs, **no credit backpressure** |
| Behavioral tier | `models/behavioral/`, `isa/behavioral_program_executor` | **Yes** (memcpy + triple loop) | **No** — instant, program order |
| OFG dataflow executors | `models/dataflow/` | **No** (counters/events) | **Yes** — order + credit tokens |
| CSP timing tier | `include/sw/kpu/timing/` | **No** (tag/credit descriptors) | **Yes** — credit dataflow, contention |

### 2.2 Main simulator — `include/sw/kpu/kpu_simulator.hpp`

`KPUSimulator` (hpp:52) delegates over value-semantic component vectors
(hpp:142-151): host memory regions, memory banks, L3 tiles, L2 banks, L1
buffers, page buffers, DMA engines, compute tiles, block movers, streamers.
`sw::memory::AddressDecoder` (hpp:154) provides a unified global address
space; `start_dma_transfer` routes by address range. Execution entry points:
`start_dma_transfer`, `start_block_transfer`, `start_row/column_stream`,
`start_matmul` (hpp:266), `step()`/`run_until_idle()` (hpp:279-280).

Key facts:
- Every included component is from `models/temporal/` (hpp:36-44); there is
  **no `SimulationFidelity` field in `KPUSimulator::Config`** — the main
  simulator is hard-wired to the temporal tier.
- `step()` (`kpu_simulator.cpp:465-496`) advances the cycle and calls each
  component's `process_transfers`/`update`; ordering is latency-driven, not
  credit-driven — stages couple only through shared memory arrays.
- The temporal `ComputeFabric::execute_matmul` computes real values
  (`src/models/temporal/compute/compute_fabric.cpp:185-215`).

### 2.3 Behavioral models

Directly under `include/sw/kpu/behavioral/`: only `l3_cache_model.hpp` —
the **deprecated** LRU/FIFO/RANDOM tile-reuse cache analyzer with
HIT/MISS/REFETCH statistics (contradicts the execution model; removal
candidate) — and `tiled_matmul_program.hpp`.

The real behavioral models are under `include/sw/kpu/models/behavioral/`:
- `BehavioralOrchestrator` (`orchestrator.hpp:36`) — Host→DMA→L3→BlockMover→
  L2→Compute→VectorEngine→…→Host, "all operations are instant" (hpp:9-10);
  real `execute_matmul`, `execute_tiled_matmul`, `execute_mlp_layer`
  (hpp:107-141).
- `BehavioralMLPExecutor` (`mlp_executor.hpp`) — full MLP forward pass.
- `BehavioralComputeFabric` (`compute/compute_fabric.hpp:51`) — implements
  `IComputeFabric`, type-dispatched real kernels, always-ready pipeline.
- Plus behavioral DMA engine, block mover, memory model/controller, L3 tile,
  vector engine, NoC.

Used by `examples/behavioral/` and `tools/runtime/kpu-loader/main.cpp`. All
instant/functional; none model credits.

### 2.4 Fidelity framework — mostly scaffolding

- `fidelity/simulation_fidelity.hpp:20-38` — the enum;
  `fidelity/component_config.hpp:21-45, 307-414` — per-component config
  structs + `SimulatorConfig` with `set_fidelity` etc.
- `SimulatorConfig` is parsed/serialized
  (`src/system/config/simulator_config_parser.cpp`, JSON in `configs/`) —
  **but never consumed by `KPUSimulator`.**
- The only end-to-end fidelity dispatch is `create_program_executor`
  (§1.4); 2 of 3 tiers reachable, DMProgram execution only.
- `fidelity_status.txt:50` leaves "Wire OFG executors with behavioral
  callbacks" unchecked, labeled "optional" — this is exactly the
  values+ordering unification the plan's Phase 1 promotes to core scope.
- Interface/factory seam exists (`models/interfaces/` +
  `*_factory.cpp`) but the transactional tier behind it has only 4
  components (`models/transactional/`: compute fabric, DMA engine, L3 tile,
  memory controller).

### 2.5 OFG dataflow executors — `include/sw/kpu/models/dataflow/`

Order-faithful, value-free, and orphaned:
- `operand_flow_graph.hpp` — full OFG data model: `Operand`, `FlowNode`,
  `FlowEdge`, `Operation` enum (LOAD/STORE/PUSH_TO_L2/SEND_EAST…/FEED_WEST/
  FEED_NORTH/DRAIN/MATMUL/…), `BUFFER_TOKEN` for backpressure (hpp:36).
- `flow_graph_executor.hpp` — token readiness, fire-when-ready, deadlock
  detection (hpp:255-262); **`execute_operation` is a no-op in the base
  (hpp:342-345)**; `step()` fires all ready nodes per cycle (hpp:227-240) —
  a topological wave, not resource-throttled.
- `dma_flow_executor.hpp` — credit modeling (`max_outstanding`,
  hpp:152-160), emits TILE_READY/BUFFER_AVAILABLE (hpp:163-210); no memcpy.
- `block_mover_flow_executor.hpp` — L2 bank occupancy credit checks
  (hpp:251-261, 90-100), mesh SEND routing, C/A/B-stationary graph builders;
  no data.
- `streamer_flow_executor.hpp` — accumulator state + FLOP counter
  (hpp:237-273); A+B presence gating (hpp:195-205); no actual MAC.

Instantiated only from `tests/dataflow/` — never from `src/`.

---

## 3. CSP Timing Tier (v0.9)

Header-only DES under `include/sw/kpu/timing/` (`src/timing/` is empty).
Moves `TileDescriptor` metadata (`tile_descriptor.hpp:115-154` — no payload
field); no dependency on behavioral models (verified: no float/byte payloads,
no memcpy anywhere in the tier; includes only other `timing/` headers plus
`isa/data_movement_isa.hpp` for `MatrixID`).

### 3.1 CSP primitives — complete and tested

- `credit_pool.hpp` (321 ln): `CreditPool` (non-blocking acquire/release,
  overflow/underflow guards; `acquire_blocking` deliberately throws in DES)
  + `PartitionedCreditPool` (A/B/C partitioning).
- `tag_cam.hpp` (316 ln): content-addressable tile tracker with
  **reference counting** for reuse. `insert()` on a duplicate increments
  ref_count; `invalidate()` returns true only when ref_count hits 0 (credit
  release signal). `find_any_ready`/`find_oldest`/`find_by_matrix` support
  work-conserving scans.
- `work_queue.hpp` (539 ln): FIFO with `at()`, `remove_at()`, `peek()` for
  out-of-order scans.
- `livelock_detector.hpp` (335 ln): progress-metric stall detection, checked
  every 100 cycles by the executor.

Note: the "3 pre-existing timing test failures" recorded in older status
notes (tag_cam duplicate-insert, BM/STR process names) are stale — the tests
were reconciled to current semantics and pass
(`tests/timing/test_tag_cam.cpp:47-56`, `test_block_mover_process.cpp:36`,
`test_streamer_process.cpp:39`).

### 3.2 Component processes

All implement `IProcess` (`process_interface.hpp`: `tick(cycle) →
vector<TimingEvent>`, `is_idle`, `has_pending_work`, `is_complete`, `reset`).

- `memory_controller_process.hpp` (556 ln): 1-command-per-cycle command bus,
  per-bank FSM (IDLE/ACTIVE/ACTIVATING/PRECHARGING), ROW_HIT/MISS/EMPTY with
  tCL/tRCD/tRP/tBurst, submit/poll keyed by `submitter_id`. Simplified:
  linear address map (noted at `:359`), no tRRD/tFAW/tWTR/tRTW/refresh.
- `dma_engine_process.hpp`: DRAM↔L3; L3 credit for loads, TagCAM match for
  stores, shared MC via submit/poll, tile-reuse dedup.
- `block_mover_process.hpp`: L3↔L2; one in-flight transfer; work-conserving
  scan + optional priority aging; MOVE acquires L2 credit + L3 tag match,
  WRITEBACK the reverse.
- `streamer_process.hpp`: L2↔Compute; FEED (releases L2 credit on consume)
  and DRAIN (waits on `compute_result_tag_cam`, acquires L2 credit). Row
  streamers = A/West, column streamers = B/North.
- **No compute process.** Compute is a fixed-latency delay inline in
  `ConcurrentTimingExecutor::step()` (lines 632-673 as surveyed): a
  `PendingCompute` starts when its `dependency_tile` appears in `fed_tiles_`,
  completes after `config.compute_latency` cycles. No arithmetic, no
  K-accumulation, K-independent latency.

### 3.3 Executor and schedule tier

- `concurrent_timing_executor.hpp` (~1000 ln): owns L3/L2 credit pools and
  L3/L2/compute-result TagCAMs, builds the component grid from `Config`,
  ticks MC → DMA → BM → row streamers → col streamers, exports Chrome
  trace/CSV, aggregates statistics.
- `timing/schedule/`: `ScheduleOperation`
  (LOAD/STORE/MOVE/WRITEBACK/FEED/COMPUTE/DRAIN), generators for matmul
  (4 strategies: OUTPUT_STATIONARY, INTERLEAVED_AB default, PREFETCH_NEXT,
  BLOCKED_AB), conv2d, softmax, layernorm, batchnorm; `schedule_validator.hpp`
  (609 ln); `schedule_executor.hpp` bridges schedules onto the executor.
- Fidelity gaps: COMPUTE depends only on the *last B feed*
  (`matmul_schedule_generator.hpp:230-260`), not all K-slice feeds;
  compute latency is K-independent; conv2d generator has a "Not implemented
  yet" branch (`conv2d_schedule_generator.hpp:179`) and simplified im2col
  addressing (`:323-330`).

### 3.4 Multi-tile livelock — RESOLVED

As surveyed (2026-07-09), `run_matmul` livelocked at its default 64³/16³
configuration. **Fixed 2026-07-10** (issue #61, PR #62). Actual root causes —
which differ from the survey's initial suspects — and the fix are documented
in `docs/sessions/2026-07-09_v0.9_multi_tile_livelock_fix.md`: silent DMA
request drop beyond `queue_depth`; L2 credit leak on tile reuse (duplicate
MOVEs each acquired a credit, ref-counted entry releases one — fixed by
implementing BlockMover move dedup per the generator contract); duplicate
in-flight loads double-acquiring L3 credits (fixed by in-flight dedup +
tile-affine work assignment); `static` slot counters shared across
instances. Post-fix, all strategies complete for 32³–256³.

### 3.5 LPDDR5 patterns — mature, but a separate model

Two memory-controller models exist. The CSP executor uses the simplified
`timing/memory_controller_process.hpp`; the mature one is
`models/temporal/memory/controllers/lpddr5_controller.hpp` (609 ln): full
LPDDR5-6400 `TimingParams` (tRCD/tRP/tRAS/tRC/tCL/tWL/tWR/tRTP, tRRD_L/S,
tCCD_L/S, tWTR_L/S, tRTW, tFAW, per-bank/all-bank refresh, BL16/32), 7-state
bank FSM, command-ownership tracking, `IMemoryController` interface,
trace integration. `patterns/memory/lpddr5/` is a substantial validation
suite: harness, workloads, `multi_fidelity.hpp`, `trace_validator.py`, and
`INVARIANTS.md` with 15 numbered invariants (INV-001..005 structure,
INV-100..105 timing, INV-200..201 visualization). Sibling controllers exist
for DDR5/GDDR6/GDDR7/HBM2/HBM3. Bridging this into the CSP executor is the
plan's Phase 6.

---

## 4. Compiler, DSL, Kernel Graphs, Schedules

Four partially overlapping toolchains and **two different things named DFX**
(documented in `include/sw/kpu/dfx/dfx_program.hpp:8-14`).

### 4.1 The DSL — works to DMProgram

- `include/sw/kpu/dsl/schedule.hpp:97-282` — fluent builder
  (`sw::kpu::dsl::Schedule` + `LoopScope`): tensors, tile sizes, dataflow,
  hardware config, loop bodies from primitives (load/store/move/writeback/
  stream_rows/stream_cols/broadcast/compute_matmul/compute_elementwise/
  compute_reduce/drain/drain_fused/barrier/double_buffer/if_not_resident/
  load_gather).
- `src/dsl/schedule_compiler.cpp` (455 ln) walks the IR and emits
  `isa::DMInstruction`s. **Compute ops emit `DMOpcode::NOP` with a label**
  (`:266-303`) — compute is implicit in streaming (Domain Flow semantics).
  The DSL is a data-movement scheduling language, manually parameterized
  (no autotuning).
- Prebuilt schedules (`include/sw/kpu/schedules/`, `src/schedules/`):
  `matmul_output_stationary`, conv2d (im2col+matmul), softmax — thin
  (73/100/91 lines), covered by `tests/dsl/test_schedule_dsl.cpp`.

### 4.2 Kernel representation — the most mature C++ path

- `kernel.hpp` (1419 ln): `Kernel` wraps a `DMProgram` + metadata; factory
  methods for MATMUL, MLP (fused matmul+bias+activation), CONV2D
  (im2col+GEMM), ATTENTION, LAYERNORM, RMSNORM, BATCHNORM, ELEMENTWISE,
  POOL2D, SOFTMAX (`kernel.hpp:674-1098`) with FLOP accounting.
- `kernel_graph.hpp` (502 ln): multi-kernel DAG; topological sort (Kahn),
  execution levels, critical path, cycle detection, fusion analysis
  (`find_fusible_pairs`, `can_fuse`; NONE/PRODUCER_CONSUMER/HORIZONTAL/
  PIPELINE), `compile()`/`compile_sequential()` → one `DMProgram` with
  barriers (`kernel_graph.hpp:147-499`).
- `kernel_serializer.hpp:75-193`: binary "KPUK" `.kpukernel` + JSON.
- Limitation: graphs are constructed **by hand in C++**; nothing lowers a
  model into a KernelGraph.

### 4.3 The two DFX namespaces

1. **Op-level DFX** — `sw::kpu::dfx` (`include/sw/kpu/dfx/`): op-level IR
   (`Op` with MATMUL/CONV2D/RELU/SOFTMAX/…, `Tensor`, `Program`) mirroring
   the Python emitter; `dfx_parser.cpp` parses DFX JSON. `src/dfx/` has only
   the parser — no executor.
2. **Tile-level DFX** — `sw::kpu::compiler::dfx`
   (`include/sw/compiler/dfx/dfx.hpp`), self-described as "the
   PTX-equivalent layer for KPU" (`dfx.hpp:1-24`): `DataMoveOp`, `ComputeOp`,
   `BarrierOp`, `TileSpec`, `TilingConfig`, dependency DAG,
   `PerformanceHints`; serializes to `.kpu` JSON via `dfx_object_file.hpp`.

### 4.4 The C++ graph compiler — front half only

- Pipeline: `.dfg` → `DFGParser` → `extract_matrix_ops` → `DFXGenerator`
  (uses `TileOptimizer` + `L2TileScheduler`) → `compiler::dfx::Program` →
  `ObjectWriter` → `.kpu`
  (`tools/compiler/kpu-kernel-compiler/main.cpp:197-289`,
  `dfx_generator.hpp:50-219`). MATMUL only ("No matrix operations found"
  otherwise).
- `TileOptimizer` (`include/sw/compiler/tile_optimizer.hpp:143-166`) is real
  automated tiling: ANALYTICAL / BOUNDED_SEARCH / HEURISTIC_HYBRID,
  memory-hierarchy-aware.
- **Dead end:** the only `.kpu` consumer is `ScheduleBinder::bind()`
  (`tools/runtime/kpu-loader/schedule_binder.cpp:18-122`) — round-robin
  resource assignment + simplified cycle estimates (`bytes/64`,
  `flops/flops_per_cycle`); referenced nowhere else. `kpu-loader`
  (`tools/runtime/kpu-loader/main.cpp:164-272`) loads **`.kpubin`**
  (serialized DMPrograms) via `create_program_executor` and ignores `.kpu`
  entirely. **No `compiler::dfx::Program → isa::DMProgram` lowering
  exists.** The `docs/06-compiler/` design itself calls the loader a
  "Skeleton".

### 4.5 The fourth toolchain — `tools/dfg/`

`kpu-dfg-gen` → `kpu-dfg-sched` (ASAP) → `kpu-dfg-compile` → `kpu-dfg-viz`
(Chrome trace) → `kpu-dfg-analyze` (critical path); see
`examples/dfg/matmul.sh`. Scheduling/analysis oriented; disjoint from the
paths above.

### 4.6 Python — the only working full-network path

`python/kpu/` is the most complete front end:
- Front ends: `@kpu.compile` tracing (`compiler.py:87-160`);
  `fx_converter.py` (81 KB) for torch.fx GraphModules; `model_loader.py`
  for ONNX + PyTorch weights (`:226-315`); models in `python/kpu/models/`.
- Middle: `graph.py` (`OpGraph`/`OpNode`/`OpType`) → `fusion.py` (34 KB;
  emits fused opcodes like `fused_matmul_bias_relu`) → `dfx_emitter.py`
  (op-level DFX JSON, `:83-160, 295-433`).
- Back end: `runtime.py` → native `kpu_native.cpp` (4001 ln).
  `execute_behavioral` computes matmul on the C++
  `BehavioralComputeFabric::submit_matmul` and other ops via C++/numpy
  (`kpu_native.cpp:355-454`); `execute_simulated` adds transactional
  timing via `TransactionalMemoryController` + transactional compute fabric
  (`:2730-3208`). Pure-numpy fallback when the native module is absent
  (`runtime.py:314-417`).
- Real pretrained models run and validate against PyTorch via
  `torch.compile(backend="kpu")`: `examples/torch/resnet18_inference.py`,
  `mobilenetv2_inference.py`, `vit_inference.py`.
- Limitation: op-granular behavioral execution — never touches the tiled
  DMProgram machine model.

### 4.7 Graph formats in the tree

`test_graphs/simple/` holds the same matmul in three formats: `matmul.dfg`
(Stillwater Domain Flow Graph text: NODES/EDGES/ADJACENCY,
`tensor<4x16xf32>`), `matmul.json` (custom graph JSON with `kpu_config` and
`target_kpu:"T256"`), `matmul.kpu` (compiled tile-level DFX JSON). No ONNX
in `test_graphs/` — ONNX enters only via the Python `model_loader`.
`kernels/` contains exactly four compiled kernels as `.kpuasm`/`.kpubin`
pairs: `matmul_16x16x16`, `matmul_4096x1024x8192`, `conv2d_im2col`,
`softmax_batch`.

---

## 5. Operator and Datatype Coverage

### 5.1 Coverage matrix

"Runs on simulator": **Yes** = executes on the behavioral compute fabric /
native; **Timing** = modeled by transactional/CSP timing generators;
**Host** = host reference only; **No** = absent.

| Operator | Exists where | Runs on simulator |
|---|---|---|
| matmul / GEMM | behavioral fabric (`src/models/behavioral/compute/compute_fabric.cpp`), temporal systolic (`models/temporal/compute/systolic_array.hpp`, `matmul_tau111_s001.cpp`), templated `quantization/kernels.hpp:72`, schedules, Python native | **Yes** (all dtypes) + Timing |
| batch_matmul | `KernelOpType::BATCH_MATMUL` | Yes (via matmul) |
| conv2d (im2col+GEMM) | behavioral `execute_conv2d_fp32`; `src/schedules/conv2d_schedule.cpp`; `kernels/asm/conv2d_im2col.kpuasm`; Python `ops.py:888` | **Yes** (fp32) + Timing |
| depthwise conv | `verification/kernels/class4_depthwise/`; `models/mobilenet.py` | Partial |
| pool2d (max/avg/global) | behavioral `execute_pool2d_fp32`; Python `ops.py:968+` | **Yes** (fp32) |
| softmax | behavioral; schedule generator; `kernels/asm/softmax_batch.kpuasm`; `kernels.hpp:265`; Python `ops.py:199` | **Yes** + Timing |
| layernorm | behavioral (`compute_fabric.cpp:334`) + transactional (`:269`); generator; `kernels.hpp:319`; Python `ops.py:667` | **Yes** + Timing |
| rmsnorm | `KernelOpType::RMSNORM` enum only (`kernel.hpp:57`) | **No** functional impl |
| batchnorm | behavioral `execute_batchnorm_fp32`; generator; Python `ops.py:726` | **Yes** + Timing |
| elementwise + activations | behavioral; `kernels.hpp:194-255` (relu/leaky_relu/gelu/silu); vector engine bias+activation | **Yes** |
| attention | composed: `kernel.cpp:146 create_attention` = matmul+softmax+matmul; `class5_attention/`; Python `ops.py:499/577` | Partial (composition; no fused kernel) |
| **Cholesky** | nowhere in code; complexity table in `docs/analysis/memory-hierarchy-performance-modeling.md:121` | **No** |
| **QR** | code: none; prose in `docs/design/tiling_discussion.md:342`, methodology docs | **No** |
| **LU** | code: none; tiling prose (`tiling_discussion.md:567-607`) | **No** |
| **SVD** | nowhere (zero hits) | **No** |
| **eigenvalue** | code: none; "Long-term: full eigenvalue solver" (`docs/02-simulation/functional-simulation-gaps.md:437`) | **No** |
| **FFT** | code: none; DFX/SURE aspiration (`docs/06-compiler/dfx-presentation.md`, `kpu_classification.md:25`) | **No** |
| **FIR/IIR filters** | nowhere in code | **No** |
| **Kalman (EKF/UKF/multiscale)** | nowhere — no code, no design doc | **No** |

### 5.2 Datatypes

`include/sw/kpu/data_types.hpp`: FLOAT32/16, BFLOAT16, INT32/16/8, UINT8,
INT4, FP8 (E4M3/E5M2/E3M4/E2M5), FP4. Backed by the Stillwater Universal
library via `quantization/universal_types.hpp` (`sw::universal::half`,
`bfloat16`, `cfloat<...>`), guarded by `KPU_HAS_UNIVERSAL`. Quantization
stack: `scalar_traits.hpp`, `packed_types.hpp` (INT4), `quant_params.hpp`,
`type_dispatch.hpp`, templated `kernels.hpp`. Python mirrors this
(`python/kpu/quantization.py`). **Posits are not used** — one comment
mention only (`systolic_array.hpp:92`).

### 5.3 Examples, verification, benchmarks

- `examples/mlp/` (xor, mnist, sine): timing model + host
  `reference_*_forward()` for numerics (`xor_classifier.cpp:17,99` states
  "PERFORMANCE/TIMING model, not a functional simulator").
- `examples/behavioral/` (xor_behavioral, matmul_behavioral, …): actually
  compute on the behavioral fabric.
- `examples/torch/`: real pretrained models through the torch.compile
  backend with PyTorch cross-checks.
- `verification/kernels/TAXONOMY.md`: classes 0-6 (elementwise, dense
  linear, spatial conv, multi-branch, depthwise, attention, quantized) with
  a `compute_harness.hpp` comparing fabric output to triple-loop references;
  `verification/dnn/` targets VGG-16/SqueezeNet (Class-I GEMM-dominant).
  No classes for factorizations, transforms, or filters.
- `benchmarks/` is an empty shell: its CMakeLists references
  `mlperf/rodinia/custom` directories that do not exist.
- Functional test coverage concentrates on matmul, MLP, conv2d, softmax,
  quantization (`tests/quantization/test_quantized_mlp.cpp` covers
  FP32/BF16/FP8/INT8/INT4 with accumulator types).

---

## 6. Existing Plans, Roadmaps, Status Documents

- **Canonical roadmap:** `docs/ROADMAP.md` (updated 2026-02-04). Released:
  v0.8 model-level execution (2026-01-22), v0.8.5 ISA infrastructure
  (2026-02-04). Active: v0.9.0 CSP concurrent timing (features
  v0.9.0-v0.9.7), detailed in `docs/plans/v0.9_concurrent_timing_roadmap.md`
  (40 tasks, 4 phases, gates G1-G6; success criteria ±10% timing, >80% DMA
  utilization, >1M cycles/sec, no livelock). v1.0.0 = CYCLE_ACCURATE
  functional, API stability, silicon validation, MNIST/SqueezeNet/
  MobileNetV2/BERT passing. The roadmap supersedes
  `docs/plans/roadmap-phase7-onwards.md`,
  `docs/09-virtual-platform/unified-dnn-roadmap.md`,
  `docs/09-virtual-platform/api-gaps-roadmap.md`, and
  `docs/project/project_plan.md`.
- **Acknowledged headline gap:** the v0.8.5 `TransactionalProgramExecutor`
  processes instructions sequentially → 4-8× timing overestimation;
  resolution = v0.9 CSP model (stated in ROADMAP, the v0.9 plan, and the
  CHANGELOG).
- **Three-subsystem disconnection:** `docs/07-fidelity-elevation/
  gap-assessment.md` names gaps G1-G9 (critical: no BehavioralStreamer, no
  behavioral DMProgram executor, BehavioralOrchestrator bypasses the ISA);
  much of its Phase 1/2 is done per `fidelity_status.txt`, but the
  OFG/behavioral wiring item remains unchecked.
- **Authoritative references (do not contradict):**
  `docs/01-architecture/kpu-execution-model.md` (credit dataflow,
  buffers-not-caches) and `docs/02-simulation/fidelity-framework.md`
  (fidelity tiers). Note these are the actual locations of the files that
  older docs reference by the uppercase names `kpu-execution-model.md` (root)
  and `SIMULATION_FIDELITY_FRAMEWORK.md`.
- **Housekeeping observations:** duplicate numbered dir prefix
  (`docs/07-fidelity-elevation/` and `docs/07-runtime/`); root-level status
  files (`v0.8-status.txt`, `v0.9-status.txt`, `fidelity_status.txt`,
  `v4-status.txt`, `perf-analysis.txt`, `enhancement-plan.md` — the last is
  an HBM-specific plan, not a top-level roadmap) duplicate CHANGELOG/session
  content and drift; superseded roadmap docs remain findable.
