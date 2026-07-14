# Changelog

All notable changes to the KPU Simulator project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **OnlineSoftmaxScheduleGenerator (#156, epic E8).** Single-pass online
  softmax schedules (pattern P3): a streaming stats pass (running max +
  rescaled running sum) produces the `(m, l)` state, and the apply pass
  emits `exp(x - m)/l` consuming that state as a **compute-resident
  dependency** (the #155 mechanism) - no drain/reload, no DRAM round-trip
  race, which is what distinguishes it from the reduction ROW_NORMALIZE.
  Realization is chosen a priori from the envelope (ROW_RESIDENT delivers
  the row once with `consumer_count=2`; RESTREAMED re-reads it), trailing
  tiles are clamped, and every COMPUTE is executable. Supersedes the
  4-pass `SoftmaxScheduleGenerator` and resolves #139 for softmax.
  Coverage: softmax generator -> done (26/105).

- **Schedule-tier compute-resident dependency mechanism (#155, epic E8).**
  A COMPUTE can now declare `resident_tiles` - inputs it consumes from the
  compute fabric, produced by a prior COMPUTE, rather than via a fresh
  FEED. `ConcurrentTimingExecutor::schedule_compute(tile, feed, resident)`
  records them as resident dependencies (the #66 completed-compute
  accounting), `ScheduleExecutor` routes COMPUTEs carrying `resident_tiles`
  there, and the `FunctionalComputeBinder` maps them onto
  `FunctionalComputeSpec::resident_tiles`. This lets a running-stat tile
  (online-softmax `(m, l)`, norm mean/var) reach the apply computes
  ordered and race-free - no DRAM round-trip - which is exactly what
  E3-T4 deferred. Built once here, reused by softmax (E8), norms (E9), and
  the reduction ROW_NORMALIZE apply phase. Coverage: softmax design +
  isa_closure -> done (25/105).

- **Reduction regression matrix + characterization (#108, epic E3
  COMPLETE).** `test_reduction_regression` executes the op x stream-length
  x envelope matrix (MAX/SUM/VAR x 1/16/64-tile + non-aligned x
  default/constrained-min/partitioned) with invariants stronger than
  completion: exact per-stage tile accounting and full credit conservation
  (both credit modes), plus a normalized stall bound. The envelope refusal
  boundary is pinned exact (min 8 generates, 4 refuses), and values survive
  the minimum envelope under partitioned credits on a non-aligned span.
  Characterization on epic #72: the single K-scaled compute amortizes
  per-tile cost 193 -> 42 cycles from 1 to 64 tiles. `online_reduction` is
  now the fourth operator complete across all five stages after matmul,
  elementwise, and broadcast (23/105). This closes epic #72 and, with the
  ISA half from #105, closes #18 entirely.

- **Value-producing streaming reduction + host oracles (#107, epic E3).**
  `FunctionalReductionExecutor` bridges the #106 generator to the #66
  payload machinery for the stats forms: input tiles ride the real CSP
  data path, the per-row stat COMPUTE reduces every streamed feed to a
  finalized statistic (MAX/MIN/SUM/MEAN/VAR, matching the VE_REDUCE ABI -
  population variance, clamped, empty -> NaN), and the stat drains back to
  DRAM. Verified against independent host oracles across all five ops,
  batched rows, partitioned credits, and non-tile-aligned spans - flips
  `online_reduction.functional` (an M4 gate cell, 22/105). ROW_NORMALIZE's
  apply-phase numerics are deferred to E8/E9 (its stat needs
  compute-resident delivery rather than a DRAM round-trip, which would
  race in the value plane).

### Fixed

- **Reduction generator now clamps non-tile-aligned trailing tiles
  (#107).** `OnlineReductionScheduleGenerator::make_tile` sized every A/C
  tile as a full tile, so a `reduction_elems` not divisible by
  `tile_elems` produced a trailing tile that read past the row end. The
  trailing footprint is now clamped (inter-tile stride stays full-tile),
  matching the elementwise generator; caught by the functional oracle.

- **OnlineReductionScheduleGenerator (#106, epic E3).** Streaming-reduction
  schedules for pattern class P3, all executable (the #101 discipline):
  `FULL_REDUCE` and `ROW_STATS` model the accumulation matmul-shaped - a
  single COMPUTE depends on every streamed input feed with K-scaled
  latency, so no resident-accumulator chain is needed at the schedule
  tier and the stats pass has a constant working set of 2 regardless of
  stream length. `ROW_NORMALIZE` is the two-phase form (stats then apply)
  whose realization is chosen a priori from the envelope -
  `ROW_RESIDENT` (row delivered once with `consumer_count=2`, the E2
  1:1:k discipline) when `reduction_tiles + 2 <= per-matrix burst share`,
  else `RESTREAMED` (row re-read from DRAM, constant working set 3); the
  per-row stat round-trips through DRAM and is broadcast to the apply
  computes via a distinct reload tile. Coverage: online_reduction
  generator -> done (21/105).

- **VE_REDUCE numerical semantics + ISA closure (#105, epic E3).**
  `VEReduceOperands` gives the previously annotation-only VE_REDUCE real
  encoding: MAX/MIN/SUM scalar accumulators and MEAN/VAR moment triplets
  `[count, sum, sumsq]`, gated by INIT/ACCUMULATE/FINALIZE phase flags,
  over a fixed 3-lane fp32 accumulator ABI. Behavioral executor runs the
  streaming combine with defined edge semantics (empty -> NaN, single
  sample -> variance 0, population divisor, clamp so cancellation never
  yields negative variance); assembler parses
  `VE_REDUCE op, src, acc [, INIT] [, FINALIZE]`; serializer round-trips
  at format v3 (variant 17 -> 18, guarded by static_asserts). The CSP
  accumulator needs no executor change - a reduction is a chain of
  functional computes targeting the accumulator tile with resident-dep
  ordering, validated end-to-end. Closes the ISA half of #18 for
  reductions. Coverage: online_reduction design + isa_closure -> done
  (20/105).

- **Elementwise/broadcast regression matrix + characterization (#103,
  epic E2 COMPLETE).** `test_elementwise_regression` executes the full
  form x size x envelope matrix (binary/broadcast/unary x 1/16/64-tile +
  non-aligned x default/constrained-min/partitioned = 36 cells) with
  invariants stronger than completion: exact per-stage tile accounting
  (tiles reaching L3 = LOADs + WRITEBACKs, moves/feeds/drains/stores
  exact) and full credit conservation (every L3/L2 credit returned).
  The envelope refusal boundary is pinned exact (12 generates, 11
  refuses), and functional values survive the minimum envelope under
  partitioned credits on a non-aligned tensor. Characterization recorded
  on epic #71: constrained-min triples stalls on binary 64-tile
  (1166 -> 3707) yet costs only 6 extra cycles - the pipeline absorbs
  envelope pressure; per-tile cost amortizes 242 -> 26 cycles from 1 to
  64 tiles. Coverage: 18/105; elementwise and broadcast are the first
  operators complete across all five stages after matmul.

- **Value-producing elementwise/broadcast execution + host oracles (#102,
  epic E2).** `FunctionalElementwiseExecutor` bridges the #101 generator to
  the #66 payload machinery: input tensors ride the real CSP data path
  (DRAM->L3->L2->L1->compute), every COMPUTE applies real VEOp semantics,
  and results drain back to DRAM - verified elementwise against an
  independent host oracle for all 14 VEOps across binary, broadcast-bias,
  unary, scalar, non-aligned, and partitioned-credit configurations
  (12k+ assertions). `ScheduleExecutor` gains an opt-in
  `FunctionalComputeBinder` so ANY generated schedule can run
  value-producing without touching the timing path. Behavioral
  STR_BROADCAST_ROW/COL get real resident-operand semantics (deliver L2->L1
  once, consume many, never fires a compute) with `str_broadcast_row/col`
  factories - closing the broadcast isa_closure gap. Coverage: 16/105
  (elementwise+broadcast design/functional cells, broadcast isa_closure).

- **ElementwiseScheduleGenerator + broadcast emission (#101, epic E2).**
  The first generator whose every schedule is executable: paired
  two-stream emission for `C = op(A, B)` (P6 - interleaved A/B pairs, no
  stream can monopolize the pools, 3-tile working set), unary and
  broadcast-B forms, COMPUTE operations carrying their full operand
  dependency sets (resolving #139 for the elementwise family), and
  envelope checks/stamping per the #90/#91 discipline.
  `emit_broadcast_tile()` delivers a resident operand once with a seeded
  consumer count (the #100 1:1:k mechanism) - the broadcast form executes
  end-to-end (one MOVE, n feeds, one credit). All three forms are
  regression-tested with and without partitioned credits; non-tile-aligned
  tensors clamp the trailing tile's footprint (full-tile stride preserved).
  Coverage: elementwise/broadcast generator cells -> done (11/105).

- **VE_ELEMENTWISE functional semantics + broadcast ref-count seeding
  (#100, epic E2 broadcast/elementwise).** `VEOperands` gives
  VE_ELEMENTWISE real operands (14 op kinds: binary/unary/scalar forms,
  L1 src/dst addressing) with factory methods, assembler syntax, and
  binary serializer round-trip (`.kpubin` format v2, with static_assert
  drift guards); the behavioral executor applies the ops elementwise over
  L1 buffers with IEEE semantics, validated exactly against a host oracle
  - the elementwise share of #18's numerical half. Broadcast (P5):
  `TileDescriptor.consumer_count` lets one MOVE seed the L2 TagCAM
  ref-count with the downstream feed count (1:1:k discipline, one credit
  per entry preserved), tested at TagCAM and BlockMover level. Discovered
  and filed #144: the serializer cannot round-trip register-file/AUTO
  operand types (pre-existing; reader throws).

- **CSP pattern-coverage matrix (#93, completes Wave 0).**
  `tests/coverage/pattern_coverage.json` is the machine-checkable operator
  x pattern x lifecycle-stage matrix for the coverage program (21
  operators, 9 pattern classes, 5 stages, truthful current statuses -
  baseline 8/105 cells done). `test_pattern_coverage` enforces schema
  consistency and **milestone gates**: a milestone marked achieved must
  have every required capability at "done", so the matrix cannot
  overclaim and achieved milestones cannot silently regress. Pattern
  epics update rows as they land; the model-validation epic (E18)
  asserts the full matrix through these gates.

### Fixed

- **Non-matmul Kernel factories now compile their actual op (#92, fixes
  the structural half of #18).** `KernelCompiler` gains per-op streaming
  compilers (`compile_softmax/layernorm/rmsnorm/batchnorm/elementwise/
  pool2d`) emitting loop-based DMPrograms with real tile traffic and
  VE_REDUCE/VE_ELEMENTWISE pass markers - a "softmax kernel" no longer
  executes a matmul. Loop-based emission keeps vocab-scale programs
  compact (the unrolled approach behind the #17 Windows OOM is gone for
  good). conv2d is documented as an intentional im2col+GEMM lowering with
  fused epilogue. `ConcurrentExecutor` learns to interpret hardware loops
  (LOOP_BEGIN/END), AUTO-addressed ops (via SET_TILE_DIM geometry), and VE
  ops for timing - which also un-breaks the bandwidth benchmarks for
  streaming ops. Numerical validation of the VE semantics lands with the
  pattern epics (E2/E3); tracked in #18, which stays open for that half.

### Added

- **CSP timing: envelope-mismatch detection (#91, Wave 0).** Schedule
  generators stamp their generation envelope into `ScheduleMetadata`
  (`l3_buffer_count`/`l2_bank_count`; 0 = hand-built/legacy, not checked).
  `ScheduleExecutor::execute` compares it against the executor's configured
  pools and surfaces a warning in the new `ExecutionResult::warnings` -
  with a may-wedge note when the executor pools are smaller (which voids
  the #67/#90 constructive-safety guarantees) and a benign note when
  larger. `validate_livelock_safety` raises the matching VAL-007 issue.
  `run_matmul` prints execution warnings.

- **CSP timing: resource envelope on all schedule generators (#90, Wave 0).**
  The conv2d/softmax/layernorm/batchnorm generators now carry the #67
  resource envelope (`l3_buffer_count`/`l2_bank_count`, `max_burst_tiles()`)
  and an **a-priori working-set check**: generation refuses with an
  actionable message when the schedule's implied peak tile residency
  exceeds the envelope share, instead of wedging at runtime - softmax's
  multi-pass exp scratch (`reduction_tiles + 2`), layernorm's resident
  affine params (`2*hidden_tiles + 2`), batchnorm's per-channel stats,
  conv2d's streaming pair. `csp_schedule_demo` declares fitting envelopes
  and reports working set vs share. Discovered in the process: these four
  generators emit DRAIN without COMPUTE and would hang if executed -
  tracked as #139 for the operator epics.

- **CSP timing: per-matrix credit partitioning wired into the executor
  (#89, Wave 0 of the pattern-coverage program).** `CreditPool` gains an
  opt-in partition mode backed by `PartitionedCreditPool` (equal A/B/C
  split, remainder to C): indexed `acquire/release/available(partition)`
  overloads match `isa::MatrixID` values and behave identically to the
  shared calls when unpartitioned, so all processes were converted once
  and work in both modes; un-indexed calls throw in partition mode to
  catch missed conversions. `ConcurrentTimingExecutor::Config` gains
  `partition_l3_credits`/`partition_l2_credits` (default off - partitioning
  intentionally forbids single-matrix workloads from filling the whole
  pool); `run_matmul` gains `--partition-credits`. Restores the original
  v0.9 design's structural livelock prevention: an adversarial A-load
  flood can no longer starve B traffic (regression-tested both ways), and
  the full strategy matrix passes partitioned - measurably faster at
  128^3 (7,132 vs 8,034 cycles) because partitioning curbs greedy
  prefetch over-filling L3.

- **DNN Milestone M1: MLP baseline demo + benchmark (#129).**
  `examples/milestones/m1_mlp_baseline` packages the first rung of the DNN
  milestone ladder: XOR 2-4-1 (exact expected outputs) and the canonical
  MNIST-shape 784-128-64-10 MLP (deterministic synthetic weights,
  host-oracle validated to 0.0 max abs error) executing with real values on
  the CSP credit-dataflow pipeline. Benchmarks batch sweep 16/64/256
  (752 -> 94 cycles/sample amortization) and pipeline sweep; `--trace-dir`
  exports Chrome traces of the credit dataflow; registered as a CI test so
  the milestone cannot regress. Writeup with measured numbers:
  `docs/milestones/M1_mlp_baseline.md`. Supporting: FunctionalMLPExecutor
  gains `set_trace_file()` and full per-component timing statistics.

- **CSP timing: resource-envelope-aware schedule generation (#67).**
  `MatMulScheduleGenerator::Config` gains a resource envelope
  (`l3_buffer_count`/`l2_bank_count`, defaults matching the executor) and a
  derived per-matrix burst bound (`max_burst_tiles()` = a quarter of the
  smaller pool). BLOCKED_AB is now livelock-safe **by construction**: an
  outer K-block loop bounds each A/B burst to the envelope share, so the
  tile working set provably fits the credit pools - the classic
  blocked-linear-algebra discipline of deriving block sizes from the memory
  hierarchy. A capacity-aware `is_livelock_safe(schedule, l3, l2)` overload
  checks the constructive residency bound (in operations: 2 ops per
  resident tile) instead of the fixed interleaving heuristic; `run_matmul`
  generates against and reports the envelope it executes with. Provenance
  of the original design gap (executor-side partitioned credits designed,
  built, never wired; safety burden silently moved to schedule ordering)
  is documented in #67.

- **CSP timing: multi-tile execution regression suite (#64).**
  `tests/timing/test_multi_tile_execution.cpp` executes generated matmul
  schedules end-to-end through `ConcurrentTimingExecutor` across all four
  strategies x 32^3/64^3/128^3, asserting completion, no livelock, exact
  per-stage tile accounting (moves/writebacks/feeds/drains match schedule op
  counts), and per-tick stall-accounting bounds. Closes the coverage hole
  that let #61 reach main: no prior test executed a generated multi-tile
  schedule.

### Fixed

- **CSP timing: COMPUTE dependencies now cover all K-slice feeds, and
  compute latency scales with K (#63).** The matmul schedule generators
  previously keyed COMPUTE for C(ti,tj) to only the last B feed; a compute
  could start before its A tiles or earlier K slices were fed. COMPUTE now
  carries the full dependency set (every A(ti,*,k) and B(*,tj,k)), the
  executor starts compute only when all of them have been fed, and latency
  is `compute_latency + (k_slices - 1) * compute_cycles_per_k_slice` (new
  config knob, default 32). `ScheduleOperation` gains a `dependency_tiles`
  list (the single-dependency field and executor overload remain for
  backward compatibility); `is_complete()` now also accounts for pending
  computes. Known remaining approximation: fed-tile tracking is a monotonic
  was-ever-fed set, so a tile fed for an earlier output iteration satisfies
  later iterations' dependencies (per-instance accounting arrives with real
  data movement).
- **CSP timing: PREFETCH_NEXT strategy livelocked at 128^3 and above** —
  caught immediately by the new regression suite (#64). The generator
  emitted each tile's load twice per output iteration (prefetch + current)
  but only one MOVE, stranding L3 TagCAM references and credits (the same
  leak class as #61). Loads are now emitted once (at k=0 directly, k>=1 via
  the previous iteration's prefetch), keeping the LOAD:MOVE pairing 1:1.

- **CSP timing: multi-tile schedules livelocked in `ConcurrentTimingExecutor` (#61).**
  `run_matmul` failed with livelock at its default 64x64x64 / 16^3-tile
  configuration; 128^3 and 256^3 also failed. Four root causes, all in the
  timing tier headers:
  - `DMAEngineProcess::schedule_load/schedule_store` silently dropped requests
    beyond `queue_depth` (32), so most of a schedule's loads never executed and
    downstream BlockMovers/Streamers tag-stalled forever. The pending list is
    now an unbounded software staging queue; `queue_depth` bounds concurrently
    *submitted* MC requests instead.
  - Credit leak on tile reuse: each duplicate MOVE of an L2-resident tile
    acquired a new L2 credit, but the ref-counted TagCAM entry releases only
    one credit when it drains — leaking (uses-1) credits per reused tile.
    `BlockMoverProcess` now deduplicates moves for L2-resident tiles
    (ref_count++ without a credit), matching the documented schedule-generator
    contract ("execution layer deduplicates").
  - Duplicate in-flight loads: a second load of the same tile could be
    submitted while the first was still in the MC, double-acquiring L3 credits
    for one TagCAM entry. Loads now defer when the tile already has a
    submitted load (resolved via the dedup path on arrival). Work assignment
    changed from round-robin to tile-affine (hash of TileID) so one tile's
    operations serialize on one DMA/BlockMover/Streamer and cannot race
    across processes.
  - Instance-state bugs: `StreamerProcess::allocate_l2_slot` used a `static`
    counter shared across all streamer instances; the executor's compute-slot
    counter was a function-local `static` shared across executor instances and
    not reset. Both are instance members now.
  Also: DMA stall accounting now counts at most one stall cycle per tick
  (matching BlockMover/Streamer semantics) instead of one per pending request,
  and optional `l3_writeback_credit_reserve` / `l2_drain_credit_reserve`
  executor knobs (default 0) guard against prefetch starving writeback/drain
  credit acquisition on extreme schedules. `run_matmul` now completes for
  32^3 through 256^3 across all strategies (including BLOCKED_AB).

### Changed

- **BREAKING — L1 memory layer restructuring (#34).** Introduced the `L1Layer`
  aggregate (owns the `L1Buffer` stream buffers) and removed the flat L1 fields
  from `KPUSimulator::Config`. Migration:
  - `config.l1_buffer_count`        → `config.l1_layer.num_buffers`
  - `config.l1_buffer_capacity_kb`  → `config.l1_layer.capacity_kb`
  - For non-uniform layers, populate `config.l1_layer.buffer_groups`
    (`group → element-config → multiplicity`) instead of the scalar fields.
  - `config.l1_buffer_base` is unchanged. No backward-compatibility shims. Python
    keeps the `l1_buffer_count` / `l1_buffer_capacity_kb` attribute names (they
    proxy into `l1_layer`); the C ABI struct `KPUSimulatorConfig` is unchanged.
  - `L1Layer` (like `L2Layer`/`L3Layer`) is a monitoring/ownership structure, not
    a ResourceManager/dataflow API.

- **BREAKING — L2 memory layer restructuring (#33).** Introduced the `L2Layer`
  aggregate (owns the `L2Bank` elements) and removed the flat L2 fields from
  `KPUSimulator::Config`. Migration:
  - `config.l2_bank_count`        → `config.l2_layer.num_banks`
  - `config.l2_bank_capacity_kb`  → `config.l2_layer.capacity_kb`
  - For non-uniform layers, populate `config.l2_layer.bank_groups`
    (`group → element-config → multiplicity`) instead of the scalar fields.
  - `config.l2_bank_base` is unchanged. No backward-compatibility shims. Python
    keeps the `l2_bank_count` / `l2_bank_capacity_kb` attribute names (they proxy
    into `l2_layer`); the C ABI struct `KPUSimulatorConfig` is unchanged.

- **BREAKING — L3 memory layer restructuring (#32).** Introduced the `L3Layer`
  aggregate (owns the `L3Tile` elements, the per-tile `BlockMover`s, and an
  optional `L3Interconnect`) and removed the flat L3 fields from
  `KPUSimulator::Config`. Migration:
  - `config.l3_tile_count`        → `config.l3_layer.num_tiles`
  - `config.l3_tile_capacity_kb`  → `config.l3_layer.capacity_kb`
  - `config.block_mover_count`    → `config.l3_layer.block_mover_count`
  - For non-uniform layers, populate `config.l3_layer.tile_groups`
    (`group → element-config → multiplicity`) instead of the scalar fields.
  - `config.l3_tile_base` is unchanged. No backward-compatibility shims are
    provided. Python keeps the `l3_tile_count` / `l3_tile_capacity_kb` /
    `block_mover_count` attribute names (they proxy into `l3_layer`); the C ABI
    struct `KPUSimulatorConfig` is unchanged.

### Added

- **Fused batched-MLP SURE (`FusedMlpSure`) (#46, epic #45).** Models the fused
  `Y = activation(X·W + b)` operator as a single System of Uniform Recurrence
  Equations over one `(i,j,k)` domain, with the bias + activation as boundary
  recurrences on the terminal accumulation face `k=K-1` (fusion = merged domain,
  no materialized intermediate tensor).
  - `include/sw/kpu/dataflow/fused_mlp_sure.hpp` + `src/software/dataflow/fused_mlp_sure.cpp`
    (in `kpu_dataflow`): a domain_flow-free public API — `FusedMlpSureConfig`,
    `Activation` (Identity/ReLU/GELU/SiLU/Sigmoid), domain/schedule accessors,
    and `evaluate(X, W, bias) → Y` (behavioral execution of the fused recurrence).
  - Built on domain_flow's standalone polyhedral primitives
    (`ConstraintSet`/`IndexSpace`/`RecurrenceVariable`/`AffineMap`/`ScheduleVector`)
    as the math kernel — domain_flow itself is unmodified (approach "B3"); the two
    upstream alternatives are filed as `branes-ai/domain_flow#1`/`#2`.
  - `examples/dataflow/fused_mlp_sure_demo.cpp` — a runnable, self-validating demo
    (registered as the `fused_mlp_sure_demo` CTest).
  - Design: `docs/design/fused-mlp-sure.md`.

- **CSP Schedule Generators (Phase 4)** — Complete schedule generation infrastructure
  for livelock-safe DNN operation scheduling:
  - `IScheduleGenerator` interface with `ScheduleOperation`, `ScheduleResult`,
    and `ScheduleAnalysis` types
  - `MatMulScheduleGenerator` with 4 strategies: OUTPUT_STATIONARY, INTERLEAVED_AB
    (default, livelock-safe), PREFETCH_NEXT, BLOCKED_AB
  - `Conv2DScheduleGenerator` using im2col transformation for systolic array mapping
  - `SoftmaxScheduleGenerator` with 4-pass algorithm (max, exp, sum, normalize)
  - `LayerNormScheduleGenerator` with 3-pass algorithm (mean, variance, normalize)
  - `BatchNormScheduleGenerator` supporting both training and inference modes
  - `ScheduleValidator` with static validation rules (VAL-001 to VAL-006) and
    livelock safety checks (VAL-LL1 to VAL-LL5)
  - `ScheduleExecutor` for applying schedules to ConcurrentTimingExecutor
  - `ScheduleAnalysis::analyze()` for detecting interleaving patterns and
    consecutive operation runs

- **Livelock Test Suite** (`tests/timing/test_livelock_scenarios.cpp`)
  - 20 test cases covering livelock detection and avoidance scenarios
  - Tests for PartitionedCreditPool A/B/C segregation
  - Tests for interleaved vs blocked scheduling strategies
  - Tests for TagCAM tile arrival tracking

- **Schedule Generator Tests** (`tests/timing/test_schedule_generators.cpp`)
  - 15 test cases validating MatMul, Conv2D, Softmax, LayerNorm, BatchNorm generators
  - Strategy comparison tests for OUTPUT_STATIONARY vs INTERLEAVED_AB vs BLOCKED_AB
  - Tile dimension and operation count validation

- **Schedule Validation Tests** (`tests/timing/test_schedule_validation.cpp`)
  - 22 test cases covering all validation rules
  - Empty schedule detection (VAL-001)
  - Incomplete operation detection (VAL-002)
  - Livelock safety validation (VAL-LL1 to VAL-LL5)

- **CSP Schedule Demo** (`examples/schedule/csp_schedule_demo.cpp`)
  - Demonstration of CSP schedule generation for all DNN operations
  - Shows operation counts, livelock analysis, and validation results
  - Example configurations for ResNet, VGG, BERT, and transformer workloads

- **CSP Pipeline Educational Demo** (`examples/schedule/csp_pipeline_demo.cpp`)
  - Minimal 1×1×1 matmul showing complete dataflow: DRAM → L3 → L2 → Compute → L2 → L3 → DRAM
  - Transaction log format with cycle-by-cycle credit flow and TagCAM actions
  - Shows credit acquisition/release patterns (L3 and L2 pools)
  - Demonstrates TagCAM insert/match/invalidate operations
  - Educational explanation of CSP synchronization mechanisms
  - Use `--verbose` flag to include stall events in output

### Fixed

- **Zero-warning builds under `-Werror` / `/WX` (#23 Phase 3, #31)** — Resolved every
  warning surfaced by enabling warnings-as-errors across all three CI compilers. gcc was
  already clean; clang and MSVC `/W4` exposed conversion warnings the GNU `-Wall -Wextra`
  builds never see, and because each build stops at the first error they only appeared one
  file at a time. Each warning class was enumerated wholesale with targeted local clang
  scans (faithful proxies for the MSVC classes) and fixed in batches:
  - **169 integer narrowings** (`C4267`/`C4244`, `size_t`/`int` → smaller) across 38 files —
    explicit `static_cast` at each site (tile indices, bank/channel IDs, sizes, counts).
  - **~626 float narrowings** (`C4244`, 64-bit int → `double`/`float`) across 152 files —
    the pervasive cycles/bytes → `double` stats, bandwidth, utilization, and percentage
    computations; both operands cast on integer/integer ratios.
  - **`C4566`** — added `/utf-8` to the MSVC compile flags so the Unicode glyphs (arrows,
    box-drawing) in trace/demo output compile under code page 1252; fixed ~175 files via
    one config line instead of source churn.
  - **`C4101`** — removed a single unused `catch` exception binding.
  - **`C4244` pybind `ssize_t` → `int`** — two negative-axis fixups clang's scan missed.
  - All ~800 casts are behavior-preserving (they make explicit the conversions the compiler
    already performed). Verified clean locally under `release-werror` with both gcc and
    clang, plus clang conversion scans confirming zero remaining narrowings.

- **DMA + Memory Controller Architecture** — Implemented correct CSP architecture with
  proper separation of concerns:
  - DMA Engine (CSP Process): Handles ISA operations, L3 credit acquisition/release,
    L3 TagCAM tile tracking, programmable queues
  - Memory Controller (Communication Resource): Models DRAM command bus contention
    (1 command/cycle), 16 bank state machines, row hit/miss/empty classification
  - DMA submits requests to MC via `submit_request()` and polls for completions via
    `get_completed_transfer(submitter_id)`
  - Added `submitter_id` tracking so multiple DMA engines can share one MC without
    losing completions (each DMA only retrieves its own completions)
  - ConcurrentTimingExecutor creates both MCs and DMAs, wiring them together and
    ticking in correct order (MC first, then DMA)
  - Tests: 34 passing across DMA, MC, and component integration tests

- **DMA 8 concurrent transfers bug** — DMA channels now correctly issue only 1 transfer
  at a time instead of 8 concurrent. Changed `in_flight_.size() < config_.queue_depth` to
  `in_flight_.empty()` in `try_issue_loads()` and `try_issue_stores()`. The queue_depth
  parameter now only limits pending requests, not concurrent in-flight transfers.

- **DRAIN starting at cycle 0 bug** — Added compute modeling so DRAIN operations properly
  wait for computation to complete before draining results:
  - Added `COMPUTE` schedule operation type and `schedule_compute()` API
  - Added `compute_result_tag_cam` to track result tiles ready for draining
  - DRAIN now waits for the result tile to appear in compute_result_tag_cam
  - COMPUTE tracks dependency tiles and only starts when dependencies are FED
  - Added `STR_STALL_COMPUTE` event type for compute dependency stalls
  - All schedule generators (MatMul, Conv2D, etc.) now emit COMPUTE operations with
    proper dependency tracking (last B tile for that output column)

- **CreditPool double-release bug** — TagCAM now uses reference counting to support
  tile reuse. When the same tile is inserted multiple times (e.g., A[ti,tk] used for
  every tj), ref_count increments instead of failing. Invalidate decrements ref_count,
  only releasing the credit when it reaches zero. This fixes the overflow error:
  "CreditPool::release() called but pool is already full"

- **Resource utilization calculation overflow** — Statistics now computes average stalls
  per component type before subtracting from total_cycles. This prevents unsigned integer
  underflow when aggregated stall cycles across multiple parallel components exceed the
  simulation duration. Previous garbage values like 4729934377874243584.0% are now correct.

- **Conditional credit release for tile reuse** — DMA, BlockMover, and Streamer now only
  release credits when `TagCAM::invalidate()` returns true (tile fully removed). This
  prevents double-release when the same tile is used multiple times. Additionally, DMA
  checks if tiles are already in L3 before acquiring credits, skipping redundant loads.

- **Chrome Trace Visualization Improvements** — Enhanced CSP timing trace export:
  - Added process_name and thread_name metadata for human-readable component identification
  - Fixed component ID collisions (DMA 0-N, BlockMover 100+, Row Streamer 200+, Col Streamer 210+)
  - Added thread_sort_index for dataflow ordering (DMA → BlockMover → Streamer)
  - Traces now display threads in execution order from top to bottom in Perfetto
  - **Grid topology naming**: Components now show physical grid positions:
    - DMA channels: `MC0:CH0`, `MC0:CH1`, `MC1:CH0` (Memory Controller + Channel)
    - BlockMovers: `L3(0,0):BM`, `L3(0,1):BM` (L3 tile grid position)
    - Streamers: `CT(0,0):RowSTR`, `CT(0,1):ColSTR` (Compute tile position)
    - Process name: "KPU CSP Executor (2 MCs, 2x2 L3 tiles, 2x2 CTs)"
  - **Matrix base addresses**: Tiles now show base and DRAM addresses in trace args
    (e.g., `"tile":"A[0,0,0]","base":"0x1000","addr":"0x1400"`)

### Changed

- `run_matmul` CLI now supports `--l3-buffers <n>` and `--livelock <n>` for experimentation

---

## [0.8.5] - 2026-02-04

### Added

- **Chrome Trace Thread Names** — Added phase "M" metadata events to Chrome trace
  export with human-readable thread names (DMA Channel 0-3, BlockMover 0-3,
  Streamer 0-3, Loop, Sync, Compute) instead of numeric thread IDs

- **Data Mover Component Test Harness Infrastructure**
  - `PatternHarnessBase<ConfigT>` template class for all harnesses
  - `TileJourneyTracker` for per-tile timing through DRAM→L3→L2→L1→Compute
  - `DMAHarness` with L3BufferPool for credit-based DMA testing
  - `BlockMoverHarness` with L2BankArray for L3→L2 tile movement testing
  - `StreamerHarness` for L2→L1 streaming and compute integration testing
  - `DataMovementPipelineHarness` for full integrated pipeline testing
  - `ScheduleValidator` with static validation, cycle detection, ordering checks
  - `schedule-runner` CLI tool for schedule experimentation and analysis
  - Configuration structures: HarnessConfig, DMAHarnessConfig, BlockMoverHarnessConfig,
    StreamerHarnessConfig, PipelineHarnessConfig
  - TileID, TileCoord, MatrixID types with comparison and hash support
  - PipelineSchedule and ScheduleOperation for schedule representation
  - Comprehensive test suite (12 passing schedule validator tests)

- **Loop Machinery and Address Generation ISA Extensions**
  - `IndexRole` enum (TI, TJ, TK, NONE) for loop-to-tile-index binding
  - New AUTO addressing opcodes: `DMA_LOAD_TILE_AUTO`, `DMA_STORE_TILE_AUTO`,
    `BM_MOVE_TILE_AUTO`, `BM_WRITEBACK_AUTO`, `STR_FEED_ROWS_AUTO`,
    `STR_FEED_COLS_AUTO`, `STR_DRAIN_AUTO`
  - Configuration opcodes: `SET_BASE`, `SET_L3_BASE`, `SET_L2_BASE`,
    `SET_STRIDE` (enhanced), `SET_TILE_DIM`, `SET_MATRIX_DIM`
  - `LoopState` class (`include/sw/kpu/isa/loop_state.hpp`) — 8 hardware loop
    counters with index role binding
  - `AddressGenerator` class (`include/sw/kpu/isa/address_generator.hpp`) —
    computes tile addresses from base + loop_index × stride
  - `ISARegisterFile` class (`include/sw/kpu/isa/register_file.hpp`) —
    unified register file for loop state and address generation
  - Large matmul assembly example (`kernels/asm/matmul_4096x1024x8192.kpuasm`) —
    32 instructions express 8.4M tile operations via loops
  - Updated assembler to parse all new opcodes and IndexRole syntax

- **KPU Assembler** (`include/sw/kpu/isa/assembler.hpp`, `src/software/isa/assembler.cpp`)
  - Full lexer and parser for KPUASM assembly language
  - Supports all DMOpcode instructions (DMA, BlockMover, Streamer, Sync, Config)
  - Directives: `.name`, `.version`, `.dimensions`, `.tiling`, `.l1_ki`, `.dataflow`, `.a_base`, `.b_base`, `.c_base`
  - Labels, comments (`;` and `#`), tile coordinates `(ti,tj,tk)`, buffer slots
  - Assembles to DMProgram, serializes to `.kpubin` binary format via ProgramSerializer

- **KPU Assembler Tool** (`tools/development/kpu-assembler/assembler.cpp`)
  - Command-line assembler: `kpu-assembler input.kpuasm -o output.kpubin`
  - Options: `--format json`, `--print`, `--stats`, `-h/--help`
  - Error reporting with filename, line number, and message

- **KPU Loader Tool** (`tools/runtime/kpu-loader/main.cpp`)
  - Loads `.kpubin` or `.kpujson` programs and executes on simulator
  - Fidelity switching: `--fidelity behavioral` or `--fidelity transactional`
  - Input/output tensor files: `--input-a`, `--input-b`, `--output-c`
  - Trace export: `--trace trace.json` (transactional only)
  - Options: `--dry-run`, `--stats`, `-v/--verbose`

- **Assembly Kernel Examples** (`kernels/asm/`)
  - `matmul_16x16x16.kpuasm` — Single-tile output-stationary matmul
  - `conv2d_im2col.kpuasm` — Conv2D via im2col + matmul with fused ReLU
  - `softmax_batch.kpuasm` — Multi-pass softmax using Vector Engine ops

- **KPUASM Specification** (`docs/kpuasm-specification.md`)
  - Complete assembly language reference
  - Syntax: directives, opcodes, operand formats
  - Example programs with annotations

- **IProgramExecutor Interface** (`include/sw/kpu/isa/program_executor_interface.hpp`)
  - Phase 3 of fidelity elevation: unified interface for fidelity switching
  - `create_program_executor(fidelity, hw)` factory function
  - Supports BEHAVIORAL and TRANSACTIONAL fidelity levels
  - Common interface: `load_program()`, `run()`, `total_cycles()`, `export_trace()`
  - 20 tests for factory, correctness, and fidelity switching

- **TransactionalProgramExecutor Loop Execution**
  - PC-based execution following actual loop control flow (not linear iteration)
  - Loop timing model: `loop_begin_latency`, `loop_end_latency`,
    `loop_branch_taken_latency`, `loop_branch_not_taken_latency`
  - `LoopState` tracking for timing computation (separate from behavioral)
  - AUTO addressing opcodes with tile coordinates from loop state:
    `DMA_LOAD_TILE_AUTO`, `DMA_STORE_TILE_AUTO`, `BM_MOVE_TILE_AUTO`,
    `BM_WRITEBACK_AUTO`, `STR_FEED_ROWS_AUTO`, `STR_FEED_COLS_AUTO`,
    `STR_DRAIN_AUTO`
  - Loop statistics in `TimingStats`: `loop_overhead_cycles`, `loop_iterations`
  - Loop events recorded in Chrome trace with "loop" category
  - 6 new tests for loop timing overhead, nested loops, and configuration
  - Total 33 tests passing

- **TransactionalProgramExecutor** (`src/software/isa/transactional_program_executor.cpp`)
  - Phase 2 of fidelity elevation: behavioral correctness + timing overlay
  - Wraps BehavioralProgramExecutor for functional execution (real data movement)
  - Analytical timing models for DMA, BlockMover, Streamer operations
  - ResourceTimeline class tracks per-resource availability and makespan
  - TimingConfig with clock frequencies, bus widths, startup latencies
  - Chrome Trace export for Perfetto visualization
  - ASCII timeline generation for terminal output
  - 27 tests covering correctness, timing, and export functionality

- **BehavioralProgramExecutor** (`src/software/isa/behavioral_program_executor.cpp`)
  - Interprets DMProgram instruction streams using temporal memory components
  - Executes DMA, BlockMover, and Streamer operations as instant memcpy
  - Triple-loop matmul computation when A and B tiles arrive at L1
  - Strided DMA transfers for row-major tiled matrix layouts
  - Statistics tracking: instructions, loads, stores, computes, bytes transferred

- **End-to-End Matmul Correctness Tests** (`tests/isa/test_behavioral_program_executor.cpp`)
  - Single-tile matmul (16×16×16)
  - Multi-tile matmul (64×64×64 with 16×16×16 tiles)
  - Identity matmul (C = I × A = A)
  - Reference matmul against naive triple-loop
  - Execution statistics verification

- **Fidelity Elevation Gap Assessment** (`docs/07-fidelity-elevation/gap-assessment.md`)
  - Analysis of behavioral/transactional/cycle-accurate tier gaps
  - Three-phase implementation plan for fidelity elevation

- **Kernel Verification Harnesses — Phase 1** (`verification/kernels/`)
  - `class0_elementwise/verify_elementwise.py` — 12 elementwise ops (relu, gelu, silu,
    sigmoid, tanh, exp, log, sqrt, softmax, neg, add, mul) tested across 4 shape sweeps
    (48 test cases, all PASS)
  - `class1_dense_linear/verify_matmul.py` — Matmul verification with 10 dimension configs
    at BEHAVIORAL + 4 at TRANSACTIONAL with FLOP count validation and roofline reporting
    (14 test cases, all PASS)
  - `class1_dense_linear/verify_fused_ops.py` — 4 fusion patterns (matmul+relu,
    matmul+bias+relu, matmul+bias+gelu, matmul+bias+silu) across 3 sizes
    (12 test cases, all PASS)

### Fixed
- **Transactional Timing Data Dependencies** (`src/software/isa/transactional_program_executor.cpp`)
  - Added tile arrival tracking maps (`tile_at_l3_`, `tile_at_l2_`)
  - BlockMover now waits for tile to arrive at L3 before starting
  - Streamer now waits for tile to arrive at L2 before starting
  - Enforces correct dataflow ordering: DRAM→L3→L2→L1→Compute

- **Harness Test Infrastructure**
  - DMA harness completion callback now properly clears `in_flight_requests_`
  - BlockMover harness `L2BankArray::allocate()` now supports `set_tile()` for
    tile ID tracking (previously only `reserve()` recorded tile IDs)
  - Journey tracking records arrivals at `current_cycle_ + 1` in behavioral mode
  - Pipeline harness buffer allocation coordination via completion callbacks

- **Windows CI** — Use `std::filesystem::temp_directory_path()` instead of
  hardcoded `/tmp/` paths (fixes stack buffer overrun 0xc0000409)

- **Schedule Compiler WRITEBACK offset** (`src/dsl/schedule_compiler.cpp`)
  - BM_WRITEBACK now uses `loc.address` from TileLayout instead of hardcoded 0
  - Fixes data loss when writing C tiles back to L3

- **Schedule Compiler str_drain argument order** (`src/dsl/schedule_compiler.cpp`)
  - Corrected parameter order: `str_drain(tile, l2_bank, l1_buf, ...)`
  - All three drain variants (DRAIN, DRAIN_FUSED, DRAIN_TO_SCRATCH) fixed

### Known Limitations
- **TransactionalProgramExecutor Timing Model** — The current timing model processes
  instructions sequentially, missing natural concurrency of the credit-based dataflow
  architecture. This results in **4-8x overestimation** of execution time for
  memory-bound workloads:
  - DMA operations for A and B matrices are serialized (should be concurrent)
  - No pipelining between memory hierarchy levels (DMA→BM→STR overlap)
  - Single instruction stream instead of concurrent component processes
  - No credit-based flow control modeling

  **Impact:** Timing numbers are directionally correct for relative comparisons
  but not accurate for absolute performance analysis. Functional results
  (computed values) are always correct.

  **Resolution:** v0.9.0 will introduce CSP-based concurrent timing model with
  true dataflow concurrency. See `docs/plans/v0.9_concurrent_timing_roadmap.md`.

### Changed
- **TAXONOMY.md** — Updated Phase 1 roadmap to reflect Class 0 and Class 1 kernel
  verification harnesses as DONE

## [0.8.0] - 2026-01-26

### Added
- **Native Wheel Infrastructure** (`python/CMakeLists.txt`, `python/pyproject.toml`)
  - scikit-build-core integration for CMake-based Python wheel builds
  - cibuildwheel CI/CD for multi-platform wheels (Linux, macOS, Windows)
  - Standalone build mode with FetchContent for all dependencies
  - GitHub Actions workflow for automated PyPI publishing

- **Trace Library for Python Bindings** (`python/CMakeLists.txt`)
  - `kpu_trace_for_python` static library with TraceEntry to_string implementations
  - `BUILDING_KPU_SIMULATOR` define for correct MSVC symbol export

### Fixed
- **DFX Parser Library Build** (`python/CMakeLists.txt`)
  - Fixed EXISTS check from non-existent `dfx_executor.cpp` to `dfx_parser.cpp`
  - DFX library now builds correctly in standalone wheel builds

- **MSVC C++20 Feature Detection** (`CMakeLists.txt`, `python/CMakeLists.txt`)
  - Added `/Zc:__cplusplus` flag for correct `__cplusplus` macro value
  - Fixes Universal library `std::bit_cast` detection on MSVC

- **Universal Library v3.91 Integration** (`cmake/Dependencies.cmake`)
  - Updated include path for v3.91 header structure (`include/sw/`)
  - Fixed bfloat16 header path (`bfloat16/bfloat16.hpp`)

### Changed
- **Universal Library Version** - Updated from v3.77 to v3.91
- **pybind11 Version** - Updated to v2.13.6 for improved CMake support

## [0.6.4] - 2026-01-21

### Added
- **Conv3d Operator Support** (`python/kpu/fx_converter.py`)
  - `_numpy_conv3d()` - NumPy implementation of 3D convolution using im2col
  - `_im2col_3d()` - 3D patch extraction with dilation and grouped convolution support
  - `_emit_conv3d()` and `_emit_conv3d_module()` - FX graph handlers for F.conv3d and nn.Conv3d

- **3D Pooling Operators** (`python/kpu/fx_converter.py`)
  - `_numpy_max_pool3d()` - 3D max pooling with stride tricks
  - `_numpy_avg_pool3d()` - 3D average pooling
  - `_numpy_adaptive_avg_pool3d()` - Adaptive 3D average pooling (global pooling optimized)
  - Emit functions for nn.MaxPool3d, nn.AvgPool3d, nn.AdaptiveAvgPool3d

- **BatchNorm3d Support** (`python/kpu/fx_converter.py`)
  - `_emit_batch_norm3d_module()` - Handler for nn.BatchNorm3d with 5D tensor reshape

- **Video Model Compatibility** (`docs/model_compatibility.md`)
  - R3D-18: PASSED (diff=8.94e-08)
  - R2+1D-18: PASSED (diff=1.19e-07)
  - MC3-18: PASSED (diff=2.09e-07)

### Changed
- **F.batch_norm Handler** (`python/kpu/fx_converter.py`)
  - Now dynamically detects input dimensionality (4D vs 5D)
  - Correctly reshapes mean/var/weight/bias for both 2D and 3D batch normalization

- **Model Compatibility Matrix** (`docs/model_compatibility.md`)
  - Updated to 45 models tested (40 PASSED, 5 PARTIAL, 0 FAILED)
  - Added Video Models section
  - Updated operator support to include 3D operators
  - Removed Conv3d from "Not Supported" list

### Version
- Bumped to v0.6.4 in `python/kpu/__init__.py` and `python/pyproject.toml`

## [0.6.0] - 2026-01-20

### Added
- **Kernel Fusion Support** (`python/kpu/fusion.py`)
  - `FusionCompiler` - Compiler pass for automatic pattern detection and fusion
  - `FusionPattern` - Abstract base class for fusion patterns
  - `MatMulBiasActivation` - Pattern for MatMul + Add (bias) + Activation
  - `MatMulActivation` - Pattern for MatMul + Activation (no bias)
  - `FusionGroup` - Represents a group of operations to be fused
  - `estimate_memory_savings()` - Utility to estimate memory traffic reduction

- **Fused Operation Types** (`python/kpu/graph.py`, `python/kpu/dfx_emitter.py`)
  - `FUSED_MATMUL_BIAS_RELU` - MatMul + Add + ReLU (~2.8x memory savings)
  - `FUSED_MATMUL_BIAS_GELU` - MatMul + Add + GELU (~2.8x memory savings)
  - `FUSED_MATMUL_BIAS_SILU` - MatMul + Add + SiLU (~2.8x memory savings)
  - `FUSED_MATMUL_RELU` - MatMul + ReLU (~2x memory savings)
  - `OpType.is_fused()` method to identify fused operations

- **Fused Op Runtime Execution** (`python/kpu/runtime.py`)
  - Behavioral execution handlers for all fused operation types
  - Correct numerical output matching unfused computation

- **Fusion Demo and Tests**
  - `examples/fusion/ffn_fusion.py` - Demo comparing fused/unfused FFN execution
  - `python/tests/test_fusion.py` - 16 tests for pattern detection, correctness, graph rewriting

### Changed
- **Compiler** (`python/kpu/compiler.py`)
  - Fusion enabled by default (`optimize=True`)
  - Use `@kpu.compile(optimize=False)` to disable fusion

- **Tests** (`python/tests/test_kpu.py`)
  - Updated graph/DFX generation tests to use `optimize=False` for unfused behavior testing

## [0.5.7] - 2026-01-20

### Added - 2026-01-20
- **v0.5.x C++ Kernel Series Complete** (`include/sw/kpu/kernel.hpp`, `src/system/simulator/kernel.cpp`)
  - v0.5.6: Pool2D kernel with `create_pool2d()`, `create_max_pool2d()`, `create_avg_pool2d()`, `create_global_avg_pool2d()`
  - v0.5.7: Softmax kernel with `create_softmax()`, negative axis indexing, FLOP calculation (8N-2 per softmax)
  - `Pool2DConfig` struct: pool_type, batch_size, channels, dimensions, kernel size, stride, padding
  - `SoftmaxConfig` struct: shape, axis, reduction_size(), num_softmax_ops(), total_flops()

- **v0.5.x Validation Test Suite** (`python/tests/test_v05_kernel_validation.py`)
  - 28 tests validating all v0.5.x kernels (Conv2D, Attention, LayerNorm, RMSNorm, BatchNorm, Elementwise, Pool2D, Softmax)
  - Correctness tests with numerical verification
  - TRANSACTIONAL mode access tests
  - Transformer encoder block integration test
  - All v0.5.0 roadmap success criteria validated

### Fixed - 2026-01-20
- **ATTENTION Runtime Handler** (`python/kpu/runtime.py`)
  - Implemented `DFXOpCode.ATTENTION` handler in behavioral runtime
  - Multi-head attention with QKV projections, scaled dot-product attention, causal masking, output projection
  - Enables compiled attention functions to execute in BEHAVIORAL and TRANSACTIONAL modes

### Changed - 2026-01-20
- **ROADMAP.md** (`docs/ROADMAP.md`)
  - Updated current status to v0.5.7
  - Marked all v0.5.0 success criteria as validated
  - Added kernel completion table (v0.5.0-v0.5.7)

### Added - 2026-01-16
- **Python KPU Package** (`python/kpu/`)
  - High-level Python API for KPU simulator with decorator-based compilation
  - `@kpu.compile` decorator for tracing Python functions into DFX IR
  - `kpu.Tensor` class with NumPy interoperability and operator overloading (`@`, `+`, `-`, `*`, `/`)
  - Operator functions: `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `softmax`, `sum`, `mean`, `matmul`, `linear`
  - `OpGraph` class for operation DAG with topological ordering and validation
  - `DFXProgram` generation with JSON serialization/deserialization
  - `KPURuntime` with BEHAVIORAL execution using NumPy for functional correctness
  - Multi-fidelity support: `BEHAVIORAL`, `TRANSACTIONAL`, `CYCLE_ACCURATE` constants

- **Python Package Examples and Tests** (`python/`)
  - `examples/mnist_mlp.py` - Complete MNIST MLP example (784→128→64→10) with NumPy verification
  - `tests/test_kpu.py` - 20 tests covering tensors, operators, compiler, DFX emitter
  - `pyproject.toml` - Package configuration for pip installation
  - `README.md` - Quick start guide and API documentation

- **Native Bindings Infrastructure** (`python/kpu/_native/`)
  - `kpu_native.cpp` - pybind11 bindings for optional C++ acceleration
  - `CMakeLists.txt` - Build configuration outputting to package directory
  - `__init__.py` - Package init with graceful fallback when bindings unavailable
  - Supports all operators: matmul, relu, gelu, silu, sigmoid, tanh, softmax, add, sub, mul, div, neg, exp, log, sqrt
  - FLOP counting and timing statistics

- **Virtual Platform Documentation** (`docs/09-virtual-platform/`)
  - `exaloop-integration-design.md` - Comprehensive Exaloop/Codon integration design
  - `qemu-vs-userspace-runtime.md` - Analysis of QEMU vs user-space runtime tradeoffs

### Changed - 2026-01-16
- **Root CMakeLists.txt**
  - Added section to build `python/kpu/_native` when `KPU_BUILD_PYTHON_BINDINGS=ON` and pybind11 available

### Changed - 2026-01-15
- **Documentation Reorganization** (`docs/`)
  - Restructured from ~70 flat files to organized hierarchy with 9 numbered categories
  - Created `01-architecture/` through `09-virtual-platform/` for core simulator components
  - Added subdirectories: `03-memory-subsystem/{controllers,invariants,l3-l2-l1}`, `05-data-movement/{dma,noc,pcie}`
  - Consolidated external references under `reference/gpu-specs/`
  - Reorganized project management under `project/{milestones,reports,partners}`
  - Archived deprecated documents to `archive/{development-notes,status,superseded}`
  - All moves done via `git mv` to preserve file history

### Added - 2026-01-15
- **Documentation Index** (`docs/README.md`)
  - Comprehensive navigation guide with quick start section
  - Table of contents for all documentation categories
  - Key concepts section covering multi-fidelity simulation and credit-based dataflow
  - Navigation tips for common use cases

### Fixed - 2026-01-14
- **OFG Visualization NaN% Statistics** (`tools/visualization/ofg_execution_animation.html`)
  - Fixed field name mismatch: display code expected `dma_loads`/`dma_stores`/`matmuls` but traces use `dma_pushes`/`dma_pulls`/`computes`
  - Added fallback lookups supporting both old and new naming conventions
  - Progress bars and percentages now display correctly

- **OFG Visualization Loop Progress Display** (`tools/visualization/ofg_execution_animation.html`)
  - Fixed loop progress showing zero-indexed values (e.g., "1/2" instead of "2/2" when complete)
  - Changed display to show completion count: `${loopState.i + 1}/${m}` for intuitive progress tracking

- **OFG Visualization Missing Event Log Entries** (`tools/visualization/ofg_execution_animation.html`)
  - Added `logEvent()` calls to BlockMover events (BM_PUSH, BM_PULL, PUSH_TO_L2, PULL_FROM_L2)
  - Added `logEvent()` calls to Streamer events (STR_FEED_A/B, FEED_WEST/NORTH, STR_DRAIN, DRAIN)
  - Added `logEvent()` calls to TILE_READY and TILE_COMPLETE events
  - Event log now shows complete dataflow pipeline activity

### Changed - 2026-01-14
- **OFG Embedded Demo Trace** (`tools/visualization/ofg_execution_animation.html`)
  - Changed from 4×4×2 tiles (32 matmul ops) to 2×2×3 tiles (12 matmul ops)
  - Matches `--tiny` CLI option for educational examples
  - Shows buffer reuse patterns more clearly

- **OFG Visual Separation** (`tools/visualization/ofg_execution_animation.html`)
  - Added labels ("Buffer Occupancy:", "Bank Occupancy:", "Stream Buffers:")
  - Added dashed separators between buffer displays and executor OFG states
  - Clearer distinction between tile storage and executor state machines

### Added - 2026-01-09
- **DMA Pattern Test Suite** (`patterns/dma/`)
  - Complete infrastructure for DMA data movement validation
  - `common/dma_harness.hpp`: Test harness integrating DMA + Memory Controller + NoC
  - `common/dma_configs.hpp`: Standard DMA configuration presets
  - `common/matrix_layouts.hpp`: Matrix addressing with pitch support for tile extraction
  - STREAM patterns: `stream_copy.cpp`, `stream_triad.cpp`
  - GEMM tile patterns: `tile_aligned.cpp`, `tile_pitched_narrow.cpp`, `tile_pitched_wide.cpp`, `tile_page_boundary.cpp`, `a_tile_row_major.cpp`, `b_tile_col_major.cpp`
  - Conv2D pattern: `input_tile_nhwc.cpp`
  - Documentation: `README.md`, `INVARIANTS.md`

- **DMA-to-MC Trace Linkage** (`patterns/dma/common/dma_harness.hpp`)
  - Explicit `dma_transfer_id` field in MC trace entries
  - Accurate timing correlation between DMA and MC components
  - Click-to-highlight support in visualization

- **DMA Swimlane Visualization** (`traces/dma/tools/swimlane.html`)
  - Interactive swimlane view with DMA channels and MC banks
  - Left sidebar with statistics (transfers, bandwidth, page hits)
  - DMA-MC association highlighting on click
  - Bank utilization display
  - File loading, zoom, and pan controls

### Changed - 2026-01-09
- **Memory Controller Interface** (`include/sw/kpu/components/memory/memory_controller_interface.hpp`)
  - Added `trace_entries()` method to retrieve MC trace data
  - Added `clear_trace_entries()` method for trace management
  - Full `trace_entry.hpp` include for TraceEntry type

### Fixed - 2026-01-09
- **DMA Transfer Start Cycle Computation** (`patterns/dma/common/dma_harness.hpp`)
  - Fixed issue where all DMA transfers showed `submit_cycle=0`
  - Compute actual start from associated MC commands using completion-based mapping
  - Each transfer now shows when MC begins processing its request

- **DMA WRITE Trace Generation** (`src/components/datamovement/cycle_accurate_dma_engine.cpp`)
  - Fixed DMA engine only issuing memory reads, ignoring write transfers
  - STALLED_MEMORY_FULL state now checks transfer direction and calls appropriate function
  - stream_copy pattern now correctly shows 6R + 6W memory controller commands

### Added - 2026-01-08
- **HBM3E Pattern Infrastructure** (`patterns/memory/hbm3e/`)
  - Separate directory structure for HBM3E variants (8.4-9.6 Gbps)
  - `common/hbm3e_configs.hpp`: HBM3E-8400 @ 4.2 GHz and HBM3E-9600 @ 4.8 GHz configs
  - `common/hbm3e_harness.hpp`: Test harness with variant-aware clock frequencies
  - HBM3E-9600 patterns: page_hits, page_conflicts, max_bandwidth
  - HBM3E-8400 pattern: page_hits
  - Swimlane visualization labeled for HBM3E (1.23 TB/s peak)
  - Traces output to `traces/memory/hbm3e/` (separate from HBM3)

- **HBM2E Pattern Infrastructure** (`patterns/memory/hbm2e/`)
  - Separate directory structure for HBM2E variants (3.2-3.6 Gbps)
  - `common/hbm2e_configs.hpp`: HBM2E-3200 @ 1.6 GHz and HBM2E-3600 @ 1.8 GHz configs
  - `common/hbm2e_harness.hpp`: Test harness with variant-aware clock frequencies
  - HBM2E-3600 patterns: page_hits, page_conflicts, max_bandwidth
  - HBM2E-3200 pattern: page_hits
  - Swimlane visualization labeled for HBM2E (460.8 GB/s peak)
  - Traces output to `traces/memory/hbm2e/` (separate from HBM2)

- **HBM2E and HBM3E Timing Parameters** (`src/components/memory/memory_controller_factory.cpp`)
  - Distinct timing for HBM2E-3600 @ 1.8 GHz (461 GB/s peak): tRCD=7, tRP=8, tRAS=16, tRC=24
  - Distinct timing for HBM3E-9600 @ 4.8 GHz (1229 GB/s peak): tRCD=5, tRP=5, tRAS=10, tRC=14
  - Scaled from base variants using clock ratio (HBM2E: 0.56x, HBM3E: 0.58x)

- **HBM2 Trace Validator** (`patterns/memory/hbm2/common/trace_validator.py`, `patterns/memory/hbm2/INVARIANTS.md`)
  - Python trace validator for HBM2 traces with structure and timing invariant checking
  - INV-001 to INV-004: Transaction structure invariants
  - INV-100 to INV-108: Timing constraint invariants (tRCD, tRP, tRRD, tFAW, tCCD, tRAS, tRC)
  - Pseudo-channel aware bank group calculations
  - Comprehensive INVARIANTS.md documentation

- **HBM3 Trace Validator** (`patterns/memory/hbm3/common/trace_validator.py`, `patterns/memory/hbm3/INVARIANTS.md`)
  - Python trace validator for HBM3 traces with HBM3-5600 timing parameters
  - Same invariant structure as HBM2 adapted for 16-channel architecture
  - Comprehensive INVARIANTS.md documentation

- **HBM2 Memory Controller** (`include/sw/kpu/components/hbm2_memory_controller.hpp`, `src/components/memory/hbm2_memory_controller.cpp`)
  - Cycle-accurate HBM2-2000 memory controller (256 GB/s peak bandwidth)
  - 8 channels, 2 pseudo-channels per channel, 16 banks per PC (256 total banks)
  - Full timing parameter support (tRCD=12, tCL=18, tRP=14, tRAS=28, tRC=42, etc.)
  - Bank group timing (tRRD_L, tRRD_S, tCCD_L, tCCD_S, tFAW)
  - Chrome Trace export for Perfetto visualization
  - Semantic invariant checking aligned with LPDDR5/GDDR6 patterns

- **HBM3 Memory Controller** (`include/sw/kpu/components/hbm3_memory_controller.hpp`, `src/components/memory/hbm3_memory_controller.cpp`)
  - Cycle-accurate HBM3-5600 memory controller (716.8 GB/s peak bandwidth)
  - 16 channels, 2 pseudo-channels per channel, 16 banks per PC (512 total banks)
  - Full timing parameter support (tRCD=8, tCL=8, tRP=8, tRAS=16, tRC=24, etc.)
  - Bank group timing and per-bank refresh support
  - Chrome Trace export for Perfetto visualization

- **HBM2 Pattern Test Suite** (`patterns/memory/hbm2/`)
  - 9 pattern tests covering single-bank, two-bank, pseudo-channel, multi-channel, and bandwidth scenarios
  - Common infrastructure: `hbm2_harness.hpp`, `hbm2_configs.hpp`
  - Patterns: page_hits, page_conflicts, mixed_rw, same_group, diff_groups, dual_pc, four_channel, eight_channel, max_bandwidth

- **HBM3 Pattern Test Suite** (`patterns/memory/hbm3/`)
  - 9 pattern tests mirroring HBM2 suite
  - Common infrastructure: `hbm3_harness.hpp`, `hbm3_configs.hpp`
  - Patterns: page_hits, page_conflicts, mixed_rw, same_group, diff_groups, dual_pc, eight_channel, sixteen_channel, max_bandwidth

- **HBM Memory Characterization** (`docs/analysis/memory-characterization.md`)
  - Technology Summary table with LPDDR5, GDDR6, HBM2, HBM3
  - HBM2-2000 full characterization (timing, latency, bandwidth)
  - HBM3-5600 full characterization (timing, latency, bandwidth)
  - HBM Evolution: HBM2 to HBM3 to HBM4 comparison
  - LPDDR5 vs HBM2, HBM2 vs HBM3, All Technologies comparisons
  - Technology Selection Guide and Design Recommendations

### Changed - 2026-01-08
- **Memory Technology Enum** (`include/sw/kpu/fidelity/simulation_fidelity.hpp`)
  - Added `HBM2`, `HBM2E` to `MemoryTechnology` enum
  - Updated `to_string()` and `is_hbm()` helper functions

- **Trace Component Types** (`include/sw/trace/trace_entry.hpp`)
  - Added HBM2 component types (HBM2_BANK, HBM2_PSEUDO_CHANNEL, etc.)
  - Added HBM3 component types (HBM3_BANK, HBM3_PSEUDO_CHANNEL, etc.)

- **Collapsible HBM Swimlane Visualization** (`traces/memory/hbm2/tools/swimlane.html`, `traces/memory/hbm3/tools/swimlane.html`)
  - Hierarchical collapsible view: Channel → Pseudo-Channel → Banks + Data Bus
  - Expand All / Collapse All controls
  - Activity indicators for collapsed sections
  - Per-channel color coding
  - DQ pin range display showing physical bus mapping (e.g., "PC0 (DQ[63:0])")
  - Bank ID decoding: `bank_id = channel * 32 + pc * 16 + bank`
  - HBM2: 8 channels × 2 PCs × 64-bit = 1024-bit I/O bus
  - HBM3: 16 channels × 2 PCs × 32-bit = 1024-bit I/O bus

### Fixed - 2026-01-08
- **HBM Swimlane Visualization Bugs** (`traces/memory/hbm2/tools/swimlane.html`, `traces/memory/hbm3/tools/swimlane.html`)
  - Fixed bandwidth calculation double-counting from `databus-*` and `globalbus-*` events
  - Fixed min/max latency not highlighting associated transaction (passed object instead of ID)
  - Fixed horizontal panning broken by period overlay `z-index` above sticky lane labels
  - Added CA Bus activity indicators when collapsed (matching Data Bus behavior)
  - Fixed playback cursor misaligned after zooming (duplicate 200px offset in CSS + JS)
  - Added preset zoom levels [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0] with reset to 100%
  - Added keyboard shortcut '0' to reset zoom; clickable zoom level display for reset

- **LPDDR5/GDDR6 Swimlane Visualization** (`traces/memory/lpddr5/tools/swimlane.html`, `traces/memory/gddr6/tools/swimlane.html`)
  - Fixed playback cursor misaligned after zooming (removed duplicate offset in JS)
  - Added preset zoom levels [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0] for predictable zoom stepping
  - Added `resetZoom()` function and keyboard shortcut '0' to return to 100%
  - Made zoom level display clickable to reset to 100%

- **HBM Invariant Checking** (`hbm2_memory_controller.cpp`, `hbm3_memory_controller.cpp`)
  - Changed from generic "state_until in past" checks to semantic invariant checking
  - Aligned with LPDDR5/GDDR6 approach checking tRCD, tRAS, tWR, tRTP violations
  - READING/WRITING states use `burst_end` not `state_until` for timing

- **Trace Generation Script Missing Patterns** (`traces/scripts/generate_all_traces.sh`)
  - Added missing LPDDR5 patterns: stream, multi_dma, max_bandwidth, page_burst
  - Added missing GDDR6 patterns: stream, multi_dma, max_bandwidth, page_burst, eight_bank_bandwidth
  - Root cause: `--clean` option deleted all traces but script only regenerated subset

- **LPDDR5 Page Burst Test Failure** (`patterns/memory/lpddr5/bandwidth/page_burst.cpp`)
  - Created `bandwidth_test_config()` with queue_depth=2048 (was 64)
  - Fixed silent request drops due to queue overflow
  - Changed assertions to expect >90% hit rate instead of exact count
  - Accounts for DRAM refresh (tREFIpb=244) periodically closing pages

### Fixed - 2026-01-07
- **GDDR6/LPDDR5 Multi-DMA Trace Generation** (`patterns/memory/{gddr6,lpddr5}/complex/multi_dma.cpp`)
  - Fixed bug where trace export only showed 8 of 16 GDDR6 banks (and 4 of 8 LPDDR5 banks)
  - Root cause: Queue depth (64) was smaller than total requests (128) submitted before simulation
  - Increased queue depth to 256 for trace export sections
  - **Result**: GDDR6 trace now shows all 16 banks (144 events, was 72); LPDDR5 shows all 8 banks (136 events)

### Added - 2026-01-07
- **Memory Characterization Documentation** (`docs/memory-characterization.md`)
  - Comprehensive latency and bandwidth analysis for LPDDR5-6400 and GDDR6-16000
  - Timing parameter tables (tRCD, tCL, tRP, tRAS, tRC, etc.)
  - Latency characterization: page hit, page empty, page conflict scenarios
  - Bandwidth scaling analysis (1-16 banks)
  - STREAM benchmark results (Copy, Scale, Add, Triad)
  - Multi-DMA performance (4-32 concurrent engines)
  - Comparative analysis between LPDDR5 and GDDR6
  - Pattern category descriptions (Levels 1-7)

- **Updated Trace Directory Documentation** (`traces/README.md`)
  - Complete directory structure for both LPDDR5 and GDDR6 traces
  - Memory technology specifications and quick start commands
  - Pattern category descriptions with trace file listings
  - Visualization tool reference table
  - Chrome Trace Format documentation

### Changed - 2026-01-06
- **SystolicArray Template Refactoring** (`include/sw/kpu/components/systolic_array.hpp`)
  - Converted `SystolicArray` from a non-templated class to `template<typename Scalar> class SystolicArray`
  - Removed hardcoded `using Scalar = double;` typedef
  - Enables instantiation with different numeric types: `float`, `double`, `int8_t`, `int32_t`, and custom types
  - Moved all implementations to header (required for templates)
  - Updated `ProcessingElement<Scalar>` to use `Scalar{0}` for type-generic zero values
  - Added explicit instantiations for `int8_t`, `int32_t`, `float`, `double`
  - Updated `ComputeFabric` to use `SystolicArray<float>` explicitly
  - **Benefit**: Systolic array structure is now orthogonal to scalar type, enabling quantized inference and custom numeric types

### Added - 2026-01-05
- **Multi-Fidelity Calibration Framework** (`src/calibration/`, `tools/calibration/`)
  - Complete calibration workflow for deriving behavioral and transactional model parameters from cycle-accurate simulation
  - Calibration storage schema with JSON serialization (`calibration_storage.hpp`)
  - Parameter extraction from cycle-accurate statistics (`calibration_extraction.hpp`)
  - Quality assessment with severity levels, scores, and grades (`calibration_quality.hpp`)
  - CLI tools:
    - `kpu-calibrate` - Run cycle-accurate simulation and extract calibration parameters
    - `kpu-validate` - Cross-validate calibration across all fidelity levels with quality reporting
  - Test coverage: `calibration_storage_test`, `calibration_extraction_test`, `calibration_quality_test`
  - Documentation: `docs/MULTI_FIDELITY_CALIBRATION_WORKFLOW.md`

### Fixed - 2026-01-05
- **Transactional Memory Controller Accuracy** (`transactional_memory_controller.cpp`)
  - Use physical timing parameters (tCL, tRCD, tRP) for service time calculation
  - Removed redundant queueing delay that double-counted contention
  - **Result: Cycle error reduced from 2013% to 1.3%** vs cycle-accurate reference

### Added - 2026-01-03
- **LPDDR5 Memory Controller Pattern Test Suite** (`patterns/`)
  - Complete rewrite of pattern infrastructure for cycle-accurate LPDDR5 controller
  - Progressive bank access testing: 1, 2, 3, 4 banks
  - Common infrastructure:
    - `patterns/common/lpddr5_configs.hpp` - Standard single/dual channel LPDDR5-6400 configs
    - `patterns/common/pattern_harness.hpp` - Reusable test harness with tracing
  - Pattern 01 tests:
    - Single bank page hits (same row)
    - Single bank page conflicts (different rows)
    - Two banks same group (tRRD_L timing)
    - Two banks different groups (tRRD_S timing)
    - Three banks mixed groups
    - Four banks full group (tFAW testing)
    - Four banks across groups (max parallelism)
    - Mixed read/write with turnarounds (tRTW, tWTR)
  - Chrome Trace export for Perfetto visualization
  - Documentation: `patterns/PLAN.md`, `patterns/ARCHITECTURE.md`

### Fixed - 2026-01-03
- **GCC Warning in LPDDR5MemoryController** (`lpddr5_memory_controller.cpp`)
  - Fixed false positive `-Wstringop-overflow` warning in constructor
  - Added explicit bounds check with `std::min<uint8_t>()` for loop variable

- **CI Build Failure** (`CMakeLists.txt`)
  - Made `add_subdirectory(patterns)` conditional on directory existence
  - Prevents build failure when patterns directory not present

### Added - 2025-12-31
- **Standalone DFG Toolchain** (`tools/dfg/`)
  - Complete CLI toolchain for Data Flow Graph generation, scheduling, compilation, visualization, and analysis
  - 5 standalone tools with JSON interchange format:
    - `kpu-dfg-gen` - Generate DFG from templates (matmul)
    - `kpu-dfg-sched` - Schedule using ASAP/ALAP/LIST algorithms
    - `kpu-dfg-compile` - Compile to BlockMover programs
    - `kpu-dfg-viz` - Export to DOT, Chrome Trace, Mermaid
    - `kpu-dfg-analyze` - Statistics, critical path, validation
  - JSON serialization library (`tools/dfg/common/`):
    - `dfg_json.hpp/cpp` - TileDataFlowGraph serialization
    - `schedule_json.hpp/cpp` - DFGSchedule serialization
    - `compiled_json.hpp/cpp` - CompiledSchedule/BlockMoverProgram serialization
  - Chrome Trace export for Perfetto timeline visualization
  - DOT/GraphViz export for graph structure visualization
  - Comprehensive documentation: `docs/dfg-toolchain.md`

- **Example Pipeline**:
  ```bash
  kpu-dfg-gen --template matmul -M 1024 -N 1024 -K 1024 --tiles 4x4x4 -o dfg.json
  kpu-dfg-sched -i dfg.json -o scheduled.json --algorithm ASAP
  kpu-dfg-compile -i scheduled.json -o programs.json
  kpu-dfg-viz -i scheduled.json -o timeline.json --format chrome-trace
  kpu-dfg-analyze -i dfg.json --stats --critical-path
  ```

### Added - 2025-12-29 (Session 2)
- **FLIT-Level Tracking in NoC** (`include/sw/kpu/noc/noc.hpp`, `src/noc/noc.cpp`)
  - New event types: `FLIT_SEND` and `FLIT_ARRIVE` for fine-grained visualization
  - Extended `NoCTraceEvent` with `flit_index`, `num_flits`, `src_router`, `dst_router` fields
  - Sampled FLIT emission to balance trace detail vs overhead:
    - `FLIT_ARRIVE`: Every 256 FLITs → 16 progressive fill updates per tile
    - `FLIT_SEND`: Every 512 FLITs → 8 link activity updates per hop
  - For 256KB tiles (4096 FLITs): progressive fill shows ~256 cycles per 6.25% increment

- **Progressive Tile Filling Animation** (`tools/visualization/generate_noc_animation.py`)
  - Tracks partial tile fill state per L3 cache (`l3PartialTiles` map)
  - Visual progressive fill: light background fills from bottom-up as FLITs arrive
  - Displays percentage completion on partial tiles (e.g., "A0.0 25%")
  - Link activity visualization showing tensor type during FLIT transfer
  - New light color palette (`TENSOR_COLORS_LIGHT`) for partial tile backgrounds

- **Extended NoC Trace CSV Format**
  - New columns: `flit_index`, `num_flits`, `src_router`, `dst_router`
  - Full format: `cycle,type,router_id,port,packet_seq,tensor,m_tile,n_tile,k_tile,flit_index,num_flits,src_router,dst_router`

### Verified - 2025-12-29 (Session 2)
- **Systolic Wavefront Timing**
  - A and B tiles injected concurrently (1 cycle apart): A[0,0,k=0] at cycle 2, B[0,0,k=0] at cycle 3
  - Proper East/South flow: A tiles flow East, B tiles flow South
  - K-step barriers synchronizing correctly: K=1 tiles start after K=0 completes
  - Parallel DMA channels working: each row/column has independent injection

### Fixed - 2025-12-29
- **Timing Bug in StatefulBlockMover** (`stateful_block_mover.cpp:200-247`)
  - `execute_current()` was passing `0` as cycle to all command executors
  - Added `current_cycle_` member variable and public accessor
  - Now all transfer timing calculations use correct current cycle
  - Impact: Transfer completion times now calculated correctly

- **Infinite Loop in L3Interconnect** (`l3_interconnect.cpp:76-81`)
  - When link busy, `inject_packet()` was re-queuing packets with same cycle
  - The `step()` while loop immediately re-processed them, causing infinite loop
  - Fixed by queuing for `cycle + 1` instead of `cycle`
  - Impact: Simulation no longer hangs on busy links

- **Interconnect Callback Timing** (`stateful_block_mover.cpp:617-624`)
  - Transfer callback was passing `0` for cycle when injecting packets
  - Now uses `mover->current_cycle()` for correct timing

### Changed - 2025-12-29
- **Block Systolic Matmul Example** (`examples/blas/block_systolic_matmul.cpp`)
  - Cleaned up debug output for production use
  - Added note that compute time is not simulated (data movement only)
  - Improved progress reporting for long simulations

### Added - 2025-12-25
- **Benchmark Infrastructure (Phase 7)**
  - `include/sw/benchmark/benchmark.hpp` - Complete benchmark harness API:
    - `BenchmarkHarness` class with systematic sweep methods
    - `BenchmarkResult` and `BenchmarkSuite` structs for result collection
    - `HardwareSpec` for roofline performance modeling
    - Size sweeps, tile sensitivity analysis, activation comparisons
  - `src/benchmark/benchmark.cpp` - Full implementation
  - `src/benchmark/CMakeLists.txt` - Build configuration with `StillwaterKPU::Benchmark` alias

- **Benchmark Test Suite**
  - `tests/benchmarks/test_matmul_benchmarks.cpp` - 7 matmul benchmark tests:
    - Size sweeps (64 to 2048)
    - Tile sensitivity analysis
    - Non-square and transformer-like dimensions
    - Roofline analysis
    - CSV export
  - `tests/benchmarks/test_mlp_benchmarks.cpp` - 5 MLP benchmark tests:
    - Activation function comparison (RELU, GELU, SIGMOID, TANH, SILU)
    - Transformer FFN benchmarks
    - Size sweeps with GELU
  - `tests/benchmarks/test_graph_benchmarks.cpp` - 6 multi-kernel graph tests:
    - Two-layer MLP graph
    - Deep MLP (5 layers)
    - Transformer FFN block
    - Diamond pattern (parallel branches)
    - Graph vs individual kernel comparison
    - Depth scaling analysis

- **Efficiency Diagnostic Tools**
  - `tests/benchmarks/test_efficiency_diagnostic.cpp` - Comprehensive diagnostic test:
    - Kernel/tile configuration display
    - Theoretical vs actual cycle comparison
    - Operation breakdown by resource type (DMA, BM, Streamer, Compute)
    - ASCII timeline visualization
    - Pipeline analysis (startup/drain cycles)
  - `docs/efficiency-bug-analysis.md` - Detailed analysis of efficiency bug

### Fixed - 2025-12-25
- **String concatenation error** in `benchmark.cpp` (line 202)
  - Changed `"mlp_" + activation_type_name()` to `std::string("mlp_") + ...`

- **CMake test registration** in `tests/benchmarks/CMakeLists.txt`
  - Changed from `catch_discover_tests()` to `add_test()` pattern for compatibility

- **Division by zero in executor** (`concurrent_executor.cpp:82-84`)
  - Added guards for zero tile dimensions in `initialize_layout_for_program()`
  - Uses default 64 for Ti/Tj/Tk if program dimensions are 0

- **FLOP count tolerance** in `test_graph_benchmarks.cpp`
  - Changed exact equality to 1% tolerance for MLP kernels
  - Accounts for bias and activation FLOPs not in basic matmul calculation

### Added - 2025-12-25 (Session 2)
- **Pipelined Tile Scheduling for Blocked Matmul**
  - Modified `OutputStationaryProgramBuilder::build()` in `src/isa/data_movement_isa.cpp`
  - Removed unnecessary barriers within K-loop for continuous accumulation
  - Added prefetch logic: load next k-tile while current streams to systolic array
  - Double-buffering for overlap of data movement and compute
  - Results: 96% compute utilization at 1024×1024 (up from 76%)
  - Overhead reduced from 31% to 4.2% for large matrices
  - Created `docs/SYSTOLIC_TILE_SCHEDULING.md` with analysis

### Fixed - 2025-12-25 (Session 2)
- **Critical Efficiency Bug in ConcurrentExecutor** (RESOLVED)
  - Modified `ConcurrentExecutor::schedule_instruction()` in `src/isa/concurrent_executor.cpp`
  - **STR_FEED_ROWS** now calculates and schedules compute cycles:
    - Compute cycles = Ti × Tj × Tk / systolic_size²
    - Streamer duration = max(transfer_cycles, compute_cycles)
    - Schedules both streamer and compute fabric operations
  - **STR_FEED_COLS** models transfer only (output-stationary dataflow)
    - B columns are broadcast while A rows stream
    - Compute already counted in STR_FEED_ROWS
  - **BARRIER** now waits for compute fabric completion
  - Results:
    - Before: 0% compute utilization across all sizes
    - After: 50-76% compute utilization depending on matrix size
    - Overhead trends from 100% (64×64) down to 31% (1024×1024)
  - Updated `docs/efficiency-bug-analysis.md` with fix details and results

### Added - 2025-12-06
- **CLAUDE.md Documentation File**
  - Created `CLAUDE.md` for Claude Code guidance when working in this repository
  - Includes build commands, architecture overview, key subsystems, and testing info

- **LPDDR5X Memory Pipeline Documentation**
  - `docs/LPDDR5X_MEMORY_PIPELINE.md` - Detailed walkthrough of memory timing:
    - LPDDR5X specifications (8533 MT/s, BL16, x16 channel)
    - Clock domain breakdown (I/O @ 4266 MHz, MC @ 250 MHz)
    - 64-byte cache line transfer timing analysis
    - Pipeline stages from DRAM to L3 tile
    - Latency vs throughput calculations

- **Tile Caching Architecture Design**
  - `docs/TILE_CACHING_ARCHITECTURE.md` - Three-phase implementation plan:
    - Phase 1: Software tile cache tracking (implemented)
    - Phase 2: ISA extensions for cached loads and refcounting
    - Phase 3: Hardware tile cache controller modeling
  - Addresses tile reuse, protection guarantees, and eviction policies

- **Software Tile Cache Implementation (Phase 1)**
  - `include/sw/kpu/isa/tile_cache.hpp` - Tile cache data structures:
    - `TileKey`, `TileCacheEntry`, `TileCacheStats` structs
    - `TileCache` class with LRU eviction and reference counting
    - `TileCacheTracker` helper for program builder integration
  - `src/isa/tile_cache.cpp` - Full implementation
  - Tracks tile residency by (matrix, ti, tj, tk) key
  - Statistics: hits, misses, hit rate, bytes saved

- **Tile Cache Integration in Program Builder**
  - Added `TileCacheState` to `OutputStationaryProgramBuilder`
  - New methods: `try_emit_load_a_tile()`, `try_emit_load_b_tile()`
  - Cache-aware load functions skip DMA for already-resident tiles
  - `get_cache_stats()` method for reporting cache performance
  - `enable_tile_caching` config option (default: true)

- **Tile Caching Demo (Example 6)**
  - Extended `data_movement_isa_matmul.cpp` with tile caching demonstration
  - Side-by-side comparison with and without caching
  - Shows 75% cache hit rate, 67% DMA reduction, optimal reuse factor

### Fixed - 2025-12-06
- **DMA Timing Model**
  - Fixed bandwidth calculation: was treating GB/s as bytes/cycle
  - Now uses `bus_width_bytes` for accurate cycle calculation
  - `cycles = ceil(bytes / bus_width_bytes)` instead of `bytes / bandwidth_gb_s`
  - Added `bus_width_bytes` member to `HardwareResource` class
  - Result: DMA cycles per 4KB tile dropped from 256 to 64

- **Tile Size Calculation for Layout**
  - Fixed `initialize_layout_for_program()` to use correct tile dimensions
  - Changed from `Ti × Tj` to `max(Ti × Tk, Tk × Tj)`
  - Properly reflects actual A and B tile sizes

- **Tile Reuse Factor**
  - Fixed external memory traffic estimation to only count actual DMA transfers
  - Reuse factor for 64×64×64 matmul improved from 1.67× to 1.00× (optimal)
  - DMA operations reduced by 40% for typical workloads

### Changed - 2025-12-06
- Updated `HardwareResource` constructor to accept `bus_width` parameter
- Updated `MemoryChannel` to include `bus_width_bytes` member
- Updated `ConcurrentExecutor` to pass bus widths when initializing resources
- Traffic estimates now distinguish between external memory (DMA) and internal (L3/L2)

### Added - 2025-12-01
- **Tile Layout Policies for Memory Channel Interleaving**
  - `include/sw/kpu/isa/tile_layout.hpp` - Four configurable layout policies:
    - `MATRIX_PARTITIONED`: Dedicates channels to specific matrices (0% conflicts)
    - `ROUND_ROBIN`: Distributes tiles evenly across all channels (~25% conflicts)
    - `ITERATION_AWARE`: Places A on even channels, B on odd channels (0% conflicts)
    - `HARDWARE_INTERLEAVED`: Address bits determine channel selection (realistic HW model)
  - `src/isa/tile_layout.cpp` - Full implementations with conflict analysis and reports
  - Factory function `create_tile_layout()` for runtime policy selection
  - `TileLocation` struct for physical tile placement (channel, address, L3/L2 IDs)
  - `LayoutConfig` struct with channel assignments and tile dimensions

- **Concurrent Executor Integration with Tile Layout**
  - Updated `ConcurrentExecutor` to use `TileLayout` for resource selection
  - `select_dma_channel()` now uses layout policy for conflict-free A/B access
  - `select_block_mover()` and `select_streamer()` distribute operations across all resources
  - Automatic layout initialization from program dimensions

- **Realistic Clock Domain and Bandwidth Modeling**
  - `ResourceConfig` now includes clock frequencies for each domain:
    - Compute fabric: 2.0 GHz (500 ps cycle time)
    - L1/L2/Streamer/BlockMover: 500 MHz (2 ns cycle time)
    - L3/DMA engines: 250 MHz (4 ns cycle time)
  - Bus widths: 64-byte (512-bit) for cache-line aligned transfers
  - Derived bandwidths: DMA 16 GB/s, BM 32 GB/s, STR 32 GB/s per resource

- **Enhanced Timeline Visualization**
  - Clock domain legend with frequencies, cycle times, and bandwidths
  - Total execution time in nanoseconds and microseconds
  - Scale information mapping cycles to real time
  - Aggregate bandwidth display for each resource type
  - Cycle-by-cycle view header shows time range in nanoseconds

- **Debug and Test Tools**
  - `examples/basic/tile_layout_test.cpp` - Compares all four layout policies
  - `examples/basic/concurrent_execution_debug.cpp` - Debug tool for concurrent scheduling
  - `docs/MEMORY_INTERLEAVING_DESIGN.md` - Design document for layout options

### Changed - 2025-12-01
- **Fixed Concurrent Resource Utilization**
  - Previously BM[2], BM[3], STR[2], STR[3] showed 0% utilization
  - Root cause: Hash-based channel selection caused A and B to collide
  - Fix: TileLayout ensures A and B tiles are always on different channels
  - Result: ~46% faster execution, all resources now utilized

- **Updated Default Bandwidths**
  - DMA: 50 GB/s → 16 GB/s (realistic LPDDR5X x16 @ 250 MHz)
  - BlockMover: 100 GB/s → 32 GB/s (64-byte bus @ 500 MHz)
  - Streamer: 200 GB/s → 32 GB/s (64-byte bus @ 500 MHz)

### Added - 2025-11-26
- **Domain Flow Execution (DFX) Layer**
  - Created PTX-equivalent hardware-agnostic intermediate representation for KPU
  - `include/sw/compiler/dfx/dfx.hpp` - Core DFX types and structures:
    - `DataType`, `MemoryLevel`, `DataflowStrategy` enums
    - `TensorDescriptor`, `TileSpec`, `TilingConfig` structures
    - `Operation` base class with `DataMoveOp`, `ComputeOp`, `BarrierOp` derived types
    - `Program` struct containing complete compiled kernel representation
  - `include/sw/compiler/dfx/dfx_object_file.hpp` - JSON serialization for .kpu files

- **KPU Kernel Compiler (`kpu-kernel-compiler`)**
  - Full compilation pipeline from DFG to .kpu object files
  - `tools/compiler/kpu-kernel-compiler/dfg_parser.hpp/cpp` - DFG/JSON file parsing
  - `tools/compiler/kpu-kernel-compiler/dfx_generator.hpp/cpp` - DFX program generation
  - `tools/compiler/kpu-kernel-compiler/object_writer.hpp/cpp` - .kpu file writer
  - CLI options: `-o`, `-d` (dataflow), `-t` (tile-strategy), `--emit-dfx`, `--dump`, `-v`
  - Supports output-stationary, weight-stationary, and input-stationary dataflows
  - Integrates with existing TileOptimizer for optimal tile size selection

- **KPU Loader Framework** (skeleton)
  - `tools/runtime/kpu-loader/` - Loader/driver framework
  - `object_reader.hpp/cpp` - Read and validate .kpu files
  - `schedule_binder.hpp/cpp` - Bind DFX operations to concrete hardware resources
  - Maps abstract operations to DMA engines, BlockMovers, and Streamers

- **Tools Directory Reorganization**
  - New category-based structure: `compiler/`, `runtime/`, `analysis/`, `development/`, `configuration/`, `benchmark/`
  - `kpu_add_tool()` CMake helper function for consistent tool creation
  - Moved Python tools to appropriate subdirectories

- **Implementation Plan Document**
  - `docs/compiler/KPU_COMPILER_IMPLEMENTATION_PLAN.md` - Comprehensive design document
  - Covers architecture, DFX format, object file structure, CLI design

### Changed - 2025-11-26
- **Renamed KIR to DFX**
  - Renamed namespace from `sw::kpu::compiler::kir` to `sw::kpu::compiler::dfx`
  - Renamed directory from `include/sw/compiler/kir/` to `include/sw/compiler/dfx/`
  - Renamed files: `kir.hpp` → `dfx.hpp`, `object_file.hpp` → `dfx_object_file.hpp`
  - Renamed class: `KIRGenerator` → `DFXGenerator` (with backward compatibility alias)
  - Updated version constants: `KIR_VERSION_*` → `DFX_VERSION_*`
  - Updated CLI flag: `--emit-kir` → `--emit-dfx`
  - Updated JSON key: `"kir_version"` → `"dfx_version"`

### Added - 2025-11-25
- **Strategy-Aware L2/L3 Scheduling**
  - Implemented proper dataflow strategy loop ordering in L2 tile scheduler
  - Added strategy-aware execution in L3 scheduler
  - Strategies now produce different (and correct) overfetch results:
    - **WS (Weight-Stationary)**: `tk → ti → tj` keeps B tiles resident
    - **IS (Input-Stationary)**: `tk → tj → ti` keeps A tiles resident
    - **OS (Output-Stationary)**: `ti → tj → tk` keeps C tiles resident
  - Added `strategy` field to `L2Schedule` struct to propagate strategy choice

- **Distributed L3 Support in Analysis Tools**
  - Added 1MB and 2MB L3 sizes to focused analysis (3→5 sizes, 108→180 configs)
  - Added 1MB and 2MB L3 sizes to comprehensive analysis (5→7 sizes, 405→567 configs)
  - Created `run_comprehensive_overnight.sh` convenience script

- **Analysis Documentation**
  - Created `L3_ANALYSIS_UPDATED.md` documenting distributed L3 support
  - Created `STRATEGY_AWARE_SCHEDULING_RESULTS.md` documenting bug fix and results
  - Updated analysis tools to use strategy-aware scheduling

### Fixed - 2025-11-25
- **Critical Overfetch Asymmetry Bug**
  - Fixed L2 scheduler's `generate_compute_order()` ignoring strategy parameter
  - Fixed L3 scheduler's `simulate_l2_execution()` using hard-coded OS loops
  - **Impact**: 380× improvement for 32k×7k workload (34.56× → 0.90× with WS)
  - Tall and wide matrices now show proper symmetry with correct strategy selection

- **Compiler Warnings**
  - Fixed unused parameter warnings in `l3_overfetch_analyzer.cpp`
  - Fixed unused parameter warnings in `schedule_characterizer_demo.cpp`

### Changed - 2025-11-25
- **L2 Tile Scheduler**
  - Moved `ReplacementPolicy` and `SchedulingStrategy` enums before `L2Schedule` struct
  - Updated `generate_compute_order()` to respect strategy parameter
  - Strategy now stored in generated L2 schedules

- **L3 Analysis Tools**
  - `l3_focused_analysis.cpp` generates separate L2 schedules for each strategy
  - `l3_comprehensive_analysis.cpp` applies strategy-aware scheduling
  - Both tools now test 1MB and 2MB L3 configurations

### Added - 2025-11-23
- **Tile Notation Improvements** in `ScheduleGenerator`
  - Added `TileIndex::label_A()`, `label_B()`, `label_C()` methods for proper mathematical notation
  - Tile labels now show correct dimensionality:
    - `A_tile[ti,tk]` - A tile indexed by M-dimension and K-dimension
    - `B_tile[tk,tj]` - B tile indexed by K-dimension and N-dimension
    - `C_tile[ti,tj]` - C tile indexed by M-dimension and N-dimension
  - Kept legacy `label(char)` method for backwards compatibility

- **Double-Buffering Infrastructure** in `ScheduleGenerator`
  - Implemented `apply_double_buffering()` method
  - Buffer ID tracking for commands (alternates between 0 and 1)
  - Dependency adjustment for buffer switching
  - **Known Issue**: Does not properly model resource constraints

- **Pipelining Infrastructure** in `ScheduleGenerator`
  - Implemented `apply_pipelining()` method
  - Dependency refinement to enable parallelism
  - **Known Issue**: Shows physically impossible parallelism (multiple commands on same resource)

- **Enhanced Timing Estimation** in `ScheduleGenerator`
  - Improved `estimate_timing()` to handle parallel command execution
  - Proper dependency-based scheduling
  - Commands scheduled when all dependencies satisfied

- **Command Timeline Visualization** in `schedule_generator_demo`
  - Added detailed timeline printing in `compare_strategies()`
  - Shows all commands with start/end cycles, duration, and buffer IDs
  - Changed demo matrix size from 512×512×512 to 128×128×128 for readable output
  - Visual comparison of Sequential, Double-buffered, and Fully-pipelined strategies

- **Session Documentation**
  - Created `docs/sessions/` directory for session logs
  - Added comprehensive session log for 2025-11-23 pipelining work

### Changed - 2025-11-23
- **ScheduleGenerator** tile label generation
  - Updated all command generation to use new tile notation
  - `generate_dma_commands()`, `generate_block_move_commands()`, `generate_stream_commands()`, `generate_compute_commands()` now use `TileIndex::label_A/B/C()`

- **schedule_generator_demo.cpp**
  - `compare_strategies()` now prints full command timeline for all three strategies
  - Matrix size reduced to 128×128×128 for strategy comparison (from 512×512×512)
  - Added detailed explanations of pipelining benefits

### Fixed - 2025-11-23
- **Compilation Error** in `schedule_generator.cpp`
  - Added missing `#include <iostream>` header

### Known Issues - 2025-11-23

#### Critical Design Flaws in Pipelining Implementation

The current pipelining and double-buffering implementation has fundamental flaws:

1. **Resource Constraints Not Modeled**
   - Schedules show physically impossible parallelism (e.g., 16 BlockMoves starting simultaneously)
   - No modeling of finite resource capacity (DMA engines, BlockMovers, Streamers)
   - No resource allocation or scheduling logic
   - **Impact**: Generated schedules cannot execute on actual hardware

2. **No True Overlap**
   - Dependencies don't correctly model producer-consumer relationships across pipeline stages
   - No real overlap between data movement and compute despite "pipelined" strategy
   - **Impact**: Performance estimates are incorrect

3. **Improper Tile Reuse**
   - Doesn't model tile reuse across K-dimension
   - Treats reused tiles as independent loads
   - **Impact**: Overstates memory traffic, incorrect cache modeling

4. **Missing Constraints**
   - No spatial routing constraints (which L3 tile connects to which L2 bank)
   - No bandwidth modeling for interconnects
   - No systolic array scheduling
   - **Impact**: Schedules violate physical hardware constraints

#### Test Coverage Gaps

- All 32 tests in `test_schedule_generator.cpp` pass
- **However**: Tests don't validate:
  - Resource constraint satisfaction
  - Physical feasibility of parallelism
  - Correct tile reuse modeling
  - Actual data movement and compute overlap

#### Recommendations for Future Work

See `docs/sessions/2025-11-23_schedule_generator_pipelining.md` for detailed recommendations:
- Phase 1: Add explicit resource capacity modeling and resource scheduler
- Phase 2: Model network topology and spatial constraints
- Phase 3: Implement tile reuse optimization
- Phase 4: Add bandwidth modeling for interconnects
- Phase 5: Correct dependency graph with resource hazards
- Alternative: Consider polyhedral scheduling approach (MLIR, Halide, TVM)

### Testing - 2025-11-23
- ✅ All 32 tests in `test_schedule_generator` pass
- ✅ Clean build with no warnings
- ✅ Demo executable runs and produces output
- ⚠️  Output shows physically impossible parallelism (design flaw, not implementation bug)

---

## Notes

### Session Logs
Detailed session logs are maintained in `docs/sessions/` directory:
- `2026-01-16_python_kpu_package.md` - Python KPU package with @kpu.compile decorator and DFX IR generation
- `2026-01-08_hbm2_hbm3_memory_controllers.md` - HBM2/HBM3 memory controllers, collapsible swimlane visualization, trace script and test fixes
- `2026-01-07_gddr6_trace_and_bandwidth_metrics.md` - GDDR6 trace fix and memory characterization documentation
- `2025-12-29_block_systolic_matmul_simulation.md` - Block systolic matmul bug fixes and FLIT-level tracking
- `2025-12-25_benchmarking_and_efficiency_analysis.md` - Benchmark infrastructure and efficiency bug fix
- `2025-11-26_dfx_compiler_implementation.md` - DFX layer and kernel compiler implementation
- `2025-11-25_strategy_aware_scheduling.md` - Strategy-aware L2/L3 scheduling fix
- `2025-11-23_schedule_generator_pipelining.md` - Double-buffering and pipelining attempt

### Version History
This CHANGELOG was created on 2025-11-23 to track changes going forward.
Previous changes to the KPU simulator are documented in:
- Git commit history
- Session logs in `docs/sessions/`
- Documentation in `docs/` directory
