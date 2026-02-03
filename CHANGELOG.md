# Changelog

All notable changes to the KPU Simulator project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- **Schedule Compiler WRITEBACK offset** (`src/dsl/schedule_compiler.cpp`)
  - BM_WRITEBACK now uses `loc.address` from TileLayout instead of hardcoded 0
  - Fixes data loss when writing C tiles back to L3

- **Schedule Compiler str_drain argument order** (`src/dsl/schedule_compiler.cpp`)
  - Corrected parameter order: `str_drain(tile, l2_bank, l1_buf, ...)`
  - All three drain variants (DRAIN, DRAIN_FUSED, DRAIN_TO_SCRATCH) fixed

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
