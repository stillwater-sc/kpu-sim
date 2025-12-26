# Changelog

All notable changes to the KPU Simulator project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `2025-11-26_dfx_compiler_implementation.md` - DFX layer and kernel compiler implementation
- `2025-11-25_strategy_aware_scheduling.md` - Strategy-aware L2/L3 scheduling fix
- `2025-11-23_schedule_generator_pipelining.md` - Double-buffering and pipelining attempt

### Version History
This CHANGELOG was created on 2025-11-23 to track changes going forward.
Previous changes to the KPU simulator are documented in:
- Git commit history
- Session logs in `docs/sessions/`
- Documentation in `docs/` directory
