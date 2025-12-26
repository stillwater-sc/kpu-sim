# Session Log: December 25, 2025
## Benchmarking Infrastructure and Efficiency Analysis

### Session Overview
This session continued Phase 7 (Benchmarking) of the KPU simulator development, implementing a comprehensive benchmark infrastructure and discovering a critical efficiency bug in the executor.

### Work Completed

#### 1. Benchmark Infrastructure (Phase 7)

**Files Created:**
- `include/sw/benchmark/benchmark.hpp` - Complete benchmark harness API
- `src/benchmark/benchmark.cpp` - Benchmark implementation
- `src/benchmark/CMakeLists.txt` - Build configuration
- `tests/benchmarks/test_matmul_benchmarks.cpp` - 7 matmul benchmark tests
- `tests/benchmarks/test_mlp_benchmarks.cpp` - 5 MLP benchmark tests
- `tests/benchmarks/test_graph_benchmarks.cpp` - 6 multi-kernel graph tests
- `tests/benchmarks/test_efficiency_diagnostic.cpp` - Efficiency diagnostic tool

**Key Features:**
- `BenchmarkHarness` class with sweep methods for systematic testing
- `BenchmarkResult` and `BenchmarkSuite` structs for result collection
- `HardwareSpec` for roofline performance modeling
- Size sweeps, tile sensitivity analysis, activation comparisons
- CSV export for external analysis
- Roofline data generation for gnuplot visualization

**Test Results:** 18 tests, 64 assertions - all passing

#### 2. Bug Fixes During Implementation

1. **String concatenation error** (`benchmark.cpp:202`)
   - Fixed `"mlp_" + activation_type_name()` to use `std::string("mlp_") + ...`

2. **CMake test registration**
   - Changed from `catch_discover_tests()` to `add_test()` pattern

3. **Division by zero in executor** (`concurrent_executor.cpp:82-84`)
   - Added guards for zero tile dimensions in `initialize_layout_for_program()`

4. **FLOP count tolerance** (`test_graph_benchmarks.cpp:116`)
   - Changed exact equality to 1% tolerance for MLP kernels (includes bias/activation FLOPs)

5. **LTO linker issues**
   - Fixed circular dependency by repeating `kpu_simulator` in link order

#### 3. Efficiency Bug Discovery

**Problem Identified:**
The benchmark showed unexpectedly low efficiency:
```
Size       Cycles    Efficiency   Compute Util
64×64×64   1280      60%          0.0%  ← No compute!
```

**Root Cause:**
The `ConcurrentExecutor` models only data movement time, not systolic compute time:
- `HardwareResource::schedule_op` calculates: `cycles = bytes / bus_width`
- This gives transfer time only (256 cycles for 16KB)
- Systolic array compute (1024 cycles for 64³) is completely missing

**Correct Model:**
For streamer operations feeding the systolic array:
```cpp
Cycle compute_cycles = (Ti × Tj × Tk × 2) / (systolic_size² × 2);
Cycle duration = max(transfer_cycles, compute_cycles);
```

#### 4. Debugging Tools Created

- **`efficiency_diagnostic`** - Comprehensive diagnostic test showing:
  - Kernel/tile configuration
  - Theoretical vs actual cycles
  - Operation breakdown by resource type
  - Timeline visualization
  - Pipeline analysis (startup/drain cycles)

- **Timeline Visualization** - ASCII Gantt chart output:
```
DMA[0]  |AAAA                    CCCC|
BM[0]   |    AAAA                    |
STR[0]  |        AAAA CCCC           |
```

### Files Modified

| File | Changes |
|------|---------|
| `tests/benchmarks/CMakeLists.txt` | Updated test registration, added efficiency_diagnostic |
| `src/benchmark/benchmark.cpp` | Fixed string concatenation |
| `src/isa/concurrent_executor.cpp` | Added zero-dimension guards |
| `tests/benchmarks/test_graph_benchmarks.cpp` | Fixed FLOP tolerance check |

### Files Created

| File | Purpose |
|------|---------|
| `include/sw/benchmark/benchmark.hpp` | Benchmark harness API |
| `src/benchmark/benchmark.cpp` | Benchmark implementation |
| `src/benchmark/CMakeLists.txt` | Build config |
| `tests/benchmarks/test_matmul_benchmarks.cpp` | Matmul benchmarks |
| `tests/benchmarks/test_mlp_benchmarks.cpp` | MLP benchmarks |
| `tests/benchmarks/test_graph_benchmarks.cpp` | Graph benchmarks |
| `tests/benchmarks/test_efficiency_diagnostic.cpp` | Efficiency diagnostic |
| `docs/efficiency-bug-analysis.md` | Detailed bug analysis |

### Key Insights

1. **Output Stationary Dataflow**: Confirmed as default (correct for square matrices)
   - C tiles accumulate in place
   - No partial sum read/write overhead

2. **Efficiency Formula**:
   - Current: `efficiency = achieved_gflops / roofline_predicted_gflops`
   - Roofline accounts for memory vs compute bound transitions

3. **Missing Compute Model**: The ISA is "Data Movement ISA" - compute happens implicitly during streaming, but the executor wasn't modeling this time.

### Remaining Work

1. **Fix compute timing** in `ConcurrentExecutor::schedule_instruction`:
   - Add compute cycle calculation for STR_FEED_ROWS/COLS
   - Track compute fabric utilization properly
   - Model pipeline fill/drain overhead

2. **Pipelining verification**: Ensure tiles are double-buffered and compute overlaps with data movement

3. **Performance regression tests**: Add tests that verify efficiency targets (>90% for large matrices)

### Commands Used

```bash
# Build benchmarks
cmake --build build --target benchmark_matmul benchmark_mlp benchmark_graph

# Run efficiency diagnostic
./build/tests/benchmarks/efficiency_diagnostic

# Run all benchmarks
./build/tests/benchmarks/run_all_benchmarks
```

### Session Statistics

- Duration: ~2 hours
- Files created: 8
- Files modified: 4
- Tests added: 20
- Bugs fixed: 5
- Critical issues identified: 1 (compute timing)
