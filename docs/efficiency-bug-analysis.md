# KPU Simulator Efficiency Bug Analysis

## Problem Statement (RESOLVED)

The benchmark suite previously showed unexpectedly low efficiency for matmul operations:

```
Size       Cycles    Ideal    Overhead   Compute Util
64         1280      1024     25%        0.0%
128        7680      8192     -6%        0.0%
256        48128     65536    -27%       0.0%
```

**Critical Issue**: Compute utilization was 0.0% for all sizes!

## Root Cause

The `ConcurrentExecutor` modeled data movement time but **completely ignored compute time**.

### Old (Incorrect) Model

The `HardwareResource::schedule_op` calculated cycles as:
```cpp
Cycle cycles = (bytes + bus_width_bytes - 1) / bus_width_bytes;
```

This only measured transfer time:
- STR_FEED_ROWS: 16KB / 64B = 256 cycles (transfer time)
- STR_FEED_COLS: 16KB / 64B = 256 cycles (transfer time)
- STR_DRAIN_OUTPUT: 16KB / 64B = 256 cycles (transfer time)

Total: ~768 streamer cycles + DMA/BM overhead = 1280 cycles

### Missing Compute Model

When the streamer feeds data to the systolic array, **computation happens**:
- A rows stream into the left edge
- B columns stream into the top edge
- Each cycle, 16×16 = 256 FMA operations occur
- Total FLOPs = 2 × Ti × Tj × Tk

For 64×64×64:
- Total FLOPs = 524,288
- Systolic throughput = 16×16×2 = 512 ops/cycle
- **Ideal compute cycles = 1024**

## Fix Applied (December 25, 2025)

Modified `ConcurrentExecutor::schedule_instruction()` in `src/isa/concurrent_executor.cpp`:

### STR_FEED_ROWS (drives compute)

```cpp
case DMOpcode::STR_FEED_ROWS: {
    // STR_FEED_ROWS feeds A matrix rows into the systolic array's left edge
    // This is the operation that primarily drives compute for output-stationary
    const auto& ops = std::get<StreamerOperands>(instr.operands);

    // Calculate transfer cycles (existing model)
    Cycle transfer_cycles = (transfer_size + str.bus_width_bytes - 1) / str.bus_width_bytes;

    // Calculate compute cycles for the tile being processed
    // Total FMAs = Ti × Tj × Tk, throughput = systolic_size²
    uint64_t total_fmas = static_cast<uint64_t>(Ti) * Tj * Tk;
    Cycle compute_cycles = (total_fmas + systolic_throughput - 1) / systolic_throughput;

    // Streamer duration = max(transfer, compute) since they overlap
    Cycle streamer_duration = std::max(transfer_cycles, compute_cycles);

    // Schedule BOTH streamer AND compute fabric
    // ... schedule streamer with streamer_duration
    // ... schedule compute fabric with compute_cycles
}
```

### STR_FEED_COLS (data transfer only)

```cpp
case DMOpcode::STR_FEED_COLS: {
    // In output-stationary dataflow, B cols are broadcast while A rows stream
    // The compute is already counted in STR_FEED_ROWS, so this is just transfer
    // ... schedule streamer with transfer_cycles only (no compute)
}
```

### Key Design Decision: Output-Stationary Dataflow

For output-stationary matmul:
- A rows and B columns feed the systolic array **simultaneously**
- Compute happens once per tile, not twice
- STR_FEED_ROWS counts the compute (A is the streaming input)
- STR_FEED_COLS is data transfer only (B is broadcast)

## Results After Fix

```
Size       Cycles    Ideal    Overhead   Compute Util
64         2048      1024     100%       50.0%
128        13824     8192     69%        59.3%
256        97280     65536    48%        67.4%
512        718848    524288   37%        72.9%
1024       5500928   4194304  31%        76.2%
```

**Improvements:**
- Compute utilization now properly tracked (50-76% depending on size)
- Overhead decreases with matrix size (pipeline amortization)
- Efficiency trends correctly show compute-bound behavior

### Full Benchmark Results

```
Benchmark Suite: matmul_square_sweep
====================================================================================================
                Name          Config       Cycles     GFLOPS     Eff%       AI Bottleneck
----------------------------------------------------------------------------------------------------
              matmul        64x64x64         2048     256.00    37.5%    10.67 L3-memory-bound
              matmul     128x128x128        13824     303.41    29.6%    21.33 compute-bound
              matmul     256x256x256        97280     344.93    33.7%    42.67 compute-bound
              matmul     512x512x512       718848     373.42    36.5%    85.33 compute-bound
              matmul  1024x1024x1024      5500928     390.39    38.1%   170.67 compute-bound
              matmul  2048x2048x2048     42983424     399.69    39.0%   341.33 compute-bound
====================================================================================================
```

## Remaining Overhead Analysis

The ~31-100% overhead comes from:

1. **Pipeline Startup**: DMA + BM transfers before first compute (512 cycles for 64×64)
2. **Pipeline Drain**: STR_DRAIN + DMA_STORE after last compute (512 cycles for 64×64)
3. **No Double-Buffering**: Current schedule doesn't overlap compute with next tile's data movement

For further optimization, the schedule generator should implement:
- Double-buffering at each memory level
- Prefetch of next tile while current tile computes
- This would reduce overhead to near-zero for large matrices

## Debugging Tools

1. `efficiency_diagnostic` test - shows timeline and breakdown
2. `generate_timeline()` - ASCII Gantt chart of resource usage
3. `generate_cycle_report()` - detailed cycle-by-cycle analysis

## References

- `src/isa/concurrent_executor.cpp` - Main executor implementation (fix applied here)
- `include/sw/kpu/isa/concurrent_executor.hpp` - Resource models
- `tests/benchmarks/test_efficiency_diagnostic.cpp` - Diagnostic test
