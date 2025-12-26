# KPU Simulator Efficiency Bug Analysis

## Problem Statement

The benchmark suite shows unexpectedly low efficiency for matmul operations:

```
Size       Cycles    Ideal    Overhead   Compute Util
64         1280      1024     25%        0.0%
128        7680      8192     -6%        0.0%
256        48128     65536    -27%       0.0%
```

**Critical Issue**: Compute utilization is 0.0% for all sizes!

## Root Cause

The `ConcurrentExecutor` models data movement time but **completely ignores compute time**.

### Current (Incorrect) Model

The `HardwareResource::schedule_op` calculates cycles as:
```cpp
Cycle cycles = (bytes + bus_width_bytes - 1) / bus_width_bytes;
```

This only measures transfer time:
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

## Correct Model

The streamer operation time should be the **maximum of**:
1. Data transfer time (current model)
2. Systolic compute time (missing!)

### Systolic Compute Time Formula

For output-stationary matmul with tile dimensions Ti×Tj×Tk:

```
compute_cycles = ceil(Ti × Tj × Tk × 2 / systolic_ops_per_cycle)

where:
  systolic_ops_per_cycle = systolic_size × systolic_size × 2
                         = 16 × 16 × 2 = 512 for 16×16 array
```

For a 64×64×64 tile:
```
compute_cycles = ceil(64 × 64 × 64 × 2 / 512) = ceil(1024) = 1024
```

### Pipeline Considerations

For a properly pipelined systolic array:
1. **Startup**: Ti + Tj - 2 cycles to fill the pipeline
2. **Steady state**: Ti × Tj × (Tk/systolic_size) cycles of full utilization
3. **Drain**: Ti + Tj - 2 cycles to drain results

With tile reuse and double buffering, the pipeline should be constantly fed.

## Proposed Fix

Modify streamer scheduling in `ConcurrentExecutor::schedule_instruction`:

```cpp
case DMOpcode::STR_FEED_ROWS:
case DMOpcode::STR_FEED_COLS: {
    // Current: just transfer cycles
    Cycle transfer_cycles = (bytes + bus_width - 1) / bus_width;

    // NEW: compute cycles for the tile being processed
    Size tile_flops = program_.Ti * program_.Tj * program_.Tk * 2;
    Cycle compute_cycles = (tile_flops + systolic_ops_per_cycle - 1) / systolic_ops_per_cycle;

    // Use the maximum (compute-bound or memory-bound)
    Cycle duration = std::max(transfer_cycles, compute_cycles);

    // Also track compute fabric utilization
    schedule_compute_fabric(earliest, compute_cycles, ...);
}
```

## Expected Results After Fix

| Size | Current Cycles | Fixed Cycles | Expected Efficiency |
|------|---------------|--------------|---------------------|
| 64   | 1280          | ~1024+overhead | >95% |
| 128  | 7680          | ~8192+overhead | >95% |
| 256  | 48128         | ~65536+overhead | >95% |

With proper pipelining and double-buffering, efficiency should approach 100% for larger matrices.

## Debugging Tools Created

1. `efficiency_diagnostic` test - shows timeline and breakdown
2. `generate_timeline()` - ASCII Gantt chart of resource usage
3. `generate_cycle_report()` - detailed cycle-by-cycle analysis

## References

- `src/isa/concurrent_executor.cpp` - Main executor implementation
- `include/sw/kpu/isa/concurrent_executor.hpp` - Resource models
- `tests/benchmarks/test_efficiency_diagnostic.cpp` - Diagnostic test
