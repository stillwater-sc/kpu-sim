# Autonomous Execution Model - Timing Fixes

## Summary

Fixed the autonomous execution model (host_t100_autonomous.cpp) by implementing proper multi-cycle timing for BlockMover and SystolicArray components. The model now correctly executes concurrent hardware operations with realistic timing.

## Root Cause Analysis

### Initial Problem
Both GOD mode (host_t100.cpp) and autonomous mode (host_t100_autonomous.cpp) were producing zero outputs for matrix multiplication operations.

### Investigation
1. **Streamer component**: ✅ Already had proper multi-cycle timing
2. **BlockMover component**: ❌ Completed transfers instantly in one cycle
3. **SystolicArray component**: ❌ Completed computation instantly in one cycle

### Why test_debug_dataflow.cpp worked
The debug test called `run_until_idle()` after EVERY stage, which masked the instant-completion issue by forcing synchronization points.

### Why host_t100_autonomous.cpp failed
The orchestrator launched operations asynchronously with signal dependencies. When operations completed instantly:
- Callbacks fired before subsequent cycles could properly synchronize
- Data was being processed but timing issues caused incorrect orchestration
- The L2→L3 readback path used incorrect API (BlockMover only supports L3→L2)

## Fixes Applied

### 1. BlockMover Multi-Cycle Timing Model

**File**: `src/components/datamovement/block_mover.cpp`

**Changes**:
- Added `cycles_remaining` and `transfer_buffer` state variables
- Implemented timing model: 1 cycle per 64 bytes (cache line), minimum 1 cycle
- Operations now span multiple cycles:
  - Cycle 0: Read from L3, apply transform, calculate required cycles
  - Cycles 1..N-1: Transfer in progress
  - Cycle N: Write to L2, fire callback

**Code**:
```cpp
// Timing model: 1 cycle per 64 bytes (cache line), minimum 1 cycle
constexpr Size CACHE_LINE_SIZE = 64;
cycles_remaining = std::max<Cycle>(1, (block_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE);
```

### 2. SystolicArray Multi-Cycle Timing Model

**File**: `src/components/compute/systolic_array.cpp`

**Changes**:
- Fixed `update()` method to use `estimate_cycles()` instead of instant completion
- Computation now properly waits for required cycles before completing:
  - Cycle formula: `k + max(m,n) + max(rows,cols)`
  - For 4×4×8 matmul on 16×16 array: 8 + 4 + 16 = 28 cycles

**Code**:
```cpp
// Calculate required cycles for this matmul operation
Cycle required_cycles = estimate_cycles(current_op.m, current_op.n, current_op.k);

// Check if computation has completed
if (cycles_elapsed >= required_cycles) {
    // Perform the actual matrix multiplication
    perform_direct_matrix_multiply(scratchpads);

    // Call completion callback if provided
    if (current_op.completion_callback) {
        current_op.completion_callback();
    }

    is_computing = false;
    return true;
}
```

### 3. BlockMover API Limitation Fix

**File**: `models/kpu/host_t100_autonomous.cpp`

**Problem**: BlockMover only supports L3→L2 direction, not L2→L3 reverse
**Solution**: Used direct read/write for L2→L3 in readback path

**Code**:
```cpp
orch.await(STREAM_OUTPUT_DONE, [&]() {
    const Address l2_output_addr = 0x4000;
    const Address l3_output_addr = 0x8000;
    // BlockMover only supports L3→L2, so do manual L2→L3 transfer
    std::vector<uint8_t> temp(batch_size * output_dim * sizeof(float));
    kpu->read_l2_bank(l2_bank_id, l2_output_addr, temp.data(), temp.size());
    kpu->write_l3_tile(l3_tile_id, l3_output_addr, temp.data(), temp.size());
    orch.signal(BLOCK_OUTPUT_DONE);
}, "Manual: L2->L3 (output)");
```

## Timing Results

### Before Fixes
- BlockMover: Instant (0 cycles)
- SystolicArray: Instant (0 cycles)
- Total execution: ~13 cycles (incorrect)

### After Fixes
- BlockMover: 1 cycle per transfer (realistic)
- SystolicArray: 28 cycles for 4×8×4 matmul (correct)
- Streamer: 3-4 cycles (already correct)
- Total execution: 42 cycles (realistic)

## Execution Trace (After Fixes)

```
Cycle  1: Launch BlockMover L3→L2 (input)
Cycle  1: Launch BlockMover L3→L2 (weights)
Cycle  2: BlockMover input complete → signal
Cycle  3: Launch Streamer L2→L1 (input)
Cycle  4: BlockMover weights complete → signal
Cycle  5: Launch Streamer L2→L1 (weights)
Cycle  6: Streamer input complete → signal
Cycle  8: Streamer weights complete → signal
Cycle  9: Launch SystolicArray matmul
Cycle 37: SystolicArray complete → signal (28 cycle computation)
Cycle 38: Add bias + Launch Streamer L1→L2 (output)
Cycle 41: Streamer output complete → signal
Cycle 42: Manual L2→L3 transfer + L3→Memory + Memory→Host
```

## Validation

### Test Results
- **test_debug_dataflow.cpp**: ✅ PASSED (2×2 matmul)
- **model_host_t100_autonomous**: ✅ PASSED (4×8×4 MLP layer)
- Expected outputs: `[2.36, 2.22, 1.88, 2.14, 2.56, ...]`
- Actual outputs: Exact match!

## Future Work

### Near Term
1. Implement bidirectional BlockMover (L3↔L2 in both directions)
2. Add proper DMA engine support for Memory↔L3 transfers
3. Implement 256×256 matmul with 16×16 tiles

### Long Term
1. Add more realistic timing models (bandwidth-based, not just cycle counts)
2. Implement true systolic dataflow (currently using direct matmul)
3. Add contention modeling for shared resources
4. Implement double buffering and pipelining optimizations

## Key Learnings

1. **Timing models matter**: Instant completion breaks asynchronous orchestration
2. **API design matters**: Unidirectional vs bidirectional data movement
3. **Test strategies**: `run_until_idle()` after each stage masks timing bugs
4. **Debugging approach**: Verbose logging + incremental stage validation
5. **Realistic hardware**: Multi-cycle operations enable true concurrent execution

## Related Files

- `src/components/datamovement/block_mover.cpp` - BlockMover implementation
- `src/components/compute/systolic_array.cpp` - SystolicArray implementation
- `models/kpu/host_t100_autonomous.cpp` - Autonomous execution model
- `models/kpu/autonomous_orchestrator.hpp` - Signal-based orchestration
- `models/kpu/test_debug_dataflow.cpp` - Debug validation test
- `docs/autonomous-kpu-design.md` - Original design document
- `docs/RCA-zero-output-bug.md` - Initial root cause analysis

## Conclusion

The autonomous execution model now correctly models concurrent hardware behavior with realistic multi-cycle timing. All components operate independently with signal-based synchronization, eliminating the need for centralized `run_until_idle()` calls between stages. This enables future work on advanced features like pipelining, double buffering, and latency hiding.
