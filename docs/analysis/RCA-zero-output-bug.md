# Root Cause Analysis: Zero Output Bug

## Problem Statement

Both `model_host_t100` and `model_host_t100_autonomous` produce all-zero outputs for MLP matrix multiplication, despite successful orchestration through the complete pipeline (Memory → L3 → L2 → L1 → Compute → Result readback).

## Investigation Summary

### Component Analysis

#### ✅ BlockMover (L3 → L2)
**Status**: WORKING CORRECTLY

**Evidence**:
- `block_mover.cpp:58-70`: Properly reads from L3, applies transform, writes to L2
- Uses `L3Tile::read_block()` and `L2Bank::write_block()` which correctly handle 2D block transfers
- Identity transform correctly copies data

#### ✅ L3Tile & L2Bank Memory
**Status**: WORKING CORRECTLY

**Evidence**:
- `l3_tile.cpp:30-44`: `read_block()` correctly reads rows with proper stride
- `l2_bank.cpp:38-52`: `read_block()` and `write_block()` work symmetrically
- Both use `std::memcpy` for actual data movement

#### ❌ Streamer (L2 → L1)
**Status**: **CRITICAL BUG FOUND**

**Root Cause Location**: `streamer.cpp:136-180` (`stream_row_l2_to_l1`)

**The Bug**:
```cpp
// Line 159 in streamer.cpp
Address l1_addr = config.l1_base_addr + i * config.element_size;
```

**Problem**: The Streamer writes each fabric-sized chunk to the **SAME L1 addresses** (0, 1, 2, ..., fabric_size-1), **overwriting** the previous chunk on each call!

**Expected Behavior**: Accumulate the entire matrix in L1 by writing to different addresses:
```cpp
// Should be:
Size elements_written_so_far = current_stream->current_row * config.matrix_width + current_stream->current_col;
Address l1_addr = config.l1_base_addr + (elements_written_so_far + i) * config.element_size;
```

#### ✅ SystolicArray (Compute)
**Status**: WORKING CORRECTLY (but receives corrupt data)

**Evidence**:
- `systolic_array.cpp:373-404`: `perform_direct_matrix_multiply()` correctly:
  - Reads full matrices A and B from L1 (lines 386, 389)
  - Performs C = A × B computation (lines 392-400)
  - Writes result back to L1 (line 403)

**Why It Produces Zeros**:
The SystolicArray tries to read the **entire matrix** from L1, but the Streamer has only written **the last fabric-sized chunk** (overwriting all previous chunks). The rest of L1 contains zeros (initial state), so the computation produces all zeros.

## Data Flow Analysis

### What Actually Happens

```
Memory Bank [has full data]
    ↓ (manual staging)
L3 Tile [has full data]
    ↓ (BlockMover - works correctly)
L2 Bank [has full data]
    ↓ (Streamer - BUG HERE!)
L1/Scratchpad [ONLY HAS LAST CHUNK, rest is zeros]
    ↓ (SystolicArray reads full matrix)
Compute [multiplies mostly zeros] → Output = all zeros
```

### Streaming Bug Detail

For a 4×8 matrix with fabric_size=16:

**Call 1**: Stream elements [0-15] → writes to L1[0-15] ✓
**Call 2**: Stream elements [16-31] → writes to L1[0-15] ❌ (OVERWRITES!)

**Result**: L1 only contains elements [16-31], everything else is zero.

When SystolicArray reads the full 32-element matrix:
- L1[0-15] = elements [16-31] (wrong position!)
- L1[16-31] = zeros (never written)

## Root Cause

**The Streamer implements "streaming to systolic array edges" but the SystolicArray expects "full matrices in L1".**

Two architectural mismatches:

### 1. Address Calculation Bug
Streamer uses fixed base offset instead of accumulating position:
```cpp
// WRONG (current):
Address l1_addr = config.l1_base_addr + i * config.element_size;

// RIGHT (should be):
Size offset = (current_row * matrix_width + current_col + i);
Address l1_addr = config.l1_base_addr + offset * config.element_size;
```

### 2. Architectural Mismatch
- **Streamer design**: Stream fabric-sized chunks to SA edges each cycle
- **SystolicArray implementation**: Read full matrices from L1 once, compute directly

## Fix Options

### Option 1: Fix Streamer to Accumulate Full Matrix (RECOMMENDED)
Make Streamer write the entire matrix to L1 by fixing address calculation.

**Pros**:
- Simple fix (one line change)
- Works with current SystolicArray implementation
- Good for initial functionality validation

**Cons**:
- Doesn't model true systolic streaming
- Requires full matrices in L1 (memory inefficient)

### Option 2: Implement True Systolic Streaming
Rewrite SystolicArray to consume streams cycle-by-cycle from Streamer.

**Pros**:
- Realistic hardware model
- Memory efficient (only fabric-sized chunks in L1)
- Enables pipelining and double-buffering

**Cons**:
- Major refactoring of SystolicArray
- Complex cycle-accurate streaming logic
- Requires careful synchronization

## Recommendation

**Phase 1**: Fix Option 1 to get functionality working
**Phase 2**: Implement Option 2 for realistic tiled/streamed execution

## Impact on Autonomous Model

**Good News**: The autonomous orchestration is **working perfectly**. The bug is in the component implementations, not the coordination logic.

**Evidence**:
- All signals fire in correct order
- All callbacks execute
- Pipeline completes in 13 cycles
- Zero outputs occur in BOTH GOD mode and autonomous mode (proving it's not an orchestration bug)

## Next Steps

1. **Immediate**: Fix Streamer address calculation (Option 1)
2. **Validate**: Run simple 4×8×4 matmul to verify correctness
3. **Scale Up**: Implement 256×256 tiled matmul with proper L1 management
4. **Future**: Design true systolic streaming architecture (Option 2)
