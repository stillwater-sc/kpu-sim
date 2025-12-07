# Session: Tile Caching and DMA Timing Fixes

**Date:** December 6, 2025

## Session Overview

This session focused on two major areas:
1. Fixing the DMA timing model which was producing unrealistic cycle counts
2. Implementing software tile caching to eliminate redundant DMA transfers

## Key Discussions

### DMA Timing Model Issues

**Problem Identified:**
The user asked why DMA transfers were taking so long (256 cycles for a 256-byte tile when the memory interface should handle it in ~4 cycles).

**Root Causes Found:**
1. **Bandwidth calculation bug**: Code was doing `cycles = bytes / bandwidth_gb_s`, treating GB/s as bytes/cycle
2. **Wrong tile size**: Using `Ti × Tj` instead of actual tile dimensions

**Solution:**
Changed to `cycles = ceil(bytes / bus_width_bytes)` where `bus_width_bytes = 64` (512-bit cache line).

### LPDDR5X Memory Pipeline Analysis

Detailed walkthrough of fetching 64 bytes from LPDDR5X:

| Parameter | Value |
|-----------|-------|
| Data Rate | 8533 MT/s |
| Bus Width | 16 bits (x16) |
| Burst Length | BL16 |
| I/O Clock | 4266 MHz |
| Memory Controller | 250 MHz |

**Pipeline Timing:**
```
LPDDR5X → Memory Controller → DMA Engine → NoC → L3 Tile
  1.9 ns      1 cycle           1 cycle    1 cycle  1 cycle

First access: ~56 ns (includes CAS latency)
Streaming:    ~4 ns per 64B (pipelined)
```

### Tile Caching Architecture

**Problem:** Every loop iteration reloaded tiles even when the same tile was needed again.

For output-stationary matmul:
- A[ti,tk] should be reused across all tj iterations
- B[tk,tj] should be reused across all ti iterations

**Three-Phase Implementation Plan:**

1. **Phase 1 (Implemented):** Software tracking in compiler
   - Track which tiles are resident in L3
   - Skip DMA loads for cache hits
   - Statistics collection

2. **Phase 2 (Future):** ISA extensions
   - `DMA_LOAD_TILE_CACHED` - Load only on miss
   - `TILE_RELEASE` - Decrement refcount
   - `TILE_FENCE` - Memory ordering

3. **Phase 3 (Future):** Hardware modeling
   - Tile Descriptor Table (TDT)
   - LRU eviction with refcount constraints
   - Context isolation for multi-tenant

## Files Changed

### New Files
- `CLAUDE.md` - Claude Code guidance for repository
- `docs/LPDDR5X_MEMORY_PIPELINE.md` - Memory timing documentation
- `docs/TILE_CACHING_ARCHITECTURE.md` - Caching design document
- `include/sw/kpu/isa/tile_cache.hpp` - Tile cache classes
- `src/isa/tile_cache.cpp` - Tile cache implementation

### Modified Files
- `include/sw/kpu/isa/concurrent_executor.hpp` - Added `bus_width_bytes` to `HardwareResource`
- `include/sw/kpu/isa/data_movement_isa.hpp` - Added `TileCacheState`, `enable_tile_caching`
- `src/isa/concurrent_executor.cpp` - Fixed timing calculation, pass bus widths
- `src/isa/data_movement_isa.cpp` - Added cache-aware load functions
- `src/isa/CMakeLists.txt` - Added tile_cache.cpp
- `examples/basic/data_movement_isa_matmul.cpp` - Added Example 6 tile caching demo

## Results

### DMA Timing Fix
| Metric | Before | After |
|--------|--------|-------|
| 4KB tile transfer | 256 cycles | 64 cycles |
| 1KB tile transfer | 64 cycles | 16 cycles |

### Tile Caching (128×128×128 matmul)
| Metric | Without Cache | With Cache | Improvement |
|--------|---------------|------------|-------------|
| DMA operations | 144 | 48 | 67% fewer |
| External traffic | 576 KB | 192 KB | 67% less |
| Reuse factor | 3.00× | 1.00× | Optimal |
| Arithmetic intensity | 7.1 F/B | 21.3 F/B | 3× better |
| Cache hit rate | - | 75% | - |

## Technical Insights

### Tile Reuse Patterns in Output-Stationary Dataflow

```
Loop: for ti, for tj, for tk

A tiles indexed by (ti, tk):
  - Same A[ti,tk] used for all tj values
  - Reuse factor: N/Tj times

B tiles indexed by (tk, tj):
  - Same B[tk,tj] used for all ti values
  - Reuse factor: M/Ti times
```

### Correct Bandwidth Math

For 64-byte bus at 250 MHz:
```
Bandwidth = 64 bytes × 250 MHz = 16 GB/s
Cycles per 64B = 1 cycle
Cycles per 256B tile = ceil(256/64) = 4 cycles
```

## Next Steps

1. **Phase 2: ISA Extensions**
   - Add `DMA_LOAD_TILE_CACHED` opcode
   - Implement refcount management
   - Add `TILE_FENCE` for memory ordering

2. **L3 Capacity Constraints**
   - Currently assuming infinite L3 cache
   - Need eviction when cache is full
   - LRU policy with refcount=0 constraint

3. **Pipeline Granularity**
   - Current transfers are full tiles
   - Consider sub-tile streaming for better pipelining
   - Double/triple buffering integration

## Code Snippets

### Fixed Timing Calculation
```cpp
// Before (wrong):
Cycle cycles = static_cast<Cycle>(bytes / bandwidth_gb_s);

// After (correct):
Cycle cycles = (bytes + bus_width_bytes - 1) / bus_width_bytes;
```

### Cache-Aware Tile Loading
```cpp
bool try_emit_load_a_tile(DMProgram& prog, TileCoord tile, BufferSlot buf) {
    if (config_.enable_tile_caching &&
        tile_cache_.is_resident(MatrixID::A, tile.ti, 0, tile.tk)) {
        // Cache hit - no DMA needed
        tile_cache_.hits++;
        tile_cache_.bytes_saved += tile_bytes;
        return false;
    }
    // Cache miss - emit DMA
    emit_load_a_tile(prog, tile, buf);
    tile_cache_.mark_resident(MatrixID::A, tile.ti, 0, tile.tk, tile_bytes);
    return true;
}
```

## Verification

All 34 tests pass:
```
100% tests passed, 0 tests failed out of 34
Total Test time (real) =   4.49 sec
```

Demo output (Example 6) shows expected behavior with 75% cache hit rate.
