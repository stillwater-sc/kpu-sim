# Session Log: OFG Visualization Dataflow Fixes

**Date:** 2026-01-14
**Duration:** ~2 hours
**Focus:** Fix visualization issues in OFG execution animation - NaN statistics, incomplete loop progress, and missing event log entries

## Summary

Fixed multiple issues in the OFG (Operand Flow Graph) execution animation visualization related to credit-based dataflow semantics. The visualization was showing NaN% for execution statistics, loop progress appeared incomplete at trace end, and the event log was missing BlockMover, Streamer, and tile events. All issues traced to field name mismatches and missing `logEvent()` calls.

## Context

The KPU visualization tool (`tools/visualization/ofg_execution_animation.html`) animates tiled matrix multiplication execution showing data flow through the memory hierarchy: Host → L3 → L2 → L1 → Compute. Previous work had updated the execution model from cache semantics to credit-based dataflow (see `docs/kpu-execution-model.md`), but the visualization code had residual inconsistencies.

## Issues Fixed

### Issue 1: NaN% Progress Statistics

**Symptom:** Execution statistics panel showed "NaN%" for progress.

**Root Cause:** Field name mismatch between trace format and display code:

| Trace Field | Display Expected |
|-------------|------------------|
| `dma_pushes`, `dma_pulls` | `dma_loads`, `dma_stores` |
| `computes` | `matmuls` |

**Fix:** Updated display code to use fallback lookups supporting both naming conventions:
```javascript
const dmaTotal = (t.dma_pushes || t.dma_loads || 0) + (t.dma_pulls || t.dma_stores || 0);
const computeTotal = t.computes || t.matmuls || 0;
```

### Issue 2: Loop Progress Appears Incomplete

**Symptom:** At end of trace, loop progress showed "i: 1/2, j: 1/2, k: 2/3" instead of "2/2, 2/2, 3/3".

**Root Cause:** Display showed zero-indexed loop indices instead of completion count.

**Fix:** Changed text display from zero-indexed to one-indexed:
```javascript
// Before: ${loopState.i}/${m}     -> showed "1/2"
// After:  ${loopState.i + 1}/${m} -> shows "2/2"
```

### Issue 3: Missing Event Log Entries

**Symptom:** Event log only showed DMA, L3 credits, and MATMUL events. No BlockMover, Streamer, TILE_READY, or TILE_COMPLETE events.

**Root Cause:** Event handlers updated internal state but didn't call `logEvent()`.

**Fix:** Added `logEvent()` calls to all event handlers:

| Event Type | Logging Added |
|-----------|---------------|
| `BM_PUSH`, `BM_PULL` | First 10 events |
| `PUSH_TO_L2`, `PULL_FROM_L2` | First 10 events |
| `STR_FEED_A`, `STR_FEED_B` | First 10 events |
| `FEED_WEST`, `FEED_NORTH` | First 10 events |
| `STR_DRAIN`, `DRAIN` | First 10 events |
| `TILE_READY` | First 10 events |
| `TILE_COMPLETE` | All events (milestones) |

## Files Modified

| File | Changes |
|------|---------|
| `tools/visualization/ofg_execution_animation.html` | Fixed stats field lookups, loop progress display, added event logging |

## Additional Improvements (from earlier in session)

1. **Updated embedded demo trace** - Changed from 4×4×2 tiles (32 matmul ops) to 2×2×3 tiles (12 matmul ops) matching the `--tiny` CLI option for better educational value

2. **Visual separation of executor OFG states** - Added labels and dashed separators to distinguish "Buffer Occupancy" from "Executor OFG" displays in each memory level section

3. **OFG node terminology** - Already updated from cache-semantic "FETCH" to dataflow terminology:
   - **W** = Wait for token/credit
   - **P** = Push data downstream
   - **T** = Emit TILE_READY token
   - **C** = Wait for buffer credit
   - **F** = Feed tile to L1
   - **D** = Drain accumulator
   - **↑** = Return credit upstream

## Verification

The tiny trace example now works correctly:
```bash
./build/examples/behavioral/tiled_matmul_trace --tiny
```
- Generates D[32,32] = A[32,48] × B[48,32]
- 12 matmul operations
- ~2856 cycles
- Shows buffer reuse patterns (A loaded once per i,k; B loaded m_tiles times)

## Key Dataflow Concepts Reinforced

1. **NO CACHE** - L3, L2, L1 are buffers, not caches
2. **Credits UP, Data DOWN** - Producer waits for credit before pushing
3. **TILE_READY** - Data token signals arrival at downstream buffer
4. **BUFFER_AVAILABLE** - Credit token signals buffer freed upstream

## Related Documentation

- `docs/kpu-execution-model.md` - Authoritative reference for credit-based dataflow
- `examples/behavioral/tiled_matmul_trace.cpp` - Trace generator with --tiny/--small/--medium/--large options
