# Session Log: GDDR6 Trace Fix and Bandwidth Metrics

**Date:** 2026-01-07
**Duration:** ~2 hours
**Focus:** Fix GDDR6 multi-bank trace generation, add memory characterization documentation

## Summary

Fixed a bug where GDDR6 traces only showed 8 banks instead of 16, and created comprehensive latency/bandwidth characterization documentation for both LPDDR5 and GDDR6 memory controllers.

## Context

The user noticed that the GDDR6 multi_dma trace only showed banks 0-7 in the visualization, even though GDDR6 has 16 banks. Investigation revealed the issue was not in the memory controller but in the pattern's trace export section.

## Bug Investigation

### Initial Hypothesis
The first hypothesis was that `bank_bits` was incorrectly set to 3 (8 banks) instead of 4 (16 banks) in the GDDR6 configuration.

### Investigation Steps

1. **Verified address encoding/decoding** - Created test program confirming `make_address()` and `decode_address()` correctly handle banks 0-15 with 4 bank bits

2. **Checked GDDR6 configuration** - Confirmed `bank_bits = 4` is correctly set in:
   - `include/sw/kpu/components/gddr6_memory_controller.hpp` (line 365)
   - `patterns/memory/gddr6/common/gddr6_configs.hpp`

3. **Found root cause** - The pattern's trace export section submitted 128 requests (16 banks × 8 lines) before calling `run_until_complete()`, but the queue depth was only 64. Requests for banks 8-15 were silently rejected.

### Root Cause
```cpp
// Queue depth = 64, but pattern submits 128 requests
for (int dma = 0; dma < 16; ++dma) {  // 16 DMAs
    for (int line = 0; line < 8; ++line) {  // 8 lines each
        harness.submit_read(...);  // 128 total, but only first 64 accepted
    }
}
harness.run_until_complete();  // Only 8 banks traced
```

## Fixes Applied

### 1. GDDR6 multi_dma pattern (`patterns/memory/gddr6/complex/multi_dma.cpp`)

```cpp
// Before:
GDDR6Harness harness(gddr6_16000_config());

// After:
auto trace_config = gddr6_16000_config();
trace_config.queue_depth = 256;  // Fit all 128 requests
GDDR6Harness harness(trace_config);
```

**Result:** Trace now shows 144 events across all 16 banks (was 72 events for 8 banks)

### 2. LPDDR5 multi_dma pattern (`patterns/memory/lpddr5/complex/multi_dma.cpp`)

Same fix applied - queue depth increased from 64 to 256.

**Result:** Trace now shows 136 events across all 8 banks (was ~68 events for 4 banks)

## Documentation Created

### 1. Memory Characterization Document (`docs/memory-characterization.md`)

Comprehensive latency and bandwidth characterization:

**LPDDR5-6400:**
- Timing parameters (tRCD=14, tCL=16, tRP=14, etc.)
- Latency: Page hit ~243 cycles, Page conflict ~305 cycles
- Bandwidth scaling: 1-8 banks → 2.20-2.80 bytes/cycle
- STREAM: Copy 19.04, Triad 26.67 bytes/cycle
- Multi-DMA: 4-16 engines → 44.73-89.47 bytes/cycle

**GDDR6-16000:**
- Timing parameters (tRCDRD=18, tRL=20, tRP=18, etc.)
- Latency: Page hit ~172 cycles, Page conflict ~298 cycles
- Bandwidth scaling: 1-16 banks → 1.59-2.56 bytes/cycle
- STREAM: Copy 38.24, Triad 59.08 bytes/cycle
- Multi-DMA: 4-32 engines → 40.93-81.87 bytes/cycle

**Key Findings:**
- GDDR6 provides ~2x STREAM bandwidth over LPDDR5
- LPDDR5 has lower page miss latency (38 vs 42 cycles for page empty)
- Full page utilization (128 cache lines/page) yields 2.5x bandwidth improvement

### 2. Updated Traces README (`traces/README.md`)

- Complete directory structure for both LPDDR5 and GDDR6
- Memory technology specifications
- Pattern category descriptions (Levels 1-7)
- Quick start commands
- Visualization tool reference
- Chrome Trace Format documentation

## Files Changed

| File | Change |
|------|--------|
| `patterns/memory/gddr6/complex/multi_dma.cpp` | Increased queue_depth to 256 for trace export |
| `patterns/memory/lpddr5/complex/multi_dma.cpp` | Increased queue_depth to 256 for trace export |
| `docs/memory-characterization.md` | New - latency/bandwidth characterization |
| `traces/README.md` | Updated - complete directory structure |

## Verification

```bash
# Verify GDDR6 trace now has all 16 banks
$ grep -o '"tid": [0-9]*' traces/memory/gddr6/complex/multi_dma_trace.json | sort -u
"tid": 0  ... "tid": 15  # All 16 banks present

# Verify trace event count
$ ./build/patterns/memory/gddr6/gddr6_multi_dma
Events: 144  # Was 72
```

## Lessons Learned

1. **Silent request rejection** - When `submit_read()` returns `std::nullopt`, the pattern continues without warning. Consider adding debug output or assertions when requests are rejected unexpectedly.

2. **Queue depth sizing** - Pattern trace export sections need queue depth sized for the total number of requests submitted before simulation runs.

3. **Test verification** - The pattern tests passed because they only tested functional correctness (operations completed), not whether all expected requests were traced.

## Next Steps

- Consider adding queue depth validation or warnings when patterns submit more requests than queue can hold
- Add `--verbose` flag to patterns that logs rejected requests
- Consider dynamic queue sizing based on pattern requirements
