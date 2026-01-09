# Session Log: HBM2 and HBM3 Memory Controllers

**Date:** 2026-01-08
**Duration:** ~4 hours
**Focus:** Implement HBM2 and HBM3 cycle-accurate memory controllers with full pattern test suites

## Summary

Completed implementation of HBM2 and HBM3 memory controllers following the established LPDDR5/GDDR6 architecture patterns. Created 18 pattern tests (9 per technology) covering single-bank, two-bank, pseudo-channel, multi-channel, and bandwidth scenarios. Fixed invariant checking bugs by aligning with LPDDR5's semantic approach. Updated memory characterization documentation with comprehensive HBM comparisons.

## Context

The KPU simulator needed HBM memory controller support for modeling high-bandwidth scenarios. HBM2 (256 GB/s) and HBM3 (716.8 GB/s) provide significantly higher bandwidth than LPDDR5 and GDDR6, making them essential for data center and HPC workloads.

## Implementation Overview

### HBM2 Architecture
- 8 channels per stack (128-bit each)
- 2 pseudo-channels per channel (64-bit each)
- 16 banks per pseudo-channel
- **256 total banks**, 256 GB/s peak bandwidth
- 1.0 GHz clock (HBM2-2000)

### HBM3 Architecture
- 16 channels per stack (64-bit each)
- 2 pseudo-channels per channel (32-bit each)
- 16 banks per pseudo-channel
- **512 total banks**, 716.8 GB/s peak bandwidth
- 2.8 GHz clock (HBM3-5600)

## Files Created

### Memory Controllers
| File | Lines | Description |
|------|-------|-------------|
| `include/sw/kpu/components/hbm2_memory_controller.hpp` | ~600 | HBM2 controller interface with timing params |
| `src/components/memory/hbm2_memory_controller.cpp` | ~1200 | HBM2 implementation |
| `include/sw/kpu/components/hbm3_memory_controller.hpp` | ~600 | HBM3 controller interface with timing params |
| `src/components/memory/hbm3_memory_controller.cpp` | ~1200 | HBM3 implementation |

### Pattern Infrastructure
| File | Description |
|------|-------------|
| `patterns/memory/hbm2/common/hbm2_harness.hpp` | HBM2 test harness with tracing |
| `patterns/memory/hbm2/common/hbm2_configs.hpp` | HBM2 configuration presets |
| `patterns/memory/hbm3/common/hbm3_harness.hpp` | HBM3 test harness with tracing |
| `patterns/memory/hbm3/common/hbm3_configs.hpp` | HBM3 configuration presets |

### Pattern Tests (18 total)

**HBM2 Patterns:**
- `single-bank/hbm2_page_hits.cpp` - Same row accesses
- `single-bank/hbm2_page_conflicts.cpp` - Different row accesses
- `single-bank/hbm2_mixed_rw.cpp` - Read/write turnarounds
- `two-bank/hbm2_same_group.cpp` - tRRD_L timing
- `two-bank/hbm2_diff_groups.cpp` - tRRD_S timing
- `pseudo-channel/hbm2_dual_pc.cpp` - PC parallelism
- `multi-channel/hbm2_four_channel.cpp` - 4-channel parallel
- `multi-channel/hbm2_eight_channel.cpp` - All 8 channels
- `bandwidth/hbm2_max_bandwidth.cpp` - Saturation test

**HBM3 Patterns:**
- `single-bank/hbm3_page_hits.cpp` - Same row accesses
- `single-bank/hbm3_page_conflicts.cpp` - Different row accesses
- `single-bank/hbm3_mixed_rw.cpp` - Read/write turnarounds
- `two-bank/hbm3_same_group.cpp` - tRRD_L timing
- `two-bank/hbm3_diff_groups.cpp` - tRRD_S timing
- `pseudo-channel/hbm3_dual_pc.cpp` - PC parallelism
- `multi-channel/hbm3_eight_channel.cpp` - 8-channel parallel
- `multi-channel/hbm3_sixteen_channel.cpp` - All 16 channels
- `bandwidth/hbm3_max_bandwidth.cpp` - Saturation test

## Bug Fixes

### 1. Invariant Checking Bug (Critical)

**Symptom:** Tests failed with "HBM2-INV-001: Bank state_until is in the past"

**Initial Wrong Fix:** Reordered `tick()` to run `update_bank_states()` before `check_all_invariants()`

**User Feedback:**
> "Your response to reorder and first do bank updates before checking invariants to try to resolve the cycle bug was not advisable. We need to get back to the original tick() state machine where we start off with checking the invariants."

**Root Cause Analysis:**
- LPDDR5/GDDR6 don't use generic "state_until in past" checks
- READING/WRITING states use `burst_end` not `state_until`
- Generic check was invalid for these states

**Correct Fix:** Changed to semantic invariant checking (like LPDDR5):

```cpp
void HBM2MemoryController::check_bank_invariants(uint8_t ch, uint8_t pc, uint8_t bank) {
    const auto& b = channels_[channel].pseudo_channels[pc].banks[bank];
    const auto& timing = hbm2_config_.timing;

    // INV-BANK-2: tRCD check
    if (b.state == hbm2::BankState::ACTIVE && current_cycle_ < b.state_until) {
        report_violation("HBM2-INV-001", "Bank transitioned to ACTIVE before tRCD elapsed", ...);
    }

    // INV-BANK-3: tRAS check
    if (b.state == hbm2::BankState::PRECHARGING) {
        if (b.last_activate > 0 && current_cycle_ < b.last_activate + timing.tRAS) {
            report_violation("HBM2-INV-002", "PRECHARGE issued before tRAS elapsed", ...);
        }
    }
    // ... tWR, tRTP checks
}
```

### 2. Page Conflicts Test Expectations

**Symptom:** page_conflicts test expected 1 page_empty, got 4

**Root Cause:** Page conflicts involve precharge (page_conflicts++) followed by activate (page_empty++)

**Fix:** Updated expectations:
- N accesses to different rows = N page_empty + (N-1) page_conflicts
- 4 accesses = 4 page_empty + 3 page_conflicts

### 3. Max Bandwidth Queue Overflow

**Symptom:** Queue depth (64) smaller than total requests (512+)

**Fix:** Reduced banks per PC to 4 and added drain loop:
```cpp
while (!harness.submit_read(addr)) {
    harness.run_cycles(1);  // Drain queue if full
}
```

## Documentation Updates

Updated `docs/analysis/memory-characterization.md` with:

1. **Technology Summary Table** - All 4 technologies (LPDDR5, GDDR6, HBM2, HBM3)
2. **HBM2-2000 Full Characterization**
   - Timing parameters (tRCD=12, tCL=18, tRP=14, etc.)
   - Latency analysis (page hit: ~18 cycles)
   - Bandwidth: 256 GB/s theoretical
3. **HBM3-5600 Full Characterization**
   - Timing parameters (tRCD=8, tCL=8, tRP=8, etc.)
   - Latency analysis (page hit: ~8 cycles)
   - Bandwidth: 716.8 GB/s theoretical
4. **HBM Evolution: HBM2 to HBM3 to HBM4**
   - Generation comparison tables
   - Architecture progression
5. **Comparative Analysis**
   - LPDDR5 vs HBM2
   - HBM2 vs HBM3
   - All technologies comparison
6. **Technology Selection Guide**

## Test Results

All 18 patterns pass:
```
HBM2 Patterns:
  PASS: hbm2_page_hits
  PASS: hbm2_page_conflicts
  PASS: hbm2_mixed_rw
  PASS: hbm2_same_group
  PASS: hbm2_diff_groups
  PASS: hbm2_dual_pc
  PASS: hbm2_four_channel
  PASS: hbm2_eight_channel
  PASS: hbm2_max_bandwidth

HBM3 Patterns:
  PASS: hbm3_page_hits
  PASS: hbm3_page_conflicts
  PASS: hbm3_mixed_rw
  PASS: hbm3_same_group
  PASS: hbm3_diff_groups
  PASS: hbm3_dual_pc
  PASS: hbm3_eight_channel
  PASS: hbm3_sixteen_channel
  PASS: hbm3_max_bandwidth
```

## Lessons Learned

1. **Follow established patterns** - The tick() state machine order (check_invariants first) was intentional. Changing it to mask a bug creates technical debt.

2. **Semantic vs generic invariants** - Generic "state_until in past" checks don't work for all states. READING/WRITING states track completion via `burst_end`. Semantic checks (tRCD, tRAS, tWR, tRTP) are more meaningful and robust.

3. **Page conflict mechanics** - A page conflict is precharge + activate, incrementing both `page_conflicts` and `page_empty` counters.

4. **Queue management** - When submitting many requests, either increase queue depth or add drain loops to avoid silent rejection.

## Files Modified

| File | Change |
|------|--------|
| `include/sw/kpu/fidelity/simulation_fidelity.hpp` | Added HBM2, HBM2E to MemoryTechnology enum |
| `include/sw/trace/trace_entry.hpp` | Added HBM2/HBM3 ComponentTypes |
| `src/components/memory/memory_controller_factory.cpp` | Wire up HBM controllers |
| `src/components/memory/CMakeLists.txt` | Add HBM source files |
| `patterns/CMakeLists.txt` | Add HBM pattern targets |
| `docs/analysis/memory-characterization.md` | Comprehensive HBM documentation |

## Session 2: Trace Script Fix, Test Fix, and Visualization Improvements

### Deleted Traces Root Cause Analysis

**Issue:** Running `generate_all_traces.sh --clean` deleted GDDR6 and LPDDR5 traces that were not regenerated.

**Deleted Files:**
- `traces/memory/lpddr5/bandwidth/max_bandwidth_trace.json`
- `traces/memory/lpddr5/bandwidth/page_burst_trace.json`
- `traces/memory/lpddr5/complex/multi_dma_trace.json`
- `traces/memory/lpddr5/complex/stream_trace.json`
- `traces/memory/gddr6/bandwidth/eight_bank_bandwidth_trace.json`
- `traces/memory/gddr6/bandwidth/max_bandwidth_trace.json`
- `traces/memory/gddr6/bandwidth/page_burst_trace.json`
- `traces/memory/gddr6/complex/multi_dma_trace.json`
- `traces/memory/gddr6/complex/stream_trace.json`

**Root Cause:** The `generate_all_traces.sh` script's `--clean` option deleted ALL traces, but only regenerated traces for patterns explicitly listed in the script. Missing patterns:
- LPDDR5: `stream`, `multi_dma`, `max_bandwidth`, `page_burst`
- GDDR6: `stream`, `multi_dma`, `max_bandwidth`, `page_burst`, `eight_bank_bandwidth`

**Fix:**
1. Restored traces from git: `git checkout -- traces/memory/lpddr5/ traces/memory/gddr6/`
2. Updated `traces/scripts/generate_all_traces.sh` to include all missing patterns

### LPDDR5 Page Burst Test Fix

**Issue:** `lpddr5_page_burst` test failing after trace regeneration.

**Root Cause 1:** Queue depth (64) too small for 128 requests, causing silent request drops.

**Root Cause 2:** Test expected exactly 127/128 page hits, but DRAM refresh (tREFIpb=244 cycles) periodically closes pages, causing some `page_empty` hits instead of `page_hit`.

**Fix:**
1. Created `bandwidth_test_config()` with `queue_depth = 2048`
2. Changed assertions to expect >90% hit rate instead of exact count
3. Updated all bandwidth test functions to use the new config

### Collapsible HBM Swimlane Visualization

**Investigation:** User reported overlapping data bus cycles for concurrent banks in HBM2 swimlane visualization.

**Finding:** NOT a bug - HBM has 16 pseudo-channels with independent data buses:
- HBM2: 2 PCs per channel × 8 channels = 16 PCs, each with 64-bit DQ bus
- HBM3: 2 PCs per channel × 16 channels = 32 PCs, each with 32-bit DQ bus
- Each PC uses different physical pins (DQ[63:0], DQ[127:64], etc.)
- The visualization was showing all buses on limited lanes, causing apparent overlap

**Implementation:** Collapsible Pseudo-Channel Hierarchy (Option 1 of 3 presented)

**New Visualization Features:**
- Hierarchical collapsible structure: Channel → PC → Banks + Data Bus
- Expand All / Collapse All buttons
- Activity indicators (colored bars) for collapsed sections
- Per-channel color coding
- DQ pin range display (e.g., "DQ[63:0]")
- Only shows active channels/PCs/banks from trace
- Tooltip shows full details including DQ pin range

**Files Created/Updated:**
- `traces/memory/hbm2/tools/swimlane.html` - Complete rewrite with collapsible hierarchy
- `traces/memory/hbm3/tools/swimlane.html` - HBM3 version (16 channels, 32-bit per PC)

**Key Code Additions:**
```javascript
function decodeBankId(bankId) {
    const channel = Math.floor(bankId / (NUM_PCS_PER_CHANNEL * BANKS_PER_PC));
    const remainder = bankId % (NUM_PCS_PER_CHANNEL * BANKS_PER_PC);
    const pc = Math.floor(remainder / BANKS_PER_PC);
    const bank = remainder % BANKS_PER_PC;
    return { channel, pc, bank };
}

function getDQRange(channel, pc) {
    const pcIndex = channel * NUM_PCS_PER_CHANNEL + pc;
    const startBit = pcIndex * BITS_PER_PC;
    const endBit = startBit + BITS_PER_PC - 1;
    return `DQ[${endBit}:${startBit}]`;
}
```

## Summary of All Changes (Session 1 + Session 2)

### Files Created
| File | Description |
|------|-------------|
| HBM2 Memory Controller (header + cpp) | ~1800 lines |
| HBM3 Memory Controller (header + cpp) | ~1800 lines |
| HBM2 Pattern Infrastructure (18 files) | ~3000 lines |
| HBM3 Pattern Infrastructure (18 files) | ~3000 lines |
| HBM2/HBM3 Swimlane Visualizations | ~800 lines each |

### Files Modified
| File | Change |
|------|--------|
| `include/sw/kpu/fidelity/simulation_fidelity.hpp` | Added HBM2, HBM2E enum |
| `include/sw/trace/trace_entry.hpp` | Added HBM component types |
| `src/components/memory/memory_controller_factory.cpp` | Wire up HBM controllers |
| `src/components/memory/CMakeLists.txt` | Add HBM source files |
| `patterns/CMakeLists.txt` | Add HBM pattern targets |
| `traces/scripts/generate_all_traces.sh` | Add missing LPDDR5/GDDR6 patterns |
| `patterns/memory/lpddr5/bandwidth/page_burst.cpp` | Fix queue depth and assertions |
| `docs/analysis/memory-characterization.md` | HBM documentation |

## Session 3: Swimlane Visualization Bug Fixes

### Bugs Reported

User testing of the HBM2/HBM3 swimlane visualizations revealed several issues:

1. **Max bandwidth reported as 327.7 GB/s** - impossible value (HBM2 peak is 256 GB/s)
2. **Max bandwidth period selection shows wrong period** - showed beginning of trace (lowest bandwidth) instead of actual max
3. **Min/Max latency not highlighting the transaction** - clicking didn't select the associated request
4. **Horizontal panning broken** - lane labels scrolled away instead of staying fixed
5. **CA Bus missing activity indicators** - Data Bus showed aggregated activity when collapsed, CA Bus didn't
6. **Playback cursor misaligned after zooming** - cursor didn't track cycle count after zoom
7. **Can't reset to 100% zoom** - multiplying/dividing by 1.5 never reaches exactly 100%

### Root Cause Analysis

**1. Bandwidth Double-Counting**
Each data transfer creates TWO trace events: `databus-X-Y` and `globalbus-X-Y`. The utilization calculation used Sets for deduplication, but bandwidth calculation counted both:
```javascript
// Bug: counted databus-0-0 AND globalbus-0-0 as separate transfers
const dataBusEvents = events.filter(e => e.type === 'data-rd' || e.type === 'data-wr');
```

**Fix:** Filter to only `databus-*` events:
```javascript
const dataBusEvents = events.filter(e =>
    (e.type === 'data-rd' || e.type === 'data-wr') &&
    e.lane && e.lane.startsWith('databus-')
);
```

**2. Latency Highlight Not Working (HBM2 only)**
The `selectRequest()` function expected an ID string, but code passed the full request object:
```javascript
// Bug: passed object instead of ID
selectRequest(minLatencyRequest);
// Fix:
selectRequest(minLatencyRequest.id);
```

**3. Panning Broken**
Period overlays had `position: absolute` and `z-index: 40`, which was higher than lane labels' `z-index: 5`. The overlays covered the lane labels and intercepted scroll events.

**Fix:** Replaced overlays with CSS class-based dimming:
```css
.event.period-dimmed {
    opacity: 0.25;
}
```

**4. CA Bus Activity Indicators**
The aggregated activity logic only handled Data Bus group headers:
```javascript
if (lane.isDataBusGroupHeader) {
    relevantEvents = events.filter(e => e.lane && e.lane.startsWith('databus-'));
}
```

**Fix:** Added CA Bus case:
```javascript
} else if (lane.isCABusGroupHeader) {
    relevantEvents = events.filter(e => e.lane && e.lane.startsWith('cabus-'));
}
```

**5. Playback Cursor Offset**
CSS already had `margin-left: 200px` for lane-label width, but JavaScript ALSO added 200:
```javascript
// Bug: double offset
const left = 200 + currentCycle * pixelsPerCycle;
```

**Fix:**
```javascript
// CSS handles margin, just set left based on cycle
const left = currentCycle * pixelsPerCycle;
```

**6. Zoom Reset Impossible**
Using `* 1.5` and `/ 1.5` produces sequences like:
- 100% → 150% → 225% → 337.5%
- 100% → 66.7% → 44.4% → 29.6%
Never returns to exactly 100%.

**Fix:** Preset zoom levels with discrete stepping:
```javascript
const ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];

function zoomIn() {
    const nextIndex = ZOOM_LEVELS.findIndex(z => z > zoomLevel);
    if (nextIndex !== -1) setZoom(ZOOM_LEVELS[nextIndex]);
}

function resetZoom() {
    setZoom(1.0);
}
```

Added:
- Keyboard shortcut '0' resets to 100%
- Clicking zoom level display resets to 100%

### Files Modified

| File | Changes |
|------|---------|
| `traces/memory/hbm2/tools/swimlane.html` | All 7 bug fixes |
| `traces/memory/hbm3/tools/swimlane.html` | All 7 bug fixes |

### Validation

All fixes applied consistently to both HBM2 and HBM3 swimlane visualizations:
- Bandwidth now correctly reports ~244 GB/s at 95% utilization (vs impossible 327 GB/s)
- Min/Max latency clicks highlight the correct transaction
- Lane labels stay fixed during horizontal scrolling
- CA Bus shows activity indicators when collapsed
- Playback cursor aligns with cycle count at all zoom levels
- Zoom preset levels allow returning to exactly 100%

## Session 4: HBM Variants, Trace Validators, and Visualization Improvements

### HBM2E and HBM3E Timing Parameters

Added distinct timing parameters for the higher-speed HBM variants:

**HBM2E-3600 (3.6 Gbps, 1.8 GHz CK):**
- Scaled from HBM2: 1.0/1.8 ≈ 0.56x
- tRCD: 7 (was 12), tRP: 8 (was 14), tRAS: 16 (was 28), tRC: 24 (was 42)
- tCL: 10 (was 18), tWL: 4 (was 7)
- Peak bandwidth: 461 GB/s

**HBM3E-9600 (9.6 Gbps, 4.8 GHz CK):**
- Scaled from HBM3: 2.8/4.8 ≈ 0.58x
- tRCD: 5 (was 8), tRP: 5 (was 8), tRAS: 10 (was 16), tRC: 14 (was 24)
- tCL: 5 (was 8), tWL: 3 (was 4)
- Peak bandwidth: 1229 GB/s

**File Modified:** `src/components/memory/memory_controller_factory.cpp`

### HBM2 and HBM3 Trace Validators

Created Python trace validators following the GDDR6 pattern:

**Files Created:**
| File | Lines | Description |
|------|-------|-------------|
| `patterns/memory/hbm2/INVARIANTS.md` | ~300 | HBM2 invariant documentation |
| `patterns/memory/hbm2/common/trace_validator.py` | ~550 | HBM2 Python validator |
| `patterns/memory/hbm3/INVARIANTS.md` | ~300 | HBM3 invariant documentation |
| `patterns/memory/hbm3/common/trace_validator.py` | ~550 | HBM3 Python validator |

**Invariants Implemented:**
- INV-001 to INV-004: Transaction structure (txn_id semantics, command ownership, temporal ordering)
- INV-100: tRCD constraint (ACTIVATE to READ/WRITE)
- INV-101: tRP constraint (PRECHARGE to ACTIVATE)
- INV-102: tRRD constraint (ACTIVATE to ACTIVATE, bank group aware)
- INV-103: tFAW constraint (four-activate window)
- INV-106: tCCD constraint (CAS to CAS, bank group aware)
- INV-107: tRAS constraint (minimum row active time)
- INV-108: tRC constraint (row cycle time)

**Key Features:**
- Pseudo-channel aware timing checks (timing constraints apply per-PC)
- Bank group detection: `bg = bank / 4`
- Bank ID decoding: `channel = bank_id / 32; pc = (bank_id % 32) / 16; bank = bank_id % 16`

### LPDDR5/GDDR6 Swimlane Visualization Improvements

Applied the same zoom fixes from HBM swimlanes to LPDDR5 and GDDR6:

**Files Modified:**
- `traces/memory/lpddr5/tools/swimlane.html`
- `traces/memory/gddr6/tools/swimlane.html`

**Changes:**
1. Fixed `updatePlaybackCursor()` - removed duplicate offset (CSS already handles margin-left)
2. Added `ZOOM_LEVELS` array: `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]`
3. Added `zoomIn()`, `zoomOut()`, `resetZoom()` functions
4. Updated zoom button event listeners to use new functions
5. Made zoom level display clickable to reset to 100%
6. Added keyboard shortcut '0' to reset zoom

### Summary

All "Next Steps" from Session 3 have been completed:
- ✅ Add HBM2E and HBM3E variants with higher data rates
- ✅ Add trace validators for HBM (like LPDDR5)
- ✅ Apply similar visualization improvements to LPDDR5/GDDR6 swimlanes

Note: GDDR6 already had a trace validator (639 lines), so no new validator was needed.

## Session 5: HBM2E Separate Pattern Infrastructure

### User Request

User asked how to produce a trace for HBM2E-3600 configuration. Initial approach mixed HBM2E patterns into the HBM2 directory, which the user correctly identified as problematic:

> "Is that your design? I don't like it as it mixes HBM2 with HBM2E traces. How would that manifest itself in the visualization tools? As they are hardcoded labeling to HBM2, wouldn't that create confusion?"

### Solution: Separate HBM2E Infrastructure

Created a complete separate directory structure for HBM2E variants:

```
patterns/memory/hbm2e/
├── common/
│   ├── hbm2e_configs.hpp     # HBM2E-3200 and HBM2E-3600 configs
│   └── hbm2e_harness.hpp     # Test harness with variant-aware clock
├── single-bank/
│   ├── hbm2e_3600_page_hits.cpp
│   ├── hbm2e_3600_page_conflicts.cpp
│   └── hbm2e_3200_page_hits.cpp
└── bandwidth/
    └── hbm2e_3600_max_bandwidth.cpp

traces/memory/hbm2e/
└── tools/
    └── swimlane.html         # Visualization labeled "460.8 GB/s"
```

### Key Design Decisions

**1. Variant-Aware Harness**
The `HBM2EHarness` class tracks which variant is in use and exports traces with correct clock frequency:

```cpp
enum class HBM2EVariant {
    HBM2E_3200,  // 3.2 Gbps @ 1.6 GHz
    HBM2E_3600   // 3.6 Gbps @ 1.8 GHz
};

// Trace export uses correct clock for time conversion
bool export_trace(const std::string& filename) {
    return ChromeTraceExporter::export_traces(
        filename,
        mc_->trace_entries(),
        clock_ghz()  // 1.6 or 1.8 GHz based on variant
    );
}
```

**2. Timing Parameters at Higher Clock**
HBM2E-3600 timings in cycles (at 1.8 GHz CK) vs HBM2-2000 (at 1.0 GHz CK):

| Parameter | HBM2-2000 (1.0 GHz) | HBM2E-3600 (1.8 GHz) | Ratio |
|-----------|---------------------|----------------------|-------|
| tRCDRD | 12 | 22 | 1.8x |
| tRP | 14 | 25 | 1.8x |
| tRAS | 28 | 50 | 1.8x |
| tRC | 42 | 76 | 1.8x |
| tRL | 18 | 32 | 1.8x |
| tWL | 7 | 13 | 1.8x |

The cycle counts increase proportionally to clock (same absolute time in ns).

**3. Separate Visualization**
The swimlane.html for HBM2E shows:
- Header: "HBM2E Swimlane View"
- Badge: "460.8 GB/s" (vs HBM2's "256 GB/s")
- Comments reference HBM2E architecture

### Files Created

| File | Lines | Description |
|------|-------|-------------|
| `patterns/memory/hbm2e/common/hbm2e_configs.hpp` | ~300 | 3200/3600 configs, timing constants |
| `patterns/memory/hbm2e/common/hbm2e_harness.hpp` | ~300 | Variant-aware test harness |
| `patterns/memory/hbm2e/single-bank/hbm2e_3600_page_hits.cpp` | ~75 | Page hit pattern |
| `patterns/memory/hbm2e/single-bank/hbm2e_3600_page_conflicts.cpp` | ~85 | Page conflict pattern |
| `patterns/memory/hbm2e/single-bank/hbm2e_3200_page_hits.cpp` | ~75 | HBM2E-3200 variant |
| `patterns/memory/hbm2e/bandwidth/hbm2e_3600_max_bandwidth.cpp` | ~95 | Bandwidth test |
| `traces/memory/hbm2e/tools/swimlane.html` | ~2100 | Adapted from HBM2 |

### Files Modified

| File | Change |
|------|--------|
| `patterns/CMakeLists.txt` | Added HBM2E pattern targets with `add_hbm2e_pattern()` function |
| `patterns/memory/hbm2/common/hbm2_configs.hpp` | Removed HBM2E configs (now in separate file) |

### Cleanup

Removed from `hbm2_configs.hpp`:
- `hbm2e_3600_config()` function
- `single_pc_config_3600()` function
- `full_stack_config_3600()` function
- `HBM2E_3600_BANDWIDTH` constant

These are now properly located in `hbm2e_configs.hpp`.

### Usage

To generate an HBM2E-3600 trace:

```bash
# Build
cmake --build --preset release

# Run pattern
./build/patterns/memory/hbm2e/hbm2e_3600_page_hits

# Output shows:
# Configuration: HBM2E-3600 @ 1.8 GHz
# Peak bandwidth: 460.8 GB/s
# Trace exported to: traces/memory/hbm2e/single-bank/hbm2e_3600_page_hits_trace.json
```

### Test Results

All 4 HBM2E patterns pass:
```
PASS: hbm2e_3600_page_hits
PASS: hbm2e_3600_page_conflicts
PASS: hbm2e_3600_max_bandwidth
PASS: hbm2e_3200_page_hits
```

### Lessons Learned

1. **Separate variants cleanly** - Mixing HBM2 and HBM2E in the same directory creates confusion in visualization tools and trace analysis
2. **Variant-aware infrastructure** - The harness should track which variant is in use for correct clock frequency in trace exports
3. **Consistent labeling** - Each technology variant should have its own clearly labeled visualization tools

## Next Steps

1. Calibration data collection for multi-fidelity models
2. Run HBM trace validators on existing traces to verify
3. Add HBM3E separate pattern infrastructure (following HBM2E pattern)
4. Consider adding more complex HBM2E pattern tests (multi-channel, full-bank)
