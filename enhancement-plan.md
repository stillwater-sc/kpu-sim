# HBM Enhancement Plan

 Implementation Plan: HBM Variants, Trace Validators, and Visualization Improvements

 Overview

 Four-part implementation to enhance HBM memory controller variants, add trace validators for HBM2/HBM3, and port visualization improvements to
 LPDDR5/GDDR6.

 Part 1: HBM2E and HBM3E Timing Variants

 Current State

 - MemoryTechnology::HBM2E and HBM3E enum values already exist
 - Factory handles them but uses same timing as base variants
 - Need distinct timing parameters for higher data rates

 HBM2E Specifications (3.6 Gbps vs HBM2's 2.0 Gbps)

 - Clock: 1.8 GHz CK (vs 1.0 GHz)
 - Timing scale factor: ~0.56x (1.0/1.8)
 - Key timings to adjust: tRCD, tRP, tRAS, tRC, tRL, tWL

 HBM3E Specifications (9.6 Gbps vs HBM3's 5.6 Gbps)

 - Clock: 4.8 GHz CK (vs 2.8 GHz)
 - Timing scale factor: ~0.58x (2.8/4.8)
 - Key timings to adjust: tRCD, tRP, tRAS, tRC, tRL, tWL

 Files to Modify

 | File                                                | Changes                                                       |
 |-----------------------------------------------------|---------------------------------------------------------------|
 | src/components/memory/memory_controller_factory.cpp | Add separate case blocks for HBM2E/HBM3E with distinct timing |

 Implementation

 // In get_default_timing():
 case MemoryTechnology::HBM2E:
     params.hbm2_timing = hbm2::TimingParams{
         .tRCDRD = 7,   // 12 * 0.56 ≈ 7
         .tRCDWR = 4,   // 6 * 0.56 ≈ 4
         .tRP = 8,      // 14 * 0.56 ≈ 8
         .tRAS = 16,    // 28 * 0.56 ≈ 16
         .tRC = 24,     // 42 * 0.56 ≈ 24
         .tRL = 10,     // 18 * 0.56 ≈ 10
         .tWL = 4,      // 7 * 0.56 ≈ 4
         // ... other timings scaled similarly
     };
     break;

 case MemoryTechnology::HBM3E:
     params.hbm3_timing = hbm3::TimingParams{
         .tRCD = 5,     // 8 * 0.58 ≈ 5
         .tRP = 5,      // 8 * 0.58 ≈ 5
         .tRAS = 10,    // 16 * 0.58 ≈ 10
         .tRC = 14,     // 24 * 0.58 ≈ 14
         .tRL = 5,      // 8 * 0.58 ≈ 5
         .tWL = 3,      // 4 * 0.58 ≈ 3
         // ... other timings scaled similarly
     };
     break;

 ---
 Part 2: HBM2 and HBM3 Trace Validators

 Current State

 - LPDDR5 has validator: patterns/memory/lpddr5/common/trace_validator.py (501 lines)
 - GDDR6 has validator: patterns/memory/gddr6/common/trace_validator.py (639 lines)
 - HBM2/HBM3 have NO validators

 Files to Create

 | File                                           | Description                  |
 |------------------------------------------------|------------------------------|
 | patterns/memory/hbm2/INVARIANTS.md             | HBM2 invariant documentation |
 | patterns/memory/hbm2/common/trace_validator.py | HBM2 Python validator        |
 | patterns/memory/hbm3/INVARIANTS.md             | HBM3 invariant documentation |
 | patterns/memory/hbm3/common/trace_validator.py | HBM3 Python validator        |

 HBM2 Validator Structure (based on GDDR6 pattern)

 # Key data classes
 @dataclass
 class Event:
     cycle: int
     lane: str
     event_type: str
     txn_id: int
     channel: int      # 0-7 for HBM2
     pc: int           # pseudo-channel 0-1
     bank: int         # 0-15 per PC
     # Factory method from_json()

 @dataclass
 class Transaction:
     txn_id: int
     events: List[Event]
     # Properties: has_activate, has_read, has_write, etc.

 # Timing parameters (HBM2-2000 @ 1.0 GHz)
 TIMING = {
     'tRCDRD': 12, 'tRCDWR': 6, 'tRP': 14, 'tRAS': 28, 'tRC': 42,
     'tRL': 18, 'tWL': 7, 'tWR': 16, 'tRTP': 6,
     'tRRD_L': 4, 'tRRD_S': 3, 'tCCD_L': 4, 'tCCD_S': 2, 'tFAW': 16
 }

 # Invariant checks
 class HBM2TraceValidator:
     def _check_inv_001_txn_id_semantics(self): ...
     def _check_inv_002_command_ownership(self): ...
     def _check_inv_003_temporal_ordering(self): ...
     def _check_inv_100_tRCD_constraint(self): ...
     def _check_inv_101_tRP_constraint(self): ...
     def _check_inv_102_tRRD_constraint(self): ...
     def _check_inv_103_tFAW_constraint(self): ...
     def _check_inv_106_tCCD_constraint(self): ...
     def _check_inv_107_tRAS_constraint(self): ...

 HBM2 Invariants (INVARIANTS.md)

 Structure Invariants:
 - INV-001: Valid txn_id semantics (one data op per txn)
 - INV-002: ACTIVATE/PRECHARGE belong to valid transactions
 - INV-003: Temporal ordering (ACT before RD/WR before PRE)
 - INV-004: Unique transaction IDs

 Timing Invariants:
 - INV-100: tRCDRD/tRCDWR (ACT to RD/WR)
 - INV-101: tRP (PRE to ACT)
 - INV-102: tRRD_L/tRRD_S (ACT to ACT same/diff bank group)
 - INV-103: tFAW (four-activate window)
 - INV-106: tCCD_L/tCCD_S (CAS to CAS)
 - INV-107: tRAS (minimum active time)
 - INV-108: tRC (row cycle)

 HBM3 Validator

 Similar structure with HBM3-specific timing:
 - 16 channels (vs 8)
 - Different timing values (tRCD=8, tRP=8, etc.)
 - Shorter timing cycles due to higher clock

 ---
 Part 3: Swimlane Visualization Improvements

 Features to Port from HBM to LPDDR5/GDDR6

 | Feature                    | Description                                              | Priority |
 |----------------------------|----------------------------------------------------------|----------|
 | Preset zoom levels         | ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0] | High     |
 | Reset zoom function        | resetZoom() returns to 100%                              | High     |
 | Clickable zoom display     | Click percentage to reset                                | High     |
 | Correct cursor positioning | Remove duplicate 200px offset                            | High     |
 | Keyboard shortcut '0'      | Reset zoom to 100%                                       | Medium   |

 Files to Modify

 | File                                     | Changes                            |
 |------------------------------------------|------------------------------------|
 | traces/memory/lpddr5/tools/swimlane.html | Add zoom preset system, fix cursor |
 | traces/memory/gddr6/tools/swimlane.html  | Add zoom preset system, fix cursor |

 Key Code Changes

 1. Add ZOOM_LEVELS array and index:
 const ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];
 let zoomIndex = ZOOM_LEVELS.indexOf(1.0);

 2. Add zoom control functions:
 function zoomIn() {
     const nextIndex = ZOOM_LEVELS.findIndex(z => z > zoomLevel);
     if (nextIndex !== -1) {
         zoomIndex = nextIndex;
         setZoom(ZOOM_LEVELS[zoomIndex]);
     }
 }

 function zoomOut() {
     let prevIndex = -1;
     for (let i = ZOOM_LEVELS.length - 1; i >= 0; i--) {
         if (ZOOM_LEVELS[i] < zoomLevel) { prevIndex = i; break; }
     }
     if (prevIndex !== -1) {
         zoomIndex = prevIndex;
         setZoom(ZOOM_LEVELS[zoomIndex]);
     }
 }

 function resetZoom() {
     zoomIndex = ZOOM_LEVELS.indexOf(1.0);
     setZoom(1.0);
 }

 3. Fix playback cursor positioning:
 function updatePlaybackCursor() {
     // CSS already has margin-left, don't add offset again
     const left = currentCycle * pixelsPerCycle;
     playbackCursor.style.left = `${left}px`;
     currentCycleEl.textContent = currentCycle;
 }

 4. Update event listeners:
 btnZoomIn.addEventListener('click', zoomIn);
 btnZoomOut.addEventListener('click', zoomOut);

 zoomLevelEl.style.cursor = 'pointer';
 zoomLevelEl.title = 'Click to reset to 100%';
 zoomLevelEl.addEventListener('click', resetZoom);

 5. Add keyboard shortcut:
 case '0':
     e.preventDefault();
     resetZoom();
     break;

 ---
 Implementation Order

 1. Part 1: HBM2E/HBM3E timing (~30 min)
   - Update memory_controller_factory.cpp with distinct timing cases
 2. Part 2a: HBM2 trace validator (~1 hr)
   - Create patterns/memory/hbm2/INVARIANTS.md
   - Create patterns/memory/hbm2/common/trace_validator.py
 3. Part 2b: HBM3 trace validator (~45 min)
   - Create patterns/memory/hbm3/INVARIANTS.md
   - Create patterns/memory/hbm3/common/trace_validator.py
 4. Part 3a: LPDDR5 visualization (~30 min)
   - Update traces/memory/lpddr5/tools/swimlane.html
 5. Part 3b: GDDR6 visualization (~30 min)
   - Update traces/memory/gddr6/tools/swimlane.html
 6. Validation & Documentation
   - Run validators on existing traces
   - Test visualizations
   - Update CHANGELOG.md and session log

 ---
 Files Summary

 New Files (4)

 | File                                           | Est. Lines |
 |------------------------------------------------|------------|
 | patterns/memory/hbm2/INVARIANTS.md             | ~300       |
 | patterns/memory/hbm2/common/trace_validator.py | ~550       |
 | patterns/memory/hbm3/INVARIANTS.md             | ~300       |
 | patterns/memory/hbm3/common/trace_validator.py | ~550       |

 Modified Files (3)

 | File                                                | Changes             |
 |-----------------------------------------------------|---------------------|
 | src/components/memory/memory_controller_factory.cpp | HBM2E/HBM3E timing  |
 | traces/memory/lpddr5/tools/swimlane.html            | Zoom + cursor fixes |
 | traces/memory/gddr6/tools/swimlane.html             | Zoom + cursor fixes |

 ---
 Notes

 - GDDR6 already has a trace validator (639 lines) - no action needed
 - HBM swimlane has additional features (collapsible hierarchy, bandwidth stats) that are HBM-specific and won't be ported to LPDDR5/GDDR6
 - Focus on high-value, universal improvements (zoom, cursor) for LPDDR5/GDDR6

##  Summary of Completed Work

Part 1: HBM2E/HBM3E Timing Variants
  - Added distinct timing parameters in memory_controller_factory.cpp
  - HBM2E-3600: 1.8 GHz clock, 461 GB/s peak bandwidth (0.56x scaling)
  - HBM3E-9600: 4.8 GHz clock, 1229 GB/s peak bandwidth (0.58x scaling)

Part 2: HBM Trace Validators
  - patterns/memory/hbm2/INVARIANTS.md - 11.5 KB documentation
  - patterns/memory/hbm2/common/trace_validator.py - 26 KB validator
  - patterns/memory/hbm3/INVARIANTS.md - 10 KB documentation
  - patterns/memory/hbm3/common/trace_validator.py - 26 KB validator
  - GDDR6 already had a validator (639 lines) - no action needed

Part 3: Swimlane Visualization Improvements
  - LPDDR5 swimlane: Added zoom presets, fixed cursor positioning
  - GDDR6 swimlane: Added zoom presets, fixed cursor positioning
  - Both now have: ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
  - Keyboard shortcut '0' to reset zoom, clickable percentage to reset

Documentation
  - CHANGELOG.md updated with all new features
  - Session log updated with Session 4 details

Build verified successful. All validators can be run with:
```bash
  python3 patterns/memory/hbm2/common/trace_validator.py <trace.json>
  python3 patterns/memory/hbm3/common/trace_validator.py <trace.json>
```

## HBM2E

To Generate an HBM2E-3600 Trace

```bash
  # Build the patterns
  cmake --build --preset release

  # Run any HBM2E-3600 pattern
  ./build/patterns/memory/hbm2e/hbm2e_3600_page_hits
  ./build/patterns/memory/hbm2e/hbm2e_3600_page_conflicts
  ./build/patterns/memory/hbm2e/hbm2e_3600_max_bandwidth

  # For HBM2E-3200:
  ./build/patterns/memory/hbm2e/hbm2e_3200_page_hits
```

  Traces are written to: traces/memory/hbm2e/<category>/

Created Structure

```
  patterns/memory/hbm2e/
  ├── common/
  │   ├── hbm2e_configs.hpp     # 3200/3600 configs
  │   └── hbm2e_harness.hpp     # Test harness with variant support
  ├── single-bank/
  │   ├── hbm2e_3600_page_hits.cpp
  │   ├── hbm2e_3600_page_conflicts.cpp
  │   └── hbm2e_3200_page_hits.cpp
  └── bandwidth/
      └── hbm2e_3600_max_bandwidth.cpp

  traces/memory/hbm2e/
  └── tools/
      └── swimlane.html         # Visualization (labeled 460.8 GB/s)
```

Key Configs Available

  | Config                  | Clock   | Peak BW    |
  |-------------------------|---------|------------|
  | hbm2e_3200_config()     | 1.6 GHz | 409.6 GB/s |
  | hbm2e_3600_config()     | 1.8 GHz | 460.8 GB/s |
  | single_pc_config_3200() | 1.6 GHz | Single PC  |
  | single_pc_config_3600() | 1.8 GHz | Single PC  |

## HBM3E

HBM3E pattern infrastructure complete. Summary:

Created Structure

```
  patterns/memory/hbm3e/
  ├── common/
  │   ├── hbm3e_configs.hpp     # 8400/9600 configs
  │   └── hbm3e_harness.hpp     # Variant-aware harness
  ├── single-bank/
  │   ├── hbm3e_9600_page_hits.cpp
  │   ├── hbm3e_9600_page_conflicts.cpp
  │   └── hbm3e_8400_page_hits.cpp
  └── bandwidth/
      └── hbm3e_9600_max_bandwidth.cpp

  traces/memory/hbm3e/tools/swimlane.html  # 1.23 TB/s badge
```

HBM3E Variants

  | Variant    | Clock   | Peak Bandwidth |
  |------------|---------|----------------|
  | HBM3E-8400 | 4.2 GHz | 1.075 TB/s     |
  | HBM3E-9600 | 4.8 GHz | 1.229 TB/s     |

  Test Results

  All 4 patterns pass:
  PASS: hbm3e_9600_page_hits
  PASS: hbm3e_9600_page_conflicts
  PASS: hbm3e_9600_max_bandwidth
  PASS: hbm3e_8400_page_hits

Usage

  ./build/patterns/memory/hbm3e/hbm3e_9600_page_hits
  # Trace exported to: traces/memory/hbm3e/single-bank/hbm3e_9600_page_hits_trace.json

