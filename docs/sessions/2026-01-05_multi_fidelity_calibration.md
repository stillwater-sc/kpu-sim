# Session Log: Multi-Fidelity Calibration Framework

**Date:** 2026-01-05
**Duration:** ~3 hours
**Focus:** Implement calibration framework for memory controller multi-fidelity simulation

## Summary

Implemented a complete calibration workflow for deriving behavioral and transactional model parameters from cycle-accurate simulation. This enables the faster simulation models to accurately approximate cycle-accurate results, achieving 1.3% cycle error for the transactional model.

## Context

The KPU simulator supports three fidelity levels for memory controller simulation:

1. **Cycle-Accurate** (`LPDDR5MemoryController`): Models every clock cycle, DRAM commands, bank state machines. Reference model but slowest.
2. **Transactional** (`TransactionalMemoryController`): Queue-based timing with statistical latency. ~10-100x faster.
3. **Behavioral** (`BehavioralMemoryController`): Fixed latency, no contention modeling. ~100-1000x faster.

The challenge was that lower-fidelity models used hardcoded timing values that didn't match cycle-accurate behavior. This session implemented a calibration framework to derive accurate parameters from cycle-accurate simulation.

## Implementation

### Phase 1: Calibration Storage Schema

Created `include/sw/kpu/calibration/calibration_storage.hpp`:
- `CycleAccurateReference`: Reference metrics (total requests, cycles, latencies, page hit/conflict rates)
- `BehavioralCalibration`: Fixed read/write latencies
- `TransactionalCalibration`: Mean latencies, page scenario factors, per-scenario latencies
- `ValidationResults`: Error percentages and pass/fail status
- JSON serialization via `src/calibration/calibration_storage.cpp`

### Phase 2: Calibration Extraction

Created `include/sw/kpu/calibration/calibration_extraction.hpp`:
- Extended `LPDDR5MemoryController::Statistics` with per-scenario latency tracking
- `derive_behavioral()`: Extract fixed latencies from cycle-accurate stats
- `derive_transactional()`: Extract mean latencies and page factors
- `extract_calibration()`: Complete calibration data extraction

### Phase 3: CLI Tools

Created `tools/calibration/kpu-calibrate.cpp`:
- Runs cycle-accurate simulation with balanced workloads (page hits, conflicts, empty, mixed)
- Extracts calibration parameters
- Saves to JSON file

Created `tools/calibration/kpu-validate.cpp`:
- Loads calibration from JSON
- Runs identical workload on all three fidelity levels
- Computes error percentages
- Updates calibration file with validation results
- `--quality` flag for quality assessment report

### Phase 4: Quality Assessment

Created `include/sw/kpu/calibration/calibration_quality.hpp`:
- Severity levels: INFO, WARNING, ERROR
- Quality criteria for sample size, coverage, latency, factors
- Assessment functions with scores (0-100)
- Quality grades (A-F)
- Formatted report output

### Phase 5: Transactional Model Fix

The critical fix was in `transactional_memory_controller.cpp`:

**Problem**: The transactional model was using calibrated end-to-end latencies (which include queueing delay) combined with its own per-bank serialization, causing double-counting of contention. Result: 2013% cycle error.

**Solution**: Use physical timing parameters (tCL, tRCD, tRP, tBurst) for service time calculation instead of calibrated latencies:

```cpp
// Before: Used calibrated latencies that included queueing
latency = static_cast<uint32_t>(base_latency * timing.page_hit_factor);

// After: Use physical timing for service time
uint32_t cas_latency = timing.tCL + timing.tBurst;  // ~22 cycles
if (bank_info.open_row == row) {
    latency = cas_latency;  // Page hit
} else if (!bank_info.has_row_open()) {
    latency = timing.tRCD + cas_latency;  // Page empty
} else {
    latency = timing.tRP + timing.tRCD + cas_latency;  // Page conflict
}
```

Also removed redundant queueing delay estimation that was added on top of `busy_until_cycle` tracking.

**Result**: Cycle error reduced from 2013% to **1.3%**.

## Files Changed

### New Files
- `include/sw/kpu/calibration/calibration_storage.hpp`
- `include/sw/kpu/calibration/calibration_extraction.hpp`
- `include/sw/kpu/calibration/calibration_quality.hpp`
- `src/calibration/calibration_storage.cpp`
- `src/calibration/CMakeLists.txt`
- `tools/calibration/kpu-calibrate.cpp`
- `tools/calibration/kpu-validate.cpp`
- `tools/calibration/CMakeLists.txt`
- `tests/calibration/calibration_storage_test.cpp`
- `tests/calibration/calibration_extraction_test.cpp`
- `tests/calibration/calibration_quality_test.cpp`
- `tests/calibration/CMakeLists.txt`
- `configs/calibration/lpddr5_6400.json`
- `docs/MULTI_FIDELITY_CALIBRATION_WORKFLOW.md`

### Modified Files
- `include/sw/kpu/components/lpddr5_memory_controller.hpp` - Added per-scenario latency tracking to Statistics
- `include/sw/kpu/fidelity/component_config.hpp` - Added page factors and per-scenario latency config
- `src/components/memory/lpddr5_memory_controller.cpp` - Track per-scenario latencies
- `src/components/memory/transactional_memory_controller.cpp` - Fixed latency calculation
- `src/CMakeLists.txt` - Added calibration subdirectory
- `tests/CMakeLists.txt` - Added calibration tests
- `tools/CMakeLists.txt` - Added calibration tools

## Test Results

All calibration tests pass:
- `calibration_storage_test`: 35 assertions in 6 test cases
- `calibration_extraction_test`: 41 assertions in 6 test cases
- `calibration_quality_test`: 29 assertions in 6 test cases

Validation results with new calibration:
| Model | Cycle Error | Status |
|-------|-------------|--------|
| Transactional | 1.3% | Excellent |
| Behavioral | 88.5% | Expected (no contention modeling) |

## Key Insights

1. **Service time vs end-to-end latency**: Calibrated latencies from cycle-accurate simulation include queueing delays. Using these in a model that also tracks queueing causes double-counting.

2. **Physical timing is the right abstraction**: The transactional model's per-bank `busy_until_cycle` tracking already models contention correctly. Using physical timing (tCL, tRCD, tRP) for service time allows the model to naturally recreate queueing behavior.

3. **Cycle count is the key metric**: For performance estimation, total cycle count (throughput) is more meaningful than individual request latencies, which can differ based on measurement semantics.

## Commit

```
f0e8b09 Add multi-fidelity calibration framework for memory controllers
```

Pushed to `origin/main`.
