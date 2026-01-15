# Multi-Fidelity Memory Controller Calibration Workflow

This document describes the workflow for calibrating behavioral and transactional memory controller models against a cycle-accurate reference implementation.

## Overview

The KPU simulator supports three fidelity levels for memory controller simulation:

| Fidelity | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| **Behavioral** | ~100-1000x | Low | Functional validation, early design exploration |
| **Transactional** | ~10-100x | Medium | Performance estimation, workload analysis |
| **Cycle-Accurate** | 1x (reference) | High | Timing validation, detailed analysis |

The calibration workflow ensures that faster models produce results consistent with the cycle-accurate reference.

## Calibration Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION PIPELINE                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────┐                   │
│  │   Workload   │───>│ Cycle-Accurate│───>│   Reference  │                   │
│  │   Patterns   │    │   LPDDR5 MC   │    │    Traces    │                   │
│  └──────────────┘    └───────────────┘    └──────┬───────┘                   │
│                                                  │                           │
│                            ┌─────────────────────┼─────────────────────┐     │
│                            ▼                     ▼                     ▼     │
│                   ┌────────────────┐    ┌────────────────┐    ┌────────────┐ │
│                   │ Extract Metrics│    │ Extract Metrics│    │  Validate  │ │
│                   │  (Behavioral)  │    │(Transactional) │    │   Traces   │ │
│                   └───────┬────────┘    └───────┬────────┘    └────────────┘ │
│                           │                     │                            │
│                           ▼                     ▼                            │
│                   ┌────────────────┐    ┌────────────────┐                   │
│                   │  Calibration   │    │  Calibration   │                   │
│                   │   Parameters   │    │   Parameters   │                   │
│                   │  (fixed lat)   │    │ (mean,var,etc) │                   │
│                   └───────┬────────┘    └───────┬────────┘                   │
│                           │                     │                            │
│                           ▼                     ▼                            │
│                   ┌────────────────┐    ┌────────────────┐                   │
│                   │   Behavioral   │    │  Transactional │                   │
│                   │      MC        │    │       MC       │                   │
│                   └───────┬────────┘    └───────┬────────┘                   │
│                           │                     │                            │
│                           └──────────┬──────────┘                            │
│                                      ▼                                       │
│                            ┌──────────────────┐                              │
│                            │  Cross-Validate  │                              │
│                            │  (Error < 5%?)   │                              │
│                            └────────┬─────────┘                              │
│                                     │                                        │
│                        ┌────────────┴────────────┐                           │
│                        ▼                         ▼                           │
│                   ┌─────────┐              ┌──────────┐                      │
│                   │  PASS   │              │   FAIL   │                      │
│                   │ Deploy  │              │ Iterate  │                      │
│                   └─────────┘              └──────────┘                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Infrastructure Status

### What Exists

| Component | Status | Location |
|-----------|--------|----------|
| Cycle-Accurate LPDDR5 MC | Complete | `src/components/memory/lpddr5_memory_controller.cpp` |
| Behavioral MC | Complete | `src/components/memory/behavioral_memory_controller.cpp` |
| Transactional MC | Complete | `src/components/memory/transactional_memory_controller.cpp` |
| Test Patterns | Complete | `patterns/memory/lpddr5/` |
| Trace Validation | Complete | `patterns/memory/lpddr5/common/trace_validator.py` |
| MultiFidelityHarness | Partial | `patterns/memory/lpddr5/common/multi_fidelity.hpp` |
| Factory Pattern | Complete | `src/components/memory/memory_controller_factory.cpp` |

### What's Missing

| Component | Status | Priority |
|-----------|--------|----------|
| Calibration Storage (JSON) | Not Started | High |
| Robust Calibration Extraction | Partial | High |
| kpu-calibrate CLI Tool | Not Started | Medium |
| kpu-validate CLI Tool | Not Started | Medium |
| Quality Metrics & Acceptance Criteria | Not Started | Medium |

## Key Files

### Interface & Implementations

```
include/sw/kpu/components/memory/
├── memory_controller_interface.hpp      # IMemoryController base
├── behavioral_memory_controller.hpp     # Instant/fixed latency
├── transactional_memory_controller.hpp  # Queue-based model
└── ...

include/sw/kpu/components/
└── lpddr5_memory_controller.hpp         # Cycle-accurate LPDDR5

src/components/memory/
├── memory_controller_factory.cpp        # Factory with tech defaults
├── behavioral_memory_controller.cpp
├── transactional_memory_controller.cpp
└── lpddr5_memory_controller.cpp         # 1100+ lines
```

### Configuration Framework

```
include/sw/kpu/fidelity/
├── simulation_fidelity.hpp              # Fidelity enums
└── component_config.hpp                 # MemoryControllerConfig

configs/calibration/                      # Calibration parameter files
└── lpddr5_6400.json                     # (to be created)
```

### Test Patterns

```
patterns/memory/lpddr5/
├── common/
│   ├── lpddr5_configs.hpp               # LPDDR5-6400 timing constants
│   ├── lpddr5_harness.hpp               # Test harness
│   ├── workloads.hpp                    # Workload definitions
│   └── multi_fidelity.hpp               # Comparison framework
├── single-bank/
│   ├── page-hits/main.cpp               # Sequential same-row reads
│   ├── page-conflicts/main.cpp          # Different-row reads
│   └── mixed-rw/main.cpp                # Alternating R/W
├── two-bank/
├── three-bank/
├── four-bank/
├── dual-channel/
└── complex/
```

## Calibration Parameters

### Behavioral Model

The behavioral model uses fixed latencies:

| Parameter | Description | Derivation |
|-----------|-------------|------------|
| `fixed_read_latency` | Cycles for read completion | Mean of CA read latencies |
| `fixed_write_latency` | Cycles for write completion | Mean of CA write latencies |

### Transactional Model

The transactional model uses statistical parameters:

| Parameter | Description | Derivation |
|-----------|-------------|------------|
| `mean_read_latency` | Average read latency | Mean of CA read latencies |
| `mean_write_latency` | Average write latency | Mean of CA write latencies |
| `latency_std_dev` | Latency variation | Std dev of CA latencies |
| `page_hit_factor` | Multiplier for page hits | CA page_hit_lat / mean_lat |
| `page_empty_factor` | Multiplier for page empty | CA page_empty_lat / mean_lat |
| `page_conflict_factor` | Multiplier for conflicts | CA conflict_lat / mean_lat |

## Calibration Storage Schema

Calibration parameters are stored in JSON format:

```json
{
  "version": "1.0",
  "technology": "LPDDR5",
  "speed_grade_mt_s": 6400,
  "calibration_date": "2026-01-05",
  "source_patterns": ["page_hits", "page_conflicts", "mixed_rw"],

  "cycle_accurate_reference": {
    "total_requests": 1000,
    "total_cycles": 45000,
    "mean_read_latency": 36.2,
    "mean_write_latency": 38.1,
    "page_hit_rate": 0.65,
    "page_conflict_rate": 0.20
  },

  "behavioral": {
    "fixed_read_latency_cycles": 36,
    "fixed_write_latency_cycles": 38
  },

  "transactional": {
    "mean_read_latency_cycles": 36,
    "mean_write_latency_cycles": 38,
    "latency_std_dev_cycles": 8,
    "page_hit_factor": 0.61,
    "page_empty_factor": 1.0,
    "page_conflict_factor": 1.39
  },

  "validation": {
    "behavioral_latency_error_pct": 3.2,
    "transactional_latency_error_pct": 1.8,
    "max_acceptable_error_pct": 5.0
  }
}
```

## CLI Tools

### kpu-calibrate

Generates calibration parameters from cycle-accurate runs:

```bash
./build/tools/calibration/kpu-calibrate \
    --technology LPDDR5 \
    --speed-grade 6400 \
    --patterns-dir patterns/memory/lpddr5 \
    --output configs/calibration/lpddr5_6400.json
```

### kpu-validate

Validates calibration quality against acceptance criteria:

```bash
./build/tools/calibration/kpu-validate \
    --calibration configs/calibration/lpddr5_6400.json \
    --patterns-dir patterns/memory/lpddr5 \
    --max-error 5%
```

## Acceptance Criteria

A calibration is considered valid when:

1. **Latency Error < 5%**: Mean absolute percentage error between calibrated model and CA reference
2. **Bandwidth Error < 3%**: Effective bandwidth within tolerance
3. **Pattern Coverage**: All standard patterns (page hits, conflicts, mixed R/W) validated
4. **No Systematic Bias**: Error distribution centered around zero

## Implementation Phases

### Phase 1: Calibration Storage
- Define JSON schema for calibration parameters
- Create C++ structs for loading/saving calibration
- Add calibration loader to factory

### Phase 2: Robust Calibration Extraction
- Separate read vs write latency tracking in CA model
- Per-pattern-type statistics collection
- Compute actual factors from CA reference data

### Phase 3: CLI Tools
- kpu-calibrate: Automated calibration generation
- kpu-validate: Cross-validation with error reporting

### Phase 4: CI Integration
- Automated calibration validation in CI pipeline
- Regression detection for calibration drift

## References

- `docs/SIMULATION_FIDELITY_FRAMEWORK.md` - Overall fidelity framework design
- `patterns/memory/lpddr5/INVARIANTS.md` - Trace validation invariants
- `patterns/memory/lpddr5/VALIDATION_FRAMEWORK.md` - Trace validation tools


---

# Implementation progress

## Phase 1: Calibration Storage - Complete

Summary of what was created:

  | File                                               | Purpose                                       |
  |----------------------------------------------------|-----------------------------------------------|
  | configs/calibration/lpddr5_6400.json               | Example calibration data for LPDDR5 6400 MT/s |
  | include/sw/kpu/calibration/calibration_storage.hpp | Header with CalibrationData structs and API   |
  | src/calibration/calibration_storage.cpp            | JSON load/save implementation                 |
  | src/calibration/CMakeLists.txt                     | Library build config                          |
  | tests/calibration/calibration_storage_test.cpp     | Unit tests (35 assertions, all pass)          |

The calibration schema captures:
  - Metadata: version, technology, speed grade, date, description
  - Cycle-accurate reference: ground truth metrics from CA simulation
  - Behavioral calibration: fixed read/write latencies
  - Transactional calibration: mean latencies + page hit/conflict factors
  - Validation results: error percentages, pass/fail status

Next phase options:
  1. Phase 2a: Behavioral calibration extraction - run CA model, measure separate read/write latencies
  2. Phase 2b: Transactional calibration - measure page hit/conflict patterns from CA traces
  3. Phase 3: Build kpu-calibrate CLI tool for automated calibration runs
  4. Phase 4: Build kpu-validate CLI tool for cross-validation between fidelity levels

## Phase 2: Calibration Extraction - Complete

Summary of changes:

  | File                                                   | Changes                                                                                                                            |
  |--------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
  | include/sw/kpu/components/lpddr5_memory_controller.hpp | Extended Statistics struct with separate R/W latencies, per-scenario latency tracking, and helper methods                          |
  | src/components/memory/lpddr5_memory_controller.cpp     | Updated complete_requests() to track calibration metrics, updated sync_interface_stats() for min/max                               |
  | include/sw/kpu/calibration/calibration_extraction.hpp  | New file with extract_reference(), derive_behavioral(), derive_transactional(), extract_calibration(), print_calibration_summary() |
  | tests/calibration/calibration_extraction_test.cpp      | Unit tests for extraction functions                                                                                                |

The CA model now tracks:
  - Separate read/write latencies: avg_read_latency(), avg_write_latency()
  - Per-scenario latencies: avg_page_hit_latency(), avg_page_empty_latency(), avg_page_conflict_latency()
  - Page factors: page_hit_factor(), page_empty_factor(), page_conflict_factor() (relative to mean)

Next: Phase 3 - Build kpu-calibrate CLI tool

This tool will:
  1. Run CA simulation with memory access patterns
  2. Extract calibration data using the new functions
  3. Save to JSON calibration file

## Phase 3: kpu-calibrate CLI Tool - Complete

Summary of what was created:

  | File                                | Purpose                          |
  |-------------------------------------|----------------------------------|
  | tools/calibration/kpu-calibrate.cpp | CLI tool for running calibration |
  | tools/calibration/CMakeLists.txt    | Build configuration              |

The tool provides:
  - --output=<file> - Save calibration to JSON
  - --technology=<tech> - Memory technology (default: LPDDR5)
  - --speed=<mt/s> - Speed grade (default: 6400)
  - --requests=<count> - Requests per workload type
  - --seed=<value> - Random seed for reproducibility
  - --verbose - Detailed progress output

The tool runs four workload types:
  1. Page hits - Sequential accesses to same row
  2. Page conflicts - Different rows in same bank
  3. Page empty - Round-robin across banks
  4. Mixed - Realistic 50% hit / 20% empty / 30% conflict pattern

Remaining tasks:
  - Phase 4: Build kpu-validate CLI tool for cross-validation
  - Phase 5: Add calibration quality metrics and acceptance criteria

## Phase 4: kpu-validate CLI Tool - Complete

Summary of what was created:

  | File                                     | Purpose                       |
  |------------------------------------------|-------------------------------|
  | tools/calibration/kpu-validate.cpp       | CLI tool for cross-validation |
  | Updated tools/calibration/CMakeLists.txt | Added kpu-validate target     |

The tool provides:
  - --update - Update calibration file with validation results
  - --requests=<count> - Number of requests for validation
  - --seed=<value> - Random seed for reproducibility
  - --threshold=<pct> - Maximum acceptable error percentage
  - --verbose - Detailed progress output

Validation results:

  | Model         | Latency Error | Cycle Error | Notes                              |
  |---------------|---------------|-------------|------------------------------------|
  | Behavioral    | ~20-60%       | ~25%        | Works, fixed latency approximation |
  | Transactional | Very high     | Very high   | Needs model improvements           |

Key findings:
  1. Behavioral model - Works reasonably well for latency approximation, but no contention modeling means cycle counts differ significantly from CA
  2. Transactional model - The current implementation doesn't properly use calibrated latency factors; it uses raw timing parameters (tCL, tRP, etc.) instead of calibrated mean latencies for page hit/conflict scenarios

Remaining task:
  - Phase 5: Add calibration quality metrics and acceptance criteria

The transactional model would benefit from updates to properly apply calibrated page hit/conflict factors, but that's a model improvement task rather than a calibration tool task.

## Phase 5 Calibration Quality Metrics and Acceptance Criteria - Complete

Created a comprehensive quality assessment framework:

Files Created/Modified:

  - include/sw/kpu/calibration/calibration_quality.hpp - Quality assessment header with:
    - Severity enum (INFO, WARNING, ERROR)
    - QualityIssue struct for tracking problems
    - QualityCriteria struct with configurable thresholds
    - QualityAssessment struct with scores and methods
    - Assessment functions: assess_sample_quality(), assess_coverage_quality(), assess_latency_quality(), assess_factor_quality(), assess_validation_quality()
    - assess_calibration_quality() - full assessment
    - print_quality_report() - formatted output
  - tests/calibration/calibration_quality_test.cpp - 29 assertions covering all quality assessment functions
  - tools/calibration/kpu-validate.cpp - Added --quality flag to show quality report

Quality Metrics Evaluated:

  | Category    | Checks                                               |
  |-------------|------------------------------------------------------|
  | Sample Size | Minimum 100 requests, recommended 1000               |
  | Coverage    | Balanced page hit/empty/conflict distribution        |
  | Latency     | Valid range (10-10000 cycles), reasonable R/W ratio  |
  | Factors     | Page hit < 1.0, page conflict > 1.0                  |
  | Validation  | Error thresholds for behavioral/transactional models |

---

All 6 calibration workflow phases are now complete:
  1. ✅ Calibration parameter schema and storage
  2. ✅ Behavioral calibration extraction
  3. ✅ Transactional calibration extraction
  4. ✅ kpu-calibrate CLI tool
  5. ✅ kpu-validate CLI tool
  6. ✅ Quality metrics and acceptance criteria
 
## Transactional Model Improvement Complete

The transactional memory controller now properly matches cycle-accurate throughput:

Results:

  | Metric             | Before  | After  |
  |--------------------|---------|--------|
  | Cycle Error        | 2013%   | 1.3%   |
  | Total Cycles (CA)  | 23,683  | 23,683 |
  | Total Cycles (Txn) | 500,500 | 23,990 |

Key Changes:

  1. include/sw/kpu/fidelity/component_config.hpp - Added per-scenario latency config fields (page_hit_latency, page_empty_latency, page_conflict_latency)
  2. include/sw/kpu/calibration/calibration_storage.hpp - Added per-scenario latency fields to TransactionalCalibration struct
  3. include/sw/kpu/calibration/calibration_extraction.hpp - Extract per-scenario latencies directly from CA statistics
  4. src/components/memory/transactional_memory_controller.cpp:
    - Fixed calculate_latency() to use physical timing parameters (service time)
    - Removed redundant queueing delay that double-counted contention
  5. src/calibration/calibration_storage.cpp - JSON serialization for new fields
  6. tools/calibration/kpu-validate.cpp - Apply per-scenario latencies from calibration

Key Insight: The transactional model's per-bank busy_until_cycle tracking already handles request serialization. Using end-to-end latencies from CA (which include queueing) caused double-counting. Using physical service times (tCL, tRCD, tRP) matches CA throughput correctly.

