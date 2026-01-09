# Memory Test and Characterization Patterns

Multi-fidelity simulation and calibration patterns for memory subsystems.

## Memory Technologies

| Directory | Technology | Status |
|-----------|------------|--------|
| `ddr5/` | DDR5 (server) | Planned |
| `lpddr5/` | LPDDR5 (mobile/edge AI) | Active |
| `gddr6/` | GDDR6 (accelerators) | Active |
| `hbm2/` | HBM2 (legacy) | Active |
| `hbm3/` | HBM3 (datacenter AI) | Active |

## Multi-Fidelity Approach

Each memory technology supports three simulation fidelity levels:

```
BEHAVIORAL      → Fast functional simulation (~100-1000x)
TRANSACTIONAL   → Statistical timing model (~10-100x)
CYCLE_ACCURATE  → Full protocol simulation (reference)
```

Abstract models are calibrated against cycle-accurate results to maintain accuracy while enabling faster exploration.

## Pattern Categories

### Per-Memory Patterns

Each memory type follows the same pattern progression:

1. **Single Bank** - Fundamental timing (page hits, conflicts, turnaround)
2. **Two Bank** - Bank group constraints (same group vs different groups)
3. **Three Bank** - Multi-bank parallelism
4. **Four Bank** - tFAW and maximum parallelism
5. **Multi-Channel** - Channel independence and interleaving
6. **Complex** - Real-world access patterns (strided, random, tile loads)

## Building and Running

```bash
# Build all patterns
cmake --preset release && cmake --build --preset release

# Run LPDDR5 patterns
./build/patterns/memory/lpddr5/lpddr5_page_hits
./build/patterns/memory/lpddr5/lpddr5_page_conflicts
./build/patterns/memory/lpddr5/lpddr5_mixed_rw

# With multi-fidelity comparison
./build/patterns/memory/lpddr5/lpddr5_page_hits --fidelity
```

## Adding New Memory Technologies

1. Create directory structure under `memory/<technology>/`
2. Implement `<tech>_configs.hpp` with timing parameters
3. Implement `<tech>_harness.hpp` wrapping the memory controller
4. Create pattern subdirectories following the level progression
5. Update `CMakeLists.txt` with pattern targets

## Summary of Completed Work

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

