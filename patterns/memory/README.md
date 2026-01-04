# Memory Test and Characterization Patterns

Multi-fidelity simulation and calibration patterns for memory subsystems.

## Memory Technologies

| Directory | Technology | Status |
|-----------|------------|--------|
| `lpddr5/` | LPDDR5 (mobile/edge AI) | Active |
| `ddr5/` | DDR5 (server) | Planned |
| `hbm3/` | HBM3 (datacenter AI) | Planned |
| `hbm2/` | HBM2 (legacy) | Planned |

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
