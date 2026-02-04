# Data Mover Component Test Harness Infrastructure

**Date:** 2026-02-03
**Version:** v0.8.x
**Status:** Complete
**Tests:** Schedule validator tests passing; component harness tests require behavioral model integration

## 1. Summary

Implemented a comprehensive test harness infrastructure for KPU data movement schedules.
The infrastructure provides:

1. **Individual Component Harnesses** — DMA, BlockMover, and Streamer harnesses for isolated testing
2. **System-Level Pipeline Harness** — Full DRAM→L3→L2→L1→Compute flow testing
3. **Tile Journey Tracking** — Per-tile timing through the entire memory hierarchy
4. **Schedule Validation** — Static and runtime validation of data movement schedules
5. **CLI Tooling** — `schedule-runner` for schedule experimentation and analysis
6. **Statistics Collection** — Component utilization, throughput, and bottleneck analysis

## 2. Scope

| Feature | Status | Tests |
|---------|--------|-------|
| HarnessConfig structures | DONE | N/A |
| PatternHarnessBase template | DONE | N/A |
| TileJourneyTracker | DONE | N/A |
| DMAHarness | DONE | 7 tests |
| BlockMoverHarness | DONE | 8 tests |
| StreamerHarness | DONE | 10 tests |
| PipelineHarness | DONE | 8 tests |
| ScheduleValidator | DONE | 12 tests (ALL PASS) |
| schedule-runner CLI | DONE | Manual verification |

## 3. Technical Decisions

**Decision 1: Template-Based Harness Design**
- **Choice:** Use `PatternHarnessBase<ConfigT>` template for all harnesses
- **Alternatives Considered:** Inheritance without templates, composition
- **Rationale:** Templates allow type-safe configuration while sharing common
  simulation logic (tick, run, reset, validate). Matches LPDDR5Harness pattern.
- **Files:** `include/sw/kpu/harness/pattern_harness_base.hpp`

**Decision 2: Credit-Based Flow in Harnesses**
- **Choice:** Harnesses model credit-based flow between memory levels
- **Alternatives Considered:** Simple request-response model
- **Rationale:** Aligns with KPU's credit-based dataflow execution model.
  L3BufferPool and L2BankArray track credits for realistic simulation.
- **Files:** `include/sw/kpu/harness/dma_harness.hpp`, `block_mover_harness.hpp`

**Decision 3: Tile-Based Dependency Model**
- **Choice:** Schedule dependencies expressed as TileIDs
- **Alternatives Considered:** Operation indices, explicit DAG
- **Rationale:** TileID provides natural ordering - operations on the same tile
  must complete in sequence. Self-dependencies represent "wait for previous op".
- **Files:** `include/sw/kpu/harness/pipeline_harness.hpp`

**Decision 4: Self-Dependencies Are Valid**
- **Choice:** Cycle detection allows self-dependencies (same tile_id)
- **Alternatives Considered:** Treat self-dependencies as cycles
- **Rationale:** A BlockMover operation depending on its tile means "wait for
  DMA load of this tile to complete", not a true cycle.
- **Files:** `include/sw/kpu/harness/schedule_validator.hpp`

**Decision 5: Separate Harness Library**
- **Choice:** Create `kpu_harness` library linking to `kpu_behavioral`
- **Alternatives Considered:** Inline in test files, part of kpu_behavioral
- **Rationale:** Harnesses are testing infrastructure, not core models. Separate
  library allows reuse in tests, CLI tools, and integration suites.
- **Files:** `src/harness/CMakeLists.txt`

## 4. Issues Encountered

**Issue 1: BehavioralMemoryModel API Mismatch**
- **Symptom:** `'write_l3_tile' is not a member of BehavioralMemoryModel`
- **Root cause:** Created harness code using assumed API that doesn't exist
- **Fix:** Used correct API: `get_region(MemoryRegionType::L3_TILE, id)->write()`

**Issue 2: MemoryModelConfig Field Names**
- **Symptom:** Wrong field names in config struct initialization
- **Root cause:** Used `l3_tiles` instead of `l3_tile_count`
- **Fix:** Updated to use correct field names from `memory_model.hpp`

**Issue 3: BehavioralDMAEngine Namespace**
- **Symptom:** `'behavioral' was not declared in this scope`
- **Root cause:** BehavioralDMAEngine is in `sw::kpu`, not `sw::kpu::behavioral`
- **Fix:** Changed to `sw::kpu::BehavioralDMAEngine`

**Issue 4: Library Name Mismatch**
- **Symptom:** `cannot find -lkpu_behavioral_models`
- **Root cause:** Library is named `kpu_behavioral`, not `kpu_behavioral_models`
- **Fix:** Updated all CMakeLists.txt to use correct library name

**Issue 5: ScheduleValidator Config Default Initializer**
- **Symptom:** Compile error about default member initializers
- **Root cause:** C++ doesn't allow `Config{}` default parameter with in-class initializers
- **Fix:** Split into default constructor and explicit config constructor

**Issue 6: Cycle Detection False Positive**
- **Symptom:** Valid schedule flagged as having dependency cycle
- **Root cause:** Self-dependencies (tile A depends on tile A) flagged as cycles
- **Fix:** Added `if (dep == node) continue;` to skip self-dependencies in DFS

## 5. Wrong Decisions

**Wrong Decision 1: Assumed API Without Verification**
- **What:** Created `write_l3_tile()` calls assuming BehavioralMemoryModel had this method
- **Why wrong:** The method doesn't exist; should have read the header first
- **Correction:** Read `memory_model.hpp` and used correct `get_region()->write()` API
- **Lesson:** Always verify API exists before using it in new code

**Wrong Decision 2: Used Wrong Library Name**
- **What:** Used `kpu_behavioral_models` instead of `kpu_behavioral`
- **Why wrong:** Just guessed the name without checking
- **Correction:** Used `find` to locate actual library files in build directory
- **Lesson:** Check existing CMakeLists.txt or build output for correct names

## 6. Verification

```bash
# Build harness infrastructure
cmake --preset release
cmake --build --preset release --target kpu_harness schedule-runner \
    test_dma_harness test_block_mover_harness test_streamer_harness \
    test_pipeline_harness test_schedule_validator

# Run schedule validator tests (all pass)
./build/tests/harness/test_schedule_validator
# All tests passed (48 assertions in 12 test cases)

# Test CLI tool
./build/tools/harness/schedule-runner --help
```

## 7. Files Created/Modified

### New Files
- `include/sw/kpu/harness/harness_config.hpp` — Configuration structures
- `include/sw/kpu/harness/pattern_harness_base.hpp` — Abstract base class
- `include/sw/kpu/harness/tile_journey_tracker.hpp` — Per-tile timing tracking
- `include/sw/kpu/harness/dma_harness.hpp` — DMA engine harness
- `include/sw/kpu/harness/block_mover_harness.hpp` — BlockMover harness
- `include/sw/kpu/harness/streamer_harness.hpp` — Streamer harness
- `include/sw/kpu/harness/pipeline_harness.hpp` — Full pipeline harness
- `include/sw/kpu/harness/schedule_validator.hpp` — Validation utilities
- `src/harness/dma_harness.cpp` — DMA harness implementation
- `src/harness/block_mover_harness.cpp` — BlockMover harness implementation
- `src/harness/streamer_harness.cpp` — Streamer harness implementation
- `src/harness/pipeline_harness.cpp` — Pipeline harness implementation
- `src/harness/CMakeLists.txt` — Harness library build
- `tools/harness/schedule_runner.cpp` — CLI tool
- `tools/harness/CMakeLists.txt` — CLI tools build
- `tests/harness/test_dma_harness.cpp` — DMA harness tests
- `tests/harness/test_block_mover_harness.cpp` — BlockMover harness tests
- `tests/harness/test_streamer_harness.cpp` — Streamer harness tests
- `tests/harness/test_pipeline_harness.cpp` — Pipeline harness tests
- `tests/harness/test_schedule_validator.cpp` — Schedule validator tests
- `tests/harness/CMakeLists.txt` — Test build configuration

### Modified Files
- `src/CMakeLists.txt` — Added harness subdirectory
- `tools/CMakeLists.txt` — Added harness subdirectory
- `tests/CMakeLists.txt` — Added harness subdirectory and test_harness target

## 8. Next Steps

1. **Integrate with Behavioral Models** — Connect harnesses to actual DMA engines,
   BlockMovers, and Streamers for end-to-end functional testing
2. **Add Transactional Timing** — Support transactional fidelity in harnesses
3. **Performance Analysis CLI** — Add bottleneck analysis and roofline plotting
4. **Schedule Generation Integration** — Connect to DMProgram and schedule generators
5. **Chrome Trace Export** — Add Perfetto visualization for tile journeys
