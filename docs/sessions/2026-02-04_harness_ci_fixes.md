# Harness Test Infrastructure and CI Fixes

**Date:** 2026-02-04
**Version:** v0.8.x
**Status:** Complete
**Tests:** 84/84 passing

## 1. Summary

This session fixed multiple issues with the harness test infrastructure that were
causing CI failures, plus a Windows-specific path issue. The work included:

1. Chrome trace export fix to show human-readable thread names instead of numeric IDs
2. DMA harness completion callback fix for proper in-flight request tracking
3. BlockMover harness allocation vs reservation semantics fix
4. Journey tracking timing fixes (record arrivals at cycle+1 in behavioral mode)
5. Pipeline harness buffer allocation coordination fix
6. Windows CI fix for cross-platform temporary directory paths

## 2. Scope

| Feature | Status | Tests |
|---------|--------|-------|
| Chrome trace thread name metadata | DONE | 1 test |
| DMA harness in-flight tracking | DONE | 4 tests |
| BlockMover set_tile() method | DONE | 4 tests |
| Journey timing (cycle+1 for arrivals) | DONE | 12 tests |
| Pipeline harness buffer coordination | DONE | 4 tests |
| Windows cross-platform temp paths | DONE | 2 tests |

## 3. Technical Decisions

**Decision 1: Chrome Trace Thread Name Metadata**
- **Choice:** Add phase "M" (metadata) events at start of Chrome trace with thread_name
- **Alternatives Considered:** Rely on numeric thread IDs
- **Rationale:** Human-readable names (e.g., "DMA Channel 0", "BlockMover 0") make
  traces much more understandable in Perfetto/Chrome trace viewer
- **Files:** `src/software/isa/transactional_program_executor.cpp`

**Decision 2: Component Harness Buffer Allocation**
- **Choice:** Let component harnesses (DMA, BlockMover) allocate their own buffers
  internally, track allocated buffer IDs via completion callbacks
- **Alternatives Considered:** Pipeline harness pre-allocates and passes buffer IDs
- **Rationale:** `allocate()` marks buffer as occupied; passing that ID to component
  harness's `reserve()` fails because it's already occupied. Let components manage
  their own allocation and communicate results via callbacks.
- **Files:** `src/harness/pipeline_harness.cpp`, `src/harness/dma_harness.cpp`

**Decision 3: Journey Arrival Timing in Behavioral Mode**
- **Choice:** Record L3/L2/L1 arrivals at `current_cycle_ + 1` not `current_cycle_`
- **Alternatives Considered:** Keep at `current_cycle_`
- **Rationale:** In behavioral mode everything completes "instantly" at cycle 0.
  Using cycle+1 for arrivals ensures non-zero timestamps that indicate "end of tick"
  and allows journey tracking to work correctly.
- **Files:** `src/harness/block_mover_harness.cpp`, `src/harness/streamer_harness.cpp`

**Decision 4: Cross-Platform Temp Directory**
- **Choice:** Use `std::filesystem::temp_directory_path()` instead of hardcoded `/tmp/`
- **Alternatives Considered:** Conditional compilation for Windows
- **Rationale:** Standard library solution works on all platforms without #ifdefs
- **Files:** `tests/isa/test_transactional_program_executor.cpp`,
  `tests/isa/test_program_executor_interface.cpp`

## 4. Issues Encountered

**Issue 1: DMA harness transfers never completing**
- **Symptom:** `has_pending()` always returned true
- **Cause:** `in_flight_requests_` map was never cleared in completion callback
- **Fix:** Added `in_flight_requests_.erase(expected_transfer_id)` in callback

**Issue 2: BlockMover tile not resident in L2**
- **Symptom:** `tile_resident_l2()` returned false after move
- **Cause:** `allocate()` marks bank occupied but doesn't record tile_id, only
  `reserve()` did
- **Fix:** Added `set_tile()` method to L2BankArray, call after `allocate()`

**Issue 3: Journey tracking l2_arrival/l1_arrival = 0**
- **Symptom:** Journey validation failed with arrival time = 0
- **Cause:** In behavioral mode, arrivals recorded at cycle 0
- **Fix:** Record at `current_cycle_ + 1`

**Issue 4: Pipeline operations not completing**
- **Symptom:** Operations stuck, never completed
- **Cause:** Pipeline pre-allocated buffers, then component harnesses tried to
  reserve already-occupied buffers
- **Fix:** Pass `UINT32_MAX` to let component harnesses allocate, capture buffer
  IDs in completion callbacks

**Issue 5: Windows CI stack buffer overrun (0xc0000409)**
- **Symptom:** `test_transactional_program_executor` crashed on Windows CI
- **Cause:** Hardcoded `/tmp/` path doesn't exist on Windows
- **Fix:** Use `std::filesystem::temp_directory_path()` for cross-platform paths

## 5. Wrong Decisions

No wrong decisions this session. All fixes addressed actual bugs found through
CI test failures.

## 6. Verification

```bash
# Build
cmake --build --preset release

# Run all tests
ctest --test-dir build -j8

# Expected: 84/84 tests pass
# Key harness tests: 12/12 passing (schedule_validator_tests)
# Key ISA tests: 33/33 + 20/20 passing
```

## 7. Files Modified

### Modified Files
- `src/software/isa/transactional_program_executor.cpp`
  - Added Chrome trace thread name metadata events

- `src/harness/dma_harness.cpp`
  - Added `in_flight_requests_.erase()` in completion callback
  - Captured `l3_buffer_copy` for journey tracking

- `include/sw/kpu/harness/block_mover_harness.hpp`
  - Added `set_tile()` method to `L2BankArray` class

- `src/harness/block_mover_harness.cpp`
  - Called `set_tile()` after `allocate()`
  - Changed L2 arrival to record at `current_cycle_ + 1`

- `src/harness/streamer_harness.cpp`
  - Changed L1/compute arrivals to record at `current_cycle_ + 1`

- `src/harness/pipeline_harness.cpp`
  - Changed DMA_LOAD to pass `UINT32_MAX` for L3 buffer
  - Changed BM_L3_TO_L2 to pass `UINT32_MAX` for L2 bank
  - Added completion callbacks to track allocated buffer IDs

- `tests/isa/test_transactional_program_executor.cpp`
  - Changed `/tmp/` to `std::filesystem::temp_directory_path()`

- `tests/isa/test_program_executor_interface.cpp`
  - Added `#include <filesystem>`
  - Changed `/tmp/` paths to `std::filesystem::temp_directory_path()`

## 8. Commits

- `6ba41a5` Fix Windows CI: use cross-platform temp directory paths
- `73e1569` Fix harness test infrastructure
- `4c2d27c` Add run_matmul example and fix Chrome trace thread names

## 9. Next Steps

1. Monitor Windows CI to confirm fix works
2. Consider adding more platform-specific test coverage
3. Continue with data mover harness testing for schedule validation
