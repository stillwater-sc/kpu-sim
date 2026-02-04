# Transactional Executor Loop Timing

**Date:** 2026-02-03
**Version:** v0.8.x
**Status:** Complete
**Tests:** 33/33 passing

## 1. Summary

Implemented loop execution in the TransactionalProgramExecutor with a timing model for
loop iteration overhead. The executor now follows actual control flow with PC-based
execution (including LOOP_BEGIN/LOOP_END branch behavior), tracks loop overhead cycles,
handles AUTO addressing opcodes using current loop state, and records loop events in
Chrome traces for visualization.

## 2. Scope

| Feature | Status | Tests |
|---------|--------|-------|
| LoopState tracking in transactional executor | DONE | N/A |
| Loop timing configuration | DONE | 1 test |
| PC-based execution with loops | DONE | 2 tests |
| AUTO addressing opcode timing | DONE | Implicit |
| Loop statistics in TimingStats | DONE | 3 tests |
| Loop events in Chrome trace | DONE | 1 test |
| Existing functionality preserved | DONE | 27 tests |

## 3. Technical Decisions

**Decision 1: Separate LoopState for Timing**
- **Choice:** Add a separate `timing_loop_state_` member rather than sharing with behavioral
- **Alternatives Considered:** Share loop state with behavioral executor
- **Rationale:** Behavioral executor runs to completion first, then timing is computed
  separately. Using shared state would interfere with behavioral execution.
- **Files:** `include/sw/kpu/isa/transactional_program_executor.hpp`

**Decision 2: PC-Based Execution for Timing**
- **Choice:** Modified `run()` to iterate with PC counter and handle loop opcodes with
  branch back behavior, identical to behavioral executor's control flow
- **Alternatives Considered:** Linear iteration with loop unrolling
- **Rationale:** Matches actual hardware execution semantics; loop overhead is only
  incurred when loops are actually executed, not when they're skipped
- **Files:** `src/software/isa/transactional_program_executor.cpp`

**Decision 3: Configurable Loop Latencies**
- **Choice:** Added four timing parameters to TimingConfig:
  - `loop_begin_latency` (default: 2 cycles) — Loop counter initialization
  - `loop_end_latency` (default: 1 cycle) — Counter check and decrement
  - `loop_branch_taken_latency` (default: 1 cycle) — Branch back overhead
  - `loop_branch_not_taken_latency` (default: 0 cycles) — Fall through (free)
- **Alternatives Considered:** Single fixed overhead per loop iteration
- **Rationale:** Allows modeling different loop machinery implementations;
  branch prediction effects can be captured with different taken/not-taken latencies
- **Files:** `include/sw/kpu/isa/transactional_program_executor.hpp`

**Decision 4: Loop Events in Chrome Trace**
- **Choice:** Record LOOP_BEGIN and LOOP_END as separate events with "loop" category
  and thread IDs 300+ (based on loop_id)
- **Alternatives Considered:** Omit loop events from trace; group with control events
- **Rationale:** Dedicated category allows filtering in Perfetto; separate thread IDs
  show loop nesting structure in timeline
- **Files:** `src/software/isa/transactional_program_executor.cpp`

## 4. Issues Encountered

**Issue 1: None**
- The implementation was straightforward, following the behavioral executor's pattern

## 5. Wrong Decisions

No wrong decisions identified this session. The implementation approach was validated
by following the existing behavioral executor pattern.

## 6. Verification

```bash
# Build
cmake --build --preset release --target kpu_isa test_transactional_program_executor

# Run tests
./build/tests/isa/test_transactional_program_executor

# Expected output:
# Passed: 33
# Failed: 0

# Key loop test outputs:
# test_loop_timing_overhead:
#   Loop overhead: 5 cycles
#   Loop iterations: 2
# test_nested_loop_timing:
#   Loop overhead: 15 cycles
#   Loop iterations: 6
# test_loop_timing_config:
#   Default loop overhead: 9 cycles
#   Slow loop overhead: 45 cycles
```

## 7. Files Modified

### Modified Files
- `include/sw/kpu/isa/transactional_program_executor.hpp`
  - Added `#include <sw/kpu/isa/loop_state.hpp>`
  - Added loop timing parameters to `TimingConfig`
  - Added `loop_overhead_cycles` and `loop_iterations` to `TimingStats`
  - Added `timing_loop_state_` member
  - Added `total_loop_overhead_cycles_` and `total_loop_iterations_` counters

- `src/software/isa/transactional_program_executor.cpp`
  - Reset loop state and counters in `load_program()`
  - Rewrote `run()` to use PC-based execution with loop handling
  - Added LOOP_BEGIN and LOOP_END timing in `run()` (before dispatch)
  - Added AUTO opcode cases in `dispatch_with_timing()`
  - Added configuration opcode cases (SET_BASE, SET_STRIDE, etc.)
  - Updated `get_opcode_name()` with all new opcodes
  - Updated `get_timing_stats()` to include loop statistics
  - Updated `generate_timeline()` to show loop overhead
  - Updated `export_chrome_trace()` to include loop metadata

- `tests/isa/test_transactional_program_executor.cpp`
  - Added `test_loop_timing_overhead()` — basic loop timing
  - Added `test_nested_loop_timing()` — 2x2 nested loops
  - Added `test_loop_timing_config()` — configurable latencies

- `CHANGELOG.md`
  - Added entry for TransactionalProgramExecutor Loop Execution

## 8. Next Steps

1. Test with actual assembly kernels using loops (e.g., `matmul_4096x1024x8192.kpuasm`)
2. Validate loop timing against expected iteration counts for large programs
3. Consider adding loop-level parallelism analysis to statistics
