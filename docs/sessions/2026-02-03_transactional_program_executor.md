# Fidelity Elevation — Phases 2 & 3 Complete

**Date:** 2026-02-03
**Version:** v0.8.x
**Status:** Complete
**Tests:** 79/79 passing (47 new tests)

## 1. Summary

Implemented the `TransactionalProgramExecutor` that combines behavioral correctness
(via BehavioralProgramExecutor) with analytical timing models. This executor:

1. Executes programs functionally (real data movement, correct matmul results)
2. Computes timing estimates (when operations would complete on real hardware)
3. Generates Chrome Trace output for visualization in Perfetto

This completes Phase 2 of the fidelity elevation plan.

## 2. Scope

| Feature | Status | Tests |
|---------|--------|-------|
| TransactionalProgramExecutor class | DONE | N/A |
| ResourceTimeline (per-resource scheduling) | DONE | N/A |
| Timing models for DMA/BM/STR | DONE | N/A |
| TimingConfig (clocks, bus widths, latencies) | DONE | N/A |
| Chrome Trace export (Perfetto format) | DONE | N/A |
| ASCII timeline generation | DONE | N/A |
| Timing statistics (utilization, cycle counts) | DONE | N/A |
| Comprehensive test suite | DONE | 27/27 PASS |
| Full regression | DONE | 78/78 PASS |

## 3. Technical Decisions

**Decision 1: Two-Phase Execution (Behavioral then Timing)**
- **Choice:** Run behavioral executor to completion first, then iterate for timing
- **Alternatives Considered:** Interleaved execution with callbacks
- **Rationale:** Since behavioral operations are instant (memcpy), the timing
  overlay is independent. Simpler architecture with same correctness guarantees.
- **Files:** `src/software/isa/transactional_program_executor.cpp`

**Decision 2: ResourceTimeline for Per-Resource Scheduling**
- **Choice:** Track availability per DMA channel, BlockMover, Streamer independently
- **Alternatives Considered:** Single global timeline
- **Rationale:** Different resources can operate in parallel. Per-resource tracking
  enables accurate overlap modeling and utilization calculation.
- **Files:** `include/sw/kpu/isa/transactional_program_executor.hpp`

**Decision 3: Timing via Analytical Models (not Cycle-Accurate)**
- **Choice:** `cycles = startup_latency + bytes / bus_width`
- **Alternatives Considered:** Cycle-accurate simulation with queues
- **Rationale:** Transactional fidelity targets architecture exploration, not
  protocol-level accuracy. Analytical models are fast and sufficient for
  bottleneck identification and design space exploration.
- **Files:** `src/software/isa/transactional_program_executor.cpp`

**Decision 4: Chrome Trace Format for Visualization**
- **Choice:** Export directly to Chrome Trace JSON format
- **Alternatives Considered:** Reusing existing ChromeTraceExporter
- **Rationale:** TransactionalProgramExecutor has different event structure
  (timing events vs dataflow events). Direct export is simpler and matches
  the Phase 2 use case of visualizing tile movement timing.
- **Files:** `src/software/isa/transactional_program_executor.cpp`

## 4. Issues Encountered

**Issue 1: std::variant operand access**
- **Symptom:** Compilation error accessing `instr.operands.dma` directly
- **Root cause:** DMInstruction::operands is a `std::variant`, not a union
- **Fix:** Use `std::get<DMAOperands>(instr.operands)` for type-safe access

**Issue 2: Incorrect opcode names**
- **Symptom:** Compilation errors for BM_WRITEBACK, STR_DRAIN, etc.
- **Root cause:** Using outdated opcode names from gap-assessment.md
- **Fix:** Updated to correct names: BM_WRITEBACK_TILE, STR_DRAIN_OUTPUT

**Issue 3: TileCoord member names**
- **Symptom:** Compilation error accessing `tile.i`, `tile.j`
- **Root cause:** TileCoord uses `ti`, `tj`, `tk` member names
- **Fix:** Changed to `tile.ti`, `tile.tj`, `tile.tk`

## 5. Wrong Decisions

None in this session. The architecture worked correctly from the start.

## 6. Verification

```bash
# Transactional program executor tests
./build/tests/isa/test_transactional_program_executor
# Results: 27 passed, 0 failed

# Full test suite
ctest --test-dir build -j4
# 100% tests passed, 0 tests failed out of 78
```

## 7. Files Modified

| File | Action |
|------|--------|
| `include/sw/kpu/isa/transactional_program_executor.hpp` | CREATE |
| `src/software/isa/transactional_program_executor.cpp` | CREATE |
| `tests/isa/test_transactional_program_executor.cpp` | CREATE |
| `src/software/isa/CMakeLists.txt` | MODIFY — add transactional_program_executor.cpp |
| `tests/isa/CMakeLists.txt` | MODIFY — add test target |
| `CHANGELOG.md` | UPDATE |

## 8. Architecture Impact

This completes Phase 2 of the fidelity elevation plan:

```
Schedule DSL → compile_schedule() → DMProgram
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
              BehavioralProgramExecutor      TransactionalProgramExecutor
              (instant memcpy, real data)    (behavioral + timing overlay)
                       │                               │
                       ▼                               ▼
              Correct C = A × B             Correct C = A × B
              verified numerically                    +
                                            Cycle timeline
                                            Chrome Trace output
```

The TransactionalProgramExecutor proves that:
1. Compiled schedules produce correct numerical results (via behavioral tier)
2. Timing estimates are reasonable for the workload (via analytical models)
3. Visualization works for architecture exploration (via Chrome Trace)

## 9. Test Coverage

The test suite covers:

| Test Category | Tests | Description |
|---------------|-------|-------------|
| Construction | 2 | Default and custom timing config |
| Single-tile | 2 | Correctness and timing generation |
| Multi-tile | 2 | 2×2 tile grid correctness and timing |
| Export | 2 | Chrome Trace file creation and format |
| Timeline | 4 | ASCII timeline with all categories |
| Config | 1 | Different bus widths → different cycles |
| Identity | 1 | C = I × B = B verification |

## 10. Phase 3: Fidelity Switching Interface

After Phase 2, implemented the unified interface for fidelity switching.

### Phase 3 Scope

| Feature | Status | Tests |
|---------|--------|-------|
| IProgramExecutor interface | DONE | N/A |
| Factory function `create_program_executor()` | DONE | N/A |
| BehavioralExecutorWrapper | DONE | N/A |
| TransactionalExecutorWrapper | DONE | N/A |
| Fidelity switching tests | DONE | 20/20 PASS |
| Full regression | DONE | 79/79 PASS |

### Phase 3 Files

| File | Action |
|------|--------|
| `include/sw/kpu/isa/program_executor_interface.hpp` | CREATE |
| `src/software/isa/program_executor_interface.cpp` | CREATE |
| `tests/isa/test_program_executor_interface.cpp` | CREATE |
| `src/software/isa/CMakeLists.txt` | MODIFY |
| `tests/isa/CMakeLists.txt` | MODIFY |
| `CHANGELOG.md` | UPDATE |

### Usage Example

```cpp
// Create executor with desired fidelity
auto exec = create_program_executor(SimulationFidelity::TRANSACTIONAL, hw);

// Load and run program
exec->load_program(prog, a_base, b_base, c_base);
exec->run();

// Get results (both fidelities produce correct C = A × B)
std::vector<float> result(M * N);
hw.external_memory.read(c_base, result.data(), result.size() * sizeof(float));

// Transactional-only: get timing and export trace
if (exec->fidelity() == SimulationFidelity::TRANSACTIONAL) {
    std::cout << "Cycles: " << exec->total_cycles() << "\n";
    exec->export_trace("matmul_trace.json");
}

// Get statistics
std::cout << exec->statistics_string();
```

## 11. Next Steps

- **Python Integration** — Expose IProgramExecutor to Python via pybind11
- **Compute Timing** — Add systolic array timing model to TransactionalProgramExecutor
- **Conv2D/Softmax Schedules** — Extend DSL for additional kernel types
