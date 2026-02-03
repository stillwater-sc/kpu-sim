# Behavioral Program Executor — Phase 1 Fidelity Elevation

**Date:** 2026-02-02
**Version:** v0.8.x
**Status:** Complete
**Tests:** 77/77 passing

## 1. Summary

Implemented the `BehavioralProgramExecutor` that interprets DMProgram instruction streams
using real temporal memory components (L3Tile, L2Bank, L1Buffer, ExternalMemory). This
bridges the gap between the Schedule DSL and functional verification — schedules compiled
to DMProgram can now be executed with actual float data to verify correctness.

## 2. Scope

| Feature | Status | Tests |
|---------|--------|-------|
| BehavioralProgramExecutor implementation | DONE | N/A |
| DMProgram interpretation (all opcodes) | DONE | N/A |
| Strided DMA transfers for tiled matrices | DONE | N/A |
| End-to-end matmul correctness tests | DONE | 19/19 PASS |
| Schedule compiler bug fixes | DONE | N/A |
| Full regression suite | DONE | 77/77 PASS |

## 3. Technical Decisions

**Decision 1: Executor uses temporal memory components directly**
- **Choice:** BehavioralProgramExecutor operates on L3Tile, L2Bank, L1Buffer, ExternalMemory
- **Alternatives Considered:** Creating new abstract memory interface
- **Rationale:** Temporal components already have the right API (read/write with byte
  buffers). Direct use avoids abstraction overhead and proves the components work correctly.
- **Files:** `include/sw/kpu/isa/behavioral_program_executor.hpp`,
  `src/software/isa/behavioral_program_executor.cpp`

**Decision 2: Strided DMA transfers for row-major external memory**
- **Choice:** DMA load/store perform row-by-row transfers with proper stride
- **Alternatives Considered:** Contiguous transfers (original implementation)
- **Rationale:** Tiled matrices in external memory are stored row-major with stride = N,
  not Ti. Without strided transfers, only single-tile cases work correctly.
- **Files:** `src/software/isa/behavioral_program_executor.cpp` (dispatch_dma_load/store)

**Decision 3: Base address resolution at execution time**
- **Choice:** Schedule compiler generates offsets relative to 0; executor adds base addresses
- **Alternatives Considered:** Embedding absolute addresses in DMProgram
- **Rationale:** Schedules should be reusable with different input addresses. The
  `load_program(prog, a_base, b_base, c_base)` API allows this flexibility.
- **Files:** `src/software/isa/behavioral_program_executor.cpp`

## 4. Issues Encountered

**Issue 1: WRITEBACK using wrong L3 offset**
- **Symptom:** BM_WRITEBACK wrote to L3 offset 0, but DMA_STORE read from offset 1024
- **Root cause:** Schedule compiler hardcoded `dst_offset = 0` instead of using TileLayout
- **Fix:** Changed to `dst_offset = loc_wb.address` in WRITEBACK code

**Issue 2: str_drain argument order wrong**
- **Symptom:** All STR_DRAIN operations used L2 bank 0 regardless of tile
- **Root cause:** str_drain call had `(tile, 0, loc.l2_bank_id, ...)` but signature is
  `(tile, l2_bank, l1_buf, ...)`
- **Fix:** Corrected to `(tile, loc.l2_bank_id, 0, ...)`

**Issue 3: Multi-tile matmul had partial correct results**
- **Symptom:** First ~9 rows of each tile correct, rest were 0
- **Root cause:** DMA load/store did contiguous transfers, but external memory layout
  for tiled matrices requires strided access (stride = N*sizeof(float), not Ti*sizeof(float))
- **Fix:** Implemented row-by-row strided transfers in dispatch_dma_load/store

## 5. Wrong Decisions

None in this session. All initial designs worked correctly once the three bugs above
were fixed.

## 6. Verification

```bash
# Behavioral program executor tests
./build/tests/isa/test_behavioral_program_executor
# Results: 19 passed, 0 failed

# Full test suite
ctest --test-dir build -j4
# 100% tests passed, 0 tests failed out of 77
```

## 7. Files Modified

| File | Action |
|------|--------|
| `include/sw/kpu/isa/behavioral_program_executor.hpp` | CREATE |
| `src/software/isa/behavioral_program_executor.cpp` | CREATE |
| `tests/isa/test_behavioral_program_executor.cpp` | CREATE |
| `src/software/isa/CMakeLists.txt` | MODIFY — add behavioral_program_executor.cpp |
| `tests/isa/CMakeLists.txt` | MODIFY — add test target |
| `src/dsl/schedule_compiler.cpp` | MODIFY — fix WRITEBACK offset, str_drain args |
| `docs/07-fidelity-elevation/gap-assessment.md` | CREATE |
| `CHANGELOG.md` | UPDATE |

## 8. Architecture Impact

This completes Phase 1 of the fidelity elevation plan:

```
Schedule DSL → compile_schedule() → DMProgram → BehavioralProgramExecutor → Verified Results
                                                        ↓
                                              Uses temporal memory:
                                              - ExternalMemory (DRAM)
                                              - L3Tile (on-chip L3)
                                              - L2Bank (on-chip L2)
                                              - L1Buffer (compute fabric)
```

The executor proves that compiled schedules are functionally correct: the right tiles
reach the right places in the right order, and accumulated matmul results match reference.

## 9. Next Steps

- **Phase 2: Transactional Timing Overlay** — Add queue-based timing model to track
  latencies without changing functional behavior
- **Phase 3: Integration** — Connect BehavioralProgramExecutor to the Python runtime
  for full end-to-end DSL → compile → execute → verify workflow
