# Loop Machinery and Address Generation ISA Extensions

**Date:** 2026-02-03
**Version:** v0.8.x
**Status:** In Progress (Phase 2 Complete)
**Tests:** 79/79 passing

## 1. Summary

Implemented comprehensive ISA extensions for loop machinery and address generation,
enabling compact parametric programs that express millions of tile operations in
~35 instructions. This is the foundation for large matmul support (e.g., 4096×1024×8192)
on the T256 hardware configuration.

The implementation includes:
- Loop counter registers with index role binding (TI, TJ, TK)
- Base address and stride registers per matrix
- AUTO addressing modes for DMA, BlockMover, and Streamer
- Address computation from loop indices at execution time
- Updated assembler to parse new opcodes
- Example large matmul program (32 instructions for 8.4M tile operations)

## 2. Scope

| Feature | Status | Tests |
|---------|--------|-------|
| IndexRole enum (TI, TJ, TK, NONE) | DONE | Compiles |
| New opcodes in DMOpcode enum | DONE | Compiles |
| LoopState class | DONE | Compiles |
| AddressGenerator class | DONE | Compiles |
| ISARegisterFile class | DONE | Compiles |
| Assembler: new opcode parsing | DONE | Manual |
| Factory methods for new instructions | DONE | Compiles |
| Large matmul assembly example | DONE | Assembles |
| Loop execution in behavioral executor | DONE | 23/23 pass |
| End-to-end validation | PENDING | — |

## 3. Technical Decisions

**Decision 1: Loop Counter Register Design**
- **Choice:** 8 loop registers (LC[0..7]) with count, limit, stride, and IndexRole binding
- **Alternatives Considered:** Single implicit loop; unlimited software loops
- **Rationale:** 8 registers handle typical 3-level nested loops plus future extensions
  (multi-level block matmul). Hardware loops are essential for address generation.
- **Files:** `include/sw/kpu/isa/loop_state.hpp`

**Decision 2: Index Role Binding**
- **Choice:** Each loop can bind to TI (row), TJ (column), or TK (reduction)
- **Alternatives Considered:** Implicit binding by loop nesting order
- **Rationale:** Explicit roles decouple loop order from tile index semantics,
  enabling different loop orderings (output-stationary, weight-stationary, etc.)
- **Files:** `include/sw/kpu/isa/data_movement_isa.hpp` (IndexRole enum)

**Decision 3: AUTO Addressing Mode**
- **Choice:** New *_AUTO opcodes compute addresses from loop indices at runtime
- **Alternatives Considered:** Inline address expressions; separate address opcode
- **Rationale:** Separate opcodes are explicit and self-documenting. Address
  computation logic is encapsulated in AddressGenerator class.
- **Files:** `include/sw/kpu/isa/address_generator.hpp`

**Decision 4: Stride Register Configuration**
- **Choice:** Per-matrix strides (row_stride, tile_i_stride, tile_j_stride)
- **Alternatives Considered:** Global strides; computed from matrix dimensions
- **Rationale:** Per-matrix strides support different layouts (row-major A,
  transposed B, etc.) and custom stride patterns for convolution.
- **Files:** `include/sw/kpu/isa/address_generator.hpp`

**Decision 5: Backward-Compatible LOOP_BEGIN**
- **Choice:** Optional IndexRole parameter; defaults to NONE for old programs
- **Alternatives Considered:** Breaking change; new LOOP_BEGIN_INDEXED opcode
- **Rationale:** Existing test kernels use LOOP_BEGIN with stride parameter.
  Parser checks if third argument is identifier (role) or number (stride).
- **Files:** `src/software/isa/assembler.cpp`

## 4. Issues Encountered

**Issue 1: Assembler lookahead bug**
- **Symptom:** `expected number` error when parsing `LOOP_BEGIN 0, 256, TI`
- **Root cause:** Used `lexer_->peek_token()` instead of `check(TokenType::IDENTIFIER)`
- **Fix:** Changed to use parser's `check()` method which examines current_token_

## 5. Wrong Decisions

No wrong decisions identified this session.

## 6. Verification

```bash
# Build
cmake --preset release && cmake --build --preset release
# 79/79 tests pass

# Assemble large matmul program
./build/tools/development/kpu-assembler \
  kernels/asm/matmul_4096x1024x8192.kpuasm \
  -o kernels/bin/matmul_4096x1024x8192.kpubin --stats
# Assembled 32 instructions (vs 8.4M tile operations!)

# Run behavioral executor test with loop machinery
./build/tests/isa/test_behavioral_program_executor
# All 23 tests pass, including loop machinery test:
#   - 8 computes (2×2×2 tiles)
#   - 14 loop iterations
#   - 16 DMA loads
#   - All 1024 elements correct (32.0)
```

## 7. Files Modified

| File | Action |
|------|--------|
| `include/sw/kpu/isa/data_movement_isa.hpp` | MODIFY — add new opcodes, IndexRole, operand structs |
| `src/software/isa/data_movement_isa.cpp` | MODIFY — add factory methods |
| `include/sw/kpu/isa/loop_state.hpp` | CREATE — LoopCounter, LoopState classes |
| `include/sw/kpu/isa/address_generator.hpp` | CREATE — AddressGenerator class |
| `include/sw/kpu/isa/register_file.hpp` | CREATE — ISARegisterFile class |
| `include/sw/kpu/isa/assembler.hpp` | MODIFY — add new parse method declarations |
| `src/software/isa/assembler.cpp` | MODIFY — add new opcode parsers |
| `kernels/asm/matmul_4096x1024x8192.kpuasm` | CREATE — large matmul example |
| `docs/plans/large_matmul_component_harnesses.md` | UPDATE — ISA extension design |
| `include/sw/kpu/isa/behavioral_program_executor.hpp` | MODIFY — add ISARegisterFile, AUTO dispatch methods |
| `src/software/isa/behavioral_program_executor.cpp` | MODIFY — PC-based execution, loop control, AUTO addressing |
| `tests/isa/test_behavioral_program_executor.cpp` | MODIFY — add loop machinery test |

## 8. Next Steps (Updated)

1. ~~**Implement loop execution in behavioral executor**~~ ✓ DONE
   - ~~Modify BehavioralProgramExecutor to use ISARegisterFile~~
   - ~~Execute LOOP_BEGIN/LOOP_END with proper PC control flow~~
   - ~~Compute addresses for AUTO opcodes using AddressGenerator~~

2. **Implement loop execution in transactional executor**
   - Timing model for loop iteration overhead
   - Track accumulated cycles across iterations

3. **Create component test harnesses**
   - DMA harness with memory controller
   - BlockMover harness with L3/L2
   - Streamer harness with L2/L1

4. **End-to-end validation**
   - Run large matmul on behavioral simulator
   - Verify C matrix correctness
   - Compare cycle estimates with analytical model

## 9. Architecture Impact

The loop machinery enables a **compact parametric program model**:

```
Previous: 8.4M explicit instructions (impossible to store)
    ↓
New:      ~35 instructions with loops (fits in small program memory)
```

This is critical for the two-level block matrix multiplier architecture:
- Level 1: Compute tile (16×16 systolic) executes element matmul
- Level 2: Checkerboard (16×16 tiles) executes block matmul via loops

The design scales to three-level hierarchy for multi-SoC configurations.
