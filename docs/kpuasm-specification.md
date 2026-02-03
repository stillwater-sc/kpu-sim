# KPU Assembly Language Specification

**Version:** 1.0
**Date:** 2026-02-03

## Overview

KPUASM is an assembly language for programming the KPU data movement ISA. Programs
written in KPUASM are assembled into `.kpubin` binary format that can be loaded
and executed by the behavioral and transactional simulators.

## File Format

- Extension: `.kpuasm`
- Encoding: UTF-8
- Line endings: LF or CRLF

## Syntax

### Comments

```asm
; Single-line comment (semicolon to end of line)
# Alternative single-line comment (hash to end of line)
```

### Labels

Labels mark locations in the instruction stream and can be referenced by jumps/loops.

```asm
label_name:
    DMA_LOAD_TILE A, (0,0,0), 0x1000, 0, 0, 4096, BUF_0
```

### Directives

Directives configure program metadata. All directives start with `.`:

```asm
.name "matmul_16x16x16"     ; Program name
.version 1                   ; Format version (always 1)
.dimensions M=16, N=16, K=16 ; Matrix dimensions
.tiling Ti=16, Tj=16, Tk=16  ; Tile dimensions
.l1_ki 16                    ; L1 streaming chunk
.dataflow output_stationary  ; Dataflow strategy: output_stationary|weight_stationary|input_stationary

; Memory base addresses (set by loader, can be overridden)
.a_base 0x0000
.b_base 0x1000
.c_base 0x2000
```

### Instructions

Each instruction occupies one line:

```asm
OPCODE [operands...]
```

Whitespace (spaces/tabs) separates operands. Commas are optional separators.

## Opcodes and Operand Formats

### DMA Operations

**DMA_LOAD_TILE** - Load tile from external memory to L3
```asm
DMA_LOAD_TILE matrix, tile_coord, ext_addr, l3_tile_id, l3_offset, size_bytes, buffer
; matrix:      A | B | C
; tile_coord:  (ti, tj, tk)
; ext_addr:    hex or decimal address
; l3_tile_id:  0-255
; l3_offset:   offset within L3 tile
; size_bytes:  transfer size
; buffer:      BUF_0 | BUF_1 | AUTO

; Example:
DMA_LOAD_TILE A, (0,0,0), 0x0000, 0, 0, 1024, BUF_0
```

**DMA_STORE_TILE** - Store tile from L3 to external memory
```asm
DMA_STORE_TILE matrix, tile_coord, ext_addr, l3_tile_id, l3_offset, size_bytes, buffer
; Same operands as DMA_LOAD_TILE
```

**DMA_PREFETCH_TILE** - Prefetch tile (non-blocking)
```asm
DMA_PREFETCH_TILE matrix, tile_coord, ext_addr, l3_tile_id, l3_offset, size_bytes, buffer
```

**DMA_LOAD_GATHER** - Strided gather for im2col
```asm
DMA_LOAD_GATHER matrix, tile_coord, ext_addr, l3_tile_id, l3_offset, size_bytes, buffer
```

**DMA_STORE_SCATTER** - Strided scatter
```asm
DMA_STORE_SCATTER matrix, tile_coord, ext_addr, l3_tile_id, l3_offset, size_bytes, buffer
```

### Block Mover Operations

**BM_MOVE_TILE** - Move tile L3 → L2 (identity)
```asm
BM_MOVE_TILE matrix, tile_coord, src_l3_tile, src_off, dst_l2_bank, dst_off, height, width, elem_size, buffer
; matrix:       A | B | C
; tile_coord:   (ti, tj, tk)
; src_l3_tile:  source L3 tile ID
; src_off:      source offset
; dst_l2_bank:  destination L2 bank ID
; dst_off:      destination offset
; height:       block height (rows)
; width:        block width (cols)
; elem_size:    element size in bytes (typically 4)
; buffer:       BUF_0 | BUF_1 | AUTO

; Example:
BM_MOVE_TILE A, (0,0,0), 0, 0, 0, 0, 16, 16, 4, BUF_0
```

**BM_TRANSPOSE_TILE** - Move tile L3 → L2 with transpose
```asm
BM_TRANSPOSE_TILE matrix, tile_coord, src_l3_tile, src_off, dst_l2_bank, dst_off, height, width, elem_size, buffer
```

**BM_WRITEBACK_TILE** - Move tile L2 → L3
```asm
BM_WRITEBACK_TILE matrix, tile_coord, src_l3_tile, src_off, dst_l2_bank, dst_off, height, width, elem_size, buffer
```

**BM_RESHAPE_TILE** - Move with block reshape
```asm
BM_RESHAPE_TILE matrix, tile_coord, src_l3_tile, src_off, dst_l2_bank, dst_off, height, width, elem_size, buffer
```

### Streamer Operations

**STR_FEED_ROWS** - Stream rows to systolic array (A matrix)
```asm
STR_FEED_ROWS matrix, tile_coord, l2_bank, l1_buffer, l2_addr, l1_addr, height, width, fabric_size, buffer
; matrix:       A | B | C
; tile_coord:   (ti, tj, tk)
; l2_bank:      L2 bank ID
; l1_buffer:    L1 buffer ID
; l2_addr:      L2 address
; l1_addr:      L1 address
; height:       matrix height
; width:        matrix width
; fabric_size:  systolic array size
; buffer:       BUF_0 | BUF_1 | AUTO

; Example:
STR_FEED_ROWS A, (0,0,0), 0, 0, 0, 0, 16, 16, 16, BUF_0
```

**STR_FEED_COLS** - Stream columns to systolic array (B matrix)
```asm
STR_FEED_COLS matrix, tile_coord, l2_bank, l1_buffer, l2_addr, l1_addr, height, width, fabric_size, buffer
```

**STR_DRAIN_OUTPUT** - Drain output from systolic array (C matrix)
```asm
STR_DRAIN_OUTPUT tile_coord, l2_bank, l1_buffer, l2_addr, l1_addr, height, width, fabric_size, buffer [, VE options]
; VE options (optional):
;   VE_ENABLE activation [, BIAS bias_addr]
;   activation: NONE | RELU | GELU | SIGMOID | TANH | SWISH

; Examples:
STR_DRAIN_OUTPUT (0,0,0), 0, 0, 0, 0, 16, 16, 16, BUF_0
STR_DRAIN_OUTPUT (0,0,0), 0, 0, 0, 0, 16, 16, 16, BUF_0, VE_ENABLE RELU
STR_DRAIN_OUTPUT (0,0,0), 0, 0, 0, 0, 16, 16, 16, BUF_0, VE_ENABLE RELU, BIAS 0x3000
```

**STR_BROADCAST_ROW** - Broadcast row to all PE columns
```asm
STR_BROADCAST_ROW matrix, tile_coord, l2_bank, l1_buffer, l2_addr, l1_addr, height, width, fabric_size, buffer
```

**STR_BROADCAST_COL** - Broadcast column to all PE rows
```asm
STR_BROADCAST_COL matrix, tile_coord, l2_bank, l1_buffer, l2_addr, l1_addr, height, width, fabric_size, buffer
```

### Vector Engine Operations

**VE_ELEMENTWISE** - Elementwise operation
```asm
VE_ELEMENTWISE operation
; operation: ADD | SUB | MUL | DIV | EXP | LOG | SQRT | ABS | NEG
```

**VE_REDUCE** - Reduction operation
```asm
VE_REDUCE operation
; operation: MAX | MIN | SUM | MEAN
```

### Synchronization Operations

**BARRIER** - Wait for all pending operations
```asm
BARRIER
```

**WAIT_DMA** - Wait for specific DMA completion
```asm
WAIT_DMA mask
; mask: bitmask of DMA operations to wait for (hex or decimal)
```

**WAIT_BM** - Wait for specific BlockMover completion
```asm
WAIT_BM mask
```

**WAIT_STR** - Wait for specific Streamer completion
```asm
WAIT_STR mask
```

**SIGNAL** - Signal completion token
```asm
SIGNAL signal_id
```

### Configuration Operations

**SET_TILE_SIZE** - Configure tile dimensions
```asm
SET_TILE_SIZE Ti, Tj, Tk, L1_Ki
```

**SET_BUFFER** - Configure double-buffer selection
```asm
SET_BUFFER buffer_id
```

**SET_STRIDE** - Configure address stride patterns
```asm
SET_STRIDE stride_m, stride_n, stride_k
```

### Loop Control

**LOOP_BEGIN** - Start hardware loop
```asm
LOOP_BEGIN loop_id, count, stride
; loop_id:  0-255
; count:    iteration count
; stride:   tile index stride per iteration
```

**LOOP_END** - End hardware loop
```asm
LOOP_END loop_id
```

### L2 Scratch Operations

**L2_SCRATCH_WRITE** - Write to L2 scratch region
```asm
L2_SCRATCH_WRITE offset, size
```

**L2_SCRATCH_READ** - Read from L2 scratch region
```asm
L2_SCRATCH_READ offset, size
```

### Special Operations

**NOP** - No operation
```asm
NOP
```

**HALT** - End of program
```asm
HALT
```

## Example Programs

### Minimal MatMul (16x16x16, single tile)

```asm
; matmul_single_tile.kpuasm
; C[16,16] = A[16,16] × B[16,16]

.name "matmul_single_tile"
.version 1
.dimensions M=16, N=16, K=16
.tiling Ti=16, Tj=16, Tk=16
.l1_ki 16
.dataflow output_stationary

.a_base 0x0000
.b_base 0x0400      ; 16*16*4 = 1024 = 0x400
.c_base 0x0800

; Load A and B tiles
DMA_LOAD_TILE A, (0,0,0), 0x0000, 0, 0, 1024, BUF_0
DMA_LOAD_TILE B, (0,0,0), 0x0400, 1, 0, 1024, BUF_0
BARRIER

; Move to L2
BM_MOVE_TILE A, (0,0,0), 0, 0, 0, 0, 16, 16, 4, BUF_0
BM_MOVE_TILE B, (0,0,0), 1, 0, 1, 0, 16, 16, 4, BUF_0
BARRIER

; Stream to systolic array
STR_FEED_ROWS A, (0,0,0), 0, 0, 0, 0, 16, 16, 16, BUF_0
STR_FEED_COLS B, (0,0,0), 1, 0, 0, 0, 16, 16, 16, BUF_0
; Compute happens reactively in PEs
BARRIER

; Drain and store result
STR_DRAIN_OUTPUT (0,0,0), 2, 0, 0, 0, 16, 16, 16, BUF_0
BARRIER
BM_WRITEBACK_TILE C, (0,0,0), 2, 0, 2, 0, 16, 16, 4, BUF_0
BARRIER
DMA_STORE_TILE C, (0,0,0), 0x0800, 2, 0, 1024, BUF_0
BARRIER

HALT
```

### Multi-Tile MatMul (32x32x16)

```asm
; matmul_multi_tile.kpuasm
; C[32,32] = A[32,16] × B[16,32]
; Tiled as 2×2 output tiles, 1 reduction tile

.name "matmul_multi_tile"
.version 1
.dimensions M=32, N=32, K=16
.tiling Ti=16, Tj=16, Tk=16
.l1_ki 16
.dataflow output_stationary

; Matrix layout in external memory:
; A[32,16]: rows 0-31, cols 0-15 = 32*16*4 = 2048 bytes at 0x0000
; B[16,32]: rows 0-15, cols 0-31 = 16*32*4 = 2048 bytes at 0x0800
; C[32,32]: rows 0-31, cols 0-31 = 32*32*4 = 4096 bytes at 0x1000

.a_base 0x0000
.b_base 0x0800
.c_base 0x1000

; =============================================
; Output tile (0,0): C[0:16, 0:16] = A[0:16, 0:16] × B[0:16, 0:16]
; =============================================
tile_0_0:
    DMA_LOAD_TILE A, (0,0,0), 0x0000, 0, 0, 1024, BUF_0  ; A[0:16, 0:16]
    DMA_LOAD_TILE B, (0,0,0), 0x0800, 1, 0, 1024, BUF_0  ; B[0:16, 0:16]
    BARRIER
    BM_MOVE_TILE A, (0,0,0), 0, 0, 0, 0, 16, 16, 4, BUF_0
    BM_MOVE_TILE B, (0,0,0), 1, 0, 1, 0, 16, 16, 4, BUF_0
    BARRIER
    STR_FEED_ROWS A, (0,0,0), 0, 0, 0, 0, 16, 16, 16, BUF_0
    STR_FEED_COLS B, (0,0,0), 1, 0, 0, 0, 16, 16, 16, BUF_0
    BARRIER
    STR_DRAIN_OUTPUT (0,0,0), 2, 0, 0, 0, 16, 16, 16, BUF_0
    BARRIER
    BM_WRITEBACK_TILE C, (0,0,0), 2, 0, 2, 0, 16, 16, 4, BUF_0
    BARRIER
    DMA_STORE_TILE C, (0,0,0), 0x1000, 2, 0, 1024, BUF_0
    BARRIER

; =============================================
; Output tile (0,1): C[0:16, 16:32]
; =============================================
tile_0_1:
    ; A[0:16, 0:16] already loaded, B needs new tile
    DMA_LOAD_TILE B, (0,1,0), 0x0840, 1, 0, 1024, BUF_0  ; B[0:16, 16:32]
    BARRIER
    BM_MOVE_TILE A, (0,0,0), 0, 0, 0, 0, 16, 16, 4, BUF_0
    BM_MOVE_TILE B, (0,1,0), 1, 0, 1, 0, 16, 16, 4, BUF_0
    BARRIER
    STR_FEED_ROWS A, (0,0,0), 0, 0, 0, 0, 16, 16, 16, BUF_0
    STR_FEED_COLS B, (0,1,0), 1, 0, 0, 0, 16, 16, 16, BUF_0
    BARRIER
    STR_DRAIN_OUTPUT (0,1,0), 2, 0, 0, 0, 16, 16, 16, BUF_0
    BARRIER
    BM_WRITEBACK_TILE C, (0,1,0), 2, 0, 2, 0, 16, 16, 4, BUF_0
    BARRIER
    DMA_STORE_TILE C, (0,1,0), 0x1040, 2, 0, 1024, BUF_0
    BARRIER

; =============================================
; Output tile (1,0): C[16:32, 0:16]
; =============================================
tile_1_0:
    DMA_LOAD_TILE A, (1,0,0), 0x0400, 0, 0, 1024, BUF_0  ; A[16:32, 0:16]
    DMA_LOAD_TILE B, (0,0,0), 0x0800, 1, 0, 1024, BUF_0  ; B[0:16, 0:16] (reuse from cache if available)
    BARRIER
    BM_MOVE_TILE A, (1,0,0), 0, 0, 0, 0, 16, 16, 4, BUF_0
    BM_MOVE_TILE B, (0,0,0), 1, 0, 1, 0, 16, 16, 4, BUF_0
    BARRIER
    STR_FEED_ROWS A, (1,0,0), 0, 0, 0, 0, 16, 16, 16, BUF_0
    STR_FEED_COLS B, (0,0,0), 1, 0, 0, 0, 16, 16, 16, BUF_0
    BARRIER
    STR_DRAIN_OUTPUT (1,0,0), 2, 0, 0, 0, 16, 16, 16, BUF_0
    BARRIER
    BM_WRITEBACK_TILE C, (1,0,0), 2, 0, 2, 0, 16, 16, 4, BUF_0
    BARRIER
    DMA_STORE_TILE C, (1,0,0), 0x1100, 2, 0, 1024, BUF_0
    BARRIER

; =============================================
; Output tile (1,1): C[16:32, 16:32]
; =============================================
tile_1_1:
    ; A[16:32, 0:16] already loaded
    DMA_LOAD_TILE B, (0,1,0), 0x0840, 1, 0, 1024, BUF_0  ; B[0:16, 16:32]
    BARRIER
    BM_MOVE_TILE A, (1,0,0), 0, 0, 0, 0, 16, 16, 4, BUF_0
    BM_MOVE_TILE B, (0,1,0), 1, 0, 1, 0, 16, 16, 4, BUF_0
    BARRIER
    STR_FEED_ROWS A, (1,0,0), 0, 0, 0, 0, 16, 16, 16, BUF_0
    STR_FEED_COLS B, (0,1,0), 1, 0, 0, 0, 16, 16, 16, BUF_0
    BARRIER
    STR_DRAIN_OUTPUT (1,1,0), 2, 0, 0, 0, 16, 16, 16, BUF_0
    BARRIER
    BM_WRITEBACK_TILE C, (1,1,0), 2, 0, 2, 0, 16, 16, 4, BUF_0
    BARRIER
    DMA_STORE_TILE C, (1,1,0), 0x1140, 2, 0, 1024, BUF_0
    BARRIER

HALT
```

## Assembler Command Line

```bash
kpu-assembler input.kpuasm -o output.kpubin
kpu-assembler input.kpuasm -o output.kpujson --format json
kpu-assembler input.kpuasm --print  # Print parsed instructions
```

## Error Messages

The assembler reports errors with line numbers:

```
input.kpuasm:15: error: unknown opcode 'DMA_LOAD'
input.kpuasm:20: error: invalid matrix ID 'D' (expected A, B, or C)
input.kpuasm:25: error: invalid tile coordinate format (expected (ti,tj,tk))
input.kpuasm:30: warning: address 0x10000 exceeds typical L3 capacity
```

## Notes

1. **Addresses are symbolic** - The `.a_base`, `.b_base`, `.c_base` directives set base
   addresses that the loader can relocate. Instruction addresses are relative to these bases.

2. **Buffer management** - `BUF_0` and `BUF_1` enable double-buffering. `AUTO` lets the
   hardware alternate automatically.

3. **Tile coordinates** - Always specified as `(ti, tj, tk)` even when a dimension is unused.
   Use 0 for unused dimensions.

4. **Element size** - The ISA assumes 4-byte float32 elements by default. Other sizes may
   be supported in future versions.

5. **Fabric size** - The systolic array dimension (e.g., 16 for a 16×16 array) is passed
   to streamer operations for correct feeding patterns.
