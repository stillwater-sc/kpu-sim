# KPU Binary Disassembler

The `kpu-kpubin-disasm` tool inspects and displays the contents of KPU binary files in human-readable format. It supports both DMProgram files (`.kpubin`) and Kernel files (`.kpukernel`).

## Table of Contents

- [Overview](#overview)
- [Why Use the Disassembler](#why-use-the-disassembler)
- [Binary File Formats](#binary-file-formats)
- [How It Works](#how-it-works)
- [Command Reference](#command-reference)
- [How-To Guides](#how-to-guides)
  - [Inspect a Compiled Kernel](#inspect-a-compiled-kernel)
  - [Debug Data Movement Schedules](#debug-data-movement-schedules)
  - [Analyze Memory Traffic](#analyze-memory-traffic)
  - [Compare Kernel Configurations](#compare-kernel-configurations)
  - [Export to JSON for Scripting](#export-to-json-for-scripting)
  - [Verify MLP Kernel Configuration](#verify-mlp-kernel-configuration)

---

## Overview

The KPU simulator compiles computational graphs (matrix multiplications, MLPs, etc.) into **Data Movement Programs** that orchestrate data flow through the memory hierarchy. These programs are serialized to binary files for:

- Faster loading (no recompilation needed)
- Distribution of pre-compiled kernels
- Debugging and analysis of generated schedules

The disassembler reads these binary files and presents their contents in a structured, readable format.

### Supported File Types

| Extension | Format | Description |
|-----------|--------|-------------|
| `.kpubin` | DMProgram | Raw data movement program with instructions, memory map, and estimates |
| `.kpukernel` | Kernel | Complete kernel including metadata (op type, arguments, dimensions) plus embedded DMProgram |

---

## Why Use the Disassembler

### 1. Debug Compiler Output

When the tile optimizer or schedule generator produces unexpected results, the disassembler shows exactly what instructions were generated:

```
Instructions (9):
  [0   ] DMA_LOAD_TILE     A[0,0,0] ext:0x10000 -> L3[0]:0x0 (64 KB)
  [1   ] DMA_LOAD_TILE     B[0,0,0] ext:0x20000 -> L3[1]:0x0 (64 KB)
  [2   ] BARRIER
  ...
```

### 2. Understand Memory Layout

See how data is allocated across the memory hierarchy:

```
Memory Map:
  A base: 0x10000
  B base: 0x20000
  C base: 0x30000

  L3 Allocations (1):
    Tile[0] A offset=0x0 size=64 KB [buf0]
```

### 3. Validate Kernel Configuration

Verify that MLP kernels have the correct activation function and bias settings:

```
=== KPU Kernel: mlp_128x256x128_bias_gelu ===
Operation: mlp
Data Type: float32
Activation: gelu
Has Bias: yes
```

### 4. Performance Analysis

Review estimated cycle counts and memory traffic before running simulation:

```
Performance Estimates:
  Total cycles:         150,000
  External memory:      768 KB
  Arithmetic intensity: 42.67 FLOP/byte
  Estimated GFLOPS:     500.0
```

### 5. Integration Testing

Generate JSON output for automated verification in CI/CD pipelines.

---

## Binary File Formats

### DMProgram Binary Format (`.kpubin`)

```
[Header]
  magic:        4 bytes (0x4B505544 "KPUD")
  version:      4 bytes
  name_len:     4 bytes
  name:         name_len bytes
  M, N, K:      3 × 8 bytes (dimensions)
  Ti, Tj, Tk:   3 × 8 bytes (tile sizes)
  L1_Ki:        8 bytes
  dataflow:     1 byte (0=output_stationary, 1=weight_stationary)
  num_instr:    4 bytes

[Instructions]
  For each instruction:
    opcode:           1 byte
    operand_type:     1 byte
    earliest_cycle:   4 bytes
    deadline_cycle:   4 bytes
    instruction_id:   4 bytes
    num_deps:         4 bytes
    deps:             num_deps × 4 bytes
    label_len:        2 bytes
    label:            label_len bytes
    operands:         variable (depends on opcode)

[Memory Map]
  a_base, b_base, c_base: 3 × 8 bytes
  num_l3_allocs:    4 bytes
  l3_allocs:        variable
  num_l2_allocs:    4 bytes
  l2_allocs:        variable

[Estimates]
  total_cycles:         8 bytes
  external_mem_bytes:   8 bytes
  l3_bytes:             8 bytes
  l2_bytes:             8 bytes
  arithmetic_intensity: 8 bytes (double)
  estimated_gflops:     8 bytes (double)
```

### Kernel Binary Format (`.kpukernel`)

```
[Header]
  magic:        4 bytes (0x4B50554B "KPUK")
  version:      4 bytes
  name:         length-prefixed string
  op_type:      1 byte (0=unknown, 1=matmul, 2=conv2d, 3=elementwise, 4=mlp)
  dtype:        1 byte (0=float32, 1=float16, 2=bfloat16, 3=int8, 4=int4)
  M, N, K:      3 × 8 bytes
  Ti, Tj, Tk:   3 × 8 bytes
  L1_Ki:        8 bytes
  has_bias:     1 byte
  activation:   1 byte (0=none, 1=relu, 2=gelu, 3=sigmoid, 4=tanh, 5=silu)

[Arguments]
  num_args:     4 bytes
  For each argument:
    name:       length-prefixed string
    dtype:      1 byte
    is_output:  1 byte
    num_dims:   1 byte
    shape:      num_dims × 8 bytes
    size_bytes: 8 bytes

[Program]
  program_size: 4 bytes
  program_data: program_size bytes (embedded DMProgram)
```

---

## How It Works

### Detection and Loading

1. The tool reads the first 4 bytes to identify the magic number
2. Based on magic (`KPUD` or `KPUK`), it selects the appropriate deserializer
3. The binary is parsed into in-memory structures

### Instruction Decoding

Each instruction opcode is decoded to its symbolic name:

| Opcode | Name | Description |
|--------|------|-------------|
| `DMA_LOAD_TILE` | DMA Load | External memory → L3 |
| `DMA_STORE_TILE` | DMA Store | L3 → External memory |
| `BM_MOVE_TILE` | Block Mover | L3 ↔ L2 transfer |
| `STR_FEED_ROWS` | Streamer Feed Rows | L2 → L1 (row matrix) |
| `STR_FEED_COLS` | Streamer Feed Cols | L2 → L1 (column matrix) |
| `STR_DRAIN_OUTPUT` | Streamer Drain | L1 → L2 (output) |
| `BARRIER` | Barrier | Synchronization point |
| `HALT` | Halt | Program termination |

### Operand Display

Each instruction type has specialized operand formatting:

- **DMA**: Shows external address, L3 tile ID, offset, and transfer size
- **Block Mover**: Shows source/destination locations, dimensions, and transform
- **Streamer**: Shows L2 bank, L1 buffer, dimensions, and VE configuration

---

## Command Reference

```
Usage: kpu-kpubin-disasm <file> [options]

Options:
  -v, --verbose         Show detailed instruction operands
  -s, --summary         Show only header and statistics (no instructions)
  -i, --instructions    Show only instructions (no memory map or estimates)
  -m, --memory-map      Show only memory map
  -j, --json            Output in JSON format
  -h, --help            Show help message
```

### Output Modes

| Mode | Shows |
|------|-------|
| Default | Header, instruction summary, instructions, memory map, estimates |
| `--verbose` | All of the above plus detailed operand fields and labels |
| `--summary` | Header, instruction summary, estimates (no instruction list) |
| `--instructions` | Instructions only |
| `--memory-map` | Memory allocations only |
| `--json` | Full program/kernel as JSON |

---

## How-To Guides

### Inspect a Compiled Kernel

**Goal**: View the structure and configuration of a compiled matmul kernel.

```bash
# Compile and save a kernel (in your code)
# Kernel kernel = Kernel::create_matmul(512, 512, 512, DataType::FLOAT32);
# KernelSerializer().save(kernel, "matmul_512.kpukernel");

# Inspect the kernel
./build/tools/analysis/kpu-kpubin-disasm matmul_512.kpukernel
```

**Output**:
```
=== KPU Kernel: matmul_512x512x512_os ===
Operation: matmul
Data Type: float32
Dimensions: M=512, N=512, K=512
Tiles: Ti=64, Tj=64, Tk=96

Arguments (3):
  A        float32    [512 x 512] 1 MB (input)
  B        float32    [512 x 512] 1 MB (input)
  C        float32    [512 x 512] 1 MB (output)

Memory Footprint:
  Input:  2 MB
  Output: 1 MB
  FLOPs:  268,435,456
  Arithmetic Intensity: 85.33 FLOP/byte
```

**What to look for**:
- Tile sizes (Ti, Tj, Tk) should fit in L1/L2
- Arithmetic intensity indicates compute vs memory bound behavior
- Argument shapes match your expected dimensions

---

### Debug Data Movement Schedules

**Goal**: Understand why a kernel has unexpected performance.

```bash
./build/tools/analysis/kpu-kpubin-disasm kernel.kpukernel --verbose
```

**Output** (excerpt):
```
Instructions (2747):
  [0   ] DMA_LOAD_TILE     A[0,0,0] ext:0x0 -> L3[0]:0x0 (24 KB) [auto]  ; DMA_LOAD A_tile[0,0]
  [1   ] DMA_LOAD_TILE     B[0,0,0] ext:0x0 -> L3[0]:0x6000 (24 KB) [auto]  ; DMA_LOAD B_tile[0,0]
  [2   ] BARRIER             ; BARRIER
  [3   ] BM_MOVE_TILE      A[0,0,0] L3[0]:0x0 -> L2[0]:0x0 (64x96, identity) [elem=4B, auto]
  [4   ] BM_MOVE_TILE      B[0,0,0] L3[0]:0x6000 -> L2[0]:0x6000 (96x64, identity) [elem=4B, auto]
  ...
```

**What to look for**:
- Excessive `BARRIER` instructions indicate serialization
- Tile indices `[ti, tj, tk]` show iteration pattern
- Transfer sizes should match expected tile dimensions × element size
- Labels (in verbose mode) help identify which loop iteration

---

### Analyze Memory Traffic

**Goal**: Understand memory bandwidth requirements.

```bash
./build/tools/analysis/kpu-kpubin-disasm kernel.kpukernel --summary
```

**Output**:
```
Operations Summary:
  Total: 2747
  DMA:      160    (External <-> L3)
  BM:       768    (L3 <-> L2)
  Streamer: 832    (L2 <-> L1)
  Sync:     986
  Other:    1

Performance Estimates:
  Total cycles:         0
  External memory:      320 KB
  L3 traffic:           512 KB
  L2 traffic:           512 KB
  Arithmetic intensity: 25.60 FLOP/byte
```

**What to look for**:
- DMA count × average transfer size = external bandwidth
- High sync count may indicate over-synchronization
- L2/L3 traffic ratio shows cache effectiveness
- Arithmetic intensity < 10 suggests memory-bound execution

---

### Compare Kernel Configurations

**Goal**: Compare two kernels to understand configuration differences.

```bash
# Generate summary for both kernels
./build/tools/analysis/kpu-kpubin-disasm kernel_v1.kpukernel --summary > v1.txt
./build/tools/analysis/kpu-kpubin-disasm kernel_v2.kpukernel --summary > v2.txt

# Compare
diff v1.txt v2.txt
```

Or use JSON for programmatic comparison:

```bash
./build/tools/analysis/kpu-kpubin-disasm kernel_v1.kpukernel --json > v1.json
./build/tools/analysis/kpu-kpubin-disasm kernel_v2.kpukernel --json > v2.json

# Use jq to compare specific fields
jq '.tiles' v1.json v2.json
```

---

### Export to JSON for Scripting

**Goal**: Extract kernel information for automated analysis.

```bash
./build/tools/analysis/kpu-kpubin-disasm kernel.kpukernel --json > kernel.json
```

**Example: Extract instruction counts by type with jq**:

```bash
jq '[.instructions[].opcode] | group_by(.) | map({opcode: .[0], count: length})' kernel.json
```

**Output**:
```json
[
  {"opcode": "BARRIER", "count": 58},
  {"opcode": "BM_MOVE_TILE", "count": 32},
  {"opcode": "DMA_LOAD_TILE", "count": 16},
  {"opcode": "DMA_STORE_TILE", "count": 4},
  {"opcode": "HALT", "count": 1},
  {"opcode": "STR_DRAIN_OUTPUT", "count": 8},
  {"opcode": "STR_FEED_COLS", "count": 16},
  {"opcode": "STR_FEED_ROWS", "count": 16}
]
```

**Example: Validate tile sizes in CI**:

```bash
#!/bin/bash
Ti=$(jq '.tiles.Ti' kernel.json)
Tj=$(jq '.tiles.Tj' kernel.json)

if [ "$Ti" -gt 128 ] || [ "$Tj" -gt 128 ]; then
    echo "ERROR: Tile size exceeds L1 capacity"
    exit 1
fi
```

---

### Verify MLP Kernel Configuration

**Goal**: Confirm an MLP kernel has the correct activation and bias settings.

```bash
./build/tools/analysis/kpu-kpubin-disasm mlp_gelu.kpukernel
```

**Output**:
```
=== KPU Kernel: mlp_128x256x128_bias_gelu ===
Operation: mlp
Data Type: float32
Dimensions: M=128, N=256, K=128
Tiles: Ti=64, Tj=80, Tk=64
Activation: gelu
Has Bias: yes

Arguments (4):
  A        float32    [128 x 128] 64 KB (input)
  B        float32    [128 x 256] 128 KB (input)
  bias     float32    [256] 1 KB (input)
  C        float32    [128 x 256] 128 KB (output)
```

**What to verify**:
- `Operation: mlp` confirms MLP kernel type
- `Activation: gelu` shows correct activation function
- `Has Bias: yes` confirms bias addition is enabled
- Arguments include `bias` as the third input

**Check VE-enabled drain instructions** (verbose mode):

```bash
./build/tools/analysis/kpu-kpubin-disasm mlp_gelu.kpukernel --verbose | grep STR_DRAIN
```

If Vector Engine fusion is enabled, you'll see:
```
STR_DRAIN_OUTPUT  C[0,0,0] L2[0]:0x0 <-> L1[2]:0x0 (64x80) [VE: gelu, bias@0x5000]
```

---

## Troubleshooting

### "Invalid magic number" Error

The file is not a valid KPU binary. Possible causes:
- File is corrupted
- File is a different format (JSON, text, etc.)
- File was created with an incompatible version

### "Unsupported version" Error

The binary was created with a newer serializer version. Update your tools.

### Empty Instruction List

A program with no instructions is valid but unusual. Check:
- The kernel was properly compiled before serialization
- No error occurred during schedule generation

---

## Related Documentation

- [Data Movement ISA](design/data-movement-isa.md) - Instruction set reference
- [Kernel Compilation](compiler/kernel-compiler.md) - How kernels are compiled
- [Memory Hierarchy](design/memory-hierarchy.md) - L1/L2/L3 organization
