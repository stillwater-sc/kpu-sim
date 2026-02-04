# Kernel Execution Methodology

This document describes how to execute KPU assembly kernels using the behavioral and
transactional simulation fidelities.

## Overview

The KPU simulator supports a two-stage execution pipeline:

1. **Assembly** — Convert `.kpuasm` source files to `.kpubin` binary programs
2. **Execution** — Run the binary on a behavioral or transactional simulator

### Simulation Fidelities

| Fidelity | Purpose | Speed | Output |
|----------|---------|-------|--------|
| **Behavioral** | Functional correctness | ~100-1000x | Computed tensor values |
| **Transactional** | Performance analysis | ~10-100x | Timing + Chrome trace |

**Behavioral fidelity** executes operations as instant memcpy and matmul, producing
correct numerical results without timing information.

**Transactional fidelity** wraps behavioral execution with analytical timing models,
producing both correct results and performance traces for visualization.

## Available Kernels

Three example kernels demonstrate different compute patterns:

| Kernel | File | Dimensions | Description |
|--------|------|------------|-------------|
| MatMul | `matmul_16x16x16.kpuasm` | 16×16×16 | Single-tile matrix multiplication |
| Conv2D | `conv2d_im2col.kpuasm` | 36×8×36 | 2D convolution via im2col + matmul |
| Softmax | `softmax_batch.kpuasm` | 4×16 | Multi-pass softmax with Vector Engine |

## Prerequisites

Build the KPU toolchain:

```bash
cmake --preset release
cmake --build --preset release
```

This produces:
- `build/tools/development/kpu-assembler` — Assembly tool
- `build/tools/runtime/kpu-loader` — Execution tool

## Step 1: Assemble the Kernel

Convert assembly source to binary format:

```bash
# MatMul kernel
./build/tools/development/kpu-assembler \
    kernels/asm/matmul_16x16x16.kpuasm \
    -o matmul_16x16x16.kpubin

# Conv2D kernel
./build/tools/development/kpu-assembler \
    kernels/asm/conv2d_im2col.kpuasm \
    -o conv2d_im2col.kpubin

# Softmax kernel
./build/tools/development/kpu-assembler \
    kernels/asm/softmax_batch.kpuasm \
    -o softmax_batch.kpubin
```

### Inspect Assembly Output

View parsed program without generating binary:

```bash
./build/tools/development/kpu-assembler \
    kernels/asm/matmul_16x16x16.kpuasm \
    --print --stats
```

Output:
```
Program: matmul_16x16x16
Dimensions: M=16 N=16 K=16
Tiling: Ti=16 Tj=16 Tk=16
Instructions: 12
  DMA: 3
  BM:  3
  STR: 3
  Sync: 3
```

## Step 2: Prepare Input Data

Create binary tensor files (raw float32 format):

### Python Example

```python
import numpy as np

# MatMul: C[16,16] = A[16,16] × B[16,16]
A = np.random.randn(16, 16).astype(np.float32)
B = np.random.randn(16, 16).astype(np.float32)
C_expected = A @ B

A.tofile('A_16x16.bin')
B.tofile('B_16x16.bin')
C_expected.tofile('C_expected_16x16.bin')

# Conv2D: im2col matrices
A_col = np.random.randn(36, 36).astype(np.float32)  # im2col patches
B_w = np.random.randn(36, 8).astype(np.float32)     # weight matrix
C_out = np.maximum(0, A_col @ B_w)                   # ReLU activation

A_col.tofile('A_col_36x36.bin')
B_w.tofile('B_w_36x8.bin')
C_out.tofile('C_out_expected.bin')

# Softmax: Y[4,16] = softmax(X[4,16])
X = np.random.randn(4, 16).astype(np.float32)
Y_expected = np.exp(X - X.max(axis=1, keepdims=True))
Y_expected /= Y_expected.sum(axis=1, keepdims=True)

X.tofile('X_4x16.bin')
Y_expected.tofile('Y_expected_4x16.bin')
```

## Step 3: Execute with Behavioral Fidelity

Behavioral execution validates functional correctness:

```bash
# MatMul — full execution with I/O
./build/tools/runtime/kpu-loader matmul_16x16x16.kpubin \
    --fidelity behavioral \
    --input-a A_16x16.bin \
    --input-b B_16x16.bin \
    --output-c C_result.bin \
    --stats

# Conv2D
./build/tools/runtime/kpu-loader conv2d_im2col.kpubin \
    --fidelity behavioral \
    --input-a A_col_36x36.bin \
    --input-b B_w_36x8.bin \
    --output-c C_conv_result.bin \
    --stats

# Softmax
./build/tools/runtime/kpu-loader softmax_batch.kpubin \
    --fidelity behavioral \
    --input-a X_4x16.bin \
    --output-c Y_result.bin \
    --stats
```

### Validate Results (Python)

```python
import numpy as np

# Load KPU output
C_kpu = np.fromfile('C_result.bin', dtype=np.float32).reshape(16, 16)
C_expected = np.fromfile('C_expected_16x16.bin', dtype=np.float32).reshape(16, 16)

# Compare
max_diff = np.abs(C_kpu - C_expected).max()
print(f"Max difference: {max_diff}")
assert max_diff < 1e-5, "Results don't match!"
print("✓ Behavioral execution correct")
```

## Step 4: Execute with Transactional Fidelity

Transactional execution adds timing analysis:

```bash
# MatMul with trace export
./build/tools/runtime/kpu-loader matmul_16x16x16.kpubin \
    --fidelity transactional \
    --input-a A_16x16.bin \
    --input-b B_16x16.bin \
    --output-c C_result.bin \
    --trace matmul_trace.json \
    --stats -v

# Conv2D with trace
./build/tools/runtime/kpu-loader conv2d_im2col.kpubin \
    --fidelity transactional \
    --input-a A_col_36x36.bin \
    --input-b B_w_36x8.bin \
    --output-c C_conv_result.bin \
    --trace conv2d_trace.json \
    --stats

# Softmax with trace
./build/tools/runtime/kpu-loader softmax_batch.kpubin \
    --fidelity transactional \
    --input-a X_4x16.bin \
    --output-c Y_result.bin \
    --trace softmax_trace.json \
    --stats
```

### Example Statistics Output

```
Execution Statistics:
  Total cycles: 1247
  Instructions executed: 12
  DMA operations: 3 (load: 2, store: 1)
  BlockMover operations: 3
  Streamer operations: 3
  Bytes transferred: 4096
  Compute operations: 4096 FLOPs
```

## Step 5: Visualize Performance Traces

### Chrome Trace Format

Open traces in Perfetto or Chrome's `chrome://tracing`:

1. Navigate to https://ui.perfetto.dev
2. Click "Open trace file"
3. Select `matmul_trace.json`

The trace shows:
- **DMA channel activity** — DRAM transfers
- **BlockMover timeline** — L3→L2 tile movements
- **Streamer timeline** — L2→L1 streaming
- **Compute activity** — Systolic array utilization

### ASCII Timeline (Terminal)

The `--stats -v` flags print an ASCII timeline:

```
Timeline:
  0    100   200   300   400   500   600   700   800   900  1000
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
DMA [====         ][====         ]                       [====]
BM       [====]        [====]
STR           [==========]
COMP               [====================]
```

## Dry Run Mode

Validate kernel without execution:

```bash
./build/tools/runtime/kpu-loader matmul_16x16x16.kpubin --dry-run -v
```

Output:
```
Program: matmul_16x16x16
Dimensions: M=16 N=16 K=16
Tiling: Ti=16 Tj=16 Tk=16
Instructions: 12
  DMA: 3
  BM:  3
  STR: 3
  Sync: 3

Memory layout:
  A base: 0x0000
  B base: 0x0400
  C base: 0x0800

Dry run: Program validated successfully
```

## Complete Workflow Example

End-to-end example for MatMul 16×16×16:

```bash
#!/bin/bash
set -e

# 1. Create test data
python3 - << 'EOF'
import numpy as np
A = np.eye(16, dtype=np.float32)  # Identity matrix
B = np.arange(256, dtype=np.float32).reshape(16, 16)
A.tofile('A.bin')
B.tofile('B.bin')
(A @ B).tofile('C_expected.bin')
EOF

# 2. Assemble
./build/tools/development/kpu-assembler \
    kernels/asm/matmul_16x16x16.kpuasm \
    -o matmul.kpubin

# 3. Execute behavioral (fast, correct values)
./build/tools/runtime/kpu-loader matmul.kpubin \
    --fidelity behavioral \
    --input-a A.bin --input-b B.bin --output-c C_beh.bin \
    --stats

# 4. Execute transactional (timing + trace)
./build/tools/runtime/kpu-loader matmul.kpubin \
    --fidelity transactional \
    --input-a A.bin --input-b B.bin --output-c C_txn.bin \
    --trace trace.json --stats -v

# 5. Validate results
python3 - << 'EOF'
import numpy as np
C_exp = np.fromfile('C_expected.bin', dtype=np.float32)
C_beh = np.fromfile('C_beh.bin', dtype=np.float32)
C_txn = np.fromfile('C_txn.bin', dtype=np.float32)
print(f"Behavioral max diff: {np.abs(C_exp - C_beh).max():.2e}")
print(f"Transactional max diff: {np.abs(C_exp - C_txn).max():.2e}")
EOF

echo "✓ Workflow complete. View trace.json in Perfetto."
```

## Kernel Memory Layouts

### MatMul 16×16×16

```
Address   Size    Content
0x0000    1024    A[16,16] input matrix
0x0400    1024    B[16,16] input matrix
0x0800    1024    C[16,16] output matrix
```

### Conv2D im2col

```
Address   Size    Content
0x0000    5184    A_col[36,36] im2col patches
0x1500    1152    B_w[36,8] weight matrix
0x1A00    1152    C_out[36,8] output (with ReLU)
```

### Softmax Batch

```
Address   Size    Content
0x0000    256     X[4,16] input tensor
0x0100    256     Y[4,16] output tensor
0x0200    16      max[4] scratch (per-row max)
0x0210    16      sum[4] scratch (per-row sum)
0x0220    256     exp_buf[4,16] scratch
```

## Troubleshooting

### "Cannot read input file"

Ensure tensor files exist and have correct sizes:

```bash
ls -la *.bin
# A_16x16.bin should be 1024 bytes (16*16*4)
```

### "Invalid instruction"

Check assembly syntax:

```bash
./build/tools/development/kpu-assembler kernel.kpuasm --print
```

### "Results don't match"

1. Verify input data generation matches kernel expectations
2. Check matrix dimensions match `.dimensions` directive
3. For Conv2D, ensure im2col transformation is pre-applied

### "No trace output"

Trace export only works with transactional fidelity:

```bash
./build/tools/runtime/kpu-loader prog.kpubin \
    --fidelity transactional \  # Required for trace
    --trace output.json
```

## Related Documentation

- [KPUASM Specification](kpuasm-specification.md) — Assembly language reference
- [Fidelity Framework](02-simulation/fidelity-framework.md) — Multi-fidelity design
- [KPU Execution Model](kpu-execution-model.md) — Credit-based dataflow
