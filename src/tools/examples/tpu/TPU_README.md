# Google TPU Assembly and Architecture

This directory contains representations of MLP neural network execution on Google's Tensor Processing Unit (TPU), including XLA HLO intermediate representation and architectural pseudo-assembly.

## Important Note on TPU Assembly

**Google does not publicly release TPU assembly language or ISA documentation.** Unlike NVIDIA GPUs (which have PTX and SASS) or CPUs (which have well-documented ISAs), TPU machine code is proprietary.

What we provide here:
1. **XLA HLO IR**: The documented intermediate representation used by the XLA compiler
2. **Pseudo-Assembly**: Representative assembly based on published architecture papers
3. **JAX Examples**: High-level code that compiles to TPU
4. **Architecture Documentation**: How operations map to hardware

## Directory Structure

```
tpu/
├── xla_hlo/
│   └── mlp_forward.hlo          # XLA HLO intermediate representation
├── pseudo_assembly/
│   └── mlp_forward_tpu.asm      # Pseudo-assembly based on architecture
├── jax_examples/
│   └── mlp_jax_tpu.py           # JAX code that compiles to TPU
└── TPU_README.md                # This file
```

## TPU Architecture Overview

### Hardware Components

#### 1. Matrix Multiply Unit (MXU)
- **Systolic Array**: 2D grid of Processing Elements (PEs)
  - TPU v2/v3: 128×128 array
  - TPU v4: Larger arrays (exact size not publicly disclosed)
- **Function**: Highly optimized matrix multiplication
- **Performance**: Peak 275 TFLOPS (BF16) on v4
- **Execution**: O(N) time for N×N matrix (vs O(N³) naive)

#### 2. Vector Processing Units (VPU)
- **Count**: 2 VPUs per TPU core (v4)
- **Function**: Element-wise operations
  - Activations (ReLU, tanh, sigmoid)
  - Bias addition
  - Normalization
- **Width**: Process multiple elements in parallel

#### 3. Scalar Unit
- **Function**: Control flow, addressing, integer operations
- **Role**: Coordinates MXU and VPU execution

#### 4. High Bandwidth Memory (HBM)
- **Capacity**: 32 GB per chip (TPU v4)
- **Bandwidth**: 1.2 TB/s
- **Organization**: Multiple memory banks
- **Access**: DMA engines for async transfer

#### 5. Interconnect
- **Topology**: 2D torus (4-way connectivity)
- **Speed**: High-bandwidth inter-chip links
- **Purpose**: Multi-chip scaling to "pods" and "slices"

### Memory Hierarchy

```
┌─────────────────────────────────────────┐
│  HBM (32 GB)                            │  ← Slowest, largest
│  1.2 TB/s bandwidth                     │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Unified Buffer (varies by version)     │
│  Shared workspace for MXU/VPU           │
└────────────────┬────────────────────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
┌────▼────────┐    ┌─────────▼──────────┐
│ MXU         │    │ VPU Registers       │  ← Fastest, smallest
│ Accumulator │    │ 32 KB (typical)     │
│ Registers   │    │                     │
└─────────────┘    └────────────────────┘
```

## TPU Compilation Flow

```
Python/JAX Code
       │
       ▼
   JIT Compile
       │
       ▼
  XLA Compiler
       │
       ▼
   HLO IR  ◄─── What we show in mlp_forward.hlo
       │
       ▼
XLA Optimizations
   (fusion, layout, tiling)
       │
       ▼
  TPU Backend
       │
       ▼
TPU Machine Code  ◄─── Proprietary (pseudo-assembly approximation)
       │
       ▼
TPU Hardware Execution
```

## Systolic Array Explained

### What is a Systolic Array?

A **systolic array** is a homogeneous network of data processing units called cells or nodes, which rhythmically compute and pass data through the system. Think of it like a heartbeat ("systole" is the contraction phase of the heartbeat), where data pulses through the array.

### How It Works

For matrix multiplication C = A × B:

```
Input Matrix A:        Input Matrix B:
┌────────┐             ┌────────┐
│ a0 a1  │             │ b0 b1  │
│ a2 a3  │             │ b2 b3  │
└────────┘             └────────┘

4×4 Systolic Array:

    b0   b1   b2   b3  ← B flows DOWN
    ↓    ↓    ↓    ↓
a0→[PE] [PE] [PE] [PE]
a1→[PE] [PE] [PE] [PE]
a2→[PE] [PE] [PE] [PE]
a3→[PE] [PE] [PE] [PE]
    ↓    ↓    ↓    ↓
   c0   c1   c2   c3  ← Results drain

Each PE (Processing Element):
┌──────────────────┐
│  accumulator     │
│  weight (stored) │
│                  │
│  operation:      │
│  acc += a * b    │
└──────────────────┘
```

### Execution Timeline

For a 4×4 array computing 4×4 matrices:

```
Cycle 0:  PE[0,0] receives a0, b0
Cycle 1:  PE[0,0] computes, passes data
          PE[1,0] receives a1
          PE[0,1] receives b1
Cycle 2:  Wavefront propagates...
Cycle 6:  First result c[0,0] complete
Cycle 9:  All 16 results complete
```

**Key Insight**: For an N×N array:
- Takes approximately 3N-1 cycles to complete
- But can be pipelined: start new matrix every N cycles
- Throughput: N² operations per N cycles = **N operations per cycle**

For TPU's 128×128 array:
- 16,384 multiply-accumulates per cycle
- At 1 GHz: **16.4 trillion operations per second** (just the MXU!)

## XLA HLO Intermediate Representation

HLO (High-Level Optimizer) is XLA's intermediate representation. This is the closest thing to "assembly" that's publicly documented for TPUs.

### Key HLO Operations

```hlo
// Matrix multiplication (maps to MXU)
%result = f32[M,N] dot(
  f32[M,K] %input,
  f32[N,K] %weights
), lhs_contracting_dims={1}, rhs_contracting_dims={1}

// Element-wise add (maps to VPU)
%sum = f32[M,N] add(f32[M,N] %a, f32[M,N] %b)

// Broadcast (metadata only, no compute)
%broadcasted = f32[M,N] broadcast(f32[N] %vec), dimensions={1}

// ReLU via maximum (maps to VPU)
%relu = f32[M,N] maximum(f32[M,N] %x, f32[] constant(0))

// Fusion (combines operations, reduces memory traffic)
%fused = f32[M,N] fusion(...), kind=kLoop, calls=%subcomputation
```

### Example MLP in HLO

See `xla_hlo/mlp_forward.hlo` for complete examples including:
- Basic single-layer MLP
- Fused operations (bias + ReLU)
- Batched processing
- 2-layer networks

## Running Code on TPU

### Option 1: Google Colab (Free TPU Access)

```python
# In Colab, select Runtime > Change runtime type > TPU

import jax
import jax.numpy as jnp

# Verify TPU is available
print(jax.devices())  # Should show TPU devices

# Your code automatically runs on TPU!
@jax.jit
def mlp_layer(x, weights, bias):
    return jax.nn.relu(jnp.dot(x, weights.T) + bias)
```

### Option 2: Google Cloud TPU VMs

```bash
# Create TPU VM
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central1-a \
  --accelerator-type=v4-8 \
  --version=tpu-vm-v4-base

# SSH into TPU VM
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central1-a

# Install JAX
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Run code
python mlp_jax_tpu.py
```

### Option 3: Kaggle (Free TPU)

Kaggle notebooks offer free TPU access. Just select TPU as accelerator in notebook settings.

## Performance Characteristics

### TPU v4 (as of 2024)

| Specification | Value |
|--------------|-------|
| **Compute** |
| Peak BF16 | 275 TFLOPS |
| Peak FP32 | 137.5 TFLOPS |
| Peak INT8 | 550 TOPS |
| **Memory** |
| HBM Capacity | 32 GB |
| HBM Bandwidth | 1.2 TB/s |
| **Networking** |
| Inter-chip BW | 4.8 Tbps (total) |
| **Power** |
| TDP | ~200-300W per chip |
| **Cost** |
| Cloud pricing | ~$1-3/hour per chip |

### MLP Performance (512×256 layer, batch=128)

| Metric | TPU v4 (BF16) | TPU v4 (FP32) |
|--------|---------------|---------------|
| Latency | ~5-10 µs | ~10-20 µs |
| Throughput | ~100-150 TFLOPS | ~50-75 TFLOPS |
| Utilization | 40-60% MXU | 40-60% MXU |
| Bottleneck | Memory (small model) | Memory |

**Note**: Small models are memory-bound. Large models/batches can saturate MXU.

## Optimization Strategies

### 1. Use BF16 Precision

```python
# BF16 is 2x faster than FP32 on TPU
x = x.astype(jnp.bfloat16)
weights = weights.astype(jnp.bfloat16)
result = jnp.dot(x, weights.T)
```

**Why**: TPU MXU has 2x throughput for BF16 vs FP32.

### 2. Pad Dimensions to Multiples of 128

```python
# Bad: dimension 250 (not multiple of 128)
weights = jnp.zeros((250, 512))

# Good: dimension 256 (multiple of 128)
weights = jnp.zeros((256, 512))
```

**Why**: MXU works in 128×128 tiles. Padding reduces wasted cycles.

### 3. Use Large Batch Sizes

```python
# Bad: batch size 8 (underutilizes TPU)
input = jnp.zeros((8, 512))

# Good: batch size 128 or 256
input = jnp.zeros((128, 512))
```

**Why**: TPU is a throughput-oriented architecture. Large batches amortize overhead.

### 4. Minimize Host-TPU Transfers

```python
# Bad: Transfer data every iteration
for i in range(1000):
    data = jax.device_put(cpu_data[i])  # Slow!
    result = model(data)

# Good: Transfer once, keep on TPU
data_on_tpu = jax.device_put(cpu_data)
for i in range(1000):
    result = model(data_on_tpu[i])  # Fast!
```

**Why**: PCIe transfer is slow (~16 GB/s) vs HBM (~1200 GB/s).

### 5. Let XLA Fuse Operations

```python
# Don't manually fuse, XLA does it better
def mlp(x, w, b):
    return jax.nn.relu(jnp.dot(x, w.T) + b)  # XLA fuses this
```

**Why**: XLA compiler can fuse matmul+bias+relu into fewer kernels.

### 6. Use All Cores (pmap)

```python
# Distribute across TPU cores
@jax.pmap
def parallel_mlp(x_shard):
    return mlp(x_shard, weights, bias)

# x has shape [num_devices, batch_per_device, features]
results = parallel_mlp(x)
```

**Why**: TPU chips have multiple cores. pmap enables data parallelism.

## Comparison with Other Accelerators

| Feature | TPU v4 | NVIDIA A100 | ARM Mali G78 |
|---------|--------|-------------|--------------|
| **Architecture** | Systolic array | Streaming MPs | Tile-based |
| **Peak FP32** | 137 TFLOPS | 19.5 TFLOPS | 1.5 TFLOPS |
| **Peak BF16** | 275 TFLOPS | 312 TFLOPS (Tensor) | N/A |
| **Memory** | 32 GB HBM | 40-80 GB HBM2e | 8-16 GB unified |
| **Memory BW** | 1.2 TB/s | 1.5 TB/s | 60 GB/s |
| **Power** | 200-300W | 250-400W | 5-10W |
| **Strength** | Large batches | Flexible | Energy efficient |
| **Weakness** | Small batches | Cost | Absolute perf |
| **Best For** | Training | Training & HPC | Mobile/Edge |
| **Programming** | JAX, TensorFlow | CUDA, PyTorch | OpenCL, Vulkan |
| **ISA Access** | No (HLO only) | Yes (PTX, SASS) | Limited |

### When to Use TPU

✅ **Good Use Cases:**
- Large batch training (NLP models, image classification)
- Transformer models (BERT, GPT)
- Research with JAX or TensorFlow
- Google Cloud infrastructure
- Models with mostly matrix multiplies

❌ **Bad Use Cases:**
- Small batch inference (latency-critical)
- Custom kernels (limited flexibility)
- Non-Google cloud environments
- Models with irregular control flow

## TPU Evolution

| Generation | Year | Peak (BF16) | Memory | Key Feature |
|-----------|------|-------------|--------|-------------|
| **TPU v1** | 2015 | N/A (int8 only) | 8 GB | Inference only |
| **TPU v2** | 2017 | 45 TFLOPS | 16 GB HBM | First training TPU |
| **TPU v3** | 2018 | 90 TFLOPS | 32 GB HBM | 2x performance |
| **TPU v4** | 2021 | 275 TFLOPS | 32 GB HBM | 3x performance, sparse support |
| **TPU v5e** | 2023 | ~120 TFLOPS | 16 GB HBM | Cost-optimized |
| **TPU v5p** | 2023 | ~450 TFLOPS | 96 GB HBM | Flagship, 4x v4 |

## Learning Resources

### Official Documentation
- XLA Compiler: https://www.tensorflow.org/xla
- JAX on TPU: https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html
- Cloud TPU: https://cloud.google.com/tpu/docs

### Research Papers
- **TPU v1**: "In-Datacenter Performance Analysis of a Tensor Processing Unit" (ISCA 2017)
- **TPU v2/v3**: "A Domain-Specific Supercomputer for Training Deep Neural Networks" (CACM 2020)
- **Systolic Arrays**: "Why Systolic Architectures?" (IEEE Computer 1982)

### Tutorials
- Google Colab TPU Guide: https://colab.research.google.com/notebooks/tpu.ipynb
- JAX Quickstart: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html

## Summary

**Key Takeaways:**

1. **TPU ISA is proprietary** - We can only access HLO IR, not native assembly
2. **Systolic array** - Specialized hardware for matrix multiplication (O(N) time!)
3. **XLA compiler** - Automatically optimizes Python → HLO → TPU code
4. **BF16 precision** - 2x faster than FP32, minimal accuracy loss
5. **Throughput oriented** - Best with large batches (128+)
6. **Limited flexibility** - Not programmable like GPUs, but very efficient for what it does

For MLP neural networks, TPUs excel at batch processing and can achieve **100+ TFLOPS** on real workloads, making them ideal for training large models.

