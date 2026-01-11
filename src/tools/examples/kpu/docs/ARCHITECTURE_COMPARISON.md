# Architecture Comparison: CPU vs GPU vs TPU vs KPU

## MLP Forward Pass Across Four Architectures

This document compares how the same MLP operation executes on different architectures.

**Operation**: `output = ReLU(weights × input + bias)`
- Input: [1 × 512]
- Weights: [256 × 512]
- Bias: [256]
- Output: [1 × 256]

---

## 1. CPU (x86-64 with AVX)

### Execution Model: **Sequential with SIMD**

```
┌─────────────────┐
│  Core           │
│  ┌───────────┐  │
│  │ Registers │  │  ← 16 YMM registers (256-bit)
│  └───────────┘  │
│  ┌───────────┐  │
│  │  L1 Cache │  │  ← 32 KB
│  └───────────┘  │
└─────────────────┘
```

### Assembly Snippet

```asm
// For each output neuron:
xor rax, rax                    # output_idx = 0

output_loop:
    vxorps ymm0, ymm0, ymm0     # sum = 0 (accumulator)
    mov r10, weights[rax]       # weight row pointer

    # Dot product loop (processes 8 floats/iter)
    xor r11, r11                # i = 0
dot_loop:
    vmovups ymm1, [input + r11*4]    # Load 8 inputs
    vmovups ymm2, [r10 + r11*4]      # Load 8 weights
    vfmadd231ps ymm0, ymm1, ymm2     # FMA: sum += in * w
    add r11, 8
    cmp r11, 512
    jl dot_loop

    # Horizontal sum + bias + ReLU
    # ... (reduction code)

    inc rax
    cmp rax, 256
    jl output_loop
```

### Characteristics

- **Control**: Explicit loops, manual SIMD
- **Parallelism**: 8 floats per AVX instruction
- **Throughput**: ~512 FMAs per 256 neurons = ~131K FMAs
- **Time**: ~50-100 µs @ 2-3 GHz
- **Performance**: ~2-5 GFLOPS

---

## 2. GPU (NVIDIA Ampere)

### Execution Model: **SIMT (Single Instruction, Multiple Threads)**

```
┌──────────────────────────────────┐
│  Streaming Multiprocessor (SM)   │
│  ┌────────────────────────────┐  │
│  │  Warps (32 threads each)   │  │
│  │  ┌──┐┌──┐┌──┐┌──┐┌──┐     │  │
│  │  │T ││T ││T │...│T │      │  │  ← 32 threads execute together
│  │  └──┘└──┘└──┘   └──┘      │  │
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │  Shared Memory (164 KB)    │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

### PTX Assembly Snippet

```ptx
// Each thread computes one output neuron

mlp_layer_forward_ptx:
    // Thread ID = which output neuron
    mov.u32 %r0, %tid.x

    // Load input to shared memory cooperatively
    ld.global.ca.f32 %f1, [input + %tid.x*4]
    st.shared.f32 [shared_input + %tid.x*4], %f1
    bar.sync 0                      // __syncthreads()

    // Compute dot product (thread's neuron)
    mov.f32 %f10, 0.0              // sum = 0
    mov.u32 %r1, 0                 // i = 0

dot_loop:
    ld.global.ca.v4.f32 {%f20, %f21, %f22, %f23}, [weights + offset]
    ld.shared.v4.f32 {%f24, %f25, %f26, %f27}, [shared_input + i*4]

    fma.rn.f32 %f10, %f20, %f24, %f10   // 4x FMA
    fma.rn.f32 %f10, %f21, %f25, %f10
    fma.rn.f32 %f10, %f22, %f26, %f10
    fma.rn.f32 %f10, %f23, %f27, %f10

    add.u32 %r1, %r1, 4
    setp.lt.u32 %p1, %r1, 512
    @%p1 bra dot_loop

    // Add bias + ReLU
    ld.global.ca.f32 %f30, [bias + %r0*4]
    add.f32 %f10, %f10, %f30
    max.f32 %f10, %f10, 0.0

    st.global.wb.f32 [output + %r0*4], %f10
    ret
```

### Characteristics

- **Control**: Thousands of threads in parallel
- **Parallelism**: 32 threads per warp, many warps per SM
- **Throughput**: All 256 neurons computed in parallel
- **Time**: ~5-10 µs @ 1-1.5 GHz
- **Performance**: ~10-20 GFLOPS (memory-bound for small model)

---

## 3. TPU (Google Cloud TPU v4)

### Execution Model: **Systolic Array (Weight Stationary)**

```
        Input (flows DOWN)
        ↓    ↓    ↓    ↓
        b₀   b₁   b₂   b₃
        ↓    ↓    ↓    ↓
a₀ →  [PE] [PE] [PE] [PE]  → c₀
a₁ →  [PE] [PE] [PE] [PE]  → c₁
a₂ →  [PE] [PE] [PE] [PE]  → c₂
a₃ →  [PE] [PE] [PE] [PE]  → c₃

(Actual: 128×128 array)
```

### XLA HLO IR

```hlo
ENTRY %mlp_layer_forward (
  input: f32[1,512],
  weights: f32[256,512],
  bias: f32[256]
) -> f32[1,256] {

  %input = f32[1,512] parameter(0)
  %weights = f32[256,512] parameter(1)
  %bias = f32[256] parameter(2)

  // Matrix multiply on MXU (systolic array)
  %dot = f32[1,256] dot(%input, %weights),
    lhs_contracting_dims={1},
    rhs_contracting_dims={1}

  // Broadcast bias
  %bias_bcast = f32[1,256] broadcast(%bias), dimensions={1}

  // Add bias (on VPU)
  %add = f32[1,256] add(%dot, %bias_bcast)

  // ReLU (on VPU)
  %zero = f32[1,256] broadcast(f32[] constant(0)), dimensions={}
  ROOT %relu = f32[1,256] maximum(%add, %zero)
}
```

### Characteristics

- **Control**: Implicit (XLA compiler)
- **Parallelism**: 128×128 = 16K MACs per cycle
- **Throughput**: Tiles 512×256 into 128×128 chunks
- **Time**: ~5-10 µs @ 1 GHz (BF16)
- **Performance**: ~100-150 TFLOPS (but memory-bound)
- **Key**: Weight stationary, fixed dataflow

---

## 4. KPU (Stillwater Domain Flow)

### Execution Model: **Tagged Token Dataflow**

```
┌─────────────────────────────────────┐
│  PE Mesh (2D Torus)                 │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐    │
│  │ PE │──│ PE │──│ PE │──│ PE │    │
│  │CAM │  │CAM │  │CAM │  │CAM │    │
│  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘    │
│    │       │       │       │        │
│  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐   │
│  │ PE │──│ PE │──│ PE │──│ PE │    │
│  │CAM │  │CAM │  │CAM │  │CAM │    │
│  └────┘  └────┘  └────┘  └────┘    │
│                                     │
│  Tokens route by TAG, not position │
└─────────────────────────────────────┘
```

### KPU Assembly

```asm
// STAGE 1: Install kernel in compute fabric (FIRST!)

.kernel mlp_matmul_kernel:
    // Instruction 1: MAC when tokens meet
    .inst mac_instruction:
        .match  {
                    ("A_element", (row, k)),
                    ("W_element", (k, col))
                }
        .operation  mac_bf16
        .accumulate partial_sum[row, col]

    // Instruction 2: Emit when k complete
    .inst emit_partial:
        .condition  k == k_max
        .emit_token signature: "C_partial",
                    dims: (row, col),
                    value: partial_sum[row, col]

    // Instruction 3: Add bias
    .inst add_bias:
        .match  {("C_partial", (row, col)), ("bias", (col))}
        .operation  add_bf16
        .emit_token signature: "C_biased", ...

    // Instruction 4: ReLU
    .inst relu:
        .match  ("C_biased", (row, col))
        .operation  max_bf16(value, 0.0)
        .emit_token signature: "C_output", ...
.kernel_end

// Broadcast kernel to all PEs
fabric.broadcast source=mlp_matmul_kernel,
                 target=all_pes

// STAGE 2-6: Configure sequencers
// (DMA, Block Mover, Streamer sequencers...)

// STAGE 7: Start execution
// Streamers create tagged tokens:
//   input[i] → ("A_element", (0, i), value=input[i])
//   weight[j,i] → ("W_element", (i, j), value=weight[j,i])
//
// Tokens flow to PEs, match in CAM, instructions fire!
```

### Characteristics

- **Control**: Dataflow (tokens trigger computation)
- **Parallelism**: Position-independent, all PEs active
- **Throughput**: Similar to TPU systolic array
- **Time**: ~5-10 µs (similar to TPU)
- **Performance**: ~100-150 TFLOPS
- **Key**: Programmable + adaptive routing

---

## Side-by-Side Feature Comparison

| Feature | CPU AVX | GPU Ampere | TPU v4 | KPU |
|---------|---------|------------|--------|-----|
| **Paradigm** | SIMD | SIMT | Systolic | Dataflow |
| **Parallelism** | 8-wide | 32-wide warps | 16K PEs | Mesh of PEs |
| **Programming** | Explicit | Thread-based | Compiled HLO | Tagged tokens |
| **Control Flow** | Sequential | Divergent | Implicit | Dataflow |
| **Data Movement** | Manual | Explicit | Streaming | Sequencers |
| **Position** | Fixed | Thread-dependent | PE-position | Independent |
| **Flexibility** | Full | High | Low | Medium-High |
| **Efficiency** | Low | Medium | Very High | Very High |
| **Power** | 50-100W | 250-400W | 200-300W | ~100-200W (est) |

---

## Detailed Comparison: Matrix Multiply

### How Each Architecture Computes C[0,3] = Σ(A[0,k] × B[k,3])

#### CPU (AVX)
```
Thread 0 computes neuron 3:
  sum = 0
  for k in 0..511 step 8:
    sum += A[0,k:k+8] ⊙ B[3,k:k+8]  # 8-wide SIMD
  C[0,3] = sum
```
- **64 iterations** of 8-wide FMA
- **Explicit loop control**
- **One neuron at a time**

#### GPU (Ampere)
```
Thread 3 computes neuron 3:
  sum = 0
  for k in 0..511 step 4:
    sum += A[0,k:k+4] ⊙ B[3,k:k+4]  # Vectorized
  C[0,3] = sum

All 256 threads run in parallel (one per neuron)
```
- **128 iterations** of 4-wide FMA
- **256 threads parallel**
- **Shared memory for A**

#### TPU (v4)
```
Systolic array processes tiles:

Tile 1: A[0, 0:127] × B[0:127, 0:127]
  → Produces partial C[0, 0:127]
  → Element C[0,3] accumulates contributions from k=0..127

Tile 2: A[0, 128:255] × B[128:255, 0:127]
  → Adds to C[0,3] (k=128..255)

... (4 tiles total for k dimension)

Final C[0,3] = sum of all partial results
```
- **Wavefront through systolic array**
- **All 256 outputs computed together**
- **O(N) time with N² PEs**

#### KPU (Domain Flow)
```
Streamer emits tagged tokens:
  A[0,0] → ("A_element", (row=0, k=0), value)
  A[0,1] → ("A_element", (row=0, k=1), value)
  ...
  B[0,3] → ("W_element", (k=0, col=3), value)
  B[1,3] → ("W_element", (k=1, col=3), value)
  ...

Tokens route to PEs based on tag hash.
When A[0,k] and B[k,3] meet at a PE:
  - CAM matches tags (both have k value)
  - MAC instruction fires
  - Accumulates in partial_sum[0,3]

After all k processed:
  - Emit ("C_partial", (0, 3), sum)
  - Routes to PE handling output
  - Adds bias, applies ReLU
  - Drains as output
```
- **Dataflow execution**
- **Position-independent**
- **Automatic load balancing**

---

## Memory Hierarchy Comparison

### CPU
```
Registers (16 × 256-bit)
    ↓ ~1 cycle
L1 Cache (32 KB)
    ↓ ~4 cycles
L2 Cache (256 KB)
    ↓ ~12 cycles
L3 Cache (8-32 MB)
    ↓ ~40 cycles
DRAM (16-64 GB)
    ↓ ~100-300 cycles
```
**Implicit management** (hardware cache)

### GPU
```
Registers (64K per SM)
    ↓ ~1 cycle
Shared Memory (164 KB, programmer-controlled)
    ↓ ~20 cycles
L1 Cache (128 KB)
    ↓ ~30 cycles
L2 Cache (40 MB)
    ↓ ~200 cycles
HBM (40-80 GB)
    ↓ ~400-800 cycles
```
**Explicit management** (shared memory)

### TPU
```
Vector Registers
    ↓
Unified Buffer
    ↓
HBM (32 GB)
```
**Implicit streaming** (XLA compiler)

### KPU
```
PE Local Storage
    ↓
L1 Stream Buffers
    ↓ Streamer (creates tagged tokens)
L2 Scratchpad
    ↓ Block Mover (sequencer-controlled)
L3 Distributed Banks
    ↓ DMA (sequencer-controlled)
DRAM
```
**Explicit sequencers** (programmable schedule)

---

## When to Use Each Architecture

### CPU (x86-64 AVX)
✅ **Use When:**
- Small batches or single inference
- Latency critical (<1ms response time)
- Irregular control flow
- Development/prototyping
- Limited power budget

❌ **Avoid When:**
- Large batch processing
- Need maximum throughput
- Pure matrix operations

### GPU (NVIDIA Ampere)
✅ **Use When:**
- High parallelism needed
- Flexibility required (custom kernels)
- Mixed workload (training + inference)
- Existing CUDA ecosystem
- Medium-large batches (32-256)

❌ **Avoid When:**
- Tiny batches (GPU underutilized)
- Pure matrix multiply (TPU/KPU more efficient)
- Strict power limits

### TPU (Google v4)
✅ **Use When:**
- Very large batches (128-1024)
- Matrix-heavy workloads
- Training large models (Transformers)
- Google Cloud infrastructure
- Maximum throughput needed

❌ **Avoid When:**
- Small batches (<32)
- Irregular operations
- Need custom kernels
- Low latency critical

### KPU (Stillwater Knowledge Processing Unit)
✅ **Use When:**
- Need both efficiency AND flexibility
- Want programmable domain-specific hardware
- Structured but varying computations
- R&D requiring hardware adaptivity
- Custom neural network architectures
- Knowledge-intensive workloads

❌ **Avoid When:**
- Simple fixed workloads (TPU sufficient)
- Fully irregular code (GPU better)
- Extremely tiny batches

---

## Performance Summary (512×256 MLP)

| Architecture | Latency | Throughput | Power | Perf/Watt |
|--------------|---------|------------|-------|-----------|
| **CPU AVX** | 50-100 µs | ~5 GFLOPS | 50W | 0.1 GFLOPS/W |
| **GPU A100** | 5-10 µs | ~20 GFLOPS | 300W | 0.067 GFLOPS/W |
| **TPU v4** | 5-10 µs | ~150 GFLOPS | 250W | 0.6 GFLOPS/W |
| **KPU** | 5-10 µs | ~150 GFLOPS | 150W (est) | 1.0 GFLOPS/W |

**Note**: For this small model, all are memory-bound. Performance gap widens with larger models/batches.

---

## Conceptual Summary

| | CPU | GPU | TPU | KPU |
|---|-----|-----|-----|-----|
| **Mental Model** | "Do one thing at a time, really fast" | "Do many things at once" | "Stream data through specialized hardware" | "Let tokens find their computation" |
| **Programmer thinks** | Loops and SIMD | Threads and blocks | Tensor operations | Dataflow graphs |
| **Hardware does** | Superscalar + SIMD | Warp scheduling | Systolic waves | Tag matching + routing |
| **Best at** | Versatility | Parallelism | Matrix efficiency | Programmable efficiency |

The **KPU** uniquely combines:
- TPU's efficiency (systolic organization)
- GPU's flexibility (programmable compute)
- Novel approach (tagged dataflow)

It represents a **new point in the design space** between specialized accelerators (TPU) and general-purpose processors (GPU/CPU).

