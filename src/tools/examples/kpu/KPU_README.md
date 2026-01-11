# Stillwater KPU (Knowledge Processing Unit)

## Domain Flow Architecture for Neural Networks

The **Knowledge Processing Unit (KPU)** is a novel processor architecture designed by Stillwater Supercomputing that combines the efficiency of systolic arrays with the flexibility of dataflow machines.

## What Makes KPU Unique?

### Domain Flow = Systolic + Dataflow

The KPU represents a new point in the processor design space:

```
Systolic Arrays (TPU)     +     Dataflow Machines
   High efficiency               High adaptivity
   Fixed function                Programmable
   Position-dependent            Position-independent
          ↓                              ↓
                    KPU Domain Flow
              (Best of both worlds)
```

### Key Innovations

1. **Programmable Systolic Array**
   - Unlike TPU's fixed systolic array
   - Each PE runs a small domain flow program (~100-200 bytes)
   - Same program in all PEs (position-independent)

2. **Tagged Token Dataflow**
   - Data carries "tags" representing N-dimensional abstract coordinates
   - PEs use CAM (Content Addressable Memory) to match tags
   - When operands meet at a PE → instruction fires automatically
   - Results route to appropriate PEs based on tags

3. **Sequencer-Based Data Movement**
   - Explicit control over memory hierarchy (DRAM → L3 → L2 → L1)
   - DMA, Block Mover, and Streamer engines have programmable sequencers
   - Streamers create tagged tokens as data flows into compute fabric

4. **Position-Independent Computation**
   - Computation doesn't care which physical PE executes it
   - Automatic load balancing via token routing
   - Enables adaptive scheduling and fault tolerance

## Directory Structure

```
kpu/
├── assembly/
│   └── mlp_forward_kpu.asm      # KPU assembly for MLP
├── examples/
│   └── (future JAX/TensorFlow examples)
├── docs/
│   ├── KPU_ARCHITECTURE.md      # Detailed architecture guide
│   └── ARCHITECTURE_COMPARISON.md  # KPU vs CPU/GPU/TPU
└── KPU_README.md                # This file
```

## Architecture Overview

### Compute Fabric: Programmable Systolic Array

```
┌────────────────────────────────────────────────┐
│           2D/3D Torus Mesh of PEs              │
│                                                │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐              │
│  │ PE │──│ PE │──│ PE │──│ PE │              │
│  │CAM │  │CAM │  │CAM │  │CAM │ ← Tag match  │
│  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘              │
│    │       │       │       │                  │
│  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐             │
│  │ PE │──│ PE │──│ PE │──│ PE │              │
│  └────┘  └────┘  └────┘  └────┘              │
│                                                │
│  Tagged tokens route by hash, not position    │
└────────────────────────────────────────────────┘
```

### Memory Hierarchy with Sequencers

```
         DRAM (External Memory)
              ↓
        DMA Engine (sequencer-controlled)
              ↓
    L3 Distributed On-Chip Memory (Banks 0-3)
              ↓
      Block Mover Engine (sequencer-controlled)
              ↓
        L2 Scratchpad Memory
              ↓
       Streamer Engine (sequencer-controlled)
       Creates Tagged Tokens!
              ↓
        L1 Stream Buffers
              ↓
     Programmable Systolic Array
```

## How It Works: MLP Example

### Step 1: Install Domain Flow Program (FIRST!)

```asm
.kernel mlp_matmul_kernel:
    // When A[i,k] and B[k,j] tokens meet:
    .inst mac_instruction:
        .match  {("A_element", (row, k)), ("W_element", (k, col))}
        .operation  mac_bf16
        .accumulate partial_sum[row, col]

    // When k accumulation complete:
    .inst emit_result:
        .condition  k == k_max
        .emit_token signature: "C_output",
                    dims: (row, col),
                    value: partial_sum
.kernel_end

// Broadcast to all PEs
fabric.broadcast mlp_matmul_kernel → all_pes
```

**Key**: This happens BEFORE any data movement! Fabric must be ready.

### Step 2: Configure Data Movement Sequencers

```asm
// DMA Sequencer: DRAM → L3
dma.sequencer.define load_inputs:
    .step transfer: src=DRAM, dst=L3, size=...
    .step wait: transfer_complete
    .step signal: data_ready
.seq_end

// Streamer Sequencer: L2 → L1 (creates tagged tokens!)
streamer.sequencer.define stream_inputs:
    .step for_each_element i:
        value = memory[i]
        tag = ("A_element", (row=0, k=i))
        emit_token(tag, value)
.seq_end
```

### Step 3: Execute

```asm
// Start DMA engines
dma.start engine0

// Block movers auto-trigger on signals
// Streamers auto-trigger and create tokens

// Tokens flow into fabric:
//   ("A_element", (0, 42), 2.5) ──┐
//   ("W_element", (42, 5), 1.2) ──┤→ Meet at PE → MAC fires!
//                                 │
//   Result: ("C_partial", (0, 5), 3.0)
```

**Magic**: Computation happens automatically as tokens arrive and match!

## Tagged Token Example

```
Input element A[0, 42] = 2.5 becomes:

┌─────────────────────────────────┐
│  Tagged Token                   │
├─────────────────────────────────┤
│  Signature: "A_element"         │ ← What kind of data
│  Dims: (row=0, k=42)            │ ← Abstract coordinates
│  Value: 2.5                     │ ← Actual data
│  Tag Hash: hash(0, 42)          │ ← Routing destination
└─────────────────────────────────┘

Weight element W[42, 5] = 1.2 becomes:

┌─────────────────────────────────┐
│  Tagged Token                   │
├─────────────────────────────────┤
│  Signature: "W_element"         │
│  Dims: (k=42, col=5)            │ ← Both have k=42!
│  Value: 1.2                     │
│  Tag Hash: hash(42, 5)          │
└─────────────────────────────────┘

When both arrive at a PE:
  - CAM matches: both have k=42
  - MAC instruction fires: 2.5 × 1.2
  - Accumulates in partial_sum[0, 5]
```

## Advantages Over Other Architectures

### vs TPU

| Feature | TPU | KPU |
|---------|-----|-----|
| Systolic array | Fixed | Programmable |
| Operations | Matrix multiply only | Any dataflow graph |
| Position | PE-dependent | Independent |
| Adaptivity | None | High (token routing) |
| Flexibility | Low | High |

**KPU wins**: Programmability + Adaptivity

### vs GPU

| Feature | GPU | KPU |
|---------|-----|-----|
| Control | Thread-based | Dataflow |
| Efficiency | Medium | Very high |
| Power | High (250-400W) | Lower (~150W) |
| Programming | Explicit threads | Declarative dataflow |

**KPU wins**: Energy efficiency for structured workloads

### vs CPU

| Feature | CPU | KPU |
|---------|-----|-----|
| Parallelism | 8-wide SIMD | Mesh of PEs |
| Throughput | ~5 GFLOPS | ~150 TFLOPS |
| Flexibility | Full | Domain-specific |

**KPU wins**: Massive throughput for AI, DSP, and HPC

## Use Cases

✅ **Ideal For**:
- Real-time signal processing
- Sensor Fusion and Control
- Neural network training and inference
- Knowledge processing and reasoning
- Graph neural networks
- Transformer models
- Structured computations with regular patterns
- R&D requiring flexible hardware

❌ **Not Ideal For**:
- Fully irregular computations
- General-purpose computing
- Very small workloads
- Single-operation throughput (TPU better for pure matmul)

## Performance Estimates

For 512×256 MLP layer:
- **Latency**: ~5-10 µs (similar to TPU)
- **Throughput**: ~150 TFLOPS (BF16)
- **Power**: ~150W (estimated)
- **Efficiency**: ~1.0 TFLOPS/W (best-in-class)

## Key Concepts

### 1. Domain Flow Program
Small program (~100-200 bytes) that runs in each PE, describing:
- What tagged tokens to match
- What operation to perform
- What result token to emit
- How to route the result

### 2. Tagged Token
Data packet with:
- Signature (type identifier)
- Dimensions (N-D abstract coordinates)
- Value (actual data)
- Tag hash (routing information)

### 3. Sequencer
Programmable state machine in data movement engines that executes a schedule of transfers and transformations.

### 4. CAM (Content Addressable Memory)
Hardware in each PE that stores arriving tokens and matches them by tag to fire instructions.

### 5. Position Independence
Computation defined by abstract coordinates, not physical PE locations. Enables adaptive routing and load balancing.

## Programming Model

Unlike traditional architectures:
- **Not**: "Thread X computes element Y"
- **Not**: "PE at position (i,j) computes specific element"
- **Instead**: "When tokens with matching tags meet, compute"

This is **declarative dataflow** with **spatial execution**.

## Summary

The Stillwater KPU represents a fundamentally new approach to neural network acceleration:

1. **Programmable** like a GPU (not fixed like TPU)
2. **Efficient** like a systolic array (structured computation)
3. **Adaptive** through tagged token routing
4. **Knowledge-centric** for AI workloads

It's a **domain flow architecture** that combines the best aspects of systolic arrays and dataflow machines to create a uniquely capable processor for knowledge processing and neural network computation.

---

**For detailed technical information**, see:
- `docs/KPU_ARCHITECTURE.md` - Complete architecture guide
- `docs/ARCHITECTURE_COMPARISON.md` - Comparison with CPU/GPU/TPU
- `assembly/mlp_forward_kpu.asm` - Example assembly code

