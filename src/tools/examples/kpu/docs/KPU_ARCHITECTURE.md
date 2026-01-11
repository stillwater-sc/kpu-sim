# Stillwater KPU Architecture

## Domain Flow Computing for Neural Networks

The Stillwater KPU (Knowledge Processing Unit) is a **domain flow architecture** - a distributed dataflow machine that combines:
- Spatial organization of systolic arrays
- Adaptivity and programmability of dataflow machines
- Tagged token matching for position-independent computation

## Architecture Classification

**Domain Flow Machine = Systolic Array + Dataflow Computer**

### Key Innovations

1. **Programmable Systolic Array** - Unlike fixed-function TPU
2. **Tagged Token Routing** - Position-independent dataflow execution
3. **Hierarchical Sequencers** - Systematic data movement orchestration
4. **N-Dimensional Abstract Space** - Computational coordinates, not physical positions

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         EXTERNAL DRAM                        │
│                    (Model Parameters, Data)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                   ┌──────┴──────┐
                   │ DMA Engines │ ← Sequencer-controlled
                   │  (DRAM↔L3)  │
                   └──────┬──────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│           L3 DISTRIBUTED ON-CHIP MEMORY (Banks 0-3)          │
│              High-bandwidth, multiple access ports            │
└─────────────────────────┬────────────────────────────────────┘
                          │
                ┌─────────┴─────────┐
                │  Block Movers     │ ← Sequencer-controlled
                │    (L3 ↔ L2)      │
                └─────────┬─────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│           L2 SCRATCHPAD MEMORY (Configurable)                │
│           Partitioned for inputs, weights, outputs            │
└─────────────────────────┬────────────────────────────────────┘
                          │
                  ┌───────┴────────┐
                  │   Streamers    │ ← Sequencer-controlled
                  │  (L2 → L1)     │ ← CREATES TAGGED TOKENS!
                  └───────┬────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│     L1 STREAM BUFFERS (North/West injection points)          │
│              Tagged token FIFOs feeding fabric                │
└─────────────────────────┬────────────────────────────────────┘
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
   ┌────▼─────────────────────────────────┐ │
   │  PROGRAMMABLE SYSTOLIC ARRAY         │ │
   │  (Compute Fabric)                    │ │
   │                                      │ │
   │  • 2D/3D Torus Mesh of PEs          │ │
   │  • Each PE has CAM for tag matching │ │
   │  • Position-independent programs    │ │
   │  • Tagged token dataflow            │ │
   │  • Network overlay for config       │ │
   └──────────────────────────────────────┘ │
                          │                  │
                          └──────────────────┘
                        Results drain to L2
```

---

## Processing Element (PE) Architecture

Each PE in the compute fabric contains:

```
┌────────────────────────────────────────────────┐
│              Processing Element (PE)            │
├────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Content Addressable Memory (CAM)        │  │
│  │  - Stores arriving tagged tokens         │  │
│  │  - Matches tags to find operand sets     │  │
│  │  - Fires instructions when complete      │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Domain Flow Program Memory (~100-200B)  │  │
│  │  - Position-independent instructions     │  │
│  │  - Tag matching patterns                 │  │
│  │  - Operations to perform                 │  │
│  │  - Output tag generation                 │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Compute Unit                            │  │
│  │  - BF16 multiply-accumulate              │  │
│  │  - Element-wise operations               │  │
│  │  - Accumulator registers                 │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Router                                   │  │
│  │  - Routes tokens by tag hash             │  │
│  │  - Torus network connections (N/S/E/W)   │  │
│  │  - Backpressure handling                 │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
└────────────────────────────────────────────────┘
```

---

## Tagged Token Dataflow Model

### What is a Tagged Token?

A **tagged token** is a packet containing:

```
┌─────────────────────────────────────────┐
│          Tagged Token Structure          │
├─────────────────────────────────────────┤
│  Signature:  "A_element"                │ ← Type identifier
│  Dimensions: (row=0, k=42)              │ ← N-D abstract coordinates
│  Value:      1.5 (BF16)                 │ ← Actual data
│  Tag Hash:   0x1A2B3C4D                 │ ← Routing destination
└─────────────────────────────────────────┘
```

### How Tokens Match and Fire

1. **Token Arrival**: Tagged token arrives at a PE
2. **CAM Lookup**: PE checks CAM for matching tags
3. **Operand Set**: When all operands for an instruction arrive, they MATCH
4. **Instruction Fires**: Operation executes automatically
5. **Result Token**: New tagged token created and routed

**Example - Matrix Multiply Element**:

```
PE receives two tokens:

Token 1:                      Token 2:
┌────────────────────┐       ┌────────────────────┐
│ Sig: "A_element"   │       │ Sig: "W_element"   │
│ Dims: (row=0, k=5) │  +    │ Dims: (k=5, col=3) │
│ Value: 2.5         │       │ Value: 1.2         │
│ Hash: ...          │       │ Hash: ...          │
└────────────────────┘       └────────────────────┘

Both have k=5 → MATCH!

Domain flow program in PE:
  .match (A_element[row, k], W_element[k, col])
  .operation mac_bf16
  .accumulate partial_sum[row, col]

Fires: partial_sum[0, 3] += 2.5 × 1.2
```

### Position Independence

**Key Insight**: The computation doesn't care which physical PE executes it!

- Tags represent **abstract computational coordinates**
- Physical PE location is irrelevant
- Tokens route to available PEs based on tag hash
- Same program runs in all PEs
- Load balancing happens automatically via routing

**Contrast with TPU**:
- TPU: Fixed PE position, fixed data flow (north/west)
- KPU: Any PE can compute any element, tags determine routing

---

## Domain Flow Program Example

The small program (~180 bytes) installed in each PE for MLP:

```
.kernel mlp_matmul_kernel:

    // Instruction 1: Multiply-Accumulate
    .inst mac_instruction:
        .match  tag_pattern = {
                    (signature: "A_element", dims: (row_idx, k_idx)),
                    (signature: "W_element", dims: (k_idx, col_idx))
                }
        .operation  mac_bf16
        .accumulate partial_sum[row_idx, col_idx]
        .consume    both_tokens

    // Instruction 2: Emit when complete
    .inst emit_partial:
        .condition  k_idx == k_max
        .emit_token signature: "C_partial",
                    dims: (row_idx, col_idx),
                    value: partial_sum,
                    tag: hash(row_idx, col_idx)
        .reset      partial_sum

    // Instruction 3: Add bias
    .inst add_bias:
        .match  {("C_partial", (row, col)), ("bias_element", (col))}
        .operation  add_bf16
        .emit_token signature: "C_biased", ...

    // Instruction 4: ReLU
    .inst apply_relu:
        .match  ("C_biased", (row, col))
        .operation  max_bf16(value, 0.0)
        .emit_token signature: "C_output", ...

.kernel_end
```

**This same program runs in ALL PEs!**

---

## Sequencer-Based Data Movement

### What is a Sequencer?

A **sequencer** is a small programmable state machine in each data movement engine:
- DMA engines
- Block movers
- Streamers

It executes a **schedule** - a sequence of steps for data movement.

### DMA Sequencer Example

```
dma.sequencer.define seq_input_load:
    .step   transfer:
                src=dram:0x1000,
                dst=l3_bank0:0x0000,
                size=1024,
                mode=burst

    .step   wait:
                condition=transfer_complete

    .step   signal:
                flag=input_in_l3

    .seq_end
```

When started, this sequencer:
1. Initiates DMA transfer
2. Waits for completion
3. Signals that data is ready
4. Other sequencers can trigger on this signal

### Streamer Sequencers: Creating Tagged Tokens

**This is where the magic happens!**

Streamers don't just move data - they **create tagged tokens**:

```
streamer.sequencer.define seq_input_stream:
    .step   stream_with_tags:
            for_each_element %i:
                value = src[%i],
                tag_signature = "A_element",
                tag_dims = (row_idx=0, k_idx=%i),
                tag_hash = hash(0, %i),
                emit_token(signature, dims, value, tag_hash)
    .seq_end
```

For each element read from L2:
1. Read value
2. Attach signature ("A_element")
3. Attach dimensions (row, k coordinates)
4. Compute routing hash
5. Emit as tagged token into fabric

The fabric receives these tokens and the dataflow computation begins!

---

## Execution Flow

### Phase 1: Configuration (Before Any Data Moves)

```
1. Install kernel in compute fabric
   ↓
   fabric.broadcast → all PEs get same program
   ↓
   PEs initialize CAMs and accumulators
   ↓
2. Configure DMA sequencers
   ↓
3. Configure Block Mover sequencers
   ↓
4. Configure Streamer sequencers
   ↓
5. Configure Output Drain sequencer
   ↓
   FABRIC IS READY, WAITING FOR TAGGED TOKENS
```

### Phase 2: Execution (Trigger Data Movement)

```
Start DMA engines
   ↓
Data moves DRAM → L3
   ↓
Trigger: Block movers start (L3 → L2)
   ↓
Trigger: Streamers start (L2 → L1 → Fabric)
   ↓
Streamers create TAGGED TOKENS
   ↓
Tokens flow into compute fabric
   ↓
PEs match tags in CAM
   ↓
Instructions FIRE when operands complete
   ↓
Results create new tagged tokens
   ↓
Tokens route to appropriate PEs
   ↓
Eventually, output tokens drain to L2
   ↓
L2 → L3 → DRAM writeback
   ↓
DONE
```

---

## Comparison: KPU vs TPU vs GPU

| Feature | KPU | TPU | GPU |
|---------|-----|-----|-----|
| **Architecture** | Domain flow | Systolic array | SIMT |
| **Programming** | Tagged tokens | Fixed dataflow | Threads |
| **Compute Fabric** | Programmable PEs | Fixed-function PEs | Shader cores |
| **Position** | Independent | Fixed | Fixed (warp-based) |
| **Data Movement** | Sequencers | Implicit streaming | Explicit loads |
| **Flexibility** | High | Low | Very high |
| **Efficiency** | Very high | Very high | High |
| **Adaptivity** | Yes (dataflow) | No | Limited |

### Key Differences

**KPU**:
- ✅ Position-independent computation
- ✅ Adaptive load balancing (tokens route anywhere)
- ✅ Programmable for different operations
- ✅ Explicit control over data movement hierarchy
- ✅ Tagged token matching in hardware (CAM)

**TPU**:
- ✅ Extremely efficient for fixed matrix operations
- ❌ Not programmable (fixed systolic array)
- ❌ Position-dependent (fixed north/west dataflow)
- ❌ Limited to matrix multiply patterns

**GPU**:
- ✅ Fully programmable
- ✅ Wide variety of operations
- ❌ Less efficient for structured computation
- ❌ Requires explicit thread management
- ❌ More power consumption

---

## Performance Characteristics

### Strengths

1. **Matrix Operations**: As efficient as TPU when configured for matmul
2. **Flexibility**: Can reconfigure for different operations
3. **Load Balancing**: Automatic via tagged token routing
4. **Memory Hierarchy**: Explicit control via sequencers
5. **Data Reuse**: Programmable patterns in streamers

### When to Use KPU

✅ **Ideal For**:
- Real-time signal processing and control
- Sensor Fusion
- Neural network training and inference
- Structured computations with regular patterns
- Workloads needing both efficiency and adaptivity
- R&D requiring hardware flexibility

❌ **Not Ideal For**:
- Fully irregular computations
- Very small workloads (overhead of configuration)
- Single-operation streaming (TPU better)

---

## Programming Model Summary

### Traditional (CPU/GPU)

```
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i,j] += A[i,k] * B[k,j]
```
**Explicit loops, explicit memory access**

### TPU

```
C = matmul(A, B)  # Compiled to systolic array
```
**Implicit execution on fixed hardware**

### KPU

```
1. Define domain flow kernel:
   "When A[i,k] and B[k,j] tokens meet → MAC → emit C[i,j]"

2. Configure data streamers:
   "Stream A as 'A_element' tokens with (i,k) tags"
   "Stream B as 'W_element' tokens with (k,j) tags"

3. Start streaming:
   → Tokens flow
   → Dataflow fires
   → Results emerge
```
**Declarative dataflow + explicit data movement control**

---

## Summary

The Stillwater KPU represents a **new point in the design space**:

- **Efficiency of TPU**: Systolic array organization, specialized for ML
- **Flexibility of GPU**: Programmable computation via domain flow programs
- **Explicit Control**: Sequencer-based hierarchical data movement
- **Dataflow Execution**: Tagged token matching, position-independent

It's a **domain-specific** architecture (neural networks) that maintains **programmability** through its tagged token dataflow model and sequencer-based data orchestration.

The key innovation: **Spatial + Dataflow = Domain Flow**

