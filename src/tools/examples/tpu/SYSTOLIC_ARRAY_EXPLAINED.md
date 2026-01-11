# Systolic Array Architecture - Deep Dive

This document explains how Google TPU's systolic array works to accelerate matrix multiplication, which is the core operation in neural networks.

## What Problem Does It Solve?

Matrix multiplication is the bottleneck in neural networks:
- Forward pass: activations × weights
- Backward pass: gradients × weights
- Parameter update: activations^T × gradients

**Naive algorithm**: O(N³) operations, lots of memory traffic
**Systolic array**: O(N) time with N² processing elements, minimal memory traffic

## Systolic Array Basics

### The Name "Systolic"

The term comes from biology:
- **Systole**: The contraction phase of the heartbeat
- **Systolic array**: Data pulses through the array rhythmically, like a heartbeat

Data flows through the array in waves, with each "beat" (clock cycle) moving data one step further.

## Simple 4×4 Example

Let's compute C = A × B where:
- A is 4×4
- B is 4×4
- C will be 4×4

### Array Structure

```
        Input B (flows DOWN)
        b₀  b₁  b₂  b₃
        ↓   ↓   ↓   ↓
a₀ →  [PE][PE][PE][PE] → output row 0
a₁ →  [PE][PE][PE][PE] → output row 1
a₂ →  [PE][PE][PE][PE] → output row 2
a₃ →  [PE][PE][PE][PE] → output row 3
        ↓   ↓   ↓   ↓
        c₀  c₁  c₂  c₃
```

Each PE (Processing Element) contains:
```
┌─────────────────┐
│ Accumulator     │  ← Stores partial sum
│ Weight (static) │  ← Loaded once, stays in PE
│                 │
│ Operation:      │
│ acc = acc + a×b │  ← Multiply-accumulate (MAC)
└─────────────────┘
```

### Data Flow

**Key Principle**: Reuse! Each data item is used multiple times as it flows through.

#### Inputs Flow Pattern

```
         B Matrix Values
Cycle    Col0  Col1  Col2  Col3
  0:     b₀₀   b₀₁   b₀₂   b₀₃  ← Enter top row
  1:     b₁₀   b₁₁   b₁₂   b₁₃  ← Enter top row
  2:     b₂₀   b₂₁   b₂₂   b₂₃  ← Previous values flow down
  3:     b₃₀   b₃₁   b₃₂   b₃₃

         A Matrix Values
Cycle   Row0  Row1  Row2  Row3
  0:    a₀₀   —     —     —     ← Enter left column
  1:    a₀₁   a₁₀   —     —     ← Staggered timing
  2:    a₀₂   a₁₁   a₂₀   —
  3:    a₀₃   a₁₂   a₂₁   a₃₀
  4:    —     a₁₃   a₂₂   a₃₁
  5:    —     —     a₂₃   a₃₂
  6:    —     —     —     a₃₃
```

**Note the stagger**: Each row of A starts one cycle later than the previous row. This ensures elements meet at the right PE.

### Cycle-by-Cycle Execution

Let's trace what happens in PE[1,1] (second row, second column):

```
Cycle 0:  PE[1,1] idle
Cycle 1:  PE[1,1] idle
Cycle 2:  Receives a₁₀ (from left), b₀₁ (from top)
          Computes: acc₁₁ = 0 + a₁₀ × b₀₁
Cycle 3:  Receives a₁₁, b₁₁
          Computes: acc₁₁ = acc₁₁ + a₁₁ × b₁₁
Cycle 4:  Receives a₁₂, b₂₁
          Computes: acc₁₁ = acc₁₁ + a₁₂ × b₂₁
Cycle 5:  Receives a₁₃, b₃₁
          Computes: acc₁₁ = acc₁₁ + a₁₃ × b₃₁
          Result: C₁₁ = a₁₀×b₀₁ + a₁₁×b₁₁ + a₁₂×b₂₁ + a₁₃×b₃₁ ✓
```

### Complete Timeline

```
Cycle  PE[0,0]        PE[0,1]        PE[1,0]        PE[1,1]
  0    a₀₀×b₀₀        —              —              —
  1    a₀₁×b₁₀        a₀₀×b₀₁        a₁₀×b₀₀        —
  2    a₀₂×b₂₀        a₀₁×b₁₁        a₁₁×b₁₀        a₁₀×b₀₁
  3    a₀₃×b₃₀        a₀₂×b₂₁        a₁₂×b₂₀        a₁₁×b₁₁
  4    → C₀₀ ready    a₀₃×b₃₁        a₁₃×b₃₀        a₁₂×b₂₁
  5    —              → C₀₁ ready    → C₁₀ ready    a₁₃×b₃₁
  6    —              —              —              → C₁₁ ready
```

Results appear diagonally (wavefront pattern):
- Cycle 4: C₀₀
- Cycle 5: C₀₁, C₁₀
- Cycle 6: C₀₂, C₁₁, C₂₀
- ...
- Cycle 9: C₃₃ (last result)

## Scalability: 128×128 TPU Array

Google TPU v2/v3 uses a **128×128 systolic array** = 16,384 PEs!

### Performance Analysis

For multiplying 128×128 matrices:

**Traditional CPU**:
- Operations: 128³ = 2,097,152 multiply-adds
- Time (at 1 MAC/cycle): 2,097,152 cycles
- At 2 GHz: ~1 millisecond

**TPU 128×128 Systolic Array**:
- Operations: Still 128³ = 2,097,152 MACs
- But 16,384 PEs work in parallel!
- Time: 128 + 128 - 1 = 255 cycles (wavefront propagation)
- At 1 GHz: **0.255 microseconds** (0.000255 ms)
- **Speedup: ~4,000x** vs serial execution

### Throughput

Once the pipeline is full:
- **16,384 MACs per cycle**
- At 1 GHz: 16.4 trillion MACs per second
- With BF16: **32.8 TFLOPS** (just from the MXU!)

### Memory Efficiency

**Key advantage**: Weights stay in PEs (no repeated loads)

For 128×128×128 matrix multiply:
- **Traditional**: Load 128³ weights from memory
- **Systolic**: Load each weight once, reuse in PE

Memory accesses:
- Weights: 128² = 16,384 loads (loaded to PEs, stay there)
- A matrix: 128² = 16,384 loads (each used 128 times as it flows)
- B matrix: 128² = 16,384 loads (each used 128 times as it flows)
- **Total**: ~49K loads vs 2M operations = **41x data reuse!**

## Tiling for Large Matrices

Real matrices are often larger than 128×128. TPU tiles them.

Example: Multiply 512×512 matrices on 128×128 array

### Tiling Strategy

Divide into 128×128 tiles:
- 512/128 = 4 tiles per dimension
- Total: 4×4 = 16 tiles per matrix

```
Matrix A (512×512):          Matrix B (512×512):
┌────┬────┬────┬────┐       ┌────┬────┬────┬────┐
│ A₀ │ A₁ │ A₂ │ A₃ │       │ B₀ │ B₁ │ B₂ │ B₃ │
├────┼────┼────┼────┤       ├────┼────┼────┼────┤
│ A₄ │ A₅ │ A₆ │ A₇ │       │ B₄ │ B₅ │ B₆ │ B₇ │
├────┼────┼────┼────┤       ├────┼────┼────┼────┤
│ A₈ │ A₉ │ A₁₀│ A₁₁│       │ B₈ │ B₉ │ B₁₀│ B₁₁│
├────┼────┼────┼────┤       ├────┼────┼────┼────┤
│ A₁₂│ A₁₃│ A₁₄│ A₁₅│       │ B₁₂│ B₁₃│ B₁₄│ B₁₅│
└────┴────┴────┴────┘       └────┴────┴────┴────┘
  Each tile is 128×128         Each tile is 128×128
```

### Computation Pattern

To compute result tile C₀ (top-left 128×128 of C):

```
C₀ = A₀×B₀ + A₁×B₄ + A₂×B₈ + A₃×B₁₂
     ↑      ↑      ↑      ↑
   tile0  tile1  tile2  tile3
```

Algorithm:
1. Clear accumulator
2. For each k-tile (0 to 3):
   - Load A-tile and B-tile
   - Compute tile product (255 cycles)
   - Accumulate into result
3. Store final result

Total time for one output tile:
- 4 tile products × 255 cycles = 1,020 cycles
- Plus DMA transfer overhead
- At 1 GHz: ~1 microsecond per 128×128 tile

For full 512×512 result (16 output tiles):
- 16 × 1,020 cycles = 16,320 cycles
- At 1 GHz: ~16 microseconds
- Operations: 512³ = 134 million MACs
- Throughput: **8.2 TFLOPS**

## Weight Stationarity

TPU uses "weight stationary" dataflow:

```
             ┌─────────────────┐
             │  HBM (DRAM)     │
             │  1.2 TB/s       │
             └────────┬────────┘
                      │
                      │ Load weights once
                      ↓
             ┌─────────────────┐
             │ Unified Buffer  │
             └────────┬────────┘
                      │
                      ↓
             ┌─────────────────┐
             │ Systolic Array  │
             │                 │
             │ Weights stay    │ ← Key optimization!
             │ in PEs during   │
             │ entire batch    │
             └─────────────────┘
                      ↑
                      │ Stream activations
                      │
             ┌─────────────────┐
             │ Activation Data │
             └─────────────────┘
```

**Benefit**: For batch processing:
- Load weights once
- Stream batch of inputs through
- Massive reuse → minimal memory traffic

Example with batch size 128:
- Load weights: 1×
- Process 128 inputs: stream through array
- Each weight used 128 times
- **128× reduction in weight memory traffic!**

## Comparison with GPU

### NVIDIA GPU (Tensor Cores)

```
Tensor Core: 4×4 matrix multiply per cycle
Multiple cores: 432 Tensor Cores on A100
Flexibility: Programmable, many operations
Memory: Explicit shared memory management
```

### TPU Systolic Array

```
MXU: 128×128 matrix multiply
Single unit: One large systolic array
Flexibility: Fixed function, matrix-only
Memory: Automatic weight stationarity
```

| Aspect | GPU Tensor Core | TPU Systolic Array |
|--------|----------------|-------------------|
| Size | 4×4 per core | 128×128 monolithic |
| Total | 432 cores = 6.9K PEs | 16,384 PEs |
| Flexibility | High (programmable) | Low (fixed function) |
| Efficiency | Good | Excellent (for matmul) |
| Power | Higher | Lower (specialized) |
| Use Case | General ML/HPC | Large-scale ML |

## Why Systolic Arrays for ML?

### Advantages

1. **Data Reuse**: Each value used multiple times as it flows
2. **No Memory Bottleneck**: Minimal off-chip memory access
3. **Regular Structure**: Easy to scale (N×N → 2N×2N)
4. **Energy Efficient**: No instruction fetch, simple control
5. **Predictable**: No cache misses, no divergence

### Disadvantages

1. **Inflexible**: Only good for matrix multiplication
2. **Utilization**: Poor on non-rectangular matrices
3. **Latency**: Wavefront takes time to propagate
4. **Wasted Work**: Padding needed for non-multiples of 128

## MLP Execution on TPU

### Forward Pass: Input × Weights

```
Input: [1 × 512]
Weights: [256 × 512]
Output: [1 × 256]

Tiling:
- 256/128 = 2 output tiles
- 512/128 = 4 input tiles
- 8 total tile multiplications

Timeline:
  Cycle 0-255:    Tile [0,0] (first 128 outputs, first 128 inputs)
  Cycle 256-511:  Tile [0,1] (first 128 outputs, next 128 inputs)
  Cycle 512-767:  Tile [0,2]
  Cycle 768-1023: Tile [0,3]
  Cycle 1024-:    Tile [1,0] (next 128 outputs, first 128 inputs)
  ...
  Cycle ~2040:    All tiles complete

Total: ~2K cycles ≈ 2 microseconds at 1 GHz
```

### Batch Processing (128 examples)

```
Input: [128 × 512]
Weights: [256 × 512]
Output: [128 × 256]

Strategy: Keep weights in array, stream 128 inputs

Time: ~128× longer = 256 microseconds
Throughput: (128 × 512 × 256 × 2) / 256µs = 65 TFLOPS!
```

## Visualizing Data Movement

### Single Cycle Snapshot

```
Cycle N in a 4×4 array:

        ┌────┐
        │ b₁ │ ← Input column flowing down
        └─┬──┘
          ↓
   ┌───┐ ╔════╗
   │a₂ │→║ PE ║ ← This PE:
   └───┘ ║────║    - Multiplies a₂ × b₁
         ║acc ║    - Adds to accumulator
         ╚══╦═╝    - Passes a₂ right
            ↓       - Passes b₁ down
         ┌──┴──┐
         │  b₁ │ → To next PE below
         └─────┘
```

### Full Array (Cycle 3)

```
        b₀₀  b₁₁  b₂₂  b₃₃
         ↓    ↓    ↓    ↓
a₀₃ →  [**][**][**][**]
a₁₂ →  [**][**][**][ ]
a₂₁ →  [**][**][ ][ ]
a₃₀ →  [**][ ][ ][ ]
         ↓    ↓    ↓    ↓

[**] = PE actively computing
[ ]  = PE idle (wavefront hasn't reached)

Diagonal wavefront of computation!
```

## Summary

**Systolic Array = Purpose-Built Matrix Multiplication Engine**

Key insights:
1. **Parallelism**: N² PEs work simultaneously
2. **Data Flow**: Values flow rhythmically through array
3. **Reuse**: Each value used multiple times (weights stay put)
4. **Efficiency**: Minimal memory traffic
5. **Specialization**: Trade flexibility for performance

For neural networks dominated by matrix multiplication:
- **TPU systolic array**: Unbeatable energy efficiency
- **GPU Tensor Cores**: More flexible, also very fast
- **CPU SIMD**: Most flexible, but much slower

The systolic array is why TPUs can achieve **100+ TFLOPS** on real ML workloads while using less power than GPUs!

