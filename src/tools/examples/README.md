# MLP Assembly Implementations

This repository contains hand-written assembly implementations of a Multi-Layer Perceptron (MLP) neural network for different CPU and GPU architectures with SIMD optimizations.

## Supported Architectures

### CPUs

#### x86-64 (Intel/AMD)
- **AVX** (Advanced Vector Extensions): 256-bit SIMD, processes 8 single-precision floats
- **SSE** (Streaming SIMD Extensions): 128-bit SIMD, processes 4 single-precision floats

#### ARM
- **AArch64** (64-bit ARM): NEON SIMD, 128-bit, processes 4 single-precision floats
- **ARMv7** (32-bit ARM): NEON SIMD, 128-bit, processes 4 single-precision floats

### GPUs

#### NVIDIA Ampere
- **PTX** (Parallel Thread Execution): Portable assembly for NVIDIA GPUs
- **SASS** (Streaming ASSembler): Native machine code for Ampere architecture
- Warp-based execution (32 threads per warp)
- Tensor Core support for ML acceleration

#### ARM Mali
- **Bifrost** Architecture (G71, G72, G76): Quad-based execution model
- **Valhall** Architecture (G77, G78, G710): 16-wide warp execution
- Energy-efficient mobile GPU design

### TPUs (Tensor Processing Units)

#### Google TPU
- **XLA HLO**: High-Level Optimizer intermediate representation (documented)
- **Systolic Array**: 128×128 matrix multiply unit for extreme throughput
- **Pseudo-Assembly**: Representative assembly based on published architecture
- Purpose-built for large-scale neural network training and inference
- Note: Actual TPU assembly is proprietary and not publicly available

### KPUs (Knowledge Processing Units)

#### Stillwater KPU
- **Domain Flow Architecture**: Combines systolic arrays with dataflow machines
- **Tagged Token Dataflow**: Position-independent computation via CAM matching
- **Programmable Systolic Array**: Flexible PE mesh with domain flow programs
- **Hierarchical Sequencers**: Explicit control over DMA, Block Movers, Streamers
- Novel architecture for adaptive neural network processing

## Project Structure

```
.
├── CPU Implementations
│   ├── x86_64/
│   │   └── mlp_avx.s                # x86-64 AVX/SSE implementations
│   ├── arm/
│   │   ├── mlp_neon_aarch64.s       # ARM 64-bit NEON implementation
│   │   └── mlp_neon_armv7.s         # ARM 32-bit NEON implementation
│   ├── include/
│   │   └── mlp_asm.h                # C header file with function declarations
│   ├── src/
│   │   └── example.c                # Example usage and benchmarks
│   └── Makefile                     # CPU build system
│
├── GPU Implementations
│   └── gpu/
│       ├── nvidia_ampere/
│       │   ├── mlp_forward.ptx      # NVIDIA PTX assembly
│       │   └── mlp_forward.sass     # NVIDIA SASS assembly
│       ├── arm_mali/
│       │   ├── mlp_forward_bifrost.asm   # Mali Bifrost assembly
│       │   └── mlp_forward_valhall.asm   # Mali Valhall assembly
│       ├── cuda/
│       │   └── mlp_cuda_example.cu  # CUDA host code
│       ├── opencl/
│       │   └── mlp_opencl_example.c # OpenCL host code
│       ├── Makefile                 # GPU build system
│       ├── GPU_README.md            # GPU documentation
│       └── GPU_COMPARISON.md        # Architecture comparison
│
├── TPU Implementations
│   └── tpu/
│       ├── xla_hlo/
│       │   └── mlp_forward.hlo      # XLA HLO intermediate representation
│       ├── pseudo_assembly/
│       │   └── mlp_forward_tpu.asm  # TPU pseudo-assembly
│       ├── jax_examples/
│       │   └── mlp_jax_tpu.py       # JAX code for TPU
│       ├── TPU_README.md            # TPU documentation
│       └── SYSTOLIC_ARRAY_EXPLAINED.md  # Systolic array deep dive
│
├── KPU Implementations
│   └── kpu/
│       ├── assembly/
│       │   └── mlp_forward_kpu.asm  # KPU assembly with tagged tokens
│       ├── docs/
│       │   ├── KPU_ARCHITECTURE.md  # Detailed architecture guide
│       │   └── ARCHITECTURE_COMPARISON.md  # KPU vs CPU/GPU/TPU
│       └── KPU_README.md            # KPU documentation
│
└── Documentation
    ├── README.md                    # This file
    ├── INSTRUCTION_REFERENCE.md     # Detailed ISA reference
    └── SIDE_BY_SIDE_COMPARISON.md   # CPU ISA comparison
```

## Building

### CPU Implementations

#### For Current Architecture
```bash
make
```

#### Run Example
```bash
make run
```

#### Run Optimized Benchmark
```bash
make benchmark
```

#### Cross-Compilation
```bash
make x86_64     # For x86-64
make aarch64    # For ARM 64-bit
make armv7      # For ARM 32-bit
```

### GPU Implementations

#### NVIDIA CUDA (Ampere)
```bash
cd gpu
make cuda           # Build CUDA example
make run-cuda       # Build and run
make show-ptx       # Display PTX assembly
make show-sass      # Display SASS assembly
```

#### OpenCL (Mali and others)
```bash
cd gpu
make opencl         # Build OpenCL example
make run-opencl     # Build and run
make android        # Cross-compile for Android Mali
```

See `gpu/GPU_README.md` for detailed GPU documentation.

### TPU Implementations

#### JAX on Google TPU
```bash
# Run on Google Colab (free TPU access)
# Or Google Cloud TPU VM

cd tpu/jax_examples
pip install jax[tpu]
python mlp_jax_tpu.py

# View XLA HLO IR
cat ../xla_hlo/mlp_forward.hlo

# Read architecture documentation
cat ../TPU_README.md
cat ../SYSTOLIC_ARRAY_EXPLAINED.md
```

See `tpu/TPU_README.md` for detailed TPU documentation and `tpu/SYSTOLIC_ARRAY_EXPLAINED.md` for how the systolic array works.

### KPU Implementations

#### Stillwater KPU - Domain Flow Architecture
```bash
# The KPU is a next-generation architecture combining systolic 
# arrays with dataflow machines using tagged token matching

cd kpu

# View assembly implementation
cat assembly/mlp_forward_kpu.asm

# Read architecture documentation
cat KPU_README.md
cat docs/KPU_ARCHITECTURE.md

# Compare all four architectures
cat docs/ARCHITECTURE_COMPARISON.md
```

See `kpu/KPU_README.md` for detailed KPU documentation and `kpu/docs/KPU_ARCHITECTURE.md` for the complete architecture guide.

## Implementation Details

### MLP Layer Forward Pass

Each implementation computes:
```
output = ReLU(weights × input + bias)
```

Where:
- **input**: Input vector of size `input_size`
- **weights**: Weight matrix of size `[output_size × input_size]` (row-major)
- **bias**: Bias vector of size `output_size`
- **output**: Output vector of size `output_size`
- **ReLU**: Rectified Linear Unit activation function `max(0, x)`

### Algorithm

For each output neuron:
1. Compute dot product of input vector and corresponding weight row
2. Add bias term
3. Apply ReLU activation
4. Store result

## SIMD Instruction Set Comparison

### x86-64 AVX (256-bit)

**Key Instructions:**
- `vmovups`: Load/store 8 unaligned floats
- `vfmadd231ps`: Fused multiply-add (FMA): `a = a + (b × c)` for 8 floats
- `vaddps`: Add 8 floats in parallel
- `vmaxps`: Maximum of 8 floats (for ReLU)
- `vhaddps`: Horizontal add (sum elements within vector)
- `vextractf128`: Extract upper 128 bits from 256-bit register

**Advantages:**
- Largest SIMD width (8 floats)
- FMA instruction for efficient MAC operations
- Rich instruction set

**Example (Dot Product Loop):**
```asm
vmovups ymm1, [r12 + r10*4]      # Load 8 input values
vmovups ymm2, [r11 + r10*4]      # Load 8 weight values
vfmadd231ps ymm0, ymm1, ymm2     # ymm0 += ymm1 * ymm2 (8 MACs)
```

### x86-64 SSE (128-bit)

**Key Instructions:**
- `movups`: Load/store 4 unaligned floats
- `mulps`: Multiply 4 floats in parallel
- `addps`: Add 4 floats in parallel
- `maxps`: Maximum of 4 floats
- `haddps`: Horizontal add

**Advantages:**
- Widely supported (even on older CPUs)
- Lower power consumption than AVX

**Example:**
```asm
movups xmm1, [r12 + r10*4]       # Load 4 input values
movups xmm2, [r11 + r10*4]       # Load 4 weight values
mulps xmm1, xmm2                 # xmm1 *= xmm2
addps xmm0, xmm1                 # xmm0 += xmm1
```

### ARM NEON (128-bit)

**Key Instructions:**
- `ldr q#`: Load 4 floats into NEON Q register
- `fmla v#.4s, v#.4s, v#.4s`: Fused multiply-accumulate for 4 floats
- `fadd v#.4s, v#.4s, v#.4s`: Add 4 floats
- `fmax`: Maximum (for ReLU)
- `faddp`: Pairwise add (for horizontal reduction)

**Advantages:**
- Energy efficient
- Built into modern ARM processors (mobile, embedded)
- Unified instruction set across ARM implementations

**Example (AArch64):**
```asm
ldr q1, [x19, x7, lsl #2]        # Load 4 input values
ldr q2, [x8, x7, lsl #2]         # Load 4 weight values
fmla v0.4s, v1.4s, v2.4s         # v0 += v1 * v2 (4 MACs)
```

**Example (ARMv7):**
```asm
vld1.32 {q1}, [r2]               # Load 4 input values
vld1.32 {q2}, [r3]               # Load 4 weight values
vmla.f32 q0, q1, q2              # q0 += q1 * q2
```

## Architectural Differences

### Register Usage

| Architecture | SIMD Registers | Width | Elements | Callee-Saved |
|--------------|---------------|-------|----------|--------------|
| x86-64 AVX   | ymm0-ymm15    | 256-bit | 8 floats | None |
| x86-64 SSE   | xmm0-xmm15    | 128-bit | 4 floats | None |
| ARM AArch64  | v0-v31 (q0-q31) | 128-bit | 4 floats | v8-v15 (d8-d15) |
| ARM ARMv7    | q0-q15 (d0-d31) | 128-bit | 4 floats | d8-d15 |

### Calling Conventions

**x86-64 System V ABI:**
- Integer args: `rdi, rsi, rdx, rcx, r8, r9`
- Return value: `rax`
- Callee-saved: `rbx, rbp, r12-r15`
- SIMD: All `xmm`/`ymm` registers are caller-saved

**ARM AArch64 AAPCS64:**
- Integer args: `x0-x7`
- Return value: `x0`
- Callee-saved: `x19-x28`
- SIMD: `v8-v15` (d8-d15) are callee-saved

**ARM ARMv7 AAPCS:**
- Integer args: `r0-r3`, then stack
- Return value: `r0`
- Callee-saved: `r4-r11`
- SIMD: `d8-d15` are callee-saved

### Key Differences

1. **SIMD Width**: x86-64 AVX uses 256-bit (8 floats), ARM NEON and x86-64 SSE use 128-bit (4 floats)

2. **FMA Support**:
   - x86-64: `vfmadd231ps` (AVX2/FMA3)
   - ARM: `fmla` (native in NEON)

3. **Memory Addressing**:
   - x86-64: Complex addressing modes like `[base + index*scale + offset]`
   - ARM: Simpler modes, uses shift operations `[base, index, lsl #2]`

4. **Instruction Syntax**:
   - x86-64 (Intel syntax): Destination first `mov rax, rbx` (rax = rbx)
   - ARM: Destination first `mov x0, x1` (x0 = x1)
   - Both follow destination-first convention

5. **Horizontal Operations**:
   - x86-64: `vhaddps` for horizontal sum (complex)
   - ARM: `faddp` pairwise add (simpler, requires multiple steps)

## Performance Considerations

### Optimization Techniques Used

1. **SIMD Vectorization**: Process multiple elements in parallel
   - x86-64 AVX: 8 floats per iteration
   - ARM/SSE: 4 floats per iteration

2. **Loop Unrolling**: Optimized versions process 8 floats (2×4) per iteration
   - Reduces loop overhead
   - Better instruction-level parallelism
   - More efficient use of registers

3. **FMA Instructions**: Single instruction for multiply-accumulate
   - Reduces instruction count
   - Better throughput
   - Higher accuracy (intermediate result not rounded)

4. **Register Reuse**: Multiple accumulators reduce data dependencies
   - Allows CPU to execute instructions out-of-order
   - Hides memory latency

5. **Memory Alignment**: 32-byte aligned allocations
   - Faster SIMD loads/stores
   - Avoid cache line splits

### Expected Performance

Approximate speedups over scalar C code (depends on CPU, problem size):
- **AVX (x86-64)**: 5-7x speedup
- **SSE (x86-64)**: 3-4x speedup
- **NEON (ARM)**: 3-4x speedup

Factors affecting performance:
- Input/output sizes (larger = better amortization of overhead)
- Memory bandwidth (large weight matrices)
- Cache behavior (locality of access)
- CPU microarchitecture (execution units, out-of-order depth)

## Usage Example

```c
#include "mlp_asm.h"

// Allocate aligned memory
float* input = aligned_alloc(32, 128 * sizeof(float));
float* weights = aligned_alloc(32, 64 * 128 * sizeof(float));
float* bias = aligned_alloc(32, 64 * sizeof(float));
float* output = aligned_alloc(32, 64 * sizeof(float));

// Initialize data...

// Run forward pass
mlp_layer_forward(input, weights, bias, output, 128, 64);

// Use output...

free(input);
free(weights);
free(bias);
free(output);
```

The `mlp_layer_forward` macro automatically selects the best implementation for your architecture.

## Testing

The example program includes:
1. **Correctness verification**: Compares assembly output with C reference
2. **Performance benchmark**: Measures speedup vs. scalar implementation
3. **Multi-layer example**: Demonstrates composition of layers

Run with:
```bash
make run
```

## Instruction Set Architecture (ISA) Details

### x86-64 ISA Features

**Instruction Format**: Variable length (1-15 bytes)
- Prefix bytes (optional)
- Opcode (1-3 bytes)
- ModR/M byte (addressing mode)
- SIB byte (scale-index-base)
- Displacement (0-4 bytes)
- Immediate (0-4 bytes)

**Register File**:
- 16 general-purpose 64-bit registers (rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8-r15)
- 16 vector registers (ymm0-ymm15 for AVX, xmm0-xmm15 for SSE)
- RFLAGS status register

**Memory Model**: Little-endian, byte-addressable

### ARM ISA Features

**Instruction Format**: Fixed 32-bit (AArch64/ARMv7)

**AArch64 Register File**:
- 31 general-purpose 64-bit registers (x0-x30)
- 32 SIMD/FP registers (v0-v31), can be accessed as:
  - Q registers (128-bit): q0-q31
  - D registers (64-bit): d0-d31
  - S registers (32-bit): s0-s31
- Stack pointer (SP)
- Program counter (PC)
- Processor state (PSTATE)

**ARMv7 Register File**:
- 16 general-purpose 32-bit registers (r0-r15)
- 16 SIMD/FP Q registers (128-bit): q0-q15
  - Or 32 D registers (64-bit): d0-d31
- CPSR (Current Program Status Register)

**Memory Model**: Little-endian (configurable), byte-addressable

## Learning Resources

### Understanding the Code

Start with these files in order:
1. `include/mlp_asm.h` - Function interfaces
2. `src/example.c` - How to use the functions
3. `x86_64/mlp_avx.s` - AVX implementation (most features)
4. `arm/mlp_neon_aarch64.s` - ARM 64-bit implementation

### Key Concepts

- **SIMD**: Single Instruction, Multiple Data
- **FMA**: Fused Multiply-Add
- **Horizontal reduction**: Summing elements within a SIMD register
- **Loop unrolling**: Replicating loop body to reduce overhead
- **Calling conventions**: How arguments are passed to functions

### External Resources

- Intel Intrinsics Guide: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
- ARM NEON Programmer's Guide: https://developer.arm.com/architectures/instruction-sets/simd-isas/neon
- x86-64 ABI: https://gitlab.com/x86-psABIs/x86-64-ABI
- ARM AAPCS64: https://github.com/ARM-software/abi-aa

## License

This code is provided for educational purposes. Feel free to use, modify, and distribute.

## Acknowledgments

This implementation demonstrates:
- Low-level neural network computation
- SIMD programming techniques
- Assembly optimization strategies
- Architecture-specific considerations

Perfect for studying:
- Computer architecture
- Parallel computing
- Neural network implementation
- Performance optimization
