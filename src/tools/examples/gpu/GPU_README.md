# GPU Assembly Implementations - MLP Forward Pass

This directory contains hand-written GPU assembly implementations of a Multi-Layer Perceptron for two major GPU architectures:
- **NVIDIA Ampere** (PTX and SASS assembly)
- **ARM Mali** (Bifrost and Valhall shader assembly)

## Directory Structure

```
gpu/
├── nvidia_ampere/
│   ├── mlp_forward.ptx          # PTX assembly (portable)
│   └── mlp_forward.sass         # SASS assembly (Ampere-specific)
├── arm_mali/
│   ├── mlp_forward_bifrost.asm  # Bifrost architecture (G71-G76)
│   └── mlp_forward_valhall.asm  # Valhall architecture (G77-G710)
├── cuda/
│   └── mlp_cuda_example.cu      # CUDA host code and benchmarks
├── opencl/
│   └── mlp_opencl_example.c     # OpenCL code for Mali
└── GPU_README.md                # This file
```

## GPU Architectures

### NVIDIA Ampere (SM 8.0)

**Architecture Highlights:**
- Streaming Multiprocessors (SMs) with 128 CUDA cores each
- Tensor Cores for matrix operations
- 256 KB L2 cache per SM
- Warp size: 32 threads
- Up to 65,536 registers per SM

**Programming Model:**
- Threads organized in warps (32 threads)
- Warps organized in thread blocks
- Thread blocks organized in grid
- Shared memory: software-managed cache within SM

**Assembly Languages:**
- **PTX**: Portable intermediate representation (like LLVM IR for GPUs)
- **SASS**: Native machine code for specific GPU architecture

### ARM Mali

#### Bifrost Architecture (G71, G72, G76)

**Architecture Highlights:**
- Quad-based execution (4 threads minimum)
- Multiple execution engines per shader core
- Unified shader architecture
- 16-wide wavefronts

**Key Features:**
- Clause-based execution (groups of instructions)
- Split register file (general purpose + vector)
- Hardware scheduling
- Energy-efficient design

#### Valhall Architecture (G77, G78, G710)

**Architecture Highlights:**
- Improved from Bifrost: better ILP and throughput
- 16-wide warps (fixed, vs. variable in Bifrost)
- Enhanced FMA units (dual-issue)
- Better register renaming

**Key Features:**
- Unified instruction encoding
- Improved vectorization
- Better divergence handling
- Lower latency operations

## Assembly Language Comparison

### NVIDIA PTX vs. ARM Mali

| Feature | NVIDIA PTX | ARM Mali (Bifrost) | ARM Mali (Valhall) |
|---------|-----------|-------------------|-------------------|
| Execution Model | SIMT (32-wide warps) | Quad-based (4-wide min) | Warp-based (16-wide) |
| Register File | 255 x 32-bit per thread | 64 x 32-bit per thread | 64 x 32-bit per thread |
| Shared Memory | Explicit (programmable) | L2 cache (hardware) | L2 cache (hardware) |
| ISA Type | RISC-like | VLIW-like | Improved VLIW |
| Instruction Width | Variable | Variable | Fixed (improved) |
| Vector Operations | Via registers | Native vec2/3/4 | Enhanced vec operations |
| Synchronization | __syncthreads(), barriers | barrier() | barrier(), improved |

## Key Differences

### Execution Model

**NVIDIA Ampere:**
```
Grid
└── Thread Blocks (up to 1024 threads)
    └── Warps (32 threads, lockstep execution)
        └── Threads
```

**ARM Mali:**
```
Global Work Size
└── Work Groups (like thread blocks)
    └── Quads (Bifrost: 4 threads) or Warps (Valhall: 16 threads)
        └── Work Items (like threads)
```

### Memory Hierarchy

**NVIDIA Ampere:**
- **Global Memory**: Device DRAM (GB scale)
- **L2 Cache**: Shared across SMs (MB scale)
- **L1 Cache**: Per SM (KB scale)
- **Shared Memory**: Programmable per-block cache (up to 164 KB)
- **Registers**: Private per thread (255 x 32-bit)

**ARM Mali:**
- **System Memory**: Unified with CPU
- **L2 Cache**: Shared across shader cores
- **L1 Cache**: Per shader core
- **No Shared Memory**: Relies on L2 cache coherency
- **Registers**: Private per thread (64 x 32-bit)

### Parallelism

**NVIDIA Ampere:**
- Massive parallelism: thousands of threads per SM
- 32-thread warps execute in lockstep
- Warp divergence can reduce efficiency
- High thread count hides memory latency

**ARM Mali:**
- Moderate parallelism: hundreds of threads per core
- More energy efficient
- Smaller warp/quad sizes
- Relies more on instruction-level parallelism

## MLP Implementation Details

### Algorithm

Both implementations compute:
```
output[i] = ReLU(dot(weights[i], input) + bias[i])
```

For each output neuron `i`:
1. Load input vector to fast memory (shared/local/L2)
2. Compute dot product with weight row
3. Add bias
4. Apply ReLU activation (max(0, x))
5. Store result

### NVIDIA PTX Implementation

**Key Features:**
- Uses shared memory for input vector
- Vectorized loads (4 floats at a time)
- Fused multiply-add (FMA) instructions
- Horizontal warp reduction for optimization
- Predicated execution for bounds checking

**Instruction Highlights:**
```ptx
// Vectorized load (128-bit)
ld.global.ca.v4.f32 {%f20, %f21, %f22, %f23}, [%rd9];

// Fused multiply-add
fma.rn.f32 %f10, %f20, %f24, %f10;

// Warp shuffle for reduction
shfl.down.b32 %f30, %f10, 16, 31;

// Synchronization
bar.sync 0;
```

### ARM Mali Implementation

**Key Features:**
- Uses L2 cache (no explicit shared memory)
- vec4 operations for vectorization
- Multiple accumulators for ILP
- Native FMA support
- Efficient for mobile/embedded

**Instruction Highlights (Valhall):**
```asm
// Vectorized load
VLD.vec4.f32 v4, [r14 + #0]

// Vector FMA (4 operations)
VFMA.vec4.f32 v0, v4, v8, v0

// Horizontal reduction
VEXTRACT.f32 f0, v0, #0
```

## Building and Running

### NVIDIA CUDA/PTX

#### Requirements:
- NVIDIA GPU with Compute Capability 8.0+ (Ampere)
- CUDA Toolkit 11.0+
- nvcc compiler

#### Build:
```bash
cd gpu/cuda

# Compile CUDA example
nvcc -arch=sm_80 -O3 mlp_cuda_example.cu -o mlp_cuda

# Run
./mlp_cuda
```

#### Compile PTX directly:
```bash
nvcc -ptx -arch=sm_80 kernel.cu -o kernel.ptx
```

#### Disassemble to SASS:
```bash
cuobjdump -sass kernel.cubin
```

### ARM Mali / OpenCL

#### Requirements:
- ARM Mali GPU (G71 or newer)
- OpenCL 1.2+ drivers
- gcc or clang

#### Build:
```bash
cd gpu/opencl

# Linux/Android
gcc -O3 mlp_opencl_example.c -o mlp_opencl -lOpenCL

# Run
./mlp_opencl
```

#### Note:
Mali assembly is generated by the driver compiler. To view:
```bash
# If using Mali Offline Compiler (malioc)
malioc --core mali-g78 -V kernel.cl
```

## Performance Characteristics

### NVIDIA Ampere (typical A100)

| Metric | Value |
|--------|-------|
| CUDA Cores | 6912 |
| Tensor Cores | 432 |
| Peak FP32 | 19.5 TFLOPS |
| Memory Bandwidth | 1555 GB/s |
| L2 Cache | 40 MB |
| TDP | 250-400W |

**Expected MLP Performance:**
- Small networks (512x256): ~5-10 µs per inference
- Memory-bound for small batches
- Compute-bound for large batches

### ARM Mali (typical G78 MP24)

| Metric | Value |
|--------|-------|
| Shader Cores | 24 |
| Peak FP32 | ~1.5 TFLOPS |
| Memory Bandwidth | ~50 GB/s (unified) |
| L2 Cache | 2 MB |
| TDP | 5-10W |

**Expected MLP Performance:**
- Small networks (512x256): ~50-100 µs per inference
- Always memory-bound
- Lower absolute performance but much better perf/watt

## Optimization Strategies

### NVIDIA Ampere

1. **Maximize Occupancy**: Use many threads to hide latency
2. **Coalesce Memory Access**: Ensure contiguous 128-byte loads
3. **Use Shared Memory**: Reduce global memory traffic
4. **Minimize Divergence**: Keep warps in lockstep
5. **Use Tensor Cores**: For larger matrix operations (WMMA)
6. **Optimize Register Usage**: Allow more concurrent warps

### ARM Mali

1. **Vectorize Operations**: Use vec4 for 4x throughput
2. **Multiple Accumulators**: Improve instruction-level parallelism
3. **Minimize Divergence**: Small warps make divergence expensive
4. **Cache-Friendly Access**: Leverage L2 cache instead of shared memory
5. **Reduce Precision**: FP16 often 2x faster than FP32
6. **Minimize Register Pressure**: Only 64 registers per thread

## PTX vs. SASS

### PTX (Parallel Thread Execution)

**Advantages:**
- Portable across NVIDIA GPU generations
- Human-readable and writable
- Can be JIT-compiled for new architectures
- Easier to debug and understand

**Disadvantages:**
- Not the actual machine code (requires compilation)
- May not expose all hardware features
- Driver optimizations can vary

**Example PTX:**
```ptx
fma.rn.f32 %f10, %f20, %f24, %f10;  // %f10 = %f20 * %f24 + %f10
```

### SASS (Streaming ASSembler)

**Advantages:**
- Actual GPU machine code
- Maximum performance control
- Access to all hardware features
- No runtime compilation overhead

**Disadvantages:**
- Architecture-specific (not portable)
- Generated by compiler (hard to write by hand)
- Undocumented (reverse-engineered)
- Harder to read and debug

**Example SASS:**
```sass
FFMA R20, R24, R32, R20 ;  // Same FMA operation
```

## Mali Assembly Notes

ARM Mali assembly is **not officially documented** by ARM. The examples in this repository are based on:

1. Reverse-engineering using tools like:
   - Mali Offline Compiler (malioc)
   - GLES/Vulkan shader disassemblers
   - Open-source drivers

2. ARM's high-level documentation:
   - Mali GPU architecture guides
   - OpenCL optimization guides
   - Bifrost/Valhall architecture whitepapers

3. Community reverse-engineering efforts

**Important**: Real Mali shader assembly is generated by the driver and varies by:
- GPU generation (Midgard → Bifrost → Valhall)
- Driver version
- Compiler optimizations
- Runtime kernel specialization

The assembly shown here is **representative** and **educational** rather than exact machine code.

## Instruction Set References

### NVIDIA PTX Instructions (used in MLP)

| Instruction | Description | Example |
|------------|-------------|---------|
| `ld.global` | Load from global memory | `ld.global.ca.f32 %f1, [%rd1]` |
| `st.global` | Store to global memory | `st.global.wb.f32 [%rd14], %f10` |
| `ld.shared` | Load from shared memory | `ld.shared.f32 %f31, [%r17]` |
| `st.shared` | Store to shared memory | `st.shared.f32 [%r10], %f1` |
| `fma.rn.f32` | FP32 fused multiply-add | `fma.rn.f32 %f10, %f20, %f24, %f10` |
| `add.f32` | FP32 add | `add.rn.f32 %f10, %f10, %f32` |
| `max.f32` | FP32 maximum | `max.f32 %f10, %f10, %f33` |
| `bar.sync` | Barrier synchronization | `bar.sync 0` |
| `shfl.down` | Warp shuffle down | `shfl.down.b32 %f30, %f10, 16, 31` |

### ARM Mali Instructions (representative)

#### Bifrost
| Instruction | Description | Example |
|------------|-------------|---------|
| `load.vec4.f32` | Load 4 floats | `load.vec4.f32 v1, [r14]` |
| `store.f32` | Store float | `store.f32 [r30], f0` |
| `fma.vec4.f32` | Vector FMA | `fma.vec4.f32 v0, v1, v2, v0` |
| `add.f32` | FP32 add | `add.f32 f0, f0, f5` |
| `max.f32` | FP32 max | `max.f32 f0, f0, f6` |
| `branch` | Branch | `branch .L_dot_loop` |

#### Valhall
| Instruction | Description | Example |
|------------|-------------|---------|
| `VLD.vec4.f32` | Vector load | `VLD.vec4.f32 v4, [r14]` |
| `ST.f32` | Store float | `ST.f32 [r33], f0` |
| `VFMA.vec4.f32` | Vector FMA | `VFMA.vec4.f32 v0, v4, v8, v0` |
| `FADD.f32` | FP32 add | `FADD.f32 f0, f0, f5` |
| `FMAX.f32` | FP32 max | `FMAX.f32 f0, f0, f6` |
| `BR` | Branch | `BR .L_main_loop` |

## Real-World Usage

### When to Use GPU Assembly

**NVIDIA PTX:**
- ✓ Need portable GPU code across architectures
- ✓ Implementing custom operations not in CUDA
- ✓ Fine-tuning performance-critical kernels
- ✓ Learning GPU architecture
- ✗ General application development (use CUDA C++)

**NVIDIA SASS:**
- ✓ Absolute maximum performance needed
- ✓ Exploiting specific hardware features
- ✓ Reverse-engineering for optimization
- ✗ Production code (use CUDA/PTX)
- ✗ Portable code

**ARM Mali Assembly:**
- ✓ Educational purposes (understanding GPU)
- ✓ Analyzing compiler output
- ✗ Writing shaders (use OpenCL/GLSL/Vulkan)
- ✗ Direct assembly programming (not officially supported)

### Production Best Practices

For real applications, use high-level APIs:

**NVIDIA:**
- CUDA C++ for general GPU programming
- cuBLAS/cuDNN for neural networks
- TensorRT for inference optimization

**ARM Mali:**
- OpenCL for compute kernels
- Vulkan compute shaders
- ARM NN / ARM Compute Library for ML

## Learning Path

1. **Start with high-level code**: CUDA C++ or OpenCL
2. **Compile and inspect**: Use `cuobjdump -ptx` or `malioc`
3. **Read generated assembly**: Understand compiler decisions
4. **Identify bottlenecks**: Profile with NSight or ARM Streamline
5. **Hand-optimize critical parts**: Write PTX for hot paths
6. **Benchmark and verify**: Ensure assembly is actually faster

## References

### NVIDIA
- PTX ISA Documentation: https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Ampere Architecture Whitepaper: https://www.nvidia.com/en-us/data-center/ampere-architecture/

### ARM Mali
- Mali GPU Architectures: https://developer.arm.com/ip-products/graphics-and-multimedia/mali-gpus
- OpenCL Optimization Guide: https://developer.arm.com/solutions/graphics-and-gaming/developer-guides
- Bifrost Architecture: https://community.arm.com/arm-community-blogs/b/graphics-gaming-and-vr-blog

## License

This code is provided for educational purposes.

