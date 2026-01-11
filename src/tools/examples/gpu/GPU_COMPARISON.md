# GPU Architecture Comparison: NVIDIA Ampere vs. ARM Mali

This document provides a detailed side-by-side comparison of NVIDIA Ampere and ARM Mali GPU architectures for MLP execution.

## Architecture Philosophy

### NVIDIA Ampere: Throughput-Oriented

**Design Goal**: Maximum computational throughput for data center and HPC workloads

**Key Characteristics:**
- Massive parallelism (thousands of threads per SM)
- Large power budget (250-400W)
- Explicit memory hierarchy management
- Optimized for batch processing
- Warps execute in lockstep (SIMT model)

### ARM Mali: Energy-Efficient

**Design Goal**: Balance of performance and power for mobile/embedded devices

**Key Characteristics:**
- Moderate parallelism (hundreds of threads per core)
- Small power budget (5-15W)
- Implicit memory management (cache-based)
- Optimized for interactive workloads
- More flexible execution model

---

## Hardware Specifications Comparison

### NVIDIA A100 (Ampere) vs. ARM Mali-G78 MP24

| Feature | NVIDIA A100 | ARM Mali-G78 MP24 |
|---------|-------------|-------------------|
| **Architecture** | Ampere (SM 8.0) | Valhall |
| **Manufacturing** | TSMC 7nm | Samsung 5nm / TSMC 5nm |
| **Die Size** | 826 mm² | ~100 mm² (est.) |
| **Transistors** | 54.2 billion | ~10 billion (est.) |
| **Compute Units** | 108 SMs | 24 cores |
| **Cores/Threads** | 6912 CUDA cores | 384 execution engines |
| **Peak FP32** | 19.5 TFLOPS | ~1.5 TFLOPS |
| **Peak FP16** | 312 TFLOPS (Tensor) | ~3 TFLOPS |
| **Memory** | 40-80 GB HBM2e | Unified system (8-16 GB) |
| **Memory BW** | 1555 GB/s | 50-80 GB/s |
| **L2 Cache** | 40 MB | 2-4 MB |
| **TDP** | 250-400W | 5-10W |
| **Price** | $10,000+ | Integrated (SoC) |

---

## Execution Model Comparison

### Thread Organization

**NVIDIA Ampere:**
```
Grid (device-wide)
  └─ Thread Blocks (up to 1024 threads)
      └─ Warps (32 threads, SIMT execution)
          └─ Threads (SIMD lanes)

Example: 1024 threads per block = 32 warps
```

**ARM Mali Valhall:**
```
Global Work Size (device-wide)
  └─ Work Groups (up to 1024 work items)
      └─ Warps (16 work items, native)
          └─ Work Items (SIMD lanes)

Example: 1024 work items per group = 64 warps
```

**Key Difference**: NVIDIA has 32-wide warps, Mali has 16-wide warps.

### Divergence Handling

**NVIDIA Ampere:**
```cuda
// Bad: Causes warp divergence
if (threadIdx.x < 16) {
    result = A;  // Half of warp executes this
} else {
    result = B;  // Other half waits, then executes this
}
// Effective throughput: 50% (serialized execution)
```

**ARM Mali:**
```opencl
// Same code, but:
// - Smaller warps (16 vs 32) = less divergence impact
// - More flexible scheduling = better hiding of divergence
if (get_local_id(0) < 16) {
    result = A;
} else {
    result = B;
}
// Effective throughput: ~60-70% (better scheduling)
```

---

## Memory Hierarchy Comparison

### NVIDIA Ampere Memory

```
Thread Registers (255 x 32-bit per thread)
       ↓ ~1 cycle latency
Shared Memory (164 KB per SM, software-managed)
       ↓ ~20 cycles
L1 Cache (128 KB per SM)
       ↓ ~30 cycles
L2 Cache (40 MB, shared across SMs)
       ↓ ~200 cycles
HBM2e Global Memory (40-80 GB)
       ↓ ~400-800 cycles
```

**Characteristics:**
- Explicit shared memory (programmer-controlled)
- Large register file enables high occupancy
- Separate L1 data cache and shared memory
- Very high bandwidth (1.5 TB/s)

### ARM Mali Valhall Memory

```
Thread Registers (64 x 32-bit per work item)
       ↓ ~1 cycle latency
L1 Cache (32-64 KB per core, hardware-managed)
       ↓ ~10-20 cycles
L2 Cache (2-4 MB, shared across cores)
       ↓ ~40-80 cycles
System Memory (8-16 GB, unified with CPU)
       ↓ ~200-400 cycles
```

**Characteristics:**
- No explicit shared memory (use L2 cache)
- Smaller register file (more energy efficient)
- Unified memory with CPU (easier programming)
- Lower absolute bandwidth (50-80 GB/s)

### Memory Access Patterns

**NVIDIA - Coalescing:**
```ptx
// GOOD: Coalesced access (32 consecutive floats by 32-thread warp)
ld.global.ca.f32 %f0, [base + tid*4];  // 128-byte cache line, 1 transaction

// BAD: Strided access (32 separate transactions)
ld.global.ca.f32 %f0, [base + tid*128];  // 32 transactions!
```

**Mali - Cache-Friendly:**
```asm
// GOOD: Sequential access leverages L2 cache
VLD.f32 f0, [base + id*4]  // Cached automatically

// ACCEPTABLE: Strided access still cached (smaller warps help)
VLD.f32 f0, [base + id*64]  // L2 handles this better
```

---

## Instruction Set Comparison

### Same Operation: Vector FMA (4 floats)

**NVIDIA PTX:**
```ptx
// Load 4 floats (vectorized)
ld.global.ca.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0];

// 4 separate FMA instructions
fma.rn.f32 %f10, %f0, %f4, %f10;
fma.rn.f32 %f10, %f1, %f5, %f10;
fma.rn.f32 %f10, %f2, %f6, %f10;
fma.rn.f32 %f10, %f3, %f7, %f10;

// PTX represents operations per thread
// Warp executes 32 copies in parallel → 128 FMAs total
```

**ARM Mali Valhall:**
```asm
// Load 4 floats as vec4 (single instruction)
VLD.vec4.f32 v0, [r0]

// Single vector FMA (operates on all 4 components)
VFMA.vec4.f32 v1, v0, v2, v1

// Warp executes 16 copies in parallel → 64 FMAs total
// But: Can dual-issue FMAs for 2x throughput
```

**Analysis:**
- Mali has native vector types (vec2/3/4)
- NVIDIA PTX works with scalars, hardware vectorizes
- Mali: 1 instruction = 4 FMAs
- NVIDIA: 4 instructions = 4 FMAs (but 32x wider execution)

### MLP Dot Product Loop

**NVIDIA PTX (32-wide warp):**
```ptx
DOT_LOOP:
    setp.ge.u32 %p1, %r10, %r1;        // Compare i >= input_size
    @%p1 bra    DOT_DONE;              // Predicated branch

    ld.global.ca.v4.f32 {%f20, %f21, %f22, %f23}, [%rd8];  // 4 weights
    ld.shared.v4.f32 {%f24, %f25, %f26, %f27}, [%r15];     // 4 inputs

    fma.rn.f32 %f10, %f20, %f24, %f10; // MAC 0
    fma.rn.f32 %f10, %f21, %f25, %f10; // MAC 1
    fma.rn.f32 %f10, %f22, %f26, %f10; // MAC 2
    fma.rn.f32 %f10, %f23, %f27, %f10; // MAC 3

    add.u32 %r10, %r10, 4;             // i += 4
    bra     DOT_LOOP;

// Throughput: 32 warps × 4 FMAs = 128 FMAs per iteration
```

**ARM Mali Valhall (16-wide warp):**
```asm
DOT_LOOP:
    ICMP.GE.u32 p0, r10, r1            // Compare i >= input_size
    BR.p0       DOT_DONE               // Conditional branch

    VLD.vec4.f32 v4, [r14]             // 4 weights (vectorized)
    VLD.vec4.f32 v8, [r16]             // 4 inputs (vectorized)

    VFMA.vec4.f32 v0, v4, v8, v0       // 4 MACs in one instruction

    IADD.u32    r10, r10, 4            // i += 4
    BR          DOT_LOOP

// Throughput: 16 warps × 4 FMAs = 64 FMAs per iteration
// Can dual-issue: potentially 128 FMAs per cycle
```

**Comparison:**
- NVIDIA: More instructions, but wider (32 threads)
- Mali: Fewer instructions, narrower (16 threads), but native vec4
- NVIDIA: Explicit shared memory loads
- Mali: Implicit caching (cleaner code)

---

## Synchronization

### NVIDIA Ampere

**Warp-Level (implicit):**
```ptx
// Threads in warp execute in lockstep (SIMT)
// No explicit synchronization needed within warp

// Warp shuffle (no synchronization overhead)
shfl.down.b32 %f1, %f0, 16, 31;  // Get value from lane+16
```

**Block-Level (explicit):**
```ptx
// Barrier: all threads in block must reach this point
bar.sync 0;  // __syncthreads() in CUDA
// Cost: ~20-50 cycles depending on occupancy
```

**Grid-Level:**
```cuda
// Must end kernel and launch new one
kernel_part1<<<grid, block>>>();
cudaDeviceSynchronize();  // Host-side synchronization
kernel_part2<<<grid, block>>>();
```

### ARM Mali

**Work-Group Level:**
```asm
// Barrier for work group
BARRIER  // barrier(CLK_LOCAL_MEM_FENCE)
// Cost: ~10-30 cycles (smaller work groups = faster)
```

**Global Level:**
```opencl
// Must split into multiple kernels
clEnqueueNDRangeKernel(queue, kernel1, ...);
clFinish(queue);  // Host-side synchronization
clEnqueueNDRangeKernel(queue, kernel2, ...);
```

**Key Difference**: Mali has no warp-level primitives (no shuffle), but barriers are faster due to smaller work groups.

---

## Performance Analysis: MLP Forward Pass

### Problem Size: 512 inputs → 256 outputs

**NVIDIA A100 Analysis:**

1. **Memory Transfers:**
   - Input: 512 × 4 bytes = 2 KB
   - Weights: 256 × 512 × 4 bytes = 512 KB
   - Bias: 256 × 4 bytes = 1 KB
   - Output: 256 × 4 bytes = 1 KB
   - **Total: ~516 KB**

2. **Compute:**
   - 256 neurons × 512 MACs = 131,072 FP32 operations
   - A100 peak: 19.5 TFLOPS = 19.5 × 10¹² FLOP/s
   - Theoretical time: 131,072 / 19.5×10¹² = 6.7 nanoseconds

3. **Memory Bound:**
   - Memory BW: 1555 GB/s
   - Transfer time: 516 KB / 1555 GB/s = 331 nanoseconds
   - **Bottleneck: Memory (49x slower than compute)**

4. **Actual Performance:**
   - With caching and pipelining: ~5-10 microseconds
   - Occupancy optimization critical
   - Multiple blocks hide latency

**ARM Mali-G78 Analysis:**

1. **Memory Transfers:**
   - Same 516 KB total
   - Unified memory helps (no PCIe overhead)

2. **Compute:**
   - 131,072 FP32 operations
   - G78 peak: ~1.5 TFLOPS = 1.5 × 10¹² FLOP/s
   - Theoretical time: 131,072 / 1.5×10¹² = 87 nanoseconds

3. **Memory Bound:**
   - Memory BW: ~60 GB/s
   - Transfer time: 516 KB / 60 GB/s = 8.6 microseconds
   - **Bottleneck: Memory (99x slower than compute)**

4. **Actual Performance:**
   - ~50-100 microseconds typical
   - L2 cache helps for repeated inference
   - Lower parallelism = harder to hide latency

### Performance Ratio

| Metric | A100 | Mali-G78 | Ratio |
|--------|------|----------|-------|
| Peak Compute | 19.5 TFLOPS | 1.5 TFLOPS | 13x |
| Memory BW | 1555 GB/s | 60 GB/s | 26x |
| MLP Latency | ~10 µs | ~80 µs | 8x |
| Power | 300W | 8W | 38x |
| **Perf/Watt** | 0.033 GFLOPS/W | 0.19 GFLOPS/W | **6x better** |

**Conclusion**: NVIDIA is faster in absolute terms, but Mali is far more power-efficient for small models.

---

## Code Density Comparison

### MLP Kernel (main computation only)

**NVIDIA PTX:**
- ~150 lines of PTX assembly
- ~80 actual instructions (excluding labels/directives)
- Variable instruction length in SASS (avg ~8 bytes)
- **Estimated code size: ~640 bytes**

**ARM Mali Valhall:**
- ~100 lines of assembly
- ~50 actual instructions
- Fixed 64-bit instruction encoding
- **Estimated code size: ~400 bytes**

**Analysis**: Mali has more compact code due to:
- Native vector operations
- Simpler memory model (no shared memory management)
- Fixed-length instructions
- Implicit caching

---

## Optimization Strategies Comparison

### Maximizing Throughput

**NVIDIA Ampere:**

1. **Occupancy**: Launch many threads (high occupancy)
   ```cuda
   // Bad: 64 threads per block = 2 warps = low occupancy
   kernel<<<1024, 64>>>();

   // Good: 256 threads per block = 8 warps = better occupancy
   kernel<<<256, 256>>>();
   ```

2. **Shared Memory**: Reduce global memory access
   ```ptx
   // Cooperative load to shared memory
   ld.global.ca.f32 %f1, [global_addr];
   st.shared.f32 [shared_addr], %f1;
   bar.sync 0;
   ld.shared.f32 %f2, [shared_addr];  // Fast!
   ```

3. **Coalescing**: Ensure contiguous memory access
   ```cuda
   // Access pattern: thread i accesses data[i]
   // All 32 threads in warp access consecutive addresses
   ```

**ARM Mali:**

1. **Vectorization**: Use vec4 operations
   ```opencl
   // Bad: Scalar operations
   for (int i = 0; i < 4; i++) sum += a[i] * b[i];

   // Good: Vector operation
   float4 a4 = vload4(0, a);
   float4 b4 = vload4(0, b);
   sum = dot(a4, b4);
   ```

2. **ILP**: Multiple independent operations
   ```opencl
   // Issue multiple loads/FMAs
   float4 sum0 = 0, sum1 = 0;
   sum0 = fma(w0, in0, sum0);  // Can execute in parallel
   sum1 = fma(w1, in1, sum1);  // on separate FMA units
   ```

3. **Reduce Divergence**: Minimize branches
   ```opencl
   // Avoid: if statements in hot loops
   // Prefer: select() or ternary operators
   result = select(a, b, condition);
   ```

---

## When to Use Each Architecture

### Choose NVIDIA Ampere When:

- ✅ Maximum absolute performance needed
- ✅ Batch processing (many inputs at once)
- ✅ Power budget is not a constraint
- ✅ Need large memory capacity (40-80 GB)
- ✅ Workload is compute-bound
- ✅ Training large neural networks
- ✅ HPC / data center deployment

**Example Use Cases:**
- Training large language models (GPT, BERT)
- Batch inference for web services
- Scientific simulations
- Video rendering farms

### Choose ARM Mali When:

- ✅ Power efficiency is critical
- ✅ Mobile or embedded deployment
- ✅ Interactive latency more important than throughput
- ✅ Unified memory simplifies programming
- ✅ Cost-sensitive applications
- ✅ On-device inference
- ✅ Thermal constraints

**Example Use Cases:**
- Mobile phone AI (camera, voice assistant)
- Edge devices and IoT
- Augmented reality
- Gaming on mobile devices
- Automotive infotainment

---

## Future Trends

### NVIDIA Direction:
- Larger Tensor Cores for AI
- Higher memory bandwidth (HBM3)
- Multi-instance GPU (MIG) for cloud
- Sparsity acceleration
- FP8 / INT8 for inference

### ARM Mali Direction:
- Better energy efficiency (new process nodes)
- Enhanced ML acceleration
- Raytracing in mobile
- Variable Rate Shading
- Tighter CPU integration

---

## Summary Table

| Aspect | NVIDIA Ampere | ARM Mali Valhall | Winner |
|--------|--------------|------------------|--------|
| Raw Performance | 19.5 TFLOPS | 1.5 TFLOPS | NVIDIA (13x) |
| Perf/Watt | 65 GFLOPS/W | 188 GFLOPS/W | Mali (3x) |
| Memory BW | 1555 GB/s | 60 GB/s | NVIDIA (26x) |
| Latency (single) | ~10 µs | ~80 µs | NVIDIA (8x) |
| Throughput (batch) | Very High | Medium | NVIDIA |
| Ease of Use | Medium | Easy | Mali |
| Code Density | Lower | Higher | Mali |
| Power | 300W | 8W | Mali (38x) |
| Cost | $10,000+ | $0 (integrated) | Mali |
| **Best For** | **Data Center** | **Mobile/Edge** | Tie |

Both architectures excel in their target domains!

