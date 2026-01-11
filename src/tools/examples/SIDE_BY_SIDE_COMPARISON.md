# Side-by-Side ISA Comparison

This document shows the same MLP operations implemented side-by-side in different assembly languages for direct comparison.

## Operation 1: Initialize Accumulator to Zero

### x86-64 AVX (256-bit)
```asm
vxorps ymm0, ymm0, ymm0         # Clear 8 floats to 0.0
```

### x86-64 SSE (128-bit)
```asm
xorps xmm0, xmm0                # Clear 4 floats to 0.0
```

### ARM AArch64 NEON
```asm
movi v0.4s, #0                  # Set 4 floats to 0
```

### ARM ARMv7 NEON
```asm
vmov.f32 q0, #0.0               # Set 4 floats to 0.0
```

**Analysis:**
- x86-64 uses XOR trick (any value XOR itself = 0)
- ARM has dedicated move-immediate instruction
- All execute in 1 cycle

---

## Operation 2: Load Data from Memory

### x86-64 AVX (256-bit)
```asm
vmovups ymm1, [r12 + r10*4]     # Load 8 floats (unaligned)
# Address = r12 + (r10 × 4)
```

### x86-64 SSE (128-bit)
```asm
movups xmm1, [r12 + r10*4]      # Load 4 floats (unaligned)
# Address = r12 + (r10 × 4)
```

### ARM AArch64 NEON
```asm
ldr q1, [x19, x7, lsl #2]       # Load 4 floats
# Address = x19 + (x7 << 2)
```

### ARM ARMv7 NEON
```asm
add r2, r4, r0, lsl #2          # Calculate address: r2 = r4 + (r0 << 2)
vld1.32 {q1}, [r2]              # Load 4 floats from [r2]
```

**Analysis:**
- x86-64 has complex addressing mode (base + index×scale)
- ARM AArch64 has scaled register offset addressing
- ARM ARMv7 requires separate address calculation
- x86-64 AVX loads twice as much data (8 vs 4 floats)

---

## Operation 3: Multiply-Accumulate (MAC)

### x86-64 AVX with FMA
```asm
vfmadd231ps ymm0, ymm1, ymm2    # ymm0 = ymm0 + (ymm1 × ymm2)
                                 # 8 MAC operations in parallel
```

### x86-64 SSE (no FMA)
```asm
mulps xmm1, xmm2                # xmm1 = xmm1 × xmm2 (4 muls)
addps xmm0, xmm1                # xmm0 = xmm0 + xmm1 (4 adds)
                                 # 4 MAC operations via 2 instructions
```

### ARM AArch64 NEON
```asm
fmla v0.4s, v1.4s, v2.4s        # v0 = v0 + (v1 × v2)
                                 # 4 MAC operations in parallel
```

### ARM ARMv7 NEON
```asm
vmla.f32 q0, q1, q2             # q0 = q0 + (q1 × q2)
                                 # 4 MAC operations in parallel
```

**Analysis:**
- AVX FMA, ARM NEON both have fused MAC operations
- SSE without FMA needs 2 instructions
- AVX processes 2× data per instruction vs. NEON
- Fused operations: higher accuracy, better throughput

**Instruction Count (for 16 MACs):**
- AVX FMA: 2 iterations (16 ops / 8 per iter)
- SSE: 4 iterations × 2 instructions = 8 instructions
- ARM NEON: 4 iterations (16 ops / 4 per iter)

---

## Operation 4: Horizontal Sum (Reduce Vector to Scalar)

### x86-64 AVX
```asm
vextractf128 xmm1, ymm0, 1      # xmm1 = upper 128 bits of ymm0
vaddps xmm0, xmm0, xmm1         # Add upper and lower halves
haddps xmm0, xmm0, xmm0         # Horizontal add
haddps xmm0, xmm0, xmm0         # Horizontal add again
# Result in xmm0[31:0]
```

### x86-64 SSE
```asm
haddps xmm0, xmm0, xmm0         # Horizontal add: [a+b, c+d, a+b, c+d]
haddps xmm0, xmm0, xmm0         # Horizontal add: [a+b+c+d, ...]
# Result in xmm0[31:0]
```

### ARM AArch64 NEON
```asm
faddp v0.4s, v0.4s, v0.4s       # Pairwise add: [a+b, c+d, a+b, c+d]
faddp v0.4s, v0.4s, v0.4s       # Pairwise add: [(a+b)+(c+d), ...]
# Result in v0.s[0]
```

### ARM ARMv7 NEON
```asm
vadd.f32 d0, d0, d1             # Add lower and upper D regs: d0 = d0 + d1
                                 # [a+c, b+d]
vpadd.f32 d0, d0, d0            # Pairwise add: [(a+c)+(b+d), ...]
# Result in d0[31:0]
```

**Analysis:**
- AVX requires extra step to combine 256-bit halves
- Both architectures need 2-3 steps for full reduction
- x86 has dedicated `haddps`, ARM uses pairwise add
- ARMv7 uses narrower D registers for pairwise operations

---

## Operation 5: ReLU Activation (max(0, x))

### x86-64 AVX (vectorized)
```asm
vxorps ymm1, ymm1, ymm1         # ymm1 = 0.0
vmaxps ymm0, ymm0, ymm1         # ymm0 = max(ymm0, 0.0) for 8 floats
```

### x86-64 SSE (scalar)
```asm
xorps xmm1, xmm1                # xmm1 = 0.0
maxss xmm0, xmm1                # xmm0 = max(xmm0, 0.0) for 1 float
```

### ARM AArch64 NEON (scalar)
```asm
movi v1.4s, #0                  # v1 = 0.0
fmax s0, s0, s1                 # s0 = max(s0, 0.0) for 1 float
```

### ARM ARMv7 NEON (scalar)
```asm
vmov.f32 s1, #0.0               # s1 = 0.0
vmax.f32 s0, s0, s1             # s0 = max(s0, 0.0) for 1 float
```

**Analysis:**
- All architectures support max operation
- Simple 2-instruction sequence: set zero, compare-max
- Can be vectorized or used on scalars

---

## Operation 6: Store Result to Memory

### x86-64 AVX (256-bit)
```asm
vmovss [r15 + rax*4], xmm0      # Store 1 float (scalar)
# Address = r15 + (rax × 4)
```

### x86-64 SSE (128-bit)
```asm
movss [r15 + rax*4], xmm0       # Store 1 float (scalar)
# Address = r15 + (rax × 4)
```

### ARM AArch64 NEON
```asm
str s0, [x22, x6, lsl #2]       # Store 1 float
# Address = x22 + (x6 << 2)
```

### ARM ARMv7 NEON
```asm
add r2, r7, r10, lsl #2         # Calculate address
vstr s0, [r2]                   # Store 1 float to [r2]
```

**Analysis:**
- Similar to loads, x86-64 supports complex addressing
- ARM AArch64 supports scaled offset
- ARM ARMv7 needs separate address calculation

---

## Complete Dot Product Example

Computing `result = sum(input[i] * weights[i])` for 16 elements:

### x86-64 AVX (8 floats per iteration)
```asm
# Initialization
vxorps ymm0, ymm0, ymm0         # accumulator = 0
xor r10, r10                    # i = 0

# Main loop (2 iterations for 16 elements)
loop_start:
    vmovups ymm1, [r12 + r10*4] # Load 8 inputs
    vmovups ymm2, [r11 + r10*4] # Load 8 weights
    vfmadd231ps ymm0, ymm1, ymm2 # accumulate 8 MACs
    add r10, 8                  # i += 8
    cmp r10, 16
    jl loop_start

# Horizontal reduction
vextractf128 xmm1, ymm0, 1      # Extract upper half
vaddps xmm0, xmm0, xmm1         # Add halves
haddps xmm0, xmm0, xmm0         # Horizontal add
haddps xmm0, xmm0, xmm0         # Horizontal add again
# Result in xmm0[31:0]

# Total: ~12 instructions, 2 main loop iterations
```

### x86-64 SSE (4 floats per iteration)
```asm
# Initialization
xorps xmm0, xmm0                # accumulator = 0
xor r10, r10                    # i = 0

# Main loop (4 iterations for 16 elements)
loop_start:
    movups xmm1, [r12 + r10*4]  # Load 4 inputs
    movups xmm2, [r11 + r10*4]  # Load 4 weights
    mulps xmm1, xmm2            # Multiply 4 pairs
    addps xmm0, xmm1            # Accumulate 4 results
    add r10, 4                  # i += 4
    cmp r10, 16
    jl loop_start

# Horizontal reduction
haddps xmm0, xmm0, xmm0         # Horizontal add
haddps xmm0, xmm0, xmm0         # Horizontal add again
# Result in xmm0[31:0]

# Total: ~20 instructions, 4 main loop iterations
```

### ARM AArch64 NEON (4 floats per iteration)
```asm
// Initialization
movi v0.4s, #0                  // accumulator = 0
mov x7, #0                      // i = 0

// Main loop (4 iterations for 16 elements)
loop_start:
    ldr q1, [x19, x7, lsl #2]   // Load 4 inputs
    ldr q2, [x8, x7, lsl #2]    // Load 4 weights
    fmla v0.4s, v1.4s, v2.4s    // Accumulate 4 MACs
    add x7, x7, #4              // i += 4
    cmp x7, #16
    b.lt loop_start

// Horizontal reduction
faddp v0.4s, v0.4s, v0.4s       // Pairwise add
faddp v0.4s, v0.4s, v0.4s       // Pairwise add again
// Result in v0.s[0]

# Total: ~16 instructions, 4 main loop iterations
```

### ARM ARMv7 NEON (4 floats per iteration)
```asm
@ Initialization
vmov.f32 q0, #0.0               @ accumulator = 0
mov r0, #0                      @ i = 0

@ Main loop (4 iterations for 16 elements)
loop_start:
    add r2, r4, r0, lsl #2      @ Calculate input address
    vld1.32 {q1}, [r2]          @ Load 4 inputs
    add r2, r11, r0, lsl #2     @ Calculate weights address
    vld1.32 {q2}, [r2]          @ Load 4 weights
    vmla.f32 q0, q1, q2         @ Accumulate 4 MACs
    add r0, r0, #4              @ i += 4
    cmp r0, #16
    blt loop_start

@ Horizontal reduction
vadd.f32 d0, d0, d1             @ Add lower and upper halves
vpadd.f32 d0, d0, d0            @ Pairwise add
@ Result in d0[31:0]

@ Total: ~24 instructions, 4 main loop iterations
```

**Performance Comparison (16-element dot product):**

| Architecture | Instructions | Loop Iterations | MACs per Iter | Total Cycles (est.) |
|--------------|-------------|-----------------|---------------|---------------------|
| x86-64 AVX   | ~12         | 2               | 8             | ~15-20              |
| x86-64 SSE   | ~20         | 4               | 4             | ~25-35              |
| ARM AArch64  | ~16         | 4               | 4             | ~30-40              |
| ARM ARMv7    | ~24         | 4               | 4             | ~50-70              |

**Key Insights:**
1. AVX's 256-bit width gives it the fewest iterations
2. Fused MAC operations (AVX FMA, NEON FMLA) reduce instruction count
3. x86-64's complex addressing saves instructions vs. ARMv7
4. Horizontal reduction is expensive on all architectures

---

## Code Density Comparison

For a complete MLP layer function (excluding setup/teardown):

### Code Size (approximate)
- **x86-64 AVX**: ~200 bytes (variable-length instructions)
- **x86-64 SSE**: ~180 bytes (variable-length instructions)
- **ARM AArch64**: ~160 bytes (40 × 4-byte instructions)
- **ARM ARMv7**: ~220 bytes (55 × 4-byte instructions)

### Instruction Count (main computation)
- **x86-64 AVX**: ~50 instructions (including control flow)
- **x86-64 SSE**: ~60 instructions
- **ARM AArch64**: ~45 instructions
- **ARM ARMv7**: ~65 instructions

**Analysis:**
- ARM has fixed 32-bit instructions, leading to predictable code size
- x86-64 variable-length encoding can be more compact
- ARMv7 needs more instructions due to simpler addressing modes
- Code density doesn't directly correlate with performance

---

## Architectural Philosophy Differences

### x86-64 (CISC - Complex Instruction Set)
**Characteristics:**
- Complex, powerful instructions that do more per instruction
- Variable-length encoding (1-15 bytes)
- Complex addressing modes
- Many instruction variants
- Backward compatibility priority

**Example:** One instruction can load, multiply, add, and store:
```asm
vfmadd213ps ymm0, ymm1, [rax + rbx*4 + 16]
```

**Trade-offs:**
- ✓ Fewer instructions for same work
- ✓ Powerful addressing modes
- ✗ Complex decode logic
- ✗ Variable instruction length complicates fetch
- ✗ Larger, more power-hungry chips

### ARM (RISC - Reduced Instruction Set)
**Characteristics:**
- Simple, regular instructions
- Fixed 32-bit encoding
- Load/store architecture
- Regular instruction format
- Efficiency priority

**Example:** Same operation requires multiple instructions:
```asm
add x2, x0, x1, lsl #2          # Calculate address
ldr q0, [x2, #16]               # Load data
fmla v0.4s, v0.4s, v1.4s        # Multiply-add
```

**Trade-offs:**
- ✓ Simpler decode and dispatch
- ✓ Easier to pipeline deeply
- ✓ More energy efficient
- ✗ More instructions for complex operations
- ✗ Limited addressing modes

---

## Summary: When to Use Each

### Choose x86-64 AVX when:
- Maximum throughput needed for compute-bound workloads
- Working with large vectors (8+ floats)
- Power consumption is not primary concern
- Target is server/desktop/HPC

### Choose x86-64 SSE when:
- Broad compatibility needed (older CPUs)
- Lower power consumption desired
- Working with medium vectors (4 floats)

### Choose ARM NEON when:
- Energy efficiency is critical (mobile, embedded)
- Thermal constraints exist
- Target is ARM-based devices (phones, tablets, edge devices)
- Good balance of performance and power needed

### Performance-per-Watt Ranking (typical):
1. ARM NEON (AArch64) - Best efficiency
2. ARM NEON (ARMv7)
3. x86-64 SSE
4. x86-64 AVX - Best raw performance

---

This comparison should give you a comprehensive understanding of how different ISAs approach the same computational problems in MLP implementations.
