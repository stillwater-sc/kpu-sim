# SIMD Instruction Reference for MLP Implementation

This document provides detailed information about the SIMD instructions used in the MLP implementations.

## x86-64 AVX Instructions

### Data Movement

#### VMOVUPS - Move Unaligned Packed Single-Precision
```asm
vmovups ymm1, [mem]     # Load 8 floats from memory to ymm1
vmovups [mem], ymm1     # Store 8 floats from ymm1 to memory
```
- **Encoding**: VEX.256.0F.WIG 10 /r (load), 11 /r (store)
- **Latency**: 1 cycle (register), 7 cycles (memory)
- **Throughput**: 1/0.5 (can execute 2 per cycle on modern CPUs)
- **Alignment**: Works with unaligned memory (16-byte aligned is faster)

#### VMOVSS - Move Scalar Single-Precision
```asm
vmovss xmm1, [mem]      # Load 1 float
vmovss [mem], xmm1      # Store 1 float
```
- **Encoding**: VEX.LIG.F3.0F.WIG 10 /r (load), 11 /r (store)
- **Latency**: 3 cycles (load), 1 cycle (store)
- **Throughput**: 1/0.5

### Arithmetic Operations

#### VFMADD231PS - Fused Multiply-Add (231 form)
```asm
vfmadd231ps ymm0, ymm1, ymm2    # ymm0 = ymm0 + (ymm1 * ymm2)
```
- **Encoding**: VEX.256.66.0F38.W0 B8 /r
- **Latency**: 4 cycles
- **Throughput**: 1/0.5
- **Operation**: Performs multiply and add in single operation
- **Precision**: More accurate than separate multiply + add (no intermediate rounding)
- **Variants**:
  - `vfmadd132ps`: ymm0 = (ymm0 * ymm1) + ymm2
  - `vfmadd213ps`: ymm0 = (ymm1 * ymm0) + ymm2
  - `vfmadd231ps`: ymm0 = ymm0 + (ymm1 * ymm2)  ← Used in MLP

#### VADDPS - Add Packed Single-Precision
```asm
vaddps ymm0, ymm1, ymm2         # ymm0 = ymm1 + ymm2 (8 adds)
```
- **Encoding**: VEX.256.0F.WIG 58 /r
- **Latency**: 3 cycles
- **Throughput**: 1/0.5

#### VMAXPS/VMAXSS - Maximum
```asm
vmaxps ymm0, ymm1, ymm2         # ymm0[i] = max(ymm1[i], ymm2[i])
vmaxss xmm0, xmm1, xmm2         # xmm0 = max(xmm1, xmm2) [scalar]
```
- **Encoding**: VEX.256.0F.WIG 5F /r
- **Latency**: 3 cycles
- **Throughput**: 1/0.5
- **Use**: Implements ReLU: max(x, 0)

### Horizontal Operations

#### VHADDPS - Horizontal Add
```asm
vhaddps xmm0, xmm1, xmm2
# xmm0[31:0]   = xmm1[63:32]   + xmm1[31:0]
# xmm0[63:32]  = xmm1[127:96]  + xmm1[95:64]
# xmm0[95:64]  = xmm2[63:32]   + xmm2[31:0]
# xmm0[127:96] = xmm2[127:96]  + xmm2[95:64]
```
- **Encoding**: VEX.128.F2.0F.WIG 7C /r
- **Latency**: 5 cycles
- **Throughput**: 1/2
- **Use**: Reduce vector to scalar by summing elements
- **Note**: Called twice to fully reduce 4 elements to 1

#### VEXTRACTF128 - Extract 128 bits
```asm
vextractf128 xmm1, ymm0, imm8   # Extract upper or lower 128 bits
```
- **Encoding**: VEX.256.66.0F3A.W0 19 /r ib
- **Latency**: 3 cycles
- **Throughput**: 1
- **Use**: Extract upper half of 256-bit register for reduction

### Logical Operations

#### VXORPS - Logical XOR
```asm
vxorps ymm0, ymm1, ymm2         # ymm0 = ymm1 XOR ymm2
vxorps ymm0, ymm0, ymm0         # Clear ymm0 to zero (common idiom)
```
- **Encoding**: VEX.256.0F.WIG 57 /r
- **Latency**: 1 cycle
- **Throughput**: 1/0.33
- **Use**: Efficiently zero registers

---

## x86-64 SSE Instructions

### Data Movement

#### MOVUPS - Move Unaligned Packed Single-Precision
```asm
movups xmm1, [mem]              # Load 4 floats
movups [mem], xmm1              # Store 4 floats
```
- **Encoding**: 0F 10 /r (load), 0F 11 /r (store)
- **Latency**: 1 cycle (register), 5-7 cycles (memory)
- **Throughput**: 1/0.5

#### MOVSS - Move Scalar Single-Precision
```asm
movss xmm1, [mem]               # Load 1 float
movss [mem], xmm1               # Store 1 float
```
- **Encoding**: F3 0F 10 /r (load), F3 0F 11 /r (store)
- **Latency**: 3 cycles (load)
- **Throughput**: 1/0.5

### Arithmetic Operations

#### MULPS - Multiply Packed Single-Precision
```asm
mulps xmm1, xmm2                # xmm1 = xmm1 * xmm2 (4 multiplies)
```
- **Encoding**: 0F 59 /r
- **Latency**: 4 cycles
- **Throughput**: 1/0.5

#### ADDPS - Add Packed Single-Precision
```asm
addps xmm1, xmm2                # xmm1 = xmm1 + xmm2 (4 adds)
```
- **Encoding**: 0F 58 /r
- **Latency**: 3 cycles
- **Throughput**: 1/0.5

#### MAXPS/MAXSS - Maximum
```asm
maxps xmm1, xmm2                # xmm1[i] = max(xmm1[i], xmm2[i])
maxss xmm1, xmm2                # xmm1 = max(xmm1, xmm2) [scalar]
```
- **Encoding**: 0F 5F /r
- **Latency**: 3 cycles
- **Throughput**: 1/0.5

### Horizontal Operations

#### HADDPS - Horizontal Add
```asm
haddps xmm1, xmm2
# xmm1[31:0]   = xmm1[63:32]  + xmm1[31:0]
# xmm1[63:32]  = xmm1[127:96] + xmm1[95:64]
# xmm1[95:64]  = xmm2[63:32]  + xmm2[31:0]
# xmm1[127:96] = xmm2[127:96] + xmm2[95:64]
```
- **Encoding**: F2 0F 7C /r
- **Latency**: 5 cycles
- **Throughput**: 1/2

### Logical Operations

#### XORPS - Logical XOR
```asm
xorps xmm1, xmm2                # xmm1 = xmm1 XOR xmm2
xorps xmm0, xmm0                # Clear xmm0 to zero
```
- **Encoding**: 0F 57 /r
- **Latency**: 1 cycle
- **Throughput**: 1/0.33

---

## ARM AArch64 NEON Instructions

### Data Movement

#### LDR (SIMD) - Load SIMD register
```asm
ldr q1, [x19, x7, lsl #2]       # Load 4 floats (128 bits) into q1
```
- **Encoding**: 1x11 1100 01ii iiii iiii iinn nnnt tttt
- **Latency**: 4 cycles (L1 cache)
- **Throughput**: 2 per cycle
- **Addressing**: Base + (index << shift)

#### STR (SIMD) - Store SIMD register
```asm
str s0, [x22, x6, lsl #2]       # Store 1 float (32 bits)
```
- **Encoding**: Similar to LDR
- **Latency**: 1 cycle (store-to-load forwarding)
- **Throughput**: 1 per cycle

#### LDR (scalar) - Load scalar to SIMD
```asm
ldr s1, [x21, x6, lsl #2]       # Load 1 float into s1
```
- **Latency**: 4 cycles
- **Throughput**: 2 per cycle

### Arithmetic Operations

#### FMLA - Floating-point Multiply-Accumulate
```asm
fmla v0.4s, v1.4s, v2.4s        # v0 = v0 + (v1 * v2) [4 floats]
fmla s0, s1, s2                 # s0 = s0 + (s1 * s2) [scalar]
```
- **Encoding**: 0x00 1110 00ss ssmm mmmm 1100 11nn nnnd dddd
- **Latency**: 4 cycles
- **Throughput**: 1/0.5
- **Note**: Fused operation, more accurate than separate multiply + add

#### FADD - Floating-point Add
```asm
fadd v0.4s, v1.4s, v2.4s        # v0 = v1 + v2 [4 floats]
fadd s0, s1, s2                 # s0 = s1 + s2 [scalar]
```
- **Encoding**: 0x00 1110 00ss ssmm mmmm 1101 01nn nnnd dddd
- **Latency**: 3 cycles
- **Throughput**: 1/0.5

#### FMAX - Floating-point Maximum
```asm
fmax v0.4s, v1.4s, v2.4s        # v0[i] = max(v1[i], v2[i])
fmax s0, s1, s2                 # s0 = max(s1, s2) [scalar]
```
- **Encoding**: 0x00 1110 00ss ssmm mmmm 1111 01nn nnnd dddd
- **Latency**: 3 cycles
- **Throughput**: 1/0.5
- **Use**: ReLU activation

### Reduction Operations

#### FADDP - Floating-point Add Pairwise
```asm
faddp v0.4s, v1.4s, v2.4s
# v0[0] = v1[0] + v1[1]
# v0[1] = v1[2] + v1[3]
# v0[2] = v2[0] + v2[1]
# v0[3] = v2[2] + v2[3]
```
- **Encoding**: 0x10 1110 00ss ssmm mmmm 1101 01nn nnnd dddd
- **Latency**: 3 cycles
- **Throughput**: 1/0.5
- **Use**: Horizontal reduction (call twice for full reduction)

### Initialization

#### MOVI - Move Immediate
```asm
movi v0.4s, #0                  # Set all 4 floats to 0
```
- **Encoding**: 0x00 1111 0000 0iii iiii iiii i1nn nnnd dddd
- **Latency**: 1 cycle
- **Throughput**: 2 per cycle
- **Use**: Initialize accumulators

---

## ARM ARMv7 NEON Instructions

### Data Movement

#### VLD1 - Load multiple single elements
```asm
vld1.32 {q1}, [r2]              # Load 4 floats from [r2] into q1
vld1.32 {q1}, [r2]!             # Load and post-increment r2
```
- **Encoding**: 1111 0100 0x10 nnnn dddd tttt ssss nnnn
- **Latency**: 1 cycle (+ 3 for data availability)
- **Throughput**: 1 per cycle
- **Alignment**: Works best with 16-byte alignment

#### VST1 - Store multiple single elements
```asm
vst1.32 {q0}, [r2]              # Store 4 floats from q0 to [r2]
```
- **Encoding**: 1111 0100 0x00 nnnn dddd tttt ssss nnnn
- **Latency**: 1 cycle
- **Throughput**: 1 per cycle

#### VLDR/VSTR - Load/Store single register
```asm
vldr s0, [r2]                   # Load 1 float
vstr s0, [r2]                   # Store 1 float
```
- **Encoding**: 1110 110x UDxx nnnn dddd 1010 xxxx xxxx
- **Latency**: 4 cycles (load), 1 cycle (store)
- **Throughput**: 1 per cycle

### Arithmetic Operations

#### VMLA - Vector Multiply Accumulate
```asm
vmla.f32 q0, q1, q2             # q0 = q0 + (q1 * q2) [4 floats]
vmla.f32 s0, s1, s2             # s0 = s0 + (s1 * s2) [scalar]
```
- **Encoding**: 1111 0010 0Dss ssss dddd 1100 NQM1 mmmm
- **Latency**: 9 cycles
- **Throughput**: 1/2
- **Note**: Not fully pipelined on older ARM cores

#### VADD - Vector Add
```asm
vadd.f32 q0, q1, q2             # q0 = q1 + q2 [4 floats]
vadd.f32 d0, d1, d2             # d0 = d1 + d2 [2 floats]
```
- **Encoding**: 1111 0010 0Dss ssss dddd 1101 NQM0 mmmm
- **Latency**: 5 cycles
- **Throughput**: 1 per cycle

#### VMAX - Vector Maximum
```asm
vmax.f32 q0, q1, q2             # q0[i] = max(q1[i], q2[i])
vmax.f32 s0, s0, s1             # s0 = max(s0, s1)
```
- **Encoding**: 1111 0010 0Dss ssss dddd 1111 NQM0 mmmm
- **Latency**: 5 cycles
- **Throughput**: 1 per cycle

### Reduction Operations

#### VPADD - Vector Pairwise Add
```asm
vpadd.f32 d0, d1, d2
# d0[0] = d1[0] + d1[1]
# d0[1] = d2[0] + d2[1]
```
- **Encoding**: 1111 0011 0Dss ssss dddd 1101 NQM0 mmmm
- **Latency**: 5 cycles
- **Throughput**: 1 per cycle
- **Note**: Operates on D registers (64-bit), not Q registers

### Initialization

#### VMOV - Move immediate
```asm
vmov.f32 q0, #0.0               # Set all 4 floats to 0.0
vmov.f32 s0, #0.0               # Set scalar to 0.0
```
- **Encoding**: 1111 0011 1Dii iiii dddd 0000 NQM1 iiii (imm form)
- **Latency**: 1 cycle
- **Throughput**: 1 per cycle
- **Note**: Limited immediate values supported

---

## Instruction Encoding Formats

### x86-64 VEX Prefix (AVX)

VEX prefix structure (2 or 3 bytes):
```
2-byte VEX: 11000101 RXBmmmmm Wvvvv1pp
3-byte VEX: 11000100 RXBmmmmm Wvvvvlpp + opcode + ModRM + SIB + disp + imm
```

Components:
- **R**: Extension of ModRM.reg
- **X**: Extension of SIB.index
- **B**: Extension of ModRM.r/m or SIB.base
- **mmmmm**: Implied leading opcode bytes
- **W**: Operand size
- **vvvv**: Source register (inverted)
- **L**: Vector length (0=128-bit, 1=256-bit)
- **pp**: Implied prefix (00=none, 01=66, 10=F3, 11=F2)

### ARM AArch64 Instruction Format

Fixed 32-bit instruction format:
```
[31:21] [20:16] [15:10] [9:5] [4:0]
  op      Rm      imm     Rn    Rd
```

Example - FMLA v0.4s, v1.4s, v2.4s:
```
0000 1110 0010 0010 1100 1100 0100 0000
 op   Q op2  Rm  opc  1  Rn    Rd
```

### ARM ARMv7 Instruction Format

Fixed 32-bit instruction format:
```
[31:28] [27:20] [19:16] [15:12] [11:0]
 cond    opcode   Rn      Rd     operand2
```

For NEON:
```
1111 0010 0Dss ssss dddd tttt NQM1 mmmm
 111  op   Vn      Vd   opc      Vm
```

---

## Performance Characteristics

### x86-64 (Intel Skylake/Cascade Lake)

| Instruction | Latency | Throughput | Execution Units |
|------------|---------|------------|-----------------|
| VMOVUPS (load) | 7 | 0.5 | 2× Load |
| VMOVUPS (store) | 1 | 1.0 | 1× Store |
| VFMADD231PS | 4 | 0.5 | 2× FMA |
| VADDPS | 4 | 0.5 | 2× FP Add |
| VMAXPS | 4 | 0.5 | 2× FP Add |
| VHADDPS | 6 | 2.0 | FP Add |

### ARM Cortex-A72 (AArch64)

| Instruction | Latency | Throughput | Execution Units |
|------------|---------|------------|-----------------|
| LDR (SIMD) | 4 | 0.5 | 2× Load |
| STR (SIMD) | 1 | 1.0 | 1× Store |
| FMLA | 6 | 0.5 | 2× FP MAC |
| FADD | 5 | 0.5 | 2× FP Add |
| FMAX | 5 | 0.5 | 2× FP Add |
| FADDP | 5 | 0.5 | 2× FP Add |

### ARM Cortex-A9 (ARMv7)

| Instruction | Latency | Throughput | Execution Units |
|------------|---------|------------|-----------------|
| VLD1.32 | 1-3 | 1.0 | 1× Load |
| VST1.32 | 0-2 | 1.0 | 1× Store |
| VMLA.F32 | 9 | 2.0 | 1× FP MAC |
| VADD.F32 | 5 | 1.0 | 1× FP Add |
| VMAX.F32 | 5 | 1.0 | 1× FP Add |

Note: Latencies and throughputs are approximate and vary by microarchitecture.

---

## ISA Comparison Summary

| Feature | x86-64 AVX | x86-64 SSE | ARM NEON |
|---------|-----------|-----------|----------|
| SIMD Width | 256-bit | 128-bit | 128-bit |
| Floats/Operation | 8 | 4 | 4 |
| FMA Support | Yes (AVX2) | No | Yes (native) |
| Instruction Length | Variable (1-15 bytes) | Variable | Fixed (32-bit) |
| Register Count | 16 (ymm0-15) | 16 (xmm0-15) | 32 (v0-31) |
| Addressing Modes | Complex | Complex | Simple |
| Endianness | Little | Little | Little (config) |
| Encoding | VEX prefix | Legacy prefix | Fixed format |

This reference should help understand the low-level details of how these MLP implementations leverage SIMD capabilities across different architectures.
