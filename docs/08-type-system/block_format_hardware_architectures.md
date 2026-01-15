# Block Format Tradeoffs and Decompression Hardware Architectures

## Overview

This document provides a detailed comparison of block compression formats (ZFP, MX/Microscaling, NVFP4) and their hardware decompression architectures. These formats trade memory bandwidth for compute (decompression), making them attractive for memory-bound workloads like LLM inference and scientific computing.

---

## Format Comparison Summary

| Property | ZFP | MX (MXFP4/8) | NVFP4 |
|----------|-----|--------------|-------|
| **Origin** | LLNL (scientific computing) | OCP consortium (AI/ML) | NVIDIA (LLM inference) |
| **Block Size** | 4^d (4, 16, 64, 256) | 32 elements | 16 elements |
| **Compression Ratio** | 2-10x (configurable) | 2-4x (fixed by variant) | ~3.5x vs FP16 |
| **Encode Complexity** | High (transform + coding) | Low (find max, scale) | Low (scale per block) |
| **Decode Complexity** | High (inverse transform) | Very low (multiply) | Very low (multiply) |
| **Error Model** | Smooth (transform-based) | Block-correlated | Block-correlated |
| **Hardware Support** | FPGA (research) | AMD CDNA4, NVIDIA Blackwell | NVIDIA Blackwell |
| **Primary Use Case** | Scientific data, HPC | Training & inference | LLM KV cache |

---

## ZFP (Zfp Floating Point)

### Algorithm Overview

ZFP is a lossy/lossless compression format developed at Lawrence Livermore National Laboratory for scientific floating-point data. It achieves high compression through a sophisticated multi-stage pipeline.

**Source**: [ZFP Algorithm Documentation](https://zfp.readthedocs.io/en/release1.0.1/algorithm.html)

### Compression Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Block Partition                                        │
│ - Partition array into 4^d blocks (4 for 1D, 16 for 2D, 64 for 3D)│
│ - Zero-pad boundary blocks                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Block Floating-Point Conversion                        │
│ - Find maximum exponent in block                                │
│ - Convert all values to fixed-point with shared exponent        │
│ - Result: 4^d signed integers (31 or 63 bits each)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Decorrelating Transform (Lifting Scheme)               │
│ - Near-orthogonal transform similar to DCT                      │
│ - Exploits separability for efficiency                          │
│ - Cost: 2.5d additions + 1.5d bit shifts per integer            │
│ - "Smooth" data → small coefficients clustered around zero      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Coefficient Reordering (Zig-Zag)                       │
│ - Reorder by expected magnitude (like JPEG)                     │
│ - Low frequencies (large magnitude) first                       │
│ - High frequencies (small magnitude) last                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: Negabinary Conversion                                  │
│ - Convert two's complement to negabinary (base -2)              │
│ - Cost: 1 addition + 1 XOR per integer                          │
│ - Property: magnitude in leading bits regardless of sign        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 6: Bit-Plane Transposition                                │
│ - Reorganize bits by bit-plane (MSB to LSB)                     │
│ - Groups bits by significance across all coefficients           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 7: Embedded Coding                                        │
│ - Run-length encode bit-planes                                  │
│ - Variable-length codes for efficiency                          │
│ - Can truncate at any point (progressive refinement)            │
└─────────────────────────────────────────────────────────────────┘
```

### Decompression Pipeline (Inverse)

```
Embedded Decoding → Bit-Plane Untranspose → Negabinary→Two's Complement
    → Coefficient Reorder → Inverse Transform → Fixed→Float → Block Assembly
```

### Hardware Implementation Challenges

The embedded coding stage is **inherently serial** - each bit depends on previous bits. This is the primary bottleneck for hardware implementation.

**Solutions**:

1. **DE-ZFP** ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0141933122000266)): Replaces embedded coding with dictionary-based encoding
   - Single-cycle encoding vs serial bit emission
   - 4-13% compression efficiency loss
   - Up to 19x throughput improvement

2. **ZFP-V** ([IEEE Xplore](https://ieeexplore.ieee.org/document/8977918/)): Hardware-optimized variant
   - Modified serial portion for parallelism
   - 2x performance over best-effort hardware ZFP
   - Less on-chip resources

3. **ZHW** ([LLNL GitHub](https://github.com/LLNL/zhw)): Reference SystemC implementation
   - 15-200x speedup over software (dimension dependent)
   - Modular design supports posits and other formats
   - Separate encode/decode pipelines

### ZFP Decompression Hardware Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    ZFP Decode Pipeline                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Bit Stream  │───→│ Embedded    │───→│ Bit-Plane          │  │
│  │ Buffer      │    │ Decoder     │    │ Untranspose        │  │
│  │ (FIFO)      │    │ (Serial!)   │    │ (4^d × bits)       │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                ↓                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Float       │←───│ Inverse     │←───│ Negabinary          │  │
│  │ Converter   │    │ Lifting     │    │ to Two's Comp       │  │
│  │ (exp+mant)  │    │ Transform   │    │ (XOR + ADD)         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│        ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Decompressed Block (4^d floats)                ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘

Pipeline Characteristics:
- Latency: High (multiple stages, serial decoding)
- Throughput: Limited by embedded decoder
- Area: Moderate (transform logic dominates)
```

### ZFP Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| High compression (up to 10x) | Complex decode logic |
| Configurable rate/precision/accuracy | Serial embedded coding bottleneck |
| Smooth error distribution | Not ML-optimized |
| Lossless mode available | Large block sizes (alignment) |
| Works on any floating-point data | High latency |

---

## MX (Microscaling) Formats

### Overview

MX formats are standardized by the [Open Compute Project](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) (AMD, ARM, Intel, Meta, Microsoft, NVIDIA, Qualcomm). They use **block floating-point** with shared exponents.

**Source**: [OCP MX Scaling Formats](https://fprox.substack.com/p/ocp-mx-scaling-formats)

### Format Structure

```
┌────────────────────────────────────────────────────────────────┐
│                    MX Block (32 elements)                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Shared Scale: E8M0 (8-bit exponent, power-of-two only)   │ │
│  │ = largest power-of-two in block / largest representable  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌────┬────┬────┬────┬─────────────────────────┬────┬────┐   │
│  │ E0 │ E1 │ E2 │ E3 │          ...            │E30 │E31 │   │
│  └────┴────┴────┴────┴─────────────────────────┴────┴────┘   │
│    ↑                                                           │
│    └─ Element format varies by MX variant                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### MX Variants

| Variant | Element Format | Bits/Element | Block Overhead | Effective Bits | Compression vs FP32 |
|---------|---------------|--------------|----------------|----------------|---------------------|
| MXFP8 E5M2 | 1s + 5e + 2m | 8 | 8 bits/32 elem | 8.25 | 3.9x |
| MXFP8 E4M3 | 1s + 4e + 3m | 8 | 8 bits/32 elem | 8.25 | 3.9x |
| MXFP6 E3M2 | 1s + 3e + 2m | 6 | 8 bits/32 elem | 6.25 | 5.1x |
| MXFP6 E2M3 | 1s + 2e + 3m | 6 | 8 bits/32 elem | 6.25 | 5.1x |
| MXFP4 E2M1 | 1s + 2e + 1m | 4 | 8 bits/32 elem | 4.25 | 7.5x |
| MXINT8 | 1s + 7m | 8 | 8 bits/32 elem | 8.25 | 3.9x |

### MX Encode (Compression)

```python
def mx_encode(block_32_floats):
    # 1. Find maximum absolute value
    max_val = max(abs(x) for x in block_32_floats)

    # 2. Compute shared exponent (E8M0 = power-of-two only)
    shared_exp = floor(log2(max_val)) if max_val > 0 else 0
    scale = 2 ** shared_exp

    # 3. Scale each element and quantize to element format
    elements = []
    for x in block_32_floats:
        scaled = x / scale
        quantized = quantize_to_element_format(scaled)  # e.g., E2M1
        elements.append(quantized)

    return shared_exp, elements
```

### MX Decode (Decompression)

```python
def mx_decode(shared_exp, elements):
    scale = 2 ** shared_exp
    return [dequantize(e) * scale for e in elements]
```

**Key insight**: Decompression is just **multiply by power-of-two** (bit shift)!

### MX Hardware Architecture

**Source**: [MXDOTP RISC-V Extension](https://arxiv.org/html/2505.13159v1), [Precision-Scalable MX Hardware](https://arxiv.org/html/2505.22404v1)

```
┌──────────────────────────────────────────────────────────────────┐
│                 MX Decode + MAC Unit                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Inputs:                                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │ Block A        │  │ Block B        │  │ Shared Exponents   │ │
│  │ (32 × 4-8 bits)│  │ (32 × 4-8 bits)│  │ (exp_A + exp_B)    │ │
│  └───────┬────────┘  └───────┬────────┘  └─────────┬──────────┘ │
│          │                   │                     │             │
│          ▼                   ▼                     │             │
│  ┌─────────────────────────────────────────────┐  │             │
│  │         Element Multipliers                  │  │             │
│  │  ┌─────┐ ┌─────┐ ┌─────┐      ┌─────┐      │  │             │
│  │  │2b×2b│ │2b×2b│ │2b×2b│ ···  │2b×2b│      │  │             │
│  │  └──┬──┘ └──┬──┘ └──┬──┘      └──┬──┘      │  │             │
│  │     │       │       │            │          │  │             │
│  │  16 elementary 2-bit multipliers            │  │             │
│  │  (flexibly interconnected for all MX modes) │  │             │
│  └─────────────────────┬───────────────────────┘  │             │
│                        │                          │             │
│                        ▼                          │             │
│  ┌─────────────────────────────────────────────┐  │             │
│  │         L1 Adder (Partial Products)          │  │             │
│  │  - INT8/FP8/FP6: accumulate 4-bit partials  │  │             │
│  │  - FP4: sum four E3M4 products directly     │  │             │
│  └─────────────────────┬───────────────────────┘  │             │
│                        │                          │             │
│                        ▼                          ▼             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              L2 Adder (Final Accumulation)                  ││
│  │  - FP32 adder with 26-bit mantissa                          ││
│  │  - Shared exponents applied here: result × 2^(exp_A+exp_B)  ││
│  │  - INT8/FP4 bypass alignment logic (critical path balance)  ││
│  └─────────────────────────────────────────────────────────────┘│
│                        │                                         │
│                        ▼                                         │
│               FP32 Accumulated Result                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Pipeline Characteristics (MXDOTP):
- Stages: 3 pipeline stages
- Throughput: 1 result/cycle (8 elements × 8 elements dot product)
- Shared exponent: Streamed via SSR (Stream Semantic Registers)
- Performance: 356 GFLOPS/W @ 12nm, 0.8V, 1GHz
- Speedup: 25x over software, 12.5x energy efficiency improvement
```

### MX Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Extremely simple decode (shift) | Lower compression than ZFP |
| Native tensor core support | Block-correlated errors |
| Industry standard (OCP) | Fixed block size (32) |
| Multiple precision variants | Outliers hurt accuracy |
| Fast encode and decode | Less flexible than ZFP |

### MX+ Extension

**Source**: [MX+ Paper](https://arxiv.org/abs/2510.14557)

MX+ addresses the **outlier problem** in MX formats. When one value in a block is much larger than others, the shared exponent is dominated by it, reducing precision for the rest.

**Solution**: Repurpose the outlier's exponent field as extended mantissa:
- Outlier uses scale directly (no element exponent needed)
- Extra bits go to mantissa for higher precision
- Negligible storage overhead

---

## NVFP4 (NVIDIA FP4)

### Overview

NVFP4 is NVIDIA's proprietary 4-bit format optimized for LLM inference, introduced with the Blackwell architecture.

**Source**: [NVIDIA NVFP4 Blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)

### Format Structure

```
┌────────────────────────────────────────────────────────────────┐
│                    NVFP4 Block (16 elements)                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Two-Level Scaling:                                            │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Level 1: Micro-block scale (E4M3 FP8) per 16 elements  │   │
│  │ Level 2: Tensor-level scale (FP32) - one per tensor    │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                │
│  Elements: E2M1 format (1 sign + 2 exponent + 1 mantissa)     │
│                                                                │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                   │
│  │ E0 │ E1 │ E2 │ E3 │ E4 │ E5 │ E6 │ E7 │  (4 bits each)    │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                   │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                   │
│  │ E8 │ E9 │E10 │E11 │E12 │E13 │E14 │E15 │                   │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                   │
│                                                                │
│  Storage: 16 × 4 bits + 8 bits scale = 72 bits                │
│  Effective: 4.5 bits/element                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### NVFP4 vs MXFP4 Comparison

| Property | NVFP4 | MXFP4 |
|----------|-------|-------|
| Block size | 16 | 32 |
| Element format | E2M1 | E2M1 |
| Scale format | E4M3 (FP8) | E8M0 (exponent-only) |
| Scale precision | 8 bits with mantissa | 8 bits exponent only |
| Effective bits/element | 4.5 | 4.25 |
| Accuracy vs BF16 | ~5% better than MXFP4 | Baseline |
| Hardware support | NVIDIA Blackwell | AMD CDNA4, Blackwell |

**Key difference**: NVFP4 uses **finer-grained scaling** (16 vs 32) with **higher-precision scales** (E4M3 vs E8M0), trading slightly more storage for ~5% better accuracy.

### Blackwell Decompression Architecture

**Source**: [Blackwell Architecture Analysis](https://arxiv.org/html/2512.02189v1)

```
┌──────────────────────────────────────────────────────────────────┐
│              Blackwell 5th-Gen Tensor Core                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Tensor Memory (TMEM)                        │ │
│  │  - Compressed FP4/FP6/FP8 storage                          │ │
│  │  - Direct tensor core access                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        ↓                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Decompression Engine (DE)                      │ │
│  │  - Hardware accelerated: 800 GB/s throughput               │ │
│  │  - Inline dequantization FP4 → FP8                         │ │
│  │  - Transparent to tensor core operations                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        ↓                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              5th-Gen Tensor Core                            │ │
│  │  - Native FP4/FP6/FP8 compute                              │ │
│  │  - Block-scaled operations built-in                        │ │
│  │  - Automatic grouping and scaling                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        ↓                                         │
│                 FP32 Accumulated Result                          │
│                                                                  │
│  Performance:                                                    │
│  - Blackwell: 10 petaFLOPS NVFP4                               │
│  - Blackwell Ultra: 15 petaFLOPS NVFP4                         │
│  - 7.5x improvement over Hopper H100/H200                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### KV Cache Application

For LLM inference, NVFP4 is primarily used to compress the KV cache:

**Source**: [NVFP4 KV Cache Blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)

```
┌─────────────────────────────────────────────────────────────────┐
│                    KV Cache Compression                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Standard (FP16):                                               │
│  - Llama 3 70B, 128K context: ~40 GB per user                  │
│  - Memory-bound at scale                                        │
│                                                                 │
│  With NVFP4:                                                    │
│  - 50% memory reduction vs FP8                                  │
│  - ~3.5x reduction vs FP16                                      │
│  - <1% accuracy loss vs BF16/FP8 baseline                      │
│  - Up to 3x TTFT improvement                                    │
│  - 20% higher cache hit rates                                   │
│                                                                 │
│  Pipeline:                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │ Compute │───→│ Quantize│───→│ Store   │───→│Dequant  │     │
│  │ K/V     │    │ to NVFP4│    │ in HBM  │    │ to FP8  │     │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│                                                    ↓            │
│                                              ┌─────────┐       │
│                                              │Attention│       │
│                                              │ Compute │       │
│                                              └─────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hardware Architecture Comparison

### Decompression Complexity

| Format | Decode Operations | Cycles/Element | Hardware Complexity |
|--------|------------------|----------------|---------------------|
| **ZFP** | Bit decode + inverse transform + float convert | 10-50+ | High (transform logic) |
| **MX** | Multiply by 2^exp (bit shift) | ~1 | Very low |
| **NVFP4** | Scale lookup + multiply | ~1 | Low |

### Pipeline Comparison

```
ZFP Decode Pipeline (High Latency):
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│Embedded│──→│Bit-Plan│──→│Negabin │──→│Inverse │──→│Float   │
│Decode  │   │Untrans │   │Convert │   │Lifting │   │Convert │
│(Serial)│   │        │   │        │   │        │   │        │
└────────┘   └────────┘   └────────┘   └────────┘   └────────┘
  Cycles:      Variable      4^d          4^d         4^d

MX Decode Pipeline (Low Latency):
┌────────────┐   ┌────────────┐
│Extract     │──→│Multiply by │──→ FP32 Result
│Elements    │   │2^shared_exp│
└────────────┘   └────────────┘
  Cycles: 1        Cycles: 1

NVFP4 Decode Pipeline (Low Latency):
┌────────────┐   ┌────────────┐   ┌────────────┐
│Extract     │──→│Apply       │──→│Apply       │──→ FP8 Result
│4-bit Elem  │   │Block Scale │   │Tensor Scale│
└────────────┘   └────────────┘   └────────────┘
  Cycles: 1        Cycles: 1        Cycles: 1
```

### Area and Power Tradeoffs

| Format | Decode Unit Area | Power per Decode | Throughput |
|--------|-----------------|------------------|------------|
| **ZFP** | Large (transform) | High | Limited by serial decode |
| **MX** | Small (shifter) | Low | Memory bandwidth limited |
| **NVFP4** | Small (multiplier) | Low | Memory bandwidth limited |

---

## Recommendations for KPU

### When to Use Each Format

| Use Case | Recommended Format | Rationale |
|----------|-------------------|-----------|
| **LLM KV Cache** | NVFP4 or MX | Simple decode, good accuracy |
| **Weight Storage** | MXFP4/MXFP8 | Standard, hardware support |
| **Scientific Data** | ZFP | Best compression, smooth errors |
| **Training** | MXFP8 | Accuracy-critical |
| **Inference** | MXFP4/NVFP4 | Speed-critical |

### KPU Decompression Point Recommendations

| Format | Recommended Decompress Point | Rationale |
|--------|------------------------------|-----------|
| **ZFP** | At DMA (L3 entry) | High decode latency, amortize early |
| **MX** | At BlockMover (L3→L2) | Low latency, keep L3 compressed |
| **NVFP4** | At BlockMover (L3→L2) | Low latency, keep L3 compressed |

### Hardware Implementation Priority

For KPU, recommended implementation order:

1. **MX support first** - Industry standard, simple hardware, wide adoption
2. **NVFP4 support** - If targeting LLM inference, excellent KV cache compression
3. **ZFP support** - If targeting scientific computing, best compression

---

## Sources

### ZFP
- [ZFP Algorithm Documentation](https://zfp.readthedocs.io/en/release1.0.1/algorithm.html)
- [ZHW Hardware Implementation (GitHub)](https://github.com/LLNL/zhw)
- [DE-ZFP FPGA Implementation](https://www.sciencedirect.com/science/article/abs/pii/S0141933122000266)
- [ZFP-V Hardware-Optimized Variant](https://ieeexplore.ieee.org/document/8977918/)

### MX Formats
- [OCP MX Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [OCP MX Scaling Formats Overview](https://fprox.substack.com/p/ocp-mx-scaling-formats)
- [MXDOTP RISC-V Extension](https://arxiv.org/html/2505.13159v1)
- [Precision-Scalable MX Hardware](https://arxiv.org/html/2505.22404v1)
- [MX+ Extension](https://arxiv.org/abs/2510.14557)

### NVFP4
- [NVFP4 Introduction](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [Blackwell Architecture Analysis](https://arxiv.org/html/2512.02189v1)
- [NVIDIA Tensor Core Evolution](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)

### Additional Resources
- [RocketKV KV Cache Compression](https://github.com/NVlabs/RocketKV)
- [NVIDIA kvpress Library](https://github.com/NVIDIA/kvpress)
- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)

---

*Document created: 2026-01-15*
*Related: BLOCK_FORMAT_TYPE_SYSTEM.md, TENSOR_TYPE_SYSTEM.md*
