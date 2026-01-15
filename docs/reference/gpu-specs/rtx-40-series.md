# NVIDIA RTX 40 Series Specifications

| GPU Model | CUDA Cores | Base Clock (MHz) | Boost Clock (MHz) | Memory Type | Memory Size | Memory Bus | Memory Bandwidth | Tensor Cores | RT Cores | FP32 TFLOPs | INT8 TOPs |
|-----------|------------|------------------|-------------------|-------------|-------------|------------|------------------|--------------|----------|-------------|-----------|
| **RTX 4060** | 3,072 | 1,830 | 2,460 | GDDR6 | 8 GB | 128-bit | 272 GB/s | 96 | 24 | 15.1 | 242 |
| **RTX 4060 Ti 8GB** | 4,352 | 2,310 | 2,535 | GDDR6 | 8 GB | 128-bit | 288 GB/s | 136 | 34 | 22.1 | 347 |
| **RTX 4060 Ti 16GB** | 4,352 | 2,310 | 2,535 | GDDR6 | 16 GB | 128-bit | 288 GB/s | 136 | 34 | 22.1 | 347 |
| **RTX 4070** | 5,888 | 1,920 | 2,475 | GDDR6X | 12 GB | 192-bit | 504 GB/s | 184 | 46 | 29.1 | 469 |
| **RTX 4070 Super** | 7,168 | 1,980 | 2,475 | GDDR6X | 12 GB | 192-bit | 504 GB/s | 224 | 56 | 35.5 | 571 |
| **RTX 4070 Ti** | 7,680 | 2,310 | 2,610 | GDDR6X | 12 GB | 192-bit | 504 GB/s | 240 | 60 | 40.1 | 646 |
| **RTX 4070 Ti Super** | 8,448 | 2,340 | 2,610 | GDDR6X | 16 GB | 256-bit | 672 GB/s | 264 | 66 | 44.1 | 711 |
| **RTX 4080** | 9,728 | 2,205 | 2,505 | GDDR6X | 16 GB | 256-bit | 717 GB/s | 304 | 76 | 48.7 | 782 |
| **RTX 4080 Super** | 10,240 | 2,295 | 2,550 | GDDR6X | 16 GB | 256-bit | 736 GB/s | 320 | 80 | 52.2 | 840 |
| **RTX 4090** | 16,384 | 2,230 | 2,520 | GDDR6X | 24 GB | 384-bit | 1,008 GB/s | 512 | 128 | 82.6 | 1,321 |
| **RTX 4090D** | 14,592 | 2,230 | 2,520 | GDDR6X | 24 GB | 384-bit | 1,008 GB/s | 456 | 114 | 73.6 | 1,177 |

## Key Specifications Summary

**Architecture:** All RTX 40 series cards are based on NVIDIA's Ada Lovelace architecture built on TSMC's custom 4nm process node.

**Ray Tracing Cores:** All cards feature 3rd generation RT cores for enhanced hardware-accelerated ray tracing.

**Tensor Cores:** All cards feature 4th generation Tensor cores with support for FP8, FP16, bfloat16, and TensorFloat-32 precision formats.

**Memory Technologies:**
- **GDDR6:** Used in RTX 4060, 4060 Ti
- **GDDR6X:** Used in RTX 4070, 4070 Super, 4070 Ti, 4070 Ti Super, 4080, 4080 Super, 4090, 4090D

**GPU Dies:**
- **AD107:** RTX 4060  
- **AD106:** RTX 4060 Ti
- **AD104:** RTX 4070, 4070 Super, 4070 Ti, 4070 Ti Super
- **AD103:** RTX 4080, 4080 Super
- **AD102:** RTX 4090, 4090D

## Performance Positioning

**Entry-Level:** RTX 4060, 4060 Ti - Optimized for 1080p gaming with DLSS 3  
**Mid-Range:** RTX 4070, 4070 Super - Excellent for 1440p gaming  
**High-End:** RTX 4070 Ti, 4070 Ti Super, 4080, 4080 Super - Strong 4K gaming performance  
**Enthusiast:** RTX 4090 - Flagship 4K and 8K gaming with content creation capabilities

## New Features

**DLSS 3:** Frame Generation technology that can multiply performance up to 4x vs. brute-force rendering
**AV1 Encoding:** 8th generation dual NVENC with 8K 10-bit 120fps AV1 encoding support  
**Shader Execution Reordering:** Improves ray tracing performance up to 3x in supported games
**Optical Flow Accelerator:** Enhanced for DLSS 3 frame generation

## Performance Calculations

**FP32 TFLOPs:** Calculated as (CUDA Cores × Boost Clock × 2) ÷ 1,000,000  
**INT8 TOPs:** Calculated using 4th generation Tensor Core performance with sparsity optimizations

*Note: RTX 4090D is a China-specific variant with reduced specifications due to export restrictions. Clock speeds and performance figures represent reference/Founders Edition specifications. AIB partner cards may have different factory overclocked specifications. Performance calculations are theoretical peak values.*

## Additional info

**Major Improvements over RTX 30 Series:**
- Built on NVIDIA's Ada Lovelace architecture with 4th generation Tensor Cores and 3rd generation RT cores
- Up to 2x leap in power efficiency compared to previous generation
- DLSS 3 with Frame Generation can deliver up to 4x performance boost
- Enhanced AV1 encoding support for better streaming

**Performance Range:**
- **FP32 Performance:** Ranges from 15.1 TFLOPs (RTX 4060) up to 82.6 TFLOPs (RTX 4090)
- **INT8 Performance:** Ranges from 242 TOPs (RTX 4060) up to 1,321 TOPs (RTX 4090)

**Notable Features:**
- RTX 4090 delivers 82.6 TFlops of FP32 compute power and 1,008 GB/s memory bandwidth
- The Super variants provide enhanced performance at better price points
- RTX 4090D is a China-specific model with 14,592 CUDA cores due to export restrictions

This generation represents a significant advancement in both gaming performance and AI capabilities, with DLSS 3 being a standout feature that can dramatically improve frame rates in supported games.