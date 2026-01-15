# NVIDIA RTX 50 Series Specifications

| GPU Model | CUDA Cores | Base Clock (MHz) | Boost Clock (MHz) | Memory Type | Memory Size | Memory Bus | Memory Bandwidth | Tensor Cores | RT Cores | FP32 TFLOPs | INT8 TOPs |
|-----------|------------|------------------|-------------------|-------------|-------------|------------|------------------|--------------|----------|-------------|-----------|
| **RTX 5050** | 2,560 | 1,980 | 2,570 | GDDR6 | 8 GB | 128-bit | 224 GB/s | 80 | 20 | 13.2 | 421 |
| **RTX 5060** | 3,840 | 2,210 | 2,550 | GDDR7 | 8 GB | 128-bit | 448 GB/s | 120 | 30 | 19.6 | 627 |
| **RTX 5060 Ti 8GB** | 4,608 | 2,400 | 2,650 | GDDR7 | 8 GB | 128-bit | 448 GB/s | 144 | 36 | 24.4 | 781 |
| **RTX 5060 Ti 16GB** | 4,608 | 2,400 | 2,650 | GDDR7 | 16 GB | 128-bit | 448 GB/s | 144 | 36 | 24.4 | 781 |
| **RTX 5070** | 6,144 | 1,980 | 2,410 | GDDR7 | 12 GB | 192-bit | 672 GB/s | 192 | 48 | 29.6 | 947 |
| **RTX 5070 Ti** | 8,960 | 2,300 | 2,475 | GDDR7 | 16 GB | 256-bit | 896 GB/s | 280 | 70 | 44.4 | 1,420 |
| **RTX 5080** | 10,752 | 2,295 | 2,550 | GDDR7 | 16 GB | 256-bit | 960 GB/s | 336 | 84 | 54.9 | 1,756 |
| **RTX 5090** | 21,760 | 2,010 | 2,410 | GDDR7 | 32 GB | 512-bit | 1,792 GB/s | 680 | 170 | 104.9 | 3,352 |
| **RTX 5090D** | 19,456 | 2,010 | 2,410 | GDDR7 | 32 GB | 448-bit | 1,568 GB/s | 608 | 152 | 93.8 | 2,998 |

## Key Specifications Summary

**Architecture:** All RTX 50 series cards are based on NVIDIA's Blackwell architecture built on TSMC's custom 4N process node.

**Ray Tracing Cores:** All cards feature 4th generation RT cores with doubled ray/triangle intersection rates.

**Tensor Cores:** All cards feature 5th generation Tensor cores with native support for FP4, FP6, FP8, and FP16 precision formats.

**Memory Technologies:**
- **GDDR6:** Used only in RTX 5050 (entry-level cost optimization)
- **GDDR7:** Used in all other RTX 50 series cards for enhanced bandwidth

**GPU Dies:**
- **GB207:** RTX 5050  
- **GB206:** RTX 5060, 5060 Ti
- **GB205:** RTX 5070
- **GB203:** RTX 5070 Ti, 5080
- **GB202:** RTX 5090, 5090D

## Performance Positioning

**Entry-Level:** RTX 5050, 5060 - Optimized for 1080p gaming with DLSS 4  
**Mid-Range:** RTX 5060 Ti, 5070 - Excellent for 1440p gaming with Multi-Frame Generation  
**High-End:** RTX 5070 Ti, 5080 - Strong 4K gaming performance  
**Enthusiast:** RTX 5090 - Flagship 4K and 8K gaming with professional AI capabilities

## Revolutionary New Features

**DLSS 4:** Multi-Frame Generation can generate up to 3 additional frames between traditional rendered frames, multiplying performance up to 8x
**Neural Rendering:** RTX Neural Materials can reduce texture memory requirements by up to 33%
**Reflex 2:** Up to 75% reduction in PC latency for competitive gaming
**FP4 Precision:** Native support for 4-bit floating point operations for AI workloads
**Neural Shaders:** Enhanced shader and tensor core intermixing capabilities

## Technical Innovations

**Memory Bandwidth:** Up to 1,792 GB/s on RTX 5090 - the highest ever on consumer GPU
**AI Performance:** Up to 3,352 AI TOPS on RTX 5090 using FP4 precision  
**Ray Tracing:** 4th generation RT cores with 2x intersection performance over Ada Lovelace
**Power Efficiency:** Improved performance per watt despite increased capabilities

## Performance Calculations

**FP32 TFLOPs:** Calculated as (CUDA Cores × Boost Clock × 2) ÷ 1,000,000  
**INT8 TOPs:** Calculated using 5th generation Tensor Core performance optimized for AI workloads

*Note: RTX 5090D is a China-specific variant with reduced specifications due to export restrictions. The RTX 5050 is the only Blackwell GPU to use GDDR6 memory for cost optimization. Clock speeds and performance figures represent reference specifications. AIB partner cards may have different factory overclocked specifications. Performance calculations represent theoretical peak values with architectural enhancements.*

## Additional info

**Revolutionary Advancements:**
- Based on NVIDIA's Blackwell architecture with 5th generation Tensor Cores and 4th generation RT cores
- DLSS 4 with Multi-Frame Generation can multiply performance up to 8x
- RTX 5090 features 32GB of GDDR7 memory with 1,792 GB/s bandwidth - the highest ever on consumer GPUs

**Performance Highlights:**
- **FP32 Performance:** Ranges from 13.2 TFLOPs (RTX 5050) up to 104.9 TFLOPs (RTX 5090)
- **AI Performance:** Ranges from 421 AI TOPS (RTX 5050) up to 3,352 AI TOPS (RTX 5090) using FP4 precision

**Memory Innovation:**
- First consumer GPUs to feature GDDR7 memory (except RTX 5050 which uses GDDR6 for cost optimization)
- Massive memory capacity increases: RTX 5090 with 32GB, RTX 5070 Ti with 16GB

**Architectural Features:**
- All CUDA cores are now fully FP32/INT32 compatible, doubling versatility for AI workloads
- RT cores deliver doubled ray/triangle intersection rates compared to previous generation
- Neural rendering technologies including RTX Neural Materials

**Availability Issues:**
- Launch marred by severe availability issues and pricing well above MSRP
- Support for 32-bit PhysX, OpenCL, and CUDA applications was dropped

This represents NVIDIA's most significant generational leap in AI capabilities, with the RTX 5090 delivering over 3x the AI performance of the RTX 4090, making it a powerhouse for both gaming and AI workloads.