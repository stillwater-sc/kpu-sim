# NVIDIA RTX 30 Series Specifications

| GPU Model | CUDA Cores | Base Clock (MHz) | Boost Clock (MHz) | Memory Type | Memory Size | Memory Bus | Memory Bandwidth | Tensor Cores | RT Cores | FP32 TFLOPs | INT8 TOPs |
|-----------|------------|------------------|-------------------|-------------|-------------|------------|------------------|--------------|----------|-------------|-----------|
| **RTX 3050** | 2,560 | 1,552 | 1,777 | GDDR6 | 8 GB | 128-bit | 224 GB/s | 80 | 20 | 9.1 | 91 |
| **RTX 3060** | 3,584 | 1,320 | 1,777 | GDDR6 | 12 GB | 192-bit | 360 GB/s | 112 | 28 | 12.7 | 102 |
| **RTX 3060 Ti** | 4,864 | 1,410 | 1,665 | GDDR6 | 8 GB | 256-bit | 448 GB/s | 152 | 38 | 16.2 | 130 |
| **RTX 3070** | 5,888 | 1,500 | 1,730 | GDDR6 | 8 GB | 256-bit | 448 GB/s | 184 | 46 | 20.4 | 163 |
| **RTX 3070 Ti** | 6,144 | 1,580 | 1,770 | GDDR6X | 8 GB | 256-bit | 608 GB/s | 192 | 48 | 21.8 | 174 |
| **RTX 3080** | 8,704 | 1,440 | 1,710 | GDDR6X | 10 GB | 320-bit | 760 GB/s | 272 | 68 | 29.8 | 238 |
| **RTX 3080 12GB** | 8,960 | 1,260 | 1,710 | GDDR6X | 12 GB | 384-bit | 912 GB/s | 280 | 70 | 30.6 | 245 |
| **RTX 3080 Ti** | 10,240 | 1,365 | 1,665 | GDDR6X | 12 GB | 384-bit | 912 GB/s | 320 | 80 | 34.1 | 273 |
| **RTX 3090** | 10,496 | 1,395 | 1,695 | GDDR6X | 24 GB | 384-bit | 936 GB/s | 328 | 82 | 35.6 | 285 |
| **RTX 3090 Ti** | 10,752 | 1,560 | 1,860 | GDDR6X | 24 GB | 384-bit | 1,008 GB/s | 336 | 84 | 40.0 | 320 |

## Key Specifications Summary

**Architecture:** All RTX 30 series cards are based on NVIDIA's Ampere architecture built on Samsung's 8nm process node.

**Ray Tracing Cores:** All cards feature 2nd generation RT cores for hardware-accelerated ray tracing.

**Tensor Cores:** All cards feature 3rd generation Tensor cores for AI workloads and DLSS support.

**Memory Technologies:**
- **GDDR6:** Used in RTX 3050, 3060, 3060 Ti, and 3070
- **GDDR6X:** Used in RTX 3070 Ti, 3080, 3080 Ti, 3090, and 3090 Ti

**GPU Dies:**
- **GA106:** RTX 3050, 3060
- **GA104:** RTX 3060 Ti, 3070, 3070 Ti  
- **GA102:** RTX 3080, 3080 Ti, 3090, 3090 Ti

## Performance Positioning

**Entry-Level:** RTX 3050, 3060 - Ideal for 1080p gaming  
**Mid-Range:** RTX 3060 Ti, 3070, 3070 Ti - Excellent for 1440p gaming  
**High-End:** RTX 3080, 3080 Ti - Strong 4K gaming performance  
**Enthusiast:** RTX 3090, 3090 Ti - 8K gaming and content creation workloads

## Performance Calculations

**FP32 TFLOPs:** Calculated as (CUDA Cores × Boost Clock × 2) ÷ 1,000,000  
**INT8 TOPs:** Calculated using Tensor Core performance with sparsity optimizations

The RTX 3090 Ti delivers 40 shader TFLOPs, 78 RT TFLOPs, and 320 Tensor TFLOPs according to NVIDIA's official specifications.

*Note: Clock speeds and performance figures represent reference/Founders Edition specifications. AIB partner cards may have different factory overclocked specifications. Performance calculations are theoretical peak values.*

## Additional info

- **Core Frequencies**: Both base and boost clock speeds
- **Memory Bandwidth**: Ranging from 224 GB/s on the RTX 3050 up to 1,008 GB/s on the RTX 3090 Ti
- **CUDA Cores**: From 2,560 on the RTX 3050 to 10,752 on the RTX 3090 Ti
- **Tensor Cores**: All RTX 30 series cards feature 3rd generation Tensor Cores, with counts ranging from 80 to 336

The table also includes additional relevant information like RT core counts, memory specifications, and GPU positioning. All RTX 30 series GPUs are built on NVIDIA's Ampere architecture using Samsung's 8nm process node, and they feature 2nd generation RT cores and 3rd generation Tensor cores.

Key highlights include the RTX 3090 Ti being the first GeForce card to achieve 1 TB/s memory bandwidth, and the series spanning from entry-level 1080p gaming (RTX 3050) all the way up to 8K gaming capabilities (RTX 3090/3090 Ti).

1. **FP32 TFLOPs** - Single-precision floating-point performance calculated from CUDA cores and boost clocks
2. **INT8 TOPs** - Integer 8-bit operations per second, primarily leveraging the Tensor cores

The calculations show the theoretical peak performance capabilities of each card:

- **FP32 performance** ranges from 9.1 TFLOPs (RTX 3050) up to 40.0 TFLOPs (RTX 3090 Ti)
- **INT8 performance** ranges from 91 TOPs (RTX 3050) up to 320 TOPs (RTX 3090 Ti)

These performance metrics are crucial for understanding computational capabilities, especially for AI workloads where INT8 performance is important for inference tasks, and FP32 performance matters for training and traditional compute workloads. The RTX 3090 Ti values align with NVIDIA's official specifications of 40 shader TFLOPs and 320 Tensor TFLOPs.