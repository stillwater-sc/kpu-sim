# NVIDIA AI Accelerator Specifications

## Complete AI Accelerator Specifications Table

| GPU Model | Architecture | CUDA Cores | Base Clock (MHz) | Boost Clock (MHz) | Memory Type | Memory Size | Memory Bus | Memory Bandwidth | Tensor Cores | RT Cores | TDP (Watts) | FP32 TFLOPs | AI Performance |
|-----------|--------------|-------------|------------------|-------------------|-------------|-------------|------------|------------------|--------------|----------|-------------|-------------|----------------|
| **P100** | Pascal | 3,584 | 1,190 | 1,329 | HBM2 | 16 GB | 4096-bit | 732 GB/s | - | - | 250W | 9.5 | - |
| **V100** | Volta | 5,120 | 1,245 | 1,380 | HBM2 | 16/32 GB | 4096-bit | 900 GB/s | 640 | - | 300W | 14.1 | 125 TOPS |
| **A100 40GB** | Ampere | 6,912 | 765 | 1,410 | HBM2e | 40 GB | 5120-bit | 1,555 GB/s | 432 | - | 400W | 19.5 | 312 TOPS |
| **A100 80GB** | Ampere | 6,912 | 765 | 1,410 | HBM2e | 80 GB | 5120-bit | 2,039 GB/s | 432 | - | 400W | 19.5 | 312 TOPS |
| **H100** | Hopper | 16,896 | 1,290 | 1,980 | HBM3 | 80 GB | 5120-bit | 3,350 GB/s | 528 | - | 700W | 67.0 | 1,000 TOPS |
| **H200** | Hopper | 16,896 | 1,290 | 1,980 | HBM3e | 141 GB | 5120-bit | 4,800 GB/s | 528 | - | 700W | 67.0 | 1,200 TOPS |
| **B100** | Blackwell | 28,672 | 1,350 | 2,000 | HBM3e | 192 GB | 8192-bit | 8,000 GB/s | 896 | - | 700W | 114.6 | 2,500 TOPS |
| **B200** | Blackwell | 36,864 | 1,500 | 2,100 | HBM3e | 192 GB | 8192-bit | 8,000 GB/s | 1,152 | - | 1,000W | 156.7 | 5,000 TOPS |

## Architecture Overview

### Pascal (P100)
- **Released:** 2016
- **Process Node:** 16nm TSMC
- **Key Features:** First GPU with HBM2 memory, unified memory architecture
- **Primary Use:** Double-precision HPC workloads

### Volta (V100)
- **Released:** 2017
- **Process Node:** 12nm TSMC
- **Key Features:** First generation Tensor Cores, mixed-precision training capability
- **Primary Use:** Deep learning training and inference

### Ampere (A100)
- **Released:** 2020
- **Process Node:** 7nm TSMC
- **Key Features:** 3rd generation Tensor Cores, TensorFloat-32 (TF32), Multi-Instance GPU (MIG)
- **Primary Use:** Large-scale AI training, data analytics

### Hopper (H100/H200)
- **Released:** 2022/2024
- **Process Node:** 4nm TSMC (H100), 4nm+ TSMC (H200)
- **Key Features:** 4th generation Tensor Cores, Transformer Engine, FP8 precision
- **Primary Use:** Large language model training and inference

### Blackwell (B100/B200)
- **Released:** 2025
- **Process Node:** 4nm TSMC (custom 4NP)
- **Key Features:** Dual-die chiplet design, 5th generation Tensor Cores, FP4/FP6 precision support
- **Primary Use:** Extreme-scale AI training, next-generation LLMs

## Memory Technology Evolution

| Generation | Memory Type | Capacity Range | Bandwidth Range | Bus Width |
|------------|-------------|----------------|-----------------|-----------|
| **Pascal** | HBM2 | 16 GB | 732 GB/s | 4096-bit |
| **Volta** | HBM2 | 16-32 GB | 900 GB/s | 4096-bit |
| **Ampere** | HBM2e | 40-80 GB | 1,555-2,039 GB/s | 5120-bit |
| **Hopper** | HBM3/HBM3e | 80-141 GB | 3,350-4,800 GB/s | 5120-bit |
| **Blackwell** | HBM3e | 192 GB | 8,000 GB/s | 8192-bit |

## AI Performance Evolution

### Training Performance (TF32/FP32)
- **V100:** 125 TFLOPS
- **A100:** 312 TFLOPS (2.5x improvement)
- **H100:** 1,000 TFLOPS (3.2x improvement)
- **B200:** 5,000 TFLOPS (5x improvement)

### Key Precision Support
- **V100:** FP16, FP32, FP64
- **A100:** FP16, BF16, TF32, FP32, FP64, INT8
- **H100:** FP8, FP16, BF16, TF32, FP32, FP64, INT8
- **B100/B200:** FP4, FP6, FP8, FP16, BF16, TF32, FP32, FP64, INT8

## Power and Packaging

### Package Types
- **P100/V100:** SXM2/SXM3 modules
- **A100:** SXM4 modules
- **H100/H200:** SXM5 modules
- **B100/B200:** SXM6 modules with dual-die chiplet design

### Power Efficiency Trends
- **Pascal to Volta:** 2.4x performance increase with 20% power increase
- **Volta to Ampere:** 2.5x performance increase with 33% power increase
- **Ampere to Hopper:** 3.2x performance increase with 75% power increase
- **Hopper to Blackwell:** 5x performance increase with 43% power increase (B200)

## Interconnect Technologies

### NVLink Evolution
- **P100:** No NVLink (PCIe only)
- **V100:** NVLink 2.0 (300 GB/s)
- **A100:** NVLink 3.0 (600 GB/s)
- **H100:** NVLink 4.0 (900 GB/s)
- **B100/B200:** NVLink 5.0 (1,800 GB/s) + NV-HBI chiplet interconnect (10 TB/s)

*Note: Specifications represent standard configurations. Some variants may have different memory capacities or clock speeds. AI performance figures are theoretical peak values using optimized precision formats with sparsity where applicable.*

# Additional info

**Evolutionary Milestones:**
- P100 was the world's first GPU architecture to support HBM2 memory technology
- V100 introduced the first Tensor Cores, providing up to 12x performance improvement for deep learning
- A100 features 3rd generation Tensor Cores with TensorFloat-32 (TF32) delivering up to 20x performance improvement over V100
- H100 features 4th generation Tensor Cores with FP8 precision providing up to 4x faster training over A100

**Blackwell Architecture Breakthroughs:**
- B100/B200 feature dual-die chiplet design with 208 billion transistors connected by 10 TB/s interconnect
- B200 delivers up to 20 PFLOPS FP8 performance, 2.5x improvement over Hopper
- 192GB HBM3e memory with 8 TB/s bandwidth - highest capacity and bandwidth in AI accelerator history

**Memory and Performance Evolution:**
- **Memory Capacity:** 16GB (P100) → 192GB (B200) - 12x increase
- **Memory Bandwidth:** 732 GB/s (P100) → 8,000 GB/s (B200) - 11x increase  
- **AI Performance:** 125 TOPS (V100) → 5,000 TOPS (B200) - 40x increase

**Power Scaling:**
- Efficient scaling from 250W (P100) to 1,000W (B200) while delivering exponential performance gains
- B100 maintains 700W TDP like H100/H200 while delivering significantly higher performance

This table shows how NVIDIA has systematically advanced each generation with architectural innovations, memory technology improvements, and specialized AI compute capabilities, culminating in the revolutionary dual-die Blackwell architecture that represents the current pinnacle of AI acceleration technology.

## Dies

The AI accelerator GPUs (P100, V100, A100, H100, B100/B200) use completely different chip dies than the RTX consumer/gaming GPUs, and this is why they have no RT (Ray Tracing) cores.

Here's the key distinction:

## Different Dies for Different Markets

**AI Accelerator Dies:**
- **Pascal:** GP100 (P100) - No RT cores, optimized for compute
- **Volta:** GV100 (V100) - No RT cores, first Tensor cores
- **Ampere:** GA100 (A100) - No RT cores, 3rd gen Tensor cores
- **Hopper:** GH100 (H100/H200) - No RT cores, 4th gen Tensor cores  
- **Blackwell:** GB100 (B100/B200) - No RT cores, 5th gen Tensor cores

**RTX Consumer Dies:**
- **Ampere:** GA102/GA104/GA106 (RTX 30 series) - Include RT cores
- **Ada Lovelace:** AD102/AD103/AD104 (RTX 40 series) - Include RT cores
- **Blackwell:** GB202/GB203/GB205 (RTX 50 series) - Include RT cores

## Why the Difference?

**AI Accelerators prioritize:**
- Maximum compute density for AI/ML workloads
- Large amounts of HBM memory
- High memory bandwidth
- Tensor cores for AI operations
- Double-precision (FP64) performance for HPC

**RTX GPUs prioritize:**
- Real-time ray tracing for gaming/graphics
- GDDR memory (cost-effective for gaming)
- RT cores for ray-triangle intersection
- Display outputs and video encoding
- Single-precision (FP32) gaming performance

The AI accelerators essentially sacrifice RT cores, display outputs, and consumer-oriented features to pack in more compute units, larger caches, and specialized AI hardware. This is why you'll never see RT cores on datacenter GPUs - that silicon area is better used for more CUDA cores, Tensor cores, and memory controllers for AI workloads.

These are fundamentally different chip architectures designed for completely different use cases, even when they share the same generation name (like Ampere or Blackwell).

## TensorCore innovations

## **1st Generation TensorCore (Volta - 2017)**

The first generation was introduced with the Volta architecture (V100). These TensorCores performed mixed-precision matrix multiplication with FP16 inputs and FP32 accumulation - specifically multiplying two 4×4 FP16 matrices and adding a third FP16 or FP32 matrix using fused multiply-add operations.

**Key Innovations:**
- Mixed-precision training using FP16 computation with FP32 accumulation, providing up to 12x higher peak teraFLOPS for training and 6x for inference over Pascal
- Up to 125 TFlops FP16 performance, offering a 16x multiple versus FP64 within the same power budget
- Foundation for automatic mixed precision (AMP) training

## **2nd Generation TensorCore (Turing - 2018)**

Turing introduced enhanced TensorCores adding INT8 and INT4 precision support beyond the original FP16, and supported a new warp-level synchronous MMA operation.

**Key Innovations:**
- Extended support to INT8, INT4, and INT1 (binary) precisions for inference acceleration
- Enabled Deep Learning Super Sampling (DLSS), marking NVIDIA's entry into AI-powered gaming graphics
- Improved programming model with warp-level synchronous matrix operations

## **3rd Generation TensorCore (Ampere - 2020)**

Ampere TensorCores extended computational capability to FP64, TF32, and bfloat16 precisions, providing 2x to 4x more throughput compared to the previous generation depending on workload.

**Key Innovations:**
- Brain Floating Point Format (BF16) became the de facto standard, providing FP32-level dynamic range at half storage cost and removing the need for loss scaling in mixed-precision training
- TF32 format providing up to 20x speedups without code changes, and Fine-Grained Structured Sparsity for more efficient inference
- Asynchronous data copy capability, allowing direct data transfer from global memory to shared memory
- Full warp (32 threads) participation in MMA operations, simplifying programming

## **4th Generation TensorCore (Hopper - 2022)**

Hopper introduced the Transformer Engine using FP8 precision to deliver 6x higher performance over FP16 for trillion-parameter model training.

**Key Innovations:**
- Tensor Memory Accelerator (TMEM) - 256KB of specialized memory for TensorCore operations with restricted access patterns to reduce hardware complexity
- Thread Block Cluster hierarchy allowing finer control between CTAs and the whole GPU, mapping to groups of SMs in the same GPC
- FP8 precision providing 6x performance improvement over FP16 for large model training while maintaining accuracy
- Warpgroup-level asynchronous MMA operations

## **5th Generation TensorCore (Blackwell - 2024)**

Blackwell introduced fifth-generation TensorCores with native support for sub-8-bit data types, including community-defined MXFP6 and MXFP4 microscaling formats.

**Key Innovations:**
- Convolution support in addition to general matrix multiplication, with weight stationary patterns and collector buffer for matrix B reuse
- MXFP (Microscaling) formats: FP4, FP6, and FP8 with micro-tensor scaling factors applied to fixed-length vectors
- Performance gains: 2x speedup per clock per SM for existing formats (FP16, BF16, FP8), 2x speedup for FP6 over Hopper's FP8, and 4x speedup for FP4 over Hopper's FP8
- Second-generation Transformer Engine with 2x attention-layer acceleration and 1.5x more AI compute FLOPS

## **6th Generation TensorCore (Blackwell Ultra - 2025)**

Blackwell Ultra represents the latest advancement with breakthrough NVFP4 precision format and enhanced energy efficiency.

**Key Innovations:**
- Enhanced GigaThread Engine for improved context switching and optimized workload distribution across 160 SMs
- Advanced Multi-Instance GPU (MIG) partitioning capabilities
- Ultra-efficient FP4 processing with improved accuracy through advanced quantization techniques

## **Overall Evolution Themes:**

The progression shows three main innovation vectors:

1. **Precision/Quantization**: From FP16→INT8/4→BF16/TF32→FP8→MXFP6/4→NVFP4, enabling higher throughput while maintaining accuracy
2. **Performance**: Each generation provides 2-6x improvements in specific workloads through architectural enhancements
3. **Energy Efficiency**: Blackwell delivers 25x energy efficiency uplift over Hopper while providing 30x real-time inference performance improvement

The evolution shows NVIDIA's systematic approach to accelerating AI workloads through specialized mixed-precision arithmetic, memory subsystem innovations, and programming model improvements tailored for transformer-based models and large-scale AI training/inference.