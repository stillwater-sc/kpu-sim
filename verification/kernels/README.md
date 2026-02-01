# Kernel Verification

**Axis:** Compute kernel correctness and optimization

This directory verifies individual compute kernels across seven classes
(Class 0-6), from elementwise operations through quantized inference.
The kernel taxonomy focuses on **performance engineering**: for each kernel
class, what are the energy, latency, and memory optimization concerns
specific to the KPU dataflow architecture?

See [TAXONOMY.md](TAXONOMY.md) for the full kernel classification,
per-class verification criteria, and implementation roadmap.

## Kernel Classes

| Class | Name | Defining Kernel | Optimization Focus |
|-------|------|----------------|-------------------|
| 0 | Elementwise | ReLU, GELU, Add, Exp | Streaming bandwidth, SFU utilization |
| 1 | Dense Linear | MatMul, Linear, MLP | Tiling, systolic utilization, data reuse |
| 2 | Spatial Convolution | Conv2D, Pool2D | im2col, spatial tiling, halo regions |
| 3 | Multi-Branch | Concat, Residual Add | Branch scheduling, credit partitioning |
| 4 | Depthwise Separable | Grouped Conv | Small-GEMM utilization, mixed boundedness |
| 5 | Attention | SDPA, MHA, LayerNorm | Flash tiling, online softmax, KV cache |
| 6 | Quantized | INT8/INT4 MatMul, Q/DQ | Type dispatch, accumulator management |

## Relationship to Other Verification Axes

- **Fidelity** (`../fidelity/`): Each kernel must be verified at all fidelity levels
- **DNN** (`../dnn/`): DNN classes are composed of kernel classes; a DNN class
  verification implicitly exercises all kernel classes that model requires
