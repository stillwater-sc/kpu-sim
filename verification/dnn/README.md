# DNN Model Classification and Verification

**Axis:** Neural network architecture capability progression

This directory verifies that the KPU simulator can correctly execute and
accurately model the performance of progressively more complex DNN architectures.
Models are classified into four classes based on their **dominant computational
pattern** and the **optimization challenge** they present to the KPU hardware.

## DNN Classification

| Class | Kernel Characteristics | Optimization Focus | Example Models |
|-------|----------------------|-------------------|----------------|
| **I** | Heavy GEMM, high arithmetic intensity | Maximizing MAC utilization, roofline optimization | ResNet-50, VGG-16 |
| **II** | Depthwise separable convolutions, high data-to-computation ratio | Minimizing memory bandwidth bottlenecks | MobileNetV2, EfficientNet |
| **III** | Dynamic indexing, softmax, large-scale KV caching | Latency reduction in non-linear ops, memory tiling | ViT, GPT-series, Llama |
| **IV** | Graph traversals, unstructured pruning, scatter/gather ops | Managing irregular memory access patterns | GCN, PointNet |

## Class I: Compute-Bound (GEMM-Dominant)

### Characteristics

Class I models are dominated by large, dense matrix multiplications where
the systolic array's peak throughput is the primary performance limiter.
These models have high arithmetic intensity (FLOPs per byte of memory traffic),
meaning compute time dominates over data movement time for properly tiled
implementations.

**Defining compute patterns:**
- Large GEMM operations (M, N, K >> systolic array dimensions)
- Standard 2D convolutions with many channels (reduces to large GEMM via im2col)
- Batch normalization (folds into preceding convolution at inference time)
- Residual connections (elementwise add, negligible cost relative to GEMM)
- Pooling (spatial reduction, memory-bound but small fraction of total time)

**KPU optimization targets:**
- **MAC utilization:** What fraction of systolic array MACs are performing
  useful work? Targets > 80% for well-tiled workloads.
- **Roofline position:** These models should operate on the compute-bound
  (flat) portion of the roofline, not the memory-bound (sloped) portion.
- **Tile scheduling:** Overlap DMA → L3 → L2 → L1 → Compute pipeline stages
  to hide data movement latency behind compute latency.
- **Weight reuse:** Exploit batch dimension to amortize weight loading cost
  across multiple input samples.

### Representative Models

| Model | Params | Key Ops | FLOPs (224x224) | Status |
|-------|--------|---------|-----------------|--------|
| **VGG-16** | 138M | Conv3x3 (13 layers), FC (3 layers) | 15.5G | **VERIFIED (behavioral, native @kpu.compile)** |
| **ResNet-18** | 11.7M | Conv3x3/7x7, BatchNorm, ResidualAdd | 1.8G | Not implemented |
| **ResNet-50** | 25.6M | Conv1x1/3x3, Bottleneck blocks, BatchNorm | 4.1G | Not implemented |
| **MNIST MLP** | 169K | MatMul (3 layers) | 6.5M | **VERIFIED** |
| **MNIST CNN** | 43K | Conv3x3 (2), MaxPool, FC (2) | ~2M | **VERIFIED (behavioral)** |
| **SqueezeNet 1.0** | 1.2M | Fire modules (Conv1x1, Conv3x3, Concat) | 830M | **VERIFIED (behavioral, native @kpu.compile)** |

### Verification Criteria

**Functional (BEHAVIORAL):**
- MatMul: max_diff < 1e-5 vs NumPy
- Conv2D: max_diff < 1e-4 vs NumPy
- End-to-end model: max_diff < 1e-3 vs PyTorch
- BatchNorm folding: max_diff < 1e-5 vs unfolded

**Performance (TRANSACTIONAL/TEMPORAL):**
- XUE FLOP count: exact match with analytical model
- MAC utilization: > 80% for GEMM-dominant layers
- Roofline: achieved GFLOPS within 10% of compute roof
- Arithmetic intensity: within 5% of theoretical

**Milestone models:**
1. VGG-16 — pure Conv+FC stack, baseline for Class I
2. ResNet-50 — adds residual connections and bottleneck blocks

### Implementation Roadmap

| Phase | Model | Kernels Required | Status |
|-------|-------|-----------------|--------|
| I.1 | MNIST MLP (3-layer) | MatMul, ReLU, Add | **DONE** |
| I.2 | MNIST CNN | Conv2D, MaxPool2D, MatMul | **DONE (behavioral)** |
| I.3 | SqueezeNet 1.0 (native) | Conv2D, Concat, MaxPool, AdaptiveAvgPool | **DONE (behavioral)** |
| I.4 | VGG-16 | Deep Conv2D stacks, large FC layers | **DONE (behavioral)** |
| I.5 | ResNet-50 | Bottleneck blocks, BatchNorm fusion, ResidualAdd | TODO |

---

## Class II: Depthwise/Spatial (Bandwidth-Bound)

### Characteristics

Class II models replace standard convolutions with **depthwise separable
convolutions** to dramatically reduce FLOPs, but this shifts the bottleneck
from compute to memory bandwidth. The core challenge is that depthwise
convolutions have very low arithmetic intensity — each channel is convolved
independently, providing no cross-channel data reuse.

**Defining compute patterns:**
- Depthwise convolution: each input channel convolved with its own filter
  (groups = C_in), producing one output channel per input channel
- Pointwise convolution: 1x1 convolution that mixes channels (standard GEMM)
- Inverted residual: expand channels (1x1) → depthwise (3x3) → project (1x1)
- Squeeze-and-excitation: global average pool → FC → sigmoid → channel scaling
- Linear bottleneck: no activation on the projection (narrow) layer

**KPU optimization targets:**
- **Memory bandwidth utilization:** Depthwise convolutions should saturate
  DRAM bandwidth since they are memory-bound. The question is whether the
  DMA → L3 → L2 → L1 pipeline can sustain peak bandwidth.
- **Small-GEMM efficiency:** Depthwise conv tiles are tiny (K_h * K_w per channel).
  The systolic array is severely underutilized unless many channels are packed
  into a single tile operation.
- **Pipeline balancing:** Depthwise (fast, memory-bound) followed by pointwise
  (slow, compute-bound) creates a producer-consumer imbalance. Must overlap
  depthwise output streaming with pointwise consumption.
- **Channel expansion memory:** Inverted residuals with expansion ratio 6 create
  6x more intermediate data. Buffer management must handle this peak.

### Representative Models

| Model | Params | Key Ops | FLOPs (224x224) | Status |
|-------|--------|---------|-----------------|--------|
| **MobileNetV2** | 3.5M | Depthwise3x3, Pointwise1x1, InvertedResidual | 300M | Defined, not exercised |
| **MobileNetV3-Small** | 2.5M | + Squeeze-Excite, H-Swish | 56M | Not implemented |
| **EfficientNet-B0** | 5.3M | MBConv (depthwise + SE + residual) | 390M | Not implemented |
| **ShuffleNetV2** | 2.3M | Channel split, channel shuffle | 146M | Not implemented |

### Verification Criteria

**Functional (BEHAVIORAL):**
- Depthwise conv: max_diff < 1e-4 vs PyTorch grouped convolution
- Pointwise 1x1: max_diff < 1e-5 (equivalent to matmul)
- Inverted residual block: max_diff < 1e-4 end-to-end
- BatchNorm folding into depthwise conv: max_diff < 1e-5

**Performance (TRANSACTIONAL/TEMPORAL):**
- XUE must distinguish depthwise vs pointwise events
- Depthwise: memory bandwidth utilization > 70% of peak
- Pointwise: MAC utilization > 70%
- Energy model: depthwise energy dominated by data movement, not compute
- Pipeline utilization: < 20% idle time between depthwise and pointwise stages

**Milestone models:**
1. MobileNetV2 — canonical depthwise separable architecture
2. EfficientNet-B0 — adds compound scaling and squeeze-excitation

### Implementation Roadmap

| Phase | Model | Kernels Required | Status |
|-------|-------|-----------------|--------|
| II.1 | Depthwise Conv2D kernel | Grouped convolution (groups=C_in) | Fallback exists, needs optimization |
| II.2 | MobileNetV2 (behavioral) | Depthwise + Pointwise + InvertedResidual + BN | TODO |
| II.3 | MobileNetV2 (transactional) | XUE events for depthwise/pointwise breakdown | TODO |
| II.4 | EfficientNet-B0 | + Squeeze-Excite block, compound scaling | TODO |

---

## Class III: Sequential/Attention (Latency-Sensitive)

### Characteristics

Class III models are defined by the **attention mechanism**: scaled dot-product
attention (SDPA) where Q@K^T produces a quadratic intermediate matrix that
must be softmaxed before multiplying by V. The key challenge is managing
this quadratic memory growth and the non-linear softmax operation that
creates a serialization point between two dependent GEMMs.

Additionally, **autoregressive inference** (GPT-style) processes tokens
sequentially, making per-token latency the critical metric rather than
throughput. This requires KV caching and efficient single-token inference.

**Defining compute patterns:**
- QKV projection: three parallel large GEMMs (standard, Class I-like)
- Attention scores: Q @ K^T — O(B * H * S^2 * D_head), quadratic in S
- Softmax: row-wise normalization of S x S attention matrix
- Attention output: Scores @ V — O(B * H * S^2 * D_head)
- Output projection: standard GEMM
- LayerNorm / RMSNorm: reduction + elementwise, memory-bound
- Feed-forward: two large GEMMs with activation (GELU/SiLU) between them
- KV cache: store and retrieve key/value tensors from previous tokens
- Causal masking: upper-triangular mask preventing future token attention

**KPU optimization targets:**
- **Flash Attention tiling:** Tile the Q@K^T computation so the full S x S
  matrix is never materialized. Requires online softmax (Milakov & Gimelshein)
  where softmax statistics are accumulated incrementally as tiles are processed.
- **Softmax latency:** Softmax is the serialization point. Must minimize latency
  of the reduction (max) + subtract + exp + reduction (sum) + divide pipeline.
  On the KPU, this maps to SFU + reduction tree.
- **KV cache bandwidth:** For autoregressive decoding, each token reads the
  entire KV cache. With long sequences, this becomes bandwidth-limited.
  KV cache should reside in L3 with efficient streaming to L2/L1.
- **Multi-head parallelism:** H independent attention heads can execute in
  parallel across compute units, but each head has its own Q, K, V slices
  requiring separate data paths.
- **Token-level latency:** For serving, time-to-first-token (TTFT) and
  inter-token latency (ITL) are the critical metrics, not batch throughput.

### Representative Models

| Model | Params | Key Ops | Context Length | Status |
|-------|--------|---------|---------------|--------|
| **ViT-B/16** | 86M | Patch embed, 12x (MHA + FFN + LN) | 197 tokens | Not implemented |
| **BERT-base** | 110M | 12x (MHA + FFN + LN), encoder only | 512 tokens | Not implemented |
| **GPT-2 (124M)** | 124M | 12x (causal MHA + FFN + LN), decoder | 1024 tokens | Not implemented |
| **Llama-2 7B** | 7B | 32x (GQA + SwiGLU FFN + RMSNorm) | 4096 tokens | Not implemented |

### Verification Criteria

**Functional (BEHAVIORAL):**
- SDPA: max_diff < 1e-4 vs PyTorch `F.scaled_dot_product_attention`
- Causal mask: positions above diagonal must be exactly zero after masking
- Multi-head: equivalent to single-head loop (permutation invariant)
- LayerNorm: max_diff < 1e-5 vs PyTorch
- RMSNorm: max_diff < 1e-5 vs reference
- Full transformer layer: max_diff < 1e-3 vs PyTorch

**Performance (TRANSACTIONAL/TEMPORAL):**
- XUE: Q@K^T and Scores@V tracked as separate matmul events
- XUE: Softmax tracked as separate non-linear event
- Flash attention: memory traffic must NOT include full S x S materialization
- KV cache: memory traffic for incremental decoding must scale as O(S * D), not O(S^2)
- Multi-head: utilization across heads should be balanced (< 10% variance)
- Roofline: QKV projections and FFN should be compute-bound;
  softmax and LayerNorm should be memory-bound

**Milestone models:**
1. ViT-B/16 — encoder-only attention, fixed-length, no KV cache
2. GPT-2 (124M) — decoder with causal mask, KV cache, autoregressive

### Implementation Roadmap

| Phase | Model | Kernels Required | Status |
|-------|-------|-----------------|--------|
| III.1 | SDPA kernel (C++) | MatMul + Softmax + MatMul composition | ops.py only (Python) |
| III.2 | LayerNorm / RMSNorm (C++) | Reduction + Elementwise | LayerNorm exists, RMSNorm TODO |
| III.3 | ViT-B/16 (behavioral) | Patch embed + 12x (MHA + FFN + LN) | TODO |
| III.4 | ViT-B/16 (transactional) | Attention XUE events, flash tiling | TODO |
| III.5 | GPT-2 (behavioral) | Causal mask, KV cache management | TODO |
| III.6 | GPT-2 (transactional) | Autoregressive latency model | TODO |

---

## Class IV: Sparse/Irregular (Access-Pattern-Bound)

### Characteristics

Class IV models operate on **non-Euclidean data structures** — graphs,
point clouds, meshes — where the connectivity is not a regular grid.
The core challenge is that adjacency patterns determine memory access
patterns, and these patterns are irregular, data-dependent, and often
not known until runtime. This defeats the prefetching and tiling
strategies that work well for Classes I-III.

**Defining compute patterns:**
- Sparse matrix-vector multiply (SpMV): A_sparse @ x_dense
- Scatter/gather: write to / read from non-contiguous memory locations
  based on index tensors (edge lists, neighbor lists)
- Message passing: aggregate features from neighbor nodes
  (sum, mean, max over variable-size neighborhoods)
- Unstructured pruning: irregular zero patterns in weight matrices
  that break systolic array alignment
- Dynamic graph construction: k-NN search or radius-based neighbor
  finding at runtime
- Variable-length reduction: each node aggregates over a different
  number of neighbors (ragged tensors)

**KPU optimization targets:**
- **Scatter/gather bandwidth:** Irregular access patterns cause DRAM page
  misses and bank conflicts. The DMA engine must handle non-contiguous
  transfers efficiently. Tag CAM matching in the dataflow pipeline must
  handle out-of-order tile arrivals.
- **Sparse GEMM utilization:** Unstructured sparsity means many zeros in
  the systolic array input. N:M structured sparsity (e.g., 2:4) enables
  hardware-level skipping. Without structure, utilization can be < 50%.
- **Dynamic workload balancing:** Different nodes have different
  neighborhood sizes, creating load imbalance across compute units.
  Work partitioning must be data-dependent.
- **Memory indirection:** Gather operations require reading an index
  tensor, then using those indices to fetch data. This creates a
  two-stage memory access pattern that is hard to pipeline.

### Representative Models

| Model | Key Ops | Data Domain | Status |
|-------|---------|-------------|--------|
| **GCN** (Graph Convolutional Network) | SpMV, message passing | Graphs | Not implemented |
| **GAT** (Graph Attention Network) | Attention on edges, SpMV | Graphs | Not implemented |
| **PointNet** | Per-point MLP, global max pool | Point clouds | Not implemented |
| **PointNet++** | Farthest point sampling, ball query, MLP | Point clouds | Not implemented |
| **Sparse Transformer** | Sparse attention patterns | Sequences | Not implemented |

### Verification Criteria

**Functional (BEHAVIORAL):**
- SpMV: max_diff < 1e-5 vs SciPy sparse matrix multiply
- Scatter/gather: exact index correctness (no off-by-one)
- Message passing: aggregation matches reference for all reduction types
- Irregular neighborhood: correct handling of variable-size neighbor lists

**Performance (TRANSACTIONAL/TEMPORAL):**
- XUE: Memory access pattern classification (sequential vs random vs strided)
- XUE: DRAM page hit/miss ratio for scatter/gather workloads
- XUE: Effective bandwidth (useful bytes / total bytes transferred)
- Bank conflict rate: quantify DRAM bank conflict overhead
- Systolic utilization: expected < 50% for unstructured sparse, > 70% for N:M structured
- Load balance: variance in per-compute-unit work across graph partitions

**Milestone models:**
1. GCN on Cora dataset — simplest graph neural network
2. PointNet on ModelNet40 — point cloud classification

### Implementation Roadmap

| Phase | Model | Kernels Required | Status |
|-------|-------|-----------------|--------|
| IV.1 | Sparse tensor support | CSR/COO format, SpMV kernel | Not implemented |
| IV.2 | Scatter/gather ops | Index-based memory access | Not implemented |
| IV.3 | GCN (behavioral) | SpMV + message passing + MLP | TODO |
| IV.4 | PointNet (behavioral) | Per-point MLP + global max pool | TODO |
| IV.5 | Structured sparsity (N:M) | 2:4 sparse systolic array | TODO |

---

## Cross-Reference: DNN Class vs Kernel Class

Each DNN class draws on specific kernel classes from `../kernels/TAXONOMY.md`:

| DNN Class | Kernel Classes Required |
|-----------|----------------------|
| **I: Compute-Bound** | K0 (Elementwise), K1 (Dense Linear), K2 (Spatial Conv), K3 (Multi-Branch) |
| **II: Depthwise/Spatial** | K0-K3 + K4 (Depthwise Separable) |
| **III: Sequential/Attention** | K0-K1 + K5 (Attention) |
| **IV: Sparse/Irregular** | K0-K1 + new sparse kernels (not yet in kernel taxonomy) |

Note: DNN Class III (Attention) does not require K2-K4 (convolution kernels)
for pure transformer models, but hybrid models (e.g., ConvNeXt, CoAtNet)
would require both convolution and attention kernel classes.

## Cross-Reference: DNN Class vs Fidelity

Each DNN class must eventually be verified at all three fidelity levels.
Current status:

| DNN Class | BEHAVIORAL | TRANSACTIONAL | TEMPORAL |
|-----------|-----------|---------------|----------|
| **I: Compute-Bound** | MLP verified, CNN verified, SqueezeNet verified | MLP roofline verified | Not started |
| **II: Depthwise/Spatial** | MobileNetV2 defined, not exercised | Not started | Not started |
| **III: Sequential/Attention** | Ops exist in Python, no model | Not started | Not started |
| **IV: Sparse/Irregular** | Nothing implemented | Not started | Not started |

## Integrated Verification Roadmap

The three verification axes (DNN, Kernel, Fidelity) define a 3D matrix.
We fill this matrix by progressing along the DNN axis while simultaneously
deepening fidelity support:

```
                         FIDELITY AXIS
                    BEHAV   TRANS   TEMPORAL
                   ┌───────┬───────┬───────┐
        Class I    │██████ │███░░░ │░░░░░░ │  ← Current focus
DNN     Class II   │██░░░░ │░░░░░░ │░░░░░░ │
AXIS    Class III  │░░░░░░ │░░░░░░ │░░░░░░ │
        Class IV   │░░░░░░ │░░░░░░ │░░░░░░ │
                   └───────┴───────┴───────┘

        ██ = Verified    ███ = Partial    ░░░ = Not started
```

**Phase 1 (v0.8.x):** Complete DNN Class I at BEHAVIORAL and TRANSACTIONAL
- Verify VGG-16 and ResNet-50 end-to-end
- Validate XUE roofline predictions for compute-bound workloads

**Phase 2 (v0.9.x):** Add DNN Class II at BEHAVIORAL; deepen Class I to TEMPORAL
- Verify MobileNetV2 and EfficientNet-B0
- First TEMPORAL validation for MatMul and Conv2D kernels
- Cross-validate TRANSACTIONAL predictions against TEMPORAL

**Phase 3 (v0.10.x):** Add DNN Class III at BEHAVIORAL; deepen Class II to TRANSACTIONAL
- Verify ViT-B/16 and GPT-2
- Flash attention tiling design for KPU
- Class II memory bandwidth validation

**Phase 4 (v1.0.x):** Full matrix coverage for Classes I-III
- All three classes verified at all three fidelity levels
- Performance predictions within calibrated error bounds
- Class IV as stretch goal (sparse kernels are architecturally distinct)
