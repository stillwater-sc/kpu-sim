# DNN Kernel Taxonomy for KPU Verification

**Date:** 2026-01-31
**Author:** Claude Code / Stillwater Engineering
**Status:** DRAFT - Awaiting Review

## 1. Motivation

As we progress from v0.8.0 toward production readiness, we need a systematic
framework for verifying DNN models of increasing complexity. The challenge is
threefold:

1. **Functional correctness** - The BEHAVIORAL simulator must produce bit-accurate
   results for every supported operation and their compositions.

2. **Performance fidelity** - The TRANSACTIONAL and TEMPORAL models must predict
   energy, latency, and memory traffic within calibrated error bounds.

3. **Kernel diversity** - Different DNN architectures exercise fundamentally
   different compute patterns, each with distinct performance engineering
   concerns on the KPU dataflow architecture.

A naive approach (verify models one at a time) misses the structure: a 3-layer
MLP and a 100-layer MLP exercise the same kernels. What matters is which
*kernel classes* a model introduces, because each class has unique implications
for the memory hierarchy, systolic array utilization, and dataflow scheduling.

## 2. Taxonomy Design Principles

### 2.1 Classification by Dominant Kernel

Models are classified by the **most performance-critical kernel class** they
introduce beyond what simpler classes already cover. Each class builds on all
previous classes - a Class 3 model uses Class 1 and 2 kernels plus its own.

### 2.2 Performance Engineering Dimensions

For each kernel class, three optimization dimensions are evaluated:

| Dimension | Question | KPU Concern |
|-----------|----------|-------------|
| **Energy** | How many operations per byte moved? | Arithmetic intensity, data reuse |
| **Latency** | What is the critical path? | Pipeline depth, tile scheduling |
| **Memory** | What is the working set? | Buffer sizing, credit flow, tiling |

### 2.3 Fidelity Progression

Each kernel class must be verified at all three fidelity levels before the
class is considered complete:

```
BEHAVIORAL    →  Functional correctness (bit-accurate vs reference)
TRANSACTIONAL →  Performance estimates (within 20% of temporal)
TEMPORAL      →  Cycle-accurate timing (within 10% of target hardware)
```

## 3. The Taxonomy

### Class 0: Elementwise and Activation Kernels

**Defining operations:** ReLU, GELU, SiLU, Sigmoid, Tanh, Add, Mul, Neg, Exp, Log, Sqrt

**Representative model:** None standalone (these are building blocks)

**Performance characteristics:**
- Memory-bound: O(N) compute, O(N) memory → arithmetic intensity = 1
- No data reuse opportunity
- Streaming access pattern (sequential, unit stride)
- SFU (Special Function Unit) utilization is the bottleneck for transcendentals

**KPU engineering concerns:**
- Energy: Dominated by DRAM-to-compute data movement, not ALU energy
- Latency: Single-pass streaming, limited by memory bandwidth
- Memory: No tiling needed; fits naturally in L1 stream buffers

**Verification criteria:**
- Bit-exact for ReLU, Add, Mul, Neg (integer-like operations)
- ULP-bounded for Sigmoid, Tanh, GELU, SiLU, Exp, Log, Sqrt (transcendentals)
- XUE: bandwidth utilization should approach peak for large tensors

---

### Class 1: Dense Linear (GEMM-Dominant)

**Defining operations:** MatMul, Linear (bias add), fused MatMul+Bias+Activation

**Representative models:**
- Multi-Layer Perceptron (MLP) - any depth
- Fully-connected classifier heads

**Performance characteristics:**
- Compute-bound for large matrices: O(MNK) compute, O(MK + KN + MN) memory
- Arithmetic intensity = 2MNK / (4(MK+KN+MN)) bytes → scales with matrix size
- High data reuse via tiling (each tile of A reused N/tile_N times)
- Systolic array utilization depends on tile alignment with array dimensions

**KPU engineering concerns:**
- Energy: Tile size selection determines L1/L2/L3 traffic and thus energy
  - Too small: underutilize systolic array, high tile overhead
  - Too large: exceed buffer capacity, stall on credits
- Latency: Dominated by matmul tiles on the critical path
  - Pipeline: DMA → L3 → L2 → L1 → Compute must be overlapped
  - Tile scheduling determines how much latency is hidden
- Memory: Working set = input tile + weight tile + output tile
  - Must fit in L1 stream buffers for compute to proceed
  - Weight reuse across batch dimension is critical for efficiency
  - Output-stationary, weight-stationary, or input-stationary tiling strategies

**Fusion opportunities:**
- MatMul + Bias + ReLU: eliminates intermediate tensor materialization
- MatMul + Bias + GELU/SiLU: same, with SFU activation
- Multi-layer fusion: output of layer N feeds directly to layer N+1 via L2

**Verification criteria:**
- Functional: max_diff < 1e-5 vs NumPy (FP32), accounting for FMA ordering
- XUE: matmul FLOP count must match 2*M*N*K exactly
- XUE: arithmetic intensity must match theoretical prediction
- Roofline: achieved GFLOPS within 10% of roofline prediction
- Tile efficiency: systolic array utilization > 80% for aligned dimensions

**Existing models in this class:**
| Model | Status | Gaps |
|-------|--------|------|
| MNIST MLP (3-layer) | BEHAVIORAL: PASS, TRANSACTIONAL: STATS | None |
| Minimal MLP (2-layer) | BEHAVIORAL: PASS, TRANSACTIONAL: STATS | Add ref check |
| XUE Validation MLP | BEHAVIORAL: PASS, TRANSACTIONAL: STATS | Add ref check |

---

### Class 2: Spatial Convolution

**Defining operations:** Conv2D, MaxPool2D, AvgPool2D, AdaptiveAvgPool2D

**Representative models:**
- LeNet-5 (simplest CNN)
- MNIST CNN (our current example)
- VGG-style networks (deep stacks of 3x3 convolutions)

**Performance characteristics:**
- Conv2D decomposes to GEMM via im2col: O(N * C_out * H_out * W_out * C_in * K_h * K_w)
- Spatial locality in input feature maps enables significant data reuse
- Pooling is memory-bound (stride > 1 reduces data volume)
- Channel dimension maps naturally to systolic array's M dimension

**KPU engineering concerns:**
- Energy: im2col duplication inflates memory traffic by K_h * K_w
  - Direct convolution avoids duplication but has irregular access patterns
  - Winograd transforms reduce multiplications at cost of additions and transforms
- Latency: Feature map tiling introduces 2D tile scheduling complexity
  - Halo regions between spatial tiles require overlapping data loads
  - Pooling stride reduces downstream compute but complicates tiling alignment
- Memory: Feature maps can be very large (batch * channels * H * W)
  - Intermediate activations between conv layers dominate memory
  - In-place pooling reduces peak memory but requires careful buffer management
  - Channel-last (NHWC) vs channel-first (NCHW) layout affects access patterns

**Fusion opportunities:**
- Conv2D + BatchNorm + ReLU: BN folds into conv weights at inference time
- Conv2D + ReLU: eliminates activation buffer
- Pooling + next Conv2D: stream pooled output directly to next layer's input

**Verification criteria:**
- Functional: max_diff < 1e-4 vs NumPy (FP32) for conv2d
- Functional: Exact match for MaxPool2D (no floating-point ambiguity)
- XUE: FLOP count = 2 * N * C_out * H_out * W_out * C_in * K_h * K_w
- XUE: Memory traffic must account for im2col expansion
- Tile efficiency: GEMM tile alignment after im2col

**Existing models in this class:**
| Model | Status | Gaps |
|-------|--------|------|
| MNIST CNN | BEHAVIORAL: PASS | No transactional timing for conv/pool |

---

### Class 3: Multi-Branch and Residual Architectures

**Defining operations:** Concat (channel), Element-wise Add (residual), Skip connections

**Representative models:**
- ResNet (residual add)
- Inception/GoogLeNet (multi-branch concat)
- SqueezeNet (Fire module: squeeze + parallel expand + concat)
- DenseNet (dense concatenation)

**Performance characteristics:**
- Residual add is elementwise (memory-bound, same as Class 0)
- Concat is a data movement operation (zero compute, pure memory traffic)
- The challenge is **scheduling**: parallel branches must be buffered simultaneously
- Skip connections create non-trivial dataflow graph dependencies

**KPU engineering concerns:**
- Energy: Concat forces materialization of both branches before combining
  - Residual add can be fused with the producing operation
  - Multi-branch networks have poor temporal locality (branch A data is cold
    while branch B executes)
- Latency: Critical path depends on which branch is longest
  - Parallel branch execution requires independent buffer pools
  - Credit-based flow must manage credits across branching dataflow graph
- Memory: Peak memory occurs at branch merge point
  - Both branches' outputs must be live simultaneously
  - SqueezeNet Fire module: squeeze output (small) + expand1x1 + expand3x3 all live
  - Memory planning must account for branch lifetimes

**New engineering challenges vs Class 2:**
- Buffer lifetime analysis across branches
- Credit partitioning: how to divide L3/L2/L1 credits among parallel branches
- Non-sequential DFX execution (branches can execute in parallel or interleaved)

**Verification criteria:**
- Functional: Concat must preserve element ordering across channels
- Functional: Residual add must handle matched shapes exactly
- XUE: Memory traffic must account for branch materialization
- Memory planner: Peak memory prediction must match actual peak
- Branch scheduling: no deadlock from circular credit dependencies

**Existing models in this class:**
| Model | Status | Gaps |
|-------|--------|------|
| SqueezeNet 1.0 | BEHAVIORAL: PASS (torch.compile) | Concat not in C++ backend; no native @kpu.compile path |
| SqueezeNet 1.1 | Defined | Not exercised |

---

### Class 4: Depthwise Separable and Grouped Convolution

**Defining operations:** Depthwise Conv2D (groups=C_in), Pointwise Conv2D (1x1), Grouped Conv2D

**Representative models:**
- MobileNetV1 (depthwise separable)
- MobileNetV2 (inverted residuals with depthwise separable)
- ShuffleNet (grouped conv + channel shuffle)
- EfficientNet (MBConv blocks)

**Performance characteristics:**
- Depthwise conv: O(N * C * H * W * K_h * K_w) — no cross-channel compute
  - Dramatically fewer FLOPs than standard conv (C_out * C_in → C factor)
  - Very low arithmetic intensity (memory-bound on most hardware)
  - Each channel is an independent small convolution
- Pointwise 1x1 conv: Equivalent to per-pixel matrix multiplication
  - Compute-bound (same as GEMM with M=H*W, K=C_in, N=C_out)
  - No spatial locality exploitation needed

**KPU engineering concerns:**
- Energy: Depthwise conv is memory-dominated
  - Per-channel independence means no weight reuse across channels
  - Small per-channel GEMM (K_h * K_w elements) underutilizes systolic array
  - Pointwise conv has good arithmetic intensity but different tiling
- Latency: Two-stage pipeline (depthwise → pointwise) requires careful overlap
  - Depthwise is fast but memory-bound; pointwise is slow but compute-bound
  - Imbalance creates pipeline bubbles if not carefully scheduled
  - Inverted residual: expand (1x1) → depthwise (3x3) → project (1x1)
    creates a three-stage pipeline with varying compute/memory balance
- Memory: Channel expansion in inverted residuals (6x expansion ratio)
  - Peak memory at expanded representation (e.g., 32 → 192 channels)
  - Must tile across batch and spatial dimensions to fit in buffers

**New engineering challenges vs Class 3:**
- Systolic array utilization for small per-channel GEMMs
- Mixed compute/memory-bound kernel scheduling
- Channel shuffle requires non-trivial data reorganization
- BatchNorm2d must be fused or eliminated for inference efficiency

**Verification criteria:**
- Functional: Depthwise conv must match grouped convolution reference
- Functional: Pointwise 1x1 conv must match matmul reference
- XUE: Separate FLOP tracking for depthwise vs pointwise stages
- XUE: Memory traffic breakdown showing depthwise is bandwidth-limited
- Systolic utilization: Expected to be low for depthwise, high for pointwise
- Energy model: Must capture the depthwise memory-boundedness accurately

**Existing models in this class:**
| Model | Status | Gaps |
|-------|--------|------|
| MobileNetV2 | Defined | Depthwise conv uses slow fallback; not exercised |

---

### Class 5: Attention and Sequence Models

**Defining operations:** Scaled Dot-Product Attention (SDPA), Multi-Head Attention (MHA),
LayerNorm, RMSNorm, Positional Encoding, Causal Masking

**Representative models:**
- Transformer encoder (BERT-style)
- Transformer decoder (GPT-style)
- Vision Transformer (ViT)
- Hybrid CNN-Transformer

**Performance characteristics:**
- SDPA: Q@K^T is O(S^2 * D) — quadratic in sequence length
- Softmax is applied row-wise: O(S^2) — memory-bound
- Attention @ V is O(S^2 * D) — another matmul
- LayerNorm: O(S * D) — reduction + elementwise, memory-bound
- Total per-layer: two large matmuls + softmax + normalization

**KPU engineering concerns:**
- Energy: The Q@K^T matrix is transient (used immediately for softmax+V multiply)
  - Flash Attention insight: tile Q@K^T to avoid materializing full S x S matrix
  - Online softmax (Milakov & Gimelshein) enables single-pass attention
  - Without flash attention: S^2 intermediate consumes O(S^2 * 4) bytes of buffer
- Latency: Two dependent matmuls with softmax in between
  - Softmax creates a serialization point (must complete before V multiply)
  - Multi-head: H independent attention computations (parallelism opportunity)
  - QKV projection and output projection are standard GEMMs
- Memory: Quadratic growth in sequence length
  - S=512: 1MB attention matrix per head per layer
  - S=2048: 16MB attention matrix per head per layer
  - KV cache for autoregressive decoding grows linearly with sequence length
  - Must tile attention computation to fit in L2/L3 buffers

**New engineering challenges vs Class 4:**
- Flash Attention tiling strategy for the KPU memory hierarchy
- Online softmax implementation (numerically stable streaming softmax)
- KV cache management for decoder inference
- Causal mask integration without branch divergence
- Multi-head parallelism across compute units
- LayerNorm/RMSNorm reduction across hidden dimension

**Verification criteria:**
- Functional: SDPA must match reference within 1e-4 (FP32)
- Functional: Causal mask must zero out correct positions
- Functional: Multi-head must produce same result as single-head loop
- XUE: FLOP count = 4 * B * H * S^2 * D_head (two matmuls per head)
- XUE: Memory traffic must reflect flash attention tiling (if implemented)
- Roofline: Attention should be compute-bound for large S*D products
- Softmax stability: No overflow/underflow for large logit ranges

**Existing models in this class:**
| Model | Status | Gaps |
|-------|--------|------|
| (None) | ops.py has SDPA and MHA | No model defined; no C++ attention kernel |

---

### Class 6: Quantized Inference

**Defining operations:** Quantize, Dequantize, INT8/INT4 MatMul, INT8/INT4 Conv2D,
Mixed-Precision Accumulation (INT8 compute → INT32 accumulate → FP32 output)

**Representative models:**
- Any Class 1-5 model quantized to INT8 or INT4
- Quantization-aware variants of MobileNet, ResNet, BERT

**Performance characteristics:**
- INT8: 4x throughput vs FP32 on systolic arrays, 4x bandwidth efficiency
- INT4: 8x throughput vs FP32, but accumulator overflow risk limits tile size
- Mixed precision: Requires type conversion at accumulator boundaries
- Quantization error propagates through layers (cumulative accuracy loss)

**KPU engineering concerns:**
- Energy: Dramatic reduction in both compute and memory energy
  - INT8 multiply: ~30x less energy than FP32 multiply
  - 4x less DRAM bandwidth per element
  - But: dequantize/requantize overhead at layer boundaries
- Latency: Higher throughput per systolic cycle
  - INT8 systolic: 4x MACs per cycle vs FP32
  - But: calibration quality affects whether quantized model is usable at all
  - Fallback to FP32 for sensitive layers (first/last) is common
- Memory: Weights compressed 4x (INT8) or 8x (INT4) vs FP32
  - Activations also compressed in fully-quantized pipelines
  - Scale/zero-point metadata overhead is small but non-zero
  - INT4 packing: two values per byte, requires unpack on compute path

**New engineering challenges vs Class 5:**
- Per-channel vs per-tensor quantization parameters
- Accumulator overflow detection for INT4 with large K dimensions
- Mixed-precision pipeline: FP32 → INT8 → INT32 accumulate → FP32 output
- Calibration dataset selection affects accuracy
- Q/DQ node placement optimization

**Verification criteria:**
- Functional: Quantized model accuracy within 1% of FP32 baseline
- Functional: Dequantize(Quantize(x)) round-trip error bounded by scale/2
- XUE: Throughput must reflect reduced precision (2-4x improvement)
- XUE: Memory traffic must reflect compressed data types
- Energy model: Must show INT8/INT4 energy savings vs FP32 baseline
- Calibration: MinMax, Percentile, MSE, Entropy methods must all produce valid scales

**Existing models in this class:**
| Model | Status | Gaps |
|-------|--------|------|
| (None end-to-end) | Q/DQ ops, calibration framework (v0.7) | No quantized model verification |

---

## 4. Taxonomy Summary

```
Class 0: Elementwise          ReLU, GELU, Add, Exp, ...
    │                         Memory-bound, streaming
    ▼
Class 1: Dense Linear         MatMul, Linear, MLP
    │                         Compute-bound, tiling, systolic utilization
    ▼
Class 2: Spatial Convolution  Conv2D, Pool2D
    │                         im2col, spatial tiling, halo regions
    ▼
Class 3: Multi-Branch         Concat, Residual Add, Skip connections
    │                         Branch scheduling, buffer lifetime, credit partitioning
    ▼
Class 4: Depthwise Separable  Grouped Conv, Depthwise+Pointwise
    │                         Mixed bound, small-GEMM utilization, channel shuffle
    ▼
Class 5: Attention            SDPA, MHA, LayerNorm, KV Cache
    │                         Quadratic memory, flash tiling, online softmax
    ▼
Class 6: Quantized Inference  INT8/INT4 MatMul, Q/DQ, Mixed Precision
                              Type dispatch, accumulator management, calibration
```

Each class is **cumulative**: a Class 4 model exercises kernels from Classes 0-3
as well as its own defining kernels. Verification of a higher class implicitly
re-validates all lower classes.

## 5. Model-to-Class Mapping

| Model | Class | Defining Kernels Used |
|-------|-------|-----------------------|
| MNIST MLP (3-layer) | 1 | MatMul, ReLU, Bias Add |
| Minimal MLP (2-layer) | 1 | MatMul, ReLU, Bias Add |
| XUE Validation MLP | 1 | MatMul, ReLU, Bias Add |
| MNIST CNN | 2 | Conv2D, MaxPool2D, MatMul |
| VGG-16 (not impl.) | 2 | Deep Conv2D stacks, large FC layers |
| SqueezeNet 1.0 | 3 | Fire module (Concat), Conv2D |
| SqueezeNet 1.1 | 3 | Fire module (Concat), Conv2D |
| ResNet-18 (not impl.) | 3 | Residual Add, BatchNorm |
| MobileNetV2 | 4 | Depthwise Conv, Pointwise Conv, InvertedResidual |
| EfficientNet-B0 (not impl.) | 4 | MBConv, Squeeze-Excite |
| BERT-base (not impl.) | 5 | SDPA, MHA, LayerNorm |
| GPT-2 (not impl.) | 5 | Causal SDPA, MHA, LayerNorm, KV Cache |
| ViT-B/16 (not impl.) | 5 | Patch Embed, SDPA, MHA, LayerNorm |
| Quantized MLP (not impl.) | 6 | INT8 MatMul, Q/DQ |
| Quantized MobileNet (not impl.) | 6 | INT8 Conv2D, INT8 Depthwise, Q/DQ |

## 6. Implementation and Verification Roadmap

### Phase 1: Complete Class 0 + Class 1 - CURRENT

**Goal:** Kernel-level verification harnesses for Class 0 (Elementwise) and Class 1 (Dense Linear).

| Task | Priority | Status | Files |
|------|----------|--------|-------|
| Create Class 0 elementwise verification harness | P0 | DONE | `verification/kernels/class0_elementwise/verify_elementwise.py` |
| Create Class 1 matmul verification harness | P0 | DONE | `verification/kernels/class1_dense_linear/verify_matmul.py` |
| Create Class 1 fused ops verification harness | P0 | DONE | `verification/kernels/class1_dense_linear/verify_fused_ops.py` |
| Add NumPy ref checks to minimal_mlp.py | P0 | TODO | `python/examples/minimal_mlp.py` |
| Add NumPy ref checks to xue_validation.py | P0 | TODO | `python/examples/xue_validation.py` |
| Verify fused ops (matmul+bias+relu/gelu/silu) | P0 | DONE | `kpu_native.cpp` |
| Verify TRANSACTIONAL roofline within 10% | P1 | Partial | `xue_validation.py` |
| Implement TEMPORAL matmul (systolic pipeline) | P2 | TODO | `temporal/compute/` |
| Verify TEMPORAL cycle count vs TRANSACTIONAL | P2 | TODO | New test |

**Acceptance criteria:**
- 3+ MLP configurations verified at BEHAVIORAL (max_diff < 1e-5)
- XUE FLOP counts match theoretical for all configurations
- TRANSACTIONAL roofline prediction within 10% of theoretical
- All results reproducible (deterministic, no uninitialized memory)

### Phase 2: Complete Class 2 (Spatial Convolution)

**Goal:** CNN models verified at BEHAVIORAL and TRANSACTIONAL.

| Task | Priority | Status | Files |
|------|----------|--------|-------|
| Wire Conv2D to transactional timing model | P0 | TODO | `transactional/compute/` |
| Wire Pool2D to transactional timing model | P0 | TODO | `transactional/compute/` |
| Add XUE event recording for Conv2D | P0 | TODO | `behavioral/compute/` |
| Add XUE event recording for Pool2D | P1 | TODO | `behavioral/compute/` |
| Conv2D+BN+ReLU fusion in C++ backend | P1 | TODO | `kpu_native.cpp` |
| Create VGG-16 example (deep conv stacks) | P2 | TODO | `python/examples/` |
| Create Class 2 verification harness | P1 | TODO | `verification/class2_spatial_conv/` |

**Acceptance criteria:**
- MNIST CNN verified at BEHAVIORAL (max_diff < 1e-4) and TRANSACTIONAL
- XUE reports conv FLOP count matching theoretical
- im2col memory overhead tracked in XUE memory hierarchy
- VGG-16 runs end-to-end with correct classification

### Phase 3: Complete Class 3 (Multi-Branch)

**Goal:** SqueezeNet verified at BEHAVIORAL and TRANSACTIONAL.

| Task | Priority | Status | Files |
|------|----------|--------|-------|
| Implement Concat in C++ backend | P0 | TODO | `kpu_native.cpp` |
| Port SqueezeNet from torch.compile to native | P0 | TODO | `python/examples/` |
| Add NumPy reference for SqueezeNet | P0 | TODO | New example |
| Implement memory lifetime analysis for branches | P1 | TODO | `memory_planner.py` |
| Add branch scheduling to DFX executor | P1 | TODO | `kpu_native.cpp` |
| Add ResNet-18 model and verification | P2 | TODO | `python/kpu/models/` |
| Create Class 3 verification harness | P1 | TODO | `verification/class3_multi_branch/` |

**Acceptance criteria:**
- SqueezeNet 1.0 verified at BEHAVIORAL against PyTorch (max_diff < 1e-3)
- SqueezeNet runs through native @kpu.compile (not torch.compile)
- XUE reports branch memory traffic correctly
- Memory planner tracks peak memory at branch merge points

### Phase 4: Complete Class 4 (Depthwise Separable)

**Goal:** MobileNetV2 verified at BEHAVIORAL and TRANSACTIONAL.

| Task | Priority | Status | Files |
|------|----------|--------|-------|
| Optimize depthwise conv in C++ (groups param) | P0 | TODO | `behavioral/compute/` |
| Add depthwise conv XUE events (separate category) | P0 | TODO | `xue/event_hierarchy.hpp` |
| Verify BatchNorm2d folding for inference | P1 | TODO | Model export utility |
| Create MobileNetV2 verification example | P0 | TODO | `python/examples/` |
| Add EfficientNet-B0 model | P2 | TODO | `python/kpu/models/` |
| Systolic utilization analysis for small GEMMs | P1 | TODO | XUE analysis |
| Create Class 4 verification harness | P1 | TODO | `verification/class4_depthwise/` |

**Acceptance criteria:**
- MobileNetV2 verified at BEHAVIORAL against PyTorch (max_diff < 1e-3)
- XUE shows depthwise stages are memory-bound, pointwise are compute-bound
- Systolic utilization reported separately for depthwise vs pointwise
- BatchNorm folded into conv weights at inference time

### Phase 5: Complete Class 5 (Attention)

**Goal:** Transformer model verified at BEHAVIORAL and TRANSACTIONAL.

| Task | Priority | Status | Files |
|------|----------|--------|-------|
| Implement SDPA in C++ behavioral fabric | P0 | TODO | `behavioral/compute/` |
| Implement LayerNorm in C++ behavioral fabric | P0 | Partial | Already exists |
| Implement RMSNorm in C++ behavioral fabric | P1 | TODO | `behavioral/compute/` |
| Add attention XUE events | P0 | TODO | `xue/event_hierarchy.hpp` |
| Flash Attention tiling strategy for KPU | P1 | TODO | Design doc |
| Create BERT-base or ViT-B/16 model | P0 | TODO | `python/kpu/models/` |
| KV cache management for decoder inference | P2 | TODO | Runtime support |
| Create Class 5 verification harness | P1 | TODO | `verification/class5_attention/` |

**Acceptance criteria:**
- Transformer model verified at BEHAVIORAL against PyTorch (max_diff < 1e-4)
- SDPA computes correct attention weights and output
- Causal masking verified for decoder models
- XUE tracks Q@K^T and Attn@V as separate matmul events
- Flash attention tiling design documented for KPU memory hierarchy

### Phase 6: Complete Class 6 (Quantized Inference)

**Goal:** Quantized models verified at BEHAVIORAL with accuracy validation.

| Task | Priority | Status | Files |
|------|----------|--------|-------|
| INT8 matmul kernel in C++ behavioral fabric | P0 | TODO | `behavioral/compute/` |
| INT8 conv2d kernel | P1 | TODO | `behavioral/compute/` |
| INT4 matmul with INT32 accumulation | P1 | TODO | `behavioral/compute/` |
| Q/DQ graph rewriting pass | P0 | Partial | `quantization/` |
| Quantize an MLP and verify accuracy | P0 | TODO | `python/examples/` |
| Quantize MobileNetV2 and verify accuracy | P1 | TODO | `python/examples/` |
| Mixed-precision XUE events | P1 | TODO | `xue/event_hierarchy.hpp` |
| Create Class 6 verification harness | P1 | TODO | `verification/class6_quantized/` |

**Acceptance criteria:**
- INT8 quantized MLP within 1% accuracy of FP32 baseline
- INT8 quantized MobileNetV2 within 2% accuracy of FP32 baseline
- XUE reports INT8 throughput improvement vs FP32
- XUE memory traffic reflects compressed data types
- Calibration methods (MinMax, Percentile, MSE, Entropy) all produce valid scales

## 7. Verification Harness Design

Each class directory under `verification/` will contain:

```
verification/
├── README.md                          # This file
├── TAXONOMY.md                        # This taxonomy document
├── class1_dense_linear/
│   ├── README.md                      # Class-specific verification criteria
│   ├── verify_mlp.py                  # Parameterized MLP verification
│   ├── verify_fused_ops.py            # Fused op correctness
│   ├── golden_references/             # Saved reference outputs
│   └── results/                       # Verification run results
├── class2_spatial_conv/
│   ├── README.md
│   ├── verify_cnn.py
│   ├── verify_conv2d_variants.py      # Stride, padding, dilation
│   ├── verify_pooling.py
│   └── ...
├── class3_multi_branch/
│   ├── verify_squeezenet.py
│   ├── verify_residual.py
│   └── ...
├── class4_depthwise/
│   ├── verify_mobilenet.py
│   ├── verify_depthwise_conv.py
│   └── ...
├── class5_attention/
│   ├── verify_transformer.py
│   ├── verify_sdpa.py
│   ├── verify_flash_attention.py
│   └── ...
└── class6_quantized/
    ├── verify_int8_mlp.py
    ├── verify_int4_matmul.py
    ├── verify_calibration.py
    └── ...
```

Each `verify_*.py` script will:
1. Construct the model with known weights (deterministic initialization)
2. Run through all available fidelity levels
3. Compare against golden reference (NumPy or PyTorch)
4. Collect and validate XUE performance statistics
5. Report PASS/FAIL with detailed diagnostics
6. Output structured JSON results for CI integration

## 8. Success Metrics

| Metric | Target | Measured By |
|--------|--------|-------------|
| BEHAVIORAL accuracy | max_diff < class tolerance | verify_*.py scripts |
| TRANSACTIONAL prediction | within 20% of TEMPORAL | Cross-fidelity comparison |
| TEMPORAL accuracy | within 10% of target hardware | Hardware validation (future) |
| XUE FLOP accuracy | exact match (integer count) | FLOP audit in verify scripts |
| XUE memory traffic | within 5% of analytical model | Memory traffic analysis |
| Systolic utilization | > 80% for compute-bound kernels | XUE compute breakdown |
| Roofline prediction | within 10% of achieved | Roofline validation |
| Regression freedom | 0 regressions across classes | CI/CD verification suite |

## 9. Open Questions for Review

1. **Class granularity:** Should depthwise separable (Class 4) be merged with
   spatial convolution (Class 2), since depthwise is technically a special case
   of grouped convolution? We kept it separate because the performance
   characteristics are fundamentally different (memory-bound vs compute-bound).

2. **Sparse and structured pruning:** Should we add a Class 7 for sparse
   inference (structured pruning, N:M sparsity)? This requires different
   systolic array utilization analysis.

3. **Recurrent networks:** LSTM/GRU are declining in relevance but have unique
   sequential dependency challenges. Include as a class or skip?

4. **Generative models:** Autoregressive decoding (GPT-style) has fundamentally
   different performance characteristics than encoder inference (BERT-style).
   Should decoder-specific concerns (KV cache, speculative decoding) be a
   separate class?

5. **Multi-chip scaling:** When we move to multi-KPU configurations, NoC and
   inter-chip communication become first-order concerns. Should this be a
   Class 7 (distributed inference)?

6. **Verification CI integration:** Should verification harnesses run on every
   commit (slow), or only on release branches? Suggested: Class 1-2 on every
   commit, Class 3+ on nightly/release.
