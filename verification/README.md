# KPU Simulator Verification Suite

**Date:** 2026-01-31
**Version:** v0.8.0
**Status:** Active Development

## 1. Purpose

This directory contains the verification infrastructure for the KPU simulator.
Verification is organized along **three orthogonal axes**, each capturing a
different dimension of correctness and accuracy.

## 2. The Three Verification Axes

```
                              FIDELITY
                         (How accurate is
                          the simulation?)
                               │
                               │
                    BEHAVIORAL │ TRANSACTIONAL │ TEMPORAL
                               │
                ┌──────────────┼──────────────┐
                │              │              │
    KERNELS ────┤   The 3D     │              ├──── DNN CLASSES
   (What ops    │  Verification│              │    (What networks
    are correct │   Matrix     │              │     can we run?)
    and fast?)  │              │              │
                └──────────────┼──────────────┘
                               │
```

### Axis 1: Kernels (`kernels/`)

**Question:** Are individual compute operations correct and optimized?

Classifies compute kernels from simple (elementwise) to complex (quantized
inference) across 7 classes (K0-K6). Each class introduces new performance
engineering challenges for energy, latency, and memory optimization on the
KPU dataflow architecture.

| Kernel Class | Defining Operations | Optimization Focus |
|-------------|--------------------|--------------------|
| K0: Elementwise | ReLU, GELU, Add, Exp | Streaming bandwidth, SFU |
| K1: Dense Linear | MatMul, Linear | Tiling, systolic utilization |
| K2: Spatial Conv | Conv2D, Pool2D | im2col, spatial tiling |
| K3: Multi-Branch | Concat, Residual Add | Credit partitioning, scheduling |
| K4: Depthwise | Grouped Conv | Small-GEMM, mixed boundedness |
| K5: Attention | SDPA, MHA, LayerNorm | Flash tiling, online softmax |
| K6: Quantized | INT8/INT4, Q/DQ | Type dispatch, accumulators |

### Axis 2: Fidelity (`fidelity/`)

**Question:** How accurately does the simulation predict hardware behavior?

Verifies that the three simulation tiers produce consistent results and
that lower-cost tiers predict higher-cost tiers within calibrated bounds.

| Fidelity | What It Computes | Verification Target |
|----------|-----------------|-------------------|
| BEHAVIORAL | Actual tensor values | Bit-accurate vs reference |
| TRANSACTIONAL | Statistical timing | Within 20% of TEMPORAL |
| TEMPORAL | Cycle-accurate timing | Within 10% of hardware |

### Axis 3: DNN Classes (`dnn/`)

**Question:** What classes of neural networks can we execute and model?

Classifies DNN architectures by their **dominant computational pattern**
and the resulting hardware optimization challenge.

| DNN Class | Kernel Characteristics | Optimization Focus | Examples |
|-----------|----------------------|-------------------|----------|
| I: Compute-Bound | Heavy GEMM, high arithmetic intensity | MAC utilization, roofline | ResNet-50, VGG-16 |
| II: Depthwise/Spatial | Depthwise separable, high data/compute ratio | Memory bandwidth bottlenecks | MobileNetV2, EfficientNet |
| III: Sequential/Attention | Dynamic indexing, softmax, KV caching | Non-linear op latency, memory tiling | ViT, GPT, Llama |
| IV: Sparse/Irregular | Graph traversals, scatter/gather | Irregular memory access patterns | GCN, PointNet |

## 3. Current Status

### Overall Verification Matrix

```
                         FIDELITY AXIS
                    BEHAV   TRANS   TEMPORAL
                   ┌───────┬───────┬───────┐
  DNN   Class I    │██████ │███░░░ │░░░░░░ │  ← Current focus
  AXIS  Class II   │██░░░░ │░░░░░░ │░░░░░░ │
        Class III  │░░░░░░ │░░░░░░ │░░░░░░ │
        Class IV   │░░░░░░ │░░░░░░ │░░░░░░ │
                   └───────┴───────┴───────┘

        ██ = Verified    ███ = Partial    ░░░ = Not started
```

### Model Status (2026-01-31)

| Model | DNN Class | BEHAVIORAL | TRANSACTIONAL | TEMPORAL |
|-------|-----------|------------|---------------|----------|
| MNIST MLP (3-layer) | I | PASS (tol=0) | STATS | N/A |
| Minimal MLP (2-layer) | I | PASS (no ref) | STATS | N/A |
| MNIST CNN | I | PASS (tol=1e-4) | N/A | N/A |
| SqueezeNet 1.0 | I | Partial (torch.compile) | N/A | N/A |
| MobileNetV2 | II | Defined, not exercised | N/A | N/A |
| (Transformer) | III | Ops exist, no model | N/A | N/A |
| (GCN/PointNet) | IV | Nothing implemented | N/A | N/A |

### Known Issues

1. **Fused op temp buffers** (FIXED 2026-01-31): Uninitialized memory caused
   NaN propagation in `fused_matmul_bias_relu/gelu/silu`. Fixed by zero-init.

2. **Conv2D transactional timing**: Conv/pool not wired to timing model.

3. **Broadcasting in C++ add**: Falls back to NumPy for mismatched shapes.

4. **Concat not in C++ backend**: SqueezeNet Fire module requires Python path.

5. **Depthwise convolution**: Grouped conv uses slow loop-based fallback.

## 4. Directory Structure

```
verification/
├── README.md                  # This file — overview of all three axes
├── kernels/
│   ├── README.md              # Kernel verification overview
│   └── TAXONOMY.md            # Full kernel taxonomy (K0-K6)
├── fidelity/
│   └── README.md              # Fidelity verification strategy
└── dnn/
    └── README.md              # DNN class taxonomy (I-IV)
```

## 5. Running Verification

```bash
# BEHAVIORAL correctness — DNN Class I
python python/examples/mnist_mlp.py        # MLP (PASS)
python python/examples/mnist_cnn.py        # CNN (PASS)

# TRANSACTIONAL performance — DNN Class I
python python/examples/xue_validation.py   # XUE roofline analysis

# C++ unit tests
ctest --preset default -R xue              # XUE tests (3/3 pass)

# Python unit tests
cd python && ~/.local/bin/pytest tests/ -v
```
