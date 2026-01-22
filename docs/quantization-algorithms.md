# Quantization and Calibration Algorithms

This document describes the quantization and calibration algorithms implemented in the KPU simulator's Python API.

## Table of Contents

1. [Quantization Fundamentals](#quantization-fundamentals)
2. [Symmetric vs Asymmetric Quantization](#symmetric-vs-asymmetric-quantization)
3. [Per-Tensor vs Per-Channel Quantization](#per-tensor-vs-per-channel-quantization)
4. [Calibration Methods](#calibration-methods)
5. [Q/DQ Operations](#qdq-operations)
6. [Implementation Details](#implementation-details)

---

## Quantization Fundamentals

### Affine Quantization

Quantization maps floating-point values to a discrete set of integer values. The KPU simulator uses **affine quantization** (also called uniform quantization), which is the most common approach in neural network inference.

#### Quantization Formula

```
x_quant = clamp(round(x / scale) + zero_point, qmin, qmax)
```

Where:
- `x` is the original floating-point value
- `scale` is the quantization scale factor
- `zero_point` is the offset (integer)
- `qmin`, `qmax` are the quantized range limits (e.g., -128 to 127 for INT8)

#### Dequantization Formula

```
x_dequant = (x_quant - zero_point) * scale
```

### Quantization Error

The quantization error for a single value is bounded by:

```
|x - x_dequant| ≤ scale / 2
```

This is the **rounding error**. Additionally, values outside the representable range are **clipped**, which can introduce larger errors for outliers.

### Scale and Zero Point Calculation

Given a data range `[min_val, max_val]` and target quantized range `[qmin, qmax]`:

**Asymmetric:**
```
scale = (max_val - min_val) / (qmax - qmin)
zero_point = round(qmin - min_val / scale)
```

**Symmetric:**
```
abs_max = max(|min_val|, |max_val|)
scale = abs_max / max(|qmin|, |qmax|)
zero_point = 0
```

---

## Symmetric vs Asymmetric Quantization

### Symmetric Quantization

- Zero point is always 0
- The quantized range is symmetric around zero: `[-qmax, +qmax]`
- For INT8: maps to [-127, 127] (not using -128)
- **Best for:** weights, data centered around zero

**Advantages:**
- Simpler computation (no zero_point in matmul inner loop)
- Faster inference on hardware
- Works well for normally distributed data

**Disadvantages:**
- Wastes range for non-negative data (e.g., ReLU outputs)

### Asymmetric Quantization

- Zero point can be any value in `[qmin, qmax]`
- Uses the full quantized range
- **Best for:** activations, especially after ReLU

**Advantages:**
- Better utilization of quantized range
- Lower error for skewed distributions

**Disadvantages:**
- Requires zero_point handling in computation
- Slightly more complex hardware implementation

### Example: ReLU Output

For ReLU output with range `[0, 8]` quantized to INT8:

| Method | Scale | Zero Point | Representable Range | Utilization |
|--------|-------|------------|---------------------|-------------|
| Symmetric | 0.063 | 0 | [-8, 8] | 50% |
| Asymmetric | 0.031 | -128 | [0, 8] | 100% |

Asymmetric achieves ~2x better precision for this case.

---

## Per-Tensor vs Per-Channel Quantization

### Per-Tensor Quantization

- Single scale and zero_point for the entire tensor
- Simple and fast
- May lose precision if different parts of the tensor have different ranges

### Per-Channel Quantization

- Separate scale and zero_point for each channel
- Commonly used for weight tensors along the output channel axis
- Better preserves accuracy for weights with varying magnitudes

**Example:** For a weight tensor of shape `[out_channels, in_channels]`:
- Per-tensor: 1 scale value
- Per-channel (axis=0): `out_channels` scale values

### When to Use Each

| Tensor Type | Recommended Approach |
|-------------|---------------------|
| Weights | Per-channel (along output axis) |
| Activations | Per-tensor |
| Biases | Per-channel (same as weights) or FP32 |

---

## Calibration Methods

Calibration determines optimal quantization parameters by analyzing representative data. The goal is to minimize quantization error while maximizing utilization of the quantized range.

### Method 1: MinMax Calibration

**Algorithm:**
1. Observe min and max values across all calibration data
2. Set scale to cover the full observed range

```python
min_val = min(all_observed_values)
max_val = max(all_observed_values)
scale = (max_val - min_val) / (qmax - qmin)  # asymmetric
# or
scale = max(|min_val|, |max_val|) / qmax     # symmetric
```

**Characteristics:**
- Simplest and fastest method
- Sensitive to outliers (a single extreme value affects the entire range)
- No clipping of observed values

**Best for:** Well-behaved distributions without outliers

### Method 2: Percentile Calibration

**Algorithm:**
1. Collect all observed values
2. Use percentile values instead of absolute min/max
3. Typically use 99.99th percentile

```python
low_percentile = (100 - percentile) / 2   # e.g., 0.005 for 99.99%
high_percentile = 100 - low_percentile    # e.g., 99.995

min_val = np.percentile(all_values, low_percentile)
max_val = np.percentile(all_values, high_percentile)
# Then compute scale as in MinMax
```

**Characteristics:**
- Robust to outliers
- Clips extreme values (introduces clipping error)
- Good balance between range utilization and outlier handling

**Best for:** Distributions with outliers, general-purpose calibration

### Method 3: MSE (Mean Squared Error) Calibration

**Algorithm:**
1. Collect calibration data
2. Search over candidate clipping thresholds
3. For each threshold, compute the MSE between original and quantized values
4. Select the threshold that minimizes MSE

```python
best_mse = infinity
best_scale = None

for clip_percentage in [0.9, 0.91, ..., 1.0]:
    clip_val = abs_max * clip_percentage
    scale = clip_val / qmax

    # Quantize and dequantize
    q = round(clip(x, -clip_val, clip_val) / scale)
    dq = q * scale

    mse = mean((x - dq)^2)
    if mse < best_mse:
        best_mse = mse
        best_scale = scale
```

**Characteristics:**
- Directly optimizes for reconstruction error
- More computationally expensive than MinMax/Percentile
- Often achieves lower average error

**Best for:** Accuracy-critical applications

### Method 4: Entropy (KL Divergence) Calibration

**Algorithm:**
1. Build a histogram of the calibration data
2. Search over candidate clipping thresholds
3. For each threshold:
   a. Quantize the histogram (map bins to quantization levels)
   b. Expand back to original resolution
   c. Compute KL divergence between original and quantized histograms
4. Select the threshold that minimizes KL divergence

```python
# Build reference histogram
histogram, bin_edges = np.histogram(data, bins=2048)
ref_distribution = histogram / sum(histogram)

best_kl = infinity
best_scale = None

for clip_percentage in [0.5, 0.51, ..., 1.0]:
    clip_val = abs_max * clip_percentage
    scale = clip_val / qmax

    # Map histogram bins to quantized levels
    quant_histogram = zeros(num_quant_levels)
    for bin_idx, count in enumerate(histogram):
        bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx+1]) / 2
        q_level = round(clip(bin_center, -clip_val, clip_val) / scale)
        quant_histogram[q_level] += count

    # Expand back to original resolution
    expanded_histogram = zeros(num_bins)
    for q_level, count in enumerate(quant_histogram):
        dequant_val = q_level * scale
        bin_idx = find_bin(dequant_val, bin_edges)
        expanded_histogram[bin_idx] += count

    # Compute KL divergence: D_KL(P || Q) = sum(P * log(P/Q))
    expanded_distribution = expanded_histogram / sum(expanded_histogram)
    kl = sum(ref_distribution * log(ref_distribution / expanded_distribution))

    if kl < best_kl:
        best_kl = kl
        best_scale = scale
```

**KL Divergence:**
```
D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```

Where P is the original distribution and Q is the quantized distribution.

**Characteristics:**
- Preserves the shape of the distribution
- Minimizes information loss
- May not minimize raw numerical error
- Used by TensorRT for activation calibration

**Best for:** Preserving statistical properties of activations

### Calibration Method Comparison

| Method | Speed | Outlier Handling | Error Minimization | Distribution Preservation |
|--------|-------|------------------|-------------------|--------------------------|
| MinMax | Fast | Poor | No optimization | No |
| Percentile | Fast | Good | Indirect | No |
| MSE | Medium | Good | Direct (MSE) | No |
| Entropy | Slow | Good | Indirect | Yes |

### Practical Recommendations

1. **Start with Percentile (99.99%)** - good default for most cases
2. **Use MSE for accuracy-critical layers** - directly optimizes error
3. **Use MinMax for weights** - weights typically don't have outliers
4. **Use Entropy when distribution shape matters** - e.g., attention scores

---

## Q/DQ Operations

Q/DQ (Quantize/Dequantize) operations provide explicit graph-level quantization nodes, commonly used in:
- ONNX quantized models
- Quantization-aware training (QAT)
- Hardware deployment

### Q/DQ Pattern

```
Input (FP32) → Q → (INT8) → DQ → (FP32) → Operation → Q → (INT8) → DQ → Output (FP32)
```

### Fake Quantization

For quantization-aware training, we use "fake quantization" which simulates quantization error while keeping values in floating-point:

```python
def fake_quantize(x, scale, zero_point, qmin, qmax):
    # Quantize
    x_q = round(x / scale) + zero_point
    x_q = clamp(x_q, qmin, qmax)
    # Immediately dequantize
    x_fq = (x_q - zero_point) * scale
    return x_fq  # Still FP32, but with quantization error
```

This allows gradients to flow during training while the model learns to be robust to quantization error.

### Straight-Through Estimator (STE)

During backpropagation, the rounding operation has zero gradient. QAT uses the Straight-Through Estimator:

```
Forward:  y = round(x)
Backward: dy/dx = 1  (pass gradient through unchanged)
```

---

## Implementation Details

### Supported Data Types

| Type | Bits | Range | Bytes/Element |
|------|------|-------|---------------|
| INT8 | 8 | [-128, 127] | 1 |
| UINT8 | 8 | [0, 255] | 1 |
| INT4 | 4 | [-8, 7] | 0.5 |
| UINT4 | 4 | [0, 15] | 0.5 |
| FP16 | 16 | ±65504 | 2 |
| BF16 | 16 | ±3.4e38 | 2 |
| FP8 E4M3 | 8 | ±448 | 1 |
| FP8 E5M2 | 8 | ±57344 | 1 |
| FP4 E2M1 | 4 | ±6 | 0.5 |

### Memory Bandwidth Reduction

| Quantization | vs FP32 | Typical Accuracy Loss |
|--------------|---------|----------------------|
| FP16 | 2x | <0.1% |
| BF16 | 2x | <0.5% |
| INT8 | 4x | <1% |
| FP8 | 4x | 1-3% |
| INT4 | 8x | 3-10% |
| FP4 | 8x | 5-15% |

### CalibrationObserver Usage

```python
from kpu import CalibrationObserver, CalibrationMethod, QuantDtype

# Create observer
observer = CalibrationObserver(
    method=CalibrationMethod.PERCENTILE,
    dtype=QuantDtype.INT8,
    symmetric=True,
    percentile=99.99,
    per_channel=False,
)

# Feed calibration data
for batch in calibration_dataset:
    activations = model(batch)
    observer.observe(activations)

# Get optimal parameters
params = observer.compute_params()
print(f"Scale: {params.scale}, Zero Point: {params.zero_point}")
```

### Quick Calibration Functions

```python
from kpu import calibrate_minmax, calibrate_percentile, calibrate_mse, calibrate_entropy

# One-shot calibration
params = calibrate_percentile(tensor, dtype=QuantDtype.INT8, percentile=99.9)

# Compare all methods
from kpu import compare_calibration_methods
results = compare_calibration_methods(tensor, dtype=QuantDtype.INT8)
for method, info in results.items():
    print(f"{method}: SNR={info['snr_db']:.1f} dB")
```

---

## References

1. Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR 2018.
2. Krishnamoorthi, R. "Quantizing deep convolutional networks for efficient inference." arXiv 2018.
3. NVIDIA TensorRT Documentation - Post-Training Quantization.
4. PyTorch Quantization Documentation.
5. ONNX Runtime Quantization Documentation.
