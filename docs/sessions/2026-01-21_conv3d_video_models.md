# Session: Conv3d and Video Model Support

**Date:** 2026-01-21
**Version:** v0.6.4
**Focus:** Adding 3D convolution operators for video model support

## Summary

Extended the KPU Python package to support 3D convolution and related operators, enabling video model inference (R3D, R2+1D, MC3).

## Changes Made

### New 3D Operators (`python/kpu/fx_converter.py`)

1. **3D Convolution**
   - `_numpy_conv3d()` - Full 3D convolution using im2col algorithm
   - `_im2col_3d()` - 3D patch extraction with dilation support
   - Supports grouped convolution (for R2+1D factorized convolutions)

2. **3D Pooling**
   - `_numpy_max_pool3d()` - Using stride tricks for efficiency
   - `_numpy_avg_pool3d()` - Average pooling
   - `_numpy_adaptive_avg_pool3d()` - Optimized for global pooling case

3. **3D Batch Normalization**
   - `_emit_batch_norm3d_module()` - Handles 5D tensor reshape (N, C, D, H, W)

### Handler Registration

Added handlers for all 3D operators in:
- `_handle_call_function()` - F.conv3d, F.max_pool3d, F.avg_pool3d, F.adaptive_avg_pool3d
- `_handle_call_module()` - nn.Conv3d, nn.MaxPool3d, nn.AvgPool3d, nn.AdaptiveAvgPool3d, nn.BatchNorm3d

### Bug Fix: Dynamic Batch Normalization

Fixed `_emit_batch_norm()` (F.batch_norm) to dynamically detect input dimensionality:
- 4D tensors (N, C, H, W): reshape to (1, -1, 1, 1)
- 5D tensors (N, C, D, H, W): reshape to (1, -1, 1, 1, 1)

This was causing broadcast errors when video models used F.batch_norm with 5D inputs.

## Test Results

| Model | Parameters | Max Diff | Status |
|-------|------------|----------|--------|
| R3D-18 | 33.4M | 8.94e-08 | PASSED |
| R2+1D-18 | 31.5M | 1.19e-07 | PASSED |
| MC3-18 | 11.7M | 2.09e-07 | PASSED |

All video models pass with excellent numerical precision (< 1e-6).

## Model Compatibility Update

Updated `docs/model_compatibility.md`:
- Total models: 45
- Passed: 40 (89%)
- Partial: 5 (11%)
- Failed: 0 (0%)

Added Video Models section to compatibility matrix.

## Files Modified

| File | Changes |
|------|---------|
| `python/kpu/fx_converter.py` | Added 3D ops, fixed batch norm |
| `python/kpu/__init__.py` | Version bump to 0.6.4 |
| `python/pyproject.toml` | Version bump to 0.6.4 |
| `docs/model_compatibility.md` | Added video models, updated operator support |
| `CHANGELOG.md` | Added v0.6.4 entry |

## Architecture Notes

### 3D im2col Algorithm

The `_im2col_3d()` function extracts 3D patches for efficient convolution:

```python
# Fast path (no dilation): stride tricks
shape = (N, C, K_d, K_h, K_w, D_out, H_out, W_out)
strides = (s[0], s[1], s[2], s[3], s[4],
           s[2] * stride[0], s[3] * stride[1], s[4] * stride[2])
patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

# Slow path (with dilation): explicit loops
for od, oh, ow in product(range(D_out), range(H_out), range(W_out)):
    for c, kd, kh, kw in product(range(C), range(K_d), range(K_h), range(K_w)):
        d_pos = d_start + kd * dilation[0]
        # ... extract at dilated positions
```

### R2+1D Support

R2+1D uses factorized convolutions (2D spatial + 1D temporal), which are handled by:
- Standard Conv3d with kernel_size=(1, k, k) for spatial
- Standard Conv3d with kernel_size=(t, 1, 1) for temporal

Both work correctly with our grouped convolution support.

## Next Steps

- v0.7.0: Quantization support (INT8, FP16, BF16)
- Consider Conv3d dilation optimization if needed for specific models
