# PyTorch Vision Models on KPU Simulator

This directory contains examples of running PyTorch vision models on the KPU simulator using `torch.compile(backend="kpu")`.

## Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies if needed
pip install torch torchvision Pillow
```

## Examples

### ResNet18 Inference with Pretrained Weights

Demonstrates loading a pretrained ResNet18 model and classifying 10 reference images:

```bash
PYTHONPATH=python python examples/torch/resnet18_inference.py
```

The example:
- Downloads pretrained ImageNet weights
- Downloads 10 reference images (cached for subsequent runs)
- Classifies each image with both PyTorch and KPU
- Validates that KPU predictions match PyTorch exactly

Expected output:
```
======================================================================
ResNet18 Inference on KPU Simulator (Pretrained Weights)
======================================================================

Loading ImageNet class labels...
  Loaded 1000 class labels

Loading pretrained ResNet18 model...
  Parameters: 11,689,512

Compiling model with KPU backend...
  Compilation complete

======================================================================
Classifying Reference Images
======================================================================

Image 1/10: Expected 'Samoyed'
--------------------------------------------------
  PyTorch prediction: Samoyed (88.5%)
  KPU prediction:     Samoyed (88.5%)
  Match expected:     YES (PyTorch: YES)
  KPU matches PyTorch: YES
  KPU Top-5:
    1. * Samoyed                        ( 88.5%)
    2.   Arctic fox                     (  4.6%)
    ...

======================================================================
Summary
======================================================================

Total images processed: 10
PyTorch correct:        10/10 (100%)
KPU correct:            10/10 (100%)
KPU matches PyTorch:    10/10 (100%)

======================================================================
Validation Results
======================================================================

PASSED: KPU predictions match PyTorch exactly for all images!

Numerical precision check:
  Max difference:  8.34e-06
  Mean difference: 1.82e-06
  Precision: PASSED (< 1e-3)
```

## How It Works

The KPU simulator provides a `torch.compile` backend that:

1. **Traces** the PyTorch model using FX
2. **Converts** FX graph operations to KPU operations
3. **Executes** using KPU's behavioral runtime (NumPy)

### Supported Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| Conv2d | Supported | im2col + matmul implementation |
| BatchNorm2d | Supported | Inference mode (running stats) |
| ReLU | Supported | Element-wise |
| MaxPool2d | Supported | With padding support |
| AdaptiveAvgPool2d | Supported | Any output size |
| Linear | Supported | MatMul + bias |
| Add (residuals) | Supported | Element-wise |
| Flatten | Supported | View operation |

### Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| ResNet18/34/50 | Supported | Full validation |
| VGG11/13/16/19 | Supported | Standard convolutions |
| AlexNet | Supported | Standard convolutions |
| SqueezeNet | Supported | Fire modules |
| MobileNetV2 | Not supported | Requires grouped conv |
| EfficientNet | Not supported | Requires grouped conv |

## Performance Notes

The KPU simulator uses behavioral (functional) simulation for correctness validation. This is not optimized for speed but provides:

- **Numerical correctness** vs PyTorch reference
- **Operator coverage** validation
- **Software bring-up** before hardware

For performance analysis, use transactional mode:

```python
compiled = torch.compile(model, backend="kpu_transactional")
```

## Adding New Models

To add support for new models:

1. Check required operators in `python/kpu/fx_converter.py`
2. Add missing operator handlers if needed
3. Create example script in this directory
4. Validate against PyTorch reference (max_diff < 1e-3)
