# Application Pipelines on KPU Simulator

This directory contains complete application examples that combine PyTorch DNN inference with NumPy operators, demonstrating real-world usage patterns on the KPU simulator.

## Directory Structure

```
applications/
├── README.md                      # This file
├── image_classification/
│   └── classify_image.py          # Full image classification pipeline
└── hybrid_pipeline/
    └── numpy_torch_pipeline.py    # NumPy preprocessing + PyTorch inference
```

## Examples

### Image Classification Pipeline

Complete pipeline from raw image to class prediction:

```bash
PYTHONPATH=python python examples/applications/image_classification/classify_image.py
```

Pipeline steps:
1. **Load image** - Create/load RGB image
2. **Preprocess** - Resize, crop, normalize (torchvision transforms)
3. **Inference** - ResNet18 on KPU
4. **Postprocess** - Softmax and top-k predictions

### Hybrid NumPy + PyTorch Pipeline

Demonstrates custom NumPy preprocessing with PyTorch inference:

```bash
PYTHONPATH=python python examples/applications/hybrid_pipeline/numpy_torch_pipeline.py
```

Pipeline steps:
1. **NumPy preprocessing** - Custom resize, crop, normalize
2. **Tensor conversion** - NumPy to PyTorch
3. **KPU inference** - Compiled model execution
4. **NumPy postprocessing** - Analysis and predictions

## Common Patterns

### Pattern 1: Production Inference Pipeline

```python
import numpy as np
import torch
import kpu

# Load image (your preferred method)
img = load_image('path/to/image.jpg')  # Returns numpy array

# Preprocess with NumPy or torchvision
img_tensor = preprocess(img)

# Compile model once
model = torchvision.models.resnet18(weights=None).eval()
compiled = torch.compile(model, backend='kpu')

# Run inference
with torch.no_grad():
    output = compiled(img_tensor)

# Postprocess
predictions = postprocess(output.numpy())
```

### Pattern 2: Batch Processing

```python
# Preprocess batch with NumPy
images = [preprocess(img) for img in image_list]
batch = np.stack(images, axis=0)
batch_tensor = torch.from_numpy(batch)

# Single inference call
with torch.no_grad():
    outputs = compiled(batch_tensor)

# Process results
for i, output in enumerate(outputs):
    predictions = postprocess(output.numpy())
```

### Pattern 3: Feature Extraction

```python
# Use model as feature extractor
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = torch.nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.features(x).flatten(1)

extractor = FeatureExtractor(models.resnet18(weights=None))
compiled = torch.compile(extractor.eval(), backend='kpu')

# Extract features
with torch.no_grad():
    features = compiled(img_tensor)  # Shape: (batch, 512)
```

## Performance Considerations

The KPU simulator supports two execution modes:

### Behavioral Mode (Default)
- Full numerical computation
- Validates functional correctness
- Suitable for software development

```python
compiled = torch.compile(model, backend='kpu')
```

### Transactional Mode
- Performance estimation
- Memory traffic analysis
- Suitable for architecture exploration

```python
compiled = torch.compile(model, backend='kpu_transactional')
```

## Adding New Applications

To add a new application:

1. Create directory under `examples/applications/`
2. Implement pipeline with clear phases:
   - Data loading
   - Preprocessing (NumPy or torchvision)
   - Model inference (torch.compile with KPU)
   - Postprocessing (NumPy)
3. Add validation against PyTorch reference
4. Document usage in script docstring

## Dependencies

Required packages:
- `torch` - PyTorch for model definition
- `torchvision` - Pretrained models and transforms
- `numpy` - Array operations
- `kpu` - KPU simulator (from this repo)

Optional for real image handling:
- `Pillow` - Image loading
- `opencv-python` - Advanced image processing
