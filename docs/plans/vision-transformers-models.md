# Vision Transformers and PyTorch Vision Models Plan

## Overview

This plan catalogs PyTorch vision models available through torchvision and torch.hub,
organized by task type. It identifies which models are candidates for KPU simulator
support and the operators required for each.

## Model Categories and Use Cases

| Task | Use Cases | Example Applications |
|------|-----------|---------------------|
| **Image Classification** | Object recognition, content filtering | Product categorization, medical diagnosis |
| **Object Detection** | Locating objects with bounding boxes | Autonomous driving, surveillance |
| **Semantic Segmentation** | Pixel-wise class labeling | Scene understanding, mapping |
| **Instance Segmentation** | Per-object pixel masks | Robotics, image editing |
| **Panoptic Segmentation** | Combined semantic + instance | Full scene parsing |
| **Keypoint Detection** | Body pose estimation | Sports analytics, AR/VR |
| **Video Classification** | Action recognition | Content moderation, surveillance |
| **Optical Flow** | Motion estimation between frames | Video stabilization, tracking |
| **Depth Estimation** | Distance from camera | 3D reconstruction, AR |
| **Object Tracking** | Following objects across frames | Sports, traffic monitoring |

---

## 1. Image Classification Models

### CNN-Based (Currently Supported)

| Model | Parameters | Top-1 Acc | KPU Status | Notes |
|-------|------------|-----------|------------|-------|
| **ResNet18** | 11.7M | 69.8% | ✅ SUPPORTED | Validated with pretrained weights |
| **ResNet34** | 21.8M | 73.3% | ✅ SUPPORTED | Should work (same ops as ResNet18) |
| **ResNet50** | 25.6M | 80.9% | ✅ SUPPORTED | Bottleneck blocks |
| **ResNet101/152** | 44.5M/60.2M | 81.9%/82.3% | ✅ SUPPORTED | Deeper variants |
| **VGG11/13/16/19** | 133M-144M | 69-74% | ✅ SUPPORTED | Simple sequential, large |
| **AlexNet** | 61M | 56.5% | ✅ SUPPORTED | Classic architecture |
| **DenseNet121/169/201** | 8M-20M | 74-77% | ✅ VALIDATED | Dense connections work |
| **SqueezeNet** | 1.2M | 58% | ⚠️ PARTIAL | Fire modules, larger numerical diff |

### CNN-Based (Mobile/Efficient Architectures)

| Model | Parameters | Top-1 Acc | KPU Status | Notes |
|-------|------------|-----------|------------|-------|
| **MobileNetV2** | 3.5M | 72% | ✅ VALIDATED | 10/10 images, 100% match |
| **MobileNetV3** | 5.5M | 75% | ✅ VALIDATED | Small and Large variants work |
| **EfficientNet-B0** | 5.3M | 77% | ✅ VALIDATED | Predictions match |
| **EfficientNet B1-B7** | 8M-66M | 79-84% | ✅ SUPPORTED | Same ops as B0 |
| **EfficientNetV2** | 21M-120M | 84-86% | ⚠️ UNTESTED | Should work |
| **ShuffleNetV2** | 2.3M | 69% | ❌ NOT SUPPORTED | Channel shuffle not supported |
| **RegNet** | 4M-80M | 72-84% | ✅ VALIDATED | Grouped conv works |
| **ConvNeXt** | 29M-350M | 82-87% | ✅ VALIDATED | Depthwise + LayerNorm works |

### Vision Transformers

| Model | Parameters | Top-1 Acc | KPU Status | Notes |
|-------|------------|-----------|------------|-------|
| **ViT-B/16** | 86M | 81.8% | ✅ VALIDATED | 10/10 images correct, 100% match |
| **ViT-B/32** | 88M | 75.9% | ✅ SUPPORTED | Same ops as ViT-B/16 |
| **ViT-L/16** | 304M | 85.3% | ✅ SUPPORTED | Same ops, larger scale |
| **ViT-H/14** | 632M | 88.6% | ✅ SUPPORTED | Same ops, larger scale |
| **DeiT** | 86M | 83.4% | ⚠️ UNTESTED | Same as ViT + distillation token |
| **Swin-T** | 28M | 81.3% | ❌ NOT SUPPORTED | Shifted windows, relative position |
| **Swin-S/B/L** | 50M-197M | 83-87% | ❌ NOT SUPPORTED | Same as Swin-T |
| **MaxViT** | 31M-475M | 83-88% | ❌ NOT SUPPORTED | Grid attention, block attention |

### ViT Operator Requirements

```
Operators needed for basic ViT support (ALL VALIDATED ✅):
├── Patch Embedding (Conv2d 16x16 stride 16) ✅ WORKING
├── Linear/MLP ✅ WORKING
├── LayerNorm ✅ WORKING
├── Multi-Head Self-Attention ✅ WORKING
├── GELU activation ✅ WORKING
├── Softmax ✅ WORKING
├── Dropout (identity at inference) ✅ N/A
├── Class token concatenation ✅ WORKING
└── Position embedding addition ✅ WORKING
```

**ViT-B/16 VALIDATED** - See `examples/torch/vit_inference.py`
- 10/10 reference images classified correctly
- 100% match with PyTorch predictions
- Max numerical difference: ~2e-2 (acceptable for classification)

---

## 2. Object Detection Models

### torchvision.models.detection

| Model | Backbone | mAP (COCO) | KPU Status | Notes |
|-------|----------|------------|------------|-------|
| **Faster R-CNN** | ResNet50-FPN | 37.0 | ✅ COMPONENTS VALIDATED | All NN components work |
| **RetinaNet** | ResNet50-FPN | 36.4 | ✅ SUPPORTED | Same components as FRCNN |
| **SSD300** | VGG16 | 25.1 | ✅ SUPPORTED | VGG backbone works |
| **SSDlite** | MobileNetV3 | 21.3 | ✅ SUPPORTED | Depthwise conv now works |
| **FCOS** | ResNet50-FPN | 39.2 | ✅ SUPPORTED | Same components as FRCNN |

### Detection Operator Requirements

```
Detection operators (ALL VALIDATED ✅):
├── Feature Pyramid Network (FPN) ✅ WORKING
│   ├── Conv2d (1x1 and 3x3)
│   ├── Interpolate/Upsample (bilinear)
│   └── Add (lateral connections)
├── ROI Align ✅ WORKING (via fallback)
├── RPN Head ✅ WORKING
│   ├── Conv2d (3x3 shared, 1x1 heads)
│   └── Objectness + box regression
├── Box Predictor ✅ WORKING
│   └── Linear (classification + regression)
└── Backbone (ResNet50) ✅ WORKING
```

**Note**: Post-processing ops (NMS, proposal filtering) have dynamic shapes
and run on CPU. All neural network inference runs on KPU.

See `examples/torch/fasterrcnn_components.py` for validation.

### Transformer-Based Detection (torch.hub / HuggingFace)

| Model | mAP (COCO) | KPU Status | Notes |
|-------|------------|------------|-------|
| **DETR** | 42.0 | ⚠️ PARTIAL | Transformer encoder-decoder |
| **RT-DETR** | 54.8 | ❌ NOT SUPPORTED | Hybrid CNN + Transformer |
| **RF-DETR** | 56.0+ | ❌ NOT SUPPORTED | DINOv2 backbone |

---

## 3. Segmentation Models

### Semantic Segmentation (torchvision.models.segmentation)

| Model | Backbone | mIoU (COCO) | KPU Status | Notes |
|-------|----------|-------------|------------|-------|
| **FCN** | ResNet50/101 | 60.5/63.7 | ✅ VALIDATED | Dilated conv works |
| **DeepLabV3** | ResNet50/101 | 66.4/67.4 | ✅ VALIDATED | ASPP with dilated conv |
| **DeepLabV3+** | ResNet101 | 68+ | ✅ SUPPORTED | Same ops as DeepLabV3 |
| **LRASPP** | MobileNetV3 | 57.9 | ⚠️ PARTIAL | Works but larger numerical diff |

### Instance Segmentation

| Model | Backbone | Mask mAP | KPU Status | Notes |
|-------|----------|----------|------------|-------|
| **Mask R-CNN** | ResNet50-FPN | 34.6 | ⚠️ PARTIAL | ROI Align, mask head |

### Panoptic Segmentation

| Model | PQ (COCO) | KPU Status | Notes |
|-------|-----------|------------|-------|
| **DETR-Panoptic** | 43.4 | ⚠️ PARTIAL | DETR + mask head |
| **Panoptic SegFormer** | 50+ | ❌ NOT SUPPORTED | Transformer decoder |

---

## 4. Keypoint Detection

| Model | Backbone | Keypoint mAP | KPU Status | Notes |
|-------|----------|--------------|------------|-------|
| **Keypoint R-CNN** | ResNet50-FPN | 65.0 | ⚠️ PARTIAL | 17 COCO keypoints |

---

## 5. Video Models (torchvision.models.video)

| Model | Architecture | Top-1 (K400) | KPU Status | Notes |
|-------|--------------|--------------|------------|-------|
| **R3D-18** | 3D ResNet | 52.7% | ❌ NOT SUPPORTED | 3D convolutions |
| **R2+1D-18** | (2+1)D | 57.5% | ❌ NOT SUPPORTED | 3D conv decomposition |
| **MC3-18** | Mixed Conv | 53.9% | ❌ NOT SUPPORTED | 3D + 2D conv |
| **S3D** | Separable 3D | 58.0% | ❌ NOT SUPPORTED | Depthwise 3D conv |
| **MViT-v1/v2** | Transformer | 78-82% | ❌ NOT SUPPORTED | Pooling attention |
| **Swin3D** | 3D Swin | 78.8% | ❌ NOT SUPPORTED | 3D shifted windows |

**Video models require 3D convolution support** - significant new operator.

---

## 6. Optical Flow (torchvision.models.optical_flow)

| Model | Architecture | KITTI EPE | KPU Status | Notes |
|-------|--------------|-----------|------------|-------|
| **RAFT** | Recurrent | 5.10 | ❌ NOT SUPPORTED | Correlation volume, GRU |

**Optical flow requires correlation layers and recurrent units.**

---

## 7. Depth Estimation (torch.hub)

| Model | Architecture | Rel. Error | KPU Status | Notes |
|-------|--------------|------------|------------|-------|
| **MiDaS v3 Large** | DPT (ViT-Large) | 0.062 | ⚠️ PARTIAL | ViT backbone |
| **MiDaS v3 Hybrid** | DPT (ResNet+ViT) | 0.075 | ⚠️ PARTIAL | Hybrid architecture |
| **MiDaS Small** | EfficientNet | 0.116 | ❌ NOT SUPPORTED | Depthwise conv |
| **ZoeDepth** | MiDaS + binning | Metric | ❌ NOT SUPPORTED | Complex decoder |

---

## Implementation Priority

### Phase 1: Vision Transformers (ViT) - ✅ COMPLETE

**Goal**: Support basic ViT models for image classification

| Operator | Status | Validated |
|----------|--------|-----------|
| Patch embedding (Conv2d 16x16) | ✅ Working | ViT-B/16 |
| Linear layers | ✅ Working | ViT-B/16 |
| LayerNorm | ✅ Working | ViT-B/16 |
| Multi-head attention | ✅ Working | ViT-B/16 |
| GELU | ✅ Working | ViT-B/16 |
| Class token concat | ✅ Working | ViT-B/16 |
| Position embedding | ✅ Working | ViT-B/16 |

**Result**: ViT-B/16 validated with 10/10 images correct, 100% PyTorch match

### Phase 2: Depthwise/Grouped Convolution - ✅ COMPLETE

**Goal**: Unlock MobileNet, EfficientNet, ShuffleNet families

| Operator | Status | Validated |
|----------|--------|-----------|
| Depthwise Conv2d (groups==C_in) | ✅ Working | MobileNetV2, EfficientNet-B0 |
| Grouped Conv2d (groups>1) | ✅ Working | Unit tests |
| Squeeze-and-Excitation | ✅ Working | EfficientNet-B0 |

**Result**: MobileNetV2 and EfficientNet-B0 validated with pretrained weights

### Phase 3: Detection Infrastructure - ✅ COMPLETE

**Goal**: Basic object detection support

| Operator | Status | Validated |
|----------|--------|-----------|
| ROI Align | ✅ Working | Faster R-CNN |
| Feature Pyramid Network | ✅ Working | ResNet50-FPN |
| Interpolate/Upsample | ✅ Working | Bilinear, nearest |
| RPN Head | ✅ Working | Objectness + box regression |
| Box Predictor | ✅ Working | Classification + regression |

**Result**: All Faster R-CNN neural network components validated.
See `examples/torch/fasterrcnn_components.py`

**Note**: NMS runs on CPU (dynamic post-processing, not neural network op)

### Phase 4: 3D Convolutions - LOW PRIORITY

**Goal**: Video understanding models

| Operator | Models Unlocked | Effort |
|----------|-----------------|--------|
| Conv3d | R3D, MC3, S3D | High |
| 3D pooling | All video models | Medium |

**Estimated effort**: High

---

## Operator Gap Analysis

### Currently Supported ✅

```
Conv2d (standard)      MatMul/Linear        ReLU/GELU/SiLU
Conv2d (depthwise)     Conv2d (grouped)     Sigmoid/Tanh
BatchNorm2d            LayerNorm            Softmax
MaxPool2d              AvgPool2d            AdaptiveAvgPool2d
Add (skip conn)        Concat               Mul (SE blocks)
Transpose              Reshape              Flatten
```

### High Priority Gaps ❌

| Operator | Priority | Models Blocked |
|----------|----------|----------------|
| ~~**Depthwise Conv2d**~~ | ~~P0~~ | ✅ IMPLEMENTED |
| ~~**Grouped Conv2d**~~ | ~~P0~~ | ✅ IMPLEMENTED |
| ~~**Upsample/Interpolate**~~ | ~~P1~~ | ✅ IMPLEMENTED |
| ~~**ROI Align**~~ | ~~P1~~ | ✅ IMPLEMENTED |
| ~~**Dilated/Atrous Conv2d**~~ | ~~P2~~ | ✅ IMPLEMENTED |

### Medium Priority Gaps

| Operator | Priority | Models Blocked |
|----------|----------|----------------|
| **Conv3d** | P2 | Video models |
| **Transposed Conv2d** | P3 | Some decoders |
| **Channel Shuffle** | P3 | ShuffleNet |
| **Shifted Window Attention** | P3 | Swin Transformer |

---

## Recommended Next Steps

1. ~~**Validate ViT support**~~ ✅ DONE - `examples/torch/vit_inference.py`
   - ViT-B/16 validated with pretrained weights
   - 10/10 images correct, 100% PyTorch match

2. ~~**Add depthwise/grouped conv**~~ ✅ DONE - `examples/torch/mobilenetv2_inference.py`
   - MobileNetV2 validated: 10/10 images, 100% PyTorch match
   - EfficientNet-B0 validated: predictions match

3. ~~**Add detection support**~~ ✅ DONE - `examples/torch/fasterrcnn_components.py`
   - Upsample/Interpolate (bilinear, nearest): validated
   - ROI Align: validated
   - Feature Pyramid Network (FPN): validated
   - All Faster R-CNN NN components work on KPU
   - Note: NMS runs on CPU (dynamic post-processing)

4. ~~**Create model compatibility matrix**~~ ✅ DONE - `docs/model_compatibility.md`
   - 38 models tested: 28 PASSED, 3 PARTIAL, 7 FAILED
   - Test script: `examples/torch/model_compatibility.py`
   - Missing ops documented: channel shuffle, shifted windows

5. ~~**Add segmentation support**~~ ✅ DONE
   - Dilated/atrous convolution: implemented
   - FCN-ResNet50/101: validated
   - DeepLabV3-ResNet50/101: validated
   - LRASPP-MobileNetV3: works with larger numerical diff

---

## References

- [TorchVision Models Documentation](https://docs.pytorch.org/vision/stable/models.html)
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [DETR Paper](https://arxiv.org/abs/2005.12872)
- [MiDaS Depth Estimation](https://pytorch.org/hub/intelisl_midas_v2/)
- [RT-DETR (CVPR 2024)](https://github.com/lyuwenyu/RT-DETR)
- [RF-DETR (Roboflow)](https://github.com/roboflow/rf-detr)
- [DINOv2/v3 (Meta)](https://github.com/facebookresearch/dinov2)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2025-01-21 | Initial catalog of models and use cases |
| 0.2 | 2025-01-21 | ViT-B/16 validated, created vit_inference.py example |
| 0.3 | 2025-01-21 | Added grouped/depthwise conv, MobileNetV2 & EfficientNet-B0 validated |
| 0.4 | 2025-01-21 | Detection support: FPN, ROI Align, Interpolate validated for Faster R-CNN |
| 0.5 | 2025-01-21 | Model compatibility matrix: 35 models tested (24 pass, 2 partial, 9 fail) |
| 0.6 | 2025-01-21 | Dilated convolution support: FCN, DeepLabV3 segmentation models now work |
