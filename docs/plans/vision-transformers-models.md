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
| **DenseNet121/169/201** | 8M-20M | 74-77% | ⚠️ UNTESTED | Dense connections |
| **SqueezeNet** | 1.2M | 58% | ⚠️ UNTESTED | Fire modules |

### CNN-Based (Requires New Operators)

| Model | Parameters | Top-1 Acc | KPU Status | Missing Operators |
|-------|------------|-----------|------------|-------------------|
| **MobileNetV2** | 3.5M | 72% | ❌ NOT SUPPORTED | Depthwise conv |
| **MobileNetV3** | 5.5M | 75% | ❌ NOT SUPPORTED | Depthwise conv, SE blocks |
| **EfficientNet B0-B7** | 5M-66M | 77-84% | ❌ NOT SUPPORTED | Depthwise conv, SE blocks |
| **EfficientNetV2** | 21M-120M | 84-86% | ❌ NOT SUPPORTED | Depthwise conv, SE blocks |
| **ShuffleNetV2** | 2.3M | 69% | ❌ NOT SUPPORTED | Channel shuffle, grouped conv |
| **RegNet** | 4M-80M | 72-84% | ❌ NOT SUPPORTED | Grouped conv |
| **ConvNeXt** | 29M-350M | 82-87% | ❌ NOT SUPPORTED | Depthwise conv, LayerNorm |

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
| **Faster R-CNN** | ResNet50-FPN | 37.0 | ⚠️ PARTIAL | RPN, ROI pooling needed |
| **RetinaNet** | ResNet50-FPN | 36.4 | ⚠️ PARTIAL | Focal loss, anchor generation |
| **SSD300** | VGG16 | 25.1 | ⚠️ PARTIAL | Multi-scale detection |
| **SSDlite** | MobileNetV3 | 21.3 | ❌ NOT SUPPORTED | Depthwise conv |
| **FCOS** | ResNet50-FPN | 39.2 | ⚠️ PARTIAL | Anchor-free detection |

### Detection Operator Requirements

```
Additional operators for detection:
├── Feature Pyramid Network (FPN) - Conv + upsample + add
├── ROI Pooling / ROI Align
├── Non-Maximum Suppression (NMS)
├── Anchor generation
└── Multi-scale feature extraction
```

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
| **FCN** | ResNet50/101 | 60.5/63.7 | ⚠️ PARTIAL | Transposed conv needed |
| **DeepLabV3** | ResNet50/101 | 66.4/67.4 | ⚠️ PARTIAL | Atrous/dilated conv |
| **DeepLabV3+** | ResNet101 | 68+ | ⚠️ PARTIAL | ASPP module |
| **LRASPP** | MobileNetV3 | 57.9 | ❌ NOT SUPPORTED | Depthwise conv |

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

### Phase 2: Depthwise/Grouped Convolution - HIGH VALUE

**Goal**: Unlock MobileNet, EfficientNet, ShuffleNet families

| Operator | Models Unlocked | Effort |
|----------|-----------------|--------|
| Depthwise Conv2d | MobileNetV2/V3, EfficientNet, ConvNeXt | Medium |
| Grouped Conv2d | ShuffleNet, RegNet, ResNeXt | Medium |
| Squeeze-and-Excitation | EfficientNet, MobileNetV3 | Low |

**Estimated effort**: Medium

### Phase 3: Detection Infrastructure - MEDIUM PRIORITY

**Goal**: Basic object detection support

| Operator | Models Unlocked | Effort |
|----------|-----------------|--------|
| ROI Align | Faster R-CNN, Mask R-CNN | High |
| Feature Pyramid Network | All detection models | Medium |
| NMS (Non-Max Suppression) | All detection models | Low |
| Upsample/Interpolate | FPN, segmentation | Medium |

**Estimated effort**: High

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
BatchNorm2d            LayerNorm            Sigmoid/Tanh
MaxPool2d              AvgPool2d            AdaptiveAvgPool2d
Softmax                Add (skip conn)      Concat
Transpose              Reshape              Flatten
```

### High Priority Gaps ❌

| Operator | Priority | Models Blocked |
|----------|----------|----------------|
| **Depthwise Conv2d** | P0 | MobileNet, EfficientNet, ConvNeXt |
| **Grouped Conv2d** | P0 | ShuffleNet, RegNet, ResNeXt |
| **Upsample/Interpolate** | P1 | Segmentation, FPN |
| **ROI Align** | P1 | Detection models |

### Medium Priority Gaps

| Operator | Priority | Models Blocked |
|----------|----------|----------------|
| **Conv3d** | P2 | Video models |
| **Transposed Conv2d** | P2 | FCN segmentation |
| **Dilated/Atrous Conv2d** | P2 | DeepLabV3 |

---

## Recommended Next Steps

1. ~~**Validate ViT support**~~ ✅ DONE - `examples/torch/vit_inference.py`
   - ViT-B/16 validated with pretrained weights
   - 10/10 images correct, 100% PyTorch match

2. **Add depthwise conv** - Unlock mobile-optimized models (NEXT)
   - Modify `conv2d` to support `groups` parameter
   - Test MobileNetV2 inference

3. **Create model compatibility matrix** - Track which models work
   - Automated testing of each model
   - Document operator requirements

4. **Add detection support** - For autonomous driving use case
   - Implement ROI Align
   - Implement FPN
   - Test Faster R-CNN

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
