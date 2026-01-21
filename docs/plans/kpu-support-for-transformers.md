# Current KPU support for Transformers

Model Categories by Use Case
```text
  ┌───────────────────────┬─────────────────────────────────────────────────────┬───────────────────┐
  │         Task          │                     Key Models                      │    KPU Status     │
  ├───────────────────────┼─────────────────────────────────────────────────────┼───────────────────┤
  │ Image Classification  │ ResNet ✅, VGG ✅, ViT ⚠️, Swin ❌, EfficientNet ❌ │ Partial           │
  ├───────────────────────┼─────────────────────────────────────────────────────┼───────────────────┤
  │ Object Detection      │ Faster R-CNN ⚠️, DETR ⚠️, SSD ⚠️, RT-DETR ❌        │ Needs ROI ops     │
  ├───────────────────────┼─────────────────────────────────────────────────────┼───────────────────┤
  │ Semantic Segmentation │ FCN ⚠️, DeepLabV3 ⚠️                                │ Needs upsample    │
  ├───────────────────────┼─────────────────────────────────────────────────────┼───────────────────┤
  │ Instance Segmentation │ Mask R-CNN ⚠️                                       │ Needs ROI Align   │
  ├───────────────────────┼─────────────────────────────────────────────────────┼───────────────────┤
  │ Panoptic Segmentation │ DETR-Panoptic ⚠️                                    │ Needs mask head   │
  ├───────────────────────┼─────────────────────────────────────────────────────┼───────────────────┤
  │ Keypoint Detection    │ Keypoint R-CNN ⚠️                                   │ Needs ROI ops     │
  ├───────────────────────┼─────────────────────────────────────────────────────┼───────────────────┤
  │ Video Classification  │ R3D ❌, MViT ❌, Swin3D ❌                          │ Needs Conv3d      │
  ├───────────────────────┼─────────────────────────────────────────────────────┼───────────────────┤
  │ Optical Flow          │ RAFT ❌                                             │ Needs correlation │
  ├───────────────────────┼─────────────────────────────────────────────────────┼───────────────────┤
  │ Depth Estimation      │ MiDaS ⚠️ (ViT variant)                              │ ViT backbone      │
  └───────────────────────┴─────────────────────────────────────────────────────┴───────────────────┘
  Key Finding: ViT May Already Work
```

The basic Vision Transformer uses operators we already support:
  - Patch embedding (Conv2d 16x16)
  - Linear/MatMul
  - LayerNorm
  - Multi-head attention
  - GELU
  - Softmax

Recommended first step: Validate ViT-B/16 inference to confirm.

Critical Operator Gaps
```text
  ┌──────────────────┬──────────┬───────────────────────────────────┐
  │     Operator     │ Priority │          Models Unlocked          │
  ├──────────────────┼──────────┼───────────────────────────────────┤
  │ Depthwise Conv2d │ P0       │ MobileNet, EfficientNet, ConvNeXt │
  ├──────────────────┼──────────┼───────────────────────────────────┤
  │ Grouped Conv2d   │ P0       │ ShuffleNet, RegNet                │
  ├──────────────────┼──────────┼───────────────────────────────────┤
  │ Upsample         │ P1       │ All segmentation, FPN             │
  ├──────────────────┼──────────┼───────────────────────────────────┤
  │ ROI Align        │ P1       │ All detection models              │
  └──────────────────┴──────────┴───────────────────────────────────┘
```
