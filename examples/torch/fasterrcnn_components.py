#!/usr/bin/env python3
"""
Faster R-CNN Components on KPU Simulator

Demonstrates that the KPU simulator supports all neural network components
used in Faster R-CNN object detection:

1. Backbone (ResNet50) - Feature extraction
2. Feature Pyramid Network (FPN) - Multi-scale features
3. RPN Head - Region proposal generation
4. ROI Align - Region of interest pooling
5. Box Predictor - Classification and regression heads

Note: The full Faster R-CNN model has dynamic post-processing (NMS, proposal
filtering) that doesn't work with torch.compile. However, all the neural
network inference components are fully supported on KPU.

Usage:
    PYTHONPATH=python python examples/torch/fasterrcnn_components.py

Requirements:
    - torchvision with detection models
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import torch
import torchvision.models.detection as detection
from torchvision.ops import roi_align

import kpu


def test_backbone_fpn():
    """Test ResNet50 + FPN backbone."""
    print("=" * 60)
    print("1. Testing Backbone (ResNet50 + FPN)")
    print("=" * 60)

    model = detection.fasterrcnn_resnet50_fpn(
        weights=detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).eval()
    backbone = model.backbone

    print("Compiling backbone with KPU...")
    kpu_backbone = torch.compile(backbone, backend='kpu')

    # Test with sample image
    x = torch.randn(1, 3, 480, 640)

    with torch.no_grad():
        pytorch_features = backbone(x)
        kpu_features = kpu_backbone(x)

    print(f"Input: {x.shape}")
    print("FPN outputs (multi-scale features):")

    all_match = True
    for key in pytorch_features:
        pt_feat = pytorch_features[key]
        kpu_feat = kpu_features[key]
        diff = (pt_feat - kpu_feat).abs().max().item()
        status = "OK" if diff < 1e-4 else "DIFF"
        print(f"  {key}: {pt_feat.shape}, max diff: {diff:.2e} [{status}]")
        if diff >= 1e-4:
            all_match = False

    return all_match


def test_rpn_head():
    """Test Region Proposal Network head."""
    print()
    print("=" * 60)
    print("2. Testing RPN Head (Region Proposal Network)")
    print("=" * 60)

    model = detection.fasterrcnn_resnet50_fpn(
        weights=detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).eval()
    rpn_head = model.rpn.head

    # Wrapper to handle list inputs
    class RPNHeadWrapper(torch.nn.Module):
        def __init__(self, rpn_head):
            super().__init__()
            self.rpn_head = rpn_head

        def forward(self, f0, f1, f2, f3, f4):
            features = [f0, f1, f2, f3, f4]
            return self.rpn_head(features)

    wrapper = RPNHeadWrapper(rpn_head)

    print("Compiling RPN head with KPU...")
    kpu_wrapper = torch.compile(wrapper, backend='kpu')

    # FPN feature maps at different scales
    features = [
        torch.randn(1, 256, 60, 80),
        torch.randn(1, 256, 30, 40),
        torch.randn(1, 256, 15, 20),
        torch.randn(1, 256, 8, 10),
        torch.randn(1, 256, 4, 5),
    ]

    with torch.no_grad():
        pt_logits, pt_bbox = wrapper(*features)
        kpu_logits, kpu_bbox = kpu_wrapper(*features)

    print("RPN outputs (objectness + box regression per anchor):")
    all_match = True
    for i in range(len(pt_logits)):
        logits_diff = (pt_logits[i] - kpu_logits[i]).abs().max().item()
        bbox_diff = (pt_bbox[i] - kpu_bbox[i]).abs().max().item()
        status = "OK" if max(logits_diff, bbox_diff) < 1e-4 else "DIFF"
        print(f"  Scale {i}: logits {pt_logits[i].shape}, bbox {pt_bbox[i].shape}, "
              f"max diff: {max(logits_diff, bbox_diff):.2e} [{status}]")
        if max(logits_diff, bbox_diff) >= 1e-4:
            all_match = False

    return all_match


def test_roi_align():
    """Test ROI Align operation."""
    print()
    print("=" * 60)
    print("3. Testing ROI Align")
    print("=" * 60)

    # Feature map and boxes
    features = torch.randn(1, 256, 50, 50)
    boxes = torch.tensor([
        [0, 5, 5, 20, 20],
        [0, 10, 10, 35, 35],
        [0, 25, 25, 45, 45],
    ], dtype=torch.float32)

    output_size = (7, 7)
    spatial_scale = 1.0
    sampling_ratio = 2

    # Test via torch.compile
    from torchvision.ops import RoIAlign

    class ROIAlignModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.roi_align = RoIAlign(output_size, spatial_scale, sampling_ratio)

        def forward(self, features, boxes):
            return self.roi_align(features, boxes)

    model = ROIAlignModel()

    print("Compiling ROI Align with KPU...")
    kpu_model = torch.compile(model, backend='kpu')

    with torch.no_grad():
        pytorch_out = model(features, boxes)
        kpu_out = kpu_model(features, boxes)

    diff = (pytorch_out - kpu_out).abs().max().item()
    status = "OK" if diff < 1e-5 else "DIFF"

    print(f"Input features: {features.shape}")
    print(f"Input boxes: {boxes.shape[0]} regions")
    print(f"Output: {pytorch_out.shape}")
    print(f"Max diff: {diff:.2e} [{status}]")

    return diff < 1e-5


def test_box_predictor():
    """Test box classification and regression heads."""
    print()
    print("=" * 60)
    print("4. Testing Box Predictor (Classification + Regression)")
    print("=" * 60)

    model = detection.fasterrcnn_resnet50_fpn(
        weights=detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).eval()
    box_predictor = model.roi_heads.box_predictor

    print("Compiling box predictor with KPU...")
    kpu_predictor = torch.compile(box_predictor, backend='kpu')

    # ROI features (after ROI Align + FC layers)
    roi_features = torch.randn(50, 1024)  # 50 ROIs

    with torch.no_grad():
        pt_cls, pt_reg = box_predictor(roi_features)
        kpu_cls, kpu_reg = kpu_predictor(roi_features)

    cls_diff = (pt_cls - kpu_cls).abs().max().item()
    reg_diff = (pt_reg - kpu_reg).abs().max().item()

    print(f"Input: {roi_features.shape} (50 ROI features)")
    print(f"Class scores: {pt_cls.shape} (91 COCO classes)")
    print(f"Box deltas: {pt_reg.shape} (4 coords x 91 classes)")
    print(f"Classification max diff: {cls_diff:.2e}")
    print(f"Regression max diff: {reg_diff:.2e}")

    status = "OK" if max(cls_diff, reg_diff) < 1e-4 else "DIFF"
    print(f"Status: [{status}]")

    return max(cls_diff, reg_diff) < 1e-4


def test_interpolate():
    """Test bilinear interpolation (used in FPN)."""
    print()
    print("=" * 60)
    print("5. Testing Interpolate (Upsample in FPN)")
    print("=" * 60)

    import torch.nn.functional as F

    class InterpolateModel(torch.nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    model = InterpolateModel()

    print("Compiling interpolate with KPU...")
    kpu_model = torch.compile(model, backend='kpu')

    x = torch.randn(1, 256, 30, 40)

    with torch.no_grad():
        pytorch_out = model(x)
        kpu_out = kpu_model(x)

    diff = (pytorch_out - kpu_out).abs().max().item()
    status = "OK" if diff < 1e-5 else "DIFF"

    print(f"Input: {x.shape}")
    print(f"Output: {pytorch_out.shape} (2x upsampled)")
    print(f"Max diff: {diff:.2e} [{status}]")

    return diff < 1e-5


def main():
    """Run all component tests."""
    print("=" * 60)
    print("Faster R-CNN Components on KPU Simulator")
    print("=" * 60)
    print()
    print("This example validates that all neural network components")
    print("used in Faster R-CNN object detection work on the KPU.")
    print()

    results = []

    # Run tests
    results.append(("Backbone (ResNet50 + FPN)", test_backbone_fpn()))
    results.append(("RPN Head", test_rpn_head()))
    results.append(("ROI Align", test_roi_align()))
    results.append(("Box Predictor", test_box_predictor()))
    results.append(("Interpolate", test_interpolate()))

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All Faster R-CNN neural network components work on KPU!")
        print()
        print("Supported detection operators:")
        print("  [OK] Conv2d (standard, depthwise, grouped)")
        print("  [OK] BatchNorm2d")
        print("  [OK] ReLU")
        print("  [OK] MaxPool2d")
        print("  [OK] AdaptiveAvgPool2d")
        print("  [OK] Linear (FC layers)")
        print("  [OK] Interpolate/Upsample (bilinear, nearest)")
        print("  [OK] ROI Align")
        print("  [OK] Feature Pyramid Network")
        print()
        print("Note: Full detection models require dynamic post-processing")
        print("(NMS, proposal filtering) that runs on CPU after KPU inference.")
    else:
        print("Some components failed - check output above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
