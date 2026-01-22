#!/usr/bin/env python3
"""
Model Compatibility Matrix for KPU Simulator

Automatically tests PyTorch vision models for KPU compatibility and generates
a compatibility report showing which models work and which operators are missing.

Usage:
    # Test all models
    PYTHONPATH=python python examples/torch/model_compatibility.py

    # Test specific category
    PYTHONPATH=python python examples/torch/model_compatibility.py --category classification

    # Generate markdown report
    PYTHONPATH=python python examples/torch/model_compatibility.py --report

    # Quick test (smaller input, fewer models)
    PYTHONPATH=python python examples/torch/model_compatibility.py --quick

Categories:
    - classification: Image classification models (ResNet, VGG, ViT, etc.)
    - detection: Object detection models (Faster R-CNN components)
    - segmentation: Semantic/instance segmentation models

Requirements:
    - torchvision
"""

import sys
import os
import time
import argparse
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import torch
import torch.nn as nn
import torchvision.models as models

import kpu


class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    PARTIAL = "PARTIAL"


@dataclass
class ModelTestResult:
    """Result of testing a single model."""
    name: str
    category: str
    status: TestStatus
    compile_time: float = 0.0
    inference_time: float = 0.0
    max_diff: float = 0.0
    error_message: str = ""
    missing_ops: List[str] = field(default_factory=list)
    parameters: int = 0
    notes: str = ""


@dataclass
class CompatibilityReport:
    """Complete compatibility report."""
    results: List[ModelTestResult] = field(default_factory=list)
    timestamp: str = ""
    kpu_version: str = ""
    torch_version: str = ""

    def add(self, result: ModelTestResult):
        self.results.append(result)

    def summary(self) -> Dict[str, int]:
        """Get summary counts by status."""
        counts = {s.value: 0 for s in TestStatus}
        for r in self.results:
            counts[r.status.value] += 1
        return counts

    def by_category(self) -> Dict[str, List[ModelTestResult]]:
        """Group results by category."""
        grouped = {}
        for r in self.results:
            if r.category not in grouped:
                grouped[r.category] = []
            grouped[r.category].append(r)
        return grouped


def test_model(
    name: str,
    model_fn: Callable[[], nn.Module],
    input_shape: Tuple[int, ...],
    category: str = "classification",
    tolerance: float = 1e-3,
    quick: bool = False
) -> ModelTestResult:
    """
    Test a single model for KPU compatibility.

    Args:
        name: Model name
        model_fn: Function that returns the model
        input_shape: Input tensor shape
        category: Model category
        tolerance: Max allowed difference for numerical validation
        quick: Use smaller input for faster testing

    Returns:
        ModelTestResult with test outcome
    """
    result = ModelTestResult(name=name, category=category)

    try:
        # Load model
        sys.stdout.write(f"  {name}: loading...")
        sys.stdout.flush()
        model = model_fn().eval()
        result.parameters = sum(p.numel() for p in model.parameters())

        # Create input
        if quick:
            # Use smaller input for quick tests
            if len(input_shape) == 4:  # NCHW
                input_shape = (1, input_shape[1],
                              min(input_shape[2], 224),
                              min(input_shape[3], 224))

        x = torch.randn(*input_shape)

        # Compile with KPU
        sys.stdout.write(" compiling...")
        sys.stdout.flush()
        compile_start = time.time()
        try:
            kpu_model = torch.compile(model, backend="kpu")
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            # Try to extract missing op from error
            err_str = str(e)
            if "not supported" in err_str.lower() or "unsupported" in err_str.lower():
                result.missing_ops = [err_str.split(":")[-1].strip()[:50]]
            print(" FAILED (compile)")
            sys.stdout.flush()
            return result

        result.compile_time = time.time() - compile_start

        # Run inference
        sys.stdout.write(" testing...")
        sys.stdout.flush()
        inference_start = time.time()

        with torch.no_grad():
            pytorch_out = model(x)
            kpu_out = kpu_model(x)

        result.inference_time = time.time() - inference_start

        # Handle different output types
        if isinstance(pytorch_out, torch.Tensor):
            diff = (pytorch_out - kpu_out).abs().max().item()
        elif isinstance(pytorch_out, dict):
            # Detection models return dicts
            diffs = []
            for key in pytorch_out:
                if isinstance(pytorch_out[key], torch.Tensor):
                    d = (pytorch_out[key] - kpu_out[key]).abs().max().item()
                    diffs.append(d)
            diff = max(diffs) if diffs else 0.0
        elif isinstance(pytorch_out, (list, tuple)):
            diffs = []
            for p, k in zip(pytorch_out, kpu_out):
                if isinstance(p, torch.Tensor):
                    diffs.append((p - k).abs().max().item())
            diff = max(diffs) if diffs else 0.0
        else:
            diff = 0.0

        result.max_diff = diff

        if diff < tolerance:
            result.status = TestStatus.PASSED
            print(f" PASSED (diff={diff:.2e})")
        else:
            result.status = TestStatus.PARTIAL
            result.notes = f"Numerical diff {diff:.2e} > tolerance {tolerance:.2e}"
            print(f" PARTIAL (diff={diff:.2e})")
        sys.stdout.flush()

    except Exception as e:
        result.status = TestStatus.FAILED
        result.error_message = str(e)
        # Extract useful info from traceback
        tb = traceback.format_exc()
        if "aten." in tb:
            # Extract the aten op that failed
            for line in tb.split("\n"):
                if "aten." in line:
                    result.missing_ops.append(line.strip()[:80])
                    break
        print(" FAILED")
        sys.stdout.flush()
        if "--verbose" in sys.argv:
            print(f"    Error: {e}")
            sys.stdout.flush()

    return result


# ============================================================================
# Classification Models
# ============================================================================

CLASSIFICATION_MODELS = [
    # ResNet family
    ("ResNet18", lambda: models.resnet18(weights=None), (1, 3, 224, 224)),
    ("ResNet34", lambda: models.resnet34(weights=None), (1, 3, 224, 224)),
    ("ResNet50", lambda: models.resnet50(weights=None), (1, 3, 224, 224)),
    ("ResNet101", lambda: models.resnet101(weights=None), (1, 3, 224, 224)),

    # VGG family
    ("VGG11", lambda: models.vgg11(weights=None), (1, 3, 224, 224)),
    ("VGG16", lambda: models.vgg16(weights=None), (1, 3, 224, 224)),
    ("VGG19", lambda: models.vgg19(weights=None), (1, 3, 224, 224)),

    # MobileNet family
    ("MobileNetV2", lambda: models.mobilenet_v2(weights=None), (1, 3, 224, 224)),
    ("MobileNetV3-Small", lambda: models.mobilenet_v3_small(weights=None), (1, 3, 224, 224)),
    ("MobileNetV3-Large", lambda: models.mobilenet_v3_large(weights=None), (1, 3, 224, 224)),

    # EfficientNet family
    ("EfficientNet-B0", lambda: models.efficientnet_b0(weights=None), (1, 3, 224, 224)),
    ("EfficientNet-B1", lambda: models.efficientnet_b1(weights=None), (1, 3, 240, 240)),
    ("EfficientNet-B2", lambda: models.efficientnet_b2(weights=None), (1, 3, 260, 260)),

    # DenseNet family
    ("DenseNet121", lambda: models.densenet121(weights=None), (1, 3, 224, 224)),
    ("DenseNet169", lambda: models.densenet169(weights=None), (1, 3, 224, 224)),

    # Other CNNs
    ("AlexNet", lambda: models.alexnet(weights=None), (1, 3, 224, 224)),
    ("SqueezeNet1.0", lambda: models.squeezenet1_0(weights=None), (1, 3, 224, 224)),
    ("SqueezeNet1.1", lambda: models.squeezenet1_1(weights=None), (1, 3, 224, 224)),
    ("ShuffleNetV2-x0.5", lambda: models.shufflenet_v2_x0_5(weights=None), (1, 3, 224, 224)),
    ("ShuffleNetV2-x1.0", lambda: models.shufflenet_v2_x1_0(weights=None), (1, 3, 224, 224)),

    # RegNet family
    ("RegNet-Y-400MF", lambda: models.regnet_y_400mf(weights=None), (1, 3, 224, 224)),
    ("RegNet-Y-800MF", lambda: models.regnet_y_800mf(weights=None), (1, 3, 224, 224)),

    # Vision Transformers
    ("ViT-B/16", lambda: models.vit_b_16(weights=None), (1, 3, 224, 224)),
    ("ViT-B/32", lambda: models.vit_b_32(weights=None), (1, 3, 224, 224)),
    ("ViT-L/16", lambda: models.vit_l_16(weights=None), (1, 3, 224, 224)),

    # ConvNeXt
    ("ConvNeXt-Tiny", lambda: models.convnext_tiny(weights=None), (1, 3, 224, 224)),
    ("ConvNeXt-Small", lambda: models.convnext_small(weights=None), (1, 3, 224, 224)),

    # Swin Transformers
    ("Swin-T", lambda: models.swin_t(weights=None), (1, 3, 224, 224)),
    ("Swin-S", lambda: models.swin_s(weights=None), (1, 3, 224, 224)),

    # MaxViT
    ("MaxViT-T", lambda: models.maxvit_t(weights=None), (1, 3, 224, 224)),
]

# Quick subset for fast testing
CLASSIFICATION_MODELS_QUICK = [
    ("ResNet18", lambda: models.resnet18(weights=None), (1, 3, 224, 224)),
    ("VGG11", lambda: models.vgg11(weights=None), (1, 3, 224, 224)),
    ("MobileNetV2", lambda: models.mobilenet_v2(weights=None), (1, 3, 224, 224)),
    ("EfficientNet-B0", lambda: models.efficientnet_b0(weights=None), (1, 3, 224, 224)),
    ("ViT-B/16", lambda: models.vit_b_16(weights=None), (1, 3, 224, 224)),
    ("DenseNet121", lambda: models.densenet121(weights=None), (1, 3, 224, 224)),
    ("SqueezeNet1.0", lambda: models.squeezenet1_0(weights=None), (1, 3, 224, 224)),
    ("ShuffleNetV2-x1.0", lambda: models.shufflenet_v2_x1_0(weights=None), (1, 3, 224, 224)),
]


# ============================================================================
# Detection Models (Components)
# ============================================================================

def get_fasterrcnn_backbone():
    """Get Faster R-CNN backbone for testing."""
    import torchvision.models.detection as detection
    model = detection.fasterrcnn_resnet50_fpn(weights=None).eval()
    return model.backbone

def get_fasterrcnn_rpn_head():
    """Get Faster R-CNN RPN head wrapped for testing."""
    import torchvision.models.detection as detection
    model = detection.fasterrcnn_resnet50_fpn(weights=None).eval()
    rpn_head = model.rpn.head

    class RPNHeadWrapper(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
        def forward(self, f0, f1, f2, f3, f4):
            return self.head([f0, f1, f2, f3, f4])

    return RPNHeadWrapper(rpn_head)

def get_fasterrcnn_box_predictor():
    """Get Faster R-CNN box predictor."""
    import torchvision.models.detection as detection
    model = detection.fasterrcnn_resnet50_fpn(weights=None).eval()
    return model.roi_heads.box_predictor


# Custom input shapes for detection components
DETECTION_MODELS = [
    ("FRCNN-Backbone", get_fasterrcnn_backbone, (1, 3, 480, 640)),
    ("FRCNN-BoxPredictor", get_fasterrcnn_box_predictor, (50, 1024)),  # ROI features
]


# ============================================================================
# Segmentation Models
# ============================================================================

def get_fcn_resnet50():
    """Get FCN with ResNet50 backbone."""
    return models.segmentation.fcn_resnet50(weights=None)

def get_deeplabv3_resnet50():
    """Get DeepLabV3 with ResNet50 backbone."""
    return models.segmentation.deeplabv3_resnet50(weights=None)

def get_lraspp_mobilenetv3():
    """Get LRASPP with MobileNetV3 backbone."""
    return models.segmentation.lraspp_mobilenet_v3_large(weights=None)


SEGMENTATION_MODELS = [
    ("FCN-ResNet50", get_fcn_resnet50, (1, 3, 224, 224)),
    ("DeepLabV3-ResNet50", get_deeplabv3_resnet50, (1, 3, 224, 224)),
    ("LRASPP-MobileNetV3", get_lraspp_mobilenetv3, (1, 3, 224, 224)),
]


# ============================================================================
# Report Generation
# ============================================================================

def generate_markdown_report(report: CompatibilityReport) -> str:
    """Generate markdown compatibility report."""
    lines = [
        "# KPU Model Compatibility Matrix",
        "",
        f"Generated: {report.timestamp}",
        f"KPU Version: {report.kpu_version}",
        f"PyTorch Version: {report.torch_version}",
        "",
    ]

    # Summary
    summary = report.summary()
    total = len(report.results)
    lines.extend([
        "## Summary",
        "",
        f"| Status | Count | Percentage |",
        f"|--------|-------|------------|",
        f"| ✅ PASSED | {summary['PASSED']} | {summary['PASSED']/total*100:.0f}% |",
        f"| ⚠️ PARTIAL | {summary['PARTIAL']} | {summary['PARTIAL']/total*100:.0f}% |",
        f"| ❌ FAILED | {summary['FAILED']} | {summary['FAILED']/total*100:.0f}% |",
        f"| ⏭️ SKIPPED | {summary['SKIPPED']} | {summary['SKIPPED']/total*100:.0f}% |",
        "",
    ])

    # Results by category
    for category, results in report.by_category().items():
        lines.extend([
            f"## {category.title()} Models",
            "",
            "| Model | Status | Parameters | Max Diff | Notes |",
            "|-------|--------|------------|----------|-------|",
        ])

        for r in results:
            status_icon = {
                TestStatus.PASSED: "✅",
                TestStatus.FAILED: "❌",
                TestStatus.PARTIAL: "⚠️",
                TestStatus.SKIPPED: "⏭️",
            }[r.status]

            params = f"{r.parameters/1e6:.1f}M" if r.parameters > 0 else "-"
            diff = f"{r.max_diff:.2e}" if r.max_diff > 0 else "-"
            notes = r.notes or r.error_message[:50] if r.error_message else ""

            lines.append(f"| {r.name} | {status_icon} {r.status.value} | {params} | {diff} | {notes} |")

        lines.append("")

    # Failed models detail
    failed = [r for r in report.results if r.status == TestStatus.FAILED]
    if failed:
        lines.extend([
            "## Failed Models - Error Details",
            "",
        ])
        for r in failed:
            lines.extend([
                f"### {r.name}",
                "",
                f"**Error:** {r.error_message[:200]}",
                "",
            ])
            if r.missing_ops:
                lines.append("**Missing/Unsupported ops:**")
                for op in r.missing_ops:
                    lines.append(f"- `{op}`")
                lines.append("")

    return "\n".join(lines)


def print_summary(report: CompatibilityReport):
    """Print summary to console."""
    print()
    print("=" * 70)
    print("Compatibility Summary")
    print("=" * 70)
    print()

    summary = report.summary()
    total = len(report.results)

    print(f"  PASSED:  {summary['PASSED']:3d} ({summary['PASSED']/total*100:5.1f}%)")
    print(f"  PARTIAL: {summary['PARTIAL']:3d} ({summary['PARTIAL']/total*100:5.1f}%)")
    print(f"  FAILED:  {summary['FAILED']:3d} ({summary['FAILED']/total*100:5.1f}%)")
    print(f"  SKIPPED: {summary['SKIPPED']:3d} ({summary['SKIPPED']/total*100:5.1f}%)")
    print(f"  ─────────────────")
    print(f"  TOTAL:   {total:3d}")
    print()

    # List failed models
    failed = [r for r in report.results if r.status == TestStatus.FAILED]
    if failed:
        print("Failed models:")
        for r in failed:
            print(f"  - {r.name}: {r.error_message[:60]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Test model compatibility with KPU")
    parser.add_argument("--category", choices=["classification", "detection", "segmentation", "all"],
                       default="all", help="Model category to test")
    parser.add_argument("--quick", action="store_true", help="Quick test with subset of models")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--output", type=str, help="Output file for report")
    parser.add_argument("--verbose", action="store_true", help="Show detailed errors")
    args = parser.parse_args()

    print("=" * 70)
    print("KPU Model Compatibility Matrix")
    print("=" * 70)
    print()
    print(f"KPU Version: {kpu.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    # Initialize report
    from datetime import datetime
    report = CompatibilityReport(
        timestamp=datetime.now().isoformat(),
        kpu_version=kpu.__version__,
        torch_version=torch.__version__,
    )

    # Select models to test
    models_to_test = []

    if args.category in ("classification", "all"):
        model_list = CLASSIFICATION_MODELS_QUICK if args.quick else CLASSIFICATION_MODELS
        models_to_test.extend([(n, f, s, "classification") for n, f, s in model_list])

    if args.category in ("detection", "all"):
        models_to_test.extend([(n, f, s, "detection") for n, f, s in DETECTION_MODELS])

    if args.category in ("segmentation", "all"):
        models_to_test.extend([(n, f, s, "segmentation") for n, f, s in SEGMENTATION_MODELS])

    # Run tests
    print(f"Testing {len(models_to_test)} models...")
    print("-" * 70)

    for name, model_fn, input_shape, category in models_to_test:
        result = test_model(name, model_fn, input_shape, category, quick=args.quick)
        report.add(result)

    # Print summary
    print_summary(report)

    # Generate report if requested
    if args.report:
        markdown = generate_markdown_report(report)

        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "docs", "model_compatibility.md"
            )

        with open(output_path, "w") as f:
            f.write(markdown)
        print(f"Report saved to: {output_path}")

    # Return exit code
    failed_count = report.summary()["FAILED"]
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
