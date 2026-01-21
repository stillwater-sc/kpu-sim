#!/usr/bin/env python3
"""
MobileNetV2 Inference on KPU Simulator with Pretrained Weights

Demonstrates loading a pretrained MobileNetV2 model from torchvision and
classifying real images on the KPU simulator via torch.compile backend.

This validates that the KPU simulator supports depthwise separable convolutions,
which are the key building block of efficient mobile architectures:
    - Depthwise convolution (groups == in_channels)
    - Pointwise convolution (1x1 conv)
    - Inverted residual blocks

Features:
    - Uses pretrained ImageNet weights (MobileNet_V2_Weights.IMAGENET1K_V1)
    - Downloads and classifies 10 reference images
    - Validates KPU classifications match PyTorch exactly
    - Shows top-5 predictions for each image

Usage:
    PYTHONPATH=python python examples/torch/mobilenetv2_inference.py

Requirements:
    - Internet connection (first run downloads weights and images)
    - torchvision with pretrained weights support
    - Pillow: pip install Pillow
"""

import sys
import os
import urllib.request
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import torch
import torchvision.models as models
import torchvision.transforms as T

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not installed. Install with: pip install Pillow")

import kpu

# Cache directory for downloaded images
CACHE_DIR = Path(__file__).parent / ".cache"

# Reference images with expected classifications
# Format: (url, expected_class_name, expected_class_ids)
REFERENCE_IMAGES = [
    # PyTorch example images
    ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
     "Samoyed", [258]),
    # ImageNet sample images (GitHub hosted)
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099601_golden_retriever.JPEG",
     "golden retriever", [207]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02123045_tabby.JPEG",
     "tabby/Egyptian cat", [281, 285]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02123159_tiger_cat.JPEG",
     "tiger cat", [282]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02106662_German_shepherd.JPEG",
     "German shepherd", [235]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG",
     "goldfish", [1]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02124075_Egyptian_cat.JPEG",
     "Egyptian cat", [285]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02105641_Old_English_sheepdog.JPEG",
     "Old English sheepdog", [229]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02102040_English_springer.JPEG",
     "English springer", [217]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02091831_Saluki.JPEG",
     "Saluki", [176]),
]

# ImageNet class labels URL
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def download_file(url: str, dest: Path, desc: str = "file") -> bool:
    """Download a file from URL to destination."""
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc}...")

    try:
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; KPU-Sim/1.0)'}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"    Failed to download: {e}")
        return False


def load_imagenet_labels() -> list:
    """Load ImageNet class labels."""
    labels_file = CACHE_DIR / "imagenet_classes.txt"

    if not labels_file.exists():
        if not download_file(IMAGENET_LABELS_URL, labels_file, "ImageNet labels"):
            return [f"class_{i}" for i in range(1000)]

    return labels_file.read_text().strip().split('\n')


def load_image(url: str, index: int) -> 'Image.Image':
    """Load image from URL, using cache."""
    if not HAS_PIL:
        raise RuntimeError("PIL is required. Install with: pip install Pillow")

    cache_file = CACHE_DIR / f"mobilenet_image_{index}.jpg"

    if not download_file(url, cache_file, f"image {index+1}"):
        return None

    try:
        return Image.open(cache_file).convert('RGB')
    except Exception as e:
        print(f"    Failed to load image: {e}")
        return None


def preprocess_image(img: 'Image.Image') -> torch.Tensor:
    """Preprocess image for MobileNetV2 inference.

    MobileNetV2 expects:
        - 224x224 input size
        - ImageNet normalization
    """
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def classify_image(model, img_tensor: torch.Tensor, labels: list, top_k: int = 5) -> list:
    """Run classification and return top-k predictions."""
    with torch.no_grad():
        output = model(img_tensor)

    probs = torch.softmax(output[0], dim=0)
    top_probs, top_indices = probs.topk(top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            'class_id': idx.item(),
            'class_name': labels[idx.item()],
            'probability': prob.item()
        })
    return results


def print_model_info(model):
    """Print information about the MobileNetV2 model architecture."""
    print("  Architecture: MobileNetV2")
    print("  Key innovation: Inverted residuals with depthwise separable convolutions")
    print("  Depthwise conv: Each input channel convolved separately (groups=in_channels)")
    print("  Pointwise conv: 1x1 convolution to mix channels")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} (7x smaller than ResNet18)")


def main():
    """Main entry point."""
    if not HAS_PIL:
        print("Error: PIL/Pillow is required for this example.")
        print("Install with: pip install Pillow")
        sys.exit(1)

    print("=" * 70)
    print("MobileNetV2 Inference on KPU Simulator")
    print("=" * 70)
    print()

    # Load ImageNet labels
    print("Loading ImageNet class labels...")
    labels = load_imagenet_labels()
    print(f"  Loaded {len(labels)} class labels")
    print()

    # Load pretrained MobileNetV2 model
    print("Loading pretrained MobileNetV2 model...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).eval()
    print_model_info(model)
    print()

    # Compile with KPU backend
    print("Compiling model with KPU backend...")
    print("  (This validates KPU supports depthwise separable convolutions)")
    try:
        kpu_model = torch.compile(model, backend="kpu")
        print("  Compilation complete!")
    except Exception as e:
        print(f"  Compilation FAILED: {e}")
        print()
        print("MobileNetV2 requires these operators:")
        print("  - Depthwise Conv2d (groups == in_channels)")
        print("  - Pointwise Conv2d (1x1 kernel)")
        print("  - BatchNorm2d")
        print("  - ReLU6 (clamped ReLU)")
        print()
        print("Check which operator is missing from the error above.")
        sys.exit(1)
    print()

    # Process reference images
    print("=" * 70)
    print("Classifying Reference Images")
    print("=" * 70)
    print()

    total_images = 0
    correct_kpu = 0
    correct_pytorch = 0
    kpu_matches_pytorch = 0

    for i, (url, expected_name, expected_ids) in enumerate(REFERENCE_IMAGES):
        print(f"Image {i+1}/{len(REFERENCE_IMAGES)}: Expected '{expected_name}'")
        print("-" * 50)

        # Load and preprocess image
        img = load_image(url, i)
        if img is None:
            print("  Skipped (download failed)")
            print()
            continue

        img_tensor = preprocess_image(img)
        total_images += 1

        # Run PyTorch reference
        pytorch_results = classify_image(model, img_tensor, labels)
        pytorch_top1 = pytorch_results[0]

        # Run KPU inference
        kpu_results = classify_image(kpu_model, img_tensor, labels)
        kpu_top1 = kpu_results[0]

        # Check correctness
        pytorch_correct = pytorch_top1['class_id'] in expected_ids
        kpu_correct = kpu_top1['class_id'] in expected_ids
        predictions_match = pytorch_top1['class_id'] == kpu_top1['class_id']

        if pytorch_correct:
            correct_pytorch += 1
        if kpu_correct:
            correct_kpu += 1
        if predictions_match:
            kpu_matches_pytorch += 1

        # Display results
        print(f"  PyTorch: {pytorch_top1['class_name']} ({pytorch_top1['probability']*100:.1f}%)")
        print(f"  KPU:     {kpu_top1['class_name']} ({kpu_top1['probability']*100:.1f}%)")
        print(f"  Correct: {'YES' if kpu_correct else 'NO'}")
        print(f"  KPU matches PyTorch: {'YES' if predictions_match else 'NO'}")

        # Show top-5 for interesting cases
        if not predictions_match or not kpu_correct:
            print(f"  KPU Top-5:")
            for j, pred in enumerate(kpu_results):
                marker = "*" if pred['class_id'] in expected_ids else " "
                print(f"    {j+1}. {marker} {pred['class_name']:30s} ({pred['probability']*100:5.1f}%)")

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Total images processed: {total_images}")
    if total_images > 0:
        print(f"PyTorch correct:        {correct_pytorch}/{total_images} ({correct_pytorch/total_images*100:.0f}%)")
        print(f"KPU correct:            {correct_kpu}/{total_images} ({correct_kpu/total_images*100:.0f}%)")
        print(f"KPU matches PyTorch:    {kpu_matches_pytorch}/{total_images} ({kpu_matches_pytorch/total_images*100:.0f}%)")
    else:
        print("No images were processed. Check your internet connection.")
        return
    print()

    # Numerical validation
    print("=" * 70)
    print("Numerical Precision Validation")
    print("=" * 70)
    print()

    sample_img = load_image(REFERENCE_IMAGES[0][0], 0)
    if sample_img:
        sample_tensor = preprocess_image(sample_img)
        with torch.no_grad():
            pytorch_out = model(sample_tensor)
            kpu_out = kpu_model(sample_tensor)

        diff = (pytorch_out - kpu_out).abs()
        print(f"Max difference:  {diff.max().item():.2e}")
        print(f"Mean difference: {diff.mean().item():.2e}")
        print()

        if diff.max().item() < 1e-4:
            print("Precision: PASSED (max diff < 1e-4)")
        elif diff.max().item() < 1e-3:
            print("Precision: ACCEPTABLE (max diff < 1e-3)")
        else:
            print("Precision: WARNING (max diff >= 1e-3)")
    print()

    # Final validation
    print("=" * 70)
    print("Validation Results")
    print("=" * 70)
    print()

    if kpu_matches_pytorch == total_images:
        print("PASSED: KPU predictions match PyTorch exactly for all images!")
        print()
        print("This confirms KPU simulator supports MobileNetV2 operators:")
        print("  [OK] Depthwise convolution (groups == in_channels)")
        print("  [OK] Pointwise convolution (1x1)")
        print("  [OK] Inverted residual blocks")
        print("  [OK] BatchNorm2d")
        print("  [OK] ReLU6 activation")
        print("  [OK] Global average pooling")
        print("  [OK] Fully connected classifier")
    else:
        print(f"WARNING: KPU and PyTorch predictions differ on {total_images - kpu_matches_pytorch} image(s)")
        print("  This may indicate numerical precision differences.")

    print()
    print("MobileNetV2 inference demo completed!")


if __name__ == "__main__":
    main()
