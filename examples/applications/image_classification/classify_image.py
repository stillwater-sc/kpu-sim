#!/usr/bin/env python3
"""
Image Classification Pipeline on KPU Simulator

Complete pipeline demonstrating:
    1. Image loading from URLs (with caching)
    2. Preprocessing with torchvision transforms
    3. ResNet18 inference on KPU with pretrained weights
    4. Postprocessing to get class labels
    5. Validation against PyTorch reference

Usage:
    PYTHONPATH=python python examples/applications/image_classification/classify_image.py

Requirements:
    - Internet connection (first run downloads weights and images)
    - Pillow: pip install Pillow
"""

import sys
import os
import urllib.request
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python'))

import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import kpu

# Cache directory for downloaded images
CACHE_DIR = Path(__file__).parent / ".cache"

# Reference images for classification demo
# Format: (url, expected_class_name, expected_class_ids)
DEMO_IMAGES = [
    ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
     "Samoyed", [258]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099601_golden_retriever.JPEG",
     "golden retriever", [207]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG",
     "goldfish", [1]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02124075_Egyptian_cat.JPEG",
     "Egyptian cat", [285]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02106662_German_shepherd.JPEG",
     "German shepherd", [235]),
]

# ImageNet class labels URL
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def download_file(url: str, dest: Path, desc: str = "file") -> bool:
    """Download a file from URL to destination."""
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {desc}...")

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

    cache_file = CACHE_DIR / f"demo_image_{index}.jpg"

    if not download_file(url, cache_file, f"image {index+1}"):
        return None

    try:
        return Image.open(cache_file).convert('RGB')
    except Exception as e:
        print(f"    Failed to load image: {e}")
        return None


def preprocess_image(img: 'Image.Image') -> torch.Tensor:
    """Preprocess image for ResNet inference.

    Standard ImageNet preprocessing:
        1. Resize to 256 pixels on shorter side
        2. Center crop to 224x224
        3. Convert to tensor (0-1 range)
        4. Normalize with ImageNet mean/std

    Returns:
        Preprocessed tensor with shape (1, 3, 224, 224)
    """
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def postprocess_output(output: torch.Tensor, labels: list, top_k: int = 5) -> list:
    """Convert model output to class predictions.

    Returns:
        List of dicts with class_id, class_name, probability
    """
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


def main():
    """Run complete image classification pipeline."""
    if not HAS_PIL:
        print("Error: PIL/Pillow is required for this example.")
        print("Install with: pip install Pillow")
        sys.exit(1)

    print("=" * 70)
    print("Image Classification Pipeline on KPU Simulator")
    print("=" * 70)
    print()

    # Step 1: Load ImageNet labels
    print("Step 1: Loading ImageNet class labels...")
    labels = load_imagenet_labels()
    print(f"  Loaded {len(labels)} class labels")
    print()

    # Step 2: Load pretrained model
    print("Step 2: Loading pretrained ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print()

    # Step 3: Compile with KPU backend
    print("Step 3: Compiling model with KPU backend...")
    kpu_model = torch.compile(model, backend="kpu")
    print("  Compilation complete")
    print()

    # Step 4: Classify demo images
    print("=" * 70)
    print("Step 4: Classifying Demo Images")
    print("=" * 70)
    print()

    total_images = 0
    correct_kpu = 0
    kpu_matches_pytorch = 0

    for i, (url, expected_name, expected_ids) in enumerate(DEMO_IMAGES):
        print(f"Image {i+1}/{len(DEMO_IMAGES)}: Expected '{expected_name}'")
        print("-" * 50)

        # Load image
        print("  Loading image...")
        img = load_image(url, i)
        if img is None:
            print("  Skipped (download failed)")
            print()
            continue

        total_images += 1

        # Preprocess
        print("  Preprocessing...")
        x = preprocess_image(img)

        # Run PyTorch reference
        print("  Running PyTorch inference...")
        with torch.no_grad():
            pytorch_output = model(x)
        pytorch_preds = postprocess_output(pytorch_output, labels)

        # Run KPU inference
        print("  Running KPU inference...")
        with torch.no_grad():
            kpu_output = kpu_model(x)
        kpu_preds = postprocess_output(kpu_output, labels)

        # Check correctness
        pytorch_correct = pytorch_preds[0]['class_id'] in expected_ids
        kpu_correct = kpu_preds[0]['class_id'] in expected_ids
        predictions_match = pytorch_preds[0]['class_id'] == kpu_preds[0]['class_id']

        if kpu_correct:
            correct_kpu += 1
        if predictions_match:
            kpu_matches_pytorch += 1

        # Show results
        print()
        print(f"  PyTorch: {pytorch_preds[0]['class_name']} ({pytorch_preds[0]['probability']*100:.1f}%)")
        print(f"  KPU:     {kpu_preds[0]['class_name']} ({kpu_preds[0]['probability']*100:.1f}%)")
        print(f"  Correct: {'YES' if kpu_correct else 'NO'}")
        print(f"  KPU matches PyTorch: {'YES' if predictions_match else 'NO'}")

        # Numerical validation
        diff = (pytorch_output - kpu_output).abs().max().item()
        print(f"  Numerical diff: {diff:.2e}")
        print()

    # Summary
    print("=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print()
    print(f"Total images processed: {total_images}")
    print(f"KPU correct:            {correct_kpu}/{total_images} ({correct_kpu/total_images*100:.0f}%)")
    print(f"KPU matches PyTorch:    {kpu_matches_pytorch}/{total_images} ({kpu_matches_pytorch/total_images*100:.0f}%)")
    print()

    if kpu_matches_pytorch == total_images:
        print("PASSED: KPU predictions match PyTorch exactly for all images!")
    else:
        print(f"WARNING: KPU and PyTorch predictions differ on {total_images - kpu_matches_pytorch} image(s)")
    print()

    print("Image classification pipeline completed successfully!")


if __name__ == "__main__":
    main()
