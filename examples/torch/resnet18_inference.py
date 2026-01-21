#!/usr/bin/env python3
"""
ResNet18 Inference on KPU Simulator with Pretrained Weights

Demonstrates loading a pretrained ResNet18 model from torchvision and
classifying real images on the KPU simulator via torch.compile backend.

Features:
    - Uses pretrained ImageNet weights
    - Downloads and classifies 10 reference images
    - Validates KPU classifications match PyTorch exactly
    - Shows top-5 predictions for each image

Usage:
    PYTHONPATH=python python examples/torch/resnet18_inference.py

Requirements:
    - Internet connection (first run downloads weights and images)
    - torchvision with pretrained weights support
"""

import sys
import os
import time
import urllib.request
import json
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np

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
# Class IDs from ImageNet-1K (multiple IDs for similar classes)
# Using GitHub-hosted PyTorch example images and other reliable CDNs
REFERENCE_IMAGES = [
    # PyTorch example images (from official tutorials/examples)
    ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
     "Samoyed", [258]),  # White fluffy dog
    ("https://raw.githubusercontent.com/pytorch/hub/master/images/deeplab1.png",
     "ram/livestock", [348, 341, 385, 386]),  # ram, hog, elephant, warthog - segmentation scene
    # Hugging Face sample images
    ("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
     "beignet/pastry", [928, 929, 930, 931, 965]),  # trifle, ice cream, food items, burrito
    # ImageNet sample images (GitHub hosted)
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099601_golden_retriever.JPEG",
     "golden retriever", [207]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02123045_tabby.JPEG",
     "tabby/Egyptian cat", [281, 285]),  # Often confused
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
]

# ImageNet class labels (subset - full list has 1000 classes)
# We'll download the full list on first run
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def download_file(url: str, dest: Path, desc: str = "file") -> bool:
    """Download a file from URL to destination."""
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc}...")

    try:
        # Add headers to avoid 403 errors from some servers
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
            # Fallback: return numeric labels
            return [f"class_{i}" for i in range(1000)]

    labels = labels_file.read_text().strip().split('\n')
    return labels


def load_image(url: str, index: int) -> 'Image.Image':
    """Load image from URL, using cache."""
    if not HAS_PIL:
        raise RuntimeError("PIL is required. Install with: pip install Pillow")

    # Create cache filename from URL hash
    cache_file = CACHE_DIR / f"image_{index}.jpg"

    if not download_file(url, cache_file, f"image {index+1}"):
        return None

    try:
        img = Image.open(cache_file).convert('RGB')
        return img
    except Exception as e:
        print(f"    Failed to load image: {e}")
        return None


def preprocess_image(img: 'Image.Image') -> torch.Tensor:
    """Preprocess image for ResNet inference."""
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


def main():
    """Main entry point."""
    if not HAS_PIL:
        print("Error: PIL/Pillow is required for this example.")
        print("Install with: pip install Pillow")
        sys.exit(1)

    print("=" * 70)
    print("ResNet18 Inference on KPU Simulator (Pretrained Weights)")
    print("=" * 70)
    print()

    # Load ImageNet labels
    print("Loading ImageNet class labels...")
    labels = load_imagenet_labels()
    print(f"  Loaded {len(labels)} class labels")
    print()

    # Load pretrained model
    print("Loading pretrained ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print()

    # Compile with KPU backend
    print("Compiling model with KPU backend...")
    kpu_model = torch.compile(model, backend="kpu")
    print("  Compilation complete")
    print()

    # Process reference images
    print("=" * 70)
    print("Classifying Reference Images")
    print("=" * 70)
    print()

    results_summary = []
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
        print(f"  PyTorch prediction: {pytorch_top1['class_name']} ({pytorch_top1['probability']*100:.1f}%)")
        print(f"  KPU prediction:     {kpu_top1['class_name']} ({kpu_top1['probability']*100:.1f}%)")
        print(f"  Match expected:     {'YES' if kpu_correct else 'NO'} (PyTorch: {'YES' if pytorch_correct else 'NO'})")
        print(f"  KPU matches PyTorch: {'YES' if predictions_match else 'NO'}")

        # Show top-5 for KPU
        print(f"  KPU Top-5:")
        for j, pred in enumerate(kpu_results):
            marker = "*" if pred['class_id'] in expected_ids else " "
            print(f"    {j+1}. {marker} {pred['class_name']:30s} ({pred['probability']*100:5.1f}%)")

        print()

        results_summary.append({
            'image': i + 1,
            'expected': expected_name,
            'kpu_prediction': kpu_top1['class_name'],
            'pytorch_prediction': pytorch_top1['class_name'],
            'kpu_correct': kpu_correct,
            'pytorch_correct': pytorch_correct,
            'match': predictions_match
        })

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
        print()
        return
    print()

    # Validation
    print("=" * 70)
    print("Validation Results")
    print("=" * 70)
    print()

    if kpu_matches_pytorch == total_images:
        print("PASSED: KPU predictions match PyTorch exactly for all images!")
    else:
        print(f"WARNING: KPU and PyTorch predictions differ on {total_images - kpu_matches_pytorch} image(s)")
        print("  This may indicate numerical precision differences.")

    # Numerical validation on a sample
    print()
    print("Numerical precision check:")
    sample_img = load_image(REFERENCE_IMAGES[0][0], 0)
    if sample_img:
        sample_tensor = preprocess_image(sample_img)
        with torch.no_grad():
            pytorch_out = model(sample_tensor)
            kpu_out = kpu_model(sample_tensor)

        diff = (pytorch_out - kpu_out).abs()
        print(f"  Max difference:  {diff.max().item():.2e}")
        print(f"  Mean difference: {diff.mean().item():.2e}")

        if diff.max().item() < 1e-3:
            print("  Precision: PASSED (< 1e-3)")
        else:
            print("  Precision: FAILED (>= 1e-3)")

    print()
    print("ResNet18 inference demo completed!")


if __name__ == "__main__":
    main()
