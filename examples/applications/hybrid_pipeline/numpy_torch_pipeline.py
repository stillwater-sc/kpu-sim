#!/usr/bin/env python3
"""
Hybrid NumPy + PyTorch Pipeline on KPU Simulator

Demonstrates combining NumPy preprocessing operators with PyTorch DNN
inference using pretrained weights, all executing on the KPU simulator.

Pipeline:
    1. NumPy: Image loading from URL
    2. NumPy: Custom preprocessing (resize, crop, normalize)
    3. KPU: PyTorch model inference with pretrained weights
    4. NumPy: Postprocessing and analysis
    5. Validation against PyTorch reference

This pattern is common in production where custom preprocessing logic
is implemented in NumPy for flexibility.

Usage:
    PYTHONPATH=python python examples/applications/hybrid_pipeline/numpy_torch_pipeline.py

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
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import kpu

# Cache directory for downloaded images
CACHE_DIR = Path(__file__).parent / ".cache"

# Reference images for hybrid pipeline demo
# Format: (url, expected_class_name, expected_class_ids)
DEMO_IMAGES = [
    ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
     "Samoyed", [258]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02123159_tiger_cat.JPEG",
     "tiger cat", [282]),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02105641_Old_English_sheepdog.JPEG",
     "Old English sheepdog", [229]),
]

# ImageNet class labels URL
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def download_file(url: str, dest: Path, desc: str = "file") -> bool:
    """Download a file from URL to destination."""
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

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
        print("  Downloading ImageNet labels...")
        if not download_file(IMAGENET_LABELS_URL, labels_file, "ImageNet labels"):
            return [f"class_{i}" for i in range(1000)]

    return labels_file.read_text().strip().split('\n')


def numpy_load_image_from_url(url: str, index: int) -> np.ndarray:
    """Load image from URL as NumPy array.

    This demonstrates the common pattern of loading images into NumPy
    arrays for custom preprocessing pipelines.

    Returns:
        RGB image as float32 array (H, W, 3) in range [0, 1]
    """
    if not HAS_PIL:
        raise RuntimeError("PIL is required. Install with: pip install Pillow")

    cache_file = CACHE_DIR / f"hybrid_image_{index}.jpg"

    if not cache_file.exists():
        print(f"    Downloading image {index+1}...")
        if not download_file(url, cache_file, f"image {index+1}"):
            return None

    try:
        img = Image.open(cache_file).convert('RGB')
        # Convert to NumPy array and normalize to [0, 1]
        return np.array(img).astype(np.float32) / 255.0
    except Exception as e:
        print(f"    Failed to load image: {e}")
        return None


def numpy_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image using NumPy (bilinear-like interpolation).

    This is a pure NumPy implementation for demonstration.
    For production, use cv2.resize or PIL for better quality.
    """
    h, w = img.shape[:2]
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)

    # Simple nearest-neighbor resize (NumPy only)
    y_indices = np.linspace(0, h - 1, new_h).astype(int)
    x_indices = np.linspace(0, w - 1, new_w).astype(int)
    resized = img[y_indices[:, np.newaxis], x_indices]
    return resized


def numpy_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Center crop image to square size using NumPy."""
    h, w = img.shape[:2]
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return img[start_h:start_h+size, start_w:start_w+size]


def numpy_normalize(img: np.ndarray,
                    mean: tuple = (0.485, 0.456, 0.406),
                    std: tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Normalize image with ImageNet statistics using NumPy."""
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    return (img - mean) / std


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert NumPy image (H, W, C) to PyTorch tensor (1, C, H, W)."""
    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))
    # Add batch dimension
    img = img[np.newaxis, ...]
    return torch.from_numpy(img.astype(np.float32))


def numpy_preprocess(img: np.ndarray) -> np.ndarray:
    """Complete NumPy preprocessing pipeline.

    This demonstrates custom preprocessing entirely in NumPy,
    which is common when you need:
    - Custom augmentation logic
    - Domain-specific preprocessing
    - Integration with existing NumPy pipelines
    """
    # Resize to 256 on shorter side
    img = numpy_resize(img, 256)

    # Center crop to 224x224
    img = numpy_center_crop(img, 224)

    # Normalize with ImageNet stats
    img = numpy_normalize(img)

    return img


def numpy_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax using NumPy."""
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / exp_logits.sum()


def numpy_postprocess(logits: np.ndarray, labels: list, top_k: int = 5) -> list:
    """Postprocess model output with NumPy.

    Returns:
        List of dicts with class_id, class_name, probability
    """
    logits = logits[0]  # Remove batch dim
    probs = numpy_softmax(logits)

    # Top-k
    top_indices = np.argsort(probs)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'class_id': int(idx),
            'class_name': labels[idx],
            'probability': float(probs[idx])
        })
    return results


def numpy_analyze_features(features: np.ndarray) -> dict:
    """Analyze feature statistics with NumPy."""
    return {
        'mean': float(np.mean(features)),
        'std': float(np.std(features)),
        'min': float(np.min(features)),
        'max': float(np.max(features)),
        'sparsity': float(np.mean(np.abs(features) < 0.01)),  # Near-zero values
    }


def run_hybrid_pipeline():
    """Run the complete hybrid NumPy + PyTorch pipeline with pretrained weights."""
    if not HAS_PIL:
        print("Error: PIL/Pillow is required for this example.")
        print("Install with: pip install Pillow")
        sys.exit(1)

    print("=" * 70)
    print("Hybrid NumPy + PyTorch Pipeline on KPU Simulator")
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
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Compile with KPU backend
    print("Compiling model with KPU backend...")
    kpu_model = torch.compile(model, backend="kpu")
    print("  Compilation complete")
    print()

    # Process each demo image
    print("=" * 70)
    print("Processing Images with Hybrid NumPy + KPU Pipeline")
    print("=" * 70)
    print()

    total_images = 0
    correct_kpu = 0
    kpu_matches_pytorch = 0

    for i, (url, expected_name, expected_ids) in enumerate(DEMO_IMAGES):
        print(f"Image {i+1}/{len(DEMO_IMAGES)}: Expected '{expected_name}'")
        print("-" * 70)

        # Phase 1: NumPy Image Loading
        print("  Phase 1: NumPy Image Loading")
        img = numpy_load_image_from_url(url, i)
        if img is None:
            print("    Skipped (download failed)")
            print()
            continue

        total_images += 1
        print(f"    Raw image shape: {img.shape}, dtype: {img.dtype}")

        # Phase 2: NumPy Preprocessing
        print("  Phase 2: NumPy Preprocessing")
        img_resized = numpy_resize(img, 256)
        print(f"    After resize: {img_resized.shape}")

        img_cropped = numpy_center_crop(img_resized, 224)
        print(f"    After crop: {img_cropped.shape}")

        img_normalized = numpy_normalize(img_cropped)
        print(f"    After normalize: range [{img_normalized.min():.2f}, {img_normalized.max():.2f}]")

        # Phase 3: Convert to PyTorch tensor
        print("  Phase 3: Convert to PyTorch Tensor")
        x = numpy_to_tensor(img_normalized)
        print(f"    Tensor shape: {x.shape}")

        # Phase 4: KPU Inference
        print("  Phase 4: KPU Model Inference")
        with torch.no_grad():
            kpu_output = kpu_model(x)
        print(f"    Output shape: {kpu_output.shape}")

        # Phase 5: NumPy Postprocessing
        print("  Phase 5: NumPy Postprocessing")
        kpu_output_np = kpu_output.numpy()
        kpu_preds = numpy_postprocess(kpu_output_np, labels)
        stats = numpy_analyze_features(kpu_output_np)

        # PyTorch reference
        with torch.no_grad():
            pytorch_output = model(x)
        pytorch_preds = numpy_postprocess(pytorch_output.numpy(), labels)

        # Check correctness
        kpu_correct = kpu_preds[0]['class_id'] in expected_ids
        predictions_match = kpu_preds[0]['class_id'] == pytorch_preds[0]['class_id']

        if kpu_correct:
            correct_kpu += 1
        if predictions_match:
            kpu_matches_pytorch += 1

        # Results
        print()
        print(f"  Results:")
        print(f"    KPU prediction: {kpu_preds[0]['class_name']} ({kpu_preds[0]['probability']*100:.1f}%)")
        print(f"    PyTorch prediction: {pytorch_preds[0]['class_name']} ({pytorch_preds[0]['probability']*100:.1f}%)")
        print(f"    Correct: {'YES' if kpu_correct else 'NO'}")
        print(f"    KPU matches PyTorch: {'YES' if predictions_match else 'NO'}")

        # Numerical validation
        diff = np.abs(kpu_output_np - pytorch_output.numpy()).max()
        print(f"    Numerical diff: {diff:.2e}")

        # Feature statistics
        print(f"    Output stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
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

    print("Hybrid NumPy + PyTorch pipeline completed successfully!")


def main():
    """Main entry point."""
    run_hybrid_pipeline()


if __name__ == "__main__":
    main()
