#!/usr/bin/env python3
"""
Hybrid NumPy + PyTorch Pipeline on KPU Simulator

Demonstrates combining NumPy preprocessing operators with PyTorch DNN
inference, all executing on the KPU simulator.

Pipeline:
    1. NumPy: Image loading and basic preprocessing
    2. NumPy: Custom augmentation/normalization
    3. KPU: PyTorch model inference
    4. NumPy: Postprocessing and analysis

This pattern is common in production where custom preprocessing logic
is implemented in NumPy for flexibility.

Usage:
    PYTHONPATH=python python examples/applications/hybrid_pipeline/numpy_torch_pipeline.py
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python'))

import torch
import torchvision.models as models
import numpy as np
import kpu


def numpy_load_image(height: int = 256, width: int = 256) -> np.ndarray:
    """Simulate image loading with NumPy.

    In production, this would use:
        import cv2
        img = cv2.imread('image.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    Returns:
        RGB image as float32 array (H, W, 3) in range [0, 1]
    """
    # Create synthetic image
    np.random.seed(42)
    img = np.random.rand(height, width, 3).astype(np.float32)
    return img


def numpy_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image using NumPy (nearest neighbor).

    For production, use cv2.resize or PIL for better quality.
    """
    h, w = img.shape[:2]
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)

    # Simple nearest-neighbor resize
    y_indices = np.linspace(0, h - 1, new_h).astype(int)
    x_indices = np.linspace(0, w - 1, new_w).astype(int)
    resized = img[y_indices[:, np.newaxis], x_indices]
    return resized


def numpy_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Center crop image to square size."""
    h, w = img.shape[:2]
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return img[start_h:start_h+size, start_w:start_w+size]


def numpy_normalize(img: np.ndarray,
                    mean: tuple = (0.485, 0.456, 0.406),
                    std: tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Normalize image with ImageNet statistics."""
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


def numpy_postprocess(logits: np.ndarray, top_k: int = 5) -> list:
    """Postprocess model output with NumPy.

    Args:
        logits: Raw model output, shape (1, 1000)
        top_k: Number of top predictions

    Returns:
        List of (class_idx, probability) tuples
    """
    # Softmax
    logits = logits[0]  # Remove batch dim
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / exp_logits.sum()

    # Top-k
    top_indices = np.argsort(probs)[-top_k:][::-1]
    return [(int(idx), float(probs[idx])) for idx in top_indices]


def numpy_analyze_features(features: np.ndarray) -> dict:
    """Analyze feature statistics with NumPy.

    This demonstrates using NumPy for model output analysis.
    """
    return {
        'mean': float(np.mean(features)),
        'std': float(np.std(features)),
        'min': float(np.min(features)),
        'max': float(np.max(features)),
        'sparsity': float(np.mean(features == 0)),
    }


def run_hybrid_pipeline():
    """Run the complete hybrid NumPy + PyTorch pipeline."""
    print("Hybrid NumPy + PyTorch Pipeline on KPU Simulator")
    print("=" * 60)
    print()

    # Phase 1: NumPy Preprocessing
    print("Phase 1: NumPy Preprocessing")
    print("-" * 60)

    print("  Loading image with NumPy...")
    img = numpy_load_image(300, 400)
    print(f"    Raw image shape: {img.shape}")

    print("  Resizing...")
    img = numpy_resize(img, 256)
    print(f"    After resize: {img.shape}")

    print("  Center cropping...")
    img = numpy_center_crop(img, 224)
    print(f"    After crop: {img.shape}")

    print("  Normalizing...")
    img = numpy_normalize(img)
    print(f"    After normalize: range [{img.min():.2f}, {img.max():.2f}]")
    print()

    # Phase 2: Convert to PyTorch
    print("Phase 2: Convert to PyTorch Tensor")
    print("-" * 60)
    x = numpy_to_tensor(img)
    print(f"  Tensor shape: {x.shape}")
    print(f"  Tensor dtype: {x.dtype}")
    print()

    # Phase 3: KPU Inference
    print("Phase 3: KPU Model Inference")
    print("-" * 60)

    print("  Loading ResNet18...")
    model = models.resnet18(weights=None).eval()

    print("  Compiling with KPU backend...")
    compiled_model = torch.compile(model, backend="kpu")

    print("  Running inference...")
    with torch.no_grad():
        output = compiled_model(x)
    print(f"  Output shape: {output.shape}")
    print()

    # Phase 4: NumPy Postprocessing
    print("Phase 4: NumPy Postprocessing")
    print("-" * 60)

    # Convert output to NumPy
    output_np = output.numpy()

    print("  Computing softmax and top-5...")
    predictions = numpy_postprocess(output_np)

    print("  Analyzing output features...")
    stats = numpy_analyze_features(output_np)

    print()
    print("Results:")
    print("-" * 60)
    print("Top-5 Predictions (random weights):")
    for i, (class_idx, prob) in enumerate(predictions):
        print(f"  {i+1}. Class {class_idx:4d}: {prob*100:5.2f}%")

    print()
    print("Output Statistics:")
    print(f"  Mean:     {stats['mean']:.4f}")
    print(f"  Std:      {stats['std']:.4f}")
    print(f"  Range:    [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Sparsity: {stats['sparsity']*100:.1f}%")
    print()

    # Validation
    print("Validation:")
    print("-" * 60)
    with torch.no_grad():
        ref_output = model(x)
    max_diff = (output - ref_output).abs().max().item()
    print(f"  Max difference vs PyTorch: {max_diff:.2e}")
    print(f"  Status: {'PASSED' if max_diff < 1e-3 else 'FAILED'}")
    print()

    print("Hybrid pipeline completed successfully!")


def run_batch_processing():
    """Demonstrate batch processing with NumPy + PyTorch."""
    print("\n" + "=" * 60)
    print("Batch Processing Demo")
    print("=" * 60 + "\n")

    batch_size = 4
    print(f"Processing batch of {batch_size} images...")

    # Create batch of images with NumPy
    images = []
    for i in range(batch_size):
        np.random.seed(i)
        img = numpy_load_image(256, 256)
        img = numpy_preprocess(img)
        images.append(img)

    # Stack into batch
    batch_np = np.stack(images, axis=0)
    batch_np = np.transpose(batch_np, (0, 3, 1, 2))  # NHWC -> NCHW
    batch = torch.from_numpy(batch_np.astype(np.float32))
    print(f"  Batch tensor shape: {batch.shape}")

    # Run inference
    model = models.resnet18(weights=None).eval()
    compiled = torch.compile(model, backend="kpu")

    with torch.no_grad():
        outputs = compiled(batch)
    print(f"  Output shape: {outputs.shape}")

    # Process each output
    outputs_np = outputs.numpy()
    print("\nPredictions per image:")
    for i in range(batch_size):
        preds = numpy_postprocess(outputs_np[i:i+1])
        top_class, top_prob = preds[0]
        print(f"  Image {i}: class {top_class:4d} ({top_prob*100:.1f}%)")

    print()


def main():
    """Main entry point."""
    run_hybrid_pipeline()

    # Optional: batch processing demo
    # run_batch_processing()

    print("All hybrid pipeline demos completed!")


if __name__ == "__main__":
    main()
