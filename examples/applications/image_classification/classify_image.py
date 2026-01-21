#!/usr/bin/env python3
"""
Image Classification Pipeline on KPU Simulator

Complete pipeline demonstrating:
    1. Image loading and preprocessing
    2. ResNet18 inference on KPU
    3. Postprocessing to get class labels

This example uses synthetic data since we don't bundle ImageNet labels,
but shows the full pipeline structure for real image classification.

Usage:
    PYTHONPATH=python python examples/applications/image_classification/classify_image.py
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python'))

import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import kpu

# ImageNet class names (subset for demo)
IMAGENET_CLASSES = {
    0: 'tench',
    1: 'goldfish',
    2: 'great white shark',
    3: 'tiger shark',
    4: 'hammerhead',
    5: 'electric ray',
    6: 'stingray',
    7: 'rooster',
    8: 'hen',
    9: 'ostrich',
    # ... (full list would have 1000 entries)
}


def create_synthetic_image(height: int = 256, width: int = 256) -> np.ndarray:
    """Create a synthetic RGB image for demonstration.

    In a real application, this would be replaced with:
        from PIL import Image
        img = Image.open('path/to/image.jpg')
        img_array = np.array(img)

    Returns:
        RGB image as numpy array with shape (H, W, 3), dtype uint8
    """
    # Create gradient pattern
    y_gradient = np.linspace(0, 255, height).reshape(-1, 1)
    x_gradient = np.linspace(0, 255, width).reshape(1, -1)

    # Create RGB channels with different patterns
    r = ((y_gradient + x_gradient) / 2).astype(np.uint8)
    g = (y_gradient * np.ones((1, width))).astype(np.uint8)
    b = (np.ones((height, 1)) * x_gradient).astype(np.uint8)

    # Stack to create RGB image
    img = np.stack([r, g, b], axis=2)
    return img


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """Preprocess image for ResNet inference.

    Standard ImageNet preprocessing:
        1. Resize to 256 pixels on shorter side
        2. Center crop to 224x224
        3. Convert to tensor (0-1 range)
        4. Normalize with ImageNet mean/std

    Args:
        img: RGB image as numpy array (H, W, 3), dtype uint8

    Returns:
        Preprocessed tensor with shape (1, 3, 224, 224)
    """
    # Define ImageNet preprocessing transforms
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Apply transforms
    tensor = transform(img)

    # Add batch dimension
    return tensor.unsqueeze(0)


def postprocess_output(output: torch.Tensor, top_k: int = 5) -> list:
    """Convert model output to class predictions.

    Args:
        output: Raw model output (logits), shape (1, 1000)
        top_k: Number of top predictions to return

    Returns:
        List of (class_idx, class_name, probability) tuples
    """
    # Apply softmax to get probabilities
    probs = torch.softmax(output[0], dim=0)

    # Get top-k predictions
    top_probs, top_indices = probs.topk(top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        class_idx = idx.item()
        class_name = IMAGENET_CLASSES.get(class_idx, f"class_{class_idx}")
        results.append((class_idx, class_name, prob.item()))

    return results


def classify_image_pipeline(img: np.ndarray) -> list:
    """Full image classification pipeline.

    Args:
        img: RGB image as numpy array (H, W, 3)

    Returns:
        List of top-5 (class_idx, class_name, probability) predictions
    """
    # Load model
    model = models.resnet18(weights=None).eval()

    # Compile with KPU backend
    compiled_model = torch.compile(model, backend="kpu")

    # Preprocess
    x = preprocess_image(img)

    # Inference
    with torch.no_grad():
        output = compiled_model(x)

    # Postprocess
    predictions = postprocess_output(output)

    return predictions


def main():
    """Run complete image classification pipeline."""
    print("Image Classification Pipeline on KPU Simulator")
    print("=" * 50)
    print()

    # Step 1: Load/create image
    print("Step 1: Loading image...")
    img = create_synthetic_image(256, 256)
    print(f"  Image shape: {img.shape}")
    print(f"  Image dtype: {img.dtype}")
    print()

    # Step 2: Preprocess
    print("Step 2: Preprocessing...")
    x = preprocess_image(img)
    print(f"  Tensor shape: {x.shape}")
    print(f"  Tensor range: [{x.min():.2f}, {x.max():.2f}]")
    print()

    # Step 3: Model setup
    print("Step 3: Setting up model...")
    model = models.resnet18(weights=None).eval()
    compiled_model = torch.compile(model, backend="kpu")
    print("  Model: ResNet18")
    print("  Backend: KPU")
    print()

    # Step 4: Inference
    print("Step 4: Running inference...")
    with torch.no_grad():
        output = compiled_model(x)
    print(f"  Output shape: {output.shape}")
    print()

    # Step 5: Postprocess
    print("Step 5: Postprocessing...")
    predictions = postprocess_output(output)
    print()

    # Show results
    print("Results:")
    print("-" * 50)
    print("Top-5 Predictions (using random weights):")
    for i, (class_idx, class_name, prob) in enumerate(predictions):
        print(f"  {i+1}. {class_name:20s} (class {class_idx:4d}): {prob*100:5.2f}%")
    print()

    # Validation
    print("Validation:")
    print("-" * 50)
    with torch.no_grad():
        ref_output = model(x)
    max_diff = (output - ref_output).abs().max().item()
    print(f"  Max difference vs PyTorch: {max_diff:.2e}")
    print(f"  Status: {'PASSED' if max_diff < 1e-3 else 'FAILED'}")
    print()

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
