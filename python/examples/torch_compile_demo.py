#!/usr/bin/env python3
"""
torch.compile KPU Backend Demo

This example demonstrates using the KPU simulator as a torch.compile backend.
PyTorch models are compiled and executed on the KPU behavioral simulator.

Usage:
    python examples/torch_compile_demo.py

Requirements:
    pip install torch torchvision
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")
    print("Skipping torch.compile demo.")
    sys.exit(0)

import kpu


def demo_simple_mlp():
    """Demo 1: Simple MLP with torch.compile."""
    print("=" * 60)
    print("Demo 1: Simple MLP with torch.compile")
    print("=" * 60)

    # Define a simple MLP
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create model and set to eval mode
    model = SimpleMLP()
    model.eval()

    # Create random input
    x = torch.randn(32, 784)

    # Get PyTorch reference output
    with torch.no_grad():
        ref_output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reference output shape: {ref_output.shape}")

    # Compile with KPU backend
    print("\nCompiling with KPU backend...")
    compiled_model = torch.compile(model, backend="kpu")

    # Execute on KPU
    print("Executing on KPU simulator...")
    with torch.no_grad():
        kpu_output = compiled_model(x)

    print(f"KPU output shape: {kpu_output.shape}")

    # Compare results
    max_diff = torch.max(torch.abs(ref_output - kpu_output)).item()
    print(f"\nMax difference from PyTorch: {max_diff:.2e}")

    if max_diff < 1e-4:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED - outputs differ")

    return max_diff < 1e-4


def demo_cnn():
    """Demo 2: CNN with torch.compile."""
    print("\n" + "=" * 60)
    print("Demo 2: CNN with torch.compile")
    print("=" * 60)

    # Define a simple CNN for MNIST
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 28->14
            x = self.pool(F.relu(self.conv2(x)))  # 14->7
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create model
    model = SimpleCNN()
    model.eval()

    # Create MNIST-like input
    x = torch.randn(4, 1, 28, 28)

    # Get reference
    with torch.no_grad():
        ref_output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reference output shape: {ref_output.shape}")

    # Compile with KPU
    print("\nCompiling CNN with KPU backend...")
    compiled_model = torch.compile(model, backend="kpu")

    # Execute
    print("Executing on KPU simulator...")
    with torch.no_grad():
        kpu_output = compiled_model(x)

    print(f"KPU output shape: {kpu_output.shape}")

    # Compare
    max_diff = torch.max(torch.abs(ref_output - kpu_output)).item()
    print(f"\nMax difference from PyTorch: {max_diff:.2e}")

    if max_diff < 1e-3:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED - outputs differ")

    return max_diff < 1e-3


def demo_pretrained_model():
    """Demo 3: Pretrained model (if torchvision available)."""
    print("\n" + "=" * 60)
    print("Demo 3: Pretrained Model (SqueezeNet)")
    print("=" * 60)

    try:
        import torchvision
        import torchvision.models as models
    except ImportError:
        print("torchvision not installed. Skipping pretrained model demo.")
        print("Install with: pip install torchvision")
        return True

    # Load pretrained SqueezeNet (small model)
    print("Loading pretrained SqueezeNet 1.0...")
    model = models.squeezenet1_0(weights=None)  # No pretrained weights for speed
    model.eval()

    # Create ImageNet-like input
    x = torch.randn(1, 3, 224, 224)

    # Get reference
    print("Running PyTorch reference...")
    with torch.no_grad():
        ref_output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reference output shape: {ref_output.shape}")

    # Compile with KPU
    print("\nCompiling SqueezeNet with KPU backend...")
    print("(This may take a moment for a large model)")

    try:
        compiled_model = torch.compile(model, backend="kpu")

        # Execute
        print("Executing on KPU simulator...")
        with torch.no_grad():
            kpu_output = compiled_model(x)

        print(f"KPU output shape: {kpu_output.shape}")

        # Compare
        max_diff = torch.max(torch.abs(ref_output - kpu_output)).item()
        print(f"\nMax difference from PyTorch: {max_diff:.2e}")

        if max_diff < 1e-2:
            print("VALIDATION PASSED")
        else:
            print("VALIDATION FAILED - outputs differ (may be expected for complex models)")

        return max_diff < 1e-2

    except Exception as e:
        print(f"Error during compilation/execution: {e}")
        print("This may be due to unsupported operations in SqueezeNet.")
        print("The basic demos above should still work.")
        return False


def demo_function():
    """Demo 4: Compile a function (not just nn.Module)."""
    print("\n" + "=" * 60)
    print("Demo 4: Compile a Function")
    print("=" * 60)

    # Define a function using torch operations
    def my_function(x, w1, w2):
        h = torch.relu(torch.matmul(x, w1))
        return torch.matmul(h, w2)

    # Create inputs
    x = torch.randn(16, 64)
    w1 = torch.randn(64, 32)
    w2 = torch.randn(32, 10)

    # Reference
    ref_output = my_function(x, w1, w2)

    print(f"Input shape: {x.shape}")
    print(f"Reference output shape: {ref_output.shape}")

    # Compile
    print("\nCompiling function with KPU backend...")
    compiled_fn = torch.compile(my_function, backend="kpu")

    # Execute
    print("Executing on KPU simulator...")
    kpu_output = compiled_fn(x, w1, w2)

    print(f"KPU output shape: {kpu_output.shape}")

    # Compare
    max_diff = torch.max(torch.abs(ref_output - kpu_output)).item()
    print(f"\nMax difference from PyTorch: {max_diff:.2e}")

    if max_diff < 1e-4:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

    return max_diff < 1e-4


def main():
    print("=" * 60)
    print("KPU torch.compile Backend Demo")
    print("=" * 60)

    # Show KPU info
    print("\nKPU Package Info:")
    print(kpu.info())

    if not kpu.TORCH_AVAILABLE:
        print("PyTorch integration not available!")
        return 1

    print("\nRunning demos...\n")

    results = {
        "Simple MLP": demo_simple_mlp(),
        "CNN": demo_cnn(),
        "Function": demo_function(),
    }

    # Optionally run pretrained model demo
    # results["Pretrained"] = demo_pretrained_model()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("=" * 60)

    if all_passed:
        print("All demos passed!")
        return 0
    else:
        print("Some demos failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
