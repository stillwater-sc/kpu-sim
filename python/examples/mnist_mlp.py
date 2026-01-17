#!/usr/bin/env python3
"""
MNIST MLP Example - First KPU Functional Test

This example demonstrates:
1. Defining a 3-layer MLP using @kpu.compile decorator
2. Executing on the KPU behavioral simulator
3. Verifying correctness against NumPy reference
4. Inspecting the generated DFX IR

Network architecture: 784 -> 128 -> 64 -> 10
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kpu


# Set fidelity to BEHAVIORAL (computes actual values)
kpu.set_fidelity(kpu.BEHAVIORAL)


@kpu.compile
def mnist_mlp(x: kpu.Tensor,
              w1: kpu.Tensor, b1: kpu.Tensor,
              w2: kpu.Tensor, b2: kpu.Tensor,
              w3: kpu.Tensor, b3: kpu.Tensor) -> kpu.Tensor:
    """
    3-layer MLP for MNIST classification.

    Args:
        x: Input images [batch, 784]
        w1, b1: Layer 1 weights [784, 128] and bias [128]
        w2, b2: Layer 2 weights [128, 64] and bias [64]
        w3, b3: Layer 3 weights [64, 10] and bias [10]

    Returns:
        Logits [batch, 10]
    """
    # Layer 1: 784 -> 128 with ReLU
    h1 = kpu.relu(x @ w1 + b1)

    # Layer 2: 128 -> 64 with ReLU
    h2 = kpu.relu(h1 @ w2 + b2)

    # Layer 3: 64 -> 10 (no activation - raw logits)
    logits = h2 @ w3 + b3

    return logits


def xavier_init(shape: tuple, dtype=np.float32) -> np.ndarray:
    """Xavier/Glorot initialization for weights."""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape).astype(dtype) * std


def create_test_data(batch_size: int = 32):
    """Create synthetic test data for MNIST MLP."""
    np.random.seed(42)

    # Weights and biases with Xavier initialization
    w1 = kpu.Tensor(xavier_init((784, 128)), name="w1")
    b1 = kpu.Tensor(np.zeros(128, dtype=np.float32), name="b1")

    w2 = kpu.Tensor(xavier_init((128, 64)), name="w2")
    b2 = kpu.Tensor(np.zeros(64, dtype=np.float32), name="b2")

    w3 = kpu.Tensor(xavier_init((64, 10)), name="w3")
    b3 = kpu.Tensor(np.zeros(10, dtype=np.float32), name="b3")

    # Random input (simulating flattened MNIST images)
    x = kpu.Tensor(np.random.randn(batch_size, 784).astype(np.float32), name="input")

    return x, (w1, b1, w2, b2, w3, b3)


def reference_mlp(x, w1, b1, w2, b2, w3, b3):
    """NumPy reference implementation for verification."""
    h1 = np.maximum(x @ w1 + b1, 0)  # ReLU
    h2 = np.maximum(h1 @ w2 + b2, 0)  # ReLU
    logits = h2 @ w3 + b3
    return logits


def main():
    print("=" * 60)
    print("MNIST MLP on KPU Simulator")
    print("=" * 60)
    print()

    # Create test data
    batch_size = 32
    x, weights = create_test_data(batch_size=batch_size)
    w1, b1, w2, b2, w3, b3 = weights

    print("Network Architecture:")
    print(f"  Input:  {x.shape}")
    print(f"  Layer 1: {w1.shape[0]} -> {w1.shape[1]} (ReLU)")
    print(f"  Layer 2: {w2.shape[0]} -> {w2.shape[1]} (ReLU)")
    print(f"  Layer 3: {w3.shape[0]} -> {w3.shape[1]} (Linear)")
    print()

    # Execute on KPU simulator
    print("Compiling and executing on KPU (BEHAVIORAL)...")
    logits = mnist_mlp(x, w1, b1, w2, b2, w3, b3)

    print(f"  Output shape: {logits.shape}")
    print()

    # Verify against NumPy reference
    print("Verifying against NumPy reference...")
    ref_logits = reference_mlp(
        x.numpy(), w1.numpy(), b1.numpy(),
        w2.numpy(), b2.numpy(), w3.numpy(), b3.numpy()
    )

    max_diff = np.max(np.abs(logits.numpy() - ref_logits))
    print(f"  Max difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  [PASS] Results match reference implementation!")
    else:
        print("  [FAIL] Results do not match - check implementation")
        return 1

    # Print execution statistics
    stats = mnist_mlp.get_stats()
    if stats:
        print()
        print("Execution Statistics:")
        print(f"  Operations executed: {stats.ops_executed}")
        print(f"  MatMul FLOPs: {stats.matmul_flops:,}")

    # Show generated graph
    print()
    print("=" * 60)
    print("Operation Graph")
    print("=" * 60)
    print(mnist_mlp.summary())

    # Show DFX IR
    print()
    print("=" * 60)
    print("Generated DFX IR")
    print("=" * 60)
    dfx = mnist_mlp.get_dfx()
    if dfx:
        print(dfx.summary())
        print()
        print("DFX JSON (first 2000 chars):")
        print(dfx.to_json()[:2000])
        if len(dfx.to_json()) > 2000:
            print("...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
