#!/usr/bin/env python3
"""
Comprehensive CNN Validation Tests for KPU Python Package.

This module provides thorough validation of CNN operators against
NumPy reference implementations. When PyTorch is available, it also
validates against PyTorch for additional confidence.

Run with: python tests/test_cnn_validation.py
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kpu

# Try to import PyTorch for optional comparison
try:
    import torch
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("Note: PyTorch not available, using NumPy reference only")


class NumpyReference:
    """NumPy reference implementations for validation."""

    @staticmethod
    def conv2d(x, weight, bias=None, stride=1, padding=0):
        """
        NumPy reference conv2d implementation.

        Args:
            x: [N, C_in, H, W]
            weight: [C_out, C_in, K_h, K_w]
            bias: [C_out] or None
            stride: int or (int, int)
            padding: int or (int, int)
        """
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        N, C_in, H, W = x.shape
        C_out, C_in_w, K_h, K_w = weight.shape

        # Pad input
        if padding[0] > 0 or padding[1] > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                          (padding[0], padding[0]),
                          (padding[1], padding[1])), mode='constant')

        H_out = (x.shape[2] - K_h) // stride[0] + 1
        W_out = (x.shape[3] - K_w) // stride[1] + 1

        output = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride[0]
                        w_start = w * stride[1]
                        receptive_field = x[n, :, h_start:h_start+K_h, w_start:w_start+K_w]
                        output[n, c_out, h, w] = np.sum(receptive_field * weight[c_out])

        if bias is not None:
            output = output + bias.reshape(1, -1, 1, 1)

        return output

    @staticmethod
    def max_pool2d(x, kernel_size, stride=None):
        """NumPy reference max_pool2d."""
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)

        N, C, H, W = x.shape
        K_h, K_w = kernel_size

        H_out = (H - K_h) // stride[0] + 1
        W_out = (W - K_w) // stride[1] + 1

        output = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride[0]
                        w_start = w * stride[1]
                        output[n, c, h, w] = np.max(
                            x[n, c, h_start:h_start+K_h, w_start:w_start+K_w]
                        )

        return output

    @staticmethod
    def batch_norm2d(x, gamma, beta, eps=1e-5):
        """NumPy reference batch_norm2d (inference mode)."""
        # For inference, we use running mean/var
        # Here we compute from batch for testing
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)

    @staticmethod
    def layer_norm(x, normalized_shape, gamma=None, beta=None, eps=1e-5):
        """NumPy reference layer_norm."""
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        ndim = len(normalized_shape)
        axes = tuple(range(-ndim, 0))
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        if gamma is not None:
            x_norm = x_norm * gamma
        if beta is not None:
            x_norm = x_norm + beta
        return x_norm


def test_conv2d_against_numpy():
    """Test conv2d against NumPy reference."""
    print("\n" + "=" * 60)
    print("Test: conv2d against NumPy reference")
    print("=" * 60)

    kpu.set_fidelity(kpu.BEHAVIORAL)
    np.random.seed(42)

    test_cases = [
        # (N, C_in, H, W), (C_out, C_in, K, K), stride, padding
        ((1, 1, 8, 8), (4, 1, 3, 3), 1, 0),
        ((2, 3, 16, 16), (8, 3, 3, 3), 1, 1),
        ((1, 16, 14, 14), (32, 16, 3, 3), 2, 1),
        ((4, 1, 28, 28), (16, 1, 5, 5), 1, 2),
    ]

    all_passed = True
    for x_shape, w_shape, stride, padding in test_cases:
        x_np = np.random.randn(*x_shape).astype(np.float32)
        w_np = np.random.randn(*w_shape).astype(np.float32)
        b_np = np.random.randn(w_shape[0]).astype(np.float32)

        # KPU result
        x_kpu = kpu.Tensor(x_np)
        w_kpu = kpu.Tensor(w_np)
        b_kpu = kpu.Tensor(b_np)
        y_kpu = kpu.conv2d(x_kpu, w_kpu, b_kpu, stride=stride, padding=padding)

        # NumPy reference
        y_np = NumpyReference.conv2d(x_np, w_np, b_np, stride=stride, padding=padding)

        max_diff = np.max(np.abs(y_kpu.numpy() - y_np))
        passed = max_diff < 1e-4  # Relaxed tolerance for accumulated floating-point errors

        status = "PASS" if passed else "FAIL"
        print(f"  {x_shape} * {w_shape} stride={stride} pad={padding}: {status} (diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    assert all_passed, "Some conv2d tests failed"


def test_conv2d_against_pytorch():
    """Test conv2d against PyTorch (if available)."""
    if not HAS_PYTORCH:
        print("\n" + "=" * 60)
        print("Test: conv2d against PyTorch - SKIPPED (PyTorch not installed)")
        print("=" * 60)
        return  # Skip test, no assertion needed

    print("\n" + "=" * 60)
    print("Test: conv2d against PyTorch")
    print("=" * 60)

    kpu.set_fidelity(kpu.BEHAVIORAL)
    np.random.seed(42)

    test_cases = [
        ((1, 1, 8, 8), (4, 1, 3, 3), 1, 0),
        ((2, 3, 16, 16), (8, 3, 3, 3), 1, 1),
        ((1, 16, 14, 14), (32, 16, 3, 3), 2, 1),
    ]

    all_passed = True
    for x_shape, w_shape, stride, padding in test_cases:
        x_np = np.random.randn(*x_shape).astype(np.float32)
        w_np = np.random.randn(*w_shape).astype(np.float32)
        b_np = np.random.randn(w_shape[0]).astype(np.float32)

        # KPU result
        x_kpu = kpu.Tensor(x_np)
        w_kpu = kpu.Tensor(w_np)
        b_kpu = kpu.Tensor(b_np)
        y_kpu = kpu.conv2d(x_kpu, w_kpu, b_kpu, stride=stride, padding=padding)

        # PyTorch reference
        x_pt = torch.tensor(x_np)
        w_pt = torch.tensor(w_np)
        b_pt = torch.tensor(b_np)
        y_pt = F.conv2d(x_pt, w_pt, b_pt, stride=stride, padding=padding)

        max_diff = np.max(np.abs(y_kpu.numpy() - y_pt.numpy()))
        passed = max_diff < 1e-5

        status = "PASS" if passed else "FAIL"
        print(f"  {x_shape} * {w_shape}: {status} (diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    assert all_passed, "Some conv2d PyTorch tests failed"


def test_pooling_against_numpy():
    """Test pooling operations against NumPy reference."""
    print("\n" + "=" * 60)
    print("Test: pooling against NumPy reference")
    print("=" * 60)

    kpu.set_fidelity(kpu.BEHAVIORAL)
    np.random.seed(42)

    all_passed = True

    # Test max_pool2d
    test_cases = [
        ((1, 1, 8, 8), 2, 2),
        ((2, 16, 14, 14), 2, 2),
        ((1, 32, 7, 7), 3, 1),
    ]

    for x_shape, kernel_size, stride in test_cases:
        x_np = np.random.randn(*x_shape).astype(np.float32)

        # KPU result
        x_kpu = kpu.Tensor(x_np)
        y_kpu = kpu.max_pool2d(x_kpu, kernel_size=kernel_size, stride=stride)

        # NumPy reference
        y_np = NumpyReference.max_pool2d(x_np, kernel_size, stride)

        max_diff = np.max(np.abs(y_kpu.numpy() - y_np))
        passed = max_diff < 1e-5

        status = "PASS" if passed else "FAIL"
        print(f"  max_pool2d {x_shape} k={kernel_size} s={stride}: {status} (diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    assert all_passed, "Some pooling tests failed"


def test_layer_norm_against_numpy():
    """Test layer_norm against NumPy reference."""
    print("\n" + "=" * 60)
    print("Test: layer_norm against NumPy reference")
    print("=" * 60)

    kpu.set_fidelity(kpu.BEHAVIORAL)
    np.random.seed(42)

    all_passed = True

    test_cases = [
        ((2, 128), 128),
        ((4, 16, 32), 32),
        ((1, 64, 7, 7), (7, 7)),
    ]

    for x_shape, normalized_shape in test_cases:
        x_np = np.random.randn(*x_shape).astype(np.float32)

        # KPU result
        x_kpu = kpu.Tensor(x_np)
        y_kpu = kpu.layer_norm(x_kpu, normalized_shape=normalized_shape)

        # NumPy reference
        y_np = NumpyReference.layer_norm(x_np, normalized_shape)

        max_diff = np.max(np.abs(y_kpu.numpy() - y_np))
        passed = max_diff < 1e-4

        status = "PASS" if passed else "FAIL"
        print(f"  layer_norm {x_shape} norm={normalized_shape}: {status} (diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    assert all_passed, "Some layer_norm tests failed"


def test_full_cnn_pipeline():
    """Test a full CNN pipeline matching MNIST classifier."""
    print("\n" + "=" * 60)
    print("Test: Full CNN pipeline")
    print("=" * 60)

    kpu.set_fidelity(kpu.BEHAVIORAL)
    np.random.seed(42)

    # Define CNN
    @kpu.compile
    def mnist_cnn(x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b):
        # Conv1: [N, 1, 28, 28] -> [N, 16, 26, 26]
        h = kpu.relu(kpu.conv2d(x, conv1_w) + conv1_b.reshape(1, -1, 1, 1))
        # Pool1: [N, 16, 26, 26] -> [N, 16, 13, 13]
        h = kpu.max_pool2d(h, kernel_size=2, stride=2)
        # Conv2: [N, 16, 13, 13] -> [N, 32, 11, 11]
        h = kpu.relu(kpu.conv2d(h, conv2_w) + conv2_b.reshape(1, -1, 1, 1))
        # Pool2: [N, 32, 11, 11] -> [N, 32, 5, 5]
        h = kpu.max_pool2d(h, kernel_size=2, stride=2)
        # Flatten: [N, 32, 5, 5] -> [N, 800]
        h = h.reshape(h.shape[0], -1)
        # FC1: [N, 800] -> [N, 128]
        h = kpu.relu(h @ fc1_w + fc1_b)
        # FC2: [N, 128] -> [N, 10]
        return h @ fc2_w + fc2_b

    # Create weights
    conv1_w = kpu.Tensor(np.random.randn(16, 1, 3, 3).astype(np.float32) * 0.1)
    conv1_b = kpu.Tensor(np.zeros(16, dtype=np.float32))
    conv2_w = kpu.Tensor(np.random.randn(32, 16, 3, 3).astype(np.float32) * 0.1)
    conv2_b = kpu.Tensor(np.zeros(32, dtype=np.float32))
    fc1_w = kpu.Tensor(np.random.randn(800, 128).astype(np.float32) * 0.1)
    fc1_b = kpu.Tensor(np.zeros(128, dtype=np.float32))
    fc2_w = kpu.Tensor(np.random.randn(128, 10).astype(np.float32) * 0.1)
    fc2_b = kpu.Tensor(np.zeros(10, dtype=np.float32))

    # Create input batch
    batch_size = 4
    x = kpu.Tensor(np.random.randn(batch_size, 1, 28, 28).astype(np.float32))

    # Run CNN
    output = mnist_cnn(x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b)

    # Verify output shape
    assert output.shape == (batch_size, 10), f"Expected (4, 10), got {output.shape}"
    print(f"  Output shape: {output.shape} - PASS")

    # Verify DFX generation
    dfx = mnist_cnn.get_dfx()
    op_types = set(op.opcode.value for op in dfx.ops)
    expected_ops = {'conv2d', 'relu', 'maxpool2d', 'reshape', 'matmul', 'add'}
    assert expected_ops.issubset(op_types), f"Missing ops: {expected_ops - op_types}"
    print(f"  DFX ops: {sorted(op_types)} - PASS")

    # Run NumPy reference for comparison
    def numpy_cnn(x, weights):
        h = NumpyReference.conv2d(x, weights['conv1_w'], weights['conv1_b'])
        h = np.maximum(h, 0)
        h = NumpyReference.max_pool2d(h, 2, 2)
        h = NumpyReference.conv2d(h, weights['conv2_w'], weights['conv2_b'])
        h = np.maximum(h, 0)
        h = NumpyReference.max_pool2d(h, 2, 2)
        h = h.reshape(h.shape[0], -1)
        h = np.maximum(h @ weights['fc1_w'] + weights['fc1_b'], 0)
        return h @ weights['fc2_w'] + weights['fc2_b']

    weights = {
        'conv1_w': conv1_w.numpy(), 'conv1_b': conv1_b.numpy(),
        'conv2_w': conv2_w.numpy(), 'conv2_b': conv2_b.numpy(),
        'fc1_w': fc1_w.numpy(), 'fc1_b': fc1_b.numpy(),
        'fc2_w': fc2_w.numpy(), 'fc2_b': fc2_b.numpy(),
    }
    expected = numpy_cnn(x.numpy(), weights)

    max_diff = np.max(np.abs(output.numpy() - expected))
    passed = max_diff < 1e-4
    status = "PASS" if passed else "FAIL"
    print(f"  NumPy reference match: {status} (max diff={max_diff:.2e})")

    assert passed, f"Full CNN pipeline failed with max diff {max_diff:.2e}"


def test_traced_vs_direct_execution():
    """Test that traced execution matches direct execution."""
    print("\n" + "=" * 60)
    print("Test: Traced vs direct execution")
    print("=" * 60)

    kpu.set_fidelity(kpu.BEHAVIORAL)
    np.random.seed(42)

    # Create inputs
    x_np = np.random.randn(2, 8, 16, 16).astype(np.float32)
    w_np = np.random.randn(16, 8, 3, 3).astype(np.float32)
    b_np = np.random.randn(16).astype(np.float32)

    x = kpu.Tensor(x_np)
    w = kpu.Tensor(w_np)
    b = kpu.Tensor(b_np)

    # Direct execution (not traced)
    direct = kpu.relu(kpu.conv2d(x, w, b, padding=1))

    # Traced execution
    @kpu.compile
    def traced_fn(x, w, b):
        return kpu.relu(kpu.conv2d(x, w, b, padding=1))

    traced = traced_fn(x, w, b)

    max_diff = np.max(np.abs(direct.numpy() - traced.numpy()))
    passed = max_diff < 1e-6

    status = "PASS" if passed else "FAIL"
    print(f"  Direct vs traced: {status} (max diff={max_diff:.2e})")

    assert passed, f"Traced vs direct execution failed with max diff {max_diff:.2e}"


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("KPU CNN Validation Suite")
    print("=" * 60)

    if HAS_PYTORCH:
        print(f"PyTorch: v{torch.__version__}")
    else:
        print("PyTorch: Not installed (using NumPy reference only)")

    results = []

    results.append(("conv2d vs NumPy", test_conv2d_against_numpy()))
    results.append(("conv2d vs PyTorch", test_conv2d_against_pytorch()))
    results.append(("pooling vs NumPy", test_pooling_against_numpy()))
    results.append(("layer_norm vs NumPy", test_layer_norm_against_numpy()))
    results.append(("full CNN pipeline", test_full_cnn_pipeline()))
    results.append(("traced vs direct", test_traced_vs_direct_execution()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
