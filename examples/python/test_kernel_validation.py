#!/usr/bin/env python3
"""
Kernel Validation Suite
Verifies kernel creation and metadata correctness

Run from repository root:
    python3 examples/python/test_kernel_validation.py
"""
import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '../../build/src/bindings/python/Release')
sys.path.insert(0, build_dir)

import stillwater_kpu as kpu

def test_matmul_kernel():
    """Test matmul kernel creation"""
    test_cases = [
        (64, 64, 64),
        (128, 256, 512),
        (1024, 1024, 1024),
        (32, 128, 64),
    ]

    for M, N, K in test_cases:
        kernel = kpu.Kernel.create_matmul(M, N, K)

        # Verify dimensions
        assert kernel.M() == M, f"M mismatch: {kernel.M()} != {M}"
        assert kernel.N() == N, f"N mismatch: {kernel.N()} != {N}"
        assert kernel.K() == K, f"K mismatch: {kernel.K()} != {K}"

        # Verify validity
        assert kernel.is_valid(), f"Kernel should be valid"

        # Verify FLOPs calculation (2*M*N*K for matmul)
        expected_flops = 2 * M * N * K
        assert kernel.total_flops() == expected_flops, \
            f"FLOPs mismatch: {kernel.total_flops()} != {expected_flops}"

        # Verify arguments
        args = kernel.arguments()
        assert len(args) == 3, f"Expected 3 arguments, got {len(args)}"

        arg_names = {arg.name for arg in args}
        assert arg_names == {"A", "B", "C"}, f"Unexpected arguments: {arg_names}"

        print(f"  PASS: matmul {M}x{N}x{K}")

    return True

def test_mlp_kernel():
    """Test MLP kernel creation with various activations"""
    activations = [
        kpu.ActivationType.NONE,
        kpu.ActivationType.RELU,
        kpu.ActivationType.GELU,
        kpu.ActivationType.SIGMOID,
        kpu.ActivationType.TANH,
        kpu.ActivationType.SILU,
    ]

    M, N, K = 64, 128, 256

    for act in activations:
        kernel = kpu.Kernel.create_mlp(M, N, K, act, has_bias=True)

        assert kernel.is_valid()
        assert kernel.op_type() == kpu.KernelOpType.MLP
        assert kernel.activation() == act
        assert kernel.has_bias() == True

        # MLP with bias has 4 arguments: A, B, bias, C
        args = kernel.arguments()
        assert len(args) == 4

        print(f"  PASS: MLP with {act}")

    return True

def test_mlp_without_bias():
    """Test MLP kernel without bias"""
    kernel = kpu.Kernel.create_mlp(64, 128, 256, kpu.ActivationType.RELU, has_bias=False)

    assert kernel.is_valid()
    assert kernel.has_bias() == False

    # MLP without bias has 3 arguments: A, B, C
    args = kernel.arguments()
    assert len(args) == 3

    print("  PASS: MLP without bias")
    return True

def test_kernel_validation():
    """Test kernel validation"""
    kernel = kpu.Kernel.create_matmul(64, 64, 64)
    valid, error = kernel.validate()
    assert valid, f"Valid kernel failed validation: {error}"
    print("  PASS: Kernel validation")
    return True

def test_data_types():
    """Test data type support"""
    test_dtypes = [
        (kpu.DataType.FLOAT32, 4, "float32"),
        (kpu.DataType.FLOAT16, 2, "float16"),
        (kpu.DataType.BFLOAT16, 2, "bfloat16"),
        (kpu.DataType.INT8, 1, "int8"),
        (kpu.DataType.INT4, 1, "int4"),
    ]

    for dtype, expected_size, expected_name in test_dtypes:
        assert kpu.dtype_size(dtype) == expected_size, \
            f"Size mismatch for {dtype}"
        assert kpu.dtype_name(dtype) == expected_name, \
            f"Name mismatch for {dtype}"
        print(f"  PASS: {expected_name} (size={expected_size})")

    return True

def test_kernel_summary():
    """Test kernel summary output"""
    kernel = kpu.Kernel.create_matmul(256, 256, 256)
    summary = kernel.summary()

    # Summary should contain key information
    assert "256" in summary  # Dimensions
    assert "matmul" in summary.lower() or "MATMUL" in summary

    print("  PASS: Kernel summary")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("  Kernel Validation Suite")
    print("=" * 60)

    tests = [
        ("Matmul Kernel Creation", test_matmul_kernel),
        ("MLP Kernel Creation", test_mlp_kernel),
        ("MLP Without Bias", test_mlp_without_bias),
        ("Kernel Validation", test_kernel_validation),
        ("Data Types", test_data_types),
        ("Kernel Summary", test_kernel_summary),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
