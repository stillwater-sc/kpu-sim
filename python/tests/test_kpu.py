#!/usr/bin/env python3
"""
Test suite for KPU Python package.

Run with: python -m pytest tests/test_kpu.py -v
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import kpu


class TestTensor:
    """Tests for kpu.Tensor class."""

    def test_create_from_numpy(self):
        """Test creating tensor from numpy array."""
        arr = np.random.randn(32, 64).astype(np.float32)
        t = kpu.Tensor(arr)

        assert t.shape == (32, 64)
        assert t.dtype == np.float32
        assert not t.is_symbolic()
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_create_from_shape(self):
        """Test creating tensor from shape tuple."""
        t = kpu.Tensor((32, 64), dtype=np.float32, name="test")

        assert t.shape == (32, 64)
        assert t.dtype == np.float32
        assert t.name == "test"
        assert t.is_symbolic()

    def test_tensor_factories(self):
        """Test tensor factory methods."""
        t_zeros = kpu.Tensor.zeros((4, 4))
        t_ones = kpu.Tensor.ones((4, 4))
        t_randn = kpu.Tensor.randn(4, 4)

        assert t_zeros.shape == (4, 4)
        assert t_ones.shape == (4, 4)
        assert t_randn.shape == (4, 4)

        np.testing.assert_array_equal(t_zeros.numpy(), np.zeros((4, 4)))
        np.testing.assert_array_equal(t_ones.numpy(), np.ones((4, 4)))


class TestOperators:
    """Tests for KPU operators."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_matmul_basic(self):
        """Test basic matrix multiplication."""
        A = kpu.Tensor(np.random.randn(32, 64).astype(np.float32))
        B = kpu.Tensor(np.random.randn(64, 128).astype(np.float32))

        # Direct execution (not traced)
        C = A @ B

        expected = A.numpy() @ B.numpy()
        np.testing.assert_allclose(C.numpy(), expected, rtol=1e-5)

    def test_relu(self):
        """Test ReLU activation."""
        X = kpu.Tensor(np.array([[-1, 2], [3, -4]], dtype=np.float32))
        Y = kpu.relu(X)

        expected = np.array([[0, 2], [3, 0]], dtype=np.float32)
        np.testing.assert_array_equal(Y.numpy(), expected)

    def test_gelu(self):
        """Test GELU activation."""
        X = kpu.Tensor(np.array([0, 1, -1], dtype=np.float32))
        Y = kpu.gelu(X)

        # GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert Y.numpy()[0] == pytest.approx(0, abs=1e-5)
        assert Y.numpy()[1] == pytest.approx(0.841, abs=0.01)
        assert Y.numpy()[2] == pytest.approx(-0.159, abs=0.01)

    def test_sigmoid(self):
        """Test sigmoid activation."""
        X = kpu.Tensor(np.array([0, 1, -1], dtype=np.float32))
        Y = kpu.sigmoid(X)

        expected = 1 / (1 + np.exp(-X.numpy()))
        np.testing.assert_allclose(Y.numpy(), expected, rtol=1e-5)

    def test_softmax(self):
        """Test softmax activation."""
        X = kpu.Tensor(np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float32))
        Y = kpu.softmax(X)

        # Check that rows sum to 1
        row_sums = Y.numpy().sum(axis=-1)
        np.testing.assert_allclose(row_sums, [1, 1], rtol=1e-5)

    def test_elementwise_add(self):
        """Test elementwise addition."""
        A = kpu.Tensor(np.array([1, 2, 3], dtype=np.float32))
        B = kpu.Tensor(np.array([4, 5, 6], dtype=np.float32))

        C = A + B
        np.testing.assert_array_equal(C.numpy(), [5, 7, 9])

    def test_elementwise_mul(self):
        """Test elementwise multiplication."""
        A = kpu.Tensor(np.array([1, 2, 3], dtype=np.float32))
        B = kpu.Tensor(np.array([4, 5, 6], dtype=np.float32))

        C = A * B
        np.testing.assert_array_equal(C.numpy(), [4, 10, 18])

    def test_scalar_operations(self):
        """Test operations with scalars."""
        A = kpu.Tensor(np.array([1, 2, 3], dtype=np.float32))

        np.testing.assert_array_equal((A + 1).numpy(), [2, 3, 4])
        np.testing.assert_array_equal((A * 2).numpy(), [2, 4, 6])
        np.testing.assert_array_equal((A - 1).numpy(), [0, 1, 2])


class TestCompiler:
    """Tests for @kpu.compile decorator."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_simple_matmul(self):
        """Test compiling simple matmul."""
        @kpu.compile
        def simple_matmul(a, b):
            return a @ b

        A = kpu.Tensor(np.random.randn(16, 32).astype(np.float32))
        B = kpu.Tensor(np.random.randn(32, 64).astype(np.float32))

        C = simple_matmul(A, B)
        expected = A.numpy() @ B.numpy()

        np.testing.assert_allclose(C.numpy(), expected, rtol=1e-5)
        assert C.shape == (16, 64)

    def test_single_layer(self):
        """Test single layer: y = relu(x @ w + b)"""
        @kpu.compile
        def single_layer(x, w, b):
            return kpu.relu(x @ w + b)

        X = kpu.Tensor(np.random.randn(16, 784).astype(np.float32))
        W = kpu.Tensor(np.random.randn(784, 128).astype(np.float32))
        B = kpu.Tensor(np.zeros(128, dtype=np.float32))

        result = single_layer(X, W, B)
        expected = np.maximum(X.numpy() @ W.numpy() + B.numpy(), 0)

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_two_layer_mlp(self):
        """Test two-layer MLP."""
        @kpu.compile
        def two_layer_mlp(x, w1, w2):
            h = kpu.relu(x @ w1)
            return h @ w2

        X = kpu.Tensor(np.random.randn(8, 64).astype(np.float32))
        W1 = kpu.Tensor(np.random.randn(64, 32).astype(np.float32))
        W2 = kpu.Tensor(np.random.randn(32, 10).astype(np.float32))

        result = two_layer_mlp(X, W1, W2)

        h = np.maximum(X.numpy() @ W1.numpy(), 0)
        expected = h @ W2.numpy()

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_graph_generation(self):
        """Test that graph is correctly generated."""
        @kpu.compile
        def simple_net(x, w):
            return kpu.relu(x @ w)

        X = kpu.Tensor(np.zeros((4, 8), dtype=np.float32))
        W = kpu.Tensor(np.zeros((8, 16), dtype=np.float32))

        _ = simple_net(X, W)

        graph = simple_net.get_graph()
        assert graph is not None
        assert len(graph.nodes) == 2  # matmul + relu

    def test_dfx_generation(self):
        """Test that DFX IR is correctly generated."""
        @kpu.compile
        def simple_net(x, w):
            return kpu.relu(x @ w)

        X = kpu.Tensor(np.zeros((4, 8), dtype=np.float32))
        W = kpu.Tensor(np.zeros((8, 16), dtype=np.float32))

        _ = simple_net(X, W)

        dfx = simple_net.get_dfx()
        assert dfx is not None

        dfx_dict = dfx.to_dict()
        assert 'ops' in dfx_dict
        assert len(dfx_dict['ops']) == 2  # matmul + relu

        op_types = [op['opcode'] for op in dfx_dict['ops']]
        assert 'matmul' in op_types
        assert 'relu' in op_types


class TestMNISTMLP:
    """Integration test for MNIST MLP."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_full_mnist_mlp(self):
        """Test complete MNIST MLP network."""
        @kpu.compile
        def mnist_mlp(x, w1, b1, w2, b2, w3, b3):
            h1 = kpu.relu(x @ w1 + b1)
            h2 = kpu.relu(h1 @ w2 + b2)
            return h2 @ w3 + b3

        np.random.seed(42)
        batch_size = 8

        X = kpu.Tensor(np.random.randn(batch_size, 784).astype(np.float32))
        W1 = kpu.Tensor(np.random.randn(784, 128).astype(np.float32) * 0.01)
        B1 = kpu.Tensor(np.zeros(128, dtype=np.float32))
        W2 = kpu.Tensor(np.random.randn(128, 64).astype(np.float32) * 0.01)
        B2 = kpu.Tensor(np.zeros(64, dtype=np.float32))
        W3 = kpu.Tensor(np.random.randn(64, 10).astype(np.float32) * 0.01)
        B3 = kpu.Tensor(np.zeros(10, dtype=np.float32))

        result = mnist_mlp(X, W1, B1, W2, B2, W3, B3)

        # NumPy reference
        h1 = np.maximum(X.numpy() @ W1.numpy() + B1.numpy(), 0)
        h2 = np.maximum(h1 @ W2.numpy() + B2.numpy(), 0)
        expected = h2 @ W3.numpy() + B3.numpy()

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
        assert result.shape == (batch_size, 10)

    def test_xor_classifier(self):
        """Test XOR classifier (2 -> 4 -> 1)."""
        @kpu.compile
        def xor_net(x, w1, b1, w2, b2):
            h = kpu.relu(x @ w1 + b1)
            return h @ w2 + b2

        # XOR training data
        X = kpu.Tensor(np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], dtype=np.float32))

        # Pre-trained weights for XOR
        W1 = kpu.Tensor(np.array([
            [1, 1, -1, -1],
            [1, -1, 1, -1]
        ], dtype=np.float32))
        B1 = kpu.Tensor(np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32))

        W2 = kpu.Tensor(np.array([[1], [1], [1], [1]], dtype=np.float32))
        B2 = kpu.Tensor(np.array([-1.5], dtype=np.float32))

        result = xor_net(X, W1, B1, W2, B2)

        # Reference
        h = np.maximum(X.numpy() @ W1.numpy() + B1.numpy(), 0)
        expected = h @ W2.numpy() + B2.numpy()

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestDFXEmitter:
    """Tests for DFX IR emission."""

    def test_dfx_serialization(self):
        """Test DFX program serialization to JSON."""
        @kpu.compile
        def test_fn(x, w):
            return x @ w

        X = kpu.Tensor(np.zeros((4, 8), dtype=np.float32))
        W = kpu.Tensor(np.zeros((8, 16), dtype=np.float32))

        _ = test_fn(X, W)

        dfx = test_fn.get_dfx()
        json_str = dfx.to_json()

        # Parse and verify
        import json
        parsed = json.loads(json_str)

        assert parsed['name'] == 'test_fn'
        assert 'tensors' in parsed
        assert 'ops' in parsed
        assert 'inputs' in parsed
        assert 'outputs' in parsed

    def test_dfx_deserialization(self):
        """Test DFX program deserialization from JSON."""
        @kpu.compile
        def test_fn(x, w):
            return kpu.relu(x @ w)

        X = kpu.Tensor(np.zeros((4, 8), dtype=np.float32))
        W = kpu.Tensor(np.zeros((8, 16), dtype=np.float32))

        _ = test_fn(X, W)

        dfx = test_fn.get_dfx()
        json_str = dfx.to_json()

        # Deserialize
        from kpu.dfx_emitter import DFXProgram
        restored = DFXProgram.from_json(json_str)

        assert restored.name == dfx.name
        assert len(restored.ops) == len(dfx.ops)
        assert restored.inputs == dfx.inputs
        assert restored.outputs == dfx.outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
