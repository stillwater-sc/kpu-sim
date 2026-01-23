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

        # rtol=1e-4 + atol=1e-4 handles both relative and absolute error from large matmuls (784x128)
        # Max absolute diff ~3e-5 observed in practice
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=1e-4)

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

        # rtol=1e-4 accounts for C++ vs NumPy floating point differences
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=1e-6)

    def test_graph_generation(self):
        """Test that graph is correctly generated."""
        # Use optimize=False to test unfused graph structure
        @kpu.compile(optimize=False)
        def simple_net(x, w):
            return kpu.relu(x @ w)

        X = kpu.Tensor(np.zeros((4, 8), dtype=np.float32))
        W = kpu.Tensor(np.zeros((8, 16), dtype=np.float32))

        _ = simple_net(X, W)

        graph = simple_net.get_graph()
        assert graph is not None
        assert len(graph.nodes) == 2  # matmul + relu (unfused)

    def test_dfx_generation(self):
        """Test that DFX IR is correctly generated."""
        # Use optimize=False to test unfused DFX generation
        @kpu.compile(optimize=False)
        def simple_net(x, w):
            return kpu.relu(x @ w)

        X = kpu.Tensor(np.zeros((4, 8), dtype=np.float32))
        W = kpu.Tensor(np.zeros((8, 16), dtype=np.float32))

        _ = simple_net(X, W)

        dfx = simple_net.get_dfx()
        assert dfx is not None

        dfx_dict = dfx.to_dict()
        assert 'ops' in dfx_dict
        assert len(dfx_dict['ops']) == 2  # matmul + relu (unfused)

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

        # rtol=1e-4 accounts for C++ vs NumPy floating point differences in deep networks
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=1e-6)
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


class TestCNNOperators:
    """Tests for CNN operators (conv2d, pooling, etc.)."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_conv2d_basic(self):
        """Test basic 2D convolution."""
        # Input: [batch=1, channels=1, height=4, width=4]
        x = kpu.Tensor(np.ones((1, 1, 4, 4), dtype=np.float32))
        # Kernel: [out_channels=1, in_channels=1, kh=3, kw=3]
        w = kpu.Tensor(np.ones((1, 1, 3, 3), dtype=np.float32))

        y = kpu.conv2d(x, w)

        # Output should be [1, 1, 2, 2] with all 9s (sum of 3x3 ones)
        assert y.shape == (1, 1, 2, 2)
        np.testing.assert_allclose(y.numpy(), np.full((1, 1, 2, 2), 9.0), rtol=1e-5)

    def test_conv2d_with_padding(self):
        """Test conv2d with padding."""
        x = kpu.Tensor(np.ones((1, 1, 4, 4), dtype=np.float32))
        w = kpu.Tensor(np.ones((1, 1, 3, 3), dtype=np.float32))

        y = kpu.conv2d(x, w, padding=1)

        # With padding=1, output should be same size as input
        assert y.shape == (1, 1, 4, 4)

    def test_conv2d_with_stride(self):
        """Test conv2d with stride."""
        x = kpu.Tensor(np.ones((1, 1, 8, 8), dtype=np.float32))
        w = kpu.Tensor(np.ones((2, 1, 3, 3), dtype=np.float32))

        y = kpu.conv2d(x, w, stride=2)

        # Stride=2 should halve spatial dimensions (approximately)
        assert y.shape == (1, 2, 3, 3)

    def test_max_pool2d(self):
        """Test 2D max pooling."""
        x_data = np.array([[[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]]], dtype=np.float32)
        x = kpu.Tensor(x_data)

        y = kpu.max_pool2d(x, kernel_size=2, stride=2)

        expected = np.array([[[[6, 8],
                               [14, 16]]]], dtype=np.float32)
        assert y.shape == (1, 1, 2, 2)
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_avg_pool2d(self):
        """Test 2D average pooling."""
        x_data = np.array([[[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]]], dtype=np.float32)
        x = kpu.Tensor(x_data)

        y = kpu.avg_pool2d(x, kernel_size=2, stride=2)

        # Each 2x2 region averaged
        expected = np.array([[[[3.5, 5.5],
                               [11.5, 13.5]]]], dtype=np.float32)
        assert y.shape == (1, 1, 2, 2)
        np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)

    def test_adaptive_avg_pool2d(self):
        """Test adaptive average pooling."""
        x = kpu.Tensor(np.ones((1, 1, 8, 8), dtype=np.float32))

        y = kpu.adaptive_avg_pool2d(x, output_size=(1, 1))

        assert y.shape == (1, 1, 1, 1)
        np.testing.assert_allclose(y.numpy(), np.ones((1, 1, 1, 1)), rtol=1e-5)

    def test_layer_norm(self):
        """Test layer normalization."""
        x = kpu.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))

        y = kpu.layer_norm(x, normalized_shape=3)

        # Each row should be normalized (mean=0, std=1)
        y_np = y.numpy()
        np.testing.assert_allclose(y_np.mean(axis=-1), [0, 0], atol=1e-5)
        np.testing.assert_allclose(y_np.std(axis=-1), [1, 1], atol=0.1)

    def test_concat(self):
        """Test tensor concatenation."""
        a = kpu.Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = kpu.Tensor(np.array([[5, 6], [7, 8]], dtype=np.float32))

        # Concat along dim 0
        c = kpu.concat([a, b], dim=0)
        assert c.shape == (4, 2)
        np.testing.assert_array_equal(c.numpy(), [[1, 2], [3, 4], [5, 6], [7, 8]])

        # Concat along dim 1
        d = kpu.concat([a, b], dim=1)
        assert d.shape == (2, 4)
        np.testing.assert_array_equal(d.numpy(), [[1, 2, 5, 6], [3, 4, 7, 8]])

    def test_reshape(self):
        """Test tensor reshape."""
        x = kpu.Tensor(np.arange(12, dtype=np.float32))

        y = kpu.reshape(x, (3, 4))
        assert y.shape == (3, 4)

        # Test with -1
        z = kpu.reshape(x, (2, -1))
        assert z.shape == (2, 6)

    def test_tensor_reshape_method(self):
        """Test Tensor.reshape() method."""
        x = kpu.Tensor(np.arange(12, dtype=np.float32))

        y = x.reshape(3, 4)
        assert y.shape == (3, 4)

        z = x.reshape((2, 6))
        assert z.shape == (2, 6)

    def test_flatten(self):
        """Test tensor flatten."""
        x = kpu.Tensor(np.ones((2, 3, 4), dtype=np.float32))

        y = kpu.flatten(x, start_dim=1)
        assert y.shape == (2, 12)

        z = x.flatten(start_dim=0)
        assert z.shape == (24,)


class TestSimpleCNN:
    """Integration test for simple CNN."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_simple_cnn(self):
        """Test simple Conv -> ReLU -> Pool -> FC architecture."""
        @kpu.compile(optimize=False)  # Disable fusion to test unfused op generation
        def simple_cnn(x, conv_w, fc_w, fc_b):
            # Conv: [1, 1, 8, 8] -> [1, 4, 6, 6]
            h = kpu.relu(kpu.conv2d(x, conv_w))
            # Pool: [1, 4, 6, 6] -> [1, 4, 3, 3]
            h = kpu.max_pool2d(h, kernel_size=2, stride=2)
            # Flatten: [1, 4, 3, 3] -> [1, 36]
            h = h.reshape(h.shape[0], -1)
            # FC: [1, 36] -> [1, 10]
            return h @ fc_w + fc_b

        np.random.seed(42)
        x = kpu.Tensor(np.random.randn(1, 1, 8, 8).astype(np.float32))
        conv_w = kpu.Tensor(np.random.randn(4, 1, 3, 3).astype(np.float32) * 0.1)
        fc_w = kpu.Tensor(np.random.randn(36, 10).astype(np.float32) * 0.1)
        fc_b = kpu.Tensor(np.zeros(10, dtype=np.float32))

        result = simple_cnn(x, conv_w, fc_w, fc_b)

        assert result.shape == (1, 10)
        # Verify DFX generation includes new ops
        dfx = simple_cnn.get_dfx()
        op_types = [op.opcode.value for op in dfx.ops]
        assert 'conv2d' in op_types
        assert 'relu' in op_types
        assert 'maxpool2d' in op_types
        assert 'reshape' in op_types
        assert 'matmul' in op_types


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
