#!/usr/bin/env python3
"""
Test Transformer Operations (v0.6)

These tests verify that transformer operations (softmax, layer_norm, attention)
use the C++ BehavioralComputeFabric and NOT the Python/NumPy fallback.

To run these tests:
    cd python && python -m pytest tests/test_transformer_ops.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kpu


class TestSoftmax:
    """Test softmax operation via C++ backend."""

    def setup_method(self):
        """Reset state before each test."""
        kpu.set_strict_native(False)
        kpu.set_fidelity(kpu.BEHAVIORAL)

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_softmax_basic(self):
        """Softmax should compute correct values via C++."""
        @kpu.compile
        def softmax_fn(x):
            return kpu.softmax(x, axis=-1)

        x_data = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        x = kpu.Tensor(x_data)

        result = softmax_fn(x)
        stats = softmax_fn.stats

        # Verify execution backend
        assert stats.execution_backend == "cpp_behavioral", \
            f"Expected cpp_behavioral, got {stats.execution_backend}"

        # Verify result shape
        assert result.shape == x.shape

        # Verify softmax properties: each row sums to 1
        result_np = result.numpy()
        np.testing.assert_allclose(result_np.sum(axis=-1), [1.0, 1.0], rtol=1e-5)

        # For uniform input, softmax should give uniform distribution
        np.testing.assert_allclose(result_np[1], [1/3, 1/3, 1/3], rtol=1e-5)

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_softmax_numerical_stability(self):
        """Softmax should handle large values numerically stable."""
        @kpu.compile
        def softmax_fn(x):
            return kpu.softmax(x, axis=-1)

        # Large values that would overflow without max subtraction
        x_data = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        x = kpu.Tensor(x_data)

        result = softmax_fn(x)
        result_np = result.numpy()

        # Should still sum to 1
        np.testing.assert_allclose(result_np.sum(axis=-1), [1.0], rtol=1e-5)

        # Should not have NaN or Inf
        assert not np.isnan(result_np).any()
        assert not np.isinf(result_np).any()


class TestLayerNorm:
    """Test layer normalization via C++ backend."""

    def setup_method(self):
        """Reset state before each test."""
        kpu.set_strict_native(False)
        kpu.set_fidelity(kpu.BEHAVIORAL)

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_layer_norm_basic(self):
        """Layer norm should normalize to zero mean and unit variance."""
        @kpu.compile
        def layer_norm_fn(x, weight, bias):
            return kpu.layer_norm(x, normalized_shape=(4,), weight=weight, bias=bias)

        x_data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)

        x = kpu.Tensor(x_data)
        w = kpu.Tensor(weight)
        b = kpu.Tensor(bias)

        result = layer_norm_fn(x, w, b)
        stats = layer_norm_fn.stats

        # Verify execution backend
        assert stats.execution_backend == "cpp_behavioral", \
            f"Expected cpp_behavioral, got {stats.execution_backend}"

        # Verify result shape
        assert result.shape == x.shape

        # Verify normalization: each row should have ~zero mean and ~unit variance
        result_np = result.numpy()
        for row in result_np:
            np.testing.assert_allclose(row.mean(), 0.0, atol=1e-5)
            np.testing.assert_allclose(row.std(), 1.0, atol=1e-2)  # Some tolerance for small N

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_layer_norm_with_affine(self):
        """Layer norm should apply weight and bias correctly."""
        @kpu.compile
        def layer_norm_fn(x, weight, bias):
            return kpu.layer_norm(x, normalized_shape=(4,), weight=weight, bias=bias)

        x_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        weight = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)  # Scale by 2
        bias = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)    # Shift by 1

        x = kpu.Tensor(x_data)
        w = kpu.Tensor(weight)
        b = kpu.Tensor(bias)

        result = layer_norm_fn(x, w, b)
        result_np = result.numpy()

        # After scaling and shifting, mean should be ~1 (bias)
        # Standard deviation should be ~2 (weight)
        np.testing.assert_allclose(result_np.mean(), 1.0, atol=1e-5)


class TestAttention:
    """Test scaled dot-product attention via C++ backend."""

    def setup_method(self):
        """Reset state before each test."""
        kpu.set_strict_native(False)
        kpu.set_fidelity(kpu.BEHAVIORAL)

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_attention_basic(self):
        """Attention should compute Q @ K^T @ V correctly."""
        @kpu.compile
        def attention_fn(q, k, v):
            return kpu.scaled_dot_product_attention(q, k, v)

        # Simple attention: 2 sequence positions, 4-dimensional keys/queries
        seq_len = 2
        d_k = 4
        d_v = 4

        q_data = np.random.randn(seq_len, d_k).astype(np.float32)
        k_data = np.random.randn(seq_len, d_k).astype(np.float32)
        v_data = np.random.randn(seq_len, d_v).astype(np.float32)

        q = kpu.Tensor(q_data)
        k = kpu.Tensor(k_data)
        v = kpu.Tensor(v_data)

        result = attention_fn(q, k, v)
        stats = attention_fn.stats

        # Verify execution backend
        assert stats.execution_backend == "cpp_behavioral", \
            f"Expected cpp_behavioral, got {stats.execution_backend}"

        # Verify result shape: [seq_len, d_v]
        assert result.shape == (seq_len, d_v)

        # Verify FLOPs were recorded (attention involves 2 matmuls)
        assert stats.matmul_count >= 2, \
            f"Expected at least 2 matmuls for attention, got {stats.matmul_count}"

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_attention_batched(self):
        """Batched attention should work correctly."""
        @kpu.compile
        def attention_fn(q, k, v):
            return kpu.scaled_dot_product_attention(q, k, v)

        batch = 2
        heads = 4
        seq_len = 8
        d_k = 16

        q_data = np.random.randn(batch, heads, seq_len, d_k).astype(np.float32)
        k_data = np.random.randn(batch, heads, seq_len, d_k).astype(np.float32)
        v_data = np.random.randn(batch, heads, seq_len, d_k).astype(np.float32)

        q = kpu.Tensor(q_data)
        k = kpu.Tensor(k_data)
        v = kpu.Tensor(v_data)

        result = attention_fn(q, k, v)
        stats = attention_fn.stats

        # Verify execution backend
        assert stats.execution_backend == "cpp_behavioral", \
            f"Expected cpp_behavioral, got {stats.execution_backend}"

        # Verify result shape: [batch, heads, seq_len, d_k]
        assert result.shape == (batch, heads, seq_len, d_k)

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_attention_uniform_weights(self):
        """When Q = K, attention should produce uniform weights."""
        @kpu.compile
        def attention_fn(q, k, v):
            return kpu.scaled_dot_product_attention(q, k, v)

        seq_len = 4
        d_k = 8

        # Uniform queries and keys - should give uniform attention
        q_data = np.ones((seq_len, d_k), dtype=np.float32)
        k_data = np.ones((seq_len, d_k), dtype=np.float32)
        # V = identity-like for easy verification
        v_data = np.eye(seq_len, dtype=np.float32)[:, :seq_len]
        # Pad to d_k if needed
        v_padded = np.zeros((seq_len, d_k), dtype=np.float32)
        v_padded[:, :seq_len] = v_data

        q = kpu.Tensor(q_data)
        k = kpu.Tensor(k_data)
        v = kpu.Tensor(v_padded)

        result = attention_fn(q, k, v)
        result_np = result.numpy()

        # With uniform Q and K, attention weights should be uniform (1/seq_len)
        # So output should be average of V rows
        expected_first_cols = np.mean(v_padded[:, :seq_len], axis=0)
        np.testing.assert_allclose(result_np[0, :seq_len], expected_first_cols, rtol=1e-4)


class TestFusedMatMulOps:
    """Test fused matmul operations via C++ backend."""

    def setup_method(self):
        """Reset state before each test."""
        kpu.set_strict_native(False)
        kpu.set_fidelity(kpu.BEHAVIORAL)

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_fused_matmul_bias_relu(self):
        """Fused matmul + bias + relu should work correctly."""
        @kpu.compile
        def mlp_layer(x, w, b):
            h = kpu.matmul(x, w)
            h = h + b
            return kpu.relu(h)

        x_data = np.random.randn(4, 8).astype(np.float32)
        w_data = np.random.randn(8, 4).astype(np.float32)
        b_data = np.random.randn(4).astype(np.float32)

        x = kpu.Tensor(x_data)
        w = kpu.Tensor(w_data)
        b = kpu.Tensor(b_data)

        result = mlp_layer(x, w, b)
        stats = mlp_layer.stats

        # Verify execution backend
        assert stats.execution_backend == "cpp_behavioral", \
            f"Expected cpp_behavioral, got {stats.execution_backend}"

        # Verify result shape
        assert result.shape == (4, 4)

        # Verify result correctness
        expected = np.maximum(0, x_data @ w_data + b_data)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_fused_matmul_gelu(self):
        """Matmul + GELU should work correctly."""
        @kpu.compile
        def gelu_layer(x, w):
            h = kpu.matmul(x, w)
            return kpu.gelu(h)

        x_data = np.random.randn(4, 8).astype(np.float32)
        w_data = np.random.randn(8, 4).astype(np.float32)

        x = kpu.Tensor(x_data)
        w = kpu.Tensor(w_data)

        result = gelu_layer(x, w)
        stats = gelu_layer.stats

        # Verify execution backend
        assert stats.execution_backend == "cpp_behavioral", \
            f"Expected cpp_behavioral, got {stats.execution_backend}"

        # Verify result shape
        assert result.shape == (4, 4)


class TestTransformerMLP:
    """Test complete transformer MLP block via C++."""

    def setup_method(self):
        """Reset state before each test."""
        kpu.set_strict_native(False)
        kpu.set_fidelity(kpu.BEHAVIORAL)

    @pytest.mark.skipif(not kpu.is_native_available(),
                        reason="Native module not available")
    def test_transformer_mlp_block(self):
        """Complete MLP block: linear -> gelu -> linear."""
        @kpu.compile
        def mlp_block(x, w1, w2):
            h = kpu.matmul(x, w1)
            h = kpu.gelu(h)
            return kpu.matmul(h, w2)

        hidden_dim = 64
        intermediate_dim = 256

        x_data = np.random.randn(8, hidden_dim).astype(np.float32)
        w1_data = np.random.randn(hidden_dim, intermediate_dim).astype(np.float32)
        w2_data = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32)

        x = kpu.Tensor(x_data)
        w1 = kpu.Tensor(w1_data)
        w2 = kpu.Tensor(w2_data)

        result = mlp_block(x, w1, w2)
        stats = mlp_block.stats

        # Verify execution backend
        assert stats.execution_backend == "cpp_behavioral", \
            f"Expected cpp_behavioral, got {stats.execution_backend}"

        # Verify result shape
        assert result.shape == (8, hidden_dim)

        # Verify FLOPs were recorded for matmul operations
        # XUE summary should show matmul events
        xue = stats.xue_summary
        assert xue is not None, "XUE summary should be present"
        compute = xue.get('compute_breakdown', {})
        matmul_events = compute.get('matmul_events', 0)
        assert matmul_events > 0, \
            f"Expected matmul events in XUE, got {matmul_events}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
