#!/usr/bin/env python3
"""
Test suite for KPU Kernel Fusion (v0.6.0+).

Tests pattern detection, graph rewriting, and behavioral correctness
of fused operations.

Run with: cd python && python -m pytest tests/test_fusion.py -v
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import kpu
from kpu.graph import OpType, OpGraph
from kpu.fusion import (
    FusionCompiler,
    FusionGroup,
    MatMulBiasActivation,
    MatMulActivation,
    estimate_memory_savings,
    # v0.6.1
    FusionAnalyzer,
    FusionReport,
    FusionOpportunity,
    RooflineMetrics,
    analyze_fusion_potential,
    # v0.6.2
    Conv2DBatchNormActivation,
    Conv2DActivation,
)


class TestFusionPatternDetection:
    """Tests for fusion pattern matching."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_matmul_bias_relu_detection(self):
        """Test detection of MatMul + Bias + ReLU pattern."""
        @kpu.compile(optimize=False)
        def unfused(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        # Trace to get graph
        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        _ = unfused(X, W, bias)
        graph = unfused.graph

        # Run fusion compiler
        compiler = FusionCompiler()
        optimized = compiler.optimize(graph)

        # Should have fused to 1 operation
        assert compiler.num_fusions == 1
        assert 'MatMulBiasActivation' in compiler.fusion_stats

        # Check the fused op type
        fused_nodes = [n for n in optimized.nodes if n.op_type.is_fused()]
        assert len(fused_nodes) == 1
        assert fused_nodes[0].op_type == OpType.FUSED_MATMUL_BIAS_RELU

    def test_matmul_bias_gelu_detection(self):
        """Test detection of MatMul + Bias + GELU pattern."""
        @kpu.compile(optimize=False)
        def unfused(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.gelu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        _ = unfused(X, W, bias)
        graph = unfused.graph

        compiler = FusionCompiler()
        optimized = compiler.optimize(graph)

        assert compiler.num_fusions == 1
        fused_nodes = [n for n in optimized.nodes if n.op_type.is_fused()]
        assert len(fused_nodes) == 1
        assert fused_nodes[0].op_type == OpType.FUSED_MATMUL_BIAS_GELU

    def test_matmul_relu_detection(self):
        """Test detection of MatMul + ReLU pattern (no bias)."""
        @kpu.compile(optimize=False)
        def unfused(x, w):
            y = x @ w
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))

        _ = unfused(X, W)
        graph = unfused.graph

        compiler = FusionCompiler()
        optimized = compiler.optimize(graph)

        assert compiler.num_fusions == 1
        assert 'MatMulActivation' in compiler.fusion_stats

        fused_nodes = [n for n in optimized.nodes if n.op_type.is_fused()]
        assert len(fused_nodes) == 1
        assert fused_nodes[0].op_type == OpType.FUSED_MATMUL_RELU

    def test_no_fusion_without_activation(self):
        """Test that MatMul alone is not fused."""
        @kpu.compile(optimize=False)
        def unfused(x, w):
            return x @ w

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))

        _ = unfused(X, W)
        graph = unfused.graph

        compiler = FusionCompiler()
        optimized = compiler.optimize(graph)

        # No fusions should occur
        assert compiler.num_fusions == 0

    def test_no_fusion_with_multiple_consumers(self):
        """Test that patterns with multiple consumers are not fused."""
        @kpu.compile(optimize=False)
        def unfused(x, w):
            y = x @ w
            # y is consumed by both relu and the return
            z = kpu.relu(y)
            return y + z  # y has two consumers, should not fuse

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))

        _ = unfused(X, W)
        graph = unfused.graph

        compiler = FusionCompiler()
        optimized = compiler.optimize(graph)

        # Should not fuse because matmul output has two consumers
        assert compiler.num_fusions == 0


class TestFusedOperationCorrectness:
    """Tests for behavioral correctness of fused operations."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_fused_matmul_bias_relu_correctness(self):
        """Test that fused MatMul+Bias+ReLU produces correct output."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(32, 64).astype(np.float32))
        W = kpu.Tensor(np.random.randn(64, 128).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(128).astype(np.float32))

        # Compile without optimization
        unfused_fn = kpu.compile(ffn, optimize=False)
        result_unfused = unfused_fn(X, W, bias)

        # Compile with optimization (will fuse)
        fused_fn = kpu.compile(ffn, optimize=True)
        result_fused = fused_fn(X, W, bias)

        # Outputs should match
        np.testing.assert_allclose(
            result_fused.numpy(),
            result_unfused.numpy(),
            rtol=1e-5,
            atol=1e-5
        )

    def test_fused_matmul_bias_gelu_correctness(self):
        """Test that fused MatMul+Bias+GELU produces correct output."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.gelu(y)
            return y

        X = kpu.Tensor(np.random.randn(16, 32).astype(np.float32))
        W = kpu.Tensor(np.random.randn(32, 64).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(64).astype(np.float32))

        unfused_fn = kpu.compile(ffn, optimize=False)
        result_unfused = unfused_fn(X, W, bias)

        fused_fn = kpu.compile(ffn, optimize=True)
        result_fused = fused_fn(X, W, bias)

        np.testing.assert_allclose(
            result_fused.numpy(),
            result_unfused.numpy(),
            rtol=1e-5,
            atol=1e-5
        )

    def test_fused_matmul_bias_silu_correctness(self):
        """Test that fused MatMul+Bias+SiLU produces correct output."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.silu(y)
            return y

        X = kpu.Tensor(np.random.randn(16, 32).astype(np.float32))
        W = kpu.Tensor(np.random.randn(32, 64).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(64).astype(np.float32))

        unfused_fn = kpu.compile(ffn, optimize=False)
        result_unfused = unfused_fn(X, W, bias)

        fused_fn = kpu.compile(ffn, optimize=True)
        result_fused = fused_fn(X, W, bias)

        np.testing.assert_allclose(
            result_fused.numpy(),
            result_unfused.numpy(),
            rtol=1e-5,
            atol=1e-5
        )

    def test_fused_matmul_relu_correctness(self):
        """Test that fused MatMul+ReLU produces correct output."""
        def layer(x, w):
            y = x @ w
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(32, 64).astype(np.float32))
        W = kpu.Tensor(np.random.randn(64, 128).astype(np.float32))

        unfused_fn = kpu.compile(layer, optimize=False)
        result_unfused = unfused_fn(X, W)

        fused_fn = kpu.compile(layer, optimize=True)
        result_fused = fused_fn(X, W)

        np.testing.assert_allclose(
            result_fused.numpy(),
            result_unfused.numpy(),
            rtol=1e-5,
            atol=1e-5
        )


class TestGraphRewriting:
    """Tests for graph rewriting during fusion."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_operation_count_reduction(self):
        """Test that fusion reduces operation count."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        # Unfused should have 3 ops: matmul, add, relu
        unfused_fn = kpu.compile(ffn, optimize=False)
        _ = unfused_fn(X, W, bias)
        assert len(unfused_fn.graph.nodes) == 3

        # Fused should have 1 op: fused_matmul_bias_relu
        fused_fn = kpu.compile(ffn, optimize=True)
        _ = fused_fn(X, W, bias)
        assert len(fused_fn.graph.nodes) == 1

    def test_multiple_fusions_in_sequence(self):
        """Test that multiple fusion opportunities are handled."""
        def two_layer_ffn(x, w1, b1, w2, b2):
            # First layer
            y = x @ w1
            y = y + b1
            y = kpu.relu(y)
            # Second layer
            y = y @ w2
            y = y + b2
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W1 = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        b1 = kpu.Tensor(np.random.randn(16).astype(np.float32))
        W2 = kpu.Tensor(np.random.randn(16, 32).astype(np.float32))
        b2 = kpu.Tensor(np.random.randn(32).astype(np.float32))

        # Unfused should have 6 ops
        unfused_fn = kpu.compile(two_layer_ffn, optimize=False)
        result_unfused = unfused_fn(X, W1, b1, W2, b2)
        assert len(unfused_fn.graph.nodes) == 6

        # Fused should have 2 ops (two fused matmul+bias+relu)
        fused_fn = kpu.compile(two_layer_ffn, optimize=True)
        result_fused = fused_fn(X, W1, b1, W2, b2)
        assert len(fused_fn.graph.nodes) == 2

        # Results should match
        np.testing.assert_allclose(
            result_fused.numpy(),
            result_unfused.numpy(),
            rtol=1e-5,
            atol=1e-5
        )

    def test_topological_order_preserved(self):
        """Test that topological order is correct after fusion."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        fused_fn = kpu.compile(ffn, optimize=True)
        _ = fused_fn(X, W, bias)

        # Get topological order
        topo_order = fused_fn.graph.topological_order()

        # Should be able to traverse without issues
        assert len(topo_order) > 0

        # All nodes should be reachable
        visited_ids = {n._id for n in topo_order}
        all_ids = {n._id for n in fused_fn.graph.nodes}
        assert visited_ids == all_ids


class TestConv2DFusionPatterns:
    """Tests for Conv2D fusion patterns (v0.6.2+)."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_conv2d_relu_detection(self):
        """Test detection of Conv2D + ReLU pattern."""
        @kpu.compile(optimize=False)
        def unfused(x, w):
            y = kpu.conv2d(x, w, padding=(1, 1))
            y = kpu.relu(y)
            return y

        # NCHW format: (batch, channels, height, width)
        X = kpu.Tensor(np.random.randn(1, 3, 8, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(16, 3, 3, 3).astype(np.float32))

        _ = unfused(X, W)
        graph = unfused.graph

        # Run fusion compiler
        compiler = FusionCompiler()
        optimized = compiler.optimize(graph)

        # Should have fused to 1 operation
        assert compiler.num_fusions == 1
        assert 'Conv2DActivation' in compiler.fusion_stats

        # Check the fused op type
        fused_nodes = [n for n in optimized.nodes if n.op_type.is_fused()]
        assert len(fused_nodes) == 1
        assert fused_nodes[0].op_type == OpType.FUSED_CONV2D_RELU

    def test_conv2d_bn_relu_detection(self):
        """Test detection of Conv2D + BatchNorm + ReLU pattern."""
        @kpu.compile(optimize=False)
        def unfused(x, w):
            y = kpu.conv2d(x, w, padding=(1, 1))
            y = kpu.batch_norm2d(y)
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(16, 3, 3, 3).astype(np.float32))

        _ = unfused(X, W)
        graph = unfused.graph

        compiler = FusionCompiler()
        optimized = compiler.optimize(graph)

        # Should have fused to 1 operation
        assert compiler.num_fusions == 1
        assert 'Conv2DBatchNormActivation' in compiler.fusion_stats

        fused_nodes = [n for n in optimized.nodes if n.op_type.is_fused()]
        assert len(fused_nodes) == 1
        assert fused_nodes[0].op_type == OpType.FUSED_CONV2D_BN_RELU

    def test_conv2d_relu_correctness(self):
        """Test that fused Conv2D+ReLU produces correct output."""
        def conv_block(x, w):
            y = kpu.conv2d(x, w, padding=(1, 1))
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(16, 3, 3, 3).astype(np.float32))

        # Compile without optimization
        unfused_fn = kpu.compile(conv_block, optimize=False)
        result_unfused = unfused_fn(X, W)

        # Compile with optimization (will fuse)
        fused_fn = kpu.compile(conv_block, optimize=True)
        result_fused = fused_fn(X, W)

        # Outputs should match
        np.testing.assert_allclose(
            result_fused.numpy(),
            result_unfused.numpy(),
            rtol=1e-5,
            atol=1e-5
        )

    def test_conv2d_bn_relu_correctness(self):
        """Test that fused Conv2D+BatchNorm+ReLU produces correct output."""
        def conv_block(x, w):
            y = kpu.conv2d(x, w, padding=(1, 1))
            y = kpu.batch_norm2d(y)
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(16, 3, 3, 3).astype(np.float32))

        unfused_fn = kpu.compile(conv_block, optimize=False)
        result_unfused = unfused_fn(X, W)

        fused_fn = kpu.compile(conv_block, optimize=True)
        result_fused = fused_fn(X, W)

        np.testing.assert_allclose(
            result_fused.numpy(),
            result_unfused.numpy(),
            rtol=1e-5,
            atol=1e-5
        )

    def test_conv2d_no_fusion_without_activation(self):
        """Test that Conv2D alone is not fused."""
        @kpu.compile(optimize=False)
        def unfused(x, w):
            return kpu.conv2d(x, w, padding=(1, 1))

        X = kpu.Tensor(np.random.randn(1, 3, 8, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(16, 3, 3, 3).astype(np.float32))

        _ = unfused(X, W)
        graph = unfused.graph

        compiler = FusionCompiler()
        optimized = compiler.optimize(graph)

        # No fusions should occur
        assert compiler.num_fusions == 0

    def test_analyzer_detects_conv2d_opportunities(self):
        """Test that FusionAnalyzer detects Conv2D fusion opportunities."""
        def conv_block(x, w):
            y = kpu.conv2d(x, w, padding=(1, 1))
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(1, 3, 8, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(16, 3, 3, 3).astype(np.float32))

        unfused_fn = kpu.compile(conv_block, optimize=False)
        _ = unfused_fn(X, W)

        analyzer = FusionAnalyzer()
        report = analyzer.analyze(unfused_fn.graph)

        # Should find 1 fusion opportunity
        assert report.total_fusions_possible == 1
        assert len(report.opportunities) == 1
        assert report.opportunities[0].pattern_name == 'Conv2DActivation'


class TestOpTypePredicates:
    """Tests for OpType predicate methods."""

    def test_is_fused(self):
        """Test is_fused() returns correct values."""
        assert OpType.FUSED_MATMUL_BIAS_RELU.is_fused()
        assert OpType.FUSED_MATMUL_BIAS_GELU.is_fused()
        assert OpType.FUSED_MATMUL_BIAS_SILU.is_fused()
        assert OpType.FUSED_MATMUL_RELU.is_fused()
        # v0.6.2 Conv2D fused ops
        assert OpType.FUSED_CONV2D_BN_RELU.is_fused()
        assert OpType.FUSED_CONV2D_RELU.is_fused()

        # Non-fused ops should return False
        assert not OpType.MATMUL.is_fused()
        assert not OpType.RELU.is_fused()
        assert not OpType.ADD.is_fused()

    def test_is_fused_conv(self):
        """Test is_fused_conv() returns correct values."""
        assert OpType.FUSED_CONV2D_BN_RELU.is_fused_conv()
        assert OpType.FUSED_CONV2D_RELU.is_fused_conv()

        # MatMul fused ops should return False
        assert not OpType.FUSED_MATMUL_BIAS_RELU.is_fused_conv()
        assert not OpType.FUSED_MATMUL_RELU.is_fused_conv()

    def test_is_fused_matmul(self):
        """Test is_fused_matmul() returns correct values."""
        assert OpType.FUSED_MATMUL_BIAS_RELU.is_fused_matmul()
        assert OpType.FUSED_MATMUL_BIAS_GELU.is_fused_matmul()
        assert OpType.FUSED_MATMUL_RELU.is_fused_matmul()

        # Conv2D fused ops should return False
        assert not OpType.FUSED_CONV2D_BN_RELU.is_fused_matmul()
        assert not OpType.FUSED_CONV2D_RELU.is_fused_matmul()


class TestDFXEmission:
    """Tests for DFX IR emission of fused ops."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_fused_op_in_dfx(self):
        """Test that fused ops appear correctly in DFX IR."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        fused_fn = kpu.compile(ffn, optimize=True)
        _ = fused_fn(X, W, bias)

        dfx_program = fused_fn.dfx
        assert dfx_program is not None

        # Should have fused op in DFX
        fused_ops = [
            op for op in dfx_program.ops
            if 'fused' in op.opcode.value
        ]
        assert len(fused_ops) == 1
        assert fused_ops[0].opcode.value == 'fused_matmul_bias_relu'

        # Check inputs: should have 3 (A, B, bias)
        assert len(fused_ops[0].inputs) == 3

    def test_dfx_shape_attributes(self):
        """Test that fused ops have correct shape attributes."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(32, 64).astype(np.float32))
        W = kpu.Tensor(np.random.randn(64, 128).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(128).astype(np.float32))

        fused_fn = kpu.compile(ffn, optimize=True)
        _ = fused_fn(X, W, bias)

        dfx_program = fused_fn.dfx
        fused_op = [op for op in dfx_program.ops if 'fused' in op.opcode.value][0]

        # Check M, K, N attributes
        assert fused_op.attrs.get('M') == 32
        assert fused_op.attrs.get('K') == 64
        assert fused_op.attrs.get('N') == 128


class TestMemorySavingsEstimation:
    """Tests for memory savings estimation."""

    def test_estimate_memory_savings(self):
        """Test memory savings estimation function."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        # Get both graphs
        unfused_fn = kpu.compile(ffn, optimize=False)
        _ = unfused_fn(X, W, bias)
        unfused_graph = unfused_fn.graph

        fused_fn = kpu.compile(ffn, optimize=True)
        _ = fused_fn(X, W, bias)
        fused_graph = fused_fn.graph

        savings = estimate_memory_savings(unfused_graph, fused_graph)

        assert savings['original_ops'] > savings['fused_ops']
        assert savings['reduction_factor'] > 1.0


class TestFusionAnalyzer:
    """Tests for FusionAnalyzer (v0.6.1)."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_analyzer_finds_opportunities(self):
        """Test that FusionAnalyzer detects fusion opportunities."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        # Compile without optimization to get unfused graph
        unfused_fn = kpu.compile(ffn, optimize=False)
        _ = unfused_fn(X, W, bias)

        # Analyze the unfused graph
        analyzer = FusionAnalyzer()
        report = analyzer.analyze(unfused_fn.graph)

        # Should find 1 fusion opportunity
        assert report.total_fusions_possible == 1
        assert len(report.opportunities) == 1
        assert report.opportunities[0].pattern_name == 'MatMulBiasActivation'

    def test_analyzer_with_multiple_opportunities(self):
        """Test analyzer with multiple fusion opportunities."""
        def two_layer_ffn(x, w1, b1, w2, b2):
            y = x @ w1
            y = y + b1
            y = kpu.relu(y)
            y = y @ w2
            y = y + b2
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W1 = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        b1 = kpu.Tensor(np.random.randn(16).astype(np.float32))
        W2 = kpu.Tensor(np.random.randn(16, 32).astype(np.float32))
        b2 = kpu.Tensor(np.random.randn(32).astype(np.float32))

        unfused_fn = kpu.compile(two_layer_ffn, optimize=False)
        _ = unfused_fn(X, W1, b1, W2, b2)

        analyzer = FusionAnalyzer()
        report = analyzer.analyze(unfused_fn.graph)

        # Should find 2 fusion opportunities
        assert report.total_fusions_possible == 2
        assert len(report.opportunities) == 2

    def test_analyzer_no_opportunities(self):
        """Test analyzer when no fusion opportunities exist."""
        def simple_matmul(x, w):
            return x @ w

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))

        unfused_fn = kpu.compile(simple_matmul, optimize=False)
        _ = unfused_fn(X, W)

        analyzer = FusionAnalyzer()
        report = analyzer.analyze(unfused_fn.graph)

        # No fusion opportunities (matmul alone)
        assert report.total_fusions_possible == 0


class TestRooflineAnalysis:
    """Tests for roofline analysis (v0.6.1)."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_roofline_metrics(self):
        """Test that roofline metrics are computed correctly."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        unfused_fn = kpu.compile(ffn, optimize=False)
        _ = unfused_fn(X, W, bias)

        analyzer = FusionAnalyzer(
            peak_flops_per_cycle=1024.0,
            peak_bytes_per_cycle=64.0,
        )
        report = analyzer.analyze(unfused_fn.graph)

        # Check roofline metrics are populated
        assert report.roofline_unfused.total_flops > 0
        assert report.roofline_unfused.total_bytes > 0
        assert report.roofline_unfused.arithmetic_intensity > 0
        assert report.roofline_unfused.ridge_point == 1024.0 / 64.0  # 16.0

    def test_memory_bound_detection(self):
        """Test detection of memory-bound workload."""
        # Small matmul = low arithmetic intensity = memory bound
        def small_ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        unfused_fn = kpu.compile(small_ffn, optimize=False)
        _ = unfused_fn(X, W, bias)

        # Use high ridge point to force memory-bound classification
        analyzer = FusionAnalyzer(
            peak_flops_per_cycle=10000.0,  # Very high compute
            peak_bytes_per_cycle=64.0,
        )
        report = analyzer.analyze(unfused_fn.graph)

        # With high compute capability, small workloads are memory-bound
        assert report.roofline_unfused.is_memory_bound
        assert not report.roofline_unfused.is_compute_bound
        assert report.roofline_unfused.efficiency < 100.0

    def test_fusion_improves_efficiency(self):
        """Test that fusion improves arithmetic intensity."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        unfused_fn = kpu.compile(ffn, optimize=False)
        _ = unfused_fn(X, W, bias)

        analyzer = FusionAnalyzer()
        report = analyzer.analyze(unfused_fn.graph)

        # Fused should have higher arithmetic intensity (same FLOPs, less bytes)
        assert report.roofline_fused.arithmetic_intensity >= report.roofline_unfused.arithmetic_intensity
        # Fused should have higher or equal efficiency
        assert report.roofline_fused.efficiency >= report.roofline_unfused.efficiency


class TestFusionReport:
    """Tests for FusionReport (v0.6.1)."""

    def setup_method(self):
        """Set up behavioral fidelity for each test."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

    def test_report_summary(self):
        """Test that FusionReport generates a summary."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        unfused_fn = kpu.compile(ffn, optimize=False)
        _ = unfused_fn(X, W, bias)

        report = analyze_fusion_potential(unfused_fn.graph)
        summary = report.summary()

        # Summary should contain key information
        assert 'Fusion Analysis Report' in summary
        assert 'Fusion Opportunities' in summary
        assert 'Roofline Analysis' in summary
        assert 'Arithmetic Intensity' in summary

    def test_analyze_fusion_potential_convenience(self):
        """Test the convenience function analyze_fusion_potential."""
        def ffn(x, w, bias):
            y = x @ w
            y = y + bias
            y = kpu.relu(y)
            return y

        X = kpu.Tensor(np.random.randn(4, 8).astype(np.float32))
        W = kpu.Tensor(np.random.randn(8, 16).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(16).astype(np.float32))

        unfused_fn = kpu.compile(ffn, optimize=False)
        _ = unfused_fn(X, W, bias)

        # Use the convenience function
        report = analyze_fusion_potential(
            unfused_fn.graph,
            peak_flops_per_cycle=512.0,
            peak_bytes_per_cycle=32.0,
        )

        assert isinstance(report, FusionReport)
        assert report.roofline_unfused.ridge_point == 512.0 / 32.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
