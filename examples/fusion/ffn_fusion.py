#!/usr/bin/env python3
"""
FFN Fusion Demonstration (v0.6.1)

Compares fused vs unfused feed-forward network patterns to demonstrate
memory traffic reduction and roofline analysis for kernel fusion.

Target Pattern:
    Unfused: Y = relu(matmul(X, W1) + bias1) - 3 ops, 3 memory passes
    Fused:   Y = fused_matmul_bias_relu(X, W1, bias1) - 1 op, 1 memory pass

v0.6.1 Features:
    - FusionAnalyzer for detecting fusion opportunities
    - Roofline analysis showing memory-bound vs compute-bound
    - Detailed performance metrics

Usage:
    cd python && python examples/fusion/ffn_fusion.py
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np
import kpu
from kpu.fusion import FusionAnalyzer, analyze_fusion_potential


def create_ffn_layer(batch_size: int, input_dim: int, hidden_dim: int):
    """Create inputs for a single FFN layer."""
    X = kpu.Tensor(np.random.randn(batch_size, input_dim).astype(np.float32))
    W = kpu.Tensor(np.random.randn(input_dim, hidden_dim).astype(np.float32))
    bias = kpu.Tensor(np.random.randn(hidden_dim).astype(np.float32))
    return X, W, bias


def ffn_unfused(x, w, bias):
    """Unfused FFN: matmul -> add -> relu (3 separate ops)."""
    y = x @ w
    y = y + bias
    y = kpu.relu(y)
    return y


def ffn_fused(x, w, bias):
    """Same operation but will be fused by compiler."""
    y = x @ w
    y = y + bias
    y = kpu.relu(y)
    return y


def estimate_memory_traffic(batch_size: int, input_dim: int, hidden_dim: int,
                           fused: bool) -> int:
    """
    Estimate DRAM memory traffic for FFN layer.

    Unfused (3 ops):
        1. MatMul: read X, W, write intermediate Y1
        2. Add: read Y1, bias, write intermediate Y2
        3. ReLU: read Y2, write output Y

    Fused (1 op):
        1. FusedMatMulBiasReLU: read X, W, bias, write output Y

    Returns bytes transferred.
    """
    bytes_per_element = 4  # float32

    # Input sizes
    X_bytes = batch_size * input_dim * bytes_per_element
    W_bytes = input_dim * hidden_dim * bytes_per_element
    bias_bytes = hidden_dim * bytes_per_element
    output_bytes = batch_size * hidden_dim * bytes_per_element

    if fused:
        # Fused: Read inputs once, write output once
        # No intermediate tensors written to/read from DRAM
        total = X_bytes + W_bytes + bias_bytes + output_bytes
    else:
        # Unfused: Each op reads inputs and writes outputs
        # Intermediate Y1 written by matmul, read by add
        # Intermediate Y2 written by add, read by relu
        intermediate_bytes = batch_size * hidden_dim * bytes_per_element

        # MatMul: read X + W, write Y1
        matmul_traffic = X_bytes + W_bytes + intermediate_bytes

        # Add: read Y1 + bias, write Y2
        add_traffic = intermediate_bytes + bias_bytes + intermediate_bytes

        # ReLU: read Y2, write output
        relu_traffic = intermediate_bytes + output_bytes

        total = matmul_traffic + add_traffic + relu_traffic

    return total


def main():
    print("FFN Fusion Demonstration (v0.6.1)")
    print("=" * 50)
    print()

    # Configuration - use smaller dimensions that result in memory-bound workload
    batch_size = 64
    input_dim = 256
    hidden_dim = 1024

    print(f"Configuration:")
    print(f"  Batch size:  {batch_size}")
    print(f"  Input dim:   {input_dim}")
    print(f"  Hidden dim:  {hidden_dim}")
    print()

    # Create test data
    X, W, bias = create_ffn_layer(batch_size, input_dim, hidden_dim)

    # Compile unfused version (optimization disabled)
    print("Compiling unfused version (optimize=False)...")
    unfused_fn = kpu.compile(ffn_unfused, optimize=False)
    result_unfused = unfused_fn(X, W, bias)
    unfused_graph = unfused_fn.graph

    print(f"  Unfused graph: {len(unfused_graph.nodes)} ops")
    for node in unfused_graph.topological_order():
        print(f"    - {node.op_type.value}")
    print()

    # Analyze fusion opportunities BEFORE fusing
    print("Fusion Analysis (v0.6.1)")
    print("-" * 50)

    # Use realistic hardware parameters:
    # - 1024 FLOPs/cycle (e.g., 32x32 systolic array)
    # - 64 bytes/cycle memory bandwidth
    analyzer = FusionAnalyzer(
        peak_flops_per_cycle=1024.0,
        peak_bytes_per_cycle=64.0,
    )
    report = analyzer.analyze(unfused_graph)

    print(f"Detected {report.total_fusions_possible} fusion opportunity:")
    for opp in report.opportunities:
        print(f"  - {opp.description}")
    print()

    # Compile fused version (optimization enabled - default)
    print("Compiling fused version (optimize=True)...")
    fused_fn = kpu.compile(ffn_fused, optimize=True)
    result_fused = fused_fn(X, W, bias)
    fused_graph = fused_fn.graph

    print(f"  Fused graph: {len(fused_graph.nodes)} ops")
    for node in fused_graph.topological_order():
        print(f"    - {node.op_type.value}")
    print()

    # Verify correctness
    print("Verifying correctness...")
    if np.allclose(result_unfused.numpy(), result_fused.numpy(), rtol=1e-5, atol=1e-5):
        print("  PASSED: Fused output matches unfused output")
    else:
        print("  FAILED: Outputs do not match!")
        max_diff = np.max(np.abs(result_unfused.numpy() - result_fused.numpy()))
        print(f"  Max difference: {max_diff}")
    print()

    # Roofline Analysis
    print("Roofline Analysis")
    print("-" * 50)
    print()

    print("Unfused Workload:")
    print(f"  Total FLOPs:          {report.roofline_unfused.total_flops:,}")
    print(f"  Total Memory:         {report.roofline_unfused.total_bytes:,} bytes")
    print(f"  Arithmetic Intensity: {report.roofline_unfused.arithmetic_intensity:.2f} FLOPs/byte")
    print(f"  Ridge Point:          {report.roofline_unfused.ridge_point:.2f} FLOPs/byte")
    print(f"  Bottleneck:           {report.roofline_unfused.bottleneck.upper()}")
    print(f"  Efficiency:           {report.roofline_unfused.efficiency:.1f}%")
    print()

    print("Fused Workload (estimated):")
    print(f"  Total FLOPs:          {report.roofline_fused.total_flops:,}")
    print(f"  Total Memory:         {report.roofline_fused.total_bytes:,} bytes")
    print(f"  Arithmetic Intensity: {report.roofline_fused.arithmetic_intensity:.2f} FLOPs/byte")
    print(f"  Bottleneck:           {report.roofline_fused.bottleneck.upper()}")
    print(f"  Efficiency:           {report.roofline_fused.efficiency:.1f}%")
    print()

    # Memory traffic analysis
    print("Memory Traffic Analysis")
    print("-" * 50)

    unfused_traffic = estimate_memory_traffic(batch_size, input_dim, hidden_dim, fused=False)
    fused_traffic = estimate_memory_traffic(batch_size, input_dim, hidden_dim, fused=True)
    reduction = unfused_traffic / fused_traffic

    print(f"Unfused DRAM traffic: {unfused_traffic:,} bytes ({unfused_traffic / 1e6:.1f} MB)")
    print(f"Fused DRAM traffic:   {fused_traffic:,} bytes ({fused_traffic / 1e6:.1f} MB)")
    print(f"Memory reduction:     {reduction:.2f}x")
    print()

    # Summary
    print("Summary")
    print("-" * 50)
    print(f"Operations reduced:     3 -> 1")
    print(f"Memory traffic:         {reduction:.1f}x reduction")
    print(f"Estimated speedup:      {report.estimated_speedup:.2f}x")
    print()

    # Success criteria validation
    print("Success Criteria Validation (ROADMAP.md)")
    print("-" * 50)

    criteria = [
        ("2x memory traffic reduction on FFN", reduction >= 2.0),
        ("Automatic fusion detection", report.total_fusions_possible >= 1),
        ("MatMul+Bias+ReLU fuses correctly", len(fused_graph.nodes) == 1),
        ("Unfused is memory bound", report.roofline_unfused.is_memory_bound),
        ("Fused has higher efficiency", report.roofline_fused.efficiency > report.roofline_unfused.efficiency),
    ]

    all_passed = True
    for description, passed in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {description}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All success criteria PASSED!")
    else:
        print("Some criteria failed - check configuration parameters.")


if __name__ == "__main__":
    main()
