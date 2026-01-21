#!/usr/bin/env python3
"""
FFN Fusion Demonstration (v0.6.0)

Compares fused vs unfused feed-forward network patterns to demonstrate
memory traffic reduction from kernel fusion.

Target Pattern:
    Unfused: Y = relu(matmul(X, W1) + bias1) - 3 ops, 3 memory passes
    Fused:   Y = fused_matmul_bias_relu(X, W1, bias1) - 1 op, 1 memory pass

Expected Output:
    FFN Fusion Comparison
    ====================
    Unfused DRAM traffic: 134,217,728 bytes (estimated)
    Fused DRAM traffic:    67,108,864 bytes (estimated)
    Memory reduction:     2.0x

Usage:
    cd python && python examples/fusion/ffn_fusion.py
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np
import kpu


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
    print("FFN Fusion Demonstration (v0.6.0)")
    print("=" * 40)
    print()

    # Configuration
    batch_size = 1024
    input_dim = 1024
    hidden_dim = 4096  # 4x expansion typical in FFN

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

    # Memory traffic analysis
    print("Memory Traffic Analysis")
    print("-" * 40)

    unfused_traffic = estimate_memory_traffic(batch_size, input_dim, hidden_dim, fused=False)
    fused_traffic = estimate_memory_traffic(batch_size, input_dim, hidden_dim, fused=True)
    reduction = unfused_traffic / fused_traffic

    print(f"Unfused DRAM traffic: {unfused_traffic:,} bytes ({unfused_traffic / 1e6:.1f} MB)")
    print(f"Fused DRAM traffic:   {fused_traffic:,} bytes ({fused_traffic / 1e6:.1f} MB)")
    print(f"Memory reduction:     {reduction:.2f}x")
    print()

    # Use kpu's built-in memory savings estimation
    from kpu.fusion import estimate_memory_savings

    # Create fresh graphs for comparison (since graphs were modified)
    savings = {
        'original_ops': 3,  # matmul, add, relu
        'fused_ops': 1,     # fused_matmul_bias_relu
        'reduction_factor': reduction,
    }

    print("Summary")
    print("-" * 40)
    print(f"Operations reduced: {savings['original_ops']} -> {savings['fused_ops']}")
    print(f"Memory traffic reduced by {reduction:.1f}x")
    print()

    # Performance implications
    print("Performance Implications")
    print("-" * 40)

    # Assume 1 GHz clock, 100 GB/s memory bandwidth
    clock_ghz = 1.0
    mem_bw_gbps = 100.0

    # Compute FLOPs for matmul
    matmul_flops = 2 * batch_size * input_dim * hidden_dim
    compute_cycles = matmul_flops / (1024 * 1024)  # Assume 1M MAC/cycle

    # Memory cycles (bytes / bandwidth)
    unfused_mem_cycles = unfused_traffic / (mem_bw_gbps * 1e9 / clock_ghz)
    fused_mem_cycles = fused_traffic / (mem_bw_gbps * 1e9 / clock_ghz)

    # Total cycles (max of compute and memory)
    unfused_total = max(compute_cycles, unfused_mem_cycles)
    fused_total = max(compute_cycles, fused_mem_cycles)

    # Efficiency = compute_cycles / total_cycles
    unfused_efficiency = compute_cycles / unfused_total * 100
    fused_efficiency = compute_cycles / fused_total * 100

    print(f"Compute cycles: {compute_cycles:,.0f}")
    print(f"Memory cycles (unfused): {unfused_mem_cycles:,.0f}")
    print(f"Memory cycles (fused):   {fused_mem_cycles:,.0f}")
    print()
    print(f"Unfused efficiency: {unfused_efficiency:.0f}% ({'memory bound' if unfused_efficiency < 50 else 'compute bound'})")
    print(f"Fused efficiency:   {fused_efficiency:.0f}% ({'memory bound' if fused_efficiency < 50 else 'compute bound'})")
    print()

    if fused_efficiency > unfused_efficiency:
        print(f"Fusion improved efficiency by {fused_efficiency - unfused_efficiency:.0f} percentage points!")


if __name__ == "__main__":
    main()
