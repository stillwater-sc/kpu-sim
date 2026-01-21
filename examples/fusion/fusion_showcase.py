#!/usr/bin/env python3
"""
Kernel Fusion Showcase (v0.6.3)

Comprehensive demonstration of kernel fusion with:
- Energy estimation based on memory hierarchy
- Multiple fusion patterns (FFN, Conv2D)
- Direct control over fusion (enable/disable patterns)
- Quantified performance and energy savings
- Scalability analysis across problem sizes

Energy Model (approximate pJ/op for 7nm):
    DRAM access:      ~20 pJ/byte
    L3 (SRAM) access: ~2 pJ/byte
    L2 (SRAM) access: ~0.5 pJ/byte
    L1 (Reg) access:  ~0.1 pJ/byte
    FP32 MAC:         ~1 pJ/op

Key Insight: Memory access dominates energy in unfused kernels.
Fusion keeps intermediate data in registers/L1, avoiding expensive DRAM writes.

Usage:
    PYTHONPATH=python python examples/fusion/fusion_showcase.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np
import kpu
from kpu.fusion import (
    FusionCompiler,
    FusionAnalyzer,
    MatMulBiasActivation,
    MatMulActivation,
    Conv2DBatchNormActivation,
    Conv2DActivation,
)
from dataclasses import dataclass
from typing import List, Tuple


# =============================================================================
# Energy Model
# =============================================================================

@dataclass
class EnergyModel:
    """Energy costs for memory hierarchy (pJ/byte or pJ/op)."""
    dram_pj_per_byte: float = 20.0      # DRAM access
    l3_pj_per_byte: float = 2.0         # L3 SRAM
    l2_pj_per_byte: float = 0.5         # L2 SRAM
    l1_pj_per_byte: float = 0.1         # L1/Registers
    mac_pj_per_op: float = 1.0          # FP32 multiply-accumulate

    def estimate_unfused_energy(self,
                                compute_flops: int,
                                input_bytes: int,
                                weight_bytes: int,
                                intermediate_bytes: int,
                                output_bytes: int) -> dict:
        """
        Estimate energy for unfused kernel chain.

        Unfused: Each intermediate tensor written to and read from DRAM.
        """
        # Compute energy
        compute_energy = compute_flops * self.mac_pj_per_op

        # Memory energy (all through DRAM for unfused)
        # Inputs: read once from DRAM
        input_energy = input_bytes * self.dram_pj_per_byte
        weight_energy = weight_bytes * self.dram_pj_per_byte

        # Intermediates: written and read from DRAM (2x traffic)
        intermediate_energy = intermediate_bytes * 2 * self.dram_pj_per_byte

        # Output: written to DRAM
        output_energy = output_bytes * self.dram_pj_per_byte

        total_memory = input_energy + weight_energy + intermediate_energy + output_energy
        total = compute_energy + total_memory

        return {
            'compute_pj': compute_energy,
            'memory_pj': total_memory,
            'total_pj': total,
            'breakdown': {
                'input': input_energy,
                'weight': weight_energy,
                'intermediate': intermediate_energy,
                'output': output_energy,
            }
        }

    def estimate_fused_energy(self,
                              compute_flops: int,
                              input_bytes: int,
                              weight_bytes: int,
                              output_bytes: int) -> dict:
        """
        Estimate energy for fused kernel.

        Fused: Intermediates stay in registers/L1, only final output to DRAM.
        """
        # Compute energy (same as unfused)
        compute_energy = compute_flops * self.mac_pj_per_op

        # Memory energy (no intermediate DRAM traffic)
        input_energy = input_bytes * self.dram_pj_per_byte
        weight_energy = weight_bytes * self.dram_pj_per_byte
        output_energy = output_bytes * self.dram_pj_per_byte

        total_memory = input_energy + weight_energy + output_energy
        total = compute_energy + total_memory

        return {
            'compute_pj': compute_energy,
            'memory_pj': total_memory,
            'total_pj': total,
            'breakdown': {
                'input': input_energy,
                'weight': weight_energy,
                'intermediate': 0,  # Kept in registers!
                'output': output_energy,
            }
        }


# =============================================================================
# Demo Functions
# =============================================================================

def demo_ffn_fusion(energy_model: EnergyModel):
    """Demonstrate FFN (Feed-Forward Network) fusion."""
    print("\n" + "=" * 70)
    print("DEMO 1: Feed-Forward Network (FFN) Fusion")
    print("Pattern: MatMul -> Bias -> ReLU => FUSED_MATMUL_BIAS_RELU")
    print("=" * 70)

    # Configuration
    batch_size = 256
    input_dim = 256
    hidden_dim = 1024

    print(f"\nConfiguration:")
    print(f"  Batch:  {batch_size}")
    print(f"  Input:  {input_dim}")
    print(f"  Hidden: {hidden_dim}")

    # Create tensors
    X = kpu.Tensor(np.random.randn(batch_size, input_dim).astype(np.float32))
    W = kpu.Tensor(np.random.randn(input_dim, hidden_dim).astype(np.float32))
    bias = kpu.Tensor(np.random.randn(hidden_dim).astype(np.float32))

    # Define the FFN layer
    def ffn(x, w, b):
        y = x @ w
        y = y + b
        y = kpu.relu(y)
        return y

    # Compile WITHOUT fusion
    print("\n--- Unfused Execution ---")
    unfused_fn = kpu.compile(ffn, optimize=False)
    result_unfused = unfused_fn(X, W, bias)
    print(f"  Graph ops: {[n.op_type.value for n in unfused_fn.graph.nodes]}")

    # Compile WITH fusion
    print("\n--- Fused Execution ---")
    fused_fn = kpu.compile(ffn, optimize=True)
    result_fused = fused_fn(X, W, bias)
    print(f"  Graph ops: {[n.op_type.value for n in fused_fn.graph.nodes]}")

    # Verify correctness
    correct = np.allclose(result_unfused.numpy(), result_fused.numpy(), rtol=1e-5)
    print(f"\n  Correctness: {'PASS' if correct else 'FAIL'}")

    # Calculate metrics
    bytes_per_elem = 4
    input_bytes = batch_size * input_dim * bytes_per_elem
    weight_bytes = input_dim * hidden_dim * bytes_per_elem
    bias_bytes = hidden_dim * bytes_per_elem
    output_bytes = batch_size * hidden_dim * bytes_per_elem
    intermediate_bytes = batch_size * hidden_dim * bytes_per_elem
    compute_flops = 2 * batch_size * input_dim * hidden_dim  # MatMul FLOPs

    # Energy estimation
    unfused_energy = energy_model.estimate_unfused_energy(
        compute_flops, input_bytes + bias_bytes, weight_bytes,
        intermediate_bytes * 2,  # 2 intermediates (after matmul, after add)
        output_bytes
    )
    fused_energy = energy_model.estimate_fused_energy(
        compute_flops, input_bytes + bias_bytes, weight_bytes, output_bytes
    )

    # Print results
    print("\n--- Performance & Energy Analysis ---")
    print(f"\n  Memory Traffic:")
    unfused_traffic = input_bytes + weight_bytes + bias_bytes + 4*intermediate_bytes + output_bytes
    fused_traffic = input_bytes + weight_bytes + bias_bytes + output_bytes
    print(f"    Unfused: {unfused_traffic:,} bytes ({unfused_traffic/1e6:.2f} MB)")
    print(f"    Fused:   {fused_traffic:,} bytes ({fused_traffic/1e6:.2f} MB)")
    print(f"    Reduction: {unfused_traffic/fused_traffic:.2f}x")

    print(f"\n  Energy Consumption:")
    print(f"    Unfused: {unfused_energy['total_pj']/1e9:.3f} mJ")
    print(f"      - Compute: {unfused_energy['compute_pj']/1e9:.3f} mJ ({100*unfused_energy['compute_pj']/unfused_energy['total_pj']:.1f}%)")
    print(f"      - Memory:  {unfused_energy['memory_pj']/1e9:.3f} mJ ({100*unfused_energy['memory_pj']/unfused_energy['total_pj']:.1f}%)")
    print(f"    Fused:   {fused_energy['total_pj']/1e9:.3f} mJ")
    print(f"      - Compute: {fused_energy['compute_pj']/1e9:.3f} mJ ({100*fused_energy['compute_pj']/fused_energy['total_pj']:.1f}%)")
    print(f"      - Memory:  {fused_energy['memory_pj']/1e9:.3f} mJ ({100*fused_energy['memory_pj']/fused_energy['total_pj']:.1f}%)")
    print(f"    Energy Savings: {unfused_energy['total_pj']/fused_energy['total_pj']:.2f}x")

    return {
        'pattern': 'FFN (MatMul+Bias+ReLU)',
        'traffic_reduction': unfused_traffic/fused_traffic,
        'energy_reduction': unfused_energy['total_pj']/fused_energy['total_pj'],
    }


def demo_conv2d_fusion(energy_model: EnergyModel):
    """Demonstrate Conv2D fusion."""
    print("\n" + "=" * 70)
    print("DEMO 2: Convolutional Network Fusion")
    print("Pattern: Conv2D -> BatchNorm -> ReLU => FUSED_CONV2D_BN_RELU")
    print("=" * 70)

    # Configuration (smaller for faster behavioral runtime)
    # Note: Energy/traffic analysis uses larger sizes for realistic estimates
    batch_size = 4
    in_channels = 16
    out_channels = 32
    height, width = 8, 8
    kernel_size = 3

    print(f"\nConfiguration:")
    print(f"  Batch:    {batch_size}")
    print(f"  Input:    {in_channels}x{height}x{width}")
    print(f"  Output:   {out_channels}x{height}x{width}")
    print(f"  Kernel:   {kernel_size}x{kernel_size}")

    # Create tensors
    X = kpu.Tensor(np.random.randn(batch_size, in_channels, height, width).astype(np.float32))
    W = kpu.Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32))

    # Define the conv block
    def conv_block(x, w):
        y = kpu.conv2d(x, w, padding=(1, 1))
        y = kpu.batch_norm2d(y)
        y = kpu.relu(y)
        return y

    # Compile WITHOUT fusion
    print("\n--- Unfused Execution ---")
    unfused_fn = kpu.compile(conv_block, optimize=False)
    result_unfused = unfused_fn(X, W)
    print(f"  Graph ops: {[n.op_type.value for n in unfused_fn.graph.nodes]}")

    # Compile WITH fusion
    print("\n--- Fused Execution ---")
    fused_fn = kpu.compile(conv_block, optimize=True)
    result_fused = fused_fn(X, W)
    print(f"  Graph ops: {[n.op_type.value for n in fused_fn.graph.nodes]}")

    # Verify correctness
    correct = np.allclose(result_unfused.numpy(), result_fused.numpy(), rtol=1e-4, atol=1e-4)
    print(f"\n  Correctness: {'PASS' if correct else 'FAIL'}")

    # Calculate metrics using REALISTIC sizes for energy analysis
    # (execution uses smaller sizes for speed, analysis uses practical sizes)
    analysis_batch = 16
    analysis_in_ch = 64
    analysis_out_ch = 128
    analysis_h, analysis_w = 32, 32
    analysis_k = 3

    print(f"\n  (Energy analysis uses realistic size: {analysis_batch}x{analysis_in_ch}x{analysis_h}x{analysis_w})")

    bytes_per_elem = 4
    input_bytes = analysis_batch * analysis_in_ch * analysis_h * analysis_w * bytes_per_elem
    weight_bytes = analysis_out_ch * analysis_in_ch * analysis_k * analysis_k * bytes_per_elem
    output_bytes = analysis_batch * analysis_out_ch * analysis_h * analysis_w * bytes_per_elem
    intermediate_bytes = output_bytes  # Same shape as output

    # Conv2D FLOPs: 2 * batch * out_channels * out_H * out_W * in_channels * kH * kW
    compute_flops = 2 * analysis_batch * analysis_out_ch * analysis_h * analysis_w * analysis_in_ch * analysis_k * analysis_k

    # Energy estimation
    unfused_energy = energy_model.estimate_unfused_energy(
        compute_flops, input_bytes, weight_bytes,
        intermediate_bytes * 2,  # after conv, after bn
        output_bytes
    )
    fused_energy = energy_model.estimate_fused_energy(
        compute_flops, input_bytes, weight_bytes, output_bytes
    )

    # Print results
    print("\n--- Performance & Energy Analysis ---")
    print(f"\n  Memory Traffic:")
    unfused_traffic = input_bytes + weight_bytes + 4*intermediate_bytes + output_bytes
    fused_traffic = input_bytes + weight_bytes + output_bytes
    print(f"    Unfused: {unfused_traffic:,} bytes ({unfused_traffic/1e6:.2f} MB)")
    print(f"    Fused:   {fused_traffic:,} bytes ({fused_traffic/1e6:.2f} MB)")
    print(f"    Reduction: {unfused_traffic/fused_traffic:.2f}x")

    print(f"\n  Energy Consumption:")
    print(f"    Unfused: {unfused_energy['total_pj']/1e9:.3f} mJ")
    print(f"    Fused:   {fused_energy['total_pj']/1e9:.3f} mJ")
    print(f"    Energy Savings: {unfused_energy['total_pj']/fused_energy['total_pj']:.2f}x")

    return {
        'pattern': 'Conv2D+BN+ReLU',
        'traffic_reduction': unfused_traffic/fused_traffic,
        'energy_reduction': unfused_energy['total_pj']/fused_energy['total_pj'],
    }


def demo_direct_fusion_control():
    """Demonstrate how to control fusion directly."""
    print("\n" + "=" * 70)
    print("DEMO 3: Direct Fusion Control")
    print("How to enable/disable specific fusion patterns")
    print("=" * 70)

    # Create a function with multiple fusible patterns
    def multi_pattern(x, w1, b1, w2, b2):
        # Pattern 1: MatMul + Bias + ReLU
        h = x @ w1
        h = h + b1
        h = kpu.relu(h)
        # Pattern 2: MatMul + ReLU (no bias)
        h = h @ w2
        h = kpu.relu(h)
        return h

    X = kpu.Tensor(np.random.randn(32, 64).astype(np.float32))
    W1 = kpu.Tensor(np.random.randn(64, 128).astype(np.float32))
    b1 = kpu.Tensor(np.random.randn(128).astype(np.float32))
    W2 = kpu.Tensor(np.random.randn(128, 64).astype(np.float32))
    b2 = kpu.Tensor(np.random.randn(64).astype(np.float32))

    print("\n--- Method 1: Default (all patterns enabled) ---")
    fn_default = kpu.compile(multi_pattern, optimize=True)
    _ = fn_default(X, W1, b1, W2, b2)
    print(f"  Ops: {[n.op_type.value for n in fn_default.graph.nodes]}")

    print("\n--- Method 2: No fusion ---")
    fn_none = kpu.compile(multi_pattern, optimize=False)
    _ = fn_none(X, W1, b1, W2, b2)
    print(f"  Ops: {[n.op_type.value for n in fn_none.graph.nodes]}")

    print("\n--- Method 3: Custom pattern selection ---")
    print("  (Using FusionCompiler with specific patterns)")

    # Compile without optimization first to get the graph
    fn_base = kpu.compile(multi_pattern, optimize=False)
    _ = fn_base(X, W1, b1, W2, b2)
    graph = fn_base.graph

    # Apply only MatMulBiasActivation pattern (not MatMulActivation)
    compiler = FusionCompiler(patterns=[MatMulBiasActivation()])
    optimized_graph = compiler.optimize(graph)
    print(f"  With only MatMulBiasActivation: {compiler.num_fusions} fusions")
    print(f"  Ops: {[n.op_type.value for n in optimized_graph.nodes]}")

    print("\n--- API Summary ---")
    print("  kpu.compile(fn, optimize=True)   # Enable all fusion patterns")
    print("  kpu.compile(fn, optimize=False)  # Disable all fusion")
    print("  FusionCompiler(patterns=[...])   # Custom pattern selection")
    print("  Available patterns:")
    print("    - MatMulBiasActivation: matmul -> add -> activation")
    print("    - MatMulActivation: matmul -> activation")
    print("    - Conv2DBatchNormActivation: conv2d -> bn -> activation")
    print("    - Conv2DActivation: conv2d -> activation")


def demo_scalability_sweep(energy_model: EnergyModel):
    """Demonstrate fusion benefits across different problem sizes."""
    print("\n" + "=" * 70)
    print("DEMO 4: Scalability Analysis")
    print("Fusion benefits across different problem sizes")
    print("=" * 70)

    # Different sizes to test
    sizes = [
        (32, 64, 128),     # Small
        (64, 128, 256),    # Medium-small
        (128, 256, 512),   # Medium
        (256, 256, 1024),  # Medium-large (our main demo)
        (512, 512, 2048),  # Large
    ]

    print("\n  Size (B×I×H)      | Traffic Red. | Energy Red. | Notes")
    print("  " + "-" * 66)

    results = []
    for batch, input_dim, hidden_dim in sizes:
        # Create tensors
        X = kpu.Tensor(np.random.randn(batch, input_dim).astype(np.float32))
        W = kpu.Tensor(np.random.randn(input_dim, hidden_dim).astype(np.float32))
        bias = kpu.Tensor(np.random.randn(hidden_dim).astype(np.float32))

        def ffn(x, w, b):
            return kpu.relu(x @ w + b)

        # Calculate metrics
        bytes_per_elem = 4
        input_bytes = batch * input_dim * bytes_per_elem
        weight_bytes = input_dim * hidden_dim * bytes_per_elem
        bias_bytes = hidden_dim * bytes_per_elem
        output_bytes = batch * hidden_dim * bytes_per_elem
        intermediate_bytes = batch * hidden_dim * bytes_per_elem
        compute_flops = 2 * batch * input_dim * hidden_dim

        unfused_traffic = input_bytes + weight_bytes + bias_bytes + 4*intermediate_bytes + output_bytes
        fused_traffic = input_bytes + weight_bytes + bias_bytes + output_bytes
        traffic_reduction = unfused_traffic / fused_traffic

        unfused_energy = energy_model.estimate_unfused_energy(
            compute_flops, input_bytes + bias_bytes, weight_bytes,
            intermediate_bytes * 2, output_bytes
        )
        fused_energy = energy_model.estimate_fused_energy(
            compute_flops, input_bytes + bias_bytes, weight_bytes, output_bytes
        )
        energy_reduction = unfused_energy['total_pj'] / fused_energy['total_pj']

        # Determine if memory-bound
        ai = compute_flops / unfused_traffic
        note = "mem-bound" if ai < 32 else "compute-bound"

        print(f"  {batch:3d}×{input_dim:3d}×{hidden_dim:4d}     |    {traffic_reduction:5.2f}x     |    {energy_reduction:5.2f}x    | {note}")

        results.append({
            'size': f"{batch}×{input_dim}×{hidden_dim}",
            'traffic_reduction': traffic_reduction,
            'energy_reduction': energy_reduction,
        })

    print("\n  Key Insight: Fusion benefits increase with batch size because")
    print("  intermediate tensor traffic (which is eliminated) grows with batch,")
    print("  while weight traffic (which remains) is independent of batch size.")

    return results


def print_summary(ffn_results, conv_results, scale_results):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("SUMMARY: Kernel Fusion Benefits")
    print("=" * 70)

    print("\n  Pattern                  | Traffic Red. | Energy Red.")
    print("  " + "-" * 55)
    print(f"  {ffn_results['pattern']:<24} |    {ffn_results['traffic_reduction']:5.2f}x     |    {ffn_results['energy_reduction']:5.2f}x")
    print(f"  {conv_results['pattern']:<24} |    {conv_results['traffic_reduction']:5.2f}x     |    {conv_results['energy_reduction']:5.2f}x")

    print("\n  Key Takeaways:")
    print("  1. Fusion eliminates intermediate DRAM traffic (2-3x reduction)")
    print("  2. Energy savings often exceed performance gains (memory dominates)")
    print("  3. Benefits scale with batch size and activation map size")
    print("  4. Fusion is automatic with optimize=True (default)")
    print("  5. Custom patterns available via FusionCompiler API")

    print("\n  Energy Breakdown (why fusion matters):")
    print("    DRAM access:  ~20 pJ/byte  (expensive!)")
    print("    L3 access:    ~2 pJ/byte")
    print("    L1/Reg:       ~0.1 pJ/byte (where fused intermediates live)")
    print("    FP32 MAC:     ~1 pJ/op")
    print()


def main():
    print("=" * 70)
    print("Kernel Fusion Showcase (v0.6.3)")
    print("Demonstrating performance and energy improvements from fusion")
    print("=" * 70)

    # Energy model (7nm approximation)
    energy_model = EnergyModel()

    # Run demos
    ffn_results = demo_ffn_fusion(energy_model)
    conv_results = demo_conv2d_fusion(energy_model)
    demo_direct_fusion_control()
    scale_results = demo_scalability_sweep(energy_model)

    # Summary
    print_summary(ffn_results, conv_results, scale_results)


if __name__ == "__main__":
    main()
