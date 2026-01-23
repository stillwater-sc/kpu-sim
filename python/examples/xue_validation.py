#!/usr/bin/env python3
"""
XUE Pre/Post-Silicon Validation Workflow

This example demonstrates the XUE Observation Architecture for pre/post-silicon
validation.

XUE Methodology: X (Throughput) → U (Utilization) → E (Efficiency)
    - X = Measured throughput (work done per unit time)
    - U = Resource utilization (fraction of time resource is busy)
    - E = Efficiency (measured throughput / peak throughput for the period)

The progression X → U → E allows systematic drill-down on resource effectiveness:
    1. Compare measured throughput (X) to speed-of-light throughput
    2. If below threshold, measure utilization (U)
    3. If throughput isn't proportional to utilization, calculate efficiency (E)

The Observation Architecture provides event hierarchies that aggregate cleanly
without logic on the datapath, enabling this analysis methodology.

Usage:
    python xue_validation.py                  # Basic XUE analysis
    python xue_validation.py --transactional  # With transactional timing

Key Concepts:
    - Event Recording: Single-cycle counter increments, no conditional logic
    - Event Aggregation: Hierarchical summation via dedicated paths
    - Zero Datapath Impact: Observation never stalls the observed pipeline
    - Post-hoc Analysis: All complex analysis happens after data collection
"""

import sys
import os
import argparse

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kpu


def parse_args():
    parser = argparse.ArgumentParser(description="XUE Pre/Post-Silicon Validation")
    parser.add_argument("--transactional", action="store_true",
                        help="Use TRANSACTIONAL mode with timing simulation")
    parser.add_argument("--clock", type=float, default=1.0,
                        help="Clock frequency in GHz (default: 1.0)")
    parser.add_argument("--peak-gflops", type=float, default=512.0,
                        help="Peak GFLOPS for roofline model (default: 512.0)")
    parser.add_argument("--dram-bandwidth", type=float, default=64.0,
                        help="DRAM bandwidth in GB/s (default: 64.0)")
    return parser.parse_args()


@kpu.compile
def mlp_workload(x, w1, b1, w2, b2):
    """
    Two-layer MLP workload for XUE validation.

    Args:
        x: Input [batch, 784]
        w1, b1: Layer 1 weights [784, 256] and bias [256]
        w2, b2: Layer 2 weights [256, 10] and bias [10]

    Returns:
        Output logits [batch, 10]
    """
    h = kpu.relu(x @ w1 + b1)
    return h @ w2 + b2


def create_workload(batch_size=32):
    """Create test workload with known dimensions for validation."""
    np.random.seed(42)

    # Layer dimensions for FLOP verification
    input_dim = 784
    hidden_dim = 256
    output_dim = 10

    x = kpu.Tensor(np.random.randn(batch_size, input_dim).astype(np.float32), name="input")
    w1 = kpu.Tensor(np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01, name="w1")
    b1 = kpu.Tensor(np.zeros(hidden_dim, dtype=np.float32), name="b1")
    w2 = kpu.Tensor(np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01, name="w2")
    b2 = kpu.Tensor(np.zeros(output_dim, dtype=np.float32), name="b2")

    # Calculate expected FLOPs
    # MatMul 1: [batch, 784] @ [784, 256] = 2 * batch * 784 * 256 FLOPs
    # MatMul 2: [batch, 256] @ [256, 10] = 2 * batch * 256 * 10 FLOPs
    # Add bias: batch * 256 + batch * 10 FLOPs
    # ReLU: batch * 256 comparisons
    matmul1_flops = 2 * batch_size * input_dim * hidden_dim
    matmul2_flops = 2 * batch_size * hidden_dim * output_dim
    total_matmul_flops = matmul1_flops + matmul2_flops

    expected = {
        'batch_size': batch_size,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'matmul1_flops': matmul1_flops,
        'matmul2_flops': matmul2_flops,
        'total_matmul_flops': total_matmul_flops,
    }

    return (x, w1, b1, w2, b2), expected


def print_section(title):
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    args = parse_args()

    # Configure fidelity
    if args.transactional:
        kpu.set_fidelity(kpu.TRANSACTIONAL)
        kpu.set_clock_frequency(args.clock)
    else:
        kpu.set_fidelity(kpu.BEHAVIORAL)

    mode = "TRANSACTIONAL" if args.transactional else "BEHAVIORAL"

    print_section("XUE Pre/Post-Silicon Validation Workflow")
    print(f"  Mode:           {mode}")
    print(f"  Clock:          {args.clock:.1f} GHz")
    print(f"  Peak GFLOPS:    {args.peak_gflops:.1f}")
    print(f"  DRAM Bandwidth: {args.dram_bandwidth:.1f} GB/s")

    # Create workload
    inputs, expected = create_workload(batch_size=32)
    x, w1, b1, w2, b2 = inputs

    print_section("1. Workload Configuration")
    print(f"  Input shape:     [{expected['batch_size']}, {expected['input_dim']}]")
    print(f"  Hidden shape:    [{expected['batch_size']}, {expected['hidden_dim']}]")
    print(f"  Output shape:    [{expected['batch_size']}, {expected['output_dim']}]")
    print()
    print("  Expected FLOPs:")
    print(f"    MatMul 1: {expected['matmul1_flops']:,}")
    print(f"    MatMul 2: {expected['matmul2_flops']:,}")
    print(f"    Total:    {expected['total_matmul_flops']:,}")

    # Execute workload
    print_section("2. Execute Workload on Simulator")
    print(f"  Running MLP workload ({mode} mode)...")

    result = mlp_workload(x, w1, b1, w2, b2)
    print(f"  Output shape: {result.shape}")
    print(f"  Output dtype: {result.dtype}")

    # Get XUE summary from C++ (identical format to hardware counters)
    print_section("3. XUE Event Summary (from C++ EventCollector)")
    xue = kpu.get_xue_summary()

    print("  Event Categories:")
    for category, counts in xue.get('event_categories', {}).items():
        print(f"    {category}: {counts.get('events', 0):,} events, {counts.get('flops', 0):,} FLOPs")

    print()
    print("  Memory Hierarchy:")
    mem = xue.get('memory_hierarchy', {})
    for level in ['dram', 'l3', 'l2', 'l1']:
        level_stats = mem.get(level, {})
        print(f"    {level.upper()}: {level_stats.get('bytes', 0):,} bytes, {level_stats.get('events', 0):,} events")

    print()
    print("  Compute Breakdown:")
    compute = xue.get('compute_breakdown', {})
    print(f"    MatMul events: {compute.get('matmul_events', 0):,}")
    print(f"    MatMul FLOPs:  {compute.get('matmul_flops', 0):,}")
    print(f"    Elementwise:   {compute.get('elementwise_events', 0):,}")
    print(f"    Total FLOPs:   {xue.get('total_flops', 0):,}")

    # Compare expected vs recorded FLOPs
    print_section("4. FLOP Verification")
    recorded_matmul_flops = compute.get('matmul_flops', 0)
    expected_matmul_flops = expected['total_matmul_flops']

    print(f"  Expected MatMul FLOPs: {expected_matmul_flops:,}")
    print(f"  Recorded MatMul FLOPs: {recorded_matmul_flops:,}")

    if recorded_matmul_flops == 0:
        print("  [NOTE] XUE events not yet recorded by simulator components")
        print("         This is expected until XUE is fully integrated into compute fabrics")
    elif recorded_matmul_flops == expected_matmul_flops:
        print("  [PASS] FLOPs match exactly!")
    else:
        # Allow some difference due to tile-level recording (padding to 16x16 tiles)
        diff_pct = abs(recorded_matmul_flops - expected_matmul_flops) / expected_matmul_flops * 100
        if diff_pct < 20:  # Within 20% due to tiling overhead
            print(f"  [PASS] FLOPs within {diff_pct:.1f}% (tiling overhead expected)")
        else:
            print(f"  [WARN] FLOPs differ by {diff_pct:.1f}%")

    # Run operational analysis (roofline model)
    print_section("5. Operational Analysis (Roofline Model)")
    analysis = kpu.get_operational_analysis(
        peak_gflops=args.peak_gflops,
        dram_bandwidth_gbs=args.dram_bandwidth,
        clock_ghz=args.clock
    )

    print(f"  Arithmetic Intensity: {analysis.get('arithmetic_intensity', 0):.2f} FLOP/byte")
    print(f"  Ridge Point:          {analysis.get('ridge_point', 0):.2f} FLOP/byte")
    print(f"  Predicted GFLOPS:     {analysis.get('predicted_gflops', 0):.1f}")
    print(f"  Predicted Bottleneck: {analysis.get('predicted_bottleneck', 'unknown')}")

    # Calculate memory/compute balance
    if analysis.get('arithmetic_intensity', 0) > 0:
        if analysis['arithmetic_intensity'] < analysis.get('ridge_point', 8.0):
            print("  Status: Memory-bound workload")
        else:
            print("  Status: Compute-bound workload")

    # Timing analysis (TRANSACTIONAL mode only)
    if args.transactional:
        print_section("6. Timing Analysis (TRANSACTIONAL Mode)")
        stats = mlp_workload.get_stats()
        if stats and stats.elapsed_cycles > 0:
            T = stats.elapsed_cycles
            wall_time_us = T / (args.clock * 1e9) * 1e6

            print(f"  T (Elapsed Cycles): {T:,} cycles")
            print(f"  Wall Time:          {wall_time_us:.2f} us")
            print()
            print("  Cycle Breakdown:")
            print(f"    Compute: {stats.compute_cycles:,} cycles")
            print(f"    Memory:  {stats.memory_cycles:,} cycles")
            print(f"    Busy:    {stats.busy_cycles:,} cycles")
            print(f"    Idle:    {stats.idle_cycles:,} cycles")
            print(f"    Stall:   {stats.stall_cycles:,} cycles")

            # Validate against roofline prediction
            actual_gflops = stats.gflops if hasattr(stats, 'gflops') else 0
            if actual_gflops > 0:
                print_section("7. Roofline Validation")
                predicted = analysis.get('predicted_gflops', 0)

                print(f"  Roofline Prediction: {predicted:.1f} GFLOPS")
                print(f"  Actual Achieved:     {actual_gflops:.1f} GFLOPS")

                if predicted > 0:
                    error_pct = abs(actual_gflops - predicted) / predicted * 100
                    print(f"  Prediction Error:    {error_pct:.1f}%")

                    within_10 = error_pct <= 10
                    print(f"  Within 10% Target:   {'YES' if within_10 else 'NO'}")
        else:
            print("  No timing statistics available")

    # Summary
    print_section("Summary")
    print("  XUE Observation Architecture provides:")
    print("    - Zero-logic event recording (atomic counter increments)")
    print("    - Hierarchical event aggregation (matching hardware counters)")
    print("    - Roofline model predictions for performance estimation")
    print("    - Pre/post-silicon comparison capability")
    print()
    print("  For post-silicon validation, compare:")
    print("    - XUE event counts (should match hardware counters)")
    print("    - Roofline predictions vs measured performance")
    print("    - Memory hierarchy traffic patterns")

    return 0


if __name__ == "__main__":
    sys.exit(main())
