#!/usr/bin/env python3
"""
MNIST MLP Example - First KPU Functional Test

This example demonstrates:
1. Defining a 3-layer MLP using @kpu.compile decorator
2. Executing on the KPU behavioral simulator
3. Verifying correctness against NumPy reference
4. Inspecting the generated DFX IR
5. (Optional) XUE performance analysis with TRANSACTIONAL mode

Network architecture: 784 -> 128 -> 64 -> 10

Usage:
    python mnist_mlp.py                   # BEHAVIORAL mode (default)
    python mnist_mlp.py --transactional   # TRANSACTIONAL mode with XUE analysis
"""

import sys
import os
import argparse

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kpu


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST MLP on KPU Simulator")
    parser.add_argument("--transactional", action="store_true",
                        help="Use TRANSACTIONAL mode with XUE performance analysis")
    parser.add_argument("--clock", type=float, default=1.0,
                        help="Clock frequency in GHz (default: 1.0)")
    return parser.parse_args()


# Parse command line arguments
args = parse_args()

# Set fidelity based on command line
if args.transactional:
    kpu.set_fidelity(kpu.TRANSACTIONAL)
    kpu.set_clock_frequency(args.clock)  # Required for TRANSACTIONAL mode
else:
    kpu.set_fidelity(kpu.BEHAVIORAL)


@kpu.compile
def mnist_mlp(x: kpu.Tensor,
              w1: kpu.Tensor, b1: kpu.Tensor,
              w2: kpu.Tensor, b2: kpu.Tensor,
              w3: kpu.Tensor, b3: kpu.Tensor) -> kpu.Tensor:
    """
    3-layer MLP for MNIST classification.

    Args:
        x: Input images [batch, 784]
        w1, b1: Layer 1 weights [784, 128] and bias [128]
        w2, b2: Layer 2 weights [128, 64] and bias [64]
        w3, b3: Layer 3 weights [64, 10] and bias [10]

    Returns:
        Logits [batch, 10]
    """
    # Layer 1: 784 -> 128 with ReLU
    h1 = kpu.relu(x @ w1 + b1)

    # Layer 2: 128 -> 64 with ReLU
    h2 = kpu.relu(h1 @ w2 + b2)

    # Layer 3: 64 -> 10 (no activation - raw logits)
    logits = h2 @ w3 + b3

    return logits


def xavier_init(shape: tuple, dtype=np.float32) -> np.ndarray:
    """Xavier/Glorot initialization for weights."""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape).astype(dtype) * std


def create_test_data(batch_size: int = 32):
    """Create synthetic test data for MNIST MLP."""
    np.random.seed(42)

    # Weights and biases with Xavier initialization
    w1 = kpu.Tensor(xavier_init((784, 128)), name="w1")
    b1 = kpu.Tensor(np.zeros(128, dtype=np.float32), name="b1")

    w2 = kpu.Tensor(xavier_init((128, 64)), name="w2")
    b2 = kpu.Tensor(np.zeros(64, dtype=np.float32), name="b2")

    w3 = kpu.Tensor(xavier_init((64, 10)), name="w3")
    b3 = kpu.Tensor(np.zeros(10, dtype=np.float32), name="b3")

    # Random input (simulating flattened MNIST images)
    x = kpu.Tensor(np.random.randn(batch_size, 784).astype(np.float32), name="input")

    return x, (w1, b1, w2, b2, w3, b3)


def reference_mlp(x, w1, b1, w2, b2, w3, b3):
    """NumPy reference implementation for verification."""
    h1 = np.maximum(x @ w1 + b1, 0)  # ReLU
    h2 = np.maximum(h1 @ w2 + b2, 0)  # ReLU
    logits = h2 @ w3 + b3
    return logits


def main():
    print("=" * 60)
    print("MNIST MLP on KPU Simulator")
    print("=" * 60)
    print()

    # Create test data
    batch_size = 32
    x, weights = create_test_data(batch_size=batch_size)
    w1, b1, w2, b2, w3, b3 = weights

    print("Network Architecture:")
    print(f"  Input:  {x.shape}")
    print(f"  Layer 1: {w1.shape[0]} -> {w1.shape[1]} (ReLU)")
    print(f"  Layer 2: {w2.shape[0]} -> {w2.shape[1]} (ReLU)")
    print(f"  Layer 3: {w3.shape[0]} -> {w3.shape[1]} (Linear)")
    print()

    # Execute on KPU simulator
    mode = "TRANSACTIONAL" if args.transactional else "BEHAVIORAL"
    print(f"Compiling and executing on KPU ({mode})...")
    if args.transactional:
        print(f"  Clock frequency: {args.clock:.1f} GHz")
    logits = mnist_mlp(x, w1, b1, w2, b2, w3, b3)

    print(f"  Output shape: {logits.shape}")
    print()

    # Verify against NumPy reference
    print("Verifying against NumPy reference...")
    ref_logits = reference_mlp(
        x.numpy(), w1.numpy(), b1.numpy(),
        w2.numpy(), b2.numpy(), w3.numpy(), b3.numpy()
    )

    max_diff = np.max(np.abs(logits.numpy() - ref_logits))
    print(f"  Max difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  [PASS] Results match reference implementation!")
    else:
        print("  [FAIL] Results do not match - check implementation")
        return 1

    # Print execution statistics
    stats = mnist_mlp.get_stats()
    if stats:
        print()
        print("Execution Statistics:")
        print(f"  Operations executed: {stats.ops_executed}")
        print(f"  MatMul FLOPs: {stats.matmul_flops:,}")

        # XUE Performance Analysis (TRANSACTIONAL mode only)
        if args.transactional and stats.elapsed_cycles > 0:
            T = stats.elapsed_cycles  # XUE elapsed time in cycles
            print()
            print("=" * 60)
            print("XUE Performance Analysis (v0.5.0)")
            print("=" * 60)
            print(f"  Clock Frequency:     {stats.clock_frequency_ghz:.1f} GHz")
            print(f"  T (Elapsed Cycles):  {T:,} cycles")
            print(f"  Wall Time:           {T / (stats.clock_frequency_ghz * 1e9) * 1e6:.2f} us")
            print()

            # Get XUE summary from C++ EventCollector
            xue = kpu.get_xue_summary()
            mem = xue.get('memory_hierarchy', {})
            compute_bd = xue.get('compute_breakdown', {})

            print("Memory Hierarchy (C++ XUE Events):")
            print(f"  DRAM: {mem.get('dram', {}).get('bytes', 0):,} bytes | {mem.get('dram', {}).get('events', 0):,} events")
            print(f"  L3:   {mem.get('l3', {}).get('bytes', 0):,} bytes | {mem.get('l3', {}).get('events', 0):,} events")
            print(f"  L2:   {mem.get('l2', {}).get('bytes', 0):,} bytes | {mem.get('l2', {}).get('events', 0):,} events")
            print(f"  L1:   {mem.get('l1', {}).get('bytes', 0):,} bytes | {mem.get('l1', {}).get('events', 0):,} events")
            print()

            print("Compute Events (C++ XUE):")
            print(f"  MatMul Events:      {compute_bd.get('matmul_events', 0):,}")
            print(f"  MatMul FLOPs:       {compute_bd.get('matmul_flops', 0):,}")
            print(f"  Total FLOPs:        {xue.get('total_flops', 0):,}")
            print()

            print("Cycle Breakdown:")
            print(f"  Compute Cycles:  {stats.compute_cycles:,}")
            print(f"  Memory Cycles:   {stats.memory_cycles:,}")
            print(f"  Busy Cycles:     {stats.busy_cycles:,}")
            print(f"  Idle Cycles:     {stats.idle_cycles:,}")
            print(f"  Stall Cycles:    {stats.stall_cycles:,}")
            print()

            # =================================================================
            # XUE Metrics per Resource (X, U, E)
            # =================================================================
            # XUE methodology requires reporting metrics in order:
            #   X = Throughput (work done per unit time)
            #   U = Utilization (fraction of time resource is busy)
            #   E = Efficiency (fraction of peak capability achieved)
            # =================================================================

            # Compute Fabric XUE (16x16 systolic array, 256 MACs/cycle)
            peak_macs_per_cycle = 256
            compute_throughput = stats.matmul_flops / T if T > 0 else 0

            print("Compute Fabric XUE (16x16 Systolic Array):")
            print(f"  Throughput   (X): {compute_throughput:.1f} FLOPs/cycle ({stats.gflops:.1f} GFLOPS)")
            print(f"  Utilization  (U): {stats.utilization * 100:.1f}%")
            print(f"  Efficiency   (E): {stats.efficiency * 100:.1f}%")
            print()

            # Memory Controller XUE (LPDDR5)
            mem_throughput = stats.memory_bandwidth_gbps
            mem_utilization = stats.memory_cycles / T * 100 if T > 0 else 0

            print("Memory Controller XUE (LPDDR5):")
            print(f"  Throughput   (X): {mem_throughput:.2f} GB/s")
            print(f"  Utilization  (U): {mem_utilization:.1f}%")
            print(f"  Efficiency   (E): {stats.page_hit_rate * 100:.1f}% (page hit rate)")
            print("=" * 60)

    # Show generated graph
    print()
    print("=" * 60)
    print("Operation Graph")
    print("=" * 60)
    print(mnist_mlp.summary())

    # Show DFX IR
    print()
    print("=" * 60)
    print("Generated DFX IR")
    print("=" * 60)
    dfx = mnist_mlp.get_dfx()
    if dfx:
        print(dfx.summary())
        print()
        print("DFX JSON (first 2000 chars):")
        print(dfx.to_json()[:2000])
        if len(dfx.to_json()) > 2000:
            print("...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
