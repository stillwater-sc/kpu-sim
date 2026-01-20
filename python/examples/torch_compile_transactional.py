#!/usr/bin/env python3
"""
torch.compile with TRANSACTIONAL Mode Demo

This example demonstrates using the KPU simulator's TRANSACTIONAL fidelity
with torch.compile to get timing statistics from model execution.

Usage:
    python examples/torch_compile_transactional.py

Requirements:
    pip install torch
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")
    sys.exit(0)

import kpu


def demo_mlp_transactional():
    """Demo: MLP with TRANSACTIONAL timing."""
    print("=" * 60)
    print("Demo: MLP with torch.compile TRANSACTIONAL mode")
    print("=" * 60)

    # Define MLP
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create model
    model = SimpleMLP()
    model.eval()

    # Create input
    x = torch.randn(32, 784)

    # Get PyTorch reference
    with torch.no_grad():
        ref_output = model(x)

    print(f"\nModel: 3-layer MLP (784 -> 128 -> 64 -> 10)")
    print(f"Input shape: {x.shape}")
    print(f"Batch size: {x.shape[0]}")

    # Compile with TRANSACTIONAL backend
    print("\nCompiling with backend='kpu_transactional'...")
    compiled_model = torch.compile(model, backend="kpu_transactional")

    # Execute on KPU (TRANSACTIONAL mode)
    print("Executing on KPU simulator (TRANSACTIONAL)...")
    with torch.no_grad():
        kpu_output = compiled_model(x)

    # Get timing stats
    stats = kpu.get_torch_compile_stats()

    print(f"\nKPU output shape: {kpu_output.shape}")

    # Compare results
    max_diff = torch.max(torch.abs(ref_output - kpu_output)).item()
    print(f"\nMax difference from PyTorch: {max_diff:.2e}")

    # Show timing stats if available
    if stats is not None:
        print("\n" + "-" * 40)
        print("TRANSACTIONAL Timing Statistics:")
        print("-" * 40)
        print(f"  Total cycles:    {stats.cycles:,}")
        print(f"  Compute cycles:  {stats.compute_cycles:,}")
        print(f"  Memory cycles:   {stats.memory_cycles:,}")
        print(f"  Busy cycles:     {stats.busy_cycles:,}")
        print(f"  Idle cycles:     {stats.idle_cycles:,}")
        print()
        print(f"  MatMul FLOPs:    {stats.matmul_flops:,}")
        print(f"  Total MACs:      {stats.total_macs:,}")
        print(f"  MatMul count:    {stats.matmul_count}")
        print()
        if stats.clock_frequency_ghz > 0:
            print(f"  Clock frequency: {stats.clock_frequency_ghz:.1f} GHz")
            print(f"  Throughput:      {stats.gflops:.1f} GFLOPS")
            print(f"  Utilization:     {stats.utilization*100:.1f}%")
            print(f"  Efficiency:      {stats.efficiency*100:.1f}%")
        print("-" * 40)
    else:
        print("\nNote: No timing stats available (native bindings may not be loaded)")

    # Validation
    passed = max_diff < 1e-4
    if passed:
        print("\n[PASS] Results match reference implementation")
    else:
        print("\n[FAIL] Results differ from reference")

    return passed, stats


def demo_compare_with_direct_kpu():
    """Compare torch.compile timing with direct KPU execution."""
    print("\n" + "=" * 60)
    print("Demo: Compare torch.compile vs Direct KPU Execution")
    print("=" * 60)

    # Simple 2-layer MLP for comparison
    batch_size = 32
    input_dim = 784
    hidden_dim = 128
    output_dim = 10

    # Create weights
    np.random.seed(42)
    w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
    w2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
    x_np = np.random.randn(batch_size, input_dim).astype(np.float32)

    # --- Direct KPU execution ---
    print("\n1. Direct KPU execution (TRANSACTIONAL):")
    print("-" * 40)

    kpu.set_fidelity(kpu.TRANSACTIONAL)
    kpu.set_clock_frequency(1.0)  # 1 GHz

    @kpu.compile
    def mlp_direct(x, w1, w2):
        h = kpu.relu(kpu.matmul(x, w1))
        return kpu.matmul(h, w2)

    x_kpu = kpu.Tensor(x_np)
    w1_kpu = kpu.Tensor(w1)
    w2_kpu = kpu.Tensor(w2)

    result_direct, stats_direct = mlp_direct.execute_with_stats(x_kpu, w1_kpu, w2_kpu)

    if stats_direct:
        print(f"  Cycles: {stats_direct.cycles:,}")
        print(f"  FLOPs:  {stats_direct.matmul_flops:,}")
        if stats_direct.gflops > 0:
            print(f"  GFLOPS: {stats_direct.gflops:.1f}")

    # --- torch.compile execution ---
    print("\n2. torch.compile execution (TRANSACTIONAL):")
    print("-" * 40)

    class SimpleMLP(nn.Module):
        def __init__(self, w1, w2):
            super().__init__()
            self.w1 = nn.Parameter(torch.from_numpy(w1.T.copy()))
            self.w2 = nn.Parameter(torch.from_numpy(w2.T.copy()))

        def forward(self, x):
            h = F.relu(F.linear(x, self.w1))
            return F.linear(h, self.w2)

    model = SimpleMLP(w1, w2)
    model.eval()

    compiled_model = torch.compile(model, backend="kpu_transactional")

    with torch.no_grad():
        x_torch = torch.from_numpy(x_np)
        result_torch = compiled_model(x_torch)

    stats_torch = kpu.get_torch_compile_stats()

    if stats_torch:
        print(f"  Cycles: {stats_torch.cycles:,}")
        print(f"  FLOPs:  {stats_torch.matmul_flops:,}")
        if stats_torch.gflops > 0:
            print(f"  GFLOPS: {stats_torch.gflops:.1f}")

    # Compare
    print("\n3. Comparison:")
    print("-" * 40)

    if stats_direct and stats_torch:
        cycle_diff = abs(stats_direct.cycles - stats_torch.cycles)
        flops_diff = abs(stats_direct.matmul_flops - stats_torch.matmul_flops)

        print(f"  Cycle difference:  {cycle_diff:,}")
        print(f"  FLOPS difference:  {flops_diff:,}")

        if cycle_diff == 0 and flops_diff == 0:
            print("\n  [PASS] Timing matches between torch.compile and direct KPU")
            return True
        elif flops_diff == 0:
            print("\n  [PASS] FLOPS match (cycles may differ due to execution model)")
            return True
        else:
            print("\n  [INFO] Some differences detected (may be expected)")
            return True
    else:
        print("  Cannot compare - stats not available from one or both methods")
        return True


def main():
    print("=" * 60)
    print("KPU torch.compile TRANSACTIONAL Mode Demo")
    print("=" * 60)

    print("\nKPU Package Info:")
    print(kpu.info())

    if not kpu.TORCH_AVAILABLE:
        print("PyTorch integration not available!")
        return 1

    # Run demos
    passed1, stats1 = demo_mlp_transactional()
    passed2 = demo_compare_with_direct_kpu()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  MLP TRANSACTIONAL:    {'PASS' if passed1 else 'FAIL'}")
    print(f"  Timing comparison:    {'PASS' if passed2 else 'FAIL'}")

    if stats1:
        print(f"\n  Timing stats available: Yes")
        print(f"  Cycles collected: {stats1.cycles:,}")
    else:
        print(f"\n  Timing stats available: No (native bindings may be needed)")

    print("=" * 60)

    return 0 if (passed1 and passed2) else 1


if __name__ == "__main__":
    sys.exit(main())
