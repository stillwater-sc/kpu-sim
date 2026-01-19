import sys
import os
import numpy as np

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kpu

# Set TRANSACTIONAL mode for timing simulation
kpu.set_fidelity(kpu.TRANSACTIONAL)

# IMPORTANT: Must set clock frequency before execution in TRANSACTIONAL mode
# This prevents silent assumptions about clock speed
kpu.set_clock_frequency(1.0)  # 1 GHz

@kpu.compile
def mlp(x, w1, w2):
    h = kpu.relu(kpu.matmul(x, w1))
    return kpu.matmul(h, w2)

# Create tensors
batch_size = 32
x = kpu.Tensor(np.random.randn(batch_size, 784).astype(np.float32))
w1 = kpu.Tensor(np.random.randn(784, 128).astype(np.float32))
w2 = kpu.Tensor(np.random.randn(128, 10).astype(np.float32))

# Execute
result = mlp(x, w1, w2)
stats = mlp.stats

# =============================================================================
# XUE Methodology: T (Elapsed Cycles) is the Central Measurement
# =============================================================================
# T represents the wall-clock execution time in cycles
# All service rates and throughputs are computed relative to T:
#   - Service Rate = bytes / T (bytes per cycle)
#   - Throughput = transactions / T (transactions per cycle)
#   - GFLOPS = (FLOPs / T) * clock_frequency_ghz
# =============================================================================

T = stats.elapsed_cycles  # XUE elapsed time in cycles

print("=" * 60)
print("XUE Performance Analysis")
print("=" * 60)
print(f"  Clock Frequency:     {stats.clock_frequency_ghz:.1f} GHz")
print(f"  T (Elapsed Cycles):  {T:,} cycles")
print(f"  Wall Time:           {T / (stats.clock_frequency_ghz * 1e9) * 1e6:.2f} us")
print()

# XUE Memory Hierarchy Stats with Service Rates
print("Memory Hierarchy (XUE Events):")
print(f"  DRAM: {stats.dram.total_bytes:,} bytes | {stats.dram.total_count:,} txns | {stats.dram.service_rate:.2f} B/cycle")
print(f"  L3:   {stats.l3.total_bytes:,} bytes | {stats.l3.total_count:,} txns | {stats.l3.service_rate:.2f} B/cycle")
print(f"  L2:   {stats.l2.total_bytes:,} bytes | {stats.l2.total_count:,} txns | {stats.l2.service_rate:.2f} B/cycle")
print(f"  L1:   {stats.l1.total_bytes:,} bytes | {stats.l1.total_count:,} txns | {stats.l1.service_rate:.2f} B/cycle")
print()

# Compute Performance
print("Compute Performance:")
print(f"  MatMul FLOPs:  {stats.matmul_flops:,}")
print(f"  GFLOPS:        {stats.gflops:.1f} @ {stats.clock_frequency_ghz:.1f} GHz")
print(f"  FLOPs/Cycle:   {stats.matmul_flops / T:.1f}")
print()

# Cycle Breakdown (v0.4.2 Timing Stats)
print("Cycle Breakdown:")
print(f"  Compute Cycles:  {stats.compute_cycles:,}")
print(f"  Memory Cycles:   {stats.memory_cycles:,}")
print(f"  Busy Cycles:     {stats.busy_cycles:,}")
print(f"  Idle Cycles:     {stats.idle_cycles:,}")
print(f"  Stall Cycles:    {stats.stall_cycles:,}")
print()

# Utilization Metrics
print("Utilization Metrics:")
print(f"  Utilization:     {stats.utilization * 100:.1f}%")
print(f"  Efficiency:      {stats.efficiency * 100:.1f}%")
print(f"  Page Hit Rate:   {stats.page_hit_rate * 100:.1f}%")
print(f"  Memory BW:       {stats.memory_bandwidth_gbps:.2f} GB/s")
print("=" * 60)

# Reference calculations
# For matmul [M, K] @ [K, N], FLOPs = 2 * M * K * N (multiply-accumulate)
matmul1_flops = 2 * batch_size * 784 * 128  # x @ w1: [32, 784] @ [784, 128]
matmul2_flops = 2 * batch_size * 128 * 10   # h @ w2: [32, 128] @ [128, 10]
relu_flops = batch_size * 128               # ReLU on [32, 128] (comparisons)

print("\nReference Calculations:")
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {result.shape}")
print(f"  MatMul 1 (x @ w1): [32, 784] @ [784, 128] = {matmul1_flops:,} FLOPs")
print(f"  MatMul 2 (h @ w2): [32, 128] @ [128, 10]  = {matmul2_flops:,} FLOPs")
print(f"  ReLU:              [32, 128]              = {relu_flops:,} ops")
print(f"  Total MatMul FLOPs: {matmul1_flops + matmul2_flops:,}")

# Verify against simulator stats
print(f"\nSimulator reported: {stats.matmul_flops:,} FLOPs")
if stats.matmul_flops == matmul1_flops + matmul2_flops:
    print("  [PASS] Reference matches simulator!")
else:
    print(f"  [FAIL] Mismatch: expected {matmul1_flops + matmul2_flops:,}")

# Memory traffic reference
input_bytes = batch_size * 784 * 4       # x: [32, 784] float32
w1_bytes = 784 * 128 * 4                 # w1: [784, 128] float32
w2_bytes = 128 * 10 * 4                  # w2: [128, 10] float32
h_bytes = batch_size * 128 * 4           # hidden: [32, 128] float32
output_bytes = batch_size * 10 * 4       # output: [32, 10] float32

total_input_bytes = input_bytes + w1_bytes + w2_bytes
total_output_bytes = h_bytes + output_bytes  # intermediate + final

print(f"\nMemory Traffic Reference:")
print(f"  Input tensors:  {total_input_bytes:,} bytes (x + w1 + w2)")
print(f"  Output tensors: {total_output_bytes:,} bytes (h + output)")
print(f"  Total DRAM:     {stats.dram.total_bytes:,} bytes (read + write)")