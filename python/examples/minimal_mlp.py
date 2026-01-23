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
# XUE Observation Architecture (v0.5.0+)
# =============================================================================
# XUE Methodology: X (Throughput) → U (Utilization) → E (Efficiency)
#
# The Observation Architecture provides event hierarchies that aggregate cleanly
# without logic on the datapath. This enables systematic drill-down analysis:
#   - X = Measured throughput compared to speed-of-light
#   - U = Resource utilization if throughput is below threshold
#   - E = Efficiency (measured/peak) if utilization doesn't explain throughput
#
# T represents the wall-clock execution time in cycles.
# Key metrics computed in C++:
#   - Service Rate = bytes / T (bytes per cycle)
#   - Throughput = transactions / T (transactions per cycle)
#   - GFLOPS = (FLOPs / T) * clock_frequency_ghz
# =============================================================================

T = stats.elapsed_cycles  # XUE elapsed time in cycles

print("=" * 60)
print("XUE Performance Analysis (v0.5.0)")
print("=" * 60)
print(f"  Clock Frequency:     {stats.clock_frequency_ghz:.1f} GHz")
print(f"  T (Elapsed Cycles):  {T:,} cycles")
print(f"  Wall Time:           {T / (stats.clock_frequency_ghz * 1e9) * 1e6:.2f} us")
print()

# Get XUE summary from C++ EventCollector
xue = kpu.get_xue_summary()

# XUE Memory Hierarchy Stats (from C++ XUE)
mem = xue.get('memory_hierarchy', {})
print("Memory Hierarchy (C++ XUE Events):")
print(f"  DRAM: {mem.get('dram', {}).get('bytes', 0):,} bytes | {mem.get('dram', {}).get('events', 0):,} events")
print(f"  L3:   {mem.get('l3', {}).get('bytes', 0):,} bytes | {mem.get('l3', {}).get('events', 0):,} events")
print(f"  L2:   {mem.get('l2', {}).get('bytes', 0):,} bytes | {mem.get('l2', {}).get('events', 0):,} events")
print(f"  L1:   {mem.get('l1', {}).get('bytes', 0):,} bytes | {mem.get('l1', {}).get('events', 0):,} events")
print()

# Compute Events from C++ XUE
compute = xue.get('compute_breakdown', {})
print("Compute Events (C++ XUE):")
print(f"  MatMul Events:      {compute.get('matmul_events', 0):,}")
print(f"  MatMul FLOPs:       {compute.get('matmul_flops', 0):,}")
print(f"  Elementwise Events: {compute.get('elementwise_events', 0):,}")
print(f"  Total FLOPs:        {xue.get('total_flops', 0):,}")
print()

# Operational Analysis (Roofline Model)
analysis = kpu.get_operational_analysis(
    peak_gflops=512.0,  # 16x16 systolic @ 1GHz = 256 MACs * 2 = 512 GFLOPS
    dram_bandwidth_gbs=64.0,
    clock_ghz=stats.clock_frequency_ghz
)

print("Operational Analysis (Roofline Model):")
print(f"  Arithmetic Intensity: {analysis.get('arithmetic_intensity', 0):.2f} FLOP/byte")
print(f"  Predicted GFLOPS:     {analysis.get('predicted_gflops', 0):.1f}")
print(f"  Predicted Bottleneck: {analysis.get('predicted_bottleneck', 'unknown')}")
print()

# Cycle Breakdown
print("Cycle Breakdown:")
print(f"  Compute Cycles:  {stats.compute_cycles:,}")
print(f"  Memory Cycles:   {stats.memory_cycles:,}")
print(f"  Busy Cycles:     {stats.busy_cycles:,}")
print(f"  Idle Cycles:     {stats.idle_cycles:,}")
print(f"  Stall Cycles:    {stats.stall_cycles:,}")
print()

# =============================================================================
# XUE Metrics per Resource (X, U, E)
# =============================================================================
# XUE methodology requires reporting metrics in order:
#   X = Throughput (work done per unit time)
#   U = Utilization (fraction of time resource is busy)
#   E = Efficiency (fraction of peak capability achieved)
# =============================================================================

# Compute Fabric XUE (16x16 systolic array, 256 MACs/cycle)
peak_macs_per_cycle = 256
peak_gflops = peak_macs_per_cycle * 2 * stats.clock_frequency_ghz  # 2 FLOPs per MAC
compute_throughput = stats.matmul_flops / T if T > 0 else 0  # FLOPs/cycle

print("Compute Fabric XUE (16x16 Systolic Array):")
print(f"  Throughput   (X): {compute_throughput:.1f} FLOPs/cycle ({stats.gflops:.1f} GFLOPS)")
print(f"  Utilization  (U): {stats.utilization * 100:.1f}%")
print(f"  Efficiency   (E): {stats.efficiency * 100:.1f}%")
print()

# Memory Controller XUE (LPDDR5)
mem_throughput = stats.memory_bandwidth_gbps  # GB/s
mem_utilization = stats.memory_cycles / T * 100 if T > 0 else 0  # % of time doing memory ops

print("Memory Controller XUE (LPDDR5):")
print(f"  Throughput   (X): {mem_throughput:.2f} GB/s")
print(f"  Utilization  (U): {mem_utilization:.1f}%")
print(f"  Efficiency   (E): {stats.page_hit_rate * 100:.1f}% (page hit rate)")
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

# Verify against C++ XUE stats
xue_matmul_flops = compute.get('matmul_flops', 0)
print(f"\nC++ XUE reported: {xue_matmul_flops:,} FLOPs")
# Note: XUE counts at tile level (16x16), so may differ from exact reference

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
print(f"  C++ XUE DRAM:   {xue.get('dram_bytes', 0):,} bytes")