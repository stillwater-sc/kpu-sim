# How to Configure and Run KPU Simulations

## Overview

The KPU Simulator provides a flexible configuration system and command-line runner for executing simulations. This guide covers:

1. Configuration file formats (YAML and JSON)
2. Factory configurations for common use cases
3. Using the `kpu-runner` command-line tool
4. Customizing configurations for your needs

## Prerequisites

Build the KPU simulator with the runner tool:

```bash
cd /path/to/kpu-sim
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

The runner tool will be at: `build/tools/runner/kpu-runner`

## Quick Start

### Using Factory Configurations

The fastest way to run a simulation is with built-in factory configurations:

```bash
# Run a 64x64x64 matrix multiplication with minimal hardware
./build/tools/runner/kpu-runner --factory minimal -m 64x64x64

# Run on edge AI configuration
./build/tools/runner/kpu-runner --factory edge_ai -m 128x128x128

# Run on datacenter configuration with larger matrices
./build/tools/runner/kpu-runner --factory datacenter -m 1024x1024x1024
```

### Using Configuration Files

```bash
# YAML configuration
./build/tools/runner/kpu-runner configs/kpu/minimal.yaml -m 256x256x256

# JSON configuration
./build/tools/runner/kpu-runner configs/kpu/minimal.json -m 256x256x256
```

## Configuration File Format

### YAML Format

YAML is the recommended format for human-readable configurations:

```yaml
# KPU Simulator Configuration
name: "My KPU Config"
description: "Custom configuration for testing"

# Host system memory (for host-to-device transfers)
host_memory:
  region_count: 1
  region_capacity_mb: 256
  bandwidth_gbps: 50

# External memory (GDDR/HBM)
external_memory:
  bank_count: 2
  bank_capacity_mb: 512
  bandwidth_gbps: 100

# Memory controller
memory_controller:
  controller_count: 1
  page_buffer_count: 2
  page_buffer_capacity_kb: 32

# On-chip memory hierarchy
on_chip_memory:
  l3:
    tile_count: 2
    tile_capacity_kb: 128
  l2:
    bank_count: 8
    bank_capacity_kb: 64
  # L1 buffer_count is DERIVED from processor array configuration
  # For 16x16 array with 1 tile: 4 * (16 + 16) * 1 = 128 buffers
  l1:
    buffer_capacity_kb: 64

# Data movement engines
data_movement:
  dma_engine_count: 2
  block_mover_count: 4
  streamer_count: 8

# Compute fabric
compute:
  tile_count: 1
  processor_array:
    rows: 16
    cols: 16
    topology: rectangular  # rectangular or hexagonal
  systolic_mode: true
```

### JSON Format

For programmatic configuration or integration with other tools:

```json
{
  "name": "My KPU Config",
  "description": "Custom configuration for testing",
  "host_memory": {
    "region_count": 1,
    "region_capacity_mb": 256,
    "bandwidth_gbps": 50
  },
  "external_memory": {
    "bank_count": 2,
    "bank_capacity_mb": 512,
    "bandwidth_gbps": 100
  },
  "memory_controller": {
    "controller_count": 1,
    "page_buffer_count": 2,
    "page_buffer_capacity_kb": 32
  },
  "on_chip_memory": {
    "l3": {
      "tile_count": 2,
      "tile_capacity_kb": 128
    },
    "l2": {
      "bank_count": 8,
      "bank_capacity_kb": 64
    },
    "l1": {
      "buffer_capacity_kb": 64
    }
  },
  "data_movement": {
    "dma_engine_count": 2,
    "block_mover_count": 4,
    "streamer_count": 8
  },
  "compute": {
    "tile_count": 1,
    "processor_array": {
      "rows": 16,
      "cols": 16,
      "topology": "rectangular"
    },
    "systolic_mode": true
  }
}
```

Note: The `l1.buffer_count` field is **derived** from the processor array configuration and should not be set manually. It is computed as:
`L1_buffers = 4 × (rows + cols) × compute_tiles`

## Configuration Parameters

### Memory Hierarchy

| Section | Parameter | Description | Typical Values |
|---------|-----------|-------------|----------------|
| **host_memory** | region_count | Number of host memory regions | 1-4 |
| | region_capacity_mb | Capacity per region (MB) | 256-4096 |
| | bandwidth_gbps | PCIe bandwidth (GB/s) | 32-128 |
| **external_memory** | bank_count | Number of GDDR/HBM banks | 2-16 |
| | bank_capacity_mb | Capacity per bank (MB) | 256-4096 |
| | bandwidth_gbps | Memory bandwidth (GB/s) | 100-1000 |
| **on_chip_memory.l3** | tile_count | Global buffer tiles | 1-256 |
| | tile_capacity_kb | Capacity per tile (KB) | 64-512 |
| **on_chip_memory.l2** | bank_count | Tile buffer banks | 4-4096 |
| | bank_capacity_kb | Capacity per bank (KB) | 32-128 |
| **on_chip_memory.l1** | buffer_count | **DERIVED** - see below | - |
| | buffer_capacity_kb | Capacity per buffer (KB) | 32-128 |

### L1 Buffer Count (Derived Property)

L1 streaming buffers are **automatically derived** from the processor array configuration. Do not set `buffer_count` manually - it will be computed:

**Formula**: `L1_buffers = 4 × (rows + cols) × compute_tiles`

Each edge (TOP/BOTTOM/LEFT/RIGHT) has ingress and egress buffers:
- TOP edge: `cols` ingress (B weights) + `cols` egress (C output)
- BOTTOM edge: `cols` ingress + `cols` egress
- LEFT edge: `rows` ingress (A inputs) + `rows` egress
- RIGHT edge: `rows` ingress + `rows` egress

**Examples**:
- 8×8 array, 1 tile (minimal): 4 × (8+8) × 1 = **64 L1 buffers**
- 16×16 array, 2 tiles (edge_ai): 4 × (16+16) × 2 = **256 L1 buffers**
- 32×32 array, 256 tiles (datacenter): 4 × (32+32) × 256 = **65,536 L1 buffers**

### Data Movement

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| dma_engine_count | DMA engines for external transfers | 1-32 |
| block_mover_count | L3↔L2 block movers | 1-256 |
| streamer_count | L2↔L1 streamers | 4-1024 |

### Compute

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| tile_count | Number of compute tiles | 1-256 |
| processor_array.rows | Systolic array rows | 8-64 |
| processor_array.cols | Systolic array columns | 8-64 |
| processor_array.topology | Array layout (rectangular or hexagonal) | rectangular |
| systolic_mode | Enable systolic data flow | true/false |

## Factory Configurations

Four factory configurations are provided for common use cases, forming a proper progression from small to large:

### Minimal (`--factory minimal`)

Smallest viable KPU for testing and development:

- 1 compute tile (8×8 rectangular systolic array)
- 1 external memory channel (256 MB, LPDDR4x, 25 GB/s)
- 1 L3 tile, 4 L2 banks, **64 L1 buffers** (derived: 4×(8+8)×1)
- Best for: Unit testing, small matrix operations, development

### Edge AI (`--factory edge_ai`)

Dual-tile configuration for edge AI inference:

- 2 compute tiles (16×16 rectangular systolic arrays each)
- 4 external memory channels (256 MB each, LPDDR5, 64-bit total)
- 2 L3 tiles, 16 L2 banks (8 per L3), **256 L1 buffers** (derived: 4×(16+16)×2)
- Power-efficient 48 GB/s bandwidth (~3.2W memory subsystem)
- Best for: Mobile/embedded AI, edge inference

### Embodied AI (`--factory embodied_ai`)

64-tile configuration for robotics and autonomous systems (Jetson Orin style):

- 64 compute tiles (24×24 rectangular systolic arrays each) in 8×8 layout
- 8 external memory channels (512 MB each, LPDDR5, 256-bit total)
- 64 L3 tiles, 1024 L2 banks (16 per L3), **12,288 L1 buffers** (derived: 4×(24+24)×64)
- 200 GB/s memory bandwidth, power-efficient (~8W memory subsystem)
- Best for: Real-time robotics, autonomous driving, embodied agents

### Datacenter (`--factory datacenter`)

256-tile configuration for datacenter-scale AI workloads:

- 256 compute tiles (32×32 rectangular systolic arrays each) in 16×16 checkerboard
- 6 HBM3 channels (4 GB each, 24 GB total)
- 256 L3 tiles, 4096 L2 banks (16 per L3), **65,536 L1 buffers** (derived: 4×(32+32)×256)
- 4.8 TB/s memory bandwidth (800 GB/s per channel)
- Best for: Training, large batch inference, datacenter workloads

## KPU Runner Command Reference

```
Usage:
  kpu-runner [options] [config-file]

Options:
  -h, --help              Show help message
  -v, --verbose           Verbose output with detailed info
  -t, --test <type>       Test type: matmul, mlp, benchmark
  -m, --matrix <MxNxK>    Matrix dimensions (e.g., 128x128x128)
  -o, --output <file>     Write results to JSON file
  --validate              Validate config without running
  --show-config           Display parsed configuration
  --factory <name>        Use factory config: minimal, edge_ai, embodied_ai, datacenter
```

### Test Types

| Type | Description |
|------|-------------|
| `matmul` | Single matrix multiplication (default) |
| `mlp` | MLP layer with GELU activation |
| `benchmark` | Suite of matrix sizes (64 to 512) |

## Examples

### Basic Matrix Multiplication

```bash
# 128x128x128 matmul on minimal config
./build/tools/runner/kpu-runner --factory minimal -m 128x128x128

# Output:
# KPU Simulator initialized.
#   Memory banks: 2
#   L3 tiles:     4
#   L2 banks:     8
#   L1 buffers:   4
#   Compute tiles:1
# Running MatMul test: 128 x 128 x 128
#
# === Results ===
# Status:      SUCCESS
# Cycles:      11904
# Time:        0.051 ms
# Performance: 82.16 GFLOPS
```

### MLP Layer Test

```bash
# MLP with 128x64x128 dimensions
./build/tools/runner/kpu-runner --factory minimal -t mlp -m 128x64x128

# Output shows fused matmul + GELU activation performance
```

### Benchmark Suite

```bash
# Run benchmark across multiple matrix sizes
./build/tools/runner/kpu-runner --factory datacenter -t benchmark

# Output:
# === Running Benchmark Suite ===
#
#       Size      Cycles   Time (ms)      GFLOPS
# ----------------------------------------------
#      64x64        1792        0.02       27.28
#    128x128       11904        0.03      127.36
#    256x256       78464        0.11      292.78
#    512x512      571008        0.81      331.25
```

### View Configuration

```bash
# Display parsed configuration details
./build/tools/runner/kpu-runner configs/kpu/datacenter.yaml --show-config

# Output shows all memory, data movement, and compute settings
```

### Export Results to JSON

```bash
# Save results to file for analysis
./build/tools/runner/kpu-runner --factory minimal -m 256x256x256 -o results.json

# results.json:
# {
#   "success": true,
#   "cycles": 78464,
#   "elapsed_ms": 0.118,
#   "gflops": 283.42
# }
```

### Validate Configuration

```bash
# Check config validity without running
./build/tools/runner/kpu-runner configs/kpu/custom.yaml --validate
```

## Creating Custom Configurations

### Step 1: Copy a Template

```bash
cp configs/kpu/minimal.yaml configs/kpu/my_config.yaml
```

### Step 2: Modify Parameters

Edit the file to match your target hardware:

```yaml
name: "My Custom KPU"
description: "Optimized for transformer inference"

external_memory:
  bank_count: 4
  bank_capacity_mb: 1024
  bandwidth_gbps: 400

on_chip_memory:
  l3:
    tile_count: 8
    tile_capacity_kb: 256
  l2:
    bank_count: 16
    bank_capacity_kb: 64

compute:
  tile_count: 2
  processor_array:
    rows: 32
    cols: 32
```

### Step 3: Validate and Test

```bash
# Validate syntax
./build/tools/runner/kpu-runner configs/kpu/my_config.yaml --validate

# Run a test
./build/tools/runner/kpu-runner configs/kpu/my_config.yaml -m 512x512x512 -v
```

## Programmatic Configuration (C++)

You can also load configurations programmatically:

```cpp
#include <sw/kpu/kpu_config_loader.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

int main() {
    // Load from file
    auto config = KPUConfigLoader::load_yaml("configs/kpu/minimal.yaml");

    // Or use factory
    auto config = KPUConfigLoader::create_minimal();

    // Or create programmatically
    auto config = KPUConfigLoader::create_for_matmul(1024, 1024, 1024);

    // Create simulator
    KPUSimulator sim(config);

    // Run your workload...
}
```

## Troubleshooting

### Configuration Not Loading

**Error**: `Failed to load configuration`

**Solutions**:
1. Check file exists: `ls -la configs/kpu/your_config.yaml`
2. Validate YAML syntax with an online validator
3. Ensure all required sections are present

### Invalid Matrix Dimensions

**Error**: `Invalid matrix dimensions`

**Solution**: Use format `MxNxK` with lowercase 'x':
```bash
# Correct
-m 128x128x128

# Incorrect
-m 128X128X128
-m 128,128,128
```

### Factory Config Not Found

**Error**: `Unknown factory config: xyz`

**Solution**: Use one of: `minimal`, `edge_ai`, `datacenter`

### Low Performance Numbers

If GFLOPS seems low:

1. **Matrix too small**: Small matrices have high overhead-to-compute ratio
2. **Wrong config**: Edge AI config has lower bandwidth
3. **Use benchmark mode**: Compare across sizes to see scaling

```bash
# See how performance scales with size
./build/tools/runner/kpu-runner --factory datacenter -t benchmark
```

## Configuration Files Location

```
kpu-sim/
├── configs/
│   └── kpu/
│       ├── minimal.yaml      # Basic testing config
│       ├── minimal.json      # JSON equivalent
│       ├── edge_ai.yaml      # Edge deployment config
│       ├── embodied_ai.yaml  # Robotics/autonomous config
│       └── datacenter.yaml   # High-performance config
├── tools/
│   └── runner/
│       └── kpu_runner.cpp    # Runner implementation
└── include/
    └── sw/kpu/
        └── kpu_config_loader.hpp  # Config loader API
```

## See Also

- [KPU Architecture](kpu_architecture.md) - Detailed hardware architecture
- [Memory Hierarchy](unified-address-space.md) - Memory addressing
- [Python Integration](how-to-build-and-use-python-bindings.md) - Python API
- [Benchmarking](sessions/2025-12-25_benchmarking_and_efficiency_analysis.md) - Performance analysis
