# How to Configure and Run KPU Simulations

## Overview

The KPU Simulator provides a multi-fidelity configuration system supporting BEHAVIORAL, TRANSACTIONAL, and CYCLE_ACCURATE simulation modes. This guide covers:

1. Multi-fidelity configuration format (JSON)
2. Preset configurations for common use cases
3. Using the `kpu-runner` command-line tool
4. Configuration validation and analysis with `kpu-config`
5. Customizing configurations for your needs

## Prerequisites

Build the KPU simulator with the runner tool:

```bash
cd /path/to/kpu-sim
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

The tools will be at:
- `build/tools/runner/kpu-runner` - Simulation runner
- `build/tools/configuration/kpu-config` - Configuration tool

## Quick Start

### Using Preset Configurations

The fastest way to run a simulation is with built-in presets:

```bash
# Run a 64x64x64 matrix multiplication with minimal hardware
./build/tools/runner/kpu-runner --preset minimal -m 64x64x64

# Run on edge AI configuration
./build/tools/runner/kpu-runner --preset edge_ai -m 128x128x128

# Run on datacenter configuration with larger matrices
./build/tools/runner/kpu-runner --preset datacenter -m 1024x1024x1024

# Run with fast behavioral simulation
./build/tools/runner/kpu-runner --preset fast -m 256x256x256

# Run with cycle-accurate simulation
./build/tools/runner/kpu-runner --preset accurate -m 256x256x256
```

### Using Configuration Files

```bash
# Use a multi-fidelity configuration file
./build/tools/runner/kpu-runner configs/components/kpu/minimal.json -m 256x256x256

# Use cross-validation configs for comparing fidelity levels
./build/tools/runner/kpu-runner configs/components/kpu/crossval_behavioral.json -m 128x128x128
./build/tools/runner/kpu-runner configs/components/kpu/crossval_cycle_accurate.json -m 128x128x128
```

## Multi-Fidelity Configuration Format

The configuration system supports three simulation fidelity levels:

| Fidelity | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| BEHAVIORAL | Fastest | Low | Functional verification, early prototyping |
| TRANSACTIONAL | Medium | Medium | Performance estimation, design exploration |
| CYCLE_ACCURATE | Slowest | High | Detailed timing analysis, hardware validation |

### JSON Configuration Format

```json
{
  "simulation": {
    "default_fidelity": "BEHAVIORAL",
    "verification_level": "ASSERTIONS",
    "num_memory_controllers": 1,
    "num_dma_engines": 2,
    "num_l3_tiles": 1,
    "num_l2_banks": 4,
    "num_compute_tiles": 1
  },
  "memory": {
    "fidelity": "BEHAVIORAL",
    "technology": "LPDDR5",
    "capacity_gb": 16,
    "channels": 2,
    "burst_length": 16,
    "clock_mhz": 3200
  },
  "dma": {
    "fidelity": "BEHAVIORAL",
    "max_concurrent_transfers": 4,
    "burst_size_bytes": 256
  },
  "interconnect": {
    "fidelity": "BEHAVIORAL",
    "technology": "MESH_2D",
    "mesh_rows": 4,
    "mesh_cols": 4,
    "link_bandwidth_gbps": 100
  },
  "compute": {
    "fidelity": "BEHAVIORAL",
    "technology": "SYSTOLIC",
    "array_size": [8, 8],
    "macs_per_cycle": 64
  }
}
```

### Component-Level Fidelity

Each subsystem can have independent fidelity levels, allowing mixed-fidelity simulation:

```json
{
  "simulation": {
    "default_fidelity": "TRANSACTIONAL"
  },
  "memory": {
    "fidelity": "CYCLE_ACCURATE"
  },
  "compute": {
    "fidelity": "BEHAVIORAL"
  }
}
```

This enables fast compute simulation while maintaining accurate memory timing.

## Configuration Parameters

### Simulation Section

| Parameter | Description | Values |
|-----------|-------------|--------|
| default_fidelity | Default fidelity for all components | BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE |
| verification_level | Runtime checking level | NONE, ASSERTIONS, INVARIANTS, PROTOCOL |
| num_memory_controllers | Number of memory controllers | 1-8 |
| num_dma_engines | Number of DMA engines | 1-32 |
| num_l3_tiles | Number of L3 cache tiles | 1-256 |
| num_l2_banks | Number of L2 cache banks | 4-4096 |
| num_compute_tiles | Number of compute tiles | 1-256 |

### Memory Section

| Parameter | Description | Values |
|-----------|-------------|--------|
| technology | Memory type | LPDDR4, LPDDR5, HBM2, HBM3, GDDR6, DDR5 |
| capacity_gb | Total memory capacity | 1-128 |
| channels | Number of memory channels | 1-16 |
| burst_length | Burst transfer length | 8, 16, 32 |
| clock_mhz | Memory clock frequency | 800-6400 |

### Compute Section

| Parameter | Description | Values |
|-----------|-------------|--------|
| technology | Compute architecture | SYSTOLIC, VECTOR, HYBRID |
| array_size | Systolic array dimensions [rows, cols] | [8,8] to [64,64] |
| macs_per_cycle | Peak MACs per cycle | 64-4096 |

### Interconnect Section

| Parameter | Description | Values |
|-----------|-------------|--------|
| technology | Network topology | MESH_2D, TORUS_2D, CROSSBAR, RING |
| mesh_rows | Mesh network rows | 2-16 |
| mesh_cols | Mesh network columns | 2-16 |
| link_bandwidth_gbps | Link bandwidth | 10-1000 |

## Preset Configurations

### Behavioral Presets (Fast)

| Preset | Description |
|--------|-------------|
| `fast` | All-behavioral simulation for quick functional testing |
| `minimal` | Smallest hardware config (1 compute tile, 8x8 array) |

### Transactional Presets (Balanced)

| Preset | Description |
|--------|-------------|
| `balanced` | Mixed fidelity for design exploration |
| `edge_ai` | Edge AI configuration (2 tiles, 16x16 arrays) |
| `embodied_ai` | Robotics configuration (64 tiles, 24x24 arrays) |

### Cycle-Accurate Presets (Detailed)

| Preset | Description |
|--------|-------------|
| `accurate` | Full cycle-accurate simulation |
| `datacenter` | High-performance config (256 tiles, 32x32 arrays) |

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
  --preset <name>         Use preset: fast, balanced, accurate, mixed,
                          minimal, edge_ai, embodied_ai, datacenter
```

### Examples

```bash
# Basic matrix multiplication with minimal config
./build/tools/runner/kpu-runner --preset minimal -m 128x128x128

# MLP layer test with edge AI config
./build/tools/runner/kpu-runner --preset edge_ai -t mlp -m 128x64x128

# Benchmark suite on datacenter config
./build/tools/runner/kpu-runner --preset datacenter -t benchmark

# Use configuration file
./build/tools/runner/kpu-runner configs/components/kpu/minimal.json -m 256x256x256

# Verbose output with results export
./build/tools/runner/kpu-runner --preset accurate -m 512x512x512 -v -o results.json
```

## KPU Config Tool

The `kpu-config` tool helps manage configuration files:

```bash
# Show configuration details
./build/tools/configuration/kpu-config show configs/components/kpu/minimal.json

# Validate configuration
./build/tools/configuration/kpu-config validate configs/components/kpu/my_config.json

# Generate preset configuration file
./build/tools/configuration/kpu-config generate --preset edge_ai > edge_ai.json

# Convert between presets
./build/tools/configuration/kpu-config generate --preset datacenter --fidelity BEHAVIORAL
```

## Creating Custom Configurations

### Step 1: Start from a Template

```bash
# Generate a base config
./build/tools/configuration/kpu-config generate --preset minimal > my_config.json
```

### Step 2: Modify Parameters

Edit the JSON file to match your requirements:

```json
{
  "simulation": {
    "default_fidelity": "TRANSACTIONAL",
    "verification_level": "ASSERTIONS",
    "num_memory_controllers": 2,
    "num_dma_engines": 4,
    "num_l3_tiles": 4,
    "num_l2_banks": 16,
    "num_compute_tiles": 4
  },
  "memory": {
    "fidelity": "CYCLE_ACCURATE",
    "technology": "LPDDR5",
    "capacity_gb": 16,
    "channels": 4,
    "burst_length": 16,
    "clock_mhz": 3200
  },
  "compute": {
    "fidelity": "BEHAVIORAL",
    "technology": "SYSTOLIC",
    "array_size": [16, 16],
    "macs_per_cycle": 256
  }
}
```

### Step 3: Validate and Test

```bash
# Validate configuration
./build/tools/configuration/kpu-config validate my_config.json

# Run a test
./build/tools/runner/kpu-runner my_config.json -m 512x512x512 -v
```

## Cross-Validation Testing

To compare simulation results across fidelity levels, use matched configurations:

```bash
# Create matching configs at different fidelities
./build/tools/configuration/kpu-config generate --preset minimal --fidelity BEHAVIORAL > crossval_behavioral.json
./build/tools/configuration/kpu-config generate --preset minimal --fidelity TRANSACTIONAL > crossval_transactional.json
./build/tools/configuration/kpu-config generate --preset minimal --fidelity CYCLE_ACCURATE > crossval_cycle_accurate.json

# Run same workload on each
./build/tools/runner/kpu-runner crossval_behavioral.json -m 128x128x128 -o behavioral_results.json
./build/tools/runner/kpu-runner crossval_transactional.json -m 128x128x128 -o transactional_results.json
./build/tools/runner/kpu-runner crossval_cycle_accurate.json -m 128x128x128 -o cycle_accurate_results.json

# Compare results
# Results should match functionally; timing will vary by fidelity
```

## Programmatic Configuration (C++)

You can also create configurations programmatically:

```cpp
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

int main() {
    // Create configuration directly
    KPUSimulator::Config config;
    config.host_memory_region_count = 1;
    config.host_memory_region_capacity_mb = 256;
    config.host_memory_bandwidth_gbps = 50;
    config.memory_bank_count = 1;
    config.memory_bank_capacity_mb = 256;
    config.memory_bandwidth_gbps = 25;
    config.memory_controller_count = 1;
    config.page_buffer_count = 2;
    config.page_buffer_capacity_kb = 32;
    config.l3_tile_count = 1;
    config.l3_tile_capacity_kb = 128;
    config.l2_bank_count = 4;
    config.l2_bank_capacity_kb = 64;
    config.l1_buffer_count = 64;
    config.l1_buffer_capacity_kb = 64;
    config.compute_tile_count = 1;
    config.processor_array_rows = 8;
    config.processor_array_cols = 8;
    config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    config.use_systolic_array_mode = true;
    config.dma_engine_count = 2;
    config.block_mover_count = 2;
    config.streamer_count = 4;

    // Create simulator
    KPUSimulator sim(config);

    // Run your workload...
    return 0;
}
```

## Troubleshooting

### Configuration Not Loading

**Error**: `Failed to load configuration`

**Solutions**:
1. Check file exists: `ls -la configs/components/kpu/your_config.json`
2. Validate JSON syntax with `jq . your_config.json`
3. Use the config tool: `./build/tools/configuration/kpu-config validate your_config.json`

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

### Preset Not Found

**Error**: `Unknown preset: xyz`

**Solution**: Use one of: `fast`, `balanced`, `accurate`, `mixed`, `minimal`, `edge_ai`, `embodied_ai`, `datacenter`

### Low Performance Numbers

If GFLOPS seems low:

1. **Matrix too small**: Small matrices have high overhead-to-compute ratio
2. **Wrong fidelity**: CYCLE_ACCURATE is slower but more precise
3. **Use benchmark mode**: Compare across sizes to see scaling

```bash
# See how performance scales with size
./build/tools/runner/kpu-runner --preset datacenter -t benchmark
```

## Configuration Files Location

```
kpu-sim/
├── configs/
│   ├── components/
│   │   └── kpu/                      # KPU component configs
│   │       ├── minimal.json              # Minimal hardware (1 tile, 8x8)
│   │       ├── edge_ai.json              # Edge AI (2 tiles, 16x16)
│   │       ├── embodied_ai.json          # Robotics (64 tiles, 24x24)
│   │       ├── datacenter.json           # Datacenter (256 tiles, 32x32)
│   │       ├── crossval_behavioral.json  # Cross-validation BEHAVIORAL
│   │       ├── crossval_transactional.json
│   │       └── crossval_cycle_accurate.json
│   └── systems/                      # Full system configs
│       ├── minimal_kpu.json          # Minimal single-KPU system
│       ├── edge_ai.json              # Edge AI system
│       └── datacenter_hbm.json       # HBM datacenter system
├── tools/
│   ├── runner/
│   │   └── kpu_runner.cpp            # Runner implementation
│   └── configuration/
│       └── kpu-config.cpp            # Config tool
└── include/
    └── sw/kpu/
        └── config/
            └── simulator_config_parser.hpp  # Config parser API
```

## See Also

- [KPU Architecture](kpu_architecture.md) - Detailed hardware architecture
- [Memory Hierarchy](unified-address-space.md) - Memory addressing
- [Python Integration](how-to-build-and-use-python-bindings.md) - Python API
