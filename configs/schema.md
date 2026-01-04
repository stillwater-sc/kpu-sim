# Configuration Schema Documentation

## Overview

The KPU simulator uses two types of JSON configuration files:

1. **Component Configs** (`configs/components/kpu/`) - Define KPU hardware parameters for simulation
2. **System Configs** (`configs/systems/`) - Define complete heterogeneous systems with host, accelerators, and interconnects

## Directory Structure

```
configs
├── README.md
├── components/
│   ├── gpu/
│   ├── kpu/                          # KPU component configurations
│   │   ├── minimal.json              # Minimal hardware (  1 tile,   8x8,  DDR4)
│   │   ├── edge_ai.json              # Edge AI          (  2 tiles, 16x16, LPDDR5)
│   │   ├── embodied_ai.json          # Robotics         ( 64 tiles, 24x24, LPDDR5)
│   │   └── datacenter.json           # Datacenter       (256 tiles, 32x32, HBM3)
│   └── npu
├── cross-validation/                 # Configurations to validate simulators through cross-validation
│   └── components/
│       └── kpu/
│           ├── behavioral.json
│           ├── cycle_accurate.json
│           └── transactional.json
├── systems/                          # Full system configurations
│   ├── minimal_kpu.json              # Single KPU system
│   ├── edge_ai.json                  # Edge AI system
│   └── datacenter_hbm.json           # HBM datacenter system
└── schema.md                         # This file
```

---

## Part 1: KPU Component Configuration Schema

Component configs define KPU hardware for the `kpu-runner` simulation tool.

### Schema Structure

```json
{
  "simulation": {
    "default_fidelity": "BEHAVIORAL|TRANSACTIONAL|CYCLE_ACCURATE",
    "verification_level": "NONE|ASSERTIONS|INVARIANTS|PROTOCOL",
    "enable_tracing": boolean
  },

  "kpu": {
    "memory": {
      "fidelity": "BEHAVIORAL|TRANSACTIONAL|CYCLE_ACCURATE",
      "technology": "LPDDR5|LPDDR5X|DDR5|HBM3|HBM3E|GDDR6|GDDR7",
      "capacity_gb": integer,
      "controllers": integer,
      "channels_per_controller": integer,
      "banks_per_channel": integer,
      "speed_mt_s": integer
    },

    "dma": {
      "fidelity": "BEHAVIORAL|TRANSACTIONAL|CYCLE_ACCURATE",
      "engines": integer,
      "channels_per_engine": integer,
      "bandwidth_gbps": integer
    },

    "l3": {
      "fidelity": "BEHAVIORAL|TRANSACTIONAL|CYCLE_ACCURATE",
      "tiles": integer,
      "capacity_kb": integer,
      "banks_per_tile": integer,
      "ports_per_tile": integer
    },

    "l2": {
      "fidelity": "BEHAVIORAL|TRANSACTIONAL|CYCLE_ACCURATE",
      "banks": integer,
      "capacity_kb": integer
    },

    "compute": {
      "fidelity": "BEHAVIORAL|TRANSACTIONAL|CYCLE_ACCURATE",
      "technology": "INT8_SYSTOLIC|FP16_SYSTOLIC|BF16_SYSTOLIC|FP32_SIMD|MIXED_PRECISION",
      "tiles": integer,
      "array_rows": integer,
      "array_cols": integer,
      "macs_per_cycle": integer
    },

    "noc": {
      "fidelity": "BEHAVIORAL|TRANSACTIONAL|CYCLE_ACCURATE",
      "topology": "MESH_2D|TORUS_2D|HIERARCHICAL",
      "rows": integer,
      "cols": integer,
      "link_bandwidth_gbps": integer
    }
  }
}
```

### Simulation Fidelity Levels

| Fidelity | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| BEHAVIORAL | Fastest | Low | Functional verification, early prototyping |
| TRANSACTIONAL | Medium | Medium | Performance estimation, design exploration |
| CYCLE_ACCURATE | Slowest | High | Detailed timing analysis, hardware validation |

Each component can have independent fidelity, enabling mixed-fidelity simulation (e.g., accurate memory with fast compute).

### Example: Minimal Configuration

```json
{
  "simulation": {
    "default_fidelity": "BEHAVIORAL",
    "verification_level": "ASSERTIONS",
    "enable_tracing": false
  },
  "kpu": {
    "memory": {
      "fidelity": "BEHAVIORAL",
      "technology": "LPDDR5",
      "capacity_gb": 1,
      "controllers": 1,
      "channels_per_controller": 1,
      "banks_per_channel": 16,
      "speed_mt_s": 6400
    },
    "dma": {
      "fidelity": "BEHAVIORAL",
      "engines": 1,
      "channels_per_engine": 1,
      "bandwidth_gbps": 50
    },
    "l3": {
      "fidelity": "BEHAVIORAL",
      "tiles": 1,
      "capacity_kb": 64,
      "banks_per_tile": 4,
      "ports_per_tile": 2
    },
    "l2": {
      "fidelity": "BEHAVIORAL",
      "banks": 4,
      "capacity_kb": 64
    },
    "compute": {
      "fidelity": "BEHAVIORAL",
      "technology": "INT8_SYSTOLIC",
      "tiles": 1,
      "array_rows": 8,
      "array_cols": 8,
      "macs_per_cycle": 64
    },
    "noc": {
      "fidelity": "BEHAVIORAL",
      "topology": "MESH_2D",
      "rows": 1,
      "cols": 1,
      "link_bandwidth_gbps": 64
    }
  }
}
```

### Hardware Scale Presets

| Config | Tiles | Array Size | Memory | Use Case |
|--------|-------|------------|--------|----------|
| minimal | 1 | 8x8 | 1GB LPDDR5 | Testing, debugging |
| edge_ai | 2 | 16x16 | 4GB LPDDR5 | Edge devices, IoT |
| embodied_ai | 64 | 24x24 | 16GB LPDDR5 | Robotics, autonomous systems |
| datacenter | 256 | 32x32 | 32GB HBM3 | Cloud, data centers |

---

## Part 2: System Configuration Schema

System configs define complete heterogeneous computing systems for system-level simulation.

### Schema Structure

```json
{
  "system": {
    "name": "string",
    "description": "string",
    "clock_frequency_mhz": integer
  },

  "host": {
    "cpu": {
      "core_count": integer,
      "frequency_mhz": integer,
      "cache_l1_kb": integer,
      "cache_l2_kb": integer,
      "cache_l3_kb": integer
    },
    "memory": {
      "dram_controller": {
        "channel_count": integer,
        "data_width_bits": integer
      },
      "modules": [
        {
          "id": "string",
          "type": "DDR4|DDR5|LPDDR4|LPDDR5",
          "form_factor": "DIMM|SODIMM|LPDIMM|OnPackage",
          "capacity_gb": integer,
          "frequency_mhz": integer,
          "bandwidth_gbps": number,
          "latency_ns": integer,
          "channels": integer
        }
      ]
    },
    "storage": [
      {
        "id": "string",
        "type": "SSD|NVME|HDD",
        "capacity_gb": integer,
        "read_bandwidth_mbps": integer,
        "write_bandwidth_mbps": integer,
        "latency_us": integer
      }
    ]
  },

  "accelerators": [
    {
      "type": "KPU|GPU|NPU|DSP|FPGA",
      "id": "string",
      "description": "string",

      "kpu": { ... },      // KPU-specific config (see below)
      "gpu_config": { ... },  // GPU-specific config
      "npu_config": { ... }   // NPU-specific config
    }
  ],

  "interconnect": {
    "host_to_accelerator": {
      "type": "PCIe|CXL|NVLink|CustomFabric",
      "pcie_config": {
        "generation": integer,
        "lanes": integer,
        "bandwidth_gbps": number
      },
      "cxl_config": {
        "version": "1.0|2.0|3.0",
        "bandwidth_gbps": number
      }
    },
    "accelerator_to_accelerator": {
      "type": "NVLink|InfinityFabric|NoC|None",
      "noc_config": {
        "topology": "mesh|torus|ring|crossbar",
        "router_count": integer,
        "link_bandwidth_gbps": number
      }
    },
    "on_chip": {
      "type": "AMBA|CHI|TileLink|Custom",
      "amba_config": {
        "protocol": "AXI4|AXI5|ACE|CHI"
      }
    },
    "network": {
      "enabled": boolean,
      "type": "Ethernet|RoCE|InfiniBand",
      "speed_gbps": integer
    }
  },

  "system_services": {
    "memory_manager": {
      "enabled": boolean,
      "pool_size_mb": integer,
      "alignment_bytes": integer
    },
    "interrupt_controller": {
      "enabled": boolean
    },
    "power_management": {
      "enabled": boolean
    }
  }
}
```

### KPU Configuration in System Configs

Within system configs, KPU accelerators use the `kpu` key (note: `kpu_config` is also supported for backwards compatibility):

```json
{
  "type": "KPU",
  "id": "kpu_0",
  "kpu": {
    "memory": {
      "type": "GDDR6|HBM2|HBM3|Custom",
      "form_factor": "Substrate|PCB|Interposer|3DStack",
      "banks": [
        { "id": "bank_0", "capacity_mb": 1024, "bandwidth_gbps": 100, "latency_ns": 20 }
      ],
      "l3_tiles": [
        { "id": "l3_0", "capacity_kb": 128 }
      ],
      "l2_banks": [
        { "id": "l2_0", "capacity_kb": 64 }
      ],
      "scratchpads": [
        { "id": "scratch_0", "capacity_kb": 64 }
      ]
    },
    "compute_fabric": {
      "tiles": [
        { "id": "tile_0", "type": "systolic", "systolic_rows": 16, "systolic_cols": 16, "datatype": "fp32" }
      ]
    },
    "data_movement": {
      "dma_engines": [
        { "id": "dma_0", "bandwidth_gbps": 50, "channels": 1 }
      ],
      "block_movers": [
        { "id": "block_mover_0" }
      ],
      "streamers": [
        { "id": "streamer_0" }
      ]
    }
  }
}
```

### Using $ref for Composability

System configs can reference external component configs using `$ref`:

```json
{
  "system": { "name": "Composable System" },
  "accelerators": [
    {
      "type": "KPU",
      "id": "kpu_0",
      "kpu": { "$ref": "../components/kpu/datacenter.json" }
    }
  ]
}
```

The `$ref` path is resolved relative to the config file location.

---

## Architecture Principles

### 1. Memory Subsystem Ownership
- **Host**: DDR/LPDDR modules on DIMMs, optimized for capacity and cost
- **KPU**: GDDR/HBM on substrate/interposer, optimized for bandwidth
- **GPU**: Similar to KPU, high-bandwidth memory close to compute
- **NPU**: Often LPDDR for power efficiency or on-chip SRAM

### 2. Form Factors
- **DIMM/SODIMM**: Standardized modules for host memory
- **Substrate**: Custom memory layout for accelerators
- **Interposer**: 2.5D integration (HBM)
- **3DStack**: 3D stacked memory (HBM3)

### 3. Hierarchical Interconnect
- **Host ↔ Accelerator**: PCIe, CXL (for memory coherence)
- **Accelerator ↔ Accelerator**: High-speed links or NoC
- **On-chip**: AMBA/CHI for internal communication

---

## Validation Rules

1. **Component Ownership**: Each accelerator must define its memory subsystem
2. **Interconnect Consistency**: Multiple accelerators require accelerator-to-accelerator interconnect definition
3. **Bandwidth Matching**: Interconnect bandwidth should not create bottlenecks
4. **Memory Technology Constraints**:
   - DDR/LPDDR: Host only
   - GDDR: Accelerators only
   - HBM: Accelerators only (GPUs, high-end KPUs)
   - OnChip: NPUs and embedded processors

---

## Tools

### kpu-runner (Component Simulation)
```bash
# Run with component config
./build/tools/runner/kpu-runner configs/components/kpu/minimal.json -m 256x256x256

# Run with preset
./build/tools/runner/kpu-runner --preset datacenter -m 1024x1024x1024
```

### kpu-config (Configuration Management)
```bash
# Show configuration details
./build/tools/configuration/kpu-config show configs/components/kpu/minimal.json

# Generate preset configuration
./build/tools/configuration/kpu-config generate --preset edge_ai > my_config.json
```

### ConfigLoader (System Simulation)
```cpp
#include <sw/system/config_loader.hpp>

auto config = sw::sim::ConfigLoader::load_from_file("configs/systems/minimal_kpu.json");
```

---

## Example Configurations

See:
- `configs/components/kpu/` - KPU hardware scale presets
- `configs/systems/minimal_kpu.json` - Single KPU with basic memory
- `configs/systems/edge_ai.json` - Edge AI system configuration
- `configs/systems/datacenter_hbm.json` - High-performance datacenter with HBM
