# JSON Configuration System Architecture

## Overview

The Stillwater System Simulator uses a hierarchical JSON-based configuration system to define heterogeneous computing systems. This document provides UML diagrams and architectural overview of the configuration classes.

## High-Level Architecture

```mermaid
graph TB
    User[User/Application]

    User -->|creates| Sim[SystemSimulator]
    User -->|loads JSON| Config[SystemConfig]
    User -->|uses| Loader[ConfigLoader]

    Loader -->|reads/writes| JSON[(JSON Files)]
    Loader -->|creates| Config

    Sim -->|uses| Config
    Sim -->|instantiates| KPU[KPU Instances]

    Config -->|validates| Config
    Config -->|contains| Host[Host Config]
    Config -->|contains| Accel[Accelerator Configs]
    Config -->|contains| Inter[Interconnect Config]

    style Sim fill:#e1f5ff
    style Config fill:#fff4e1
    style Loader fill:#e7f5e7
    style JSON fill:#f0f0f0
```

## Core Classes

### Main System Classes

```mermaid
classDiagram
    class SystemSimulator {
        -bool initialized_
        -SystemConfig config_
        -vector~unique_ptr~KPUSimulator~~ kpu_instances_
        +SystemSimulator()
        +SystemSimulator(SystemConfig)
        +SystemSimulator(filesystem::path)
        +initialize() bool
        +initialize(SystemConfig) bool
        +load_config_and_initialize(path) bool
        +shutdown() void
        +get_config() SystemConfig
        +get_kpu_count() size_t
        +get_kpu(index) KPUSimulator*
        +get_kpu_by_id(id) KPUSimulator*
        +print_config() void
        +run_self_test() bool
    }

    class ConfigLoader {
        <<static>>
        +load_from_file(path)$ SystemConfig
        +load_from_string(json)$ SystemConfig
        +save_to_file(config, path)$ void
        +to_json_string(config, pretty)$ string
        +validate_file(path)$ bool
        +get_validation_errors(path)$ vector~string~
    }

    class SystemConfig {
        +SystemInfo system
        +HostConfig host
        +vector~AcceleratorConfig~ accelerators
        +InterconnectConfig interconnect
        +SystemServicesConfig system_services
        +validate() bool
        +get_validation_errors() string
        +get_kpu_count() size_t
        +get_gpu_count() size_t
        +get_tpu_count() size_t
        +get_npu_count() size_t
        +find_accelerator(id) AcceleratorConfig*
        +create_minimal_kpu()$ SystemConfig
        +create_edge_ai()$ SystemConfig
        +create_datacenter()$ SystemConfig
    }

    SystemSimulator --> SystemConfig : uses
    SystemSimulator --> KPUSimulator : creates
    ConfigLoader ..> SystemConfig : serializes
```

## Configuration Hierarchy

### System Configuration Structure

```mermaid
classDiagram
    class SystemConfig {
        +SystemInfo system
        +HostConfig host
        +AcceleratorConfig[] accelerators
        +InterconnectConfig interconnect
        +SystemServicesConfig system_services
    }

    class SystemInfo {
        +string name
        +string description
        +optional~uint32_t~ clock_frequency_mhz
    }

    class HostConfig {
        +CPUConfig cpu
        +HostMemoryConfig memory
        +StorageConfig[] storage
    }

    class AcceleratorConfig {
        +AcceleratorType type
        +string id
        +string description
        +optional~KPUConfig~ kpu_config
        +optional~GPUConfig~ gpu_config
        +optional~TPUConfig~ tpu_config
        +optional~NPUConfig~ npu_config
    }

    class InterconnectConfig {
        +HostToAcceleratorConfig host_to_accelerator
        +AcceleratorToAcceleratorConfig accelerator_to_accelerator
        +OnChipConfig on_chip
        +optional~NetworkConfig~ network
    }

    class SystemServicesConfig {
        +MemoryManagerConfig memory_manager
        +InterruptControllerConfig interrupt_controller
        +PowerManagementConfig power_management
    }

    SystemConfig *-- SystemInfo
    SystemConfig *-- HostConfig
    SystemConfig *-- AcceleratorConfig
    SystemConfig *-- InterconnectConfig
    SystemConfig *-- SystemServicesConfig
```

### Host Configuration

```mermaid
classDiagram
    class HostConfig {
        +CPUConfig cpu
        +HostMemoryConfig memory
        +StorageConfig[] storage
    }

    class CPUConfig {
        +uint32_t core_count
        +uint32_t frequency_mhz
        +uint32_t cache_l1_kb
        +uint32_t cache_l2_kb
        +uint32_t cache_l3_kb
    }

    class HostMemoryConfig {
        +DRAMControllerConfig dram_controller
        +MemoryModuleConfig[] modules
    }

    class DRAMControllerConfig {
        +uint32_t channel_count
        +uint32_t data_width_bits
    }

    class MemoryModuleConfig {
        +string id
        +string type
        +string form_factor
        +uint32_t capacity_gb
        +uint32_t frequency_mhz
        +float bandwidth_gbps
        +uint32_t latency_ns
        +uint32_t channels
    }

    class StorageConfig {
        +string id
        +string type
        +uint32_t capacity_gb
        +uint32_t read_bandwidth_mbps
        +uint32_t write_bandwidth_mbps
        +uint32_t latency_us
    }

    HostConfig *-- CPUConfig
    HostConfig *-- HostMemoryConfig
    HostConfig *-- StorageConfig
    HostMemoryConfig *-- DRAMControllerConfig
    HostMemoryConfig *-- MemoryModuleConfig

    note for HostMemoryConfig "DIMM-based memory\nDDR4, DDR5, LPDDR"
    note for MemoryModuleConfig "Standardized modules\nDIMM, SODIMM, OnPackage"
```

### Accelerator Configuration

```mermaid
classDiagram
    class AcceleratorConfig {
        +AcceleratorType type
        +string id
        +string description
        +optional~KPUConfig~ kpu_config
        +optional~GPUConfig~ gpu_config
        +optional~NPUConfig~ npu_config
    }

    class AcceleratorType {
        <<enumeration>>
        KPU
        GPU
        TPU
        NPU
        CGRA
        DSP
        FPGA
    }

    class KPUConfig {
        +KPUMemoryConfig memory
        +KPUComputeConfig compute_fabric
        +KPUDataMovementConfig data_movement
    }

    class GPUConfig {
        +uint32_t compute_units
        +uint32_t clock_mhz
        +GPUMemoryConfig memory
    }

    class NPUConfig {
        +uint32_t tops_int8
        +uint32_t tops_fp16
        +NPUMemoryConfig memory
    }

    AcceleratorConfig --> AcceleratorType
    AcceleratorConfig --> KPUConfig
    AcceleratorConfig --> GPUConfig
    AcceleratorConfig --> NPUConfig

    note for AcceleratorConfig "Each accelerator\nowns its memory\nsubsystem"
```

### KPU Configuration Details

```mermaid
classDiagram
    class KPUConfig {
        +KPUMemoryConfig memory
        +KPUComputeConfig compute_fabric
        +KPUDataMovementConfig data_movement
    }

    class KPUMemoryConfig {
        +string type
        +string form_factor
        +KPUMemoryBankConfig[] banks
        +KPUTileConfig[] l3_tiles
        +KPUTileConfig[] l2_banks
        +KPUScratchpadConfig[] scratchpads
    }

    class KPUMemoryBankConfig {
        +string id
        +uint32_t capacity_mb
        +float bandwidth_gbps
        +uint32_t latency_ns
    }

    class KPUTileConfig {
        +string id
        +uint32_t capacity_kb
    }

    class KPUScratchpadConfig {
        +string id
        +uint32_t capacity_kb
    }

    class KPUComputeConfig {
        +ComputeTileConfig[] tiles
    }

    class ComputeTileConfig {
        +string id
        +string type
        +uint32_t systolic_rows
        +uint32_t systolic_cols
        +string datatype
    }

    class KPUDataMovementConfig {
        +DMAEngineConfig[] dma_engines
        +BlockMoverConfig[] block_movers
        +StreamerConfig[] streamers
    }

    KPUConfig *-- KPUMemoryConfig
    KPUConfig *-- KPUComputeConfig
    KPUConfig *-- KPUDataMovementConfig

    KPUMemoryConfig *-- KPUMemoryBankConfig
    KPUMemoryConfig *-- KPUTileConfig
    KPUMemoryConfig *-- KPUScratchpadConfig

    KPUComputeConfig *-- ComputeTileConfig

    KPUDataMovementConfig *-- DMAEngineConfig
    KPUDataMovementConfig *-- BlockMoverConfig
    KPUDataMovementConfig *-- StreamerConfig

    note for KPUMemoryConfig "Custom layouts\nGDDR6, HBM2, HBM3\nSubstrate, Interposer"
```

### Interconnect Configuration

```mermaid
classDiagram
    class InterconnectConfig {
        +HostToAcceleratorConfig host_to_accelerator
        +AcceleratorToAcceleratorConfig accelerator_to_accelerator
        +OnChipConfig on_chip
        +optional~NetworkConfig~ network
    }

    class HostToAcceleratorConfig {
        +string type
        +optional~PCIeConfig~ pcie_config
        +optional~CXLConfig~ cxl_config
    }

    class PCIeConfig {
        +uint32_t generation
        +uint32_t lanes
        +float bandwidth_gbps
    }

    class CXLConfig {
        +string version
        +float bandwidth_gbps
    }

    class AcceleratorToAcceleratorConfig {
        +string type
        +optional~NoCConfig~ noc_config
    }

    class NoCConfig {
        +string topology
        +uint32_t router_count
        +float link_bandwidth_gbps
    }

    class OnChipConfig {
        +string type
        +optional~AMBAConfig~ amba_config
    }

    class NetworkConfig {
        +bool enabled
        +string type
        +uint32_t speed_gbps
    }

    InterconnectConfig *-- HostToAcceleratorConfig
    InterconnectConfig *-- AcceleratorToAcceleratorConfig
    InterconnectConfig *-- OnChipConfig
    InterconnectConfig *-- NetworkConfig

    HostToAcceleratorConfig *-- PCIeConfig
    HostToAcceleratorConfig *-- CXLConfig

    AcceleratorToAcceleratorConfig *-- NoCConfig

    OnChipConfig *-- AMBAConfig
```

## Memory Architecture

### Memory Subsystem Ownership

```mermaid
graph TB
    subgraph System["Computing System"]
        subgraph Host["Host Subsystem"]
            CPU[CPU Cores]
            HostMem["Host Memory<br/>(DIMM-based)<br/>DDR4/DDR5/LPDDR"]
            CPU <--> HostMem
        end

        subgraph KPU["KPU Subsystem"]
            KPUCompute[Compute Tiles<br/>Systolic Arrays]
            KPUMem["KPU Memory<br/>(Custom Layout)<br/>GDDR6/HBM2/HBM3"]
            KPUCompute <--> KPUMem
        end

        subgraph GPU["GPU Subsystem"]
            GPUCompute[GPU Cores]
            GPUMem["GPU Memory<br/>(Custom Layout)<br/>GDDR6/HBM3"]
            GPUCompute <--> GPUMem
        end

        subgraph NPU["NPU Subsystem"]
            NPUCompute[NPU Cores]
            NPUMem["NPU Memory<br/>(OnChip/LPDDR)"]
            NPUCompute <--> NPUMem
        end
    end

    Host <-->|PCIe/CXL| KPU
    Host <-->|PCIe| GPU
    Host <-->|NoC| NPU
    KPU <-->|NVLink/NoC| GPU

    style HostMem fill:#e1f5ff
    style KPUMem fill:#ffe1f5
    style GPUMem fill:#f5ffe1
    style NPUMem fill:#fff5e1
```

## Configuration Workflows

### JSON File Loading

```mermaid
sequenceDiagram
    participant User
    participant SystemSimulator
    participant ConfigLoader
    participant SystemConfig
    participant JSONFile

    User->>SystemSimulator: new SystemSimulator("config.json")
    SystemSimulator->>ConfigLoader: load_from_file("config.json")
    ConfigLoader->>JSONFile: read file
    JSONFile-->>ConfigLoader: JSON data
    ConfigLoader->>ConfigLoader: parse JSON
    ConfigLoader->>SystemConfig: create from JSON
    SystemConfig->>SystemConfig: validate()
    SystemConfig-->>ConfigLoader: validated config
    ConfigLoader-->>SystemSimulator: SystemConfig

    User->>SystemSimulator: initialize()
    SystemSimulator->>SystemConfig: validate()
    SystemConfig-->>SystemSimulator: OK
    SystemSimulator->>SystemSimulator: create_components_from_config()
    SystemSimulator->>KPUSimulator: create KPU instances
    SystemSimulator-->>User: initialized
```

### Programmatic Configuration

```mermaid
sequenceDiagram
    participant User
    participant SystemConfig
    participant SystemSimulator

    User->>SystemConfig: create_minimal_kpu()
    SystemConfig->>SystemConfig: build default config
    SystemConfig-->>User: SystemConfig

    User->>User: customize config

    User->>SystemConfig: validate()
    SystemConfig-->>User: validation result

    User->>SystemSimulator: new SystemSimulator(config)
    SystemSimulator->>SystemSimulator: store config

    User->>SystemSimulator: initialize()
    SystemSimulator->>SystemSimulator: create components
    SystemSimulator-->>User: initialized
```

### Configuration Round-Trip

```mermaid
sequenceDiagram
    participant User
    participant SystemConfig
    participant ConfigLoader
    participant JSONFile

    User->>SystemConfig: create_edge_ai()
    SystemConfig-->>User: config

    User->>ConfigLoader: save_to_file(config, "my_config.json")
    ConfigLoader->>ConfigLoader: serialize to JSON
    ConfigLoader->>JSONFile: write JSON
    JSONFile-->>ConfigLoader: OK
    ConfigLoader-->>User: saved

    Note over User,JSONFile: Later...

    User->>ConfigLoader: load_from_file("my_config.json")
    ConfigLoader->>JSONFile: read JSON
    JSONFile-->>ConfigLoader: JSON data
    ConfigLoader->>SystemConfig: deserialize
    SystemConfig-->>ConfigLoader: config
    ConfigLoader-->>User: loaded config
```

## Validation Rules

The configuration system enforces several validation rules:

1. **Required Fields**
   - System name must be non-empty
   - At least one host memory module required

2. **Component Constraints**
   - KPU must have at least one memory bank
   - KPU must have at least one compute tile
   - Scratchpad count ≥ compute tile count (recommended)
   - DMA engine count ≤ memory bank count × 2

3. **Interconnect Consistency**
   - Multi-accelerator systems should define accelerator-to-accelerator interconnect
   - Interconnect bandwidth should not create obvious bottlenecks

4. **Memory Technology Constraints**
   - Host: DDR/LPDDR only (DIMM-based)
   - KPU/GPU: GDDR/HBM (custom layouts)
   - NPU: OnChip/LPDDR (power-efficient)

## Example Configurations

### Minimal KPU System

```json
{
  "system": { "name": "Minimal KPU System" },
  "host": {
    "cpu": { "core_count": 4, "frequency_mhz": 2400 },
    "memory": {
      "modules": [
        { "id": "mem_0", "type": "DDR4", "capacity_gb": 16 }
      ]
    }
  },
  "accelerators": [
    {
      "type": "KPU",
      "id": "kpu_0",
      "kpu_config": {
        "memory": {
          "type": "GDDR6",
          "banks": [
            { "id": "bank_0", "capacity_mb": 1024, "bandwidth_gbps": 100 }
          ],
          "scratchpads": [
            { "id": "scratch_0", "capacity_kb": 64 }
          ]
        },
        "compute_fabric": {
          "tiles": [
            { "id": "tile_0", "systolic_rows": 16, "systolic_cols": 16 }
          ]
        }
      }
    }
  ]
}
```

### Edge AI System (Multi-Accelerator)

- KPU with LPDDR5 (power-efficient)
- NPU for CNN inference
- NoC interconnect between accelerators

### Datacenter System (High-End)

- KPU with HBM3 (ultra-high bandwidth)
- GPU with HBM3
- PCIe Gen5 host interconnect
- NVLink accelerator-to-accelerator

## File Structure

```
KPU-simulator/
├── configs/
│   ├── schema.md                    # JSON schema reference
│   ├── README.md                    # User guide
│   └── examples/
│       ├── minimal_kpu.json         # Basic configuration
│       ├── edge_ai.json             # Edge device
│       └── datacenter_hbm.json      # Datacenter node
├── include/sw/system/
│   ├── system_config.hpp            # Configuration structures
│   ├── config_loader.hpp            # JSON I/O
│   └── toplevel.hpp                 # System simulator
├── src/system/
│   ├── system_config.cpp            # Config implementation
│   ├── config_loader.cpp            # JSON parsing
│   └── toplevel.cpp                 # Simulator implementation
├── tests/system/
│   └── test_system_config.cpp       # Configuration tests
└── examples/basic/
    └── system_config_demo.cpp       # Usage examples
```

## Key Design Decisions

1. **Memory Subsystem Separation**
   - Each accelerator owns its memory subsystem
   - Reflects real hardware architecture (DIMMs vs custom layouts)

2. **Hierarchical Composition**
   - Tree structure mirrors physical system organization
   - Easy to navigate and understand

3. **Type-Safe Configuration**
   - Strong typing with enums and structs
   - Compile-time type checking

4. **Validation First**
   - All configs validated before use
   - Clear error messages for debugging

5. **Multiple Workflows**
   - JSON files for reproducibility
   - Factory methods for quick start
   - Programmatic for dynamic generation

## References

- [Configuration User Guide](../configs/README.md)
- [JSON Schema Reference](../configs/schema.md)
- [Example Configurations](../configs/examples/)
- [Implementation Guide](json-configuration-system.md)
