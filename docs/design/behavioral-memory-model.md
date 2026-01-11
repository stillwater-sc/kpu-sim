# Behavioral Memory Model Design

## Overview

The `BehavioralMemoryModel` class (`sw::kpu::behavioral::BehavioralMemoryModel`) provides
a unified memory abstraction for functional simulation of the KPU. Unlike cycle-accurate
simulation which models timing, this model stores and manipulates **actual data values**
to enable functional verification of software.

## Purpose

The behavioral memory model serves several purposes:

1. **Functional Simulation**: Store actual tensor values (weights, activations, outputs)
2. **Data Movement Modeling**: Track data as it moves through Host → L3 → L2 → L1
3. **Resource Tracking**: Monitor allocation and usage across memory hierarchy
4. **Software Validation**: Enable verification that computed results are correct

## Memory Hierarchy

The KPU has a four-level memory hierarchy:

```
┌─────────────────────────────────────────────────────────────────┐
│                         HOST MEMORY                              │
│  - Graph structures, weights, inputs, outputs                   │
│  - Accessible via DMA                                           │
│  - Capacity: configurable (default 1 GB)                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ DMA Engine
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         L3 TILES                                 │
│  - On-chip memory tiles                                         │
│  - Multiple tiles (default 4 × 128 KB = 512 KB)                 │
│  - Data arrives here from Host via DMA                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │ BlockMover
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         L2 BANKS                                 │
│  - Working memory for compute tiles                             │
│  - Multiple banks (default 8 × 64 KB = 512 KB)                  │
│  - Data must be here before compute can start                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │ Streamers
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       L1 BUFFERS                                 │
│  - Streaming buffers attached to compute tile ports             │
│  - Multiple buffers (default 16 × 16 KB = 256 KB)               │
│  - Push data into compute tile ingress ports                    │
└─────────────────────────────────────────────────────────────────┘
```

## Address Encoding

The model uses a 64-bit address space with embedded region information:

```
┌──────────────┬────────────────────────┬────────────────────────────────┐
│ Region Type  │      Region ID         │           Offset               │
│   (8 bits)   │      (24 bits)         │          (32 bits)             │
└──────────────┴────────────────────────┴────────────────────────────────┘
     Bits 56-63        Bits 32-55               Bits 0-31
```

**Region Types:**
| Type | Value | Description |
|------|-------|-------------|
| HOST | 0 | Host CPU memory |
| L3_TILE | 1 | L3 on-chip tile |
| L2_BANK | 2 | L2 on-chip bank |
| L1_BUFFER | 3 | L1 streaming buffer |

**Example Addresses:**
```cpp
// Host memory at offset 0x1000
Address host_addr = 0x00'000000'00001000;  // Type=0, ID=0, Offset=0x1000

// L3 Tile 2 at offset 0x500
Address l3_addr = 0x01'000002'00000500;    // Type=1, ID=2, Offset=0x500

// L2 Bank 5 at offset 0x100
Address l2_addr = 0x02'000005'00000100;    // Type=2, ID=5, Offset=0x100
```

## Allocation Model

The memory model uses a simple **bump allocator** per region:

```
Region Before Allocation:
┌────────────────────────────────────────────────────────┐
│ Used │                  Available                      │
└──────┴─────────────────────────────────────────────────┘
       ↑
   next_offset

After allocate(1024):
┌────────────────────────────────────────────────────────┐
│ Used │ New Alloc (1024) │        Available             │
└──────┴──────────────────┴──────────────────────────────┘
                          ↑
                      next_offset
```

**Allocation API:**
```cpp
// Allocate in specific region
Allocation host_alloc = memory.allocate_host(1024, "weights");
Allocation l3_alloc = memory.allocate_l3(tile_id, 4096, "input_tile");
Allocation l2_alloc = memory.allocate_l2(bank_id, 2048);
Allocation l1_alloc = memory.allocate_l1(buffer_id, 512);

// Find by name
auto found = memory.find_allocation("weights");
```

## Data Operations

### Write and Read

```cpp
// Write raw data
float weights[256];
memory.write(alloc.base_address, weights, sizeof(weights));

// Read raw data
float result[256];
memory.read(alloc.base_address, result, sizeof(result));

// Typed convenience methods
memory.write_floats(addr, {1.0f, 2.0f, 3.0f, 4.0f});
std::vector<float> data = memory.read_floats(addr, 4);
```

### Copy (Data Movement)

The `copy()` operation models data movement between regions:

```cpp
// DMA: Host → L3
memory.copy(l3_alloc.base_address, host_alloc.base_address, 4096);

// BlockMover: L3 → L2
memory.copy(l2_alloc.base_address, l3_alloc.base_address, 2048);

// Streamer: L2 → L1
memory.copy(l1_alloc.base_address, l2_alloc.base_address, 512);
```

### Direct Pointer Access

For compute operations, direct pointer access is available:

```cpp
// Get typed pointer for computation
const float* A = memory.get_ptr<float>(l2_a_addr);
const float* B = memory.get_ptr<float>(l2_b_addr);
float* C = memory.get_ptr<float>(l2_c_addr);

// Perform matmul directly on memory
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[i*K + k] * B[k*N + j];
        }
        C[i*N + j] = sum;
    }
}
```

## Tensor Location Tracking

The model tracks where named tensors reside across the hierarchy:

```cpp
// Register tensor location
memory.register_tensor("weights", host_alloc);

// After DMA, update location
memory.update_tensor_location("weights", l3_alloc);

// Query location
auto location = memory.locate_tensor("weights");
if (location->exists_in(MemoryRegionType::L3_TILE)) {
    auto l3_loc = location->find_in(MemoryRegionType::L3_TILE);
    // Use l3_loc->base_address
}
```

This enables tracking data replication (tensor may exist in multiple levels simultaneously).

## Configuration

```cpp
MemoryModelConfig config;
config.host_capacity_bytes = 1ULL << 30;      // 1 GB
config.l3_tile_count = 4;
config.l3_tile_capacity_bytes = 128 * 1024;   // 128 KB per tile
config.l2_bank_count = 8;
config.l2_bank_capacity_bytes = 64 * 1024;    // 64 KB per bank
config.l1_buffer_count = 16;
config.l1_buffer_capacity_bytes = 16 * 1024;  // 16 KB per buffer

BehavioralMemoryModel memory(config);
```

## Statistics

The model collects usage statistics:

```cpp
const auto& stats = memory.stats();
std::cout << "Host allocated: " << stats.total_host_allocated << " bytes\n";
std::cout << "L3 allocated: " << stats.total_l3_allocated << " bytes\n";
std::cout << "Writes: " << stats.write_count << " (" << stats.bytes_written << " bytes)\n";
std::cout << "Reads: " << stats.read_count << " (" << stats.bytes_read << " bytes)\n";
std::cout << "Copies: " << stats.copy_count << " (" << stats.bytes_copied << " bytes)\n";
```

## Integration with Behavioral Execution

The memory model is used by the behavioral execution flow:

```
┌─────────────────────────────────────────────────────────────────┐
│                    BehavioralGraphExecutor                       │
│                                                                  │
│  1. Allocate tensors in HOST                                    │
│  2. Write weights and inputs                                    │
│                                                                  │
│  For each operator tile:                                        │
│    3. Allocate L3/L2/L1 regions                                 │
│    4. Copy: HOST → L3 (DMA)                                     │
│    5. Copy: L3 → L2 (BlockMover)                                │
│    6. Get pointers, compute matmul                              │
│    7. Copy: L2 → L3 → HOST (results)                            │
│                                                                  │
│  8. Read outputs from HOST                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Single Unified Model

Rather than separate models for each memory level, we use one unified model that
manages all regions. This simplifies cross-region operations and tensor tracking.

### 2. Address Encoding

Embedding region type and ID in the address allows:
- Single address space across all memories
- Easy routing of operations to correct region
- Debugging (address reveals location)

### 3. Bump Allocation

Simple bump allocation is used because:
- Fast allocation (O(1))
- Predictable layout
- Suitable for graph execution where allocation patterns are known
- Free is optional (reset between graphs)

### 4. Explicit Data Movement

Data doesn't automatically propagate between levels. Explicit `copy()` calls model
the DMA/BlockMover/Streamer operations, making data movement visible and traceable.

### 5. Direct Pointer Access

For compute efficiency, `get_ptr<T>()` provides direct access to memory. This avoids
copying data for computation while maintaining the hierarchical memory model.

## Files

| File | Description |
|------|-------------|
| `include/sw/kpu/behavioral/memory_model.hpp` | Header with API |
| `src/components/behavioral/memory_model.cpp` | Implementation |
| `tests/runtime/test_behavioral_memory.cpp` | Unit tests |

## See Also

- `docs/plans/BEHAVIORAL_EXECUTION_MODEL.md` - Overall behavioral execution plan
- `docs/SIMULATION_FIDELITY_FRAMEWORK.md` - Multi-fidelity simulation framework
- `include/sw/kpu/fidelity/simulation_fidelity.hpp` - Fidelity level definitions
