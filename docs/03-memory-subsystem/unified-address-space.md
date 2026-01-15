# Unified Address Space Architecture

## Overview

The KPU Simulator implements a **unified address space** where all memory regions (host memory, external memory, on-chip caches, and scratchpads) are mapped into a single contiguous address space. This simplifies DMA operations and provides a clean, address-based API.

## Architecture

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Address Space                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────┐                                     │
│  │   HOST MEMORY      │  System memory (DDR4/DDR5)          │
│  │   (NUMA regions)   │  Typical: 4GB - 128GB               │
│  └────────────────────┘                                     │
│           ↕                                                  │
│  ┌────────────────────┐                                     │
│  │ EXTERNAL MEMORY    │  Local to KPU (HBM/GDDR)           │
│  │   (Memory Banks)   │  High bandwidth: 100-900 GB/s       │
│  └────────────────────┘                                     │
│           ↕                                                  │
│  ┌────────────────────┐                                     │
│  │   L3 CACHE TILES   │  Large on-chip cache               │
│  │                    │  Typical: 512KB - 2MB per tile      │
│  └────────────────────┘                                     │
│           ↕                                                  │
│  ┌────────────────────┐                                     │
│  │   L2 CACHE BANKS   │  Distributed cache                  │
│  │                    │  Typical: 128KB - 512KB per bank    │
│  └────────────────────┘                                     │
│           ↕                                                  │
│  ┌────────────────────┐                                     │
│  │   SCRATCHPADS      │  Software-managed L1                │
│  │   (L1 / SRAM)      │  Closest to compute: 32KB - 256KB   │
│  └────────────────────┘                                     │
│           ↕                                                  │
│  ┌────────────────────┐                                     │
│  │  COMPUTE FABRIC    │  Matrix engines, systolic arrays    │
│  └────────────────────┘                                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Address Space Layout

The address space is organized sequentially based on configuration:

```
Address Range          | Memory Type        | Notes
-----------------------|--------------------|--------------------------
0x00000000 - ...       | Host Memory        | Region 0, 1, 2, ...
...        - ...       | External Memory    | Bank 0, 1, 2, ...
...        - ...       | L3 Tiles           | Tile 0, 1, 2, ...
...        - ...       | L2 Banks           | Bank 0, 1, 2, ...
...        - END       | Scratchpads        | Scratchpad 0, 1, 2, ...
```

## Default Configuration Example

### Configuration
```cpp
Config config;
// Defaults (from kpu_simulator.hpp:82-95):
config.host_memory_region_count = 1;
config.host_memory_region_capacity_mb = 4096;  // 4GB
config.memory_bank_count = 2;
config.memory_bank_capacity_mb = 1024;         // 1GB each
config.l3_tile_count = 4;
config.l3_tile_capacity_kb = 128;              // 128KB each
config.l2_bank_count = 8;
config.l2_bank_capacity_kb = 64;               // 64KB each
config.scratchpad_count = 2;
config.scratchpad_capacity_kb = 64;            // 64KB each
```

### Computed Address Map

```
┌──────────────────────────────────────────────────────────────────┐
│ HOST_MEMORY[0]                                                   │
│ Base: 0x00000000                                                 │
│ Size: 4GB (4,294,967,296 bytes)                                  │
│ End:  0xFFFFFFFF (4,294,967,295)                                 │
├──────────────────────────────────────────────────────────────────┤
│ EXTERNAL[0]                                                      │
│ Base: 0x100000000 (4,294,967,296)                               │
│ Size: 1GB (1,073,741,824 bytes)                                  │
│ End:  0x13FFFFFFF (5,368,709,119)                                │
├──────────────────────────────────────────────────────────────────┤
│ EXTERNAL[1]                                                      │
│ Base: 0x140000000 (5,368,709,120)                               │
│ Size: 1GB                                                        │
│ End:  0x17FFFFFFF (6,442,450,943)                                │
├──────────────────────────────────────────────────────────────────┤
│ L3_TILE[0]                                                       │
│ Base: 0x180000000 (6,442,450,944)                               │
│ Size: 128KB (131,072 bytes)                                      │
│ End:  0x18001FFFF (6,442,582,015)                                │
├──────────────────────────────────────────────────────────────────┤
│ L3_TILE[1..3]                                                    │
│ Base: 0x180020000, 0x180040000, 0x180060000                     │
│ ...                                                              │
├──────────────────────────────────────────────────────────────────┤
│ L2_BANK[0]                                                       │
│ Base: 0x180080000 (6,442,844,160)                               │
│ Size: 64KB (65,536 bytes)                                        │
│ End:  0x18008FFFF (6,442,909,695)                                │
├──────────────────────────────────────────────────────────────────┤
│ L2_BANK[1..7]                                                    │
│ Base: 0x180090000, 0x1800A0000, ...                             │
│ ...                                                              │
├──────────────────────────────────────────────────────────────────┤
│ SCRATCHPAD[0]                                                    │
│ Base: 0x180100000 (6,443,368,448)                               │
│ Size: 64KB (65,536 bytes)                                        │
│ End:  0x18010FFFF (6,443,433,983)                                │
├──────────────────────────────────────────────────────────────────┤
│ SCRATCHPAD[1]                                                    │
│ Base: 0x180110000 (6,443,433,984)                               │
│ Size: 64KB                                                       │
│ End:  0x18011FFFF (6,443,499,519)                                │
└──────────────────────────────────────────────────────────────────┘

Total Address Space: ~6.44 GB (including gaps)
```

## Address Computation Helpers

The simulator provides helper methods to compute base addresses:

### C++ API
```cpp
KPUSimulator sim(config);

// Get base addresses
Address host_base = sim.get_host_memory_region_base(0);
Address ext_base = sim.get_external_bank_base(0);
Address l3_base = sim.get_l3_tile_base(0);
Address l2_base = sim.get_l2_bank_base(0);
Address scratch_base = sim.get_scratchpad_base(0);

// Compute global addresses for DMA
Address src_addr = ext_base + 0x1000;      // External bank 0, offset 0x1000
Address dst_addr = scratch_base + 0x0;     // Scratchpad 0, offset 0x0

// Start DMA transfer
sim.start_dma_transfer(0, src_addr, dst_addr, transfer_size);
```

### Python API
```python
import stillwater_kpu as kpu

sim = kpu.KPUSimulator(config)

# Get base addresses
host_base = sim.get_host_memory_region_base(0)
ext_base = sim.get_external_bank_base(0)
l3_base = sim.get_l3_tile_base(0)
l2_base = sim.get_l2_bank_base(0)
scratch_base = sim.get_scratchpad_base(0)

# Compute global addresses
src_addr = ext_base + 0x1000
dst_addr = scratch_base + 0x0

# Start DMA transfer
sim.start_dma_transfer(0, src_addr, dst_addr, transfer_size)
```

## DMA Transfer Examples

### Example 1: Host → External Memory

```cpp
// Prepare data in host memory
std::vector<float> data(1024);
sim.write_host_memory(0, 0, data.data(), data.size() * sizeof(float));

// Compute global addresses
Address host_addr = sim.get_host_memory_region_base(0) + 0;
Address ext_addr = sim.get_external_bank_base(0) + 0;
Size transfer_size = data.size() * sizeof(float);

// DMA transfer with callback
bool complete = false;
sim.dma_host_to_external(0, host_addr, ext_addr, transfer_size,
    [&complete]() { complete = true; });

// Wait for completion
sim.run_until_idle();
```

### Example 2: Multi-Stage Pipeline (Host → External → Scratchpad)

```cpp
// Stage 1: Host → External
Address host_addr = sim.get_host_memory_region_base(0) + 0;
Address ext_addr = sim.get_external_bank_base(0) + 0;

sim.dma_host_to_external(0, host_addr, ext_addr, size);
sim.run_until_idle();

// Stage 2: External → Scratchpad
Address scratch_addr = sim.get_scratchpad_base(0) + 0;

sim.dma_external_to_scratchpad(1, ext_addr, scratch_addr, size);
sim.run_until_idle();

// Stage 3: Compute on scratchpad data
sim.start_matmul(0, 0, m, n, k, a_addr, b_addr, c_addr);
sim.run_until_idle();
```

### Example 3: Cache Hierarchy (External → L3 → L2)

```cpp
// Load data into L3 cache
Address ext_addr = sim.get_external_bank_base(0) + 0;
Address l3_addr = sim.get_l3_tile_base(0) + 0;

sim.dma_external_to_l3(0, ext_addr, l3_addr, size);
sim.run_until_idle();

// Later, move to L2 for compute fabric access
Address l2_addr = sim.get_l2_bank_base(0) + 0;

// Use BlockMover for L3 → L2 with transformations
sim.start_block_transfer(
    0,                          // block_mover_id
    0,                          // src_l3_tile_id
    0,                          // src_offset
    0,                          // dst_l2_bank_id
    0,                          // dst_offset
    matrix_height,              // block_height
    matrix_width,               // block_width
    sizeof(float),              // element_size
    BlockMover::TransformType::TRANSPOSE  // optional transformation
);
sim.run_until_idle();
```

### Example 4: Using Primary Address-Based API

The primary `start_dma_transfer()` API automatically routes based on addresses:

```cpp
// The address decoder figures out the routing automatically
Address any_src = sim.get_host_memory_region_base(0) + offset;
Address any_dst = sim.get_l3_tile_base(2) + offset;

// This automatically becomes a Host → L3 transfer
sim.start_dma_transfer(0, any_src, any_dst, size);
```

## Address Validation

The AddressDecoder validates all addresses during DMA operations:

```cpp
// Valid: address within a memory region
Address valid_addr = sim.get_external_bank_base(0) + 1024;

// Invalid: address not mapped to any region
Address invalid_addr = 0xFFFFFFFFFFFFFFFF;

try {
    sim.start_dma_transfer(0, invalid_addr, valid_addr, size);
} catch (const std::out_of_range& e) {
    // "Address 0xFFFFFFFFFFFFFFFF is not mapped to any memory region"
}
```

## Programmable Memory Map

For advanced use cases, you can override the default sequential layout:

```cpp
Config config;

// Configure component counts and sizes
config.host_memory_region_count = 1;
config.host_memory_region_capacity_mb = 4096;
config.memory_bank_count = 2;
config.memory_bank_capacity_mb = 1024;

// Override base addresses for custom layout
config.host_memory_base = 0x00000000;           // Start at 0
config.external_memory_base = 0x200000000;      // Start at 8GB
config.scratchpad_base = 0x300000000;           // Start at 12GB

// L3 and L2 will use default sequential placement between external and scratchpad
config.l3_tile_base = 0;  // 0 = automatic
config.l2_bank_base = 0;  // 0 = automatic
```

## Performance Considerations

### Bandwidth Hierarchy

The memory hierarchy has different bandwidth characteristics:

```
Memory Type         | Typical Bandwidth  | Latency
--------------------|--------------------|----------
Host Memory         | 50 GB/s            | ~100 ns
External (HBM)      | 900 GB/s           | ~50 ns
L3 Cache            | 2 TB/s             | ~10 ns
L2 Cache            | 5 TB/s             | ~5 ns
Scratchpad (L1)     | 10 TB/s            | ~1 ns
```

### DMA Engine Assignment

For optimal performance, assign DMA engines to specific transfer types:

```cpp
// DMA 0-1: Host ↔ External
sim.dma_host_to_external(0, host_addr, ext_addr, size);

// DMA 2-3: External ↔ Scratchpad
sim.dma_external_to_scratchpad(2, ext_addr, scratch_addr, size);

// BlockMovers: L3 ↔ L2 (with transformations)
sim.start_block_transfer(0, l3_tile, 0, l2_bank, 0, h, w, elem_size);

// Streamers: L2 ↔ L1/Scratchpad (for systolic arrays)
sim.start_row_stream(0, l2_bank, scratchpad, l2_addr, l1_addr, ...);
```

## Best Practices

1. **Use address helpers**: Always use `get_*_base()` methods rather than hardcoding addresses
2. **Validate before transfer**: Check capacity before initiating large transfers
3. **Use callbacks**: Register callbacks for asynchronous DMA completion
4. **Pipeline transfers**: Overlap compute and data movement for optimal performance
5. **Respect hierarchy**: Move data through the hierarchy (Host → External → L3 → L2 → L1)
6. **Check alignment**: Ensure addresses are properly aligned for SIMD operations

## See Also

- [DMA Architecture](dma-architecture.md)
- [Memory Orchestrator](memory-orchestrator-vs-buffet.md)
- [Configuration System](json-configuration-system.md)
- [C++ API Reference](../include/sw/kpu/kpu_simulator.hpp)
- [Python Examples](../examples/python/address_based_dma_example.py)
