# Address-Based DMA API - Quick Start Guide

## Overview

This guide provides practical examples for using the KPU Simulator's address-based DMA API. The unified address space simplifies data movement across the memory hierarchy.

## Basic Concepts

### 1. Global Address Space

All memory is accessible through a single unified address space:
- Each memory region has a **base address**
- Offsets are added to base addresses to form **global addresses**
- DMA operations use global addresses for source and destination

### 2. Address Computation Pattern

```cpp
// Pattern: Base Address + Offset = Global Address
Address global_addr = sim.get_<memory_type>_base(id) + offset;
```

## Quick Start Examples

### Example 1: Simple External → Scratchpad Transfer

```cpp
#include <sw/kpu/kpu_simulator.hpp>
#include <vector>

int main() {
    // Create simulator with default config
    sw::kpu::KPUSimulator sim;

    // Prepare test data
    std::vector<float> data(256);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    // Write data to external memory bank 0
    sim.write_memory_bank(0, 0, data.data(), data.size() * sizeof(float));

    // Compute global addresses
    Address src_addr = sim.get_external_bank_base(0) + 0;
    Address dst_addr = sim.get_scratchpad_base(0) + 0;
    Size transfer_size = data.size() * sizeof(float);

    // Start DMA transfer
    bool complete = false;
    sim.dma_external_to_scratchpad(
        0,              // DMA engine ID
        src_addr,       // Source (global address)
        dst_addr,       // Destination (global address)
        transfer_size,  // Size in bytes
        [&complete]() { complete = true; }  // Completion callback
    );

    // Wait for completion
    while (!complete) {
        sim.step();
    }

    // Verify data
    std::vector<float> result(256);
    sim.read_scratchpad(0, 0, result.data(), result.size() * sizeof(float));

    std::cout << "Transfer completed successfully!\n";
    return 0;
}
```

### Example 2: Multi-Stage Data Pipeline

```cpp
// Multi-stage pipeline: Host → External → Scratchpad
void pipeline_example() {
    sw::kpu::KPUSimulator sim;

    const size_t data_size = 512;
    const size_t transfer_bytes = data_size * sizeof(float);

    // Stage 1: Write data to host memory
    std::vector<float> input(data_size, 1.0f);
    sim.write_host_memory(0, 0, input.data(), transfer_bytes);

    // Stage 2: Host → External
    Address host_addr = sim.get_host_memory_region_base(0);
    Address ext_addr = sim.get_external_bank_base(0);

    sim.dma_host_to_external(0, host_addr, ext_addr, transfer_bytes);
    sim.run_until_idle();  // Simplified completion waiting

    // Stage 3: External → Scratchpad
    Address scratch_addr = sim.get_scratchpad_base(0);

    sim.dma_external_to_scratchpad(1, ext_addr, scratch_addr, transfer_bytes);
    sim.run_until_idle();

    std::cout << "Pipeline complete: Host → External → Scratchpad\n";
}
```

### Example 3: Using the Primary Address-Based API

The primary API automatically determines the transfer type:

```cpp
void automatic_routing_example() {
    sw::kpu::KPUSimulator sim;

    // The API automatically figures out this is a Host → Scratchpad transfer
    Address src = sim.get_host_memory_region_base(0) + 1024;
    Address dst = sim.get_scratchpad_base(0) + 0;
    Size size = 256 * sizeof(float);

    // Single API handles all transfer types
    sim.start_dma_transfer(0, src, dst, size);
    sim.run_until_idle();
}
```

## Python Quick Start

### Example 1: Basic Transfer

```python
import stillwater_kpu as kpu

# Create simulator
sim = kpu.KPUSimulator()

# Prepare data
data = [float(i) for i in range(256)]

# Write to external memory
sim.write_memory_bank(0, 0, data)

# Compute global addresses
src_addr = sim.get_external_bank_base(0) + 0
dst_addr = sim.get_scratchpad_base(0) + 0
transfer_size = len(data) * 4  # 4 bytes per float

# DMA transfer with callback
complete = False
def on_complete():
    global complete
    complete = True

sim.dma_external_to_scratchpad(0, src_addr, dst_addr, transfer_size, on_complete)

# Wait for completion
sim.run_until_idle()

# Verify
result = sim.read_scratchpad(0, 0, len(data))
print(f"Transfer successful: {result == data}")
```

### Example 2: Memory Hierarchy Configuration

```python
import stillwater_kpu as kpu

# Configure custom memory hierarchy
config = kpu.SimulatorConfig()

# Host memory
config.host_memory_region_count = 2
config.host_memory_region_capacity_mb = 2048  # 2GB per region

# External memory
config.memory_bank_count = 4
config.memory_bank_capacity_mb = 512  # 512MB per bank

# On-chip hierarchy
config.l3_tile_count = 4
config.l3_tile_capacity_kb = 256
config.l2_bank_count = 8
config.l2_bank_capacity_kb = 128
config.scratchpad_count = 4
config.scratchpad_capacity_kb = 128

# DMA engines
config.dma_engine_count = 8

# Create simulator
sim = kpu.KPUSimulator(config)

# Print address space layout
print(f"Host[0] base:       0x{sim.get_host_memory_region_base(0):08x}")
print(f"External[0] base:   0x{sim.get_external_bank_base(0):08x}")
print(f"L3[0] base:         0x{sim.get_l3_tile_base(0):08x}")
print(f"Scratchpad[0] base: 0x{sim.get_scratchpad_base(0):08x}")
```

## Common Patterns

### Pattern 1: Double Buffering

```cpp
void double_buffer_example() {
    sw::kpu::KPUSimulator sim;

    // Use two scratchpads for double buffering
    Address scratch0 = sim.get_scratchpad_base(0);
    Address scratch1 = sim.get_scratchpad_base(1);
    Address ext_base = sim.get_external_bank_base(0);

    const Size buffer_size = 1024 * sizeof(float);

    // Load first buffer
    sim.dma_external_to_scratchpad(0, ext_base, scratch0, buffer_size);

    for (int i = 0; i < 10; ++i) {
        Address current_scratch = (i % 2 == 0) ? scratch0 : scratch1;
        Address next_scratch = (i % 2 == 0) ? scratch1 : scratch0;

        // Compute on current buffer
        sim.start_matmul(0, i % 2, m, n, k, a_addr, b_addr, c_addr);

        // Simultaneously load next buffer
        if (i < 9) {
            Address next_ext = ext_base + (i + 1) * buffer_size;
            sim.dma_external_to_scratchpad(1, next_ext, next_scratch, buffer_size);
        }

        sim.run_until_idle();
    }
}
```

### Pattern 2: Scatter-Gather Operations

```cpp
void scatter_gather_example() {
    sw::kpu::KPUSimulator sim;

    Address ext_base = sim.get_external_bank_base(0);
    Address scratch_base = sim.get_scratchpad_base(0);

    // Gather: Multiple external locations → Single scratchpad
    std::vector<Address> sources = {
        ext_base + 0x0000,
        ext_base + 0x1000,
        ext_base + 0x2000,
        ext_base + 0x3000
    };

    Size chunk_size = 256 * sizeof(float);
    Address scratch_offset = 0;

    for (auto src : sources) {
        sim.dma_external_to_scratchpad(
            0,
            src,
            scratch_base + scratch_offset,
            chunk_size
        );
        scratch_offset += chunk_size;
    }

    sim.run_until_idle();
}
```

### Pattern 3: Cache Warming

```cpp
void cache_warming_example() {
    sw::kpu::KPUSimulator sim;

    // Pre-load frequently accessed data into cache hierarchy
    Address ext_addr = sim.get_external_bank_base(0);
    Address l3_addr = sim.get_l3_tile_base(0);
    Size cache_line = 4096;  // 4KB cache line

    // Warm L3 cache
    for (int i = 0; i < 16; ++i) {
        sim.dma_external_to_l3(
            0,
            ext_addr + i * cache_line,
            l3_addr + i * cache_line,
            cache_line
        );
    }

    sim.run_until_idle();

    // Now subsequent accesses will hit L3 cache
}
```

## Error Handling

### Address Validation

```cpp
void error_handling_example() {
    sw::kpu::KPUSimulator sim;

    try {
        // Invalid: address outside any memory region
        Address invalid_addr = 0xFFFFFFFFFFFFFFFF;
        Address valid_addr = sim.get_scratchpad_base(0);

        sim.start_dma_transfer(0, invalid_addr, valid_addr, 1024);

    } catch (const std::out_of_range& e) {
        std::cerr << "Address validation failed: " << e.what() << "\n";
    }

    try {
        // Invalid: transfer size exceeds destination capacity
        Address src = sim.get_external_bank_base(0);
        Address dst = sim.get_scratchpad_base(0);
        Size oversized = sim.get_scratchpad_capacity(0) + 1024;

        sim.start_dma_transfer(0, src, dst, oversized);

    } catch (const std::exception& e) {
        std::cerr << "Transfer size validation failed: " << e.what() << "\n";
    }
}
```

## Debugging Tips

### 1. Print Memory Map

```cpp
void print_memory_map(const sw::kpu::KPUSimulator& sim) {
    std::cout << "Memory Map:\n";
    std::cout << "  Host regions: " << sim.get_host_memory_region_count() << "\n";
    for (size_t i = 0; i < sim.get_host_memory_region_count(); ++i) {
        std::cout << "    Host[" << i << "]: 0x" << std::hex
                  << sim.get_host_memory_region_base(i) << std::dec << "\n";
    }

    std::cout << "  External banks: " << sim.get_memory_bank_count() << "\n";
    for (size_t i = 0; i < sim.get_memory_bank_count(); ++i) {
        std::cout << "    External[" << i << "]: 0x" << std::hex
                  << sim.get_external_bank_base(i) << std::dec << "\n";
    }

    // Similar for L3, L2, Scratchpads...
}
```

### 2. Enable Tracing

```cpp
void enable_tracing_example() {
    sw::kpu::KPUSimulator sim;

    // Enable tracing for specific DMA engines
    sim.enable_dma_tracing(0);
    sim.enable_dma_tracing(1);

    // Perform operations (traces will be logged)
    Address src = sim.get_external_bank_base(0);
    Address dst = sim.get_scratchpad_base(0);
    sim.start_dma_transfer(0, src, dst, 1024);
    sim.run_until_idle();

    // Disable when done
    sim.disable_dma_tracing(0);
}
```

### 3. Component Status Monitoring

```cpp
void monitor_status() {
    sw::kpu::KPUSimulator sim;

    // Check component readiness
    std::cout << "Component Status:\n";
    std::cout << "  Host[0] ready: " << sim.is_host_memory_region_ready(0) << "\n";
    std::cout << "  External[0] ready: " << sim.is_memory_bank_ready(0) << "\n";
    std::cout << "  Scratchpad[0] ready: " << sim.is_scratchpad_ready(0) << "\n";
    std::cout << "  DMA[0] busy: " << sim.is_dma_busy(0) << "\n";

    // Print detailed status
    sim.print_component_status();
}
```

## Performance Tips

1. **Use appropriate DMA engines**: Assign specific engines to specific transfer types
2. **Overlap transfers**: Start next transfer while compute is running
3. **Respect bandwidth hierarchy**: Don't bypass cache levels unnecessarily
4. **Align transfers**: Use cache-line aligned sizes for optimal performance
5. **Batch small transfers**: Combine multiple small transfers into larger ones

## Migration from Type-Based API

If migrating from the old type-based API:

### Old API (Deprecated)
```cpp
// Old type-based API
dma.enqueue_transfer(
    DMAEngine::MemoryType::EXTERNAL, 0, 0x1000,  // src
    DMAEngine::MemoryType::SCRATCHPAD, 0, 0x0,   // dst
    1024,
    callback
);
```

### New API (Current)
```cpp
// New address-based API
Address src = sim.get_external_bank_base(0) + 0x1000;
Address dst = sim.get_scratchpad_base(0) + 0x0;

sim.dma_external_to_scratchpad(0, src, dst, 1024, callback);
// Or use primary API:
sim.start_dma_transfer(0, src, dst, 1024, callback);
```

## Complete Example

See [address_based_dma_example.py](../examples/python/address_based_dma_example.py) for a comprehensive Python example demonstrating all features.

## Next Steps

- Read [Unified Address Space Architecture](unified-address-space.md) for detailed architecture
- Explore [C++ Examples](../examples/basic/memory_management.cpp)
- Review [API Reference](../include/sw/kpu/kpu_simulator.hpp)
- Study [Test Cases](../tests/dma/test_dma_basic.cpp) for more patterns
