# Migration Guide: Type-Based → Address-Based DMA API

## Overview

This guide helps you migrate from the deprecated type-based DMA API to the new unified address-based API. The new API provides better flexibility, cleaner code, and support for the full memory hierarchy.

## Key Changes

### 1. Unified Address Space

**Old Model**: Memory regions identified by `(MemoryType, id, offset)` tuples

**New Model**: All memory mapped into a single unified address space with global addresses

### 2. API Simplification

**Before**:
```cpp
// Required 6 parameters: src type, src id, src offset, dst type, dst id, dst offset
dma.enqueue_transfer(
    DMAEngine::MemoryType::EXTERNAL, 0, 0x1000,
    DMAEngine::MemoryType::SCRATCHPAD, 0, 0x0,
    transfer_size,
    callback
);
```

**After**:
```cpp
// Only 3 parameters: src address, dst address, size
Address src = sim.get_external_bank_base(0) + 0x1000;
Address dst = sim.get_scratchpad_base(0) + 0x0;

sim.start_dma_transfer(0, src, dst, transfer_size, callback);
```

### 3. New Memory Types

The new API adds support for:
- **Host Memory**: System memory (DDR4/DDR5)
- **L3 Tiles**: Large on-chip cache
- **L2 Banks**: Distributed cache

## Step-by-Step Migration

### Step 1: Update Includes

No changes needed - existing includes work:
```cpp
#include <sw/kpu/kpu_simulator.hpp>
```

### Step 2: Replace Direct DMAEngine Access

**Old Pattern**:
```cpp
// Directly accessing DMA engine
DMAEngine& dma = get_dma_engine(0);
dma.enqueue_transfer(...);
```

**New Pattern**:
```cpp
// Use KPUSimulator's DMA methods
sim.start_dma_transfer(...);
// Or use convenience helpers:
sim.dma_external_to_scratchpad(...);
```

### Step 3: Convert Address Computation

**Old**:
```cpp
// Type + ID + Offset
size_t bank_id = 0;
Address offset = 0x1000;
// Passed separately to API
```

**New**:
```cpp
// Compute global address once
Address global_addr = sim.get_external_bank_base(0) + 0x1000;
// Use in all subsequent operations
```

### Step 4: Update Transfer Calls

#### External → Scratchpad

**Before**:
```cpp
dma.enqueue_transfer(
    DMAEngine::MemoryType::EXTERNAL, 0, src_offset,
    DMAEngine::MemoryType::SCRATCHPAD, 0, dst_offset,
    size,
    callback
);
```

**After (Option 1 - Convenience Helper)**:
```cpp
Address src = sim.get_external_bank_base(0) + src_offset;
Address dst = sim.get_scratchpad_base(0) + dst_offset;
sim.dma_external_to_scratchpad(0, src, dst, size, callback);
```

**After (Option 2 - Primary API)**:
```cpp
Address src = sim.get_external_bank_base(0) + src_offset;
Address dst = sim.get_scratchpad_base(0) + dst_offset;
sim.start_dma_transfer(0, src, dst, size, callback);
```

#### Scratchpad → External

**Before**:
```cpp
dma.enqueue_transfer(
    DMAEngine::MemoryType::SCRATCHPAD, 0, src_offset,
    DMAEngine::MemoryType::EXTERNAL, 0, dst_offset,
    size
);
```

**After**:
```cpp
Address src = sim.get_scratchpad_base(0) + src_offset;
Address dst = sim.get_external_bank_base(0) + dst_offset;
sim.dma_scratchpad_to_external(0, src, dst, size);
```

## Complete Migration Examples

### Example 1: Simple Transfer

**Before**:
```cpp
void old_transfer() {
    DMAEngine& dma = sim.get_dma_engine(0);

    std::vector<float> data(256);
    sim.write_memory_bank(0, 0, data.data(), data.size() * sizeof(float));

    bool complete = false;
    dma.enqueue_transfer(
        DMAEngine::MemoryType::EXTERNAL, 0, 0,
        DMAEngine::MemoryType::SCRATCHPAD, 0, 0,
        data.size() * sizeof(float),
        [&complete]() { complete = true; }
    );

    while (!complete) {
        sim.step();
    }
}
```

**After**:
```cpp
void new_transfer() {
    std::vector<float> data(256);
    sim.write_memory_bank(0, 0, data.data(), data.size() * sizeof(float));

    Address src = sim.get_external_bank_base(0) + 0;
    Address dst = sim.get_scratchpad_base(0) + 0;
    Size transfer_size = data.size() * sizeof(float);

    bool complete = false;
    sim.dma_external_to_scratchpad(0, src, dst, transfer_size,
        [&complete]() { complete = true; });

    sim.run_until_idle();  // Simplified completion waiting
}
```

### Example 2: Multiple Transfers

**Before**:
```cpp
void old_multi_transfer() {
    DMAEngine& dma = sim.get_dma_engine(0);

    // Transfer 1
    dma.enqueue_transfer(
        DMAEngine::MemoryType::EXTERNAL, 0, 0x0,
        DMAEngine::MemoryType::SCRATCHPAD, 0, 0x0,
        1024
    );

    // Transfer 2
    dma.enqueue_transfer(
        DMAEngine::MemoryType::EXTERNAL, 0, 0x1000,
        DMAEngine::MemoryType::SCRATCHPAD, 0, 0x400,
        1024
    );

    while (dma.is_busy()) {
        sim.step();
    }
}
```

**After**:
```cpp
void new_multi_transfer() {
    Address ext_base = sim.get_external_bank_base(0);
    Address scratch_base = sim.get_scratchpad_base(0);

    // Transfer 1
    sim.dma_external_to_scratchpad(0,
        ext_base + 0x0,
        scratch_base + 0x0,
        1024
    );

    // Transfer 2
    sim.dma_external_to_scratchpad(0,
        ext_base + 0x1000,
        scratch_base + 0x400,
        1024
    );

    sim.run_until_idle();
}
```

## New Capabilities

The new API enables transfers that weren't possible before:

### Host Memory Transfers

```cpp
// New: Host → External
Address host_addr = sim.get_host_memory_region_base(0) + offset;
Address ext_addr = sim.get_external_bank_base(0) + offset;
sim.dma_host_to_external(0, host_addr, ext_addr, size);

// New: Host → Scratchpad (direct)
Address host_addr = sim.get_host_memory_region_base(0) + offset;
Address scratch_addr = sim.get_scratchpad_base(0) + offset;
sim.dma_host_to_scratchpad(0, host_addr, scratch_addr, size);
```

### Cache Hierarchy Transfers

```cpp
// New: External → L3
Address ext_addr = sim.get_external_bank_base(0) + offset;
Address l3_addr = sim.get_l3_tile_base(0) + offset;
sim.dma_external_to_l3(0, ext_addr, l3_addr, size);

// New: L3 → Host
Address l3_addr = sim.get_l3_tile_base(0) + offset;
Address host_addr = sim.get_host_memory_region_base(0) + offset;
sim.dma_l3_to_host(0, l3_addr, host_addr, size);
```

### Scratchpad-to-Scratchpad

```cpp
// New: Data reshuffling between scratchpads
Address scratch0 = sim.get_scratchpad_base(0) + offset;
Address scratch1 = sim.get_scratchpad_base(1) + offset;
sim.dma_scratchpad_to_scratchpad(0, scratch0, scratch1, size);
```

## API Reference Quick Lookup

### DMA Convenience Helpers

| Old API | New API |
|---------|---------|
| `EXTERNAL → SCRATCHPAD` | `dma_external_to_scratchpad(dma_id, src, dst, size, cb)` |
| `SCRATCHPAD → EXTERNAL` | `dma_scratchpad_to_external(dma_id, src, dst, size, cb)` |
| *(Not available)* | `dma_host_to_external(dma_id, src, dst, size, cb)` |
| *(Not available)* | `dma_external_to_host(dma_id, src, dst, size, cb)` |
| *(Not available)* | `dma_host_to_l3(dma_id, src, dst, size, cb)` |
| *(Not available)* | `dma_l3_to_host(dma_id, src, dst, size, cb)` |
| *(Not available)* | `dma_external_to_l3(dma_id, src, dst, size, cb)` |
| *(Not available)* | `dma_l3_to_external(dma_id, src, dst, size, cb)` |
| *(Not available)* | `dma_host_to_scratchpad(dma_id, src, dst, size, cb)` |
| *(Not available)* | `dma_scratchpad_to_host(dma_id, src, dst, size, cb)` |
| *(Not available)* | `dma_scratchpad_to_scratchpad(dma_id, src, dst, size, cb)` |

### Address Computation

```cpp
// Get base addresses for computing global addresses
Address sim.get_host_memory_region_base(size_t region_id);
Address sim.get_external_bank_base(size_t bank_id);
Address sim.get_l3_tile_base(size_t tile_id);
Address sim.get_l2_bank_base(size_t bank_id);
Address sim.get_scratchpad_base(size_t scratchpad_id);
```

### Status Queries

```cpp
// Check if DMA engine is busy
bool sim.is_dma_busy(size_t dma_id);

// Check if memory regions are ready
bool sim.is_host_memory_region_ready(size_t region_id);
bool sim.is_memory_bank_ready(size_t bank_id);
bool sim.is_l3_tile_ready(size_t tile_id);
bool sim.is_l2_bank_ready(size_t bank_id);
bool sim.is_scratchpad_ready(size_t scratchpad_id);
```

## Python Migration

### Before (if old API was exposed)

```python
# Old API (not actually available in Python)
dma.enqueue_transfer(
    kpu.MemoryType.EXTERNAL, 0, 0x1000,
    kpu.MemoryType.SCRATCHPAD, 0, 0x0,
    1024
)
```

### After

```python
# New address-based API
src = sim.get_external_bank_base(0) + 0x1000
dst = sim.get_scratchpad_base(0) + 0x0

sim.dma_external_to_scratchpad(0, src, dst, 1024)
# Or:
sim.start_dma_transfer(0, src, dst, 1024)
```

## Common Pitfalls

### 1. Forgetting to Compute Global Addresses

❌ **Wrong**:
```cpp
// This won't work - offsets aren't global addresses
sim.start_dma_transfer(0, 0x1000, 0x0, size);
```

✅ **Correct**:
```cpp
// Must use base address helpers
Address src = sim.get_external_bank_base(0) + 0x1000;
Address dst = sim.get_scratchpad_base(0) + 0x0;
sim.start_dma_transfer(0, src, dst, size);
```

### 2. Hardcoding Addresses

❌ **Wrong**:
```cpp
// Addresses depend on configuration - don't hardcode
Address src = 0x100000000 + 0x1000;  // Bad!
```

✅ **Correct**:
```cpp
// Always use helpers
Address src = sim.get_external_bank_base(0) + 0x1000;
```

### 3. Using Wrong Memory ID

❌ **Wrong**:
```cpp
// Bank 0 and bank 1 have different base addresses
Address base = sim.get_external_bank_base(0);
// But this offset might be in bank 1!
Address wrong = base + (2 * 1024 * 1024 * 1024);  // 2GB offset - goes past bank 0
```

✅ **Correct**:
```cpp
// Compute addresses for each bank separately
Address bank0_addr = sim.get_external_bank_base(0) + offset;
Address bank1_addr = sim.get_external_bank_base(1) + offset;
```

## Backward Compatibility

The old type-based API is **deprecated but still available** in DMAEngine for compatibility. However:

- It will be removed in a future version
- It doesn't support new memory types (Host, L3, L2)
- New code should use the address-based API
- Tests using the old API will show deprecation warnings

You can suppress warnings temporarily:
```cpp
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4996)
#else
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

// Old API calls here

#ifdef _MSC_VER
    #pragma warning(pop)
#else
    #pragma GCC diagnostic pop
#endif
```

## Testing Your Migration

1. **Compile with warnings**: Ensure no deprecation warnings
2. **Run existing tests**: All functionality should still work
3. **Verify addresses**: Print memory map to check address layout
4. **Test new transfers**: Try Host→External and cache transfers
5. **Performance testing**: Ensure no performance regression

## Getting Help

- Review [Quick Start Guide](address-based-dma-quickstart.md)
- Check [API Documentation](unified-address-space.md)
- Study [Example Code](../examples/basic/memory_management.cpp)
- Look at [Test Cases](../tests/dma/test_dma_basic.cpp)

## Timeline

- **v0.1.0**: Address-based API introduced, type-based API deprecated
- **v0.2.0** (planned): Type-based API removed
- **Recommendation**: Migrate code now to avoid breaking changes
