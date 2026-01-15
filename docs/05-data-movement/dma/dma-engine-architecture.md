# DMA Engine Architecture

## Overview

The DMA Engine is a **cycle-accurate hardware component** responsible for asynchronous data movement across the KPU's memory hierarchy. It implements a multi-cycle timing model that accurately reflects bandwidth-limited data transfers between different memory types.

### Key Design Principles

1. **Cycle-Accurate Simulation**: Transfers take multiple cycles based on bandwidth and data size
2. **Multi-Level Memory Hierarchy**: Supports transfers between 5 memory types (Host, External, L3, L2, Scratchpad)
3. **Queue-Based Operation**: FIFO transfer queue with completion callbacks
4. **Tracing Integration**: Full transaction tracking with timing information
5. **Early Validation**: Capacity and bounds checking before transfer execution

---

## Architecture Components

### Memory Hierarchy Support

The DMA engine supports transfers between the following memory types:

```
┌─────────────────┐
│   HOST_MEMORY   │  Host DDR (CPU-side memory)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│    EXTERNAL     │  KPU memory banks (GDDR6/HBM)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│    L3_TILE      │  L3 cache tiles (128KB typical)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│    L2_BANK      │  L2 cache banks (64KB typical)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   SCRATCHPAD    │  L1 scratchpad (software-managed, 64-256KB)
└─────────────────┘
```

### Transfer Data Structure

Each transfer contains:

```cpp
struct Transfer {
    // Source information
    MemoryType src_type;
    size_t src_id;      // Index into memory array
    Address src_addr;   // Starting address

    // Destination information
    MemoryType dst_type;
    size_t dst_id;
    Address dst_addr;

    // Transfer parameters
    Size size;          // Transfer size in bytes
    std::function<void()> completion_callback;

    // Timing information
    trace::CycleCount start_cycle;
    trace::CycleCount end_cycle;
    uint64_t transaction_id;  // For trace correlation
};
```

### Engine State

```cpp
class DMAEngine {
private:
    // Transfer management
    std::vector<Transfer> transfer_queue;  // FIFO queue
    std::vector<uint8_t> transfer_buffer;  // Current transfer buffer

    // Timing state
    trace::CycleCount cycles_remaining;    // Multi-cycle counter
    trace::CycleCount current_cycle_;      // Global cycle counter

    // Performance characteristics
    double clock_freq_ghz_;    // Engine clock frequency
    double bandwidth_gb_s_;    // Theoretical bandwidth

    // Status
    bool is_active;            // Currently processing
    size_t engine_id;          // Engine identifier
};
```

---

## Cycle-Accurate Operation Model

### Bandwidth-Based Timing

Transfer latency is calculated based on theoretical bandwidth:

```
cycles_required = ceil(transfer_size_bytes / bytes_per_cycle)

where:
    bytes_per_cycle = bandwidth_gb_s / clock_freq_ghz

Example:
    Transfer: 4096 bytes
    Bandwidth: 100 GB/s
    Clock: 1 GHz

    bytes_per_cycle = 100 / 1 = 100 bytes/cycle
    cycles_required = ceil(4096 / 100) = 41 cycles
```

### Multi-Cycle Processing Pattern

The DMA engine uses a **two-phase** processing model:

#### **Phase 1: Transfer Initiation** (First `process_transfers()` call)
1. Validate destination capacity (early error detection)
2. Calculate required cycles based on bandwidth
3. Allocate transfer buffer
4. Perform source read operation
5. Log trace entry for transfer issue
6. Set `cycles_remaining` counter

#### **Phase 2: Cycle-by-Cycle Processing** (Subsequent `process_transfers()` calls)
1. Decrement `cycles_remaining` counter
2. When counter reaches 0:
   - Perform destination write operation
   - Log trace entry for transfer completion
   - Invoke completion callback (if provided)
   - Remove transfer from queue
   - Proceed to next queued transfer

---

## Transfer Lifecycle

### State Diagram

```
┌──────────┐
│  QUEUED  │ ← enqueue_transfer()
└────┬─────┘
     │
     ↓ process_transfers() called
┌──────────┐
│ STARTING │ Validate, read source, calculate cycles
└────┬─────┘
     │
     ↓ process_transfers() × N cycles
┌──────────┐
│PROCESSING│ Decrement cycles_remaining
└────┬─────┘
     │
     ↓ cycles_remaining == 0
┌──────────┐
│COMPLETING│ Write destination, callback, log
└────┬─────┘
     │
     ↓
┌──────────┐
│   DONE   │
└──────────┘
```

### Example Execution Trace

```cpp
// Setup
DMAEngine dma(0, 1.0, 100.0);  // 1 GHz, 100 GB/s
dma.set_current_cycle(1000);

// Enqueue 4KB transfer
dma.enqueue_transfer(
    DMAEngine::MemoryType::EXTERNAL, 0, 0x1000,
    DMAEngine::MemoryType::SCRATCHPAD, 0, 0x0,
    4096
);

// Processing loop
while (dma.is_busy()) {
    dma.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    dma.set_current_cycle(dma.get_current_cycle() + 1);
}

// Timeline:
// Cycle 1000: Transfer starts, source read, cycles_remaining = 41
// Cycles 1001-1039: Processing (decrementing counter)
// Cycle 1040: Transfer completes, destination write, callback
```

---

## API Usage

> **Note**: The DMA Engine now supports two APIs:
> - **Address-Based API** (Recommended) - Industry-standard, hardware-agnostic
> - **Type-Based API** (Deprecated) - Legacy API maintained for backward compatibility
>
> See [DMA Architecture Comparison](dma-architecture-comparison.md) for industry analysis.

### Address-Based API (Recommended)

The address-based API follows industry standards used by Intel I/OAT, ARM PL330, AMD SDMA, Xilinx AXI DMA, and Google TPU. It provides hardware topology independence and enables virtual memory support.

#### Setup with AddressDecoder

```cpp
using sw::memory::AddressDecoder;

// Create DMA engine
DMAEngine dma(engine_id, clock_freq_ghz, bandwidth_gb_s);

// Configure memory map
AddressDecoder decoder;
decoder.add_region(0x0000'0000, 512*1024*1024, MemoryType::EXTERNAL, 0, "External Bank 0");
decoder.add_region(0xFFFF'0000, 256*1024, MemoryType::SCRATCHPAD, 0, "Scratchpad 0");

// Attach decoder to DMA engine
dma.set_address_decoder(&decoder);
```

#### Basic Transfer

```cpp
// Pure address-based transfer (recommended)
bool complete = false;
dma.enqueue_transfer(
    0x0000'1000,                    // Source address
    0xFFFF'0000,                    // Destination address
    4096,                           // Size in bytes
    [&]() { complete = true; }      // Completion callback (optional)
);

// Process until complete
while (dma.is_busy()) {
    dma.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    dma.set_current_cycle(dma.get_current_cycle() + 1);
}
```

#### Benefits

- **Hardware Independence**: Applications don't need to know physical topology
- **Virtual Memory Support**: Memory remapping doesn't break DMA programs
- **Industry Standard**: Matches real-world DMA controller behavior
- **Simpler Code**: Fewer parameters, cleaner interface

### Type-Based API (Deprecated)

> **⚠️ Deprecated**: This API is maintained for backward compatibility but will be removed in a future release. Please migrate to the address-based API.

#### Basic Transfer (Legacy)

```cpp
// Create DMA engine
DMAEngine dma(engine_id, clock_freq_ghz, bandwidth_gb_s);

// Enqueue transfer (deprecated)
bool complete = false;
dma.enqueue_transfer(
    DMAEngine::MemoryType::EXTERNAL,    // Source type
    0,                                   // Source ID
    0x1000,                             // Source address (relative)
    DMAEngine::MemoryType::SCRATCHPAD,  // Dest type
    0,                                   // Dest ID
    0x0,                                // Dest address (relative)
    4096,                               // Size in bytes
    [&]() { complete = true; }          // Completion callback
);

// Process until complete
while (dma.is_busy()) {
    dma.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    dma.set_current_cycle(dma.get_current_cycle() + 1);
}
```

#### Why It's Deprecated

- **Non-Standard**: Real hardware DMA controllers don't require memory type in commands
- **Topology Coupling**: Applications must know physical memory organization
- **VM Incompatibility**: Cannot support dynamic memory remapping
- **Complex**: Requires explicit memory type and ID for each transfer

See [Migration Guide](#migration-from-type-based-to-address-based-api) below for conversion examples.

### Multiple Queued Transfers

```cpp
// Queue multiple transfers (processed sequentially, FIFO order)
dma.enqueue_transfer(0x0000'1000, 0xFFFF'0000, 4096);  // Transfer 1
dma.enqueue_transfer(0x0000'2000, 0xFFFF'1000, 8192);  // Transfer 2
dma.enqueue_transfer(0x0000'4000, 0xFFFF'3000, 16384); // Transfer 3

// Process all transfers
while (dma.is_busy()) {
    dma.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    dma.set_current_cycle(dma.get_current_cycle() + 1);
}
```

### Checking Status

```cpp
// Check if DMA is busy
if (dma.is_busy()) {
    std::cout << "DMA active with " << dma.get_queue_size() << " transfers queued\n";
}

// Get engine ID
size_t id = dma.get_engine_id();
```

---

## Migration from Type-Based to Address-Based API

### Step 1: Set Up AddressDecoder

Replace explicit memory type/ID management with a unified address map:

```cpp
// Before: No address decoder needed
DMAEngine dma(0, 1.0, 100.0);

// After: Create and configure address decoder
AddressDecoder decoder;
decoder.add_region(0x0000'0000, 512*1024*1024, MemoryType::EXTERNAL, 0, "Bank 0");
decoder.add_region(0x2000'0000, 512*1024*1024, MemoryType::EXTERNAL, 1, "Bank 1");
decoder.add_region(0xF000'0000, 128*1024, MemoryType::L3_TILE, 0, "L3 Tile 0");
decoder.add_region(0xFFFF'0000, 256*1024, MemoryType::SCRATCHPAD, 0, "Scratchpad 0");

DMAEngine dma(0, 1.0, 100.0);
dma.set_address_decoder(&decoder);
```

### Step 2: Convert Transfer Calls

#### Example 1: External to Scratchpad

```cpp
// Before (Type-Based):
dma.enqueue_transfer(
    DMAEngine::MemoryType::EXTERNAL,    // Source type
    0,                                   // Source bank ID
    0x1000,                             // Source offset
    DMAEngine::MemoryType::SCRATCHPAD,  // Dest type
    0,                                   // Dest scratchpad ID
    0x0,                                // Dest offset
    4096                                // Size
);

// After (Address-Based):
dma.enqueue_transfer(
    0x0000'1000,    // Source address (Bank 0, offset 0x1000)
    0xFFFF'0000,    // Dest address (Scratchpad 0, offset 0x0)
    4096            // Size
);
```

#### Example 2: Bank-to-Bank Transfer

```cpp
// Before (Type-Based):
dma.enqueue_transfer(
    DMAEngine::MemoryType::EXTERNAL, 0, 0x0,      // Bank 0
    DMAEngine::MemoryType::EXTERNAL, 1, 0x0,      // Bank 1
    1024*1024
);

// After (Address-Based):
dma.enqueue_transfer(
    0x0000'0000,    // Bank 0 base
    0x2000'0000,    // Bank 1 base
    1024*1024
);
```

#### Example 3: With Completion Callback

```cpp
// Before (Type-Based):
bool done = false;
dma.enqueue_transfer(
    DMAEngine::MemoryType::EXTERNAL, 0, src_offset,
    DMAEngine::MemoryType::SCRATCHPAD, 0, dst_offset,
    size,
    [&]() { done = true; }
);

// After (Address-Based):
bool done = false;
dma.enqueue_transfer(
    0x0000'0000 + src_offset,
    0xFFFF'0000 + dst_offset,
    size,
    [&]() { done = true; }
);
```

### Step 3: Handle Virtual Memory Scenarios

The address-based API enables dynamic memory remapping:

```cpp
// Define memory allocator (simulation of dynamic allocation)
auto allocate = [](size_t size) -> Address {
    static Address next_addr = 0x0000'0000;
    Address result = next_addr;
    next_addr += size;
    return result;
};

// Allocate tensors dynamically
Address tensor_a = allocate(4096);
Address tensor_b = allocate(4096);
Address scratch_buffer = 0xFFFF'0000;

// DMA program doesn't need to know which bank was used
dma.enqueue_transfer(tensor_a, scratch_buffer, 4096);

// If memory map changes (e.g., tensor_a remapped to different bank),
// the DMA program doesn't need to change
```

### Step 4: Error Handling

Error messages are more descriptive with the address-based API:

```cpp
// Before: Cryptic error
// Error: Invalid source memory bank ID: 5

// After: Clear address-based error
// Error: Source address 0x5000'0000 is not mapped in address space
// Available regions:
//   0x0000'0000 - 0x1FFF'FFFF: External Bank 0 (512 MB)
//   0xFFFF'0000 - 0xFFFF'FFFF: Scratchpad 0 (256 KB)
```

### Conversion Checklist

- [ ] Create `AddressDecoder` instance
- [ ] Map all memory regions with `add_region()`
- [ ] Call `dma.set_address_decoder(&decoder)` for each DMA engine
- [ ] Convert all `enqueue_transfer()` calls to use addresses instead of (type, id, offset) tuples
- [ ] Update address calculations to use region base addresses
- [ ] Test all transfer paths
- [ ] Remove deprecated API calls

### Testing Your Migration

Use the address-based test suite as a reference:

```bash
# Run address-based API tests
ctest -R dma_address_based_test -V

# View comprehensive examples
cat tests/dma/test_dma_address_based.cpp
```

---

## Tracing Integration

### Enabling Tracing

```cpp
DMAEngine dma(0, 1.0, 100.0);
dma.enable_tracing(true, &trace::TraceLogger::instance());
```

### Trace Events Generated

For each transfer, the DMA engine generates **two trace entries**:

#### 1. Transfer Issue (at start)
```cpp
TraceEntry {
    component_type: DMA_ENGINE
    component_id: engine_id
    transaction_type: TRANSFER
    transaction_id: unique_id
    cycle_issue: start_cycle
    status: ISSUED
    payload: DMAPayload {
        source: { address, size, id, type }
        destination: { address, size, id, type }
        bytes_transferred: size
        bandwidth_gb_s: bandwidth
    }
}
```

#### 2. Transfer Completion (at end)
```cpp
TraceEntry {
    component_type: DMA_ENGINE
    component_id: engine_id
    transaction_type: TRANSFER
    transaction_id: same_unique_id
    cycle_issue: start_cycle
    cycle_complete: end_cycle
    status: COMPLETED
    payload: same_DMAPayload
}
```

### Trace Analysis

```cpp
auto& logger = trace::TraceLogger::instance();

// Query DMA traces
auto dma_traces = logger.get_component_traces(ComponentType::DMA_ENGINE, 0);

// Analyze bandwidth
for (const auto& trace : dma_traces) {
    if (trace.status == TransactionStatus::COMPLETED) {
        auto& payload = std::get<DMAPayload>(trace.payload);
        uint64_t duration = trace.get_duration_cycles();
        double duration_ns = duration / trace.clock_freq_ghz.value();
        double effective_bw = payload.bytes_transferred / duration_ns;  // GB/s

        std::cout << "Transfer " << trace.transaction_id
                  << ": " << effective_bw << " GB/s\n";
    }
}
```

---

## Error Handling and Validation

### Early Capacity Validation

The DMA engine validates destination capacity **before starting the transfer** (lines 65-75 in `dma_engine.cpp`):

```cpp
// Validate destination capacity before starting the transfer
if (transfer.dst_type == MemoryType::SCRATCHPAD) {
    if (transfer.dst_id >= scratchpads.size()) {
        throw std::out_of_range("Invalid destination scratchpad ID");
    }
    if (transfer.dst_addr + transfer.size > scratchpads[transfer.dst_id].get_capacity()) {
        throw std::out_of_range("DMA transfer would exceed scratchpad capacity");
    }
}
```

**Why early validation?**
- Errors detected within a few cycles (not after full transfer latency)
- No wasted cycles reading source data for invalid transfers
- Immediate feedback for debugging

### Exception Handling in Tests

```cpp
sim->start_dma_external_to_scratchpad(0, 0, 0, 0, invalid_addr, size);

bool exception_thrown = false;
for (int i = 0; i < 100 && !exception_thrown; ++i) {
    try {
        sim->step();
    } catch (const std::out_of_range& e) {
        exception_thrown = true;
        std::cout << "Error caught: " << e.what() << "\n";
        break;
    }
}
```

### Validation Points

1. **At Transfer Start**:
   - Destination capacity validation (for scratchpad destinations)
   - Transaction ID generation

2. **During Source Read**:
   - Source ID bounds checking
   - Source address validation (by memory component)

3. **During Destination Write**:
   - Destination ID bounds checking
   - Destination address validation (by memory component)
   - Final capacity check (redundant but safe)

---

## Performance Characteristics

### Typical Configurations

| Configuration | Clock Freq | Bandwidth | Bytes/Cycle | 4KB Transfer Cycles |
|--------------|-----------|-----------|-------------|---------------------|
| GDDR6        | 2.0 GHz   | 512 GB/s  | 256 bytes   | 16 cycles          |
| HBM3         | 1.0 GHz   | 900 GB/s  | 900 bytes   | 5 cycles           |
| Test Config  | 1.0 GHz   | 100 GB/s  | 100 bytes   | 41 cycles          |

### Measured Performance (from test suite)

```
Transfer Size | Duration (cycles) | Effective BW (GB/s)
-------------------------------------------------------
1024 bytes   | 10 cycles         | 102.4 GB/s
4096 bytes   | 40 cycles         | 102.4 GB/s
16384 bytes  | 163 cycles        | 100.515 GB/s
65536 bytes  | 655 cycles        | 100.055 GB/s
```

**Note**: Effective bandwidth approaches theoretical bandwidth for larger transfers.

### Concurrent DMA Operations

Multiple DMA engines can operate in parallel:

```cpp
// Engine 0: Bank 0 → Scratchpad 0
dma_engines[0].enqueue_transfer(EXTERNAL, 0, addr, SCRATCHPAD, 0, addr, size);

// Engine 1: Bank 1 → Scratchpad 1 (concurrent)
dma_engines[1].enqueue_transfer(EXTERNAL, 1, addr, SCRATCHPAD, 1, addr, size);

// Both process in parallel
while (dma_engines[0].is_busy() || dma_engines[1].is_busy()) {
    for (auto& dma : dma_engines) {
        dma.process_transfers(...);
        dma.set_current_cycle(dma.get_current_cycle() + 1);
    }
}
```

---

## Integration with KPU Simulator

### High-Level API

The `KPUSimulator` class provides convenience methods that internally use DMA engines:

```cpp
// External Memory → Scratchpad
sim->start_dma_external_to_scratchpad(
    dma_id, memory_bank_id, src_addr,
    scratchpad_id, dst_addr, size, callback
);

// Scratchpad → External Memory
sim->start_dma_scratchpad_to_external(
    dma_id, scratchpad_id, src_addr,
    memory_bank_id, dst_addr, size, callback
);

// Check status
bool busy = sim->is_dma_busy(dma_id);

// Step simulation (advances all DMA engines)
sim->step();
```

### Orchestration Pattern

```cpp
// Typical data flow for matrix multiplication
// 1. Load matrices from external memory to scratchpad
sim->start_dma_external_to_scratchpad(...);
while (sim->is_dma_busy(0)) { sim->step(); }

// 2. Perform computation
sim->start_matmul(...);
while (sim->is_compute_busy(0)) { sim->step(); }

// 3. Write result back to external memory
sim->start_dma_scratchpad_to_external(...);
while (sim->is_dma_busy(0)) { sim->step(); }
```

---

## Design Rationale

### Why Cycle-Accurate?

1. **Realistic Timing**: Matches real hardware behavior where transfers are bandwidth-limited
2. **Trace Accuracy**: Enables accurate analysis of data movement bottlenecks
3. **Architecture Exploration**: Allows evaluation of different bandwidth/memory configurations
4. **Bug Detection**: Revealed timing issues in autonomous orchestration (BlockMover starting before DMA completion)

### Why Queue-Based?

1. **Simplicity**: FIFO ordering is predictable and easy to reason about
2. **Hardware Realism**: Real DMA controllers typically have command queues
3. **Flexibility**: Allows batching multiple transfers without blocking

### Why Early Validation?

1. **Fast Failure**: Detect errors in 1-2 cycles instead of after full transfer latency
2. **Resource Efficiency**: Don't waste bandwidth on invalid transfers
3. **Debugging**: Immediate feedback helps identify configuration errors

---

## Future Enhancements

Potential areas for extension:

1. **Scatter/Gather**: Support for non-contiguous memory regions
2. **2D Transfers**: Direct support for matrix/tensor slicing
3. **Priority Queues**: Multiple priority levels for transfer scheduling
4. **Pipelining**: Overlap read and write phases for back-to-back transfers
5. **Prefetching**: Speculative data movement based on access patterns
6. **Multi-Channel**: Independent read/write channels for simultaneous operations

---

## References

- **Implementation**: `src/components/datamovement/dma_engine.cpp`
- **Header**: `include/sw/kpu/components/dma_engine.hpp`
- **Tests**: `tests/dma/test_dma_basic.cpp`
- **Tracing Tests**: `tests/trace/test_dma_tracing.cpp`
- **Related**: `docs/autonomous-timing-fixes.md` (timing bug analysis)
