# PCIe Arbiter Implementation Summary

## Overview

Successfully integrated PCIe bus arbitration into the KPU simulator to properly model the physical constraint that the PCIe bus is a shared resource that can only handle one transaction at a time.

## Problem Statement

Prior to this implementation, the trace logs showed concurrent PCIe transfers, which is physically impossible:
- Multiple DMA descriptor writes happening simultaneously
- Data transfers overlapping with command transfers
- No serialization of bus access

## Solution Architecture

### Component Design: PCIeArbiter

**Location:**
- Header: `include/sw/system/pcie_arbiter.hpp`
- Implementation: `src/system/pcie_arbiter.cpp`
- Integration: `models/kpu/host_t100_autonomous.cpp`

**Key Features:**

1. **Separate Transaction Queues:**
   - Command queue: Non-posted transactions (CONFIG_WRITE, CONFIG_READ, MEMORY_READ)
   - Data queue: Posted transactions (MEMORY_WRITE for bulk data)
   - Completion queue: Completion packets for non-posted requests

2. **Transaction Slots:**
   - Each queue has an associated transaction slot
   - Slots track: busy state, current request, start cycle, completion cycle, trace ID

3. **Bus-Level Serialization:**
   - Only ONE transaction can be active across all slots at any time
   - Priority ordering: Completion > Command > Data
   - Single shared link bandwidth (e.g., 32 GB/s for PCIe Gen4 x16)
   - Transaction duration based purely on size: small descriptors (32B) = 1 cycle, large payloads (128B) = 4 cycles

### Implementation Details

#### Step Function Logic

```cpp
void PCIeArbiter::step() {
    // 1. Increment cycle
    current_cycle_++;

    // 2. Check for completed transactions and free slots
    if (command_slot_.busy && current_cycle_ >= command_slot_.completion_cycle) {
        // Complete transaction, call callbacks, free slot
    }
    // (similar for data and completion slots)

    // 3. Only start new transaction if bus is completely free
    bool bus_busy = command_slot_.busy || data_slot_.busy || completion_slot_.busy;

    if (!bus_busy) {
        // Arbitrate: priority is completion > command > data
        if (!completion_queue_.empty()) {
            // Start completion transaction
        } else if (!command_queue_.empty()) {
            // Start command transaction
        } else if (!data_queue_.empty()) {
            // Start data transaction
        }
    }
}
```

#### Transaction Request Structure

```cpp
struct TransactionRequest {
    TransactionType type;
    sw::kpu::Size transfer_size;
    std::string description;
    std::function<void()> completion_callback;
    // Routing information
    sw::kpu::Address src_addr, dst_addr;
    trace::ComponentType src_component, dst_component;
    uint32_t src_id, dst_id;
};
```

### Integration Pattern

In host_t100_autonomous.cpp:

```cpp
// 1. Create arbiter (clock_freq_ghz, link_bandwidth_gb_s, max_tags)
sw::system::PCIeArbiter pcie_arbiter(1.0, 32.0, 32);
pcie_arbiter.enable_tracing(true, &trace_logger);

// 2. Enqueue transactions
// Command phase
PCIeArbiter::TransactionRequest cmd_req;
cmd_req.type = PCIeArbiter::TransactionType::CONFIG_WRITE;
cmd_req.transfer_size = 32;  // Descriptor size
cmd_req.description = "DMA descriptor";
pcie_arbiter.enqueue_request(cmd_req);

// Data phase
PCIeArbiter::TransactionRequest data_req;
data_req.type = PCIeArbiter::TransactionType::MEMORY_WRITE;
data_req.transfer_size = 128;
data_req.completion_callback = [&]() { /* signal done */ };
pcie_arbiter.enqueue_request(data_req);

// 3. Step in simulation loop
while (!done) {
    kpu->step();
    pcie_arbiter.step();  // Advance PCIe arbiter
    orch.step();
}
```

## Results

### Before Integration

```
Cycle 1: HOST_CPU CONFIG_WRITE DMA 0
Cycle 1: HOST_CPU CONFIG_WRITE DMA 1  ← INVALID: concurrent
Cycle 3: PCIE_BUS DATA DMA 0
Cycle 3: PCIE_BUS DATA DMA 1          ← INVALID: concurrent
```

**Issues:** Multiple transactions concurrent on shared PCIe bus

### After Integration

```
Cycle   2-  3 (1 cycle):  [PCIE_CMD] Config Write: DMA descriptor (32 bytes)
Cycle   3-  7 (4 cycles): [PCIE_DATA] Memory Write: Input tensor (128 bytes) ✓
Cycle   8-  9 (1 cycle):  [PCIE_CMD] Config Write: DMA descriptor (32 bytes) ✓
Cycle   9- 13 (4 cycles): [PCIE_DATA] Memory Write: Weight matrix (128 bytes) ✓
```

**Verification:**
- ✅ All transactions properly serialized (no overlaps)
- ✅ Realistic timing: small descriptors (32B) = 1 cycle, large payloads (128B) = 4 cycles
- ✅ Single shared link bandwidth correctly modeled (32 GB/s)
- ✅ Completion callbacks fire correctly

## Files Modified

1. **include/sw/system/pcie_arbiter.hpp** - New file
   - PCIeArbiter class definition
   - Transaction types and request structures
   - Transaction slot structure

2. **src/system/pcie_arbiter.cpp** - New file
   - Core arbitration logic
   - Queue processing
   - Trace generation

3. **src/system/CMakeLists.txt**
   - Added pcie_arbiter.cpp to SYSTEM_SOURCES

4. **models/kpu/host_t100_autonomous.cpp**
   - Added #include for pcie_arbiter.hpp
   - Created PCIeArbiter instance
   - Replaced traced_host_to_kpu_dma calls with arbiter requests
   - Added pcie_arbiter.step() to simulation loop

5. **docs/pcie-arbiter-integration-guide.md** - New file
   - Integration guide for users

## Performance Impact

- Execution cycles: ~54 cycles (realistic due to proper serialization)
- PCIe transaction overhead: 2 descriptors (1 cycle each) + 2 data transfers (4 cycles each) = 10 cycles total
- Trace event count: 24 events
- All functional correctness tests pass
- Results verified correct
- Timing now accurately reflects PCIe Gen4 x16 bandwidth (32 GB/s)

## Future Enhancements

Potential improvements to the PCIe arbiter:

1. **Virtual Channels:** Multiple logical channels over one physical link
2. **QoS/Priority:** Weighted fair queuing for different traffic classes
3. **Credit-Based Flow Control:** Model PCIe's actual flow control mechanism
4. **TLP Splitting:** Large transfers split into maximum payload sizes (e.g., 256 bytes)
5. **Realistic Latencies:** Add descriptor processing overhead, interrupt latency

## References

- PCIe Base Specification (PCI-SIG)
- https://xillybus.com/tutorials/pci-express-tlp-pcie-primer-tutorial-guide-1
- https://www.fpga4fun.com/PCI-Express4.html
- "PCI Express System Architecture" by Budruk, Anderson, Shanley

## Conclusion

The PCIe arbiter successfully models the physical constraint of the shared PCIe bus, ensuring that all transactions are properly serialized. This provides realistic timing behavior for performance analysis and demonstrates proper hardware modeling techniques for bus arbitration that can be extended to other interconnects (AXI, CHI, etc.).
