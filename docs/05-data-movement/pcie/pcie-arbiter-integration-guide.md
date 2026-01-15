# PCIe Arbiter Integration Guide

## Overview

The PCIe Arbiter component implements proper serialization of PCIe bus transactions, modeling both command and data phases according to the PCIe TLP (Transaction Layer Packet) protocol.

## Architecture

### Transaction Types

**Command Phase (Non-Posted):**
- CONFIG_WRITE: DMA descriptor setup
- CONFIG_READ: Read device configuration
- MEMORY_READ: Read requests requiring completion

**Data Phase (Posted):**
- MEMORY_WRITE: Bulk data transfers (fire-and-forget)

**Completion Phase:**
- COMPLETION: Response packets for non-posted requests

### Key Features

1. **Separate Command and Data Queues**: Models PCIe's separation of configuration/control vs data traffic
2. **Tag-Based Request/Completion Matching**: Tracks outstanding non-posted transactions
3. **Proper Serialization**: Only one transaction per queue active at a time
4. **Bandwidth Modeling**: Different bandwidths for command (2 GB/s) vs data (32 GB/s for Gen4 x16)

## Integration Steps

### Step 1: Include Header

```cpp
#include <sw/system/pcie_arbiter.hpp>
```

### Step 2: Create Arbiter Instance

```cpp
// In execute_mlp_layer_autonomous or main simulation function
sw::system::PCIeArbiter pcie_arbiter(
    1.0,   // clock_freq_ghz
    32.0,  // link_bandwidth_gb_s (PCIe Gen4 x16)
    32     // max_outstanding_tags
);

// Enable tracing
pcie_arbiter.enable_tracing(true, &trace_logger);
```

### Step 3: Replace Direct traced_host_to_kpu_dma Calls

**OLD CODE (Direct Tracing):**
```cpp
traced_host_to_kpu_dma(kpu, transfer_buffer.data(), desc.host_src_addr,
                       bank_id, bank_input_addr, desc.transfer_size,
                       trace_logger, kpu->get_current_cycle(), desc.description);
```

**NEW CODE (Using PCIe Arbiter):**
```cpp
// Enqueue command phase (descriptor write)
sw::system::PCIeArbiter::TransactionRequest cmd_req;
cmd_req.type = sw::system::PCIeArbiter::TransactionType::CONFIG_WRITE;
cmd_req.transfer_size = 32;  // Descriptor size
cmd_req.requester_id = 0;    // CPU core ID
cmd_req.description = "DMA descriptor: " + desc.description;
cmd_req.src_addr = 0;        // Host side (not relevant for config)
cmd_req.dst_addr = 0;        // Device config space
cmd_req.src_component = trace::ComponentType::HOST_CPU;
cmd_req.dst_component = trace::ComponentType::DMA_ENGINE;
cmd_req.src_id = 0;
cmd_req.dst_id = 0;
pcie_arbiter.set_current_cycle(kpu->get_current_cycle());
pcie_arbiter.enqueue_request(cmd_req);

// Enqueue data phase (actual memory transfer)
sw::system::PCIeArbiter::TransactionRequest data_req;
data_req.type = sw::system::PCIeArbiter::TransactionType::MEMORY_WRITE;
data_req.transfer_size = desc.transfer_size;
data_req.requester_id = 0;
data_req.description = desc.description;
data_req.src_addr = desc.host_src_addr;
data_req.dst_addr = desc.kpu_dest_addr;
data_req.src_component = trace::ComponentType::HOST_MEMORY;
data_req.dst_component = trace::ComponentType::EXTERNAL_MEMORY;
data_req.src_id = 0;
data_req.dst_id = static_cast<uint32_t>(bank_id);
data_req.completion_callback = [&]() {
    orch.signal(DMA_INPUT_DONE);
};
pcie_arbiter.enqueue_request(data_req);
```

### Step 4: Add Arbiter to Simulation Loop

```cpp
while (!orch.is_complete()) {
    kpu->step();             // Advance KPU components
    pcie_arbiter.step();     // Advance PCIe arbiter ← ADD THIS
    orch.step();             // Check dependencies

    cycle_count++;
    // ... rest of loop
}

// Also step arbiter until idle
while (pcie_arbiter.is_busy()) {
    pcie_arbiter.step();
    kpu->step();
}
```

## Expected Trace Output

### Before (Incorrect - Concurrent Transfers):
```
Cycle 1: HOST_CPU CONFIG_WRITE DMA 0
Cycle 1: HOST_CPU CONFIG_WRITE DMA 1  ← INVALID: concurrent
Cycle 3: PCIE_BUS DATA DMA 0
Cycle 3: PCIE_BUS DATA DMA 1          ← INVALID: concurrent
```

### After (Correct - Serialized):
```
Cycle 1: HOST_CPU CONFIG_WRITE DMA 0
Cycle 1: [PCIE_CMD] Config Write (queued)
Cycle 2: [PCIE_CMD] Config Write DMA 0 (active)
Cycle 3: HOST_CPU CONFIG_WRITE DMA 1
Cycle 3: [PCIE_CMD] Config Write (queued)
Cycle 4: [PCIE_CMD] Config Write DMA 1 (active)
Cycle 5: [PCIE_DATA] Memory Write DMA 0 (active)  ← Serialized
Cycle 6: [PCIE_DATA] Memory Write DMA 1 (active)  ← Serialized
```

## Benefits

1. **Physically Correct**: Models actual PCIe bus contention
2. **Reusable**: Can be used for other bus arbitration (AXI, CHI, etc.)
3. **Extensible**: Easy to add QoS policies, priorities, virtual channels
4. **Educational**: Shows proper PCIe protocol layering

## Advanced Features (Future)

- **Virtual Channels**: Multiple logical channels over one physical link
- **QoS/Priority**: Weighted fair queuing for different traffic classes
- **Credit-Based Flow Control**: Model PCIe's actual flow control mechanism
- **TLP Splitting**: Large transfers split into maximum payload sizes

## References

- PCIe Base Specification (PCI-SIG)
- https://xillybus.com/tutorials/pci-express-tlp-pcie-primer-tutorial-guide-1
- https://www.fpga4fun.com/PCI-Express4.html
- "PCI Express System Architecture" by Budruk, Anderson, Shanley
