# PCIe Bandwidth Model Correction

## Issue Identified

The initial PCIe arbiter implementation incorrectly modeled command and data transactions as having **different bandwidths**, as if they were on separate physical channels:

**Original (Incorrect) Model:**
- Command transactions: 2 GB/s → 32-byte descriptor took 16 cycles
- Data transactions: 32 GB/s → 128-byte payload took 4 cycles

This was backwards - small descriptors were taking more time than large data payloads!

## Root Cause

The mistake was conceptualizing PCIe as having separate "command" and "data" buses with different speeds. In reality:

- **PCIe has ONE physical link** (the lanes)
- All TLPs (Transaction Layer Packets) share the **same bandwidth**
- The only difference between transactions is their **size**

## Corrected Model

**Single Shared Link Bandwidth:**
- All transactions use 32 GB/s (PCIe Gen4 x16)
- Duration = packet_size / bandwidth

**Results at 1 GHz clock, 32 GB/s:**
- 32 bytes/cycle transfer rate
- 32-byte descriptor: 32 / 32 = **1 cycle** ✓
- 128-byte payload: 128 / 32 = **4 cycles** ✓

This makes intuitive sense: **smaller packets take less time**.

## Code Changes

### Header (pcie_arbiter.hpp)

```cpp
// Before:
double command_bandwidth_gb_s_;
double data_bandwidth_gb_s_;

PCIeArbiter(double clock_freq_ghz = 1.0,
            double command_bandwidth_gb_s = 2.0,
            double data_bandwidth_gb_s = 32.0,
            uint32_t max_outstanding_tags = 32);

// After:
double link_bandwidth_gb_s_;  // Single shared link

PCIeArbiter(double clock_freq_ghz = 1.0,
            double link_bandwidth_gb_s = 32.0,
            uint32_t max_outstanding_tags = 32);
```

### Implementation (pcie_arbiter.cpp)

```cpp
// Before:
trace::CycleCount PCIeArbiter::calculate_duration(const TransactionRequest& request) const {
    double bandwidth_gb_s = (request.type == TransactionType::MEMORY_WRITE)
                             ? data_bandwidth_gb_s_
                             : command_bandwidth_gb_s_;  // WRONG!
    // ...
}

// After:
trace::CycleCount PCIeArbiter::calculate_duration(const TransactionRequest& request) const {
    // All transaction types use the same PCIe link bandwidth
    double bytes_per_cycle = link_bandwidth_gb_s_ / clock_freq_ghz_;
    // Duration depends only on size, not transaction type
    // ...
}
```

## Verification

**Trace Output After Fix:**

```
Cycle   2-  3 ( 1 cycle,  32 bytes): [PCIE_CMD] Config Write: DMA descriptor
Cycle   3-  7 ( 4 cycles, 128 bytes): [PCIE_DATA] Memory Write: Input tensor
Cycle   8-  9 ( 1 cycle,  32 bytes): [PCIE_CMD] Config Write: DMA descriptor
Cycle   9- 13 ( 4 cycles, 128 bytes): [PCIE_DATA] Memory Write: Weight matrix
```

✅ **All timing now correct:**
- Small descriptors (32B) = 1 cycle
- Large data (128B) = 4 cycles
- Perfectly serialized (no overlaps)

## Lesson Learned

When modeling interconnects:

1. **Understand the physical reality**: PCIe has one shared link, not separate channels
2. **Transaction types** (command vs data) are **logical classifications** for:
   - TLP packet format
   - Completion requirements (posted vs non-posted)
   - Arbitration priority

3. **Bandwidth is shared**: All packet types compete for the same physical link bandwidth

4. **Bandwidth models should match hardware**:
   - ❌ Different bandwidths for different transaction types (unless truly separate channels like AXI)
   - ✅ Single link bandwidth, duration = size / bandwidth

## Comparison with Other Interconnects

| Interconnect | Architecture | Correct Model |
|--------------|--------------|---------------|
| **PCIe** | Single bidirectional link | One shared bandwidth |
| **AXI** | 5 independent channels | Separate bandwidth per channel (read addr, write addr, read data, write data, write response) |
| **CHI (Coherent Hub Interface)** | 3 virtual networks | Shared link, but separate QoS/priority |

For the KPU's PCIe interface, the single shared link model is correct.
