# Cycle-Accurate DMA-Memory-NoC Integration Plan

## Overview

Implement cycle-accurate DMA engines that connect to memory controllers and NoC edge routers, enabling true end-to-end data movement simulation with backpressure handling.

## Architecture

```
                    DMA[N0]   DMA[N1]   DMA[N2]   DMA[N3]
                      ↓         ↓         ↓         ↓
                    [R0,0]----[R0,1]----[R0,2]----[R0,3]
    DMA[W0] -----→    |         |         |         |
                    [R1,0]----[R1,1]----[R1,2]----[R1,3]
    DMA[W1] -----→    |         |         |         |
                    [R2,0]----[R2,1]----[R2,2]----[R2,3]
    DMA[W2] -----→    |         |         |         |
                    [R3,0]----[R3,1]----[R3,2]----[R3,3]
    DMA[W3] -----→

West DMAs: Load A tiles (row ingress, flowing East)
North DMAs: Load B tiles (column ingress, flowing South)
Each DMA connects to: Memory Controller + NoC Edge Router
```

## Current State

**Working:**
- Memory Controllers: Cycle-accurate FSMs for LPDDR5, DDR5, HBM, GDDR with `submit_read/write` + callbacks
- NoC Wormhole: `dma_inject()` on edge routers with `InjectResult` backpressure
- DMA Interface: `IDMAEngine` with memory callbacks (defined but unused)

**Missing:**
- `CycleAccurateDMAEngine` (stubbed, falls back to transactional)
- DMA ↔ Memory wiring
- DMA ↔ NoC wiring
- DMA placement configuration

## Implementation Plan

### Phase 1: DMA Channel State Machine

**File:** `include/sw/kpu/components/dma/cycle_accurate_dma_engine.hpp`

```cpp
enum class DMAChannelState : uint8_t {
    IDLE,                   // Ready for new transfer
    WAITING_MEMORY_READ,    // Issued memory read, awaiting response
    MEMORY_READ_COMPLETE,   // Data ready, preparing for NoC injection
    WAITING_NOC_INJECT,     // Retry NoC injection (got BUSY)
    NOC_INJECTING,          // Active NoC transfer
    WAITING_MEMORY_WRITE,   // Issued memory write (for stores)
    COMPLETE,               // Transfer done, invoking callback
    STALLED_MEMORY_FULL,    // Memory queue backpressure
    STALLED_NOC_FULL        // NoC injection backpressure
};

struct DMAChannel {
    uint32_t channel_id;
    DMAChannelState state = DMAChannelState::IDLE;

    // Active transfer
    uint64_t transfer_id;
    DMATransfer request;
    std::vector<uint8_t> data_buffer;
    std::optional<uint64_t> memory_request_id;
    bool memory_complete = false;

    // NoC injection
    uint8_t target_router_id;
    TileDescriptor tile;

    // Timing
    uint64_t last_memory_issue_cycle = 0;
    uint64_t last_noc_inject_cycle = 0;

    // Backpressure tracking
    uint32_t memory_retry_count = 0;
    uint32_t noc_retry_count = 0;

    std::function<void()> callback;
};
```

### Phase 2: CycleAccurateDMAEngine Class

**File:** `include/sw/kpu/components/dma/cycle_accurate_dma_engine.hpp`

```cpp
class CycleAccurateDMAEngine : public IDMAEngine {
public:
    // === IDMAEngine Interface ===
    std::optional<uint64_t> submit(const DMATransfer& transfer,
                                   std::function<void()> callback) override;
    bool can_accept() const override;
    void tick() override;

    // === External Bindings ===
    void bind_memory_controller(IMemoryController* controller);
    void bind_noc(noc::INoC* noc, uint8_t edge_router_id);

    // === Address Mapping ===
    void set_tile_mapper(std::function<TileDescriptor(uint64_t, uint32_t)> mapper);

private:
    std::vector<DMAChannel> channels_;
    IMemoryController* memory_controller_ = nullptr;
    noc::INoC* noc_ = nullptr;
    uint8_t bound_router_id_ = 0;

    // FSM processing per channel state
    void process_channel(uint32_t ch_id);
    void issue_memory_read(uint32_t ch_id);
    void attempt_noc_injection(uint32_t ch_id);
    void complete_transfer(uint32_t ch_id);
};
```

### Phase 3: Memory Controller Binding

**Flow:**
```
IDLE → check can_accept() → submit_read(addr, size, callback)
     → WAITING_MEMORY_READ → callback fires → MEMORY_READ_COMPLETE
```

**Implementation:**
```cpp
void CycleAccurateDMAEngine::issue_memory_read(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    if (!memory_controller_->can_accept()) {
        ch.state = DMAChannelState::STALLED_MEMORY_FULL;
        return;
    }

    auto req_id = memory_controller_->submit_read(
        ch.request.src_addr,
        ch.request.size,
        [this, ch_id]() { channels_[ch_id].memory_complete = true; });

    if (req_id) {
        ch.memory_request_id = *req_id;
        ch.state = DMAChannelState::WAITING_MEMORY_READ;
    }
}
```

### Phase 4: NoC Injection Binding

**Flow:**
```
MEMORY_READ_COMPLETE → dma_inject(router_id, dst, tile)
                     → SUCCESS: NOC_INJECTING → delivery callback → COMPLETE
                     → BUSY: STALLED_NOC_FULL → retry
```

**Implementation:**
```cpp
void CycleAccurateDMAEngine::attempt_noc_injection(uint32_t ch_id) {
    auto& ch = channels_[ch_id];

    auto result = noc_->dma_inject(
        bound_router_id_,
        ch.target_router_id,
        ch.tile,
        current_cycle_);

    switch (result) {
        case noc::InjectResult::SUCCESS:
            ch.state = DMAChannelState::NOC_INJECTING;
            break;
        case noc::InjectResult::BUSY:
            ch.state = DMAChannelState::STALLED_NOC_FULL;
            ch.noc_retry_count++;
            break;
        case noc::InjectResult::INVALID:
            // Error - log and fail
            break;
    }
}
```

### Phase 5: DMA Placement Configuration

**File:** `include/sw/kpu/components/dma/dma_placement.hpp`

```cpp
struct DMAPlacement {
    enum class Edge { WEST, NORTH, EAST, SOUTH };

    Edge edge;
    uint8_t position;           // Row (for WEST/EAST) or column (for NORTH/SOUTH)
    uint8_t router_id;          // Computed from edge + position
    uint8_t memory_controller_id;
};

struct DMASystemConfig {
    std::vector<DMAPlacement> west_edge_dmas;   // Column ingress per row
    std::vector<DMAPlacement> north_edge_dmas;  // Row ingress per column

    static DMASystemConfig create_standard(uint8_t mesh_rows, uint8_t mesh_cols,
                                           uint8_t num_memory_controllers);
};
```

### Phase 6: Factory Integration

**File:** `src/components/datamovement/dma_engine_factory.cpp`

```cpp
std::unique_ptr<IDMAEngine> create_dma_engine(const DMAEngineConfig& config) {
    switch (config.fidelity) {
        case SimulationFidelity::BEHAVIORAL:
            return std::make_unique<BehavioralDMAEngine>(config);
        case SimulationFidelity::TRANSACTIONAL:
            return std::make_unique<TransactionalDMAEngine>(config);
        case SimulationFidelity::CYCLE_ACCURATE:
            return std::make_unique<CycleAccurateDMAEngine>(config);  // NEW
    }
}

// System-level factory
DMASystem create_dma_system(const DMASystemConfig& placement,
                            std::vector<IMemoryController*>& mem_controllers,
                            noc::INoC* noc);
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `include/sw/kpu/components/dma/cycle_accurate_dma_engine.hpp` | **CREATE** - Class definition |
| `src/components/datamovement/cycle_accurate_dma_engine.cpp` | **CREATE** - Implementation |
| `include/sw/kpu/components/dma/dma_placement.hpp` | **CREATE** - Placement config |
| `src/components/datamovement/dma_engine_factory.cpp` | **MODIFY** - Add CYCLE_ACCURATE case |
| `include/sw/kpu/fidelity/component_config.hpp` | **MODIFY** - Add CycleAccurateDMAConfig |
| `tests/components/test_cycle_accurate_dma.cpp` | **CREATE** - Unit tests |
| `tests/integration/test_dma_memory_noc.cpp` | **CREATE** - Integration tests |

## Test Strategy

### Unit Tests (`test_cycle_accurate_dma.cpp`)

```cpp
TEST_CASE("DMA channel state machine", "[dma][cycle_accurate]") {
    SECTION("IDLE to WAITING_MEMORY_READ transition") { ... }
    SECTION("Memory backpressure handling") { ... }
    SECTION("NoC backpressure handling") { ... }
    SECTION("Complete transfer callback") { ... }
}
```

### Integration Tests (`test_dma_memory_noc.cpp`)

```cpp
TEST_CASE("DMA-Memory-NoC integration", "[dma][memory][noc][integration]") {
    // Create 4x4 mesh with memory controller and DMA system
    auto noc = create_noc(NoCType::WORMHOLE, config);
    auto mem = create_memory_controller(mem_config);
    auto dma_system = create_dma_system(placement, {mem.get()}, noc.get());

    SECTION("Load tile through west edge DMA") {
        // Submit transfer, tick until complete, verify data at destination
    }

    SECTION("Memory backpressure stalls DMA") {
        // Fill memory queue, verify DMA stalls and retries
    }

    SECTION("NoC congestion causes retry") {
        // Create NoC congestion, verify BUSY handling
    }
}
```

## Verification

1. **Build:** `cmake --build --preset release`
2. **Unit tests:** `ctest --preset default -R cycle_accurate_dma`
3. **Integration:** `ctest --preset default -R dma_memory_noc`
4. **Verify metrics:**
   - Memory backpressure causes STALLED_MEMORY_FULL state
   - NoC BUSY causes STALLED_NOC_FULL state
   - Transfers complete with correct latency
   - Statistics track stall cycles accurately

## Expected Results

| Scenario | Behavior |
|----------|----------|
| Normal transfer | IDLE → WAITING_MEMORY_READ → MEMORY_READ_COMPLETE → NOC_INJECTING → COMPLETE |
| Memory full | IDLE → STALLED_MEMORY_FULL → retry → WAITING_MEMORY_READ |
| NoC congested | MEMORY_READ_COMPLETE → STALLED_NOC_FULL → retry → NOC_INJECTING |
| Multi-channel | Channels operate independently with round-robin arbitration |
