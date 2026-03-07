# Correct CSP Architecture: DMA Engine + Memory Controller

## Goal

Correctly model the data movement architecture with proper separation of concerns:
- **DMA Engine** = CSP Process (programmable, ISA-driven, schedulable)
- **Memory Controller** = Communication Resource (DRAM access contention)

## Architectural Correction

The previous implementation **incorrectly replaced** DMA Engine with Memory Controller. The correct model keeps BOTH:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ConcurrentTimingExecutor                             │
│                                                                         │
│  ┌─────────────────────────────┐      ┌──────────────────────────┐    │
│  │     DMA Engine (CSP)        │      │   Memory Controller      │    │
│  │                             │      │   (Shared Resource)      │    │
│  │  • Defines data movement    │      │                          │    │
│  │    ISA operations           │ uses │  • Command bus: 1/cycle  │    │
│  │  • Programmable queues      │─────►│  • 16 Bank State Machines│    │
│  │  • Schedulable process      │      │  • Row hit/miss/empty    │    │
│  │  • schedule_load/store()    │      │  • tCL, tRCD, tRP, tBurst│    │
│  └──────────────┬──────────────┘      └──────────────────────────┘    │
│                 │                                                       │
│                 │ via Interconnect                                      │
│                 ▼                                                       │
│  ┌─────────────────────────────┐                                       │
│  │     L3 Memory Tiles         │                                       │
│  │  (L3 Credit Pool + TagCAM)  │                                       │
│  └─────────────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Roles

| Component | Type | Role |
|-----------|------|------|
| `DMAEngineProcess` | CSP Process | Executes data movement ISA, manages work queues |
| `MemoryController` | Resource | Models DRAM command bus contention, bank states |
| `BlockMoverProcess` | CSP Process | Moves tiles L3↔L2 |
| `StreamerProcess` | CSP Process | Feeds/drains tiles L2↔Compute |

## Design: DMA Engine Uses Memory Controller

### Option A: MC as Internal Resource (Preferred)
DMA Engine internally holds a reference to a shared Memory Controller:

```cpp
class DMAEngineProcess : public IProcess {
public:
    DMAEngineProcess(const Config& config,
                     MemoryController& mc,        // Shared MC resource
                     CreditPool& l3_credits,
                     TagCAM& l3_tag_cam);

    void schedule_load(const TileDescriptor& tile);
    void schedule_store(const TileDescriptor& tile);
    std::vector<TimingEvent> tick(Cycle current_cycle) override;

private:
    MemoryController& mc_;  // DMA uses MC for DRAM access
    // ... DMA-specific queues and state
};
```

### Option B: Executor Mediates MC Access
Executor passes MC access through scheduling:

```cpp
// Executor routes DMA requests through MC
void schedule_load(tile) {
    uint32_t mc_id = address_to_mc(tile.dram_address);
    memory_controllers_[mc_id]->schedule_request(tile, /*is_load=*/true);
    // MC completion triggers DMA state update
}
```

## Selected Approach: MC as IProcess (Consistent with Other Processes)

**Rationale:**
1. Consistent with BlockMover and Streamer architecture
2. Executor ticks all processes uniformly
3. MC can emit its own timing events for tracing
4. Clear separation: DMA = "what" (tile operations), MC = "how" (DRAM protocol)

## Implementation Steps

### Step 1: Keep MemoryController as IProcess
MC implements IProcess and gets ticked by executor:

```cpp
class MemoryControllerProcess : public IProcess {
public:
    struct Config { /* bank count, timing params */ };

    MemoryControllerProcess(const Config& config);

    // IProcess interface - executor calls this
    std::vector<TimingEvent> tick(Cycle current_cycle) override;
    bool is_idle() const override;
    bool has_pending_work() const override;

    // DMA Engine submits requests to MC's queue
    void submit_request(const TileDescriptor& tile, bool is_load);

    // DMA checks if its request completed
    std::optional<CompletedTransfer> get_completed_transfer();

private:
    std::vector<BankState> bank_states_;
    std::vector<PendingRequest> request_queue_;
    std::vector<CompletedTransfer> completed_transfers_;
    Cycle command_bus_ready_ = 0;
};
```

### Step 2: Update DMAEngineProcess to Submit to MC
DMA submits requests and polls for completion:

```cpp
class DMAEngineProcess : public IProcess {
public:
    DMAEngineProcess(const Config& config,
                     MemoryControllerProcess& mc,  // Reference to shared MC
                     CreditPool& l3_credits,
                     TagCAM& l3_tag_cam);

    std::vector<TimingEvent> tick(Cycle current_cycle) override {
        std::vector<TimingEvent> events;

        // Check for completed transfers from MC
        while (auto completed = mc_.get_completed_transfer()) {
            // Update L3 tag cam, release credits, emit events
            handle_completion(*completed, events);
        }

        // Try to submit new requests to MC
        for (auto& pending : pending_requests_) {
            if (can_submit(pending)) {
                mc_.submit_request(pending.tile, pending.is_load);
                pending.submitted = true;
                emit_start_event(pending, events);
            }
        }

        return events;
    }

private:
    MemoryControllerProcess& mc_;  // DMA uses MC for DRAM access
    std::vector<PendingRequest> pending_requests_;
};
```

### Step 3: Update ConcurrentTimingExecutor
Executor creates MCs and DMAs, ticks both:

```cpp
class ConcurrentTimingExecutor {
private:
    std::vector<std::unique_ptr<MemoryControllerProcess>> memory_controllers_;
    std::vector<std::unique_ptr<DMAEngineProcess>> dma_engines_;

    void create_components() {
        // Create MCs first (1 per DRAM channel)
        for (size_t i = 0; i < config_.num_memory_controllers; ++i) {
            memory_controllers_.push_back(
                std::make_unique<MemoryControllerProcess>(mc_config));
        }

        // Create DMA engines, each assigned to an MC
        for (size_t i = 0; i < config_.num_dma_engines; ++i) {
            size_t mc_id = i % memory_controllers_.size();
            dma_engines_.push_back(
                std::make_unique<DMAEngineProcess>(
                    dma_config,
                    *memory_controllers_[mc_id],
                    l3_credits_,
                    l3_tag_cam_));
        }
    }

    bool step() {
        // Tick MCs first (process DRAM commands)
        for (auto& mc : memory_controllers_) {
            auto mc_events = mc->tick(current_cycle_);
            events_.insert(events_.end(), mc_events.begin(), mc_events.end());
        }

        // Tick DMAs (submit requests, handle completions)
        for (auto& dma : dma_engines_) {
            auto dma_events = dma->tick(current_cycle_);
            events_.insert(events_.end(), dma_events.begin(), dma_events.end());
        }

        // Tick BlockMovers, Streamers...
    }
};
```

## Files to Modify

| File | Action |
|------|--------|
| `include/sw/kpu/timing/memory_controller_process.hpp` | KEEP - Already has bank states, command bus |
| `include/sw/kpu/timing/dma_engine_process.hpp` | MODIFY - Add MC reference, submit/poll pattern |
| `include/sw/kpu/timing/concurrent_timing_executor.hpp` | MODIFY - Restore DMA engines, keep MCs, wire them together |
| `tests/timing/test_memory_controller.cpp` | KEEP - Tests MC contention correctly |
| `tests/timing/test_dma_engine_process.cpp` | MODIFY - Update for new DMA→MC API |
| `tests/timing/test_component_integration.cpp` | MODIFY - Use DMA+MC together |
| `tests/timing/CMakeLists.txt` | MODIFY - Re-enable DMA engine tests |

## Revert Changes Needed

The current working copy has incorrect changes that replaced DMA with MC. Need to:
1. **Revert concurrent_timing_executor.hpp** to use DMAEngineProcess (not just MC)
2. **Keep MemoryControllerProcess** with its bank states and command bus logic
3. **Add MC reference to DMAEngineProcess** so DMA submits requests to MC
4. **Add completion polling** so DMA can get results from MC
5. **Fix tests** to use the correct DMA+MC architecture

## Verification

```bash
# Build
cmake --build --preset release

# Run DMA tests
./build/tests/timing/test_dma_engine_process

# Run MC tests
./build/tests/timing/test_memory_controller

# Run integration tests
./build/tests/timing/test_component_integration

# Verify command bus constraint
# Multiple DMA requests should serialize through MC command bus
```

## Key Invariants

1. **DMA Engine defines the ISA** - schedule_load/store are the programming interface
2. **MC enforces DRAM contention** - 1 command/cycle, bank state tracking
3. **Multiple DMAs can share one MC** - realistic for multi-channel configs
4. **Latency = DMA overhead + MC latency** - composition of delays
