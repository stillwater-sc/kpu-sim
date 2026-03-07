# Transactional Memory Controller Design

**Date:** 2026-02-07
**Status:** Design
**Goal:** Correct resource contention modeling for CSP timing model

## 1. Problem Statement

The current CSP timing model treats each "DMA channel" as an independent resource with
its own bandwidth. This is architecturally incorrect:

- **Reality:** One Memory Controller per channel, command bus is shared
- **Current:** Multiple DMAEngineProcess instances operate in parallel

## 2. Architecture

### 2.1 Physical Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Controller                             │
│  ┌─────────────┐  ┌──────────────────────────────────────────┐  │
│  │   Request   │  │           Bank State Machines             │  │
│  │    Queue    │  │  ┌─────┐ ┌─────┐ ┌─────┐      ┌─────┐   │  │
│  │   (CAM)     │──▶│ BSM0 │ │ BSM1 │ │ BSM2 │ ... │BSM15│   │  │
│  │             │  │  └─────┘ └─────┘ └─────┘      └─────┘   │  │
│  └─────────────┘  └──────────────────────────────────────────┘  │
│         │                          │                             │
│         ▼                          ▼                             │
│  ┌─────────────┐           ┌─────────────┐                      │
│  │   Command   │           │  Data Bus   │                      │
│  │   Arbiter   │──────────▶│  Occupancy  │                      │
│  │ (1 cmd/cyc) │           │  Tracker    │                      │
│  └─────────────┘           └─────────────┘                      │
│         │                          │                             │
└─────────│──────────────────────────│─────────────────────────────┘
          ▼                          ▼
    Command Bus                 Data Bus (16/32-bit)
    (to LPDDR5)                 (to/from LPDDR5)
```

### 2.2 Bank State Machine

Each bank tracks:
- **State:** IDLE, ACTIVATING, ACTIVE(row), PRECHARGING
- **Open Row:** Which row is currently activated (if ACTIVE)
- **Ready Cycle:** When the bank will be ready for next command

```cpp
struct BankState {
    enum class State { IDLE, ACTIVATING, ACTIVE, PRECHARGING };
    State state = State::IDLE;
    uint32_t open_row = 0;        // Valid when ACTIVE
    Cycle ready_cycle = 0;        // When bank accepts next command
};
```

### 2.3 Request Types

```cpp
enum class AccessType {
    ROW_HIT,      // Row already open → just issue RD/WR
    ROW_MISS,     // Different row open → PRE + ACT + RD/WR
    ROW_EMPTY     // Bank idle → ACT + RD/WR
};
```

## 3. Simple Delay Model

For transactional simulation, use aggregate latencies:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tCL` | 10 cycles | CAS latency (row hit) |
| `tRCD` | 15 cycles | RAS to CAS delay |
| `tRP` | 15 cycles | Row precharge time |
| `tBURST` | 4 cycles | Burst transfer time (64B @ 16B/cycle) |

### Access Latencies

| Access Type | Command Sequence | Total Latency |
|-------------|------------------|---------------|
| Row Hit | RD/WR | tCL + tBURST = 14 cycles |
| Row Empty | ACT → RD/WR | tRCD + tCL + tBURST = 29 cycles |
| Row Miss | PRE → ACT → RD/WR | tRP + tRCD + tCL + tBURST = 44 cycles |

## 4. Resource Contention

### 4.1 Command Bus (Critical Constraint)

**Only 1 command can be issued per cycle.**

This means:
- If Bank 0 needs ACT and Bank 1 needs RD, they serialize
- Command arbiter selects which bank gets the bus
- Other banks wait

### 4.2 Data Bus Occupancy

After a RD/WR command, the data bus is occupied for `tBURST` cycles.
- Cannot issue another RD/WR until data bus is free
- ACT and PRE can still be issued (command bus only)

### 4.3 Bank Conflicts

If two requests target the same bank:
- First request proceeds normally
- Second request waits until bank is ready

## 5. Implementation

### 5.1 Class Structure

```cpp
namespace sw::kpu::timing {

class MemoryControllerProcess : public IProcess {
public:
    struct Config {
        uint32_t controller_id = 0;
        size_t num_banks = 16;           // LPDDR5: 16 banks (4 BG × 4 banks)
        size_t request_queue_depth = 32;

        // Timing parameters (simple model)
        Cycle t_cl = 10;      // CAS latency
        Cycle t_rcd = 15;     // RAS to CAS delay
        Cycle t_rp = 15;      // Row precharge
        Cycle t_burst = 4;    // Burst duration

        // Bandwidth
        double bandwidth_gbps = 25.6;  // Per-channel bandwidth
        double clock_ghz = 1.0;
    };

    MemoryControllerProcess(const Config& config,
                            CreditPool& l3_credits,
                            TagCAM& l3_tag_cam);

    // IProcess interface
    std::vector<TimingEvent> tick(Cycle current_cycle) override;
    bool is_idle() const override;
    bool has_pending_work() const override;

    // Scheduling interface
    void schedule_load(const TileDescriptor& tile);
    void schedule_store(const TileDescriptor& tile);

private:
    // Bank state tracking
    std::vector<BankState> bank_states_;

    // Request queue (pending transfers)
    struct PendingRequest {
        TileDescriptor tile;
        bool is_load;
        Cycle enqueue_cycle;
        uint32_t bank_id;      // Computed from address
        uint32_t row_id;       // Computed from address
    };
    std::vector<PendingRequest> request_queue_;

    // In-flight tracking
    struct InFlightTransfer {
        TileDescriptor tile;
        bool is_load;
        Cycle complete_cycle;
    };
    std::vector<InFlightTransfer> in_flight_;

    // Resource tracking
    Cycle command_bus_ready_ = 0;  // When command bus is free
    Cycle data_bus_ready_ = 0;     // When data bus is free

    // Internal methods
    AccessType classify_access(uint32_t bank_id, uint32_t row_id) const;
    Cycle compute_latency(AccessType type) const;
    uint32_t address_to_bank(uint64_t addr) const;
    uint32_t address_to_row(uint64_t addr) const;
    void try_issue_command(Cycle current_cycle, std::vector<TimingEvent>& events);
};

} // namespace
```

### 5.2 Command Arbitration (Simple FCFS with Bank Awareness)

```cpp
void MemoryControllerProcess::try_issue_command(Cycle current_cycle,
                                                 std::vector<TimingEvent>& events) {
    // Can only issue if command bus is free
    if (current_cycle < command_bus_ready_) return;

    // Find first ready request (simple FCFS)
    for (auto it = request_queue_.begin(); it != request_queue_.end(); ++it) {
        auto& req = *it;
        auto& bank = bank_states_[req.bank_id];

        // Check if bank is ready
        if (current_cycle < bank.ready_cycle) continue;

        // Classify access type
        AccessType access_type = classify_access(req.bank_id, req.row_id);

        // For data commands (RD/WR), check data bus
        if (access_type == AccessType::ROW_HIT) {
            if (current_cycle < data_bus_ready_) continue;
        }

        // Issue the command
        Cycle latency = compute_latency(access_type);

        // Update bank state
        if (access_type == AccessType::ROW_EMPTY || access_type == AccessType::ROW_MISS) {
            bank.state = BankState::State::ACTIVE;
            bank.open_row = req.row_id;
        }
        bank.ready_cycle = current_cycle + latency;

        // Update bus occupancy
        command_bus_ready_ = current_cycle + 1;  // 1 command per cycle
        data_bus_ready_ = current_cycle + latency;  // Data bus occupied until complete

        // Create in-flight transfer
        in_flight_.push_back({req.tile, req.is_load, current_cycle + latency});

        // Emit event
        auto event_type = req.is_load ? EventType::DMA_LOAD_START : EventType::DMA_STORE_START;
        events.push_back(TimingEvent(event_type, current_cycle,
                                     config_.controller_id, req.tile.tile_id, name()));

        // Remove from queue
        request_queue_.erase(it);
        return;  // Only 1 command per cycle
    }
}
```

## 6. Integration with ConcurrentTimingExecutor

### 6.1 Replace DMAEngineProcess

```cpp
// Old (incorrect):
std::vector<std::unique_ptr<DMAEngineProcess>> dma_engines_;

// New (correct):
std::vector<std::unique_ptr<MemoryControllerProcess>> memory_controllers_;
```

### 6.2 Configuration Changes

```cpp
struct Config {
    // Old:
    // size_t num_dma_engines = 4;
    // size_t channels_per_mc = 2;

    // New:
    size_t num_memory_controllers = 1;  // 1 per physical channel

    // MC timing (simple model)
    Cycle mc_t_cl = 10;
    Cycle mc_t_rcd = 15;
    Cycle mc_t_rp = 15;
    Cycle mc_t_burst = 4;
};
```

### 6.3 Scheduling API

```cpp
// Old:
void schedule_load(const TileDescriptor& tile, size_t engine_id);

// New:
void schedule_load(const TileDescriptor& tile, size_t mc_id = 0);
// MC is selected explicitly or by address interleaving
```

## 7. Verification

### 7.1 Resource Contention Test

```cpp
TEST_CASE("MC command bus serializes requests") {
    // Schedule 2 loads to different banks at cycle 0
    mc.schedule_load(tile_bank0);
    mc.schedule_load(tile_bank1);

    // First load starts at cycle 0
    // Second load starts at cycle 1 (command bus busy)
    // NOT both at cycle 0 (old incorrect behavior)
}
```

### 7.2 Row Hit/Miss Test

```cpp
TEST_CASE("Row hit is faster than row miss") {
    // First access to bank 0, row 0
    mc.schedule_load(tile_row0);
    // Second access to bank 0, row 0 (row hit)
    mc.schedule_load(tile_row0_different_col);
    // Third access to bank 0, row 1 (row miss)
    mc.schedule_load(tile_row1);

    // Row hit should complete before row miss
}
```

## 8. Migration Path

1. Create `MemoryControllerProcess` class
2. Add tests for resource contention
3. Update `ConcurrentTimingExecutor` to use MC instead of DMA engines
4. Update demos and existing tests
5. Remove `DMAEngineProcess` (or keep for backward compatibility)

## 9. Future Extensions

- **Bank Group Timing:** Different tCCD for same-BG vs different-BG
- **Refresh:** Model periodic refresh interference
- **Power States:** Track bank power-down modes
- **Reordering:** FR-FCFS (First-Ready, First-Come-First-Serve) scheduler
