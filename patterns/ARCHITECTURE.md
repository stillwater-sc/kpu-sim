# LPDDR5 Memory Controller Architecture

This document provides the technical reference for the cycle-accurate LPDDR5 memory controller used in pattern testing.

---

## LPDDR5 Memory Organization

### Physical Structure

```
LPDDR5 Package (Single Channel)
├── Channel 0
│   ├── Bank Group 0
│   │   ├── Bank 0
│   │   ├── Bank 1
│   │   ├── Bank 2
│   │   └── Bank 3
│   ├── Bank Group 1
│   │   ├── Bank 4
│   │   ├── Bank 5
│   │   ├── Bank 6
│   │   └── Bank 7
│   ├── Bank Group 2
│   │   ├── Bank 8
│   │   ├── Bank 9
│   │   ├── Bank 10
│   │   └── Bank 11
│   └── Bank Group 3
│       ├── Bank 12
│       ├── Bank 13
│       ├── Bank 14
│       └── Bank 15
```

### Bank Group Significance

Banks within the same group share internal resources, creating timing constraints:
- **tRRD_L** (6 cycles): ACT-to-ACT delay within same bank group
- **tCCD_L** (6 cycles): CAS-to-CAS delay within same bank group
- **tWTR_L** (10 cycles): Write-to-read turnaround within same bank group

Banks in different groups have relaxed timings:
- **tRRD_S** (4 cycles): ACT-to-ACT delay across bank groups
- **tCCD_S** (4 cycles): CAS-to-CAS delay across bank groups
- **tWTR_S** (4 cycles): Write-to-read turnaround across bank groups

---

## Bank State Machine

```
                    ┌─────────┐
         ┌─────────►│  IDLE   │◄─────────┐
         │          └────┬────┘          │
         │               │ ACTIVATE      │ tRP complete
         │               ▼               │
         │          ┌─────────────┐      │
         │          │ ACTIVATING  │      │
         │          │   (tRCD)    │      │
         │          └──────┬──────┘      │
         │                 │ tRCD        │
         │                 ▼ complete    │
         │          ┌─────────┐          │
    REF  │    ┌────►│ ACTIVE  │◄────┐    │ PRECHARGE
    done │    │     └────┬────┘     │    │
         │    │ burst    │ READ/    │    │
         │    │ done     │ WRITE    │ burst
         │    │          ▼          │ done
    ┌────┴────┴──┐  ┌─────────┐  ┌──┴────────┐
    │ REFRESHING │  │ READING │  │  WRITING  │
    │  (tRFCpb)  │  │(tBurst) │  │ (tBurst)  │
    └────────────┘  └─────────┘  └───────────┘
         ▲                            │
         │                            │ PRECHARGE
         │          ┌─────────────┐   │ (after tWR)
         └──────────│ PRECHARGING │◄──┘
          REFRESH   │    (tRP)    │
                    └─────────────┘
```

### State Descriptions

| State | Duration | Description |
|-------|----------|-------------|
| IDLE | - | No row open, ready for ACTIVATE or REFRESH |
| ACTIVATING | tRCD (14) | Row being opened, latching row address |
| ACTIVE | - | Row open, ready for READ or WRITE |
| READING | tBurst (8/16) | Read burst in progress on data bus |
| WRITING | tBurst (8/16) | Write burst in progress on data bus |
| PRECHARGING | tRP (14) | Row being closed, restoring bitlines |
| REFRESHING | tRFCpb (140) | Per-bank refresh operation |

---

## Page Hit/Miss/Conflict

### Page Hit
The requested row is already open in the bank.
- Command sequence: READ/WRITE only
- Latency: tCL (14) + tBurst (8) = 22 cycles

### Page Empty
The bank is idle (no row open).
- Command sequence: ACTIVATE → READ/WRITE
- Latency: tRCD (14) + tCL (14) + tBurst (8) = 36 cycles

### Page Conflict
A different row is open in the bank.
- Command sequence: PRECHARGE → ACTIVATE → READ/WRITE
- Latency: tRP (14) + tRCD (14) + tCL (14) + tBurst (8) = 50 cycles

---

## Data Bus and Command Bus

### Command Bus
- **Width**: Variable (command + address)
- **Constraints**: One command per cycle maximum
- **State**: IDLE or BUSY

### Data Bus
- **Width**: 16 bits per channel (32 bits for dual channel)
- **Burst modes**: BL16 (8 cycles) or BL32 (16 cycles)
- **States**:
  - IDLE: No data transfer
  - READ_BURST: Reading from DRAM to controller
  - WRITE_BURST: Writing from controller to DRAM
  - TURNAROUND: Direction change (tRTW or tWTR)

### Bus Turnaround

When switching between reads and writes, the data bus requires turnaround time:

| Transition | Delay | Description |
|------------|-------|-------------|
| Read → Write | tRTW (14) | Wait for read data, then switch to write |
| Write → Read | tWTR_L (10) | Wait for write data, then switch to read (same BG) |
| Write → Read | tWTR_S (4) | Wait for write data, then switch to read (diff BG) |

---

## Timing Constraints Visualization

### Single Bank Access Timeline

```
Cycle:    0   5   10  15  20  25  30  35  40  45  50
          |---|---|---|---|---|---|---|---|---|---|
Page Empty (first access):
          [ACT]─────tRCD─────>[RD]─tCL─>[BURST]
          |                   |         |──────|
          0                   14        28     36

Page Hit (row already open):
          [RD]─────tCL────>[BURST]
          |                |──────|
          0                14     22

Page Conflict (different row open):
          [PRE]──tRP──>[ACT]──tRCD──>[RD]─tCL─>[BURST]
          |            |             |         |──────|
          0            14            28        42     50
```

### Multi-Bank Access Timeline (Same Bank Group)

```
Cycle:    0   5   10  15  20  25  30  35  40
          |---|---|---|---|---|---|---|---|
Bank 0:   [ACT]─────tRCD─────>[RD]──>[BURST]
Bank 1:        [ACT]─tRRD_L──────────>[RD]──>[BURST]
               |<--6->|
```

### Multi-Bank Access Timeline (Different Bank Groups)

```
Cycle:    0   5   10  15  20  25  30  35  40
          |---|---|---|---|---|---|---|---|
Bank 0:   [ACT]─────tRCD─────>[RD]──>[BURST]
Bank 4:      [ACT]─tRRD_S──────────>[RD]──>[BURST]
             |<4>|
```

---

## Address Mapping

### Single Channel Address Format

```
|<-- MSB                                             LSB -->|
+----------------+--------+----------+-------------------+
|      Row       |  Bank  |  Column  |   Byte Offset     |
|   (16 bits)    |(4 bits)| (10 bits)|     (6 bits)      |
+----------------+--------+----------+-------------------+
     bits 36:20    19:16     15:6           5:0

Bank → Bank Group mapping:
  Bank 0-3   → Bank Group 0
  Bank 4-7   → Bank Group 1
  Bank 8-11  → Bank Group 2
  Bank 12-15 → Bank Group 3
```

### Dual Channel Address Format

```
|<-- MSB                                                  LSB -->|
+----------------+--------+----------+---------+-------------------+
|      Row       |  Bank  |  Column  | Channel |   Byte Offset     |
|   (16 bits)    |(4 bits)| (10 bits)| (1 bit) |     (6 bits)      |
+----------------+--------+----------+---------+-------------------+
     bits 37:21    20:17     16:7        6           5:0
```

---

## tFAW (Four Activate Window)

The tFAW constraint limits the number of ACTIVATE commands within a rolling window:

**Constraint**: At most 4 ACTIVATEs in any tFAW (24 cycle) window.

```
Cycle:    0   5   10  15  20  25  30  35  40
          |---|---|---|---|---|---|---|---|
          [A0]────────────────tFAW────────────────|
              [A1]                                |
                  [A2]                            |
                      [A3]                        |
                                         [A4]  ← Must wait until tFAW expires
                          |<----24 cycles---->|
```

This prevents excessive power draw from multiple simultaneous row activations.

---

## Per-Bank Refresh

LPDDR5 supports per-bank refresh (PBR) instead of all-bank refresh:

| Parameter | Value | Description |
|-----------|-------|-------------|
| tRFCpb | 140 cycles | Per-bank refresh cycle time |
| tREFIpb | 244 cycles | Per-bank refresh interval |

Per-bank refresh allows other banks to continue operation while one bank refreshes.

---

## Bandwidth Calculations

### Single Channel LPDDR5-6400

```
Data rate:     6400 MT/s
Bus width:     16 bits = 2 bytes
Peak BW:       6400 × 2 = 12,800 MB/s = 12.8 GB/s

Per burst (BL16):
  Bytes:       16 × 2 = 32 bytes (minimum)
  Extended:    64 bytes (typical access size)
  Cycles:      8

Sustained BW (page hits):
  64 bytes / (tCL + tBurst) = 64 / 22 = 2.9 bytes/cycle
  At 3200 MHz: 2.9 × 3200 = 9.3 GB/s (~73% of peak)
```

### Dual Channel LPDDR5-6400

```
Peak BW:       25.6 GB/s (2× single channel)

With perfect interleaving:
  Sustained:   ~18.6 GB/s (~73% of peak)
```

---

## Trace Events

The memory controller emits the following trace events for visualization:

| Event | Fields | Description |
|-------|--------|-------------|
| BANK_STATE_CHANGE | channel, bank, old_state, new_state, cycle | Bank state transition |
| CMD_ISSUE | channel, bank, cmd_type, row, cycle | Command issued |
| DATA_BUS_STATE | channel, state, cycle, duration | Data bus state change |
| REQUEST_SUBMIT | request_id, address, type, cycle | Request submitted |
| REQUEST_COMPLETE | request_id, latency, page_result | Request completed |

These events can be exported to Chrome Trace format for Perfetto visualization.

---

## Invariants Checked

The memory controller validates these invariants at every cycle:

### Bank Invariants
1. **ACTIVE_HAS_ROW**: ACTIVE state must have valid open_row
2. **STATE_TIMING**: State transitions respect minimum timings
3. **tRAS_MINIMUM**: Row must be active for at least tRAS before precharge

### Timing Invariants
1. **tRCD_RESPECTED**: Read/Write only after tRCD from activate
2. **tRP_RESPECTED**: Activate only after tRP from precharge
3. **tRRD_L_RESPECTED**: Same bank group activate spacing
4. **tRRD_S_RESPECTED**: Different bank group activate spacing
5. **tFAW_RESPECTED**: At most 4 activates in rolling window

### Bus Invariants
1. **NO_BUS_COLLISION**: Only one burst active on data bus
2. **TURNAROUND_RESPECTED**: R→W and W→R turnaround times
3. **CMD_BUS_SINGLE**: Only one command per cycle

---

## Integration with KPU Simulator

The LPDDR5 memory controller implements `IMemoryController` interface:

```cpp
#include <sw/kpu/components/lpddr5_memory_controller.hpp>

// Create single-channel configuration
sw::kpu::lpddr5::LPDDR5MemoryController::Config config;
config.num_channels = 1;
config.burst_length = sw::kpu::lpddr5::BurstLength::BL16;
config.queue_depth = 64;

auto mc = std::make_unique<sw::kpu::lpddr5::LPDDR5MemoryController>(config);

// Enable tracing
mc->enable_tracing(true);
sw::trace::ResourceTracker tracker;
mc->set_resource_tracker(&tracker);

// Submit requests
mc->submit_read(address, 64, [](){ /* callback */ });

// Tick simulation
while (mc->has_pending()) {
    mc->tick();
}

// Check for violations
if (mc->has_violations()) {
    for (const auto& v : mc->violations()) {
        std::cerr << "Cycle " << v.cycle << ": " << v.message << std::endl;
    }
}

// Export trace
sw::trace::ChromeTraceExporter::export_traces("trace.json", mc->trace_entries(), 3.2);
```
