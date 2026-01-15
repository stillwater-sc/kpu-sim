# LPDDR5 State Model

This document defines the formal state model for LPDDR5 memory devices and their bank structures. This model establishes the invariants and state transitions that the memory controller must respect.

---

## Device Architecture

### Single-Channel Configuration
```
LPDDR5 Device (Single Channel)
└── Channel 0 (16-bit data bus, 3200 MHz)
    ├── Bank Group 0
    │   ├── Bank 0
    │   ├── Bank 1
    │   ├── Bank 2
    │   └── Bank 3
    ├── Bank Group 1
    │   ├── Bank 4
    │   ├── Bank 5
    │   ├── Bank 6
    │   └── Bank 7
    ├── Bank Group 2
    │   ├── Bank 8
    │   ├── Bank 9
    │   ├── Bank 10
    │   └── Bank 11
    └── Bank Group 3
        ├── Bank 12
        ├── Bank 13
        ├── Bank 14
        └── Bank 15
```

### Dual-Channel Configuration
```
LPDDR5 Device (Dual Channel)
├── Channel 0 (16-bit data bus, 3200 MHz)
│   ├── Bank Group 0 (Banks 0-3)
│   ├── Bank Group 1 (Banks 4-7)
│   ├── Bank Group 2 (Banks 8-11)
│   └── Bank Group 3 (Banks 12-15)
│
└── Channel 1 (16-bit data bus, 3200 MHz)
    ├── Bank Group 0 (Banks 0-3)
    ├── Bank Group 1 (Banks 4-7)
    ├── Bank Group 2 (Banks 8-11)
    └── Bank Group 3 (Banks 12-15)

Total: 32 banks (16 per channel), 32-bit combined data bus
```

### Key Parameters

| Parameter | Single Channel | Dual Channel |
|-----------|----------------|--------------|
| Data Bus Width | 16-bit (×16) | 32-bit (×32) |
| Banks per Channel | 16 | 16 |
| Bank Groups per Channel | 4 | 4 |
| Banks per Bank Group | 4 | 4 |
| Total Banks | 16 | 32 |
| Peak Bandwidth | 12.8 GB/s | 25.6 GB/s |
| Burst Length | BL16 or BL32 | BL16 or BL32 |
| Prefetch | 16n | 16n |

---

## Burst Length Modes

### BL16 (16 Transfers)

| Property | Value |
|----------|-------|
| Data per burst (×16) | 16 × 2 bytes = 32 bytes |
| Data per burst (×32) | 16 × 4 bytes = 64 bytes |
| Burst cycles | 8 (DDR: 2 transfers/cycle) |
| 4KB tile bursts (×16) | 128 |
| 4KB tile bursts (×32) | 64 |

### BL32 (32 Transfers)

| Property | Value |
|----------|-------|
| Data per burst (×16) | 32 × 2 bytes = 64 bytes |
| Data per burst (×32) | 32 × 4 bytes = 128 bytes |
| Burst cycles | 16 (DDR: 2 transfers/cycle) |
| 4KB tile bursts (×16) | 64 |
| 4KB tile bursts (×32) | 32 |

### Burst Mode Invariants

1. **INV-BL-1**: BL32 reduces command overhead but increases minimum transfer granularity
2. **INV-BL-2**: Cannot change burst length mid-operation
3. **INV-BL-3**: BL32 requires contiguous 64-byte (×16) or 128-byte (×32) alignment

---

## Device-Level States

| State | Description | Power | Valid From | Valid To |
|-------|-------------|-------|------------|----------|
| **POWER_OFF** | No power applied | None | - | POWER_ON_RESET |
| **POWER_ON_RESET** | Power applied, initializing | Full | POWER_OFF | INITIALIZATION |
| **INITIALIZATION** | Mode register setup, ZQ calibration | Full | POWER_ON_RESET | IDLE |
| **IDLE** | Ready for commands, all banks idle | Active | INITIALIZATION, SELF_REFRESH_EXIT, IDLE_POWER_DOWN_EXIT | ACTIVE, SELF_REFRESH, IDLE_POWER_DOWN, DEEP_SLEEP |
| **ACTIVE** | At least one bank has open row | Active | IDLE | IDLE, SELF_REFRESH, IDLE_POWER_DOWN |
| **SELF_REFRESH** | Self-refresh mode, retains data | Low | IDLE, ACTIVE (after precharge all) | SELF_REFRESH_EXIT |
| **SELF_REFRESH_EXIT** | Exiting self-refresh | Active | SELF_REFRESH | IDLE |
| **IDLE_POWER_DOWN** | Clock stopped, retains state | Medium | IDLE | IDLE_POWER_DOWN_EXIT |
| **IDLE_POWER_DOWN_EXIT** | Resuming from power down | Active | IDLE_POWER_DOWN | IDLE |
| **DEEP_SLEEP** | Minimal power, state lost | Minimal | IDLE | POWER_ON_RESET |

### Device State Transitions

```
                    ┌─────────────────┐
                    │   POWER_OFF     │
                    └────────┬────────┘
                             │ Power On
                             ▼
                    ┌─────────────────┐
                    │ POWER_ON_RESET  │
                    └────────┬────────┘
                             │ tINIT complete
                             ▼
                    ┌─────────────────┐
                    │ INITIALIZATION  │
                    └────────┬────────┘
                             │ MRW/ZQ complete
                             ▼
         ┌──────────────────┐ ┌──────────────────┐
         │  IDLE_POWER_DOWN │◄┤                  │
         └────────┬─────────┘ │      IDLE        │◄──────┐
                  │           │                  │       │
                  └──────────►└────────┬─────────┘       │
                                       │                 │
                    ┌──────────────────┼─────────────────┤
                    │                  │                 │
                    ▼                  ▼                 │
         ┌──────────────────┐ ┌──────────────────┐      │
         │   SELF_REFRESH   │ │     ACTIVE       │──────┘
         └──────────────────┘ └──────────────────┘
```

### Device State Invariants

1. **INV-DEV-1**: Device cannot accept read/write commands unless in ACTIVE or IDLE state
2. **INV-DEV-2**: All banks must be precharged before entering SELF_REFRESH
3. **INV-DEV-3**: At least tXSR cycles must pass after SELF_REFRESH_EXIT before commands
4. **INV-DEV-4**: ZQ calibration must complete during INITIALIZATION
5. **INV-DEV-5**: Mode registers must be written during INITIALIZATION
6. **INV-DEV-6**: Dual-channel: channels operate independently but share power states

---

## Bank-Level States

| State | Description | Row Buffer | Valid From | Valid To |
|-------|-------------|------------|------------|----------|
| **IDLE** | Precharged, no row open | Empty | PRECHARGING (after tRP), REFRESHING (after tRFCpb) | ACTIVATING |
| **ACTIVATING** | Row being opened | Loading | IDLE | ACTIVE |
| **ACTIVE** | Row open, ready for R/W | Valid | ACTIVATING (after tRCD) | READING, WRITING, PRECHARGING |
| **READING** | Read burst in progress | Valid | ACTIVE | ACTIVE, PRECHARGING |
| **WRITING** | Write burst in progress | Valid | ACTIVE | ACTIVE, PRECHARGING |
| **PRECHARGING** | Row being closed | Draining | ACTIVE, READING (after tRTP), WRITING (after tWR) | IDLE |
| **REFRESHING** | Bank being refreshed | Invalid | IDLE | IDLE |

### Bank State Transition Diagram

```
                         ┌───────────────┐
            ┌───────────►│     IDLE      │◄───────────┐
            │            └───────┬───────┘            │
            │                    │ ACT                │
            │ tRP                ▼                    │ tRFCpb
            │            ┌───────────────┐            │
            │            │  ACTIVATING   │            │
            │            └───────┬───────┘            │
            │                    │ tRCD               │
            │                    ▼                    │
            │            ┌───────────────┐            │
            ├────────────┤    ACTIVE     │────────────┤
            │      PRE   └───┬───────┬───┘            │
            │                │       │                │
            │           RD   │       │ WR             │
            │                ▼       ▼                │
            │         ┌────────┐ ┌────────┐           │
            │         │READING │ │WRITING │           │
            │         └───┬────┘ └────┬───┘           │
            │             │           │               │
            │    tRTP     │           │ tWR           │
            │             ▼           ▼               │
            │            ┌───────────────┐            │
            └────────────┤  PRECHARGING  │            │
                         └───────────────┘            │
                                                      │
                         ┌───────────────┐            │
                         │  REFRESHING   │────────────┘
                         └───────────────┘
```

### Bank State Invariants

1. **INV-BANK-1**: Only one row can be active per bank at any time
2. **INV-BANK-2**: Must wait tRCD after ACTIVATE before READ/WRITE
3. **INV-BANK-3**: Must wait tRP after PRECHARGE before next ACTIVATE
4. **INV-BANK-4**: Must wait tRAS minimum after ACTIVATE before PRECHARGE
5. **INV-BANK-5**: Must wait tWR after last write data before PRECHARGE
6. **INV-BANK-6**: Must wait tRTP after READ command before PRECHARGE
7. **INV-BANK-7**: Read/Write commands only valid in ACTIVE state
8. **INV-BANK-8**: PRECHARGE only valid when bank is ACTIVE (not during burst)
9. **INV-BANK-9**: Per-bank REFRESH only valid when target bank is IDLE

---

## Timing Parameters (LPDDR5-6400)

### Core Timing Parameters

| Parameter | Symbol | Cycles @ 3200 MHz | ns | Description |
|-----------|--------|-------------------|-----|-------------|
| Row Address to Column Address | tRCD | 14 | 14 | ACTIVATE to READ/WRITE |
| Row Precharge Time | tRP | 14 | 14 | PRECHARGE to ACTIVATE |
| Row Active Time (min) | tRAS | 28 | 28 | ACTIVATE to PRECHARGE (min) |
| Row Cycle Time | tRC | 42 | 42 | ACTIVATE to ACTIVATE (same bank) |
| CAS Latency (Read) | tCL | 14 | 14 | READ command to data |
| CAS Write Latency | tWL | 8 | 8 | WRITE command to data |
| Write Recovery | tWR | 24 | 24 | Last write data to PRECHARGE |
| Read to Precharge | tRTP | 6 | 6 | READ command to PRECHARGE |

### Bank Group Timing Parameters

| Parameter | Symbol | Cycles | ns | Description |
|-----------|--------|--------|-----|-------------|
| ACT to ACT (same bank group) | tRRD_L | 6 | 6 | ACTIVATE to ACTIVATE within bank group |
| ACT to ACT (diff bank group) | tRRD_S | 4 | 4 | ACTIVATE to ACTIVATE across bank groups |
| CAS to CAS (same bank group) | tCCD_L | 6 | 6 | Column command spacing within bank group |
| CAS to CAS (diff bank group) | tCCD_S | 4 | 4 | Column command spacing across bank groups |

### Turnaround Timing Parameters

| Parameter | Symbol | Cycles | ns | Description |
|-----------|--------|--------|-----|-------------|
| Write to Read (same bank group) | tWTR_L | 10 | 10 | Write to read turnaround within bank group |
| Write to Read (diff bank group) | tWTR_S | 4 | 4 | Write to read turnaround across bank groups |
| Read to Write | tRTW | 14 | 14 | Read to write turnaround (bus turnaround) |

### Burst Timing

| Parameter | Symbol | BL16 | BL32 | Description |
|-----------|--------|------|------|-------------|
| Burst cycles | tBurst | 8 | 16 | Data burst duration |
| Data transfer rate | - | 2/cycle | 2/cycle | DDR transfers per cycle |

### Refresh Timing (Per-Bank)

| Parameter | Symbol | Cycles | ns | Description |
|-----------|--------|--------|-----|-------------|
| Per-bank Refresh Cycle | tRFCpb | 140 | 140 | Single bank refresh time |
| All-bank Refresh Cycle | tRFCab | 280 | 280 | All banks refresh time |
| Refresh Interval (per-bank) | tREFIpb | 244 | 244 | Per-bank refresh interval (16× shorter) |
| Four Activate Window | tFAW | 24 | 24 | Max 4 ACTIVATEs in window |

### Timing Invariants

1. **INV-TIME-1**: tRC ≥ tRAS + tRP (row cycle includes active + precharge)
2. **INV-TIME-2**: tRCD ≤ tRAS (can't precharge before RCD completes)
3. **INV-TIME-3**: tCL > tRCD is valid (pipelining supported)
4. **INV-TIME-4**: At most 4 ACTIVATE commands in any tFAW window
5. **INV-TIME-5**: tRRD_L > tRRD_S (same bank group has longer constraint)
6. **INV-TIME-6**: tWTR_L > tWTR_S (same bank group has longer constraint)
7. **INV-TIME-7**: tCCD_L > tCCD_S (same bank group has longer constraint)
8. **INV-TIME-8**: Per-bank refresh: each bank needs refresh every 16 × tREFIpb

---

## Bus Architecture

### Command Bus

The command bus carries row/column addresses and control signals.

| Property | Value |
|----------|-------|
| Command Rate | 1 command per cycle (typical) |
| Command Width | CA[5:0] + CKE + CS |
| Pipelining | Yes - commands can be pipelined |

### Command Bus States

| State | Description | Valid Commands |
|-------|-------------|----------------|
| **IDLE** | No command pending | Any valid command |
| **BUSY** | Command being transmitted | None (wait) |

### Command Bus Invariants

1. **INV-CMD-1**: One command per cycle per channel
2. **INV-CMD-2**: Command bus is shared by all banks in channel
3. **INV-CMD-3**: Multi-cycle commands block bus for duration

### Data Bus

The data bus carries read/write data.

| Property | Single Channel | Dual Channel |
|----------|----------------|--------------|
| Width | 16 bits | 32 bits (16 per channel) |
| Transfer Rate | DDR (2 transfers/cycle) | DDR (2 transfers/cycle) |
| Bytes per cycle | 4 bytes | 8 bytes |

### Data Bus States

| State | Description | Duration |
|-------|-------------|----------|
| **IDLE** | No data transfer | - |
| **READ_BURST** | Data being read | tBurst cycles |
| **WRITE_BURST** | Data being written | tBurst cycles |
| **TURNAROUND** | Direction change | tRTW or tWTR cycles |

### Data Bus Invariants

1. **INV-DATA-1**: Data bus exclusive to one operation at a time
2. **INV-DATA-2**: Read bursts complete in tBurst cycles
3. **INV-DATA-3**: Write bursts complete in tBurst cycles
4. **INV-DATA-4**: Bus turnaround required between read and write
5. **INV-DATA-5**: Dual-channel: each channel has independent data bus

### Bus Contention Model

```
Time →
         0    5    10   15   20   25   30   35   40
         │    │    │    │    │    │    │    │    │
CMD Bus  ├ACT─┼────┼─RD─┼────┼─RD─┼────┼────┼────┤
         │    │    │    │    │    │    │    │    │
         │    │    │    tCCD │    │    │    │    │
         │    │    │◄───────►│    │    │    │    │
         │    │    │         │    │    │    │    │
DATA Bus ├────┼────┼────┼────┼─D0─┼─D1─┼────┼────┤
         │    │    │    │    │████│████│    │    │
         │    │    │    │    │read│read│    │    │
         │    │    │    │    │    │    │    │    │
                              tCL delay
```

---

## Per-Bank Refresh Model

LPDDR5 supports per-bank refresh, allowing other banks to remain accessible during refresh.

### Refresh Distribution

| Mode | Banks Affected | Duration | Interval |
|------|----------------|----------|----------|
| Per-bank (REFpb) | 1 bank | tRFCpb (140 cycles) | tREFIpb (244 cycles) |
| All-bank (REFab) | 16 banks | tRFCab (280 cycles) | tREFI (3900 cycles) |

### Per-Bank Refresh Scheduling

```
Bank:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15   0   1  ...
        │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
Cycle:  0  15  30  45  60  75  90 105 120 135 150 165 180 195 210 225 244 259 ...
        │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
        └REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF┴REF...
```

### Refresh Invariants

1. **INV-REF-1**: Each bank must be refreshed within 16 × tREFIpb cycles
2. **INV-REF-2**: Per-bank refresh blocks only target bank for tRFCpb
3. **INV-REF-3**: Bank must be IDLE before refresh command
4. **INV-REF-4**: Interleaved refresh allows other banks to service requests
5. **INV-REF-5**: Refresh takes priority over normal commands when deadline approaches

---

## Command Truth Table

| Command | Bank State Required | Bank Group Check | Timing Constraints |
|---------|---------------------|------------------|-------------------|
| ACTIVATE | IDLE | tRRD_L (same BG), tRRD_S (diff BG) | tRC (same bank), tFAW window |
| READ | ACTIVE | tCCD_L (same BG), tCCD_S (diff BG) | tRCD after ACT, tWTR after WR |
| WRITE | ACTIVE | tCCD_L (same BG), tCCD_S (diff BG) | tRCD after ACT, tRTW after RD |
| PRECHARGE | ACTIVE | - | tRAS after ACT, tWR after WR, tRTP after RD |
| REFpb | IDLE | - | tRFCpb, target bank only |
| REFab | IDLE (all) | - | tRFCab, all banks |

---

## Access Pattern Latency Summary

### Page Hit (Row Buffer Hit)
- Row already open in bank
- Only need CAS latency (tCL for read, tWL for write)
- **Read latency: tCL = 14 cycles**
- **Write latency: tWL = 8 cycles**

### Page Empty (Closed Page)
- Bank is idle, no row open
- Need ACTIVATE + CAS
- **Read latency: tRCD + tCL = 14 + 14 = 28 cycles**
- **Write latency: tRCD + tWL = 14 + 8 = 22 cycles**

### Page Conflict (Row Buffer Miss)
- Different row open in bank
- Need PRECHARGE + ACTIVATE + CAS
- **Read latency: tRP + tRCD + tCL = 14 + 14 + 14 = 42 cycles**
- **Write latency: tRP + tRCD + tWL = 14 + 14 + 8 = 36 cycles**

---

## Burst Transfer Calculation

### 4KB Tile Transfer - Single Channel (×16, BL16)

| Component | Value | Calculation |
|-----------|-------|-------------|
| Bytes per burst | 32 | 16 beats × 2 bytes |
| Bursts required | 128 | 4096 / 32 |
| Burst cycles | 1024 | 128 × 8 cycles |
| First data (page empty) | tRCD + tCL = 28 | Latency to first data |
| **Total cycles** | **1052** | 28 + 1024 |

### 4KB Tile Transfer - Dual Channel (×32, BL16)

| Component | Value | Calculation |
|-----------|-------|-------------|
| Bytes per burst | 64 | 16 beats × 4 bytes |
| Bursts required | 64 | 4096 / 64 |
| Burst cycles | 512 | 64 × 8 cycles |
| First data (page empty) | tRCD + tCL = 28 | Latency to first data |
| **Total cycles** | **540** | 28 + 512 |

### 4KB Tile Transfer - Dual Channel (×32, BL32)

| Component | Value | Calculation |
|-----------|-------|-------------|
| Bytes per burst | 128 | 32 beats × 4 bytes |
| Bursts required | 32 | 4096 / 128 |
| Burst cycles | 512 | 32 × 16 cycles |
| First data (page empty) | tRCD + tCL = 28 | Latency to first data |
| **Total cycles** | **540** | 28 + 512 |

**Note**: BL32 halves command overhead but same total data cycles.

---

## Memory Controller State Tracking

### Per-Bank State

```cpp
struct BankState {
    enum State {
        IDLE,
        ACTIVATING,
        ACTIVE,
        READING,
        WRITING,
        PRECHARGING,
        REFRESHING
    };

    State state;
    uint32_t open_row;           // Valid when state == ACTIVE
    uint64_t state_until;        // Cycle when current state completes
    uint64_t last_activate;      // For tRAS, tRC tracking
    uint64_t last_read_cmd;      // For tRTP tracking (command issue time)
    uint64_t last_write_cmd;     // For tWR tracking (command issue time)
    uint64_t last_refresh;       // For per-bank refresh tracking
    uint8_t bank_group;          // Which bank group this bank belongs to
};
```

### Per-Bank-Group State

```cpp
struct BankGroupState {
    uint64_t last_activate;      // For tRRD_L tracking
    uint64_t last_cas;           // For tCCD_L tracking
    uint64_t last_write;         // For tWTR_L tracking
};
```

### Per-Channel State

```cpp
struct ChannelState {
    enum BusState { IDLE, READ_BURST, WRITE_BURST, TURNAROUND };

    // Command bus
    bool cmd_bus_busy;
    uint64_t cmd_bus_until;

    // Data bus
    BusState data_bus_state;
    uint64_t data_bus_until;
    bool last_was_write;         // For read-to-write turnaround

    // Refresh tracking
    uint64_t activate_window[4]; // Circular buffer for tFAW
    uint8_t activate_count;
    uint8_t next_refresh_bank;   // Round-robin per-bank refresh
    uint64_t last_refresh[16];   // Per-bank refresh tracking
};
```

### Per-Device State

```cpp
struct DeviceState {
    enum State { IDLE, ACTIVE, SELF_REFRESH, POWER_DOWN };

    State state;
    std::array<ChannelState, 2> channels;  // Support dual-channel
    uint8_t num_channels;                   // 1 or 2
    BurstLength burst_length;               // BL16 or BL32
};
```

---

## Invariant Verification Functions

```cpp
// Bank state invariants
bool can_activate(const BankState& bank, uint64_t cycle) {
    return bank.state == IDLE;
}

bool can_read(const BankState& bank, uint64_t cycle) {
    return bank.state == ACTIVE && cycle >= bank.state_until;
}

bool can_write(const BankState& bank, uint64_t cycle) {
    return bank.state == ACTIVE && cycle >= bank.state_until;
}

bool can_precharge(const BankState& bank, uint64_t cycle) {
    if (bank.state != ACTIVE) return false;
    if (cycle < bank.last_activate + tRAS) return false;
    if (cycle < bank.last_write_cmd + tWL + tBurst + tWR) return false;
    if (cycle < bank.last_read_cmd + tRTP) return false;
    return true;
}

bool can_refresh(const BankState& bank) {
    return bank.state == IDLE;
}

// Bank group timing
uint64_t next_activate_same_bg(const BankGroupState& bg, uint64_t cycle) {
    return std::max(cycle, bg.last_activate + tRRD_L);
}

uint64_t next_activate_diff_bg(uint64_t cycle) {
    return cycle + tRRD_S;  // Can issue after tRRD_S
}

// Turnaround timing
uint64_t next_read_after_write_same_bg(const BankGroupState& bg, uint64_t cycle) {
    return std::max(cycle, bg.last_write + tWL + tBurst + tWTR_L);
}

uint64_t next_write_after_read(const ChannelState& ch, uint64_t cycle) {
    if (!ch.last_was_write) {
        return std::max(cycle, ch.data_bus_until + tRTW);
    }
    return cycle;
}
```

---

## Summary of Modeling Requirements

| Feature | Modeled | Notes |
|---------|---------|-------|
| 16 banks per channel | ✓ | 4 bank groups × 4 banks |
| Dual-channel support | ✓ | Independent channels, combined bandwidth |
| Bank group timing (tRRD_L/S, tCCD_L/S) | ✓ | Different constraints within/across groups |
| Write-to-read turnaround (tWTR_L/S) | ✓ | Bank group dependent |
| Read-to-write turnaround (tRTW) | ✓ | Bus turnaround time |
| Per-bank refresh | ✓ | tRFCpb, interleaved refresh |
| Separate command bus | ✓ | One command per cycle |
| Separate data bus | ✓ | Burst states, turnaround tracking |
| BL16 support | ✓ | 32 bytes per burst (×16) |
| BL32 support | ✓ | 64 bytes per burst (×16) |
| tFAW tracking | ✓ | Max 4 activates in window |
