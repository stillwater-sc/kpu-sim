# NoC Router Architecture Analysis

## Current Implementation Issues

The Chrome Trace shows impossible behavior: Router R[0,0] appears to concurrently inject A[0,0], A[0,1], A[0,2], and A[0,3] - four 256KB tiles (1MB total). This document analyzes why and proposes the correct model.

---

## Current (Flawed) Router Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           NoCRouter[0,0]                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      INPUT PORTS (6 total)                       │   │
│  │                                                                  │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │   │  NORTH   │  │  SOUTH   │  │  EAST    │  │  WEST    │        │   │
│  │   │ queue<>  │  │ queue<>  │  │ queue<>  │  │ queue<>  │        │   │
│  │   │ cap: 8   │  │ cap: 8   │  │ cap: 8   │  │ cap: 8   │        │   │
│  │   │ packets! │  │ packets! │  │ packets! │  │ packets! │        │   │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘        │   │
│  │                                                                  │   │
│  │   ┌──────────┐  ┌──────────┐                                    │   │
│  │   │  LOCAL   │  │   DMA    │  ← L3 Cache injects here           │   │
│  │   │ queue<>  │  │ queue<>  │                                    │   │
│  │   │ cap: 8   │  │ cap: 8   │                                    │   │
│  │   │ packets! │  │ packets! │  ← BUG: This is 8 PACKETS,         │   │
│  │   └──────────┘  └──────────┘       not 8 FLITs (512 bytes)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       ROUTING LOGIC                              │   │
│  │                                                                  │   │
│  │   for each input port with packet:                              │   │
│  │       out_dir = xy_route(pkt.dst_router)                        │   │
│  │       if output_port[out_dir].can_send():                       │   │
│  │           move packet to output_port[out_dir].buffer            │   │
│  │                                                                  │   │
│  │   ← BUG: Moves ENTIRE 256KB packet in ONE cycle                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      OUTPUT PORTS (6 total)                      │   │
│  │                                                                  │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │   │  NORTH   │  │  SOUTH   │  │  EAST    │  │  WEST    │        │   │
│  │   │  →Link   │  │  →Link   │  │  →Link   │  │  →Link   │        │   │
│  │   │ cap: 4   │  │ cap: 4   │  │ cap: 4   │  │ cap: 4   │        │   │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘        │   │
│  │                                                                  │   │
│  │   ┌──────────┐  ┌──────────┐                                    │   │
│  │   │  LOCAL   │  │   DMA    │  → To L3 Cache (ejection)          │   │
│  │   │  →L3     │  │  →DMA    │                                    │   │
│  │   └──────────┘  └──────────┘                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Problems with Current Model

1. **Packet-level queues, not flit-level**
   - `std::queue<NoCPacket>` stores entire packets
   - Each packet can be 256KB (4096 flits)
   - Buffer capacity `input_buffer_flits = 8` is ignored - we can queue 8 PACKETS

2. **Instant packet transfer**
   - `inject_packet()` immediately places packet in queue
   - No modeling of time to serialize flits onto the link

3. **No injection bandwidth limit**
   - Config has `injection_bandwidth = 64` (bytes/cycle)
   - But code ignores this - any number of packets can be injected

4. **Concurrent transfers on same resource**
   - Multiple packets in same queue appear as concurrent activity
   - In reality, only one packet can use a link at a time

---

## Correct Router Model (Store-and-Forward)

For a 256KB tile with 64B flits:
- Tile size: 256KB = 262,144 bytes
- Flit size: 64 bytes
- Flits per tile: 4,096 flits
- Injection time at 64 B/cycle: **4,096 cycles**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     NoCRouter[0,0] (Correct Model)                      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      INPUT PORTS                                 │   │
│  │                                                                  │   │
│  │   ┌────────────────────────────────────────────────────┐        │   │
│  │   │                    LOCAL PORT                       │        │   │
│  │   │                                                     │        │   │
│  │   │  ┌─────────────────────────────────────────────┐   │        │   │
│  │   │  │           FLIT BUFFER (8 FLITs = 512B)       │   │        │   │
│  │   │  │  ┌────┬────┬────┬────┬────┬────┬────┬────┐  │   │        │   │
│  │   │  │  │ F0 │ F1 │ F2 │ F3 │ F4 │ F5 │ F6 │ F7 │  │   │        │   │
│  │   │  │  │64B │64B │64B │64B │64B │64B │64B │64B │  │   │        │   │
│  │   │  │  └────┴────┴────┴────┴────┴────┴────┴────┘  │   │        │   │
│  │   │  └─────────────────────────────────────────────┘   │        │   │
│  │   │                                                     │        │   │
│  │   │  State machine per packet:                          │        │   │
│  │   │  ┌────────────┐                                    │        │   │
│  │   │  │ current_pkt │→ NoCPacket* (being received)      │        │   │
│  │   │  │ flits_rcvd  │→ 0..4096 (progress counter)       │        │   │
│  │   │  │ injection_  │→ When started (for busy tracking) │        │   │
│  │   │  │ start_cycle │                                   │        │   │
│  │   │  └────────────┘                                    │        │   │
│  │   │                                                     │        │   │
│  │   │  Injection rate: 1 flit/cycle (64 B/cycle)         │        │   │
│  │   │  A 256KB tile takes 4096 cycles to fully inject    │        │   │
│  │   └────────────────────────────────────────────────────┘        │   │
│  │                                                                  │   │
│  │   Similar for NORTH, SOUTH, EAST, WEST ports                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CROSSBAR SWITCH                               │   │
│  │                                                                  │   │
│  │   Per-cycle arbitration:                                        │   │
│  │   - Each output port can accept 1 flit from one input          │   │
│  │   - Round-robin priority among competing inputs                 │   │
│  │   - Wormhole: once head flit wins, path is reserved            │   │
│  │                                                                  │   │
│  │      IN[N]──┐                                                   │   │
│  │      IN[S]──┼──►┌─────┐                                        │   │
│  │      IN[E]──┼───│ARBTR│──►OUT[N]                               │   │
│  │      IN[W]──┼───│     │──►OUT[S]                               │   │
│  │      IN[L]──┼──►│     │──►OUT[E]                               │   │
│  │      IN[D]──┘   └─────┘──►OUT[W]                               │   │
│  │                        ──►OUT[L]                               │   │
│  │                        ──►OUT[D]                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     OUTPUT PORTS                                 │   │
│  │                                                                  │   │
│  │   ┌────────────────────────────────────────────────────┐        │   │
│  │   │                    EAST PORT                        │        │   │
│  │   │                                                     │        │   │
│  │   │  ┌─────────────────────────────────────────────┐   │        │   │
│  │   │  │           FLIT BUFFER (4 FLITs = 256B)       │   │        │   │
│  │   │  │  ┌────┬────┬────┬────┐                      │   │        │   │
│  │   │  │  │ F0 │ F1 │ F2 │ F3 │                      │   │        │   │
│  │   │  │  └────┴────┴────┴────┘                      │   │        │   │
│  │   │  └─────────────────────────────────────────────┘   │        │   │
│  │   │                                                     │        │   │
│  │   │  ──────────────► NoCLink ──────────────►           │        │   │
│  │   │                                                     │        │   │
│  │   │  Link sends 1 flit/cycle (64 B/cycle = 32 GB/s)   │        │   │
│  │   │  Link latency: 1 cycle                             │        │   │
│  │   └────────────────────────────────────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Resource Serialization Timeline

With correct modeling, injecting 4 tiles from R[0,0]:

```
                     Cycles
             0    4096   8192   12288  16384
             │     │      │       │      │
A[0,0] ──────████████████████████████████████─────► Inject 4096 cycles
                                                    Link 4096 cycles

A[0,1] ─────────────████████████████████████████─► Starts at cycle 4096
                    (must wait for A[0,0])

A[0,2] ───────────────────████████████████████─► Starts at cycle 8192
                          (must wait for A[0,1])

A[0,3] ─────────────────────────████████████──► Starts at cycle 12288
                                (must wait for A[0,2])

CORRECT: Sequential injection, one tile at a time per port
```

---

## Key Data Structures That Need Fixing

### Current (Wrong)

```cpp
struct NoCInputPort {
    std::queue<NoCPacket> buffer;     // Queue of FULL PACKETS
    uint32_t buffer_capacity = 8;      // Treated as packet count
};
```

### Correct

```cpp
struct NoCInputPort {
    // Flit-level buffer (actual storage)
    std::array<uint8_t, 512> flit_buffer;  // 8 flits × 64B = 512B physical
    size_t flits_buffered = 0;

    // Current packet being assembled
    NoCPacket* current_packet = nullptr;
    size_t flits_received_for_packet = 0;

    // Completed packets ready for routing
    std::queue<NoCPacket*> ready_queue;  // Only packets with all flits received

    // Injection state (LOCAL port only)
    bool injection_in_progress = false;
    uint64_t injection_start_cycle = 0;

    // Rate limit: 1 flit/cycle
    uint64_t last_flit_received_cycle = 0;

    bool can_accept_flit(uint64_t cycle) const {
        return flits_buffered < 8 && cycle > last_flit_received_cycle;
    }
};
```

---

## Bandwidth Reality Check

| Resource | Bandwidth | Time for 256KB tile |
|----------|-----------|---------------------|
| LOCAL inject | 64 B/cycle | 4,096 cycles |
| Link | 64 B/cycle | 4,096 cycles |
| L3→L3 (3 hops) | - | 3 × 4,096 + routing = ~12,300 cycles |

At 500 MHz clock:
- 4,096 cycles = 8.2 µs per tile
- 1 tile/8.2 µs = 122K tiles/sec
- 122K × 256KB = **31 GB/s** per link

**This matches the design intent: each link is ~32 GB/s.**

---

## Implementation Options

### Option 1: Full Flit-Level Simulation (Most Accurate)

- Model every flit movement
- Track buffer occupancy precisely
- Accurate contention and backpressure
- Cost: Higher simulation overhead

### Option 2: Packet-Level with Time Tracking (Recommended)

```cpp
struct PacketTransferState {
    NoCPacket* packet;
    uint64_t transfer_start_cycle;
    uint64_t transfer_end_cycle;    // start + (num_flits * cycles_per_flit)
    uint64_t flits_transferred;

    bool is_complete(uint64_t cycle) const {
        return cycle >= transfer_end_cycle;
    }

    float progress(uint64_t cycle) const {
        if (cycle >= transfer_end_cycle) return 1.0;
        return float(cycle - transfer_start_cycle) /
               float(transfer_end_cycle - transfer_start_cycle);
    }
};

struct NoCInputPort {
    // Only ONE packet can be injecting at a time
    std::optional<PacketTransferState> active_injection;

    // Completed packets waiting to be routed
    std::queue<NoCPacket*> completed_queue;

    bool can_start_injection(uint64_t cycle) const {
        return !active_injection.has_value() ||
               active_injection->is_complete(cycle);
    }

    void start_injection(NoCPacket* pkt, uint64_t cycle) {
        active_injection = PacketTransferState{
            .packet = pkt,
            .transfer_start_cycle = cycle,
            .transfer_end_cycle = cycle + pkt->num_flits,
            .flits_transferred = 0
        };
    }

    void step(uint64_t cycle) {
        if (active_injection && active_injection->is_complete(cycle)) {
            completed_queue.push(active_injection->packet);
            active_injection.reset();
        }
    }
};
```

---

## Summary

The current NoC implementation has a fundamental modeling error:
- **Claims**: 8 flit buffer capacity (512 bytes)
- **Reality**: Stores 8 full packets (up to 2MB)

This creates the illusion that multiple tiles can be injected simultaneously, when in reality:
- Each router port can only inject/transmit at 64 B/cycle
- A 256KB tile takes 4,096 cycles to transfer
- Resources must be serialized, not parallelized

The fix requires tracking transfer progress per port and only allowing one active transfer at a time on each link/port.
