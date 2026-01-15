# Wormhole Router Resource Management Algorithm

This document describes the resource management algorithm implemented in the KPU simulator's wormhole NoC (Network-on-Chip) router.

## Table of Contents

1. [Overview](#overview)
2. [Flit-Based Transmission](#flit-based-transmission)
3. [Buffer Architecture](#buffer-architecture)
4. [Credit-Based Flow Control](#credit-based-flow-control)
5. [Path Reservation Protocol](#path-reservation-protocol)
6. [Arbitration Algorithm](#arbitration-algorithm)
7. [Pipeline Execution Model](#pipeline-execution-model)
8. [Backpressure Mechanism](#backpressure-mechanism)
9. [NoC Step Execution](#noc-step-execution)

---

## Overview

The wormhole router implements a **flit-pipelined wormhole switching** protocol for transferring data tiles between L3 caches in a 2D mesh NoC. The key characteristics are:

- **Flit-level granularity**: Packets are divided into 64-byte flits
- **Path reservation**: HEAD flit reserves output ports, TAIL flit releases them
- **Credit-based flow control**: Prevents buffer overflow through explicit credit tracking
- **Single-hop nearest-neighbor routing**: Each router forwards to adjacent routers only
- **Simultaneous independent link activity**: Multiple links can be active concurrently

### Design Goals

1. **Accurate bandwidth modeling**: Proper serialization at flit granularity
2. **Deadlock-free operation**: Credit flow control ensures no circular dependencies
3. **Contention modeling**: Arbitration captures realistic blocking behavior
4. **Low latency**: Cut-through switching allows flits to flow before entire packet arrives

---

## Flit-Based Transmission

### Flit Types

Packets are divided into **flits** (flow control units) of 64 bytes each:

```
┌─────────────────────────────────────────────────────────────────┐
│                          PACKET                                  │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────┤
│  HEAD   │  BODY   │  BODY   │  BODY   │  ...    │  BODY   │TAIL │
│ (routing│  (data) │  (data) │  (data) │         │  (data) │     │
│  info)  │         │         │         │         │         │     │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────┘
```

| Flit Type | Purpose |
|-----------|---------|
| `HEAD` | First flit; contains routing info (src, dst, tile descriptor) |
| `BODY` | Middle flits; data payload |
| `TAIL` | Last flit; signals end of packet, releases path reservation |
| `HEAD_TAIL` | Single-flit packet (for small tiles ≤64 bytes) |

### Flit Structure

```cpp
struct Flit {
    FlitType type;           // HEAD, BODY, TAIL, HEAD_TAIL
    uint16_t packet_seq;     // Unique packet identifier
    uint8_t src_router;      // Source router ID
    uint8_t dst_router;      // Destination router ID
    uint16_t total_flits;    // Total flits in this packet
    TileDescriptor tile;     // Tile metadata (HEAD only)
    uint64_t inject_cycle;   // Injection timestamp
};
```

### Flit Calculation

For a tile of size `S` bytes:
```
num_flits = ceil(S / 64)
```

Example: A 256KB tile (262,144 bytes) requires 4,096 flits.

---

## Buffer Architecture

### Input Buffers

Each input port has a **circular buffer** holding up to 8 flits (512 bytes):

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT PORT                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │   FlitBuffer (capacity = 8 flits)                        │    │
│  │   ┌────┬────┬────┬────┬────┬────┬────┬────┐              │    │
│  │   │ f0 │ f1 │ f2 │ f3 │ f4 │ f5 │ f6 │ f7 │              │    │
│  │   └────┴────┴────┴────┴────┴────┴────┴────┘              │    │
│  │        ↑                             ↑                    │    │
│  │       head                          tail                  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  State:                                                           │
│   - active_packet_: Currently receiving packet ID                 │
│   - credits_to_return_: Count of consumed flits                   │
└──────────────────────────────────────────────────────────────────┘
```

### Output Buffers

Each output port has a smaller buffer (4 flits, 256 bytes) plus path reservation state:

```
┌──────────────────────────────────────────────────────────────────┐
│                       OUTPUT PORT                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │   FlitBuffer (capacity = 4 flits)                        │    │
│  │   ┌────┬────┬────┬────┐                                  │    │
│  │   │ f0 │ f1 │ f2 │ f3 │                                  │    │
│  │   └────┴────┴────┴────┘                                  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  PathReservation:                                                 │
│   - state: IDLE | RESERVED                                        │
│   - input_port: Which input port reserved this output             │
│   - packet_seq: ID of reserving packet                            │
│   - flits_remaining: Countdown to release                         │
│                                                                   │
│  Credits:                                                         │
│   - credits_: Available space in downstream input buffer          │
│   - pending_flit_: Flit ready for link transmission               │
└──────────────────────────────────────────────────────────────────┘
```

### Port Layout Per Router

```
                    NORTH Input/Output
                          ↑↓
                    ┌─────────────┐
     WEST ←→        │   Router    │        ←→ EAST
                    │   Crossbar  │
                    └─────────────┘
                          ↑↓
                    SOUTH Input/Output
                          ↑↓
                    LOCAL (L3 Cache)
                          ↑↓
                    DMA (edge routers only)
```

---

## Credit-Based Flow Control

Credit-based flow control prevents buffer overflow by tracking available buffer space at downstream routers.

### Credit Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CREDIT FLOW                                        │
│                                                                              │
│   Router A                          Link                          Router B   │
│   ─────────                         ────                          ─────────  │
│                                                                              │
│   OutputPort                                                    InputPort    │
│   ┌────────────┐                                               ┌──────────┐ │
│   │ credits: 8 │ ◄──────────────── credit ◄─────────────────── │ consumed │ │
│   │            │                                               │          │ │
│   │            │ ─────────────────  flit  ─────────────────► │ received │ │
│   │ credits: 7 │                                               │          │ │
│   └────────────┘                                               └──────────┘ │
│                                                                              │
│   CRITICAL: Credits are returned when flits are CONSUMED (switched to       │
│   output), NOT when they are received into the input buffer.                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Credit Rules

1. **Initial credits**: Output port starts with credits = downstream input buffer capacity (8)
2. **Sending consumes credit**: Each flit sent decrements credits by 1
3. **Credits returned on consumption**: When a flit is switched from input buffer to output buffer, a credit is queued for return
4. **Credit return timing**: Credits are returned to upstream at end of each cycle

### Why Credits Return on Consume, Not Receive

This is a critical design decision that prevents deadlock:

```
WRONG (credits on receive):
┌─────────────────────────────────────────────────────────────────────────────┐
│  If credits returned immediately on receive:                                 │
│                                                                              │
│  Router A sends flit → Router B receives (credit returned) →                 │
│  Router A sends another flit → Router B buffer blocked waiting for output → │
│  Credits exhausted but flits stuck in buffer → Buffer overflow possible!     │
└─────────────────────────────────────────────────────────────────────────────┘

CORRECT (credits on consume):
┌─────────────────────────────────────────────────────────────────────────────┐
│  If credits returned when flit leaves input buffer:                          │
│                                                                              │
│  Router A sends flit → Router B receives (no credit yet) →                   │
│  Router B switches flit to output → credit returned →                        │
│  Credits accurately reflect AVAILABLE buffer space                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Credit Tracking Implementation

```cpp
// In InputPort::consume_flit() - called when flit is switched to output
Flit InputPort::consume_flit() {
    auto flit = buffer_.pop();
    if (flit) {
        credits_to_return_++;  // Mark for credit return
        return *flit;
    }
    return Flit{};
}

// In WormholeNoC::return_credits_to_upstream() - called each cycle
void WormholeNoC::return_credits_to_upstream(uint64_t cycle) {
    for (each router) {
        for (each input port N,S,E,W) {
            uint32_t credits = in_port.credits_to_return();
            if (credits > 0) {
                // Return credits to upstream router's output port
                upstream_router.output(out_dir).receive_credit();
                in_port.clear_returned_credits();
            }
        }
    }
}
```

---

## Path Reservation Protocol

Wormhole routing uses **path reservation** to ensure a packet's flits traverse the network in order without interleaving.

### Reservation States

```
┌─────────────────────────────────────────────────────────────────┐
│                     PATH RESERVATION FSM                         │
│                                                                  │
│                      ┌───────────┐                               │
│       HEAD flit      │           │       TAIL flit               │
│       reserves   ──► │  RESERVED │ ───►  releases                │
│                      │           │                               │
│                      └─────┬─────┘                               │
│                            │                                     │
│                            │ TAIL                                │
│                            ▼                                     │
│                      ┌───────────┐                               │
│                      │           │                               │
│                      │   IDLE    │ ◄─── initial state            │
│                      │           │                               │
│                      └───────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### Reservation Data Structure

```cpp
struct PathReservation {
    State state;              // IDLE or RESERVED
    PortDir input_port;       // Which input port holds the reservation
    uint16_t packet_seq;      // Packet ID for verification
    uint16_t flits_remaining; // Countdown to release

    void reserve(PortDir in_port, uint16_t seq, uint16_t total_flits);
    void flit_passed();       // Decrement flits_remaining
    void release();           // Called when TAIL passes
};
```

### Reservation Enforcement

When an output port is reserved:
- Only flits from the reserving input port can use it
- Other HEAD flits requesting this output are blocked (arbitration conflict)
- BODY and TAIL flits from the same packet flow through without arbitration

---

## Arbitration Algorithm

The arbitration algorithm determines which input port can use each output port.

### Two-Phase Arbitration

```
Phase 1: Continue existing reservations (BODY/TAIL flits)
──────────────────────────────────────────────────────────
For each output port with active reservation:
  1. Check if reserving input port has a flit
  2. Verify flit belongs to reserved packet (packet_seq match)
  3. Switch flit from input to output buffer
  4. If TAIL flit, release reservation after switching

Phase 2: Arbitrate new HEAD flits
─────────────────────────────────
For each input port with HEAD flit:
  1. Determine required output port via routing
  2. Check if output port is available:
     - Has buffer space (can_accept)
     - Not already reserved
  3. If available:
     - Reserve the path
     - Switch HEAD flit to output
  4. If blocked:
     - Increment arbitration_conflicts counter
     - HEAD flit waits until next cycle
```

### Priority Scheme

Input ports are processed in fixed order: NORTH, SOUTH, EAST, WEST, LOCAL, DMA. This implicit priority prevents starvation (all ports eventually get service) but can affect fairness under high contention.

### Arbitration Implementation

```cpp
void WormholeRouter::arbitrate_and_switch(uint64_t cycle) {
    // Phase 1: Continue existing reservations
    for (each output port) {
        if (output.reservation().is_reserved()) {
            PortDir in_dir = output.reservation().input_port;
            if (input[in_dir].has_flit()) {
                Flit* flit = input[in_dir].peek_flit();
                if (flit->packet_seq == output.reservation().packet_seq) {
                    Flit f = input[in_dir].consume_flit();
                    output.accept_flit(f, cycle);
                }
            }
        }
    }

    // Phase 2: Arbitrate new HEAD flits
    for (each input port) {
        if (!input.has_flit()) continue;
        Flit* flit = input.peek_flit();
        if (!flit->is_head()) continue;

        PortDir out_dir = route(flit->dst_router);
        if (!output[out_dir].can_accept()) continue;
        if (output[out_dir].reservation().is_reserved()) continue;

        // Reserve and switch
        output[out_dir].reservation().reserve(in_dir, flit->packet_seq, flit->total_flits);
        Flit f = input.consume_flit();
        output[out_dir].accept_flit(f, cycle);
    }
}
```

---

## Pipeline Execution Model

Each router executes a 3-phase pipeline per cycle:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ROUTER PIPELINE (per cycle)                           │
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │  Phase 1:        │    │  Phase 2:        │    │  Phase 3:        │       │
│  │  INJECTION       │ ──►│  ARBITRATION &   │ ──►│  OUTPUT          │       │
│  │                  │    │  SWITCHING       │    │  TRANSMISSION    │       │
│  │  L3/DMA → Input  │    │  Input → Output  │    │  Output → Link   │       │
│  │  Buffer          │    │  Buffer          │    │                  │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│                                                                              │
│  Cycle N:     ┌────┐     ┌────┐     ┌────┐                                  │
│  Flit A:      │ P1 │ ──► │ P2 │ ──► │ P3 │ ──► Link                         │
│               └────┘     └────┘     └────┘                                  │
│                                                                              │
│  Cycle N+1:              ┌────┐     ┌────┐                                  │
│  Flit A at next router:  │ P1 │ ──► │ P2 │ ──► ...                          │
│  (received)              └────┘     └────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase Details

**Phase 1: Injection (`process_injections`)**
- LOCAL port: L3 cache injects flits into LOCAL input buffer
- DMA port: External DMA injects flits into DMA input buffer
- One flit per injection source per cycle (if buffer space available)

**Phase 2: Arbitration & Switching (`arbitrate_and_switch`)**
- Process existing path reservations (BODY/TAIL flits)
- Arbitrate new HEAD flits for idle output ports
- Transfer flits from input buffers to output buffers

**Phase 3: Output Transmission (`process_outputs`)**
- Each output port attempts to send one flit to link
- Requires: output buffer has flit AND credits > 0
- Flit placed in `pending_flit_` for NoC-level link transfer

---

## Backpressure Mechanism

Backpressure propagates through the network when a downstream router is congested.

### Backpressure Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKPRESSURE EXAMPLE                                │
│                                                                              │
│  Router A ───────────► Router B ───────────► Router C (congested)            │
│                                                                              │
│  Step 1: Router C's LOCAL output is busy (compute not consuming)             │
│          └── C's input buffer fills up                                       │
│              └── C stops returning credits to B                              │
│                                                                              │
│  Step 2: B's output to C has 0 credits                                       │
│          └── B's output buffer fills up                                      │
│              └── B can't switch flits from input to output                   │
│                  └── B's input buffer fills up                               │
│                      └── B stops returning credits to A                      │
│                                                                              │
│  Step 3: A's output to B has 0 credits                                       │
│          └── A's output stalls                                               │
│              └── Backpressure fully propagated!                              │
│                                                                              │
│  Recovery: When C consumes flits, credits flow back, pressure releases       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Credit Stall Detection

Output ports track credit stalls for performance analysis:

```cpp
void OutputPort::step(uint64_t cycle) {
    if (credits_ == 0) {
        stats_.credit_stall_cycles++;  // Track backpressure
        return;  // Can't send
    }
    // ... send flit
}
```

---

## NoC Step Execution

The NoC orchestrates all routers and manages inter-router flit transfers.

### Cycle Execution Order

```cpp
void WormholeNoC::step(uint64_t cycle) {
    // Step 1: All routers process internally
    for (auto& router : routers_) {
        router.step(cycle);  // Injection, Arbitration, Output
    }

    // Step 2: Transfer flits between routers
    for (each router) {
        for (each direction N,S,E,W) {
            if (output.has_flit_for_link()) {
                flit = output.take_pending_flit();
                neighbor.input(opposite_dir).receive_flit(flit);
            }
        }
    }

    // Step 2b: Return credits to upstream
    return_credits_to_upstream(cycle);

    // Step 3: Check for completed deliveries
    check_deliveries(cycle);
}
```

### Timing Model

| Operation | Latency |
|-----------|---------|
| Flit injection | 1 cycle |
| Router traversal | 1 cycle (pipelined) |
| Link transfer | 1 cycle |
| Flit ejection | 1 cycle |

**Total single-hop latency**: ~3 cycles for HEAD flit (injection + switch + ejection)

**Packet transfer time**: For N flits, total time ≈ 3 + (N-1) cycles (pipelined)

### Delivery Tracking

The NoC tracks active transfers and invokes callbacks on completion:

```cpp
struct ActiveTransfer {
    uint16_t packet_seq;
    uint8_t src_router, dst_router;
    TileDescriptor tile;
    uint64_t inject_start_cycle;
    uint16_t total_flits;
    uint16_t flits_delivered;
};

// On last flit delivery:
if (transfer.flits_delivered >= transfer.total_flits) {
    latency = cycle - transfer.inject_start_cycle;
    delivery_callback(tile, src_router, cycle);
}
```

---

## Summary

The wormhole router resource management algorithm provides:

1. **Efficient bandwidth utilization** through flit-level pipelining
2. **Deadlock-free operation** via credit-based flow control with proper credit timing
3. **Fair access** through path reservation with deterministic arbitration
4. **Accurate contention modeling** with statistics on conflicts and stalls
5. **Low latency** by allowing cut-through switching

Key invariants maintained:
- Output credits ≤ downstream input buffer capacity
- At most one packet can reserve an output port at a time
- Flits of a packet traverse the path in order
- No flit is ever dropped (credits prevent overflow)

---

## References

- `include/sw/kpu/noc/wormhole_router.hpp` - Header definitions
- `src/noc/wormhole_router.cpp` - Implementation
- Dally, W.J. and Towles, B. "Principles and Practices of Interconnection Networks" (2003)
