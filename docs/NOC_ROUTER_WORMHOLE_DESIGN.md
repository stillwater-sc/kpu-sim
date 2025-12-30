# NoC Router Design: Wormhole Routing with Nearest-Neighbor Communication

## Design Requirements (from review)

1. **Push-based data movement**: Source L3 knows destination, writes data
2. **Large payloads**: Vectors/matrices in tens to hundreds of KB
3. **Minimal router storage**: L3 tiles are the storage, not routers
4. **Wormhole routing**: Flits flow through, head reserves path
5. **Nearest-neighbor communication**: Single-hop transfers only
6. **All links active simultaneously**: East and South links independent
7. **Credit-based flow control**: 8 flit buffers, backpressure via credits
8. **No virtual channels**: XY routing is deadlock-free
9. **64-byte flits**: Matches cache line size
10. **Accurate DMA modeling**: Edge routers only

---

## Wormhole vs Store-and-Forward

```
STORE-AND-FORWARD (rejected):
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Router0 │───►│ Router1 │───►│ Router2 │
│         │    │         │    │         │
│ Buffer  │    │ Buffer  │    │ Buffer  │
│ 256KB   │    │ 256KB   │    │ 256KB   │  ← Requires huge buffers!
└─────────┘    └─────────┘    └─────────┘

WORMHOLE (selected):
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Router0 │───►│ Router1 │───►│ Router2 │
│         │    │         │    │         │
│ Buffer  │    │ Buffer  │    │ Buffer  │
│ 512B    │    │ 512B    │    │ 512B    │  ← Tiny buffers OK!
└─────────┘    └─────────┘    └─────────┘
     │              │              │
     └──────────────┴──────────────┘
     Head flit reserves entire path
     Body/tail flits follow like a "worm"
```

---

## Nearest-Neighbor Communication Model

For systolic dataflow, tiles move in predictable patterns:
- **A tiles**: Flow EAST (row broadcast)
- **B tiles**: Flow SOUTH (column broadcast)

Each transfer is a single hop to an adjacent router:

```
        Col 0     Col 1     Col 2     Col 3
       ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
Row 0  │ R00 │──►│ R01 │──►│ R02 │──►│ R03 │   A tiles flow East
       └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
          │        │        │        │
          ▼        ▼        ▼        ▼        B tiles flow South
       ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
Row 1  │ R10 │──►│ R11 │──►│ R12 │──►│ R13 │
       └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
          │        │        │        │
          ▼        ▼        ▼        ▼
       ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
Row 2  │ R20 │──►│ R21 │──►│ R22 │──►│ R23 │
       └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
          │        │        │        │
          ▼        ▼        ▼        ▼
       ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
Row 3  │ R30 │──►│ R31 │──►│ R32 │──►│ R33 │
       └─────┘   └─────┘   └─────┘   └─────┘

DMA injection points: Edge routers (R00, R01, R02, R03 for top edge)
```

---

## Router Architecture (Wormhole)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              NoCRouter (Wormhole)                                        │
│                                                                                          │
│   ┌────────────────────────────────────────────────────────────────────────────────┐    │
│   │                            INPUT PORTS                                          │    │
│   │                                                                                 │    │
│   │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │    │
│   │   │   NORTH INPUT   │  │   SOUTH INPUT   │  │   EAST INPUT    │               │    │
│   │   │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │               │    │
│   │   │  │Flit Buffer│  │  │  │Flit Buffer│  │  │  │Flit Buffer│  │               │    │
│   │   │  │ 8 × 64B   │  │  │  │ 8 × 64B   │  │  │  │ 8 × 64B   │  │               │    │
│   │   │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │               │    │
│   │   │  credits: 0-8   │  │  credits: 0-8   │  │  credits: 0-8   │               │    │
│   │   └─────────────────┘  └─────────────────┘  └─────────────────┘               │    │
│   │                                                                                 │    │
│   │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │    │
│   │   │   WEST INPUT    │  │   LOCAL INPUT   │  │    DMA INPUT    │               │    │
│   │   │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │               │    │
│   │   │  │Flit Buffer│  │  │  │Flit Buffer│  │  │  │Flit Buffer│  │ (edge only)  │    │
│   │   │  │ 8 × 64B   │  │  │  │ 8 × 64B   │  │  │  │ 8 × 64B   │  │               │    │
│   │   │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │               │    │
│   │   │  credits: 0-8   │  │  (from L3)      │  │  (from DMA)     │               │    │
│   │   └─────────────────┘  └─────────────────┘  └─────────────────┘               │    │
│   └────────────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                                │
│                                         ▼                                                │
│   ┌────────────────────────────────────────────────────────────────────────────────┐    │
│   │                         WORMHOLE SWITCH CONTROL                                 │    │
│   │                                                                                 │    │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐  │    │
│   │   │                      PATH RESERVATION TABLE                              │  │    │
│   │   │                                                                          │  │    │
│   │   │   Output Port    Reserved By    Packet Seq    State                     │  │    │
│   │   │   ───────────    ───────────    ──────────    ─────                     │  │    │
│   │   │   EAST           LOCAL          42            ACTIVE                    │  │    │
│   │   │   SOUTH          NORTH          17            ACTIVE                    │  │    │
│   │   │   LOCAL          WEST           --            IDLE                      │  │    │
│   │   │   NORTH          --             --            IDLE                      │  │    │
│   │   │   WEST           --             --            IDLE                      │  │    │
│   │   │                                                                          │  │    │
│   │   │   Path reserved when HEAD flit arrives                                  │  │    │
│   │   │   Path released when TAIL flit departs                                  │  │    │
│   │   └─────────────────────────────────────────────────────────────────────────┘  │    │
│   │                                                                                 │    │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐  │    │
│   │   │                         ARBITER                                          │  │    │
│   │   │                                                                          │  │    │
│   │   │   For HEAD flits competing for same output:                             │  │    │
│   │   │     - Round-robin among requesters                                       │  │    │
│   │   │     - Loser's flits remain in input buffer (backpressure)               │  │    │
│   │   │                                                                          │  │    │
│   │   │   For BODY/TAIL flits:                                                  │  │    │
│   │   │     - Follow reserved path (no arbitration needed)                      │  │    │
│   │   └─────────────────────────────────────────────────────────────────────────┘  │    │
│   └────────────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                                │
│                                         ▼                                                │
│   ┌────────────────────────────────────────────────────────────────────────────────┐    │
│   │                            OUTPUT PORTS                                         │    │
│   │                                                                                 │    │
│   │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │    │
│   │   │  NORTH OUTPUT   │  │  SOUTH OUTPUT   │  │   EAST OUTPUT   │               │    │
│   │   │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │               │    │
│   │   │  │Flit Buffer│  │  │  │Flit Buffer│  │  │  │Flit Buffer│  │               │    │
│   │   │  │ 4 × 64B   │  │  │  │ 4 × 64B   │  │  │  │ 4 × 64B   │  │               │    │
│   │   │  └─────┬─────┘  │  │  └─────┬─────┘  │  │  └─────┬─────┘  │               │    │
│   │   │        │        │  │        │        │  │        │        │               │    │
│   │   │        ▼        │  │        ▼        │  │        ▼        │               │    │
│   │   │   ┌────────┐    │  │   ┌────────┐    │  │   ┌────────┐    │               │    │
│   │   │   │  Link  │────┼──┼───│  Link  │────┼──┼───│  Link  │────┼──►            │    │
│   │   │   │Serialzr│    │  │   │Serialzr│    │  │   │Serialzr│    │               │    │
│   │   │   └────────┘    │  │   └────────┘    │  │   └────────┘    │               │    │
│   │   └─────────────────┘  └─────────────────┘  └─────────────────┘               │    │
│   │                                                                                 │    │
│   │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │    │
│   │   │  WEST OUTPUT    │  │  LOCAL OUTPUT   │  │   DMA OUTPUT    │               │    │
│   │   │  (to neighbor)  │  │  (to L3 cache)  │  │  (edge only)    │               │    │
│   │   └─────────────────┘  └─────────────────┘  └─────────────────┘               │    │
│   └────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Wormhole Path Reservation State Machine

```
                                    ┌─────────────────────────────────────────┐
                                    │         OUTPUT PORT STATE               │
                                    │                                         │
                                    │   For each output port (N,S,E,W,L,DMA): │
                                    └─────────────────────────────────────────┘
                                                       │
                                                       ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │                                                                               │
        │                            ┌──────────┐                                      │
        │               ┌───────────►│   IDLE   │◄───────────┐                        │
        │               │            └────┬─────┘            │                        │
        │               │                 │                  │                        │
        │               │                 │ HEAD flit        │                        │
        │               │                 │ wins arbitration │                        │
        │               │                 │ for this output  │                        │
        │               │                 ▼                  │                        │
        │               │            ┌──────────┐            │                        │
        │               │            │ RESERVED │            │                        │
        │               │            │          │            │                        │
        │               │            │ input_port: X        │                        │
        │               │            │ packet_seq: N        │                        │
        │               │            └────┬─────┘            │                        │
        │               │                 │                  │                        │
        │               │                 │ BODY flits       │                        │
        │               │                 │ flow through     │                        │
        │               │                 │ (no arbitration) │                        │
        │               │                 │                  │                        │
        │               │                 │ TAIL flit        │                        │
        │               │                 │ transmitted      │                        │
        │               │                 │                  │                        │
        │               └─────────────────┴──────────────────┘                        │
        │                         path released                                        │
        │                                                                               │
        └──────────────────────────────────────────────────────────────────────────────┘

State variables per output port:
  - state:       IDLE | RESERVED
  - input_port:  Which input has the reservation (N/S/E/W/L/DMA)
  - packet_seq:  Sequence number of packet holding reservation
  - flits_remaining: Countdown to TAIL (optional, for stats)
```

---

## Simultaneous Link Activity

Key insight: **East and South outputs are independent**. A router can transmit on both simultaneously:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Router R[1,1]                                   │
│                                                                          │
│   LOCAL input                    NORTH input                            │
│   (from L3)                      (from R[0,1])                          │
│      │                              │                                    │
│      │ A tile                       │ B tile                            │
│      │ dst=EAST                     │ dst=SOUTH                         │
│      ▼                              ▼                                    │
│   ┌──────────────────────────────────────────────────────────┐          │
│   │                    SWITCH FABRIC                          │          │
│   │                                                           │          │
│   │   Path Reservations:                                     │          │
│   │     EAST output  ← LOCAL input (A tile packet 42)        │          │
│   │     SOUTH output ← NORTH input (B tile packet 17)        │          │
│   │                                                           │          │
│   │   Both paths active SIMULTANEOUSLY                       │          │
│   │   No conflict - different outputs                         │          │
│   └──────────────────────────────────────────────────────────┘          │
│                    │                           │                         │
│                    ▼                           ▼                         │
│              EAST output                 SOUTH output                   │
│                    │                           │                         │
│                    ▼                           ▼                         │
│              To R[1,2]                   To R[2,1]                       │
│              (A tile)                    (B tile)                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Bandwidth: 64 B/cycle per link × 2 active links = 128 B/cycle aggregate
```

---

## Flit Format

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HEAD FLIT (64 bytes)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  Bits [1:0]    flit_type:     2'b00 = HEAD                              │
│  Bits [9:2]    src_router:    8-bit source router ID                    │
│  Bits [17:10]  dst_router:    8-bit destination router ID               │
│  Bits [33:18]  total_flits:   16-bit flit count (max 65535 = 4MB)       │
│  Bits [49:34]  packet_seq:    16-bit sequence number                    │
│  Bits [57:50]  tensor_id:     8-bit tensor identifier (A=0,B=1,C=2,...) │
│  Bits [63:58]  reserved:      6 bits                                    │
│  Bits [127:64] tile_m:        16-bit M tile index                       │
│                tile_n:        16-bit N tile index                       │
│                tile_k:        16-bit K tile index                       │
│                flags:         16-bit flags                              │
│  Bits [511:128] reserved/padding                                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         BODY FLIT (64 bytes)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  Bits [1:0]    flit_type:     2'b01 = BODY                              │
│  Bits [511:2]  payload:       510 bits of data (not modeled in sim)     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         TAIL FLIT (64 bytes)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  Bits [1:0]    flit_type:     2'b10 = TAIL                              │
│  Bits [511:2]  payload:       510 bits of data (not modeled in sim)     │
└─────────────────────────────────────────────────────────────────────────┘

Note: For simulation, we don't model actual payload data, just track metadata.
      Flit type determines path reservation/release behavior.
```

---

## Timing: Single Tile Transfer (Nearest Neighbor)

256KB tile from R[0,0] LOCAL to R[0,1] (single hop East):

```
                     Cycles
          0       1024     2048     3072     4096
          │        │        │        │        │
          ▼        ▼        ▼        ▼        ▼

R[0,0] L3 Cache:
          ├────────────────────────────────────────────────────────┤
          │ Pushing 4096 flits to LOCAL input @ 1 flit/cycle       │
          └────────────────────────────────────────────────────────┘

R[0,0] LOCAL Input Buffer:
          █████████ (8 flits max buffered at any time)
          │ Flits flow through as fast as output can accept

R[0,0] EAST Output:
          ├────────────────────────────────────────────────────────┤
          │ Path reserved by HEAD flit at cycle 0                  │
          │ Transmitting 4096 flits @ 1 flit/cycle                 │
          │ Path released by TAIL flit at cycle 4095               │
          └────────────────────────────────────────────────────────┘

R[0,0]→R[0,1] Link:
          ├────────────────────────────────────────────────────────┤
          │ 1 flit in flight per cycle (1 cycle link latency)      │
          └────────────────────────────────────────────────────────┘

R[0,1] WEST Input Buffer:
             █████████ (8 flits max buffered)
             │ Returns credits to R[0,0] as flits consumed

R[0,1] LOCAL Output (to L3):
             ├────────────────────────────────────────────────────┤
             │ Delivering flits to L3 cache @ 1 flit/cycle        │
             │ First flit arrives cycle 1, last at cycle 4096     │
             └────────────────────────────────────────────────────┘

TOTAL LATENCY: 4096 cycles for 256KB (limited by bandwidth, not hops)
               = 8.2 µs @ 500 MHz
               = 31.25 GB/s effective bandwidth
```

---

## Concurrent Transfers on Same Router

R[1,1] simultaneously forwarding A tile East and B tile South:

```
                     Cycles
          0       1024     2048     3072     4096
          │        │        │        │        │
          ▼        ▼        ▼        ▼        ▼

A tile (LOCAL → EAST):
          ├────────────────────────────────────────────────────────┤
          │ 4096 flits flowing LOCAL input → EAST output           │
          │ EAST output reserved for this packet                   │
          └────────────────────────────────────────────────────────┘

B tile (NORTH → SOUTH):
          ├────────────────────────────────────────────────────────┤
          │ 4096 flits flowing NORTH input → SOUTH output          │
          │ SOUTH output reserved for this packet                  │
          └────────────────────────────────────────────────────────┘

          │◄──────────── CONCURRENT ─────────────►│

Both transfers complete in 4096 cycles
Aggregate bandwidth: 2 × 64 B/cycle = 128 B/cycle through this router
```

---

## Contention Scenario

Two packets both want EAST output:

```
                     Cycles
          0       1024     2048     3072     4096     5120     6144     7168     8192
          │        │        │        │        │        │        │        │        │
          ▼        ▼        ▼        ▼        ▼        ▼        ▼        ▼        ▼

Packet A (LOCAL → EAST, arrives first):
          ├────────────────────────────────────────────────────────┤
          │ HEAD wins arbitration at cycle 0                       │
          │ EAST reserved for packet A                             │
          │ 4096 flits transmitted                                 │
          │ TAIL releases path at cycle 4095                       │
          └────────────────────────────────────────────────────────┘

Packet B (NORTH → EAST, arrives cycle 0):
          │◄─── BLOCKED ───►├────────────────────────────────────────────────────────┤
          │ HEAD loses       │ HEAD wins arbitration at cycle 4096                   │
          │ arbitration      │ EAST reserved for packet B                            │
          │ Flits back up    │ 4096 flits transmitted                                │
          │ in NORTH input   │ TAIL releases path at cycle 8191                      │
          │ buffer           └───────────────────────────────────────────────────────┘
          │
          │ Backpressure propagates upstream via credits
          │ NORTH input buffer fills (8 flits)
          │ Credits to upstream router exhausted
          │ Upstream router stalls on this output

BLOCKING TIME: 4096 cycles (entire packet A duration)
This is HEAD-OF-LINE (HOL) blocking - fundamental to wormhole routing
```

---

## DMA Port Modeling (Edge Routers Only)

DMA engines connect to edge routers for external memory access:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              TOP EDGE (Row 0)                                      │
│                                                                                    │
│   External Memory Bus                                                              │
│   ════════════════════════════════════════════════════════════════════════════    │
│         │              │              │              │                             │
│         ▼              ▼              ▼              ▼                             │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                       │
│   │   DMA    │   │   DMA    │   │   DMA    │   │   DMA    │                       │
│   │ Engine 0 │   │ Engine 1 │   │ Engine 2 │   │ Engine 3 │                       │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘                       │
│        │              │              │              │                              │
│        ▼              ▼              ▼              ▼                              │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                        │
│   │  R[0,0] │───►│  R[0,1] │───►│  R[0,2] │───►│  R[0,3] │                        │
│   │         │    │         │    │         │    │         │                        │
│   │ DMA_IN  │    │ DMA_IN  │    │ DMA_IN  │    │ DMA_IN  │                        │
│   │ DMA_OUT │    │ DMA_OUT │    │ DMA_OUT │    │ DMA_OUT │                        │
│   └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘                        │
│        │              │              │              │                              │
│        ▼              ▼              ▼              ▼                              │
│      South          South          South          South                           │
│                                                                                    │
└───────────────────────────────────────────────────────────────────────────────────┘

DMA Operations:
  - LOAD:  DMA_IN → LOCAL output (to L3 cache)
  - STORE: LOCAL input → DMA_OUT (from L3 cache to memory)

DMA port has same structure as other ports:
  - 8 flit input buffer
  - 4 flit output buffer
  - Credit-based flow control
  - 64 B/cycle bandwidth (can be different, e.g., 32 B/cycle for DDR)
```

---

## C++ Implementation

```cpp
//=============================================================================
// Flit Types
//=============================================================================
enum class FlitType : uint8_t {
    HEAD = 0,
    BODY = 1,
    TAIL = 2,
    HEAD_TAIL = 3  // Single-flit packet (rare, for small control messages)
};

struct Flit {
    FlitType type;
    uint16_t packet_seq;      // Which packet this flit belongs to

    // Only valid for HEAD flits:
    uint8_t src_router;
    uint8_t dst_router;
    uint16_t total_flits;
    TileDescriptor tile;

    bool is_head() const { return type == FlitType::HEAD || type == FlitType::HEAD_TAIL; }
    bool is_tail() const { return type == FlitType::TAIL || type == FlitType::HEAD_TAIL; }
};

//=============================================================================
// Credit-based Flit Buffer
//=============================================================================
class FlitBuffer {
public:
    static constexpr size_t CAPACITY = 8;  // 8 flits = 512 bytes

    bool can_accept() const { return count_ < CAPACITY; }
    bool is_empty() const { return count_ == 0; }
    size_t count() const { return count_; }
    size_t free_space() const { return CAPACITY - count_; }

    bool push(const Flit& flit);
    std::optional<Flit> pop();
    const Flit* peek() const;

private:
    std::array<Flit, CAPACITY> buffer_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t count_ = 0;
};

//=============================================================================
// Path Reservation for Wormhole Routing
//=============================================================================
struct PathReservation {
    enum class State { IDLE, RESERVED };

    State state = State::IDLE;
    PortDirection input_port;   // Which input holds the reservation
    uint16_t packet_seq;        // Packet sequence number
    uint16_t flits_remaining;   // For statistics/debugging

    bool is_reserved() const { return state == State::RESERVED; }

    void reserve(PortDirection in_port, uint16_t seq, uint16_t total_flits) {
        state = State::RESERVED;
        input_port = in_port;
        packet_seq = seq;
        flits_remaining = total_flits;
    }

    void release() {
        state = State::IDLE;
    }
};

//=============================================================================
// Input Port (Wormhole)
//=============================================================================
class WormholeInputPort {
public:
    WormholeInputPort(PortDirection dir, size_t buffer_size = 8);

    PortDirection direction() const { return direction_; }

    // Flit reception
    bool can_receive() const { return buffer_.can_accept(); }
    void receive_flit(const Flit& flit, uint64_t cycle);

    // Flit availability for switch
    bool has_flit() const { return !buffer_.is_empty(); }
    const Flit* peek_flit() const { return buffer_.peek(); }
    Flit consume_flit();

    // Credit management
    uint32_t credits_to_return() const { return credits_to_return_; }
    void clear_returned_credits() { credits_to_return_ = 0; }

    // Current packet state (for path reservation)
    bool has_active_packet() const { return active_packet_seq_.has_value(); }
    uint16_t active_packet_seq() const { return active_packet_seq_.value_or(0); }

    // Statistics
    struct Stats {
        uint64_t flits_received = 0;
        uint64_t head_flits = 0;
        uint64_t tail_flits = 0;
        uint64_t stall_cycles = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    PortDirection direction_;
    FlitBuffer buffer_;
    uint32_t credits_to_return_ = 0;
    std::optional<uint16_t> active_packet_seq_;  // Packet currently being received
    Stats stats_;
};

//=============================================================================
// Output Port (Wormhole)
//=============================================================================
class WormholeOutputPort {
public:
    WormholeOutputPort(PortDirection dir, size_t buffer_size = 4);

    PortDirection direction() const { return direction_; }

    // Connection to downstream
    void connect(NoCLink* link);
    void set_downstream_credits(uint32_t initial_credits);

    // Flit acceptance from crossbar
    bool can_accept() const { return buffer_.can_accept() && credits_ > 0; }
    void accept_flit(const Flit& flit);

    // Link transmission
    void step(uint64_t cycle);

    // Credit reception from downstream
    void receive_credit() { credits_++; }

    // Path reservation
    PathReservation& reservation() { return reservation_; }
    const PathReservation& reservation() const { return reservation_; }

    // Statistics
    struct Stats {
        uint64_t flits_sent = 0;
        uint64_t link_busy_cycles = 0;
        uint64_t credit_stall_cycles = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    PortDirection direction_;
    FlitBuffer buffer_;
    NoCLink* link_ = nullptr;
    uint32_t credits_ = 0;
    PathReservation reservation_;
    Stats stats_;
};

//=============================================================================
// Wormhole Router
//=============================================================================
class WormholeRouter {
public:
    WormholeRouter(uint8_t id, uint8_t row, uint8_t col, const NoCConfig& config);

    uint8_t id() const { return id_; }
    uint8_t row() const { return row_; }
    uint8_t col() const { return col_; }

    // Port access
    WormholeInputPort& input(PortDirection dir);
    WormholeOutputPort& output(PortDirection dir);

    // Injection from L3 cache
    bool can_inject() const;
    bool inject_flit(const Flit& flit, uint64_t cycle);

    // Ejection to L3 cache
    bool has_ejection() const;
    Flit eject_flit();

    // DMA ports (edge routers only)
    bool has_dma_port() const { return has_dma_; }
    WormholeInputPort* dma_input() { return has_dma_ ? &dma_input_ : nullptr; }
    WormholeOutputPort* dma_output() { return has_dma_ ? &dma_output_ : nullptr; }

    // Simulation step
    void step(uint64_t cycle);

    // Statistics
    struct Stats {
        uint64_t flits_switched = 0;
        uint64_t arbitration_events = 0;
        uint64_t arbitration_conflicts = 0;
        uint64_t paths_reserved = 0;
        uint64_t paths_released = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    uint8_t id_, row_, col_;
    const NoCConfig& config_;
    bool has_dma_;

    // Standard ports: N, S, E, W, L
    std::array<WormholeInputPort, 5> inputs_;
    std::array<WormholeOutputPort, 5> outputs_;

    // DMA ports (only for edge routers)
    WormholeInputPort dma_input_;
    WormholeOutputPort dma_output_;

    Stats stats_;

    // Routing (nearest-neighbor only)
    PortDirection route_nearest_neighbor(uint8_t dst_router) const;

    // Arbitration
    void arbitrate_and_switch(uint64_t cycle);
};

//=============================================================================
// NoC Top Level
//=============================================================================
class WormholeNoC {
public:
    explicit WormholeNoC(const NoCConfig& config);

    // Tile injection (from BlockMover)
    enum class InjectResult { SUCCESS, BUSY, ERROR };
    InjectResult inject_tile(uint8_t src_router, uint8_t dst_router,
                             const TileDescriptor& tile, uint64_t cycle);

    // Check injection readiness
    bool can_inject(uint8_t router_id) const;

    // Delivery callback
    using DeliveryCallback = std::function<void(const TileDescriptor&, uint64_t)>;
    void set_delivery_callback(uint8_t router_id, DeliveryCallback cb);

    // DMA interface
    InjectResult dma_load(uint8_t dst_router, const TileDescriptor& tile, uint64_t cycle);
    InjectResult dma_store(uint8_t src_router, const TileDescriptor& tile, uint64_t cycle);

    // Simulation
    void step(uint64_t cycle);
    bool is_idle() const;

    // Statistics
    struct Stats {
        uint64_t total_flits = 0;
        uint64_t total_tiles = 0;
        uint64_t total_bytes = 0;
        uint64_t total_latency_cycles = 0;
        uint64_t max_latency_cycles = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    NoCConfig config_;
    std::vector<WormholeRouter> routers_;
    std::vector<NoCLink> links_;

    // Per-router delivery callbacks
    std::vector<DeliveryCallback> delivery_callbacks_;

    // Active tile transfers (for tracking completion)
    struct ActiveTransfer {
        uint16_t packet_seq;
        uint8_t src_router;
        uint8_t dst_router;
        TileDescriptor tile;
        uint64_t inject_start_cycle;
        uint16_t total_flits;
        uint16_t flits_injected;
        uint16_t flits_delivered;
    };
    std::map<uint16_t, ActiveTransfer> active_transfers_;
    uint16_t next_packet_seq_ = 0;

    Stats stats_;
};
```

---

## Trace Events for Visualization

```cpp
enum class WormholeEventType {
    INJECT_START,      // First flit of tile injected
    INJECT_FLIT,       // Individual flit injected (optional, verbose)
    INJECT_COMPLETE,   // Last flit of tile injected

    PATH_RESERVE,      // HEAD flit reserves output path
    PATH_RELEASE,      // TAIL flit releases output path

    LINK_TRANSFER,     // Flit traversing link

    DELIVER_START,     // First flit arrives at destination
    DELIVER_COMPLETE,  // Last flit arrives at destination

    ARBITRATION_WIN,   // HEAD flit wins arbitration
    ARBITRATION_LOSE,  // HEAD flit loses, must wait
    BACKPRESSURE,      // Credit exhaustion causing stall
};

struct WormholeTraceEvent {
    uint64_t cycle;
    WormholeEventType type;
    uint8_t router_id;
    PortDirection port;
    uint16_t packet_seq;
    FlitType flit_type;
    uint16_t flit_index;      // Which flit (0 to total_flits-1)
    uint16_t total_flits;
    TileDescriptor tile;      // Only for INJECT_START, DELIVER_COMPLETE
};
```

---

## Summary

| Aspect | Design Choice |
|--------|---------------|
| Routing | Wormhole (flits flow through) |
| Scope | Nearest-neighbor (single hop) |
| Buffer size | 8 flits input, 4 flits output (512B / 256B) |
| Flow control | Credit-based |
| Flit size | 64 bytes |
| Bandwidth | 1 flit/cycle/link = 64 B/cycle = 32 GB/s @ 500 MHz |
| Concurrent links | Yes - all output ports independent |
| Contention | HOL blocking on same output port |
| DMA | Edge routers only, same bandwidth as mesh links |
| Virtual channels | None (XY routing is deadlock-free) |

---

## Confirmed Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Routing scope | Single-hop only | Nearest neighbor sufficient for systolic |
| L3 interface | 64 B/cycle | Multi-banked, 64-byte width, L3 controller parallelizes |
| L3/NoC clock | Same clock domain | Simplifies interface |
| DMA bandwidth | 64 B/cycle | Matched to NoC (DMA→L3→NoC path) |
| Tile size range | 1KB - 256KB | Variable based on operation |
| Tile sweet spot | 4KB - 8KB | Common case for block matmul |

### Tile Size Impact on Transfer Time

| Tile Size | Flits | Transfer Time @ 64 B/cycle |
|-----------|-------|---------------------------|
| 1 KB | 16 | 16 cycles (32 ns @ 500 MHz) |
| 4 KB | 64 | 64 cycles (128 ns) |
| 8 KB | 128 | 128 cycles (256 ns) |
| 16 KB | 256 | 256 cycles (512 ns) |
| 64 KB | 1024 | 1024 cycles (2 µs) |
| 256 KB | 4096 | 4096 cycles (8 µs) |

Smaller tiles (4-8KB sweet spot) mean:
- Less HOL blocking time per packet
- More opportunities for interleaving multiple streams
- Better overall link utilization
