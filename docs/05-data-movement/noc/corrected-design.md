# NoC Router Corrected Design

## Design Goals

1. **Bandwidth-accurate**: Enforce 64 B/cycle per link
2. **Resource serialization**: One packet per port at a time
3. **Store-and-forward**: Complete packet buffered before forwarding
4. **Credit-based flow control**: Backpressure when downstream full
5. **RTL-synthesizable model**: Maps cleanly to hardware

---

## System Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Flit size | 64 bytes | Single transfer unit |
| Link bandwidth | 1 flit/cycle | 64 B/cycle = 32 GB/s @ 500 MHz |
| Max tile size | 256 KB | 4096 flits |
| Input buffer | 8 flits | 512 bytes per port |
| Output buffer | 4 flits | 256 bytes per port |
| Router latency | 1 cycle | Routing decision time |
| Link latency | 1 cycle | Wire delay |
| Mesh size | 4×4 | 16 routers |

---

## Router Block Diagram

```
                              ┌──────────────────────────────────────────────────────────┐
                              │                      NoCRouter                            │
                              │                                                           │
    From North Router         │  ┌─────────────────────────────────────────────────────┐ │         To North Router
    ─────────────────────────►│  │                   INPUT UNIT [N]                    │ │◄─────────────────────────
                              │  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │ │
                              │  │  │ Flit Buffer │  │   Packet    │  │  Credit    │  │ │
                              │  │  │  8 × 64B    │  │  Assembler  │  │  Counter   │  │ │
                              │  │  └─────────────┘  └─────────────┘  └────────────┘  │ │
                              │  └─────────────────────────────────────────────────────┘ │
                              │                            │                              │
    From South Router         │  ┌─────────────────────────────────────────────────────┐ │         To South Router
    ─────────────────────────►│  │                   INPUT UNIT [S]                    │ │◄─────────────────────────
                              │  │         (same structure as above)                   │ │
                              │  └─────────────────────────────────────────────────────┘ │
                              │                            │                              │
    From East Router          │  ┌─────────────────────────────────────────────────────┐ │         To East Router
    ─────────────────────────►│  │                   INPUT UNIT [E]                    │ │◄─────────────────────────
                              │  │         (same structure as above)                   │ │
                              │  └─────────────────────────────────────────────────────┘ │
                              │                            │                              │
    From West Router          │  ┌─────────────────────────────────────────────────────┐ │         To West Router
    ─────────────────────────►│  │                   INPUT UNIT [W]                    │ │◄─────────────────────────
                              │  │         (same structure as above)                   │ │
                              │  └─────────────────────────────────────────────────────┘ │
                              │                            │                              │
    From L3 Cache             │  ┌─────────────────────────────────────────────────────┐ │         To L3 Cache
    ─────────────────────────►│  │                   INPUT UNIT [L]                    │ │◄─────────────────────────
    (Local Injection)         │  │         (same structure as above)                   │ │         (Local Ejection)
                              │  └─────────────────────────────────────────────────────┘ │
                              │                            │                              │
                              │                            ▼                              │
                              │  ┌─────────────────────────────────────────────────────┐ │
                              │  │                   SWITCH FABRIC                      │ │
                              │  │                                                      │ │
                              │  │   ┌────────────┐    ┌────────────┐    ┌──────────┐  │ │
                              │  │   │   Route    │───►│  Arbiter   │───►│ Crossbar │  │ │
                              │  │   │  Compute   │    │ (per port) │    │  5 × 5   │  │ │
                              │  │   └────────────┘    └────────────┘    └──────────┘  │ │
                              │  │                                                      │ │
                              │  └─────────────────────────────────────────────────────┘ │
                              │                            │                              │
                              │                            ▼                              │
                              │  ┌─────────────────────────────────────────────────────┐ │
                              │  │                  OUTPUT UNITS [N,S,E,W,L]           │ │
                              │  │                                                      │ │
                              │  │   ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │ │
                              │  │   │ Flit Buffer │  │    Link     │  │   Credit   │  │ │
                              │  │   │  4 × 64B    │  │ Serializer  │  │   Return   │  │ │
                              │  │   └─────────────┘  └─────────────┘  └────────────┘  │ │
                              │  └─────────────────────────────────────────────────────┘ │
                              └──────────────────────────────────────────────────────────┘
```

---

## Input Unit Detail

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    INPUT UNIT                                           │
│                                                                                         │
│   From Link                                                                             │
│   ──────────────────────────────────────────────────────────────────►                  │
│       │                                                                                 │
│       │  ┌──────────────────────────────────────────────────────────────────────────┐  │
│       │  │                         FLIT RECEIVER                                     │  │
│       │  │                                                                           │  │
│       │  │   flit_valid ──────┐                                                     │  │
│       │  │   flit_data[63:0] ─┼──►┌─────────────────────────────────────────────┐   │  │
│       │  │   flit_head ───────┤   │              FLIT BUFFER                     │   │  │
│       │  │   flit_tail ───────┘   │                                              │   │  │
│       │  │                        │   ┌────┬────┬────┬────┬────┬────┬────┬────┐  │   │  │
│       │  │                        │   │ S0 │ S1 │ S2 │ S3 │ S4 │ S5 │ S6 │ S7 │  │   │  │
│       │  │                        │   │64B │64B │64B │64B │64B │64B │64B │64B │  │   │  │
│       │  │                        │   └────┴────┴────┴────┴────┴────┴────┴────┘  │   │  │
│       │  │                        │              ▲                    │           │   │  │
│       │  │                        │              │                    ▼           │   │  │
│       │  │                        │         write_ptr            read_ptr         │   │  │
│       │  │                        │                                              │   │  │
│       │  │                        │   buffer_count: 0..8                         │   │  │
│       │  │                        │   buffer_full:  (count == 8)                 │   │  │
│       │  │                        │   buffer_empty: (count == 0)                 │   │  │
│       │  │                        └──────────────────────────────────────────────┘   │  │
│       │  │                                           │                               │  │
│       │  └───────────────────────────────────────────┼───────────────────────────────┘  │
│       │                                              │                                  │
│       │  ┌───────────────────────────────────────────┼───────────────────────────────┐  │
│       │  │                      PACKET STATE MACHINE │                               │  │
│       │  │                                           ▼                               │  │
│       │  │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│       │  │   │                    Packet Assembly State                         │    │  │
│       │  │   │                                                                  │    │  │
│       │  │   │   current_packet:                                               │    │  │
│       │  │   │     ├─ src_router:      uint8                                   │    │  │
│       │  │   │     ├─ dst_router:      uint8                                   │    │  │
│       │  │   │     ├─ tile_descriptor: TileDescriptor                          │    │  │
│       │  │   │     ├─ total_flits:     uint16  (from header)                   │    │  │
│       │  │   │     └─ flits_received:  uint16  (progress)                      │    │  │
│       │  │   │                                                                  │    │  │
│       │  │   │   state: IDLE | RECEIVING | COMPLETE                            │    │  │
│       │  │   │                                                                  │    │  │
│       │  │   │   ┌──────┐  head_flit   ┌───────────┐  tail_flit  ┌──────────┐ │    │  │
│       │  │   │   │ IDLE │─────────────►│ RECEIVING │────────────►│ COMPLETE │ │    │  │
│       │  │   │   └──────┘              └───────────┘             └──────────┘ │    │  │
│       │  │   │       ▲                       │                        │        │    │  │
│       │  │   │       └───────────────────────┴────────────────────────┘        │    │  │
│       │  │   │                        packet_forwarded                          │    │  │
│       │  │   └─────────────────────────────────────────────────────────────────┘    │  │
│       │  │                                           │                               │  │
│       │  │                                           │ packet_ready                  │  │
│       │  │                                           ▼                               │  │
│       │  │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│       │  │   │                    Ready Queue (depth 2)                         │    │  │
│       │  │   │   Packets with all flits received, waiting for routing          │    │  │
│       │  │   └─────────────────────────────────────────────────────────────────┘    │  │
│       │  │                                           │                               │  │
│       │  └───────────────────────────────────────────┼───────────────────────────────┘  │
│                                                      │                                  │
│   To Arbiter ◄───────────────────────────────────────┘                                  │
│                                                                                         │
│   Credit Return ◄────────────────────────────────────────────────────────────────────── │
│   (to upstream router)      credit_out = (flit consumed from buffer)                   │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Local Injection Unit (L3 Cache Interface)

The LOCAL port has special handling for injection from the L3 cache:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              LOCAL INJECTION UNIT                                       │
│                                                                                         │
│   From L3 Cache                                                                         │
│   ──────────────────────────────────────────────────────────────────►                  │
│       │                                                                                 │
│       │   inject_request ────────┐                                                     │
│       │   inject_tile_desc ──────┤                                                     │
│       │   inject_tile_size ──────┘                                                     │
│       │                          │                                                      │
│       │  ┌───────────────────────┼──────────────────────────────────────────────────┐  │
│       │  │                INJECTION STATE MACHINE                                    │  │
│       │  │                       │                                                   │  │
│       │  │   ┌──────┐           ▼             ┌────────────────┐                    │  │
│       │  │   │ IDLE │◄──────────────────────►│   INJECTING    │                    │  │
│       │  │   └──────┘   inject_request &     │                │                    │  │
│       │  │       │      buffer_has_space     │  flits_sent++  │                    │  │
│       │  │       │                           │  each cycle    │                    │  │
│       │  │       │                           │                │                    │  │
│       │  │       │                           └────────────────┘                    │  │
│       │  │       │                                  │                               │  │
│       │  │       │                                  │ (flits_sent == total_flits)  │  │
│       │  │       │                                  ▼                               │  │
│       │  │       │                           ┌────────────────┐                    │  │
│       │  │       └───────────────────────────│   COMPLETE     │                    │  │
│       │  │                                   │ signal to L3   │                    │  │
│       │  │                                   └────────────────┘                    │  │
│       │  │                                                                          │  │
│       │  │   State variables:                                                       │  │
│       │  │     injection_active:    bool                                           │  │
│       │  │     injection_packet:    NoCPacket                                      │  │
│       │  │     injection_start:     uint64  (cycle when started)                   │  │
│       │  │     flits_injected:      uint16  (0 to total_flits)                     │  │
│       │  │     total_flits:         uint16  (tile_size / 64)                       │  │
│       │  │                                                                          │  │
│       │  │   Timing:                                                                │  │
│       │  │     256KB tile = 4096 flits                                             │  │
│       │  │     @ 1 flit/cycle = 4096 cycles to inject                             │  │
│       │  │                                                                          │  │
│       │  │   Backpressure:                                                          │  │
│       │  │     If flit_buffer is full (8 flits), injection stalls                  │  │
│       │  │     Stall cycles counted for performance analysis                        │  │
│       │  │                                                                          │  │
│       │  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   inject_ready ◄─────────────────────────────────────────────────────────────────────  │
│   (to L3 Cache)       = !injection_active                                              │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Unit Detail

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    OUTPUT UNIT                                          │
│                                                                                         │
│   From Crossbar                                                                         │
│   ──────────────────────────────────────────────────────────────────►                  │
│       │                                                                                 │
│       │  ┌──────────────────────────────────────────────────────────────────────────┐  │
│       │  │                         OUTPUT BUFFER                                     │  │
│       │  │                                                                           │  │
│       │  │   ┌────┬────┬────┬────┐                                                  │  │
│       │  │   │ S0 │ S1 │ S2 │ S3 │   4 flit slots (256 bytes)                      │  │
│       │  │   │64B │64B │64B │64B │                                                  │  │
│       │  │   └────┴────┴────┴────┘                                                  │  │
│       │  │                                                                           │  │
│       │  │   buffer_count: 0..4                                                     │  │
│       │  │   can_accept:   (count < 4)                                              │  │
│       │  │                                                                           │  │
│       │  └──────────────────────────────────────────────────────────────────────────┘  │
│       │                                              │                                  │
│       │  ┌───────────────────────────────────────────┼───────────────────────────────┐  │
│       │  │                      LINK SERIALIZER      │                               │  │
│       │  │                                           ▼                               │  │
│       │  │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│       │  │   │                    Transmission State                            │    │  │
│       │  │   │                                                                  │    │  │
│       │  │   │   current_packet:                                               │    │  │
│       │  │   │     ├─ packet_ptr:      NoCPacket*                              │    │  │
│       │  │   │     ├─ total_flits:     uint16                                  │    │  │
│       │  │   │     └─ flits_sent:      uint16  (progress)                      │    │  │
│       │  │   │                                                                  │    │  │
│       │  │   │   state: IDLE | SENDING                                         │    │  │
│       │  │   │                                                                  │    │  │
│       │  │   │   Link interface:                                               │    │  │
│       │  │   │     link_valid:  (state == SENDING && has_credits)              │    │  │
│       │  │   │     link_data:   current flit data                              │    │  │
│       │  │   │     link_head:   (flits_sent == 0)                              │    │  │
│       │  │   │     link_tail:   (flits_sent == total_flits - 1)                │    │  │
│       │  │   │                                                                  │    │  │
│       │  │   └─────────────────────────────────────────────────────────────────┘    │  │
│       │  │                                                                           │  │
│       │  │   Credits from downstream:                                               │  │
│       │  │     credits_available: 0..8  (downstream buffer space)                   │  │
│       │  │     can_send: (credits_available > 0) && (buffer_count > 0)             │  │
│       │  │                                                                           │  │
│       │  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   To Link ◄──────────────────────────────────────────────────────────────────────────  │
│                                                                                         │
│   Credit In ◄────────────────────────────────────────────────────────────────────────  │
│   (from downstream router)                                                             │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Switch Fabric / Crossbar

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SWITCH FABRIC                                         │
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                              ROUTE COMPUTE                                       │  │
│   │                                                                                  │  │
│   │   For each input port with ready packet:                                        │  │
│   │     output_port = XY_route(packet.dst_router)                                   │  │
│   │                                                                                  │  │
│   │   XY Routing (deterministic, deadlock-free):                                    │  │
│   │     if (dst_col > my_col) return EAST                                          │  │
│   │     if (dst_col < my_col) return WEST                                          │  │
│   │     if (dst_row > my_row) return SOUTH                                         │  │
│   │     if (dst_row < my_row) return NORTH                                         │  │
│   │     return LOCAL  // arrived at destination                                     │  │
│   │                                                                                  │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                              │                                          │
│                                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                           ARBITER (per output port)                              │  │
│   │                                                                                  │  │
│   │   For each output port O:                                                       │  │
│   │     candidates = {input ports requesting O}                                     │  │
│   │                                                                                  │  │
│   │     if output_buffer[O].can_accept():                                          │  │
│   │       winner = round_robin_select(candidates, last_winner[O])                  │  │
│   │       grant[winner] = true                                                      │  │
│   │       last_winner[O] = winner                                                   │  │
│   │                                                                                  │  │
│   │   Priority order (for tie-breaking):                                           │  │
│   │     1. Packets already in-flight (wormhole continuation)                       │  │
│   │     2. Older packets (by inject_cycle)                                         │  │
│   │     3. Round-robin among equals                                                │  │
│   │                                                                                  │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                              │                                          │
│                                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                              5×5 CROSSBAR                                        │  │
│   │                                                                                  │  │
│   │        OUT[N]    OUT[S]    OUT[E]    OUT[W]    OUT[L]                           │  │
│   │           │         │         │         │         │                             │  │
│   │   IN[N]──┬┼─────────┼─────────┼─────────┼─────────┼──                          │  │
│   │          ││         │         │         │         │                             │  │
│   │   IN[S]──┼┼─────────┬─────────┼─────────┼─────────┼──                          │  │
│   │          ││         │         │         │         │                             │  │
│   │   IN[E]──┼┼─────────┼─────────┬─────────┼─────────┼──                          │  │
│   │          ││         │         │         │         │                             │  │
│   │   IN[W]──┼┼─────────┼─────────┼─────────┬─────────┼──                          │  │
│   │          ││         │         │         │         │                             │  │
│   │   IN[L]──┼┼─────────┼─────────┼─────────┼─────────┬──                          │  │
│   │          ▼▼         ▼         ▼         ▼         ▼                             │  │
│   │                                                                                  │  │
│   │   Each crosspoint: 64-bit mux controlled by arbiter grant                      │  │
│   │   One flit transferred per granted input→output pair per cycle                 │  │
│   │                                                                                  │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Packet Header Format

The first flit (head flit) contains routing and packet metadata:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      HEAD FLIT (64 bytes)                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Byte 0-1:   src_router (uint8) + dst_router (uint8)                    │
│  Byte 2-3:   total_flits (uint16) - number of flits in packet           │
│  Byte 4-5:   packet_sequence (uint16) - for ordering/debugging          │
│  Byte 6:     traffic_class (uint8) - BULK=0, LOW_LATENCY=1, DMA=2       │
│  Byte 7:     flags (uint8) - reserved                                   │
│  Byte 8-15:  inject_cycle (uint64) - timestamp for latency calculation  │
│  Byte 16-47: tile_descriptor (32 bytes)                                 │
│              ├─ tensor_id (uint8)                                       │
│              ├─ m_tile, n_tile, k_tile (uint16 × 3)                     │
│              ├─ base_addr (uint64)                                      │
│              ├─ size (uint32)                                           │
│              └─ padding                                                  │
│  Byte 48-63: reserved / checksum                                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     BODY FLIT (64 bytes)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Byte 0-63:  payload data (not modeled in simulation - just count)      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     TAIL FLIT (64 bytes)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Byte 0-63:  final payload data                                         │
│              (implicit: seeing tail flit marks packet complete)         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Timing Diagram: Single Packet Transfer

A 256KB packet (4096 flits) from R[0,0] to R[0,3] (3 hops East):

```
Cycle:    0         4096      8192      12288     16384
          │          │          │          │          │
          ▼          ▼          ▼          ▼          ▼

R[0,0] LOCAL injection:
          ├──────────────────────────────────────────────┤
          │ Injecting 4096 flits @ 1 flit/cycle          │
          │ Flits enter input buffer, route to EAST out │
          └──────────────────────────────────────────────┘

R[0,0]→R[0,1] Link:
                    ├──────────────────────────────────────────────┤
                    │ Transmitting 4096 flits @ 1 flit/cycle       │
                    │ First flit arrives at R[0,1] cycle 4097      │
                    └──────────────────────────────────────────────┘

R[0,1]→R[0,2] Link:
                              ├──────────────────────────────────────────────┤
                              │ Transmitting 4096 flits                       │
                              │ First flit arrives at R[0,2] cycle 8193       │
                              └──────────────────────────────────────────────┘

R[0,2]→R[0,3] Link:
                                        ├──────────────────────────────────────────────┤
                                        │ Transmitting 4096 flits                       │
                                        │ Last flit arrives at R[0,3] cycle 16384      │
                                        └──────────────────────────────────────────────┘

TOTAL LATENCY: ~16,384 cycles for 256KB over 3 hops
  - Injection:     4,096 cycles
  - Hop 1:         4,096 cycles
  - Hop 2:         4,096 cycles
  - Hop 3:         4,096 cycles
  - Router delays: ~3 cycles (negligible)
```

---

## Timing Diagram: Contention

Two packets competing for the same output port:

```
Cycle:    0         4096      8192      12288     16384
          │          │          │          │          │
          ▼          ▼          ▼          ▼          ▼

Packet A (from WEST, going EAST):
          ├───────────────────────────────────────┤
          │ Wins arbitration at cycle 0           │
          │ Transmitting on EAST output           │
          └───────────────────────────────────────┘

Packet B (from LOCAL, going EAST):
          │◄── BLOCKED ──►├───────────────────────────────────────┤
          │  Waiting for   │ Wins after A completes              │
          │  EAST output   │ Transmitting on EAST output         │
          │  4096 cycles   │                                      │
          └────────────────┴──────────────────────────────────────┘

Packet B experiences 4096 cycles of HOL (head-of-line) blocking
This is the COST OF CONTENTION - accurately modeled
```

---

## Credit-Based Flow Control

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      CREDIT FLOW CONTROL                                  │
│                                                                           │
│   Upstream Router                              Downstream Router          │
│   ┌─────────────┐                              ┌─────────────┐           │
│   │ Output Unit │                              │ Input Unit  │           │
│   │             │                              │             │           │
│   │ credits: 8  │◄────── credit_return ────────│ buffer: 8   │           │
│   │             │         (1 per consumed)     │ slots       │           │
│   │             │                              │             │           │
│   │ can_send =  │────────── flit_valid ───────►│ flit →      │           │
│   │ credits > 0 │────────── flit_data ────────►│ buffer      │           │
│   │             │                              │             │           │
│   └─────────────┘                              └─────────────┘           │
│                                                                           │
│   Protocol:                                                               │
│   1. Upstream starts with credits = downstream_buffer_size (8)           │
│   2. Each flit sent decrements credits                                   │
│   3. Downstream returns 1 credit when flit consumed from buffer          │
│   4. If credits == 0, upstream stalls (backpressure)                     │
│                                                                           │
│   This prevents buffer overflow without dropped packets                  │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## C++ Class Design

```cpp
//=============================================================================
// FlitBuffer - Circular buffer for flits
//=============================================================================
class FlitBuffer {
public:
    static constexpr size_t FLIT_SIZE = 64;

    explicit FlitBuffer(size_t capacity_flits);

    bool can_accept() const { return count_ < capacity_; }
    bool is_empty() const { return count_ == 0; }
    size_t count() const { return count_; }
    size_t free_space() const { return capacity_ - count_; }

    // Write one flit (returns false if full)
    bool push_flit(const FlitData& flit);

    // Read one flit (returns nullopt if empty)
    std::optional<FlitData> pop_flit();

    // Peek at front flit without removing
    const FlitData* peek_front() const;

private:
    size_t capacity_;
    size_t count_ = 0;
    size_t read_ptr_ = 0;
    size_t write_ptr_ = 0;
    std::vector<FlitData> buffer_;
};

//=============================================================================
// PacketAssemblyState - Tracks packet being received
//=============================================================================
struct PacketAssemblyState {
    enum class State { IDLE, RECEIVING, COMPLETE };

    State state = State::IDLE;
    NoCPacket packet;           // Packet metadata (extracted from head flit)
    uint16_t total_flits = 0;   // From header
    uint16_t flits_received = 0;

    bool is_complete() const {
        return state == State::COMPLETE ||
               (state == State::RECEIVING && flits_received >= total_flits);
    }

    void start_packet(const FlitData& head_flit);
    void receive_flit();
    void reset();
};

//=============================================================================
// InjectionState - Tracks packet being injected from L3
//=============================================================================
struct InjectionState {
    enum class State { IDLE, INJECTING };

    State state = State::IDLE;
    NoCPacket packet;           // Packet being injected
    uint64_t start_cycle = 0;   // When injection started
    uint16_t total_flits = 0;   // Total flits to inject
    uint16_t flits_injected = 0;// Progress

    bool is_active() const { return state == State::INJECTING; }

    bool is_complete() const {
        return state == State::INJECTING && flits_injected >= total_flits;
    }

    uint64_t estimated_completion_cycle() const {
        return start_cycle + total_flits;
    }

    // Returns true if injection finished this cycle
    bool step(uint64_t cycle, FlitBuffer& buffer);
};

//=============================================================================
// TransmissionState - Tracks packet being sent on output link
//=============================================================================
struct TransmissionState {
    enum class State { IDLE, SENDING };

    State state = State::IDLE;
    NoCPacket packet;
    uint16_t total_flits = 0;
    uint16_t flits_sent = 0;

    bool is_active() const { return state == State::SENDING; }
    bool is_complete() const { return flits_sent >= total_flits; }

    // Returns true if a flit was sent this cycle
    bool step(uint64_t cycle, uint32_t& credits, NoCLink* link);
};

//=============================================================================
// NoCInputUnit - Complete input port with buffering and assembly
//=============================================================================
class NoCInputUnit {
public:
    NoCInputUnit(PortDirection dir, const NoCConfig& config);

    // Flit reception from link
    bool can_receive_flit() const;
    void receive_flit(const FlitData& flit, uint64_t cycle);

    // For LOCAL port: injection from L3
    bool can_start_injection() const;
    bool start_injection(const NoCPacket& packet, uint64_t cycle);

    // Packet ready for routing?
    bool has_ready_packet() const;
    const NoCPacket* peek_ready_packet() const;
    NoCPacket pop_ready_packet();

    // Credit management
    uint32_t credits_to_return() const { return credits_to_return_; }
    void clear_credits_returned() { credits_to_return_ = 0; }

    // Simulation step
    void step(uint64_t cycle);

    // Statistics
    struct Stats {
        uint64_t flits_received = 0;
        uint64_t packets_received = 0;
        uint64_t stall_cycles = 0;      // Cycles blocked due to full buffer
        uint64_t injection_cycles = 0;   // Total cycles spent injecting
    };
    const Stats& stats() const { return stats_; }

private:
    PortDirection direction_;
    const NoCConfig& config_;

    FlitBuffer flit_buffer_;
    PacketAssemblyState assembly_;
    InjectionState injection_;      // Only used for LOCAL port

    std::queue<NoCPacket> ready_queue_;  // Assembled packets ready to route
    uint32_t credits_to_return_ = 0;

    Stats stats_;
};

//=============================================================================
// NoCOutputUnit - Complete output port with buffering and transmission
//=============================================================================
class NoCOutputUnit {
public:
    NoCOutputUnit(PortDirection dir, const NoCConfig& config);

    // Connect to downstream link
    void connect_link(NoCLink* link);

    // Accept flit from crossbar
    bool can_accept_flit() const;
    void accept_flit(const FlitData& flit);

    // Start transmitting a packet (store-and-forward: all flits buffered)
    bool can_start_transmission() const;
    void start_transmission(NoCPacket packet);

    // Credit management
    void receive_credit();
    uint32_t credits_available() const { return credits_; }

    // Simulation step
    void step(uint64_t cycle);

    // For LOCAL port: packet ejection to L3
    bool has_packet_for_ejection() const;
    NoCPacket eject_packet();

private:
    PortDirection direction_;
    const NoCConfig& config_;

    FlitBuffer output_buffer_;
    TransmissionState transmission_;
    NoCLink* link_ = nullptr;

    uint32_t credits_ = 0;  // Credits from downstream

    std::queue<NoCPacket> ejection_queue_;  // For LOCAL port
};

//=============================================================================
// NoCRouterV2 - Corrected router with proper bandwidth modeling
//=============================================================================
class NoCRouterV2 {
public:
    NoCRouterV2(uint8_t id, const NoCConfig& config);

    uint8_t id() const { return id_; }
    uint8_t row() const { return row_; }
    uint8_t col() const { return col_; }

    // Input/Output unit access
    NoCInputUnit& input(PortDirection dir);
    NoCOutputUnit& output(PortDirection dir);

    // L3 interface
    bool can_inject(uint64_t cycle) const;
    bool inject_packet(const NoCPacket& packet, uint64_t cycle);

    bool has_ejection_ready() const;
    NoCPacket eject_packet();

    // Connect to neighbor routers
    void connect_output(PortDirection dir, NoCLink* link);
    void connect_input(PortDirection dir, NoCRouterV2* upstream);

    // Simulation
    void step(uint64_t cycle);

    // Statistics
    struct Stats {
        uint64_t packets_injected = 0;
        uint64_t packets_ejected = 0;
        uint64_t packets_forwarded = 0;
        uint64_t arbitration_conflicts = 0;
        uint64_t total_latency = 0;     // Sum of all packet latencies
    };
    const Stats& stats() const { return stats_; }

private:
    uint8_t id_, row_, col_;
    const NoCConfig& config_;

    std::array<NoCInputUnit, 5> inputs_;   // N, S, E, W, L
    std::array<NoCOutputUnit, 5> outputs_; // N, S, E, W, L

    Stats stats_;

    // Routing
    PortDirection compute_route(uint8_t dst_router) const;

    // Arbitration
    void arbitrate_and_switch(uint64_t cycle);
};
```

---

## Simulation Loop

```cpp
void NoCv2::step(uint64_t cycle) {
    // Phase 1: All routers process their input units
    //          (receive flits, assemble packets, handle injections)
    for (auto& router : routers_) {
        for (auto dir : {NORTH, SOUTH, EAST, WEST, LOCAL}) {
            router.input(dir).step(cycle);
        }
    }

    // Phase 2: All routers perform arbitration and crossbar switching
    for (auto& router : routers_) {
        router.arbitrate_and_switch(cycle);
    }

    // Phase 3: All output units transmit flits onto links
    for (auto& router : routers_) {
        for (auto dir : {NORTH, SOUTH, EAST, WEST, LOCAL}) {
            router.output(dir).step(cycle);
        }
    }

    // Phase 4: Links deliver flits to downstream routers
    for (auto& link : all_links_) {
        if (link.has_flit_in_flight()) {
            auto& flit = link.get_arriving_flit();
            auto& dst_router = routers_[link.dst_router_id()];
            dst_router.input(opposite(link.direction())).receive_flit(flit, cycle);
        }
    }

    // Phase 5: Credit returns flow upstream
    for (auto& router : routers_) {
        for (auto dir : {NORTH, SOUTH, EAST, WEST}) {
            uint32_t credits = router.input(dir).credits_to_return();
            if (credits > 0) {
                auto* upstream = get_neighbor(router.id(), opposite(dir));
                if (upstream) {
                    upstream->output(dir).receive_credit();
                }
                router.input(dir).clear_credits_returned();
            }
        }
    }
}
```

---

## Expected Trace Output (Corrected)

With proper modeling, injecting 4 tiles from R[0,0]:

```csv
cycle,type,router_id,port,packet_seq,flits_progress,total_flits
0,INJECT_START,0,L,0,0,4096
1,INJECT_PROGRESS,0,L,0,1,4096
...
4095,INJECT_COMPLETE,0,L,0,4096,4096
4096,INJECT_START,0,L,1,0,4096
...
8191,INJECT_COMPLETE,0,L,1,4096,4096
8192,INJECT_START,0,L,2,0,4096
...
```

Chrome Trace will show **sequential** injection:
```
R[0,0]: A[0,0]████████████────────────────────────────
R[0,0]: A[0,1]────────────████████████────────────────
R[0,0]: A[0,2]────────────────────────████████████────
R[0,0]: A[0,3]────────────────────────────────────████
```

---

## Questions for Review

1. **Buffer sizing**: Is 8 flits (512B) appropriate for input buffers? With 256KB tiles, this is tiny - relies on credit flow control to prevent overflow.

2. **Store-and-forward vs Wormhole**: Current design is store-and-forward (simpler). Wormhole would reduce latency but adds complexity. Which do you prefer?

3. **Virtual channels**: Not included in this design. Needed for adaptive routing to avoid deadlock. Required?

4. **DMA ports**: Currently simplified (inject anywhere). Should we model edge-only DMA injection properly?

5. **Tile data**: We only track metadata, not actual bytes. Is this sufficient or do you need data integrity verification?
