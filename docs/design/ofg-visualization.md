# OFG Execution Visualization Design

## Overview

This document describes the visualization system for Operand Flow Graph (OFG) execution on the KPU. The visualization shows how operands flow through the memory hierarchy driven by concurrent, dataflow-triggered OFG executors.

## Core Concept: Dataflow-Driven Execution

The KPU executes matrix operations through a **dataflow-driven** model, not a traditional state machine. Key principles:

1. **Operand Presence Triggers Execution** - OFG nodes fire when their input operands become ready
2. **Concurrent Executors** - Each memory level has independent OFG executors running in parallel
3. **Event-Driven Coordination** - `TILE_READY` events propagate through the hierarchy
4. **No Central Scheduler** - Each level watches for operands and reacts autonomously

## Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST MEMORY                               │
│   Input tensors (A, B) and output tensor (C)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │ DMA Engine
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     L3 TILE ARRAY (4 × 128KB)                   │
│   On-chip SRAM cache for tensor tiles                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Block Mover
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     L2 BANK ARRAY (8 × 64KB)                    │
│   Operand buffers feeding compute                               │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Streamer
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     L1 STREAM BUFFERS                           │
│   Edge buffers feeding systolic array                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Push triggers compute
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     COMPUTE TILE (Passive)                      │
│   Systolic array executes when L1 pushes operands               │
└─────────────────────────────────────────────────────────────────┘
```

## Concurrent OFG Executors

Each memory level has its own OFG executor running concurrently:

### DMA Level (Host ↔ L3)
- **Watches**: Host memory for input tensors, L3 for result tiles
- **Operations**: `LOAD` (Host→L3), `STORE` (L3→Host)
- **Triggers**: Buffer availability, tile completion

### BlockMover Level (L3 ↔ L2)
- **Watches**: L3 tiles for operand arrival, L2 for result tiles
- **Operations**: `PUSH_TO_L2`, `PULL_FROM_L2`, mesh forwarding
- **Triggers**: `TILE_READY@L3`, `TILE_READY@L2`

### Streamer Level (L2 ↔ L1 ↔ Compute)
- **Watches**: L2 banks for operand tiles, accumulator for results
- **Operations**: `FEED_WEST`, `FEED_NORTH`, `DRAIN`, `MATMUL`
- **Triggers**: `TILE_READY@L2`, `ACC_READY`

### Compute Tile (Passive)
- **Watches**: L1 stream buffers for incoming data
- **Operations**: MAC operations in systolic array
- **Triggers**: L1 buffer push (data-driven, not scheduled)

## Dataflow Trigger Chain

The execution follows a trigger chain where each completed operation produces events that trigger downstream operations:

```
LOAD A,B from Host
        │
        ▼
TILE_READY(A@L3), TILE_READY(B@L3)
        │
        ▼ triggers BlockMover
PUSH_TO_L2
        │
        ▼
TILE_READY(A@L2), TILE_READY(B@L2)
        │
        ▼ triggers Streamer
FEED_WEST(A), FEED_NORTH(B)
        │
        ▼
L1 buffer push
        │
        ▼ triggers Compute (passive)
MATMUL: C += A × B
        │
        ▼
ACC_READY
        │
        ▼ triggers Streamer
DRAIN(C→L2)
        │
        ▼
TILE_READY(C@L2)
        │
        ▼ triggers BlockMover
PULL_FROM_L2(C→L3)
        │
        ▼
TILE_READY(C@L3)
        │
        ▼ triggers DMA
STORE(C→Host)
```

## Visualization Requirements

### Visual Elements

1. **Memory Regions** - Rectangles showing Host, L3, L2, L1, Compute
2. **Tiles/Operands** - Colored rectangles (A=red, B=green, C=yellow)
3. **OFG Nodes** - State indicators showing WAITING→READY→FIRING→DONE
4. **Data Flow** - Animated arrows showing operand movement
5. **Trigger Events** - Flashing connections when events fire
6. **Timeline** - Cycle counter with playback controls

### Animation Behaviors

1. **Tile Movement** - Smooth animation of tiles between memory levels
2. **OFG State Changes** - Visual indication of node state transitions
3. **Concurrent Operations** - Multiple animations running in parallel
4. **Trigger Propagation** - Visual pulse showing event propagation
5. **Buffer Fill Levels** - Progress bars for L1 stream buffers

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  KPU OFG Execution Visualization                    [Controls]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HOST MEMORY                                                    │
│  ┌────────┐  ┌────────┐              ┌────────┐                │
│  │ A[M,K] │  │ B[K,N] │              │ C[M,N] │                │
│  └────────┘  └────────┘              └────────┘                │
│       │           │                       ▲                     │
│  ┌────┴───────────┴────┐          ┌──────┴──────┐              │
│  │    DMA OFG (load)   │          │ DMA OFG     │              │
│  │   ○───○───○         │          │ (store)     │              │
│  └─────────┬───────────┘          └─────────────┘              │
│            ▼                                                    │
│  L3 TILES  ┌──────┬──────┬──────┬──────┐                       │
│            │ [0]  │ [1]  │ [2]  │ [3]  │                       │
│            └──────┴──────┴──────┴──────┘                       │
│                    │                                            │
│  ┌─────────────────┴─────────────────┐                         │
│  │       BlockMover OFG              │                         │
│  │      ○───○───○───○                │                         │
│  └─────────────────┬─────────────────┘                         │
│                    ▼                                            │
│  L2 BANKS  ┌──┬──┬──┬──┬──┬──┬──┬──┐                           │
│            │0 │1 │2 │3 │4 │5 │6 │7 │                           │
│            └──┴──┴──┴──┴──┴──┴──┴──┘                           │
│                    │                                            │
│  ┌─────────────────┴─────────────────┐                         │
│  │        Streamer OFG               │                         │
│  │      ○───○───○───○───○            │                         │
│  └─────────────────┬─────────────────┘                         │
│                    ▼                                            │
│  L1 + COMPUTE                                                   │
│  West:[▓▓▓░░]  ┌─────────┐  South:[░░░░░]                      │
│  North:[▓▓░░]  │ Systolic│                                     │
│                │  Array  │                                      │
│                └─────────┘                                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Cycle: 42    [|◀] [▶] [▶|]  Speed: ━━━●━━━                    │
└─────────────────────────────────────────────────────────────────┘
```

## Trace File Format

The visualization consumes a JSON trace file with the following structure:

```json
{
  "metadata": {
    "generator": "kpu-sim",
    "version": "1.0",
    "timestamp": "2025-01-12T10:30:00Z",
    "config": {
      "l3_tiles": 4,
      "l2_banks": 8,
      "systolic_size": 16
    }
  },
  "tensors": [
    {"id": "A", "shape": [64, 32], "type": "input"},
    {"id": "B", "shape": [32, 64], "type": "input"},
    {"id": "C", "shape": [64, 64], "type": "output"}
  ],
  "events": [
    {
      "cycle": 0,
      "level": "DMA",
      "node_id": 0,
      "type": "node_fire",
      "operation": "LOAD",
      "operand": {"type": "TILE_A", "coord": [0, 0], "location": "HOST"}
    },
    {
      "cycle": 100,
      "level": "DMA",
      "node_id": 0,
      "type": "node_complete",
      "operation": "LOAD",
      "operand": {"type": "TILE_A", "coord": [0, 0], "location": "L3", "node": 0}
    },
    {
      "cycle": 100,
      "level": "DMA",
      "type": "tile_ready",
      "operand": {"type": "TILE_A", "coord": [0, 0], "location": "L3", "node": 0}
    },
    {
      "cycle": 101,
      "level": "BLOCK_MOVER",
      "node_id": 0,
      "type": "node_fire",
      "operation": "PUSH_TO_L2",
      "operand": {"type": "TILE_A", "coord": [0, 0], "location": "L3", "node": 0}
    }
  ]
}
```

### Event Types

| Type | Description |
|------|-------------|
| `node_fire` | OFG node begins execution |
| `node_complete` | OFG node finishes execution |
| `tile_ready` | Operand becomes available at a location |
| `buffer_available` | Buffer space becomes available |
| `trigger` | Event triggers downstream OFG |

### Execution Levels

| Level | Description |
|-------|-------------|
| `DMA` | DMA engine operations (Host ↔ L3) |
| `BLOCK_MOVER` | Block mover operations (L3 ↔ L2) |
| `STREAMER` | Streamer operations (L2 ↔ L1) |
| `COMPUTE` | Compute tile operations |

## Integration with Simulators

The visualization can receive data from multiple sources:

### 1. Behavioral Simulator
- Instant execution, functional correctness
- Generates simplified trace with operation sequence
- No timing, just dependencies

### 2. Transactional Simulator
- Approximate timing, transaction-level
- Generates trace with estimated latencies
- Good for architecture exploration

### 3. Temporal Simulator
- Cycle-accurate timing
- Generates detailed trace with exact cycles
- Full pipeline and contention modeling

### 4. Offline Trace File
- Pre-generated JSON file
- Can be from any simulator or hand-crafted
- Enables replay and debugging

## Implementation Notes

### Animation Engine
- Use requestAnimationFrame for smooth animation
- Interpolate tile positions between keyframes
- Support variable playback speed
- Allow step-by-step execution

### State Management
- Track current cycle
- Track operand positions
- Track OFG node states per level
- Track buffer fill levels

### Event Processing
- Process events in cycle order
- Handle multiple events per cycle
- Trigger animations for state changes
- Update visual elements reactively
