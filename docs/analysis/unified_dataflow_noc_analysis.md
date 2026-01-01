# Unified Dataflow NoC Analysis

## Executive Summary

This analysis compares the current NoC implementation with a proposed unified dataflow approach where the network operates on the same principles as the compute tiles: tagged tokens, local decision-making, and self-organizing behavior.

**Key Insight**: The compute fabric is already a push-based dataflow machine with local tag matching. The NoC should be the same - not a separate architectural paradigm with complex routing algorithms, but another layer of the hierarchical Domain Flow Graph.

---

## Current Architecture Analysis

### Compute Tile Model (Dataflow)

The compute tiles follow elegant dataflow principles:

```
┌─────────────────────────────────────────────────────────┐
│  COMPUTE TILE DATAFLOW MODEL                            │
│                                                         │
│  Token arrives → Tag matches? → Trigger computation     │
│       ↓              ↓                   ↓              │
│  TileDescriptor   Local check      Execute & emit       │
│  (tensor,m,n,k)   "Is this mine?"  new tagged result    │
└─────────────────────────────────────────────────────────┘
```

**Characteristics**:
- **Push-based**: Data tokens flow without request/response
- **Tagged**: Every tile carries `(tensor, m_tile, n_tile, k_tile)` identity
- **Local decisions**: Each tile checks "is this for me?" via trigger channels
- **Self-organizing**: Computation fires when dependencies arrive
- **No centralized control**: Synchronization emerges from token flow

### Current NoC Model (Traditional Router)

The current router implementation follows a different paradigm:

```
┌─────────────────────────────────────────────────────────┐
│  CURRENT NOC MODEL                                      │
│                                                         │
│  Flit arrives → Route calculation → Path reservation    │
│       ↓              ↓                    ↓             │
│  HEAD flit      XY algorithm          State machine     │
│  dst_router     "Which port?"         IDLE→RESERVED     │
│                      ↓                                  │
│               Arbitration logic                         │
│               Credit management                         │
│               Conflict resolution                       │
└─────────────────────────────────────────────────────────┘
```

**Characteristics**:
- **Routing algorithms**: XY, YX, adaptive variants
- **Path reservation**: HEAD reserves, BODY follows, TAIL releases
- **State machines**: InjectionState, PathReservation with explicit states
- **Credit-based flow control**: Separate tracking mechanism
- **Arbitration**: Complex multi-phase logic for conflicts

### The Paradigm Mismatch

| Aspect | Compute Tiles | Current NoC |
|--------|---------------|-------------|
| **Decision** | "Is this token for me?" | "Which port to forward?" |
| **Matching** | Tag-based (signature) | Address-based (router ID) |
| **State** | Minimal (trigger flags) | Complex (path reservation) |
| **Flow control** | Implicit (dataflow) | Explicit (credits) |
| **Coordination** | Self-organizing | Algorithmic routing |

---

## Proposed Unified Dataflow NoC

### Core Principle

> "A tile gets an identifying tag and a destination, then is simply pushed into the network. Each router node knows if the tile is meant for its L3 or needs forwarding."

This is exactly how compute tiles work - they receive tagged data and make simple local decisions.

### Unified Model

```
┌─────────────────────────────────────────────────────────┐
│  UNIFIED DATAFLOW NOC                                   │
│                                                         │
│  Token arrives → Local match? → Consume or Forward      │
│       ↓              ↓              ↓                   │
│  (tile_tag,      "dst == me?"   Yes: deliver to L3     │
│   destination)                  No:  push to neighbor   │
└─────────────────────────────────────────────────────────┘
```

**The router becomes a simple dataflow node**:
```cpp
struct DataflowRouter {
    uint8_t my_id;                    // This router's identity

    void receive(TaggedTile token) {
        if (token.destination == my_id) {
            deliver_to_l3(token);      // Local consumption
        } else {
            push_toward(token);        // Forward toward destination
        }
    }

    void push_toward(TaggedTile token) {
        // Simple direction: move closer to destination
        Direction dir = closer_to(token.destination);
        output_port[dir].push(token);
    }
};
```

### Hierarchical Domain Flow Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL DOMAIN FLOW                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LEVEL 2: Tile Data Movement (NoC Layer)                │    │
│  │                                                          │    │
│  │  Domains = Tiles (64KB-4MB blocks)                      │    │
│  │  Tokens  = (TileDescriptor, destination_l3)             │    │
│  │  Nodes   = Routers (local match + forward)              │    │
│  │  Flow    = Push toward destination                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LEVEL 1: Element Data Movement (L3→L2→Compute)         │    │
│  │                                                          │    │
│  │  Domains = Elements/Vectors                              │    │
│  │  Tokens  = (value, tag)                                  │    │
│  │  Nodes   = Compute units (tag match → execute)          │    │
│  │  Flow    = Push to consumers                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Both levels: Tagged tokens + Local decisions + Push-based      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Comparison

### 1. Token Structure

**Current (Complex)**:
```cpp
struct Flit {
    FlitType type;           // HEAD, BODY, TAIL, HEAD_TAIL
    uint16_t packet_seq;     // Packet tracking
    uint8_t src_router;      // Source
    uint8_t dst_router;      // Destination
    uint16_t total_flits;    // Packet size
    TileDescriptor tile;     // Payload metadata
    uint64_t inject_cycle;   // Timing
};
```

**Proposed (Simple)**:
```cpp
struct DataflowFlit {
    TileTag tag;             // Tile identity (tensor, m, n, k)
    uint8_t destination;     // Target L3 ID
    uint16_t sequence;       // Flit position in tile
    uint16_t total;          // Total flits for this tile
    // Payload: 64 bytes of tile data
};
```

**Difference**: The proposed structure treats the flit as a tagged dataflow token. The `tag` is the tile's identity (which persists), and `destination` is where it should go. No separate tracking of HEAD/BODY/TAIL states - just sequence numbers.

### 2. Routing Decision

**Current (Algorithmic)**:
```cpp
PortDir WormholeRouter::route(uint8_t dst_router) {
    if (dst_router == id_) return LOCAL;

    uint8_t dst_row = dst_router / 4;
    uint8_t dst_col = dst_router % 4;

    // XY routing: X first, then Y
    if (dst_col > col_) return EAST;
    if (dst_col < col_) return WEST;
    if (dst_row > row_) return SOUTH;
    if (dst_row < row_) return NORTH;

    return LOCAL; // Error
}
```

**Proposed (Tag Match)**:
```cpp
void DataflowRouter::process(DataflowFlit flit) {
    if (flit.destination == my_id_) {
        // This token is for me - consume it
        deliver_to_l3(flit);
    } else {
        // Not for me - push toward destination
        Direction dir = direction_toward(flit.destination);
        neighbors_[dir].receive(flit);
    }
}
```

**Difference**: The routing "algorithm" becomes a simple predicate: "Is this for me?" This matches exactly how compute tiles decide whether to process a token.

### 3. Flow Control

**Current (Credit-Based)**:
```cpp
struct OutputPort {
    uint8_t credits;
    std::queue<Flit> buffer;

    bool can_send() { return credits > 0 && !buffer.empty(); }
    void send() {
        send_flit(buffer.front());
        buffer.pop();
        credits--;
    }
    void receive_credit() { credits++; }
};
```

**Proposed (Dataflow Back-Pressure)**:
```cpp
struct DataflowPort {
    std::queue<DataflowFlit> buffer;
    static constexpr size_t MAX_DEPTH = 8;

    bool can_accept() { return buffer.size() < MAX_DEPTH; }

    void receive(DataflowFlit flit) {
        if (can_accept()) {
            buffer.push(flit);
        }
        // Back-pressure: upstream naturally stalls
    }
};
```

**Difference**: Instead of explicit credit messages, back-pressure emerges naturally from buffer fullness - exactly like dataflow execution where a node stalls when its output can't be consumed.

### 4. State Management

**Current (Complex State Machines)**:
```cpp
struct PathReservation {
    enum class State { IDLE, RESERVED };
    State state;
    PortDir input_port;
    uint16_t packet_seq;
    uint16_t flits_remaining;
};

struct InjectionState {
    enum class State { IDLE, INJECTING };
    State state;
    uint16_t packet_seq;
    TileDescriptor tile;
    // ... many more fields
};
```

**Proposed (Stateless Forwarding)**:
```cpp
// No per-path state needed!
// Each flit carries its own identity and destination.
// Router simply: receive → match → forward or consume
```

**Difference**: The proposed model eliminates path reservation entirely. Each flit is self-describing - it knows where it's going and what tile it belongs to. Routers don't need to remember which path a packet is using.

### 5. Multi-Flit Handling

**Current (Path Reservation)**:
```
HEAD flit → Reserve path → BODY flits follow reserved path → TAIL releases
```

**Proposed (Tagged Reassembly)**:
```
All flits carry (tile_tag, sequence, total)
Destination L3 reassembles by matching tile_tag
No path reservation needed - flits can take different paths!
```

**Difference**: The proposed model allows flits of the same tile to take different routes (if needed for load balancing). The destination reassembles by tag matching, just like dataflow nodes collect their input tokens.

---

## Benefits of Unified Approach

### 1. Conceptual Simplicity

**Current**: Two different mental models
- Compute: dataflow, tag matching, triggers
- Network: routing algorithms, path reservation, credits

**Proposed**: One mental model
- Everything is dataflow: tagged tokens, local decisions, push-based

### 2. Implementation Simplicity

**Current Router Complexity**:
```
- Path reservation state machine
- Credit tracking per port
- Arbitration logic (HEAD vs continuing paths)
- Flit type handling (HEAD, BODY, TAIL)
- Packet sequence tracking
- Conflict resolution
```

**Proposed Router Complexity**:
```
- Tag match (is this for me?)
- Direction lookup (which way is destination?)
- Buffer management (simple queue)
```

### 3. Robustness

**Current**: Path reservation creates dependencies
- If a path is reserved, other traffic is blocked
- HEAD flit loss would leave path permanently reserved
- Complex recovery logic needed

**Proposed**: Stateless forwarding is inherently robust
- No reservation state to corrupt
- Each flit is self-describing
- Lost flit affects only that flit, not the path

### 4. Flexibility

**Current**: Fixed routing algorithms
- XY routing is deterministic - same path every time
- No dynamic load balancing

**Proposed**: Natural load balancing
- "Push toward destination" can consider link utilization
- Different flits can take different paths
- Emergent load distribution

---

## Design Sketch: Dataflow NoC

### Router Structure

```cpp
class DataflowRouter {
    uint8_t id_;
    uint8_t row_, col_;

    // Neighbors (null if at edge)
    DataflowRouter* north_;
    DataflowRouter* south_;
    DataflowRouter* east_;
    DataflowRouter* west_;

    // Local L3 tile connection
    L3Tile* local_l3_;

    // Input buffers (one per direction + local injection)
    std::array<FlitQueue, 5> inputs_;  // N, S, E, W, LOCAL

public:
    void step(uint64_t cycle) {
        // Process all input buffers
        for (auto& input : inputs_) {
            while (!input.empty() && can_forward(input.front())) {
                process_flit(input.pop());
            }
        }
    }

private:
    void process_flit(DataflowFlit flit) {
        if (flit.destination == id_) {
            // Tag match! Deliver locally
            local_l3_->receive_flit(flit);
        } else {
            // Forward toward destination
            Direction dir = toward(flit.destination);
            get_neighbor(dir)->inject(flit);
        }
    }

    Direction toward(uint8_t dst) {
        // Simple: move closer in X first, then Y
        // (Or could be adaptive based on neighbor load)
        uint8_t dst_row = dst / cols_;
        uint8_t dst_col = dst % cols_;

        if (dst_col > col_) return EAST;
        if (dst_col < col_) return WEST;
        if (dst_row > row_) return SOUTH;
        if (dst_row < row_) return NORTH;

        return LOCAL; // Shouldn't reach here
    }
};
```

### Flit Structure

```cpp
struct DataflowFlit {
    // Tile identity (the "tag" in dataflow terms)
    TileTag tag;

    // Destination (where this token should be consumed)
    uint8_t destination;

    // Reassembly info
    uint16_t flit_index;    // Which flit of the tile (0, 1, 2, ...)
    uint16_t total_flits;   // Total flits in this tile

    // Payload
    std::array<uint8_t, 56> data;  // 56 bytes payload + 8 bytes header = 64
};

struct TileTag {
    TensorId tensor;        // A, B, C, etc.
    uint16_t m_tile;
    uint16_t n_tile;
    uint16_t k_tile;

    bool operator==(const TileTag& other) const {
        return tensor == other.tensor &&
               m_tile == other.m_tile &&
               n_tile == other.n_tile &&
               k_tile == other.k_tile;
    }
};
```

### L3 Tile Reassembly

```cpp
class L3Tile {
    // Pending tile reassembly (tag → partial tile)
    std::unordered_map<TileTag, PartialTile> pending_;

public:
    void receive_flit(DataflowFlit flit) {
        auto& partial = pending_[flit.tag];

        if (partial.empty()) {
            // First flit of this tile
            partial.init(flit.total_flits);
        }

        partial.add_flit(flit.flit_index, flit.data);

        if (partial.complete()) {
            // All flits received - tile is ready!
            emit_trigger(TriggerChannel::TILE_READY, flit.tag);
            // Move to L3 buffer
            store_tile(flit.tag, partial.assemble());
            pending_.erase(flit.tag);
        }
    }
};
```

---

## Comparison Summary

| Aspect | Current NoC | Proposed Dataflow NoC |
|--------|-------------|----------------------|
| **Mental Model** | Traditional router | Dataflow node |
| **Decision Logic** | "Which port?" (routing algorithm) | "Is this mine?" (tag match) |
| **State** | Path reservation, credits | Stateless (per-flit) |
| **Flow Control** | Explicit credits | Implicit back-pressure |
| **Multi-flit** | Path reservation | Tag-based reassembly |
| **Robustness** | State corruption risk | Inherently robust |
| **Load Balancing** | Fixed paths | Potentially adaptive |
| **Conceptual Unity** | Different from compute | Same as compute |

---

## Alignment with KPU Philosophy

The KPU documentation describes the system as:

> "A push-based machine with tagged token matching for position-independent computation... distributed pattern matching across processing elements... self-coordinating execution through token flow."

The proposed unified NoC directly embodies this:

1. **Push-based**: Tiles are pushed into network, flow toward destination
2. **Tagged tokens**: Flits carry tile identity that persists through network
3. **Position-independent**: Any router can forward; destination does tag-match
4. **Distributed matching**: Each router makes local "is this mine?" decision
5. **Self-coordinating**: No central router controller; flow emerges from local decisions

---

## Implementation Path

### Phase 1: Minimal Dataflow Router
- Simple tag-match + forward logic
- Buffer-based back-pressure
- No credits, no path reservation

### Phase 2: L3 Reassembly
- Tag-based flit collection
- Trigger emission on tile completion
- Integration with BlockMover

### Phase 3: Adaptive Forwarding
- Consider neighbor buffer depth
- Load-balanced "toward destination" decisions
- Optional: multiple paths for same tile

### Phase 4: Remove Old NoC
- Replace WormholeRouter with DataflowRouter
- Simplify NoC to collection of DataflowRouters
- Update tests and examples

---

## Conclusion

The current NoC is built on traditional router design: routing algorithms, path reservation, credit-based flow control. While functional, it represents a different paradigm from the compute tiles' dataflow execution model.

The proposed unified approach treats the NoC as another layer of the Domain Flow Graph:
- **Same principles**: Tagged tokens, local tag-matching, push-based flow
- **Simpler implementation**: No path reservation, no credits, no arbitration state
- **Better alignment**: One mental model for the entire system
- **More robust**: Stateless forwarding, self-describing flits

This creates a **hierarchical Domain Flow system** where:
- Level 2 (NoC): Data movement of tiles between L3s
- Level 1 (Compute): Data movement of elements within compute tiles

Both levels operate on the same principle: **tagged tokens flowing through nodes that make simple local decisions based on tag matching**.

This is conceptually cleaner, easier to reason about, and more aligned with the KPU's fundamental architecture.
