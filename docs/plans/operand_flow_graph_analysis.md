# Operand Flow Graph Analysis for Matrix Multiply

## Programming Model Principles

### Data-Flow Driven Execution

**Wrong approach (state-change driven):**
```
DMA: LOAD_TILE → EMIT(DMA_DONE)           ← "I finished loading"
BM:  WAIT(DMA_DONE) → PUSH_TO_L2          ← Waits for operation completion
```

**Correct approach (operand-presence driven):**
```
DMA: LOAD_TILE → produces TILE_READY(A[i,k] @ L3[0,j])
BM:  consumes TILE_READY(A[i,k] @ L3) → PUSH_TO_L2 → produces TILE_READY(A[i,k] @ L2)
```

The distinction:
- **State-change**: "The DMA is done" (focuses on component state)
- **Operand-presence**: "Tile A[i,k] is now at L3" (focuses on data availability)

### Coordination Events

| Event Type | Meaning | Direction |
|------------|---------|-----------|
| `TILE_READY(T @ Loc)` | Tile T is available at location Loc | Forward (producer → consumer) |
| `BUFFER_AVAILABLE(Loc)` | Buffer at Loc can accept a tile | Backward (consumer → producer) |

This gives us **forward flow** (operand ready) and **backpressure** (buffer available).

---

## Matrix Multiply: C[M,N] = A[M,K] × B[K,N]

### Tiling

```
Tile sizes: Tm × Tn × Tk
A tiles: A[i,k] where i ∈ [0, M/Tm), k ∈ [0, K/Tk)
B tiles: B[k,j] where k ∈ [0, K/Tk), j ∈ [0, N/Tn)
C tiles: C[i,j] where i ∈ [0, M/Tm), j ∈ [0, N/Tn)
```

### KPU Mesh Topology

```
        j=0     j=1     j=2     j=3
       ┌───┐   ┌───┐   ┌───┐   ┌───┐
i=0    │0,0│───│0,1│───│0,2│───│0,3│  ← B tiles enter from North
       └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
         │       │       │       │
       ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
i=1    │1,0│───│1,1│───│1,2│───│1,3│
       └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
         │       │       │       │
       ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
i=2    │2,0│───│2,1│───│2,2│───│2,3│
       └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
         │       │       │       │
       ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
i=3    │3,0│───│3,1│───│3,2│───│3,3│
       └───┘   └───┘   └───┘   └───┘
         ↑
    A tiles enter from West
```

---

## Schedule 1: C-Stationary (Output-Stationary)

### Principle
- C[i,j] **stays** at L3[i mod 4, j mod 4] throughout computation
- A[i,k] tiles flow **West → East** (row broadcast)
- B[k,j] tiles flow **North → South** (column broadcast)
- Accumulation: C[i,j] += A[i,k] × B[k,j] for all k

### Tile Flow Pattern

```
For k = 0, 1, 2, ... K_tiles-1:

  A[0,k]→  ┌───┐   ┌───┐   ┌───┐   ┌───┐
           │C00│→A→│C01│→A→│C02│→A→│C03│
  B[k,0]↓  └─↓─┘   └─↓─┘   └─↓─┘   └─↓─┘
             B       B       B       B
  A[1,k]→  ┌─↓─┐   ┌─↓─┐   ┌─↓─┐   ┌─↓─┐
           │C10│→A→│C11│→A→│C12│→A→│C13│
  B[k,1]↓  └─↓─┘   └─↓─┘   └─↓─┘   └─↓─┘
             B       B       B       B
           ... (continues for i=2,3)
```

### Level 1: Memory → L3 (DMA Operand Flow Graph)

**DMA Engine for A tiles (West edge, loads A[i,k] to L3[i,0]):**

```
┌─────────────────────────────────────────────────────────────┐
│  DMA Engine 0: A-tile Loader                                │
│                                                             │
│  for i in 0..3:                                             │
│    for k in 0..K_tiles-1:                                   │
│                                                             │
│      ┌──────────────────┐                                   │
│      │ BUFFER_AVAILABLE │─────┐                             │
│      │   (L3[i,0])      │     │                             │
│      └──────────────────┘     ▼                             │
│                          ┌─────────┐    ┌────────────────┐  │
│                          │  LOAD   │───▶│ TILE_READY     │  │
│                          │ A[i,k]  │    │ (A[i,k] @ L3)  │  │
│                          └─────────┘    └────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Operand Flow Graph (formal):
  Nodes:
    N1: WAIT(BUFFER_AVAILABLE(L3[i,0]))
    N2: FIRE(LOAD A[i,k] from Memory to L3[i,0])
    N3: PRODUCE(TILE_READY(A[i,k] @ L3[i,0]))

  Edges:
    N1 → N2 (enable load when buffer available)
    N2 → N3 (produce ready signal after load)
```

**DMA Engine for B tiles (North edge, loads B[k,j] to L3[0,j]):**

```
┌─────────────────────────────────────────────────────────────┐
│  DMA Engine 1: B-tile Loader                                │
│                                                             │
│  for j in 0..3:                                             │
│    for k in 0..K_tiles-1:                                   │
│                                                             │
│      ┌──────────────────┐                                   │
│      │ BUFFER_AVAILABLE │─────┐                             │
│      │   (L3[0,j])      │     │                             │
│      └──────────────────┘     ▼                             │
│                          ┌─────────┐    ┌────────────────┐  │
│                          │  LOAD   │───▶│ TILE_READY     │  │
│                          │ B[k,j]  │    │ (B[k,j] @ L3)  │  │
│                          └─────────┘    └────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Level 2: L3 → L2 (BlockMover Operand Flow Graph)

**BlockMover at L3[i,j] (interior node, receives A from West, B from North):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BlockMover at L3[i,j] where i,j ∈ [0,3]                                    │
│                                                                             │
│  for k in 0..K_tiles-1:                                                     │
│                                                                             │
│    ┌─────────────────┐                                                      │
│    │ TILE_READY      │──────┐                                               │
│    │ (A[i,k] @ L3)   │      │      ┌────────────┐    ┌──────────────────┐   │
│    └─────────────────┘      ├─────▶│ PUSH A[i,k]│───▶│ TILE_READY       │   │
│                             │      │ to L2[0]   │    │ (A[i,k] @ L2)    │   │
│    ┌─────────────────┐      │      └────────────┘    └──────────────────┘   │
│    │ L2_BUFFER_AVAIL │──────┘                                               │
│    │ (bank 0)        │                                                      │
│    └─────────────────┘                                                      │
│                                                                             │
│    ┌─────────────────┐                                                      │
│    │ TILE_READY      │──────┐                                               │
│    │ (B[k,j] @ L3)   │      │      ┌────────────┐    ┌──────────────────┐   │
│    └─────────────────┘      ├─────▶│ PUSH B[k,j]│───▶│ TILE_READY       │   │
│                             │      │ to L2[1]   │    │ (B[k,j] @ L2)    │   │
│    ┌─────────────────┐      │      └────────────┘    └──────────────────┘   │
│    │ L2_BUFFER_AVAIL │──────┘                                               │
│    │ (bank 1)        │                                                      │
│    └─────────────────┘                                                      │
│                                                                             │
│    ══════════════════════════════════════════════════════════════════════   │
│    Tile Forwarding (after local consumption):                               │
│                                                                             │
│    ┌─────────────────┐      ┌────────────┐    ┌────────────────────┐        │
│    │ A consumed      │─────▶│ SEND_EAST  │───▶│ TILE_READY         │        │
│    │ (if j < 3)      │      │ A[i,k]     │    │ (A[i,k]@L3[i,j+1]) │        │
│    └─────────────────┘      └────────────┘    └────────────────────┘        │
│                                                                             │
│    ┌─────────────────┐      ┌────────────┐    ┌────────────────────┐        │
│    │ B consumed      │─────▶│ SEND_SOUTH │───▶│ TILE_READY         │        │
│    │ (if i < 3)      │      │ B[k,j]     │    │ (B[k,j]@L3[i+1,j]) │        │
│    └─────────────────┘      └────────────┘    └────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Operand Flow Graph (formal):
  Nodes:
    // A path
    N1: WAIT(TILE_READY(A[i,k] @ L3[i,j]) ∧ L2_BUFFER_AVAIL(bank0))
    N2: FIRE(PUSH A[i,k] from L3 to L2[bank0])
    N3: PRODUCE(TILE_READY(A[i,k] @ L2))
    N4: FIRE(SEND_EAST A[i,k])  // if j < 3
    N5: PRODUCE(TILE_READY(A[i,k] @ L3[i,j+1]))

    // B path (symmetric)
    M1: WAIT(TILE_READY(B[k,j] @ L3[i,j]) ∧ L2_BUFFER_AVAIL(bank1))
    M2: FIRE(PUSH B[k,j] from L3 to L2[bank1])
    M3: PRODUCE(TILE_READY(B[k,j] @ L2))
    M4: FIRE(SEND_SOUTH B[k,j])  // if i < 3
    M5: PRODUCE(TILE_READY(B[k,j] @ L3[i+1,j]))

  Edges:
    N1 → N2 → N3, N2 → N4 → N5
    M1 → M2 → M3, M2 → M4 → M5
```

### Level 3: L2 → L1 (Streamer Operand Flow Graph)

**Streamer at Compute Tile [i,j]:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Streamer at CT[i,j]                                                        │
│                                                                             │
│  // C[i,j] stays in accumulator throughout                                  │
│                                                                             │
│  for k in 0..K_tiles-1:                                                     │
│                                                                             │
│    ┌─────────────────┐                                                      │
│    │ TILE_READY      │───────┐                                              │
│    │ (A[i,k] @ L2)   │       │                                              │
│    └─────────────────┘       │                                              │
│                              │      ┌────────────┐                          │
│                              ├─────▶│ FEED A,B   │────┐                     │
│                              │      │ to L1      │    │                     │
│    ┌─────────────────┐       │      └────────────┘    │                     │
│    │ TILE_READY      │───────┘                        │                     │
│    │ (B[k,j] @ L2)   │                                │                     │
│    └─────────────────┘                                ▼                     │
│                                                 ┌───────────┐               │
│                                                 │ COMPUTE   │               │
│                                                 │C += A × B │               │
│                                                 └─────┬─────┘               │
│                                                       │                     │
│                                                       ▼                     │
│                                            ┌──────────────────┐             │
│                                            │ SIGNAL           │             │
│                                            │ (A,B consumed)   │             │
│                                            └──────────────────┘             │
│                                                                             │
│  end for                                                                    │
│                                                                             │
│  // After all k iterations:                                                 │
│  ┌────────────┐    ┌──────────────────┐                                     │
│  │ DRAIN C    │───▶│ TILE_READY       │                                     │
│  │ from L1    │    │ (C[i,j] @ L2)    │                                     │
│  └────────────┘    └──────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Operand Flow Graph (formal):
  Nodes:
    N1: WAIT(TILE_READY(A[i,k] @ L2) ∧ TILE_READY(B[k,j] @ L2))
    N2: FIRE(FEED A[i,k] to systolic west edge)
    N3: FIRE(FEED B[k,j] to systolic north edge)
    N4: FIRE(COMPUTE C[i,j] += A[i,k] × B[k,j])  // triggered by data arrival
    N5: PRODUCE(TILE_CONSUMED(A[i,k]) ∧ TILE_CONSUMED(B[k,j]))

    // After loop:
    D1: WAIT(k == K_tiles)  // all partial products computed
    D2: FIRE(DRAIN C[i,j] from accumulator to L2)
    D3: PRODUCE(TILE_READY(C[i,j] @ L2))
```

---

## Schedule 2: A-Stationary

### Principle
- A[i,k] **stays** at L3[i mod 4, k mod 4]
- B[k,j] tiles flow through (reused across i dimension)
- C[i,j] is accumulated by flowing partial sums

### Tile Flow Pattern

```
A tiles are pre-loaded and stationary:

  A[0,0]   A[0,1]   A[0,2]   A[0,3]
  ┌───┐   ┌───┐   ┌───┐   ┌───┐
  │   │←B─│   │←B─│   │←B─│   │←B  B[k,j] enters from East
  └─↓─┘   └─↓─┘   └─↓─┘   └─↓─┘
    C       C       C       C     C partial sums flow South
  A[1,0]   A[1,1]   A[1,2]   A[1,3]
  ┌─↓─┐   ┌─↓─┐   ┌─↓─┐   ┌─↓─┐
  │   │←B─│   │←B─│   │←B─│   │←B
  └─↓─┘   └─↓─┘   └─↓─┘   └─↓─┘
    C       C       C       C
  ...
```

### Loop Structure
```
for j in 0..N_tiles-1:      // For each output column
  for i in 0..M_tiles-1:    // For each output row
    C[i,j] = 0
    for k in 0..K_tiles-1:  // Reduction along K
      C[i,j] += A[i,k] × B[k,j]   // A[i,k] is local, B[k,j] flows in
```

### Level 2: BlockMover Operand Flow Graph (A-Stationary)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BlockMover at L3[i,k] holding A[i,k] stationary                            │
│                                                                             │
│  // A[i,k] is loaded once and stays                                         │
│  WAIT(TILE_READY(A[i,k] @ L3)) → PUSH A[i,k] to L2[0] (once)                │
│                                                                             │
│  for j in 0..N_tiles-1:                                                     │
│                                                                             │
│    ┌─────────────────┐      ┌──────────────┐    ┌────────────────────┐      │
│    │ TILE_READY      │─────▶│ PUSH B[k,j]  │───▶│ TILE_READY         │      │
│    │ (B[k,j] @ L3)   │      │   to L2[1]   │    │ (B[k,j] @ L2)      │      │
│    └─────────────────┘      └──────────────┘    └────────────────────┘      │
│                                    │                                        │
│                                    ▼                                        │
│                             ┌──────────────┐    ┌────────────────────┐      │
│                             │ FORWARD B    │───▶│ TILE_READY         │      │
│                             │ to L3[i,k-1] │    │ (B[k,j]@L3[i,k-1]) │      │
│                             └──────────────┘    └────────────────────┘      │
│                                                                             │
│    // Receive partial C, accumulate, forward                                │
│    ┌─────────────────┐                                                      │
│    │ TILE_READY      │───┐                                                  │
│    │ (C_partial from │   │   ┌────────────────┐                             │
│    │  North)         │   ├──▶│ ACCUMULATE     │                             │
│    └─────────────────┘   │   │ C += A×B       │                             │
│                          │   └───────┬────────┘                             │
│    ┌─────────────────┐   │           │                                      │
│    │ COMPUTE_DONE    │───┘           ▼                                      │
│    └─────────────────┘        ┌────────────┐    ┌──────────────────┐        │
│                               │ SEND_SOUTH │───▶│ TILE_READY       │        │
│                               │ C_partial  │    │ (C@L3[i+1,k])    │        │
│                               └────────────┘    └──────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Key difference from C-stationary:
  - A tile is loaded ONCE and reused for all j iterations
  - B tiles flow horizontally (West ← East for this diagram)
  - C partial sums flow vertically (North → South)
  - Each node contributes to partial sum before forwarding
```

---

## Schedule 3: B-Stationary

### Principle
- B[k,j] **stays** at L3[k mod 4, j mod 4]
- A[i,k] tiles flow through (reused across j dimension)
- C[i,j] is accumulated by flowing partial sums

### Tile Flow Pattern

```
B tiles are pre-loaded and stationary:

       B[0,0]   B[0,1]   B[0,2]   B[0,3]
        ↓        ↓        ↓        ↓
  A→   ┌───┐   ┌───┐   ┌───┐   ┌───┐
       │   │──C│   │──C│   │──C│   │→C  C partials flow East
  A→   └─↓─┘   └─↓─┘   └─↓─┘   └─↓─┘
       B[1,0]   B[1,1]   B[1,2]   B[1,3]
  A→   ┌─↓─┐   ┌─↓─┐   ┌─↓─┐   ┌─↓─┐
       │   │──C│   │──C│   │──C│   │→C
  A→   └─↓─┘   └─↓─┘   └─↓─┘   └─↓─┘
       ...

  A tiles enter from West, flow East
```

### Level 2: BlockMover Operand Flow Graph (B-Stationary)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BlockMover at L3[k,j] holding B[k,j] stationary                            │
│                                                                             │
│  // B[k,j] is loaded once and stays                                         │
│  WAIT(TILE_READY(B[k,j] @ L3)) → PUSH B[k,j] to L2[1] (once)                │
│                                                                             │
│  for i in 0..M_tiles-1:                                                     │
│                                                                             │
│    ┌─────────────────┐      ┌─────────────┐    ┌──────────────────┐         │
│    │ TILE_READY      │─────▶│ PUSH A[i,k] │───▶│ TILE_READY       │         │
│    │ (A[i,k] @ L3)   │      │ to L2[0]    │    │ (A[i,k] @ L2)    │         │
│    └─────────────────┘      └─────────────┘    └──────────────────┘         │
│                                    │                                        │
│                                    ▼                                        │
│                             ┌─────────────┐    ┌──────────────────┐         │
│                             │ FORWARD A   │───▶│ TILE_READY       │         │
│                             │ SOUTH       │    │ (A@L3[k+1,j])    │         │
│                             └─────────────┘    └──────────────────┘         │
│                                                                             │
│    // Receive partial C from West, accumulate, forward East                 │
│    ┌─────────────────┐                                                      │
│    │ TILE_READY      │───┐                                                  │
│    │ (C_partial from │   │   ┌────────────────┐                             │
│    │  West)          │   ├──▶│ ACCUMULATE     │                             │
│    └─────────────────┘   │   │ C += A×B       │                             │
│                          │   └───────┬────────┘                             │
│    ┌─────────────────┐   │           │                                      │
│    │ COMPUTE_DONE    │───┘           ▼                                      │
│    └─────────────────┘        ┌────────────┐    ┌──────────────────┐        │
│                               │ SEND_EAST  │───▶│ TILE_READY       │        │
│                               │ C_partial  │    │ (C@L3[k,j+1])    │        │
│                               └────────────┘    └──────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Key difference from A-stationary:
  - B tile is loaded ONCE and reused for all i iterations
  - A tiles flow vertically (North → South)
  - C partial sums flow horizontally (West → East)
```

---

## Comparison of Tile Flows

| Schedule | Stationary | Flowing (reused) | Flowing (accumulated) | Data Reuse |
|----------|------------|------------------|----------------------|------------|
| C-stationary | C[i,j] | A[i,k], B[k,j] | - | A reused ×N, B reused ×M |
| A-stationary | A[i,k] | B[k,j] | C[i,j] partial sums | B reused ×M |
| B-stationary | B[k,j] | A[i,k] | C[i,j] partial sums | A reused ×N |

### Tile Flow Directions

```
C-stationary:           A-stationary:           B-stationary:
A: West→East            A: stationary           A: North→South
B: North→South          B: East→West            B: stationary
C: stationary           C: North→South          C: West→East
```

---

## Formal Operand Flow Graph Representation

### Graph Structure

```cpp
// Operand Types
enum class OperandType {
    TILE_A,          // A matrix tile
    TILE_B,          // B matrix tile
    TILE_C,          // C matrix tile (partial or final)
    BUFFER_TOKEN,    // Buffer availability token
};

// Location in memory hierarchy
enum class Location {
    MEMORY,          // External memory
    L3,              // L3 tile
    L2,              // L2 bank
    L1,              // L1 buffer / accumulator
};

// Operand descriptor
struct Operand {
    OperandType type;
    TileCoord coord;     // (i, j, k) tile indices
    Location location;
    uint8_t node_id;     // Which L3/CT node
};

// Flow Graph Node
struct FlowNode {
    enum class Type {
        WAIT,            // Wait for operands on input ports
        FIRE,            // Execute operation
        PRODUCE,         // Produce operands on output ports
        JOIN,            // AND of multiple inputs
        FORK,            // Replicate to multiple outputs
    };

    Type type;
    Operation operation;  // LOAD, PUSH, SEND, COMPUTE, etc.

    std::vector<Operand> inputs;   // Required input operands
    std::vector<Operand> outputs;  // Produced output operands
};

// Flow Graph
struct OperandFlowGraph {
    std::vector<FlowNode> nodes;
    std::vector<std::pair<size_t, size_t>> edges;  // (from_node, to_node)

    // Metadata
    uint8_t level;       // 1=DMA, 2=BlockMover, 3=Streamer
    uint8_t node_id;     // Which sequencer instance
};
```

### C-Stationary Operand Flow Graph (Complete for L3[1,1])

```cpp
OperandFlowGraph bm_graph;
bm_graph.level = 2;  // BlockMover level
bm_graph.node_id = 5; // L3[1,1] = 4*1 + 1 = 5

// For each k iteration
for (int k = 0; k < K_tiles; k++) {
    // Node 0: Wait for A tile
    bm_graph.nodes.push_back({
        .type = FlowNode::WAIT,
        .operation = RECEIVE,
        .inputs = {{TILE_A, {1,0,k}, L3, 5}},  // A[1,k] at L3[1,1]
        .outputs = {}
    });

    // Node 1: Wait for B tile
    bm_graph.nodes.push_back({
        .type = FlowNode::WAIT,
        .operation = RECEIVE,
        .inputs = {{TILE_B, {k,1,0}, L3, 5}},  // B[k,1] at L3[1,1]
        .outputs = {}
    });

    // Node 2: Join A and B ready + L2 available
    bm_graph.nodes.push_back({
        .type = FlowNode::JOIN,
        .inputs = {/*A ready*/, /*B ready*/, {BUFFER_TOKEN, {}, L2, 0}},
        .outputs = {}
    });

    // Node 3: Push A to L2
    bm_graph.nodes.push_back({
        .type = FlowNode::FIRE,
        .operation = PUSH_TO_L2,
        .inputs = {},
        .outputs = {{TILE_A, {1,0,k}, L2, 0}}
    });

    // Node 4: Push B to L2
    bm_graph.nodes.push_back({
        .type = FlowNode::FIRE,
        .operation = PUSH_TO_L2,
        .inputs = {},
        .outputs = {{TILE_B, {k,1,0}, L2, 1}}
    });

    // Node 5: Forward A east (to L3[1,2])
    bm_graph.nodes.push_back({
        .type = FlowNode::FIRE,
        .operation = SEND_EAST,
        .inputs = {},
        .outputs = {{TILE_A, {1,0,k}, L3, 6}}  // L3[1,2]
    });

    // Node 6: Forward B south (to L3[2,1])
    bm_graph.nodes.push_back({
        .type = FlowNode::FIRE,
        .operation = SEND_SOUTH,
        .inputs = {},
        .outputs = {{TILE_B, {k,1,0}, L3, 9}}  // L3[2,1]
    });
}
```

---

## Summary

### Key Insights

1. **Operand-presence is the trigger**: Every operation fires when its input operands are ready, not when upstream operations complete.

2. **Three independent levels**: Each level (Memory→L3, L3→L2, L2→L1) has its own Operand Flow Graph that can be tested independently.

3. **Two coordination primitives only**:
   - `TILE_READY(T @ Loc)` - forward flow
   - `BUFFER_AVAILABLE(Loc)` - backpressure

4. **Stationary tile determines reuse pattern**:
   - C-stationary: Both A and B flow, maximum reuse
   - A-stationary: B flows, partial sums flow
   - B-stationary: A flows, partial sums flow

5. **Each schedule has distinct tile flow topology**:
   - Different directions for flowing operands
   - Different accumulation patterns

### Next Steps

1. Implement `OperandFlowGraph` data structure
2. Implement flow graph executors for each level
3. Create C-stationary flow graphs for 4×4 matmul
4. Validate against behavioral reference model
5. Extend to A-stationary and B-stationary
