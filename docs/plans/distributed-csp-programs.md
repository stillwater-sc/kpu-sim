# Distributed CSP programs on a nearest-neighbour mesh of DMM nodes

**Date:** 2026-11-27
**Tracking:** epic #603 (steps #604–#617); the NoC rework is #618
**Status:** Design
**Precedes:** the NoC rework, and `docs/plans/retire-concurrent-timing-executor.md` (re-sequenced behind this
plan; see §7)
**Related:** `csp_program.hpp` (level 1, "the plan's MPI hypothesis"; level 2 is "a later step"),
`docs/plans/csp_pattern_coverage_roadmap.md` (pattern classes P1–P9), ADR 0002 §3.3, ADR 0004 §5

## 1. Problem Statement

A CSP program today is **level 1**. It addresses the machine as one flat L3 and one large compute fabric, with
one process of each kind (`csp_program.hpp:16`). The machine is not like that.

The machine, as the user decided on 2026-11-27:
- **A node** is an L3 tile with its own compute tile (CT) beside it: L3+CT. It is a distributed-memory-machine
  (DMM) node, so the decades of DMM/MPI algorithms apply.
- **The NoC is an XY mesh of nodes, and push only.** A source prepares a packet and sends it to its
  destination; nothing on the NoC is a request or a response.
- **Communication is one hop.** A node talks only to its four neighbours, N, E, S and W. It can send one
  packet on each of its four links and receive one on each, all at once.
- **Credits travel on a separate credit network,** not the data mesh.
- **All CSPs together make one global data movement,** paced by credits that start at the leaf compute
  events.
- **Eight BlockMovers per L3 tile.**
  - Four send, one per link.
  - Three feed the CT: two input operands in, one output operand out.
  - That is seven, rounded up to eight, with one CSP per potential packet flow, never shared. The eighth
    is a third input stream (decided 2026-11-27).
- **DMA engines belong to the memory controllers.** They run on the memory controller's clock, not the
  ALU's, which saves energy.
- **The memory PHYs are spread around the chip's periphery.** That is needed for 4, 8, 16, 32 or more
  channels. So a DMA engine may have to feed a node on the far side of the chip.
- **A periphery network joins them.** A ring buffer around the mesh, which the mesh's edge ports and the
  DMA engines both connect to. It absorbs the mismatch between where the PHYs are and where the edge ports
  are. Other structures are hypotheses to evaluate, a Clos network among them.

This plan settles the **first requirement**: under this allocation, can distributed (level 2) CSP programs be
generated for every operator pattern class, P1–P9? Nothing gets built on the mesh until the answer is yes,
or until the allocation is changed by what this plan finds.

**One-hop may have to change.** An operator such as a matrix transform has to be rewritten as a
nearest-neighbour communication pattern. Such rewrites exist; on a mesh, any permutation can be done by
store-and-forward through the L3 tiles between source and destination. So a permutation without a
nearest-neighbour solution is not the expected failure; a permutation whose solution costs too much is. Its
hops, its transit slots or its movers can be too many. If a required pattern's cost is out of bounds, the
one-hop design is reopened. This plan measures that cost for every pattern (§3.5).

## 2. Architecture

```
     PHY ── MC ── DMA ─┐   (memory-controller clock)          ┌─ DMA ── MC ── PHY
                       ▼                                      ▼
   ══ periphery network (ring buffer; or Clos, ...) ═══════════════════════════════  (memory-controller clock)
          ▲ edge port              ▲ edge port                     ▲ edge port
          │                        │                               │
   ┌──────────── node(0,0) ────────────┐        ┌──────────── node(0,1) ────────────┐
   │  L3 tile                          │  E ──► │  L3 tile                          │
   │   slots ◄── rx_N rx_E rx_S rx_W   │  ◄── W │   ...                             │
   │   bm_N bm_E bm_S bm_W  (send)     │        │                                   │
   │   bm_a bm_b (feed) bm_c (drain)   │        │                                   │
   │   bm_x (eighth: third input)      │        │                                   │
   │      │ │        ▲                 │        │                                   │
   │      ▼ ▼        │                 │        │                                   │
   │  CT: L2 ─ Streamer ─ L1 ─ PE array ─ Combiner                                  │
   └───────────────────────────────────┘        └───────────────────────────────────┘
        S │ ▲ N                                       S │ ▲ N
          ▼ │                                           ▼ │
        node(1,0)  ...                                node(1,1) ...

   data mesh:    one-hop links, each a push into a neighbour's named L3 slot
   credit mesh:  the same adjacencies, reversed; a credit goes from a slot's consumer to its one producer
```

**One hop means the mesh needs no routers.** Each link is a channel between two neighbouring L3 tiles. A
packet's destination is the neighbour's slot: there is no route to compute and no traffic passing through.
Where data must go further, it lands in the next node's L3 and that node's mover sends it on, and the
program says so.

**The periphery network is the one routed fabric, and it is still push only.** A DMA engine's packet names
an edge node's slot. The periphery carries it from the engine's attachment to that node's edge port, and the
edge node's receive port lands it. The mesh's one-hop rule is unchanged: the periphery is between DRAM and
the mesh, not inside the mesh. Its credits come back on a periphery credit path, from the edge node's slot
to the DMA engine. The periphery runs on the memory controllers' clock (decided 2026-11-27), so a DMA flow
crosses one clock domain, at the edge port: from the periphery onto the mesh. Its credits cross back there.

## 3. Design

### 3.1 The machine contract

The resource model that the checker (§3.3) enforces, and that a deployment declares:

| Resource | Per node | Rule |
|---|---|---|
| Send movers `bm_N`, `bm_E`, `bm_S`, `bm_W` | 4 | each owns its outgoing link, and runs one CSP |
| Feed movers `bm_a`, `bm_b` | 2 | L3 → CT's L2, one operand stream each |
| Drain mover `bm_c` | 1 | CT's L2 → L3, the result stream |
| Eighth mover `bm_x` | 1 | its role is a recommendation (§7, decision 1): a third input stream (bias, accumulator init, a gating operand) |
| Receive ports `rx_N` … `rx_W` | 4 | a received packet lands in the L3 slot it names, under a credit the receiver granted. Not a mover, and not a CSP |
| Edge port | edge nodes: 1 (corners: 2) | the link with no neighbour connects to the periphery network; its sender is that side's send mover, and its receiver lands packets as the other receive ports do |
| DMA engine | per memory controller, not per node | runs on its controller's clock; pushes over the periphery into any edge node's slot, under that slot's credit |
| Periphery network | one, around the mesh | push only and routed; a ring buffer first, its topology a deployment parameter (ring, Clos, ...) |
| Credit links | as the data links, reversed | one hop on the mesh; across the periphery from an edge slot to its DMA engine |

**WRONG**, a level-1 process placed onto a mesh:

```cpp
Process{ProcessKind::BlockMover, "bm", /*every move in the program*/ ...};   // one mover for the machine
```

**CORRECT**, a process named by where it is and what it owns:

```cpp
struct ProcessId {
    NodeId node;              // (row, col) on the mesh; or a DMA engine on an edge
    MoverRole role;           // SendN, SendE, SendS, SendW, FeedA, FeedB, Drain, Extra, Dma, Streamer, Compute
};
```

### 3.2 The distributed program (level 2)

Level 2 keeps level 1's invariants: residency, placement and reuse are the program's (ADR 0004 §5,
reuse-is-deliberate). It adds a location to every process, and one-hop channels:

```cpp
struct Channel2 {                      // one packet flow: one sending CSP, one receiving slot pool
    ProcessId from;                    // the one process that pushes on it
    NodeId to;                         // a neighbour of from.node (or from.node itself, for feeds and drains)
    Link link;                         // N/E/S/W, or Local
    std::uint32_t credits;             // slots the receiver grants this flow
};

struct Action2 {
    enum class Kind { Push, Feed, Drain, Call, Release, Load, Store } kind;
    TileCoord tile;
    std::size_t channel;               // a Push, Feed, Drain, Load or Store: the flow it uses
    L3Address dest;                    // where the packet lands (node, slot, bank): the program places it
};

struct DistributedProgram {
    MeshShape mesh;                    // rows x cols of nodes, and where the DMA engines attach
    std::vector<ProcessId> processes;
    std::vector<std::vector<std::size_t>> actions_of;   // each process's actions, in its own order
    std::vector<Channel2> channels;
    std::vector<Action2> actions;
    TileProgram source;                // the L0 trace: values answer to L-B
};
```

Every process runs its own list. A Push waits for its channel's credit and spends it. A Release returns the
freed slot's credit, on the credit mesh, to that slot's producer. There is no global order and no shared
sequencer: the leaf Calls' releases pace everything above them.

**As built (#605).** `include/sw/kpu/program/csp/distributed.hpp` and the text form `dcsp 1`
(`csp/lang/distributed_text.hpp`) refine this sketch in three ways:
- **A channel owns its slots.** Each channel lists the receiving pool's slots it owns, so a slot's credit
  always returns to one sender. A channel into a DMA engine (a Store's path out of an edge port) has
  `credits` instead: the store buffer.
- **No Release action.** Every write that fills a slot states the reads its residency serves
  (`reads n`, with 0 holding the tile to the end). The n-th read frees the slot and returns its credit,
  as level 1 records the consumers per residency. The actions are Load, Push, Feed, Drain, Store and
  Call; an accumulating Call merges into a live result and spends no credit.
- **The values' source comes with L-B (#606).** A program declares its operands as level 1 does.
- **A result that accumulates expects its merges (#606).** `call ... to T at l2[t]:s merges k` makes
  every read of the result wait until k accumulating calls have landed, as the Combiner signals a tile
  ready only after its last merge. Without it a drain could take a partial, and level-2 L-B showed that
  race.

### 3.3 The checker

A static check, run on every generated program before any simulation:

| Check | What it refuses |
|---|---|
| D1 resources | more than 8 movers on a node; two CSPs on one send link; one CSP on two flows |
| D2 one hop | a mesh channel whose ends are not neighbours, or a node and its own CT; a periphery channel that does not join a DMA engine and an edge port |
| D3 credits | a channel without a credit return on the credit mesh; a slot freed with no producer to credit |
| D4 placement | a Push into a slot that is live, or into a slot its receiver never granted |
| D5 liveness | a deadlock, found by running the program on L-B with every channel bounded by its credits |
| D6 values | a result that differs from L-B's level-1 run of the same operator |

D5 and D6 need **L-B for level 2** (§4 step 3). It executes every process with bounded channels, in a fair
interleaving, and computes the values.

### 3.4 Each pattern class as a DMM algorithm

**These rows are hypotheses until step 5 produces each program and the checker passes it.** "Sends" counts
the send movers a node uses at once.

| Class | DMM algorithm, nearest neighbour only | Sends | Feeds + drain | Risk |
|---|---|---|---|---|
| P1 GEMM | **built (#608):** systolic 2D: A streams in from the W edge and moves E, B from the N edge and moves S, C stays, then drains north to row 0 and out of its edge port | E, S, N | a, b, c | Cannon needs wraparound shifts, which a mesh without wraparound does not have in one hop; the edge-streamed systolic form does not need them |
| P2 sliding window | **built (#612):** a 3 × 3 box filter as a row pass and a column pass (separable, so no corner halos): X streams in from the west edge; each node sends its X west and east and its row-pass result north and south as halos; one flow per send mover, each carrying everything bound for its neighbour. One tile per node (7 of 8 L2 slots); weights, when a filter has them, come by P5's broadcast | N, E, S, W | a, b, x, c | the tightest: an interior node uses all 8 movers |
| P3 row reduction | **built (#611):** X streams down each column (bm.S), leaving the row's links to the reduction; partials chain west (`rowsum`, then `sum ... acc` of the east's partial), the west end stores the total out of its edge port and broadcasts it back east, each node holding it in L2 (bm.x) | S, W, E | a, b, x, c | the row's length is the latency |
| P4 gather/scatter | index-driven DMA into an edge node, then forwarded; or a permutation across the mesh | ≤ 2 | a, c | **the indices are run-time data**, and a static program has to state every move. See §7, decision 2 |
| P5 broadcast | **built (#609):** pipelined: along row 0 to the E, then down each column to the S; every node holds its L3 copy and feeds it to its compute tile's L2 through `bm.x`, both held; node (r, c) is reached in r + c hops | E, S | x | none |
| P6 two-operand aligned | **built (#610):** both operands share one block distribution, so they are aligned by construction: each row's X and Y pairs stream in from the west edge and pass east, a node keeping its own; `call add`; Z north as P1's C. Realigning operands distributed differently is a layout transform (P7) | E, N | a, b, c | none |
| P7 layout transform | **built (#613):** Y = X^T, one tile per node of a square mesh: X[r,c] goes home along its row, then along row r to the diagonal (r,r), then along column r to (c,r), where it is transposed; every mover sends in a synchronous schedule's order and every tile delivered into L3 has a slot of its own, so no L3 delivery waits for a credit; row 0's pushes out to the DMA engine can wait for its store buffer (2 credits), which the engine frees as it stores, so the program cannot deadlock. **Measured:** the longest route 2(R-1) hops (6 on the 4x4), 40 hops in all on the 4x4, and the busiest node 11 of its 16 L3 slots (5 on the 2x2) | up to 4 | a, c | the one-hop cost is in L3: within capacity at 4x4, growing with the mesh's side |
| P8 chained-GEMM fusion | the intermediate stays in the node; K and V move as a pipeline, not a ring (a ring cannot be embedded with dilation 1 in a line) | E, S | a, b, c, x | the online softmax's state per node |
| P9 append | the cache is distributed by head; append and the windowed re-read are DMA work at the owner node, through its edge row | ≤ 1 | a, c | interior nodes reach DRAM only through the nodes between them and the edge |

**DRAM is at the edges.** An interior node is reached only through the nodes between it and an edge. So
loads are pipelines from the edge inward: the node-level form of "everything streams from the edges". The
plan measures what that costs the edge nodes' movers.

**The periphery decides which edge, not the PHY's position.** P1's A enters on the W edge and B on the N
edge, whichever side of the chip their channels' PHYs are on. The periphery carries each from its PHY to
that edge. So every pattern also has a periphery load: bytes per edge port, and, for each flow, the distance
from its DMA engine to its edge port. That load is how ring and Clos get compared (§3.5).

### 3.5 The coverage report

The rows are measured, not written by hand: #617 builds the report from the programs the
generators write, every built class at once (P1 #608 and P5 #609 are built; their rows come with it).
Each pattern's generator writes a row: nodes, movers used per node (the maximum), hops per tile (the
maximum), transit slots per node, credits per flow, and whether D1–D6 pass. It also gives the periphery
load: bytes per edge port, and the busiest periphery segment for a ring of the deployment's channel count
(4, 8, 16, 32). The report is the answer to the
first requirement. For each pattern it says one of:
- **fits**;
- **fits at a cost**, with the number;
- **needs the allocation changed**, with the change.

A "needs" against one hop or eight movers goes to the user before the NoC rework starts.

## 4. Implementation Steps

1. **The machine contract**, as data: `DeploymentSpec` gains a `dmm_mesh` topology beside today's
   checkerboard:
   - rows, cols and movers per node;
   - the memory controllers, each with its DMA engine, its clock and its PHY's place on the periphery;
   - the periphery network: its topology (`ring` first) and queues, on the memory controllers' clock. `MeshLayout::of` (`mesh_layout.hpp`, beside the checkerboard's `ArrayLayout`) lays it out. **Done: #604.**
   The spec is hierarchical (#635): SoC → `cluster` (`l3_tile` with its BlockMovers and ports,
   `compute_tile` with its L2, Streamers, L1, fabric and Combiner) → resources. `noc` is the NoC cluster, and
   `memory` holds the DMA engines and the periphery. On a `dmm_mesh` the flat fields are derived views. `kpu-floorplan --structure`
   writes it in `kpu-structure` v1, so the floorplan viewer draws the nodes, the data mesh and the credit
   mesh (`include/sw/kpu/program/platform/`, `src/program/floorplan.cpp`).
2. **The level-2 program**: `include/sw/kpu/program/csp/distributed.hpp` (§3.2), its printer, and a text form
   for hand-written programs, so a person can write and read one (`csp/lang/`). **Done: #605**
   (`dcsp 1`, samples in `tests/program/csp/distributed/`).
3. **L-B for level 2**: `csp/distributed_behavioral.hpp` executes the processes with bounded channels and
   computes the values. Checks D5 and D6. **Done: #606** (round robin, so no process starves; a deadlock
   names every stuck process, with its line and what it waits for; `compare_values` is D6, bit for bit).
4. **The checker**: `csp/distributed_check.hpp` (D1–D4), and `kpu-csp-check --distributed`. **Done: #607** (also
   D0, the program's mesh is the deployment's; D4 counts each slot's reads and merges across processes, a
   call's operand once per mention).
5. **The generators**, one PR per class, in the order that builds on what came before: P1, P5, P6, P3, P2,
   P7, P8, P9, P4 (`csp/gen/distributed/`). Each is tested on 1×1, 2×2 and 4×4 meshes, checked with D1–D6,
   and its row added to the coverage report.
6. **The coverage report** (`docs/analysis/distributed-csp-coverage.md`): the answer, and the changes to the
   allocation if any are needed.

Step 1 is small and independent. Steps 2–4 are the infrastructure. P1 on 2×2 is the first end-to-end proof.

## 5. Verification

```bash
cmake --build build -j16 && (cd build && ctest -L "csp;distributed" --output-on-failure)
./build/tools/csp-gen/kpu-csp-gen --algo matmul --size 64 --tile 16 --dtype int8 \
    --target tests/program/deploy/kpu_dmm_2x2.json --distributed -o /tmp/mm64.dcsp
./build/tools/csp-check/kpu-csp-check --distributed /tmp/mm64.dcsp \
    --deploy tests/program/deploy/kpu_dmm_2x2.json       # D1-D6 pass; prints the coverage row
```

Expected: every class's generator passes D1–D6 on each mesh, or the report says which check fails and why.
P1's values at L-B equal level 1's, bit for bit (the same computation, distributed).

## 6. Key Invariants

- **One hop.** Every channel joins two neighbours, a node and its own CT, or an edge DMA and its node.
- **Push only.** No action reads another node's memory. Data moves only when its producer pushes it under a
  credit.
- **One flow, one CSP.** A mover runs exactly one channel's sends; no flow shares a CSP.
- **The program places.** Every packet names its destination slot; no receiver chooses one.
- **Credits start at the leaves.** Every slot's credit is returned, when its residency's last read is done,
  to its one producer, on the credit mesh.
- **Values answer to L-B.** A distributed program computes exactly what its level-1 form computes.

## 7. Decisions

**Decided 2026-11-27 (the user):**
1. **The eighth mover is a third input stream:** bias, accumulator init or a gating operand.
2. **Run-time indices stay at the DMA.** Embedding lookups, KV paging and JEPA masks are index-driven
   between DRAM and the edge node; the mesh sees only a static pattern, which the program states.
3. **The order of work:** this plan first. Then the NoC rework: one-hop links, a credit mesh and the
   periphery network on the cycle core, replacing the folded-torus ring and its hubs. Then the retirement
   plan's tracks: A (the hand-traced path, now DRAM → MC → DMA → periphery → edge port → L3), B (the shared
   views) and R.
4. **DMA engines are the memory controllers',** on their clock. The PHYs are spread around the periphery,
   and a periphery network (a ring buffer first; Clos and others as hypotheses) joins the DMA engines to the
   mesh's edge ports.

5. **The periphery carries DMA flows only, as the first step.** Edge-to-edge traffic over it (the mesh's
   wraparound, for Cannon-style shifts and ring algorithms, §3.4 P1 and P8) is reopened only if a pattern's
   coverage row shows it needs it. D2 refuses it until then.
6. **The periphery runs on the memory controllers' clock.** A DMA flow's one crossing is at the edge port.

