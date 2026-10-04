# Tile-Flow Debugger: a hierarchical view of tile movement over the CPU/KPU SoC floorplan

**Date:** 2026-10-03
**Status:** Design; §7 answered on review (2026-10-03)
**Companion:** `docs/plans/tile-flow-debugger-views.html`, the hypothesized views on a synthetic schedule (open in a browser)
**Issue:** extends #286 (spatial event record and viewer). This plan is #286's viewer half made
concrete, plus the floorplan layer #286 does not yet name.
**Depends on:** #282 (naming map), #305 (the descriptor trace gives the orchestrating CPU its
events), #283 (L-T2, where the record becomes dense enough to need every LOD level)
**Relates to:** `docs/01-architecture/kpu-architecture.md` §5.1 (checkerboard floorplan),
ADR 0002 §3.3–3.4 (resource vocabulary, naming map), ADR 0003 D4 (the CPU is the attached
orchestrator, not a host)

## Decided on review (2026-10-03)

All of §7 is answered. Each answer is restated here so a misread is visible and cheap to
correct, not buried in the section it came from.

| | decision |
|---|---|
| **Q1 format** | **A new columnar `.tflow` bundle** (§3.3). Chrome trace stays as a derived export. #286's "no new trace format" clause is amended by this decision |
| **Q2 L3 at L-T1** | **Pooled first.** Step 1 draws L3 as one pooled station, labelled as such. Slot binding (step 6) is a separate, reviewed executor model change |
| **Q3a floorplan source** | **The generator is the reference** until a real SoC floorplan exists. The importer comes later |
| **Q3b T64 size** | **64 compute tiles.** The checkerboard dimensions in `kpu-architecture.md` §5.2.1 are wrong and are corrected with the §5.1 amendment (step 1). The 8×8 diagram in §3.1 stays illustrative |
| **Q4 CPU detail** | **Cores + SRAM + descriptor/completion rings** as separate blocks |
| **Q5 viewer** | **Plain static HTML + Canvas2D**, with WebGL for dense pixel layers, and no build step |
| **Q6 colour** | **By operator class**, with individual operators on hover and in the phase strip |
| **Q7 NoC topology** | **A folded 2D torus over the L3 hubs.** Two rows of the checkerboard form one loop around one torus dimension, and two columns one loop around the other. The 8×8 checkerboard is a 4×4 torus: 4 loops per dimension, 8 L3 hubs per loop, and every link equidistant. The **fold ends are the ports** where traffic enters or leaves a loop, and the DMA channels attach there. This connectivity is a **first pass**: the block schedules decide what is added or removed (§3.1) |
| **Q8 BlockMover count** | **Derived** from the topology, one BlockMover per L3 edge that abuts a compute tile. A declared value that disagrees is reported. First pass: an interior L3 tile has four, to the compute tiles abutting it W/N/E/S. A boundary tile abuts fewer, and how it is treated is part of the connectivity study (§3.1) |

---

## 1. Problem Statement

A KPU run moves tens of thousands of tiles through DRAM, L3, L2, L1 and the compute fabric
(CF). Today the only ways to see that movement are the following, and each one breaks at
scale:

| what exists | why it does not answer the question |
|---|---|
| `ChromeTraceExporter` → chrome://tracing / Perfetto | it is **time-major**, with one track per lane. It shows *when* movers were busy, but not *where* tiles are, and never how full a station is |
| `tile_tracker.hpp` / softmax simulator | text bands printed on each change. Readable for 50 tiles, unusable for 50,000 |
| `tools/visualization/*.html` | hand-built animations driven by the **older temporal model's** CSV. They are not connected to the L-T1 executor or the naming map |
| `TileRunStats` | totals and peaks. #279 is the precedent: a peak can look healthy while the series behind it violates capacity |

What engineers need to ask, in the order this plan delivers it:

1. **Is the schedule right?** Do the operators' tile sequences flow without oversubscribing a
   station? Where are the bubbles, the idle L3 tiles, the DMA-bound phases?
2. **Why is *this* compute late?** Follow the specific A, B and C tiles that meet in one
   compute tile. When did each one arrive, what did it wait for (data, credit or a lane), and
   how far apart were the operand arrivals?
3. **Where is the activity, and what does it cost?** Show activity and occupancy as heat on
   the physical floorplan, then convert it to energy and later to temperature.

Two more facts shape the design:

- **The SoC has two halves.** The attached CPU (RV64GC; ADR 0003) shares the address space
  and does resource management and orchestration. Its descriptor stream (`RESERVE`, `PLACE`,
  `LAUNCH`, `RELEASE`) is *causally upstream* of every tile move, and belongs on the same
  picture.
- **Logical and physical are two views of one thing.** The pipeline ladder (DRAM → L3 → L2
  → L1 → CF) is the *logical* manifestation. The checkerboard floorplan is the *physical*
  one. A tile event has a position in both, joined by one key: the #282 naming-map resource.

## 2. Architecture

The organizing principle is #286's: **the record is not the view.** The executor emits a
record. Derived products (occupancy series, a level-of-detail pyramid) are computed from the
record. The viewer consumes those derived products together with a floorplan, and never
consumes the executor directly.

```
┌──────────────────────────────┐        ┌──────────────────────────────────┐
│  Interpreter (L-B/T1/T2/CA)  │        │  Orchestrator (host or RV64)     │
│  TileOpRecord + HopRecord    │        │  DescriptorTrace (#305)          │
│  + station binding  (NEW)    │        │  RESERVE/PLACE/LAUNCH/RELEASE    │
│  + residency intervals (NEW) │        │                                  │
│  + cause edges (#286)        │        │                                  │
└──────────────┬───────────────┘        └──────────────┬───────────────────┘
               │ events keyed by naming-map index      │ events on the CPU block
               ▼                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TileFlowRecord  (the RECORD: columnar, append-only)                     │
│   residency[tile, station, t0, t1)   transit[tile, link, t0, t1)         │
│   compute[op, cf, t0, t1)            cause[event → event]                │
│   descriptor[id, kind, t, cpu]       provenance: RunIdentity, levels     │
└──────────────┬──────────────────────────────────────┬────────────────────┘
               │ derive (offline, deterministic)      │ checked by
               ▼                                      ▼
┌──────────────────────────────┐        ┌──────────────────────────────────┐
│  LOD pyramid                 │        │  Invariant checker (#286 DoD)    │
│  per (station, time-bin 2^k):│        │  occupancy ≤ capacity, every     │
│  occupancy, by-operator mix, │        │  residency closed, causes acyclic│
│  transfers, bytes, stalls,   │        │  (modelled on trace_validator.py)│
│  energy                      │        └──────────────────────────────────┘
└──────────────┬───────────────┘
               │                     ┌──────────────────────────────────────┐
               │                     │  SocFloorplan (NEW)                  │
               │                     │  naming-map resource → rect (µm),    │
               │                     │  hierarchy, CPU / MC / PHY / NoC     │
               │                     │  generated from DeploymentSpec, or   │
               │                     │  imported from the SoC floorplan     │
               │                     └───────────────┬──────────────────────┘
               ▼                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Viewer: static HTML, no server (tools/visualization/tileflow/)          │
│                                                                          │
│   ┌────────────────────┐  ┌──────────────────────┐  ┌────────────────┐   │
│   │ PHYSICAL           │  │ LOGICAL              │  │ INSPECTOR      │   │
│   │ floorplan, tiles   │◄►│ station swimlanes,   │◄►│ selected tile, │   │
│   │ as pixels at t     │  │ occupancy vs time    │  │ lineage, causes│   │
│   └────────────────────┘  └──────────────────────┘  └────────────────┘   │
│        linked by: naming-map index (space) and cycle (time)              │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. Design

### 3.1 The floorplan model: geometry for the naming map

The survey found **no machine-readable floorplan** in the repo. The only spatial facts are
§5.1's prose: an alternating, pitch-matched checkerboard of L3 tiles and compute tiles. The
only coordinate system in code is the temporal model's NoC mesh.

**Where the movers live (architecture correction, 2026-10-03).** §5.1 places the BlockMovers in
the compute tile. That is wrong, and this plan does not inherit it. A BlockMover moves
tensor, matrix and vector blocks L3→L2 *and* L3→L3, so it is clocked with the L3 tile, not
the compute tile. The NoC, which moves blocks between L3 tiles, attaches to the L3 tiles too.
The two tile types are therefore:

| tile | contains | clock / power domain |
|---|---|---|
| **L3 tile** | SRAM banks; **one BlockMover per edge that faces a compute tile** (N/E/S/W); the **NoC router hub** | L3 |
| **compute tile (CF)** | PE array, L2 banks, L1 stream buffers, Streamers (L2→L1, L1→L2) | CF |

So an L3→L2 move is driven by the BlockMover on the L3 tile's edge facing the destination CF,
and crosses that shared edge. An L3→L3 move goes router hub to router hub over the NoC. That
matches the executor's `news()` model ("four surrounding L3 tiles feed the CF"): a CF's
block bandwidth is the sum of the BlockMovers on its neighbours' facing edges, and that
bandwidth is a property of the *L3 tiles*. §5.1 of `kpu-architecture.md` should be amended to
match (§4 step 1).

**Where the DMA engines live, and what the NoC is.** The DMA engines are attached to the
**memory controllers** on the die periphery (DMA0–3 beside MC0–3), not to the L3 tiles. Inbound,
a DMA pulls a burst from DRAM through its memory controller and **pushes it into the NoC**.
Outbound, it **pulls from the NoC** and writes DRAM. The NoC is therefore an **address-routed
burst engine**: each burst carries its destination address (an L3 tile and offset, resolved
through the naming map), and the routers forward it hop by hop to that tile's hub. An inbound
tile's path is MC → DMA → loop port → NoC hops → destination L3 hub → slot, and the NoC
carries both DMA bursts and BlockMover L3→L3 traffic. The DMA channels connect at the torus's
**fold-end ports** (below), the one place a burst can enter or leave a loop.

So the floorplan becomes a first-class input with one rule:

> **Every rectangle names a naming-map resource, or a block the naming map will gain.** No
> geometry exists that the record cannot address, and no record address lacks geometry.

```cpp
// include/sw/kpu/program/platform/floorplan.hpp
struct Rect { double x_um, y_um, w_um, h_um; };

struct FloorplanBlock {
    std::string name;                 // "dev0/l3[5]", "dev0/cf[2]/l2[1]", "cpu0", "mc[0]"
    std::optional<ResourceName> resource;   // the naming-map binding, when one exists
    BlockKind kind;                   // L3Tile, L3Bank, BlockMover, NocRouter (inside L3Tile);
                                      // ComputeTile, PeArray, L2Bank, L1Vector, Streamer
                                      // (inside ComputeTile); NocLink; DmaEngine,
                                      // MemoryController, DramPhy, Cpu, CpuSram, Io
    std::string clock_domain;         // "l3", "cf", "cpu", "dram": where activity is charged
    Rect rect;
    std::vector<FloorplanBlock> children;   // hierarchy == LOD hierarchy (§3.4)
};

struct NocLink {                      // so every burst has a drawable, routable path
    std::string a, b;                 // hub to hub: "dev0/l3[5]/noc", "dev0/l3[9]/noc";
                                      // or a fold-end port, where a DMA channel
                                      // attaches: "dev0/noc/port[row0.E]"
    std::vector<std::pair<double, double>> route_um;   // the wire's path, for drawing
};

struct SocFloorplan {
    std::string source;               // "generated:checkerboard" or "imported:<file digest>"
    Rect die;
    std::vector<FloorplanBlock> blocks;
    std::vector<NocLink> noc;
    std::string digest() const;       // enters provenance: a view names its floorplan
};

// Two sources, one type:
SocFloorplan generate_floorplan(const DeploymentSpec&, const FloorplanStyle& = {});
SocFloorplan import_floorplan(const std::string& json_path, const DeploymentSpec&);
```

The **generator** is parametric on `DeploymentSpec`. The topology (`single` / `news` /
`checkerboard`) picks the pattern, the counts fill it, and the CPU cluster, memory
controllers, DRAM PHYs and IO are placed on the periphery.

**The NoC is a folded 2D torus over the L3 router hubs (Q7).** The checkerboard staggering is
the traditional folded-torus layout: it closes each loop using only equidistant links. Two
rows form one loop around the X dimension. For rows 0–1 of the 8×8 array, the loop runs
(0,0) → (0,2) → (0,4) → (0,6) → (1,7) → (1,5) → (1,3) → (1,1) → (0,0). Two columns form
one loop around the Y dimension in the same way. The 8×8 checkerboard is therefore a **4×4
torus**: 4 loops per dimension, 8 L3 hubs per loop. Every hub sits on exactly one row loop
and one column loop, so it has four links.

- **Fold ends are ports.** The link where each loop folds back, such as (0,6)–(1,7) on the
  east edge or (1,1)–(0,0) on the west, is where traffic can **enter or exit the loop**. The
  DMA channels attach to these ports, so the memory controllers sit at the array edges where
  the loops end. The 8×8 array has 16 such ports, one at each end of every loop. At all
  **four** corners a row loop and a column loop fold over the **same** link: (0,0)–(1,1),
  (0,6)–(1,7), (6,0)–(7,1) and (6,6)–(7,7). The eight corner hubs therefore have three
  distinct neighbours, the 64 loop links are 60 wires, and two ports share each corner link.
  That is an input to the connectivity study, not something to assume away.
- **Routing is dimension-ordered and address-routed.** A burst rides its row loop to a hub on
  the destination's column loop, then rides that loop to the destination, taking the shorter
  way around each ring.
- **Connectivity is a first pass, settled by measurement.** The block schedules decide which
  links and ports are worth having. Showing, per link and per port, how the schedules use
  them is part of what this debugger is for (§3.5, link-level activity).
- **First pass for the BlockMovers:** one per abutting compute tile, so an interior L3 tile has
  four, W/N/E/S. On the array boundary a finite checkerboard leaves some L3 edges with no abutting
  compute tile. Whether those tiles get a mover toward the fold port, or the array is bordered
  so that every L3 tile is interior, is part of the same connectivity study.

The generator emits every link, with its `NocLink::route_um`, and every fold port as a named
block. The **import** path takes the
real SoC floorplan as layout guidance. It reads a JSON of named rectangles exported from the
physical-design flow and **validates it against the deployment**: every declared resource
must have a rectangle, and every rectangle must resolve to a resource. Extra or missing
blocks are a refusal, in the same shape as the loadable's capability check.

```
WRONG — geometry the record cannot address
  { "name": "L3 macro 7", "x": 1200, "y": 400 }          // which naming-map resource?

CORRECT — geometry bound to the naming map
  { "name": "dev0/l3[7]", "kind": "L3Tile", "rect": [1200, 400, 380, 380] }
```

**The naming map must grow** to give the floorplan's non-memory blocks an address. Today
`ResourceKind` has no movers, memory controllers, NoC or CPU. It gains `DmaEngine`,
`BlockMover`, `Streamer`, `NocRouter`, `MemoryController`, `Cpu` and `CpuSram`, added the
way #282 requires (additively). Each mover is named under the tile that hosts it:

| resource | name | path |
|---|---|---|
| DMA engine 0 of memory controller 1 | `dev0/mc[1]/dma[0]` | — |
| BlockMover on L3 tile 5's east edge | `dev0/l3[5]/bm[1]` | edge index: 0 N, 1 E, 2 S, 3 W |
| NoC router hub of L3 tile 5 | `dev0/l3[5]/noc` | — |
| Streamer 0 of compute tile 2 | `dev0/cf[2]/str[0]` | — |

An L3 tile on the array boundary has a BlockMover only on the edges that face a compute tile,
so the generator derives the mover count from the topology. `DeploymentSpec::movers.block_movers`
is a global count today; under this layout it becomes **derived** (Σ facing edges), and a
declared value that disagrees is reported, the way `unmodelled_fields` reports a field a
level ignores (Q8).

**Conceptual floorplan, generated, 8×8 checkerboard** (32 L3 + 32 CF; illustrative counts):

```
┌──────────────────────────────── die ─────────────────────────────────┐
│ ┌─PHY0─┐ ┌─MC0─┐  ┌─DMA0─┐              ┌─DMA1─┐ ┌─MC1─┐ ┌─PHY1─┐    │
│ └──────┘ └─────┘  └──────┘              └──────┘ └─────┘ └──────┘    │
│ ┌────────────┐   ┌────┬────┬────┬────┬────┬────┬────┬────┐           │
│ │ CPU        │   │ L3 │ CF │ L3 │ CF │ L3 │ CF │ L3 │ CF │           │
│ │ RV64GC ×4  │   ├────┼────┼────┼────┼────┼────┼────┼────┤           │
│ │ + L2$/SRAM │   │ CF │ L3 │ CF │ L3 │ CF │ L3 │ CF │ L3 │           │
│ │ desc rings │   ├────┼────┼────┼────┼────┼────┼────┼────┤           │
│ ├────────────┤   │ L3 │ CF │ L3 │ CF │ L3 │ CF │ L3 │ CF │           │
│ │ IO / PCIe  │   ├────┼────┼── … 8 rows ─────────────────┤           │
│ └────────────┘   └────┴────┴────┴────┴────┴────┴────┴────┘           │
│                   NoC: folded 2D torus over the L3 hubs (Q7);        │
│                   fold-end ports at the array edges take the DMAs    │
│                                                                      │
│ ┌─PHY2─┐ ┌─MC2─┐  ┌─DMA2─┐              ┌─DMA3─┐ ┌─MC3─┐ ┌─PHY3─┐    │
│ └──────┘ └─────┘  └──────┘              └──────┘ └─────┘ └──────┘    │
└──────────────────────────────────────────────────────────────────────┘

        L3 tile (L3 clock domain)             compute tile (CF clock domain)
    ┌──────────[ BM N ]───────────┐          ┌──────────────────────────┐
    │  ┌──┬──┬──┬──┬──┬──┬──┬──┐  │          │        PE array          │
    │  │  │  │  banks │  │  │  │  │          │    (domain flow fabric)  │
    │  ├──┼──┼──┼──┼──┼──┼──┼──┤  │          │                          │
 [BM W]│   slots: one pixel ┌───┐[BM E] ─►   │                          │
    │  │   each at S1 LOD   │NoC│ │          ├────────┬───────┬─────────┤
    │  ├──┼──┼──┼──┼──┼──┼──│hub│ │          │   L2   │  STR  │ L1 strm │
    │  └──┴──┴──┴──┴──┴──┴──└───┘ │          │  banks │ L2↔L1 │ buffers │
    └──────────[ BM S ]───────────┘          └──────────────────────────┘
   one BlockMover per edge that faces a CF; the move crosses that shared edge
```

### 3.2 The record: what the executor must emit that it does not today

From the survey, these are the gaps between `TileOpRecord`/`HopRecord` and a record you can
place in space and time:

| needed | today | change |
|---|---|---|
| **station binding** on every hop: which L3 tile, which CF, which L2 bank | `HopRecord{hop, lane, start, finish}`, a lane in a mover *pool* | add `src`/`dst` naming-map indices. At L-T1, L3 is a **pooled** capacity with no tile index, so either the executor assigns a concrete L3 tile/slot (a model change, made deliberately) or the view draws L3 as one pooled station and *says so* (§3.6) |
| **residency intervals** `tile ∈ station over [t0,t1)` | resident set internal; only `peak_l3_residency` | emit them. This closes #286 gap 3, and it is what a pixel *is* |
| **cause edges** | `TileDependencies` never reaches the record | #286 gap 2, consumed as specified there |
| **tile identity** | an op index into `prog.ops()` | `(operator, tensor, ti, tj, role)`. Role is A/B/C/out; the operator comes from the loadable |
| **CPU events** | none | the `DescriptorTrace` from #305: each descriptor is an interval on `cpu0` with a cause edge to the moves it licensed |
| **mover binding** | BlockMovers are one pool (`movers.block_movers`) with a lane index | a BlockMover hop names the mover that ran it, `l3[t]/bm[edge]`: the source L3 tile's edge facing the destination CF. A lane in a pool becomes a mover in a place |
| **NoC L3→L3 hop** | never emitted (and `timeline_trace` throws on it) | emit it as a transit from router hub to router hub, over the floorplan's `NocLink`s, driven by the source L3 tile's BlockMover |
| **DMA path** | `Hop::DmaDramToL3` is one interval with nothing in between | split it into the DMA leg (DRAM → `mc[m]/dma[d]` → NoC injection) and the NoC transit hop by hop to the destination L3 hub. The DMA burst is address-routed, so the record carries its destination address and the hubs it crossed |

```cpp
// WRONG — a view that reads the executor's internals
for (auto& op : executor.timeline()) draw(op.resource_id, ...);   // a lane, not a place

// CORRECT — the record is the contract; the view sees only it
struct Residency { std::uint32_t tile, station; Cycle t0, t1; };   // station = naming-map index
struct Transit   { std::uint32_t tile, link;    Cycle t0, t1; Hop hop; };
struct Compute   { std::uint32_t op,   cf;      Cycle t0, t1; };
struct Cause     { std::uint64_t effect, cause; CauseKind why; };  // Data | Credit | Lane | Descriptor
```

`CauseKind` is what Step 2 relies on. It is not enough to know that a tile waited. The record
must say whether it waited for **data** (a producer), a **credit** (buffer space), a
**lane** (mover concurrency) or a **descriptor** (the orchestrator had not licensed it yet).

### 3.3 The file: columnar, browser-native, no server

```
run.tflow/
  manifest.json      RunIdentity, deployment, floorplan digest, level, what is unmodelled,
                     tile table (operator/tensor/ti/tj/role), station table (naming map)
  floorplan.json     SocFloorplan
  residency.bin      Uint32 tile, Uint32 station, Float64 t0, t1     (little-endian, columnar)
  transit.bin  compute.bin  cause.bin  descriptor.bin
  lod/k{0..K}.bin    per (station, 2^k-cycle bin): occupancy-time sum and max, occupancy-time
                     per operator CLASS (exact), transfers, bytes, stall-cycles by cause,
                     energy-pJ; plus a top-4 per-operator hint (lossy, display only)
```

Columnar typed arrays load straight into `Float64Array`/`Uint32Array` with no parse step,
which matters at 10^6 events. **This is a new format**, and #286's DoD says "no new trace
format or viewer stack". That conflict is real, so it is raised for review (Q1), not glossed.
The argument for it: Chrome-trace JSON has no notion of a station, residency or a floorplan,
and Perfetto renders tracks, not die plots. The time-major export stays available as a
*derived* view (Perfetto for lanes), so nothing that works today is lost.

### 3.4 Level of detail: one hierarchy for space, one pyramid for time

**Space** follows the floorplan hierarchy. Each level has a fixed visual unit:

| LOD | unit drawn | a tile is drawn as | question it answers |
|---|---|---|---|
| **S0 die** | blocks: CPU, MC, L3 tile, CF | nothing individually; blocks are colored by aggregate | where is the activity? (Step 3) |
| **S1 array** | every L3 slot and CF | **one pixel** per resident tile, colored by operator | is the schedule flowing? (Step 1) |
| **S2 station** | one L3 tile's banks and slots, or one CF's L2/L1 | a labelled cell | what is in this buffer, and who is waiting? |
| **S3 tile** | one tile | its lineage across the ladder | why was this late? (Step 2) |
| **S4 fabric** | PE array wavefront | elements (L-CA only) | is the systolic alignment right? (future) |

**Time** is a pyramid of power-of-two bins. The viewer picks the finest `k` that keeps at most
~1 bin per screen pixel. Zooming swaps `k` the way a map swaps tiles. Aggregates are
**conserving**: a bin at level `k+1` is exactly the merge of its two children, and that is
tested (§5). Every conserved field merges by sum or max: occupancy-time is stored as a sum,
not a mean, and the operator mix is stored per operator *class* (a bounded set, Q6), so the
merge is exact. The per-operator top-4 is the one field that cannot merge exactly, since a
fifth operator can enter a parent's top four. It is kept only as a display hint, recomputed
from raw events at the level it is stored, and excluded from the conservation rule.

**Budget** (the reason the pyramid exists). The 512³/T16 GEMM is about 99k ops, which gives
roughly 300k transits and about 200k residency intervals. A T256-class array has 256 L3 ×
~64 slots, or about 16k slot-pixels, and fits a 1080p canvas at 2×2 px per slot. Raw events
draw in Canvas2D up to ~50k visible marks; past that, the viewer draws from `lod/` and never
from raw events.

### 3.5 The views

These are the hypothesized views. The companion artifact shows each one on synthetic data.

**Step 1: System schedule.** This view validates schedules and occupancy.

- **(1a) Floorplan animation.** S1 pixels on the physical floorplan at cycle *t*, with a
  scrubber. Tiles in transit draw as short streaks along the link they cross: a DMA burst
  from its memory controller into the NoC, routed hub to hub to the destination L3 tile; a BlockMove from the L3 tile's facing-edge mover into the
  CF; a NoC move hub to hub along the links. A BlockMover that is driving a move lights up on
  its edge. The CPU block
  pulses with descriptor activity.
- **(1b) Station swimlanes.** One row per station, grouped by hierarchy (DRAM channels, L3
  tiles, CFs, plus the CPU). x is time, and each cell shows occupancy relative to capacity,
  with the operator mix as hue. A capacity line is drawn per row. **Oversubscription is a
  status mark (critical, with an icon and label)**, because occupancy above capacity should
  be impossible: it means a bug, not a hot spot.
- **(1c) Operator phase strip.** Each operator is a band over the time it holds stations.
  Overlap between operators is the prefetch freedom from #305 increment 3 made visible.
- **Automatic diagnostics**, listed beside the views: oversubscribed stations; idle stations
  while others queue; L3 imbalance (coefficient of variation across L3 tiles); DMA-bound
  windows (all DMA lanes busy while the CF is idle); credit-stall share by station.

**Step 2: Tile lineage.** This view follows tiles that meet in the CF.

- Pick a compute op (or a C tile). The viewer takes the **backward causal slice**: the A and
  B tiles it consumed, the C partial it accumulated, and the descriptors that licensed them.
- **(2a) Metro view.** The rows are the stage ladder (CPU, DRAM, L3, L2, L1, CF). x is time.
  Each tile is a line that is thick while resident and dashed while in transit, with a gap
  while waiting. Wait segments are tagged by `CauseKind`.
- **(2b) Alignment table.** For each compute, it lists operand arrival times at L1, the
  **skew** (last arrival minus first), and the start delay (compute start minus last
  arrival). It also attributes the delay to data, credit, lane or descriptor. Large skew is
  the "misalignment" signal: one operand waited in L1 for its partner.

**Step 3: Heat.** This view shows activity first, then energy, then temperature.

- **(3a) Activity heat**, per floorplan block, **per NoC link and per fold port**: occupancy-time, bytes moved, transactions per
  cycle. It uses one sequential ramp, over a selectable time window. Activity is **charged to
  the block that hosts the hardware doing it**. A BlockMove into a CF heats the source L3
  tile's mover, not the destination CF. A NoC transfer, whether a BlockMover L3→L3 move or a
  DMA burst, heats **every router hub and link on its route**, and the burst's originator (the
  source L3 tile's mover, or the memory controller's DMA engine) is charged for driving it. The
  CF is charged for its Streamers and its compute. Charging moves to the CF would put the L3 tiles' power in the
  wrong clock domain.
- **(3b) Energy.** Per-event attribution: `pj_per_byte × bytes` on transits, `pj_per_mac ×
  macs` on computes, and `static_pj_per_tile_per_cyc` as leakage per block. These are the
  same constants as `characterize/device_model.hpp`, now attributed per block instead of
  totalled.
- **(3c) Thermal** (later). It solves a 2D RC grid over the floorplan with the energy map as
  the source. It is labelled **"model, not measurement"** wherever it shows, by the same
  discipline as `has_timing`.

### 3.6 Honesty rules the viewer must follow

```
WRONG — an unmodelled station drawn empty      (reads as "idle")
CORRECT — hatched, labelled "not modelled at L-T1"

WRONG — L3 at L-T1 drawn as 32 tiles with invented slot positions
CORRECT — one pooled L3 station, labelled "pooled at L-T1", until the executor binds slots

WRONG — a heat map with no time window, level or energy-model source in its legend
CORRECT — every heat legend names: window, level, metric, and model constants' source
```

## 4. Implementation Steps

Ordered so that Step 1 delivers value on L-T1 data before any executor changes beyond
residency.

1. **Floorplan.** Add `include/sw/kpu/program/platform/floorplan.hpp` and
   `src/program/floorplan.cpp`, with a generator for `single` / `news` / `checkerboard`, an
   importer, and validation. Extend `ResourceKind` (`resource_map.hpp`) with the
   mover/MC/NoC/CPU kinds, with BlockMovers and router hubs under `l3[t]` and Streamers under
   `cf[c]`. Derive the BlockMover count from the topology and report a disagreeing
   `movers.block_movers`. Amend `docs/01-architecture/kpu-architecture.md` §5.1, which places
   the BlockMovers in the compute tile. Add `tools/floorplan/` to dump the floorplan as JSON + SVG.

   **Done (2026-10-04).** `platform/array_layout.hpp` derives the grid, the BlockMover sites and
   the folded torus once; the naming map and the floorplan both read it. The spec gains
   optional `array.rows/cols`, `memory.controllers` and `cpu.harts`; a spec without them keeps
   its canonical bytes. `kpu-floorplan` writes JSON and SVG and validates an imported file.
   What differs from the text above, and why:
   - **A layout is derived, not required.** It exists only when the spec describes an
     alternating array (`l3.tiles == compute_tiles`, an even grid of 2 × compute_tiles
     cells). Existing specs that do not, such as the 16-CF / 8-L3 fixture, stay valid and
     simply have no floorplan, and the refusal says why.
   - **Ports are numbered** (`dev/noc/port[k]`), because the address grammar takes digits in
     brackets. The human label (`row0.W`, `col3.S`) is on the floorplan block. Port numbering
     is fixed: row loop p gets ports 2p (W) and 2p+1 (E); column loop q gets R + 2q (N) and
     R + 2q + 1 (S).
   - **No Streamer names yet.** `movers.streamers` is a pool size, not a count per compute
     tile, so there is nothing to place per tile without inventing one. The PE array is the
     compute tile's own body, not a sub-block.
   - **Four shared corner links, not two.** The test that checks every loop link found that
     the 8×8 board's 64 loop links are 60 wires (§3.1 corrected).
   - **First-pass DMA attachment:** controllers on the top edge feed the N ports nearest them,
     controllers on the bottom edge the S ports. W and E ports start unattached, visibly.
2. **Record v0 (residency + transit at L-T1).** In `tile_transaction_executor.hpp`, emit
   residency intervals and station-bound transits (L3 pooled; CF from `Placement`). Add
   `include/sw/kpu/program/record/tile_flow_record.hpp` and a writer for the `.tflow`
   bundle. Fold in the descriptor trace from `orchestrate()`.

   **Done (2026-10-04).** The executor publishes `L3Residency` intervals (the series
   `peak_l3_residency` is the maximum of), carried on `RunOutcome`.
   `record/tile_flow_record.hpp` builds the record from a run and writes and reads the
   columnar `.tflow` bundle; `kpu-run --tflow <dir>` writes one. Pinned: the record's peak
   equals the executor's own at tight and unbounded capacities, every hop is one transit,
   seeded and retained tiles are flagged, and the bundle is byte-deterministic.
   What differs from the text above:
   - **The descriptor trace is not folded in yet.** An orchestrated model is several launches,
     each of which starts at cycle 0, and a `PLACE` has no time of its own at L-T1. Folding
     descriptors in needs a time base across launches. That is a separate, orchestrated
     record, not a single-run field, so it comes after the viewer has a single run to show.
   - **L2 and L1 appear only as pooled, unmodelled transit endpoints**, so the BlockMover and
     Streamer legs have somewhere to go. They are listed in the manifest's `unmodelled`.
   - **Moves carry a mover and a lane, not a place.** Binding a BlockMover hop to
     `l3[t]/bm[e]` needs L3 slot binding (step 6).

3. **LOD pyramid + invariant checker.** Add `src/program/record/lod.cpp` and
   `tools/trace/tflow_check.py`, the #286 checker. Reintroducing #279 must make it fail.
4. **Viewer, Step 1.** In `tools/visualization/tileflow/index.html`, plus modules, build views
   1a/1b/1c and the diagnostics. Add a `kpu-run --tflow <dir>` flag.
5. **Cause edges and lineage, Step 2.** Add the #286 causality plumbing and `CauseKind`, then
   views 2a/2b.
6. **L3 slot binding.** This is a deliberate executor model change, with its own design note:
   L-T1 assigns a concrete L3 tile and slot per residency, using a placement policy that
   follows the floorplan (nearest L3 to the consuming CF). Pooled L3 stays available.
7. **Heat, Step 3.** Views 3a/3b. 3c (thermal) is planned behind its own review.
8. **L-T2 density.** When #283 lands, L2 banks and L1 vectors become real stations, and S2
   LOD fills in.

## 5. Verification

```bash
cmake --build build && cd build && ctest -L tileflow --output-on-failure
python3 tools/trace/tflow_check.py runs/gemm512.tflow            # exit 0, else violations
kpu-run --program gemm512.l0 --level transactional --tflow runs/gemm512.tflow
python3 -m http.server -d tools/visualization/tileflow            # open, load runs/*.tflow
```

| test | asserts |
|---|---|
| floorplan coverage | generated and imported floorplans: every enumerated resource has exactly one rect; no rect overlaps a sibling; children lie inside their parent |
| record conservation | summed residency per station at every event equals the executor's own occupancy; `max` equals `peak_l3_residency` |
| pyramid conservation | for every station and every `k`: bin(k+1) == merge(bin(k) children); totals at the coarsest level equal raw totals |
| invariant checker bites | reintroduce the #279 early-release bug → `tflow_check` exits 1 naming the station and cycle |
| determinism | two runs → byte-identical `.tflow` (ADR 0002 §3.5) |
| honesty | a run at L-T1 marks L2/L1 unmodelled in the manifest; the viewer renders them hatched |
| viewer scale | 512³/T16 loads and scrubs at ≥ 30 fps at S1 (manual, recorded in the PR) |

## 6. Key Invariants

1. **The record is the only interface.** The viewer never reads executor internals, and
   everything it draws is traceable to record events.
2. **One key joins logical and physical:** the naming-map index. A floorplan rect without a
   resource, or a resource without a rect, is a refusal.
3. **Aggregates conserve.** Every LOD bin is an exact merge of its children.
4. **Unmodelled is never empty, and pooled is never invented.** What a level does not model is
   drawn as such.
5. **Occupancy above capacity is a bug signal,** rendered as a status mark and failed by the
   checker. It is never just a hot color.
6. **Model outputs say they are models.** Energy names its constants, and thermal is labelled
   "model, not measurement".
7. **Hardware is drawn and charged where it lives.** BlockMovers and the NoC router hub belong
   to the L3 tile (L3 clock domain); Streamers belong to the compute tile. A move's activity
   and energy go to the block hosting the mover that ran it.
8. **Credit-based semantics only.** The vocabulary is arrive, resident, credit returned and
   waiting for credit. Never hit, miss or evict.

## 7. Questions for review (answered; see "Decided on review" at the top)

- **Q1. Format.** *Decided: yes, `.tflow`.* Approve a new columnar `.tflow` bundle, superseding #286's "no new trace
  format" clause, with Chrome-trace kept as a derived export? *Recommended: yes* (§3.3).
- **Q2. L3 at L-T1.** *Decided: pooled first.* Accept "pooled L3" in Step 1 and schedule slot binding (step 6) as a
  separate model change? Or bind slots first? *Recommended: pooled first.* Step 1 is useful
  without it, and binding is a placement policy that deserves its own review.
- **Q3. Floorplan source of truth.** *Decided: generator for now; T64 = 64 compute tiles.* Is there an existing SoC floorplan (DEF/LEF, a
  spreadsheet, a slide) to import for T64/T256, or should the generator be the reference until
  one exists? Also: the T64 tile counts in `kpu-architecture.md` §5.2.1 disagree (64 compute
  tiles vs a 4×4 CF). Which is right?
- **Q4. CPU detail.** *Decided: cores + SRAM + rings.* Model the CPU as one block with descriptor activity, or as cores +
  SRAM + the descriptor/completion rings as separate blocks? *Recommended: cores + SRAM +
  rings*, so the MMIO traffic of ADR 0003 shows where it physically lands.
- **Q5. Viewer tech.** *Decided: plain.* A plain static HTML + Canvas2D/WebGL app under
  `tools/visualization/tileflow/` (no build step), or a framework? *Recommended: plain.* It
  must open from a file or an artifact with no server.
- **Q6. Operator color budget.** *Decided: by operator class.* The categorical palette validates three colors all-pairs.
  Real models have dozens of operators. Should the system view color by **operator class**
  (GEMM, elementwise, reduction, …), with individual operators distinguished on hover and in
  the phase strip? *Recommended: yes.*
- **Q7. NoC topology among L3 hubs.** *Decided: a folded 2D torus (4×4 loops, 8 hubs per loop), with fold-end ports for the DMAs; connectivity is a first pass (§3.1).* In a checkerboard, the nearest L3 tiles are diagonal
  neighbours. Do the hub-to-hub links run diagonally across the CF corners, or orthogonally
  two cells apart, routed between compute tiles? The generator needs one answer. An imported
  floorplan states it through its `NocLink` routes.
- **Q8. BlockMover count.** *Decided: derived.* Should `movers.block_movers` become derived from the topology (one
  per L3 edge facing a compute tile), with a declared value reported when it disagrees, or
  stay declarable for what-if studies? *Recommended: derived*, because the count is a fact
  of the layout.
