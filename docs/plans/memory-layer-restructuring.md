# Memory Layer Restructuring: Aggregate Layers over Addressable Elements

**Status:** Design / proposed
**Tracking epic:** #35
**Sub-issues:** #32 (L3Layer), #33 (L2Layer), #34 (L1 → compute fabric)
**Milestone:** v0.8

## Goal

Introduce the missing **aggregate "Layer" abstraction** over the KPU's distributed
on-chip memory, and stop conflating an individual storage *element* with the
*whole* storage layer.

Concretely:

- Introduce **`L3Layer`** and **`L2Layer`** aggregate types that own a (possibly
  non-uniform) collection of elements plus the layer's interconnect and access
  ports.
- **Keep** the element types `L3Tile`, `L2Bank`, `L1Buffer` exactly as they are —
  they are real, addressable, configurable entities that matter to the machine's
  operation. This is **not** a rename of the elements.
- Introduce an **`L1Layer`** aggregate that owns the `L1Buffer` stream buffers,
  symmetric with `L2Layer` and `L3Layer`. *(Updated 2026-06-28 — supersedes an
  earlier draft that folded `L1Buffer` into the `ComputeFabric`; see
  "On `L1Layer`" below.)*

### Principle: Layers are monitoring/ownership structures, not a dataflow API

`L3Layer` / `L2Layer` / `L1Layer` are **conceptual resource owners** that reflect
the physical distribution of storage across the SoC. They exist to let us
**interrogate L3/L2/L1 resources for monitoring and debug** (occupancy, capacity,
per-element state). They **must NOT** be part of any **ResourceManager API** or be
used for **domain-flow / dataflow functionality** — precisely because these
components are physically distributed. The actual data movement, reuse, and
credit flow through the hierarchy is driven by the **distributed CSP engines**
(DMA, BlockMover, Streamer) executing a global schedule. *The Layers observe; the
CSPs drive.*

## The Defect Being Corrected

We think about "the L3" as a single SoC-wide resource, but it is physically a
**distributed** structure: a mesh of `L3Tile`s, a NoC moving lines/blocks/packets
between them, and access ports into the layer. Today the code has no type for
"the L3 as a whole," so the *element* (`L3Tile`) and its flat collection
(`l3_tiles`) are forced to stand in for the aggregate. That conflation is
unsustainable and is the root of the naming confusion in #32/#33/#34.

The original issues proposed renaming `L3Tile → L3Layer` (etc.). That is the
**wrong fix** — it relabels the mesh *element* as the *whole layer*, destroys the
(correct) tile concept, and leaves the interconnect homeless. The correct fix is
to **introduce the missing aggregate** and let it *own* the retained elements.

## Current State (evidence)

| Aspect | Finding | Evidence |
|--------|---------|----------|
| Aggregate type | None. Simulator holds flat `std::vector<L3Tile/L2Bank/L1Buffer>` | `kpu_simulator.hpp:144–146` |
| Element accessed by | Flat index (`l3_tiles[tile_id]`) | `kpu_simulator.cpp` process loop |
| Layer property leaking into element config | `L3TileConfig::num_tiles` (a *layer* count living on an *element* config) | `component_config.hpp:183` |
| Flat layer config on simulator | `l3_tile_count`, `l3_tile_capacity_kb`, … | `kpu_simulator.hpp:71–74` |
| L3 interconnect | `L3Interconnect` (2D mesh, `L3Position{row,col}`, N/S/E/W/LOCAL, Manhattan routing) **exists but is NOT wired into the simulator** | `l3_interconnect.hpp` |
| Wormhole NoC | `noc::NoC` (flit-level, virtual channels, credit flow) **exists, not integrated** | `noc.hpp` |
| Ports | Modeled **per element** (`L3TileConfig::num_ports`=8, `L2BankConfig::num_ports`=2), not at layer level | `component_config.hpp:192,207` |
| L2 topology | None. Flat `bank_id`, no coordinates, no interconnect | — |
| L1 ownership | Documented as part of the **compute fabric**; count derived `4×(rows+cols)` per compute tile | `l1_buffer.hpp:24–35`, `processor_array_topology.hpp:97–131` |

**Takeaway:** the pieces of `L3Layer` already exist (tiles + `L3Interconnect` +
ports) but are unbound. We are introducing the binding type, not inventing new
mechanics from scratch.

## Target Architecture

```
                          ┌──────────────────────────────────────────┐
                          │              KPUSimulator                 │
                          └──────────────────────────────────────────┘
                                 │                │              │
                                 ▼                ▼              ▼
                     ┌───────────────────┐ ┌──────────────┐ ┌────────────────────┐
                     │     L3Layer       │ │   L2Layer    │ │   ComputeFabric    │
                     │ (SoC-wide L3)     │ │ (SoC-wide L2)│ │                    │
                     │                   │ │              │ │  owns L1Buffers     │
                     │  ┌─────┐ ┌─────┐  │ │ ┌────┐ ┌────┐│ │  (stream buffers)  │
                     │  │L3Tile│ │L3Tile│ │ │ │L2  │ │L2  ││ │  ┌────┐ ┌────┐    │
                     │  └─────┘ └─────┘  │ │ │Bank│ │Bank││ │  │L1  │ │L1  │    │
                     │   NoC + ports +   │ │ └────┘ └────┘│ │  │Buf │ │Buf │    │
                     │   BlockMovers     │ │  + ports     │ │  └────┘ └────┘    │
                     └───────────────────┘ └──────────────┘ └────────────────────┘
                              ▲                    ▲                  ▲
                              │ DMA (addr-based)   │ BlockMover       │ Streamer
                              │                    │ (owned by        │ (L2Bank shape
                              │                    │  L3Layer;        │  → L1Buffer shape)
                              │                    │  L3Tile→L2Bank)  │
   Host / External Memory ────┘                    └──────────────────┘
   (BlockMovers are owned by L3Layer — one attached per L3Tile — and push tiles downstream into the L2Layer)
```

### Ownership decisions

| Concept | Type | Owns | Notes |
|---------|------|------|-------|
| L3 layer (whole) | **`L3Layer`** *(new)* | collection of `L3Tile`, `L3Interconnect`/NoC, **the per-tile `BlockMover`s**, layer ports | aggregate; may be **non-uniform** (see below) |
| L3 element | `L3Tile` *(retained)* | own capacity, internal banks, own ports | addressable mesh element |
| L3→L2 mover | `BlockMover` *(retained, ownership moves)* | pushes a tile from its `L3Tile` down into the `L2Layer` | **owned by `L3Layer`**, one attached per `L3Tile` (was a flat `std::vector<BlockMover>` on the simulator, `kpu_simulator.hpp:150`) |
| L2 layer (whole) | **`L2Layer`** *(new)* | collection of `L2Bank`, layer ports, (interconnect TBD) | aggregate; may be **non-uniform** |
| L2 element | `L2Bank` *(retained)* | own capacity, ports | addressable element |
| L1 layer (whole) | **`L1Layer`** *(new)* | collection of `L1Buffer` stream buffers | aggregate; owner for monitoring/debug; may be **non-uniform** |
| L1 element | `L1Buffer` *(retained)* | own capacity, double-buffering | addressable stream buffer feeding the compute fabric |
| L2→L1 shape translation | `Streamer` *(existing role, made explicit)* | translates L2 bank shape → L1 buffer shape | the entity that bridges memory layer and compute |

### On `L1Layer` (decision updated 2026-06-28)

An earlier draft argued there should be **no** `L1Layer`: L1 buffers are derived
from and lifecycle-bound to the compute tiles (`4×(rows+cols)` per tile), so they
"belong" to the `ComputeFabric`. **That reasoning is rejected:**

- It does not distinguish L1 from L2. The size/shape of the `L2Layer`'s `L2Bank`
  structures is **also** a function of the fabric size and kernel schedule — the
  same properties that size the L1 buffers. If derivation-from-fabric justified
  folding L1 into compute, it would equally justify folding L2 into compute,
  which we do not do.
- Under the monitoring/ownership framing above, an L1 owner is exactly what we
  want for efficient inspection of the L1 resources. Symmetry of the hierarchy
  (L3/L2/L1 each have an owner) is the right model.

So **`L1Layer` is introduced** as a peer of `L2Layer`/`L3Layer`, owning the
`L1Buffer` stream buffers. It is an *ownership/monitoring* structure only (see the
principle above) — it is **not** wired into any ResourceManager or dataflow path,
and the `Streamer` remains the CSP that actually drives L2→L1 movement.

## Key Requirement: Non-Uniform Layers (heterogeneous compute fabrics)

The KPU roadmap allows **different compute fabrics** in the same machine. Different
fabrics impose different demands on the memory layers feeding them, so:

- An `L3Layer` may contain **`L3Tile`s of differing shape/capacity/port count**.
- An `L2Layer` may contain **`L2Bank`s of differing bank shape and port shape**.

Therefore the aggregates **must not** assume a single uniform element template:

- `L3Layer`/`L2Layer` hold their elements as collections that permit
  **per-element configuration**, not `count × identical_config`.
- These heterogeneous groupings are managed as **named groups** of the layer.
  The config shape is **`group → element-config → multiplicity`**: each named
  group binds one element configuration (e.g. an `L3TileConfig`) to a count of
  identical elements, and a layer holds many such groups. This lets a layer
  describe, say, a high-bandwidth tile group feeding a systolic fabric alongside
  a different group feeding another fabric.
- The current `L3TileConfig::num_tiles` (single uniform multiplier) is explicitly
  **insufficient** and is superseded by this `group → element-config →
  multiplicity` layer config.

This requirement is a first-class design constraint, not a future nice-to-have:
the config schema and the aggregate types must support non-uniformity from the
start, even if the initial systems happen to be uniform.

## Configuration Restructuring

Split today's flat/element-conflated config into **layer config** vs **element
config**:

| Today | Target |
|-------|--------|
| `L3TileConfig { num_tiles, capacity_kb, num_banks, num_ports, … }` | `L3LayerConfig { mesh/topology, interconnect, block_mover config, layer ports, tile_groups: { name → (L3TileConfig, multiplicity) } }` + `L3TileConfig { capacity_kb, num_banks, num_ports, … }` (per element/group) |
| `L2BankConfig { capacity_kb, num_ports }` + simulator-level `l2_bank_count` | `L2LayerConfig { layer ports, bank_groups: { name → (L2BankConfig, multiplicity) } }` + `L2BankConfig { capacity_kb, num_ports, shape }` (per element/group) |
| `L1BufferConfig` under memory | `L1BufferConfig` under `ComputeFabricConfig` |
| `KPUSimulator::Config { l3_tile_count, l3_tile_capacity_kb, l2_bank_count, … }` | `KPUSimulator::Config { L3LayerConfig, L2LayerConfig, ComputeFabricConfig(incl. L1) }` |

Each named group follows the **`group → element-config → multiplicity`** shape
(a group binds one element config to a count of identical elements; a layer holds
many groups).

JSON system configs (`configs/systems/*.json`, `configs/schema.md`) gain
`l3_layer` / `l2_layer` objects describing topology + named element groups.
**No backward compatibility** is provided for the old flat keys — the current flat
design is wrong and is replaced outright (see Public API Impact).

## Migration & Sequencing

Each phase must compile and keep tests green before the next begins.

1. **Phase 0 — Design sign-off.** This document approved; #32/#33/#34 re-scoped;
   #35 promoted to design epic.
2. **Phase 1 — `L3Layer` (issue #32).** Introduce `L3Layer` owning the existing
   `L3Tile` collection, the `L3Interconnect`/`NoC`, **and the per-tile
   `BlockMover`s** (relocated from the simulator's flat `block_movers` vector).
   Move `num_tiles`/topology out of `L3TileConfig` into `L3LayerConfig` with
   `group → element-config → multiplicity` tile groups. `L3Layer` owns the
   interconnect instance (wiring routing live is a separate follow-up). Update
   simulator to hold `L3Layer` instead of `std::vector<L3Tile>` and update all
   call sites in place — **no compatibility shims** for the old flat accessors.
3. **Phase 2 — `L2Layer` (issue #33).** Mirror Phase 1 for L2: `L2Layer` owns
   `L2Bank`s with named, non-uniform bank/port groups. Decide L2 interconnect
   (likely deferred).
4. **Phase 3 — L1 → compute fabric (issue #34).** Move `L1Buffer` ownership into
   `ComputeFabric`; relocate `L1BufferConfig` under `ComputeFabricConfig`;
   formalize `Streamer` as the L2-bank-shape → L1-buffer-shape translator. Remove
   the simulator-level flat `l1_buffers` vector.
5. **Phase 4 — Config & docs.** Update JSON schema + system configs, public-API
   surface, and current reference docs. No deprecation aliases (hard break).

## Public API Impact

This is a **hard break** to `KPUSimulator::Config` and the flat
`read_l3_tile`/`l3_tile_count`/etc. accessors. **No backward compatibility is
provided** — the current flat design is wrong, so the old config fields and
accessors are **replaced outright**, not retained as `[[deprecated]]` shims. All
consumers are updated in place, with a CHANGELOG migration note describing the new
`L3Layer`/`L2Layer`/`ComputeFabric` config surface.

Python bindings (`python/kpu_bindings.cpp`) and `kpu-loader` consume these names
and must be updated in lockstep.

## Resolved Decisions

- **`L3Layer` ownership.** `L3Layer` owns the `L3Interconnect`/`NoC` **and** the
  per-tile `BlockMover`s (one attached per `L3Tile`), in addition to the `L3Tile`
  collection and layer ports.
- **Named-group config shape.** `group → element-config → multiplicity` — a named
  group binds one element config to a count of identical elements; a layer holds
  many groups. Supersedes `L3TileConfig::num_tiles`.
- **No backward compatibility.** The flat `KPUSimulator::Config` fields and
  `read_l3_tile`/`l3_tile_count`/etc. accessors are replaced outright (no
  `[[deprecated]]` shims) — the current flat design is wrong.
- **Element names retained.** `L3Tile`/`L2Bank`/`L1Buffer` keep their names — they
  are addressable, configurable entities that matter to machine operation.

## Open Questions

1. **L2 interconnect.** Does `L2Layer` get a topology/NoC now, or is it a flat
   aggregate until a use case demands it? (Recommend: flat now, structured later.)
2. **Live L3 routing.** `L3Layer` owns the interconnect; do we integrate routing
   into the base simulation in this effort, or only establish the ownership seam
   and defer live routing to a separate issue? (Recommend: ownership seam now,
   live routing later.)

## Issue Mapping (re-scope)

| Issue | Was | Becomes |
|-------|-----|---------|
| #35 | Tracking: rename L*→L*Layer | **Design epic**: introduce aggregate layers; this plan |
| #32 | Rename `L3Tile`→`L3Layer` | **Introduce `L3Layer`** aggregate owning retained `L3Tile`s + interconnect + ports; non-uniform tile groups |
| #33 | Rename `L2Bank`→`L2Layer` | **Introduce `L2Layer`** aggregate owning retained `L2Bank`s + ports; non-uniform bank/port groups |
| #34 | Rename `L1Buffer`→`L1Layer` | **Introduce `L1Layer`** aggregate owning retained `L1Buffer`s (monitoring/ownership only, not a dataflow API); `Streamer` remains the L2→L1 CSP driver |
