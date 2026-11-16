# The Shared Structure Record: the Machine's Structure and Floorplan (#392)

**Status:** version 1, 2026-11-16. The first of the shared formats ADR 0005 calls for: it comes
before the views that read it (#447 Overview, #448 Investigate, #390 occupancy, #391 tile flow).
**Format:** `kpu-structure`, version 1. Schema: `tools/trace/structure.schema.json`. Checker:
`tools/trace/structure_check.py`.

## 1. What it is for

One description of a machine's hardware, written by whichever producer has it:
- **the model:** kpu-sim, from a deployment spec (`kpu-floorplan`);
- **the RTL:** kpu-gen, from the elaborated design and its floorplanner
  (branes-ai/embodied-ai-soc-generator#29 for the structure, #58 for the floorplan).

Every view of the machine lays out exactly what the record declares. A model record and an RTL
record of the same configuration can be compared block for block, and a mismatch is a bug in one
of them.

It replaces two formats with one:
- esg's `floorplan.json` v1 (`kpu-gen-floorplan`), in mm, with blocks named by RTL instance path;
- kpu-sim's `kpu-floorplan` v1 JSON, in µm, with blocks named by the naming map.

Both convert losslessly into it: `tools/trace/structure_from_floorplan.py`. kpu-sim writes it
natively, `kpu-floorplan --structure` (§5.1); the RTL generator is to follow (§7).

## 2. The document

```json
{
  "format": "kpu-structure",
  "version": 1,
  "provenance": { ... },
  "units": {"length": "mm", "area": "mm2"},
  "blocks": [ ... ],
  "channels": [ ... ],
  "networks": [ ... ],
  "floorplan": { ... }
}
```

`format` and `version` are required, and a reader refuses any other. `blocks` is required. The
rest are optional: a producer writes what it knows.

### 2.1 `provenance`

| Key | Value |
|---|---|
| `producer` | Required. The tool and its version: `kpu-floorplan kpu-sim 0.10.0`, `kpu-gen <sha>` |
| `source` | Required. `model` (kpu-sim) or `rtl` (an RTL generator) |
| `configuration` | The configuration's name: the deployment's device, or the RTL spec |
| `configuration_digest` | Its digest: the deployment digest, or the spec's SHA-256 |
| `from` | When the record was converted: the original format and version, `kpu-gen-floorplan 1` |
| `details` | A producer's own fields, kept as given: esg's process node, area model, `as_built`, `rtl` |

### 2.2 `units`

Lengths in **mm**, areas in **mm²**, origin at the die's lower-left corner. The model's floorplan
is on an illustrative pitch, not silicon: its producer says so in `floorplan.estimated`.

### 2.3 `blocks`: the block tree

One object per block, flat, linked by `parent`. The tree is the level-of-detail ladder a view
drills down.

| Key | Value |
|---|---|
| `id` | Required, unique. Stable: `.dflow` events and other records reference it. The RTL's instance path (`soc.checkerboard.tile_0_0`), or the naming map's resource name (`t4/l3[0]/bm[1]`) |
| `kind` | Required, from the vocabulary in §3 |
| `parent` | The parent block's `id`; absent for a root |
| `label` | A short human label: `l3[0]`, `E`, `hart0` |
| `resource` | The naming map's resource, when the block is one (the model always; the RTL when it knows) |
| `rtl_path` | The RTL instance path, when there is one |
| `layer` | The KPU memory layer it belongs to: `dram`, `l3`, `l2`, `l1`, `fabric` |
| `role` | Its CSP process role: `dma`, `block_mover`, `streamer`, `fabric`, `combiner` |
| `clock_domain` | `l3`, `cf`, `dram`, `cpu`, ... |
| `class` | A tile's class: `systolic_int8_ws` |
| `sites` | The checkerboard sites it covers, `[[row, col], ...]` |
| `generated` | Whether the RTL generator emits RTL for it (false for a PHY or the I/O ring) |
| `resources` | Its typed resources (§2.4) |
| `rect`, `area` | Its placement, `[x, y, width, height]` in mm, and its envelope area in mm². Part of the floorplan |
| `notes` | Free text |

### 2.4 `resources`

Each block's resources, typed and with units (#392's table), as `{kind, ...fields}`:

| `kind` | Fields |
|---|---|
| `memory` | `bytes`, `depth`, `width_bits`, `read_ports`, `write_ports`, `banks`, `latency_cycles` |
| `slot_pool` | `slots` (whole-tile credits), `producer`, `consumer` (block ids) |
| `queue` | `depth`, `feeds` (a block id) |
| `stream_buffer`, `result_buffer` | `count`, `depth`, `credits` |
| `pe_grid` | `rows`, `cols`, `grouping`, `operand_bits`, `accumulator_bits`, `macs` |
| `bus` | `width_bits`, `protocol` (`axi4`, `axi4_lite`, `noc_flit`) |

KPU and CSP terms throughout, never cache terms: a slot pool holds credits; nothing hits or misses.

### 2.5 `channels`

A channel groups ports between two blocks: `{id, kind, from, to, width_bits}`, where `kind` is
`data`, `command` or `sync` (credits and tile-ready tokens), and `from` and `to` are block ids.

### 2.6 `networks`

The machine's interconnects, each drawn over the blocks:

```json
{"id": "noc_req", "kind": "noc_request", "routing": "xy", "description": "...",
 "nodes": [{"id": "...", "kind": "router", "block": "...", "at": [x, y], "level": 2}],
 "links": [{"id": "...", "from": "<node>", "to": "<node>", "points": [[x, y], ...], "length": 0.21}]}
```

- `kind`: `noc_request`, `noc_response`, `noc` (one network for both), `ring`, `control_tree`,
  `dram_l3`.
- `routing`: the rule a view follows to derive a route between two nodes (`xy`, `ring`, `tree`),
  so it is never hard-coded in a view.
- A node's `kind`: `router`, `port`, `source`, `root`, `branch`, `leaf`, `phy`, `dma`, `cell`,
  `attach`. Its `block` names the block it stands for, when one does.
- `at` and a link's `points` (mm) and `length` (mm) are the floorplan's; a network without them
  is topology only.

### 2.7 `floorplan`

| Key | Value |
|---|---|
| `die` | `{width, height, area}` in mm and mm² |
| `estimated` | `true` when the geometry is an estimate, the model's illustrative pitch or an area model's. Each block's `rect` is then an estimate too |
| `area` | The breakdown: `{items: [{name, category, area, ...}], silicon, whitespace, die}`, with silicon + whitespace = die |

A record has a floorplan when it has `floorplan.die` and every placed block has a `rect`.

## 3. The block vocabulary

The KPU model's kinds. A producer maps its own onto them and keeps its own in `notes` or
`details` if it needs to.

| `kind` | What it is | esg `floorplan.json` v1 | `kpu-floorplan` v1 |
|---|---|---|---|
| `die` | The die | `die` | the die rect |
| `array` | The checkerboard of sites | `checkerboard` | `array` |
| `compute_tile` | A compute tile (its class says which) | `systolic`, `compute`, `fixed_function` | `compute_tile` |
| `l3_tile` | An L3 memory cell | `memory_cell` | `l3_tile` |
| `l3_bank`, `l2_bank`, `l1_vector`, `register_file` | Memories inside a tile | | the same |
| `block_mover`, `noc_router`, `noc_port` | The tile's movers and the NoC's parts | | the same |
| `memory_controller`, `dma_engine` | The memory control layer | `dma` (the host DMA) | the same |
| `dram_phy`, `io` | The periphery | `memory_phy`, `io_ring` | `dram_phy`, `io` |
| `control`, `host_bridge` | SoC control | `control`, `host_bridge` | |
| `cpu`, `cpu_hart`, `cpu_sram`, `descriptor_ring`, `completion_ring` | The host side | | the same |

## 4. Invariants (the checker)

`structure_check.py <record.json> [--json] [--verbose]`. Exit 0 when every invariant holds, 1 on a
violation, 2 when the record cannot be read: the contract of `dflow_check.py` and `tflow_check.py`.

| Id | Invariant |
|---|---|
| S1 format | `format` is `kpu-structure` and `version` is one this checker reads; `provenance.producer` and `source` are present |
| S2 ids | every block, channel, network, node and link id is unique within its kind; every block `kind` is in the vocabulary |
| S3 tree | every `parent` names a block; the tree has no cycle |
| S4 references | every channel's `from` and `to`, every node's `block`, every resource's block ids, and every link's `from` and `to` (nodes of its own network) exist |
| S5 resources | every resource's `kind` is known and its numeric fields are non-negative numbers; a memory's `bytes` agrees with `depth` x `width_bits` / 8 when all three are given |
| F1 inside | a placed block's `rect` lies inside its placed parent's (a 1 µm tolerance) and inside the die |
| F2 area | the breakdown's silicon + whitespace equals the die's area, and the die's area is width x height (to 0.1%) |
| F3 networks | a placed network's nodes lie on the die, and a link's points start at its `from` node and end at its `to` node |

F1 to F3 run when the record has a floorplan.

## 5. Converting the existing formats

`structure_from_floorplan.py <in.json> -o <out.json>` reads either format, from its `format` field:
- `kpu-gen-floorplan` v1:
  - blocks map by §3, their ids and `rect` unchanged;
  - `layers` become `networks` (`noc_req` to `noc_request` with `routing: "xy"`, `control_tree` to
    `routing: "tree"`, `dram_l3`);
  - `area` and `die` become the floorplan;
  - the manifest goes into `provenance.details`.
- `kpu-floorplan` v1:
  - the nested blocks flatten with `parent`, µm becomes mm, and each resource name is kept;
  - the NoC links become one `noc` network with `routing: "ring"`, its nodes the routers and ports
    they join;
  - `floorplan.estimated` is true, because the model's pitch is illustrative.

### 5.1 The model writes it natively

`kpu-floorplan --deploy <spec.json> --structure <out.json>` (`floorplan_to_structure`, in
`src/program/floorplan.cpp`). It is the converted record block for block: CTest
`structure_native_t4` compares them with `structure_diff.py`, and only the provenance differs.
The native provenance names the build (`kpu-floorplan kpu-sim 0.8.0`) and keeps the layout
(`generated:checkerboard`) and the floorplan's digest in `details`.

### 5.2 Comparing two records

`structure_diff.py <a> <b>`: everything but `provenance`, blocks, networks, nodes, links and
channels matched by id, numbers within a relative tolerance (1e-9 by default). Exit 0 the same, 1
a difference (each named by its path), 2 unreadable.

A model record and an RTL record of the same configuration do not compare yet: their ids are two
naming schemes (the naming map's `t4/l3[0]`, and the RTL's instance path
`soc.checkerboard.tile_0_0`). They meet when the RTL fills each block's `resource` with the naming
map's name, which §2.3 provides for; the comparison then keys on `resource`. That is part of
branes-ai/embodied-ai-soc-generator#90 (§7).

## 6. Samples

`tests/program/structure/`, the checker's and the viewers' fixtures:
- `kpu_t4.structure.json`: the model's T4 (`kpu-floorplan --deploy tests/program/deploy/kpu_t4.json`,
  converted);
- `kpugen_kpu_2x2_systolic.structure.json` and `kpugen_kpu_edge_hetero.structure.json`: the RTL
  generator's `kpu_2x2_systolic` and `kpu_edge_hetero`, converted from their `floorplan.json` v1
  (kpu-gen 756039f, as their manifests record).

CTest: `structure_record_t4` and `structure_check_t4` regenerate the model's sample from
`kpu-floorplan` and check it. `structure_check_selftest` checks every sample, and breaks each
invariant on purpose in a copy to see it reported under its id.

## 7. What follows

- ~~`kpu-floorplan --structure` writes the record natively~~ (§5.1). Its `--json` stays for one
  release, then retires.
- branes-ai/embodied-ai-soc-generator#90 (`kpu-sim-sync`): kpu-gen writes the record natively (`kpu-gen floorplan`, and its
  block-diagram generator, esg#29), fills `resource` with the naming map's names so a model
  record and an RTL record compare (§5.2), and its CI pins this checker.
- `.dflow` v6.x references block ids (the original #392 item "link to .dflow").
- The views: #447 and #448 read the floorplan, #390 the resources.
