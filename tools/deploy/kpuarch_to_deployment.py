#!/usr/bin/env python3
"""
The KPU RTL generator's machine as a kpu-sim deployment (#396).

The RTL generator (branes-ai/embodied-ai-soc-generator, kpugen) describes a KPU with an
embodied-schemas KPUArchitecture: the expanded YAML `kpugen expand` writes (configs/*.full.yaml).
kpu-sim describes a machine with a DeploymentSpec (deployment_spec.hpp). This tool maps the first
onto the second, so the two sides can be shown to model the same machine.

It maps every field of the architecture it reads, or names it:
- MAPPED: a field with a DeploymentSpec counterpart, converted (KiB to tile counts, bits to bytes).
- FROM THE GENERATOR'S CODE: the tile's slots, banks and latencies are kpugen's SystolicConfig
  defaults and tile.py's constants, not KPUArchitecture fields. They are listed below, pinned
  to the kpugen revision they were read from, and written into the `systolic` block.
- NOT CARRIED, each with its reason: energy estimates, kernel lists, circuit classes other than
  the memories', and (single-tile) the 2x2 grid and its NoC.
- REFUSED: any field this tool does not know, or a value it cannot represent (a second tile
  class, a MAC with several modes or lanes, local memory per PE). A field is never dropped
  silently.

What kpu-sim needs and the architecture does not say (the DMA and mover rates, the DRAM's bank
geometry) keeps kpu-sim's defaults. The tool prints that list, and --notes writes it to a file.

Single tile first (v0.9.0): the deployment is ONE of the architecture's compute tiles, topology
single. All four tiles of a 2x2 is v0.12.0 (--tiles all is refused until then).

Usage:
    python3 kpuarch_to_deployment.py configs/kpu_2x2_systolic.full.yaml -o kpu_soc_2x2.json
    python3 kpuarch_to_deployment.py arch.json -o dep.json --notes notes.txt

Reads YAML (PyYAML) or JSON. Exit 0 written, 1 refused, 2 unreadable.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import json
import sys
from pathlib import Path

# kpugen's SystolicConfig defaults and tile.py constants the deployment's `systolic` block takes,
# read from branes-ai/embodied-ai-soc-generator (src/kpugen/tiles/systolic/config.py and tile.py).
# KPUArchitecture does not carry them; when kpugen changes them, this table changes with it.
GENERATOR_DEFAULTS = {
    "l2_slots": 4,              # config.py: l2_slots
    "acc_slots": 2,             # config.py: acc_slots
    "sp_banks": 4,              # tile.py: SP_BANKS
    "acc_banks": 2,             # tile.py: ACC_BANKS
    "spad_read_delay": 4,       # config.py: spad_read_delay
    "acc_latency": 2,           # config.py: acc_latency
    "tile_rows": 1,             # tile.py _config: the grid is the array, one PE per tile
    "tile_cols": 1,
}
# The MAC formats a deployment can state: whole-byte integers. A floating-point format has no field
# in DeploymentSpec yet (its bits alone would read as an integer's), and int4 needs a packing
# contract: both are refused until the deployment can represent them.
BITS = {"int8": 8, "int16": 16, "int32": 32}
DATAFLOWS = {"output_stationary", "weight_stationary", "input_stationary"}
DRAM = {"lpddr5": "lpddr5", "lpddr5x": "lpddr5x", "ddr5": "ddr5", "gddr6": "gddr6", "gddr7": "gddr7",
        "hbm2": "hbm2", "hbm3": "hbm3"}             # deployment_spec.hpp known_dram_technology

# Fields read but not carried into the deployment, with the reason. Anything outside these tables
# and the mapped fields is refused.
NOT_CARRIED = {
    "tile_type": "a label",
    "tile_class_id": "a label (the deployment's device name says which tile)",
    "tile_class_ref": "a label",
    "circuit_class": "the array's logic circuit class: an energy and area input, not a timing one",
    "supported_kernels": "what the tile's FSMs run; a program says what it computes",
    "notes": "free text",
    "mac.unit_id": "a label",
    "mac.op": "mac (the only op a systolic tile has)",
    "mac.notes": "free text",
    "mac.modes[0].issue_interval_cycles": "1: one MAC per PE per cycle, which macs_per_cycle already states",
    "mac.modes[0].energy": "an energy estimate; kpu-sim's analytical energy is its own",
    "multi_precision_alu": "the vector ALU's formats; kpu-sim's vector units are movers.vector",
    "memory.l2_kib_per_tile": "0: the L2 is the tile's local_memory (weight_buffer)",
    "memory.l1_kib_per_tile": "0: no L1 beyond the array's edge",
}
GRID_FIELDS = {
    "noc": "the 2x2 grid's mesh: single tile first; all four tiles are v0.12.0",
    "checkerboard": "the 2x2 grid's sites: single tile first; all four tiles are v0.12.0",
    "total_tiles": "the architecture's 4 tiles: the deployment is one of them",
    "num_tiles": "4 tiles of this class: the deployment is one of them",
}


class Refused(Exception):
    pass


def _no_duplicates(pairs, where="JSON"):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"{where}: the key {k!r} appears twice (a converter must not pick one)")
        out[k] = v
    return out


def load(path):
    """The spec, refusing a key that appears twice in one mapping: YAML and JSON loaders keep the
    last value silently, which would change the converted machine without a word."""
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:
            raise ValueError("reading YAML needs PyYAML; convert to JSON first") from e

        class Strict(yaml.SafeLoader):
            pass

        def mapping(loader, node, deep=False):
            loader.flatten_mapping(node)
            pairs = [(loader.construct_object(k, deep=deep), loader.construct_object(v, deep=deep))
                     for k, v in node.value]
            return _no_duplicates(pairs, f"YAML line {node.start_mark.line + 1}")

        Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        return yaml.load(text, Loader=Strict)  # noqa: S506 -- Strict is a SafeLoader
    return json.loads(text, object_pairs_hook=_no_duplicates)


def convert(doc, tiles="one"):
    """The deployment (a dict, kpu-sim's JSON form) and the notes: what was carried, what keeps
    kpu-sim's defaults, and what was read but not carried."""
    if tiles != "one":
        raise Refused("--tiles all: the 2x2 as four compute tiles is v0.12.0; v0.9.0 converts one tile")
    if not isinstance(doc, dict) or "architecture" not in doc or "name" not in doc:
        raise ValueError("not an expanded KPU spec (name + architecture): run `kpugen expand` first")
    extra = set(doc) - {"name", "architecture"}
    if extra:
        raise Refused(f"top-level fields {sorted(extra)}: not a KPUArchitecture field")
    a, name = doc["architecture"], doc["name"]
    carried, defaulted, dropped = [], [], []

    def take(obj, key, where, required=True):
        if key not in obj:
            if required:
                raise Refused(f"{where}.{key} is missing")
            return None
        return obj[key]

    known = {"total_tiles", "tiles", "noc", "memory", "multi_precision_alu", "checkerboard"}
    for k in a:
        if k not in known:
            raise Refused(f"architecture.{k}: this converter does not know it (power_domains and the like "
                          "are not modelled; add them here, mapped or named, rather than drop them)")
    ts = take(a, "tiles", "architecture")
    if len(ts) != 1:
        raise Refused(f"architecture.tiles has {len(ts)} tile classes; a deployment is one kind of compute tile")
    t = ts[0]
    if t.get("tile_kind") != "systolic":
        raise Refused(f"tiles[0].tile_kind is {t.get('tile_kind')!r}; this converter maps a systolic tile")
    tile_known = {"tile_kind", "array_rows", "array_cols", "dataflow", "mac", "local_memory", "num_tiles"} | \
        {k for k in NOT_CARRIED if "." not in k}
    for k in t:
        if k not in tile_known:
            raise Refused(f"tiles[0].{k}: this converter does not know it")
    rows, cols = take(t, "array_rows", "tiles[0]"), take(t, "array_cols", "tiles[0]")
    dataflow = take(t, "dataflow", "tiles[0]")
    if dataflow not in DATAFLOWS:
        raise Refused(f"tiles[0].dataflow is {dataflow!r}; kpu-sim's systolic block names {sorted(DATAFLOWS)}")
    mac = take(t, "mac", "tiles[0]")
    for k in mac:
        if k not in ("unit_id", "op", "modes", "notes"):
            raise Refused(f"tiles[0].mac.{k}: this converter does not know it")
    modes = take(mac, "modes", "tiles[0].mac")
    if len(modes) != 1:
        raise Refused(f"tiles[0].mac has {len(modes)} modes; a deployment states one number system per tile")
    m = modes[0]
    for k in m:
        if k not in ("operand_format", "accumulate_format", "lanes", "issue_interval_cycles", "energy"):
            raise Refused(f"tiles[0].mac.modes[0].{k}: this converter does not know it")
    if m.get("lanes", 1) != 1 or m.get("issue_interval_cycles", 1) != 1:
        raise Refused("tiles[0].mac.modes[0]: lanes and issue_interval_cycles must be 1 (one MAC per PE per cycle)")
    inp, acc = take(m, "operand_format", "tiles[0].mac.modes[0]"), take(m, "accumulate_format", "tiles[0].mac.modes[0]")
    if inp not in BITS or acc not in BITS:
        raise Refused(f"the MAC's formats {inp} -> {acc}: a deployment states whole-byte integer formats "
                      f"({', '.join(BITS)}); floating-point and int4 have no DeploymentSpec field yet")
    mem = {}
    for lm in take(t, "local_memory", "tiles[0]"):
        for k in lm:
            if k not in ("level", "kib", "per", "circuit_class"):
                raise Refused(f"tiles[0].local_memory[].{k}: this converter does not know it")
        if lm.get("per", "tile") != "tile":
            raise Refused(f"local_memory {lm['level']} is per {lm['per']}; this converter maps per-tile memories")
        if lm["level"] not in ("weight_buffer", "accumulator"):
            raise Refused(f"local_memory {lm['level']}: a systolic tile's memories are its weight_buffer "
                          "(scratchpad) and accumulator")
        mem[lm["level"]] = lm
    for need in ("weight_buffer", "accumulator"):
        if need not in mem:
            raise Refused(f"tiles[0].local_memory has no {need}")

    # The tile's memories, in rows as the RTL banks them (kpugen config.py: rows per bank =
    # KiB * 8192 / (banks * width), the scratchpad row cols inputs wide, the accumulator's cols
    # accumulators wide).
    g = GENERATOR_DEFAULTS
    sp_bits = int(mem["weight_buffer"]["kib"] * 8192)
    acc_bits_total = int(mem["accumulator"]["kib"] * 8192)
    sp_row = cols * BITS[inp]
    acc_row = cols * BITS[acc]
    if sp_bits % (g["sp_banks"] * sp_row) or acc_bits_total % (g["acc_banks"] * acc_row):
        raise Refused("the local memories do not divide into whole rows per bank")
    systolic = {
        "dataflow": dataflow,
        "input_bits": BITS[inp], "weight_bits": BITS[inp], "acc_bits": BITS[acc],
        "grid_rows": rows, "grid_cols": cols, "tile_rows": g["tile_rows"], "tile_cols": g["tile_cols"],
        "l2_slots": g["l2_slots"], "acc_slots": g["acc_slots"],
        "sp_banks": g["sp_banks"], "sp_rows_per_bank": sp_bits // (g["sp_banks"] * sp_row),
        "acc_banks": g["acc_banks"], "acc_rows_per_bank": acc_bits_total // (g["acc_banks"] * acc_row),
        "spad_read_delay": g["spad_read_delay"], "acc_latency": g["acc_latency"],
        "sp_macro": mem["weight_buffer"].get("circuit_class", "sram_hd"),
        "acc_macro": mem["accumulator"].get("circuit_class", "sram_hd"),
    }
    carried += [f"tiles[0].array_rows/array_cols -> fabric {rows} x {cols}, macs_per_cycle {rows * cols}",
                f"tiles[0].dataflow -> systolic.dataflow {dataflow}",
                f"tiles[0].mac.modes[0] {inp} -> {acc} -> systolic.input_bits/weight_bits {BITS[inp]}, acc_bits "
                f"{BITS[acc]}; element_bytes {BITS[inp] // 8}",
                f"local_memory weight_buffer {mem['weight_buffer']['kib']} KiB -> systolic.sp_rows_per_bank "
                f"{systolic['sp_rows_per_bank']} ({g['sp_banks']} banks of {sp_row}-bit rows)",
                f"local_memory accumulator {mem['accumulator']['kib']} KiB -> systolic.acc_rows_per_bank "
                f"{systolic['acc_rows_per_bank']} ({g['acc_banks']} banks of {acc_row}-bit rows)"]
    defaulted += [f"systolic.{k} = {v}: kpugen's SystolicConfig / tile.py value, not a KPUArchitecture field"
                  for k, v in g.items()]

    # L3: the per-tile L3 in tile-sized buffers (a tile is the array's rows x cols elements of
    # the widest type, the accumulator's).
    memory = take(a, "memory", "architecture")
    for k in memory:
        if k not in ("memory_type", "memory_size_gb", "memory_bus_bits", "memory_bandwidth_gbps", "memory_controllers",
                     "l3_kib_per_tile", "l2_kib_per_tile", "l1_kib_per_tile"):
            raise Refused(f"architecture.memory.{k}: this converter does not know it")
    tile_bytes = rows * cols * BITS[acc] // 8
    l3_bytes = int(memory["l3_kib_per_tile"] * 1024)
    if l3_bytes % tile_bytes:
        raise Refused(f"l3_kib_per_tile {memory['l3_kib_per_tile']} is not a whole number of {tile_bytes}-byte tiles")
    for k in ("l2_kib_per_tile", "l1_kib_per_tile"):
        if memory.get(k, 0):
            raise Refused(f"memory.{k} is {memory[k]}; the tile's L2 is its local_memory, and it has no L1 beyond its edge")
    carried.append(f"memory.l3_kib_per_tile {memory['l3_kib_per_tile']} -> l3.capacity_tiles {l3_bytes // tile_bytes} "
                   f"({tile_bytes}-byte tiles)")

    # DRAM: what the architecture states. Its bank geometry is not stated: kpu-sim's defaults.
    tech = memory["memory_type"]
    if tech not in DRAM:
        raise Refused(f"memory.memory_type {tech!r}: kpu-sim's DRAM technologies are {sorted(DRAM)}")
    size = int(round(memory["memory_size_gb"] * (1 << 30)))
    if size & (size - 1):
        raise Refused(f"memory_size_gb {memory['memory_size_gb']}: kpu-sim's DRAM capacity is a power of two bytes")
    bus = memory["memory_bus_bits"]
    if bus % 16:
        raise Refused(f"memory_bus_bits {bus}: not a whole number of x16 channels")
    rate = memory["memory_bandwidth_gbps"] * 1e9 / (bus / 8) / 1e6
    if abs(rate - round(rate)) > 1e-6:
        raise Refused(f"memory_bandwidth_gbps {memory['memory_bandwidth_gbps']} over {bus} bits is not a whole MT/s")
    controllers = memory["memory_controllers"]
    channels = bus // 16
    dram = {"technology": DRAM[tech], "channels": channels // controllers, "channel_width_bits": 16,
            "data_rate_mtps": int(round(rate)), "capacity_bytes": size}
    carried.append(f"memory.memory_controllers {controllers} -> memory.controllers")
    # kpu-sim's LPDDR5 controller drives one or two x16 channels (dram_bridge). A wider bus on one
    # controller is a machine kpu-sim's DRAM model cannot host: the block is not carried, and said so.
    if tech.startswith("lpddr") and (channels % controllers or channels // controllers > 2):
        dram = None
        dropped.append(f"memory.memory_type {tech}, memory_bus_bits {bus}, memory_bandwidth_gbps "
                       f"{memory['memory_bandwidth_gbps']}, memory_size_gb {memory['memory_size_gb']}: "
                       f"{channels} x16 channels on {controllers} controller(s), and kpu-sim's LPDDR controller "
                       "drives 1 or 2; memory.dram is not carried (kpu-sim's built-in memory timing stands in)")
    else:
        carried += [f"memory.memory_type {tech} -> memory.dram.technology",
                    f"memory.memory_bus_bits {bus} -> {channels // controllers} x16 channels per controller",
                    f"memory.memory_bandwidth_gbps {memory['memory_bandwidth_gbps']} -> memory.dram.data_rate_mtps "
                    f"{int(round(rate))}",
                    f"memory.memory_size_gb {memory['memory_size_gb']} -> memory.dram.capacity_bytes {size}"]
        defaulted.append("memory.dram ranks, bank groups, banks, page and burst bytes, address map: kpu-sim's "
                         "defaults (the architecture does not state them)")
    defaulted += [f"dma.engines = {controllers}: one per memory controller, kpu-sim's choice (no DMA in the "
                  "architecture; the engines must divide among the controllers)",
                  "dma.bytes_per_cycle: kpu-sim's default (no DMA in the architecture)",
                  "movers.*: kpu-sim's defaults (no mover rates in the architecture)",
                  "clock: none (KPUArchitecture has no clock; kpu-sim counts cycles)"]

    # What the architecture says that the deployment does not carry, each named.
    for k, why in NOT_CARRIED.items():
        parts = k.split(".")
        obj, ok = t, True
        if parts[0] == "memory":
            obj = memory
            parts = parts[1:]
        elif parts[0] == "multi_precision_alu":
            obj = a
        for p_ in parts:
            if p_ == "modes[0]":
                obj = modes[0] if obj is mac else None
            elif isinstance(obj, dict) and p_ in obj:
                obj = obj[p_]
            else:
                ok = False
                break
        if ok and obj is not None:
            dropped.append(f"{k}: {why}")
    for k, why in GRID_FIELDS.items():
        if k in a or k in t:
            dropped.append(f"{k}: {why}")

    device = {
        "name": f"{name}.tile0",
        "topology": "single",
        "compute_tiles": 1,
        "macs_per_cycle": rows * cols,
        "fabric": {"pe_rows": rows, "pe_cols": cols},
        "systolic": systolic,
        "element_bytes": BITS[inp] // 8,
        "dma": {"engines": controllers},
        "l3": {"tiles": 1, "capacity_tiles": l3_bytes // tile_bytes},
        "memory": {"controllers": controllers, **({"dram": dram} if dram else {})},
    }
    return {"devices": [device]}, {"carried": carried, "defaulted": defaulted, "not carried": dropped}


def main():
    ap = argparse.ArgumentParser(description="The KPU RTL generator's KPUArchitecture as a kpu-sim deployment.")
    ap.add_argument("arch", help="the expanded spec (kpugen expand): YAML or JSON")
    ap.add_argument("-o", "--output", required=True, help="the deployment JSON to write")
    ap.add_argument("--notes", help="also write what was carried, defaulted and not carried")
    ap.add_argument("--tiles", choices=("one", "all"), default="one")
    a = ap.parse_args()
    try:
        doc = load(a.arch)
        dep, notes = convert(doc, a.tiles)
    except (OSError, ValueError) as e:
        print(f"kpuarch_to_deployment: {a.arch}: unreadable: {e}", file=sys.stderr)
        return 2
    except Refused as e:
        print(f"kpuarch_to_deployment: {a.arch}: refused: {e}", file=sys.stderr)
        return 1
    Path(a.output).write_text(json.dumps(dep, indent=2) + "\n", encoding="utf-8")
    text = "".join(f"{k}:\n" + "".join(f"  - {x}\n" for x in v) for k, v in notes.items())
    if a.notes:
        Path(a.notes).write_text(text, encoding="utf-8")
    print(f"wrote {a.output} (one compute tile of {doc['name']})")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
