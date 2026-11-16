#!/usr/bin/env python3
"""
Convert a floorplan into the shared structure record, `kpu-structure` v1 (#392;
docs/plans/shared-structure-record.md §5).

Reads either existing format, from its `format` field:
    kpu-gen-floorplan v1   the RTL generator's floorplan.json (mm; blocks by RTL instance path)
    kpu-floorplan v1       kpu-sim's `kpu-floorplan --json` (um; blocks by naming-map resource)

Usage:
    python3 structure_from_floorplan.py <floorplan.json> -o <out.structure.json>

Exit codes: 0 written; 2 the input could not be read or is not a format this converts.
"""
from __future__ import annotations

import argparse
import json
import sys

FORMAT = "kpu-structure"
VERSION = 1

# esg floorplan.json v1 kinds -> the KPU model's vocabulary (spec §3).
ESG_KIND = {
    "die": "die",
    "checkerboard": "array",
    "systolic": "compute_tile",
    "compute": "compute_tile",
    "fixed_function": "compute_tile",
    "memory_cell": "l3_tile",
    "memory_phy": "dram_phy",
    "io_ring": "io",
    "control": "control",
    "host_bridge": "host_bridge",
    "dma": "dma_engine",
}
ESG_NETWORK = {
    "noc_req": ("noc_request", "xy"),
    "noc_rsp": ("noc_response", "xy"),
    "control_tree": ("control_tree", "tree"),
    "dram_l3": ("dram_l3", None),
}
# kpu-floorplan v1 kinds that are not already the vocabulary's.
SIM_KIND = {"dram_phy": "dram_phy", "io": "io"}


class ConvertError(Exception):
    pass


def from_esg(d: dict) -> dict:
    m = d.get("manifest", {})
    blocks = []
    for b in d["blocks"]:
        kind = ESG_KIND.get(b["kind"])
        if kind is None:
            raise ConvertError(f"block {b['id']}: kind '{b['kind']}' has no structure-record kind")
        out = {"id": b["id"], "kind": kind, "rtl_path": b["id"]}
        if b.get("parent"):
            out["parent"] = b["parent"]
        for k in ("class",):
            if b.get(k):
                out[k] = b[k]
        if b.get("sites"):
            out["sites"] = b["sites"]
        if "generated" in b:
            out["generated"] = bool(b["generated"])
        if b.get("rect") is not None:
            out["rect"] = list(b["rect"])
        if b.get("area") is not None:
            out["area"] = b["area"]
        if b.get("notes"):
            out["notes"] = b["notes"]
        if kind != b["kind"]:
            out["notes"] = (out.get("notes", "") + f" (kpu-gen kind: {b['kind']})").strip()
        if kind == "l3_tile":
            out["layer"] = "l3"
        if kind == "compute_tile":
            out["layer"] = "fabric"
        blocks.append(out)
    networks = []
    for name, layer in d.get("layers", {}).items():
        kind, routing = ESG_NETWORK.get(name, (name, None))
        net = {"id": name, "kind": kind}
        if routing:
            net["routing"] = routing
        if layer.get("description"):
            net["description"] = layer["description"]
        net["nodes"] = []
        for n in layer.get("nodes", []):
            node = {"id": n["id"], "kind": n["kind"]}
            if n.get("block"):
                node["block"] = n["block"]
            if n.get("at") is not None:
                node["at"] = list(n["at"])
            if n.get("level") is not None:
                node["level"] = n["level"]
            net["nodes"].append(node)
        net["links"] = []
        for s in layer.get("segments", []):
            link = {"id": s["id"], "from": s["from"], "to": s["to"]}
            if s.get("points") is not None:
                link["points"] = [list(p) for p in s["points"]]
            if s.get("length") is not None:
                link["length"] = s["length"]
            net["links"].append(link)
        networks.append(net)
    rec = {
        "format": FORMAT,
        "version": VERSION,
        "provenance": {
            "producer": m.get("producer", "kpu-gen"),
            "source": "rtl",
            "configuration": m.get("spec", ""),
            "configuration_digest": m.get("spec_sha256", ""),
            "from": f"{d['format']} {d['version']}",
            "details": {k: v for k, v in m.items() if k not in ("producer", "spec", "spec_sha256", "units")},
        },
        "units": {"length": "mm", "area": "mm2"},
        "blocks": blocks,
        "networks": networks,
        "floorplan": {
            "die": dict(d["die"]),
            # Every number in floorplan.json is an estimate (its docs, "Everything here is an estimate").
            "estimated": True,
            "area": d.get("area", {}),
        },
    }
    return rec


def from_sim(d: dict) -> dict:
    if d.get("units") != "um":
        raise ConvertError(f"kpu-floorplan units '{d.get('units')}': this converter reads um")
    k = 1e-3                                           # um -> mm

    def rect(r):
        return [v * k for v in r]

    blocks = []

    def walk(b, parent):
        out = {"id": b["name"], "kind": SIM_KIND.get(b["kind"], b["kind"])}
        if parent:
            out["parent"] = parent
        if b.get("label"):
            out["label"] = b["label"]
        if b.get("resource"):
            out["resource"] = b["resource"]
        if b.get("clock"):
            out["clock_domain"] = b["clock"]
        if b.get("rect") is not None:
            out["rect"] = rect(b["rect"])
            out["area"] = out["rect"][2] * out["rect"][3]
        role = {"block_mover": "block_mover", "dma_engine": "dma", "compute_tile": "fabric"}.get(out["kind"])
        if role:
            out["role"] = role
        layer = {"l3_tile": "l3", "l3_bank": "l3", "l2_bank": "l2", "l1_vector": "l1", "compute_tile": "fabric"}.get(out["kind"])
        if layer:
            out["layer"] = layer
        blocks.append(out)
        for c in b.get("children", []):
            walk(c, b["name"])

    die = rect(d["die"])
    root = d["device"] + "/die"
    blocks.append({"id": root, "kind": "die", "rect": die, "area": die[2] * die[3]})
    for b in d["blocks"]:
        walk(b, root)
    # The NoC: one network, its nodes the blocks its links join (routers, ports, DMA attachments).
    nodes, links = {}, []
    by_id = {b["id"]: b for b in blocks}
    for i, ln in enumerate(d.get("noc", [])):
        for end in (ln["a"], ln["b"]):
            if end not in nodes:
                blk = by_id.get(end)
                kind = {"noc_router": "router", "noc_port": "port"}.get(blk["kind"] if blk else "", "attach")
                node = {"id": end, "kind": kind, "block": end}
                if blk and "rect" in blk:
                    x, y, w, h = blk["rect"]
                    node["at"] = [x + w / 2, y + h / 2]
                nodes[end] = node
        link = {"id": f"{ln['kind']}_{i}", "from": ln["a"], "to": ln["b"]}
        if ln.get("route"):
            pts = [[p[0] * k, p[1] * k] for p in ln["route"]]
            link["points"] = pts
            link["length"] = sum(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        links.append(link)
    networks = []
    if links:
        networks.append({"id": "noc", "kind": "noc", "routing": "ring",
                         "description": "the NoC's ring, port and attach links (kpu-floorplan)",
                         "nodes": list(nodes.values()), "links": links})
    w, h = die[2], die[3]
    return {
        "format": FORMAT,
        "version": VERSION,
        "provenance": {
            "producer": "kpu-floorplan",
            "source": "model",
            "configuration": d["device"],
            "from": f"{d['format']} {d['version']} ({d.get('source', '')})",
        },
        "units": {"length": "mm", "area": "mm2"},
        "blocks": blocks,
        "networks": networks,
        # The model's pitch is illustrative, not silicon: every rect is an estimate.
        "floorplan": {"die": {"width": w, "height": h, "area": w * h}, "estimated": True},
    }


def convert(d: dict) -> dict:
    fmt, ver = d.get("format"), d.get("version")
    if fmt == "kpu-gen-floorplan" and ver == 1:
        return from_esg(d)
    if fmt == "kpu-floorplan" and ver == 1:
        return from_sim(d)
    raise ConvertError(f"format '{fmt}' version {ver}: this converts kpu-gen-floorplan 1 and kpu-floorplan 1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("floorplan")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    try:
        with open(a.floorplan) as f:
            d = json.load(f)
        rec = convert(d)
    except (OSError, ValueError, KeyError, ConvertError) as e:
        print(f"structure_from_floorplan: {a.floorplan}: {e}", file=sys.stderr)
        return 2
    with open(a.output, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"structure_from_floorplan: wrote {a.output} ({len(rec['blocks'])} blocks, "
          f"{len(rec.get('networks', []))} networks, from {rec['provenance']['from']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
