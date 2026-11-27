#!/usr/bin/env python3
"""
Structure record checker (#392; docs/plans/shared-structure-record.md §4).

Checks a `kpu-structure` record -- the machine's blocks, resources, channels, networks and
floorplan, written by kpu-sim's model or by an RTL generator -- against the invariants every
record must satisfy, so a producer's bug is a failing check rather than a wrong drawing. The RTL
repositories run it in CI on every record they write (ADR 0005).

Usage:
    python3 structure_check.py <record.json> [--json] [--verbose]

Exit codes (the same contract as dflow_check.py, tflow_check.py and mflow_check.py):
    0  every invariant holds
    1  one or more invariants are violated -- parse them and fix the producer
    2  the record could not be read -- there are no violations to parse

Invariants (the spec's §4):
    S1 format      format is kpu-structure, version is one this checker reads, and the provenance
                   names its producer and its source (model or rtl)
    S2 ids         every block, channel, network, node and link id is unique within its kind, and
                   every block kind is in the vocabulary
    S3 tree        every parent names a block, and the tree has no cycle
    S4 references  channel ends, node blocks, resources' block ids, and links' ends (nodes of
                   their own network) all exist
    S5 resources   every resource kind is known, its numeric fields are non-negative numbers, and
                   a memory's bytes agrees with depth x width_bits / 8 when all three are given
    F1 inside      a placed block lies inside its placed parent and inside the die (1 um slack)
    F2 area        silicon + whitespace equals the die's area, and the die's area is width x
                   height (0.1%)
    F3 networks    a placed node lies on the die, and a link's points start at its from node and
                   end at its to node
F1 to F3 run when the record has a floorplan (floorplan.die).
"""
from __future__ import annotations

import argparse
import json
import sys

FORMAT = "kpu-structure"
VERSIONS = {1}
KINDS = {
    "die", "array", "compute_tile", "l3_tile", "l3_bank", "l2_bank", "l1_vector", "register_file",
    "block_mover", "noc_router", "noc_port", "memory_controller", "dma_engine", "dram_phy", "io",
    "control", "host_bridge", "cpu", "cpu_hart", "cpu_sram", "descriptor_ring", "completion_ring",
    "cluster",
}
RESOURCES = {
    "memory": ("bytes", "depth", "width_bits", "read_ports", "write_ports", "banks", "latency_cycles"),
    "slot_pool": ("slots",),
    "queue": ("depth",),
    "stream_buffer": ("count", "depth", "credits"),
    "result_buffer": ("count", "depth", "credits"),
    "pe_grid": ("rows", "cols", "operand_bits", "accumulator_bits", "macs"),
    "bus": ("width_bits",),
}
RESOURCE_REFS = {"slot_pool": ("producer", "consumer"), "queue": ("feeds",)}
SLACK = 1e-3                                     # mm: 1 um


class Checker:
    def __init__(self, rec: dict):
        self.r = rec
        self.violations: list[dict] = []

    def fail(self, inv: str, msg: str):
        self.violations.append({"invariant": inv, "message": msg})

    def run(self) -> list[dict]:
        r = self.r
        # S1
        if r.get("format") != FORMAT:
            self.fail("S1", f"format is '{r.get('format')}', not '{FORMAT}'")
            return self.violations
        if r.get("version") not in VERSIONS:
            self.fail("S1", f"version {r.get('version')}: this checker reads {sorted(VERSIONS)}")
            return self.violations
        p = r.get("provenance", {})
        if not p.get("producer"):
            self.fail("S1", "provenance.producer is missing")
        if p.get("source") not in ("model", "rtl"):
            self.fail("S1", f"provenance.source is '{p.get('source')}', not model or rtl")
        blocks = r.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            self.fail("S1", "blocks is missing or empty")
            return self.violations
        # S2
        self.blocks = {}
        for b in blocks:
            bid = b.get("id")
            if not bid:
                self.fail("S2", f"a block has no id: {json.dumps(b)[:80]}")
                continue
            if bid in self.blocks:
                self.fail("S2", f"block id '{bid}' is not unique")
            self.blocks[bid] = b
            if b.get("kind") not in KINDS:
                self.fail("S2", f"block '{bid}': kind '{b.get('kind')}' is not in the vocabulary")
        self.unique("channels", r.get("channels", []))
        nets = r.get("networks", [])
        self.unique("networks", nets)
        for n in nets:
            self.unique(f"network '{n.get('id')}' nodes", n.get("nodes", []))
            self.unique(f"network '{n.get('id')}' links", n.get("links", []))
        # S3
        for bid, b in self.blocks.items():
            par = b.get("parent")
            if par is not None and par not in self.blocks:
                self.fail("S3", f"block '{bid}': parent '{par}' is not a block")
        for bid in self.blocks:
            seen, at = set(), bid
            while at is not None and at in self.blocks:
                if at in seen:
                    self.fail("S3", f"block '{bid}': its parents form a cycle through '{at}'")
                    break
                seen.add(at)
                at = self.blocks[at].get("parent")
        # S4
        for c in r.get("channels", []):
            if c.get("kind") not in ("data", "command", "sync"):
                self.fail("S4", f"channel '{c.get('id')}': kind '{c.get('kind')}' is not data, command or sync")
            for end in ("from", "to"):
                if c.get(end) not in self.blocks:
                    self.fail("S4", f"channel '{c.get('id')}': {end} '{c.get(end)}' is not a block")
        for n in nets:
            node_ids = {x.get("id") for x in n.get("nodes", [])}
            for x in n.get("nodes", []):
                if x.get("block") is not None and x["block"] not in self.blocks:
                    self.fail("S4", f"network '{n.get('id')}' node '{x.get('id')}': block '{x['block']}' is not a block")
            for ln in n.get("links", []):
                for end in ("from", "to"):
                    if ln.get(end) not in node_ids:
                        self.fail("S4", f"network '{n.get('id')}' link '{ln.get('id')}': {end} '{ln.get(end)}' "
                                        "is not a node of its network")
        # S5
        for bid, b in self.blocks.items():
            for res in b.get("resources", []):
                k = res.get("kind")
                if k not in RESOURCES:
                    self.fail("S5", f"block '{bid}': resource kind '{k}' is not known")
                    continue
                for f in RESOURCES[k]:
                    v = res.get(f)
                    if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0):
                        self.fail("S5", f"block '{bid}' {k}: {f} = {v!r} is not a non-negative number")
                for f in RESOURCE_REFS.get(k, ()):
                    if res.get(f) is not None and res[f] not in self.blocks:
                        self.fail("S4", f"block '{bid}' {k}: {f} '{res[f]}' is not a block")
                if k == "memory" and all(isinstance(res.get(f), (int, float)) for f in ("bytes", "depth", "width_bits")):
                    if abs(res["bytes"] - res["depth"] * res["width_bits"] / 8) > 0.5:
                        self.fail("S5", f"block '{bid}' memory: bytes {res['bytes']} is not depth {res['depth']} "
                                        f"x width {res['width_bits']} bits / 8")
        fp = r.get("floorplan")
        if fp and fp.get("die"):
            self.floorplan(fp, nets)
        return self.violations

    def unique(self, what: str, items: list):
        seen = set()
        for x in items:
            i = x.get("id")
            if i is None:
                self.fail("S2", f"{what}: an entry has no id")
            elif i in seen:
                self.fail("S2", f"{what}: id '{i}' is not unique")
            seen.add(i)

    def floorplan(self, fp: dict, nets: list):
        die = fp["die"]
        w, h, a = die.get("width"), die.get("height"), die.get("area")
        if not all(isinstance(v, (int, float)) for v in (w, h)):
            self.fail("F2", "floorplan.die needs a width and a height")
            return
        if isinstance(a, (int, float)) and abs(a - w * h) > 1e-3 * max(a, 1e-12):
            self.fail("F2", f"floorplan.die area {a} is not width {w} x height {h}")
        ar = fp.get("area") or {}
        if all(isinstance(ar.get(k), (int, float)) for k in ("silicon", "whitespace", "die")):
            if abs(ar["silicon"] + ar["whitespace"] - ar["die"]) > 1e-3 * max(ar["die"], 1e-12):
                self.fail("F2", f"area: silicon {ar['silicon']} + whitespace {ar['whitespace']} is not the die's "
                                f"{ar['die']}")

        def inside(inner, outer):
            x, y, iw, ih = inner
            ox, oy, ow, oh = outer
            return x >= ox - SLACK and y >= oy - SLACK and x + iw <= ox + ow + SLACK and y + ih <= oy + oh + SLACK

        die_rect = [0.0, 0.0, w, h]
        for bid, b in self.blocks.items():
            rc = b.get("rect")
            if rc is None:
                continue
            if len(rc) != 4 or rc[2] < 0 or rc[3] < 0:
                self.fail("F1", f"block '{bid}': rect {rc} is not [x, y, width, height]")
                continue
            if not inside(rc, die_rect):
                self.fail("F1", f"block '{bid}': rect {rc} is outside the die")
            par = self.blocks.get(b.get("parent"))
            if par and par.get("rect") is not None and not inside(rc, par["rect"]):
                self.fail("F1", f"block '{bid}': rect {rc} is outside its parent '{b['parent']}' {par['rect']}")
        for n in nets:
            at = {x["id"]: x.get("at") for x in n.get("nodes", []) if "id" in x}
            for nid, p in at.items():
                if p is not None and not inside([p[0], p[1], 0, 0], die_rect):
                    self.fail("F3", f"network '{n.get('id')}' node '{nid}': at {p} is off the die")
            for ln in n.get("links", []):
                pts = ln.get("points")
                if not pts:
                    continue
                for end, p in (("from", pts[0]), ("to", pts[-1])):
                    q = at.get(ln.get(end))
                    if q is not None and (abs(p[0] - q[0]) > SLACK or abs(p[1] - q[1]) > SLACK):
                        self.fail("F3", f"network '{n.get('id')}' link '{ln.get('id')}': its points' {end} end {p} "
                                        f"is not at its {end} node {q}")


def check(rec: dict) -> list[dict]:
    return Checker(rec).run()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check a kpu-structure record (#392).")
    ap.add_argument("record")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    try:
        with open(a.record) as f:
            rec = json.load(f)
        if not isinstance(rec, dict):
            raise ValueError("the record is not a JSON object")
    except (OSError, ValueError) as e:
        print(f"structure_check: {a.record}: {e}", file=sys.stderr)
        return 2
    v = check(rec)
    if a.json:
        print(json.dumps({"record": a.record, "violations": v, "ok": not v}, indent=1))
    else:
        nb = len(rec.get("blocks", []))
        nn = len(rec.get("networks", []))
        fp = "with a floorplan" if (rec.get("floorplan") or {}).get("die") else "no floorplan"
        if v:
            print(f"structure_check: {a.record}: {len(v)} violation(s)")
            for x in v:
                print(f"  {x['invariant']}  {x['message']}")
        else:
            print(f"structure_check: {a.record}: ok ({nb} blocks, {nn} networks, {fp}; "
                  f"{rec.get('provenance', {}).get('source')} from {rec.get('provenance', {}).get('producer')})")
        if a.verbose and not v:
            for b in rec["blocks"][:20]:
                print(f"    {b['kind']:<16} {b['id']}")
    return 1 if v else 0


if __name__ == "__main__":
    sys.exit(main())
