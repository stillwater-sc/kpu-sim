#!/usr/bin/env python3
"""
Consistency test for bind.py's derived binding (#286).

Usage: python3 test_bind.py <bundle.tflow> <floorplan.json>

Binds the bundle and checks the binding against the record and the floorplan: every tile is
homed in an L3 tile, within its per-tile capacity or counted; every DMA transfer has an
engine on the controller that owns its DRAM address; every DMA route starts at the engine's
attached port and ends at the tile's home hub; every BlockMover named exists and abuts the
compute tile it feeds.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "trace"))
import bind          # noqa: E402
import tflow_check   # noqa: E402

BUNDLE = FLOORPLAN = None


def over_count(ivs):
    """Transfers that start while the bus already holds a block, counted directly: another
    transfer started earlier and has not ended (an end at the same cycle frees the bus first),
    or started at the same cycle and is served first. Zero-length transfers hold nothing."""
    live = sorted((a, b) for a, b in ivs if b > a)
    return sum(1 for j, (s, _) in enumerate(live)
               if any(a < s < b for a, b in live) or any(a == s for a, _ in live[:j]))


class BindTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = tflow_check.load(BUNDLE)
        cls.fp = json.loads(Path(FLOORPLAN).read_text(encoding="utf-8"))
        cls.b = bind.bind(cls.rec, cls.fp)
        cls.names = {x["name"] for x in bind.flat(cls.fp["blocks"])}

    def test_every_tile_is_homed_within_capacity(self):
        b, res = self.b, self.rec["residency"]
        homes = b["l3"]["residency_home"]
        self.assertEqual(len(homes), self.rec["residency_rows"])
        self.assertTrue(all(0 <= h < len(b["l3"]["tiles"]) for h in homes))
        cap = b["l3"]["per_tile_capacity"]
        over = 0
        for t in range(len(b["l3"]["tiles"])):
            ev = sorted([(res["t0"][i], 1) for i, h in enumerate(homes) if h == t] +
                        [(res["t1"][i], -1) for i, h in enumerate(homes) if h == t])
            cur = peak = 0
            for _, d in ev:
                cur += d
                peak = max(peak, cur)
            over += peak > cap
        self.assertEqual(over > 0, b["l3"]["overflow"] > 0)

    def test_dma_engines_follow_the_address_interleave(self):
        b, tr, m = self.b, self.rec["transit"], self.rec["manifest"]
        tensors = {t["name"]: t for t in b["dram"]["tensors"]}
        mcs = sorted({bind.index_of(e, "mc") for e in b["engines"]})
        for i in range(self.rec["transit_rows"]):
            if tr["hop"][i] not in (bind.HOP_DMA_IN, bind.HOP_EJECT, bind.HOP_DMA_WRITE):
                self.assertEqual(b["transit_engine"][i], -1)
                continue
            name, ti, tj = m["tiles"][tr["tile"][i]]
            t = tensors[name]
            addr = t["base"] + (ti * t["tile_cols"] + tj) * t["tile_bytes"]
            want_mc = mcs[(addr // t["tile_bytes"]) % len(mcs)]
            eng = b["engines"][b["transit_engine"][i]]
            self.assertEqual(bind.index_of(eng, "mc"), want_mc, eng)

    def test_dma_routes_run_port_to_home_hub(self):
        b, tr, res = self.b, self.rec["transit"], self.rec["residency"]
        attach = {l["a"]: l["b"] for l in self.fp["noc"] if l["kind"] == "attach"}
        homes = {}
        for i, h in enumerate(b["l3"]["residency_home"]):
            homes.setdefault(res["tile"][i], []).append((res["t0"][i], res["t1"][i], h))
        for i in range(self.rec["transit_rows"]):
            if tr["hop"][i] != 0:
                continue
            path = [b["nodes"][n] for n in b["transit_path"][i]]
            self.assertTrue(path, f"transit {i} has no route")
            self.assertEqual(path[0], attach[b["engines"][b["transit_engine"][i]]])
            h = next(h for t0, t1, h in homes[tr["tile"][i]] if t0 <= tr["t1"][i] <= t1)
            self.assertEqual(path[-1], b["l3"]["tiles"][h] + "/noc")
            for n in path:
                self.assertIn(n, self.names)

    def test_an_ejection_fills_the_buffer_its_engine_writes_out(self):
        # Every hop is a push: writeback is a BlockMover ejecting into a DMA engine buffer across
        # the NoC (home hub -> that engine's port), then the SAME engine writing it to DRAM, off
        # the NoC.
        b, tr = self.b, self.rec["transit"]
        attach = {l["a"]: l["b"] for l in self.fp["noc"] if l["kind"] == "attach"}
        writes = {}
        for i in range(self.rec["transit_rows"]):
            if tr["hop"][i] == bind.HOP_DMA_WRITE:
                writes.setdefault((tr["op"][i], tr["tile"][i]), []).append(i)
                self.assertEqual(b["transit_path"][i], [], f"DMA write {i} crosses the NoC")
        ejections = 0
        for i in range(self.rec["transit_rows"]):
            if tr["hop"][i] != bind.HOP_EJECT:
                continue
            ejections += 1
            after = [w for w in writes.get((tr["op"][i], tr["tile"][i]), []) if tr["t0"][w] >= tr["t1"][i]]
            self.assertTrue(after, f"ejection {i} is never written to DRAM")
            w = min(after, key=lambda w: tr["t0"][w])
            self.assertEqual(b["transit_engine"][w], b["transit_engine"][i])
            path = [b["nodes"][n] for n in b["transit_path"][i]]
            self.assertTrue(path, f"ejection {i} has no route")
            self.assertTrue(path[0].endswith("/noc"), path)               # from the home hub
            self.assertEqual(path[-1], attach[b["engines"][b["transit_engine"][i]]])   # to the port
        self.assertEqual(ejections, sum(len(v) for v in writes.values()))

    def test_blockmovers_exist_on_the_floorplan(self):
        for name in self.b["bms"]:
            self.assertIn(name, self.names)
            self.assertRegex(name, r"/l3\[\d+\]/bm\[[0-3]\]$")

    def test_the_binding_names_the_run_it_indexes(self):
        # Its arrays are per row of this run; a reader rejects a binding from another run.
        self.assertEqual(self.b["bundle"], bind.bundle_identity(self.rec["manifest"]))
        self.assertEqual(len(self.b["transit_engine"]), self.b["bundle"]["transit"])
        self.assertEqual(len(self.b["l3"]["residency_home"]), self.b["bundle"]["residency"])

    def test_a_port_is_two_one_block_buses(self):
        # Every bus is reported against a capacity of ONE block, recomputed here from the raw
        # transits on their derived paths, and a bus over one is counted, never normalized.
        ports = self.b["ports"]
        self.assertEqual(ports["capacity_per_bus"], 1)
        self.assertFalse(ports["modelled"])
        tr, nodes = self.rec["transit"], self.b["nodes"]
        want = {}
        for i, path in enumerate(self.b["transit_path"]):
            hop = tr["hop"][i]
            if hop not in (bind.HOP_DMA_IN, bind.HOP_EJECT) or not path:
                continue
            end = nodes[path[0] if hop == bind.HOP_DMA_IN else path[-1]]
            if "/port[" in end:
                bus = "inject" if hop == bind.HOP_DMA_IN else "eject"
                want.setdefault(end, {"inject": [], "eject": []})[bus].append((tr["t0"][i], tr["t1"][i]))
        self.assertEqual(sorted(ports["buses"]), sorted(want))
        for name, buses in want.items():
            for bus, ivs in buses.items():
                got = ports["buses"][name][bus]
                self.assertEqual(got["transfers"], len(ivs))
                peak = max((sum(1 for a, b in ivs if a <= t < b) for t, _ in ivs), default=0)
                self.assertEqual(got["peak"], peak, f"{name} {bus}")
                self.assertEqual(got["oversubscribed"], over_count(ivs), f"{name} {bus}")
                self.assertEqual(got["oversubscribed"] > 0, peak > 1, f"{name} {bus}")

    def test_a_floorplan_of_another_device_is_refused(self):
        fp = dict(self.fp, device="elsewhere")
        with self.assertRaises(bind.BindError):
            bind.bind(self.rec, fp)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    FLOORPLAN = sys.argv.pop(2)
    BUNDLE = sys.argv.pop(1)
    unittest.main()
