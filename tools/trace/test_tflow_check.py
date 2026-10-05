#!/usr/bin/env python3
"""
Self-test for tflow_check.py (#286 step 3): every invariant must FAIL on a bundle that
breaks it, and the clean bundle must pass. A checker that passes everything is not a checker.

Usage: python3 test_tflow_check.py <clean-bundle.tflow>

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE / "tflow_check.py"
BUNDLE = None


def run(path):
    p = subprocess.run([sys.executable, str(CHECK), str(path), "--json"],
                       capture_output=True, text=True)
    report = json.loads(p.stdout) if p.returncode in (0, 1) else None
    return p.returncode, report


class Bundle:
    """A writable copy of the clean bundle, with column-level edits."""

    def __init__(self, src):
        self.dir = Path(tempfile.mkdtemp(prefix="tflow-selftest-")) / "b.tflow"
        shutil.copytree(src, self.dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())

    def column(self, table, name):
        t = self.manifest["tables"][table]
        for c in t["columns"]:
            if c["name"] == name:
                return self.dir / t["file"], c["offset"], c["dtype"]
        raise KeyError(name)

    def get(self, table, name, row):
        path, off, dtype = self.column(table, name)
        fmt, size = {"u8": ("<B", 1), "u32": ("<I", 4), "f64": ("<d", 8)}[dtype]
        return struct.unpack_from(fmt, path.read_bytes(), off + row * size)[0]

    def set(self, table, name, row, value):
        path, off, dtype = self.column(table, name)
        fmt, size = {"u8": ("<B", 1), "u32": ("<I", 4), "f64": ("<d", 8)}[dtype]
        b = bytearray(path.read_bytes())
        struct.pack_into(fmt, b, off + row * size, value)
        path.write_bytes(bytes(b))

    def save_manifest(self):
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest))

    def close(self):
        shutil.rmtree(self.dir.parent, ignore_errors=True)


class TflowCheckSelfTest(unittest.TestCase):
    def setUp(self):
        self.b = Bundle(BUNDLE)

    def tearDown(self):
        self.b.close()

    def assertFails(self, invariant):
        code, report = run(self.b.dir)
        self.assertEqual(code, 1, f"expected {invariant} to fail, checker exited {code}")
        found = {v["invariant"] for v in report["violations"]}
        self.assertIn(invariant, found, f"violations were {sorted(found)}")

    def test_clean_bundle_passes(self):
        code, report = run(self.b.dir)
        self.assertEqual(code, 0, report and report["violations"])
        self.assertIn("TF9", report["checked"])

    def test_tf1_capacity(self):
        for s in self.b.manifest["stations"]:
            if s["kind"] == "l3":
                s["capacity"] = 1
        self.b.save_manifest()
        self.assertFails("TF1")

    def test_tf2_outside_the_run(self):
        self.b.set("compute", "t1", 0, float(self.b.manifest["makespan"] + 10))
        self.assertFails("TF2")

    def test_tf3_lane_shared(self):
        rows = self.b.manifest["tables"]["transit"]["rows"]
        # Put the second transit on the first one's mover and lane, at its time.
        for col in ("mover", "lane", "t0", "t1"):
            self.b.set("transit", col, 1, self.b.get("transit", col, 0))
        self.assertGreater(rows, 1)
        self.assertFails("TF3")

    def test_tf4_compute_tile_shared(self):
        for col in ("station", "t0", "t1"):
            self.b.set("compute", col, 1, self.b.get("compute", col, 0))
        self.assertFails("TF4")

    def test_tf5_tile_residencies_overlap(self):
        self.assertGreater(self.b.manifest["tables"]["residency"]["rows"], 1)
        for col in ("tile", "t0", "t1"):
            self.b.set("residency", col, 1, self.b.get("residency", col, 0))
        self.assertFails("TF5")

    def test_tf7_operation_hops_overlap(self):
        self.assertGreater(self.b.manifest["tables"]["transit"]["rows"], 1)
        for col in ("op", "t0", "t1"):
            self.b.set("transit", col, 1, self.b.get("transit", col, 0))
        self.assertFails("TF7")

    def test_tf7_hops_disjoint_but_out_of_order(self):
        # Swap the stations of two consecutive hops of one tile: the times stay disjoint, but the
        # chain no longer connects (and each hop now moves between the wrong kinds).
        tr = self.b.manifest["tables"]["transit"]["rows"]
        chains = {}
        for i in range(tr):
            key = (self.b.get("transit", "op", i), self.b.get("transit", "tile", i))
            chains.setdefault(key, []).append((self.b.get("transit", "t0", i), i))
        hops = next(sorted(h) for h in chains.values() if len(h) >= 3)
        i, j = hops[1][1], hops[2][1]
        for col in ("src", "dst", "hop"):
            a, b = self.b.get("transit", col, i), self.b.get("transit", col, j)
            self.b.set("transit", col, i, b)
            self.b.set("transit", col, j, a)
        self.assertFails("TF7")

    def test_tf7_hop_between_wrong_kinds(self):
        # A BlockMover L3 -> L2 hop relabelled as a streamer L2 -> L1 hop, stations untouched:
        # the chain still connects and the times are unchanged, only the kinds are wrong.
        tr = self.b.manifest["tables"]["transit"]["rows"]
        i = next(i for i in range(tr) if self.b.get("transit", "hop", i) == 1)
        self.b.set("transit", "hop", i, 2)
        self.assertFails("TF7")

    def test_tf9_arrival_must_land_in_l3(self):
        # A tile held once and filled only by DMA, whose DMA now lands in L2: it never reached
        # its L3 slot, so the slot is unfilled.
        res = self.b.manifest["tables"]["residency"]["rows"]
        tiles = [self.b.get("residency", "tile", i) for i in range(res)]
        t = self.b.manifest["tables"]["op_tiles"]
        nnz = self.b.get("op_tiles", "offset", t["rows"])
        written = {self.b.get("op_tiles", "tile", j) for j in range(nnz)
                   if self.b.get("op_tiles", "written", j)}
        tr = self.b.manifest["tables"]["transit"]["rows"]
        l2 = next(k for k, x in enumerate(self.b.manifest["stations"]) if x["kind"] == "l2")
        for i in range(res):
            tile = tiles[i]
            if tiles.count(tile) != 1 or tile in written or self.b.get("residency", "flags", i) & 1:
                continue
            dma = [j for j in range(tr) if self.b.get("transit", "tile", j) == tile
                   and self.b.get("transit", "hop", j) in (0, 7)]
            if dma:
                for j in dma:
                    self.b.set("transit", "dst", j, l2)
                self.assertFails("TF9")
                return
        self.fail("no tile filled only by DMA in this bundle")

    def test_tf8_base_bins_shifted(self):
        # Occupancy moved between two sibling base bins: every merge and the total still hold.
        lod = json.loads((self.b.dir / "lod.json").read_text())
        base = lod["levels"][0]
        self.assertGreaterEqual(base["bins"], 2)
        path = self.b.dir / lod["file"]
        b = bytearray(path.read_bytes())
        row = next(i for i, x in enumerate(lod["rows"]) if x["kind"] == "l3")
        at0, at1 = base["occ"] + 8 * (row * base["bins"]), base["occ"] + 8 * (row * base["bins"] + 1)
        v0, v1 = struct.unpack_from("<d", b, at0)[0], struct.unpack_from("<d", b, at1)[0]
        self.assertGreater(v0 + v1, 0)
        struct.pack_into("<d", b, at0, v0 + v1)
        struct.pack_into("<d", b, at1, 0.0)
        path.write_bytes(bytes(b))
        self.assertFails("TF8")

    def test_tf8_top_wider_than_one_bin(self):
        lod = json.loads((self.b.dir / "lod.json").read_text())
        lod["levels"].pop()
        (self.b.dir / "lod.json").write_text(json.dumps(lod))
        self.assertFails("TF8")

    def test_tf6_released_while_used(self):
        # End every residency at its start: every user now runs without a slot.
        rows = self.b.manifest["tables"]["residency"]["rows"]
        for i in range(rows):
            self.b.set("residency", "t1", i, self.b.get("residency", "t0", i))
        self.assertFails("TF6")

    def test_tf9_unfilled_slot(self):
        # Clear every "written" flag and drop every inbound DMA: no residency is filled.
        t = self.b.manifest["tables"]["op_tiles"]
        nnz = self.b.get("op_tiles", "offset", t["rows"])
        for j in range(nnz):
            self.b.set("op_tiles", "written", j, 0)
        for i in range(self.b.manifest["tables"]["transit"]["rows"]):
            if self.b.get("transit", "hop", i) == 0:
                self.b.set("transit", "hop", i, 1)
        self.assertFails("TF9")

    def test_tf9_arrival_at_the_residency_end(self):
        # A tile held once, filled only by DMA, with every arrival moved to the cycle its slot
        # is released: [t0, t1) is half-open, so the slot was never filled.
        res = self.b.manifest["tables"]["residency"]["rows"]
        tiles = [self.b.get("residency", "tile", i) for i in range(res)]
        pick = next(i for i in range(res)
                    if tiles.count(tiles[i]) == 1 and not (self.b.get("residency", "flags", i) & 1)
                    and self.b.get("residency", "t1", i) > self.b.get("residency", "t0", i))
        tile, end = tiles[pick], self.b.get("residency", "t1", pick)
        t = self.b.manifest["tables"]["op_tiles"]
        for j in range(self.b.get("op_tiles", "offset", t["rows"])):
            if self.b.get("op_tiles", "tile", j) == tile:
                self.b.set("op_tiles", "written", j, 0)
        moved = 0
        for i in range(self.b.manifest["tables"]["transit"]["rows"]):
            if self.b.get("transit", "tile", i) == tile and self.b.get("transit", "hop", i) in (0, 7):
                self.b.set("transit", "t1", i, max(self.b.get("transit", "t1", i), end))
                self.b.set("transit", "t0", i, end)
                moved += 1
        self.assertGreater(moved, 0)
        self.assertFails("TF9")

    def test_tf8_pyramid_tampered(self):
        lod = json.loads((self.b.dir / "lod.json").read_text())
        top = lod["levels"][-1]
        path = self.b.dir / lod["file"]
        b = bytearray(path.read_bytes())
        v = struct.unpack_from("<d", b, top["occ"])[0]
        struct.pack_into("<d", b, top["occ"], v + 1.0)
        path.write_bytes(bytes(b))
        self.assertFails("TF8")

    def _bump_every_level(self, metric, all_bins):
        # +1 on row 0 at every level, so each parent is still the merge of its children and
        # only the comparison with the raw events can notice.
        lod = json.loads((self.b.dir / "lod.json").read_text())
        path = self.b.dir / lod["file"]
        b = bytearray(path.read_bytes())
        for lvl in lod["levels"]:
            for bin_ in range(lvl["bins"] if all_bins else 1):
                at = lvl[metric] + 4 * bin_
                struct.pack_into("<I", b, at, struct.unpack_from("<I", b, at)[0] + 1)
        path.write_bytes(bytes(b))

    def test_tf8_peak_consistent_but_wrong(self):
        self._bump_every_level("peak", all_bins=True)
        self.assertFails("TF8")

    def test_tf8_starts_consistent_but_wrong(self):
        self._bump_every_level("starts", all_bins=False)
        self.assertFails("TF8")

    def test_tf8_row_relabelled(self):
        # The L3 row relabelled as dram AND its metrics zeroed at every level, so the pyramid
        # is self-consistent and agrees with "a dram row has no events". Only checking the rows
        # against the record's stations can see that the L3's occupancy has vanished.
        path = self.b.dir / "lod.json"
        lod = json.loads(path.read_text())
        row = next(i for i, x in enumerate(lod["rows"]) if x["kind"] == "l3")
        lod["rows"][row]["kind"] = "dram"
        path.write_text(json.dumps(lod))
        bin_path = self.b.dir / lod["file"]
        b = bytearray(bin_path.read_bytes())
        for lvl in lod["levels"]:
            for k in range(lvl["bins"]):
                at = row * lvl["bins"] + k
                struct.pack_into("<d", b, lvl["occ"] + 8 * at, 0.0)
                struct.pack_into("<I", b, lvl["peak"] + 4 * at, 0)
                struct.pack_into("<I", b, lvl["starts"] + 4 * at, 0)
        bin_path.write_bytes(bytes(b))
        self.assertFails("TF8")

    def test_tf8_row_dropped(self):
        path = self.b.dir / "lod.json"
        lod = json.loads(path.read_text())
        lod["rows"].pop()
        path.write_text(json.dumps(lod))
        self.assertFails("TF8")

    def test_malformed_pyramid_is_exit_2(self):
        for edit in (lambda m: m.pop("rows"), lambda m: m["levels"][0].update(bins=0),
                     lambda m: m["levels"][0].update(bins="many"),
                     lambda m: m["levels"][0].update(bins=True),
                     lambda m: m["levels"][0].update(occ=-8),
                     lambda m: m["levels"][0].update(peak=1.5),
                     lambda m: m["rows"][0].pop("name"),
                     lambda m: m["rows"][0].pop("kind"),
                     lambda m: m["rows"][0].update(name=7),
                     lambda m: m.update(rows={}),
                     lambda m: m.update(levels=[])):
            lod = json.loads((self.b.dir / "lod.json").read_text())
            edit(lod)
            (self.b.dir / "lod.json").write_text(json.dumps(lod))
            code, _ = run(self.b.dir)
            self.assertEqual(code, 2)
            shutil.copy(Path(BUNDLE) / "lod.json", self.b.dir / "lod.json")

    def test_malformed_manifest_rows_are_exit_2(self):
        for edit in (lambda m: m["stations"][0].pop("kind"),
                     lambda m: m["movers"][0].pop("name")):
            self.b.manifest = json.loads((Path(BUNDLE) / "manifest.json").read_text())
            edit(self.b.manifest)
            self.b.save_manifest()
            code, _ = run(self.b.dir)
            self.assertEqual(code, 2)

    def test_wrong_dtype_or_version_is_exit_2(self):
        for edit in (lambda m: next(c for c in m["tables"]["transit"]["columns"]
                                    if c["name"] == "src").update(dtype="f64"),
                     lambda m: m.update(version=1),
                     lambda m: m.update(version=2),
                     lambda m: m.update(version=4)):
            self.b.manifest = json.loads((Path(BUNDLE) / "manifest.json").read_text())
            edit(self.b.manifest)
            self.b.save_manifest()
            code, _ = run(self.b.dir)
            self.assertEqual(code, 2)

    def test_unreadable_is_exit_2(self):
        (self.b.dir / "manifest.json").unlink()
        code, _ = run(self.b.dir)
        self.assertEqual(code, 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = sys.argv.pop(1)
    unittest.main()
