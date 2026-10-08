#!/usr/bin/env python3
"""
Self-test for mflow_check.py (docs/plans/memory-side-debugger.md step 4): every invariant must
FAIL on a bundle that breaks it, and the clean bundle must pass. A checker that passes everything
is not a checker.

Usage: python3 test_mflow_check.py <clean-bundle.mflow>

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
CHECK = HERE / "mflow_check.py"
BUNDLE = None
FMT = {"u8": ("<B", 1), "u32": ("<I", 4), "u64": ("<Q", 8), "f64": ("<d", 8)}
ACT, RD, WR, PRE, REF = range(5)


def run(path):
    p = subprocess.run([sys.executable, str(CHECK), str(path), "--json"],
                       capture_output=True, text=True)
    report = json.loads(p.stdout) if p.returncode in (0, 1) else None
    return p.returncode, report


class Bundle:
    """A writable copy of the clean bundle, with column-level edits."""

    def __init__(self, src):
        self.dir = Path(tempfile.mkdtemp(prefix="mflow-selftest-")) / "b.mflow"
        shutil.copytree(src, self.dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())

    def rows(self, table):
        return self.manifest["tables"][table]["rows"]

    def column(self, table, name):
        t = self.manifest["tables"][table]
        for c in t["columns"]:
            if c["name"] == name:
                return self.dir / t["file"], c["offset"], c["dtype"]
        raise KeyError(name)

    def get(self, table, name, row):
        path, off, dtype = self.column(table, name)
        fmt, size = FMT[dtype]
        return struct.unpack_from(fmt, path.read_bytes(), off + row * size)[0]

    def set(self, table, name, row, value):
        path, off, dtype = self.column(table, name)
        fmt, size = FMT[dtype]
        b = bytearray(path.read_bytes())
        struct.pack_into(fmt, b, off + row * size, value)
        path.write_bytes(bytes(b))

    def find(self, table, pred):
        return next(i for i in range(self.rows(table)) if pred(i))

    def save_manifest(self):
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest))

    def close(self):
        shutil.rmtree(self.dir.parent, ignore_errors=True)


class MflowCheckSelfTest(unittest.TestCase):
    def setUp(self):
        self.b = Bundle(BUNDLE)

    def tearDown(self):
        self.b.close()

    def assertFails(self, invariant, needle=None):
        code, report = run(self.b.dir)
        self.assertEqual(code, 1, f"expected {invariant} to fail, checker exited {code}")
        hits = [v["message"] for v in report["violations"] if v["invariant"] == invariant]
        self.assertTrue(hits, f"violations were {sorted({v['invariant'] for v in report['violations']})}")
        if needle:
            self.assertTrue(any(needle in h for h in hits), f"no {invariant} message names {needle!r}: {hits[:3]}")

    def assertUnreadable(self):
        code, _ = run(self.b.dir)
        self.assertEqual(code, 2)

    # The command pairs the mutations below edit: a bank's ACT and the next command on it.
    def bank_pair(self, first, second):
        cmds = {}
        for i in range(self.b.rows("commands")):
            key = tuple(self.b.get("commands", f, i) for f in ("mc", "channel", "bank_group", "bank"))
            cmds.setdefault(key, []).append((self.b.get("commands", "k_issue", i), i))
        for v in cmds.values():
            v.sort()
            for (_, a), (_, b) in zip(v, v[1:]):
                if self.b.get("commands", "kind", a) == first and self.b.get("commands", "kind", b) in second:
                    return a, b
        self.fail(f"the clean bundle has no {first} followed by {second} on one bank")

    # -- the clean run ------------------------------------------------------
    def test_clean_bundle_passes(self):
        code, report = run(self.b.dir)
        self.assertEqual(code, 0, report and report["violations"][:5])
        self.assertEqual(report["checked"], ["TF10", "TF11", "M1", "M2", "M3", "M4", "M5"])

    def test_the_run_exercises_what_is_checked(self):
        # A check over nothing passes vacuously: the clean run must hold conflicts (PRE then
        # ACT), refreshes, both directions, a full window and a full store buffer.
        kinds = {self.b.get("commands", "kind", i) for i in range(self.b.rows("commands"))}
        self.assertEqual(kinds, {ACT, RD, WR, PRE, REF})
        outcomes = {self.b.get("bursts", "outcome", i) for i in range(self.b.rows("bursts"))}
        self.assertIn(3, outcomes)
        held = max(self.b.get("buffers", "held", i) for i in range(self.b.rows("buffers")))
        cap = max(s["capacity"] for s in self.b.manifest["stations"] if s["kind"] == "dmabuf")
        self.assertEqual(held, cap)

    # -- TF10 ---------------------------------------------------------------
    def test_tf10_cas_to_a_row_not_open(self):
        i = self.b.find("commands", lambda i: self.b.get("commands", "kind", i) == RD)
        self.b.set("commands", "row", i, self.b.get("commands", "row", i) + 1)
        self.assertFails("TF10", "is open")

    def test_tf10_activate_over_an_open_row(self):
        # An ACT's PRE turned into a RD of the same row: the next ACT finds the row still open.
        pre, _ = self.bank_pair(PRE, {ACT})
        self.b.set("commands", "kind", pre, RD)
        self.assertFails("TF10", "while row")

    def test_tf10_cas_serves_another_bank(self):
        i = self.b.find("commands", lambda i: self.b.get("commands", "kind", i) == WR)
        burst = self.b.get("commands", "burst", i)
        self.b.set("bursts", "bank", burst, (self.b.get("bursts", "bank", burst) + 1) % 4)
        self.assertFails("TF10", "serves burst")

    # -- TF11 ---------------------------------------------------------------
    def test_tf11_two_bursts_on_one_bus(self):
        cas = [i for i in range(self.b.rows("commands")) if self.b.get("commands", "kind", i) in (RD, WR)]
        a = cas[0]
        b = next(i for i in cas[1:] if self.b.get("commands", "mc", i) == self.b.get("commands", "mc", a)
                 and self.b.get("commands", "channel", i) == self.b.get("commands", "channel", a))
        for col in ("k_data0", "k_data1"):
            self.b.set("commands", col, b, self.b.get("commands", col, a))
        self.assertFails("TF11", "drives the data bus")

    def test_tf11_is_checked_in_ticks_not_cycles(self):
        # Executor-cycle data windows can overlap by rounding when the ticks do not: the clean run
        # passes TF11 (test_clean_bundle_passes), so the checker must not be reading cycles. Here
        # the cycles of two bursts are made to overlap, ticks untouched: still clean.
        cas = [i for i in range(self.b.rows("commands")) if self.b.get("commands", "kind", i) in (RD, WR)]
        a, b = cas[0], cas[1]
        for col in ("t_data0", "t_data1"):
            self.b.set("commands", col, b, self.b.get("commands", col, a))
        code, report = run(self.b.dir)
        self.assertNotIn("TF11", {v["invariant"] for v in (report or {"violations": []})["violations"]})

    # -- M1 / M2 ------------------------------------------------------------
    def test_m1_window_exceeded(self):
        for s in self.b.manifest["stations"]:
            if s["kind"] == "dma":
                s["capacity"] = 1
        self.b.save_manifest()
        self.assertFails("M1", "over its window")

    def test_m2_store_buffer_over_capacity(self):
        for s in self.b.manifest["stations"]:
            if s["kind"] == "dmabuf":
                s["capacity"] = 1
        self.b.save_manifest()
        self.assertFails("M2", "slots held")

    def test_m2_staged_more_than_held(self):
        i = self.b.find("buffers", lambda i: self.b.get("buffers", "held", i) > 0)
        self.b.set("buffers", "staged", i, self.b.get("buffers", "held", i) + 1)
        self.assertFails("M2", "staged")

    # -- M3 -----------------------------------------------------------------
    def test_m3_first_burst_before_credit(self):
        q = self.b.find("requests", lambda q: self.b.get("requests", "t_credit", q) > 0)
        self.b.set("requests", "t_first", q, self.b.get("requests", "t_credit", q) - 1)
        self.assertFails("M3", "first burst")

    # -- M4 -----------------------------------------------------------------
    def test_m4_burst_served_twice(self):
        cas = [i for i in range(self.b.rows("commands")) if self.b.get("commands", "kind", i) == RD]
        self.b.set("commands", "burst", cas[1], self.b.get("commands", "burst", cas[0]))
        self.assertFails("M4", "not exactly one")

    def test_m4_request_missing_a_burst(self):
        self.b.set("bursts", "request", 0, self.b.get("bursts", "request", 0) + 1)
        self.assertFails("M4", "span")

    def test_m4_burst_done_before_its_data(self):
        self.b.set("bursts", "t_done", 0, self.b.get("bursts", "t_data0", 0) - 1)
        self.assertFails("M4", "not ordered")

    # -- M5: every parameter, by tightening the table the run is checked against --------------
    def test_m5_each_parameter_is_checked(self):
        for p in ("tRCD", "tRP", "tRAS", "tRC", "tRFCpb", "tRRD_L", "tRRD_S", "tCCD_L", "tCCD_S", "tFAW"):
            with self.subTest(param=p):
                b = Bundle(BUNDLE)
                try:
                    b.manifest["dram_timing"]["params"][p] = 1_000_000
                    b.save_manifest()
                    code, report = run(b.dir)
                    self.assertEqual(code, 1)
                    self.assertTrue(any(v["invariant"] == "M5" and f": {p} = " in v["message"]
                                        for v in report["violations"]), f"no M5 violation names {p}")
                finally:
                    b.close()

    def test_m5_activate_to_cas_one_tick_short(self):
        # The data, not the table: a CAS moved to one tick inside tRCD of its ACT.
        act, cas = self.bank_pair(ACT, {RD, WR})
        trcd = self.b.manifest["dram_timing"]["params"]["tRCD"]
        self.b.set("commands", "k_issue", cas, self.b.get("commands", "k_issue", act) + trcd - 1)
        self.assertFails("M5", "tRCD")

    # -- unreadable ---------------------------------------------------------
    def test_a_version_1_bundle_is_refused(self):
        self.b.manifest["version"] = 1
        self.b.save_manifest()
        self.assertUnreadable()

    def test_a_missing_timing_table_is_refused(self):
        del self.b.manifest["dram_timing"]
        self.b.save_manifest()
        self.assertUnreadable()

    def test_a_misdeclared_column_is_refused(self):
        for c in self.b.manifest["tables"]["commands"]["columns"]:
            if c["name"] == "k_issue":
                c["dtype"] = "f64"
        self.b.save_manifest()
        self.assertUnreadable()

    def test_an_index_past_its_table_is_refused(self):
        self.b.set("commands", "burst", 0 if self.b.get("commands", "kind", 0) != REF else 1,
                   self.b.rows("bursts") + 5)
        self.assertUnreadable()

    def test_a_station_without_an_index_is_refused(self):
        # check() keys engines by the index in the name; a crash there would exit 1, a "violation".
        s = next(s for s in self.b.manifest["stations"] if s["kind"] == "dmabuf")
        s["name"] = "t4/dmabuf"
        self.b.save_manifest()
        self.assertUnreadable()

    def test_a_non_numeric_burst_size_is_refused(self):
        self.b.manifest["burst_bytes"] = "64"
        self.b.save_manifest()
        self.assertUnreadable()

    def test_a_non_bundle_is_refused(self):
        code, _ = run(self.b.dir.parent)
        self.assertEqual(code, 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = sys.argv.pop(1)
    unittest.main()
