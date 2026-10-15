#!/usr/bin/env python3
"""
Self-test for dflow_check.py (docs/plans/domain-flow-debugger.md step 2): every invariant must
FAIL on a bundle that breaks it, and the clean bundle must pass. A checker that passes everything
is not a checker.

Usage: python3 test_dflow_check.py <clean-bundle.dflow>

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
CHECK = HERE / "dflow_check.py"
BUNDLE = None
FMT = {"u8": ("<B", 1), "u16": ("<H", 2), "u32": ("<I", 4), "u64": ("<Q", 8), "i64": ("<q", 8)}
OPENS, CLOSES = 1, 2
NONE64 = 0xFFFFFFFFFFFFFFFF


def run(path):
    p = subprocess.run([sys.executable, str(CHECK), str(path), "--json"], capture_output=True, text=True)
    report = json.loads(p.stdout) if p.returncode in (0, 1) else None
    return p.returncode, report


class Bundle:
    """A writable copy of the clean bundle, with column-level edits."""

    def __init__(self, src):
        self.dir = Path(tempfile.mkdtemp(prefix="dflow-selftest-")) / "b.dflow"
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


class DflowCheckSelfTest(unittest.TestCase):
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

    # -- the clean run ------------------------------------------------------
    def test_clean_bundle_passes(self):
        code, report = run(self.b.dir)
        self.assertEqual(code, 0, report and report["violations"][:5])

    def test_the_run_exercises_what_is_checked(self):
        # A check over nothing passes vacuously: the clean run must open and release residencies,
        # nest loops at least three deep, and record every step.
        self.assertFalse(self.b.manifest["cut"])
        self.assertGreater(self.b.rows("residency"), 10)
        depths = {self.b.get("iteration", "depth", i) for i in range(self.b.rows("iteration"))}
        self.assertTrue({0, 1, 2, 3} <= depths, depths)
        flags = {self.b.get("steps", "flags", i) & (OPENS | CLOSES) for i in range(self.b.rows("steps"))}
        self.assertTrue({OPENS, CLOSES} <= flags)

    # -- D2 -----------------------------------------------------------------
    def test_d2_step_out_of_order(self):
        self.b.set("steps", "action", 5, 6)
        self.assertFails("D2", "steps[5]")

    def test_d2_uncut_bundle_missing_actions(self):
        self.b.manifest["totals"]["actions"] += 1
        self.b.save_manifest()
        self.assertFails("D2", "not cut")

    # -- D3 -----------------------------------------------------------------
    def test_d3_occupancy_not_the_residencies(self):
        self.b.set("steps", "held_l3", 10, self.b.get("steps", "held_l3", 10) + 1)
        self.assertFails("D3", "after step 10")

    # -- D4 -----------------------------------------------------------------
    def test_d4_residency_ends_before_it_opens(self):
        self.b.set("residency", "close", 0, self.b.get("residency", "open", 0))
        self.assertFails("D4", "not after its open")

    def test_d4_residency_never_ends(self):
        self.b.set("residency", "close", 3, NONE64)
        self.assertFails("D4", "never ends")

    def test_d4_open_without_an_opening_step(self):
        i = self.b.get("residency", "open", 2)
        self.b.set("steps", "flags", i, self.b.get("steps", "flags", i) & ~OPENS)
        self.assertFails("D4", "does not open it")

    # -- D5 -----------------------------------------------------------------
    def test_d5_over_capacity(self):
        peak = max(self.b.get("steps", "held_l3", i) for i in range(self.b.rows("steps")))
        self.b.manifest["machine"]["l3"] = peak - 1
        self.b.save_manifest()
        self.assertFails("D5", "over its capacity")

    # -- D6 -----------------------------------------------------------------
    def test_d6_program_summary_short(self):
        i = self.b.find("iteration", lambda i: self.b.get("iteration", "depth", i) == 0)
        self.b.set("iteration", "loads", i, self.b.get("iteration", "loads", i) - 1)
        self.assertFails("D6", "the program's iteration")

    def test_d6_iterations_overlap(self):
        rows = sorted((self.b.get("iteration", "first", i), i) for i in range(self.b.rows("iteration"))
                      if self.b.get("iteration", "depth", i) == 2)
        (_, a), (first_b, _) = rows[0], rows[1]
        self.b.set("iteration", "last", a, first_b)
        self.assertFails("D6", "overlap")

    def test_d6_counts_not_the_steps(self):
        i = self.b.find("iteration", lambda i: self.b.get("iteration", "depth", i) == 3)
        self.b.set("iteration", "peak_l2", i, self.b.get("iteration", "peak_l2", i) + 5)
        self.assertFails("D6", "its steps give")

    # -- D8 -----------------------------------------------------------------
    def test_d8_time_column(self):
        self.b.manifest["tables"]["residency"]["columns"].append({"name": "t_open", "dtype": "u64", "offset": 0})
        self.b.save_manifest()
        self.assertFails("D8", "t_open")

    # -- D9 -----------------------------------------------------------------
    def test_d9_step_address_not_its_residency(self):
        i = self.b.find("steps", lambda i: self.b.get("steps", "residency", i) != NONE64)
        self.b.set("steps", "slot", i, self.b.get("steps", "slot", i) + 1)
        self.assertFails("D9", "its residency")

    def test_d9_two_tiles_in_one_slot(self):
        # Residency 1 moved into residency 0's slot while 0 is still held.
        self.assertLess(self.b.get("residency", "open", 1), self.b.get("residency", "close", 0))
        self.b.set("residency", "slot", 1, self.b.get("residency", "slot", 0))
        self.assertFails("D9", "still holds")

    def test_d6_iteration_past_the_run_is_reported_not_a_crash(self):
        i = self.b.find("iteration", lambda i: self.b.get("iteration", "depth", i) == 3)
        self.b.set("iteration", "last", i, self.b.rows("steps") + 100)
        self.assertFails("D6", "runs past")

    def test_d6_position_out_of_range_is_reported_not_a_crash(self):
        self.b.set("steps", "where", 7, self.b.rows("where") + 5)
        self.assertFails("D6", "names position")

    # -- unreadable ---------------------------------------------------------
    def test_malformed_manifest_is_unreadable(self):
        del self.b.manifest["totals"]["loads"]
        self.b.save_manifest()
        self.assertUnreadable()

    def test_malformed_table_entry_is_unreadable(self):
        self.b.manifest["tables"]["steps"]["rows"] = "many"
        self.b.save_manifest()
        self.assertUnreadable()

    def test_negative_column_offset_is_unreadable(self):
        self.b.manifest["tables"]["steps"]["columns"][0]["offset"] = -8
        self.b.save_manifest()
        self.assertUnreadable()

    def test_unknown_version_is_unreadable(self):
        self.b.manifest["version"] = 2
        self.b.save_manifest()
        self.assertUnreadable()

    def test_missing_table_is_unreadable(self):
        (self.b.dir / "steps.bin").unlink()
        self.assertUnreadable()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = Path(sys.argv.pop(1)).resolve()
    unittest.main()
