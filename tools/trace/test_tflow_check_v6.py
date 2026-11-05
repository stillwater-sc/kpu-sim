#!/usr/bin/env python3
"""
Self-test for tflow_check.py on VERSION 6 -- a CSP program's record with its staged L2 copies and its
gemm chains on the fabric (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 4). The clean bundle
passes; TF13, TF14 and the new pyramid rows of TF8 each fail when they are broken; and a bundle that
does not say its L2 slots is unreadable.

Usage: python3 test_tflow_check_v6.py <clean-staged-bundle.tflow>   (kpu-csp-gen --stage-a, kpu-run --tflow)

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import sys
import unittest

import test_tflow_check as base
from test_tflow_check import Bundle, run

BUNDLE = None


class TflowCheckV6SelfTest(unittest.TestCase):
    def setUp(self):
        self.b = Bundle(BUNDLE)
        self.assertEqual(self.b.manifest["version"], 6)
        self.assertGreater(self.b.manifest["l2_slots"], 0, "the self-test needs a program that stages")
        self.assertGreater(self.rows("l2_residency"), 1)
        self.assertGreater(self.rows("chain"), 1)

    def tearDown(self):
        self.b.close()

    def rows(self, table):
        return self.b.manifest["tables"][table]["rows"]

    def assertFails(self, invariant):
        code, report = run(self.b.dir)
        self.assertEqual(code, 1, f"expected {invariant} to fail, checker exited {code}")
        found = {v["invariant"] for v in report["violations"]}
        self.assertIn(invariant, found, f"violations were {sorted(found)}")

    def test_clean_bundle_passes(self):
        code, report = run(self.b.dir)
        self.assertEqual(code, 0, report)
        self.assertIn("TF13", report["checked"])
        self.assertIn("TF14", report["checked"])

    # -- TF13: the staged copies -------------------------------------------------
    def test_tf13_two_copies_in_one_slot(self):
        # Stretch a copy to the end of the run: the next copy staged into its slot now lands on it.
        n = self.rows("l2_residency")
        for a in range(n):
            for b in range(n):
                if a != b and self.b.get("l2_residency", "slot", a) == self.b.get("l2_residency", "slot", b) and \
                        self.b.get("l2_residency", "t0", b) >= self.b.get("l2_residency", "t1", a):
                    self.b.set("l2_residency", "t1", a, self.b.get("l2_residency", "t1", b))
                    return self.assertFails("TF13")
        self.fail("the clean bundle reuses no L2 slot")

    def test_tf13_slot_beyond_the_program(self):
        self.b.set("l2_residency", "slot", 0, self.b.manifest["l2_slots"] + 2)
        self.assertFails("TF13")

    def test_tf13_more_copies_than_slots(self):
        self.b.manifest["l2_slots"] = 1
        self.b.save_manifest()
        for i in range(self.rows("l2_residency")):       # every copy in slot 0 would also collide
            self.b.set("l2_residency", "slot", i, 0)
        self.assertFails("TF13")

    def test_tf13_a_copy_never_fed(self):
        self.b.set("l2_residency", "feeds", 0, 0)
        self.assertFails("TF13")

    # -- TF14: the fabric's phases -------------------------------------------------
    def test_tf14_front_off_the_skew(self):
        self.b.set("chain", "front", 0, self.b.get("chain", "front", 0) + 1)
        self.assertFails("TF14")

    def test_tf14_results_leave_too_early(self):
        self.b.set("chain", "out1", 0, self.b.get("chain", "last", 0))
        self.assertFails("TF14")

    def test_tf14_results_leave_late(self):
        # Later than the top link allows: the lower bound still holds, the exact recurrence does not.
        last = self.rows("chain") - 1
        self.b.set("chain", "out1", last, self.b.get("chain", "out1", last) + 1)
        self.assertFails("TF14")

    def test_tf14_two_chains_stream_at_once(self):
        # Start the second chain on the first one's compute tile inside the first one's operand stream.
        s = self.b.get("chain", "station", 0)
        for i in range(1, self.rows("chain")):
            if self.b.get("chain", "station", i) == s:
                delta = self.b.get("chain", "in0", i) - self.b.get("chain", "in0", 0) - 1
                for col in ("in0", "front", "in1", "last"):
                    self.b.set("chain", col, i, self.b.get("chain", col, i) - delta)
                return self.assertFails("TF14")
        self.fail("no compute tile runs two chains")

    def test_tf14_chain_off_the_fabric(self):
        dram = next(i for i, s in enumerate(self.b.manifest["stations"]) if s["kind"] == "dram")
        self.b.set("chain", "station", 0, dram)
        self.assertFails("TF14")

    # -- TF8: the version-6 rows ---------------------------------------------------
    def test_tf8_l2_slot_and_cf_rows_must_be_there(self):
        for kind in ("l2_slot", "cf_in", "cf_out"):
            with self.subTest(kind=kind):
                b = Bundle(BUNDLE)
                try:
                    lod = json.loads((b.dir / "lod.json").read_text())
                    self.assertTrue(any(x["kind"] == kind for x in lod["rows"]))
                    lod["rows"] = [x for x in lod["rows"] if x["kind"] != kind]
                    (b.dir / "lod.json").write_text(json.dumps(lod))
                    code, report = run(b.dir)
                    self.assertEqual(code, 1)
                    self.assertIn("TF8", {v["invariant"] for v in report["violations"]})
                finally:
                    b.close()

    def test_tf8_the_l2_rows_count_the_staged_copies(self):
        # The pyramid is checked against the l2_residency table: shift a copy and the base disagrees.
        self.b.set("l2_residency", "t0", 0, self.b.get("l2_residency", "t0", 0) + 1)
        self.assertFails("TF8")

    def test_version_6_must_say_its_l2_slots(self):
        del self.b.manifest["l2_slots"]
        self.b.save_manifest()
        code, _ = run(self.b.dir)
        self.assertEqual(code, 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = sys.argv.pop(1)
    base.BUNDLE = BUNDLE
    unittest.main()
