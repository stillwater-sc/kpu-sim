#!/usr/bin/env python3
"""
Self-test for tflow_check.py on a CSP program's record -- VERSION 4, whose ops are the program's
actions (docs/plans/kpu-run-csp-programs.md step 4b), and VERSION 5, whose residencies also carry
their L3 address (docs/plans/l3-placement-at-lb.md step 5b). The clean bundle passes; each
invariant those versions state differently fails when it is broken (TF6, TF9; TF12 and the slot
rows of TF8 in version 5); and the version label is load-bearing.

Usage: python3 test_tflow_check_v4.py <clean-csp-bundle.tflow>   (kpu-run writes version 5)

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import sys
import unittest

import test_tflow_check as base
from test_tflow_check import Bundle, run

BUNDLE = None


class TflowCheckV4SelfTest(unittest.TestCase):
    def setUp(self):
        self.b = Bundle(BUNDLE)
        self.assertIn(self.b.manifest["version"], (4, 5))
        self.assertEqual(self.b.manifest["ops"], "csp-actions")

    def tearDown(self):
        self.b.close()

    def assertFails(self, invariant):
        code, report = run(self.b.dir)
        self.assertEqual(code, 1, f"expected {invariant} to fail, checker exited {code}")
        found = {v["invariant"] for v in report["violations"]}
        self.assertIn(invariant, found, f"violations were {sorted(found)}")

    def test_clean_bundle_passes(self):
        # Including TF6's version-4 rule: Feeds after the program released the L3 copy, and
        # calls on an accumulator that never had an L3 slot, are not violations.
        code, report = run(self.b.dir)
        self.assertEqual(code, 0, report)

    def test_tf6_l3_transit_without_a_slot(self):
        # End every residency at its start: every Load, Move, Writeback and ejection now moves
        # its tile through L3 without a slot.
        rows = self.b.manifest["tables"]["residency"]["rows"]
        for i in range(rows):
            self.b.set("residency", "t1", i, self.b.get("residency", "t0", i))
        self.assertFails("TF6")

    def test_tf9_writeback_no_longer_fills(self):
        # Relabel every writeback (hop 4, l2 -> l3) as a drain (hop 3): the result slots it filled
        # are now unfilled. (TF7 fails too -- the endpoints no longer match the hop.)
        for i in range(self.b.manifest["tables"]["transit"]["rows"]):
            if self.b.get("transit", "hop", i) == 4:
                self.b.set("transit", "hop", i, 3)
        self.assertFails("TF9")

    def test_version_4_must_say_its_ops(self):
        del self.b.manifest["ops"]
        self.b.save_manifest()
        code, _ = run(self.b.dir)
        self.assertEqual(code, 2)

    # -- version 5: the L3 address of every residency ------------------------
    def v5(self):
        if self.b.manifest["version"] != 5:
            self.skipTest("a version-4 bundle has no L3 addresses")

    def l3_residencies(self):
        l3 = next(i for i, s in enumerate(self.b.manifest["stations"]) if s["kind"] == "l3")
        return [i for i in range(self.b.manifest["tables"]["residency"]["rows"])
                if self.b.get("residency", "station", i) == l3]

    def test_tf12_two_tiles_in_one_slot(self):
        self.v5()
        # Move a residency into the slot of one that overlaps it in time.
        rs = self.l3_residencies()
        for a in rs:
            for b in rs:
                if a != b and self.b.get("residency", "slot", a) != self.b.get("residency", "slot", b) and \
                        self.b.get("residency", "t0", b) < self.b.get("residency", "t1", a) and \
                        self.b.get("residency", "t0", a) < self.b.get("residency", "t1", b):
                    slot = self.b.get("residency", "slot", a)
                    self.b.set("residency", "slot", b, slot)
                    self.b.set("residency", "bank", b, self.b.get("residency", "bank", a))
                    return self.assertFails("TF12")
        self.fail("the clean bundle has no two residencies live at once")

    def test_tf12_slot_beyond_the_capacity(self):
        self.v5()
        cap = next(s["capacity"] for s in self.b.manifest["stations"] if s["kind"] == "l3")
        self.b.set("residency", "slot", self.l3_residencies()[0], cap + 3)
        self.assertFails("TF12")

    def test_tf12_wrong_bank(self):
        self.v5()
        self.b.manifest["l3_banks"] = 4
        self.b.save_manifest()
        i = self.l3_residencies()[0]
        self.b.set("residency", "bank", i, (self.b.get("residency", "slot", i) + 1) % 4)
        self.assertFails("TF12")

    def test_tf8_slot_rows_must_be_there(self):
        self.v5()
        lod = json.loads((self.b.dir / "lod.json").read_text())
        self.assertTrue(any(x["kind"] == "l3_slot" for x in lod["rows"]))
        lod["rows"] = [x for x in lod["rows"] if x["kind"] != "l3_slot"]
        (self.b.dir / "lod.json").write_text(json.dumps(lod))
        self.assertFails("TF8")

    def test_version_5_must_say_its_banks(self):
        self.v5()
        del self.b.manifest["l3_banks"]
        self.b.save_manifest()
        code, _ = run(self.b.dir)
        self.assertEqual(code, 2)

    def test_read_as_version_3_the_rules_are_wrong(self):
        # The label is load-bearing: read under version 3's TF6, an accumulator's calls run
        # without an L3 slot, which is exactly what a CSP program does.
        self.b.manifest["version"] = 3
        del self.b.manifest["ops"]
        self.b.save_manifest()
        self.assertFails("TF6")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = sys.argv.pop(1)
    base.BUNDLE = BUNDLE
    unittest.main()
