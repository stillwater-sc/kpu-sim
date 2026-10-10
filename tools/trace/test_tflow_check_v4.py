#!/usr/bin/env python3
"""
Self-test for tflow_check.py on a VERSION-4 bundle -- a CSP program's record, whose ops are the
program's actions (docs/plans/kpu-run-csp-programs.md step 4b). The clean bundle passes; each
invariant that version 4 states differently fails when it is broken; and the version label is
load-bearing.

Usage: python3 test_tflow_check_v4.py <clean-v4-bundle.tflow>

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import sys
import unittest

import test_tflow_check as base
from test_tflow_check import Bundle, run

BUNDLE = None


class TflowCheckV4SelfTest(unittest.TestCase):
    def setUp(self):
        self.b = Bundle(BUNDLE)
        self.assertEqual(self.b.manifest["version"], 4)
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
