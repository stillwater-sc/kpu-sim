#!/usr/bin/env python3
"""
Self-test for dflow_diff.py (#402): two records of one program joined on the identity key.

Usage: python3 test_dflow_diff.py <lt1.dflow> <lca.dflow> <rtl-sample-abc.dflow> <lca-of-the-sample.dflow> <lb.dflow>

- the first two are kpu-sim's L-T1 and L-CA records of one program (kpu-run --dflow-lt1, --dflow-lca);
- the third is the RTL sample under the names A, B, C (dflow_rtl_sample.py --names A,B,C), and the
  fourth L-CA's record of its own program (--program-out), which L-CA runs under those names;
- the fifth an L-B record, which has no events.

A record joins itself with nothing one-sided and no divergence; a shifted event is named as the
first divergence; a dropped event is one-sided; L-T1 and L-CA join every event and diverge; the RTL
sample and L-CA join every transaction across L2's two address forms, and the sample's row events
are one-sided under matched transactions (the done-when of #402).

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflow_check as dc  # noqa: E402
from test_dflow_check import Bundle  # noqa: E402

DIFF = HERE / "dflow_diff.py"
LT1 = LCA = RTL = RTL_LCA = LB = None


def run(a, b, *extra):
    p = subprocess.run([sys.executable, str(DIFF), str(a), str(b), "--json", *extra], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode in (0, 1) else p.stderr)


class Diff(unittest.TestCase):
    def test_a_record_joins_itself(self):
        code, o = run(LCA, LCA)
        self.assertEqual(code, 0, o)
        self.assertEqual((o["only_a"]["count"], o["only_b"]["count"]), (0, 0))
        self.assertIsNone(o["first_divergence"])
        # L2's and the accumulator's addresses are placement, never identity (#401).
        self.assertEqual(o["dropped_addresses"], ["acc", "l2"])
        self.assertEqual(o["placement"]["count"], 0)

    def test_a_transfer_placed_elsewhere_differs(self):
        # L2's addresses are placement (§5.3): a transfer at another row still joins, and is
        # reported placed differently, which is a difference (exit 1) though nothing is one-sided.
        b = Bundle(LCA)
        try:
            rows = b.manifest["geometry"]["l2"]["rows"]
            e = next(i for i in range(b.rows("events"))
                     if dc.EVENT_KINDS[b.get("events", "kind", i)] == "BLOCK_MOVE"
                     and b.get("events", "dst", i) == dc.LAYERS.index("l2")
                     and b.get("events", "dst_addr", i) != dc.NONE64)
            b.set("events", "dst_addr", e, (b.get("events", "dst_addr", e) + 1) % rows)
            code, o = run(LCA, b.dir)
            self.assertEqual(code, 1, o)
            self.assertEqual((o["only_a"]["count"], o["only_b"]["count"]), (0, 0))
            self.assertIsNone(o["first_divergence"])
            self.assertEqual(o["placement"]["count"], 1)
            self.assertEqual([x["layer"] for x in o["placement"]["first"][0]["ends"]], ["l2"])
        finally:
            b.close()

    def test_a_shifted_event_is_the_first_divergence(self):
        b = Bundle(LCA)
        try:
            n = b.rows("events")
            e = n // 2
            while dc.EVENT_KINDS[b.get("events", "kind", e)] not in ("DMA", "BLOCK_MOVE", "STREAM", "COMPUTE", "DRAIN"):
                e += 1
            # Every event before e is unchanged, so e is the earliest that differs.
            b.set("events", "end", e, b.get("events", "end", e) + 5)
            code, o = run(LCA, b.dir)
            self.assertEqual(code, 1, o)
            f = o["first_divergence"]
            self.assertIsNotNone(f)
            self.assertEqual(f["duration"], 5)
            self.assertEqual(f["a"], [b.get("events", "cycle", e), b.get("events", "end", e) - 5])
            # Within the tolerance it is no divergence.
            code, o = run(LCA, b.dir, "--tolerance", "5")
            self.assertEqual(code, 0, o)
        finally:
            b.close()

    def test_the_levels_join_every_event_and_diverge(self):
        code, o = run(LT1, LCA)
        self.assertEqual(code, 1, o)
        # Every transaction, token and credit joins. L-CA's banked L2 adds its transfers' rows
        # (#401), which L-T1, timing a tile move whole, does not have: one-sided, each under a
        # joined transaction.
        self.assertEqual(o["only_a"]["count"], 0)
        rows = sum(o["only_b"]["by_kind"].get(k, 0) for k in ("BM_ROW", "STR_ROW", "DRAIN_ROW"))
        self.assertGreater(rows, 0, "the banked L-CA record has its transfers' rows")
        self.assertEqual(o["only_b"]["count"], rows)
        self.assertEqual(o["only_b"]["rows_under_matched_transactions"], rows)
        self.assertGreater(o["matched_transactions"], 0)
        self.assertIsNotNone(o["first_divergence"])
        self.assertNotEqual(o["cycles"][0], o["cycles"][1])

    def test_the_rtl_sample_joins_l_ca_across_l2s_forms(self):
        # #402's done-when: an L-CA record and an RTL record of the same program join, and the diff
        # names the first event where they diverge.
        code, o = run(RTL_LCA, RTL)
        self.assertEqual(code, 1, o)
        self.assertEqual(sorted(o["dropped_addresses"]), ["acc", "l2"], "slots against rows")
        sample = Bundle(RTL)
        try:
            tx = sum(1 for e in range(sample.rows("events"))
                     if dc.EVENT_KINDS[sample.get("events", "kind", e)] in ("DMA", "BLOCK_MOVE", "STREAM", "COMPUTE", "DRAIN"))
        finally:
            sample.close()
        self.assertEqual(o["matched_transactions"], tx, "every one of the sample's transactions joins")
        self.assertEqual(o["only_a"]["by_kind"].get("DMA", 0), 0)
        # The sample's row progress has no L-CA counterpart; each sits under a joined transaction.
        rows = sum(o["only_b"]["by_kind"].get(k, 0) for k in ("BM_ROW", "STR_ROW", "DRAIN_ROW"))
        self.assertGreater(rows, 0)
        self.assertEqual(o["only_b"]["rows_under_matched_transactions"], rows)
        self.assertIsNotNone(o["first_divergence"])
        # The text report says, row by row, whether its transaction joined (#470's review).
        p = subprocess.run([sys.executable, str(DIFF), str(RTL_LCA), str(RTL)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("(joined)", p.stdout)

    def test_an_event_of_another_identity_is_one_sided(self):
        b = Bundle(LCA)
        try:
            e = next(i for i in range(b.rows("events")) if dc.EVENT_KINDS[b.get("events", "kind", i)] == "TOKEN")
            b.set("events", "process", e, dc.NONE8)        # the same token, sent by no process: another key
            code, o = run(LCA, b.dir)
            self.assertEqual(code, 1, o)
            self.assertEqual((o["only_a"]["by_kind"], o["only_b"]["by_kind"]), ({"TOKEN": 1}, {"TOKEN": 1}))
            # The labels say why they did not join: the process is part of the identity (#470's review).
            a, b2 = o["only_a"]["first"][0]["key"], o["only_b"]["first"][0]["key"]
            self.assertNotEqual(a, b2)
            self.assertIn("by no process", b2)
        finally:
            b.close()

    def test_a_record_without_events_is_refused(self):
        code, err = run(LB, LCA)
        self.assertEqual(code, 2)
        self.assertIn("writes no events", err)

    def test_a_broken_record_is_refused(self):
        b = Bundle(LCA)
        try:
            b.set("events", "id", 0, 7)
            code, err = run(LCA, b.dir)
            self.assertEqual(code, 2)
            self.assertIn("fails dflow_check", err)
        finally:
            b.close()


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    LT1, LCA, RTL, RTL_LCA, LB = (Path(sys.argv.pop(1)).resolve() for _ in range(5))
    unittest.main(verbosity=2)
