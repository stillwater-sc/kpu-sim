#!/usr/bin/env python3
"""
Self-test of the checkers on an L2 accumulator (#455): dflow_check's S5 (and S2's allowance for a
partial's Drains and Writeback) and tflow_check's TF13 for a partial.

Usage: python3 test_l2acc_checks.py <l2acc.dflow> <l2acc.tflow>

Both bundles are what kpu-run writes for a program with `acc C[:, j] in l2 init D[:, j]`
(kpu-csp-gen --dataflow ws --init bias). They must check clean, and each rule, broken on purpose in
a copy, must fail under its own id. A copy without the optional columns (partial, merges, merge)
must fail as records did before them: the columns are what tell a partial from a staged copy.

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

DFLOW = None
TFLOW = None


def run(checker, path):
    p = subprocess.run([sys.executable, str(HERE / checker), str(path), "--json"], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode in (0, 1) else None)


def ids(report):
    return sorted({v["invariant"] for v in report["violations"]})


def drop_columns(b, table, names):
    t = b.manifest["tables"][table]
    t["columns"] = [c for c in t["columns"] if c["name"] not in names]
    b.save_manifest()


class DflowPartials(unittest.TestCase):
    def setUp(self):
        self.b = Bundle(DFLOW)
        g = self.b.rows("l2_staging")
        self.partials = [i for i in range(g) if self.b.get("l2_staging", "partial", i)]
        self.assertTrue(self.partials, "the fixture has no L2 accumulator's partial")
        self.p = self.partials[0]
        steps = self.b.rows("steps")
        self.on_p = [i for i in range(steps) if self.b.get("steps", "staging", i) == self.p]

    def tearDown(self):
        self.b.close()

    def step(self, kind, merge=None):
        return next(i for i in self.on_p if self.b.get("steps", "kind", i) == kind
                    and (merge is None or self.b.get("steps", "merge", i) == merge))

    def fails(self, invariant, needle):
        code, report = run("dflow_check.py", self.b.dir)
        self.assertEqual(code, 1, f"expected {invariant} to fail; the checker exited {code}")
        hits = [v["message"] for v in report["violations"] if v["invariant"] == invariant]
        self.assertTrue(any(needle in h for h in hits), f"{invariant}: {hits[:3]} (all: {ids(report)})")

    def test_clean(self):
        code, report = run("dflow_check.py", self.b.dir)
        self.assertEqual(code, 0, report and report["violations"][:3])

    def test_a_partial_is_opened_by_an_overwrite(self):
        self.b.set("steps", "merge", self.step(dc.DRAIN_KIND, 1), 2)
        self.fails("S5", "not overwrite")

    def test_a_later_merge_accumulates(self):
        self.b.set("steps", "merge", self.step(dc.DRAIN_KIND, 2), 1)
        self.fails("S5", "not accumulate")

    def test_a_partial_is_never_fed(self):
        self.b.set("l2_staging", "feeds", self.p, 1)
        self.fails("S5", "never fed")

    def test_its_merges_are_its_drains(self):
        self.b.set("l2_staging", "merges", self.p, self.b.get("l2_staging", "merges", self.p) + 1)
        self.fails("S5", "merges; its Drains give")

    def test_only_a_drain_merges(self):
        wb = self.step(dc.WRITEBACK_KIND)
        self.b.set("steps", "merge", wb, 2)
        self.fails("S5", "only a Drain merges")

    def test_a_staged_copy_takes_no_merge(self):
        self.b.set("l2_staging", "partial", self.p, 0)
        self.fails("S2", "is not an L2 accumulator's partial")

    def test_without_the_columns_it_fails_as_before(self):
        drop_columns(self.b, "l2_staging", {"partial", "merges"})
        drop_columns(self.b, "steps", {"merge"})
        self.fails("S2", "a DRAIN carries an L2 address")

    def test_its_open_step_is_checked_from_the_partial(self):
        # #459's review: the opening Drain loses its staging reference, its L2 flags and its merge,
        # so no step-side rule sees it; the partial's own open must still find an overwrite there.
        o = self.b.get("l2_staging", "open", self.p)
        self.b.set("steps", "staging", o, dc.NONE64)
        self.b.set("steps", "flags", o, self.b.get("steps", "flags", o) & ~(dc.HAS_L2 | dc.STAGES))
        self.b.set("steps", "merge", o, 0)
        self.fails("S5", f"partial {self.p} opens at step {o}, which is not an overwrite Drain")

    def test_the_writeback_follows_the_last_drain(self):
        # #459's review: a merge after the Writeback is not in what was written back. Swap the last
        # accumulate Drain with the Writeback: the counts stay, the order is wrong.
        d = max(i for i in self.on_p if self.b.get("steps", "kind", i) == dc.DRAIN_KIND)
        w = self.step(dc.WRITEBACK_KIND)
        self.assertLess(d, w)
        self.b.set("steps", "kind", d, dc.WRITEBACK_KIND)
        self.b.set("steps", "merge", d, 0)
        self.b.set("steps", "kind", w, dc.DRAIN_KIND)
        self.b.set("steps", "merge", w, 2)
        self.fails("S5", "after its Writeback")


class TflowPartials(unittest.TestCase):
    def setUp(self):
        self.b = Bundle(TFLOW)
        n = self.b.rows("l2_residency")
        self.partials = [i for i in range(n) if self.b.get("l2_residency", "partial", i)]
        self.assertTrue(self.partials, "the fixture has no L2 accumulator's partial")
        self.p = self.partials[0]

    def tearDown(self):
        self.b.close()

    def fails(self, needle):
        code, report = run("tflow_check.py", self.b.dir)
        self.assertEqual(code, 1, f"expected TF13 to fail; the checker exited {code}")
        hits = [v["message"] for v in report["violations"] if v["invariant"] == "TF13"]
        self.assertTrue(any(needle in h for h in hits), f"TF13: {hits[:3]} (all: {ids(report)})")

    def test_clean(self):
        code, report = run("tflow_check.py", self.b.dir)
        self.assertEqual(code, 0, report and report["violations"][:3])
        self.assertTrue(all(self.b.get("l2_residency", "merges", i) >= 1 for i in self.partials))

    def test_a_partial_is_never_fed(self):
        self.b.set("l2_residency", "feeds", self.p, 2)
        self.fails("a partial is merged into, never fed")

    def test_a_partial_is_merged_into(self):
        self.b.set("l2_residency", "merges", self.p, 0)
        self.fails("is never merged into")

    def test_without_the_columns_it_fails_as_before(self):
        drop_columns(self.b, "l2_residency", {"partial", "merges"})
        self.fails("never fed")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: test_l2acc_checks.py <l2acc.dflow> <l2acc.tflow>")
    TFLOW = sys.argv.pop(2)
    DFLOW = sys.argv.pop(1)
    unittest.main(verbosity=2)
