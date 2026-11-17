#!/usr/bin/env python3
"""
Self-test of the checkers on an L2 accumulator (#455): dflow_check's S5 (and S2's allowance for a
partial's Drains and Writeback) and tflow_check's TF13 for a partial.

Usage: python3 test_l2acc_checks.py <l2acc.dflow> <l2acc.tflow>

Both bundles are what kpu-run writes for a program with `acc C[:, j] in l2 init D[:, j]`
(kpu-csp-gen --dataflow ws --init bias). They must check clean, and each rule, broken on purpose in
a copy, must fail under its own id. A copy without the optional columns (partial, merges, merge)
must fail as records did before them: the columns are what tell a partial from a staged copy.
The fixture's chains are ws (#395 step 2): TF14 and P2/P3 check their stationary timing.

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

    def test_a_cut_record_still_has_one_writeback(self):
        # #459's review: in a cut record (not every step recorded) the exactly-once count is not
        # checked, and a second Writeback went unseen. The last accumulate Drain (before the
        # Writeback) becomes a second Writeback.
        d = max(i for i in self.on_p if self.b.get("steps", "kind", i) == dc.DRAIN_KIND)
        self.assertLess(d, self.step(dc.WRITEBACK_KIND))
        self.b.set("steps", "kind", d, dc.WRITEBACK_KIND)
        self.b.set("steps", "merge", d, 0)
        self.b.manifest["cut"] = True
        self.b.save_manifest()
        self.fails("S5", "written back 2 times, more than once")

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


class DflowDataflow(unittest.TestCase):
    """#395: the fixture's program states `dataflow ws` on its accumulator; S6 checks the column."""

    def setUp(self):
        self.b = Bundle(DFLOW)
        self.steps = self.b.rows("steps")

    def tearDown(self):
        self.b.close()

    def test_calls_and_drains_carry_ws(self):
        ws = [i for i in range(self.steps) if self.b.get("steps", "dataflow", i) == 2]
        self.assertTrue(ws, "no step carries ws")
        kinds = {self.b.get("steps", "kind", i) for i in ws}
        self.assertLessEqual(kinds, {dc.CALL_KIND, dc.DRAIN_KIND})
        self.assertIn(dc.CALL_KIND, kinds)

    def fails(self, needle):
        code, report = run("dflow_check.py", self.b.dir)
        self.assertEqual(code, 1, f"expected S6 to fail; the checker exited {code}")
        hits = [v["message"] for v in report["violations"] if v["invariant"] == "S6"]
        self.assertTrue(any(needle in h for h in hits), f"S6: {hits[:3]} (all: {ids(report)})")

    def test_only_a_call_or_drain_carries_one(self):
        load = next(i for i in range(self.steps) if self.b.get("steps", "kind", i) == dc.LOAD_KIND)
        self.b.set("steps", "dataflow", load, 2)
        self.fails("carries a dataflow; only a Call and its Drain do")

    def test_a_dataflow_is_in_range(self):
        call = next(i for i in range(self.steps) if self.b.get("steps", "kind", i) == dc.CALL_KIND)
        self.b.set("steps", "dataflow", call, 9)
        self.fails("dataflow 9 is not default, os, ws or is")


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


class TflowStationary(unittest.TestCase):
    """#395 step 2: the fixture's ws chains, timed PRELOAD R, the stream, out south (TF14)."""

    def setUp(self):
        self.b = Bundle(TFLOW)
        n = self.b.rows("chain")
        self.ws = [i for i in range(n) if self.b.get("chain", "dataflow", i) == 2]
        self.assertTrue(self.ws, "the fixture has no ws chain")
        self.c = self.ws[0]

    def tearDown(self):
        self.b.close()

    def fails(self, needle):
        code, report = run("tflow_check.py", self.b.dir)
        self.assertEqual(code, 1, f"expected TF14 to fail; the checker exited {code}")
        hits = [v["message"] for v in report["violations"] if v["invariant"] == "TF14"]
        self.assertTrue(any(needle in h for h in hits), f"TF14: {hits[:3]} (all: {ids(report)})")

    def test_the_formulas_hold(self):
        b, c = self.b, self.c
        R, S = b.get("chain", "preload", c), b.get("chain", "streamed", c)
        self.assertGreater(R, 0)
        self.assertEqual(b.get("chain", "front", c), b.get("chain", "in0", c) + R)
        self.assertEqual(b.get("chain", "in1", c), b.get("chain", "front", c) + S)
        self.assertEqual(b.get("chain", "out1", c), b.get("chain", "last", c) + R)

    def test_a_short_preload_fails(self):
        self.b.set("chain", "front", self.c, self.b.get("chain", "front", self.c) - 1)
        self.fails("do not follow PRELOAD")

    def test_a_dataflow_is_in_range(self):
        self.b.set("chain", "dataflow", self.c, 7)
        self.fails("dataflow 7 is not os, ws or is")

    def test_a_stationary_chain_has_one_call(self):
        self.b.set("chain", "calls", self.c, 2)
        self.fails("is stationary with 2 calls")

    def test_without_the_columns_it_is_os_and_fails(self):
        # An older record had no stationary chains: read as os, these break the OS wavefront.
        drop_columns(self.b, "chain", {"dataflow", "preload", "streamed"})
        self.fails("plus the skew")


class DflowStationary(unittest.TestCase):
    """#395 step 2: the same chains in the .dflow overlay (P2, P3)."""

    def setUp(self):
        self.b = Bundle(DFLOW)
        n = self.b.rows("lt1_chains")
        self.ws = [i for i in range(n) if self.b.get("lt1_chains", "dataflow", i) == 2]
        self.assertTrue(self.ws, "the fixture has no ws chain")
        self.c = self.ws[0]

    def tearDown(self):
        self.b.close()

    def fails(self, inv, needle):
        code, report = run("dflow_check.py", self.b.dir)
        self.assertEqual(code, 1, f"expected {inv} to fail; the checker exited {code}")
        hits = [v["message"] for v in report["violations"] if v["invariant"] == inv]
        self.assertTrue(any(needle in h for h in hits), f"{inv}: {hits[:3]} (all: {ids(report)})")

    def test_a_short_stream_fails(self):
        self.b.set("lt1_chains", "in1", self.c, self.b.get("lt1_chains", "in1", self.c) - 1)
        self.fails("P2", "rows streamed")

    def test_a_late_exit_fails(self):
        self.b.set("lt1_chains", "out1", self.c, self.b.get("lt1_chains", "out1", self.c) + 1)
        self.fails("P2", "out1 - last")

    def test_a_wrong_preload_fails(self):
        self.b.set("lt1_chains", "preload", self.c, 3)
        self.fails("P2", "PRELOAD 3")

    # The shared rules hold for a stationary chain too (#466's review).
    def test_a_drained_chain_is_closed(self):
        self.b.set("lt1_chains", "out1", self.c, dc.NONE64)
        self.fails("P2", "both are known or neither is")

    def test_its_call_runs_on_its_lane(self):
        call = next(self.b.get("lt1_calls", "action", r) for r in range(self.b.rows("lt1_calls"))
                    if self.b.get("lt1_calls", "chain", r) == self.c)
        cf = self.b.manifest["processes"].index("cf")
        g = next(g for g in range(self.b.rows("lt1_legs"))
                 if self.b.get("lt1_legs", "action", g) == call and self.b.get("lt1_legs", "proc", g) == cf)
        self.b.set("lt1_legs", "lane", g, self.b.get("lt1_chains", "lane", self.c) + 1)
        self.fails("P3", "the chain owns lane")

    def test_a_stationary_chain_has_one_call(self):
        self.b.set("lt1_chains", "calls", self.c, 2)
        self.fails("P3", "is stationary with 2 calls")

    def test_its_drain_waits_for_its_results(self):
        drain = self.b.get("lt1_chains", "drain_action", self.c)
        self.assertNotEqual(drain, dc.NONE64)
        g = next(g for g in range(self.b.rows("lt1_legs")) if self.b.get("lt1_legs", "action", g) == drain)
        self.b.set("lt1_chains", "out1", self.c, self.b.get("lt1_legs", "start", g) + 1)
        self.fails("P4", "starts before its last result is out")

    def test_the_manifest_names_the_dataflow(self):
        f = self.b.manifest["lt1"]["fabric"]
        self.assertEqual((f["mapping"], f["drain_edge"]), ("weight-stationary", "south"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: test_l2acc_checks.py <l2acc.dflow> <l2acc.tflow>")
    TFLOW = sys.argv.pop(2)
    DFLOW = sys.argv.pop(1)
    unittest.main(verbosity=2)
