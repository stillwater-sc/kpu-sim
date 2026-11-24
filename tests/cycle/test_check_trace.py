#!/usr/bin/env python3
"""check_trace.py rejects what it should (#551): run on the demo's files, then on corrupted copies.

    test_check_trace.py <trace.json> <counters.json>
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "check_trace.py")
TRACE = COUNTERS = None


def run(trace, counters):
    with tempfile.TemporaryDirectory() as d:
        t, c = os.path.join(d, "t.json"), os.path.join(d, "c.json")
        # Closed before the checker reads them: unflushed output is not guaranteed to be there.
        with open(t, "w") as f:
            json.dump(trace, f)
        with open(c, "w") as f:
            json.dump(counters, f)
        p = subprocess.run([sys.executable, CHECK, t, c], capture_output=True, text=True)
        return p.returncode, p.stdout


class CheckTrace(unittest.TestCase):
    def setUp(self):
        # Collected by a plain pytest run, which passes no paths: skip rather than error.
        if TRACE is None or COUNTERS is None:
            self.skipTest("needs the trace and counters paths: run through CTest, or pass them on the command line")
        with open(TRACE) as f:
            self.trace = json.load(f)
        with open(COUNTERS) as f:
            self.counters = json.load(f)

    def test_the_demo_passes(self):
        code, out = run(self.trace, self.counters)
        self.assertEqual(code, 0, out)

    def test_counters_off_by_one(self):
        c = copy.deepcopy(self.counters)
        c["rows"][1]["busy"] += 1
        code, out = run(self.trace, c)
        self.assertEqual(code, 1, out)
        self.assertIn("busy slices span", out)

    def test_an_overlapping_slice(self):
        t = copy.deepcopy(self.trace)
        xs = [e for e in t["traceEvents"] if e["ph"] == "X"]
        xs[0]["dur"] += 10
        code, out = run(t, self.counters)
        self.assertEqual(code, 1, out)
        self.assertIn("overlaps", out)

    def test_a_stall_with_no_counter_entry(self):
        # An extra, non-overlapping slice of a cause the counters do not report (#551's review).
        t = copy.deepcopy(self.trace)
        x = next(e for e in t["traceEvents"] if e["ph"] == "X")
        end = max(e["ts"] + e.get("dur", 0) for e in t["traceEvents"] if e.get("tid") == x["tid"] and e["ph"] == "X")
        t["traceEvents"].append({"ph": "X", "pid": 1, "tid": x["tid"], "name": "stall: unexpected",
                                 "ts": end + 1, "dur": 0.001})
        code, out = run(t, self.counters)
        self.assertEqual(code, 1, out)
        self.assertIn("stall: unexpected", out)

    def test_an_unclosed_beat(self):
        t = copy.deepcopy(self.trace)
        i = next(i for i, e in enumerate(t["traceEvents"]) if e["ph"] == "e")
        del t["traceEvents"][i]
        code, out = run(t, self.counters)
        self.assertEqual(code, 1, out)
        self.assertIn("never closed", out)

    def clock_events(self, t, key):
        return [e for e in t["traceEvents"] if e["ph"] == "C" and key in e.get("args", {})]

    def clock_tids(self, t):
        return {e["args"]["name"][len("clock "):]: e["tid"] for e in t["traceEvents"]
                if e["ph"] == "M" and e["name"] == "thread_name" and e["args"]["name"].startswith("clock ")}

    def test_the_demo_has_its_clocks(self):
        # Every domain's own cycle numbers and, as the demo turns them on, its square wave (#536).
        tids = self.clock_tids(self.trace)
        for d in self.counters["domains"]:
            self.assertIn(d["name"], tids)
            for key in ("cycle", "clock"):
                self.assertTrue([e for e in self.clock_events(self.trace, key) if e["tid"] == tids[d["name"]]],
                                (d["name"], key))

    def test_a_domain_without_its_clock_samples(self):
        # The demo check above, on a trace that lost one domain's samples (#553's review).
        t = copy.deepcopy(self.trace)
        tid = self.clock_tids(t)[self.counters["domains"][-1]["name"]]
        t["traceEvents"] = [e for e in t["traceEvents"] if not (e["ph"] == "C" and e["tid"] == tid)]
        saved, self.trace = self.trace, t
        try:
            with self.assertRaises(AssertionError):
                self.test_the_demo_has_its_clocks()
        finally:
            self.trace = saved

    def test_booleans_are_not_cycle_numbers_or_levels(self):
        for key in ("cycle", "clock"):
            t = copy.deepcopy(self.trace)
            for e in self.clock_events(t, key):
                if e["args"][key] in (0, 1):
                    e["args"][key] = bool(e["args"][key])
            code, out = run(t, self.counters)
            self.assertEqual(code, 1, (key, out))
            self.assertIn("counter values must be numbers", out)

    def test_a_skipped_cycle_number(self):
        t = copy.deepcopy(self.trace)
        cs = self.clock_events(t, "cycle")
        t["traceEvents"].remove(cs[3])
        code, out = run(t, self.counters)
        self.assertEqual(code, 1, out)
        self.assertIn("do not run 0, 1, 2", out)

    def test_a_clock_stuck_high(self):
        t = copy.deepcopy(self.trace)
        ws = self.clock_events(t, "clock")
        ws[1]["args"]["clock"] = 1
        code, out = run(t, self.counters)
        self.assertEqual(code, 1, out)
        self.assertIn("does not alternate", out)

    def test_a_misplaced_rising_edge(self):
        t = copy.deepcopy(self.trace)
        ws = self.clock_events(t, "clock")
        ws[2]["ts"] += ws[1]["ts"] - ws[0]["ts"] - 0.0001   # still before the next fall
        code, out = run(t, self.counters)
        self.assertEqual(code, 1, out)
        self.assertIn("not at its cycle's time", out)

    def test_cycles_disagree_with_the_counters(self):
        c = copy.deepcopy(self.counters)
        c["domains"][0]["cycles"] += 1
        code, out = run(self.trace, c)
        self.assertEqual(code, 1, out)
        self.assertIn("the counters say", out)


if __name__ == "__main__":
    TRACE, COUNTERS = sys.argv[1], sys.argv[2]
    unittest.main(argv=[sys.argv[0], "-v"])
