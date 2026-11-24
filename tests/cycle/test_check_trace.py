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


if __name__ == "__main__":
    TRACE, COUNTERS = sys.argv[1], sys.argv[2]
    unittest.main(argv=[sys.argv[0], "-v"])
