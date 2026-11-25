#!/usr/bin/env python3
"""
Self-test for mflow_check.py's M7 and M8 (docs/plans/cycle-mflow-and-debug-server.md step 4), on a
kpu-run --run-dir directory: a cycle-core bundle, with its waits table, beside the run.json and
counters it is checked against. Each check must fail on a directory that breaks it, and the clean
one must pass. (test_mflow_check.py covers the rest, on a kpu-memsim bundle, which has no waits
and no run.)

Usage: python3 test_mflow_check_run.py <clean-run-dir>

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE / "mflow_check.py"
sys.path.insert(0, str(HERE))
from test_mflow_check import Bundle  # noqa: E402  (the column editor)

RUN_DIR = None


def run(bundle, *extra):
    p = subprocess.run([sys.executable, str(CHECK), str(bundle), "--json", *extra], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode in (0, 1) else None)


class RunBundle(Bundle):
    """A writable copy of the whole run directory; the bundle is its mem.mflow."""

    def __init__(self, src):
        self.root = Path(tempfile.mkdtemp(prefix="mflow-run-selftest-")) / "run"
        shutil.copytree(src, self.root)
        self.dir = self.root / "mem.mflow"
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.run = json.loads((self.root / "run.json").read_text())

    def save_run(self):
        (self.root / "run.json").write_text(json.dumps(self.run))

    def close(self):
        shutil.rmtree(self.root.parent, ignore_errors=True)


class MflowCheckRunSelfTest(unittest.TestCase):
    def setUp(self):
        self.b = RunBundle(RUN_DIR)

    def tearDown(self):
        self.b.close()

    def assertFails(self, invariant, needle):
        code, report = run(self.b.dir)
        self.assertEqual(code, 1, f"expected {invariant} to fail, checker exited {code}")
        hits = [v["message"] for v in report["violations"] if v["invariant"] == invariant]
        self.assertTrue(any(needle in h for h in hits), f"no {invariant} message names {needle!r}: {hits[:3]}")

    def test_clean_run_passes_with_m7_and_m8(self):
        code, report = run(self.b.dir)
        self.assertEqual(code, 0, report and report["violations"][:5])
        self.assertIn("M7", report["checked"])
        self.assertIn("M8", report["checked"])

    def test_the_run_has_waits_to_check(self):
        self.assertGreater(self.b.rows("waits"), 1)

    # -- M7 -----------------------------------------------------------------
    def test_m7_two_waits_for_one_cause_overlap(self):
        # The first two intervals of one cause on one engine, the second moved to start inside the first.
        n = self.b.rows("waits")
        seen = {}
        for i in range(n):
            key = (self.b.get("waits", "engine", i), self.b.get("waits", "reason", i))
            if key in seen:
                a = seen[key]
                self.b.set("waits", "t_from", i, self.b.get("waits", "t_from", a))
                self.b.set("waits", "t_to", i, max(self.b.get("waits", "t_to", i), self.b.get("waits", "t_from", a) + 1))
                self.assertFails("M7", "overlap")
                return
            seen[key] = i
        self.fail("no cause waited twice")

    def test_m7_an_empty_wait(self):
        self.b.set("waits", "t_to", 0, self.b.get("waits", "t_from", 0))
        self.assertFails("M7", "not a non-empty interval")

    def test_m7_a_wait_that_starts_before_the_run(self):
        self.b.set("waits", "t_from", 0, -1.0)
        self.assertFails("M7", "not a non-empty interval")

    def test_m7_a_wait_of_an_engine_without_a_station(self):
        self.b.set("waits", "engine", 0, 9)
        self.assertFails("M7", "no dmawait station")

    # -- M8 -----------------------------------------------------------------
    def test_m8_a_burst_the_run_did_not_send(self):
        self.b.run["result"]["dma"]["read_bursts"] += 1
        self.b.save_run()
        self.assertFails("M8", "read bursts")

    def test_m8_wait_cycles_the_record_does_not_have(self):
        self.b.run["result"]["dma"]["waits"]["slot_credit"] += 1
        self.b.save_run()
        self.assertFails("M8", "cycles waited for slot_credit")

    def test_m8_a_page_outcome_off_by_one(self):
        self.b.run["result"]["dram"]["page_hits"] += 1
        self.b.save_run()
        self.assertFails("M8", "page hits")

    def test_m8_the_counters_disagree(self):
        c = json.loads((self.b.root / "counters.json").read_text())
        for row in c["rows"]:
            if row.get("name") == "lpddr5":
                row["C"] += 1
        (self.b.root / "counters.json").write_text(json.dumps(c))
        self.assertFails("M8", "the channel's completions")

    def test_m8_a_dropped_fill(self):
        # The record loses a Fill: the run's count of them says otherwise.
        q = self.b.find("requests", lambda q: self.b.get("requests", "kind", q) == 2)
        self.b.set("requests", "kind", q, 0)   # now counted as a Load
        self.assertFails("M8", "fills")

    def test_m8_named_explicitly(self):
        # --run names the manifest when the bundle is not inside its run directory.
        moved = self.b.root.parent / "elsewhere.mflow"
        shutil.copytree(self.b.dir, moved)
        code, report = run(moved)
        self.assertEqual(code, 0)
        self.assertNotIn("M8", report["checked"])
        code, report = run(moved, "--run", str(self.b.root / "run.json"))
        self.assertEqual(code, 0, report and report["violations"][:5])
        self.assertIn("M8", report["checked"])

    def test_m8_skipped_for_a_run_without_its_result(self):
        self.b.run.pop("result")
        self.b.run["status"] = "running"
        self.b.save_run()
        code, report = run(self.b.dir)
        self.assertEqual(code, 0)
        self.assertNotIn("M8", report["checked"])

    def test_unreadable_counters_are_refused(self):
        # Counters that exist but cannot be read are unreadable input (exit 2), not a crash.
        c = self.b.root / "counters.json"
        c.chmod(0)
        try:
            try:
                c.read_text()
                self.skipTest("this user reads files without permission (root, or no POSIX modes)")
            except OSError:
                pass
            code, _ = run(self.b.dir)
            self.assertEqual(code, 2)
        finally:
            c.chmod(0o644)

    def test_counters_of_the_wrong_shape_are_refused(self):
        for bad in ([1], {"rows": 3}, {"rows": [1]}, {"rows": [{"name": []}]}, {"rows": [{"name": "dma", "C": "7"}]}):
            with self.subTest(counters=bad):
                (self.b.root / "counters.json").write_text(json.dumps(bad))
                code, _ = run(self.b.dir)
                self.assertEqual(code, 2)

    def test_a_run_of_the_wrong_shape_is_refused(self):
        # Anything M8 reads that is not of its type is unreadable input (exit 2), never a crash.
        for path, bad in ((("result", "dma", "waits"), [1, 2]), (("result", "dma", "waits", "window"), "3"),
                          (("result", "dma", "read_bursts"), 1.5), (("result", "dram", "page_hits"), True),
                          (("artifacts",), ["counters.json"])):
            with self.subTest(field=".".join(path), value=bad):
                saved = json.loads(json.dumps(self.b.run))
                d = self.b.run
                for k in path[:-1]:
                    d = d[k]
                d[path[-1]] = bad
                self.b.save_run()
                code, _ = run(self.b.dir, "--run", str(self.b.root / "run.json"))
                self.assertEqual(code, 2)
                self.b.run = saved

    def test_an_unreadable_run_is_refused(self):
        (self.b.root / "run.json").write_text("{not json")
        code, _ = run(self.b.dir, "--run", str(self.b.root / "run.json"))
        self.assertEqual(code, 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    RUN_DIR = Path(sys.argv.pop(1))
    unittest.main()
