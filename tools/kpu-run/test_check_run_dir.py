#!/usr/bin/env python3
"""
Self-test for check_run_dir.py (docs/plans/cycle-mflow-and-debug-server.md step 3): the clean run
directory passes, and each way a directory can disagree with its run.json fails.

Usage: python3 test_check_run_dir.py <clean-run-dir>

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
CHECK = HERE / "check_run_dir.py"
RUN_DIR = None


def run(d, *extra):
    p = subprocess.run([sys.executable, str(CHECK), str(d), *extra], capture_output=True, text=True)
    return p.returncode, p.stdout


class CheckRunDirSelfTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="run-dir-selftest-"))
        self.d = self.tmp / "run"
        shutil.copytree(RUN_DIR, self.d)
        self.m = json.loads((self.d / "run.json").read_text())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def save(self):
        (self.d / "run.json").write_text(json.dumps(self.m))

    def assertFails(self, needle):
        code, out = run(self.d)
        self.assertEqual(code, 1, out)
        self.assertIn(needle, out)

    def test_the_clean_directory_passes(self):
        code, out = run(self.d, "--status", "ok")
        self.assertEqual(code, 0, out)

    def test_an_unexpected_status_fails(self):
        code, out = run(self.d, "--status", "failed")
        self.assertEqual(code, 1, out)

    def test_a_missing_artifact_fails(self):
        (self.d / "summary.txt").unlink()
        self.assertFails("summary.txt is named but not there")

    def test_a_bundle_without_its_manifest_fails(self):
        (self.d / "mem.mflow" / "manifest.json").unlink()
        self.assertFails("mem.mflow has no manifest.json")

    def test_a_status_that_contradicts_the_exit_code_fails(self):
        self.m["exit_code"] = 1
        self.save()
        self.assertFails("status ok with exit_code 1")

    def test_ok_with_errors_in_the_result_fails(self):
        self.m["result"]["data_errors"] = 2
        self.save()
        self.assertFails("status ok, but the result says")

    def test_an_unknown_status_fails(self):
        self.m["status"] = "maybe"
        self.save()
        self.assertFails("is not one of")

    def test_a_leftover_temporary_fails(self):
        (self.d / "run.json.tmp").write_text("{}")
        self.assertFails("run.json.tmp was left behind")

    def test_a_running_run_needs_no_exit_code(self):
        # The manifest written before the run: running, no exit code, no artifacts yet.
        for k in ("exit_code", "seconds", "result"):
            self.m.pop(k)
        self.m["status"] = "running"
        self.m["artifacts"] = {}
        self.save()
        code, out = run(self.d, "--status", "running")
        self.assertEqual(code, 0, out)

    def test_no_manifest_is_unreadable(self):
        (self.d / "run.json").unlink()
        code, _ = run(self.d)
        self.assertEqual(code, 2)

    def test_another_format_is_unreadable(self):
        self.m["format"] = "kpu-mflow"
        self.save()
        code, _ = run(self.d)
        self.assertEqual(code, 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    RUN_DIR = Path(sys.argv.pop(1))
    unittest.main()
