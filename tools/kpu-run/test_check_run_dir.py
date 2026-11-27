#!/usr/bin/env python3
"""
Self-test for check_run_dir.py (docs/plans/cycle-mflow-and-debug-server.md step 3): the clean run
directory passes, and each way a directory can disagree with its run.json fails; a malformed
run.json or heatmap.json is reported, never a traceback (#547's review).

Usage: python3 test_check_run_dir.py <clean-memory-run-dir> [<clean-compute-run-dir>]

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
COMPUTE_DIR = None


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
        self.assertFails("summary.txt is named but is not there as a file")

    def test_a_file_artifact_that_is_a_directory_fails(self):
        for name in ("summary.txt", "counters.json", "trace.json"):
            with self.subTest(artifact=name):
                (self.d / name).unlink()
                (self.d / name).mkdir()
                self.assertFails(f"{name} is named but is not there as a file")
                (self.d / name).rmdir()
                (self.d / name).write_text("{}")

    def test_ok_after_a_deadlock_or_a_violation_fails(self):
        for flag in ("deadlock", "violation"):
            with self.subTest(flag=flag):
                self.m["result"][flag] = True
                self.save()
                self.assertFails("status ok, but the result says")
                self.m["result"][flag] = False

    def test_a_manifest_without_its_commit_fails(self):
        for bad in (None, "4dfebf4"):
            with self.subTest(git=bad):
                if bad is None:
                    self.m.pop("git", None)
                else:
                    self.m["git"] = bad
                self.save()
                self.assertFails("git is")

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

    def test_a_kind_of_the_wrong_type_is_reported(self):
        # An unhashable kind ([] or {}) is a diagnostic, not a TypeError from the artifact lookup.
        for bad in ([], {}, None, "cache-subsystem"):
            with self.subTest(kind=bad):
                self.m["kind"] = bad
                self.save()
                code, out = run(self.d)
                self.assertEqual(code, 1, out)
                self.assertNotIn("Traceback", out)
                self.assertIn("kind", out)

    def test_no_manifest_is_unreadable(self):
        (self.d / "run.json").unlink()
        code, _ = run(self.d)
        self.assertEqual(code, 2)

    def test_another_format_is_unreadable(self):
        self.m["format"] = "kpu-mflow"
        self.save()
        code, _ = run(self.d)
        self.assertEqual(code, 2)


class CheckComputeRunDirSelfTest(unittest.TestCase):
    def setUp(self):
        # Decided after the arguments are parsed: a unittest option (-v) is not a compute directory.
        if COMPUTE_DIR is None:
            self.skipTest("no compute run directory given")
        self.tmp = Path(tempfile.mkdtemp(prefix="compute-run-dir-selftest-"))
        self.d = self.tmp / "run"
        shutil.copytree(COMPUTE_DIR, self.d)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_clean_directory_passes(self):
        code, out = run(self.d, "--status", "ok")
        self.assertEqual(code, 0, out)

    def test_a_missing_heatmap_fails(self):
        (self.d / "heatmap.svg").unlink()
        code, out = run(self.d)
        self.assertEqual(code, 1, out)
        self.assertIn("heatmap.svg is named but is not there", out)

    def test_a_malformed_heatmap_is_reported(self):
        # Valid JSON of the wrong shape: reported through the checker, never a traceback.
        for bad in ("[]", '{"format": "kpu-array-heatmap", "useful": null, "rows": 4}', "3", "not json"):
            with self.subTest(heatmap=bad):
                (self.d / "heatmap.json").write_text(bad)
                code, out = run(self.d)
                self.assertEqual(code, 1, out)
                self.assertNotIn("Traceback", out)
                self.assertIn("heatmap.json", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    RUN_DIR = Path(sys.argv.pop(1))
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        COMPUTE_DIR = Path(sys.argv.pop(1))
    unittest.main()
