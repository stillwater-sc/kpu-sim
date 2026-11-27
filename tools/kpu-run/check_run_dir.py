#!/usr/bin/env python3
"""
Check a kpu-run --run-dir directory (docs/plans/cycle-mflow-and-debug-server.md step 3; a compute
run's, #547): its
run.json says what ran, how it ended and where its artifacts are, and every artifact it names is
there, under the fixed names.

Usage:
    python3 check_run_dir.py <run-dir> [--status ok|failed|error]

Exit codes: 0 the directory is what run.json says; 1 it is not; 2 there is no readable run.json.

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# The artifacts a run directory holds, under its fixed names, by the run's kind: a --memory-subsystem
# run's, and a --compute-subsystem run's (#547).
ARTIFACTS = {
    "memory-subsystem": {"summary": "summary.txt", "counters": "counters.json", "perfetto": "trace.json",
                         "mflow": "mem.mflow"},
    "compute-subsystem": {"summary": "summary.txt", "counters": "counters.json", "perfetto": "trace.json",
                          "heatmap": "heatmap.json", "heatmap_svg": "heatmap.svg"},
}
STATUSES = ("running", "ok", "failed", "error")


def check(d, want_status):
    try:
        m = json.loads((d / "run.json").read_text())
    except (OSError, ValueError) as e:
        print(f"check_run_dir: {d}/run.json: {e}")
        return 2
    bad = []
    if not isinstance(m, dict) or m.get("format") != "kpu-run" or m.get("version") != 1:
        print(f"check_run_dir: {d}/run.json is not a version-1 kpu-run manifest")
        return 2
    for key, kind in (("command", list), ("build", str), ("git", str), ("device", str), ("started", str),
                      ("status", str), ("artifacts", dict), ("kind", str)):
        if not isinstance(m.get(key), kind):
            bad.append(f"{key} is {m.get(key)!r}, not a {kind.__name__}")
    if isinstance(m.get("git"), str) and not (m["git"] == "unavailable" or re.fullmatch(r"[0-9a-f]{40}", m["git"])):
        bad.append(f"git is {m['git']!r}, not a full commit or 'unavailable'")
    if m.get("status") not in STATUSES:
        bad.append(f"status {m.get('status')!r} is not one of {', '.join(STATUSES)}")
    if want_status and m.get("status") != want_status:
        bad.append(f"status is {m.get('status')!r}, expected {want_status!r}")
    if isinstance(m.get("command"), list) and (not m["command"] or m["command"][0] != "kpu-run"):
        bad.append("command does not start with kpu-run")
    if m.get("status") != "running":
        if not isinstance(m.get("exit_code"), int):
            bad.append("a finished run states its exit_code")
        if not isinstance(m.get("seconds"), (int, float)) or m.get("seconds", -1) < 0:
            bad.append("a finished run states its seconds")
        expect = {"ok": (0,), "failed": (1,), "error": (1, 2)}.get(m.get("status"))
        if expect is not None and m.get("exit_code") not in expect:
            bad.append(f"status {m.get('status')} with exit_code {m.get('exit_code')}")
    if m.get("kind") not in ARTIFACTS:
        bad.append(f"kind {m.get('kind')!r} is not one of {', '.join(ARTIFACTS)}")
    if m.get("status") in ("ok", "failed") and m.get("kind") in ARTIFACTS:
        want = ARTIFACTS[m["kind"]]
        arts = m.get("artifacts", {})
        if arts != want:
            bad.append(f"artifacts are {arts}, expected {want}")
        for key, name in want.items():
            there = (d / name).is_dir() if key == "mflow" else (d / name).is_file()
            if not there:
                bad.append(f"{name} is named but is not there as a {'directory' if key == 'mflow' else 'file'}")
        if "mflow" in want and not (d / "mem.mflow" / "manifest.json").is_file():
            bad.append("mem.mflow has no manifest.json")
        if "heatmap" in want and (d / "heatmap.json").is_file():
            try:
                h = json.loads((d / "heatmap.json").read_text())
                if h.get("format") != "kpu-array-heatmap" or len(h.get("useful", [])) != h.get("rows"):
                    bad.append("heatmap.json is not a kpu-array-heatmap with a row per PE row")
            except ValueError as e:
                bad.append(f"heatmap.json: {e}")
        res = m.get("result")
        if not isinstance(res, dict) or not isinstance(res.get("finished"), bool):
            bad.append("a run that ran states its result")
        elif m["status"] == "ok" and (not res["finished"] or res.get("data_errors") or res.get("law_errors")
                                      or res.get("deadlock") or res.get("violation")):
            bad.append(f"status ok, but the result says {res}")
    if (d / "run.json.tmp").exists():
        bad.append("run.json.tmp was left behind")
    for b in bad:
        print(f"  {b}")
    print(f"check_run_dir: {d}: {'FAIL' if bad else 'PASS'} ({m.get('status')}, {m.get('device')})")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--status", choices=STATUSES)
    args = ap.parse_args()
    return check(Path(args.dir), args.status)


if __name__ == "__main__":
    sys.exit(main())
