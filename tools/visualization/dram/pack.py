#!/usr/bin/env python3
"""
Pack a .mflow bundle into ONE self-contained HTML file: a DRAM view with the run embedded, to
share, archive or attach to a report (docs/plans/cycle-mflow-and-debug-server.md step 9).

Usage:
    python3 pack.py <view> <bundle.mflow> -o view.html      view: one of VIEWS (timing, ...)

The output needs no server and no other file: the shared scripts (../common/mflow.js, dram.js)
are inlined, through ../common/pack_mflow.py. A bundle over ~12 MB is refused; view it from its
folder instead.

Exit codes: 0 written; 2 the bundle could not be read, or the view is unknown.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from pack_mflow import PackError, pack  # noqa: E402

VIEWS = {"timing": "timing.html"}


def main():
    ap = argparse.ArgumentParser(description="Embed a .mflow bundle in a DRAM view.")
    ap.add_argument("view", choices=sorted(VIEWS))
    ap.add_argument("bundle")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    try:
        m, page = pack(HERE / VIEWS[a.view], a.bundle)
        Path(a.output).write_text(page, encoding="utf-8")
    except (PackError, OSError) as e:
        print(f"pack: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"pack: {a.output} ({a.view}, {len(page) // 1024} KiB, run {m['device']}, "
          f"{m['tables']['commands']['rows']} commands)")


if __name__ == "__main__":
    main()
