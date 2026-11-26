#!/usr/bin/env python3
"""
Pack a .mflow bundle into ONE self-contained HTML file: the memory-side debugger with the run
embedded, to share, archive or attach to a report (docs/plans/memory-side-debugger.md step 5).

Usage:
    python3 pack.py <bundle.mflow> -o view.html

The output needs no server and no other file: the shared scripts (../common/*.js) are inlined,
through ../common/pack_mflow.py. Binary columns are embedded as base64, so the page is about 4/3
the size of the bundle; a bundle over ~12 MB is refused rather than producing a page browsers
struggle with -- view it from its folder instead.

Exit codes: 0 written; 2 the bundle could not be read.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from pack_mflow import PackError, pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Embed a .mflow bundle in the memory-side debugger.")
    ap.add_argument("bundle")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    try:
        m, page = pack(HERE / "index.html", a.bundle)
        Path(a.output).write_text(page, encoding="utf-8")
    except (PackError, OSError) as e:
        print(f"pack: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"pack: {a.output} ({len(page) // 1024} KiB, run {m['device']}, makespan {m['makespan']}, "
          f"{m['tables']['bursts']['rows']} bursts)")


if __name__ == "__main__":
    main()
