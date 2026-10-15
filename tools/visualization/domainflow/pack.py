#!/usr/bin/env python3
"""
Pack a .dflow bundle into ONE self-contained HTML file: the domain-flow viewer with the run
embedded, to share, archive or attach to a report (docs/plans/domain-flow-debugger.md step 3).

Usage:
    python3 pack.py <bundle.dflow> -o view.html

The output needs no server and no other file. Binary columns are embedded as base64, so the page
is about 4/3 the size of the bundle; a bundle over ~12 MB is refused rather than producing a page
browsers struggle with -- view it from its folder instead (cut it with kpu-run --dflow-limit).

Exit codes: 0 written; 2 the bundle could not be read.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "<!--DFLOW_EMBED-->"
MAX_BYTES = 12 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser(description="Embed a .dflow bundle in the domain-flow viewer.")
    ap.add_argument("bundle")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    d = Path(a.bundle)
    try:
        manifest = (d / "manifest.json").read_text(encoding="utf-8")
        m = json.loads(manifest)
        if m.get("format") != "kpu-dflow":
            raise ValueError("not a kpu-dflow bundle")
        bins, total = {}, 0
        for t in m["tables"].values():
            data = (d / t["file"]).read_bytes()
            total += len(data)
            bins[t["file"]] = base64.b64encode(data).decode("ascii")
        if total > MAX_BYTES:
            raise ValueError(f"the bundle's columns are {total} bytes; view it from its folder instead "
                             f"of embedding it, or record fewer steps (kpu-run --dflow-limit)")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"pack: {e}", file=sys.stderr)
        sys.exit(2)

    page = (HERE / "index.html").read_text(encoding="utf-8")
    if MARKER not in page:
        print("pack: index.html has no embed marker", file=sys.stderr)
        sys.exit(2)
    payload = json.dumps({"manifest": manifest, "bins": bins})
    # "</" inside a <script> would end it early; JSON allows the escaped form.
    payload = payload.replace("</", "<\\/")
    page = page.replace(MARKER, f"<script>window.DFLOW_EMBED = {payload};</script>", 1)
    Path(a.output).write_text(page, encoding="utf-8")
    print(f"pack: {a.output} ({len(page) // 1024} KiB, program {m['program']['name']}, "
          f"{m['tables']['steps']['rows']} of {m['totals']['actions']} steps, L-B)")


if __name__ == "__main__":
    main()
