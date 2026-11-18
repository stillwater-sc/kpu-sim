#!/usr/bin/env python3
"""
Pack a timed .dflow record, and optionally a second one to overlay, into ONE self-contained HTML
file: the resource-occupancy viewer with the runs embedded (#390).

Usage:
    python3 pack.py <a.dflow> [--overlay <b.dflow>] -o view.html

The output needs no server and no other file. Binary columns are embedded as base64. A record
over ~12 MB of columns is refused; view it from its folder instead.

Exit codes: 0 written; 2 a record could not be read, or is not timed.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "<!--OCC_EMBED-->"
MAX_BYTES = 12 * 1024 * 1024


def bundle(path):
    d = Path(path)
    manifest = (d / "manifest.json").read_text(encoding="utf-8")
    m = json.loads(manifest)
    if m.get("format") != "kpu-dflow":
        raise ValueError(f"{path}: not a kpu-dflow bundle")
    if m.get("version") != 6 or not m.get("time"):
        raise ValueError(f"{path}: a {m.get('level', '?')} record with no time; occupancy needs a timed "
                         "level's record (kpu-run --dflow-lt1 or --dflow-lca)")
    # The size first, from the files' sizes, before reading or encoding any of them.
    files = [d / t["file"] for t in m["tables"].values()]
    total = sum(f.stat().st_size for f in files)
    if total > MAX_BYTES:
        raise ValueError(f"{path}: the columns are {total} bytes; view it from its folder instead")
    bins = {f.name: base64.b64encode(f.read_bytes()).decode("ascii") for f in files}
    return {"manifest": manifest, "bins": bins}, m


def main():
    ap = argparse.ArgumentParser(description="Embed timed .dflow records in the resource-occupancy viewer.")
    ap.add_argument("bundle")
    ap.add_argument("--overlay", help="a second record of the same program, joined event by event")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    try:
        ea, ma = bundle(a.bundle)
        eb, mb = bundle(a.overlay) if a.overlay else (None, None)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"pack: {e}", file=sys.stderr)
        sys.exit(2)
    page = (HERE / "index.html").read_text(encoding="utf-8")
    if MARKER not in page:
        print("pack: index.html has no embed marker", file=sys.stderr)
        sys.exit(2)
    # Every "<" escaped (a JSON string's \u003c decodes to it): no "</script>" and no "<!--" can
    # end or bend the script element the payload sits in.
    payload = json.dumps({"a": ea, "b": eb}).replace("<", "\\u003c")
    page = page.replace(MARKER, f"<script>window.OCC_EMBED = {payload};</script>", 1)
    Path(a.output).write_text(page, encoding="utf-8")
    print(f"pack: {a.output} ({len(page) // 1024} KiB, program {ma['program']['name']}, {ma['level']}"
          + (f", overlaid with {mb['level']}" if mb else "") + ")")


if __name__ == "__main__":
    main()
