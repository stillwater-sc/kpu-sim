#!/usr/bin/env python3
"""
Pack a .tflow bundle (and optionally a floorplan) into ONE self-contained HTML file: the
tile-flow viewer with the run embedded, to share, archive or attach to a report (#286 step 4).

Usage:
    python3 pack.py <bundle.tflow> [--floorplan fp.json] -o view.html

The output needs no server and no other file. Binary columns are embedded as base64, so the
page is about 4/3 the size of the bundle; a bundle over ~12 MB is refused rather than
producing a page browsers struggle with -- view it from its folder instead.

Exit codes: 0 written; 2 the bundle or floorplan could not be read.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "<!--TFLOW_EMBED-->"

sys.path.insert(0, str(HERE))
import bind  # noqa: E402  (bundle_identity, shared with the viewer)
MAX_BYTES = 12 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser(description="Embed a .tflow bundle in the tile-flow viewer.")
    ap.add_argument("bundle")
    ap.add_argument("--floorplan", help="the deployment's floorplan (kpu-floorplan --json)")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    d = Path(a.bundle)
    try:
        manifest = (d / "manifest.json").read_text(encoding="utf-8")
        m = json.loads(manifest)
        if m.get("format") != "kpu-tflow":
            raise ValueError("not a kpu-tflow bundle")
        lod = (d / "lod.json").read_text(encoding="utf-8") if (d / "lod.json").exists() else None
        files = [t["file"] for t in m["tables"].values()]
        if lod:
            files.append(json.loads(lod)["file"])
        bins = {}
        total = 0
        for f in files:
            data = (d / f).read_bytes()
            total += len(data)
            bins[f] = base64.b64encode(data).decode("ascii")
        if total > MAX_BYTES:
            raise ValueError(f"the bundle's columns are {total} bytes; view it from its folder "
                             f"instead of embedding it")
        binding = (d / "binding.json").read_text(encoding="utf-8") if (d / "binding.json").exists() else None
        if binding is not None:
            # A binding left over from an earlier run in this folder indexes the wrong rows.
            # Embedding it would draw wrong places without a word, so it is left out, loudly.
            b = json.loads(binding)
            if b.get("bundle") != bind.bundle_identity(m):
                print("pack: binding.json was derived from a different run; leaving it out -- "
                      "re-run bind.py", file=sys.stderr)
                binding = None
        floorplan = Path(a.floorplan).read_text(encoding="utf-8") if a.floorplan else None
        if floorplan and json.loads(floorplan).get("format") != "kpu-floorplan":
            raise ValueError("--floorplan is not a kpu-floorplan file")
    except (OSError, ValueError, KeyError) as e:
        print(f"pack: {e}", file=sys.stderr)
        sys.exit(2)

    page = (HERE / "index.html").read_text(encoding="utf-8")
    if MARKER not in page:
        print("pack: index.html has no embed marker", file=sys.stderr)
        sys.exit(2)
    payload = json.dumps({"manifest": manifest, "lod": lod, "bins": bins, "floorplan": floorplan,
                          "binding": binding})
    # "</" inside a <script> would end it early; JSON allows the escaped form.
    payload = payload.replace("</", "<\\/")
    page = page.replace(MARKER, f"<script>window.TFLOW_EMBED = {payload};</script>", 1)
    Path(a.output).write_text(page, encoding="utf-8")
    print(f"pack: {a.output} ({len(page) // 1024} KiB, run {m['device']} {m['level']}, "
          f"makespan {m['makespan']}{', with floorplan' if floorplan else ''}"
          f"{', with derived binding' if binding else ''})")


if __name__ == "__main__":
    main()
