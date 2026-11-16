#!/usr/bin/env python3
"""
Pack a structure record into ONE self-contained HTML file: the floorplan view with the record
embedded, to share, archive or attach to an issue (#447).

Usage:
    python3 pack.py <record.structure.json> -o view.html
    python3 pack.py <floorplan.json> -o view.html      # kpu-gen-floorplan 1 or kpu-floorplan 1,
                                                       # converted on the way (structure_from_floorplan)

The output needs no server and no other file. Exit codes: 0 written; 2 the record could not be
read, or is not one the viewer draws.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "<!--FLOORPLAN_EMBED-->"
sys.path.insert(0, str(HERE.parent.parent / "trace"))
import structure_from_floorplan  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Embed a structure record in the floorplan view.")
    ap.add_argument("record")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    try:
        rec = json.loads(Path(a.record).read_text(encoding="utf-8"))
        if not isinstance(rec, dict):
            raise ValueError("not a JSON object")
        if rec.get("format") in ("kpu-gen-floorplan", "kpu-floorplan"):
            rec = structure_from_floorplan.convert(rec)
        if rec.get("format") != "kpu-structure" or rec.get("version") != 1:
            raise ValueError(f"format '{rec.get('format')}' version {rec.get('version')}: the viewer draws kpu-structure 1")
    except (OSError, ValueError, KeyError, structure_from_floorplan.ConvertError) as e:
        print(f"pack: {a.record}: {e}", file=sys.stderr)
        return 2
    page = (HERE / "index.html").read_text(encoding="utf-8")
    if MARKER not in page:
        print("pack: index.html has no embed marker", file=sys.stderr)
        return 2
    payload = json.dumps({"record": rec, "name": Path(a.record).name}, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")            # a record's text never closes the script
    page = page.replace(MARKER, "<script>window.FLOORPLAN_EMBED = " + payload + ";</script>", 1)
    Path(a.output).write_text(page, encoding="utf-8")
    print(f"pack: wrote {a.output} ({len(rec['blocks'])} blocks, {len(rec.get('networks', []))} networks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
