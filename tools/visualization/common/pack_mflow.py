"""
Pack a .mflow bundle into ONE self-contained HTML page: a memory view with the run embedded, to
share, archive or attach to a report. The memory-side debugger (memflow/pack.py) and the DRAM
views (dram/pack.py) both pack through here.

    pack(page, bundle) -> (manifest, html)   the page with the bundle and its shared scripts inlined

The page must hold the marker <!--MFLOW_EMBED-->, where the bundle goes, and load the shared
scripts with <script src="../common/<name>.js"></script>; each is inlined in place of its tag, so
the page needs no other file. The debug server's reload bar (live.js) is left out: a packed page
is a file, with no server to watch. Binary columns are embedded as base64, so the page is about
4/3 the size of the bundle; a bundle over MAX_BYTES is refused rather than producing a page
browsers struggle with -- view it from its folder instead.

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import base64
import json
import re
from pathlib import Path

COMMON = Path(__file__).resolve().parent
MARKER = "<!--MFLOW_EMBED-->"
SHARED_TAG = re.compile(r'<script src="\.\./common/([A-Za-z0-9_-]+\.js)"></script>\n?')
DROPPED = {"live.js"}
MAX_BYTES = 12 * 1024 * 1024


class PackError(Exception):
    """The bundle or the page cannot be packed; the message says why."""


def read_bundle(d):
    """The bundle at directory d as (manifest dict, manifest text, lod text or None, {file: base64})."""
    d = Path(d)
    try:
        manifest = (d / "manifest.json").read_text(encoding="utf-8")
        m = json.loads(manifest)
        if not isinstance(m, dict) or m.get("format") != "kpu-mflow":
            raise PackError("not a kpu-mflow bundle")
        lod = (d / "lod.json").read_text(encoding="utf-8") if (d / "lod.json").exists() else None
        files = [t["file"] for t in m["tables"].values()]
        if lod:
            files.append(json.loads(lod)["file"])
        bins, total = {}, 0
        for f in files:
            data = (d / f).read_bytes()
            total += len(data)
            bins[f] = base64.b64encode(data).decode("ascii")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise PackError(str(e)) from e
    if total > MAX_BYTES:
        raise PackError(f"the bundle's columns are {total} bytes; view it from its folder instead of embedding it")
    return m, manifest, lod, bins


def pack(page_path, bundle):
    """(the bundle's manifest, the page at page_path with the bundle at directory `bundle`
    embedded and its shared scripts inlined). Raises PackError."""
    m, manifest, lod, bins = read_bundle(bundle)
    page_path = Path(page_path)
    page = page_path.read_text(encoding="utf-8")
    if MARKER not in page:
        raise PackError(f"{page_path.name} has no embed marker")
    if not SHARED_TAG.search(page):
        raise PackError(f"{page_path.name} does not load ../common/mflow.js")

    def inline(x):
        name = x.group(1)
        if name in DROPPED:
            return ""
        src = (COMMON / name).read_text(encoding="utf-8")
        if "</script" in src.lower():
            raise PackError(f"{COMMON / name} holds '</script', which would end its inlined <script> early")
        return f"<script>\n{src}</script>\n"

    try:
        page = SHARED_TAG.sub(inline, page)
    except OSError as e:
        raise PackError(str(e)) from e
    payload = json.dumps({"manifest": manifest, "lod": lod, "bins": bins})
    # "</" inside a <script> would end it early; JSON allows the escaped form.
    payload = payload.replace("</", "<\\/")
    return m, page.replace(MARKER, f"<script>window.MFLOW_EMBED = {payload};</script>", 1)
