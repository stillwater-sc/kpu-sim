"""
The debug server's core (docs/plans/cycle-mflow-and-debug-server.md §3.6, step 7): finding runs,
confining paths, and knowing which viewer reads what. Plain functions with no HTTP in them, so the
full server (docs/plans/kpu-debug-server-full.md) reuses them unchanged.

    confine(base, rel)      the file `rel` names under `base`, or None if it is outside it
    discover_runs(root)     every run under `root`: run directories (their run.json) and bare
                            artifacts (an .mflow or .tflow bundle, a trace .json)
    VIEWERS                 the viewers, where they live, and what each reads

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import datetime
import json
import os
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[3]

# The viewers the server hosts under /view/<name>/. `reads` is the artifact kind a viewer opens
# from a URL, and `param` the query parameter it takes it in; a viewer with none is served as a
# page that opens its own files.
VIEWERS = {
    "memflow": {"dir": REPO / "tools" / "visualization" / "memflow", "page": "index.html",
                "title": "Memory-side debugger", "reads": "mflow", "param": "bundle"},
    "tileflow": {"dir": REPO / "tools" / "visualization" / "tileflow", "page": "index.html",
                 "title": "Tile-flow viewer", "reads": None, "param": None},
    "occupancy": {"dir": REPO / "tools" / "visualization" / "occupancy", "page": "index.html",
                  "title": "Occupancy viewer", "reads": None, "param": None},
    "floorplan": {"dir": REPO / "tools" / "visualization" / "floorplan", "page": "index.html",
                  "title": "Floorplan viewer", "reads": None, "param": None},
    "domainflow": {"dir": REPO / "tools" / "visualization" / "domainflow", "page": "index.html",
                   "title": "Domain-flow viewer", "reads": None, "param": None},
    # The LPDDR5 views on the Chrome DRAM trace, until the .mflow views replace them (step 9).
    "lpddr5": {"dir": REPO / "traces" / "memory" / "lpddr5" / "tools", "page": "bank_analyzer.html",
               "title": "LPDDR5 bank analyzer (Chrome DRAM trace)", "reads": "dram-trace", "param": "trace"},
}
# Shared scripts the viewers load by relative path (../common/mflow.js): served beside them.
SHARED = {"common": REPO / "tools" / "visualization" / "common"}

MAX_DEPTH = 6   # how deep discover_runs looks under the root


def confine(base, rel):
    """The existing file `rel` names under directory `base`, resolved (symlinks too), or None when
    it would leave `base`, is not a file, or cannot be resolved. `rel` is a URL path, already
    percent-decoded: absolute paths, drive letters and NUL bytes are refused outright."""
    if not isinstance(rel, str) or "\0" in rel or rel.startswith(("/", "\\")) or ":" in rel.split("/")[0]:
        return None
    try:
        root = Path(base).resolve(strict=True)
        p = (root / rel).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if p != root and root not in p.parents:
        return None
    return p if p.is_file() else None


def inside(base, p):
    """Whether path `p` (a file or a directory) exists and resolves inside directory `base`."""
    try:
        root = Path(base).resolve(strict=True)
        q = Path(p).resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    return q == root or root in q.parents


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _rel(root, p):
    return Path(os.path.relpath(p, root)).as_posix()


def _bundle_format(d):
    m = _json(d / "manifest.json")
    return m.get("format") if isinstance(m, dict) else None


def _trace_kind(path):
    """A .json that is a Chrome or Perfetto trace: "dram-trace" for the LPDDR5 controller's (its
    events name LPDDR5 banks), "trace" for any other; None if it is not a trace."""
    try:
        if path.stat().st_size > 256 * 1024 * 1024:
            return None
        head = path.read_text(encoding="utf-8", errors="replace")[:65536]
    except OSError:
        return None
    if '"traceEvents"' not in head and not head.lstrip().startswith("["):
        return None
    if '"ph"' not in head:
        return None
    return "dram-trace" if "LPDDR5" in head else "trace"


def links(kind, url):
    """The viewers that open an artifact of `kind` served at `url`: [(viewer, href)]."""
    out = []
    for name, v in VIEWERS.items():
        if v["reads"] == kind and v["param"]:
            out.append((name, f"/view/{name}/{v['page']}?{v['param']}={quote(url, safe='/')}"))
    return out


def _artifact(root, p, kind, name=None):
    rel = _rel(root, p)
    url = "/files/" + quote(rel, safe="/") + ("/" if p.is_dir() else "")
    return {"name": name or p.name, "kind": kind, "path": rel, "url": url,
            "views": [{"viewer": v, "href": h} for v, h in links(kind, url)]}


def discover_runs(root):
    """Every run under `root`, newest first. A run directory is one with a kpu-run run.json: its
    artifacts are the ones the manifest names, and nothing under it is listed again. Anything
    else found is a run of its own artifact: an .mflow or .tflow bundle (a directory with its
    manifest), or a trace .json. Symlinks are not followed."""
    root = Path(root).resolve()
    runs = []

    def walk(d, depth):
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        names = {e.name for e in entries}
        if "run.json" in names:
            m = _json(Path(d) / "run.json")
            if isinstance(m, dict) and m.get("format") == "kpu-run":
                runs.append(_run(root, Path(d), m))
                return
        for e in entries:
            if e.is_symlink():
                continue
            p = Path(e.path)
            if e.is_dir():
                fmt = _bundle_format(p)
                if fmt in ("kpu-mflow", "kpu-tflow"):
                    runs.append(_bare(root, p, "mflow" if fmt == "kpu-mflow" else "tflow"))
                elif depth < MAX_DEPTH:
                    walk(p, depth + 1)
            elif e.is_file() and e.name.endswith(".json"):
                kind = _trace_kind(p)
                if kind:
                    runs.append(_bare(root, p, kind))

    walk(root, 0)
    runs.sort(key=lambda r: (r["started"] or "", r["id"]), reverse=True)
    return runs


def _run(root, d, m):
    arts = []
    named = m.get("artifacts") if isinstance(m.get("artifacts"), dict) else {}
    for name, f in sorted(named.items()):
        if not isinstance(f, str):
            continue
        p = d / f
        if not inside(root, p):
            continue   # named but missing, or outside the root
        kind = {"mflow": "mflow", "perfetto": "trace", "counters": "counters", "summary": "summary"}.get(name, name)
        arts.append(_artifact(root, p, kind, name))
    return {"id": _rel(root, d), "kind": m.get("kind", "run"), "status": m.get("status"),
            "device": m.get("device"), "started": m.get("started"), "seconds": m.get("seconds"),
            "build": m.get("build"), "git": m.get("git"), "artifacts": arts}


def _bare(root, p, kind):
    try:
        started = p.stat().st_mtime
    except OSError:
        started = 0
    stamp = datetime.datetime.fromtimestamp(started, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"id": _rel(root, p), "kind": kind, "status": None, "device": None, "started": stamp,
            "seconds": None, "build": None, "git": None, "artifacts": [_artifact(root, p, kind)]}
