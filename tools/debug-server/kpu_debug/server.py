"""
The read-only debug server (docs/plans/cycle-mflow-and-debug-server.md §3.6, step 7): the runs
under a root directory and the viewers that read them, served to a browser on loopback, for a
laptop to reach through an SSH tunnel.

ROUTES (GET and HEAD only; the server never writes):
    /                       the runs index: each run, its status, its artifacts and their viewers
    /api/runs               the same, as JSON
    /files/<path>           a file under the root, confined to it; Range requests are honoured
    /view/<viewer>/<path>   a viewer's files (and /view/common/: the scripts the viewers share)

It binds 127.0.0.1 and nothing else: there is no option to bind another address. Authentication
off loopback belongs to the full server (docs/plans/kpu-debug-server-full.md).

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import html
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from . import core

HOST = "127.0.0.1"
CHUNK = 1 << 20
TYPES = {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
         ".json": "application/json", ".html": "text/html; charset=utf-8", ".bin": "application/octet-stream",
         ".css": "text/css; charset=utf-8", ".txt": "text/plain; charset=utf-8", ".svg": "image/svg+xml",
         ".png": "image/png", ".csv": "text/csv; charset=utf-8", ".l0": "text/plain; charset=utf-8",
         ".csp": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8"}


def content_type(path):
    t = TYPES.get(path.suffix.lower())
    if t:
        return t
    guess, _ = mimetypes.guess_type(path.name)
    return guess or "application/octet-stream"


def parse_range(header, size):
    """A single "bytes=a-b" range as (start, end) inclusive, None when there is no usable Range
    header (serve the whole file), or "unsatisfiable"."""
    if not header:
        return None
    m = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", header)
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return None   # multiple ranges or another unit: the whole file, as RFC 9110 allows
    a, b = m.group(1), m.group(2)
    if a == "":   # the last b bytes
        n = int(b)
        if n == 0:
            return "unsatisfiable"
        return (max(0, size - n), size - 1)
    start = int(a)
    end = size - 1 if b == "" else min(int(b), size - 1)
    if start >= size or end < start:
        return "unsatisfiable"
    return (start, end)


class Handler(BaseHTTPRequestHandler):
    server_version = "kpu-debug-server/1"
    root = None   # set by make_server

    # -------------------------------------------------------------- dispatch
    def do_GET(self):
        self.route(head=False)

    def do_HEAD(self):
        self.route(head=True)

    def refuse_method(self):
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = refuse_method

    def route(self, head):
        path = unquote(urlsplit(self.path).path)
        if path in ("/", "/index.html"):
            return self.send_bytes(index_html(self.root).encode("utf-8"), "text/html; charset=utf-8", head, cache=False)
        if path == "/api/runs":
            body = json.dumps({"root": str(self.root), "runs": core.discover_runs(self.root)}, indent=1)
            return self.send_bytes(body.encode("utf-8"), "application/json", head, cache=False)
        if path.startswith("/files/"):
            return self.send_file(core.confine(self.root, path[len("/files/"):]), head)
        if path.startswith("/view/"):
            name, _, rest = path[len("/view/"):].partition("/")
            base = core.VIEWERS[name]["dir"] if name in core.VIEWERS else core.SHARED.get(name)
            if base is None:
                return self.not_found()
            if rest == "" and name in core.VIEWERS:
                return self.redirect(f"/view/{name}/{core.VIEWERS[name]['page']}")
            return self.send_file(core.confine(base, rest), head)
        return self.not_found()

    # -------------------------------------------------------------- responses
    def common_headers(self, ctype, cache):
        self.send_header("Content-Type", ctype)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-cache" if cache else "no-store")

    def send_bytes(self, body, ctype, head, cache=True):
        self.send_response(HTTPStatus.OK)
        self.common_headers(ctype, cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def send_file(self, p, head):
        if p is None:
            return self.not_found()
        try:
            size = p.stat().st_size
            f = open(p, "rb")
        except OSError:
            return self.not_found()
        with f:
            rng = parse_range(self.headers.get("Range"), size)
            if rng == "unsatisfiable":
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = rng if rng else (0, size - 1)
            length = max(0, end - start + 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT if rng else HTTPStatus.OK)
            self.common_headers(content_type(p), True)
            self.send_header("Accept-Ranges", "bytes")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            if head:
                return
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def redirect(self, where):
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", where)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def not_found(self):
        body = b"not found\n"
        self.send_response(HTTPStatus.NOT_FOUND)
        self.common_headers("text/plain; charset=utf-8", False)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("kpu-debug-server: %s %s\n" % (self.address_string(), fmt % args))


def make_server(root, port=8080):
    """The server for `root`, bound to 127.0.0.1:`port` (0: any free port). Call serve_forever()."""
    root = core.Path(root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    handler = type("BoundHandler", (Handler,), {"root": root})
    return ThreadingHTTPServer((HOST, port), handler)


# ------------------------------------------------------------------ the index page
STYLE = """
:root { --bg: #f4f5f6; --panel: #fcfcfb; --ink: #15181c; --ink-2: #4a5059; --line: #d9dce0;
  --ok: #2f9e6e; --bad: #d03b3b; --run: #2a78d6; --link: #2a5db0;
  --font-ui: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; --font-data: ui-monospace, Menlo, Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121416; --panel: #1a1c1f; --ink: #eef0f2; --ink-2: #b9bfc7; --line: #2e3237;
  --ok: #3fbf86; --bad: #ef6b6b; --run: #3987e5; --link: #7aa7ea; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #121416; --panel: #1a1c1f; --ink: #eef0f2; --ink-2: #b9bfc7; --line: #2e3237;
  --ok: #3fbf86; --bad: #ef6b6b; --run: #3987e5; --link: #7aa7ea; color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--ink); font: 14px/1.45 var(--font-ui); }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 20px 0 8px; }
p.sub { color: var(--ink-2); margin: 0 0 12px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; font-size: 12px; }
td.id, td.when { font-family: var(--font-data); font-size: 12px; }
a { color: var(--link); text-decoration: none; } a:hover { text-decoration: underline; }
.st { font-family: var(--font-data); font-size: 12px; }
.st.ok { color: var(--ok); } .st.failed, .st.error { color: var(--bad); } .st.running { color: var(--run); }
.art { white-space: nowrap; } .art + .art::before { content: " · "; color: var(--ink-2); }
.empty { color: var(--ink-2); }
"""


def index_html(root):
    e = html.escape
    rows = []
    for r in core.discover_runs(root):
        arts = []
        for a in r["artifacts"]:
            views = "".join(f' <a href="{e(v["href"])}">[{e(v["viewer"])}]</a>' for v in a["views"])
            # A bundle is a directory, served file by file: its name links to its manifest.
            href = a["url"] + "manifest.json" if a["url"].endswith("/") else a["url"]
            arts.append(f'<span class="art"><a href="{e(href)}">{e(a["name"])}</a>{views}</span>')
        status = r["status"] or ""
        rows.append(f'<tr><td class="id">{e(r["id"])}</td><td>{e(r["kind"] or "")}</td>'
                    f'<td class="st {e(status)}">{e(status)}</td><td>{e(r["device"] or "")}</td>'
                    f'<td class="when">{e(r["started"] or "")}</td><td>{"".join(arts)}</td></tr>')
    runs = ("<table><thead><tr><th>run</th><th>kind</th><th>status</th><th>device</th><th>started (UTC)</th>"
            "<th>artifacts and their viewers</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            if rows else '<p class="empty">No runs under this root yet. Write one with '
                         '<code>kpu-run ... --memory-subsystem --run-dir &lt;root&gt;/&lt;name&gt;</code>.</p>')
    viewers = "".join(f'<li><a href="/view/{e(n)}/{e(v["page"])}">{e(v["title"])}</a></li>'
                      for n, v in core.VIEWERS.items())
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>KPU Debug Runs</title><style>{STYLE}</style></head><body>'
            f'<h1>KPU debug runs</h1><p class="sub">Runs under <code>{e(str(root))}</code>, newest first. '
            f'Read-only; <a href="/api/runs">as JSON</a>.</p>'
            f'<div class="panel">{runs}</div><h2>Viewers</h2><div class="panel"><ul>{viewers}</ul></div>'
            f'</body></html>')
