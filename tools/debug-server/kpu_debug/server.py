"""
The read-only debug server (docs/plans/cycle-mflow-and-debug-server.md §3.6, steps 7 and 8): the
runs under a root directory and the viewers that read them, served to a browser on loopback, for a
laptop to reach through an SSH tunnel.

ROUTES (GET and HEAD only; the server never writes):
    /                       the runs index: each run, its status, its artifacts and their viewers
    /api/runs               the same, as JSON
    /api/events             server-sent events: run-added, run-changed, run-removed, from a poller
                            that rediscovers the runs every second (no inotify, any filesystem)
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
import queue
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from . import core

HOST = "127.0.0.1"
CHUNK = 1 << 20
POLL = 1.0          # seconds between rediscoveries of the root
HEARTBEAT = 15.0    # seconds between keep-alive comments on an idle event stream
TICK = 0.25         # seconds between checks of an event stream for its end
BACKLOG = 256       # events a slow client may fall behind before its stream is ended
WRITE_TIMEOUT = 30.0  # seconds one write to an event stream may block before the client is dropped
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


class Subscriber:
    """One open event stream: the events not yet sent to it, and whether it fell too far behind."""

    def __init__(self):
        self.events = queue.Queue(maxsize=BACKLOG)
        self.dropped = False


class Live:
    """The poller behind /api/events. It rediscovers the runs every `interval` seconds and sends
    each change (core.changes) to every subscriber. A subscriber that falls BACKLOG events behind is
    dropped, and its stream ends; the browser's EventSource reconnects and the page refetches."""

    def __init__(self, root, interval=POLL):
        self.root = root
        self.interval = interval
        self.lock = threading.Lock()
        self.subscribers = set()
        self.stopped = threading.Event()
        self.runs = core.discover_runs(root)
        self.thread = threading.Thread(target=self._loop, name="kpu-debug-live", daemon=True)

    def start(self):
        self.thread.start()

    def poll(self):
        """One rediscovery: send its events, and return them."""
        now = core.discover_runs(self.root)
        events = core.changes(self.runs, now)
        self.runs = now
        for e in events:
            self.publish(e)
        return events

    def _loop(self):
        while not self.stopped.wait(self.interval):
            try:
                self.poll()
            except Exception as e:  # noqa: BLE001 -- the poller must outlive one bad scan
                sys.stderr.write(f"kpu-debug-server: poll failed: {e}\n")

    def subscribe(self):
        s = Subscriber()
        with self.lock:
            self.subscribers.add(s)
        return s

    def unsubscribe(self, s):
        with self.lock:
            self.subscribers.discard(s)

    def publish(self, event):
        with self.lock:
            for s in list(self.subscribers):
                try:
                    s.events.put_nowait(event)
                except queue.Full:
                    s.dropped = True
                    self.subscribers.discard(s)

    def close(self):
        self.stopped.set()


class DebugServer(ThreadingHTTPServer):
    daemon_threads = True
    live = None

    def server_close(self):
        if self.live is not None:
            self.live.close()
        super().server_close()


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
        if path == "/api/events":
            return self.send_events(head)
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

    def send_events(self, head):
        """The event stream: one `event:` per run added, changed or removed, its run as `data:`.
        It ends when the client goes, the server closes, the client falls BACKLOG behind, or one
        write blocks WRITE_TIMEOUT (a client that stopped reading)."""
        live = self.server.live
        self.send_response(HTTPStatus.OK)
        self.common_headers("text/event-stream; charset=utf-8", False)
        self.end_headers()
        if head or live is None:
            return
        # A client that stops reading fills the socket's buffer, and a write would then block its
        # handler for ever: each write gets WRITE_TIMEOUT. Only writes block here, so an idle
        # stream (a keep-alive every HEARTBEAT) is never timed out for its silence.
        self.connection.settimeout(WRITE_TIMEOUT)
        sub = live.subscribe()
        try:
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            sent = time.monotonic()
            while not live.stopped.is_set() and not sub.dropped:
                try:
                    name, run = sub.events.get(timeout=TICK)
                except queue.Empty:
                    if time.monotonic() - sent >= HEARTBEAT:   # a comment: finds a client that went
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        sent = time.monotonic()
                    continue
                data = json.dumps({"id": run["id"], "run": run}, separators=(",", ":"))
                self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
                sent = time.monotonic()
        except OSError:
            pass   # the client went
        finally:
            live.unsubscribe(sub)
            self.close_connection = True

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


def make_server(root, port=8080, poll=POLL):
    """The server for `root`, bound to 127.0.0.1:`port` (0: any free port), its poller started
    (every `poll` seconds). Call serve_forever(); server_close() stops the poller."""
    root = core.Path(root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    handler = type("BoundHandler", (Handler,), {"root": root})
    srv = DebugServer((HOST, port), handler)
    srv.live = Live(root, poll)
    srv.live.start()
    return srv


# ------------------------------------------------------------------ the index page
STYLE = """
:root { --bg: #f4f5f6; --panel: #fcfcfb; --ink: #15181c; --ink-2: #4a5059; --line: #d9dce0;
  --ok: #2f9e6e; --bad: #d03b3b; --run: #2a78d6; --link: #2a5db0; --fresh: #e3eefb;
  --font-ui: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; --font-data: ui-monospace, Menlo, Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121416; --panel: #1a1c1f; --ink: #eef0f2; --ink-2: #b9bfc7; --line: #2e3237;
  --ok: #3fbf86; --bad: #ef6b6b; --run: #3987e5; --link: #7aa7ea; --fresh: #1d2a3b; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #121416; --panel: #1a1c1f; --ink: #eef0f2; --ink-2: #b9bfc7; --line: #2e3237;
  --ok: #3fbf86; --bad: #ef6b6b; --run: #3987e5; --link: #7aa7ea; --fresh: #1d2a3b; color-scheme: dark; }
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
button.perfetto { font: 12px var(--font-ui); background: var(--panel); color: var(--link); border: 1px solid var(--line);
  border-radius: 4px; padding: 0 6px; margin-left: 4px; cursor: pointer; }
button.perfetto:hover { border-color: var(--link); }
tr.fresh td { background: var(--fresh); transition: background 2s; }
#live { color: var(--ink-2); font-size: 12px; }
"""


def index_html(root):
    e = html.escape
    rows = []
    for r in core.discover_runs(root):
        arts = []
        for a in r["artifacts"]:
            views = "".join(f' <a href="{e(v["href"])}">[{e(v["viewer"])}]</a>' for v in a["views"])
            if a["kind"] in PERFETTO_KINDS:   # the browser hands the trace to ui.perfetto.dev (SCRIPT)
                views += (f'<button type="button" class="perfetto" data-url="{e(a["url"])}" '
                          f'data-name="{e(r["id"] + "/" + a["name"])}">Open in Perfetto</button>')
            # A bundle is a directory, served file by file: its name links to its manifest.
            href = a["url"] + "manifest.json" if a["url"].endswith("/") else a["url"]
            arts.append(f'<span class="art"><a href="{e(href)}">{e(a["name"])}</a>{views}</span>')
        status = r["status"] or ""
        rows.append(f'<tr data-id="{e(r["id"])}"><td class="id">{e(r["id"])}</td><td>{e(r["kind"] or "")}</td>'
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
            f'Read-only; <a href="/api/runs">as JSON</a>. <span id="live"></span></p>'
            f'<div class="panel" id="runs">{runs}</div><h2>Viewers</h2><div class="panel"><ul>{viewers}</ul></div>'
            f'<script>{SCRIPT}</script></body></html>')


# The artifact kinds Perfetto opens: the Perfetto export, and a Chrome trace of any kind.
PERFETTO_KINDS = ("trace", "dram-trace")

# The index's script.
#  - Live: /api/events names each run added, changed or removed; the list is refetched from "/"
#    (one renderer, the server's) and the runs named are marked. Each connection refetches too,
#    the first included, since events sent while no stream was open are lost.
#  - Open in Perfetto: ui.perfetto.dev runs in this browser and cannot reach the server, so the
#    page fetches the trace and hands it over with Perfetto's postMessage protocol: PING until the
#    tab answers PONG, then {perfetto: {buffer, title, fileName}}. The trace goes only to the tab
#    the click opened, and only to Perfetto's origin.
SCRIPT = r"""
(function () {
"use strict";
const PERFETTO = "https://ui.perfetto.dev";
const live = document.getElementById("live");
const fresh = new Set();
let timer = 0, asked = 0, shown = 0;

// Refetches may overlap: each is numbered, and a response older than the one shown is dropped.
function refresh() {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    const n = ++asked;
    try {
      const r = await fetch("/", { cache: "no-store" });
      if (!r.ok) return;
      const doc = new DOMParser().parseFromString(await r.text(), "text/html");
      const next = doc.getElementById("runs");
      if (!next || n < shown) return;
      shown = n;
      document.getElementById("runs").replaceChildren(...next.childNodes);
      for (const tr of document.querySelectorAll("#runs tr[data-id]"))
        if (fresh.has(tr.dataset.id)) tr.classList.add("fresh");
      fresh.clear();
    } catch (e) { live.textContent = "could not refresh the list: " + e.message; }
  }, 200);
}

if (window.EventSource) {
  const es = new EventSource("/api/events");
  // Every connection refetches, the first too: events before it (since the page was served) are lost.
  es.onopen = () => { live.textContent = "Live: new runs appear here."; refresh(); };
  es.onerror = () => { live.textContent = "Live updates paused; reconnecting."; };
  for (const name of ["run-added", "run-changed", "run-removed"])
    es.addEventListener(name, ev => {
      try { const d = JSON.parse(ev.data); if (name !== "run-removed") fresh.add(d.id); } catch (e) { return; }
      refresh();
    });
}

function openInPerfetto(url, name) {
  // Open the tab inside the click, or the browser blocks it; fetch the trace meanwhile.
  const win = window.open(PERFETTO + "/#!/?referrer=kpu-debug-server");
  if (!win) { live.textContent = "The browser blocked the Perfetto tab; allow pop-ups for this page."; return; }
  const trace = fetch(url, { cache: "no-store" }).then(r => {
    if (!r.ok) throw new Error(url + ": HTTP " + r.status);
    return r.arrayBuffer();
  });
  let ping = 0, tries = 0;
  const done = () => { clearInterval(ping); window.removeEventListener("message", onMessage); };
  async function onMessage(ev) {
    if (ev.origin !== PERFETTO || ev.source !== win || ev.data !== "PONG") return;
    done();
    try {
      const buffer = await trace;
      win.postMessage({ perfetto: { buffer, title: name, fileName: url.split("/").pop() } }, PERFETTO);
      live.textContent = "Sent " + name + " to Perfetto.";
    } catch (e) { live.textContent = "Could not fetch the trace: " + e.message; }
  }
  window.addEventListener("message", onMessage);
  ping = setInterval(() => {
    if (++tries > 400 || win.closed) { done(); live.textContent = "Perfetto did not answer."; return; }
    win.postMessage("PING", PERFETTO);
  }, 50);
}

document.addEventListener("click", ev => {
  const b = ev.target.closest("button.perfetto");
  if (b) openInPerfetto(b.dataset.url, b.dataset.name);
});
})();
"""
