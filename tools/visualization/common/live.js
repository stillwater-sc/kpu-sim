// tools/visualization/common/live.js -- the "newer version" bar for a view the debug server serves
// (docs/plans/cycle-mflow-and-debug-server.md §3.6, step 8).
//
// A view opened as <page>?bundle=<url> (or ?trace=<url>) calls MflowLive.watch("bundle"). On the
// debug server, /api/events names each run that changed; when a changed run holds the artifact
// this view shows, a bar offers a reload. The view never reloads itself: that would lose the
// cursor and the zoom. Anywhere else (a file:// page, a packed page, a server with no
// /api/events) it does nothing.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
(function (root) {
"use strict";

// Two URLs name the same artifact: same resolved URL, ignoring a trailing "/" (a bundle is a
// directory, served with one).
function sameArtifact(a, b, base) {
  const norm = u => { const x = new URL(u, base); x.hash = ""; return x.href.replace(/\/+$/, ""); };
  try { return norm(a) === norm(b); } catch (e) { return false; }
}

// What the bar says for event `name` on `run` when this view shows the artifact at `href`, or null
// for no bar. A run directory's run.json says "running" while kpu-run rewrites it: its record is
// not there yet, so the bar waits for the run's next version. A run that comes back after it was
// removed (rewritten between two polls) is added, not changed.
function notice(name, run, href, base) {
  if (!concerns(run, href, base)) return null;
  if (name === "run-removed") return "This run was removed from the server.";
  if (run.status === "running") return null;
  return name === "run-added" || name === "run-changed" ? "This run has a newer version." : null;
}

// Whether `run` (an event's run, as /api/runs lists it) holds the artifact at `href`.
function concerns(run, href, base) {
  return !!(run && Array.isArray(run.artifacts) &&
            run.artifacts.some(a => a && typeof a.url === "string" && sameArtifact(a.url, href, base)));
}

function bar(text) {
  let el = document.getElementById("mflow-live-bar");
  if (!el) {
    el = document.createElement("div");
    el.id = "mflow-live-bar";
    el.setAttribute("role", "status");
    el.style.cssText = "position:fixed;left:50%;top:8px;transform:translateX(-50%);z-index:100;display:flex;" +
      "gap:10px;align-items:center;padding:6px 12px;border-radius:6px;font:13px system-ui,sans-serif;" +
      "background:var(--panel,#fcfcfb);color:var(--ink,#15181c);border:1px solid var(--accent,#2a78d6);" +
      "box-shadow:0 2px 8px rgba(0,0,0,.2)";
    const msg = document.createElement("span");
    const reload = document.createElement("button");
    reload.type = "button"; reload.textContent = "Reload";
    reload.onclick = () => location.reload();
    const dismiss = document.createElement("button");
    dismiss.type = "button"; dismiss.textContent = "Dismiss";
    dismiss.onclick = () => { el.hidden = true; };
    el.append(msg, reload, dismiss);
    document.body.appendChild(el);
  }
  el.firstChild.textContent = text;
  el.hidden = false;
  return el;
}

// Watch the artifact named by query parameter `param`. Returns the EventSource, or null when there
// is nothing to watch here.
function watch(param) {
  if (typeof window === "undefined" || !window.EventSource || !/^https?:$/.test(location.protocol)) return null;
  const v = new URLSearchParams(location.search).get(param);
  if (!v) return null;
  let href;
  try { href = new URL(v, location.href); } catch (e) { return null; }
  if (href.origin !== location.origin) return null;
  const es = new EventSource("/api/events");
  for (const name of ["run-added", "run-changed", "run-removed"])
    es.addEventListener(name, ev => {
      let d;
      try { d = JSON.parse(ev.data); } catch (e) { return; }
      const text = notice(name, d && d.run, href.href, location.href);
      if (text) bar(text);
    });
  return es;
}

const api = { watch, notice, concerns, sameArtifact };
if (typeof module !== "undefined" && module.exports) module.exports = api;
else root.MflowLive = api;
})(typeof window !== "undefined" ? window : globalThis);
