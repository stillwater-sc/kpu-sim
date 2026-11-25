// ============================================================================
// tools/visualization/common/mflow.js
// The .mflow reader every memory view shares (docs/plans/cycle-mflow-and-debug-server.md §3.5,
// step 5): the bundle's manifest and columns, read once and checked once, for the memory-side
// debugger (memflow/index.html) and the DRAM views (dram/, step 9).
//
// A page loads it with a script tag whose src is ../common/mflow.js; a packed page (pack.py) has it
// inlined, so it still needs no other file (this file must therefore never hold a closing script
// tag). Under Node (the views' tests) it is a module.
//
//   Mflow.open(source)   a bundle, from any of three sources, as the record R below:
//                          a FileList or array of Files   (a folder picked or dropped)
//                          { manifest, lod, bins }         (pack.py's window.MFLOW_EMBED)
//                          a URL string                    (a server: "<base>/" holding the bundle;
//                                                           the page's ?bundle=<url>)
//   Mflow.parse(files)   the record from { "manifest.json": text, "<table>.bin": ArrayBuffer, ... }
//
// THE RECORD R: R.m the manifest; R.B bursts, R.C commands, R.Q requests, R.F buffers, R.P ports,
// R.W waits -- each { rows, <column>: TypedArray } as memory_flow_record.cpp writes it. Versions 3
// and 4 are read. A version-3 bundle reads as a version-4 one would: its requests' kinds follow
// is_load, their tickets and slots are NONE, and it has no waits.
//
// Every index the views follow is checked here, once: a bad bundle is refused by name, never
// drawn wrong.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
(function (root) {
"use strict";

const VERSIONS = [3, 4];
const NONE = 0xFFFFFFFF;
// The record's enums (memory_flow_record.hpp), in order.
const CMD = ["ACT", "RD", "WR", "PRE", "REF"];
const OUTCOME = ["unknown", "hit", "empty", "conflict"];
const PORTK = ["offer", "accept", "refuse", "land", "eject_arrive", "eject_wait", "eject_deliver"];
const REQKIND = ["load", "store", "fill"];
const WAIT = ["slot_credit", "l3_credit", "request_credit", "window", "store_buffer", "order"];

function col(buf, cols, name, n) {
  const c = cols.find(x => x.name === name);
  if (!c) throw new Error("no column " + name);
  const T = { u8: Uint8Array, u32: Uint32Array, u64: BigUint64Array, f64: Float64Array }[c.dtype];
  if (!T) throw new Error("column " + name + " has an unknown dtype " + c.dtype);
  if (c.offset % T.BYTES_PER_ELEMENT) throw new Error("column " + name + " is misaligned");
  return new T(buf, c.offset, n);
}

function parse(files) {
  const m = JSON.parse(files["manifest.json"]);
  if (m.format !== "kpu-mflow" || !VERSIONS.includes(m.version))
    throw new Error("not a version-" + VERSIONS.join(" or -") + " kpu-mflow bundle (record it with this build's kpu-memsim or kpu-run)");
  if (m.arbitration !== "round_robin" && m.arbitration !== "fixed") throw new Error("unknown arbitration " + JSON.stringify(m.arbitration));
  const R = { m };
  const tb = name => {
    const t = m.tables[name];
    if (!t) throw new Error("no table " + name);
    const buf = files[t.file];
    if (!buf) throw new Error("missing " + t.file);
    const out = { rows: t.rows };
    for (const c of t.columns) out[c.name] = col(buf, t.columns, c.name, t.rows);
    return out;
  };
  R.B = tb("bursts"); R.C = tb("commands"); R.Q = tb("requests"); R.F = tb("buffers"); R.P = tb("ports");
  if (m.version >= 4) {
    R.W = tb("waits");
    for (const c of ["kind", "ticket", "slot"]) if (!R.Q[c]) throw new Error("no column requests." + c);
  } else {
    // Version 3 states no kind, action or slot, and records no waits.
    const n = R.Q.rows;
    R.Q.kind = Uint8Array.from(R.Q.is_load, l => (l ? 0 : 1));
    R.Q.ticket = new Uint32Array(n).fill(NONE);
    R.Q.slot = new Uint32Array(n).fill(NONE);
    R.W = { rows: 0, engine: new Uint32Array(0), reason: new Uint8Array(0), t_from: new Float64Array(0), t_to: new Float64Array(0) };
  }
  // Every index and enum the views follow is checked once, here.
  for (let i = 0; i < R.C.rows; i++) {
    if (R.C.kind[i] >= CMD.length) throw new Error("command " + i + " has an unknown kind");
    if (R.C.burst[i] !== NONE && R.C.burst[i] >= R.B.rows) throw new Error("command " + i + " names a burst past the table");
  }
  for (let i = 0; i < R.B.rows; i++) {
    if (R.B.outcome[i] >= OUTCOME.length) throw new Error("burst " + i + " has an unknown outcome");
    if (R.B.request[i] !== NONE && R.B.request[i] >= R.Q.rows) throw new Error("burst " + i + " names a request past the table");
  }
  for (let i = 0; i < R.Q.rows; i++) {
    if (R.Q.kind[i] >= REQKIND.length) throw new Error("request " + i + " has an unknown kind");
    if ((R.Q.kind[i] === 1) === Boolean(R.Q.is_load[i])) throw new Error("request " + i + " is a " + REQKIND[R.Q.kind[i]] + " but its is_load says otherwise");
  }
  for (let i = 0; i < R.W.rows; i++) {
    if (R.W.reason[i] >= WAIT.length) throw new Error("wait " + i + " has an unknown reason");
    if (!(R.W.t_to[i] >= R.W.t_from[i])) throw new Error("wait " + i + " ends before it starts");
  }
  return R;
}

// A folder picked or dropped: its manifest, pyramid and columns.
async function readPicked(list) {
  const files = {};
  for (const f of list) {
    if (f.name.endsWith(".bin")) files[f.name] = await f.arrayBuffer();
    else if (f.name === "manifest.json" || f.name === "lod.json") files[f.name] = await f.text();
  }
  return files;
}

function b64(s) {
  const bin = atob(s);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u.buffer;
}

// pack.py's embedding: { manifest, lod, bins: { file: base64 } }.
function fromEmbed(E) {
  const files = { "manifest.json": E.manifest };
  if (E.lod) files["lod.json"] = E.lod;
  for (const [k, v] of Object.entries(E.bins)) files[k] = b64(v);
  return files;
}

// A bundle served at `base` (a directory URL): its manifest, then every table it names and the
// pyramid. File names come from the manifest and must be plain names: never a path out of `base`.
async function fetchFiles(base, fetchFn) {
  const get = fetchFn || (typeof fetch !== "undefined" ? fetch : null);
  if (!get) throw new Error("no fetch to read " + base + " with");
  // File names resolve against the directory's PATH: "<dir>/?v=1" must give "<dir>/manifest.json",
  // not "<dir>/?v=1/manifest.json". A relative base resolves against the page.
  const dir = new URL(base, typeof location !== "undefined" ? location.href : undefined);
  if (!dir.pathname.endsWith("/")) dir.pathname += "/";
  const at = f => new URL(f, dir).href;
  const plain = f => typeof f === "string" && /^[A-Za-z0-9_.-]+$/.test(f) && f !== "." && f !== "..";
  const text = async f => { const r = await get(at(f)); if (!r.ok) throw new Error(at(f) + ": " + r.status); return r.text(); };
  const bytes = async f => { const r = await get(at(f)); if (!r.ok) throw new Error(at(f) + ": " + r.status); return r.arrayBuffer(); };
  const files = { "manifest.json": await text("manifest.json") };
  const m = JSON.parse(files["manifest.json"]);
  for (const t of Object.values(m.tables || {})) {
    if (!plain(t.file)) throw new Error("the manifest names a table file " + JSON.stringify(t.file));
    files[t.file] = await bytes(t.file);
  }
  try {
    files["lod.json"] = await text("lod.json");
    const lod = JSON.parse(files["lod.json"]);
    if (plain(lod.file)) files[lod.file] = await bytes(lod.file);
  } catch (e) { delete files["lod.json"]; }   // the pyramid is optional
  return files;
}

// The record from a picked folder, an embedding, or a server URL.
async function open(source, fetchFn) {
  if (typeof source === "string") return parse(await fetchFiles(source, fetchFn));
  if (source && source.manifest && source.bins) return parse(fromEmbed(source));
  return parse(await readPicked(source));
}

const Mflow = { VERSIONS, NONE, CMD, OUTCOME, PORTK, REQKIND, WAIT, col, parse, readPicked, b64, fromEmbed, fetchFiles, open };
if (typeof module !== "undefined" && module.exports) module.exports = Mflow;
else root.Mflow = Mflow;
})(typeof window !== "undefined" ? window : globalThis);
