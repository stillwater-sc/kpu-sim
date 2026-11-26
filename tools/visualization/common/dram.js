// ============================================================================
// tools/visualization/common/dram.js
// The DRAM side of a .mflow record, as the DRAM views draw it (docs/plans/cycle-mflow-and-debug-
// server.md §3.5, step 9): channels, banks, each bank's state over time, a burst's command chain
// and the timing it met. Everything comes from the record (mflow.js's R); nothing is inferred
// from names or guessed from defaults. Time is the controller's clock tick, the unit of the
// manifest's timing table (dram_timing.params).
//
// A page loads it with a script tag whose src is ../common/dram.js, after ../common/mflow.js; a
// packed page has both inlined (this file must therefore never hold a closing script tag). Under
// Node (the views' tests) it is a module.
//
//   Dram.model(R)            { unit, ticksPerCycle, params, channels, end }:
//                              channels  [{ key, mc, ch, banks, cmds }], sorted by (mc, ch)
//                              banks     [{ key, bg, ba, cmds, states }], sorted by (bg, ba)
//                              cmds      command indices into R.C, by issue tick
//                              states    [{ s, from, to, row }], s one of STATES, covering
//                                        [0, end) with no gaps or overlaps
//   Dram.tick(R, i, col)     command i's tick for column issue | end | data0 | data1
//   Dram.chain(R, burst)     the commands that served `burst`, by issue tick
//   Dram.gaps(R, X, burst)   the timing around its CAS on its bank, as mflow_check.py's M5 checks
//                            it: [{ name, from, to, a, b, measured, required, ok }]
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
(function (root) {
"use strict";

// A bank's state. "activating" runs from ACT to tRCD later (the command's own end), "precharging"
// from PRE to its end, "refreshing" from REF to its end; between, the row is "open" (after an ACT)
// or the bank is "closed".
const STATES = ["closed", "activating", "open", "precharging", "refreshing"];
const ACT = 0, RD = 1, WR = 2, PRE = 3, REF = 4;   // Mflow.CMD's order
const CMD = ["ACT", "RD", "WR", "PRE", "REF"];

// Command i's tick in column col ("issue", "end", "data0", "data1"): the controller's own tick when
// the record has it (k_*), else the KPU-cycle time converted with the manifest's ratio.
function tick(R, i, col) {
  const k = R.C["k_" + col];
  if (k) return Number(k[i]);
  const ratio = R.m.dram_timing && R.m.dram_timing.ticks_per_cycle;
  return Math.round(R.C["t_" + col][i] * (ratio || 1));
}

// The banks a record declares (its dram_bank stations), by (mc, ch, bg, ba). A name is
// ".../mc[i]/ch[j]/bg[g]/ba[b]"; a station named otherwise is not a bank of a channel.
function declaredBanks(m) {
  const out = [];
  for (const s of m.stations || []) {
    if (s.kind !== "dram_bank") continue;
    const x = /mc\[(\d+)\]\/ch\[(\d+)\]\/bg\[(\d+)\]\/ba\[(\d+)\]$/.exec(s.name);
    if (x) out.push(x.slice(1).map(Number));
  }
  return out;
}

// A bank's states from its commands (indices, by issue tick) over [0, end).
function bankStates(R, cmds, end) {
  const out = [];
  let s = "closed", from = 0, row = null;
  const go = (next, at, nextRow) => {
    at = Math.min(Math.max(at, from), end);
    if (at > from) out.push({ s, from, to: at, row });
    s = next; from = at; row = nextRow;
  };
  for (const i of cmds) {
    const k = R.C.kind[i], a = tick(R, i, "issue"), b = tick(R, i, "end");
    if (k === ACT) { go("activating", a, R.C.row[i]); go("open", Math.max(a, b), R.C.row[i]); }
    else if (k === PRE) { go("precharging", a, null); go("closed", Math.max(a, b), null); }
    else if (k === REF) { go("refreshing", a, null); go("closed", Math.max(a, b), null); }
    // A CAS (RD, WR) leaves the bank's state as it is: it is drawn on the bank, not a state.
  }
  go(s, end, row);
  return out;
}

function model(R) {
  const C = R.C;
  const P = (R.m.dram_timing && R.m.dram_timing.params) || {};
  const ticksPerCycle = (R.m.dram_timing && R.m.dram_timing.ticks_per_cycle) || 1;
  const chans = new Map();
  const chan = (mc, ch) => {
    const key = "mc" + mc + "/ch" + ch;
    if (!chans.has(key)) chans.set(key, { key, mc, ch, banks: new Map(), cmds: [] });
    return chans.get(key);
  };
  const bank = (c, bg, ba) => {
    const key = "bg" + bg + "/ba" + ba;
    if (!c.banks.has(key)) c.banks.set(key, { key, bg, ba, cmds: [], states: [] });
    return c.banks.get(key);
  };
  for (const [mc, ch, bg, ba] of declaredBanks(R.m)) bank(chan(mc, ch), bg, ba);
  let end = Math.ceil((R.m.makespan || 0) * ticksPerCycle);
  for (let i = 0; i < C.rows; i++) {
    const c = chan(C.mc[i], C.channel[i]);
    c.cmds.push(i);
    bank(c, C.bank_group[i], C.bank[i]).cmds.push(i);   // a refresh is per bank, and names it
    end = Math.max(end, tick(R, i, "end"), tick(R, i, "data1"));
  }
  const byIssue = (a, b) => tick(R, a, "issue") - tick(R, b, "issue") || a - b;
  const channels = [...chans.values()].sort((a, b) => a.mc - b.mc || a.ch - b.ch).map(c => {
    c.cmds.sort(byIssue);
    const banks = [...c.banks.values()].sort((a, b) => a.bg - b.bg || a.ba - b.ba);
    for (const b of banks) { b.cmds.sort(byIssue); b.states = bankStates(R, b.cmds, end); }
    return { key: c.key, mc: c.mc, ch: c.ch, banks, cmds: c.cmds };
  });
  return { unit: (R.m.dram_timing && R.m.dram_timing.unit) || "tick", ticksPerCycle, params: P, channels, end };
}

// The commands that served burst b, by issue tick (the record names each command's burst).
function chain(R, b) {
  const out = [];
  for (let i = 0; i < R.C.rows; i++) if (R.C.burst[i] === b) out.push(i);
  return out.sort((x, y) => tick(R, x, "issue") - tick(R, y, "issue") || x - y);
}

// The bank (of model X) that burst b is on, or null.
function bankOf(R, X, b) {
  const c = X.channels.find(c => c.mc === R.B.mc[b] && c.ch === R.B.channel[b]);
  return (c && c.banks.find(k => k.bg === R.B.bank_group[b] && k.ba === R.B.bank[b])) || null;
}

// The timing around burst b's CAS, as mflow_check.py's M5 checks it on the bank's own command
// sequence, against the device's table (model X's params). Each is { name, from, to, a, b,
// measured, required, ok }: commands `from` and `to` (kinds) at ticks a and b.
//   the bank's last PRE, last REF and last ACT before the row's ACT   tRP, tRFCpb, tRC
//   the row's ACT to the CAS                    tRCD (for a hit, the ACT was another burst's)
//   the CAS to its data, and the data           tCL (read) or tWL (write); tBurst
//   the ACT to the PRE that closes the row      tRAS
// A pair whose parameter the table lacks, or whose command the record has not, is left out.
function gaps(R, X, b) {
  const P = X.params, C = R.C;
  const cas = chain(R, b).find(i => C.kind[i] === RD || C.kind[i] === WR);
  const bank = bankOf(R, X, b);
  if (cas === undefined || !bank) return [];
  const cmds = bank.cmds, at = cmds.indexOf(cas);
  const back = (from, kinds) => { for (let j = from - 1; j >= 0; j--) if (kinds.includes(C.kind[cmds[j]])) return j; return -1; };
  const out = [];
  const add = (name, i, j, ta, tb) => {
    // A parameter is a tick count: anything else in the manifest is no parameter (and never
    // reaches a page as markup).
    if (typeof P[name] !== "number" || !Number.isFinite(P[name]) || P[name] < 0) return;
    const measured = tb - ta;
    out.push({ name, from: CMD[C.kind[i]], to: j === "data" ? "data" : CMD[C.kind[j]], a: ta, b: tb,
               measured, required: P[name], ok: measured >= P[name] });
  };
  const issue = i => tick(R, i, "issue");
  const ja = back(at, [ACT]);
  if (ja >= 0) {
    const act = cmds[ja];
    // The bank's last PRE and its last REF, each on its own (M5 checks both against the ACT: a
    // REF between a PRE and the ACT does not excuse tRP, nor a PRE after a REF tRFCpb).
    const jp = back(ja, [PRE]), jr = back(ja, [REF]);
    if (jp >= 0) add("tRP", cmds[jp], act, issue(cmds[jp]), issue(act));
    if (jr >= 0) add("tRFCpb", cmds[jr], act, issue(cmds[jr]), issue(act));
    const jc = back(ja, [ACT]);
    if (jc >= 0) add("tRC", cmds[jc], act, issue(cmds[jc]), issue(act));
    add("tRCD", act, cas, issue(act), issue(cas));
  }
  const rd = C.kind[cas] === RD;
  add(rd ? "tCL" : "tWL", cas, "data", issue(cas), tick(R, cas, "data0"));
  add("tBurst", cas, "data", tick(R, cas, "data0"), tick(R, cas, "data1"));
  if (ja >= 0) {
    let jn = -1;
    for (let j = at + 1; j < cmds.length && jn < 0; j++) if (C.kind[cmds[j]] === PRE) jn = j; else if (C.kind[cmds[j]] === ACT) break;
    if (jn >= 0) add("tRAS", cmds[ja], cmds[jn], issue(cmds[ja]), issue(cmds[jn]));
  }
  return out;
}

// The first index in sorted `cmds` whose issue tick is >= t.
function firstAt(R, cmds, t) {
  let lo = 0, hi = cmds.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (tick(R, cmds[mid], "issue") < t) lo = mid + 1; else hi = mid; }
  return lo;
}

const Dram = { STATES, CMD, ACT, RD, WR, PRE, REF, tick, declaredBanks, bankStates, model, chain, bankOf, gaps, firstAt };
if (typeof module !== "undefined" && module.exports) module.exports = Dram;
else root.Dram = Dram;
})(typeof window !== "undefined" ? window : globalThis);
