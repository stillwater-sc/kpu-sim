#!/usr/bin/env python3
"""
Tests for the DRAM views (docs/plans/cycle-mflow-and-debug-server.md step 9) and their data layer,
../common/dram.js: hand-built records with known states and timing, the pages' pure helpers,
pack.py, the themes, and (given a bundle) the real record: every command on a declared bank,
every bank's states covering the run, and every burst's timing within the device's table, as
mflow_check.py's M5 finds it.

Usage: python3 test_views.py [<bundle.mflow>] [-v]

The data-layer checks run under Node; without Node on PATH they are skipped.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
READER, DRAM = COMMON / "mflow.js", COMMON / "dram.js"
VIEWS = ["timing.html", "bank_analyzer.html"]
BUNDLE = None


def extract(page, signature):
    m = re.search(re.escape(signature) + r".*?\n\}", page, re.S)
    if m is None:
        raise AssertionError("the page has no " + signature)
    return m.group(0)


# A hand-built record, as mflow.js's R holds it (plain arrays stand in for the typed columns).
# One channel's bank bg0/ba0 (row 7): ACT 10, RD 24 (data 38-46), PRE 60, REF 80 (to 220), ACT 230;
# bank bg0/ba1 only refreshes; a second channel (mc0/ch1) has one ACT and a WR.
# Commands are in issue order per controller, as the record keeps them.
HAND = """
const NONE = 0xFFFFFFFF;
const P = { tRCD: 14, tRP: 14, tRAS: 28, tRC: 42, tCL: 14, tWL: 8, tBurst: 8, tRFCpb: 140 };
const cmd = [
  // kind, mc, ch, bg, ba, row, burst, issue, end, data0, data1
  [0, 0, 0, 0, 0, 7, 0,    10,  24,  0,  0],
  [1, 0, 0, 0, 0, 7, 0,    24,  46, 38, 46],
  [3, 0, 0, 0, 0, 7, 0,    60,  74,  0,  0],
  [4, 0, 0, 0, 1, 0, NONE, 70, 210,  0,  0],
  [4, 0, 0, 0, 0, 0, NONE, 80, 220,  0,  0],
  [0, 0, 0, 0, 0, 9, 1,   230, 244,  0,  0],
  [1, 0, 0, 0, 0, 9, 1,   250, 272, 264, 272],
  [0, 0, 1, 2, 3, 4, 2,    12,  26,  0,  0],
  [2, 0, 1, 2, 3, 4, 2,    30,  46, 38, 46],
];
const colOf = k => cmd.map(c => c[k]);
const C = { rows: cmd.length, kind: colOf(0), mc: colOf(1), channel: colOf(2), bank_group: colOf(3), bank: colOf(4),
            row: colOf(5), burst: colOf(6), k_issue: colOf(7), k_end: colOf(8), k_data0: colOf(9), k_data1: colOf(10) };
C.t_issue = C.k_issue.map(t => t / 2); C.t_end = C.k_end.map(t => t / 2);
C.t_data0 = C.k_data0.map(t => t / 2); C.t_data1 = C.k_data1.map(t => t / 2);
const B = { rows: 3, mc: [0, 0, 0], channel: [0, 0, 1], bank_group: [0, 0, 2], bank: [0, 0, 3],
            row: [7, 9, 4], is_load: [1, 1, 0], outcome: [2, 3, 2], request: [NONE, NONE, NONE] };
const stations = ["mc[0]/ch[0]/bg[0]/ba[0]", "mc[0]/ch[0]/bg[0]/ba[1]", "mc[0]/ch[0]/bg[1]/ba[0]",
                  "mc[0]/ch[1]/bg[2]/ba[3]"].map(n => ({ name: "dev/" + n, kind: "dram_bank", capacity: 1 }))
  .concat([{ name: "dev/dma[0]", kind: "dma", capacity: 16 }, { name: "dev/mc[0]/ch[0]/bus", kind: "dram_bus", capacity: 1 }]);
const R = { m: { makespan: 140, stations, dram_timing: { unit: "controller clock tick", ticks_per_cycle: 2, params: P } }, C, B };
"""


class ViewsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dram-views-"))
        self.pages = {v: (HERE / v).read_text(encoding="utf-8") for v in VIEWS}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def node(self, source):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js not on PATH")
        js = self.tmp / "t.js"
        js.write_text(source, encoding="utf-8")
        p = subprocess.run([node, str(js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout)

    def dram(self, body):
        return self.node(f"const Dram = require({json.dumps(str(DRAM))});\n{HAND}\n{body}")

    def read(self, body):
        """`body` run over the real bundle as R (mflow.js) and X (dram.js's model)."""
        if BUNDLE is None:
            self.skipTest("no bundle given")
        return self.node("\n".join([
            "const fs = require('fs');",
            f"const Mflow = require({json.dumps(str(READER))}), Dram = require({json.dumps(str(DRAM))});",
            f"const dir = {json.dumps(str(BUNDLE))};",
            "const files = { 'manifest.json': fs.readFileSync(dir + '/manifest.json', 'utf8') };",
            "for (const t of Object.values(JSON.parse(files['manifest.json']).tables)) {",
            "  const b = fs.readFileSync(dir + '/' + t.file); files[t.file] = b.buffer.slice(b.byteOffset, b.byteOffset + b.length); }",
            "const R = Mflow.parse(files), X = Dram.model(R);",
            body]))

    # -- dram.js on hand-built records ----------------------------------------------
    def test_channels_and_banks_come_from_the_record(self):
        got = self.dram("""
const X = Dram.model(R);
process.stdout.write(JSON.stringify([X.unit, X.ticksPerCycle, X.end,
  X.channels.map(c => [c.key, c.banks.map(b => b.key), c.cmds])]));""")
        unit, per, end, chans = got
        self.assertEqual((unit, per), ("controller clock tick", 2))
        self.assertEqual(end, 280)   # the makespan in ticks (140 cycles x 2) bounds the last command (272)
        # Banks are the declared stations plus any a command names; a bank with no command is kept.
        self.assertEqual(chans, [["mc0/ch0", ["bg0/ba0", "bg0/ba1", "bg1/ba0"], [0, 1, 2, 3, 4, 5, 6]],
                                 ["mc0/ch1", ["bg2/ba3"], [7, 8]]])

    def test_a_banks_states_cover_the_run_from_its_commands(self):
        got = self.dram("""
const X = Dram.model(R), b = X.channels[0].banks;
process.stdout.write(JSON.stringify([b[0].states.map(s => [s.s, s.from, s.to, s.row]),
  b[1].states.map(s => [s.s, s.from, s.to]), b[2].states.map(s => [s.s, s.from, s.to])]));""")
        ba0, ba1, idle = got
        self.assertEqual(ba0, [["closed", 0, 10, None], ["activating", 10, 24, 7], ["open", 24, 60, 7],
                               ["precharging", 60, 74, None], ["closed", 74, 80, None],
                               ["refreshing", 80, 220, None], ["closed", 220, 230, None],
                               ["activating", 230, 244, 9], ["open", 244, 280, 9]])
        self.assertEqual(ba1, [["closed", 0, 70], ["refreshing", 70, 210], ["closed", 210, 280]])
        self.assertEqual(idle, [["closed", 0, 280]])

    def test_ticks_fall_back_to_cycles_when_the_record_has_no_controller_ticks(self):
        got = self.dram("""
const k = Dram.tick(R, 1, "data0");
delete R.C.k_issue; delete R.C.k_end; delete R.C.k_data0; delete R.C.k_data1;
process.stdout.write(JSON.stringify([k, Dram.tick(R, 1, "data0"), Dram.tick(R, 1, "issue")]));""")
        self.assertEqual(got, [38, 38, 24])

    def test_a_bursts_chain_and_timing(self):
        got = self.dram("""
const X = Dram.model(R);
const g = b => Dram.gaps(R, X, b).map(x => [x.name, x.from, x.to, x.a, x.b, x.measured, x.required, x.ok]);
process.stdout.write(JSON.stringify([Dram.chain(R, 0), Dram.chain(R, 1), g(0), g(1), g(2)]));""")
        c0, c1, g0, g1, g2 = got
        self.assertEqual((c0, c1), ([0, 1, 2], [5, 6]))
        # Burst 0: an empty page; its PRE closes the row later (tRAS).
        self.assertEqual(g0, [["tRCD", "ACT", "RD", 10, 24, 14, 14, True], ["tCL", "RD", "data", 24, 38, 14, 14, True],
                              ["tBurst", "RD", "data", 38, 46, 8, 8, True], ["tRAS", "ACT", "PRE", 10, 60, 50, 28, True]])
        # Burst 1: its ACT follows the bank's last PRE (tRP, 60 -> 230) and its last REF (tRFCpb,
        # 150 >= 140), each on its own as M5 checks them, and its previous ACT (tRC); its row is
        # never closed, so no tRAS.
        self.assertEqual(g1, [["tRP", "PRE", "ACT", 60, 230, 170, 14, True], ["tRFCpb", "REF", "ACT", 80, 230, 150, 140, True],
                              ["tRC", "ACT", "ACT", 10, 230, 220, 42, True],
                              ["tRCD", "ACT", "RD", 230, 250, 20, 14, True], ["tCL", "RD", "data", 250, 264, 14, 14, True],
                              ["tBurst", "RD", "data", 264, 272, 8, 8, True]])
        # Burst 2, a write on the other channel: tWL, not tCL.
        self.assertEqual([x[0] for x in g2], ["tRCD", "tWL", "tBurst"])
        self.assertEqual(g2[1][5:], [8, 8, True])

    def test_timing_under_the_table_is_flagged(self):
        got = self.dram("""
R.C.k_issue[1] = 20;   // RD 10 ticks after its ACT: tRCD is 14
const X = Dram.model(R);
process.stdout.write(JSON.stringify(Dram.gaps(R, X, 0).filter(g => !g.ok).map(g => [g.name, g.measured, g.required])));""")
        self.assertEqual(got, [["tRCD", 10, 14]])   # tCL is measured from the RD: 38 - 20 = 18, within

    def test_a_parameter_that_is_not_a_tick_count_is_left_out(self):
        got = self.dram("""
R.m.dram_timing.params.tRCD = "<img src=x onerror=alert(1)>"; R.m.dram_timing.params.tCL = -1;
R.m.dram_timing.params.tBurst = Infinity;
const X = Dram.model(R);
process.stdout.write(JSON.stringify(Dram.gaps(R, X, 0).map(g => g.name)));""")
        self.assertEqual(got, ["tRAS"])

    def test_a_hits_timing_names_the_act_that_opened_its_row(self):
        got = self.dram("""
// burst 3: a hit on row 7, a second RD before the PRE
R.C.rows++; for (const [k, v] of Object.entries({ kind: 1, mc: 0, channel: 0, bank_group: 0, bank: 0, row: 7, burst: 3,
  k_issue: 32, k_end: 54, k_data0: 46, k_data1: 54, t_issue: 16, t_end: 27, t_data0: 23, t_data1: 27 })) R.C[k].splice(2, 0, v);
R.B.rows++; for (const [k, v] of Object.entries({ mc: 0, channel: 0, bank_group: 0, bank: 0, row: 7, is_load: 1, outcome: 1, request: 0xFFFFFFFF })) R.B[k].push(v);
const X = Dram.model(R);
process.stdout.write(JSON.stringify([Dram.chain(R, 3), Dram.gaps(R, X, 3).map(g => [g.name, g.a, g.b])]));""")
        self.assertEqual(got, [[2], [["tRCD", 10, 32], ["tCL", 32, 46], ["tBurst", 46, 54], ["tRAS", 10, 60]]])

    def test_only_bank_stations_of_a_channel_are_banks(self):
        got = self.dram("""process.stdout.write(JSON.stringify(Dram.declaredBanks({ stations: [
  { name: "d/mc[1]/ch[2]/bg[3]/ba[0]", kind: "dram_bank" }, { name: "d/mc[0]/ch[0]/bus", kind: "dram_bus" },
  { name: "d/odd", kind: "dram_bank" }, { name: "d/mc[0]/ch[0]/bg[0]/ba[1]/x", kind: "dram_bank" } ] })));""")
        self.assertEqual(got, [[1, 2, 3, 0]])

    def test_a_banks_row_openings_and_numbers(self):
        got = self.dram("""
const X = Dram.model(R), [ba0, ba1, idle] = X.channels[0].banks;
const ep = b => Dram.episodes(R, b, X.end).map(e => [e.row, e.from, e.to, e.open, e.rd, e.wr, e.close]);
process.stdout.write(JSON.stringify([ep(ba0), ep(ba1), Dram.bankStats(R, ba0, X.end), Dram.bankStats(R, idle, X.end),
  X.channels.map(c => c.banks.map(b => b.bursts))]));""")
        e0, e1, s0, si, bursts = got
        # ACT 10 .. PRE 60 (one RD), then ACT 230 open to the run's end (one RD).
        self.assertEqual(e0, [[7, 10, 60, False, 1, 0, 2], [9, 230, 280, True, 1, 0, None]])
        self.assertEqual(e1, [])   # a bank that only refreshes opens no row
        self.assertEqual(s0["cmds"], [2, 2, 0, 1, 1])
        self.assertEqual((s0["bursts"], s0["reads"], s0["writes"]), (2, 2, 0))
        self.assertEqual(s0["outcomes"], [0, 0, 1, 1])   # as the record states them: empty, conflict
        self.assertEqual(s0["hitRate"], 0)
        self.assertEqual(s0["ticks"], {"closed": 26, "activating": 28, "open": 72, "precharging": 14, "refreshing": 140})
        self.assertEqual((s0["opens"], s0["rows"], s0["meanOpen"]), (2, 2, 50))
        self.assertEqual((si["bursts"], si["hitRate"], si["opens"], si["meanOpen"]), (0, None, 0, None))
        self.assertEqual(bursts, [[[0, 1], [], []], [[2]]])

    # -- the bank analyzer's helpers --------------------------------------------------
    def analyzer(self, *sigs, body):
        page = self.pages["bank_analyzer.html"]
        return self.node("\n".join([f"const Dram = require({json.dumps(str(DRAM))});",
                                    "const { STATES } = Dram, Mflow = { NONE: 0xFFFFFFFF };", HAND,
                                    *[extract(page, s) for s in sigs], body]))

    def test_banks_are_totalled_by_bank_group_and_sorted(self):
        got = self.analyzer("function groups(R, chan, end) {", "const COLUMNS = [", "function sortBanks(banks, key) {", body="""
// a second bank in bank group 0 with a hit, to sort against
R.B.rows++; for (const [k, v] of Object.entries({ mc: 0, channel: 0, bank_group: 0, bank: 1, row: 1, is_load: 0, outcome: 1, request: 0xFFFFFFFF })) R.B[k].push(v);
const X = Dram.model(R), G = groups(R, X.channels[0], X.end);
process.stdout.write(JSON.stringify([G.map(g => [g.group, g.banks.map(b => b.bank.key), g.total.bursts, g.total.outcomes, g.total.hitRate, g.total.ticks.refreshing]),
  sortBanks(G[0].banks, "bursts").map(b => b.bank.key), sortBanks(G[0].banks, "hitRate").map(b => b.bank.key),
  sortBanks(G[0].banks, "bank").map(b => b.bank.key), sortBanks(G[0].banks, "nosuch").map(b => b.bank.key)]));""")
        totals, byBursts, byHit, byBank, unknown = got
        self.assertEqual(totals, [[0, ["bg0/ba0", "bg0/ba1"], 3, [0, 1, 1, 1], 1 / 3, 280], [1, ["bg1/ba0"], 0, [0, 0, 0, 0], None, 0]])
        self.assertEqual(byBursts, ["bg0/ba0", "bg0/ba1"])
        self.assertEqual(byHit, ["bg0/ba1", "bg0/ba0"])
        self.assertEqual(byBank, ["bg0/ba0", "bg0/ba1"])
        self.assertEqual(unknown, ["bg0/ba0", "bg0/ba1"])

    def test_conflict_acts_and_state_shares(self):
        got = self.analyzer("function conflictActs(R, chan) {", "function shares(ticks) {", body="""
const X = Dram.model(R);
process.stdout.write(JSON.stringify([[...conflictActs(R, X.channels[0]).entries()], [...conflictActs(R, X.channels[1]).entries()],
  shares({ closed: 1, activating: 1, open: 2, precharging: 0, refreshing: 0 }), shares({ closed: 0, activating: 0, open: 0, precharging: 0, refreshing: 0 })]));""")
        c0, c1, sh, none = got
        self.assertEqual(c0, [["bg0/ba0", [230]], ["bg0/ba1", []], ["bg1/ba0", []]])   # burst 1 is the conflict
        self.assertEqual(c1, [["bg2/ba3", []]])
        self.assertEqual(sh, [["closed", 0.25], ["activating", 0.25], ["open", 0.5], ["precharging", 0], ["refreshing", 0]])
        self.assertEqual([f for _, f in none], [0, 0, 0, 0, 0])

    def test_the_analyzer_is_operable_from_the_keyboard(self):
        # Every control the mouse uses is focusable, Enter and Space operate it, its state is in
        # ARIA, and a rebuild of the table gives focus back to the same control.
        page = self.pages["bank_analyzer.html"]
        for needle in ('tabindex="0" ` +\n    `aria-sort=', 'tabindex="0" aria-expanded=', 'tabindex="0" aria-selected=',
                       '(e.key === "Enter" || e.key === " ") && operate(e.target)', "if (had) { const el = $(\"banks\").querySelector(had); if (el) el.focus(); }",
                       '<canvas id="cv" height="300" tabindex="0"', 'e.key !== "ArrowUp" && e.key !== "ArrowDown"'):
            with self.subTest(needle=needle):
                self.assertIn(needle, page)

    def test_the_analyzers_link_names_a_bank_and_is_clamped(self):
        got = self.analyzer("function viewFromQuery(search, end) {", body="""
process.stdout.write(JSON.stringify([viewFromQuery("?ch=mc0/ch1&bank=bg2/ba3&t=99&zoom=10", 50), viewFromQuery("?zoom=1&t=", 50)]));""")
        self.assertEqual(got, [{"ch": "mc0/ch1", "bank": "bg2/ba3", "t": 50, "zoom": 10}, {"ch": None, "bank": None, "t": None, "zoom": None}])

    # -- the timing view's helpers ------------------------------------------------------
    def timing(self, *sigs, body):
        page = self.pages["timing.html"]
        return self.node("\n".join([f"const Dram = require({json.dumps(str(DRAM))});",
                                    "const { ACT, RD, WR, PRE, REF } = Dram;", HAND,
                                    *[extract(page, s) for s in sigs], body]))

    def test_a_link_names_a_moment_and_is_clamped_to_the_run(self):
        got = self.timing("function viewFromQuery(search, end) {", body="""
process.stdout.write(JSON.stringify([
  viewFromQuery("?ch=mc0/ch1&burst=12&t=500&zoom=64&cursor=480", 1000), viewFromQuery("?t=-5&zoom=2&burst=1.5&cursor=-3", 1000),
  viewFromQuery("?t=5000&zoom=99999&burst=-1", 1000), viewFromQuery("?t=x&zoom=", 1000), viewFromQuery("", 1000)]));""")
        self.assertEqual(got, [{"cursor": 480, "zoom": 64, "t": 500, "burst": 12, "ch": "mc0/ch1"},
                               {"cursor": 0, "zoom": None, "t": 0, "burst": None, "ch": None},
                               {"cursor": None, "zoom": 1000, "t": 1000, "burst": None, "ch": None},
                               {"cursor": None, "zoom": None, "t": None, "burst": None, "ch": None},
                               {"cursor": None, "zoom": None, "t": None, "burst": None, "ch": None}])

    def test_a_bursts_view_fits_what_is_near_it(self):
        got = self.timing("function burstSpan(gaps) {", body="""
const X = Dram.model(R);
const g = (name, a, b, required) => ({ name, a, b, measured: b - a, required });
process.stdout.write(JSON.stringify([
  burstSpan(Dram.gaps(R, X, 0)),            // ACT 10 .. data 46; tRAS's PRE at 60 is left out
  burstSpan(Dram.gaps(R, X, 1)),            // the REF (80) is near (150 <= 4 x 140); the PRE (60) is not (170 > 4 x 14)
  burstSpan([g("tRP", 0, 30, 14), g("tRCD", 30, 44, 14), g("tCL", 44, 58, 14), g("tBurst", 58, 66, 8)]),
  burstSpan([g("tRP", 0, 100, 14), g("tRCD", 100, 114, 14), g("tCL", 114, 128, 14), g("tBurst", 128, 136, 8)]),
  burstSpan([g("tRCD", 0, 900, 14), g("tCL", 900, 914, 14), g("tBurst", 914, 922, 8)]),   // a hit, opened long ago
  burstSpan([g("tRP", 20, 40, 14), g("tRFCpb", 30, 40, 140), g("tRCD", 40, 54, 14), g("tCL", 54, 68, 14), g("tBurst", 68, 76, 8)]),
  burstSpan([])]));""")
        self.assertEqual(got, [[10, 46], [80, 272], [0, 66], [100, 136], [900, 922], [30, 76], None])

    def test_a_bank_that_only_refreshes_is_idle(self):
        got = self.timing("function busy(R, bank) {", body="""
const X = Dram.model(R);
process.stdout.write(JSON.stringify(X.channels[0].banks.map(b => busy(R, b))));""")
        self.assertEqual(got, [True, False, False])

    def test_the_view_is_shared_by_its_centre_and_a_channel_change_drops_the_burst(self):
        page = self.pages["timing.html"]
        self.assertIn('q.set("t", String(Math.round((viewA + viewB) / 2)));', page)
        self.assertIn('$("chan").addEventListener("change", e => { sel = null; setChannel(', page)
        # the whole run is the view until a burst or a link narrows it
        s = page.index("setView(0, X.end);   // the whole run")
        self.assertLess(s, page.index("setChannel(ci, false);"))

    def test_the_axis_step_is_round(self):
        got = self.timing("function niceStep(perTick, px) {", body="""
process.stdout.write(JSON.stringify([niceStep(1, 70), niceStep(10, 70), niceStep(0.01, 70), niceStep(100, 70)]));""")
        self.assertEqual(got, [100, 10, 10000, 1])

    # -- the pages --------------------------------------------------------------------
    def test_every_colour_is_defined_in_every_theme(self):
        for view, page in self.pages.items():
            with self.subTest(view=view):
                style = page[page.index("<style>"):page.index("</style>")]
                used = set(re.findall(r"var\(--([a-z0-9-]+)\)", page)) | set(re.findall(r'css\("--([a-z0-9-]+)"\)', page))
                used |= set(re.findall(r'"(ink|ink-2|ink-3|line|grid|lane|act|rd|wr|pre|ref|open|crit|sel|panel)"', page))
                blocks = re.findall(r"(?::root\s*\{|:root:not\(\[data-theme=\"light\"\]\)\s*\{|:root\[data-theme=\"dark\"\]\s*\{)(.*?)\}", style, re.S)
                self.assertEqual(len(blocks), 3)
                light = set(re.findall(r"--([a-z0-9-]+):", blocks[0]))
                for name in used:
                    self.assertIn(name, light, f"--{name} is not defined")
                for dark in blocks[1:]:
                    defined = set(re.findall(r"--([a-z0-9-]+):", dark))
                    missing = {n for n in light if not n.startswith("font")} - defined
                    self.assertEqual(missing, set(), "a dark theme lacks colours")

    def test_a_page_reads_a_served_bundle_only_from_its_own_origin(self):
        for view, page in self.pages.items():
            with self.subTest(view=view):
                for script in ("mflow.js", "dram.js", "live.js"):
                    self.assertIn(f'<script src="../common/{script}"></script>', page)
                self.assertIn("url.origin === location.origin", page)
                self.assertIn('MflowLive.watch("bundle")', page)

    def test_record_strings_never_become_markup(self):
        for view, page in self.pages.items():
            with self.subTest(view=view):
                out = self.node("\n".join([extract(page, "const esc = v =>").split("\n")[0], """
process.stdout.write(JSON.stringify([esc('<img src=x onerror=alert(1)>'), esc("a&b\\"'")]));"""]))
                self.assertEqual(out, ["&lt;img src=x onerror=alert(1)&gt;", "a&amp;b&quot;&#39;"])

    def test_the_page_scripts_parse(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js not on PATH")
        for view, page in self.pages.items():
            with self.subTest(view=view):
                script = page[page.rindex("<script>") + len("<script>"):page.rindex("</script>")]
                js = self.tmp / "page.js"
                js.write_text(script, encoding="utf-8")
                p = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)

    # -- pack.py --------------------------------------------------------------------
    def pack(self, *args):
        return subprocess.run([sys.executable, str(HERE / "pack.py"), *args], capture_output=True, text=True)

    def test_pack_embeds_the_bundle_and_every_shared_script(self):
        if BUNDLE is None:
            self.skipTest("no bundle given")
        for view in ("timing", "bank_analyzer"):
            with self.subTest(view=view):
                out = self.tmp / f"{view}.html"
                p = self.pack(view, str(BUNDLE), "-o", str(out))
                self.assertEqual(p.returncode, 0, p.stderr)
                page = out.read_text(encoding="utf-8")
                self.assertNotIn("<script src=", page)   # nothing to fetch: no reader, no reload bar
                self.assertIn("function parse(files)", page)
                self.assertIn("function bankStates(R, cmds, end)", page)
                m = re.search(r"window\.MFLOW_EMBED = (\{.*?\});</script>", page, re.S)
                payload = json.loads(m.group(1).replace("<\\/", "</"))
                self.assertEqual(json.loads(payload["manifest"])["format"], "kpu-mflow")

    def test_pack_refuses_an_unknown_view_and_a_non_bundle(self):
        self.assertEqual(self.pack("nosuch", str(self.tmp), "-o", str(self.tmp / "x.html")).returncode, 2)
        p = self.pack("timing", str(self.tmp), "-o", str(self.tmp / "x.html"))
        self.assertEqual(p.returncode, 2)
        self.assertIn("pack:", p.stderr)

    # -- the real record ----------------------------------------------------------------
    def test_the_real_records_commands_are_all_on_declared_banks(self):
        got = self.read("""
const declared = Dram.declaredBanks(R.m).length;
let banks = 0, cmds = 0;
for (const c of X.channels) { banks += c.banks.length; for (const b of c.banks) cmds += b.cmds.length; }
process.stdout.write(JSON.stringify({ declared, banks, cmds, rows: R.C.rows }));""")
        self.assertGreater(got["declared"], 0)
        self.assertEqual(got["banks"], got["declared"])   # no command names an undeclared bank
        self.assertEqual(got["cmds"], got["rows"])

    def test_the_real_records_bank_states_cover_the_run(self):
        got = self.read("""
let bad = 0, kinds = new Set();
for (const c of X.channels) for (const b of c.banks) {
  let t = 0;
  for (const s of b.states) { if (s.from !== t || !(s.to > s.from)) bad++; t = s.to; kinds.add(s.s); }
  if (t !== X.end) bad++;
}
process.stdout.write(JSON.stringify({ bad, kinds: [...kinds].sort() }));""")
        self.assertEqual(got["bad"], 0)
        self.assertIn("open", got["kinds"])

    def test_the_real_records_timing_is_within_the_table(self):
        # What mflow_check.py's M5 checks on the bank, the view measures the same way.
        got = self.read("""
let n = 0, under = [];
for (let b = 0; b < R.B.rows; b++) for (const g of Dram.gaps(R, X, b)) { n++; if (!g.ok) under.push([b, g.name, g.measured, g.required]); }
process.stdout.write(JSON.stringify({ n, under: under.slice(0, 5) }));""")
        self.assertGreater(got["n"], 0)
        self.assertEqual(got["under"], [])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        BUNDLE = Path(args[0]).resolve()
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a.startswith("-")]
    unittest.main()
