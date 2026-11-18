#!/usr/bin/env python3
"""
Self-test of the resource-occupancy viewer (#390): its data layer, run in Node on timed records,
against dflow_check's reading of the same records; pack.py; and, when a headless Chromium is
available (Playwright), the page itself.

Usage: python3 test_viewer.py <a.dflow> <b.dflow> [<more timed .dflow> ...] --lb <lb.dflow>
                              [--ambiguous <slotted.dflow> <rtl.dflow>]

<a> and <b> are two timed records of one program (say kpu-run --dflow-lca and --dflow-lt1); the
rest are more timed records to read (the RTL sample, say); <lb> is an L-B record, which has no time
and is refused.

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
sys.path.insert(0, str(HERE.parent.parent / "trace"))
import dflow_check as dc  # noqa: E402

NODE = shutil.which("node")
A = B = LB = None
MORE = []
AMBIGUOUS = None                                 # (a slotted timed record, an RTL record of its program)

# Node: evaluate the page's data layer and report what it reads from each record, and the join.
HARNESS = r"""
const fs = require("fs"), path = require("path");
eval(fs.readFileSync(process.argv[2], "utf8"));
const C = globalThis.OccCore;
function files(dir) {
  const m = JSON.parse(fs.readFileSync(path.join(dir, "manifest.json"), "utf8")), out = { "manifest.json": fs.readFileSync(path.join(dir, "manifest.json"), "utf8") };
  for (const t of Object.values(m.tables)) { const b = fs.readFileSync(path.join(dir, t.file)); out[t.file] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); }
  return out;
}
const dirs = process.argv.slice(3), R = dirs.map(d => C.parse(files(d))), out = { records: [], joins: [] };
for (const r of R) {
  const proc = r.lanes.filter(l => l.kind === "process"), slots = r.lanes.filter(l => l.kind !== "process");
  out.records.push({ n: r.n, cycles: r.cycles, level: r.level, process_lanes: proc.map(l => l.name),
                     process_items: proc.reduce((s, l) => s + l.items.length, 0),
                     holds: r.holds.map(h => [h.layer, h.slot, h.tile, h.from, h.to, h.credit]),
                     slot_lanes: slots.map(l => l.name) });
}
const pairs = [[0, 0], [0, 1]];
out.refused = null;
if (process.env.OCC_AMBIGUOUS) {
  const [x, y] = process.env.OCC_AMBIGUOUS.split("|").map(d => C.parse(files(d)));
  try { C.join(x, y); } catch (e) { out.refused = String(e.message); }
}
for (const [i, j] of pairs) {
  const J = C.join(R[i], R[j]);
  out.joins.push({ a: i, b: j, matched: J.matched, onlyA: J.onlyA, onlyB: J.onlyB, drop: J.drop,
                   first: J.first, first_cycles: J.first >= 0 ? [R[i].E.cycle[J.first], R[i].E.end[J.first]] : null });
}
process.stdout.write(JSON.stringify(out));
"""


def core_script():
    page = (HERE / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<script id="occ-core">(.*?)</script>', page, re.S)
    assert m, "index.html has no occ-core script"
    return m.group(1)


class Viewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.out = None
        if NODE:
            core = cls.tmp / "core.js"
            core.write_text(core_script(), encoding="utf-8")
            h = cls.tmp / "h.js"
            h.write_text(HARNESS, encoding="utf-8")
            env = None
            if AMBIGUOUS:
                import os
                env = dict(os.environ, OCC_AMBIGUOUS=f"{AMBIGUOUS[0]}|{cls.ambiguous_copy()}")
            p = subprocess.run([NODE, str(h), str(core), str(A), str(B), *map(str, MORE)], capture_output=True, text=True,
                               env=env)
            if p.returncode:
                raise AssertionError(f"node: {p.stderr[-2000:]}")
            cls.out = json.loads(p.stdout)

    @classmethod
    def ambiguous_copy(cls):
        """The RTL record with two STREAMs made alike in everything but their L2 rows: distinct in it,
        one identity once L2's addresses are left out (against a slotted kpu-sim record)."""
        sys.path.insert(0, str(HERE.parent.parent / "trace"))
        from test_dflow_check import Bundle
        b = Bundle(AMBIGUOUS[1])
        E = lambda c, i: b.get("events", c, i)
        st = [i for i in range(b.rows("events")) if E("kind", i) == dc.EV_STREAM and E("src_addr", i) != dc.NONE64]
        e1 = st[0]
        e2 = next(i for i in st[1:] if E("src_addr", i) != E("src_addr", e1))
        for c in ("tile", "where", "process", "row", "beat", "dst_addr", "action"):
            b.set("events", c, e2, E(c, e1))
        dst = cls.tmp / "ambiguous.dflow"
        shutil.copytree(b.dir, dst)
        b.close()
        return dst

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def need_node(self):
        if not NODE:
            self.skipTest("Node.js is not on PATH")

    def test_scripts_parse(self):
        self.need_node()
        page = (HERE / "index.html").read_text(encoding="utf-8")
        for i, s in enumerate(re.findall(r"<script(?: id=\"[^\"]*\")?>(.*?)</script>", page, re.S)):
            f = self.tmp / f"s{i}.js"
            f.write_text(s, encoding="utf-8")
            p = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_the_page_reads_what_the_checker_reads(self):
        self.need_node()
        for path, r in zip([A, B, *MORE], self.out["records"]):
            with self.subTest(record=path.name):
                m, T = dc.load(path)
                E = T["events"]
                self.assertEqual((r["n"], r["cycles"], r["level"]), (E["rows"], m["time"]["cycles"], m["level"]))
                # Every transaction with a process sits on exactly one lane: its process and lane.
                tx = [e for e in range(E["rows"]) if E["kind"][e] <= dc.EV_DRAIN and E["process"][e] != dc.NONE8]
                self.assertEqual(r["process_items"], len(tx))
                want = sorted({f"{m['processes'][E['process'][e]]}[{E['lane'][e]}]" for e in tx})
                self.assertEqual(sorted(r["process_lanes"]), want)
                # The holdings, against an independent replay of the rule (#472's review): in a layer
                # addressed by slot, each landing opens one; a credit at the slot closes its tile's,
                # else the oldest; what is open at the end is held to it.
                g = m.get("geometry", {})
                slotted = {dc.L3: "slots" in g.get("l3", {}), dc.L2: "slots" in g.get("l2", {})}
                open_, want_h = {}, []
                for e in range(E["rows"]):
                    k = E["kind"][e]
                    if k <= dc.EV_DRAIN and slotted.get(E["dst"][e]) and E["dst_addr"][e] != dc.NONE64:
                        open_.setdefault((E["dst"][e], E["dst_addr"][e]), []).append(
                            [E["dst"][e], E["dst_addr"][e], E["tile"][e], E["end"][e]])
                    elif k == dc.EV_CREDIT and slotted.get(E["src"][e]) and E["src_addr"][e] != dc.NONE64:
                        lst = open_.get((E["src"][e], E["src_addr"][e]), [])
                        if lst:
                            i = next((j for j, o in enumerate(lst) if o[2] == E["tile"][e]), 0)
                            want_h.append(lst.pop(i) + [E["cycle"][e], e])
                for lst in open_.values():
                    want_h += [o + [m["time"]["cycles"], -1] for o in lst]
                self.assertEqual(sorted(map(tuple, r["holds"])), sorted(map(tuple, want_h)))
                if path in (A, B):
                    self.assertTrue(r["holds"], "the fixture should hold slots")
                if not slotted[dc.L3]:
                    self.assertFalse([h for h in r["holds"] if h[0] == dc.L3], "a word-addressed L3 has no slots")
                # A slot holding runs from a transaction into the slot to the credit that frees it
                # (or to the record's end).
                by_slot = {}
                for layer, slot, tile, t0, t1, credit in r["holds"]:
                    self.assertLessEqual(t0, t1)
                    self.assertIn(layer, (dc.L3, dc.L2))
                    if credit >= 0:
                        self.assertEqual((E["kind"][credit], E["src"][credit], E["src_addr"][credit]),
                                         (dc.EV_CREDIT, layer, slot))
                    by_slot.setdefault((layer, slot), []).append((t0, t1))
                del by_slot

    def test_an_ambiguous_join_is_refused(self):
        self.need_node()
        if not AMBIGUOUS:
            self.skipTest("no --ambiguous pair given")
        self.assertIsNotNone(self.out["refused"], "two events alike once L2's addresses are dropped were joined")
        self.assertIn("cannot be joined unambiguously", self.out["refused"])

    def test_a_record_joins_itself(self):
        self.need_node()
        j = self.out["joins"][0]
        self.assertEqual((j["matched"], j["onlyA"], j["onlyB"], j["first"]), (self.out["records"][0]["n"], 0, 0, -1))

    def test_the_overlay_joins_as_dflow_diff_does(self):
        self.need_node()
        j = self.out["joins"][1]
        diff = HERE.parent.parent / "trace" / "dflow_diff.py"
        if not diff.exists():
            self.skipTest("dflow_diff.py is not in this tree (#402)")
        p = subprocess.run([sys.executable, str(diff), str(A), str(B), "--json"], capture_output=True, text=True)
        self.assertIn(p.returncode, (0, 1), p.stderr)
        d = json.loads(p.stdout)
        self.assertEqual(j["matched"], d["matched"])
        self.assertEqual(sorted(j["drop"]), sorted(d["dropped_addresses"]))
        self.assertEqual((j["onlyA"], j["onlyB"]), (d["only_a"]["count"], d["only_b"]["count"]))
        f = d["first_divergence"]
        self.assertEqual(j["first_cycles"], f["a"] if f else None)

    def test_pack_embeds_and_refuses(self):
        out = self.tmp / "v.html"
        p = subprocess.run([sys.executable, str(HERE / "pack.py"), str(A), "--overlay", str(B), "-o", str(out)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        page = out.read_text(encoding="utf-8")
        self.assertIn("window.OCC_EMBED", page)
        self.assertNotIn("<!--OCC_EMBED-->", page)
        # Every "<" in the payload is escaped: a manifest string with "<!--<script>" cannot bend the
        # script element it is embedded in (#472's review).
        odd = self.tmp / "odd.dflow"
        shutil.copytree(A, odd)
        mf = json.loads((odd / "manifest.json").read_text(encoding="utf-8"))
        mf["program"]["text"] += "\n// <!--<script></script>"
        (odd / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
        p = subprocess.run([sys.executable, str(HERE / "pack.py"), str(odd), "-o", str(self.tmp / "odd.html")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        embedded = (self.tmp / "odd.html").read_text(encoding="utf-8").split("window.OCC_EMBED = ", 1)[1].split(";</script>", 1)[0]
        self.assertNotIn("<", embedded)
        self.assertIn("<!--<script>", json.loads(embedded)["a"]["manifest"])
        p = subprocess.run([sys.executable, str(HERE / "pack.py"), str(LB), "-o", str(self.tmp / "x.html")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("no time", p.stderr)
        p = subprocess.run([sys.executable, str(HERE / "pack.py"), str(self.tmp), "-o", str(self.tmp / "y.html")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)

    def test_the_page_loads_in_a_browser(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.skipTest("Playwright is not installed")
        out = self.tmp / "b.html"
        subprocess.run([sys.executable, str(HERE / "pack.py"), str(A), "--overlay", str(B), "-o", str(out)], check=True,
                       capture_output=True)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": 1200, "height": 800})
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(out.as_uri())
                page.wait_for_timeout(500)
                self.assertEqual(errors, [])
                self.assertEqual(page.inner_text("#err"), "")
                self.assertIn("cycles to result", page.inner_text("#meta"))
                self.assertIn("B joined:", page.inner_text("#joined"))
                if not page.is_disabled("#first"):
                    page.click("#first")
                    page.wait_for_timeout(200)
                    self.assertIn("B: event", page.inner_text("#inspector"))
                # A new A drops the overlay and its join before anything draws (#472's review).
                page.evaluate("document.getElementById('loader').hidden = false")
                try:
                    page.set_input_files("#pick", str(B))
                except Exception as e:              # a Playwright without directory uploads
                    self.skipTest(f"cannot pick a folder here: {e}")
                page.wait_for_timeout(500)
                self.assertEqual(errors, [])
                self.assertEqual(page.inner_text("#joined"), "")
                self.assertTrue(page.is_disabled("#first"))
                browser.close()
        except Exception as e:                       # no browser binary: skip, as the other viewers do
            if "Executable doesn't exist" in str(e) or "BrowserType.launch" in str(e):
                self.skipTest(f"no headless Chromium: {e}")
            raise


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--lb" not in args or len(args) < 4:
        sys.exit(__doc__)
    if "--ambiguous" in args:
        j = args.index("--ambiguous")
        AMBIGUOUS = (Path(args[j + 1]).resolve(), Path(args[j + 2]).resolve())
        args = args[:j] + args[j + 3:]
    i = args.index("--lb")
    LB = Path(args[i + 1]).resolve()
    rest = [Path(x).resolve() for x in args[:i] + args[i + 2:]]
    A, B, MORE = rest[0], rest[1], rest[2:]
    sys.argv = sys.argv[:1]
    unittest.main(verbosity=2)
