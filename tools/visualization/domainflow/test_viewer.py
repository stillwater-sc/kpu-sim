#!/usr/bin/env python3
"""
Headless test for the domain-flow viewer and pack.py (docs/plans/domain-flow-debugger.md step 3).

Usage: python3 test_viewer.py <bundle.dflow> [<bundle.dflow> ...]

For each bundle:
  - pack.py embeds it, and refuses a non-bundle with exit 2;
  - the viewer accepts the version the writer writes;
  - when Node.js is on PATH: both page scripts parse, and the viewer's data layer (the
    <script id="dflow-core"> block, which has no DOM) is RUN on the bundle. What it reads at sample
    steps -- each step's fields, its program position, and every L3 slot's residency -- must equal
    the bundle read independently here, through dflow_check.load. That is what the page renders.

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
import dflow_check  # noqa: E402

BUNDLES = []
NODE = shutil.which("node")
NONE8, NONE64 = 0xFF, 0xFFFFFFFFFFFFFFFF
RES_RETAINED, HAS_PLACE = 4, 8

HARNESS = r"""
const fs = require("fs"), path = require("path");
const [core, dir, stepsArg] = process.argv.slice(2);
eval(fs.readFileSync(core, "utf8"));
const files = { "manifest.json": fs.readFileSync(path.join(dir, "manifest.json"), "utf8") };
for (const t of Object.values(JSON.parse(files["manifest.json"]).tables)) {
  const b = fs.readFileSync(path.join(dir, t.file));
  files[t.file] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
const C = globalThis.DflowCore, R = C.parse(files), out = { n: R.n, steps: {} };
for (const s of JSON.parse(stepsArg)) {
  const slots = [];
  for (let sl = 0; sl < R.cap; sl++) slots.push(C.slotAt(R, sl, s));
  const t = R.steps.tile[s], val = v => v && { k: v.k, step: v.step, chan: v.chan, h: v.h, w: v.w, head: Array.from(v.data.slice(0, 4)), n: v.data.length };
  out.steps[s] = { step: C.step(R, s), position: C.position(R, s), slots,
                   value: R.snap ? val(C.valueAt(R, t, s)) : null, final: R.snap ? val(C.finalOf(R, t)) : null,
                   legs: C.legs(R, s), proximity: C.proximity(R, t), nextSlotBound: C.nextSlotBound(R, s),
                   contents: C.contentsAt(R, s), call: C.callAt(R, s), last: C.lastCallAt(R, s),
                   nextChainStart: C.nextChainStart(R, s), nextChainEnd: C.nextChainEnd(R, s) };
}
out.lb = R.lb;
// Where the derivation puts a tile, at every step a snapshot recorded it in L2 or the fabric: an
// identity check (the held counts check only how many).
out.snapIn = [];
if (R.snap) {
  const want = new Set(["l2", "cf"]);
  for (let k = 0; k < R.snapshots.rows && out.snapIn.length < 400; k++) {
    const s = R.snapshots.step[k], chan = R.m.channels[R.snapshots.chan[k]].name;
    if (s < 0 || s >= R.n || !want.has(chan)) continue;
    const c = C.contentsAt(R, s), t = R.snapshots.tile[k];
    if (!c) { out.snapIn.push({ s, chan, l2: false, cf: false }); continue; }
    out.snapIn.push({ s, chan, l2: c.l2.some(e => e.tile === t), cf: c.cf.some(e => e.tile === t) });
  }
}
if (R.lt1) {
  out.atTime = {};
  // Sample cycles, and every leg's finish (where the lane that ran it falls idle, or another starts).
  const ts = new Set([0, Math.floor(R.lt1.makespan / 3), Math.floor(R.lt1.makespan / 2), R.lt1.makespan]);
  for (let g = 0; g < R.lt1_legs.rows && ts.size < 200; g += Math.max(1, Math.floor(R.lt1_legs.rows / 150))) ts.add(R.lt1_legs.finish[g]);
  for (const t of ts) out.atTime[t] = C.actionAtTime(R, t);
}
process.stdout.write(JSON.stringify(out));
"""


def page():
    return (HERE / "index.html").read_text(encoding="utf-8")


def scripts():
    p = page()
    core = re.search(r'<script id="dflow-core">(.*?)</script>', p, re.S)
    ui = [s for s in re.findall(r"<script>(.*?)</script>", p, re.S)]
    return core.group(1) if core else None, ui


class ViewerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dflow-viewer-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def pack(self, *args):
        return subprocess.run([sys.executable, str(HERE / "pack.py"), *args], capture_output=True, text=True)

    def test_pack_embeds_each_bundle(self):
        for b in BUNDLES:
            with self.subTest(bundle=b):
                out = self.tmp / "view.html"
                p = self.pack(str(b), "-o", str(out))
                self.assertEqual(p.returncode, 0, p.stderr)
                html = out.read_text(encoding="utf-8")
                self.assertNotIn("<!--DFLOW_EMBED-->", html)
                m = re.search(r"window\.DFLOW_EMBED = (\{.*?\});</script>", html, re.S)
                self.assertIsNotNone(m)
                payload = json.loads(m.group(1).replace("<\\/", "</"))
                manifest = json.loads(payload["manifest"])
                self.assertEqual(manifest["format"], "kpu-dflow")
                for t in manifest["tables"].values():
                    self.assertIn(t["file"], payload["bins"])

    def test_pack_refuses_a_non_bundle(self):
        (self.tmp / "manifest.json").write_text('{"format": "kpu-tflow"}')
        p = self.pack(str(self.tmp), "-o", str(self.tmp / "x.html"))
        self.assertEqual(p.returncode, 2)
        self.assertIn("not a kpu-dflow bundle", p.stderr)
        p = self.pack(str(self.tmp / "missing"), "-o", str(self.tmp / "x.html"))
        self.assertEqual(p.returncode, 2)

    def test_viewer_reads_the_version_the_writer_writes(self):
        accepted = re.search(r"const VERSIONS = \[([\d, ]+)\]", page())
        self.assertIsNotNone(accepted)
        versions = [int(v) for v in accepted.group(1).split(",")]
        for b in BUNDLES:
            written = json.loads((Path(b) / "manifest.json").read_text(encoding="utf-8"))["version"]
            self.assertIn(written, versions)

    # Interpolations in HTML that are not escaped, each reviewed: numbers the page computes or reads
    # as checked integers, its own constants and class names, and pieces built only from escaped
    # or numeric parts. A new raw interpolation fails the scan below until it is escaped or added
    # here after review.
    REVIEWED_RAW = {
        # Numbers: record columns (typed arrays), loop counters, counts.
        "k", "k % 4", "k + 1", "sl", "at.res", "l.leg", "l.depth", "l.value", "l.hi", "pos.line",
        "st.place.slot", "st.place.bank", "v.h", "v.w", "e.moves", "K - cells", "R.tiles.operand[t] % 4",
        # Page constants and conditionals over them.
        "a", 'R.banks > 1 ? "s" : ""', 'R.residency.flags[at.res] & C.RES_RETAINED ? " retained" : ""',
        'now ? "<b>this step</b>" : `latest: step ${fmt(call.step)}`',
        # Built only from escaped or numeric pieces just above the statement that writes them.
        "shape", "mv", "rtxt", "ops", "cls", "extra",
        'l.process ? `${esc(l.process)} lane ${l.lane}` : "none (credit)"',
        # Inside a key/value pair that is escaped where it is written (esc(b)).
        "R.lt1.bound.atol", "R.lt1.bound.rtol",
    }
    SAFE_CALL = re.compile(r"^(esc|fmt|chip|list)\(")

    @staticmethod
    def interpolations(st):
        out, i = [], 0
        while (j := st.find("${", i)) >= 0:
            d, k = 0, j + 2
            while k < len(st) and not (st[k] == "}" and d == 0):
                d += st[k] == "{"
                d -= st[k] == "}"
                k += 1
            out.append(st[j + 2:k].strip())
            i = k + 1
        return out

    def html_statements(self):
        # Every statement that writes HTML: an innerHTML assignment, the move string (mv) and the
        # chip helper, whose results are written as HTML.
        html = page()
        stmts = re.findall(r"innerHTML\s*=.*?;\n", html, re.S) + re.findall(r"const mv = .*?;\n", html, re.S)
        stmts += re.findall(r"function chip\(.*?\n  \}", html, re.S)
        return stmts

    def test_every_bundle_string_written_as_html_is_escaped(self):
        # A packed page is shared, and a bundle is untrusted input: a name like <img onerror=...>
        # written raw would run script in the viewer. Every ${...} in a statement that writes HTML
        # must be escaped (esc, or a helper that escapes), or be a reviewed safe expression.
        stmts = self.html_statements()
        self.assertGreater(len(stmts), 10)
        for st in stmts:
            for e in self.interpolations(st):
                if self.SAFE_CALL.match(e) or e in self.REVIEWED_RAW:
                    continue
                self.fail(f"unescaped interpolation in HTML: ${{{e[:100]}}} -- escape it or review it into REVIEWED_RAW")
            for m in re.finditer(r"\+\s*([A-Za-z_][\w.]*\.(?:name|kind|tile|process|from|to|place|text|what))\s*\+", st):
                self.fail(f"{m.group(1)} concatenated unescaped in: {st[:140]}")

    def test_the_trusted_helpers_escape(self):
        # SAFE_CALL trusts chip() and list() to return escaped HTML: chip writes the tile name only
        # through esc(), and list() only joins what its callback (chip) returns, plus a count.
        html = page()
        body = re.search(r"function chip\(.*?\n  \}", html, re.S).group(0)
        self.assertIn("esc(C.tileName(R, t))", body)
        self.assertNotRegex(body, r"\$\{C\.tileName")
        lst = re.search(r"const MAX = 40, list = .*?;\n", html, re.S).group(0)
        self.assertEqual([e for e in self.interpolations(lst)], ["v.length - MAX"])
        # Each caller's extra argument is a constant, or built from numbers only.
        for extra in re.findall(r"chip\(e\.tile, (.*?), (?:touched|e\.tile|call)", html):
            self.assertRegex(extra, r'^(""|e\.moves > 1 \? .*e\.moves.*: "")$', extra)

    @unittest.skipUnless(NODE, "Node.js is not on PATH")
    def test_parse_refuses_a_non_integer_geometry(self):
        core, _ = scripts()
        b = self.tmp / "hostile.dflow"
        shutil.copytree(BUNDLES[0], b)
        m = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
        m["machine"]["l3"] = "17<img src=x onerror=alert(1)>"
        (b / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        (self.tmp / "core.js").write_text(core, encoding="utf-8")
        (self.tmp / "harness.js").write_text(HARNESS, encoding="utf-8")
        p = subprocess.run([NODE, str(self.tmp / "harness.js"), str(self.tmp / "core.js"), str(b), "[0]"],
                           capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("machine.l3", p.stderr)

    @unittest.skipUnless(NODE, "Node.js is not on PATH")
    def test_scripts_parse(self):
        core, ui = scripts()
        self.assertIsNotNone(core)
        for i, src in enumerate([core] + ui):
            f = self.tmp / f"s{i}.js"
            f.write_text(src, encoding="utf-8")
            p = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    @unittest.skipUnless(NODE, "Node.js is not on PATH")
    def test_the_viewer_reads_what_the_bundle_holds(self):
        core, _ = scripts()
        (self.tmp / "core.js").write_text(core, encoding="utf-8")
        (self.tmp / "harness.js").write_text(HARNESS, encoding="utf-8")
        for b in BUNDLES:
            with self.subTest(bundle=b):
                m, T = dflow_check.load(b)
                S, X, W, L = T["steps"], T["residency"], T["where"], T["loops"]
                n = S["rows"]
                self.assertGreater(n, 0)
                sample = sorted({0, 1, n // 7, n // 3, n // 2, (2 * n) // 3, n - 2, n - 1} - {-1})
                sample = [s for s in sample if 0 <= s < n]
                # And the first, a middle and the last call, and the step before each (a Feed).
                ci = [s for s in range(n) if m["kinds"][S["kind"][s]] == "CALL"]
                if ci:
                    for s in (ci[0], ci[len(ci) // 2], ci[-1]):
                        sample += [s - 1, s]
                    sample = sorted({s for s in sample if 0 <= s < n})
                derived = self.expected_contents(m, T, sample)
                p = subprocess.run([NODE, str(self.tmp / "harness.js"), str(self.tmp / "core.js"), str(b),
                                    json.dumps(sample)], capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                got = json.loads(p.stdout)
                self.assertEqual(got["n"], n)
                ops = [o["name"] for o in m["operands"]]
                tile = lambda t: f"{ops[T['tiles']['operand'][t]]}[{T['tiles']['ti'][t]},{T['tiles']['tj'][t]}]"
                for s in sample:
                    v = got["steps"][str(s)]
                    st = v["step"]
                    # The step's fields.
                    self.assertEqual(st["action"], s)
                    self.assertEqual(st["kind"], m["kinds"][S["kind"][s]])
                    self.assertEqual(st["tile"], tile(S["tile"][s]))
                    res = S["residency"][s]
                    self.assertEqual(st["residency"], -1 if res == NONE64 else res)
                    self.assertEqual(st["process"], None if S["process"][s] == NONE8 else m["processes"][S["process"][s]])
                    self.assertEqual(st["held"], {"dram": S["held_dram"][s], "l3": S["held_l3"][s],
                                                  "l2": S["held_l2"][s], "cf": S["held_cf"][s]})
                    if S["flags"][s] & HAS_PLACE:
                        self.assertEqual(st["place"]["slot"], S["slot"][s])
                    else:
                        self.assertIsNone(st["place"])
                    # The program position.
                    w = S["where"][s]
                    self.assertEqual(v["position"]["line"], W["line"][w])
                    loops = [(m["vars"][L["var"][W["loop_first"][w] + d]], L["value"][W["loop_first"][w] + d])
                             for d in range(W["loop_count"][w])]
                    self.assertEqual([(l["var"], l["value"]) for l in v["position"]["loops"]], loops)
                    # Every L3 slot: the residency held after step s, or freed by it.
                    cap = m["machine"]["l3"]
                    want = [{"res": -1, "held": False} for _ in range(cap)]
                    for r in range(X["rows"]):
                        o, c = X["open"][r], X["close"][r]
                        if o > s:
                            continue
                        if c == NONE64 or c > s or X["flags"][r] & RES_RETAINED:
                            want[X["slot"][r]] = {"res": r, "held": True}
                        elif c == s:
                            want[X["slot"][r]] = {"res": r, "held": False}
                    self.assertEqual(v["slots"], want, f"slots after step {s}")
                    # The tile inspector: the step tile's value after step s, and its final value.
                    if m["version"] >= 2:
                        self.assertEqual((v["value"], v["final"]), self.expected_values(T, m, S["tile"][s], s))
                    # The L-T1 overlay: this action's legs, the tile's proximity, the next slot wait.
                    if m["version"] >= 3 and m["lt1"]:
                        self.assertEqual((v["legs"], v["proximity"], v["nextSlotBound"]),
                                         self.expected_lt1(T, m, S["tile"][s], s, n))
                    else:
                        self.assertEqual((v["legs"], v["proximity"], v["nextSlotBound"]), ([], None, -1))
                    # L2 and the fabric (l2-reuse-and-fabric-pipeline-view.md step 1).
                    want = derived[s]
                    self.assertEqual(v["contents"], want["contents"], f"L2 and fabric contents after step {s}")
                    self.assertEqual(v["call"], want["call"], f"the call at step {s}")
                    self.assertEqual(v["last"], want["last"], f"the latest call at step {s}")
                    self.assertEqual((v["nextChainStart"], v["nextChainEnd"]), want["next"], f"chain navigation from {s}")
                self.assertTrue(got["lb"]["ok"], got["lb"]["why"])
                for e in got["snapIn"]:
                    self.assertTrue(e[e["chan"]], f"a snapshot has a tile in {e['chan']} after step {e['s']}; the derivation does not")
                if m["version"] >= 3 and m["lt1"]:
                    G = T["lt1_legs"]
                    for t, a in got["atTime"].items():
                        # Running at t: start <= t < finish. The latest to start; on a tie, any.
                        run = [(G["start"][g], g) for g in range(G["rows"])
                               if G["start"][g] <= int(t) < G["finish"][g] and G["action"][g] < n]
                        if not run:
                            self.assertEqual(a, -1, f"cycle {t}: nothing runs, yet the click lands on {a}")
                        else:
                            latest = max(run)[0]
                            self.assertIn(a, {G["action"][g] for st, g in run if st == latest}, f"action at cycle {t}")

    @staticmethod
    def expected_contents(m, T, sample):
        # An independent replay of the recorded actions: a step's `to` channel gains its tile, its
        # `from` channel loses it, and a Call consumes what was fed since the last call (its result
        # stays, or opens as an accumulator). Every step's counts must equal L-B's held counts.
        S = T["steps"]
        ch = [c["name"] for c in m["channels"]]
        L3, L2, CF, CALL = ch.index("l3"), ch.index("l2"), ch.index("cf"), m["kinds"].index("CALL")
        l2, cf, fed, moves, calls, chains, open_ = {}, {}, [], {}, [], [], {}
        snaps, wanted = {}, set(sample)
        for s in range(S["rows"]):
            t, f, to = S["tile"][s], S["from"][s], S["to"][s]
            if f == L3 and to == L2:
                moves.setdefault(t, []).append(s)
            if f == L2:
                l2.pop(t)
            if f == CF:
                cf.pop(t)
            if to == L2:
                l2[t] = s
            if to == CF:
                cf[t] = (s, False)
                fed.append(t)
            if S["kind"][s] == CALL:
                used = [x for x in fed if x != t]
                for x in used:
                    cf.pop(x)
                fed = []
                cf.setdefault(t, (s, True))
                if t not in open_:
                    open_[t] = len(chains)
                    chains.append({"calls": [], "drain": -1})
                c = open_[t]
                chains[c]["calls"].append(len(calls))
                calls.append({"step": s, "tile": t, "operands": used, "chain": c, "pos": len(chains[c]["calls"])})
            if f == CF and t in open_:
                chains[open_.pop(t)]["drain"] = s
            assert (len(l2), len(cf)) == (S["held_l2"][s], S["held_cf"][s]), f"replay disagrees with held at {s}"
            if s in wanted:
                snaps[s] = (dict(l2), dict(cf))

        def info(k):
            c = calls[k]
            ch_ = chains[c["chain"]]
            of = len(ch_["calls"]) if ch_["drain"] >= 0 else -1
            return {"step": c["step"], "result": c["tile"], "operands": c["operands"], "chain": c["chain"], "pos": c["pos"],
                    "of": of, "opens": c["pos"] == 1, "closes": of >= 0 and c["pos"] == of, "drain": ch_["drain"]}

        out = {}
        for s in sample:
            l2s, cfs = snaps[s]
            nm = lambda t: sum(1 for x in moves.get(t, []) if x <= s)
            contents = {"l2": [{"tile": t, "since": a, "moves": nm(t)} for t, a in sorted(l2s.items(), key=lambda kv: kv[1])],
                        "cf": [{"tile": t, "since": a, "acc": acc} for t, (a, acc) in sorted(cfs.items(), key=lambda kv: kv[1][0])]}
            at = [k for k, c in enumerate(calls) if c["step"] == s]
            before = [k for k, c in enumerate(calls) if c["step"] <= s]
            after = [info(k) for k, c in enumerate(calls) if c["step"] > s]
            nxt = (next((c["step"] for c in after if c["opens"]), -1), next((c["step"] for c in after if c["closes"]), -1))
            out[s] = {"contents": contents, "call": info(at[0]) if at else None,
                      "last": info(before[-1]) if before else None, "next": nxt}
        return out

    def expected_lt1(self, T, m, t, s, n):
        G, Q = T["lt1_legs"], T["lt1_tiles"]
        legs = sorted((G["leg"][g], g) for g in range(G["rows"]) if G["action"][g] == s)
        out = [{"leg": G["leg"][g], "process": None if G["proc"][g] == NONE8 else m["processes"][G["proc"][g]],
                "lane": G["lane"][g], "start": G["start"][g], "finish": G["finish"][g], "slot_wait": G["slot_wait"][g]}
               for _, g in legs]
        prox = None
        for q in range(Q["rows"]):
            if Q["tile"][q] == t:
                prox = {"max_abs": Q["max_abs"][q], "max_rel": Q["max_rel"][q], "pass": Q["pass"][q] == 1}
        nxt = [G["action"][g] for g in range(G["rows"]) if G["slot_wait"][g] > 0 and s < G["action"][g] < n]
        return out, prox, min(nxt) if nxt else -1

    def expected_values(self, T, m, t, s):
        P, E = T["snapshots"], T["elements"]["v"]
        def shown(k):
            if k is None:
                return None
            h, w, f = P["height"][k], P["width"][k], P["first"][k]
            return {"k": k, "step": -1 if P["step"][k] == NONE64 else P["step"][k], "chan": m["channels"][P["chan"][k]]["name"],
                    "h": h, "w": w, "head": list(E[f:f + min(4, h * w)]), "n": h * w}
        at, final = None, None
        for k in range(P["rows"]):
            if P["tile"][k] != t:
                continue
            if P["step"][k] == NONE64:
                final = k
            elif P["step"][k] <= s and (at is None or P["step"][k] >= P["step"][at]):
                at = k
        return shown(at), shown(final)

    def test_the_bundles_exercise_the_inspector(self):
        # At least one bundle must carry step snapshots, or the value comparison above is vacuous.
        carried = 0
        for b in BUNDLES:
            m, T = dflow_check.load(b)
            if m["version"] >= 2:
                carried += sum(1 for k in range(T["snapshots"]["rows"]) if T["snapshots"]["step"][k] != NONE64)
        self.assertGreater(carried, 0)
        # And one must carry an L-T1 overlay with a slot wait, or the overlay comparison is vacuous.
        waits = 0
        for b in BUNDLES:
            m, T = dflow_check.load(b)
            if m["version"] >= 3 and m["lt1"]:
                waits += sum(1 for g in range(T["lt1_legs"]["rows"]) if T["lt1_legs"]["slot_wait"][g] > 0)
        self.assertGreater(waits, 0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    while len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        BUNDLES.append(Path(sys.argv.pop(1)).resolve())
    if not BUNDLES:
        print("test_viewer: give at least one .dflow bundle; with none, every bundle test would pass vacuously",
              file=sys.stderr)
        sys.exit(2)
    unittest.main()
