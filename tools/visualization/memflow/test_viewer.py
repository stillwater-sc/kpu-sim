#!/usr/bin/env python3
"""
Smoke test for the memory-side debugger and pack.py (docs/plans/memory-side-debugger.md step 5).

Usage: python3 test_viewer.py <bundle.mflow> [--expect-fills] [--expect-waits]

Packs the bundle, checks the payload is embedded and parses, checks a non-bundle is refused
with exit 2, and -- when Node.js is on PATH -- syntax-checks the viewer's script and runs its
reader over the real bundle, which are the parts of the page a broken edit silently disables.

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
READER = HERE.parent / "common" / "mflow.js"
LIVE = HERE.parent / "common" / "live.js"
BUNDLE = None
EXPECT_FILLS = False   # --expect-fills: the bundle's run has Fills (a matmul's C, from L-B)
EXPECT_WAITS = False   # --expect-waits: the bundle's run stalls its DMA engine


def extract(page, signature):
    m = re.search(re.escape(signature) + r".*?\n\}", page, re.S)
    if m is None:
        raise AssertionError("index.html has no " + signature)
    return m.group(0)


class ViewerSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mflow-viewer-"))
        self.page = (HERE / "index.html").read_text(encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def pack(self, *args):
        return subprocess.run([sys.executable, str(HERE / "pack.py"), *args],
                              capture_output=True, text=True)

    def node(self, source):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js not on PATH")
        js = self.tmp / "t.js"
        js.write_text(source, encoding="utf-8")
        p = subprocess.run([node, str(js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout)

    def test_pack_embeds_the_bundle(self):
        out = self.tmp / "view.html"
        p = self.pack(BUNDLE, "-o", str(out))
        self.assertEqual(p.returncode, 0, p.stderr)
        page = out.read_text(encoding="utf-8")
        self.assertNotIn("<!--MFLOW_EMBED-->", page)
        m = re.search(r"window\.MFLOW_EMBED = (\{.*?\});</script>", page, re.S)
        self.assertIsNotNone(m)
        payload = json.loads(m.group(1).replace("<\\/", "</"))
        manifest = json.loads(payload["manifest"])
        self.assertEqual(manifest["format"], "kpu-mflow")
        for table in manifest["tables"].values():
            self.assertIn(table["file"], payload["bins"])
        # The reader is inlined in place of its <script src>: the packed page needs no other file.
        self.assertNotIn('<script src="../common/mflow.js">', page)
        # The server's reload bar is left out: a packed page loads no script from anywhere.
        self.assertNotIn("<script src=", page)
        self.assertIn("function parse(files)", page)

    def test_viewer_reads_the_version_the_writer_writes(self):
        # The page refuses any other manifest version, so a format bump that the viewer does not
        # follow would leave it refusing every new bundle -- invisible to a syntax check.
        written = json.loads((Path(BUNDLE) / "manifest.json").read_text(encoding="utf-8"))["version"]
        accepted = re.search(r"const VERSIONS = \[([\d, ]+)\]", READER.read_text(encoding="utf-8"))
        self.assertIsNotNone(accepted)
        self.assertIn(written, [int(v) for v in accepted.group(1).split(",")])

    def test_the_bundle_carries_what_the_viewer_draws_against(self):
        m = json.loads((Path(BUNDLE) / "manifest.json").read_text(encoding="utf-8"))
        self.assertGreater(m["ceiling_bytes_per_cycle"], 0)
        self.assertGreater(m["burst_bytes"], 0)
        kinds = {s["kind"] for s in m["stations"]}
        # Every memory side has banks, buses, engines and store buffers; port stubs only when the
        # NoC is stubbed (kpu-memsim). A cycle-core bundle has none, and the page draws no port rows;
        # it has a dmawait station per engine (version 4), which the page does not draw yet.
        self.assertLessEqual({"dram_bank", "dram_bus", "dma", "dmabuf"}, kinds)
        self.assertLessEqual(kinds, {"dram_bank", "dram_bus", "dma", "dmabuf", "port", "dmawait"})

    def read(self, bundle, extra=""):
        """Run the shared reader (../common/mflow.js) over a bundle directory under Node."""
        return self.node("\n".join([
            "const Mflow = require(%s);" % json.dumps(str(READER)),
            "const fs = require('fs'), path = require('path'), dir = %s;" % json.dumps(str(Path(bundle).resolve())),
            "const files = { 'manifest.json': fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8') };",
            "for (const t of Object.values(JSON.parse(files['manifest.json']).tables)) {",
            "  const b = fs.readFileSync(path.join(dir, t.file));",
            "  files[t.file] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); }",
            "let R; try { R = Mflow.parse(files); } catch (e) { process.stdout.write(JSON.stringify({ error: e.message })); process.exit(0); }",
            extra,
        ]))

    def test_the_viewer_reads_the_real_bundle(self):
        # The page's reader is ../common/mflow.js: every column the page names must exist with the
        # dtype it expects, every cross-index must hold, and version 4's fields must be there.
        got = self.read(BUNDLE, """
const used = { B: ['mc','channel','bank_group','bank','row','col','engine','request','is_load','outcome','t_posted','t_submit','t_cmd','t_data0','t_data1','t_done'],
  C: ['mc','channel','bank_group','bank','row','kind','burst','t_issue','t_end','t_data0','t_data1'],
  Q: ['engine','port','is_load','address','bytes','t_offered','t_credit','t_first','t_last','t_retired','kind','ticket','slot'],
  F: ['engine','held','staged','t'], P: ['port','request','kind','t'], W: ['engine','reason','t_from','t_to'] };
const missing = [];
for (const [k, cols] of Object.entries(used)) for (const c of cols) if (!R[k][c]) missing.push(k + '.' + c);
process.stdout.write(JSON.stringify({ missing, version: R.m.version, bursts: R.B.rows, commands: R.C.rows,
  requests: R.Q.rows, waits: R.W.rows }));""")
        self.assertNotIn("error", got)
        self.assertEqual(got["missing"], [])
        self.assertGreater(got["bursts"], 0)
        self.assertGreater(got["commands"], got["bursts"])   # every burst has at least its RD/WR
        self.assertGreater(got["requests"], 0)

    def version_3_copy(self):
        """The bundle declared version 3: its v4 columns and waits table are ignored."""
        d = self.tmp / "v3.mflow"
        shutil.copytree(BUNDLE, d)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        m["version"] = 3
        m.pop("source", None)
        m.pop("write_window", None)
        m["tables"].pop("waits", None)
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        return d

    def test_a_version_3_bundle_reads_as_version_4_would(self):
        got = self.read(self.version_3_copy(), """
const kinds = [...R.Q.kind].every((k, q) => k === (R.Q.is_load[q] ? 0 : 1));
process.stdout.write(JSON.stringify({ version: R.m.version, kinds, waits: R.W.rows,
  tickets: [...R.Q.ticket].every(t => t === Mflow.NONE), slots: [...R.Q.slot].every(t => t === Mflow.NONE) }));""")
        self.assertEqual(got, {"version": 3, "kinds": True, "waits": 0, "tickets": True, "slots": True})

    def test_other_versions_and_contradicting_kinds_are_refused(self):
        d = self.tmp / "v5.mflow"
        shutil.copytree(BUNDLE, d)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        m["version"] = 5
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        self.assertIn("not a version-3 or -4", self.read(d)["error"])
        # A request whose kind contradicts its is_load: the reader refuses it by name.
        d = self.tmp / "bad.mflow"
        shutil.copytree(BUNDLE, d)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        t = m["tables"]["requests"]
        kind = next(c for c in t["columns"] if c["name"] == "kind")
        blob = bytearray((d / t["file"]).read_bytes())
        is_load = next(c for c in t["columns"] if c["name"] == "is_load")
        q = 0
        blob[kind["offset"] + q] = 1 if blob[is_load["offset"] + q] else 0   # a store that loads, or a load that stores
        (d / t["file"]).write_bytes(bytes(blob))
        self.assertIn("is_load says otherwise", self.read(d)["error"])

    def test_a_served_bundle_is_fetched_by_the_names_in_its_manifest(self):
        # Mflow.open(url) fetches the manifest, then each table the manifest names, from the bundle's
        # directory: a stub fetch serves the files and records what was asked for.
        got = self.node("\n".join([
            "const Mflow = require(%s);" % json.dumps(str(READER)),
            "const fs = require('fs'), path = require('path'), dir = %s;" % json.dumps(str(Path(BUNDLE).resolve())),
            """const asked = [];
const stub = async url => {
  asked.push(url);
  const f = path.join(dir, url.replace('http://server/runs/x/mem.mflow/', ''));
  if (!fs.existsSync(f)) return { ok: false, status: 404 };
  const b = fs.readFileSync(f);
  return { ok: true, text: async () => b.toString('utf8'), arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) };
};
(async () => {
  const out = {};
  // The bundle's directory, with and without its trailing slash, and with a query string.
  for (const base of ['http://server/runs/x/mem.mflow', 'http://server/runs/x/mem.mflow/', 'http://server/runs/x/mem.mflow/?v=1']) {
    asked.length = 0;
    const R = await Mflow.open(base, stub);
    out[base] = { bursts: R.B.rows, asked: asked.map(u => u.split('/').pop()) };
  }
  process.stdout.write(JSON.stringify(out));
})();""",
        ]))
        self.assertEqual(len(got), 3)
        for base, g in got.items():
            self.assertGreater(g["bursts"], 0, base)
            self.assertEqual(g["asked"][0], "manifest.json", base)
            self.assertIn("bursts.bin", g["asked"], base)

    def test_a_served_bundle_cannot_name_a_file_outside_it(self):
        got = self.node("\n".join([
            "const Mflow = require(%s);" % json.dumps(str(READER)),
            """const manifest = JSON.stringify({ format: 'kpu-mflow', version: 4, tables: { bursts: { file: '../../etc/passwd', rows: 0, columns: [] } } });
const stub = async url => ({ ok: true, text: async () => manifest, arrayBuffer: async () => new ArrayBuffer(0) });
Mflow.open('http://server/b/', stub).then(() => process.stdout.write('"accepted"'), e => process.stdout.write(JSON.stringify(e.message)));""",
        ]))
        self.assertIn("names a table file", got)

    def test_a_rewritten_run_offers_a_reload_only_to_its_own_views(self):
        # ../common/live.js (plan step 8): which events of /api/events put up the bar.
        base = "http://localhost:8080/view/memflow/index.html?bundle=/files/r/mem.mflow/"
        run = {"id": "r", "status": "ok", "artifacts": [{"url": "/files/r/mem.mflow/"}, {"url": "/files/r/trace.json"}]}
        cases = [
            ["run-changed", run, "/files/r/mem.mflow/"],
            ["run-changed", run, "/files/r/mem.mflow"],                 # the trailing "/" does not matter
            ["run-changed", run, "http://localhost:8080/files/r/mem.mflow/"],
            ["run-added", run, "/files/r/mem.mflow/"],                  # rewritten between two polls
            ["run-removed", run, "/files/r/mem.mflow/"],
            ["run-changed", dict(run, status="running"), "/files/r/mem.mflow/"],   # its record is not written yet
            ["run-removed", dict(run, status="running"), "/files/r/mem.mflow/"],
            ["run-changed", run, "/files/other/mem.mflow/"],            # another run's view
            ["run-changed", run, "/files/r/mem.mflow/x/"],
            ["run-changed", {"id": "r"}, "/files/r/mem.mflow/"],        # no artifacts
            ["run-changed", None, "/files/r/mem.mflow/"],
            ["run-something", run, "/files/r/mem.mflow/"],
        ]
        got = self.node("\n".join([
            f"const L = require({json.dumps(str(LIVE))});",
            f"const cases = {json.dumps(cases)};",
            f"process.stdout.write(JSON.stringify(cases.map(([n, r, h]) => L.notice(n, r, h, {json.dumps(base)}))));",
        ]))
        newer, removed = "This run has a newer version.", "This run was removed from the server."
        self.assertEqual(got, [newer, newer, newer, newer, removed, None, removed, None, None, None, None, None])

    def test_the_page_watches_its_run_only_when_served(self):
        self.assertIn('<script src="../common/live.js"></script>', self.page)
        self.assertIn('MflowLive.watch("bundle")', self.page)
        live = LIVE.read_text(encoding="utf-8")
        # Nothing to watch from a file, a packed page, or another origin.
        self.assertIn("!/^https?:$/.test(location.protocol)", live)
        self.assertIn("if (href.origin !== location.origin) return null;", live)

    def test_the_page_reads_a_served_bundle_only_from_its_own_origin(self):
        self.assertIn('<script src="../common/mflow.js"></script>', self.page)
        self.assertIn("url.origin === location.origin", self.page)

    def test_a_bad_column_is_refused_by_name(self):
        # An unknown dtype must say so, not fail on the alignment check with a TypeError.
        src = "const { col } = require(%s);" % json.dumps(str(READER)) + """
const out = [];
for (const c of [{ name: "x", dtype: "u16", offset: 0 }, { name: "x", dtype: "f64", offset: 4 }]) {
  try { col(new ArrayBuffer(64), [c], "x", 1); out.push("accepted"); } catch (e) { out.push(e.message); }
}
process.stdout.write(JSON.stringify(out));"""
        self.assertEqual(self.node(src), ["column x has an unknown dtype u16", "column x is misaligned"])

    def test_record_strings_never_become_markup(self):
        # The manifest's strings reach innerHTML (chips, inspector, observations): each must be
        # escaped, and an arbitration the format does not know must be refused by name.
        out = self.node("\n".join([extract(self.page, "function esc(v) {"), """
process.stdout.write(JSON.stringify([esc('<img src=x onerror=alert(1)>'), esc("a&b\\"'"), esc(42)]));"""]))
        self.assertEqual(out, ["&lt;img src=x onerror=alert(1)&gt;", "a&amp;b&quot;&#39;", "42"])
        for sink in ("const kv = ", "const chip = ", '$("obs").innerHTML'):
            line = next(l for l in self.page.splitlines() if sink in l)
            self.assertIn("esc(", line, f"{sink} writes record strings without esc")
        self.assertIn('m.arbitration !== "round_robin" && m.arbitration !== "fixed"', READER.read_text(encoding="utf-8"))

    def test_the_step_series_counts_in_flight(self):
        src = "\n".join([extract(self.page, "function stepSeries(intervals) {"),
                         extract(self.page, "function valueAt(steps, t) {"), """
const s = stepSeries([[0, 10], [5, 15], [10, 20], [30, 30]]);
process.stdout.write(JSON.stringify([s, [0, 4, 5, 10, 15, 19, 20, 25].map(t => valueAt(s, t))]));"""])
        steps, values = self.node(src)
        # An interval that ends at t and one that starts at t do not overlap; an empty one is no
        # interval at all.
        self.assertEqual(steps, [[0, 1], [5, 2], [15, 1], [20, 0]])
        self.assertEqual(values, [1, 1, 2, 2, 1, 1, 0, 0])

    def test_waits_are_grouped_by_cause_and_found_at_a_cycle(self):
        # The waits lane's data: an engine's intervals by cause, and the causes holding it at t
        # (half-open: an interval [a, b) holds at a, not at b); another engine's waits are not its.
        src = "\n".join(["const WAIT = [0, 1, 2, 3, 4, 5];",
                         extract(self.page, "function groupWaits(W, engine) {"),
                         extract(self.page, "function waitsAt(W, byCause, t) {"), """
const W = { rows: 5, engine: [0, 0, 0, 1, 0], reason: [3, 0, 3, 3, 0],
            t_from: [10, 0, 40, 10, 12], t_to: [20, 5, 50, 20, 30] };
const g = groupWaits(W, 0);
process.stdout.write(JSON.stringify([g, [0, 4, 5, 10, 12, 19, 20, 29, 30, 45].map(t => waitsAt(W, g, t))]));"""])
        groups, at = self.node(src)
        self.assertEqual(groups, [[1, 4], [], [], [0, 2], [], []])
        self.assertEqual(at, [[[0, 0, 5]], [[0, 0, 5]], [], [[3, 10, 20]], [[0, 12, 30], [3, 10, 20]],
                              [[0, 12, 30], [3, 10, 20]], [[0, 12, 30]], [[0, 12, 30]], [], [[3, 40, 50]]])

    def test_the_real_bundles_fills_move_no_dram_bytes(self):
        # A Fill is a request with no bursts; every other request has exactly the bursts its bytes
        # span (mflow_check.py's M4: a request smaller than a burst still takes a whole one).
        got = self.read(BUNDLE, """
const of = new Array(R.Q.rows).fill(0);
for (let b = 0; b < R.B.rows; b++) if (R.B.request[b] !== Mflow.NONE) of[R.B.request[b]]++;
const bb = BigInt(R.m.burst_bytes);
let fills = 0, fillBursts = 0, wrongSpans = 0;
for (let q = 0; q < R.Q.rows; q++) {
  if (R.Q.kind[q] === 2) { fills++; fillBursts += of[q]; continue; }
  const a = BigInt(R.Q.address[q]), n = BigInt(R.Q.bytes[q]);
  const want = n === 0n ? 0n : (a + n + bb - 1n) / bb - a / bb;
  if (BigInt(of[q]) !== want) wrongSpans++;
}
process.stdout.write(JSON.stringify({ fills, fillBursts, wrongSpans }));""")
        self.assertEqual(got["fillBursts"], 0)
        self.assertEqual(got["wrongSpans"], 0)
        if EXPECT_FILLS:   # a fixture that promises Fills (--expect-fills): a load-only run has none
            self.assertGreater(got["fills"], 0)

    def test_the_real_bundles_waits_are_grouped_and_found(self):
        # groupWaits and waitsAt on the bundle's own waits table: every interval lands in its engine's
        # list for its cause, and is found at its first cycle; a cycle-core bundle has waits.
        got = self.read(BUNDLE, "\n".join(["const WAIT = Mflow.WAIT;",
                                          extract(self.page, "function groupWaits(W, engine) {"),
                                          extract(self.page, "function waitsAt(W, byCause, t) {"), """
const W = R.W, engines = [...new Set(W.engine)];
let grouped = 0, found = 0, cycles = 0;
for (const e of engines) {
  const g = groupWaits(W, e);
  g.forEach((l, k) => { for (const i of l) {
    grouped++; cycles += W.t_to[i] - W.t_from[i];
    if (W.engine[i] !== e || W.reason[i] !== k) throw new Error("wait " + i + " grouped under the wrong engine or cause");
    if (waitsAt(W, g, W.t_from[i]).some(([c, a, b]) => c === k && a === W.t_from[i] && b === W.t_to[i])) found++;
  } });
}
process.stdout.write(JSON.stringify({ source: R.m.source || null, rows: W.rows, grouped, found, cycles }));"""]))
        self.assertNotIn("error", got)
        self.assertEqual(got["grouped"], got["rows"])
        self.assertEqual(got["found"], got["rows"])
        if EXPECT_WAITS:   # a fixture that promises stalls (--expect-waits): a run need not stall
            self.assertGreater(got["rows"], 0)
            self.assertGreater(got["cycles"], 0)

    def test_every_colour_is_defined_in_both_themes(self):
        # The waits, the Fills: each token the page draws with has a light and a dark value (the
        # page has one light block and two dark ones: the media query and data-theme="dark").
        blocks = re.findall(r"(:root\s*\{.*?\}|:root:not\(\[data-theme=\"light\"\]\)\s*\{.*?\}|:root\[data-theme=\"dark\"\]\s*\{.*?\})", self.page, re.S)
        self.assertEqual(len(blocks), 3)
        for tok in ["--fill"] + ["--w%d" % k for k in range(6)]:
            for b in blocks:
                self.assertIn(tok + ":", b, f"{tok} missing from a theme block")

    def test_a_non_bundle_is_refused(self):
        p = self.pack(str(self.tmp), "-o", str(self.tmp / "x.html"))
        self.assertEqual(p.returncode, 2)

    def test_viewer_script_parses(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js not on PATH")
        scripts = re.findall(r"<script>\n(.*?)\n</script>", self.page, re.S)
        self.assertTrue(scripts)
        js = self.tmp / "viewer.js"
        js.write_text(scripts[-1], encoding="utf-8")
        p = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        p = subprocess.run([node, "--check", str(READER)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = sys.argv.pop(1)
    for flag in ("--expect-fills", "--expect-waits"):
        if flag in sys.argv:
            sys.argv.remove(flag)
            globals()["EXPECT_" + flag.split("-")[-1].upper()] = True
    unittest.main()
