#!/usr/bin/env python3
"""
Smoke test for the memory-side debugger and pack.py (docs/plans/memory-side-debugger.md step 5).

Usage: python3 test_viewer.py <bundle.mflow>

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
BUNDLE = None


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

    def test_viewer_reads_the_version_the_writer_writes(self):
        # The page refuses any other manifest version, so a format bump that the viewer does not
        # follow would leave it refusing every new bundle -- invisible to a syntax check.
        written = json.loads((Path(BUNDLE) / "manifest.json").read_text(encoding="utf-8"))["version"]
        accepted = re.search(r"m\.version !== (\d+)", self.page)
        self.assertIsNotNone(accepted)
        self.assertEqual(int(accepted.group(1)), written)

    def test_the_bundle_carries_what_the_viewer_draws_against(self):
        m = json.loads((Path(BUNDLE) / "manifest.json").read_text(encoding="utf-8"))
        self.assertGreater(m["ceiling_bytes_per_cycle"], 0)
        self.assertGreater(m["burst_bytes"], 0)
        kinds = {s["kind"] for s in m["stations"]}
        self.assertEqual(kinds, {"dram_bank", "dram_bus", "dma", "dmabuf", "port"})

    def test_the_viewer_reads_the_real_bundle(self):
        # Runs the page's own reader (col + parse) over the bundle under Node: every column the
        # page names must exist with the dtype it expects, and every cross-index must hold.
        src = "\n".join([
            "const NONE = 0xFFFFFFFF;",
            extract(self.page, "function col(buf, cols, name, n) {"),
            extract(self.page, "function parse(files) {"),
            "const fs = require('fs'), path = require('path'), dir = process.argv[2] || %s;" % json.dumps(str(Path(BUNDLE).resolve())),
            "const files = { 'manifest.json': fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8') };",
            "for (const t of Object.values(JSON.parse(files['manifest.json']).tables)) {",
            "  const b = fs.readFileSync(path.join(dir, t.file));",
            "  files[t.file] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); }",
            "const R = parse(files);",
            "const used = { B: ['mc','channel','bank_group','bank','row','col','engine','request','is_load','outcome','t_posted','t_submit','t_cmd','t_data0','t_data1','t_done'],",
            "  C: ['mc','channel','bank_group','bank','row','kind','burst','t_issue','t_end','t_data0','t_data1'],",
            "  Q: ['engine','port','is_load','address','bytes','t_offered','t_credit','t_first','t_last','t_retired'],",
            "  F: ['engine','held','staged','t'], P: ['port','request','kind','t'] };",
            "const missing = [];",
            "for (const [k, cols] of Object.entries(used)) for (const c of cols) if (!R[k][c]) missing.push(k + '.' + c);",
            "process.stdout.write(JSON.stringify({ missing, bursts: R.B.rows, commands: R.C.rows, requests: R.Q.rows }));",
        ])
        got = self.node(src)
        self.assertEqual(got["missing"], [])
        self.assertGreater(got["bursts"], 0)
        self.assertGreater(got["commands"], got["bursts"])   # every burst has at least its RD/WR
        self.assertGreater(got["requests"], 0)

    def test_a_bad_column_is_refused_by_name(self):
        # An unknown dtype must say so, not fail on the alignment check with a TypeError.
        src = "\n".join([extract(self.page, "function col(buf, cols, name, n) {"), """
const out = [];
for (const c of [{ name: "x", dtype: "u16", offset: 0 }, { name: "x", dtype: "f64", offset: 4 }]) {
  try { col(new ArrayBuffer(64), [c], "x", 1); out.push("accepted"); } catch (e) { out.push(e.message); }
}
process.stdout.write(JSON.stringify(out));"""])
        self.assertEqual(self.node(src), ["column x has an unknown dtype u16", "column x is misaligned"])

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = sys.argv.pop(1)
    unittest.main()
