#!/usr/bin/env python3
"""
Smoke test for the tile-flow viewer and pack.py (#286 step 4).

Usage: python3 test_viewer.py <bundle.tflow>

Packs the bundle, checks the payload is embedded and parses, checks a non-bundle is refused
with exit 2, and -- when Node.js is on PATH -- syntax-checks the viewer's script, which is the
one part of the page a broken edit silently disables.

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
sys.path.insert(0, str(HERE))
import bind  # noqa: E402
BUNDLE = None


class ViewerSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tflow-viewer-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def pack(self, *args):
        return subprocess.run([sys.executable, str(HERE / "pack.py"), *args],
                              capture_output=True, text=True)

    def test_pack_embeds_the_bundle(self):
        out = self.tmp / "view.html"
        p = self.pack(BUNDLE, "-o", str(out))
        self.assertEqual(p.returncode, 0, p.stderr)
        page = out.read_text(encoding="utf-8")
        self.assertNotIn("<!--TFLOW_EMBED-->", page)
        m = re.search(r"window\.TFLOW_EMBED = (\{.*?\});</script>", page, re.S)
        self.assertIsNotNone(m)
        payload = json.loads(m.group(1).replace("<\\/", "</"))
        manifest = json.loads(payload["manifest"])
        self.assertEqual(manifest["format"], "kpu-tflow")
        for table in manifest["tables"].values():
            self.assertIn(table["file"], payload["bins"])
        self.assertIsNotNone(payload["lod"])

    # Interpolations in HTML that are not escaped, each reviewed: numbers the page computes, its
    # own constants and colours, and the summary's values (plain, escaped where they render).
    REVIEWED_RAW = {
        '(100 * busy / Math.max(1, R.m.makespan)).toFixed(1)', '(lv.occ[i] / width).toFixed(2)',
        'TENSOR_COLORS[k % TENSOR_COLORS.length]', 'a.length', 'list.length', 'mean.toFixed(2)', 'moving',
        'collapsed.has(r.cat) ? "unfold" : "fold"', 'held.length > 12 ? "<br>…" : ""',
        'held.length ? R.res.bank[held[0]] : "—"', 'r.cap && a.length > r.cap ? " OVER CAPACITY" : ""',
        'r.cap ? "utilization" : "busy (sum)"', 'r.cap ? (100 * util).toFixed(1) + "%" : fmt(busy) + " cycles"',
        'r.derived ? " (derived binding)" : ""', 'ramp(0.7)', 'ramp(l3frac)',
        'row.capacity && lv.peak[i] > row.capacity ? " OVER CAPACITY" : ""', 'row.modelled ? "" : " (not modelled)"',
        's.reads', 's.writes', 'k', 'm.device', 'm.device_label', 'm.l3_banks > 1 ? "s" : ""',
        # Conditionals and lists whose every output piece is escaped or a number:
        'b.clock ? " (" + esc(b.clock) + " clock)" : ""',
        'l3cap ? " / " + fmt(l3cap) : ""',
        'now >= 0 ? opText(R.co.op[now]) : "idle"',
        'now >= 0 ? tileText(R.res.tile[now]) : "free"',
        'i >= 0 ? tileText(R.res.tile[i]) + " (bank " + R.res.bank[i] + ")" : "free"',
        # Version 6: the staged L2 copies and the fabric's chains (tiles through tileText, numbers
        # through fmt, the page's own phase names through esc).
        'i >= 0 ? tileText(R.l2.tile[i]) + " staged, fed " + fmt(R.l2.feeds[i]) + "x" : "free"',
        'c >= 0 ? tileText(R.ch.tile[c]) + " " + esc(chainPhase(c, h.t)) : "idle"',
        'now >= 0 ? tileText(R.l2.tile[now]) : "free"',
        'now >= 0 ? tileText(R.ch.tile[now]) + " " + esc(chainPhase(now, T)) : "idle"',
        'row.kind === "cf_in" ? "operands in [in0, in1)" : "results out [in1, out1)"',
        'mine.length > 12 ? "<br>…" : ""',
        'held.slice(0, 12).map(i => `${tileText(R.l2.tile[i])}: ${fmt(R.l2.t0[i])}–${fmt(R.l2.t1[i])}, fed ${fmt(R.l2.feeds[i])}x`).join("<br>") || "never staged into"',
        'mine.slice(0, 12).map(c => `${tileText(R.ch.tile[c])}: ${fmt(R.ch.calls[c])} calls on ${fmt(R.ch.rows[c])}x${fmt(R.ch.cols[c])} PEs, in ${fmt(R.ch.in0[c])}–${fmt(R.ch.in1[c])}, out by ${fmt(R.ch.out1[c])}`).join("<br>") || "no chains"',
        'pooled || "No events are recorded against this block at " + esc(R.m.level) + "."',
        'r.cap ? " of " + esc(r.cap) : ""',
        'row.capacity ? " of " + fmt(row.capacity) : ""',
        'near.map(i => `[${fmt(R.co.t0[i])}, ${fmt(R.co.t1[i])}) ${opText(R.co.op[i])}`).join("<br>") || "none"',
        'held.slice(0, 12).map(i => `${tileText(R.res.tile[i])}: ${fmt(R.res.t0[i])}–${fmt(R.res.t1[i])}`).join("<br>") || "never used"',
        'a.slice(0, 8).map(([, , i]) => r.kind === "l3" ? tileText(R.res.tile[i]) : r.kind === "cf" ? opText(R.co.op[i]) : transitText(i)).join("<br>") || "nothing active at the cursor"',
    }
    SAFE_CALL = re.compile(r"^(esc|fmt|hex|tileText|opText|transitText)\(")

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

    def test_the_trusted_helpers_escape_the_names_they_write(self):
        # SAFE_CALL trusts tileText, opText and transitText to return escaped HTML: each must write
        # a tile's operand name (the bundle's string) only through esc().
        page = (HERE / "index.html").read_text(encoding="utf-8")
        for fn in ("tileText", "opText", "transitText"):
            body = re.search(r"function " + fn + r"\(.*?\n\}", page, re.S).group(0)
            self.assertNotRegex(body, r"\$\{(?:tile|t)\[0\]\}", f"{fn} writes an operand name unescaped")
            self.assertIn("esc(", body, f"{fn} escapes nothing")

    def test_bundle_strings_written_as_html_are_escaped(self):
        # A packed page is shared, and a bundle, its binding and a floorplan are untrusted input.
        # Every ${...} in a statement that writes HTML (innerHTML or a tooltip) must be escaped
        # (esc, or a helper that escapes or returns a number), or be a reviewed safe expression.
        # A new raw interpolation fails here until it is escaped or reviewed.
        page = (HERE / "index.html").read_text(encoding="utf-8")
        stmts = re.findall(r"(?:innerHTML\s*=|showTip\(e,).*?;\n", page, re.S)
        self.assertGreater(len(stmts), 8)
        for st in stmts:
            for e in self.interpolations(st):
                if self.SAFE_CALL.match(e) or e in self.REVIEWED_RAW:
                    continue
                self.fail(f"unescaped interpolation in HTML: ${{{e[:100]}}} -- escape it or review it into REVIEWED_RAW")
            for m in re.finditer(r"\+\s*([A-Za-z_][\w.]*\.(?:name|kind|clock|level|device|source))\s*\+", st):
                self.fail(f"{m.group(1)} concatenated unescaped in: {st[:140]}")

    def test_viewer_reads_the_version_the_writer_writes(self):
        # The page refuses any other manifest version, so a format bump that the viewer does not
        # follow would leave it refusing every new bundle -- invisible to a syntax check.
        written = json.loads((Path(BUNDLE) / "manifest.json").read_text(encoding="utf-8"))["version"]
        page = (HERE / "index.html").read_text(encoding="utf-8")
        accepted = re.search(r"const VERSIONS = \[([\d, ]+)\]", page)
        self.assertIsNotNone(accepted)
        self.assertIn(written, [int(v) for v in accepted.group(1).split(",")])

    def _copy_with_binding(self, identity):
        d = self.tmp / "bound.tflow"
        shutil.copytree(BUNDLE, d)
        (d / "binding.json").write_text(json.dumps({"format": "kpu-tflow-binding", "version": 2,
                                                    "bundle": identity}), encoding="utf-8")
        return d

    def _embedded_binding(self, bundle):
        out = self.tmp / "view.html"
        p = self.pack(str(bundle), "-o", str(out))
        self.assertEqual(p.returncode, 0, p.stderr)
        m = re.search(r"window\.TFLOW_EMBED = (\{.*?\});</script>", out.read_text(encoding="utf-8"), re.S)
        return json.loads(m.group(1).replace("<\\/", "</"))["binding"], p.stderr

    def test_a_binding_of_this_run_is_embedded(self):
        manifest = json.loads((Path(BUNDLE) / "manifest.json").read_text(encoding="utf-8"))
        binding, _ = self._embedded_binding(self._copy_with_binding(bind.bundle_identity(manifest)))
        self.assertIsNotNone(binding)

    def test_a_binding_of_another_run_is_left_out(self):
        # A binding.json left behind by an earlier run in the same folder indexes the wrong rows.
        manifest = json.loads((Path(BUNDLE) / "manifest.json").read_text(encoding="utf-8"))
        stale = dict(bind.bundle_identity(manifest), makespan=manifest["makespan"] + 1)
        binding, err = self._embedded_binding(self._copy_with_binding(stale))
        self.assertIsNone(binding)
        self.assertIn("different run", err)

    def test_viewer_and_bind_agree_on_the_identity(self):
        # index.html and bind.py each compute the identity; they must produce the same object,
        # or the viewer would ignore every good binding (or accept a stale one).
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js not on PATH")
        page = (HERE / "index.html").read_text(encoding="utf-8")
        fn = re.search(r"function bundleIdentity\(m\) \{.*?\n\}", page, re.S)
        self.assertIsNotNone(fn)
        manifest = (Path(BUNDLE) / "manifest.json").read_text(encoding="utf-8")
        js = self.tmp / "identity.js"
        js.write_text(fn.group(0) + f"\nprocess.stdout.write(JSON.stringify(bundleIdentity({manifest})));",
                      encoding="utf-8")
        p = subprocess.run([node, str(js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        want = json.dumps(bind.bundle_identity(json.loads(manifest)), separators=(",", ":"))
        self.assertEqual(p.stdout, want)

    def test_a_port_row_is_one_block_per_bus(self):
        # The page drew a port against "engines attached", so eight concurrent transfers read as
        # a legal 100%. It must draw two buses, each against a capacity of one block.
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js not on PATH")
        page = (HERE / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("capacity = engines attached", page)
        fn = re.search(r"function portBusRows\(name, label, ivs, hop\) \{.*?\n\}", page, re.S)
        self.assertIsNotNone(fn)
        js = self.tmp / "ports.js"
        js.write_text(fn.group(0) + """
const hop = [0, 0, 5, 0];
const rows = portBusRows("t4/noc/port[2]", "port[2]", [[0, 10, 0], [2, 12, 1], [5, 9, 2], [20, 30, 3]], hop);
process.stdout.write(JSON.stringify(rows.map(r => [r.name, r.cap, r.ivs.length, r.extra.block])));""",
                      encoding="utf-8")
        p = subprocess.run([node, str(js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout), [["t4/noc/port[2]#inject", 1, 3, "t4/noc/port[2]"],
                                                ["t4/noc/port[2]#eject", 1, 1, "t4/noc/port[2]"]])

    def test_a_port_opens_its_busy_bus(self):
        # Clicking a port whose ejection bus is busy opened the idle injection row.
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js not on PATH")
        page = (HERE / "index.html").read_text(encoding="utf-8")
        fn = re.search(r"function pickBlockRow\(rows, block, t\) \{.*?\n\}", page, re.S)
        self.assertIsNotNone(fn)
        js = self.tmp / "pick.js"
        js.write_text(fn.group(0) + """
const rows = [{ header: true }, { name: "p#inject", block: "p", ivs: [[0, 5]] },
              { name: "p#eject", block: "p", ivs: [[10, 20]] }];
process.stdout.write(JSON.stringify([pickBlockRow(rows, "p", 12).name,
                                     pickBlockRow(rows, "p", 2).name,
                                     pickBlockRow(rows, "p", 30).name]));""", encoding="utf-8")
        p = subprocess.run([node, str(js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout), ["p#eject", "p#inject", "p#inject"])

    def test_a_non_bundle_is_refused(self):
        p = self.pack(str(self.tmp), "-o", str(self.tmp / "x.html"))
        self.assertEqual(p.returncode, 2)

    def test_viewer_script_parses(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js not on PATH")
        page = (HERE / "index.html").read_text(encoding="utf-8")
        scripts = re.findall(r"<script>\n(.*?)\n</script>", page, re.S)
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
