#!/usr/bin/env python3
"""
Smoke test for the floorplan view and pack.py (#447).

Usage: python3 test_viewer.py <samples dir>        (tests/program/structure)

Always:
    - pack.py embeds each sample, and the embedded payload parses back to the record;
    - pack.py refuses a file that is not a record, with exit 2.
When Playwright and its Chromium are installed (pip install playwright; playwright install
chromium), it also loads each packed page in a headless browser:
    - no script error, one rectangle per placed block, one path per link with points;
    - a click on a block fills the inspector; a layer's checkbox and the presets hide what they say;
    - zoom changes the view; a record of another format is refused on the page, not drawn;
    - a record's strings cannot inject markup or script (every one is escaped).
Without Playwright those browser checks are skipped, and say so.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLES = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:                                   # the browser checks are optional
    sync_playwright = None


def pack(src, out):
    return subprocess.run([sys.executable, str(HERE / "pack.py"), str(src), "-o", str(out)],
                          capture_output=True, text=True)


def embedded(page_text):
    m = re.search(r"window\.FLOORPLAN_EMBED = (\{.*?\});</script>", page_text, re.S)
    return json.loads(m.group(1).replace("<\\/", "</")) if m else None


class PackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fpview-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pack_embeds_each_sample(self):
        for f in sorted(Path(SAMPLES).glob("*.structure.json")):
            with self.subTest(sample=f.name):
                out = self.tmp / (f.stem + ".html")
                p = pack(f, out)
                self.assertEqual(p.returncode, 0, p.stderr)
                page = out.read_text(encoding="utf-8")
                self.assertNotIn("<!--FLOORPLAN_EMBED-->", page)
                payload = embedded(page)
                self.assertIsNotNone(payload)
                self.assertEqual(payload["record"], json.loads(f.read_text()))

    def test_pack_refuses_what_is_not_a_record(self):
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"format": "kpu-tflow", "version": 6}))
        p = pack(bad, self.tmp / "x.html")
        self.assertEqual(p.returncode, 2)
        self.assertIn("kpu-structure 1", p.stderr)
        junk = self.tmp / "junk.json"
        junk.write_text("{not json")
        self.assertEqual(pack(junk, self.tmp / "y.html").returncode, 2)


@unittest.skipIf(sync_playwright is None, "Playwright is not installed: the browser checks are skipped")
class BrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fpview-b-"))
        cls.pw = sync_playwright().start()
        try:
            cls.browser = cls.pw.chromium.launch()
        except Exception as e:                        # Playwright without its browser
            cls.pw.stop()
            raise unittest.SkipTest(f"Chromium is not installed for Playwright: {e}")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def open(self, rec, name):
        src = self.tmp / (name + ".structure.json")
        src.write_text(json.dumps(rec))
        out = self.tmp / (name + ".html")
        self.assertEqual(pack(src, out).returncode, 0)
        page = self.browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(out.as_uri())
        page.wait_for_timeout(200)
        return page, errors

    def test_each_sample_draws_and_responds(self):
        for f in sorted(Path(SAMPLES).glob("*.structure.json")):
            with self.subTest(sample=f.name):
                rec = json.loads(f.read_text())
                page, errors = self.open(rec, f.stem)
                placed = [b for b in rec["blocks"] if b.get("rect") and b["kind"] != "die"]
                links = sum(1 for n in rec.get("networks", []) for ln in n.get("links", []) if ln.get("points"))
                self.assertEqual(page.eval_on_selector_all("rect.blk", "e => e.length"), len(placed))
                self.assertEqual(page.eval_on_selector_all(".net path", "e => e.length"), links)
                # A click on a block fills the inspector with it.
                pick = placed[len(placed) // 2]
                page.eval_on_selector(f'rect.blk[data-id="{pick["id"]}"]',
                                      "e => e.dispatchEvent(new MouseEvent('click', {bubbles: true}))")
                self.assertIn(pick["id"], page.inner_text("#insp"))
                self.assertEqual(page.eval_on_selector_all("rect.blk.sel", "e => e.length"), 1)
                if rec.get("networks"):
                    # A network's checkbox hides its paths; Blocks only hides every network.
                    first = rec["networks"][0]
                    n_first = sum(1 for ln in first["links"] if ln.get("points"))
                    page.click(f'input[data-layer="net:{first["id"]}"]')
                    self.assertEqual(page.eval_on_selector_all(".net path", "e => e.length"), links - n_first)
                    page.click('button[data-preset="blocks"]')
                    self.assertEqual(page.eval_on_selector_all(".net path", "e => e.length"), 0)
                    page.click('button[data-preset="all"]')
                    self.assertEqual(page.eval_on_selector_all(".net path", "e => e.length"), links)
                # Zoom changes the view; Fit restores it.
                vb = page.get_attribute("#die", "viewBox")
                page.click("#zin")
                self.assertNotEqual(page.get_attribute("#die", "viewBox"), vb)
                page.click("#fit")
                self.assertEqual(page.get_attribute("#die", "viewBox"), vb)
                self.assertIn("mm²", page.inner_text("#size"))
                self.assertEqual(errors, [])
                page.close()

    def test_another_format_is_refused_on_the_page(self):
        rec = json.loads(next(Path(SAMPLES).glob("*.structure.json")).read_text())
        page, errors = self.open(rec, "refuse")
        page.evaluate("load({format: 'kpu-gen-floorplan', version: 1, blocks: []}, 'old.json')")
        self.assertIn("convert it", page.inner_text("#empty"))
        self.assertTrue(page.is_hidden("#die"))
        page.evaluate("load({format: 'kpu-structure', version: 2, blocks: [{id: 'a', kind: 'die'}]}, 'v2.json')")
        self.assertIn("this viewer reads 1", page.inner_text("#empty"))
        self.assertEqual(errors, [])
        page.close()

    def test_compare_shows_the_change_in_die_area(self):
        a = json.loads((Path(SAMPLES) / "kpugen_kpu_2x2_systolic.structure.json").read_text())
        b = json.loads((Path(SAMPLES) / "kpugen_kpu_edge_hetero.structure.json").read_text())
        page, errors = self.open(a, "compare")
        page.evaluate("cmp => { S.cmp = cmp; renderSize(); }", b)
        text = page.inner_text("#size")
        d = a["floorplan"]["die"]["area"] - b["floorplan"]["die"]["area"]
        self.assertIn(f"vs {b['provenance']['configuration']}", text)
        self.assertIn(f"{d:+.3f} mm²", text)
        self.assertEqual(errors, [])
        page.close()

    def test_a_records_strings_cannot_inject(self):
        rec = copy.deepcopy(json.loads(next(Path(SAMPLES).glob("*.structure.json")).read_text()))
        evil = '"><img src=x onerror="window.__xss=1"><script>window.__xss=2</script>'
        target = next(b for b in rec["blocks"] if b.get("rect") and b["kind"] != "die")
        target["label"] = evil
        target["notes"] = evil
        rec["provenance"]["configuration"] = evil
        page, errors = self.open(rec, "inject")
        page.eval_on_selector(f'rect.blk[data-id="{target["id"]}"]',
                              "e => e.dispatchEvent(new MouseEvent('click', {bubbles: true}))")
        page.wait_for_timeout(200)
        self.assertIsNone(page.evaluate("window.__xss"))
        self.assertEqual(page.eval_on_selector_all("img", "e => e.length"), 0)
        self.assertIn("<script>", page.inner_text("#insp"))       # shown as text, not run
        page.close()


if __name__ == "__main__":
    SAMPLES = sys.argv.pop(1) if len(sys.argv) > 1 else str(HERE.parent.parent.parent / "tests" / "program" / "structure")
    unittest.main(verbosity=2)
