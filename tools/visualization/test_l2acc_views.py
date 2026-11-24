#!/usr/bin/env python3
"""
The viewers on an L2 accumulator (#456): the domain-flow and tile-flow pages, loaded in a headless
browser on the records of a program with `acc C[:, j] in l2 init D[:, j]` (kpu-csp-gen --dataflow ws
--init bias; CTest fixture csp_l2acc.dflow and .tflow).

Usage: python3 test_l2acc_views.py <l2acc.dflow> <l2acc.tflow>

Checks:
  - tile-flow: the header counts staged copies and accumulator partials apart, as the record has
    them, and the fabric chains as a number (it read NaN: the chain table's `rows` column overwrote
    the row count); the L2 accumulator diagnostic; a compute tile's inspector lists its chains (it
    said "no chains" for the same reason); a record without #455's columns still loads.
  - domain-flow: at the first Drain into a partial, the L2 box shows it as a partial (p<slot>
    1/<merges>) and the step says overwrite; at a later one, accumulate, and its merges so far.
Needs Playwright and its Chromium (pip install playwright; playwright install chromium); without
them it skips, and says so.

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
sys.path.insert(0, str(HERE.parent / "trace"))
import dflow_check  # noqa: E402
import tflow_check  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

DFLOW = None
TFLOW = None


def pack(viewer, bundle, out):
    p = subprocess.run([sys.executable, str(HERE / viewer / "pack.py"), str(bundle), "-o", str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return out


def tflow_record(bundle):
    rec = tflow_check.load(Path(bundle))           # its tables, by name, with "<table>_rows"
    return None, rec


def lod_rows(bundle):
    return json.loads((Path(bundle) / "lod.json").read_text())["rows"]


@unittest.skipIf(sync_playwright is None, "Playwright is not installed: the browser checks are skipped")
class L2AccViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="l2acc-views-"))
        cls.pw = sync_playwright().start()
        try:
            cls.browser = cls.pw.chromium.launch()
        except Exception as e:
            cls.pw.stop()
            raise unittest.SkipTest(f"Chromium is not installed for Playwright: {e}")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def open(self, html):
        # Tall enough that the tile-flow swimlanes are on screen: the mouse reaches only what is.
        page = self.browser.new_page(viewport={"width": 1500, "height": 3000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(html.as_uri())
        page.wait_for_timeout(800)
        return page, errors

    # ---- tile-flow -------------------------------------------------------------------------
    def click_row(self, page, name):
        """Click the swimlane row called `name`, as a person finds it: hover down the plot until the
        tooltip names it (rows are grouped under category headers, so their y is not a fixed grid)."""
        box = page.locator("#lanes").bounding_box()
        x = box["x"] + 150 + 40
        for dy in range(2, int(box["height"]), 3):
            page.mouse.move(x, box["y"] + dy)
            tip = page.locator("#tip")
            if tip.is_visible() and tip.inner_text().split("\n")[0].strip() == name:
                page.mouse.click(x, box["y"] + dy)
                page.wait_for_timeout(150)
                return
        self.fail(f"no swimlane row called {name}")

    def test_tileflow_tells_partials_from_staged_copies(self):
        _m, rec = tflow_record(TFLOW)
        l2 = rec["l2_residency"]
        n = rec["l2_residency_rows"]
        partial = [i for i in range(n) if l2["partial"][i]]
        partials = len(partial)
        self.assertGreater(partials, 0, "the fixture has no partial")
        # The walker's copies (#387's optional `copy` column: 1 an operand's, 2 a result's) are
        # counted apart from the program's stagings.
        copy = l2.get("copy") or [0] * n
        staged = sum(1 for i in range(n) if not l2["partial"][i] and copy[i] == 0)
        operand = sum(1 for i in range(n) if copy[i] == 1)
        page, errors = self.open(pack("tileflow", TFLOW, self.tmp / "tf.html"))
        summary = page.inner_text("#summary")
        self.assertIn(f"{staged} staged cop", summary)
        self.assertIn(f"{operand} operand cop", summary)
        self.assertIn(f"{partials} accumulator partial", summary)
        chains = rec["chain_rows"]
        self.assertRegex(summary, rf"fabric chains\s+{chains:,}")
        self.assertNotIn("NaN", summary)
        self.assertIn("L2 ACCUMULATOR", page.inner_text("#diag").upper())
        rows = lod_rows(TFLOW)
        # A compute tile's inspector lists its chains (it said "no chains": the count read NaN).
        self.click_row(page, next(x["name"] for x in rows if x["kind"] == "cf_in"))
        self.assertIn("calls on", page.inner_text("#insp"))
        # An L2 slot that held a partial says so.
        slot = l2["slot"][partial[0]]
        self.click_row(page, next(x["name"] for x in rows
                                  if x["kind"] == "l2_slot" and re.search(rf"slot\[{slot}\]$", x["name"])))
        self.assertIn("accumulator partials", page.inner_text("#insp"))
        self.assertRegex(page.inner_text("#insp"), r"accumulator partial, \d+ merges")
        self.assertEqual(errors, [])
        page.close()

    def test_tileflow_reads_a_record_without_the_columns(self):
        old = self.tmp / "old.tflow"
        shutil.copytree(TFLOW, old)
        m = json.loads((old / "manifest.json").read_text())
        t = m["tables"]["l2_residency"]
        t["columns"] = [c for c in t["columns"] if c["name"] not in ("partial", "merges")]
        (old / "manifest.json").write_text(json.dumps(m))
        page, errors = self.open(pack("tileflow", old, self.tmp / "old.html"))
        self.assertIn("0 accumulator partials", page.inner_text("#summary"))
        self.assertEqual(errors, [])
        page.close()

    def test_tileflow_reads_partial_without_merges(self):
        # #460's review: a bundle with `partial` and no `merges` still draws, and its L2 slot
        # inspector names the partial (with zero merges) instead of throwing.
        part = self.tmp / "partial-only.tflow"
        shutil.copytree(TFLOW, part)
        m = json.loads((part / "manifest.json").read_text())
        t = m["tables"]["l2_residency"]
        t["columns"] = [c for c in t["columns"] if c["name"] != "merges"]
        (part / "manifest.json").write_text(json.dumps(m))
        _m, rec = tflow_record(TFLOW)
        l2 = rec["l2_residency"]
        slot = l2["slot"][next(i for i in range(rec["l2_residency_rows"]) if l2["partial"][i])]
        page, errors = self.open(pack("tileflow", part, self.tmp / "partial-only-tf.html"))
        self.assertIn("L2 ACCUMULATOR", page.inner_text("#diag").upper())
        self.click_row(page, next(x["name"] for x in lod_rows(TFLOW)
                                  if x["kind"] == "l2_slot" and re.search(rf"slot\[{slot}\]$", x["name"])))
        self.assertRegex(page.inner_text("#insp"), r"accumulator partial, 0 merges")
        self.assertEqual(errors, [])
        page.close()

    # ---- domain-flow -----------------------------------------------------------------------
    def test_domainflow_reads_partial_without_merges(self):
        # #460's review: each optional column defaults on its own, as the C++ reader does, so a
        # record with `partial` and no `merges` still loads and steps through a partial.
        part = self.tmp / "partial-only.dflow"
        shutil.copytree(DFLOW, part)
        m = json.loads((part / "manifest.json").read_text())
        t = m["tables"]["l2_staging"]
        t["columns"] = [c for c in t["columns"] if c["name"] != "merges"]
        (part / "manifest.json").write_text(json.dumps(m))
        _m, T = dflow_check.load(Path(DFLOW))
        first = next(i for i in range(T["steps"]["rows"]) if T["steps"]["merge"][i] == 1)
        page, errors = self.open(pack("domainflow", part, self.tmp / "partial-only.html"))
        page.fill("#jump", str(first))
        page.press("#jump", "Enter")
        page.wait_for_timeout(200)
        self.assertRegex(page.inner_text("#l2-tiles"), r"p\d+ 1/0")   # merged once so far, of none recorded
        self.assertEqual(errors, [])
        page.close()

    def test_domainflow_follows_a_partial(self):
        m, T = dflow_check.load(Path(DFLOW))
        merge = T["steps"]["merge"]
        first = next(i for i in range(T["steps"]["rows"]) if merge[i] == 1)
        later = next(i for i in range(T["steps"]["rows"]) if merge[i] == 2)
        page, errors = self.open(pack("domainflow", DFLOW, self.tmp / "df.html"))

        def at(s):
            page.fill("#jump", str(s))
            page.press("#jump", "Enter")
            page.wait_for_timeout(200)
            return page.inner_text("#l2-cap"), page.inner_text("#l2-tiles"), page.inner_text("#step-kv")

        cap, tiles, kv = at(first)
        self.assertRegex(cap, r"partials [1-9]")
        self.assertRegex(tiles, r"p\d+ 1/\d+")
        self.assertIn("(opens it)", kv)
        # The fixture's accumulator has an init (--init bias): its first merge names it, and the
        # Combiner's stages list it (#460's review).
        self.assertIn("MERGE_INIT", kv)
        self.assertRegex(kv, r"init\(D\[\d+,\d+\]\)")
        cap, tiles, kv = at(later)
        self.assertIn("accumulate", kv)
        self.assertIn("(merged into)", kv)
        self.assertRegex(tiles, r"p\d+ [2-9]/\d+")
        self.assertEqual(errors, [])
        page.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: test_l2acc_views.py <l2acc.dflow> <l2acc.tflow>")
    TFLOW = sys.argv.pop(2)
    DFLOW = sys.argv.pop(1)
    unittest.main(verbosity=2)
