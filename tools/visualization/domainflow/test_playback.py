#!/usr/bin/env python3
"""
The domain-flow viewer's playback speeds, in a headless browser: every speed the control offers
plays at about that many steps per second, the slowest being one step per second (for following
each event as it happens).

Usage: python3 test_playback.py <bundle.dflow>

Needs Playwright and its Chromium (pip install playwright; playwright install chromium); without
them it skips, and says so.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipIf(sync_playwright is None, "Playwright is not installed: the browser checks are skipped")
class Playback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="dflow-playback-"))
        page = cls.tmp / "view.html"
        p = subprocess.run([sys.executable, str(HERE / "pack.py"), str(BUNDLE), "-o", str(page)],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        cls.page_path = page
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

    def open(self):
        page = self.browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(self.page_path.as_uri())
        page.wait_for_timeout(500)
        return page, errors

    def test_one_step_per_second_is_offered_and_slowest(self):
        page, errors = self.open()
        values = page.eval_on_selector_all("#speed option", "os => os.map(o => +o.value)")
        self.assertIn(1, values)
        self.assertEqual(min(values), 1)
        self.assertEqual(page.eval_on_selector('#speed option[value="1"]', "o => o.textContent"), "1/s")
        self.assertEqual(errors, [])
        page.close()

    def test_one_step_per_second_plays_at_that_rate(self):
        page, errors = self.open()
        page.select_option("#speed", "1")
        start = int(page.inner_text("#pos").replace(",", ""))
        page.click("#play")
        t0 = time.monotonic()
        page.wait_for_timeout(3500)
        elapsed = time.monotonic() - t0
        page.click("#play")                              # pause
        moved = int(page.inner_text("#pos").replace(",", "")) - start
        # About one step per second: 3.5 s gives 3, allowing for the timer's start and the browser.
        self.assertGreaterEqual(moved, 2, f"{moved} steps in {elapsed:.1f} s")
        self.assertLessEqual(moved, 4, f"{moved} steps in {elapsed:.1f} s")
        self.assertEqual(errors, [])
        page.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: test_playback.py <bundle.dflow>")
    BUNDLE = sys.argv.pop(1)
    unittest.main(verbosity=2)
