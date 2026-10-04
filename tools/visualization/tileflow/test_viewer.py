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
        page = out.read_text()
        self.assertNotIn("<!--TFLOW_EMBED-->", page)
        m = re.search(r"window\.TFLOW_EMBED = (\{.*?\});</script>", page, re.S)
        self.assertIsNotNone(m)
        payload = json.loads(m.group(1).replace("<\\/", "</"))
        manifest = json.loads(payload["manifest"])
        self.assertEqual(manifest["format"], "kpu-tflow")
        for table in manifest["tables"].values():
            self.assertIn(table["file"], payload["bins"])
        self.assertIsNotNone(payload["lod"])

    def test_a_non_bundle_is_refused(self):
        p = self.pack(str(self.tmp), "-o", str(self.tmp / "x.html"))
        self.assertEqual(p.returncode, 2)

    def test_viewer_script_parses(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js not on PATH")
        page = (HERE / "index.html").read_text()
        scripts = re.findall(r"<script>\n(.*?)\n</script>", page, re.S)
        self.assertTrue(scripts)
        js = self.tmp / "viewer.js"
        js.write_text(scripts[-1])
        p = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = sys.argv.pop(1)
    unittest.main()
