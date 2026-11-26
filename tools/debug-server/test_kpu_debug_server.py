#!/usr/bin/env python3
"""
Tests for the read-only debug server (docs/plans/cycle-mflow-and-debug-server.md step 7): the
server on a free loopback port over a temporary root holding a run directory, a bare .mflow
bundle, a trace, a file outside the root and a symlink to it.

Usage: python3 test_kpu_debug_server.py [-v]

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kpu_debug import core  # noqa: E402
from kpu_debug.server import HOST, make_server, parse_range  # noqa: E402

MANIFEST = {"format": "kpu-mflow", "version": 4, "tables": {}}


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="kpu-debug-server-"))
        root = cls.tmp / "runs"
        run = root / "2026-11-25-mm64"
        (run / "mem.mflow").mkdir(parents=True)
        (run / "mem.mflow" / "manifest.json").write_text(json.dumps(MANIFEST))
        (run / "mem.mflow" / "bursts.bin").write_bytes(bytes(range(256)) * 4)
        (run / "summary.txt").write_text("memory subsystem: ...\n")
        (run / "counters.json").write_text("{}")
        (run / "trace.json").write_text('{"traceEvents": [{"ph": "X", "name": "x"}]}')
        (run / "run.json").write_text(json.dumps({
            "format": "kpu-run", "version": 1, "kind": "memory-subsystem", "status": "ok", "device": "s1",
            "started": "2026-11-25T10:00:00Z", "seconds": 1.0, "build": "0.9.0", "git": "unavailable",
            "artifacts": {"summary": "summary.txt", "counters": "counters.json", "perfetto": "trace.json",
                          "mflow": "mem.mflow", "missing": "gone.json", "escape": "../../outside.txt"}}))
        (root / "bare.mflow").mkdir()
        (root / "bare.mflow" / "manifest.json").write_text(json.dumps(MANIFEST))
        (root / "dram.json").write_text('[{"ph": "X", "name": "ACTIVATE", "cat": "LPDDR5_BANK"}]')
        (root / "notes.json").write_text('{"just": "data"}')
        # Markup in a manifest's field (portable: Windows forbids < and > in file names).
        hostile = root / "hostile"
        hostile.mkdir()
        (hostile / "run.json").write_text(json.dumps({
            "format": "kpu-run", "version": 1, "kind": "memory-subsystem", "status": "ok",
            "device": "<img src=x onerror=alert(1)>", "started": "2026-11-24T00:00:00Z", "artifacts": {}}))
        (cls.tmp / "outside.txt").write_text("secret\n")
        # A run manifest and a bundle manifest outside the root, for symlinks to point at.
        elsewhere = cls.tmp / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "run.json").write_text(json.dumps({
            "format": "kpu-run", "version": 1, "kind": "memory-subsystem", "status": "ok",
            "device": "outside-device", "started": "2026-11-26T00:00:00Z", "artifacts": {}}))
        (elsewhere / "manifest.json").write_text(json.dumps(MANIFEST))
        cls.symlink = False
        try:
            os.symlink(cls.tmp / "outside.txt", root / "link.txt")
            os.symlink(cls.tmp, root / "linkdir")
            (root / "linked-run").mkdir()
            os.symlink(elsewhere / "run.json", root / "linked-run" / "run.json")
            (root / "linked.mflow").mkdir()
            os.symlink(elsewhere / "manifest.json", root / "linked.mflow" / "manifest.json")
            cls.symlink = True
        except (OSError, NotImplementedError):
            pass
        cls.root = root
        cls.srv = make_server(root, 0)
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def get(self, path, method="GET", headers=None):
        c = http.client.HTTPConnection(HOST, self.port, timeout=10)
        c.request(method, path, headers=headers or {})
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, r, body

    # -- binding ---------------------------------------------------------------
    def test_it_listens_on_loopback_only(self):
        self.assertEqual(self.srv.server_address[0], "127.0.0.1")
        self.assertNotIn("host", make_server.__code__.co_varnames)   # no way to bind another address

    # -- the runs ------------------------------------------------------------
    def test_runs_are_discovered_with_their_artifacts_and_viewers(self):
        status, r, body = self.get("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(r.getheader("Content-Type"), "application/json")
        runs = {x["id"]: x for x in json.loads(body)["runs"]}
        run = runs["2026-11-25-mm64"]
        self.assertEqual((run["kind"], run["status"], run["device"]), ("memory-subsystem", "ok", "s1"))
        arts = {a["name"]: a for a in run["artifacts"]}
        # A named artifact that is missing, or outside the root, is not listed.
        self.assertEqual(set(arts), {"summary", "counters", "perfetto", "mflow"})
        self.assertEqual(arts["mflow"]["url"], "/files/2026-11-25-mm64/mem.mflow/")
        self.assertEqual(arts["mflow"]["views"][0]["href"],
                         "/view/memflow/index.html?bundle=/files/2026-11-25-mm64/mem.mflow/")
        # Nothing inside a run directory is listed again as a run of its own.
        self.assertFalse(any(k.startswith("2026-11-25-mm64/") for k in runs))
        # Bare artifacts: a bundle, a DRAM trace (to the LPDDR5 view); a JSON file that is no trace is not a run.
        self.assertEqual(runs["bare.mflow"]["kind"], "mflow")
        self.assertEqual(runs["dram.json"]["kind"], "dram-trace")
        self.assertEqual(runs["dram.json"]["artifacts"][0]["views"][0]["viewer"], "lpddr5")
        self.assertNotIn("notes.json", runs)

    def test_symlinks_are_not_followed_into_runs(self):
        if not self.symlink:
            self.skipTest("no symlinks here")
        runs = core.discover_runs(self.root)
        self.assertFalse(any(r["id"].startswith(("linkdir", "link.txt")) for r in runs))
        # A run.json or a bundle manifest that is a link to a file outside the root is not read.
        self.assertFalse(any(r["id"] in ("linked-run", "linked.mflow") for r in runs))
        status, _, body = self.get("/api/runs")
        self.assertNotIn(b"outside-device", body)

    def test_a_directory_named_with_markup_is_escaped(self):
        # Where the filesystem allows < and > in a name (not Windows): the name is escaped too.
        d = self.root / "named" / "<script>alert(1)<"
        try:
            d.mkdir(parents=True)
        except OSError:
            self.skipTest("this filesystem does not allow < and > in a name")
        try:
            (d / "x.mflow").mkdir()
            (d / "x.mflow" / "manifest.json").write_text(json.dumps(MANIFEST))
            status, _, body = self.get("/")
            page = body.decode("utf-8")
            self.assertIn("&lt;script&gt;alert(1)&lt;", page)
            self.assertNotIn("<script>alert(1)", page)
        finally:
            shutil.rmtree(self.root / "named", ignore_errors=True)

    def test_a_large_trace_is_classified_from_its_prefix(self):
        # Only the first 64 KiB is read to tell a trace from other JSON: a large trace past that
        # point is still a trace, and reading it costs no more than its prefix.
        big = self.tmp / "big.json"
        big.write_text('{"traceEvents": [{"ph": "X", "name": "x"}' + ', {"ph": "X"}' * 400000 + "]}")
        self.assertGreater(big.stat().st_size, 1 << 20)
        self.assertEqual(core._trace_kind(big), "trace")

    def test_the_index_escapes_what_it_lists(self):
        status, r, body = self.get("/")
        self.assertEqual(status, 200)
        page = body.decode("utf-8")
        self.assertIn("2026-11-25-mm64", page)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", page)
        self.assertNotIn("<img src=x", page)
        # A bundle's name links to its manifest: the server serves files, not directories.
        self.assertIn('href="/files/2026-11-25-mm64/mem.mflow/manifest.json"', page)
        self.assertEqual(r.getheader("X-Content-Type-Options"), "nosniff")

    # -- files ---------------------------------------------------------------
    def test_a_file_under_the_root_is_served_with_its_type(self):
        for path, ctype in (("/files/2026-11-25-mm64/summary.txt", "text/plain; charset=utf-8"),
                            ("/files/2026-11-25-mm64/mem.mflow/manifest.json", "application/json"),
                            ("/files/2026-11-25-mm64/mem.mflow/bursts.bin", "application/octet-stream")):
            with self.subTest(path=path):
                status, r, body = self.get(path)
                self.assertEqual(status, 200)
                self.assertEqual(r.getheader("Content-Type"), ctype)
                self.assertEqual(r.getheader("Accept-Ranges"), "bytes")

    def test_nothing_outside_the_root_is_served(self):
        for path in ("/files/../outside.txt", "/files/%2e%2e/outside.txt", "/files/%2E%2E%2Foutside.txt",
                     "/files/2026-11-25-mm64/../../outside.txt", "/files//etc/passwd", "/files/%2Fetc%2Fpasswd",
                     "/files/C:/Windows/win.ini", "/files/x%00.txt", "/files/2026-11-25-mm64",
                     "/files/2026-11-25-mm64/mem.mflow/", "/view/memflow/../../../CMakeLists.txt",
                     "/view/common/%2e%2e/memflow/index.html", "/view/nosuch/index.html", "/elsewhere"):
            with self.subTest(path=path):
                status, _, body = self.get(path)
                self.assertEqual(status, 404, body[:80])
                self.assertNotIn(b"secret", body)

    def test_a_symlink_out_of_the_root_is_refused(self):
        if not self.symlink:
            self.skipTest("no symlinks here")
        for path in ("/files/link.txt", "/files/linkdir/outside.txt"):
            with self.subTest(path=path):
                status, _, body = self.get(path)
                self.assertEqual(status, 404)
                self.assertNotIn(b"secret", body)

    def test_ranges(self):
        path = "/files/2026-11-25-mm64/mem.mflow/bursts.bin"
        status, r, body = self.get(path, headers={"Range": "bytes=10-19"})
        self.assertEqual(status, 206)
        self.assertEqual(r.getheader("Content-Range"), "bytes 10-19/1024")
        self.assertEqual(body, bytes(range(10, 20)))
        status, r, body = self.get(path, headers={"Range": "bytes=-4"})
        self.assertEqual((status, body), (206, bytes([252, 253, 254, 255])))
        status, r, body = self.get(path, headers={"Range": "bytes=1020-"})
        self.assertEqual((status, len(body)), (206, 4))
        status, r, _ = self.get(path, headers={"Range": "bytes=2000-3000"})
        self.assertEqual(status, 416)
        self.assertEqual(r.getheader("Content-Range"), "bytes */1024")
        status, _, body = self.get(path, headers={"Range": "bytes=0-1,5-6"})   # several: the whole file
        self.assertEqual((status, len(body)), (200, 1024))

    def test_range_parsing(self):
        self.assertIsNone(parse_range(None, 10))
        self.assertEqual(parse_range("bytes=0-0", 10), (0, 0))
        self.assertEqual(parse_range("bytes=5-50", 10), (5, 9))
        self.assertEqual(parse_range("bytes=-0", 10), "unsatisfiable")
        self.assertEqual(parse_range("bytes=-50", 10), (0, 9))
        self.assertIsNone(parse_range("items=0-1", 10))

    def test_head_sends_headers_only(self):
        status, r, body = self.get("/files/2026-11-25-mm64/summary.txt", method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")
        self.assertEqual(r.getheader("Content-Length"), str(len("memory subsystem: ...\n")))

    def test_it_never_writes(self):
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            with self.subTest(method=method):
                status, r, _ = self.get("/files/2026-11-25-mm64/summary.txt", method=method)
                self.assertEqual(status, 405)
                self.assertEqual(r.getheader("Allow"), "GET, HEAD")
        self.assertEqual((self.root / "2026-11-25-mm64" / "summary.txt").read_text(), "memory subsystem: ...\n")

    # -- viewers -------------------------------------------------------------
    def test_the_viewers_and_their_shared_reader_are_served(self):
        status, r, _ = self.get("/view/memflow/")
        self.assertEqual(status, 302)
        self.assertEqual(r.getheader("Location"), "/view/memflow/index.html")
        status, r, body = self.get("/view/memflow/index.html")
        self.assertEqual(status, 200)
        self.assertIn(b'<script src="../common/mflow.js"></script>', body)
        # ../common/mflow.js from /view/memflow/ is /view/common/mflow.js.
        status, r, body = self.get("/view/common/mflow.js")
        self.assertEqual(status, 200)
        self.assertEqual(r.getheader("Content-Type"), "text/javascript; charset=utf-8")
        self.assertIn(b"function parse(files)", body)
        for name, v in core.VIEWERS.items():
            with self.subTest(viewer=name):
                status, _, _ = self.get(f"/view/{name}/{v['page']}")
                self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
