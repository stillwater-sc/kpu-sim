#!/usr/bin/env python3
"""
Tests for the read-only debug server (docs/plans/cycle-mflow-and-debug-server.md steps 7 and 8):
the server on a free loopback port over a temporary root holding a run directory, a bare .mflow
bundle, a trace, a file outside the root and a symlink to it; and, over a root of its own, the
live updates: a run written while the server runs is listed and its event is sent.

Usage: python3 test_kpu_debug_server.py [-v]

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kpu_debug import core  # noqa: E402
from kpu_debug import server as server_module  # noqa: E402
from kpu_debug.server import HOST, Live, make_server, parse_range  # noqa: E402

MANIFEST = {"format": "kpu-mflow", "version": 4, "tables": {}}
SUMMARY = b"memory subsystem: ...\n"


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="kpu-debug-server-"))
        root = cls.tmp / "runs"
        run = root / "2026-11-25-mm64"
        (run / "mem.mflow").mkdir(parents=True)
        (run / "mem.mflow" / "manifest.json").write_text(json.dumps(MANIFEST))
        (run / "mem.mflow" / "bursts.bin").write_bytes(bytes(range(256)) * 4)
        (run / "summary.txt").write_bytes(SUMMARY)   # bytes: text mode writes \r\n on Windows
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
        self.assertEqual(r.getheader("Content-Length"), str(len(SUMMARY)))

    def test_it_never_writes(self):
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            with self.subTest(method=method):
                status, r, _ = self.get("/files/2026-11-25-mm64/summary.txt", method=method)
                self.assertEqual(status, 405)
                self.assertEqual(r.getheader("Allow"), "GET, HEAD")
        self.assertEqual((self.root / "2026-11-25-mm64" / "summary.txt").read_bytes(), SUMMARY)

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


    # -- Open in Perfetto ----------------------------------------------------
    def test_every_trace_has_an_open_in_perfetto_button(self):
        _, _, body = self.get("/")
        page = body.decode("utf-8")
        # The run's Perfetto export and the bare DRAM trace; nothing else is a trace.
        buttons = re.findall(r'<button type="button" class="perfetto" data-url="([^"]*)" data-name="([^"]*)">', page)
        self.assertEqual(sorted(buttons), [("/files/2026-11-25-mm64/trace.json", "2026-11-25-mm64/perfetto"),
                                           ("/files/dram.json", "dram.json/dram.json")])
        # The trace goes to Perfetto's origin only, after its PONG, and from the tab the click opened.
        self.assertIn('const PERFETTO = "https://ui.perfetto.dev";', page)
        self.assertIn('ev.origin !== PERFETTO || ev.source !== win || ev.data !== "PONG"', page)
        self.assertIn("win.postMessage({ perfetto: { buffer,", page)
        self.assertIn("}, PERFETTO, [buffer]);", page)   # transferred, not copied

    def test_each_row_names_its_run_for_the_live_list(self):
        _, _, body = self.get("/")
        self.assertIn(b'<tr data-id="2026-11-25-mm64">', body)
        self.assertIn(b'<div class="panel" id="runs">', body)
        self.assertIn(b'new EventSource("/api/events")', body)
        # Every connection refetches the list, the first included: events before it are lost.
        self.assertIn(b'es.onopen = () => { live.textContent = "Live: new runs appear here."; refresh(); };', body)
        # An older refetch never replaces a newer one.
        self.assertIn(b"if (!next || n < shown) return;", body)

    def test_a_run_has_a_version_from_its_manifest(self):
        _, _, body = self.get("/api/runs")
        runs = {x["id"]: x for x in json.loads(body)["runs"]}
        st = (self.root / "2026-11-25-mm64" / "run.json").stat()
        self.assertEqual(runs["2026-11-25-mm64"]["version"], f"{st.st_mtime_ns:x}-{st.st_size:x}")
        st = (self.root / "bare.mflow" / "manifest.json").stat()
        self.assertEqual(runs["bare.mflow"]["version"], f"{st.st_mtime_ns:x}-{st.st_size:x}")


def remove_tree(d):
    """shutil.rmtree, retried: on Windows a file the poller has open for a moment cannot be removed."""
    for _ in range(50):
        try:
            shutil.rmtree(d)
            return
        except PermissionError:
            time.sleep(0.05)
    shutil.rmtree(d)


def run_json(device, status="ok", started="2026-11-25T12:00:00Z"):
    return json.dumps({"format": "kpu-run", "version": 1, "kind": "memory-subsystem", "status": status,
                       "device": device, "started": started, "artifacts": {"summary": "summary.txt"}})


class ChangesTest(unittest.TestCase):
    """core.changes: the events between two discoveries."""

    def run_(self, id_, version):
        return {"id": id_, "version": version, "artifacts": []}

    def test_added_changed_and_removed(self):
        before = [self.run_("a", "1"), self.run_("b", "1"), self.run_("c", "1")]
        after = [self.run_("d", "1"), self.run_("b", "2"), self.run_("a", "1")]
        self.assertEqual([(e, r["id"]) for e, r in core.changes(before, after)],
                         [("run-added", "d"), ("run-changed", "b"), ("run-removed", "c")])
        # A removed run is sent as it last was.
        self.assertIs(core.changes(before, after)[-1][1], before[2])

    def test_a_run_replaced_while_it_is_read_is_seen_changed_on_the_next_poll(self):
        tmp = Path(tempfile.mkdtemp(prefix="kpu-debug-race-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        d = tmp / "mm64"
        d.mkdir()
        (d / "run.json").write_text(run_json("s1", "running"))
        real = core._json

        def replaced_after_read(path):   # kpu-run renames a new run.json in, just after this read
            got = real(path)
            if Path(path).name == "run.json" and got.get("status") == "running":
                (d / "run.json.tmp").write_text(run_json("s1", "ok") + " ")
                os.replace(d / "run.json.tmp", d / "run.json")
            return got

        core._json = replaced_after_read
        try:
            first = core.discover_runs(tmp)
        finally:
            core._json = real
        self.assertEqual(first[0]["status"], "running")
        second = core.discover_runs(tmp)
        self.assertEqual([(e, r["status"]) for e, r in core.changes(first, second)], [("run-changed", "ok")])

    def test_no_change_is_no_event(self):
        runs = [self.run_("a", "1")]
        self.assertEqual(core.changes(runs, [dict(r) for r in runs]), [])


class LiveTest(unittest.TestCase):
    """A run written while the server runs appears in /api/runs, and an event is sent (step 8)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kpu-debug-live-"))
        self.root = self.tmp / "runs"
        self.root.mkdir()
        self.srv = make_server(self.root, 0, poll=0.05)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def stream(self):
        c = http.client.HTTPConnection(HOST, self.port, timeout=10)
        c.request("GET", "/api/events")
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("Content-Type"), "text/event-stream; charset=utf-8")
        self.assertEqual(r.getheader("Cache-Control"), "no-store")
        self.addCleanup(c.close)
        self.assertEqual(r.readline(), b"retry: 2000\n")   # sent once the stream is subscribed
        self.assertEqual(r.readline(), b"\n")
        return r

    def next_event(self, r):
        """The next event on stream `r`, as (name, data); comments are skipped."""
        name, data = None, None
        while True:
            line = r.readline()
            self.assertTrue(line, "the stream ended")
            line = line.decode("utf-8").rstrip("\n")
            if line.startswith(":"):
                continue
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
            elif line == "" and name:
                return name, data

    def runs(self):
        c = http.client.HTTPConnection(HOST, self.port, timeout=10)
        c.request("GET", "/api/runs")
        body = c.getresponse().read()
        c.close()
        return {x["id"]: x for x in json.loads(body)["runs"]}

    def write_run(self, name, status="ok"):
        d = self.root / name
        d.mkdir(exist_ok=True)
        (d / "summary.txt").write_bytes(SUMMARY)
        tmp = d / "run.json.tmp"   # as kpu-run writes it: whole, then renamed into place
        tmp.write_text(run_json("s1", status))
        for _ in range(50):   # on Windows, not while the poller has run.json open for a moment
            try:
                os.replace(tmp, d / "run.json")
                break
            except PermissionError:
                time.sleep(0.05)
        else:
            os.replace(tmp, d / "run.json")
        return d

    def test_a_run_written_while_serving_is_listed_and_sent(self):
        r = self.stream()
        self.assertEqual(self.runs(), {})
        self.write_run("mm64", status="running")
        name, data = self.next_event(r)
        self.assertEqual((name, data["id"], data["run"]["status"]), ("run-added", "mm64", "running"))
        self.assertEqual(data["run"]["artifacts"][0]["url"], "/files/mm64/summary.txt")
        self.assertEqual(self.runs()["mm64"]["status"], "running")
        # Rewritten when it finishes: a change, with the run's new manifest.
        time.sleep(0.01)   # a distinct mtime on coarse filesystems; the size differs anyway
        self.write_run("mm64", status="ok")
        name, data = self.next_event(r)
        self.assertEqual((name, data["id"], data["run"]["status"]), ("run-changed", "mm64", "ok"))
        self.assertEqual(self.runs()["mm64"]["status"], "ok")
        # Removed: sent as it last was, so a view can tell it was its run.
        remove_tree(self.root / "mm64")
        name, data = self.next_event(r)
        self.assertEqual((name, data["id"], data["run"]["status"]), ("run-removed", "mm64", "ok"))
        self.assertEqual(self.runs(), {})

    def test_a_bare_bundle_written_while_serving_is_sent(self):
        r = self.stream()
        b = self.root / "memsim.mflow"
        b.mkdir()
        (b / "manifest.json").write_text(json.dumps(MANIFEST))
        name, data = self.next_event(r)
        self.assertEqual((name, data["id"], data["run"]["kind"]), ("run-added", "memsim.mflow", "mflow"))

    def test_every_stream_is_sent_every_event(self):
        a, b = self.stream(), self.stream()
        self.write_run("two")
        self.assertEqual(self.next_event(a)[1]["id"], "two")
        self.assertEqual(self.next_event(b)[1]["id"], "two")

    def test_head_opens_no_stream(self):
        c = http.client.HTTPConnection(HOST, self.port, timeout=10)
        c.request("HEAD", "/api/events")
        r = c.getresponse()
        self.assertEqual((r.status, r.read()), (200, b""))
        c.close()
        self.assertEqual(self.srv.live.subscribers, set())

    def test_the_stream_ends_when_the_server_closes(self):
        r = self.stream()
        self.srv.live.close()
        deadline = time.monotonic() + 5
        while r.readline():
            self.assertLess(time.monotonic(), deadline)
        self.assertEqual(self.srv.live.subscribers, set())

    def test_a_client_that_goes_is_unsubscribed(self):
        c = http.client.HTTPConnection(HOST, self.port, timeout=10)
        c.request("GET", "/api/events")
        c.getresponse().readline()
        self.assertEqual(len(self.srv.live.subscribers), 1)
        c.close()
        # Its handler finds out on its next write: an event, or the keep-alive.
        self.write_run("after")
        deadline = time.monotonic() + 5
        while self.srv.live.subscribers and time.monotonic() < deadline:
            self.write_run("after-" + str(int((deadline - time.monotonic()) * 100)))
            time.sleep(0.1)
        self.assertEqual(self.srv.live.subscribers, set())


class StalledClientTest(unittest.TestCase):
    def test_a_client_that_stops_reading_is_dropped(self):
        tmp = Path(tempfile.mkdtemp(prefix="kpu-debug-stall-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        saved = server_module.WRITE_TIMEOUT
        server_module.WRITE_TIMEOUT = 0.5
        self.addCleanup(setattr, server_module, "WRITE_TIMEOUT", saved)
        srv = make_server(tmp, 0, poll=60)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        c = http.client.HTTPConnection(HOST, srv.server_address[1], timeout=10)
        c.request("GET", "/api/events")
        r = c.getresponse()   # the headers; from here on it reads nothing, and stays connected
        self.addCleanup(c.close)
        self.addCleanup(r.close)
        deadline = time.monotonic() + 5
        while not srv.live.subscribers and time.monotonic() < deadline:
            time.sleep(0.01)
        big = {"id": "x", "pad": "x" * (1 << 20)}
        # Enough to fill both sockets' buffers; then the handler's write blocks, and times out.
        for _ in range(64):
            srv.live.publish(("run-changed", big))
        deadline = time.monotonic() + 15
        while srv.live.subscribers and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(srv.live.subscribers, set())


class BacklogTest(unittest.TestCase):
    def test_a_subscriber_that_falls_behind_is_dropped(self):
        tmp = Path(tempfile.mkdtemp(prefix="kpu-debug-backlog-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        live = Live(tmp)   # not started: publish by hand
        slow, fast = live.subscribe(), live.subscribe()
        for i in range(server_module.BACKLOG):
            live.publish(("run-added", {"id": str(i)}))
            fast.events.get_nowait()
        self.assertFalse(slow.dropped)
        live.publish(("run-added", {"id": "one too many"}))
        self.assertTrue(slow.dropped)
        self.assertEqual(live.subscribers, {fast})
        self.assertEqual(fast.events.get_nowait()[1]["id"], "one too many")


if __name__ == "__main__":
    unittest.main()
