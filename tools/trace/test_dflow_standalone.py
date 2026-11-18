#!/usr/bin/env python3
"""
A standalone tile's record (#473): a compute tile with no DRAM and no DMA, whose operands the host
places in L3 before the run (systars' SystolicTop, an L2/L3 testbench).

The format already says it. An INHERIT step opens an L3 residency at the program's start, with no
process and no transaction, so E5 exempts it; a RETAIN step leaves a result in L3 past the end; and
geometry.dram is declared not modelled. This test holds the checker to that:

  - the standalone program's L-B record, with DRAM declared not modelled, passes;
  - its timed record, with DRAM not modelled and no event through DRAM, passes;
  - a record that writes the host's placement as a LOAD with no transaction fails E5, and the
    message points to INHERIT.

Usage: python3 test_dflow_standalone.py <standalone-lb.dflow> <standalone-timed.dflow> <loaded-timed.dflow>

The first two are kpu-run's records of tests/program/csp/standalone_tile_8_t4.csp (--dflow and
--dflow-lt1); the third is a timed record of any program that loads its operands.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflow_check as dc  # noqa: E402
import dflow_writer as dw  # noqa: E402

LB = TIMED = LOADED = None
NO_DRAM = {"modelled": False, "why": "a standalone tile: no DRAM; the host places its operands in L3"}
EVENT_COLUMNS = (("id", "u64"), ("parent", "u64"), ("action", "u64"), ("where", "u64"), ("src_addr", "u64"),
                 ("dst_addr", "u64"), ("tile", "u32"), ("lane", "u32"), ("row", "u32"), ("beat", "u32"),
                 ("kind", "u8"), ("process", "u8"), ("src", "u8"), ("dst", "u8"))


def standalone(src, dst):
    """A copy of `src` as a standalone tile writes it: DRAM declared not modelled, and every event
    that moves through DRAM (a load's DMA, a store's, a credit or token to the DMA) left out."""
    shutil.copytree(src, dst)
    m, T = dc.load(dst)
    E = T["events"]
    keep = [e for e in range(E["rows"]) if E["src"][e] != dc.DRAM and E["dst"][e] != dc.DRAM]
    assert all(E["parent"][e] == dc.NONE64 for e in keep), "the fixture's kept events have no parents"
    t = dw.Table("events", m["tables"]["events"]["file"])
    for name, dtype in EVENT_COLUMNS:
        t.put(name, dtype, list(range(len(keep))) if name == "id" else [E[name][e] for e in keep])
    timed = "cycle" in E
    if timed:
        t.put("cycle", "u64", [E["cycle"][e] for e in keep])
        t.put("end", "u64", [E["end"][e] for e in keep])
    m["tables"]["events"] = dw.write_table(dst, t)
    m["geometry"]["dram"] = NO_DRAM
    if timed and keep:
        latest = max(E["end"][e] for e in keep)
        m["time"]["cycles"] = latest
        m["result"]["cycles"] = latest
    (Path(dst) / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return Path(dst)


def check(path):
    p = subprocess.run([sys.executable, str(HERE / "dflow_check.py"), str(path), "--json"], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode in (0, 1) else p.stderr)


class Standalone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_program_places_with_inherit_and_keeps_with_retain(self):
        m, T = dc.load(LB)
        S = T["steps"]
        kinds = {m["kinds"][S["kind"][i]] for i in range(S["rows"])}
        self.assertIn("INHERIT", kinds)
        self.assertIn("RETAIN", kinds)
        self.assertNotIn("LOAD", kinds)
        self.assertNotIn("STORE", kinds)

    def test_its_lb_record_passes_with_no_dram(self):
        code, r = check(standalone(LB, self.tmp / "lb.dflow"))
        self.assertEqual(code, 0, r and r["violations"][:4])

    def test_its_timed_record_passes_with_no_dram(self):
        d = standalone(TIMED, self.tmp / "timed.dflow")
        m, T = dc.load(d)
        E = T["events"]
        self.assertGreater(E["rows"], 0)
        self.assertFalse([e for e in range(E["rows"]) if dc.DRAM in (E["src"][e], E["dst"][e])])
        code, r = check(d)
        self.assertEqual(code, 0, r and r["violations"][:4])

    def test_a_load_with_no_transaction_points_to_inherit(self):
        # The host's placement written as a LOAD: with DRAM not modelled it can have no DMA, so it
        # has no transaction, which E5 refuses; the message names INHERIT.
        code, r = check(standalone(LOADED, self.tmp / "loaded.dflow"))
        self.assertEqual(code, 1)
        e5 = [v["message"] for v in r["violations"] if v["invariant"] == "E5"]
        self.assertTrue(any("(LOAD) has 0 transaction events" in x and "INHERIT step" in x for x in e5), e5[:3])

    def test_with_dram_modelled_the_message_is_unchanged(self):
        # The hint is for a record with no DRAM: a loaded record missing a DMA, DRAM modelled, gets
        # the plain message.
        d = standalone(LOADED, self.tmp / "loaded_dram.dflow")
        mf = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        mf["geometry"]["dram"] = {"modelled": True}
        (d / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
        code, r = check(d)
        self.assertEqual(code, 1)
        e5 = [v["message"] for v in r["violations"] if v["invariant"] == "E5"]
        self.assertTrue(e5)
        self.assertFalse(any("INHERIT" in x for x in e5))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    LB, TIMED, LOADED = (Path(sys.argv.pop(1)).resolve() for _ in range(3))
    unittest.main(verbosity=2)
