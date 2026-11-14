#!/usr/bin/env python3
"""
Self-test for tile_cmds_run.py (#397): the clean stream kpu-run wrote must run and match, and each
break of the slot protocol, of an address or of a value must fail. An executor that passes
everything checks nothing.

Usage: python3 test_tile_cmds_run.py <stream.json>

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tile_cmds_run as tr  # noqa: E402

STREAM = None
F = {name: i for i, name in enumerate(tr.FIELDS)}


class Run(unittest.TestCase):
    def setUp(self):
        self.doc = copy.deepcopy(STREAM)

    def find(self, type_, opcode=None, nth=0):
        hits = [i for i, c in enumerate(self.doc["commands"])
                if c[F["type"]] == type_ and (opcode is None or c[F["opcode"]] == opcode)]
        return hits[nth]

    def set(self, i, field, value):
        self.doc["commands"][i][F[field]] = value

    def fails(self, needle):
        with self.assertRaises(tr.Wrong) as e:
            tr.run(self.doc)
        self.assertIn(needle, str(e.exception))

    def test_clean_stream_runs_and_matches(self):
        n, r = tr.run(self.doc)
        self.assertEqual(n, len(self.doc["commands"]))
        self.assertGreater(r, 0)
        self.assertTrue(all(x["bytes"] for x in self.doc["results"]), "the results carry L-B's values")

    def test_compute_before_its_pair_is_ready(self):
        b = self.find(tr.LOAD, tr.TRANSPOSE)
        self.set(b, "sync", self.doc["commands"][b][F["sync"]] & ~tr.SIGNAL)
        self.fails("waits on l2 slot")

    def test_a_pair_taken_twice(self):
        c = self.find(tr.EXEC, tr.COMPUTE)
        self.set(c, "sync", self.doc["commands"][c][F["sync"]] & ~tr.RELEASE)
        self.fails("acquires l2 slot")

    def test_store_before_the_reduction_ends(self):
        c = self.find(tr.EXEC, tr.COMPUTE, 1)          # the first reduction's last step
        self.set(c, "sync", self.doc["commands"][c][F["sync"]] & ~tr.SIGNAL)
        self.fails("waits on acc slot")

    def test_a_broken_chain_changes_the_result(self):
        c = self.find(tr.EXEC, tr.COMPUTE, 1)
        self.set(c, "chain", 0)                        # restarts the reduction: drops the first step
        self.fails("results differ")

    def test_an_address_outside_l3(self):
        self.set(self.find(tr.LOAD), "dram_addr", self.doc["l3"]["bytes"])
        self.fails("past its")

    def test_a_wrong_expected_value(self):
        x = self.doc["results"][0]
        x["bytes"] = ("00" if x["bytes"][:2] != "00" else "01") + x["bytes"][2:]
        self.fails("results differ")

    def test_ids_count_from_zero(self):
        self.set(1, "id", 7)
        self.fails("has id 7")

    def test_another_version_is_unreadable(self):
        self.doc["version"] = 2
        with self.assertRaises(ValueError):
            tr.run(self.doc)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    STREAM = json.loads(Path(sys.argv.pop(1)).read_text(encoding="utf-8"))
    unittest.main()
