#!/usr/bin/env python3
"""trace_validator.py's INV-006 (#537): a REFRESH belongs to no transaction, and carries txn_id 0.

    test_trace_validator.py

Runs the validator on small synthetic traces: one read (ACTIVATE then READ, tRCD apart) and a
refresh, with the refresh's txn_id 0 (valid), a request's id, missing, or null (each invalid).
Standard library only.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

VALIDATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_validator.py")


def event(name, txn, issue, complete, bank=1):
    args = {"cycle_issue": issue, "cycle_complete": complete}
    if txn is not ...:
        args["txn_id"] = txn
    return {"ph": "X", "pid": 1, "tid": bank, "name": name, "ts": issue / 3.2e3,
            "dur": (complete - issue) / 3.2e3, "args": args}


BASE = [
    {"ph": "M", "pid": 1, "tid": 1, "name": "thread_name", "args": {"name": "bank 0"}},
    event("ACTIVATE", 1, 0, 14),
    event("READ", 1, 14, 36),
    event("REFRESH", 0, 300, 440),
]


def validate(trace):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.json")
        with open(path, "w") as f:   # closed before the validator reads it
            json.dump(trace, f)
        p = subprocess.run([sys.executable, VALIDATOR, path], capture_output=True, text=True)
        return p.returncode, p.stdout


def with_refresh_id(txn):
    t = copy.deepcopy(BASE)
    t[-1] = event("REFRESH", txn, 300, 440)
    return t


class Inv006(unittest.TestCase):
    def test_a_refresh_with_no_transaction_passes(self):
        code, out = validate(BASE)
        self.assertEqual(code, 0, out)

    def test_a_refresh_with_a_requests_id(self):
        code, out = validate(with_refresh_id(1))
        self.assertEqual(code, 1, out)
        self.assertIn("INV-006", out)

    def test_a_refresh_without_an_id(self):
        code, out = validate(with_refresh_id(...))
        self.assertEqual(code, 1, out)
        self.assertIn("INV-006", out)

    def test_a_refresh_with_a_null_id(self):
        code, out = validate(with_refresh_id(None))
        self.assertEqual(code, 1, out)
        self.assertIn("INV-006", out)

    def test_other_events_without_an_id_are_still_skipped(self):
        t = copy.deepcopy(BASE)
        t.append(event("BUS BUSY", ..., 14, 36))
        code, out = validate(t)
        self.assertEqual(code, 0, out)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"])
