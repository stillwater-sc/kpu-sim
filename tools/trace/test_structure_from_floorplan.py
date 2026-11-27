#!/usr/bin/env python3
"""
test_structure_from_floorplan.py -- the converter writes a die-local record (#630's review).

Usage: python3 test_structure_from_floorplan.py <kpu-floorplan JSON>

Moves the floorplan's die and everything on it by an offset, converts both, and checks that the two
structure records' blocks and networks are the same: the record's frame starts at the die's
lower-left corner (shared-structure-record.md §2.2), wherever the die sits.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structure_from_floorplan  # noqa: E402


def moved(d: dict, dx: float, dy: float) -> dict:
    m = copy.deepcopy(d)
    m["die"][0] += dx
    m["die"][1] += dy

    def walk(b):
        b["rect"][0] += dx
        b["rect"][1] += dy
        for c in b.get("children", []):
            walk(c)

    for b in m["blocks"]:
        walk(b)
    for ln in m.get("noc", []):
        for p in ln.get("route", []):
            p[0] += dx
            p[1] += dy
    return m


def same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main(path: str) -> int:
    d = json.load(open(path))
    a = structure_from_floorplan.convert(d)
    b = structure_from_floorplan.convert(moved(d, 100.0, 50.0))
    fails = 0
    for key in ("blocks", "networks"):
        if not same(a[key], b[key]):
            print(f"FAILED: the {key} move with the die")
            fails += 1
    print(f"{'FAILED' if fails else 'passed'}: {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
