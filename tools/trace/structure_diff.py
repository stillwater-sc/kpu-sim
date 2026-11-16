#!/usr/bin/env python3
"""
Compare two structure records, kpu-structure v1, block for block (#392).

    structure_diff.py <a.structure.json> <b.structure.json> [--rel 1e-9] [--max 20]

Everything but `provenance` is compared: the blocks by id (each field), the networks by id
(their nodes and links by id), the channels by id, and the floorplan. Numbers agree within a
relative tolerance (default 1e-9; the same geometry written by two producers differs only in
rounding). Lists that are not keyed by id (a rect, a link's points) compare element by element.

Uses: the model's native record against its converted one (CTest structure_native_t4); and,
once the RTL writes the record, the model's against the RTL's for the same configuration, where
a difference is a bug in one of them.

Exit codes: 0 the same; 1 a difference (each printed, up to --max); 2 a record cannot be read.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

KEYED = {"blocks", "networks", "nodes", "links", "channels"}


def diff(a, b, path: str, rel: float, out: list[str]) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            out.append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=rel, abs_tol=rel):
            out.append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: only in the second")
            elif k not in b:
                out.append(f"{path}.{k}: only in the first")
            elif k in KEYED and isinstance(a[k], list) and isinstance(b[k], list):
                keyed(a[k], b[k], f"{path}.{k}", rel, out)
            else:
                diff(a[k], b[k], f"{path}.{k}", rel, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", rel, out)
        return
    if a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def keyed(a: list, b: list, path: str, rel: float, out: list[str]) -> None:
    ia = {x.get("id"): x for x in a if isinstance(x, dict)}
    ib = {x.get("id"): x for x in b if isinstance(x, dict)}
    if len(ia) != len(a) or len(ib) != len(b):            # not keyed after all: compare in order
        diff(list(a), list(b), path + "[]", rel, out)
        return
    for k in [x for x in ia if x not in ib]:
        out.append(f"{path}[{k}]: only in the first")
    for k in [x for x in ib if x not in ia]:
        out.append(f"{path}[{k}]: only in the second")
    for k in ia:
        if k in ib:
            diff(ia[k], ib[k], f"{path}[{k}]", rel, out)


def compare(a: dict, b: dict, rel: float = 1e-9) -> list[str]:
    out: list[str] = []
    diff({k: v for k, v in a.items() if k != "provenance"},
         {k: v for k, v in b.items() if k != "provenance"}, "$", rel, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare two kpu-structure records block for block.")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rel", type=float, default=1e-9, help="relative tolerance for numbers")
    ap.add_argument("--max", type=int, default=20, help="differences to print")
    args = ap.parse_args(argv)
    recs = []
    for p in (args.a, args.b):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError) as e:
            print(f"structure_diff: {p}: {e}", file=sys.stderr)
            return 2
        if not isinstance(r, dict) or r.get("format") != "kpu-structure" or r.get("version") != 1:
            print(f"structure_diff: {p}: not a kpu-structure 1 record", file=sys.stderr)
            return 2
        recs.append(r)
    d = compare(*recs, rel=args.rel)
    for line in d[:args.max]:
        print(line)
    if len(d) > args.max:
        print(f"... and {len(d) - args.max} more")
    print(f"structure_diff: {'the same' if not d else f'{len(d)} difference(s)'} (provenance aside)")
    return 1 if d else 0


if __name__ == "__main__":
    sys.exit(main())
