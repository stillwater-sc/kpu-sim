#!/usr/bin/env python3
"""
Domain-flow record checker (docs/plans/domain-flow-debugger.md §6, step 2; #363).

Checks a .dflow bundle (written by `kpu-run --dflow`) against the invariants every L-B run of a
CSP program must satisfy, so a recorder or interpreter bug becomes a failing check instead of a
reviewing problem.

Usage:
    python3 dflow_check.py <bundle.dflow> [--json] [--verbose]

Exit codes (the same contract as tflow_check.py, mflow_check.py and trace_validator.py):
    0  every invariant holds
    1  one or more invariants are violated -- parse them and fix the root cause
    2  the bundle could not be read -- there are no violations to parse

Invariants (the plan's §6 numbering):
    D2 one step, one action   steps[i].action == i, in program order; a bundle that is not cut
                              records every action
    D3 occupancy              held_l3 after step i equals the residencies opened at or before i
                              and not yet released (a retained one stays held)
    D4 residency lifetimes    every residency opens at a step that opens it and ends once, at a
                              Release or a Retain after its open; a step that names a residency
                              names one that is open; only a refused run leaves one unended
    D5 capacity               held_l3 never exceeds the program's L3
    D6 summary complete       the depth-0 iteration covers every action and its loads and stores
                              are the run's; at each depth iterations are disjoint, ordered and
                              nested in one iteration of the depth above; with every step
                              recorded, each iteration's counts and peaks are its steps'
    D8 no time at L-B         no time column (f64, or a t_* name) in any table
    D9 placement              a step has an address exactly when it names a residency, and it is
                              its residency's; after every step each slot holds at most one open
                              or retained residency

D1 (the record is the oracle's run) and D7 (the viewer never executes) are properties of the code
that writes and reads the bundle, not of its contents.

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import array
import json
import sys
from pathlib import Path

VERSION = 1
NONE8, NONE32, NONE64 = 0xFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
DTYPES = {"u8": ("B", 1), "u16": ("H", 2), "u32": ("I", 4), "u64": ("Q", 8), "i64": ("q", 8),
          "f64": ("d", 8)}
OPENS, CLOSES, RETAINS, HAS_PLACE, PINNED = 1, 2, 4, 8, 16
RES_RETAINED = 4
LOAD, STORE = 0, 1                                    # enum Action::Kind, in order
CHANNELS = ["dram", "l3", "l2", "cf"]

SCHEMA = {
    "steps": {"action": "u64", "residency": "u64", "where": "u64", "tile": "u32",
              "held_dram": "u32", "held_l3": "u32", "held_l2": "u32", "held_cf": "u32",
              "l3_tile": "u32", "slot": "u32", "bank": "u32", "kind": "u8", "process": "u8",
              "from": "u8", "to": "u8", "flags": "u8"},
    "tiles": {"operand": "u32", "ti": "u32", "tj": "u32"},
    "residency": {"open": "u64", "close": "u64", "tile": "u32", "l3_tile": "u32", "slot": "u32",
                  "bank": "u32", "chan": "u8", "flags": "u8"},
    "where": {"line": "u32", "loop_first": "u32", "loop_count": "u16"},
    "loops": {"value": "i64", "hi": "i64", "var": "u32", "line": "u32"},
    "iteration": {"value": "i64", "first": "u64", "last": "u64", "actions": "u64", "loads": "u64",
                  "stores": "u64", "line": "u32", "peak_dram": "u32", "peak_l3": "u32",
                  "peak_l2": "u32", "peak_cf": "u32", "depth": "u16"},
}


class Unreadable(Exception):
    pass


def _int(v, what, lo=0):
    if not isinstance(v, int) or isinstance(v, bool) or v < lo:
        raise Unreadable(f"{what} is {v!r}, not an integer >= {lo}")
    return v


def _manifest(m):
    """The manifest fields check() reads, typed: a malformed one is unreadable (exit 2), never a
    traceback that exits 1 as if an invariant had failed."""
    try:
        for k in ("actions", "loads", "stores"):
            _int(m["totals"][k], f"totals.{k}")
        if not isinstance(m["cut"], bool):
            raise Unreadable(f"cut is {m['cut']!r}, not a boolean")
        _int(m["machine"]["l3"], "machine.l3", 1)
        r = m["refusal"]
        if r is not None:
            _int(r["action"], "refusal.action")
        if not isinstance(m["tables"], dict):
            raise Unreadable("tables is not an object")
    except (KeyError, TypeError) as e:
        raise Unreadable(f"manifest: missing or malformed field ({e!r})")


def load(path):
    path = Path(path)
    try:
        m = json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        raise Unreadable(f"{path}/manifest.json: {e}")
    if m.get("format") != "kpu-dflow":
        raise Unreadable(f"{path} is not a .dflow bundle (format {m.get('format')!r})")
    if m.get("version") != VERSION:
        raise Unreadable(f".dflow version {m.get('version')} (this checker reads {VERSION})")
    _manifest(m)
    tables = {}
    for name, schema in SCHEMA.items():
        t = m["tables"].get(name)
        if not isinstance(t, dict):
            raise Unreadable(f"no table {name}")
        if not isinstance(t.get("file"), str) or not isinstance(t.get("columns"), list):
            raise Unreadable(f"{name}: malformed table entry")
        rows = _int(t.get("rows"), f"{name}.rows")
        try:
            blob = (path / t["file"]).read_bytes()
        except OSError as e:
            raise Unreadable(str(e))
        cols = {}
        for c in t["columns"]:
            if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not isinstance(c.get("dtype"), str):
                raise Unreadable(f"{name}: malformed column {c!r}")
            _int(c.get("offset"), f"{name}.{c['name']}.offset")
            code, size = DTYPES.get(c["dtype"], (None, None))
            if code is None:
                raise Unreadable(f"{name}.{c['name']}: unknown dtype {c['dtype']}")
            off = c["offset"]
            if off + rows * size > len(blob):
                raise Unreadable(f"{name}.{c['name']} runs past the end of {t['file']}")
            a = array.array(code)
            a.frombytes(blob[off:off + rows * size])
            if sys.byteorder != "little":
                a.byteswap()
            cols[c["name"]] = (c["dtype"], a)
        for col, dtype in schema.items():
            if col not in cols:
                raise Unreadable(f"{name}: no column {col}")
            if cols[col][0] != dtype:
                raise Unreadable(f"{name}.{col} is {cols[col][0]}, expected {dtype}")
        tables[name] = {"rows": rows, "dtypes": {k: v[0] for k, v in cols.items()},
                        **{k: v[1] for k, v in cols.items()}}
    return m, tables


def check(m, T, limit=20):
    """Every violation, as (invariant, message), at most `limit` per invariant."""
    out = []
    count = {}

    def bad(inv, msg):
        count[inv] = count.get(inv, 0) + 1
        if count[inv] <= limit:
            out.append((inv, msg))

    S, R, I = T["steps"], T["residency"], T["iteration"]
    n, nr = S["rows"], R["rows"]
    totals = m["totals"]
    actions = totals["actions"]
    refused = m.get("refusal") is not None
    l3 = m["machine"]["l3"]

    # D8: no time at L-B.
    for name, t in T.items():
        for col, dtype in t["dtypes"].items():
            if dtype == "f64" or col.startswith("t_"):
                bad("D8", f"{name}.{col} is a time column ({dtype}); L-B has no time")

    # D2: one step, one action.
    for i in range(n):
        if S["action"][i] != i:
            bad("D2", f"steps[{i}].action is {S['action'][i]}")
    if not m["cut"] and n != actions:
        bad("D2", f"the bundle is not cut but records {n} of {actions} actions")
    if m["cut"] and n >= actions:
        bad("D2", f"the bundle is marked cut but records {n} of {actions} actions")

    # D4: residency lifetimes (the residency table is the whole run's).
    for r in range(nr):
        o, c = R["open"][r], R["close"][r]
        if o >= actions:
            bad("D4", f"residency {r} opens at {o}, past the run's {actions} actions")
        if c == NONE64:
            if not refused:
                bad("D4", f"residency {r} never ends (no Release or Retain)")
        elif c <= o:
            bad("D4", f"residency {r} ends at {c}, not after its open at {o}")
        if R["flags"][r] & RES_RETAINED and c == NONE64:
            bad("D4", f"residency {r} is marked retained but has no Retain")
    for r in range(1, nr):
        if R["open"][r] <= R["open"][r - 1]:
            bad("D4", f"residency {r} opens at {R['open'][r]}, not after residency {r - 1}")
    for i in range(n):
        res, f = S["residency"][i], S["flags"][i]
        if res == NONE64:
            if f & (OPENS | CLOSES | RETAINS):
                bad("D4", f"step {i} opens or ends a residency but names none")
            continue
        if res >= nr:
            bad("D4", f"step {i} names residency {res}; there are {nr}")
            continue
        o, c = R["open"][res], R["close"][res]
        if f & OPENS and o != i:
            bad("D4", f"step {i} opens residency {res}, which opens at {o}")
        if not f & OPENS and o == i:
            bad("D4", f"residency {res} opens at step {i}, which does not open it")
        if f & (CLOSES | RETAINS) and c != i:
            bad("D4", f"step {i} ends residency {res}, which ends at {c}")
        if not f & OPENS and (i < o or (c != NONE64 and i > c)):
            bad("D4", f"step {i} names residency {res} outside its life [{o}, {c}]")

    # D3 and D5: occupancy is the residencies; capacity holds. D9: one tile per slot.
    opens_at, ends_at = {}, {}
    for r in range(nr):
        opens_at.setdefault(R["open"][r], []).append(r)
        c = R["close"][r]
        if c != NONE64 and not R["flags"][r] & RES_RETAINED:
            ends_at.setdefault(c, []).append(r)
    live = 0
    slots = {}
    for i in range(n):
        for r in opens_at.get(i, []):
            live += 1
            key = (R["l3_tile"][r], R["slot"][r])
            if key in slots:
                bad("D9", f"step {i}: residency {r} opens in l3[{key[0]}] slot {key[1]}, "
                          f"which residency {slots[key]} still holds")
            slots[key] = r
        for r in ends_at.get(i, []):
            live -= 1
            slots.pop((R["l3_tile"][r], R["slot"][r]), None)
        if S["held_l3"][i] != live:
            bad("D3", f"after step {i} L3 holds {S['held_l3'][i]} tiles; {live} residencies are held")
        if S["held_l3"][i] > l3:
            bad("D5", f"after step {i} L3 holds {S['held_l3'][i]} tiles, over its capacity {l3}")

    # D9: a step's address is its residency's.
    for i in range(n):
        res, f = S["residency"][i], S["flags"][i]
        has = bool(f & HAS_PLACE)
        if has != (res != NONE64):
            bad("D9", f"step {i}: has_place is {has} but it names {'no' if res == NONE64 else 'a'} residency")
            continue
        if has and res < nr:
            a = (S["l3_tile"][i], S["slot"][i], S["bank"][i])
            b = (R["l3_tile"][res], R["slot"][res], R["bank"][res])
            if a != b:
                bad("D9", f"step {i} is at {a}; its residency {res} is at {b}")

    # D6: the iteration summary is complete and nested.
    rows = [dict(depth=I["depth"][k], line=I["line"][k], value=I["value"][k], first=I["first"][k],
                 last=I["last"][k], actions=I["actions"][k], loads=I["loads"][k],
                 stores=I["stores"][k],
                 peak=[I["peak_" + c][k] for c in CHANNELS]) for k in range(I["rows"])]
    top = [r for r in rows if r["depth"] == 0]
    if len(top) != 1:
        bad("D6", f"{len(top)} depth-0 iterations (the program is one)")
    elif actions:
        p = top[0]
        if (p["first"], p["last"], p["actions"]) != (0, actions - 1, actions):
            bad("D6", f"the program's iteration covers [{p['first']}, {p['last']}] "
                      f"({p['actions']} actions); the run has {actions}")
        if (p["loads"], p["stores"]) != (totals["loads"], totals["stores"]):
            bad("D6", f"the program's iteration has {p['loads']} loads, {p['stores']} stores; "
                      f"the run {totals['loads']}, {totals['stores']}")
    by_depth = {}
    for r in rows:
        by_depth.setdefault(r["depth"], []).append(r)
    for d, rs in sorted(by_depth.items()):
        rs.sort(key=lambda r: r["first"])
        for a, b in zip(rs, rs[1:]):
            if b["first"] <= a["last"]:
                bad("D6", f"depth {d}: iterations [{a['first']}, {a['last']}] and "
                          f"[{b['first']}, {b['last']}] overlap")
        for r in rs:
            if r["last"] < r["first"] or r["actions"] > r["last"] - r["first"] + 1 or r["actions"] == 0:
                bad("D6", f"depth {d}: iteration [{r['first']}, {r['last']}] has {r['actions']} actions")
            if r["last"] >= actions:
                bad("D6", f"depth {d}: iteration [{r['first']}, {r['last']}] runs past the run's {actions} actions")
            if d > 0:
                parents = [p for p in by_depth.get(d - 1, [])
                           if p["first"] <= r["first"] and r["last"] <= p["last"]]
                if len(parents) != 1:
                    bad("D6", f"depth {d}: iteration [{r['first']}, {r['last']}] lies in "
                              f"{len(parents)} iterations of depth {d - 1}")
    if not m["cut"] and n == actions:
        # Every step recorded: recount each iteration from its steps and their positions.
        W, L = T["where"], T["loops"]
        for d, rs in sorted(by_depth.items()):
            for r in rs:
                if r["last"] >= n or r["last"] < r["first"]:
                    continue                    # reported above; nothing to recount
                acts = loads = stores = 0
                peak = [0, 0, 0, 0]
                for i in range(r["first"], r["last"] + 1):
                    w = S["where"][i]
                    if w >= W["rows"]:
                        bad("D6", f"step {i} names position {w}; there are {W['rows']}")
                        continue
                    if d > 0:
                        if W["loop_count"][w] < d:
                            continue
                        loop = W["loop_first"][w] + d - 1
                        if loop >= L["rows"]:
                            bad("D6", f"position {w} names loop {loop}; there are {L['rows']}")
                            continue
                        if (L["line"][loop], L["value"][loop]) != (r["line"], r["value"]):
                            continue
                    acts += 1
                    loads += S["kind"][i] == LOAD
                    stores += S["kind"][i] == STORE
                    for c, ch in enumerate(CHANNELS):
                        peak[c] = max(peak[c], S["held_" + ch][i])
                got = (r["actions"], r["loads"], r["stores"], r["peak"])
                want = (acts, loads, stores, peak)
                if got != want:
                    bad("D6", f"depth {d} iteration {r['value']} at line {r['line']} "
                              f"[{r['first']}, {r['last']}] records (actions, loads, stores, peak) "
                              f"{got}; its steps give {want}")

    return out, count


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    ap.add_argument("--verbose", action="store_true", help="also list the checks that passed")
    args = ap.parse_args(argv)
    try:
        m, tables = load(args.bundle)
    except Unreadable as e:
        if args.json:
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            print(f"dflow_check: cannot read {args.bundle}: {e}", file=sys.stderr)
        return 2
    violations, count = check(m, tables)
    invs = ["D2", "D3", "D4", "D5", "D6", "D8", "D9"]
    if args.json:
        print(json.dumps({"ok": not violations, "steps": tables["steps"]["rows"],
                          "actions": m["totals"]["actions"], "cut": m["cut"],
                          "violations": [{"invariant": i, "message": msg} for i, msg in violations],
                          "counts": count}))
    else:
        print(f"{args.bundle}: {tables['steps']['rows']} of {m['totals']['actions']} steps, "
              f"{tables['residency']['rows']} residencies, {tables['iteration']['rows']} iterations"
              + ("  (steps cut)" if m["cut"] else ""))
        for inv in invs:
            if inv in count:
                print(f"  FAIL {inv}: {count[inv]} violation(s)")
            elif args.verbose:
                print(f"  ok   {inv}")
        for inv, msg in violations:
            print(f"    {inv}: {msg}")
        print("OK" if not violations else "FAILED")
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
