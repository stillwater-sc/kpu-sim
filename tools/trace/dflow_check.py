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

    REF references            every index a table holds names a row that exists, in every bundle,
                              cut or not: a step's tile, position, kind, process and channels; a
                              position's loops; a loop's variable; a tile's operand; a residency's
                              tile; and every slot is inside the program's L3. The viewer resolves
                              each of them, so a dangling one is a broken bundle, not a display bug

Version 2 adds tile values (snapshots and their float32 elements):
    V1 references             a snapshot's tile, channel and elements exist, and its step is an
                              action of the run or none (a final value)
    V2 where it was taken     a recorded step's snapshot is of that step's tile, in the channel the
                              action leaves it in (Load L3, Move L2, Feed and Call the fabric,
                              Drain L2, Writeback L3, Store DRAM, Inherit L3; none for the others)
    V3 shape                  height x width is the tile's shape in its operand's tiling
    V4 finals                 recorded (manifest values.finals true): exactly one final snapshot per
                              tile of every operand; otherwise none; and no step snapshot unless
                              tiles were selected
    V5 the store is the final a tile's last Store snapshot equals its final value, bit for bit,
                              when nothing was cut and the tile was not retained (L-B's DRAM is
                              its result)

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

VERSIONS = (1, 2)
NONE8, NONE32, NONE64 = 0xFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
DTYPES = {"u8": ("B", 1), "u16": ("H", 2), "u32": ("I", 4), "u64": ("Q", 8), "i64": ("q", 8),
          "f32": ("f", 4), "f64": ("d", 8)}
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
SCHEMA_V2 = {
    "snapshots": {"step": "u64", "first": "u64", "tile": "u32", "height": "u32", "width": "u32", "chan": "u8"},
    "elements": {"v": "f32"},
}
# The channel each kind leaves its tile in (enum Action::Kind, in order); None: no snapshot.
LANDS = [1, 0, 2, 1, 3, 2, 3, None, 1, None]   # LOAD l3, STORE dram, MOVE l2, WRITEBACK l3, FEED cf,
                                                # DRAIN l2, CALL cf, RELEASE, INHERIT l3, RETAIN


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
        # The lists REF counts indices against: arrays of names (or of named objects).
        for k in ("vars", "kinds", "processes"):
            if not isinstance(m[k], list) or not all(isinstance(x, str) for x in m[k]):
                raise Unreadable(f"{k} is not an array of names")
        for k in ("operands", "channels"):
            if not isinstance(m[k], list) or not all(isinstance(x, dict) and isinstance(x.get("name"), str) for x in m[k]):
                raise Unreadable(f"{k} is not an array of named objects")
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
    if m.get("version") not in VERSIONS:
        raise Unreadable(f".dflow version {m.get('version')} (this checker reads {', '.join(map(str, VERSIONS))})")
    _manifest(m)
    if m["version"] >= 2:
        v = m.get("values")
        if not isinstance(v, dict) or v.get("format") != "float32" or not isinstance(v.get("selected"), str) \
                or not (v.get("finals") is True or isinstance(v.get("finals"), str)) \
                or not (v.get("cut_at") is None or (isinstance(v.get("cut_at"), int) and v["cut_at"] >= 0)):
            raise Unreadable("values: missing or malformed (format float32, selected, finals, cut_at)")
    tables = {}
    schemas = dict(SCHEMA, **(SCHEMA_V2 if m["version"] >= 2 else {}))
    for name, schema in schemas.items():
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
    # REF: every reference resolves, cut or not -- the viewer resolves each one.
    W, L, TL = T["where"], T["loops"], T["tiles"]
    nvars, nops, nkinds = len(m["vars"]), len(m["operands"]), len(m["kinds"])
    nprocs, nchans = len(m["processes"]), len(m["channels"])
    for t in range(TL["rows"]):
        if TL["operand"][t] >= nops:
            bad("REF", f"tile {t} names operand {TL['operand'][t]}; there are {nops}")
    for l in range(L["rows"]):
        if L["var"][l] >= nvars:
            bad("REF", f"loop row {l} names variable {L['var'][l]}; there are {nvars}")
    where_ok = [True] * W["rows"]
    for w in range(W["rows"]):
        if W["loop_first"][w] + W["loop_count"][w] > L["rows"]:
            bad("REF", f"position {w} names loops past the {L['rows']} recorded")
            where_ok[w] = False
    for r in range(nr):
        if R["tile"][r] >= TL["rows"]:
            bad("REF", f"residency {r} names tile {R['tile'][r]}; there are {TL['rows']}")
        if R["slot"][r] >= l3:
            bad("REF", f"residency {r} is in slot {R['slot'][r]}, outside the program's {l3}")
    pos_ok = [True] * n
    for i in range(n):
        w = S["where"][i]
        if w >= W["rows"] or not where_ok[w]:
            if w >= W["rows"]:
                bad("REF", f"step {i} names position {w}; there are {W['rows']}")
            pos_ok[i] = False
        if S["tile"][i] >= TL["rows"]:
            bad("REF", f"step {i} names tile {S['tile'][i]}; there are {TL['rows']}")
        if S["kind"][i] >= nkinds:
            bad("REF", f"step {i} has kind {S['kind'][i]}; there are {nkinds}")
        if S["process"][i] != NONE8 and S["process"][i] >= nprocs:
            bad("REF", f"step {i} names process {S['process'][i]}; there are {nprocs}")
        for c in ("from", "to"):
            if S[c][i] != NONE8 and S[c][i] >= nchans:
                bad("REF", f"step {i} moves {c} channel {S[c][i]}; there are {nchans}")
        if S["flags"][i] & HAS_PLACE and S["slot"][i] >= l3:
            bad("REF", f"step {i} is at slot {S['slot'][i]}, outside the program's {l3}")
    if not m["cut"] and n == actions:
        # Every step recorded: recount each iteration from its steps and their positions.
        for d, rs in sorted(by_depth.items()):
            for r in rs:
                if r["last"] >= n or r["last"] < r["first"]:
                    continue                    # reported above; nothing to recount
                acts = loads = stores = 0
                peak = [0, 0, 0, 0]
                for i in range(r["first"], r["last"] + 1):
                    if not pos_ok[i]:
                        continue                # reported above
                    w = S["where"][i]
                    if d > 0:
                        if W["loop_count"][w] < d:
                            continue
                        loop = W["loop_first"][w] + d - 1
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

    if m["version"] >= 2:
        check_values(m, T, bad, nr)
    return out, count


def check_values(m, T, bad, nr):
    S, X, TL, P, E = T["steps"], T["residency"], T["tiles"], T["snapshots"], T["elements"]
    n, actions, ne = S["rows"], m["totals"]["actions"], E["rows"]
    ops, vals = m["operands"], m["values"]
    finals_on = vals["finals"] is True
    final_of, last_store = {}, {}
    ok = []
    for k in range(P["rows"]):
        t, step, chan = P["tile"][k], P["step"][k], P["chan"][k]
        rows, cols, first = P["height"][k], P["width"][k], P["first"][k]
        good = True
        if t >= TL["rows"]:
            bad("V1", f"snapshot {k} names tile {t}; there are {TL['rows']}")
            good = False
        if chan >= len(m["channels"]):
            bad("V1", f"snapshot {k} is in channel {chan}; there are {len(m['channels'])}")
        if first + rows * cols > ne:
            bad("V1", f"snapshot {k}'s elements [{first}, {first + rows * cols}) run past the {ne} recorded")
            good = False
        if step != NONE64 and step >= actions:
            bad("V1", f"snapshot {k} is at step {step}, past the run's {actions} actions")
            good = False
        ok.append(good)
        if not good:
            continue
        o = ops[TL["operand"][t]] if TL["operand"][t] < len(ops) else None
        if o is not None:
            ti, tj = TL["ti"][t], TL["tj"][t]
            want = (min(o["tile_rows"], o["rows"] - ti * o["tile_rows"]), min(o["tile_cols"], o["cols"] - tj * o["tile_cols"]))
            if (rows, cols) != want:
                bad("V3", f"snapshot {k} of tile {t} is {rows}x{cols}; its tile is {want[0]}x{want[1]}")
        if step == NONE64:
            if t in final_of:
                bad("V4", f"tile {t} has two final values (snapshots {final_of[t]} and {k})")
            final_of[t] = k
            continue
        if not vals["selected"]:
            bad("V4", f"snapshot {k} is at step {step}, but no tiles were selected")
        if step < n:
            kind = S["kind"][step]
            if S["tile"][step] != t:
                bad("V2", f"snapshot {k} is of tile {t}, but step {step} acts on tile {S['tile'][step]}")
            elif kind >= len(LANDS) or LANDS[kind] is None or LANDS[kind] != chan:
                bad("V2", f"snapshot {k} at step {step} ({m['kinds'][kind] if kind < len(m['kinds']) else kind}) "
                          f"is in channel {chan}, not where the action leaves its tile")
            if kind == STORE:
                last_store[t] = k
    if finals_on:
        expected = sum(-(-o["rows"] // o["tile_rows"]) * -(-o["cols"] // o["tile_cols"]) for o in ops)
        if len(final_of) != expected:
            bad("V4", f"{len(final_of)} final values recorded; the operands have {expected} tiles")
    elif final_of:
        bad("V4", f"finals are not recorded ({vals['finals']}) but {len(final_of)} final values are")
    if vals["cut_at"] is None and not m["cut"] and finals_on:
        retained = {X["tile"][r] for r in range(nr) if X["flags"][r] & RES_RETAINED}
        for t, k in last_store.items():
            if t in retained or t not in final_of:
                continue
            f = final_of[t]
            a = E["v"][P["first"][k]:P["first"][k] + P["height"][k] * P["width"][k]].tobytes()
            b = E["v"][P["first"][f]:P["first"][f] + P["height"][f] * P["width"][f]].tobytes()
            if a != b:
                bad("V5", f"tile {t}'s last Store (snapshot {k}) differs from its final value (snapshot {f})")


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
    invs = ["D2", "D3", "D4", "D5", "D6", "D8", "D9", "REF", "V1", "V2", "V3", "V4", "V5"]
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
