#!/usr/bin/env python3
"""
dflow_diff.py -- join two domain-flow records event by event, and say where they diverge (#402).

    python3 tools/trace/dflow_diff.py <a.dflow> <b.dflow> [--json] [--tolerance N] [--show K]

Two records of the same program -- kpu-sim's L-T1 or L-CA (kpu-run --dflow-lt1, --dflow-lca), the
RTL's, or the hand-written RTL sample -- are joined on the identity key of
docs/plans/shared-tile-flow-record.md §5.3:

    (kind, process, tile, src, src_addr, dst, dst_addr, line, loop indices, row, beat)

where a tile is its operand name and tile coordinates, a process and a layer are their names, and
the loop indices are (variable name, value) pairs, so the key reads the same in both records. L2's
and the accumulator's addresses are each producer's placement, not identity (§5.3; #401): they are
left out of the key, and a matched event whose L2 or accumulator address differs is reported as
placed differently, once per event, with each endpoint that differs. Only addresses both records
give, in the same form, are compared: an address one record leaves out, or slots against rows, is
not a difference, so a count of zero does not mean every placement agrees. Another layer whose addresses have different forms in the two records (6.1's
words against slots) has its addresses left out of the key too, and said so.

It reports:
  - the events only one record has, by kind, a row event under its transaction;
  - with both records timed, the first divergence -- the earliest event, in A's cycles, whose start
    or duration differs by more than --tolerance cycles -- and the largest divergences of start
    and of duration.

It is a debugging tool, not a CI gate (D3): CI compares results (dflow_check.py --result), never
events. Exit 0 when the records join with nothing one-sided, no divergence and nothing placed
differently, 1 when they differ,
2 when either cannot be read, fails the checker, or has no events (L-B's steps are its events).

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflow_check as dc  # noqa: E402

TRANSACTIONS = {dc.EV_DMA, dc.EV_BM, dc.EV_STREAM, dc.EV_COMPUTE, dc.EV_DRAIN}
ROWS = {dc.EV_BM_ROW, dc.EV_STR_ROW, dc.EV_DRAIN_ROW}


class Refused(Exception):
    pass


def form(g, layer):
    """How a layer's addresses are spelled: slots, rows, words, the array's edge, any, or none."""
    x = g.get(layer)
    if not isinstance(x, dict) or not x.get("modelled"):
        return "none"
    if layer == "dram":
        return "any"
    if "words" in x:
        return "words"
    if "slots" in x:
        return "slots"
    if layer == "l1":
        return "edge"
    return "rows"


class Record:
    def __init__(self, path):
        self.path = path
        try:
            self.m, self.T = dc.load(path)
        except dc.Unreadable as e:
            raise Refused(f"{path}: {e}")
        violations, count = dc.check(self.m, self.T)
        if count:
            first = violations[0] if violations else ("?", "")
            raise Refused(f"{path}: fails dflow_check ({sum(count.values())} violations; first {first[0]}: {first[1]})")
        if self.m.get("version", 0) < 6:
            raise Refused(f"{path}: version {self.m.get('version')} has no events (version 6 does)")
        self.E = self.T["events"]
        if not self.E["rows"]:
            raise Refused(f"{path}: {self.m.get('level')} writes no events (its steps are its events); "
                          "compare timed records")
        self.timed = self.m.get("time") is not None and "cycle" in self.E
        self.level = self.m.get("level", "?")
        ops = [o["name"] for o in self.m["operands"]]
        TL, W, L = self.T["tiles"], self.T["where"], self.T["loops"]
        self.tile = [(ops[TL["operand"][t]], TL["ti"][t], TL["tj"][t]) for t in range(TL["rows"])]
        names = self.m.get("vars", [])
        self.pos = []
        for w in range(W["rows"]):
            f, n = W["loop_first"][w], W["loop_count"][w]
            loops = tuple((names[L["var"][f + d]] if L["var"][f + d] < len(names) else L["var"][f + d],
                           L["value"][f + d]) for d in range(n))
            self.pos.append((W["line"][w], loops))
        self.procs = self.m.get("processes", [])

    def describe(self):
        cyc = f", {self.m['time']['cycles']} cycles" if self.timed else ""
        return f"{self.level}  {self.m.get('producer', '?')}  {self.E['rows']} events{cyc}"


def keys(r, drop):
    """Each event's identity key, with the layers in `drop` unaddressed; and the key's parts."""
    E = r.E
    out = []
    for e in range(E["rows"]):
        k, pr, t, w = E["kind"][e], E["process"][e], E["tile"][e], E["where"][e]
        src, dst = E["src"][e], E["dst"][e]
        name = lambda layer: dc.LAYERS[layer] if layer < len(dc.LAYERS) else None
        addr = lambda layer, a: None if a == dc.NONE64 or name(layer) in drop else a
        line, loops = r.pos[w] if w != dc.NONE64 and w < len(r.pos) else (None, ())
        key = (dc.EVENT_KINDS[k], r.procs[pr] if pr != dc.NONE8 and pr < len(r.procs) else None,
               r.tile[t] if t != dc.NONE32 and t < len(r.tile) else None,
               name(src), addr(src, E["src_addr"][e]), name(dst), addr(dst, E["dst_addr"][e]),
               line, loops,
               None if E["row"][e] == dc.NONE32 else E["row"][e], None if E["beat"][e] == dc.NONE32 else E["beat"][e])
        out.append(key)
    return out


def label(key):
    kind, proc, tile, src, sa, dst, da, line, loops, row, beat = key
    t = f"{tile[0]}[{tile[1]},{tile[2]}]" if tile else "-"
    at = lambda layer, a: f"{layer}" + (f"@{a}" if a is not None else "")
    where = (f"line {line}" if line is not None else "") + \
            ("" if not loops else (", " if line is not None else "") + " ".join(f"{v}={x}" for v, x in loops))
    rb = "" if row is None else f" row {row}" + ("" if beat is None else f" beat {beat}")
    by = f" by {proc}" if proc is not None else " by no process"
    return f"{kind} {t} {at(src, sa)} -> {at(dst, da)}{rb}{by} ({where})"


def diff(a, b, tolerance=0, show=5):
    ga, gb = a.m.get("geometry", {}), b.m.get("geometry", {})
    # L2's and the accumulator's addresses are each producer's placement, not identity (§5.3; #401):
    # always left out of the key, and compared on the matched events instead. Another layer whose
    # address forms differ (6.1's words against slots) is left out too.
    drop = sorted({"l2", "acc"} | {layer for layer in dc.LAYERS if form(ga, layer) != form(gb, layer)})
    ka, kb = keys(a, set(drop)), keys(b, set(drop))
    ia, ib = {}, {}
    for side, ks, idx in (("A", ka, ia), ("B", kb, ib)):
        for e, k in enumerate(ks):
            if k in idx:
                # Distinct in its own record, the key can coincide once addresses are dropped.
                raise Refused(f"{side}: events {idx[k]} and {e} share the key {label(k)} once "
                              f"{', '.join(drop) or 'no layer'}'s addresses are left out")
            idx[k] = e
    matched = [k for k in ka if k in ib]
    only = {"A": [k for k in ka if k not in ib], "B": [k for k in kb if k not in ia]}

    def one_sided(side, r, ks, idx, other):
        rows, by_kind = [], {}
        for k in only[side]:
            e = idx[k]
            by_kind[k[0]] = by_kind.get(k[0], 0) + 1
            entry = {"event": e, "key": label(k)}
            if r.E["kind"][e] in ROWS and r.E["parent"][e] != dc.NONE64:
                pk = ks[r.E["parent"][e]]
                entry["parent"] = label(pk)
                entry["parent_matched"] = pk in other
            rows.append(entry)
        under = sum(1 for x in rows if x.get("parent_matched"))
        return {"count": len(rows), "by_kind": by_kind, "rows_under_matched_transactions": under, "first": rows[:show]}

    # Placement: matched events whose L2 or accumulator address differs (the same transfer, placed
    # elsewhere by each producer).
    placed = []
    for k in matched:
        x, y = ia[k], ib[k]
        ends = []                                   # each endpoint placed differently
        for side in ("src", "dst"):
            layer = a.E[side][x]
            if layer < len(dc.LAYERS) and dc.LAYERS[layer] in ("l2", "acc"):
                pa, pb = a.E[f"{side}_addr"][x], b.E[f"{side}_addr"][y]
                if pa != pb and dc.NONE64 not in (pa, pb) and form(ga, dc.LAYERS[layer]) == form(gb, dc.LAYERS[layer]):
                    ends.append({"layer": dc.LAYERS[layer], "a": pa, "b": pb})
        if ends:
            placed.append({"key": label(k), "ends": ends})
    out = {"a": a.describe(), "b": b.describe(), "dropped_addresses": drop,
           "placement": {"count": len(placed), "first": placed[:show]},
           "matched": len(matched),
           "matched_transactions": sum(1 for k in matched if dc.EVENT_KINDS.index(k[0]) in TRANSACTIONS),
           "only_a": one_sided("A", a, ka, ia, ib), "only_b": one_sided("B", b, kb, ib, ia)}
    if a.timed and b.timed:
        d = []
        for k in matched:
            x, y = ia[k], ib[k]
            ca, ea, cb, eb = a.E["cycle"][x], a.E["end"][x], b.E["cycle"][y], b.E["end"][y]
            d.append({"key": label(k), "a": [ca, ea], "b": [cb, eb], "start": cb - ca, "duration": (eb - cb) - (ea - ca)})
        d.sort(key=lambda v: (v["a"][0], v["a"][1]))
        first = next((v for v in d if abs(v["start"]) > tolerance or abs(v["duration"]) > tolerance), None)
        out["first_divergence"] = first
        out["largest_start"] = max(d, key=lambda v: abs(v["start"]), default=None)
        out["largest_duration"] = max(d, key=lambda v: abs(v["duration"]), default=None)
        out["cycles"] = [a.m["time"]["cycles"], b.m["time"]["cycles"]]
    else:
        out["first_divergence"] = None
        out["untimed"] = [s for s, r in (("A", a), ("B", b)) if not r.timed]
    out["differs"] = bool(only["A"] or only["B"] or out["first_divergence"] or out["placement"]["count"])
    return out


def report(o, tolerance):
    print(f"A  {o['a']}")
    print(f"B  {o['b']}")
    print("joined on (kind, process, tile, layers, addresses, line, loop indices, row, beat); "
          f"{', '.join(o['dropped_addresses'])} addresses left out (L2's and the accumulator's are placement; "
          "another layer's only when the two records' forms differ)")
    print(f"matched {o['matched']} events ({o['matched_transactions']} transactions)")
    pl = o["placement"]
    if pl["count"]:
        print(f"placed differently: {pl['count']} matched events' L2 or accumulator addresses differ")
        for x in pl["first"]:
            print(f"    {x['key']}: " + "; ".join(f"{e['layer']} A {e['a']}, B {e['b']}" for e in x["ends"]))
    for side in ("a", "b"):
        s = o[f"only_{side}"]
        if not s["count"]:
            continue
        kinds = ", ".join(f"{n} {k}" for k, n in sorted(s["by_kind"].items()))
        under = f"; {s['rows_under_matched_transactions']} row events under matched transactions" \
            if s["rows_under_matched_transactions"] else ""
        print(f"only in {side.upper()}: {s['count']} ({kinds}{under})")
        for x in s["first"]:
            print(f"    {x['key']}" + (f"   under {x['parent']} ({'joined' if x['parent_matched'] else 'not joined'})"
                                         if "parent" in x else ""))
    if "untimed" in o:
        print(f"no timing comparison: {', '.join(o['untimed'])} has no time")
    else:
        f = o["first_divergence"]
        print(f"cycles to result: A {o['cycles'][0]}, B {o['cycles'][1]}")
        if f is None:
            print(f"no divergence beyond {tolerance} cycles")
        else:
            print(f"first divergence: {f['key']}")
            print(f"    A [{f['a'][0]}, {f['a'][1]}]  B [{f['b'][0]}, {f['b'][1]}]  "
                  f"start {f['start']:+d}, duration {f['duration']:+d}")
            for what in ("start", "duration"):
                g = o[f"largest_{what}"]
                print(f"largest {what} divergence: {g['key']}  ({what} {g[what]:+d}: A [{g['a'][0]}, {g['a'][1]}], "
                      f"B [{g['b'][0]}, {g['b'][1]}])")


def main(argv=None):
    p = argparse.ArgumentParser(description="Join two .dflow records event by event (#402).")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--json", action="store_true", help="the report as JSON")
    p.add_argument("--tolerance", type=int, default=0, help="cycles a start or duration may differ (default 0)")
    p.add_argument("--show", type=int, default=5, help="one-sided events listed per side (default 5)")
    args = p.parse_args(argv)
    try:
        a, b = Record(args.a), Record(args.b)
        o = diff(a, b, args.tolerance, args.show)
    except Refused as e:
        print(f"dflow_diff: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(o, indent=1))
    else:
        report(o, args.tolerance)
    return 1 if o["differs"] else 0


if __name__ == "__main__":
    sys.exit(main())
