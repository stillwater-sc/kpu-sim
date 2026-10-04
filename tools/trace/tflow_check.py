#!/usr/bin/env python3
"""
Tile-flow record checker (#286 step 3).

Checks a .tflow bundle (written by `kpu-run --tflow`) against the invariants every L-T1 run
must satisfy, so a capacity or lifetime bug becomes a failing check instead of a reviewing
problem. #279 is the precedent: a capacity model reported a healthy peak while its series
exceeded the budget, and nothing runnable caught it.

Usage:
    python3 tflow_check.py <bundle.tflow> [--json] [--verbose]

Exit codes (the same contract as patterns/memory/lpddr5/common/trace_validator.py):
    0  every invariant holds
    1  one or more invariants are violated -- parse them and fix the root cause
    2  the bundle could not be read -- there are no violations to parse

Invariants:
    TF1  L3 capacity    resident tiles + foreign slots never exceed the L3 capacity
    TF2  intervals      0 <= t0 <= t1 <= makespan for every residency, transit and compute
    TF3  lanes          no two transits overlap on one lane of one mover pool
    TF4  compute tiles  no two computes overlap on one compute tile
    TF5  one slot       a tile's L3 residencies never overlap each other
    TF6  held while used  every compute and every transit runs while each tile its op
                        touches holds an L3 slot -- the executor's own release rule ("a slot
                        is freed when every op that touches it has completed"). An early
                        release, the #279 class, violates it.
    TF7  hop order      an op's hops do not overlap and run in chain order
    TF8  pyramid        each level of lod.bin is the exact merge of the level below, and the
                        coarsest level equals the raw totals
    TF9  filled         every L3 residency is FILLED: the tile was seeded, or a DMA delivers
                        it into the slot, or an op writes it there. A slot taken by a mere
                        reader is a tile the machine claims to hold but never received -- what
                        an early release (#279) turns into when a later reader re-takes the slot

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import array
import json
import sys
from bisect import bisect_right
from pathlib import Path

MOVER_NAMES = ["dma", "block-mover", "streamer", "noc"]   # enum Mover, in order
DTYPES = {"u8": ("B", 1), "u32": ("I", 4), "f64": ("d", 8)}


class Unreadable(Exception):
    """The bundle cannot be read (exit 2)."""


def _count(v, what, positive=False):
    """A count or offset from a JSON file: an int, and not a bool (which Python treats as one)."""
    if type(v) is not int or v < 0 or (positive and v == 0):
        raise Unreadable(f"{what} is {v!r}, not a {'positive' if positive else 'non-negative'} integer")
    return v


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _column(blob, table, name, rows):
    for c in table["columns"]:
        if c["name"] != name:
            continue
        code, size = DTYPES[c["dtype"]]
        off = _count(c["offset"], f"the offset of column {name}")
        rows = _count(rows, f"the row count of column {name}")
        if off > len(blob) or rows > (len(blob) - off) // size:
            raise Unreadable(f"column {name} runs past the end of its file")
        a = array.array(code)
        a.frombytes(blob[off:off + rows * size])
        if sys.byteorder != "little":
            a.byteswap()
        return a
    raise Unreadable(f"no column {name}")


def _named(rows, what):
    """A list of objects, each with a string name and kind: the shape TF8 compares row by row."""
    if not isinstance(rows, list):
        raise Unreadable(f"{what} is not a list")
    for i, x in enumerate(rows):
        if not (isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(x.get("kind"), str)):
            raise Unreadable(f"{what}[{i}] needs a string name and kind")
    return rows


def load(path):
    d = Path(path)
    try:
        m = json.loads((d / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        raise Unreadable(f"manifest.json: {e}") from e
    if m.get("format") != "kpu-tflow" or m.get("version") != 1:
        raise Unreadable("not a version-1 kpu-tflow bundle")
    rec = {"manifest": m}
    _named(m.get("stations"), "manifest stations")
    movers = m.get("movers", [])
    if not isinstance(movers, list) or not all(isinstance(x, dict) and isinstance(x.get("name"), str)
                                               for x in movers):
        raise Unreadable("manifest movers need a string name each")
    try:
        for name in ("residency", "transit", "compute", "op_tiles"):
            t = m["tables"][name]
            blob = (d / t["file"]).read_bytes()
            rows = t["rows"]
            cols = {}
            for c in t["columns"]:
                n = rows + 1 if (name == "op_tiles" and c["name"] == "offset") else rows
                if name == "op_tiles" and c["name"] in ("tile", "written"):
                    continue
                cols[c["name"]] = _column(blob, t, c["name"], n)
            if name == "op_tiles":
                nnz = cols["offset"][rows] if rows else 0
                cols["tile"] = _column(blob, t, "tile", nnz)
                cols["written"] = _column(blob, t, "written", nnz)
            rec[name] = cols
            rec[name + "_rows"] = rows
    except (OSError, KeyError) as e:
        raise Unreadable(f"{e}") from e
    lod_json = d / "lod.json"
    if lod_json.exists():
        # The pyramid is input too: a malformed lod.json is unreadable (exit 2), never a crash.
        try:
            lm = json.loads(lod_json.read_text())
            lblob = (d / lm["file"]).read_bytes()
            nrows = len(_named(lm["rows"], "lod rows"))
            levels = []
            for lv in lm["levels"]:
                bins = _count(lv["bins"], "a pyramid level's bin count", positive=True)
                n = nrows * bins
                levels.append({
                    "k": lv["k"], "bins": bins,
                    "occ": _column(lblob, {"columns": [{"name": "occ", "dtype": "f64", "offset": lv["occ"]}]}, "occ", n),
                    "peak": _column(lblob, {"columns": [{"name": "peak", "dtype": "u32", "offset": lv["peak"]}]}, "peak", n),
                    "starts": _column(lblob, {"columns": [{"name": "starts", "dtype": "u32", "offset": lv["starts"]}]}, "starts", n),
                })
            rec["lod"] = {"rows": lm["rows"], "levels": levels}
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise Unreadable(f"lod: {e!r}") from e
    return rec


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.violations = []
        self.checked = []

    def fail(self, inv, msg, limit=20):
        if sum(1 for v in self.violations if v["invariant"] == inv) < limit:
            self.violations.append({"invariant": inv, "message": msg})

    def ran(self, inv):
        self.checked.append(inv)


def _overlaps(intervals):
    """Pairs (a, b) of overlapping half-open intervals, given (t0, t1, id) sorted by t0."""
    out = []
    end, owner = -1, None
    for t0, t1, ident in sorted(intervals):
        if t1 == t0:
            continue                        # occupies no time
        if t0 < end:
            out.append((owner, ident, t0))
        if t1 > end:
            end, owner = t1, ident
    return out


def _accumulate(iv, constant, makespan, width, bins):
    """One row's (occ, peak, starts) per bin -- a line-for-line port of accumulate() in
    src/program/tile_flow_lod.cpp, so the checker recomputes the pyramid instead of trusting it.
    Ends sort before starts at the same cycle, a zero-length interval occupies no time, and a
    zero-length run still reports its constant (foreign slots) as the peak."""
    occ, peak, starts = [0.0] * bins, [0] * bins, [0] * bins
    for a, _ in iv:
        starts[int(min(a // width, bins - 1))] += 1
    ev = sorted([(a, 1) for a, b in iv if b > a] + [(b, -1) for a, b in iv if b > a])

    def segment(a, b, level):
        if b <= a or level <= 0:
            return
        bi = int(a // width)
        while bi < bins and bi * width < b:
            lo, hi = max(a, bi * width), min(b, (bi + 1) * width)
            occ[bi] += level * (hi - lo)
            peak[bi] = max(peak[bi], level)
            bi += 1

    cur, prev, i = constant, 0, 0
    if makespan == 0:
        peak[0] = max(peak[0], constant)
    while i < len(ev):
        t = ev[i][0]
        segment(prev, min(t, makespan), cur)
        while i < len(ev) and ev[i][0] == t:
            cur += ev[i][1]
            i += 1
        prev = max(prev, min(t, makespan))
    segment(prev, makespan, cur)
    return occ, peak, starts


def check(rec):
    r = Report()
    m = rec["manifest"]
    st = m["stations"]
    tiles = m["tiles"]
    makespan = m["makespan"]
    foreign = m.get("foreign_slots", 0)
    tile_name = lambda i: f"{tiles[i][0]}[{tiles[i][1]},{tiles[i][2]}]"
    res, tr, co, ot = rec["residency"], rec["transit"], rec["compute"], rec["op_tiles"]
    nres, ntr, nco = rec["residency_rows"], rec["transit_rows"], rec["compute_rows"]

    # TF2 -------------------------------------------------------------------
    r.ran("TF2")
    for name, cols, n in (("residency", res, nres), ("transit", tr, ntr), ("compute", co, nco)):
        for i in range(n):
            t0, t1 = cols["t0"][i], cols["t1"][i]
            if not (0 <= t0 <= t1 <= makespan):
                r.fail("TF2", f"{name} {i}: [{t0}, {t1}) is not inside [0, {makespan}]")

    # TF1 -------------------------------------------------------------------
    r.ran("TF1")
    for s_index, s in enumerate(st):
        if s["kind"] != "l3":
            continue
        cap = s["capacity"]
        ev = []
        for i in range(nres):
            if res["station"][i] == s_index and res["t1"][i] > res["t0"][i]:
                ev.append((res["t0"][i], 1, i))
                ev.append((res["t1"][i], -1, i))
        ev.sort(key=lambda e: (e[0], e[1]))   # ends before starts at the same cycle
        cur = foreign
        if cap and cur > cap:
            r.fail("TF1", f"{s['name']}: {foreign} foreign slots alone exceed capacity {cap}")
        for t, delta, i in ev:
            cur += delta
            if cap and cur > cap:
                r.fail("TF1", f"{s['name']}: {cur} tiles resident at cycle {int(t)} "
                              f"(capacity {cap}); {tile_name(res['tile'][i])} took the slot over it")

    # TF3 -------------------------------------------------------------------
    r.ran("TF3")
    lanes = {}
    for i in range(ntr):
        lanes.setdefault((tr["mover"][i], tr["lane"][i]), []).append((tr["t0"][i], tr["t1"][i], i))
    for (mover, lane), ivs in lanes.items():
        for a, b, t in _overlaps(ivs):
            r.fail("TF3", f"{MOVER_NAMES[mover]} lane {lane}: transits {a} and {b} overlap at cycle {int(t)}")

    # TF4 -------------------------------------------------------------------
    r.ran("TF4")
    tiles_cf = {}
    for i in range(nco):
        tiles_cf.setdefault(co["station"][i], []).append((co["t0"][i], co["t1"][i], co["op"][i]))
    for s_index, ivs in tiles_cf.items():
        for a, b, t in _overlaps(ivs):
            r.fail("TF4", f"{st[s_index]['name']}: ops {a} and {b} compute at once at cycle {int(t)}")

    # TF5 / TF6 --------------------------------------------------------------
    by_tile = {}
    for i in range(nres):
        by_tile.setdefault(res["tile"][i], []).append((res["t0"][i], res["t1"][i], i))
    r.ran("TF5")
    for tile, ivs in by_tile.items():
        for a, b, t in _overlaps(ivs):
            r.fail("TF5", f"{tile_name(tile)} holds two L3 slots at cycle {int(t)} (residencies {a}, {b})")
    starts = {t: sorted(ivs) for t, ivs in by_tile.items()}
    keys = {t: [iv[0] for iv in ivs] for t, ivs in starts.items()}

    def held(tile, t0, t1):
        ivs = starts.get(tile)
        if not ivs:
            return False
        j = bisect_right(keys[tile], t0) - 1
        return j >= 0 and ivs[j][0] <= t0 and t1 <= ivs[j][1]

    r.ran("TF6")
    off, opt = ot["offset"], ot["tile"]
    def op_tiles(op):
        return opt[off[op]:off[op + 1]]
    for i in range(nco):
        op = co["op"][i]
        for tile in op_tiles(op):
            if not held(tile, co["t0"][i], co["t1"][i]):
                r.fail("TF6", f"op {op} computes over [{int(co['t0'][i])}, {int(co['t1'][i])}) but "
                              f"{tile_name(tile)} does not hold an L3 slot for all of it -- "
                              f"released before its last user finished")
    for i in range(ntr):
        op = tr["op"][i]
        for tile in op_tiles(op):
            if not held(tile, tr["t0"][i], tr["t1"][i]):
                r.fail("TF6", f"op {op} moves {tile_name(tile)} over [{int(tr['t0'][i])}, "
                              f"{int(tr['t1'][i])}) without it holding an L3 slot -- released "
                              f"before its last user finished")

    # TF7 -------------------------------------------------------------------
    r.ran("TF7")
    chains = {}
    for i in range(ntr):
        chains.setdefault(tr["op"][i], []).append((tr["t0"][i], tr["t1"][i], tr["hop"][i]))
    for op, hops in chains.items():
        hops.sort()
        for (a0, a1, ah), (b0, b1, bh) in zip(hops, hops[1:]):
            if b0 < a1:
                r.fail("TF7", f"op {op}: hop {bh} starts at {int(b0)} before hop {ah} finishes at {int(a1)}")

    # TF9 -------------------------------------------------------------------
    r.ran("TF9")
    DMA_IN, L3_TO_L3 = 0, 6                 # enum Hop: DmaDramToL3, BlockMoverL3ToL3
    arrivals = {}
    for i in range(ntr):
        if tr["hop"][i] in (DMA_IN, L3_TO_L3):
            arrivals.setdefault(tr["tile"][i], []).append(tr["t0"][i])
    writes = {}
    wr = ot["written"]
    for i in range(nco):
        op = co["op"][i]
        for j in range(off[op], off[op + 1]):
            if wr[j]:
                writes.setdefault(opt[j], []).append(co["t0"][i])
    SEEDED = 1
    for i in range(nres):
        tile, t0, t1 = res["tile"][i], res["t0"][i], res["t1"][i]
        if res["flags"][i] & SEEDED:
            continue
        # Half-open, like the residency itself: an event at t1 belongs to whatever holds the
        # slot next. A zero-length residency is filled by an event at its one instant.
        inside = lambda ts: any(t0 <= t < t1 or t0 == t == t1 for t in ts)
        if not (inside(arrivals.get(tile, ())) or inside(writes.get(tile, ()))):
            r.fail("TF9", f"{tile_name(tile)} holds an L3 slot over [{int(t0)}, {int(t1)}) that "
                          f"nothing filled: no DMA delivers it and no op writes it -- its data "
                          f"was released earlier and the slot was re-taken by a reader")

    # TF8 -------------------------------------------------------------------
    if "lod" in rec:
        r.ran("TF8")
        lod = rec["lod"]
        nrows = len(lod["rows"])
        levels = lod["levels"]
        # The rows say which raw events each one is compared against, so they are checked
        # first: a relabelled row (an l3 renamed to kind "dram", say) would otherwise be
        # compared against nothing and pass.
        want_rows = [(s["name"], s["kind"]) for s in st] + \
                    [("mover:" + mv["name"], "mover") for mv in m.get("movers", [])]
        got_rows = [(x["name"], x["kind"]) for x in lod["rows"]]
        if got_rows != want_rows:
            bad = next((i for i, (g, w) in enumerate(zip(got_rows, want_rows)) if g != w),
                       min(len(got_rows), len(want_rows)))
            r.fail("TF8", f"the pyramid's rows do not match the record's stations and movers: "
                          f"{len(got_rows)} rows for {len(want_rows)}, first difference at row "
                          f"{bad}")
            return r
        unknown = [mv["name"] for mv in m.get("movers", []) if mv["name"] not in MOVER_NAMES]
        if unknown:
            r.fail("TF8", f"mover pools {unknown} are not among {MOVER_NAMES}")
            return r
        for lo, hi in zip(levels, levels[1:]):
            if hi["bins"] != (lo["bins"] + 1) // 2 or hi["k"] != lo["k"] + 1:
                r.fail("TF8", f"level k={hi['k']} is not the parent of k={lo['k']}")
                continue
            for row in range(nrows):
                for b in range(hi["bins"]):
                    kids = [row * lo["bins"] + c for c in (2 * b, 2 * b + 1) if c < lo["bins"]]
                    p = row * hi["bins"] + b
                    if (hi["occ"][p] != sum(lo["occ"][c] for c in kids) or
                            hi["peak"][p] != max(lo["peak"][c] for c in kids) or
                            hi["starts"][p] != sum(lo["starts"][c] for c in kids)):
                        r.fail("TF8", f"{lod['rows'][row]['name']}: level k={hi['k']} bin {b} "
                                      f"is not the merge of its children")
        top = levels[-1]
        # The builder halves until one bin is left; a wider top would let every bin after the
        # first escape the raw-event comparison.
        if top["bins"] != 1:
            r.fail("TF8", f"the coarsest level has {top['bins']} bins, not 1")
            return r
        base = levels[0]
        width = 1 << base["k"]
        if base["bins"] != max(1, -(-makespan // width)):
            r.fail("TF8", f"the base level has {base['bins']} bins of {width} cycles for a "
                          f"{makespan}-cycle run")
            return r
        for row, info in enumerate(lod["rows"]):
            constant = 0
            if info["kind"] == "l3":
                iv = [(res["t0"][i], res["t1"][i]) for i in range(nres) if res["station"][i] == row]
                constant = foreign
            elif info["kind"] == "cf":
                iv = [(co["t0"][i], co["t1"][i]) for i in range(nco) if co["station"][i] == row]
            elif info["kind"] == "mover":
                pool = MOVER_NAMES.index(info["name"].split(":", 1)[1])
                iv = [(tr["t0"][i], tr["t1"][i]) for i in range(ntr) if tr["mover"][i] == pool]
            else:
                iv = []
            # Recomputed from the raw events at the base and at the top. With every merge
            # checked above, that pins every level: a value moved between sibling bins keeps
            # the merges and the total, and only the base comparison sees it.
            for name, lvl, w in (("base", base, width), ("coarsest", top, 1 << top["k"])):
                occ, peak, starts = _accumulate(iv, constant, makespan, w, lvl["bins"])
                at = row * lvl["bins"]
                for b in range(lvl["bins"]):
                    for metric, want in (("occ", occ[b]), ("peak", peak[b]), ("starts", starts[b])):
                        if lvl[metric][at + b] != want:
                            r.fail("TF8", f"{info['name']}: {name} bin {b} {metric} is "
                                          f"{lvl[metric][at + b]}, the raw events give {want}")
    return r


def main():
    ap = argparse.ArgumentParser(description="Check a .tflow bundle against the tile-flow invariants.")
    ap.add_argument("bundle")
    ap.add_argument("--json", "-j", action="store_true", help="machine-readable output")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    try:
        rec = load(args.bundle)
    except Unreadable as e:
        print(f"tflow_check: cannot read {args.bundle}: {e}", file=sys.stderr)
        sys.exit(2)
    rep = check(rec)
    ok = not rep.violations
    if args.json:
        print(json.dumps({"bundle": args.bundle, "passed": ok, "checked": rep.checked,
                          "violations": rep.violations}, indent=1))
    else:
        m = rec["manifest"]
        print(f"tflow_check: {args.bundle} ({m['level']}, {m['device_label']}, makespan {m['makespan']})")
        print(f"  checked {' '.join(rep.checked)}")
        for v in rep.violations:
            print(f"  {v['invariant']}  {v['message']}")
        print("  PASS" if ok else f"  FAIL: {len(rep.violations)} violation(s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
