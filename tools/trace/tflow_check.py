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
                        release, the #279 class, violates it. Version 4 (a CSP program's
                        record; an op is one action): every transit into or out of L3 runs
                        while its tile holds an L3 slot -- an accumulator has none, and a Feed
                        reads L2 after the program released the L3 copy.
    TF7  hop order      an op's hops do not overlap; each hop moves between the station kinds
                        its type names (a DMA in goes dram -> l3, ...); and a tile's hops for
                        one op form a chain, each starting where the previous one ended
    TF8  pyramid        each level of lod.bin is the exact merge of the level below, and the
                        base and coarsest levels equal the pyramid recomputed from raw events
    TF9  filled         every L3 residency is FILLED: the tile was seeded, or a DMA delivers
                        it into the slot, or an op writes it there. A slot taken by a mere
                        reader is a tile the machine claims to hold but never received -- what
                        an early release (#279) turns into when a later reader re-takes the slot

    TF12 one tile per slot (version 5: residencies carry their L3 address) every L3 residency's
                        module is 0 (a version-5 record is the program's flat L3), its slot is
                        inside its L3's capacity, and its bank is slot % l3_banks; the residencies
                        placed in one slot of one L3 never overlap in time (a zero-length one
                        occupies no time)
    TF13 one copy per L2 slot (version 6: the program stages) every staged copy is in one of the
                        program's l2_slots, held [t0, t1) inside the run and fed at least once; the
                        copies placed in one slot of one compute tile's L2 never overlap in time,
                        and no more than l2_slots are held at once
    TF14 the fabric's phases (version 6) each chain runs on a compute tile with PEs and calls;
                        front = in0 + skew and last = in1 + skew, with skew = (rows-1) + (cols-1);
                        on one compute tile the chains' operand streams [in0, in1) never overlap; and
                        out1 = max(last + rows, the previous chain's out1 + rows) on its compute tile
                        (the top link; last + rows for its first chain)

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

# The column schema (unchanged since version 2), as write_tflow() declares it. A column declared with another
# dtype would be decoded at the wrong width and checked as garbage, or not checked at all.
SCHEMA = {
    "residency": {"tile": "u32", "station": "u32", "t0": "f64", "t1": "f64", "flags": "u8"},
    "transit": {"tile": "u32", "op": "u32", "hop": "u8", "mover": "u8", "lane": "u32",
                "t0": "f64", "t1": "f64", "src": "u32", "dst": "u32"},
    "compute": {"op": "u32", "station": "u32", "t0": "f64", "t1": "f64"},
    "op_tiles": {"kind": "u8", "offset": "u32", "tile": "u32", "written": "u8"},
}
# Version 5 adds the L3 address each residency held.
SCHEMA_V5 = {"residency": {"l3_tile": "u32", "slot": "u32", "bank": "u32"}}
# Version 6 adds two tables: each staged copy's L2 slot, and each gemm chain's phases on the fabric.
SCHEMA_V6 = {
    "l2_residency": {"tile": "u32", "cf": "u32", "slot": "u32", "t0": "f64", "t1": "f64", "feeds": "u32"},
    "chain": {"tile": "u32", "station": "u32", "rows": "u32", "cols": "u32", "calls": "u32",
              "in0": "f64", "front": "f64", "in1": "f64", "last": "f64", "out1": "f64"},
}

# enum Hop (tile_transaction_executor.hpp), by value: the station kinds each one moves between.
# Every hop is a push (version 3): writeback is a BlockMover ejecting L3 -> DMA buffer (5), then
# the DMA engine writing its buffer to DRAM (6). There is no DMA read of L3.
HOP_KINDS = [("dram", "l3"), ("l3", "l2"), ("l2", "l1"), ("l1", "l2"), ("l2", "l3"),
             ("l3", "dmabuf"), ("dmabuf", "dram"), ("l3", "l3")]


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
    if m.get("format") != "kpu-tflow":
        raise Unreadable("not a kpu-tflow bundle")
    if m.get("version") == 1:
        # Version 1 has no op_tiles.written, so TF9 cannot tell a filled slot from a re-taken
        # one. Reading it as "nothing writes" would report every result tile as unfilled.
        raise Unreadable("a version-1 bundle has no op_tiles.written column; re-record it with "
                         "this build's kpu-run --tflow")
    if m.get("version") == 2:
        # Version 2's hop 5 is the old DMA read of L3; reading it as an ejection would be wrong.
        raise Unreadable("a version-2 bundle records writeback as a DMA read (hop 5 = "
                         "dma:l3->dram); re-record it with this build's kpu-run --tflow")
    if m.get("version") not in (3, 4, 5, 6):
        raise Unreadable(f"version {m.get('version')!r} is not one this checker reads (3, 4, 5, 6)")
    if m.get("version") in (4, 5, 6) and m.get("ops") != "csp-actions":
        # Versions 4 to 6 are a CSP program's record: an op is one of the program's actions. A
        # bundle that does not say so cannot be read as one, or TF6 would be checked under the wrong rule.
        raise Unreadable(f'a version-{m.get("version")} bundle must say what its ops are ("ops": "csp-actions")')
    if m.get("version") in (5, 6):
        # Version 5: every residency carries its L3 address; the banks make the bank checkable.
        b = m.get("l3_banks")
        if type(b) is not int or b < 1:
            raise Unreadable(f"a version-{m.get('version')} bundle must say its L3 banks (l3_banks), not {b!r}")
    if m.get("version") == 6:
        n = m.get("l2_slots")
        if type(n) is not int or n < 0:
            raise Unreadable(f"a version-6 bundle must say its L2 slots (l2_slots), not {n!r}")
    rec = {"manifest": m}
    _named(m.get("stations"), "manifest stations")
    movers = m.get("movers", [])
    if not isinstance(movers, list) or not all(isinstance(x, dict) and isinstance(x.get("name"), str)
                                               for x in movers):
        raise Unreadable("manifest movers need a string name each")
    try:
        names = ("residency", "transit", "compute", "op_tiles") + \
                (("l2_residency", "chain") if m.get("version") == 6 else ())
        for name in names:
            t = m["tables"][name]
            blob = (d / t["file"]).read_bytes()
            rows = t["rows"]
            cols = {}
            declared = {c["name"]: c["dtype"] for c in t["columns"]}
            schema = dict(SCHEMA.get(name) or SCHEMA_V6[name],
                          **(SCHEMA_V5.get(name, {}) if m.get("version") in (5, 6) else {}))
            for col, dtype in schema.items():
                if declared.get(col) != dtype:
                    raise Unreadable(f"{name}.{col} is declared {declared.get(col)!r}, "
                                     f"the format says {dtype}")
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
            if not isinstance(lm["levels"], list) or not lm["levels"]:
                raise Unreadable("lod has no levels")
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
    if m.get("version") in (4, 5, 6):
        # A CSP PROGRAM's record (versions 4 to 6): an op is one action, and L3 holds a tile only between
        # its program's credit and Release. An accumulator lives in the fabric with no L3 slot,
        # and a Feed reads L2 -- often after the L3 copy was released, which is the point of a
        # release. So the rule is the program's: every transit INTO or OUT OF L3 runs while its
        # tile holds an L3 slot. (A Release before its last L3 reader -- #279's class -- breaks it.)
        l3 = {i for i, s in enumerate(st) if s.get("kind") == "l3"}
        for i in range(ntr):
            if tr["src"][i] not in l3 and tr["dst"][i] not in l3:
                continue
            tile = tr["tile"][i]
            if not held(tile, tr["t0"][i], tr["t1"][i]):
                r.fail("TF6", f"action {tr['op'][i]} moves {tile_name(tile)} through L3 over "
                              f"[{int(tr['t0'][i])}, {int(tr['t1'][i])}) without it holding an L3 slot -- "
                              f"released before its last L3 reader finished")
        nco_v3 = 0
    else:
        nco_v3 = nco
    for i in range(nco_v3):
        op = co["op"][i]
        for tile in op_tiles(op):
            if not held(tile, co["t0"][i], co["t1"][i]):
                r.fail("TF6", f"op {op} computes over [{int(co['t0'][i])}, {int(co['t1'][i])}) but "
                              f"{tile_name(tile)} does not hold an L3 slot for all of it -- "
                              f"released before its last user finished")
    for i in range(ntr if m.get("version") == 3 else 0):
        op = tr["op"][i]
        for tile in op_tiles(op):
            if not held(tile, tr["t0"][i], tr["t1"][i]):
                r.fail("TF6", f"op {op} moves {tile_name(tile)} over [{int(tr['t0'][i])}, "
                              f"{int(tr['t1'][i])}) without it holding an L3 slot -- released "
                              f"before its last user finished")

    # TF7 -------------------------------------------------------------------
    r.ran("TF7")
    chains = {}
    links = {}
    for i in range(ntr):
        hop, src, dst = tr["hop"][i], tr["src"][i], tr["dst"][i]
        if hop >= len(HOP_KINDS) or src >= len(st) or dst >= len(st):
            r.fail("TF7", f"transit {i}: hop {hop} from station {src} to {dst} is out of range")
            continue
        if (st[src]["kind"], st[dst]["kind"]) != HOP_KINDS[hop]:
            r.fail("TF7", f"transit {i}: hop {hop} should move {HOP_KINDS[hop][0]} -> "
                          f"{HOP_KINDS[hop][1]}, not {st[src]['kind']} -> {st[dst]['kind']}")
        chains.setdefault(tr["op"][i], []).append((tr["t0"][i], tr["t1"][i], hop))
        links.setdefault((tr["op"][i], tr["tile"][i]), []).append((tr["t0"][i], tr["t1"][i], src, dst))
    for op, hops in chains.items():
        hops.sort()
        for (a0, a1, ah), (b0, b1, bh) in zip(hops, hops[1:]):
            if b0 < a1:
                r.fail("TF7", f"op {op}: hop {bh} starts at {int(b0)} before hop {ah} finishes at {int(a1)}")
    # Order, not just disjointness: DMA in, then a streamer, then a BlockMover can be disjoint
    # in time and still impossible. A tile's hops for one op must be a connected chain.
    for (op, tile), hops in links.items():
        hops.sort()
        for (_, _, _, a_dst), (b0, _, b_src, _) in zip(hops, hops[1:]):
            if b_src != a_dst:
                r.fail("TF7", f"op {op}: {tile_name(tile)} leaves {st[b_src]['name']} at "
                              f"{int(b0)}, but its previous hop delivered it to {st[a_dst]['name']}")

    # TF9 -------------------------------------------------------------------
    r.ran("TF9")
    DMA_IN, WRITEBACK, L3_TO_L3 = 0, 4, 7   # enum Hop: DmaDramToL3, BlockMoverL2ToL3, BlockMoverL3ToL3
    arrivals = {}
    for i in range(ntr):
        # Only a hop that really ends in L3 fills an L3 slot; TF7 reports the mislabelled ones.
        # Version 4 (a CSP program): a result reaches L3 by its Writeback (hop 4), not by being
        # computed there, so that hop fills a slot too.
        fills = (DMA_IN, L3_TO_L3, WRITEBACK) if m.get("version") in (4, 5, 6) else (DMA_IN, L3_TO_L3)
        if tr["hop"][i] in fills and tr["dst"][i] < len(st) and \
                st[tr["dst"][i]]["kind"] == "l3":
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

    # TF12 ------------------------------------------------------------------
    if m.get("version") in (5, 6):
        r.ran("TF12")
        banks = m["l3_banks"]
        by_slot = {}
        for i in range(nres):
            s_i = res["station"][i]
            if s_i >= len(st) or st[s_i].get("kind") != "l3":
                continue
            slot, bank, cap = res["slot"][i], res["bank"][i], st[s_i].get("capacity", 0)
            if res["l3_tile"][i] != 0:
                r.fail("TF12", f"{tile_name(res['tile'][i])} is placed in L3 module {res['l3_tile'][i]}; a "
                               f"version-5 record is the program's flat L3 (module 0)")
                continue
            if cap and slot >= cap:
                r.fail("TF12", f"{tile_name(res['tile'][i])} is placed in slot {slot} of an L3 with {cap} slots")
                continue
            if bank != slot % banks:
                r.fail("TF12", f"{tile_name(res['tile'][i])} in slot {slot} says bank {bank}; slot {slot} is in "
                               f"bank {slot % banks} of {banks}")
            if res["t1"][i] > res["t0"][i]:                 # a zero-length residency occupies no time
                by_slot.setdefault((s_i, slot), []).append((res["t0"][i], res["t1"][i], i))
        for (s_i, slot), iv in by_slot.items():
            iv.sort()
            for (a0, a1, a), (b0, _b1, b) in zip(iv, iv[1:]):
                if b0 < a1:
                    r.fail("TF12", f"{st[s_i]['name']} slot {slot} holds {tile_name(res['tile'][a])} over "
                                   f"[{int(a0)}, {int(a1)}) and {tile_name(res['tile'][b])} from {int(b0)}: "
                                   f"two tiles in one slot")

    # TF13 ------------------------------------------------------------------
    if m.get("version") == 6:
        r.ran("TF13")
        l2r, nl2 = rec["l2_residency"], rec["l2_residency_rows"]
        cap = m["l2_slots"]
        by_slot, ev = {}, []
        for i in range(nl2):
            who = f"{tile_name(l2r['tile'][i])}" if l2r["tile"][i] < len(tiles) else f"tile {l2r['tile'][i]}"
            t0, t1, slot = l2r["t0"][i], l2r["t1"][i], l2r["slot"][i]
            if slot >= cap:
                r.fail("TF13", f"{who} is staged in L2 slot {slot} of a program with {cap} L2 slots")
                continue
            if not (0 <= t0 <= t1 <= makespan):
                r.fail("TF13", f"{who} holds L2 slot {slot} over [{t0}, {t1}), not inside [0, {makespan}]")
            if l2r["feeds"][i] < 1:
                r.fail("TF13", f"{who} is staged in L2 slot {slot} and never fed")
            if t1 > t0:
                by_slot.setdefault((l2r["cf"][i], slot), []).append((t0, t1, i))
                ev += [(t0, 1, i), (t1, -1, i)]
        for (cf, slot), iv in by_slot.items():
            iv.sort()
            for (a0, a1, a), (b0, _b1, b) in zip(iv, iv[1:]):
                if b0 < a1:
                    r.fail("TF13", f"L2 slot {slot} of compute tile {cf} holds {tile_name(l2r['tile'][a])} over "
                                   f"[{int(a0)}, {int(a1)}) and {tile_name(l2r['tile'][b])} from {int(b0)}: two "
                                   f"copies in one slot")
        ev.sort(key=lambda e: (e[0], e[1]))
        cur = 0
        for t, d, i in ev:
            cur += d
            if cur > cap:
                r.fail("TF13", f"{cur} staged copies held at cycle {int(t)}, more than the program's {cap} L2 slots")

    # TF14 ------------------------------------------------------------------
    if m.get("version") == 6:
        r.ran("TF14")
        ch, nch = rec["chain"], rec["chain_rows"]
        lanes_ch = {}
        for i in range(nch):
            s_i, rows, cols = ch["station"][i], ch["rows"][i], ch["cols"][i]
            if s_i >= len(st) or st[s_i].get("kind") != "cf":
                r.fail("TF14", f"chain {i} runs on station {s_i}, not a compute tile")
                continue
            if rows < 1 or cols < 1 or ch["calls"][i] < 1:
                r.fail("TF14", f"chain {i} has {rows}x{cols} PEs and {ch['calls'][i]} calls")
                continue
            skew = (rows - 1) + (cols - 1)
            in0, front, in1, last, out1 = (ch[k][i] for k in ("in0", "front", "in1", "last", "out1"))
            if front != in0 + skew or last != in1 + skew:
                r.fail("TF14", f"chain {i}: front {int(front)} and last {int(last)} are not in0 {int(in0)} and "
                               f"in1 {int(in1)} plus the skew {skew}")
            if not (in0 <= in1 and out1 >= last + rows and out1 <= makespan):
                r.fail("TF14", f"chain {i}: [in0 {int(in0)}, in1 {int(in1)}] and out1 {int(out1)} are out of order "
                               f"(out1 must be at least last + rows = {int(last + rows)}, within the run)")
            lanes_ch.setdefault(s_i, []).append((in0, in1, out1, rows, last, i))
        for s_i, cs in lanes_ch.items():
            cs.sort()
            for (a0, a1, ao, _ar, _al, a), (b0, _b1, _bo, _br, _bl, b) in zip(cs, cs[1:]):
                if b0 < a1:
                    r.fail("TF14", f"{st[s_i]['name']}: chains {a} and {b} stream operands at once from cycle {int(b0)}")
            # The top link (P5, exactly): a chain's results leave when its far PE's result has travelled
            # its rows north, or rows after the previous chain's results on this compute tile, whichever
            # is later -- out1 = max(last + rows, previous out1 + rows), and last + rows for the first.
            prev = None
            for (_0, _1, out1, rows, last, i) in cs:
                want = last + rows if prev is None else max(last + rows, prev + rows)
                if out1 != want:
                    r.fail("TF14", f"{st[s_i]['name']}: chain {i}'s results leave at {int(out1)}; the top link gives "
                                   f"{int(want)} (last {int(last)} + {rows} rows"
                                   + ("" if prev is None else f", or the previous chain's {int(prev)} + {rows}") + ")")
                prev = out1

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
        if m.get("version") in (5, 6):
            # One row per L3 slot, after the stations and movers (tile_flow_lod.cpp).
            for s in st:
                if s.get("kind") == "l3":
                    want_rows += [(f"{s['name']}/slot[{k}]", "l3_slot") for k in range(s.get("capacity", 0))]
        if m.get("version") == 6:
            # Then one row per L2 slot, and two per compute tile: its chains in and out.
            for s in st:
                if s.get("kind") == "l2":
                    want_rows += [(f"{s['name']}/slot[{k}]", "l2_slot") for k in range(m["l2_slots"])]
            for s in st:
                if s.get("kind") == "cf":
                    want_rows += [(f"{s['name']}/in", "cf_in"), (f"{s['name']}/out", "cf_out")]
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
            elif info["kind"] == "l2" and m.get("version") == 6 and m["l2_slots"] > 0:
                l2r = rec["l2_residency"]
                iv = [(l2r["t0"][i], l2r["t1"][i]) for i in range(rec["l2_residency_rows"])]
            elif info["kind"] == "l2_slot":
                slot = int(info["name"][info["name"].rindex("[") + 1:-1])
                l2r = rec["l2_residency"]
                iv = [(l2r["t0"][i], l2r["t1"][i]) for i in range(rec["l2_residency_rows"]) if l2r["slot"][i] == slot]
            elif info["kind"] in ("cf_in", "cf_out"):
                owner = info["name"][:info["name"].rindex("/")]
                s_own = next((k for k, s in enumerate(st) if s["name"] == owner), -1)
                ch, a, b = rec["chain"], *(("in0", "in1") if info["kind"] == "cf_in" else ("in1", "out1"))
                iv = [(ch[a][i], ch[b][i]) for i in range(rec["chain_rows"]) if ch["station"][i] == s_own]
            elif info["kind"] == "l3_slot":
                # "<station>/slot[k]": the residencies of THAT station placed in slot k.
                slot = int(info["name"][info["name"].rindex("[") + 1:-1])
                owner = info["name"][:info["name"].rindex("/slot[")]
                s_own = next((k for k, s in enumerate(st) if s["name"] == owner), -1)
                iv = [(res["t0"][i], res["t1"][i]) for i in range(nres)
                      if res["station"][i] == s_own and res["slot"][i] == slot]
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
