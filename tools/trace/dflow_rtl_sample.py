#!/usr/bin/env python3
"""
A hand-written RTL-level .dflow record (version 6; docs/plans/shared-tile-flow-record.md, #386).

It stands in for a kpugen run until the RTL writes the format itself. It shows everything a timed
level adds to the record that the L-B oracle doesn't have:
- geometry for L2 (banks x rows, row-to-bank interleave), the accumulator and L1 (the array's edge);
- a cycle column on every event;
- sub-tile progress events (BlockMover rows, Streamer rows with their skew, drained rows);
- credits and tile-ready tokens;
- int8 and int32 values, with expected values beside the actual ones;
- a lower-triangular operand, whose upper elements are out of domain, not zero values.

The program is C = L . B with L lower triangular (8 x 8, int8), B 8 x 4 (int8), C 8 x 4 (int32), in
4 x 4 tiles. L[0,1] lies wholly outside L's domain and is never loaded. The 33 CSP actions below are
the ones kpu-run's L-B executes for PROGRAM, in order and at the same L3 slots. The timing is
invented: a fixed cost per transaction, run back to back. It is a sample of the format, not a
model of the hardware.

Usage:
    python3 dflow_rtl_sample.py -o rtl_sample.dflow

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflow_writer import (CHANNELS, ELEMENTS, EVENT_KINDS, KINDS, LAYERS, NONE8, NONE32, NONE64, PROCESSES,  # noqa: E402
                          Table, pack, write_bundle)

PROGRAM = """csp 1.0
// C = L . B, L lower triangular: L[0,1] is outside the domain and is never loaded
program trmm_8x4 machine flat(l3 = 4) {
  tensor L[8,8] tile 4x4 in;
  tensor B[8,4] tile 4x4 in;
  tensor C[8,4] tile 4x4 out;
  resident B[:, 0];
  for i in 0..2 {
    acc C[i, 0] in fabric {
      for k in 0..i + 1 {
        resident L[i, k];
        call gemm(L[i, k], B[k, 0]) +-> C[i, 0];
        release L[i, k];
      }
    }
    store C[i, 0];
  }
  release B[:, 0];
}
"""
T = 4                                           # tile edge
OPERANDS = [  # name, rows, cols, element, domain
    ("L", 8, 8, "int8", {"kind": "lower", "diagonal": 0}),
    ("B", 8, 4, "int8", {"kind": "dense"}),
    ("C", 8, 4, "int32", {"kind": "dense"}),
]
VARS = ["i", "k"]
I_LINE, K_LINE = 8, 10                          # the `for i` and `for k` lines of PROGRAM


def i_(i):
    return [(0, i, 2, I_LINE)]                  # (var, value, hi, line)


def ik(i, k):
    return i_(i) + [(1, k, i + 1, K_LINE)]


# The program's actions: kind, tile (operand, ti, tj), L3 slot (None: no address), line, loops.
ACTIONS = [
    ("LOAD", ("B", 0, 0), 0, 7, []),
    ("LOAD", ("B", 1, 0), 1, 7, []),
]
for i in range(2):
    for k in range(i + 1):
        lslot = 2 if (i, k) != (1, 1) else 3
        ACTIONS += [
            ("LOAD", ("L", i, k), lslot, 11, ik(i, k)),
            ("MOVE", ("L", i, k), lslot, 12, ik(i, k)),
            ("FEED", ("L", i, k), None, 12, ik(i, k)),
            ("MOVE", ("B", k, 0), k, 12, ik(i, k)),
            ("FEED", ("B", k, 0), None, 12, ik(i, k)),
            ("CALL", ("C", i, 0), None, 12, ik(i, k)),
            ("RELEASE", ("L", i, k), lslot, 13, ik(i, k)),
        ]
    cslot = 3 if i == 0 else 2
    ACTIONS += [
        ("DRAIN", ("C", i, 0), None, 16, i_(i)),
        ("WRITEBACK", ("C", i, 0), cslot, 16, i_(i)),
        ("STORE", ("C", i, 0), cslot, 16, i_(i)),
        ("RELEASE", ("C", i, 0), cslot, 16, i_(i)),
    ]
ACTIONS += [
    ("RELEASE", ("B", 0, 0), 0, 18, []),
    ("RELEASE", ("B", 1, 0), 1, 18, []),
]

# Where each kind moves its tile (csp::from_chan / to_chan), and which process runs it.
MOVES = {"LOAD": ("dram", "l3", "dma"), "STORE": ("l3", "dram", "dma"), "MOVE": ("l3", "l2", "bm"),
         "WRITEBACK": ("l2", "l3", "bm"), "FEED": ("l2", "cf", "str"), "DRAIN": ("cf", "l2", "str"),
         "CALL": (None, None, "cf"), "RELEASE": (None, None, None)}
LANDS = {"LOAD": "l3", "STORE": "dram", "MOVE": "l2", "WRITEBACK": "l3", "FEED": "cf", "DRAIN": "l2", "CALL": "cf"}
OPENS, CLOSES, HAS_PLACE = 1, 2, 8


def values():
    """Every operand's elements, row-major: L lower triangular, B dense, C = L . B."""
    L = [[((r * 3 + c * 5) % 7) - 3 if c <= r else 0 for c in range(8)] for r in range(8)]
    B = [[((r * 2 + c * 3) % 5) - 2 for c in range(4)] for r in range(8)]
    C = [[sum(L[r][x] * B[x][c] for x in range(8)) for c in range(4)] for r in range(8)]
    return {"L": L, "B": B, "C": C}


def tile_of(M, ti, tj):
    return [M[ti * T + r][tj * T + c] for r in range(T) for c in range(T)]


def partial_c(L, B, i, upto):
    """C[i,0] as the accumulator holds it after the calls for k = 0..upto."""
    return [sum(L[i * T + r][k * T + x] * B[k * T + x][c] for k in range(upto + 1) for x in range(T))
            for r in range(T) for c in range(T)]


def build():
    V = values()
    names = [o[0] for o in OPERANDS]
    tiles, tile_at = [], {}

    def tile(t):
        key = (names.index(t[0]), t[1], t[2])
        if key not in tile_at:
            tile_at[key] = len(tiles)
            tiles.append(key)
        return tile_at[key]

    # ---- steps, residencies, positions: the CSP actions, applied ----
    steps, residencies, wheres, loops = [], [], [], []
    live = {}                                  # L3 slot -> its open residency
    held = {"dram": 8, "l3": 0, "l2": 0, "cf": 0}   # every operand tile is in DRAM (L-B's count)
    fed, accs = 0, set()
    prev_where = None
    for a, (kind, t, slot, line, lps) in enumerate(ACTIONS):
        ti = tile(t)
        key = (line, tuple(lps))
        if key != prev_where:
            wheres.append((line, len(loops), len(lps)))
            loops.extend(lps)
            prev_where = key
        frm, to, proc = MOVES[kind]
        res, flags = NONE64, 0
        if slot is not None:
            flags |= HAS_PLACE
            if kind in ("LOAD", "WRITEBACK"):
                res = len(residencies)
                residencies.append({"tile": ti, "open": a, "close": NONE64, "slot": slot,
                                    "flags": 1 if kind == "LOAD" else 0})
                live[slot] = res
                flags |= OPENS
                held["l3"] += 1
            else:
                res = live[slot]
            if kind == "RELEASE":
                residencies[res]["close"] = a
                del live[slot]
                flags |= CLOSES
                held["l3"] -= 1
        if kind == "MOVE" or kind == "DRAIN":
            held["l2"] += 1
        if kind in ("FEED", "WRITEBACK"):
            held["l2"] -= 1
        if kind == "FEED":
            held["cf"] += 1
            fed += 1
        if kind == "CALL":
            held["cf"] -= fed
            fed = 0
            if ti not in accs:
                accs.add(ti)
                held["cf"] += 1
        if kind == "DRAIN":
            held["cf"] -= 1
            accs.discard(ti)
        steps.append({"action": a, "kind": KINDS.index(kind), "tile": ti, "residency": res,
                      "where": len(wheres) - 1, "flags": flags, "slot": slot if slot is not None else NONE32,
                      "process": PROCESSES.index(proc) if proc else NONE8,
                      "from": CHANNELS.index(frm) if frm else NONE8, "to": CHANNELS.index(to) if to else NONE8,
                      "held": dict(held)})

    # ---- the iteration summary: every loop iteration at every depth, and the program ----
    iters = {}
    for s in steps:
        lps = loops[wheres[s["where"]][1]:wheres[s["where"]][1] + wheres[s["where"]][2]]
        for d in range(len(lps) + 1):
            key = tuple((l[3], l[1]) for l in lps[:d])
            it = iters.setdefault(key, {"depth": d, "line": lps[d - 1][3] if d else 0, "value": lps[d - 1][1] if d else 0,
                                        "first": s["action"], "last": s["action"], "actions": 0, "loads": 0,
                                        "stores": 0, "peak": {c: 0 for c in CHANNELS}})
            it["last"] = s["action"]
            it["actions"] += 1
            it["loads"] += KINDS[s["kind"]] == "LOAD"
            it["stores"] += KINDS[s["kind"]] == "STORE"
            for c in CHANNELS:
                it["peak"][c] = max(it["peak"][c], s["held"][c])
    iters = sorted(iters.values(), key=lambda x: (x["depth"], x["first"]))

    # ---- values: the tile each action leaves, its expected value beside it, and every final ----
    snaps, payload = [], bytearray()

    def snap(step, ti, chan, vals, role):
        op = OPERANDS[tiles[ti][0]]
        snaps.append({"step": step, "first": len(payload), "tile": ti, "chan": CHANNELS.index(chan), "role": role})
        payload.extend(pack(op[3], vals))

    calls = {}
    for s in steps:
        kind = KINDS[s["kind"]]
        if kind not in LANDS:
            continue
        op, ti_, tj_ = tiles[s["tile"]]
        if names[op] == "C" and kind == "CALL":
            calls[s["tile"]] = calls.get(s["tile"], -1) + 1
            vals = partial_c(V["L"], V["B"], ti_, calls[s["tile"]])
        else:
            vals = tile_of(V[names[op]], ti_, tj_)
        for role in (0, 1):                    # actual, then the golden model's expected value
            snap(s["action"], s["tile"], LANDS[kind], vals, role)
    for o, (name, rows, cols, _, _) in enumerate(OPERANDS):
        for ti_ in range(rows // T):
            for tj_ in range(cols // T):
                ti = tile((name, ti_, tj_))
                for role in (0, 1):
                    snap(NONE64, ti, "dram", tile_of(V[name], ti_, tj_), role)

    # ---- the events: what the RTL did, with time, progress, credits and tokens ----
    l2_rows, acc_rows = 16, 8
    l2_next, l2_of, acc_of = 0, {}, {}
    ev, clock = [], 0

    def emit(kind, s, src, dst, src_addr=NONE64, dst_addr=NONE64, dur=0, parent=NONE64, row=NONE32, beat=NONE32,
             proc=None, at=None):
        nonlocal clock
        start = clock if at is None else at
        e = {"id": len(ev), "kind": EVENT_KINDS.index(kind), "action": s["action"], "where": s["where"],
             "tile": s["tile"], "process": PROCESSES.index(proc) if proc else NONE8, "lane": 0,
             "src": LAYERS.index(src), "dst": LAYERS.index(dst), "src_addr": src_addr, "dst_addr": dst_addr,
             "parent": parent, "row": row, "beat": beat, "cycle": start, "end": start + dur}
        ev.append(e)
        return e

    def transaction(kind, s, src, dst, src_addr, dst_addr, dur, proc):
        nonlocal clock
        e = emit(kind, s, src, dst, src_addr, dst_addr, dur, proc=proc)
        clock = e["end"]
        return e

    for s in steps:
        kind, ti = KINDS[s["kind"]], s["tile"]
        bytes_ = T * T * ELEMENTS[OPERANDS[tiles[ti][0]][3]]
        if kind == "LOAD":
            e = transaction("DMA", s, "dram", "l3", NONE64, s["slot"], 8 + bytes_ // 8, "dma")
            emit("TOKEN", s, "l3", "l2", proc="dma", at=e["end"])           # tile ready, to the BlockMover
        elif kind == "MOVE":
            row0 = l2_next
            l2_next = (l2_next + T) % l2_rows
            l2_of[ti] = row0
            e = transaction("BLOCK_MOVE", s, "l3", "l2", s["slot"], row0, 2 + T, "bm")
            for r in range(T):                                              # one burst per row
                emit("BM_ROW", s, "l3", "l2", s["slot"], row0 + r, 1, e["id"], r, 0, "bm", at=e["cycle"] + 2 + r)
            emit("TOKEN", s, "l2", "l1", l2_of[ti], proc="bm", at=e["end"])  # tile ready, to the Streamer
        elif kind == "FEED":
            e = transaction("STREAM", s, "l2", "l1", l2_of[ti], NONE64, 2 * T - 1, "str")
            for r in range(T):                                              # row r enters r cycles late: its skew
                emit("STR_ROW", s, "l2", "l1", l2_of[ti] + r, NONE64, T, e["id"], r, r, "str", at=e["cycle"] + r)
            emit("CREDIT", s, "l2", "l3", l2_of[ti], proc="str", at=e["end"])   # its L2 rows are free
        elif kind == "CALL":
            acc_of.setdefault(ti, (tiles[ti][1] * T) % acc_rows)
            transaction("COMPUTE", s, "l1", "acc", NONE64, acc_of[ti], T + 2 * (T - 1), "cf")
        elif kind == "DRAIN":
            row0 = l2_next
            l2_next = (l2_next + T) % l2_rows
            l2_of[ti] = row0
            e = transaction("DRAIN", s, "acc", "l2", acc_of[ti], row0, 1 + T, "str")
            for r in range(T):
                emit("DRAIN_ROW", s, "acc", "l2", acc_of[ti] + r, row0 + r, 1, e["id"], r, 0, "str", at=e["cycle"] + 1 + r)
            emit("CREDIT", s, "acc", "l1", acc_of[ti], proc="str", at=e["end"])  # the accumulator rows are free
            emit("TOKEN", s, "l2", "l3", row0, proc="str", at=e["end"])          # result ready, to the BlockMover
        elif kind == "WRITEBACK":
            e = transaction("BLOCK_MOVE", s, "l2", "l3", l2_of[ti], s["slot"], 2 + T, "bm")
            emit("TOKEN", s, "l3", "dram", s["slot"], proc="bm", at=e["end"])  # result ready, to the DMA
        elif kind == "STORE":
            transaction("DMA", s, "l3", "dram", s["slot"], NONE64, 8 + bytes_ // 8, "dma")
        elif kind == "RELEASE":
            emit("CREDIT", s, "l3", "dram", s["slot"], at=clock)            # the slot is free: credit to the DMA
    cycles = max(e["end"] for e in ev)

    # ---- the manifest and the tables ----
    loads = sum(1 for s in steps if KINDS[s["kind"]] == "LOAD")
    stores = sum(1 for s in steps if KINDS[s["kind"]] == "STORE")
    manifest = {
        "level": "RTL", "producer": "dflow_rtl_sample.py (hand-written sample; stands in for kpugen)",
        "run_id": "", "program": {"name": "trmm_8x4", "text": PROGRAM},
        "machine": {"kind": "flat", "l3": 4, "banks": 1, "place": "fifo", "l2": 0},
        "channels": [{"name": c, "capacity": 4 if c == "l3" else None} for c in CHANNELS],
        "processes": PROCESSES, "kinds": KINDS, "layers": LAYERS, "event_kinds": EVENT_KINDS,
        "operands": [{"name": n, "rows": r, "cols": c, "tile_rows": T, "tile_cols": T, "element": e, "domain": d}
                     for n, r, c, e, d in OPERANDS],
        "vars": VARS, "totals": {"actions": len(steps), "loads": loads, "stores": stores},
        "step_limit": 0, "cut": False, "refusal": None,
        "time": {"unit": "cycle", "clock": None, "cycles": cycles},
        "geometry": {
            "dram": {"modelled": True},
            "l3": {"modelled": True, "slots": 4, "banks": 1},
            "l2": {"modelled": True, "banks": 2, "rows": l2_rows, "row_bytes": 16, "interleave": "row-mod-banks"},
            "acc": {"modelled": True, "banks": 2, "rows": acc_rows, "row_bytes": 16, "interleave": "row-mod-banks"},
            "l1": {"modelled": True, "rows": T, "cols": T},
        },
        "not_modelled": ["DRAM timing: a fixed cost per DMA (a hand-written sample, not an RTL run)",
                         "overlap: transactions run back to back; the RTL pipelines them"],
        "values": {"format": "per-operand", "selected": "all", "cap_bytes": 0, "cut_at": None, "finals": True,
                   "expected": True},
        "result": {"cycles": cycles, "values": "match", "bound": None},
        "lt1": None,
    }
    tabs = []
    t = Table("steps")
    for col, dt, f in [("action", "u64", "action"), ("residency", "u64", "residency"), ("where", "u64", "where"),
                       ("tile", "u32", "tile")]:
        t.put(col, dt, [s[f] for s in steps])
    for c in CHANNELS:
        t.put("held_" + c, "u32", [s["held"][c] for s in steps])
    t.put("l3_tile", "u32", [0 if s["slot"] != NONE32 else NONE32 for s in steps])
    t.put("slot", "u32", [s["slot"] for s in steps])
    t.put("bank", "u32", [0 if s["slot"] != NONE32 else NONE32 for s in steps])
    for col in ("kind", "process", "from", "to", "flags"):
        t.put(col, "u8", [s[col] for s in steps])
    t.put("staging", "u64", [NONE64] * len(steps))
    t.put("l2_cf", "u32", [NONE32] * len(steps))
    t.put("l2_slot", "u32", [NONE32] * len(steps))
    tabs.append(t)
    tabs.append(Table("tiles").put("operand", "u32", [x[0] for x in tiles]).put("ti", "u32", [x[1] for x in tiles])
                .put("tj", "u32", [x[2] for x in tiles]))
    tabs.append(Table("residency").put("open", "u64", [r["open"] for r in residencies])
                .put("close", "u64", [r["close"] for r in residencies]).put("tile", "u32", [r["tile"] for r in residencies])
                .put("l3_tile", "u32", [0] * len(residencies)).put("slot", "u32", [r["slot"] for r in residencies])
                .put("bank", "u32", [0] * len(residencies)).put("chan", "u8", [CHANNELS.index("l3")] * len(residencies))
                .put("flags", "u8", [r["flags"] for r in residencies]))
    tabs.append(Table("where").put("line", "u32", [w[0] for w in wheres]).put("loop_first", "u32", [w[1] for w in wheres])
                .put("loop_count", "u16", [w[2] for w in wheres]))
    tabs.append(Table("loops").put("value", "i64", [l[1] for l in loops]).put("hi", "i64", [l[2] for l in loops])
                .put("var", "u32", [l[0] for l in loops]).put("line", "u32", [l[3] for l in loops]))
    t = Table("iteration")
    for col, dt in [("value", "i64"), ("first", "u64"), ("last", "u64"), ("actions", "u64"), ("loads", "u64"),
                    ("stores", "u64"), ("line", "u32")]:
        t.put(col, dt, [x[col] for x in iters])
    for c in CHANNELS:
        t.put("peak_" + c, "u32", [x["peak"][c] for x in iters])
    t.put("depth", "u16", [x["depth"] for x in iters])
    tabs.append(t)
    t = Table("snapshots")
    for col, dt in [("step", "u64"), ("first", "u64"), ("tile", "u32")]:
        t.put(col, dt, [x[col] for x in snaps])
    t.put("height", "u32", [T] * len(snaps)).put("width", "u32", [T] * len(snaps))
    t.put("chan", "u8", [x["chan"] for x in snaps]).put("role", "u8", [x["role"] for x in snaps])
    tabs.append(t)
    tabs.append(Table("elements").put("b", "u8", payload))
    t = Table("events")
    for col, dt in [("id", "u64"), ("parent", "u64"), ("action", "u64"), ("where", "u64"), ("src_addr", "u64"),
                    ("dst_addr", "u64"), ("cycle", "u64"), ("end", "u64"), ("tile", "u32"), ("lane", "u32"),
                    ("row", "u32"), ("beat", "u32"), ("kind", "u8"), ("process", "u8"), ("src", "u8"), ("dst", "u8")]:
        t.put(col, dt, [e[col] for e in ev])
    tabs.append(t)
    # The L-B oracle's overlay and staging tables: empty here (an RTL run has no L-T1 overlay).
    for name, cols in [("lt1_legs", [("action", "u64"), ("start", "u64"), ("finish", "u64"), ("slot_wait", "u64"),
                                     ("lane", "u32"), ("kind", "u8"), ("proc", "u8"), ("leg", "u8")]),
                       ("lt1_tiles", [("max_abs", "f64"), ("max_rel", "f64"), ("tile", "u32"), ("pass", "u8")]),
                       ("lt1_chains", [(c, "u64") for c in ("first_action", "last_action", "drain_action", "in0",
                                                            "front", "in1", "last", "out1", "link_delay")]
                        + [(c, "u32") for c in ("tile", "lane", "used_rows", "used_cols", "calls")]),
                       ("lt1_calls", [("action", "u64"), ("chain", "u32"), ("pos", "u32")]),
                       ("l2_staging", [("open", "u64"), ("close", "u64"), ("residency", "u64"), ("feeds", "u64"),
                                       ("tile", "u32"), ("cf", "u32"), ("slot", "u32")])]:
        t = Table(name)
        for c, dt in cols:
            t.put(c, dt, [])
        tabs.append(t)
    return manifest, tabs


def main():
    ap = argparse.ArgumentParser(description="Write the hand-written RTL-level .dflow sample.")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    manifest, tables = build()
    d = write_bundle(a.output, manifest, tables)
    ev = next(t for t in tables if t.name == "events")
    print(f"wrote {d} (RTL sample: {manifest['totals']['actions']} actions, {ev.rows} events, "
          f"{manifest['time']['cycles']} cycles to result)")


if __name__ == "__main__":
    main()
