#!/usr/bin/env python3
"""
Memory-flow record checker (docs/plans/memory-side-debugger.md step 4; DRAM plan step 4).

Checks a .mflow bundle (written by `kpu-memsim --out`) against the invariants every run of the
memory side must satisfy, so a DRAM-timing, window or buffer bug becomes a failing check instead
of a reviewing problem.

Usage:
    python3 mflow_check.py <bundle.mflow> [--json] [--verbose]

Exit codes (the same contract as tflow_check.py and the LPDDR5 trace_validator.py):
    0  every invariant holds
    1  one or more invariants are violated -- parse them and fix the root cause
    2  the bundle could not be read -- there are no violations to parse

Invariants:
    TF10 one open row   per bank, in issue order: an ACT opens a closed bank; every RD/WR names
                        the open row; a PRE closes the open row; a REF finds the bank closed. A
                        RD/WR also names its burst's bank and row, and moves its direction.
    TF11 data bus       a channel's data bus never carries two bursts at once
    M1   window         an engine never has more than W bursts in flight
    M2   store buffer   a store buffer never holds more than its capacity, nor stages more than
                        it holds
    M3   credit first   a request's lifetime is ordered: offered <= credit <= first burst <=
                        last burst <= retired. A load's first burst never precedes its L3 credit;
                        a store's never precedes its buffer slot
    M4   bursts         every burst completes exactly once: its times are ordered inside the run,
                        exactly one RD/WR serves it, and every request's bursts are exactly the
                        ones its bytes span, from its own engine
    M5   DRAM timing    against the DECLARED device's table (the manifest's dram_timing, the
                        table the controller ran -- not the LPDDR5-6400 table trace_validator.py
                        hard-codes): per bank tRCD, tRP, tRAS, tRC, tRFCpb; per bank group
                        tRRD_L, tCCD_L; per channel tRRD_S, tCCD_S, tFAW

TF10, TF11 and M5 are checked on the controller's clock ticks (the k_* columns), not executor
cycles: an executor cycle is several ticks (dram_timing.ticks_per_cycle), so a constraint a tick
short would be invisible at cycle grain, and two back-to-back bursts can look overlapped.

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import array
import json
import sys
from pathlib import Path

VERSION = 2
NONE = 0xFFFFFFFF
DTYPES = {"u8": ("B", 1), "u32": ("I", 4), "u64": ("Q", 8), "f64": ("d", 8)}
ACT, RD, WR, PRE, REF = range(5)                      # enum CommandKind, in order
CMD = ["ACT", "RD", "WR", "PRE", "REF"]
EMPTY_OUTCOMES = 4                                    # enum Outcome has 4 values

# The column schema, as write_mflow() declares it.
SCHEMA = {
    "bursts": {"engine": "u32", "request": "u32", "row": "u32", "col": "u32", "mc": "u8",
               "channel": "u8", "rank": "u8", "bank_group": "u8", "bank": "u8", "is_load": "u8",
               "outcome": "u8", "t_submit": "f64", "t_cmd": "f64", "t_data0": "f64",
               "t_data1": "f64", "t_done": "f64"},
    "commands": {"row": "u32", "burst": "u32", "mc": "u8", "channel": "u8", "bank_group": "u8",
                 "bank": "u8", "kind": "u8", "t_issue": "f64", "t_end": "f64", "t_data0": "f64",
                 "t_data1": "f64", "k_issue": "u64", "k_end": "u64", "k_data0": "u64",
                 "k_data1": "u64"},
    "requests": {"address": "u64", "engine": "u32", "port": "u32", "bytes": "u32",
                 "is_load": "u8", "t_offered": "f64", "t_credit": "f64", "t_first": "f64",
                 "t_last": "f64", "t_retired": "f64"},
    "buffers": {"engine": "u32", "held": "u32", "staged": "u32", "t": "f64"},
    "ports": {"port": "u32", "engine": "u32", "request": "u32", "kind": "u8", "t": "f64"},
}
TIMING = ["tRCD", "tRP", "tRAS", "tRC", "tRFCpb", "tRRD_L", "tRRD_S", "tCCD_L", "tCCD_S", "tFAW"]


class Unreadable(Exception):
    """The bundle cannot be read (exit 2)."""


def _count(v, what):
    """A count or offset from a JSON file: an int, and not a bool (which Python treats as one)."""
    if type(v) is not int or v < 0:
        raise Unreadable(f"{what} is {v!r}, not a non-negative integer")
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
        if off > len(blob) or rows > (len(blob) - off) // size:
            raise Unreadable(f"column {name} runs past the end of its file")
        a = array.array(code)
        a.frombytes(blob[off:off + rows * size])
        if sys.byteorder != "little":
            a.byteswap()
        return a
    raise Unreadable(f"no column {name}")


def load(path):
    d = Path(path)
    try:
        m = json.loads((d / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        raise Unreadable(f"manifest.json: {e}") from e
    if not isinstance(m, dict) or m.get("format") != "kpu-mflow":
        raise Unreadable("not a kpu-mflow bundle")
    if m.get("version") == 1:
        # Version 1 has executor cycles only: too coarse to check DRAM timing against.
        raise Unreadable("a version-1 bundle has no controller-clock columns or timing table; "
                         "re-record it with this build's kpu-memsim")
    if m.get("version") != VERSION:
        raise Unreadable(f"version {m.get('version')!r} is not one this checker reads ({VERSION})")
    rec = {"manifest": m}
    try:
        _count(m["makespan"], "makespan")
        st = m["stations"]
        if not isinstance(st, list) or not all(isinstance(s, dict) and isinstance(s.get("name"), str)
                                               and isinstance(s.get("kind"), str) for s in st):
            raise Unreadable("manifest stations need a string name and kind each")
        for s in st:
            _count(s.get("capacity"), f"the capacity of {s['name']}")
        dt = m["dram_timing"]
        params = dt["params"]
        for p in TIMING:
            _count(params.get(p), f"dram_timing.params.{p}")
        if not isinstance(dt["ticks_per_cycle"], (int, float)) or dt["ticks_per_cycle"] <= 0:
            raise Unreadable("dram_timing.ticks_per_cycle is not a positive number")
        for name, schema in SCHEMA.items():
            t = m["tables"][name]
            rows = _count(t["rows"], f"the row count of {name}")
            declared = {c["name"]: c["dtype"] for c in t["columns"]}
            for col, dtype in schema.items():
                if declared.get(col) != dtype:
                    raise Unreadable(f"{name}.{col} is declared {declared.get(col)!r}, "
                                     f"the format says {dtype}")
            blob = (d / t["file"]).read_bytes()
            rec[name] = {col: _column(blob, t, col, rows) for col in schema}
            rec[name + "_rows"] = rows
    except (OSError, KeyError, TypeError, AttributeError) as e:
        raise Unreadable(f"{e!r}") from e
    # Every index the checks follow is checked once, here: a bad one is unreadable, not a crash.
    C, B = rec["commands"], rec["bursts"]
    for i in range(rec["commands_rows"]):
        if C["kind"][i] > REF:
            raise Unreadable(f"command {i} has kind {C['kind'][i]}")
        if C["burst"][i] != NONE and C["burst"][i] >= rec["bursts_rows"]:
            raise Unreadable(f"command {i} names burst {C['burst'][i]}, past the bursts table")
    for i in range(rec["bursts_rows"]):
        if B["request"][i] != NONE and B["request"][i] >= rec["requests_rows"]:
            raise Unreadable(f"burst {i} names request {B['request'][i]}, past the requests table")
        if B["outcome"][i] >= EMPTY_OUTCOMES:
            raise Unreadable(f"burst {i} has outcome {B['outcome'][i]}")
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


def _peak_over(intervals, cap):
    """The first (t, level) at which half-open intervals exceed `cap`; ends before starts."""
    ev = sorted([(a, 1) for a, b in intervals if b > a] + [(b, -1) for a, b in intervals if b > a])
    cur = 0
    for t, d in ev:
        cur += d
        if cur > cap:
            return t, cur
    return None


def _index(stations, kind):
    """{instance index: capacity} of a station kind named like dev/dma[3]."""
    out = {}
    for s in stations:
        if s["kind"] != kind:
            continue
        name = s["name"]
        key = name[name.rindex("[") + 1:name.rindex("]")]
        out[int(key)] = s["capacity"]
    return out


def check(rec):
    r = Report()
    m = rec["manifest"]
    makespan = m["makespan"]
    T = m["dram_timing"]["params"]
    C, B, Q, F = rec["commands"], rec["bursts"], rec["requests"], rec["buffers"]
    nc, nb, nq, nf = rec["commands_rows"], rec["bursts_rows"], rec["requests_rows"], rec["buffers_rows"]
    bb = m.get("burst_bytes", 0)

    def where(i):
        return (f"mc[{C['mc'][i]}]/ch[{C['channel'][i]}]/bg[{C['bank_group'][i]}]/ba[{C['bank'][i]}]")

    def cmd(i):
        return f"{CMD[C['kind'][i]]} #{i} (tick {C['k_issue'][i]}, cycle {int(C['t_issue'][i])})"

    # Commands grouped by bank, bank group and channel, in issue order. Ticks are per controller;
    # the record keeps a controller's commands in issue order, which breaks ties within a tick.
    banks, groups, channels = {}, {}, {}
    for i in range(nc):
        mc, ch, bg, ba = C["mc"][i], C["channel"][i], C["bank_group"][i], C["bank"][i]
        banks.setdefault((mc, ch, bg, ba), []).append(i)
        groups.setdefault((mc, ch, bg), []).append(i)
        channels.setdefault((mc, ch), []).append(i)
    for d in (banks, groups, channels):
        for v in d.values():
            v.sort(key=lambda i: (C["k_issue"][i], i))

    # TF10 ------------------------------------------------------------------
    r.ran("TF10")
    for key, cmds in banks.items():
        open_row = None
        for i in cmds:
            k, row = C["kind"][i], C["row"][i]
            if k == ACT:
                if open_row is not None:
                    r.fail("TF10", f"{where(i)}: {cmd(i)} opens row {row} while row {open_row} is open")
                open_row = row
            elif k in (RD, WR):
                if open_row is None:
                    r.fail("TF10", f"{where(i)}: {cmd(i)} of row {row} with no row open")
                elif row != open_row:
                    r.fail("TF10", f"{where(i)}: {cmd(i)} names row {row}, but row {open_row} is open")
            elif k == PRE:
                if open_row is not None and row != open_row:
                    r.fail("TF10", f"{where(i)}: {cmd(i)} closes row {row}, but row {open_row} is open")
                open_row = None
            elif k == REF:
                if open_row is not None:
                    r.fail("TF10", f"{where(i)}: {cmd(i)} refreshes with row {open_row} open")
    for i in range(nc):
        if C["kind"][i] not in (RD, WR):
            continue
        b = C["burst"][i]
        if b == NONE:
            r.fail("TF10", f"{where(i)}: {cmd(i)} serves no burst")
            continue
        same = all(B[f][b] == C[f][i] for f in ("mc", "channel", "bank_group", "bank", "row"))
        if not same:
            r.fail("TF10", f"{where(i)}: {cmd(i)} serves burst {b} at mc[{B['mc'][b]}]/ch[{B['channel'][b]}]"
                           f"/bg[{B['bank_group'][b]}]/ba[{B['bank'][b]}] row {B['row'][b]}")
        if (C["kind"][i] == RD) != bool(B["is_load"][b]):
            r.fail("TF10", f"{where(i)}: {cmd(i)} serves burst {b}, a "
                           f"{'load' if B['is_load'][b] else 'store'}")

    # TF11 ------------------------------------------------------------------
    r.ran("TF11")
    for (mc, ch), cmds in channels.items():
        cas = sorted((C["k_data0"][i], C["k_data1"][i], i) for i in cmds if C["kind"][i] in (RD, WR))
        end, owner = -1, None
        for a, b, i in cas:
            if b < a:
                r.fail("TF11", f"mc[{mc}]/ch[{ch}]: {cmd(i)} has its data window end before it starts")
                continue
            if b == a:
                continue
            if a < end:
                r.fail("TF11", f"mc[{mc}]/ch[{ch}]: {cmd(i)} drives the data bus at tick {a} while "
                               f"{cmd(owner)} holds it until tick {end}")
            if b > end:
                end, owner = b, i

    # M1 --------------------------------------------------------------------
    r.ran("M1")
    window = _index(m["stations"], "dma")
    flights = {}
    for i in range(nb):
        flights.setdefault(B["engine"][i], []).append((B["t_submit"][i], B["t_done"][i]))
    for e, iv in flights.items():
        if e not in window:
            r.fail("M1", f"bursts name engine {e}, which has no dma station")
            continue
        over = _peak_over(iv, window[e])
        if over:
            r.fail("M1", f"dma[{e}]: {over[1]} bursts in flight at cycle {int(over[0])}, over its "
                         f"window of {window[e]}")

    # M2 --------------------------------------------------------------------
    r.ran("M2")
    depth = _index(m["stations"], "dmabuf")
    for i in range(nf):
        e, held, staged, t = F["engine"][i], F["held"][i], F["staged"][i], int(F["t"][i])
        if e not in depth:
            r.fail("M2", f"buffer sample {i} names engine {e}, which has no dmabuf station")
        elif held > depth[e]:
            r.fail("M2", f"dmabuf[{e}]: {held} slots held at cycle {t}, over its {depth[e]}")
        if staged > held:
            r.fail("M2", f"dmabuf[{e}]: {staged} stores staged at cycle {t} but only {held} held")

    # M3 --------------------------------------------------------------------
    r.ran("M3")
    for q in range(nq):
        pts = [("offered", Q["t_offered"][q]), ("credit", Q["t_credit"][q]), ("first burst", Q["t_first"][q]),
               ("last burst", Q["t_last"][q]), ("retired", Q["t_retired"][q])]
        for (an, a), (bn, b) in zip(pts, pts[1:]):
            if b < a:
                kind = "load" if Q["is_load"][q] else "store"
                r.fail("M3", f"request {q} ({kind}, engine {Q['engine'][q]}): {bn} at cycle {int(b)} "
                             f"precedes its {an} at cycle {int(a)}")
        if Q["t_retired"][q] > makespan:
            r.fail("M3", f"request {q} retires at cycle {int(Q['t_retired'][q])}, after the run ends at {makespan}")

    # M4 --------------------------------------------------------------------
    r.ran("M4")
    served = [0] * nb
    for i in range(nc):
        if C["kind"][i] in (RD, WR) and C["burst"][i] != NONE:
            served[C["burst"][i]] += 1
    of_request = [0] * nq
    for b in range(nb):
        ts = [B["t_submit"][b], B["t_cmd"][b], B["t_data0"][b], B["t_data1"][b], B["t_done"][b]]
        if any(y < x for x, y in zip(ts, ts[1:])) or ts[-1] > makespan:
            r.fail("M4", f"burst {b}: submit {ts[0]:.0f}, first command {ts[1]:.0f}, data "
                         f"{ts[2]:.0f}..{ts[3]:.0f}, done {ts[4]:.0f} are not ordered inside the run")
        if served[b] != 1:
            r.fail("M4", f"burst {b}: served by {served[b]} RD/WR commands, not exactly one")
        q = B["request"][b]
        if q == NONE:
            r.fail("M4", f"burst {b} belongs to no request")
            continue
        of_request[q] += 1
        if B["engine"][b] != Q["engine"][q] or B["is_load"][b] != Q["is_load"][q]:
            r.fail("M4", f"burst {b} (engine {B['engine'][b]}) is not its request {q}'s "
                         f"(engine {Q['engine'][q]}, {'load' if Q['is_load'][q] else 'store'})")
    if bb > 0:
        for q in range(nq):
            a, n = Q["address"][q], Q["bytes"][q]
            want = -(-(a + n) // bb) - a // bb if n else 0
            if of_request[q] != want:
                r.fail("M4", f"request {q}: {of_request[q]} bursts, but its {n} bytes at {a:#x} span {want}")
    else:
        r.fail("M4", "the manifest declares no burst_bytes; request spans cannot be checked")

    # M5 --------------------------------------------------------------------
    r.ran("M5")

    def gap(inv_param, a, b, what):
        need = T[inv_param]
        d = C["k_issue"][b] - C["k_issue"][a]
        if d < need:
            r.fail("M5", f"{what}: {inv_param} = {need} ticks, but {cmd(b)} follows {cmd(a)} by {d}")

    for key, cmds in banks.items():
        last_act = last_pre = last_ref = None
        for i in cmds:
            k = C["kind"][i]
            if k == ACT:
                if last_pre is not None:
                    gap("tRP", last_pre, i, where(i))
                if last_act is not None:
                    gap("tRC", last_act, i, where(i))
                if last_ref is not None:
                    gap("tRFCpb", last_ref, i, where(i))
                last_act = i
            elif k in (RD, WR) and last_act is not None:
                gap("tRCD", last_act, i, where(i))
            elif k == PRE:
                if last_act is not None:
                    gap("tRAS", last_act, i, where(i))
                last_pre = i
            elif k == REF:
                last_ref = i
    for (mc, ch, bg), cmds in groups.items():
        what = f"mc[{mc}]/ch[{ch}]/bg[{bg}]"
        acts = [i for i in cmds if C["kind"][i] == ACT]
        cas = [i for i in cmds if C["kind"][i] in (RD, WR)]
        for a, b in zip(acts, acts[1:]):
            gap("tRRD_L", a, b, what)
        for a, b in zip(cas, cas[1:]):
            gap("tCCD_L", a, b, what)
    for (mc, ch), cmds in channels.items():
        what = f"mc[{mc}]/ch[{ch}]"
        acts = [i for i in cmds if C["kind"][i] == ACT]
        cas = [i for i in cmds if C["kind"][i] in (RD, WR)]
        for a, b in zip(acts, acts[1:]):
            gap("tRRD_S", a, b, what)
        for a, b in zip(cas, cas[1:]):
            gap("tCCD_S", a, b, what)
        for a, b in zip(acts, acts[4:]):            # at most four ACTs in any tFAW window
            gap("tFAW", a, b, what + " (five activates)")
    return r


def main():
    ap = argparse.ArgumentParser(description="Check a .mflow bundle against the memory-side invariants.")
    ap.add_argument("bundle")
    ap.add_argument("--json", "-j", action="store_true", help="machine-readable output")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    try:
        rec = load(args.bundle)
    except Unreadable as e:
        print(f"mflow_check: cannot read {args.bundle}: {e}", file=sys.stderr)
        sys.exit(2)
    rep = check(rec)
    ok = not rep.violations
    if args.json:
        print(json.dumps({"bundle": args.bundle, "passed": ok, "checked": rep.checked,
                          "violations": rep.violations}, indent=1))
    else:
        m = rec["manifest"]
        print(f"mflow_check: {args.bundle} ({m['device']}, makespan {m['makespan']}, "
              f"{rec['bursts_rows']} bursts, {rec['commands_rows']} commands)")
        if args.verbose:
            T = m["dram_timing"]
            print(f"  timing ({T['ticks_per_cycle']:.4g} ticks per cycle): " +
                  " ".join(f"{p}={T['params'][p]}" for p in TIMING))
        print(f"  checked {' '.join(rep.checked)}")
        for v in rep.violations:
            print(f"  {v['invariant']}  {v['message']}")
        print("  PASS" if ok else f"  FAIL: {len(rep.violations)} violation(s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
