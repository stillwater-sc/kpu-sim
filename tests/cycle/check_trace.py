#!/usr/bin/env python3
"""Validate a cycle-core trace (Chrome trace-event JSON, as Perfetto reads it) against its counters file.

    check_trace.py <trace.json> <counters.json>

The format: every event has a known phase and the fields it needs; complete events ("X") on one
thread never overlap, since Perfetto nests overlapping slices; every async slice ("b"/"e") is closed,
after it opened; every thread with slices is named. Against the counters (#535,
docs/plans/cycle-based-kpu-prototype.md §3.3): each component's busy and stall slices add up to its
busy and stall cycles times its domain's period, and each link has one slice per beat it carried.

Exit 0 if valid, 1 with the violations if not, 2 if a file cannot be read. Standard library only.
"""

import json
import sys
from collections import defaultdict

EPS = 1e-6  # microseconds


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    try:
        with open(argv[1]) as f:
            trace = json.load(f)
        with open(argv[2]) as f:
            counters = json.load(f)
    except (OSError, ValueError) as e:
        print(f"check_trace: {e}")
        return 2

    bad = []
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        print("check_trace: no traceEvents list")
        return 1

    names = {}                      # tid -> thread name
    slices = defaultdict(list)      # tid -> [(ts, dur, name)]
    opened, closed = {}, {}         # (cat, name, id) -> ts
    for i, e in enumerate(events):
        ph = e.get("ph")
        if ph not in ("M", "X", "b", "e", "C"):
            bad.append(f"event {i}: unknown phase {ph!r}")
            continue
        if not isinstance(e.get("pid"), int) or not isinstance(e.get("tid"), int):
            bad.append(f"event {i}: pid and tid must be integers")
            continue
        if ph == "M":
            if e.get("name") == "thread_name":
                names[e["tid"]] = e["args"]["name"]
            continue
        ts = e.get("ts")
        if not isinstance(ts, (int, float)) or ts < 0:
            bad.append(f"event {i}: ts {ts!r}")
            continue
        if ph == "X":
            dur = e.get("dur")
            if not isinstance(dur, (int, float)) or dur < 0:
                bad.append(f"event {i}: dur {dur!r}")
                continue
            slices[e["tid"]].append((ts, dur, e["name"]))
        elif ph in ("b", "e"):
            ident = e.get("id2", {}).get("local") if isinstance(e.get("id2"), dict) else e.get("id")
            if ident is None:
                bad.append(f"event {i}: async {ph} without an id")
                continue
            key = (e.get("cat"), e.get("name"), ident)
            book = opened if ph == "b" else closed
            if key in book:
                bad.append(f"event {i}: async {ph} {key} twice")
            book[key] = ts
        elif ph == "C":
            if not all(isinstance(v, (int, float)) for v in e.get("args", {}).values()):
                bad.append(f"event {i}: counter values must be numbers")

    for tid, ss in slices.items():
        if tid not in names:
            bad.append(f"thread {tid} has slices but no thread_name")
        ss.sort()
        for (t0, d0, n0), (t1, _, n1) in zip(ss, ss[1:]):
            if t1 < t0 + d0 - EPS:
                bad.append(f"thread {names.get(tid, tid)}: {n0!r} at {t0} overlaps {n1!r} at {t1}")
    for key, t in opened.items():
        if key not in closed:
            bad.append(f"async slice {key} never closed")
        elif closed[key] < t - EPS:
            bad.append(f"async slice {key} closes before it opens")
    for key in closed:
        if key not in opened:
            bad.append(f"async slice {key} closed but never opened")

    # Against the counters.
    period = {d["name"]: d["period_ps"] / 1e6 for d in counters["domains"]}   # microseconds
    by_thread = {name: tid for tid, name in names.items()}
    beats = defaultdict(int)
    for (cat, name, _id) in opened:
        beats[name] += 1
    for r in counters["rows"]:
        where = f"{r['kind']} {r['name']}"
        if r["kind"] == "component":
            tid = by_thread.get(f"{r['name']} ({r['domain']})")
            sums = defaultdict(float)
            for _ts, dur, name in slices.get(tid, []):
                sums[name] += dur
            want = {"busy": r["busy"]}
            want.update({f"stall: {cause}": n for cause, n in r["stalls"].items()})
            for name, cycles in want.items():
                got = sums.get(name, 0.0)
                if abs(got - cycles * period[r["domain"]]) > EPS * max(1, cycles):
                    bad.append(f"{where}: {name} slices span {got} us; the counters say {cycles} cycles")
        elif beats[r["name"]] != r["C"]:
            bad.append(f"{where}: {beats[r['name']]} beat slices; the counters say {r['C']} beats popped")

    if bad:
        print(f"check_trace: {len(bad)} violation(s)")
        for b in bad[:50]:
            print("  " + b)
        return 1
    print(f"check_trace: OK ({len(events)} events, {len(slices)} component tracks, "
          f"{len(opened)} beats, {len(counters['rows'])} counter rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
