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
    D8 no time at L-B         no time column (f64, or a t_*, start or finish name) in any table but
                              the L-T1 overlay's (version 3), which is L-T1's time, not L-B's
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

Version 3 adds the L-T1 overlay (lt1_legs, lt1_tiles; the manifest's lt1 says whether it ran):
    L1 references             a leg's action is an action of the run, its kind and process exist,
                              its lane is one of its process's lanes, and a credit action (Release,
                              Inherit, Retain) has no process; a proximity row's tile exists. With
                              no overlay, both tables are empty
    L2 the join               a leg's kind is its action's kind; every recorded action has legs,
                              numbered 0, 1, ... in order
    L3 timing                 start <= finish <= the makespan, which is the latest finish; the legs
                              on one process lane never overlap
    L4 slot waits             the legs' slot waits add up to the manifest's slot_wait_cycles, the
                              legs with one count slot_bound, only Loads wait on a slot, and no
                              wait is longer than the cycles before the Load's start
    L5 proximity              one row per tile of every operand; the manifest's pass is every
                              tile's; the errors are finite and non-negative

D1 (the record is the oracle's run) and D7 (the viewer never executes) are properties of the code
that writes and reads the bundle, not of its contents.

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
Version 4 adds the fabric pipeline (lt1_chains, lt1_calls; manifest lt1.fabric and lt1.one_leg;
docs/plans/l2-reuse-and-fabric-pipeline-view.md step 2). Each accumulator's gemm chain that L-T1
timed as an output-stationary systolic stream, on r x c PEs with skew = (r-1) + (c-1):
    P1 references             a chain's tile, compute tile (a cf lane) and actions exist, and its
                              PEs fit the declared array; a call row names a recorded Call and a
                              chain. With no fabric declared, both tables are empty
    P2 the wavefront          front - in0 == skew; a chain whose calls are over (at its Drain, or
                              when the next accumulator took its compute tile) has last - in1 == skew and
                              out1 - last == r + link_delay; in0 <= in1. front and in1 are not
                              ordered: front > in1 exactly when the chain is shorter than the skew
    P3 the calls              a chain's calls are numbered 1..calls in action order, each on its
                              chain's cf lane, the first starting at in0 and the last ending at in1
    P4 one chain per array    on one cf lane, chains' operand streams [in0, in1) never overlap,
                              and a chain's Drain starts no earlier than its out1
    P5 the top link           each chain's out1 is at least the previous chain's on its lane plus
                              r, and link_delay is exactly what that cost; the manifest's chain
                              count and link_delay_cycles are the tables'

Version 5 adds L2 staging (l2_staging; the steps' staging, l2_cf, l2_slot; machine.l2;
docs/plans/l2-reuse-and-fabric-pipeline-view.md step 3). A `stage` block's Move copies a resident
tile into an L2 slot, the calls inside it feed the copy, and its L2_RELEASE frees the slot:
    S1 one staging per slot   after every step each L2 slot holds at most one open staging, and
                              every slot is inside the program's l2
    S2 the address            a staging's Move (flag stages) and L2_RELEASE are its open and
                              close, a step with an L2 address names its staging and that
                              staging's slot and tile, a staged Feed is inside the staging's life,
                              and only Moves, Feeds and L2_RELEASEs carry an L2 address
    S3 reuse counts           with every step recorded, a staging's feeds are its staged Feeds
    S4 all free at the end    every staging closes after it opens (only a refused run leaves one
                              open)

Version 6 is the record kpu-sim and the KPU RTL share (docs/plans/shared-tile-flow-record.md; #386):
the level and producer, time where the level has it, each layer's geometry, the events both sides
emit (with identities), typed element values, operand domains, expected values and the result:
    G1 geometry               every layer (dram, l3, l2, acc, l1) is declared, modelled or not (and
                              then why); a modelled L3 is the machine's slots and banks; a modelled
                              L2 or accumulator has banks, rows, a row width and a known interleave;
                              a modelled L1 is the array's edge (rows x cols)
    T1 time                   an L-B record has no time (no manifest time, no cycle or end column,
                              no result cycles); a timed level has all three, every event has
                              cycle <= end, the latest end is the time's cycles to result, and the
                              result's cycles are the same number
    E1 references             an event's id is its row; its kind, process, layers, tile, action,
                              position and parent exist; its layers are modelled and its addresses
                              inside them; a transaction moves between the layers its kind names
    E2 identities             no two events share a key (kind, process, tile, layers and addresses,
                              loop indices, row, beat), and none shares (kind, action, row, beat)
    E3 progress               a row event (BM_ROW, STR_ROW, DRAIN_ROW) is a child of a transaction
                              of its kind, of the same tile and action, its row inside the tile, and
                              (timed) inside its parent's [cycle, end]; only row events have rows
    E4 credits and tokens     a TOKEN goes downstream along a data edge after its tile arrived at its
                              source; a CREDIT goes upstream against one after its tile left its
                              source
    E5 the join               a transaction's action is a step of the matching kind and tile, at the
                              step's L3 slot; when every transaction names its action, each recorded
                              data step has exactly one
    X1 element types          every operand names a known element type; a snapshot holds exactly
                              height x width elements of it; its role is actual or expected
    X2 domains                every operand's domain is well formed, and every element outside it
                              is stored as zero bytes (the viewer draws it as out of domain, never
                              as a zero value)
    X3 expected values        an expected snapshot sits beside an actual one (same tile, step,
                              channel, shape), once; with expected values every final has one; the
                              result's values say match exactly when every final equals its
                              expected (bit for bit, or within result.bound for floating types)

Version 6.1 (manifest "minor": 1; #398) describes two things the RTL does: a DRAIN from the
accumulator straight to L3 (it joins the WRITEBACK action and stands for the tile's DRAIN step
too), and an L3 declared as flat word-addressed memory (geometry.l3 words and word_bytes instead
of slots and banks): its addresses are words, a residency keeps one, and no credit names it. A
6.0 record may use neither, and an L3 declared both ways is refused (G1).

    --result prints only the result -- the level, the producer, the cycles to result and a digest
    of each operand's final values -- which is what CI compares (D3), never the event sequence.
"""

import argparse
import array
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

VERSIONS = (1, 2, 3, 4, 5, 6)
NONE8, NONE32, NONE64 = 0xFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
DTYPES = {"u8": ("B", 1), "u16": ("H", 2), "u32": ("I", 4), "u64": ("Q", 8), "i64": ("q", 8),
          "f32": ("f", 4), "f64": ("d", 8)}
OPENS, CLOSES, RETAINS, HAS_PLACE, PINNED, HAS_L2, STAGES = 1, 2, 4, 8, 16, 32, 64
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
SCHEMA_V3 = {
    "lt1_legs": {"action": "u64", "start": "u64", "finish": "u64", "slot_wait": "u64", "lane": "u32",
                 "kind": "u8", "proc": "u8", "leg": "u8"},
    "lt1_tiles": {"max_abs": "f64", "max_rel": "f64", "tile": "u32", "pass": "u8"},
}
SCHEMA_V4 = {
    "lt1_chains": {"first_action": "u64", "last_action": "u64", "drain_action": "u64", "in0": "u64",
                   "front": "u64", "in1": "u64", "last": "u64", "out1": "u64", "link_delay": "u64",
                   "tile": "u32", "lane": "u32", "used_rows": "u32", "used_cols": "u32", "calls": "u32"},
    "lt1_calls": {"action": "u64", "chain": "u32", "pos": "u32"},
}
SCHEMA_V5 = {
    "l2_staging": {"open": "u64", "close": "u64", "residency": "u64", "feeds": "u64", "tile": "u32",
                   "cf": "u32", "slot": "u32"},
}
STEPS_V5 = {"staging": "u64", "l2_cf": "u32", "l2_slot": "u32"}

# Version 6: the shared record (docs/plans/shared-tile-flow-record.md). The enums below are part of
# the format: a producer writes them into the manifest verbatim, and both sides index them.
LEVELS = ("L-B", "L-T1", "L-T2", "L-CA", "RTL")
LAYERS = ["dram", "l3", "l2", "acc", "l1"]
EVENT_KINDS = ["DMA", "BLOCK_MOVE", "STREAM", "COMPUTE", "DRAIN", "CREDIT", "TOKEN", "BM_ROW", "STR_ROW",
               "DRAIN_ROW"]
EV_DMA, EV_BM, EV_STREAM, EV_COMPUTE, EV_DRAIN, EV_CREDIT, EV_TOKEN, EV_BM_ROW, EV_STR_ROW, EV_DRAIN_ROW = range(10)
DRAM, L3, L2, ACC, L1 = range(5)
# A transaction's layers, and the CSP action kind each pair is (enum Action::Kind).
TRANSACTION = {EV_DMA: {(DRAM, L3): 0, (L3, DRAM): 1},           # LOAD, STORE
               EV_BM: {(L3, L2): 2, (L2, L3): 3},                 # MOVE, WRITEBACK
               EV_STREAM: {(L2, L1): 4},                          # FEED
               EV_COMPUTE: {(L1, ACC): 6},                        # CALL
               EV_DRAIN: {(ACC, L2): 5,                           # DRAIN
                          (ACC, L3): 3}}                          # 6.1: the RTL's store, acc -> l3: WRITEBACK
PROGRESS = {EV_BM_ROW: EV_BM, EV_STR_ROW: EV_STREAM, EV_DRAIN_ROW: EV_DRAIN}
DATA_EDGES = {(DRAM, L3), (L3, L2), (L2, L1), (L1, ACC), (ACC, L2), (L2, L3), (L3, DRAM), (ACC, L3)}
# Version 6's minor revisions: 1 adds the acc -> l3 DRAIN and the word-addressed L3 (#398).
MINORS = (0, 1)
DRAIN_KIND, WRITEBACK_KIND = 5, 3
ELEMENTS = {"int8": 1, "int32": 4, "fp16": 2, "bf16": 2, "fp32": 4, "fp8_e4m3": 1, "fp8_e5m2": 1}
INTERLEAVES = ("row-mod-banks",)
DOMAINS = ("dense", "lower", "upper", "band")
ROLE_ACTUAL, ROLE_EXPECTED = 0, 1
SCHEMA_V6 = {
    "events": {"id": "u64", "parent": "u64", "action": "u64", "where": "u64", "src_addr": "u64",
               "dst_addr": "u64", "tile": "u32", "lane": "u32", "row": "u32", "beat": "u32", "kind": "u8",
               "process": "u8", "src": "u8", "dst": "u8"},
}
EVENTS_TIMED = {"cycle": "u64", "end": "u64"}
MOVE_KIND, FEED_KIND, L2_RELEASE_KIND = 2, 4, 10
CALL_KIND = 6
CREDIT_KINDS = {7, 8, 9, 10}                           # RELEASE, INHERIT, RETAIN, L2_RELEASE
LOAD_KIND = 0
# The channel each kind leaves its tile in (enum Action::Kind, in order); None: no snapshot.
LANDS = [1, 0, 2, 1, 3, 2, 3, None, 1, None, None]   # LOAD l3, STORE dram, MOVE l2, WRITEBACK l3, FEED cf,
                                                      # DRAIN l2, CALL cf, RELEASE, INHERIT l3, RETAIN,
                                                      # L2_RELEASE


class Unreadable(Exception):
    pass


def _int(v, what, lo=0):
    if not isinstance(v, int) or isinstance(v, bool) or v < lo:
        raise Unreadable(f"{what} is {v!r}, not an integer >= {lo}")
    return v


def _isint(v):
    """An integer, and not a boolean (JSON's true is Python's 1)."""
    return isinstance(v, int) and not isinstance(v, bool)


def nat(v):
    """A non-negative integer: a band width, a padding count."""
    return _isint(v) and v >= 0


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
        # The shapes the value checks divide by and compare against.
        for o in m["operands"]:
            for k in ("rows", "cols", "tile_rows", "tile_cols"):
                _int(o[k], f"operand {o['name']}.{k}", 1)
    except (KeyError, TypeError) as e:
        raise Unreadable(f"manifest: missing or malformed field ({e!r})")


def _manifest_v6(m):
    """The version-6 manifest fields, typed. Their invariants (G1, T1, X1, X2) are checked by
    check_v6; here only what load() and the value decoding read."""
    try:
        _int(m["machine"]["banks"], "machine.banks", 1)
        if m["level"] not in LEVELS:
            raise Unreadable(f"level is {m['level']!r}, not one of {', '.join(LEVELS)}")
        if not isinstance(m["producer"], str) or not m["producer"]:
            raise Unreadable("producer is not a non-empty string")
        if m.get("minor", 0) not in MINORS or isinstance(m.get("minor", 0), bool):
            raise Unreadable(f"version 6.{m.get('minor')} (this checker reads 6.{', 6.'.join(map(str, MINORS))})")
        if m["layers"] != LAYERS:
            raise Unreadable(f"layers are {m['layers']!r}; version 6 names {LAYERS}")
        if m["event_kinds"] != EVENT_KINDS:
            raise Unreadable(f"event_kinds are {m['event_kinds']!r}; version 6 names {EVENT_KINDS}")
        t = m["time"]
        if t is not None and (not isinstance(t, dict) or t.get("unit") != "cycle"):
            raise Unreadable("time is neither null nor {unit: cycle, cycles}")
        if t is not None:
            _int(t.get("cycles"), "time.cycles")
        if not isinstance(m["geometry"], dict):
            raise Unreadable("geometry is not an object")
        r = m["result"]
        if not isinstance(r, dict) or r.get("values") not in ("match", "mismatch", "unchecked"):
            raise Unreadable("result: missing or malformed (cycles, values: match | mismatch | unchecked)")
        if r.get("cycles") is not None:
            _int(r["cycles"], "result.cycles")
        b = r.get("bound")
        if b is not None and not (isinstance(b, dict) and all(isinstance(b.get(k), (int, float)) and not isinstance(b.get(k), bool)
                                                              and b[k] >= 0 for k in ("atol", "rtol"))):
            raise Unreadable("result.bound is neither null nor {atol, rtol}")
        for o in m["operands"]:
            if not isinstance(o.get("element"), str):
                raise Unreadable(f"operand {o['name']}: no element type")
            if not isinstance(o.get("domain"), dict):
                raise Unreadable(f"operand {o['name']}: no domain")
    except (KeyError, TypeError) as e:
        raise Unreadable(f"manifest: missing or malformed field ({e!r})")


def element_of(m, operand):
    """An operand's element type: float32 before version 6, the operand's own from version 6."""
    return m["operands"][operand]["element"] if m["version"] >= 6 else "fp32"


def raw(m, T, k):
    """Snapshot k's elements as bytes, little-endian, row-major."""
    P = T["snapshots"]
    n = P["height"][k] * P["width"][k]
    if m["version"] < 6:
        return T["elements"]["v"][P["first"][k]:P["first"][k] + n].tobytes()
    size = ELEMENTS.get(element_of(m, T["tiles"]["operand"][P["tile"][k]]), 1)
    return T["elements"]["b"][P["first"][k]:P["first"][k] + n * size].tobytes()


def _fp8(b, ebits, mbits, bias, ieee):
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> mbits) & ((1 << ebits) - 1)
    f = b & ((1 << mbits) - 1)
    if ieee and e == (1 << ebits) - 1:                  # e5m2: IEEE-style inf and NaN
        return s * math.inf if f == 0 else math.nan
    if not ieee and e == (1 << ebits) - 1 and f == (1 << mbits) - 1:   # e4m3fn: one NaN, no inf
        return math.nan
    if e == 0:
        return s * f * 2.0 ** (1 - bias - mbits)
    return s * (1 + f / (1 << mbits)) * 2.0 ** (e - bias)


def decode(element, data):
    """Bytes of one element type as Python numbers (ints for the integer types)."""
    if element == "int8":
        return list(struct.unpack(f"<{len(data)}b", data))
    if element == "int32":
        return list(struct.unpack(f"<{len(data) // 4}i", data))
    if element == "fp32":
        return list(struct.unpack(f"<{len(data) // 4}f", data))
    if element == "fp16":
        return list(struct.unpack(f"<{len(data) // 2}e", data))
    if element == "bf16":
        return [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in struct.unpack(f"<{len(data) // 2}H", data)]
    if element == "fp8_e4m3":
        return [_fp8(b, 4, 3, 7, False) for b in data]
    if element == "fp8_e5m2":
        return [_fp8(b, 5, 2, 15, True) for b in data]
    raise ValueError(f"unknown element type {element}")


def in_domain(domain, r, c):
    """Whether operand element (r, c) is inside the operand's domain (global coordinates)."""
    pad = domain.get("pad")
    if pad is not None and (r >= pad["rows"] or c >= pad["cols"]):
        return False
    k = domain["kind"]
    if k == "lower":
        return c - r <= domain.get("diagonal", 0)
    if k == "upper":
        return r - c <= domain.get("diagonal", 0)
    if k == "band":
        return r - c <= domain["lower"] and c - r <= domain["upper"]
    return True


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
    if m["version"] >= 6:
        _manifest_v6(m)
    if m["version"] >= 2:
        v = m.get("values")
        fmt = "per-operand" if m["version"] >= 6 else "float32"
        if not isinstance(v, dict) or v.get("format") != fmt or not isinstance(v.get("selected"), str) \
                or not (v.get("finals") is True or isinstance(v.get("finals"), str)) \
                or not (v.get("cut_at") is None or (type(v.get("cut_at")) is int and v["cut_at"] >= 0)):
            raise Unreadable(f"values: missing or malformed (format {fmt}, selected, finals, cut_at)")
    tables = {}
    if m["version"] >= 3:
        o = m.get("lt1")
        if o is not None and (not isinstance(o, dict) or not isinstance(o.get("lanes"), dict)
                              or not all(isinstance(o.get(k), int) and o[k] >= 0 for k in ("makespan", "slot_bound", "slot_wait_cycles"))
                              or not isinstance(o.get("pass"), bool)):
            raise Unreadable("lt1: malformed (lanes, makespan, slot_bound, slot_wait_cycles, pass)")
    if m["version"] >= 5:
        try:
            _int(m["machine"]["l2"], "machine.l2")
        except (KeyError, TypeError) as e:
            raise Unreadable(f"manifest: missing or malformed field ({e!r})")
    if m["version"] >= 4 and m.get("lt1") is not None:
        o = m["lt1"]
        f = o.get("fabric")
        if not isinstance(o.get("one_leg"), dict) or not all(isinstance(v, int) and v >= 0 for v in o["one_leg"].values()):
            raise Unreadable("lt1.one_leg: malformed (an object of counts)")
        if f is not None and (not isinstance(f, dict)
                              or not all(isinstance(f.get(k), int) and f[k] >= 0 for k in ("pe_rows", "pe_cols", "chains", "link_delay_cycles"))
                              or f["pe_rows"] < 1 or f["pe_cols"] < 1):
            raise Unreadable("lt1.fabric: malformed (pe_rows, pe_cols, chains, link_delay_cycles)")
    schemas = dict(SCHEMA, **(SCHEMA_V2 if m["version"] >= 2 else {}), **(SCHEMA_V3 if m["version"] >= 3 else {}),
                   **(SCHEMA_V4 if m["version"] >= 4 else {}), **(SCHEMA_V5 if m["version"] >= 5 else {}))
    if m["version"] >= 5:
        schemas["steps"] = dict(schemas["steps"], **STEPS_V5)
    if m["version"] >= 6:
        schemas.update(SCHEMA_V6)
        schemas["snapshots"] = dict(schemas["snapshots"], role="u8")
        schemas["elements"] = {"b": "u8"}       # the payload: bytes, decoded by the operand's element type
        if m["time"] is not None:
            schemas["events"] = dict(schemas["events"], **EVENTS_TIMED)
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

    # D8: no time at L-B. A timed level's events (version 6) carry its time; nothing else does.
    timed = m["version"] >= 6 and m["level"] != "L-B"
    for name, t in T.items():
        if name.startswith("lt1_") or (timed and name == "events"):
            continue                                    # the L-T1 overlay, or a timed level's events
        for col, dtype in t["dtypes"].items():
            if dtype == "f64" or col.startswith("t_") or col in ("start", "finish", "cycle", "end"):
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
    if m["version"] >= 3:
        check_lt1(m, T, bad)
    if m["version"] >= 4:
        check_fabric(m, T, bad)
    if m["version"] >= 5:
        check_staging(m, T, bad)
    if m["version"] >= 6:
        check_v6(m, T, bad)
    return out, count


def check_staging(m, T, bad):
    G, S = T["l2_staging"], T["steps"]
    ng, n = G["rows"], S["rows"]
    actions, l2 = m["totals"]["actions"], m["machine"]["l2"]
    refused = m.get("refusal") is not None
    complete = not m["cut"] and n == actions

    # S4 (and the references S1 needs): every staging closes after it opens, in a slot of the L2.
    for g in range(ng):
        o, c = G["open"][g], G["close"][g]
        if G["tile"][g] >= T["tiles"]["rows"]:
            bad("S2", f"staging {g} names tile {G['tile'][g]}; there are {T['tiles']['rows']}")
        if G["slot"][g] >= l2:
            bad("S1", f"staging {g} is in L2 slot {G['slot'][g]}, outside the program's l2 = {l2}")
        if o >= actions:
            bad("S4", f"staging {g} opens at {o}, past the run's {actions} actions")
        if c == NONE64:
            if not refused:
                bad("S4", f"staging {g} never ends (no L2_RELEASE)")
        elif c <= o:
            bad("S4", f"staging {g} ends at {c}, not after its open at {o}")
        if g and o <= G["open"][g - 1]:
            bad("S2", f"staging {g} opens at {o}, not after staging {g - 1}")

    # S1: one open staging per slot, step by step.
    opens_at, ends_at = {}, {}
    for g in range(ng):
        opens_at.setdefault(G["open"][g], []).append(g)
        if G["close"][g] != NONE64:
            ends_at.setdefault(G["close"][g], []).append(g)
    live = {}
    for i in range(actions):
        for g in opens_at.get(i, []):
            key = (G["cf"][g], G["slot"][g])
            if key in live:
                bad("S1", f"step {i}: staging {g} opens in l2[{key[0]}] slot {key[1]}, which staging {live[key]} holds")
            live[key] = g
        for g in ends_at.get(i, []):
            live.pop((G["cf"][g], G["slot"][g]), None)

    # S2: each step's L2 address is its staging's; S3: the feeds are counted.
    feeds = [0] * ng
    for i in range(n):
        f, g, k = S["flags"][i], S["staging"][i], S["kind"][i]
        has = bool(f & HAS_L2)
        if has != (g != NONE64):
            bad("S2", f"step {i}: has_l2 is {has} but it names {'no' if g == NONE64 else 'a'} staging")
            continue
        if not has:
            if f & STAGES:
                bad("S2", f"step {i} is flagged as staging but names no staging")
            continue
        if g >= ng:
            bad("S2", f"step {i} names staging {g}; there are {ng}")
            continue
        if k not in (MOVE_KIND, FEED_KIND, L2_RELEASE_KIND):
            bad("S2", f"step {i}: a {m['kinds'][k] if k < len(m['kinds']) else k} carries an L2 address")
        if (S["l2_cf"][i], S["l2_slot"][i]) != (G["cf"][g], G["slot"][g]):
            bad("S2", f"step {i} is at l2[{S['l2_cf'][i]}] slot {S['l2_slot'][i]}; its staging {g} is at "
                      f"l2[{G['cf'][g]}] slot {G['slot'][g]}")
        if S["tile"][i] != G["tile"][g]:
            bad("S2", f"step {i} names tile {S['tile'][i]}; its staging {g} holds tile {G['tile'][g]}")
        o, c = G["open"][g], G["close"][g]
        if bool(f & STAGES) != (o == i):
            bad("S2", f"step {i}: flag stages is {bool(f & STAGES)}, and staging {g} opens at {o}")
        if f & STAGES and k != MOVE_KIND:
            bad("S2", f"step {i} stages a tile but is not a Move")
        if k == L2_RELEASE_KIND and c != i:
            bad("S2", f"step {i} ends staging {g}, which ends at {c}")
        if k == FEED_KIND:
            feeds[g] += 1
            if not (o < i and (c == NONE64 or i < c)):
                bad("S2", f"step {i} feeds staging {g} outside its life ({o}, {c})")
    if complete:
        for g in range(ng):
            if feeds[g] != G["feeds"][g]:
                bad("S3", f"staging {g} records {G['feeds'][g]} feeds; its steps give {feeds[g]}")


def check_fabric(m, T, bad):
    C, K, G, S = T["lt1_chains"], T["lt1_calls"], T["lt1_legs"], T["steps"]
    o = m.get("lt1")
    f = o.get("fabric") if o else None
    if f is None:
        if C["rows"] or K["rows"]:
            bad("P1", f"no fabric declared, but {C['rows']} chains and {K['rows']} stream calls")
        return
    actions, n, lanes = m["totals"]["actions"], S["rows"], o["lanes"].get("cf", 0)
    cf = m["processes"].index("cf") if "cf" in m["processes"] else -1
    legs = {}
    for g in range(G["rows"]):
        legs.setdefault(G["action"][g], []).append(g)
    calls_of = {}
    for r in range(K["rows"]):
        a, c, pos = K["action"][r], K["chain"][r], K["pos"][r]
        if a >= actions or c >= C["rows"]:
            bad("P1", f"stream call row {r} names action {a} and chain {c}; there are {actions} actions, {C['rows']} chains")
            continue
        if a < n and S["kind"][a] != CALL_KIND:
            bad("P1", f"stream call row {r} names action {a}, which is not a Call")
        calls_of.setdefault(c, []).append((a, pos))
    by_lane = {}
    delays = 0
    for c in range(C["rows"]):
        rows, cols, lane = C["used_rows"][c], C["used_cols"][c], C["lane"][c]
        in0, front, in1, last, out1, delay = (C[k][c] for k in ("in0", "front", "in1", "last", "out1", "link_delay"))
        first, lastA, drain, ncalls = C["first_action"][c], C["last_action"][c], C["drain_action"][c], C["calls"][c]
        if C["tile"][c] >= T["tiles"]["rows"] or lane >= lanes or first >= actions or lastA >= actions \
                or (drain != NONE64 and drain >= actions):
            bad("P1", f"chain {c} names tile {C['tile'][c]}, cf lane {lane} or actions {first}/{lastA}/{drain} that do not exist")
            continue
        if not (1 <= rows <= f["pe_rows"] and 1 <= cols <= f["pe_cols"]):
            bad("P1", f"chain {c} occupies {rows} x {cols} PEs; the array is {f['pe_rows']} x {f['pe_cols']}")
        skew = (rows - 1) + (cols - 1)
        if front - in0 != skew:
            bad("P2", f"chain {c}: front - in0 is {front - in0}; the skew is {skew}")
        if in1 < in0:
            bad("P2", f"chain {c}: in1 {in1} is before in0 {in0}")
        over = out1 != NONE64                  # its calls are over: last and out1 are known
        if (last == NONE64) != (out1 == NONE64):
            bad("P2", f"chain {c} records last {last} and out1 {out1}: both are known or neither is")
        elif over:
            if last - in1 != skew:
                bad("P2", f"chain {c}: last - in1 is {last - in1}; the skew is {skew}")
            if out1 - last != rows + delay:
                bad("P2", f"chain {c}: out1 - last is {out1 - last}; r + link_delay is {rows + delay}")
            delays += delay
        elif drain != NONE64:
            bad("P2", f"chain {c} was drained at action {drain}, yet its calls are not over")
        got = sorted(calls_of.get(c, []))
        if [p for _, p in got] != list(range(1, len(got) + 1)) or len(got) != ncalls:
            bad("P3", f"chain {c}'s calls are numbered {[p for _, p in got]}; it records {ncalls}")
        elif got:
            if got[0][0] != first or got[-1][0] != lastA:
                bad("P3", f"chain {c}: its calls run {got[0][0]}..{got[-1][0]}; it records {first}..{lastA}")
            prev_end = None
            for a, p in got:
                cl = [g for g in legs.get(a, []) if G["proc"][g] == cf]
                if len(cl) != 1:
                    bad("P3", f"chain {c} call {p} (action {a}) has {len(cl)} cf legs")
                    continue
                g = cl[0]
                if G["lane"][g] != lane:
                    bad("P3", f"chain {c} call {p} runs on cf lane {G['lane'][g]}; the chain owns lane {lane}")
                if p == 1 and G["start"][g] != in0:
                    bad("P3", f"chain {c}: its first call starts at {G['start'][g]}; in0 is {in0}")
                if p == ncalls and G["finish"][g] != in1:
                    bad("P3", f"chain {c}: its last call ends at {G['finish'][g]}; in1 is {in1}")
                if prev_end is not None and G["start"][g] < prev_end:
                    bad("P3", f"chain {c} call {p} starts at {G['start'][g]}, before call {p - 1} ends at {prev_end}")
                prev_end = G["finish"][g]
        if over and drain != NONE64:
            dl = legs.get(drain, [])
            if dl and min(G["start"][g] for g in dl) < out1:
                bad("P4", f"chain {c}: its Drain (action {drain}) starts before its last result is out at {out1}")
        by_lane.setdefault(lane, []).append((in0, in1, c))
    for lane, iv in by_lane.items():
        iv.sort()
        for (a0, a1, c0), (b0, b1, c1) in zip(iv, iv[1:]):
            if b0 < a1:
                bad("P4", f"cf lane {lane}: chains {c0} [{a0}, {a1}) and {c1} [{b0}, {b1}) stream operands at once")
        prev = None
        for _, _, c in iv:
            if C["out1"][c] == NONE64:
                continue
            out, rows = C["out1"][c], C["used_rows"][c]
            natural = C["last"][c] + rows
            want = max(natural, prev + rows) if prev is not None else natural
            if out != want:
                bad("P5", f"chain {c}: out1 is {out}; its results leave the top link at {want}")
            prev = out
    if C["rows"] != f["chains"] or delays != f["link_delay_cycles"]:
        bad("P5", f"the tables hold {C['rows']} chains and {delays} link-delay cycles; the manifest says "
                  f"{f['chains']} and {f['link_delay_cycles']}")


def check_lt1(m, T, bad):
    G, Q, S, TL = T["lt1_legs"], T["lt1_tiles"], T["steps"], T["tiles"]
    o, n, actions = m["lt1"], S["rows"], m["totals"]["actions"]
    if o is None:
        if G["rows"] or Q["rows"]:
            bad("L1", f"no L-T1 overlay, but {G['rows']} legs and {Q['rows']} proximity rows")
        return
    procs, kinds = m["processes"], m["kinds"]
    legs_of = {}
    busy = {}
    latest = 0
    waits = waited = 0
    for g in range(G["rows"]):
        a, k, p, lane = G["action"][g], G["kind"][g], G["proc"][g], G["lane"][g]
        st, fi, w = G["start"][g], G["finish"][g], G["slot_wait"][g]
        if a >= actions:
            bad("L1", f"leg {g} names action {a}; the run has {actions}")
            continue
        if k >= len(kinds):
            bad("L1", f"leg {g} has kind {k}; there are {len(kinds)}")
            continue
        if (p == NONE8) != (k in CREDIT_KINDS):
            bad("L1", f"leg {g} ({kinds[k]}) {'has no' if p == NONE8 else 'has a'} process")
        elif p != NONE8:
            if p >= len(procs):
                bad("L1", f"leg {g} names process {p}; there are {len(procs)}")
            elif lane >= o["lanes"].get(procs[p], 0):
                bad("L1", f"leg {g} is on {procs[p]} lane {lane}; it has {o['lanes'].get(procs[p], 0)}")
            elif fi > st:
                busy.setdefault((p, lane), []).append((st, fi, g))
        if a < n and S["kind"][a] != k:
            bad("L2", f"leg {g} is a {kinds[k]}; action {a} is a {kinds[S['kind'][a]] if S['kind'][a] < len(kinds) else S['kind'][a]}")
        legs_of.setdefault(a, []).append(G["leg"][g])
        if st > fi:
            bad("L3", f"leg {g} starts at {st}, after it finishes at {fi}")
        latest = max(latest, fi)
        if w:
            waits += w
            waited += 1
            if w > st:
                bad("L4", f"leg {g} waits {w} cycles on its slot but starts at cycle {st}")
            if k != LOAD_KIND:
                bad("L4", f"leg {g} ({kinds[k]}) waits {w} cycles on a slot; only a Load waits on its slot")
    for a in range(n):
        if a not in legs_of and m.get("refusal") is None:
            bad("L2", f"action {a} has no L-T1 leg")
    for a, ls in legs_of.items():
        if ls != list(range(len(ls))):
            bad("L2", f"action {a}'s legs are numbered {ls}")
    if latest != o["makespan"]:
        bad("L3", f"the latest leg finishes at {latest}; the makespan is {o['makespan']}")
    for (p, lane), iv in busy.items():
        iv.sort()
        for (s0, f0, g0), (s1, f1, g1) in zip(iv, iv[1:]):
            if s1 < f0:
                bad("L3", f"{procs[p]} lane {lane}: legs {g0} [{s0}, {f0}) and {g1} [{s1}, {f1}) overlap")
    if waits != o["slot_wait_cycles"] or waited != o["slot_bound"]:
        bad("L4", f"the legs wait {waits} cycles in {waited} Loads; the manifest says {o['slot_wait_cycles']} in {o['slot_bound']}")
    ops = m["operands"]
    expected = sum(-(-x["rows"] // x["tile_rows"]) * -(-x["cols"] // x["tile_cols"]) for x in ops)
    if Q["rows"] != expected:
        bad("L5", f"{Q['rows']} proximity rows; the operands have {expected} tiles")
    seen = set()
    all_pass = True
    for q in range(Q["rows"]):
        t = Q["tile"][q]
        if t >= TL["rows"]:
            bad("L1", f"proximity row {q} names tile {t}; there are {TL['rows']}")
            continue
        if t in seen:
            bad("L5", f"tile {t} has two proximity rows")
        seen.add(t)
        for c in ("max_abs", "max_rel"):
            v = Q[c][q]
            if not (v == v and v >= 0.0 and v != float("inf")):
                bad("L5", f"proximity row {q}'s {c} is {v}")
        all_pass = all_pass and Q["pass"][q] == 1
    if all_pass != o["pass"]:
        bad("L5", f"the manifest says pass is {o['pass']}; the tiles say {all_pass}")


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
        # Version 6: first is a byte offset into the payload, an element is its operand's width;
        # an expected value (role 1) is checked against its actual one by X3, not here.
        size = 1
        if m["version"] >= 6:
            if P["role"][k] == ROLE_EXPECTED:
                ok.append(None)
                continue
            if t < TL["rows"] and TL["operand"][t] < len(ops):
                size = ELEMENTS.get(element_of(m, TL["operand"][t]), 1)
        good = True
        if t >= TL["rows"]:
            bad("V1", f"snapshot {k} names tile {t}; there are {TL['rows']}")
            good = False
        if chan >= len(m["channels"]):
            bad("V1", f"snapshot {k} is in channel {chan}; there are {len(m['channels'])}")
            good = False
        if first + rows * cols * size > ne:
            bad("V1", f"snapshot {k}'s elements [{first}, {first + rows * cols * size}) run past the {ne} recorded")
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
            if raw(m, T, k) != raw(m, T, f):
                bad("V5", f"tile {t}'s last Store (snapshot {k}) differs from its final value (snapshot {f})")


def _addressed(g):
    """Whether the record declares its L3 as flat word-addressed memory (6.1), not slots."""
    x = g.get("l3")
    return isinstance(x, dict) and ("words" in x or "word_bytes" in x)


def _extent(g, layer):
    """How many addresses a modelled layer has: L3 slots (or words, addressed), L2 or accumulator
    rows, L1 edge lanes; None when any address is valid (DRAM)."""
    if layer == L3:
        return g["l3"]["words"] if _addressed(g) else g["l3"]["slots"]
    if layer in (L2, ACC):
        return g[LAYERS[layer]]["rows"]
    if layer == L1:
        return max(g["l1"]["rows"], g["l1"]["cols"])
    return None


def check_v6(m, T, bad):
    S, TL, W, L, P = T["steps"], T["tiles"], T["where"], T["loops"], T["snapshots"]
    n, actions, ops = S["rows"], m["totals"]["actions"], m["operands"]
    timed = m["level"] != "L-B"
    g = m["geometry"]
    minor = m.get("minor", 0)
    addressed = _addressed(g)

    # G1: every layer declared; the modelled ones well formed.
    modelled = [False] * len(LAYERS)
    geometry_ok = True
    for i, name in enumerate(LAYERS):
        x = g.get(name)
        if not isinstance(x, dict) or not isinstance(x.get("modelled"), bool):
            bad("G1", f"geometry.{name} is not declared (an object with modelled: true or false)")
            geometry_ok = False
            continue
        if not x["modelled"]:
            if not isinstance(x.get("why"), str) or not x["why"]:
                bad("G1", f"geometry.{name} is not modelled but does not say why")
            continue
        modelled[i] = True
        if name == "l3" and addressed:
            # 6.1: the RTL's L3, flat memory placed by the host, in words. Not also slots.
            if minor < 1:
                bad("G1", "geometry.l3 is word-addressed; that needs version 6.1 (minor 1)")
            if "slots" in x or "banks" in x:
                bad("G1", "geometry.l3 mixes the two L3 forms: words and word_bytes, and slots or banks")
                geometry_ok = False
        need = {"l3": ("words", "word_bytes") if addressed else ("slots", "banks"), "l2": ("banks", "rows", "row_bytes"),
                "acc": ("banks", "rows", "row_bytes"), "l1": ("rows", "cols")}.get(name, ())
        for k in need:
            v = x.get(k)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                bad("G1", f"geometry.{name}.{k} is {v!r}, not an integer >= 1")
                geometry_ok = False
        if name in ("l2", "acc") and x.get("interleave") not in INTERLEAVES:
            bad("G1", f"geometry.{name}.interleave is {x.get('interleave')!r}, not one of {', '.join(INTERLEAVES)}")
        if name == "l3" and geometry_ok and not addressed and (x["slots"], x["banks"]) != (m["machine"]["l3"], m["machine"]["banks"]):
            bad("G1", f"geometry.l3 is {x['slots']} slots in {x['banks']} banks; the machine is "
                      f"{m['machine']['l3']} in {m['machine']['banks']}")

    # T1: time only where the level has it.
    E = T["events"]
    ne = E["rows"]
    cycles = m["time"]["cycles"] if m["time"] is not None else None
    if not timed:
        if m["time"] is not None:
            bad("T1", "an L-B record declares time; L-B has none")
        if m["result"].get("cycles") is not None:
            bad("T1", "an L-B record states cycles to result; L-B has no time")
        for c in EVENTS_TIMED:
            if c in E["dtypes"]:
                bad("T1", f"an L-B record's events have a {c} column")
    else:
        if m["time"] is None:
            bad("T1", f"a {m['level']} record declares no time")
        else:
            latest = 0
            for e in range(ne):
                if E["cycle"][e] > E["end"][e]:
                    bad("T1", f"event {e} starts at cycle {E['cycle'][e]}, after it ends at {E['end'][e]}")
                latest = max(latest, E["end"][e])
            if ne and latest != cycles:
                bad("T1", f"the latest event ends at cycle {latest}; the time's cycles to result is {cycles}")
            if m["result"].get("cycles") != cycles:
                bad("T1", f"result.cycles is {m['result'].get('cycles')}; the time's cycles to result is {cycles}")

    # E1: references, layers and addresses.
    def loops_of(w):
        if w == NONE64 or w >= W["rows"] or W["loop_first"][w] + W["loop_count"][w] > L["rows"]:
            return ()
        f = W["loop_first"][w]
        return tuple((L["var"][f + d], L["value"][f + d]) for d in range(W["loop_count"][w]))

    good = [False] * ne
    for e in range(ne):
        k, pr, src, dst, t = E["kind"][e], E["process"][e], E["src"][e], E["dst"][e], E["tile"][e]
        ok = True
        if E["id"][e] != e:
            bad("E1", f"event row {e} has id {E['id'][e]}")
            ok = False
        if k >= len(EVENT_KINDS):
            bad("E1", f"event {e} has kind {k}; there are {len(EVENT_KINDS)}")
            continue
        if pr != NONE8 and pr >= len(m["processes"]):
            bad("E1", f"event {e} names process {pr}; there are {len(m['processes'])}")
            ok = False
        if t != NONE32 and t >= TL["rows"]:
            bad("E1", f"event {e} names tile {t}; there are {TL['rows']}")
            ok = False
        if E["action"][e] != NONE64 and E["action"][e] >= actions:
            bad("E1", f"event {e} names action {E['action'][e]}; the run has {actions}")
            ok = False
        if E["where"][e] != NONE64 and E["where"][e] >= W["rows"]:
            bad("E1", f"event {e} names position {E['where'][e]}; there are {W['rows']}")
            ok = False
        par = E["parent"][e]
        if par != NONE64 and (par >= ne or par == e):
            bad("E1", f"event {e} names parent {par}; there are {ne} events")
            ok = False
        for side, layer, addr in (("source", src, E["src_addr"][e]), ("destination", dst, E["dst_addr"][e])):
            if layer == NONE8:
                continue
            if layer >= len(LAYERS):
                bad("E1", f"event {e}'s {side} is layer {layer}; there are {len(LAYERS)}")
                ok = False
            elif not modelled[layer]:
                bad("E1", f"event {e} ({EVENT_KINDS[k]}) moves through {LAYERS[layer]}, which the record declares not modelled")
                ok = False
            elif addr != NONE64 and geometry_ok and _extent(g, layer) is not None and addr >= _extent(g, layer):
                bad("E1", f"event {e}'s {side} address {addr} is outside {LAYERS[layer]} ({_extent(g, layer)})")
                ok = False
        if k == EV_DRAIN and (src, dst) == (ACC, L3) and minor < 1:
            bad("E1", f"event {e} is a DRAIN from acc to l3; that needs version 6.1 (minor 1)")
            ok = False
        if k in TRANSACTION and ok and (src, dst) not in TRANSACTION[k]:
            names = lambda x: LAYERS[x] if x < len(LAYERS) else "none"
            bad("E1", f"event {e} is a {EVENT_KINDS[k]} from {names(src)} to {names(dst)}; a {EVENT_KINDS[k]} moves "
                      + " or ".join(f"{LAYERS[a]} -> {LAYERS[b]}" for a, b in TRANSACTION[k]))
            ok = False
        good[e] = ok

    # E2: identities.
    keys, by_action = {}, {}
    for e in range(ne):
        if not good[e]:
            continue
        key = (E["kind"][e], E["process"][e], E["tile"][e], E["src"][e], E["src_addr"][e], E["dst"][e],
               E["dst_addr"][e], loops_of(E["where"][e]), E["row"][e], E["beat"][e])
        if key in keys:
            bad("E2", f"events {keys[key]} and {e} ({EVENT_KINDS[E['kind'][e]]}) share an identity")
        keys.setdefault(key, e)
        if E["action"][e] != NONE64:
            ak = (E["kind"][e], E["action"][e], E["row"][e], E["beat"][e])
            if ak in by_action:
                bad("E2", f"events {by_action[ak]} and {e} are both the {EVENT_KINDS[E['kind'][e]]} of action "
                          f"{E['action'][e]} (row {E['row'][e]}, beat {E['beat'][e]})")
            by_action.setdefault(ak, e)

    # E3: progress rows.
    def tile_rows(t):
        o = ops[TL["operand"][t]] if TL["operand"][t] < len(ops) else None
        return None if o is None else min(o["tile_rows"], o["rows"] - TL["ti"][t] * o["tile_rows"])

    for e in range(ne):
        if not good[e]:
            continue
        k = E["kind"][e]
        if k not in PROGRESS:
            if E["row"][e] != NONE32 or E["beat"][e] != NONE32:
                bad("E3", f"event {e} ({EVENT_KINDS[k]}) has a row or beat; only row events do")
            if k in TRANSACTION and E["parent"][e] != NONE64:
                bad("E3", f"event {e} ({EVENT_KINDS[k]}) is a transaction with a parent")
            continue
        par = E["parent"][e]
        if par == NONE64 or not good[par] or E["kind"][par] != PROGRESS[k]:
            bad("E3", f"event {e} ({EVENT_KINDS[k]}) is not a child of a {EVENT_KINDS[PROGRESS[k]]}")
            continue
        if (E["tile"][e], E["action"][e]) != (E["tile"][par], E["action"][par]):
            bad("E3", f"event {e} ({EVENT_KINDS[k]}) is of tile {E['tile'][e]}, action {E['action'][e]}; its parent "
                      f"{par} of tile {E['tile'][par]}, action {E['action'][par]}")
        h = tile_rows(E["tile"][e]) if E["tile"][e] != NONE32 else None
        if E["row"][e] == NONE32 or (h is not None and E["row"][e] >= h):
            bad("E3", f"event {e} ({EVENT_KINDS[k]}) is row {E['row'][e]} of a tile of {h} rows")
        if timed and "cycle" in E and not (E["cycle"][par] <= E["cycle"][e] and E["end"][e] <= E["end"][par]):
            bad("E3", f"event {e} ({EVENT_KINDS[k]}) runs [{E['cycle'][e]}, {E['end'][e]}], outside its parent's "
                      f"[{E['cycle'][par]}, {E['end'][par]}]")

    # E4: credits upstream, tokens downstream.
    # Timed: a token or credit is sent no earlier than the transaction it follows ends. Untimed: it
    # comes after it in event order.
    clocked = timed and "cycle" in E
    when = (lambda e: E["end"][e]) if clocked else (lambda e: e)
    arrived, left = {}, {}                     # (tile, layer) -> the earliest transaction into / out of it
    for e in range(ne):
        if good[e] and E["kind"][e] in TRANSACTION and E["tile"][e] != NONE32:
            for d, key in ((arrived, (E["tile"][e], E["dst"][e])), (left, (E["tile"][e], E["src"][e]))):
                if key not in d or when(e) < when(d[key]):
                    d[key] = e
    for e in range(ne):
        k = E["kind"][e]
        if not good[e] or k not in (EV_CREDIT, EV_TOKEN):
            continue
        src, dst, t = E["src"][e], E["dst"][e], E["tile"][e]
        if k == EV_CREDIT and addressed and src == L3:
            # A credit frees its source; a word-addressed L3 has no slots to free (an L2 credit
            # sent up to L3's BlockMover is still an L2 credit).
            bad("E4", f"event {e} is a CREDIT freeing l3; a word-addressed L3 has no credits")
            continue
        edge = (src, dst) if k == EV_TOKEN else (dst, src)
        if edge not in DATA_EDGES:
            what = "downstream along" if k == EV_TOKEN else "upstream against"
            bad("E4", f"event {e} ({EVENT_KINDS[k]}) from {LAYERS[src] if src < len(LAYERS) else src} to "
                      f"{LAYERS[dst] if dst < len(LAYERS) else dst} does not go {what} a data edge")
            continue
        if t == NONE32:
            continue
        before = arrived if k == EV_TOKEN else left
        x = before.get((t, src))
        if x is None or (when(x) > E["cycle"][e] if clocked else x > e):
            what = "arrived at" if k == EV_TOKEN else "left"
            bad("E4", f"event {e} ({EVENT_KINDS[k]}) of tile {t} precedes any transaction that {what} {LAYERS[src]}")

    # E5: the join with the CSP actions.
    covered, transactions, joined = [0] * n, 0, 0
    l3_word = {}                                # residency -> (event, its word address) (addressed L3)
    for e in range(ne):
        k = E["kind"][e]
        if not good[e] or k not in TRANSACTION:
            continue
        transactions += 1
        a = E["action"][e]
        if a == NONE64:
            continue
        joined += 1
        if a >= n:
            continue                            # past a cut: nothing recorded to join to
        covered[a] += 1
        want = TRANSACTION[k][(E["src"][e], E["dst"][e])]
        if S["kind"][a] != want:
            bad("E5", f"event {e} is a {EVENT_KINDS[k]} {LAYERS[E['src'][e]]} -> {LAYERS[E['dst'][e]]}; action {a} is a "
                      f"{m['kinds'][S['kind'][a]] if S['kind'][a] < len(m['kinds']) else S['kind'][a]}")
            continue
        if E["tile"][e] != S["tile"][a]:
            bad("E5", f"event {e} is of tile {E['tile'][e]}; its action {a} is of tile {S['tile'][a]}")
        addr = E["dst_addr"][e] if E["dst"][e] == L3 else E["src_addr"][e] if E["src"][e] == L3 else NONE64
        if addressed:
            # A word-addressed L3 has no slots: the residency keeps one address, from open to close.
            r = S["residency"][a]
            if addr != NONE64 and r != NONE64:
                first = l3_word.setdefault(r, (e, addr))
                if first[1] != addr:
                    bad("E5", f"event {e} is at L3 word {addr}; event {first[0]} put its residency at word {first[1]}")
        elif S["flags"][a] & HAS_PLACE:
            if addr != NONE64 and addr != S["slot"][a]:
                bad("E5", f"event {e} is at L3 slot {addr}; its action {a} is at slot {S['slot'][a]}")
        if k == EV_DRAIN and E["dst"][e] == L3:
            # The RTL's store drains the accumulator into L3: it is the tile's DRAIN step too, the
            # last one before its WRITEBACK.
            d = next((b for b in range(a - 1, -1, -1) if S["kind"][b] == DRAIN_KIND and S["tile"][b] == S["tile"][a]), None)
            if d is None:
                bad("E5", f"event {e} drains acc -> l3 for action {a}; no DRAIN of its tile precedes it")
            else:
                covered[d] += 1
    if transactions and joined == transactions:
        data_kinds = {v for d in TRANSACTION.values() for v in d.values()}
        for a in range(n):
            if S["kind"][a] in data_kinds and covered[a] != 1:
                bad("E5", f"action {a} ({m['kinds'][S['kind'][a]]}) has {covered[a]} transaction events; it moves a tile once")

    # X1, X2: element types, shapes and domains.
    dom_ok = []
    for i, o in enumerate(ops):
        if o["element"] not in ELEMENTS:
            bad("X1", f"operand {o['name']} has element type {o['element']!r}; the format knows {', '.join(ELEMENTS)}")
        d = o["domain"]
        okd = d.get("kind") in DOMAINS
        if not okd:
            bad("X2", f"operand {o['name']}'s domain kind is {d.get('kind')!r}, not one of {', '.join(DOMAINS)}")
        # A diagonal may be negative (d = -1 is strictly lower); widths and padding may not.
        if okd and d["kind"] in ("lower", "upper") and not _isint(d.get("diagonal", 0)):
            bad("X2", f"operand {o['name']}'s domain diagonal is {d.get('diagonal')!r}")
            okd = False
        if okd and d["kind"] == "band" and not (nat(d.get("lower")) and nat(d.get("upper"))):
            bad("X2", f"operand {o['name']}'s band needs lower and upper widths")
            okd = False
        pad = d.get("pad")
        if okd and pad is not None and not (isinstance(pad, dict) and nat(pad.get("rows")) and nat(pad.get("cols"))
                                            and pad["rows"] <= o["rows"] and pad["cols"] <= o["cols"]):
            bad("X2", f"operand {o['name']}'s padding window {pad!r} is not inside its {o['rows']} x {o['cols']}")
            okd = False
        dom_ok.append(okd)
    nb = T["elements"]["rows"]
    usable = []
    for k in range(P["rows"]):
        t = P["tile"][k]
        if P["role"][k] not in (ROLE_ACTUAL, ROLE_EXPECTED):
            bad("X1", f"snapshot {k} has role {P['role'][k]}; a snapshot is actual (0) or expected (1)")
            usable.append(False)
            continue
        if t >= TL["rows"] or TL["operand"][t] >= len(ops) or ops[TL["operand"][t]]["element"] not in ELEMENTS:
            usable.append(False)
            continue
        o = ops[TL["operand"][t]]
        size = ELEMENTS[o["element"]]
        h, w = P["height"][k], P["width"][k]
        if P["first"][k] + h * w * size > nb:
            bad("X1", f"snapshot {k}'s {h} x {w} {o['element']} elements run past the {nb} payload bytes")
            usable.append(False)
            continue
        usable.append(True)
        if not dom_ok[TL["operand"][t]] or o["domain"]["kind"] == "dense" and o["domain"].get("pad") is None:
            continue
        data, r0, c0 = raw(m, T, k), TL["ti"][t] * o["tile_rows"], TL["tj"][t] * o["tile_cols"]
        for r in range(h):
            for c in range(w):
                if not in_domain(o["domain"], r0 + r, c0 + c):
                    x = (r * w + c) * size
                    if any(data[x:x + size]):
                        bad("X2", f"snapshot {k} ({o['name']}[{TL['ti'][t]},{TL['tj'][t]}]) holds a nonzero element at "
                                  f"({r0 + r}, {c0 + c}), outside operand {o['name']}'s {o['domain']['kind']} domain")
                        break
                else:
                    continue
                break

    # X3: expected values beside actual ones, and the result they decide.
    actual, expected = {}, {}
    for k in range(P["rows"]):
        if not usable[k]:
            continue
        key = (P["tile"][k], P["step"][k])
        d = actual if P["role"][k] == ROLE_ACTUAL else expected
        if P["role"][k] == ROLE_EXPECTED and key in d:
            bad("X3", f"snapshots {d[key]} and {k} are both the expected value of tile {key[0]} at step {key[1]}")
        d.setdefault(key, k)
    for key, k in expected.items():
        a = actual.get(key)
        if a is None:
            bad("X3", f"expected snapshot {k} (tile {key[0]}, step {key[1]}) has no actual value beside it")
        elif (P["chan"][a], P["height"][a], P["width"][a]) != (P["chan"][k], P["height"][k], P["width"][k]):
            bad("X3", f"expected snapshot {k} is not in the channel and shape of its actual snapshot {a}")
    has_expected = bool(expected)
    if m["values"].get("expected", False) != has_expected:
        bad("X3", f"values.expected is {m['values'].get('expected')}, but the record holds {len(expected)} expected values")
    finals = [(key, k) for key, k in actual.items() if key[1] == NONE64]
    verdict = "unchecked"
    if has_expected:
        verdict = "match"
        bound = m["result"].get("bound")
        for key, a in finals:
            x = expected.get(key)
            if x is None:
                bad("X3", f"the final value of tile {key[0]} has no expected value")
                verdict = "mismatch"
                continue
            el = element_of(m, TL["operand"][key[0]])
            if el in ("int8", "int32") or bound is None:
                same = raw(m, T, a) == raw(m, T, x)
            else:
                same = all(p == q or abs(p - q) <= bound["atol"] + bound["rtol"] * abs(q) or (p != p and q != q)
                           for p, q in zip(decode(el, raw(m, T, a)), decode(el, raw(m, T, x))))
            if not same:
                verdict = "mismatch"
    if m["result"]["values"] != verdict:
        bad("X3", f"result.values is {m['result']['values']!r}; the finals and their expected values give {verdict!r}")


def result(m, T):
    """The transformation's result, which CI compares (D3): cycles to result and a digest of each
    operand's final values, in tile order. Independent of the event vocabulary."""
    P, TL = T.get("snapshots"), T["tiles"]
    finals = {}
    if P is not None:
        for k in range(P["rows"]):
            if P["step"][k] == NONE64 and (m["version"] < 6 or P["role"][k] == ROLE_ACTUAL):
                t = P["tile"][k]
                finals[(TL["operand"][t], TL["ti"][t], TL["tj"][t])] = k
    digests = {}
    for i, o in enumerate(m["operands"]):
        h = hashlib.sha256()
        tiles = sorted(key for key in finals if key[0] == i)
        for key in tiles:
            h.update(raw(m, T, finals[key]))
        digests[o["name"]] = {"element": element_of(m, i), "tiles": len(tiles), "sha256": h.hexdigest()}
    v6 = m["version"] >= 6
    return {"level": m["level"] if v6 else "L-B", "producer": m["producer"] if v6 else None,
            "program": m["program"]["name"], "cycles": m["result"]["cycles"] if v6 else None,
            "values": m["result"]["values"] if v6 else "unchecked", "finals": digests}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    ap.add_argument("--verbose", action="store_true", help="also list the checks that passed")
    ap.add_argument("--result", action="store_true",
                    help="print only the result (cycles to result, final-value digests) as JSON, after the checks pass")
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
    if args.result:
        # The result of a record that breaks an invariant is not a result: report and exit 1.
        if violations:
            for inv, msg in violations:
                print(f"dflow_check: {inv}: {msg}", file=sys.stderr)
            return 1
        print(json.dumps(result(m, tables), indent=1))
        return 0
    invs = ["D2", "D3", "D4", "D5", "D6", "D8", "D9", "REF", "V1", "V2", "V3", "V4", "V5", "L1", "L2", "L3", "L4", "L5",
            "P1", "P2", "P3", "P4", "P5", "S1", "S2", "S3", "S4", "G1", "T1", "E1", "E2", "E3", "E4", "E5", "X1", "X2", "X3"]
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
