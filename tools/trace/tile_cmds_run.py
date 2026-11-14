#!/usr/bin/env python3
"""
Run a `kpu-tile-cmds` stream (version 1; docs/plans/tile-command-stream.md, #397) the way the KPU
systolic tile's controller does, functionally, and check its results.

This is kpu-sim's reference reading of the RTL's command semantics (kpugen
src/kpugen/tiles/systolic/controller/sync.py, isa/matmul.py), not the RTL: the stream's semantics
written down where kpu-sim's tests can run them. The RTL's runtime executes the same file.

It executes the commands in order, one queue's worth of state:
- L3 is a byte array: `l3_init` fills it, STORE writes it, `results` are read back from it;
- the scratchpad holds rows of `dim` int8 bytes, the accumulator rows of `dim` int32;
- LOAD MEMCPY copies `len` rows of `row_bytes` from L3 (row r at dram_addr + r*stride);
  LOAD TRANSPOSE stores the tile corner-turned (B's column j becomes scratchpad row j);
- COMPUTE (output-stationary) adds rs1's A rows times rs2's corner-turned B into the accumulator
  slot at rd, starting from zero unless chain bit 0 says it continues a reduction;
- STORE writes the accumulator slot's rows to L3, int32 little-endian.

and it enforces the slot protocol on every command: ACQUIRE needs its dst slot FREE and takes it,
SIGNAL marks dst READY on completion, WAIT needs src READY, RELEASE frees src on completion. L2
pair tags and accumulator tags are separate pools. A command that breaks the protocol, an address
outside L3, or a result that differs from the stream's expected bytes is an error.

Usage: python3 tile_cmds_run.py <stream.json>     exit 0 ok, 1 wrong, 2 unreadable

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import struct
import sys

LOAD, STORE, EXEC = 0, 1, 2
MEMCPY, TRANSPOSE = 0, 4
CONFIG, COMPUTE = 0x00, 0x11
WAIT, RELEASE, ACQUIRE, SIGNAL = 1 << 8, 1 << 9, 1 << 10, 1 << 11
ACC_BIT = 1 << 31
FIELDS = ["type", "opcode", "id", "dram_addr", "local_addr", "len", "rs1", "rd", "k_dim", "stride", "rs2",
          "row_bytes", "activation", "sync", "chain"]


class Wrong(Exception):
    pass


def span(where, start, length, size):
    # An access [start, start + length) must lie inside L3: no negative start or length (Python's
    # negative slicing would read the wrong bytes), no end past it.
    if not isinstance(start, int) or not isinstance(length, int) or start < 0 or length < 0 or start + length > size:
        raise Wrong(f"{where} accesses L3 bytes [{start}, {start}+{length}) outside its {size}")
    return slice(start, start + length)


def run(doc):
    if doc.get("format") != "kpu-tile-cmds" or doc.get("version") != 1:
        raise ValueError(f"not a kpu-tile-cmds version-1 stream ({doc.get('format')!r}, {doc.get('version')!r})")
    if doc.get("fields") != FIELDS:
        raise ValueError(f"fields are {doc.get('fields')!r}; version 1 names {FIELDS}")
    dim = doc["tile"]["dim"]
    l3 = bytearray(doc["l3"]["bytes"])
    for x in doc["l3_init"]:
        if x["bytes"] is None:
            raise ValueError(f"{x['tile']}: the stream carries no input values")
        b = bytes.fromhex(x["bytes"])
        l3[span(f"l3_init {x['tile']}", x["addr"], len(b), len(l3))] = b
    spad = {}                                   # scratchpad row -> dim signed bytes
    acc = {}                                    # accumulator row -> dim int32
    state = {"l2": {}, "acc": {}}               # tag -> FREE / OWNED / READY
    dataflow = None

    def pool(c, which):
        # Which pool a command's src or dst tag names: the accumulator's for a COMPUTE's dst and a
        # STORE's src; the L2 pair's otherwise.
        if c["type"] == EXEC and which == "dst":
            return "acc"
        if c["type"] == STORE and which == "src":
            return "acc"
        return "l2"

    for n, row in enumerate(doc["commands"]):
        c = dict(zip(FIELDS, row[:len(FIELDS)]))
        tag = f"command {n} ({row[len(FIELDS)]})"
        if c["id"] != n % 256:
            raise Wrong(f"{tag} has id {c['id']}; ids count from 0, modulo 256")
        sy = c["sync"]
        src, dst = sy & 0xF, (sy >> 4) & 0xF
        if sy & WAIT and state[pool(c, "src")].get(src, "FREE") != "READY":
            raise Wrong(f"{tag} waits on {pool(c, 'src')} slot {src}, which is {state[pool(c, 'src')].get(src, 'FREE')}")
        if sy & ACQUIRE:
            if state[pool(c, "dst")].get(dst, "FREE") != "FREE":
                raise Wrong(f"{tag} acquires {pool(c, 'dst')} slot {dst}, which is {state[pool(c, 'dst')][dst]}")
            state[pool(c, "dst")][dst] = "OWNED"
        # The command's effect.
        if c["type"] == EXEC and c["opcode"] == CONFIG:
            dataflow = c["rs1"]
        elif c["type"] == LOAD:
            if c["local_addr"] & ACC_BIT:
                raise Wrong(f"{tag}: a LOAD into the accumulator is not version 1's")
            rows = []
            for r in range(c["len"]):
                a = c["dram_addr"] + r * c["stride"]
                rows.append(list(struct.unpack(f"<{c['row_bytes']}b", l3[span(tag, a, c["row_bytes"], len(l3))])))
            if c["opcode"] == TRANSPOSE:
                rows = [list(col) for col in zip(*rows)]
            elif c["opcode"] != MEMCPY:
                raise Wrong(f"{tag}: DMA opcode {c['opcode']} is not version 1's")
            for r, v in enumerate(rows):
                spad[c["local_addr"] + r] = v
        elif c["type"] == EXEC and c["opcode"] == COMPUTE:
            if dataflow != 0:
                raise Wrong(f"{tag}: COMPUTE before EXEC CONFIG set output-stationary")
            base = c["rd"] & ~ACC_BIT
            k = c["k_dim"]
            for i in range(dim):
                a_row = spad.get(c["rs1"] + i)
                if a_row is None:
                    raise Wrong(f"{tag} reads scratchpad row {c['rs1'] + i}, which no LOAD wrote")
                out = acc.get(base + i, [0] * dim) if c["chain"] & 1 else [0] * dim
                out = list(out)
                for j in range(dim):
                    b_row = spad.get(c["rs2"] + j)          # B's column j, corner-turned
                    if b_row is None:
                        raise Wrong(f"{tag} reads scratchpad row {c['rs2'] + j}, which no LOAD wrote")
                    s = out[j] + sum(a_row[q] * b_row[q] for q in range(k))
                    out[j] = (s + 2 ** 31) % 2 ** 32 - 2 ** 31   # int32 wraps
                acc[base + i] = out
        elif c["type"] == STORE:
            if not c["local_addr"] & ACC_BIT or c["activation"] != 0:
                raise Wrong(f"{tag}: version 1 stores an accumulator slot, with no activation")
            base = c["local_addr"] & ~ACC_BIT
            for r in range(c["len"]):
                a = c["dram_addr"] + r * c["stride"]
                where = span(tag, a, c["row_bytes"], len(l3))
                v = acc.get(base + r)
                if v is None:
                    raise Wrong(f"{tag} stores accumulator row {base + r}, which no COMPUTE wrote")
                l3[where] = struct.pack(f"<{dim}i", *v)[:c["row_bytes"]]
        else:
            raise Wrong(f"{tag}: type {c['type']} opcode {c['opcode']} is not version 1's")
        # Completion.
        if sy & RELEASE:
            state[pool(c, "src")][src] = "FREE"
        if sy & SIGNAL:
            state[pool(c, "dst")][dst] = "READY"
    held = {f"{p} {t}": s for p in state for t, s in state[p].items() if s != "FREE"}
    if held:
        raise Wrong(f"slots still held at the end: {held}")
    bad = []
    for x in doc["results"]:
        got = bytes(l3[span(f"result {x['tile']}", x["addr"], x["row_bytes"] * dim, len(l3))]).hex()
        if x["bytes"] is not None and got != x["bytes"]:
            bad.append(x["tile"])
    if bad:
        raise Wrong(f"results differ from the expected values: {', '.join(bad)}")
    return len(doc["commands"]), len(doc["results"])


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    try:
        with open(sys.argv[1], encoding="utf-8") as f:
            doc = json.load(f)
        n, r = run(doc)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"tile_cmds_run: {sys.argv[1]}: unreadable: {e}", file=sys.stderr)
        return 2
    except Wrong as e:
        print(f"tile_cmds_run: {sys.argv[1]}: {e}", file=sys.stderr)
        return 1
    print(f"{sys.argv[1]}: {n} commands, {r} result tiles match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
