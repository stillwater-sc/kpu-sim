#!/usr/bin/env python3
"""
A minimal writer for the shared tile-flow record, .dflow version 6
(docs/plans/shared-tile-flow-record.md; #386).

kpu-sim writes .dflow from C++ (include/sw/kpu/program/record/domain_flow_record.hpp). The KPU RTL
generators (kpugen, systars) write the same format from Python; this module is the reference for
them to vendor, so they don't fork the format. It writes the columnar container
(record/columnar.hpp): a directory holding a manifest.json and one .bin file per table. Each column
is little-endian and padded to 8 bytes, and the manifest names each column's dtype and offset.

It knows the container and the version-6 enums, nothing about any producer's schedule. The
producer builds the manifest and the tables; dflow_check.py says whether the result conforms.

    from dflow_writer import Table, write_bundle
    steps = Table("steps")
    steps.put("action", "u64", [0, 1, 2])
    ...
    write_bundle("run.dflow", manifest, [steps, ...])

Standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import struct
from pathlib import Path

VERSION = 6
# Version 6's minor revision (manifest "minor"): 1 adds the acc -> l3 DRAIN (the RTL's store) and the
# word-addressed L3 (geometry.l3 words and word_bytes instead of slots and banks; #398). A record
# that uses neither may say 0. Readers of 6.1 read 6.0.
MINOR = 1
NONE8, NONE32, NONE64 = 0xFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF

# The format's enums. A producer writes them into the manifest verbatim (layers, event_kinds), and
# every reader indexes them. They change only with the format's version.
LEVELS = ("L-B", "L-T1", "L-T2", "L-CA", "RTL")
LAYERS = ["dram", "l3", "l2", "acc", "l1"]
EVENT_KINDS = ["DMA", "BLOCK_MOVE", "STREAM", "COMPUTE", "DRAIN", "CREDIT", "TOKEN", "BM_ROW", "STR_ROW",
               "DRAIN_ROW"]
ELEMENTS = {"int8": 1, "int32": 4, "fp16": 2, "bf16": 2, "fp32": 4, "fp8_e4m3": 1, "fp8_e5m2": 1}
# The CSP vocabulary the steps table indexes (enum csp::Action::Kind, csp::Chan, csp::ProcessKind).
KINDS = ["LOAD", "STORE", "MOVE", "WRITEBACK", "FEED", "DRAIN", "CALL", "RELEASE", "INHERIT", "RETAIN", "L2_RELEASE"]
CHANNELS = ["dram", "l3", "l2", "cf"]
PROCESSES = ["dma", "bm", "str", "cf"]

_FMT = {"u8": "B", "u16": "H", "u32": "I", "u64": "Q", "i64": "q", "f32": "f", "f64": "d"}


class Table:
    """One table: named columns of one dtype each, all of the same length."""

    def __init__(self, name, file=None):
        self.name, self.file = name, file or f"{name}.bin"
        self.columns = []                      # (name, dtype, values)

    def put(self, column, dtype, values):
        if dtype not in _FMT:
            raise ValueError(f"{self.name}.{column}: unknown dtype {dtype}")
        values = list(values)
        if self.columns and len(values) != len(self.columns[0][2]):
            raise ValueError(f"{self.name}.{column} has {len(values)} rows; {self.columns[0][0]} has "
                             f"{len(self.columns[0][2])}")
        self.columns.append((column, dtype, values))
        return self

    @property
    def rows(self):
        return len(self.columns[0][2]) if self.columns else 0


def write_table(directory, table, rows=None):
    """Write one table's .bin and return its manifest entry. `rows` is the row count of a table
    with no columns."""
    blob, cols = bytearray(), []
    for name, dtype, values in table.columns:
        cols.append({"name": name, "dtype": dtype, "offset": len(blob)})
        blob += struct.pack(f"<{len(values)}{_FMT[dtype]}", *values)
        blob += b"\0" * (-len(blob) % 8)
    (Path(directory) / table.file).write_bytes(bytes(blob))
    return {"file": table.file, "rows": table.rows if table.columns else (rows or 0), "columns": cols}


def write_bundle(directory, manifest, tables):
    """Write a bundle: every table, then the manifest naming them."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    m = dict(manifest)
    m.setdefault("format", "kpu-dflow")
    m.setdefault("version", VERSION)
    m.setdefault("minor", MINOR)
    m["tables"] = {t.name: write_table(d, t) for t in tables}
    (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    return d


def pack(element, values):
    """Values of one element type as little-endian bytes. The floating types take Python floats
    and are rounded by struct (fp32, fp16); bf16 and fp8 take their raw bit patterns as ints."""
    if element == "int8":
        return struct.pack(f"<{len(values)}b", *values)
    if element == "int32":
        return struct.pack(f"<{len(values)}i", *values)
    if element == "fp32":
        return struct.pack(f"<{len(values)}f", *values)
    if element == "fp16":
        return struct.pack(f"<{len(values)}e", *values)
    if element == "bf16":
        return struct.pack(f"<{len(values)}H", *values)
    if element in ("fp8_e4m3", "fp8_e5m2"):
        return bytes(values)
    raise ValueError(f"unknown element type {element}")
