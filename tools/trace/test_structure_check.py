#!/usr/bin/env python3
"""
Self-test of the structure record's converter and checker (#392).

    python3 test_structure_check.py <samples dir>

- Every sample record checks clean, and matches the JSON Schema's required shape.
- Each invariant, broken on purpose in a copy of a sample, is reported under its own id.
- The converter maps both floorplan formats and refuses any other.
- The exit codes: 0 clean, 1 violated, 2 unreadable.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import structure_check  # noqa: E402
import structure_from_floorplan  # noqa: E402
import structure_diff  # noqa: E402

FAILS = 0


def expect(cond: bool, what: str):
    global FAILS
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        FAILS += 1


def ids(v):
    return {x["invariant"] for x in v}


def required_shape(rec: dict, schema: dict) -> list[str]:
    """The schema's required keys and enums, checked without a jsonschema dependency."""
    out = []
    for k in schema["required"]:
        if k not in rec:
            out.append(f"missing {k}")
    blk = schema["$defs"]["block"]
    kinds = set(blk["properties"]["kind"]["enum"])
    for b in rec.get("blocks", []):
        for k in blk["required"]:
            if k not in b:
                out.append(f"block missing {k}")
        if b.get("kind") not in kinds:
            out.append(f"block kind {b.get('kind')} not in the schema's enum")
    return out


def main(samples: str) -> int:
    schema = json.load(open(os.path.join(HERE, "structure.schema.json")))
    # The schema's vocabulary is the checker's.
    expect(set(schema["$defs"]["block"]["properties"]["kind"]["enum"]) == structure_check.KINDS,
           "the schema's block kinds are the checker's vocabulary")
    files = sorted(f for f in os.listdir(samples) if f.endswith(".structure.json"))
    expect(len(files) >= 2, f"samples: {files}")
    recs = {}
    for f in files:
        rec = json.load(open(os.path.join(samples, f)))
        recs[f] = rec
        v = structure_check.check(rec)
        expect(not v, f"{f} checks clean" + (f": {v[:2]}" if v else ""))
        expect(not required_shape(rec, schema), f"{f} has the schema's shape")
    sources = {r["provenance"]["source"] for r in recs.values()}
    expect(sources == {"model", "rtl"}, "a sample from each producer, the model and the RTL")

    rtl = next(r for r in recs.values() if r["provenance"]["source"] == "rtl")
    model = next(r for r in recs.values() if r["provenance"]["source"] == "model")

    def broken(base, mutate):
        r = copy.deepcopy(base)
        mutate(r)
        return ids(structure_check.check(r))

    expect("S1" in broken(rtl, lambda r: r.update(format="kpu-gen-floorplan")), "S1: another format is refused")
    expect("S1" in broken(rtl, lambda r: r["provenance"].pop("source")), "S1: a record without its source")
    expect("S2" in broken(rtl, lambda r: r["blocks"].append(copy.deepcopy(r["blocks"][1]))), "S2: a duplicate block id")
    expect("S2" in broken(rtl, lambda r: r["blocks"][1].update(kind="cache")), "S2: a kind outside the vocabulary")
    expect("S3" in broken(rtl, lambda r: r["blocks"][1].update(parent="nowhere")), "S3: a parent that is no block")

    def cycle(r):
        a, b = r["blocks"][1], r["blocks"][2]
        a["parent"], b["parent"] = b["id"], a["id"]
    expect("S3" in broken(rtl, cycle), "S3: a cycle in the tree")
    expect("S4" in broken(rtl, lambda r: r.setdefault("channels", []).append(
        {"id": "c0", "kind": "data", "from": r["blocks"][1]["id"], "to": "nowhere"})), "S4: a channel to no block")
    expect("S4" in broken(rtl, lambda r: r["networks"][0]["links"][0].update(to="nowhere")),
           "S4: a link to no node of its network")
    expect("S5" in broken(rtl, lambda r: r["blocks"][1].setdefault("resources", []).append(
        {"kind": "memory", "bytes": 100, "depth": 16, "width_bits": 64})), "S5: a memory whose bytes disagree")
    expect("S5" in broken(rtl, lambda r: r["blocks"][1].setdefault("resources", []).append(
        {"kind": "slot_pool", "slots": -1})), "S5: a negative slot count")
    clean = copy.deepcopy(rtl)
    clean["blocks"][1].setdefault("resources", []).append({"kind": "memory", "bytes": 128, "depth": 16, "width_bits": 64})
    expect(not structure_check.check(clean), "S5: a consistent memory is accepted")

    def outside(r):
        b = next(x for x in r["blocks"] if x.get("parent") and x.get("rect"))
        b["rect"] = [b["rect"][0] + 1000.0, b["rect"][1], b["rect"][2], b["rect"][3]]
    expect("F1" in broken(rtl, outside), "F1: a block outside its parent")
    expect("F2" in broken(rtl, lambda r: r["floorplan"]["area"].update(whitespace=r["floorplan"]["area"]["whitespace"] + 1.0)),
           "F2: an area breakdown that does not sum to the die")

    def moved_end(r):
        ln = next(x for x in r["networks"][0]["links"] if x.get("points"))
        ln["points"][0] = [ln["points"][0][0] + 0.5, ln["points"][0][1]]
    expect("F3" in broken(rtl, moved_end), "F3: a link whose points do not start at its node")
    expect(not structure_check.check({k: v for k, v in model.items() if k != "floorplan"}),
           "a record without a floorplan skips F1 to F3")

    # The converter.
    try:
        structure_from_floorplan.convert({"format": "something", "version": 3})
        expect(False, "the converter refuses another format")
    except structure_from_floorplan.ConvertError:
        expect(True, "the converter refuses another format")
    expect(any(b.get("resource") for b in model["blocks"]), "the model's blocks keep their naming-map resources")
    expect(all(b.get("rtl_path") == b["id"] for b in rtl["blocks"]), "the RTL's blocks keep their instance paths")
    expect(model["floorplan"]["estimated"] is True, "the model's floorplan is flagged an estimate")

    # The comparer: provenance aside, a record is itself; a moved block, a dropped link, a
    # changed kind or a renamed id is a difference, each named by its path.
    expect(not structure_diff.compare(model, copy.deepcopy(model)), "diff: a record is the same as itself")
    other = copy.deepcopy(model)
    other["provenance"] = {"producer": "someone else", "source": "rtl"}
    expect(not structure_diff.compare(model, other), "diff: provenance is not compared")
    other["blocks"] = list(reversed(other["blocks"]))
    expect(not structure_diff.compare(model, other), "diff: blocks compare by id, not by order")

    def differs(mutate, what):
        r = copy.deepcopy(model)
        mutate(r)
        d = structure_diff.compare(model, r)
        expect(any(what in x for x in d), f"diff: {what} is reported" + ("" if d else " (no difference found)"))
    placed = next(i for i, b in enumerate(model["blocks"]) if b.get("parent") and b.get("rect"))
    bid = model["blocks"][placed]["id"]
    differs(lambda r: r["blocks"][placed]["rect"].__setitem__(0, r["blocks"][placed]["rect"][0] + 1e-3),
            f"$.blocks[{bid}].rect[0]")
    differs(lambda r: r["blocks"][placed].update(kind="l2_bank"), f"$.blocks[{bid}].kind")
    differs(lambda r: r["blocks"].pop(placed), f"$.blocks[{bid}]: only in the first")
    lid = model["networks"][0]["links"][0]["id"]
    differs(lambda r: r["networks"][0]["links"].pop(0), f"$.networks[noc].links[{lid}]: only in the first")

    # Relative and absolute tolerances are apart: at --rel 0.01, 0.005 and 0.01 differ (the
    # relative bound is 0.0001), which a relative tolerance reused as an absolute one hid.
    small = copy.deepcopy(model)
    small["blocks"][placed]["rect"][0] = 0.005
    smaller = copy.deepcopy(small)
    smaller["blocks"][placed]["rect"][0] = 0.01
    expect(bool(structure_diff.compare(small, smaller, rel=0.01)), "diff: 0.005 and 0.01 differ at --rel 0.01")
    expect(not structure_diff.compare(small, smaller, rel=0.01, abs_tol=0.01), "diff: --abs 0.01 accepts them")

    # The exit codes.
    chk = os.path.join(HERE, "structure_check.py")
    first = os.path.join(samples, files[0])
    expect(subprocess.run([sys.executable, chk, first], capture_output=True).returncode == 0, "exit 0 on a clean record")
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "bad.json")
        r = copy.deepcopy(rtl)
        r["blocks"][1]["parent"] = "nowhere"
        json.dump(r, open(bad, "w"))
        expect(subprocess.run([sys.executable, chk, bad], capture_output=True).returncode == 1, "exit 1 on a violation")
        junk = os.path.join(d, "junk.json")
        open(junk, "w").write("{not json")
        expect(subprocess.run([sys.executable, chk, junk], capture_output=True).returncode == 2, "exit 2 on an unreadable record")
    print(f"{'FAILED' if FAILS else 'passed'}: {FAILS} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "tests", "program", "structure")))
