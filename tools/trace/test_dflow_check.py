#!/usr/bin/env python3
"""
Self-test for dflow_check.py (docs/plans/domain-flow-debugger.md step 2): every invariant must
FAIL on a bundle that breaks it, and the clean bundle must pass. A checker that passes everything
is not a checker.

Usage: python3 test_dflow_check.py <clean-bundle.dflow>

The clean bundle must be version 3 with step snapshots and an L-T1 overlay (kpu-run --level
block-sequential --dflow-values all), so the value and overlay invariants are exercised too.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE / "dflow_check.py"
BUNDLE = None
FMT = {"u8": ("<B", 1), "u16": ("<H", 2), "u32": ("<I", 4), "u64": ("<Q", 8), "i64": ("<q", 8), "f32": ("<f", 4),
       "f64": ("<d", 8)}
OPENS, CLOSES = 1, 2
STORE_KIND = 1                                       # enum Action::Kind
NONE64 = 0xFFFFFFFFFFFFFFFF


def run(path):
    p = subprocess.run([sys.executable, str(CHECK), str(path), "--json"], capture_output=True, text=True)
    report = json.loads(p.stdout) if p.returncode in (0, 1) else None
    return p.returncode, report


class Bundle:
    """A writable copy of the clean bundle, with column-level edits."""

    def __init__(self, src):
        self.dir = Path(tempfile.mkdtemp(prefix="dflow-selftest-")) / "b.dflow"
        shutil.copytree(src, self.dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())

    def rows(self, table):
        return self.manifest["tables"][table]["rows"]

    def column(self, table, name):
        t = self.manifest["tables"][table]
        for c in t["columns"]:
            if c["name"] == name:
                return self.dir / t["file"], c["offset"], c["dtype"]
        raise KeyError(name)

    def get(self, table, name, row):
        path, off, dtype = self.column(table, name)
        fmt, size = FMT[dtype]
        return struct.unpack_from(fmt, path.read_bytes(), off + row * size)[0]

    def set(self, table, name, row, value):
        path, off, dtype = self.column(table, name)
        fmt, size = FMT[dtype]
        b = bytearray(path.read_bytes())
        struct.pack_into(fmt, b, off + row * size, value)
        path.write_bytes(bytes(b))

    def find(self, table, pred):
        return next(i for i in range(self.rows(table)) if pred(i))

    def save_manifest(self):
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest))

    def close(self):
        shutil.rmtree(self.dir.parent, ignore_errors=True)


class DflowCheckSelfTest(unittest.TestCase):
    def setUp(self):
        self.b = Bundle(BUNDLE)

    def tearDown(self):
        self.b.close()

    def assertFails(self, invariant, needle=None):
        code, report = run(self.b.dir)
        self.assertEqual(code, 1, f"expected {invariant} to fail, checker exited {code}")
        hits = [v["message"] for v in report["violations"] if v["invariant"] == invariant]
        self.assertTrue(hits, f"violations were {sorted({v['invariant'] for v in report['violations']})}")
        if needle:
            self.assertTrue(any(needle in h for h in hits), f"no {invariant} message names {needle!r}: {hits[:3]}")

    def assertUnreadable(self):
        code, _ = run(self.b.dir)
        self.assertEqual(code, 2)

    # -- the clean run ------------------------------------------------------
    def test_clean_bundle_passes(self):
        code, report = run(self.b.dir)
        self.assertEqual(code, 0, report and report["violations"][:5])

    def test_the_run_exercises_what_is_checked(self):
        # A check over nothing passes vacuously: the clean run must open and release residencies,
        # nest loops at least three deep, and record every step.
        self.assertFalse(self.b.manifest["cut"])
        self.assertGreater(self.b.rows("residency"), 10)
        depths = {self.b.get("iteration", "depth", i) for i in range(self.b.rows("iteration"))}
        self.assertTrue({0, 1, 2, 3} <= depths, depths)
        flags = {self.b.get("steps", "flags", i) & (OPENS | CLOSES) for i in range(self.b.rows("steps"))}
        self.assertTrue({OPENS, CLOSES} <= flags)

    # -- D2 -----------------------------------------------------------------
    def test_d2_step_out_of_order(self):
        self.b.set("steps", "action", 5, 6)
        self.assertFails("D2", "steps[5]")

    def test_d2_uncut_bundle_missing_actions(self):
        self.b.manifest["totals"]["actions"] += 1
        self.b.save_manifest()
        self.assertFails("D2", "not cut")

    # -- D3 -----------------------------------------------------------------
    def test_d3_occupancy_not_the_residencies(self):
        self.b.set("steps", "held_l3", 10, self.b.get("steps", "held_l3", 10) + 1)
        self.assertFails("D3", "after step 10")

    # -- D4 -----------------------------------------------------------------
    def test_d4_residency_ends_before_it_opens(self):
        self.b.set("residency", "close", 0, self.b.get("residency", "open", 0))
        self.assertFails("D4", "not after its open")

    def test_d4_residency_never_ends(self):
        self.b.set("residency", "close", 3, NONE64)
        self.assertFails("D4", "never ends")

    def test_d4_open_without_an_opening_step(self):
        i = self.b.get("residency", "open", 2)
        self.b.set("steps", "flags", i, self.b.get("steps", "flags", i) & ~OPENS)
        self.assertFails("D4", "does not open it")

    # -- D5 -----------------------------------------------------------------
    def test_d5_over_capacity(self):
        peak = max(self.b.get("steps", "held_l3", i) for i in range(self.b.rows("steps")))
        self.b.manifest["machine"]["l3"] = peak - 1
        self.b.save_manifest()
        self.assertFails("D5", "over its capacity")

    # -- D6 -----------------------------------------------------------------
    def test_d6_program_summary_short(self):
        i = self.b.find("iteration", lambda i: self.b.get("iteration", "depth", i) == 0)
        self.b.set("iteration", "loads", i, self.b.get("iteration", "loads", i) - 1)
        self.assertFails("D6", "the program's iteration")

    def test_d6_iterations_overlap(self):
        rows = sorted((self.b.get("iteration", "first", i), i) for i in range(self.b.rows("iteration"))
                      if self.b.get("iteration", "depth", i) == 2)
        (_, a), (first_b, _) = rows[0], rows[1]
        self.b.set("iteration", "last", a, first_b)
        self.assertFails("D6", "overlap")

    def test_d6_counts_not_the_steps(self):
        i = self.b.find("iteration", lambda i: self.b.get("iteration", "depth", i) == 3)
        self.b.set("iteration", "peak_l2", i, self.b.get("iteration", "peak_l2", i) + 5)
        self.assertFails("D6", "its steps give")

    # -- D8 -----------------------------------------------------------------
    def test_d8_time_column(self):
        self.b.manifest["tables"]["residency"]["columns"].append({"name": "t_open", "dtype": "u64", "offset": 0})
        self.b.save_manifest()
        self.assertFails("D8", "t_open")

    # -- D9 -----------------------------------------------------------------
    def test_d9_step_address_not_its_residency(self):
        i = self.b.find("steps", lambda i: self.b.get("steps", "residency", i) != NONE64)
        self.b.set("steps", "slot", i, self.b.get("steps", "slot", i) + 1)
        self.assertFails("D9", "its residency")

    def test_d9_two_tiles_in_one_slot(self):
        # Residency 1 moved into residency 0's slot while 0 is still held.
        self.assertLess(self.b.get("residency", "open", 1), self.b.get("residency", "close", 0))
        self.b.set("residency", "slot", 1, self.b.get("residency", "slot", 0))
        self.assertFails("D9", "still holds")

    def test_d6_iteration_past_the_run_is_reported_not_a_crash(self):
        i = self.b.find("iteration", lambda i: self.b.get("iteration", "depth", i) == 3)
        self.b.set("iteration", "last", i, self.b.rows("steps") + 100)
        self.assertFails("D6", "runs past")

    # -- REF ----------------------------------------------------------------
    def test_ref_position_out_of_range_is_reported_not_a_crash(self):
        self.b.set("steps", "where", 7, self.b.rows("where") + 5)
        self.assertFails("REF", "names position")

    def test_ref_loop_variable_out_of_range(self):
        self.b.set("loops", "var", 0, len(self.b.manifest["vars"]) + 3)
        self.assertFails("REF", "names variable")

    def test_ref_tile_operand_out_of_range(self):
        self.b.set("tiles", "operand", 0, len(self.b.manifest["operands"]) + 1)
        self.assertFails("REF", "names operand")

    def test_ref_step_tile_out_of_range(self):
        self.b.set("steps", "tile", 4, self.b.rows("tiles") + 2)
        self.assertFails("REF", "names tile")

    def test_ref_step_kind_out_of_range(self):
        self.b.set("steps", "kind", 4, 200)
        self.assertFails("REF", "has kind")

    def test_ref_position_out_of_range_in_a_cut_bundle(self):
        # A cut bundle skips the recount; its recorded positions are still checked.
        self.b.manifest["cut"] = True
        self.b.manifest["totals"]["actions"] += 10
        self.b.save_manifest()
        self.b.set("steps", "where", 7, self.b.rows("where") + 5)
        self.assertFails("REF", "names position")

    def test_ref_loop_variable_out_of_range_in_a_cut_bundle(self):
        self.b.manifest["cut"] = True
        self.b.manifest["totals"]["actions"] += 10
        self.b.save_manifest()
        self.b.set("loops", "var", 0, len(self.b.manifest["vars"]) + 3)
        self.assertFails("REF", "names variable")

    # -- V1-V5: values (version 2) -------------------------------------------
    def step_snapshot(self, kind=None):
        for k in range(self.b.rows("snapshots")):
            st = self.b.get("snapshots", "step", k)
            if st != NONE64 and (kind is None or self.b.get("steps", "kind", st) == kind):
                return k
        self.fail("the clean bundle has no such step snapshot")

    def final_snapshot(self, tile):
        return self.find("snapshots", lambda k: self.b.get("snapshots", "step", k) == NONE64
                         and self.b.get("snapshots", "tile", k) == tile)

    def find(self, table, pred):
        return self.b.find(table, pred)

    def test_values_are_exercised(self):
        self.assertGreaterEqual(self.b.manifest["version"], 2)
        self.assertTrue(self.b.manifest["values"]["selected"])
        self.step_snapshot(STORE_KIND)

    def test_v1_elements_past_the_end(self):
        k = self.step_snapshot()
        self.b.set("snapshots", "first", k, self.b.rows("elements"))
        self.assertFails("V1", "run past")

    def test_v2_snapshot_of_another_tile(self):
        k = self.step_snapshot()
        self.b.set("snapshots", "tile", k, (self.b.get("snapshots", "tile", k) + 1) % self.b.rows("tiles"))
        self.assertFails("V2", "acts on tile")

    def test_v2_snapshot_in_the_wrong_channel(self):
        k = self.step_snapshot(STORE_KIND)
        self.b.set("snapshots", "chan", k, 2)        # a Store leaves its tile in DRAM, not L2
        self.assertFails("V2", "not where the action leaves")

    def test_v3_wrong_shape(self):
        k = self.step_snapshot()
        self.b.set("snapshots", "height", k, self.b.get("snapshots", "height", k) - 1)
        self.assertFails("V3", "its tile is")

    def test_v4_two_finals_for_one_tile(self):
        a = self.final_snapshot(0)
        other = self.find("snapshots", lambda k: self.b.get("snapshots", "step", k) == NONE64
                          and self.b.get("snapshots", "tile", k) != 0)
        self.b.set("snapshots", "tile", other, 0)
        self.assertFails("V4", "two final values")

    def test_v4_step_snapshots_without_a_selection(self):
        self.b.manifest["values"]["selected"] = ""
        self.b.save_manifest()
        self.assertFails("V4", "no tiles were selected")

    def test_v5_store_differs_from_final(self):
        k = self.step_snapshot(STORE_KIND)
        f = self.final_snapshot(self.b.get("snapshots", "tile", k))
        first = self.b.get("snapshots", "first", f)
        self.b.set("elements", "v", first, self.b.get("elements", "v", first) + 1.0)
        self.assertFails("V5", "differs from its final")

    def test_v1_channel_out_of_range_is_one_violation(self):
        k = self.step_snapshot(STORE_KIND)
        self.b.set("snapshots", "chan", k, 77)
        code, report = run(self.b.dir)
        self.assertEqual(code, 1)
        mine = [v for v in report["violations"] if f"snapshot {k} " in v["message"] or f"(snapshot {k})" in v["message"]]
        self.assertTrue(any(v["invariant"] == "V1" for v in mine))
        self.assertFalse(any(v["invariant"] in ("V2", "V5") for v in mine), mine)

    def test_boolean_cut_at_is_unreadable(self):
        self.b.manifest["values"]["cut_at"] = True
        self.b.save_manifest()
        self.assertUnreadable()

    def test_zero_tile_rows_is_unreadable(self):
        self.b.manifest["operands"][0]["tile_rows"] = 0
        self.b.save_manifest()
        self.assertUnreadable()

    def test_values_format_other_than_float32_is_unreadable(self):
        self.b.manifest["values"]["format"] = "bfloat16"
        self.b.save_manifest()
        self.assertUnreadable()

    # -- L1-L5: the L-T1 overlay (version 3) ---------------------------------
    def leg_of(self, pred):
        return self.b.find("lt1_legs", pred)

    def test_overlay_is_exercised(self):
        self.assertEqual(self.b.manifest["version"], 3)
        self.assertIsNotNone(self.b.manifest["lt1"])
        self.assertGreater(self.b.rows("lt1_legs"), self.b.rows("steps"))     # a Store has two legs

    def test_l1_leg_on_a_lane_its_process_lacks(self):
        g = self.leg_of(lambda g: self.b.get("lt1_legs", "proc", g) == 3)    # cf
        self.b.set("lt1_legs", "lane", g, 99)
        self.assertFails("L1", "lane 99")

    def test_l1_credit_action_with_a_process(self):
        g = self.leg_of(lambda g: self.b.get("lt1_legs", "kind", g) == 7)    # RELEASE
        self.b.set("lt1_legs", "proc", g, 0)
        self.assertFails("L1", "has a process")

    def test_l2_leg_of_another_kind(self):
        g = self.leg_of(lambda g: self.b.get("lt1_legs", "kind", g) == 6)    # CALL
        self.b.set("lt1_legs", "kind", g, 4)                                  # FEED
        self.assertFails("L2", "is a FEED")

    def test_l2_action_without_a_leg(self):
        a = self.b.get("lt1_legs", "action", 0)
        for g in range(self.b.rows("lt1_legs")):
            if self.b.get("lt1_legs", "action", g) == a:
                self.b.set("lt1_legs", "action", g, a + 1)
        self.assertFails("L2", f"action {a} has no L-T1 leg")

    def test_l3_legs_overlap_on_a_lane(self):
        cf = sorted((self.b.get("lt1_legs", "start", g), g) for g in range(self.b.rows("lt1_legs"))
                    if self.b.get("lt1_legs", "proc", g) == 3 and self.b.get("lt1_legs", "lane", g) == 0)
        (_, g0), (s1, _) = cf[0], cf[1]
        self.b.set("lt1_legs", "finish", g0, s1 + 10)
        self.assertFails("L3", "overlap")

    def test_l3_makespan_not_the_latest_finish(self):
        self.b.manifest["lt1"]["makespan"] += 5
        self.b.save_manifest()
        self.assertFails("L3", "makespan")

    def test_l4_slot_wait_on_a_non_load(self):
        g = self.leg_of(lambda g: self.b.get("lt1_legs", "kind", g) == 2)    # MOVE
        self.b.set("lt1_legs", "slot_wait", g, 7)
        self.assertFails("L4", "only a Load waits")

    def test_l4_waits_do_not_add_up(self):
        g = self.leg_of(lambda g: self.b.get("lt1_legs", "kind", g) == 0)    # LOAD
        self.b.set("lt1_legs", "slot_wait", g, self.b.get("lt1_legs", "slot_wait", g) + 3)
        self.assertFails("L4", "the manifest says")

    def test_l4_wait_longer_than_the_start(self):
        g = self.leg_of(lambda g: self.b.get("lt1_legs", "kind", g) == 0)    # LOAD
        self.b.set("lt1_legs", "slot_wait", g, self.b.get("lt1_legs", "start", g) + 1)
        self.assertFails("L4", "but starts at cycle")

    def test_l5_pass_disagrees_with_the_tiles(self):
        self.b.set("lt1_tiles", "pass", 0, 0)
        self.assertFails("L5", "pass is True")

    def test_l5_a_missing_proximity_row(self):
        self.b.manifest["tables"]["lt1_tiles"]["rows"] -= 1
        self.b.save_manifest()
        self.assertFails("L5", "proximity rows")

    def test_l1_overlay_rows_without_an_overlay(self):
        self.b.manifest["lt1"] = None
        self.b.save_manifest()
        self.assertFails("L1", "no L-T1 overlay")

    def test_d8_time_outside_the_overlay(self):
        self.b.manifest["tables"]["residency"]["columns"].append({"name": "start", "dtype": "u64", "offset": 0})
        self.b.save_manifest()
        self.assertFails("D8", "residency.start")

    # -- unreadable ---------------------------------------------------------
    def test_malformed_manifest_is_unreadable(self):
        del self.b.manifest["totals"]["loads"]
        self.b.save_manifest()
        self.assertUnreadable()

    def test_reference_list_not_an_array_is_unreadable(self):
        # A string as long as the array would let every index pass REF.
        self.b.manifest["operands"] = "x" * len(self.b.manifest["operands"])
        self.b.save_manifest()
        self.assertUnreadable()

    def test_malformed_table_entry_is_unreadable(self):
        self.b.manifest["tables"]["steps"]["rows"] = "many"
        self.b.save_manifest()
        self.assertUnreadable()

    def test_negative_column_offset_is_unreadable(self):
        self.b.manifest["tables"]["steps"]["columns"][0]["offset"] = -8
        self.b.save_manifest()
        self.assertUnreadable()

    def test_unknown_version_is_unreadable(self):
        self.b.manifest["version"] = 99
        self.b.save_manifest()
        self.assertUnreadable()

    def test_missing_table_is_unreadable(self):
        (self.b.dir / "steps.bin").unlink()
        self.assertUnreadable()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    BUNDLE = Path(sys.argv.pop(1)).resolve()
    unittest.main()
