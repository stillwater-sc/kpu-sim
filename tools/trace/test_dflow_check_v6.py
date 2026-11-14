#!/usr/bin/env python3
"""
Self-test for dflow_check.py's version-6 invariants (docs/plans/shared-tile-flow-record.md; #386):
the record kpu-sim and the KPU RTL share. Each invariant must FAIL on a bundle that breaks it, and
the clean bundles must pass. A checker that passes everything is not a checker.

Usage: python3 test_dflow_check_v6.py <rtl-sample.dflow> <lb.dflow> [<rtl-sample-addressed.dflow>]

The third bundle is the sample's version-6.1 form (dflow_rtl_sample.py --addressed; #398): the
accumulator stores straight to L3, and L3 is word-addressed.

The RTL sample is the hand-written timed record (dflow_rtl_sample.py): L2, accumulator and L1
geometry, a cycle column, progress events, credits and tokens, int8 and int32 values with expected
values, and a lower-triangular operand. The L-B bundle is one kpu-run writes (version 6, no time).

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflow_check as dc  # noqa: E402
import dflow_writer as dw  # noqa: E402
from test_dflow_check import Bundle, run  # noqa: E402

CHECK = HERE / "dflow_check.py"
RTL = None
LB = None
ADDRESSED = None
NONE64 = dc.NONE64
KIND = {k: i for i, k in enumerate(dc.EVENT_KINDS)}
LAYER = {k: i for i, k in enumerate(dc.LAYERS)}


def result(path):
    p = subprocess.run([sys.executable, str(CHECK), str(path), "--result"], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else None)


class Checks(unittest.TestCase):
    SOURCE = None

    def setUp(self):
        self.b = Bundle(self.SOURCE)

    def tearDown(self):
        self.b.close()

    def assertFails(self, invariant, needle=None):
        code, report = run(self.b.dir)
        self.assertEqual(code, 1, f"expected {invariant} to fail, checker exited {code}")
        hits = [v["message"] for v in report["violations"] if v["invariant"] == invariant]
        self.assertTrue(hits, f"violations were {[(v['invariant'], v['message']) for v in report['violations'][:4]]}")
        if needle:
            self.assertTrue(any(needle in h for h in hits), f"no {invariant} message names {needle!r}: {hits[:3]}")

    def assertPasses(self):
        code, report = run(self.b.dir)
        self.assertEqual(code, 0, report and report["violations"][:5])

    def assertUnreadable(self):
        code, _ = run(self.b.dir)
        self.assertEqual(code, 2)

    def event(self, kind, pred=lambda i: True):
        return self.b.find("events", lambda i: self.b.get("events", "kind", i) == KIND[kind] and pred(i))

    def ev(self, col, i):
        return self.b.get("events", col, i)

    def set_ev(self, col, i, v):
        self.b.set("events", col, i, v)

    def operand_tile(self, name, ti, tj):
        op = [o["name"] for o in self.b.manifest["operands"]].index(name)
        return self.b.find("tiles", lambda t: (self.b.get("tiles", "operand", t), self.b.get("tiles", "ti", t),
                                               self.b.get("tiles", "tj", t)) == (op, ti, tj))

    def snapshot(self, tile, role=0, final=False):
        return self.b.find("snapshots", lambda k: self.b.get("snapshots", "tile", k) == tile
                           and self.b.get("snapshots", "role", k) == role
                           and (self.b.get("snapshots", "step", k) == NONE64) == final)


class RtlSample(Checks):
    """The timed, typed, domain-carrying record: every version-6 invariant, broken one at a time."""

    @classmethod
    def setUpClass(cls):
        cls.SOURCE = RTL

    # -- the clean sample ----------------------------------------------------
    def test_clean_sample_passes(self):
        self.assertPasses()

    def test_the_sample_exercises_what_is_checked(self):
        m = self.b.manifest
        self.assertEqual((m["version"], m["level"]), (6, "RTL"))
        self.assertIsNotNone(m["time"])
        kinds = {self.ev("kind", i) for i in range(self.b.rows("events"))}
        self.assertEqual(kinds, set(range(len(dc.EVENT_KINDS))), "every event kind")
        for layer in ("l2", "acc", "l1"):
            self.assertTrue(m["geometry"][layer]["modelled"], layer)
        self.assertEqual({o["element"] for o in m["operands"]}, {"int8", "int32"})
        self.assertIn("lower", {o["domain"]["kind"] for o in m["operands"]})
        self.assertTrue(m["values"]["expected"])
        self.assertEqual(m["result"]["values"], "match")

    def test_the_writer_and_the_checker_name_the_same_enums(self):
        self.assertEqual(dw.LAYERS, dc.LAYERS)
        self.assertEqual(dw.EVENT_KINDS, dc.EVENT_KINDS)
        self.assertEqual(dw.ELEMENTS, dc.ELEMENTS)
        self.assertEqual(tuple(dw.LEVELS), tuple(dc.LEVELS))

    # -- G1: geometry --------------------------------------------------------
    def test_g1_a_layer_not_declared(self):
        del self.b.manifest["geometry"]["l2"]
        self.b.save_manifest()
        self.assertFails("G1", "not declared")

    def test_g1_not_modelled_without_why(self):
        self.b.manifest["geometry"]["l1"] = {"modelled": False}
        self.b.save_manifest()
        self.assertFails("G1", "does not say why")

    def test_g1_l3_is_not_the_machine(self):
        self.b.manifest["geometry"]["l3"]["slots"] = 5
        self.b.save_manifest()
        self.assertFails("G1", "the machine is")

    def test_e1_a_six_zero_record_has_no_acc_to_l3_store(self):
        d = self.event("DRAIN")
        self.set_ev("dst", d, LAYER["l3"])
        self.set_ev("dst_addr", d, 0)
        self.assertFails("E1", "needs version 6.1")

    def test_e4_a_six_zero_record_has_no_acc_to_l3_token(self):
        t = self.event("TOKEN")
        self.set_ev("src", t, LAYER["acc"])
        self.set_ev("dst", t, LAYER["l3"])
        self.assertFails("E4", "does not go downstream along a data edge")

    def test_g1_unknown_interleave(self):
        self.b.manifest["geometry"]["l2"]["interleave"] = "xor-hash"
        self.b.save_manifest()
        self.assertFails("G1", "interleave")

    def test_g1_zero_rows(self):
        self.b.manifest["geometry"]["acc"]["rows"] = 0
        self.b.save_manifest()
        self.assertFails("G1", "rows")

    # -- T1: time ------------------------------------------------------------
    def test_t1_l_b_with_time(self):
        self.b.manifest["level"] = "L-B"
        self.b.save_manifest()
        self.assertFails("T1", "declares time")
        self.assertFails("D8", "events.cycle")

    def test_t1_timed_level_without_time(self):
        self.b.manifest["time"] = None
        self.b.manifest["result"]["cycles"] = None
        self.b.save_manifest()
        self.assertFails("T1", "declares no time")

    def test_t1_cycles_to_result_is_not_the_latest_end(self):
        self.b.manifest["time"]["cycles"] += 1
        self.b.manifest["result"]["cycles"] += 1
        self.b.save_manifest()
        self.assertFails("T1", "latest event ends")

    def test_t1_result_cycles_disagree(self):
        self.b.manifest["result"]["cycles"] = 7
        self.b.save_manifest()
        self.assertFails("T1", "result.cycles")

    def test_t1_event_ends_before_it_starts(self):
        i = self.event("DMA")
        self.set_ev("cycle", i, self.ev("end", i) + 1)
        self.assertFails("T1", "after it ends")

    # -- E1: references, layers and addresses --------------------------------
    def test_e1_id_is_not_the_row(self):
        self.set_ev("id", 3, 4)
        self.assertFails("E1", "has id")

    def test_e1_event_in_a_layer_not_modelled(self):
        self.b.manifest["geometry"]["acc"] = {"modelled": False, "why": "pretend"}
        self.b.save_manifest()
        self.assertFails("E1", "declares not modelled")

    def test_e1_address_outside_the_layer(self):
        i = self.event("BLOCK_MOVE", lambda i: self.ev("dst", i) == LAYER["l2"])
        self.set_ev("dst_addr", i, 99)
        self.assertFails("E1", "outside l2")

    def test_e1_transaction_between_the_wrong_layers(self):
        i = self.event("DMA", lambda i: self.ev("dst", i) == LAYER["l3"])
        self.set_ev("dst", i, LAYER["l2"])
        self.assertFails("E1", "a DMA moves")

    def test_e1_parent_out_of_range(self):
        i = self.event("BM_ROW")
        self.set_ev("parent", i, 10_000)
        self.assertFails("E1", "parent")

    # -- E2: identities ------------------------------------------------------
    def test_e2_two_events_share_an_identity(self):
        i = self.event("BM_ROW", lambda i: self.ev("row", i) == 0)
        j = self.event("BM_ROW", lambda j: self.ev("parent", j) == self.ev("parent", i) and self.ev("row", j) == 1)
        self.set_ev("row", j, 0)
        self.set_ev("dst_addr", j, self.ev("dst_addr", i))
        self.assertFails("E2", "share an identity")

    def test_e2_two_events_are_one_actions(self):
        i = self.event("DMA")
        j = self.event("DMA", lambda j: j != i and self.ev("src", j) == self.ev("src", i))
        self.set_ev("action", j, self.ev("action", i))
        self.assertFails("E2", "are both the DMA of action")

    # -- E3: progress --------------------------------------------------------
    def test_e3_row_outside_the_tile(self):
        i = self.event("STR_ROW")
        self.set_ev("row", i, 9)
        self.assertFails("E3", "row 9 of a tile of 4 rows")

    def test_e3_row_outside_its_parents_time(self):
        i = self.event("DRAIN_ROW")
        p = self.ev("parent", i)
        self.set_ev("cycle", i, self.ev("cycle", p) - 1)
        self.assertFails("E3", "outside its parent")

    def test_e3_row_under_the_wrong_kind_of_parent(self):
        i = self.event("BM_ROW")
        self.set_ev("parent", i, self.event("STREAM"))
        self.assertFails("E3", "not a child of a BLOCK_MOVE")

    def test_e3_a_transaction_with_a_row(self):
        i = self.event("COMPUTE")
        self.set_ev("row", i, 0)
        self.assertFails("E3", "only row events do")

    # -- E4: credits and tokens ----------------------------------------------
    def test_e4_credit_going_downstream(self):
        i = self.event("CREDIT", lambda i: self.ev("src", i) == LAYER["acc"])
        self.set_ev("src", i, LAYER["l1"])
        self.set_ev("dst", i, LAYER["acc"])
        self.assertFails("E4", "does not go upstream against")

    def test_e4_token_before_its_tile_arrived(self):
        i = self.event("TOKEN")
        self.set_ev("cycle", i, 0)
        self.set_ev("end", i, 0)
        self.assertFails("E4", "precedes any transaction that arrived at")

    # -- E5: the join with the CSP actions -----------------------------------
    def test_e5_transaction_of_another_kind_of_action(self):
        i = self.event("DMA", lambda i: self.ev("dst", i) == LAYER["l3"])
        move = self.b.find("steps", lambda a: self.b.get("steps", "kind", a) == 2)    # a MOVE
        self.set_ev("action", i, move)
        self.assertFails("E5", "is a MOVE")

    def test_e5_transaction_of_another_tile(self):
        i = self.event("STREAM")
        t = self.ev("tile", i)
        self.set_ev("tile", i, (t + 1) % self.b.rows("tiles"))
        self.assertFails("E5", "its action")

    def test_e5_dma_at_another_slot(self):
        i = self.event("DMA", lambda i: self.ev("dst", i) == LAYER["l3"])
        self.set_ev("dst_addr", i, (self.ev("dst_addr", i) + 1) % 4)
        self.assertFails("E5", "at L3 slot")

    def test_e5_a_step_without_its_transaction(self):
        c = self.operand_tile("C", 1, 0)                         # its accumulator has two calls
        i = self.event("COMPUTE", lambda i: self.ev("tile", i) == c)
        j = self.event("COMPUTE", lambda j: j != i and self.ev("tile", j) == c)
        self.set_ev("action", j, self.ev("action", i))
        self.set_ev("where", j, self.ev("where", i))
        self.assertFails("E5", "has 0 transaction events")

    # -- X1-X3: element types, domains, expected values ----------------------
    def test_x1_unknown_element_type(self):
        self.b.manifest["operands"][0]["element"] = "int4"
        self.b.save_manifest()
        self.assertFails("X1", "int4")

    def test_x1_role_neither_actual_nor_expected(self):
        self.b.set("snapshots", "role", 0, 7)
        self.assertFails("X1", "role 7")

    def test_x1_payload_too_short_for_the_type(self):
        c = self.snapshot(self.operand_tile("C", 1, 0), final=True)
        self.b.set("snapshots", "first", c, self.b.rows("elements") - 8)    # an int32 4x4 tile is 64 bytes
        self.assertFails("X1", "run past")

    def test_x2_nonzero_outside_the_domain(self):
        k = self.snapshot(self.operand_tile("L", 0, 0), final=True)
        first = self.b.get("snapshots", "first", k)
        self.b.set("elements", "b", first + 1, 5)              # L[0,1]: above the diagonal
        self.assertFails("X2", "outside operand L's lower domain")

    def test_x2_a_negative_diagonal_is_a_domain(self):
        # d = -1 is strictly lower: L's diagonal is then outside it, and nonzero.
        self.b.manifest["operands"][0]["domain"] = {"kind": "lower", "diagonal": -1}
        self.b.save_manifest()
        self.assertFails("X2", "outside operand L's lower domain")
        code, report = run(self.b.dir)
        self.assertFalse([v for v in report["violations"] if "diagonal is" in v["message"]])

    def test_x2_diagonal_not_an_integer(self):
        self.b.manifest["operands"][0]["domain"] = {"kind": "lower", "diagonal": 0.5}
        self.b.save_manifest()
        self.assertFails("X2", "diagonal is 0.5")

    def test_x2_unknown_domain(self):
        self.b.manifest["operands"][0]["domain"] = {"kind": "triangle"}
        self.b.save_manifest()
        self.assertFails("X2", "triangle")

    def test_x2_band_without_widths(self):
        self.b.manifest["operands"][1]["domain"] = {"kind": "band", "lower": 1}
        self.b.save_manifest()
        self.assertFails("X2", "lower and upper")

    def test_x2_padding_window_past_the_operand(self):
        self.b.manifest["operands"][1]["domain"] = {"kind": "dense", "pad": {"rows": 9, "cols": 4}}
        self.b.save_manifest()
        self.assertFails("X2", "padding window")

    def test_x2_padding_is_out_of_domain(self):
        # B's last row as padding: its elements are nonzero, so they must be refused as data.
        self.b.manifest["operands"][1]["domain"] = {"kind": "dense", "pad": {"rows": 7, "cols": 4}}
        self.b.save_manifest()
        self.assertFails("X2", "outside operand B's dense domain")

    def test_x3_expected_flag_disagrees(self):
        self.b.manifest["values"]["expected"] = False
        self.b.save_manifest()
        self.assertFails("X3", "values.expected")

    def test_x3_expected_without_an_actual(self):
        k = self.b.find("snapshots", lambda k: self.b.get("snapshots", "role", k) == 1
                        and self.b.get("snapshots", "step", k) != NONE64)
        self.b.set("snapshots", "step", k, 31)                  # a Release: no actual snapshot there
        self.assertFails("X3", "no actual value beside it")

    def test_x3_a_mismatch_must_be_reported_as_one(self):
        k = self.snapshot(self.operand_tile("C", 1, 0), role=1, final=True)
        first = self.b.get("snapshots", "first", k)
        self.b.set("elements", "b", first, self.b.get("elements", "b", first) ^ 0x01)
        self.assertFails("X3", "give 'mismatch'")
        # Recorded honestly, a mismatch is a valid record: the result says so.
        self.b.manifest["result"]["values"] = "mismatch"
        self.b.save_manifest()
        self.assertPasses()

    def test_x3_a_bound_never_hides_an_integer_mismatch(self):
        # result.bound applies to the floating types only: an int32 tile is compared bit for bit.
        k = self.snapshot(self.operand_tile("C", 0, 0), role=1, final=True)
        first = self.b.get("snapshots", "first", k)
        self.b.set("elements", "b", first, self.b.get("elements", "b", first) ^ 0x01)
        self.b.manifest["result"]["bound"] = {"atol": 10.0, "rtol": 0.0}
        self.b.save_manifest()
        self.assertFails("X3", "give 'mismatch'")

    def test_x3_floats_within_the_bound_match(self):
        # Read C's int32 payload as fp32 (small integers are denormals: finite, and tiny), and flip
        # the low bit of one expected element: bit equality fails it, a bound of 1e-30 passes it.
        self.b.manifest["operands"][2]["element"] = "fp32"
        k = self.snapshot(self.operand_tile("C", 0, 0), role=1, final=True)
        first = self.b.get("snapshots", "first", k)
        self.b.set("elements", "b", first, self.b.get("elements", "b", first) ^ 0x01)
        self.b.save_manifest()
        self.assertFails("X3", "give 'mismatch'")
        self.b.manifest["result"]["bound"] = {"atol": 1e-30, "rtol": 0.0}
        self.b.save_manifest()
        self.assertPasses()

    def test_x3_identical_infinities_match_under_a_bound(self):
        # inf - inf is nan: a bound alone would call a bit-identical infinity a mismatch.
        self.b.manifest["operands"][2]["element"] = "fp32"
        self.b.manifest["result"]["bound"] = {"atol": 1e-30, "rtol": 0.0}
        self.b.save_manifest()
        tile = self.operand_tile("C", 0, 0)
        snaps = [k for k in range(self.b.rows("snapshots")) if self.b.get("snapshots", "tile", k) == tile]
        for k in snaps:                                             # every value of the tile: V5 pins them equal
            first = self.b.get("snapshots", "first", k)
            for j, byte in enumerate((0x00, 0x00, 0x80, 0x7F)):     # +inf, little-endian
                self.b.set("elements", "b", first + j, byte)
        self.assertPasses()

    # -- unreadable manifests -------------------------------------------------
    def test_machine_without_banks_is_unreadable(self):
        del self.b.manifest["machine"]["banks"]
        self.b.save_manifest()
        self.assertUnreadable()

    def test_unknown_level_is_unreadable(self):
        self.b.manifest["level"] = "GPU"
        self.b.save_manifest()
        self.assertUnreadable()

    def test_event_kinds_not_the_formats_are_unreadable(self):
        self.b.manifest["event_kinds"] = self.b.manifest["event_kinds"][::-1]
        self.b.save_manifest()
        self.assertUnreadable()

    def test_layers_not_the_formats_are_unreadable(self):
        self.b.manifest["layers"] = ["dram", "l3", "l2", "l1"]
        self.b.save_manifest()
        self.assertUnreadable()

    def test_empty_producer_is_unreadable(self):
        self.b.manifest["producer"] = ""
        self.b.save_manifest()
        self.assertUnreadable()

    def test_unknown_result_values_is_unreadable(self):
        self.b.manifest["result"]["values"] = "probably"
        self.b.save_manifest()
        self.assertUnreadable()

    # -- D3: CI compares the result, never the event sequence -----------------
    def test_result_is_cycles_and_final_values(self):
        code, r = result(self.b.dir)
        self.assertEqual(code, 0)
        self.assertEqual(r["level"], "RTL")
        self.assertEqual(r["cycles"], self.b.manifest["time"]["cycles"])
        self.assertEqual(r["values"], "match")
        self.assertEqual({k: (v["element"], v["tiles"]) for k, v in r["finals"].items()},
                         {"L": ("int8", 4), "B": ("int8", 2), "C": ("int32", 2)})

    def test_result_does_not_depend_on_the_events(self):
        _, before = result(self.b.dir)
        # Renumber every progress event's beat: the event trace changes, the transformation does not.
        for i in range(self.b.rows("events")):
            if self.ev("kind", i) in (KIND["BM_ROW"], KIND["DRAIN_ROW"]):
                self.set_ev("beat", i, 1)
        self.assertPasses()
        _, after = result(self.b.dir)
        self.assertEqual(before, after)

    def test_result_changes_with_a_final_value(self):
        _, before = result(self.b.dir)
        t = self.operand_tile("C", 0, 0)
        for role in (0, 1):                                     # the actual value and its expected one
            k = self.snapshot(t, role=role, final=True)
            first = self.b.get("snapshots", "first", k)
            self.b.set("elements", "b", first, self.b.get("elements", "b", first) ^ 0x01)
        # The Store's snapshot must still equal the final (V5), so change it too.
        for role in (0, 1):
            k = self.b.find("snapshots", lambda k: self.b.get("snapshots", "tile", k) == t
                            and self.b.get("snapshots", "role", k) == role
                            and self.b.get("snapshots", "step", k) != NONE64
                            and self.b.get("steps", "kind", self.b.get("snapshots", "step", k)) == 1)
            first = self.b.get("snapshots", "first", k)
            self.b.set("elements", "b", first, self.b.get("elements", "b", first) ^ 0x01)
        self.assertPasses()
        _, after = result(self.b.dir)
        self.assertNotEqual(before["finals"]["C"]["sha256"], after["finals"]["C"]["sha256"])
        self.assertEqual(before["finals"]["L"], after["finals"]["L"])

    def test_no_result_from_a_broken_record(self):
        self.b.manifest["result"]["cycles"] = 7
        self.b.save_manifest()
        code, _ = result(self.b.dir)
        self.assertEqual(code, 1)


class Decoding(unittest.TestCase):
    """The element types and domains the checker decodes values by (the viewer decodes the same)."""

    def test_element_types(self):
        import struct
        self.assertEqual(dc.decode("int8", bytes([0x80, 0x7F, 0xFF])), [-128, 127, -1])
        self.assertEqual(dc.decode("int32", struct.pack("<2i", -5, 70000)), [-5, 70000])
        self.assertEqual(dc.decode("fp16", struct.pack("<2H", 0x3C00, 0xC000)), [1.0, -2.0])
        self.assertEqual(dc.decode("bf16", struct.pack("<2H", 0x3F80, 0x4049)), [1.0, 3.140625])
        self.assertEqual(dc.decode("fp8_e4m3", bytes([0x38, 0x7E, 0x01])), [1.0, 448.0, 2.0 ** -9])
        self.assertTrue(dc.decode("fp8_e4m3", bytes([0x7F]))[0] != dc.decode("fp8_e4m3", bytes([0x7F]))[0])   # NaN
        self.assertEqual(dc.decode("fp8_e5m2", bytes([0x3C, 0x7C, 0xFC])), [1.0, float("inf"), float("-inf")])

    def test_domains(self):
        lower, upper = {"kind": "lower"}, {"kind": "upper", "diagonal": 1}
        self.assertTrue(dc.in_domain(lower, 3, 3) and dc.in_domain(lower, 3, 0) and not dc.in_domain(lower, 0, 1))
        self.assertTrue(dc.in_domain(upper, 1, 0) and not dc.in_domain(upper, 2, 0))
        band = {"kind": "band", "lower": 1, "upper": 0}
        self.assertEqual([dc.in_domain(band, 2, c) for c in range(4)], [False, True, True, False])
        pad = {"kind": "dense", "pad": {"rows": 6, "cols": 4}}
        self.assertTrue(dc.in_domain(pad, 5, 3) and not dc.in_domain(pad, 6, 0) and not dc.in_domain(pad, 0, 4))


class RtlSixOne(Checks):
    """Version 6.1 (#398): the RTL's acc -> l3 store and its flat, word-addressed L3."""

    @classmethod
    def setUpClass(cls):
        if ADDRESSED is None:
            raise unittest.SkipTest("no 6.1 sample given")
        cls.SOURCE = ADDRESSED

    def store(self):
        return self.event("DRAIN", lambda i: self.ev("dst", i) == LAYER["l3"])

    def has(self, kind, pred):
        return any(self.ev("kind", i) == KIND[kind] and pred(i) for i in range(self.b.rows("events")))

    def test_clean_sample_passes(self):
        self.assertPasses()

    def test_the_sample_exercises_what_is_checked(self):
        m = self.b.manifest
        self.assertEqual((m["version"], m["minor"]), (6, 1))
        self.assertEqual(set(m["geometry"]["l3"]) - {"modelled"}, {"words", "word_bytes"})
        self.assertTrue(self.has("DRAIN", lambda i: self.ev("dst", i) == LAYER["l3"]), "an acc -> l3 DRAIN")
        # No DRAIN into L2: every drain is the store.
        self.assertFalse(self.has("DRAIN", lambda i: self.ev("dst", i) == LAYER["l2"]))
        self.assertFalse(self.has("CREDIT", lambda i: self.ev("src", i) == LAYER["l3"]), "no L3 credit")
        addrs = {self.ev("dst_addr", i) for i in range(self.b.rows("events")) if self.ev("dst", i) == LAYER["l3"]}
        self.assertTrue(any(a >= self.b.manifest["machine"]["l3"] for a in addrs), "word addresses, not slots")

    def test_same_result_as_the_slotted_sample(self):
        _, slotted = result(RTL)
        _, addressed = result(self.b.dir)
        self.assertEqual(slotted["finals"], addressed["finals"])

    def test_g1_mixes_the_two_l3_forms(self):
        self.b.manifest["geometry"]["l3"]["slots"] = 4
        self.b.save_manifest()
        self.assertFails("G1", "mixes the two L3 forms")

    def test_g1_word_addressed_needs_six_one(self):
        self.b.manifest["minor"] = 0
        self.b.save_manifest()
        self.assertFails("G1", "needs version 6.1")

    def test_e1_store_needs_six_one(self):
        self.b.manifest["minor"] = 0
        self.b.save_manifest()
        self.assertFails("E1", "needs version 6.1")

    def test_unknown_minor_is_unreadable(self):
        self.b.manifest["minor"] = 2
        self.b.save_manifest()
        self.assertUnreadable()

    def test_g1_words_must_be_declared(self):
        del self.b.manifest["geometry"]["l3"]["words"]
        self.b.save_manifest()
        self.assertFails("G1", "words")

    def test_e1_word_outside_l3(self):
        self.set_ev("dst_addr", self.store(), self.b.manifest["geometry"]["l3"]["words"])
        self.assertFails("E1", "outside l3")

    def test_e4_no_l3_credit(self):
        c = self.event("CREDIT")
        self.set_ev("src", c, LAYER["l3"])
        self.set_ev("dst", c, LAYER["dram"])
        self.assertFails("E4", "no credits")

    def test_e5_a_residency_keeps_its_word(self):
        mv = self.event("BLOCK_MOVE", lambda i: self.ev("src", i) == LAYER["l3"])
        self.set_ev("src_addr", mv, self.ev("src_addr", mv) + 1)
        self.assertFails("E5", "put its residency at word")

    def test_e5_the_store_stands_for_the_drain_step(self):
        # Drained into L2 instead, it is a DRAIN of a WRITEBACK action, and the DRAIN step goes uncovered.
        st = self.store()
        self.set_ev("dst", st, LAYER["l2"])
        self.set_ev("dst_addr", st, 0)
        self.assertFails("E5", "transaction events")


class LbRecord(Checks):
    """kpu-sim's own L-B record, version 6: no time, every layer declared, fp32 dense values."""

    @classmethod
    def setUpClass(cls):
        cls.SOURCE = LB

    def test_clean_record_passes(self):
        self.assertPasses()

    def test_l_b_declares_its_layers_and_no_time(self):
        m = self.b.manifest
        self.assertEqual((m["version"], m["level"]), (6, "L-B"))
        self.assertIsNone(m["time"])
        self.assertIsNone(m["result"]["cycles"])
        self.assertEqual(m["result"]["values"], "unchecked")
        for layer in ("l2", "acc", "l1"):
            self.assertFalse(m["geometry"][layer]["modelled"], layer)
        self.assertEqual(self.b.rows("events"), 0)

    def test_t1_l_b_states_cycles(self):
        self.b.manifest["result"]["cycles"] = 100
        self.b.save_manifest()
        self.assertFails("T1", "states cycles to result")

    def test_t1_l_b_record_from_a_timed_level_without_time(self):
        self.b.manifest["level"] = "L-T1"
        self.b.save_manifest()
        self.assertFails("T1", "declares no time")

    def test_x3_unchecked_is_the_only_honest_verdict(self):
        self.b.manifest["result"]["values"] = "match"
        self.b.save_manifest()
        self.assertFails("X3", "give 'unchecked'")

    def test_result_has_no_cycles(self):
        code, r = result(self.b.dir)
        self.assertEqual(code, 0)
        self.assertEqual((r["level"], r["cycles"], r["values"]), ("L-B", None, "unchecked"))
        self.assertTrue(all(v["element"] == "fp32" for v in r["finals"].values()))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    RTL = Path(sys.argv.pop(1)).resolve()
    LB = Path(sys.argv.pop(1)).resolve()
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        ADDRESSED = Path(sys.argv.pop(1)).resolve()
    unittest.main()
