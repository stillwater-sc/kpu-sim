#!/usr/bin/env python3
"""
Self-test for kpuarch_to_deployment.py (#396): the vendored generator config converts to the
checked-in deployment byte for byte (so the two stay in step), and what the converter cannot model
is refused, never dropped.

Usage: python3 test_kpuarch_to_deployment.py <generator.full.yaml> <kpu_soc_2x2.json> [<notes.txt>]

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kpuarch_to_deployment as k  # noqa: E402

ARCH = None
DEPLOYMENT = None
NOTES = None


class Convert(unittest.TestCase):
    def setUp(self):
        self.doc = copy.deepcopy(k.load(str(ARCH)))
        self.tile = self.doc["architecture"]["tiles"][0]

    def refused(self, needle):
        with self.assertRaises(k.Refused) as e:
            k.convert(self.doc)
        self.assertIn(needle, str(e.exception))

    def test_the_checked_in_deployment_is_the_generator_config_converted(self):
        dep, _ = k.convert(self.doc)
        self.assertEqual(json.dumps(dep, indent=2) + "\n", Path(DEPLOYMENT).read_text(encoding="utf-8"),
                         "kpu_soc_2x2.json is out of step: rerun tools/deploy/kpuarch_to_deployment.py")

    def test_the_checked_in_notes_are_the_conversion_s(self):
        if NOTES is None:
            self.skipTest("no notes file given")
        _, notes = k.convert(self.doc)
        text = "".join(f"{key}:\n" + "".join(f"  - {x}\n" for x in v) for key, v in notes.items())
        self.assertEqual(text, Path(NOTES).read_text(encoding="utf-8"))

    def test_every_field_is_carried_defaulted_or_named(self):
        _, notes = k.convert(self.doc)
        said = " ".join(x for v in notes.values() for x in v)
        for f in ("array_rows", "dataflow", "l3_kib_per_tile", "memory_type", "energy", "supported_kernels",
                  "checkerboard", "noc", "circuit_class", "l2_slots", "dma.engines"):
            self.assertIn(f, said)

    def test_an_unknown_tile_field_is_refused(self):
        self.tile["sparsity"] = "2:4"
        self.refused("tiles[0].sparsity")

    def test_an_unknown_architecture_field_is_refused(self):
        self.doc["architecture"]["power_domains"] = []
        self.refused("architecture.power_domains")

    def test_two_tile_classes_are_refused(self):
        self.doc["architecture"]["tiles"].append(copy.deepcopy(self.tile))
        self.refused("2 tile classes")

    def test_a_second_mac_mode_is_refused(self):
        self.tile["mac"]["modes"].append(copy.deepcopy(self.tile["mac"]["modes"][0]))
        self.refused("2 modes")

    def test_memory_per_pe_is_refused(self):
        self.tile["local_memory"][0]["per"] = "pe"
        self.refused("per pe")

    def test_all_four_tiles_wait_for_v0_12(self):
        with self.assertRaises(k.Refused) as e:
            k.convert(self.doc, "all")
        self.assertIn("v0.12.0", str(e.exception))

    def test_a_dram_kpu_sim_cannot_host_is_named_not_carried(self):
        dep, notes = k.convert(self.doc)             # one controller, 64 bits: 4 x16 channels
        self.assertNotIn("dram", dep["devices"][0]["memory"])
        self.assertTrue(any("memory.dram is not carried" in x for x in notes["not carried"]))
        self.doc["architecture"]["memory"]["memory_bus_bits"] = 32     # 2 x16 channels: hosted
        self.doc["architecture"]["memory"]["memory_bandwidth_gbps"] = 25.6
        dep, _ = k.convert(self.doc)
        self.assertEqual(dep["devices"][0]["memory"]["dram"]["channels"], 2)

    def test_floating_point_and_int4_formats_are_refused(self):
        for inp, acc in (("fp16", "fp32"), ("bf16", "fp32"), ("int4", "int32")):
            with self.subTest(inp=inp):
                self.setUp()
                self.tile["mac"]["modes"][0]["operand_format"] = inp
                self.tile["mac"]["modes"][0]["accumulate_format"] = acc
                self.refused("whole-byte integer formats")

    def test_two_controllers_get_an_engine_each(self):
        m = self.doc["architecture"]["memory"]
        m["memory_controllers"], m["memory_bus_bits"], m["memory_bandwidth_gbps"] = 2, 64, 51.2
        dep, _ = k.convert(self.doc)
        d = dep["devices"][0]
        self.assertEqual(d["dma"]["engines"], 2)
        self.assertEqual(d["memory"]["dram"]["channels"], 2)      # 4 x16 channels over 2 controllers

    def test_a_duplicate_key_is_refused(self):
        import tempfile
        text = Path(ARCH).read_text(encoding="utf-8").replace("    array_rows: 4\n", "    array_rows: 4\n    array_rows: 8\n", 1)
        self.assertIn("array_rows: 8", text)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dup.full.yaml"
            p.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError) as e:
                k.load(str(p))
            self.assertIn("'array_rows' appears twice", str(e.exception))
            j = Path(d) / "dup.json"
            j.write_text('{"name": "x", "name": "y"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                k.load(str(j))

    def test_a_repeated_memory_level_is_refused(self):
        self.tile["local_memory"].append(copy.deepcopy(self.tile["local_memory"][0]))
        self.refused("two weight_buffer entries")

    def test_channels_that_controllers_cannot_share_are_refused(self):
        m = self.doc["architecture"]["memory"]
        m["memory_type"], m["memory_controllers"] = "ddr5", 3     # 4 x16 channels over 3 controllers
        self.refused("cannot share evenly")

    def test_non_power_of_two_dram_counts_are_refused(self):
        m = self.doc["architecture"]["memory"]
        for bus, controllers, gbps in ((96, 3, 76.8), (48, 1, 38.4)):     # 3 controllers; 3 channels
            with self.subTest(bus=bus, controllers=controllers):
                m["memory_type"], m["memory_bus_bits"], m["memory_controllers"] = "ddr5", bus, controllers
                m["memory_bandwidth_gbps"] = gbps
                self.refused("power-of-two controller and channel counts")

    def test_an_unknown_dram_is_refused(self):
        self.doc["architecture"]["memory"]["memory_type"] = "sdr"
        self.refused("memory_type 'sdr'")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    ARCH = Path(sys.argv.pop(1)).resolve()
    DEPLOYMENT = Path(sys.argv.pop(1)).resolve()
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        NOTES = Path(sys.argv.pop(1)).resolve()
    unittest.main()
