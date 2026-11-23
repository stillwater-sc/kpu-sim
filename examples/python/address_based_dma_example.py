"""
Address-based DMA through the native `stillwater_kpu` binding.

Every memory in the KPU sits in one unified address space. A DMA transfer
names a source and a destination address, and the address decoder routes it.
This example moves data Host -> External -> L3 with the per-level calls and
with the any-to-any `start_dma_transfer`, then reads every destination back
and checks it. It exits non-zero if any check fails.

Run it with the module on PYTHONPATH, e.g.
    PYTHONPATH=build/src/tools/bindings/python/Release \
        python3 examples/python/address_based_dma_example.py
"""

import os
import sys

# Fall back to the default build tree when PYTHONPATH doesn't provide the module.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'build', 'src', 'tools', 'bindings', 'python', 'Release'))

import stillwater_kpu as kpu

FLOAT_BYTES = 4
failures = 0


def check(label, got, expected):
    global failures
    if got == expected:
        print(f"   PASS {label}")
    else:
        failures += 1
        mismatch = next(i for i, (g, e) in enumerate(zip(got, expected)) if g != e) \
            if len(got) == len(expected) else None
        detail = f"first mismatch at [{mismatch}]: {got[mismatch]} != {expected[mismatch]}" \
            if mismatch is not None else f"length {len(got)} != {len(expected)}"
        print(f"   FAIL {label}: {detail}")


def main():
    print("=" * 60)
    print("Address-Based DMA API Example")
    print("=" * 60)

    config = kpu.SimulatorConfig()

    # Host memory (system memory, outside the KPU)
    config.host_memory_region_count = 1
    config.host_memory_region_capacity_mb = 64
    config.host_memory_bandwidth_gbps = 50

    # External memory (the KPU's own HBM/GDDR banks)
    config.memory_bank_count = 2
    config.memory_bank_capacity_mb = 64
    config.memory_bandwidth_gbps = 100

    # On-chip buffers
    config.l3_tile_count = 2
    config.l3_tile_capacity_kb = 512
    config.l2_bank_count = 4
    config.l2_bank_capacity_kb = 128
    config.l1_buffer_count = 2
    config.l1_buffer_capacity_kb = 64

    # Compute and data movement engines
    config.compute_tile_count = 2
    config.dma_engine_count = 4
    config.block_mover_count = 2
    config.streamer_count = 4

    sim = kpu.KPUSimulator(config)

    print("\nUnified address space:")
    print(f"  Host[0]     0x{sim.get_host_memory_region_base(0):08x}")
    print(f"  External[0] 0x{sim.get_external_bank_base(0):08x}")
    print(f"  External[1] 0x{sim.get_external_bank_base(1):08x}")
    print(f"  L3[0]       0x{sim.get_l3_tile_base(0):08x}")
    print(f"  L3[1]       0x{sim.get_l3_tile_base(1):08x}")

    # === 1: Host -> External -> L3, one call per level ===
    print("\n1. Host -> External -> L3 (per-level calls)")
    data = [float(i) for i in range(256)]
    size = len(data) * FLOAT_BYTES
    sim.write_host_memory(0, 0, data)

    done = []
    sim.dma_host_to_external(0, sim.get_host_memory_region_base(0),
                             sim.get_external_bank_base(0), size,
                             lambda: done.append("host->external"))
    sim.run_until_idle()
    check("external[0] holds the host data", sim.read_memory_bank(0, 0, len(data)), data)

    sim.dma_external_to_l3(0, sim.get_external_bank_base(0),
                           sim.get_l3_tile_base(0), size,
                           lambda: done.append("external->l3"))
    sim.run_until_idle()
    check("L3[0] holds the host data", sim.read_l3_tile(0, 0, len(data)), data)
    check("both completion callbacks ran", done, ["host->external", "external->l3"])

    # === 2: any-to-any, routed by address ===
    print("\n2. External[1] -> L3[1] (start_dma_transfer, routed by address)")
    data2 = [float(2 * i) for i in range(128)]
    sim.write_memory_bank(1, 0, data2)
    sim.start_dma_transfer(1, sim.get_external_bank_base(1),
                           sim.get_l3_tile_base(1), len(data2) * FLOAT_BYTES)
    sim.run_until_idle()
    check("L3[1] holds external[1]'s data", sim.read_l3_tile(1, 0, len(data2)), data2)

    print(f"\nTotal cycles: {sim.get_current_cycle()}")
    print("=" * 60)
    print("All checks passed" if failures == 0 else f"{failures} check(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
