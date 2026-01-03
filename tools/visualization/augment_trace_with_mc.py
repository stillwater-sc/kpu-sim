#!/usr/bin/env python3
"""
Augment a tile flow trace with memory controller and DMA events.

This script takes an existing tile flow trace (from C++ simulation) and adds
realistic memory controller and DMA events to enable full data path visualization.

The augmented trace can be loaded into tile_flow_animation_v2.html.

Usage:
    python augment_trace_with_mc.py tile_flow_trace.json -o full_datapath_trace.json
    python augment_trace_with_mc.py tile_flow_trace.json --banks 8 --dma-channels 8
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import defaultdict


@dataclass
class Event:
    cycle: int
    event_type: str
    l3_id: int = 0
    tensor: int = 0
    m_tile: int = 0
    n_tile: int = 0
    k_tile: int = 0
    src_l3: int = 0
    dst_l3: int = 0
    bytes: int = 0
    bank_id: int = 0
    row: int = 0
    dma_channel: int = 0
    address: int = 0


@dataclass
class DRAMTiming:
    """DRAM timing parameters in cycles"""
    tRCD: int = 14     # Row-to-Column Delay
    tRP: int = 14      # Row Precharge
    tCL: int = 14      # CAS Latency
    tBurst: int = 4    # Burst transfer


class MemoryControllerSimulator:
    """Simulate memory controller events based on DMA requests"""

    def __init__(self, num_banks: int = 8, timing: DRAMTiming = None):
        self.num_banks = num_banks
        self.timing = timing or DRAMTiming()
        self.bank_open_rows: Dict[int, Optional[int]] = {i: None for i in range(num_banks)}
        self.bank_busy_until: Dict[int, int] = {i: 0 for i in range(num_banks)}
        self.page_hits = 0
        self.page_conflicts = 0
        self.page_misses = 0

    def get_bank_for_tile(self, tensor: int, m: int, n: int, k: int) -> int:
        """Determine which bank holds this tile (interleaved mapping)"""
        return (tensor * 17 + m * 13 + n * 7 + k * 3) % self.num_banks

    def get_row_for_tile(self, tensor: int, m: int, n: int, k: int) -> int:
        """Get DRAM row for this tile"""
        return tensor * 1000 + m * 100 + n * 10 + k

    def simulate_access(self, cycle: int, bank_id: int, row: int) -> tuple:
        """
        Simulate a memory access.
        Returns (completion_cycle, events_list)
        """
        events = []
        current_cycle = max(cycle, self.bank_busy_until[bank_id])

        if self.bank_open_rows[bank_id] == row:
            # Page hit
            self.page_hits += 1
            events.append(Event(
                cycle=current_cycle,
                event_type="MC_PAGE_HIT",
                bank_id=bank_id,
                row=row
            ))
            completion = current_cycle + self.timing.tCL + self.timing.tBurst

        elif self.bank_open_rows[bank_id] is None:
            # Page miss (bank idle)
            self.page_misses += 1
            events.append(Event(
                cycle=current_cycle,
                event_type="MC_BANK_ACTIVATE",
                bank_id=bank_id,
                row=row
            ))
            self.bank_open_rows[bank_id] = row
            completion = current_cycle + self.timing.tRCD + self.timing.tCL + self.timing.tBurst

        else:
            # Page conflict
            self.page_conflicts += 1
            events.append(Event(
                cycle=current_cycle,
                event_type="MC_PAGE_CONFLICT",
                bank_id=bank_id,
                row=self.bank_open_rows[bank_id]
            ))
            # Precharge
            precharge_done = current_cycle + self.timing.tRP
            # Activate new row
            events.append(Event(
                cycle=precharge_done,
                event_type="MC_BANK_ACTIVATE",
                bank_id=bank_id,
                row=row
            ))
            self.bank_open_rows[bank_id] = row
            completion = precharge_done + self.timing.tRCD + self.timing.tCL + self.timing.tBurst

        # Read complete event
        events.append(Event(
            cycle=completion,
            event_type="MC_READ_COMPLETE",
            bank_id=bank_id,
            row=row
        ))

        self.bank_busy_until[bank_id] = completion
        return completion, events


class DMASimulator:
    """Simulate DMA engine events"""

    def __init__(self, num_channels: int = 8, bandwidth_bytes_per_cycle: int = 64):
        self.num_channels = num_channels
        self.bandwidth = bandwidth_bytes_per_cycle
        self.channel_busy_until: Dict[int, int] = {i: 0 for i in range(num_channels)}

    def get_channel_for_tile(self, l3_id: int, tensor: int, mesh_rows: int = 4) -> int:
        """
        Assign DMA channel based on tile location.
        A tiles use channels 0..mesh_rows-1 (feeding column 0)
        B tiles use channels mesh_rows..2*mesh_rows-1 (feeding row 0)
        """
        if tensor == 0:  # A tiles
            return l3_id // 4 if l3_id // 4 < 4 else 0
        else:  # B tiles
            return 4 + (l3_id % 4)

    def simulate_transfer(self, cycle: int, channel: int, l3_id: int,
                         tensor: int, m: int, n: int, k: int,
                         tile_bytes: int, mc_completion: int) -> tuple:
        """
        Simulate a DMA transfer.
        Returns (completion_cycle, events_list)
        """
        events = []

        # DMA request (after memory controller starts)
        start_cycle = max(cycle, self.channel_busy_until[channel])
        events.append(Event(
            cycle=start_cycle,
            event_type="DMA_REQUEST",
            dma_channel=channel,
            l3_id=l3_id,
            tensor=tensor,
            m_tile=m,
            n_tile=n,
            k_tile=k,
            bytes=tile_bytes
        ))

        # Wait for memory controller
        transfer_start = max(start_cycle + 5, mc_completion)

        # DMA transfer start
        events.append(Event(
            cycle=transfer_start,
            event_type="DMA_TRANSFER_START",
            dma_channel=channel,
            l3_id=l3_id
        ))

        # Transfer time based on bandwidth
        transfer_cycles = tile_bytes // self.bandwidth
        if transfer_cycles == 0:
            transfer_cycles = 1

        completion = transfer_start + transfer_cycles

        # DMA transfer complete
        events.append(Event(
            cycle=completion,
            event_type="DMA_TRANSFER_COMPLETE",
            dma_channel=channel,
            l3_id=l3_id,
            tensor=tensor,
            m_tile=m,
            n_tile=n,
            k_tile=k,
            bytes=tile_bytes
        ))

        self.channel_busy_until[channel] = completion
        return completion, events


def augment_trace(input_trace: dict, num_banks: int = 8,
                  num_dma_channels: int = 8, mesh_rows: int = 4) -> dict:
    """
    Augment an existing trace with memory controller and DMA events.

    Strategy:
    1. Find the first L3_SEND_START for each unique tile
    2. Generate DMA load events that precede the L3 transfers
    3. Generate memory controller events for each DMA load
    """

    mc = MemoryControllerSimulator(num_banks)
    dma = DMASimulator(num_dma_channels)

    events = input_trace.get("events", [])
    metadata = input_trace.get("metadata", {})

    # Identify unique tiles that need DMA loads
    # Group by (tensor, m, n, k) and find earliest L3_SEND_START
    tile_first_send: Dict[tuple, int] = {}
    tile_l3_id: Dict[tuple, int] = {}
    tile_bytes: Dict[tuple, int] = {}

    for e in events:
        if e.get("event_type") in ["L3_SEND_START", "L3_TO_L2_START"]:
            key = (e.get("tensor", 0), e.get("m_tile", 0),
                   e.get("n_tile", 0), e.get("k_tile", 0))
            cycle = e.get("cycle", 0)
            if key not in tile_first_send or cycle < tile_first_send[key]:
                tile_first_send[key] = cycle
                # Use src_l3 for L3_SEND_START, l3_id for L3_TO_L2_START
                tile_l3_id[key] = e.get("src_l3", e.get("l3_id", 0))
                tile_bytes[key] = e.get("bytes", 1024)

    # Generate DMA and MC events for each tile
    new_events = []

    for (tensor, m, n, k), first_cycle in sorted(tile_first_send.items(),
                                                   key=lambda x: x[1]):
        l3_id = tile_l3_id[(tensor, m, n, k)]
        bytes_size = tile_bytes[(tensor, m, n, k)]

        # Calculate when DMA should start (before first L3 send)
        # DMA needs ~30-50 cycles typically
        dma_start = max(0, first_cycle - 50)

        # Memory controller access
        bank_id = mc.get_bank_for_tile(tensor, m, n, k)
        row = mc.get_row_for_tile(tensor, m, n, k)
        mc_completion, mc_events = mc.simulate_access(dma_start, bank_id, row)

        for e in mc_events:
            e.tensor = tensor
            e.m_tile = m
            e.n_tile = n
            e.k_tile = k
        new_events.extend(mc_events)

        # DMA transfer
        channel = dma.get_channel_for_tile(l3_id, tensor, mesh_rows)
        _, dma_events = dma.simulate_transfer(
            dma_start, channel, l3_id, tensor, m, n, k, bytes_size, mc_completion
        )
        new_events.extend(dma_events)

    # Convert new events to dicts
    new_event_dicts = [asdict(e) for e in new_events]

    # Merge with original events
    all_events = new_event_dicts + events

    # Sort by cycle
    all_events.sort(key=lambda e: e.get("cycle", 0))

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata["num_events"] = len(all_events)
    new_metadata["mc_page_hits"] = mc.page_hits
    new_metadata["mc_page_conflicts"] = mc.page_conflicts
    new_metadata["mc_page_misses"] = mc.page_misses
    new_metadata["num_banks"] = num_banks
    new_metadata["num_dma_channels"] = num_dma_channels

    if all_events:
        new_metadata["start_cycle"] = all_events[0].get("cycle", 0)
        new_metadata["end_cycle"] = all_events[-1].get("cycle", 0)

    return {
        "metadata": new_metadata,
        "events": all_events
    }


def main():
    parser = argparse.ArgumentParser(
        description="Augment tile flow trace with memory controller and DMA events"
    )
    parser.add_argument("input", help="Input trace JSON file")
    parser.add_argument("-o", "--output", default="full_datapath_trace.json",
                        help="Output trace JSON file")
    parser.add_argument("--banks", type=int, default=8,
                        help="Number of memory banks")
    parser.add_argument("--dma-channels", type=int, default=8,
                        help="Number of DMA channels")
    parser.add_argument("--mesh-rows", type=int, default=4,
                        help="Mesh rows (for DMA channel assignment)")

    args = parser.parse_args()

    # Load input trace
    with open(args.input, 'r') as f:
        input_trace = json.load(f)

    print(f"Input trace: {len(input_trace.get('events', []))} events")

    # Augment trace
    output_trace = augment_trace(
        input_trace,
        num_banks=args.banks,
        num_dma_channels=args.dma_channels,
        mesh_rows=args.mesh_rows
    )

    # Write output
    with open(args.output, 'w') as f:
        json.dump(output_trace, f, indent=2)

    meta = output_trace["metadata"]
    print(f"Output trace: {meta['num_events']} events")
    print(f"  Cycles: {meta['start_cycle']} - {meta['end_cycle']}")
    print(f"  MC page hits: {meta['mc_page_hits']}")
    print(f"  MC page conflicts: {meta['mc_page_conflicts']}")
    print(f"  MC page misses: {meta['mc_page_misses']}")
    print(f"\nSaved to: {args.output}")
    print(f"\nVisualize with:")
    print(f"  Open tools/visualization/tile_flow_animation_v2.html in a browser")
    print(f"  Load {args.output} using the 'Load Trace File' button")


if __name__ == "__main__":
    main()
