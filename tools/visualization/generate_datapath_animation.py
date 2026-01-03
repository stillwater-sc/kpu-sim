#!/usr/bin/env python3
"""
Generate full data path animation traces for KPU visualization.

This script generates trace events that capture the complete data flow:
  External Memory (DDR/HBM) -> Memory Controller -> DMA Engines -> L3 Mesh Edge

Usage:
    python generate_datapath_animation.py --output trace.json
    python generate_datapath_animation.py --output trace.csv --format csv
    python generate_datapath_animation.py --mesh 4x4 --tiles 4 --output trace.json

The output can be loaded into tile_flow_animation_v2.html for visualization.
"""

import argparse
import json
import csv
import random
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from enum import Enum


class EventType(Enum):
    # Memory Controller Events
    MC_BANK_OPEN = "MC_BANK_OPEN"
    MC_BANK_CLOSE = "MC_BANK_CLOSE"
    MC_BANK_CONFLICT = "MC_BANK_CONFLICT"
    MC_PAGE_HIT = "MC_PAGE_HIT"
    MC_READ_COMPLETE = "MC_READ_COMPLETE"
    MC_REFRESH = "MC_REFRESH"

    # DMA Events
    DMA_REQUEST = "DMA_REQUEST"
    DMA_TRANSFER_START = "DMA_TRANSFER_START"
    DMA_TRANSFER_COMPLETE = "DMA_TRANSFER_COMPLETE"
    DMA_STALL = "DMA_STALL"
    DMA_IDLE = "DMA_IDLE"

    # L3 Events
    L3_SEND_START = "L3_SEND_START"
    L3_RECEIVE = "L3_RECEIVE"
    L3_TO_L2_START = "L3_TO_L2_START"
    BARRIER_START = "BARRIER_START"
    BARRIER_COMPLETE = "BARRIER_COMPLETE"


@dataclass
class TraceEvent:
    cycle: int
    event_type: str

    # Memory controller fields
    bank_id: Optional[int] = None
    row: Optional[int] = None

    # DMA fields
    dma_channel: Optional[int] = None

    # L3/Tile fields
    l3_id: Optional[int] = None
    src_l3: Optional[int] = None
    dst_l3: Optional[int] = None

    # Tensor fields
    tensor: Optional[int] = None  # 0=A, 1=B, 2=C
    m_tile: Optional[int] = None
    n_tile: Optional[int] = None
    k_tile: Optional[int] = None

    # Transfer fields
    bytes: Optional[int] = None

    def to_dict(self):
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class DataPathSimulator:
    """Simulates the data path from external memory to L3 mesh."""

    def __init__(self, mesh_rows: int = 4, mesh_cols: int = 4,
                 num_banks: int = 8, num_dma_channels: int = 8,
                 tile_size_bytes: int = 4096):
        self.mesh_rows = mesh_rows
        self.mesh_cols = mesh_cols
        self.num_banks = num_banks
        self.num_dma_channels = num_dma_channels
        self.tile_size = tile_size_bytes

        # Timing parameters (in cycles)
        self.bank_open_latency = 20
        self.bank_cas_latency = 10
        self.bank_precharge_latency = 15
        self.dma_setup_latency = 5
        self.dma_transfer_cycles_per_kb = 16
        self.l3_hop_latency = 5

        # State
        self.events: List[TraceEvent] = []
        self.bank_open_rows = [None] * num_banks
        self.dma_busy_until = [0] * num_dma_channels

        # Statistics
        self.page_hits = 0
        self.page_conflicts = 0

    def add_event(self, event: TraceEvent):
        """Add an event to the trace."""
        self.events.append(event)

    def get_bank_for_address(self, tensor: int, m: int, n: int, k: int) -> int:
        """Determine which memory bank holds this tile."""
        # Simple interleaving: hash tile coordinates to bank
        return (tensor * 17 + m * 13 + n * 7 + k * 3) % self.num_banks

    def get_row_for_address(self, tensor: int, m: int, n: int, k: int) -> int:
        """Get the DRAM row for this tile."""
        # Each tile is in a different row (simplified model)
        return tensor * 1000 + m * 100 + n * 10 + k

    def simulate_memory_access(self, cycle: int, bank_id: int, row: int) -> int:
        """Simulate memory access, return completion cycle."""
        completion_cycle = cycle

        if self.bank_open_rows[bank_id] == row:
            # Page hit - just CAS latency
            self.page_hits += 1
            self.add_event(TraceEvent(
                cycle=cycle,
                event_type=EventType.MC_PAGE_HIT.value,
                bank_id=bank_id,
                row=row
            ))
            completion_cycle = cycle + self.bank_cas_latency

        elif self.bank_open_rows[bank_id] is None:
            # Bank idle - open row
            self.add_event(TraceEvent(
                cycle=cycle,
                event_type=EventType.MC_BANK_OPEN.value,
                bank_id=bank_id,
                row=row
            ))
            self.bank_open_rows[bank_id] = row
            completion_cycle = cycle + self.bank_open_latency + self.bank_cas_latency

        else:
            # Page conflict - close and reopen
            self.page_conflicts += 1
            self.add_event(TraceEvent(
                cycle=cycle,
                event_type=EventType.MC_BANK_CONFLICT.value,
                bank_id=bank_id,
                row=self.bank_open_rows[bank_id]
            ))

            # Close old row
            close_cycle = cycle + self.bank_precharge_latency

            # Open new row
            self.add_event(TraceEvent(
                cycle=close_cycle,
                event_type=EventType.MC_BANK_OPEN.value,
                bank_id=bank_id,
                row=row
            ))
            self.bank_open_rows[bank_id] = row
            completion_cycle = close_cycle + self.bank_open_latency + self.bank_cas_latency

        return completion_cycle

    def simulate_dma_transfer(self, cycle: int, dma_channel: int,
                              l3_id: int, tensor: int, m: int, n: int, k: int) -> int:
        """Simulate DMA transfer from memory to L3, return completion cycle."""

        # Wait for DMA channel to be free
        start_cycle = max(cycle, self.dma_busy_until[dma_channel])

        # Get memory bank and row
        bank_id = self.get_bank_for_address(tensor, m, n, k)
        row = self.get_row_for_address(tensor, m, n, k)

        # Memory access
        mem_complete = self.simulate_memory_access(start_cycle, bank_id, row)

        # DMA request
        self.add_event(TraceEvent(
            cycle=start_cycle + self.dma_setup_latency,
            event_type=EventType.DMA_REQUEST.value,
            dma_channel=dma_channel,
            l3_id=l3_id,
            tensor=tensor,
            m_tile=m,
            n_tile=n,
            k_tile=k
        ))

        # Transfer start (after memory read completes)
        transfer_start = mem_complete
        self.add_event(TraceEvent(
            cycle=transfer_start,
            event_type=EventType.DMA_TRANSFER_START.value,
            dma_channel=dma_channel,
            l3_id=l3_id
        ))

        # Random stall for realism (5% chance)
        stall_cycles = 0
        if random.random() < 0.05:
            stall_cycles = random.randint(10, 30)
            self.add_event(TraceEvent(
                cycle=transfer_start + 5,
                event_type=EventType.DMA_STALL.value,
                dma_channel=dma_channel
            ))

        # Transfer complete
        transfer_cycles = (self.tile_size // 1024) * self.dma_transfer_cycles_per_kb
        complete_cycle = transfer_start + transfer_cycles + stall_cycles

        self.add_event(TraceEvent(
            cycle=complete_cycle,
            event_type=EventType.DMA_TRANSFER_COMPLETE.value,
            dma_channel=dma_channel,
            l3_id=l3_id,
            tensor=tensor,
            m_tile=m,
            n_tile=n,
            k_tile=k,
            bytes=self.tile_size
        ))

        # Close bank after transfer
        self.add_event(TraceEvent(
            cycle=complete_cycle + 5,
            event_type=EventType.MC_READ_COMPLETE.value,
            bank_id=bank_id
        ))

        self.dma_busy_until[dma_channel] = complete_cycle
        return complete_cycle

    def simulate_l3_transfer(self, cycle: int, src_l3: int, dst_l3: int,
                             tensor: int, m: int, n: int, k: int) -> int:
        """Simulate L3-to-L3 transfer (hop-and-forward)."""
        self.add_event(TraceEvent(
            cycle=cycle,
            event_type=EventType.L3_SEND_START.value,
            l3_id=src_l3,
            src_l3=src_l3,
            dst_l3=dst_l3,
            tensor=tensor,
            m_tile=m,
            n_tile=n,
            k_tile=k,
            bytes=self.tile_size
        ))

        # Receive event
        receive_cycle = cycle + self.l3_hop_latency
        self.add_event(TraceEvent(
            cycle=receive_cycle,
            event_type=EventType.L3_RECEIVE.value,
            l3_id=dst_l3,
            src_l3=src_l3,
            dst_l3=dst_l3,
            tensor=tensor,
            m_tile=m,
            n_tile=n,
            k_tile=k
        ))

        return receive_cycle

    def simulate_matmul_load(self, k_tiles: int = 4):
        """Simulate loading tiles for matrix multiplication."""
        cycle = 0

        for k in range(k_tiles):
            # Load A tiles via DMA channels 0-3 to left column
            for row in range(self.mesh_rows):
                dma_channel = row % 4
                l3_id = row * self.mesh_cols  # Left column

                complete = self.simulate_dma_transfer(
                    cycle, dma_channel, l3_id,
                    tensor=0, m=row, n=0, k=k
                )

            # Load B tiles via DMA channels 4-7 to top row
            for col in range(self.mesh_cols):
                dma_channel = 4 + (col % 4)
                l3_id = col  # Top row

                complete = self.simulate_dma_transfer(
                    cycle + 10, dma_channel, l3_id,
                    tensor=1, m=0, n=col, k=k
                )

            # Wait for all DMA to complete
            cycle = max(self.dma_busy_until) + 20

            # Hop-and-forward A tiles (east)
            for row in range(self.mesh_rows):
                for col in range(self.mesh_cols - 1):
                    src = row * self.mesh_cols + col
                    dst = row * self.mesh_cols + col + 1
                    self.simulate_l3_transfer(
                        cycle + col * self.l3_hop_latency,
                        src, dst, tensor=0, m=row, n=0, k=k
                    )

            # Hop-and-forward B tiles (south)
            for col in range(self.mesh_cols):
                for row in range(self.mesh_rows - 1):
                    src = row * self.mesh_cols + col
                    dst = (row + 1) * self.mesh_cols + col
                    self.simulate_l3_transfer(
                        cycle + 5 + row * self.l3_hop_latency,
                        src, dst, tensor=1, m=0, n=col, k=k
                    )

            # Barrier after each k-step
            barrier_cycle = cycle + (self.mesh_rows - 1) * self.l3_hop_latency + 20
            for l3_id in range(self.mesh_rows * self.mesh_cols):
                self.add_event(TraceEvent(
                    cycle=barrier_cycle,
                    event_type=EventType.BARRIER_START.value,
                    l3_id=l3_id,
                    tensor=2, m_tile=0, n_tile=0, k_tile=k
                ))
                self.add_event(TraceEvent(
                    cycle=barrier_cycle + 30,
                    event_type=EventType.BARRIER_COMPLETE.value,
                    l3_id=l3_id,
                    tensor=2, m_tile=0, n_tile=0, k_tile=k
                ))

            cycle = barrier_cycle + 50

        # Final L3 to L2 pushes (output tiles)
        for l3_id in range(self.mesh_rows * self.mesh_cols):
            row = l3_id // self.mesh_cols
            col = l3_id % self.mesh_cols
            self.add_event(TraceEvent(
                cycle=cycle + l3_id * 5,
                event_type=EventType.L3_TO_L2_START.value,
                l3_id=l3_id,
                tensor=2,
                m_tile=row,
                n_tile=col,
                k_tile=0,
                bytes=self.tile_size
            ))

    def get_sorted_events(self) -> List[TraceEvent]:
        """Return events sorted by cycle."""
        return sorted(self.events, key=lambda e: e.cycle)

    def export_json(self, filename: str):
        """Export events to JSON format."""
        events = [e.to_dict() for e in self.get_sorted_events()]
        with open(filename, 'w') as f:
            json.dump({
                'metadata': {
                    'mesh_rows': self.mesh_rows,
                    'mesh_cols': self.mesh_cols,
                    'num_banks': self.num_banks,
                    'num_dma_channels': self.num_dma_channels,
                    'tile_size_bytes': self.tile_size,
                    'page_hits': self.page_hits,
                    'page_conflicts': self.page_conflicts,
                    'total_events': len(events)
                },
                'events': events
            }, f, indent=2)
        print(f"Exported {len(events)} events to {filename}")
        print(f"  Page hits: {self.page_hits}, Conflicts: {self.page_conflicts}")

    def export_csv(self, filename: str):
        """Export events to CSV format."""
        events = self.get_sorted_events()

        # Determine all fields
        all_fields = set()
        for e in events:
            all_fields.update(e.to_dict().keys())
        fields = sorted(all_fields)

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for e in events:
                writer.writerow(e.to_dict())

        print(f"Exported {len(events)} events to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate data path animation traces for KPU visualization'
    )
    parser.add_argument('-o', '--output', required=True,
                        help='Output file (e.g., trace.json or trace.csv)')
    parser.add_argument('--format', choices=['json', 'csv'], default=None,
                        help='Output format (auto-detected from extension if not specified)')
    parser.add_argument('--mesh', default='4x4',
                        help='Mesh dimensions (e.g., 4x4, 8x8)')
    parser.add_argument('--tiles', type=int, default=4,
                        help='Number of K-dimension tiles to simulate')
    parser.add_argument('--banks', type=int, default=8,
                        help='Number of memory banks')
    parser.add_argument('--dma-channels', type=int, default=8,
                        help='Number of DMA channels')
    parser.add_argument('--tile-size', type=int, default=4096,
                        help='Tile size in bytes')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Parse mesh dimensions
    mesh_parts = args.mesh.split('x')
    mesh_rows = int(mesh_parts[0])
    mesh_cols = int(mesh_parts[1]) if len(mesh_parts) > 1 else mesh_rows

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Determine output format
    output_format = args.format
    if output_format is None:
        if args.output.endswith('.json'):
            output_format = 'json'
        elif args.output.endswith('.csv'):
            output_format = 'csv'
        else:
            output_format = 'json'

    # Create simulator and generate trace
    sim = DataPathSimulator(
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        num_banks=args.banks,
        num_dma_channels=args.dma_channels,
        tile_size_bytes=args.tile_size
    )

    print(f"Simulating {mesh_rows}x{mesh_cols} mesh, {args.tiles} K-tiles...")
    sim.simulate_matmul_load(k_tiles=args.tiles)

    # Export
    if output_format == 'json':
        sim.export_json(args.output)
    else:
        sim.export_csv(args.output)


if __name__ == '__main__':
    main()
