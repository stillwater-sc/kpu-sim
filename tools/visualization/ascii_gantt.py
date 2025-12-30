#!/usr/bin/env python3
"""
ASCII Gantt Chart Visualization

Generates a terminal-based Gantt chart showing resource utilization over time.
Provides instant visual feedback without requiring a browser.

Usage:
    python3 ascii_gantt.py tile_flow_trace.csv
    python3 ascii_gantt.py /tmp/noc_trace.csv --width 120
    python3 ascii_gantt.py tile_flow_trace.csv --resources l3 --period 0-10000
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Unicode block characters for visualization
BLOCKS = {
    'full': '\u2588',      # █ Full block
    'high': '\u2593',      # ▓ Dark shade
    'medium': '\u2592',    # ▒ Medium shade
    'low': '\u2591',       # ░ Light shade
    'empty': ' ',          # Space
}

# ANSI colors
COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'reset': '\033[0m',
    'bold': '\033[1m',
}

# Event type colors
EVENT_COLORS = {
    'L3_SEND': 'green',
    'L3_RECEIVE': 'green',
    'L3_TO_L2': 'blue',
    'BARRIER': 'yellow',
    'DMA': 'magenta',
    'COMPUTE': 'red',
    'FLIT': 'cyan',
}


@dataclass
class TimeSpan:
    """A time interval of activity."""
    start: int
    end: int
    event_type: str
    label: str = ""


@dataclass
class ResourceActivity:
    """Activity for a single resource."""
    resource_id: str
    spans: List[TimeSpan]

    def utilization(self, start: int, end: int) -> float:
        """Calculate utilization in a time window, merging overlapping spans."""
        if end <= start:
            return 0.0

        # Collect all intervals that intersect with [start, end]
        intervals = []
        for span in self.spans:
            s = max(span.start, start)
            e = min(span.end, end)
            if e > s:
                intervals.append((s, e))

        if not intervals:
            return 0.0

        # Merge overlapping intervals
        intervals.sort()
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:  # Overlapping or adjacent
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Sum merged intervals
        busy = sum(e - s for s, e in merged)
        return min(1.0, busy / (end - start))  # Cap at 100%


def load_tile_flow_trace(filename: str) -> Tuple[Dict[str, ResourceActivity], int, int]:
    """Load tile flow trace CSV and extract activity spans."""
    resources: Dict[str, List[TimeSpan]] = defaultdict(list)
    min_cycle = float('inf')
    max_cycle = 0

    # Track open events (start without end)
    open_events: Dict[Tuple[str, str], int] = {}  # (resource, event_key) -> start_cycle

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cycle = int(row['cycle'])
            l3_id = int(row['l3_id'])
            event_type = row['event_type']
            tensor = int(row.get('tensor', 0))
            m_tile = int(row.get('m_tile', 0))
            n_tile = int(row.get('n_tile', 0))
            k_tile = int(row.get('k_tile', 0))
            src_l3 = int(row.get('src_l3', 0))
            dst_l3 = int(row.get('dst_l3', 0))
            bytes_val = int(row.get('bytes', 0))

            min_cycle = min(min_cycle, cycle)
            max_cycle = max(max_cycle, cycle)

            # Create resource ID
            resource_id = f"L3[{l3_id}]"

            # Create tile label
            tensor_names = {0: 'A', 1: 'B', 2: 'C', 3: 'Bias'}
            tile_label = f"{tensor_names.get(tensor, '?')}[{m_tile},{n_tile}]"

            # Handle paired events
            if event_type.endswith('_START'):
                base_type = event_type[:-6]  # Remove _START
                event_key = f"{base_type}:{tile_label}"
                open_events[(resource_id, event_key)] = cycle

            elif event_type.endswith('_COMPLETE'):
                base_type = event_type[:-9]  # Remove _COMPLETE
                event_key = f"{base_type}:{tile_label}"
                key = (resource_id, event_key)
                if key in open_events:
                    start = open_events.pop(key)
                    # Determine event category
                    if 'L3_SEND' in base_type or 'L3_TO_L2' in base_type:
                        cat = 'L3_SEND'
                    elif 'DMA' in base_type:
                        cat = 'DMA'
                    elif 'BARRIER' in base_type:
                        cat = 'BARRIER'
                    else:
                        cat = 'OTHER'
                    resources[resource_id].append(TimeSpan(start, cycle, cat, tile_label))

            elif event_type == 'L3_RECEIVE':
                # Instant event - show small span
                duration = max(1, bytes_val // 64) if bytes_val > 0 else 100
                resources[resource_id].append(TimeSpan(cycle, cycle + duration, 'L3_RECEIVE', tile_label))

            elif event_type == 'BARRIER_START':
                # Start barrier span
                open_events[(resource_id, 'BARRIER')] = cycle

            elif event_type == 'BARRIER_COMPLETE':
                key = (resource_id, 'BARRIER')
                if key in open_events:
                    start = open_events.pop(key)
                    resources[resource_id].append(TimeSpan(start, cycle, 'BARRIER', 'barrier'))

    # Convert to ResourceActivity objects
    result = {rid: ResourceActivity(rid, spans) for rid, spans in resources.items()}

    return result, int(min_cycle) if min_cycle != float('inf') else 0, int(max_cycle)


def load_noc_trace(filename: str) -> Tuple[Dict[str, ResourceActivity], int, int]:
    """Load NoC trace CSV and extract activity spans."""
    resources: Dict[str, List[TimeSpan]] = defaultdict(list)
    min_cycle = float('inf')
    max_cycle = 0

    # Event types: 0=INJECT, 1=HOP, 2=EJECT, 6=FLIT_SEND, 7=FLIT_ARRIVE
    EVENT_INJECT = 0
    EVENT_HOP = 1
    EVENT_EJECT = 2
    EVENT_FLIT_SEND = 6
    EVENT_FLIT_ARRIVE = 7

    # Track packets for duration calculation
    packet_inject: Dict[int, Tuple[int, int, str]] = {}  # seq -> (cycle, router, tile_label)

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cycle = int(row['cycle'])
            event_type = int(row['type'])
            router_id = int(row['router_id'])
            packet_seq = int(row.get('packet_seq', 0))
            tensor = int(row.get('tensor', 0))
            m_tile = int(row.get('m_tile', 0))
            n_tile = int(row.get('n_tile', 0))
            k_tile = int(row.get('k_tile', 0))
            num_flits = int(row.get('num_flits', 256))
            src_router = int(row.get('src_router', router_id))
            dst_router = int(row.get('dst_router', router_id))

            min_cycle = min(min_cycle, cycle)
            max_cycle = max(max_cycle, cycle)

            # Create tile label
            tensor_names = {0: 'A', 1: 'B', 2: 'C', 3: 'Bias'}
            tile_label = f"{tensor_names.get(tensor, '?')}[{m_tile},{k_tile if tensor == 0 else n_tile}]"

            # Determine mesh position
            mesh_cols = 4  # Assume 4x4
            row_pos = router_id // mesh_cols
            col_pos = router_id % mesh_cols

            if event_type == EVENT_INJECT:
                resource_id = f"R[{row_pos},{col_pos}]"
                packet_inject[packet_seq] = (cycle, router_id, tile_label)
                # Add injection span (duration = num_flits cycles)
                resources[resource_id].append(TimeSpan(cycle, cycle + num_flits, 'L3_SEND', tile_label))

            elif event_type == EVENT_EJECT:
                resource_id = f"R[{row_pos},{col_pos}]"
                # Add receive span
                resources[resource_id].append(TimeSpan(cycle - num_flits, cycle, 'L3_RECEIVE', tile_label))

            elif event_type == EVENT_HOP:
                # Track link activity
                src_row = src_router // mesh_cols
                src_col = src_router % mesh_cols
                dst_row = dst_router // mesh_cols
                dst_col = dst_router % mesh_cols

                if dst_col > src_col:
                    link_id = f"E[{src_row},{src_col}]"
                elif dst_row > src_row:
                    link_id = f"S[{src_row},{src_col}]"
                else:
                    link_id = f"?[{src_row},{src_col}]"

                # Add link activity span
                resources[link_id].append(TimeSpan(cycle - num_flits, cycle, 'FLIT', tile_label))

    # Convert to ResourceActivity objects
    result = {rid: ResourceActivity(rid, spans) for rid, spans in resources.items()}

    return result, int(min_cycle) if min_cycle != float('inf') else 0, int(max_cycle)


def detect_trace_type(filename: str) -> str:
    """Detect whether trace is tile_flow or noc format."""
    with open(filename, 'r') as f:
        header = f.readline()
        if 'event_type' in header and 'l3_id' in header:
            return 'tile_flow'
        elif 'type' in header and 'router_id' in header:
            return 'noc'
        else:
            return 'unknown'


def render_gantt(
    resources: Dict[str, ResourceActivity],
    start_cycle: int,
    end_cycle: int,
    width: int = 80,
    use_color: bool = True,
    resource_filter: Optional[str] = None
) -> str:
    """Render ASCII Gantt chart."""

    lines = []

    # Filter resources if requested
    if resource_filter:
        prefix = resource_filter.upper()
        resources = {k: v for k, v in resources.items() if k.startswith(prefix) or k.startswith(prefix[0])}

    if not resources:
        return "No resources to display."

    # Sort resources by ID
    sorted_resources = sorted(resources.keys(), key=lambda x: (x[0], int(''.join(c for c in x if c.isdigit()) or '0')))

    # Calculate scale
    total_cycles = end_cycle - start_cycle
    if total_cycles <= 0:
        return "Invalid time range."

    # Leave room for labels
    label_width = max(len(r) for r in sorted_resources) + 2
    chart_width = width - label_width - 1
    cycles_per_char = max(1, total_cycles / chart_width)

    # Header
    header = f"\n{'RESOURCE ACTIVITY GANTT CHART':^{width}}\n"
    header += f"{'=' * width}\n"
    header += f"Time range: {start_cycle:,} - {end_cycle:,} cycles "
    header += f"(scale: {cycles_per_char:.0f} cycles/char)\n\n"
    lines.append(header)

    # Time axis
    axis_labels = ""
    for i in range(0, chart_width, max(1, chart_width // 6)):
        cycle = start_cycle + int(i * cycles_per_char)
        label = f"{cycle:,}"
        axis_labels += label + ' ' * max(1, (chart_width // 6) - len(label))
    lines.append(' ' * label_width + axis_labels[:chart_width] + '\n')
    lines.append(' ' * label_width + '|' + '-' * (chart_width - 2) + '|\n')

    # Calculate utilization for each resource
    utilizations = {}

    # Render each resource
    for resource_id in sorted_resources:
        activity = resources[resource_id]
        utilizations[resource_id] = activity.utilization(start_cycle, end_cycle)

        # Build character array for this resource
        chars = [' '] * chart_width
        colors = ['white'] * chart_width

        for span in activity.spans:
            # Map span to character positions
            span_start = max(0, int((span.start - start_cycle) / cycles_per_char))
            span_end = min(chart_width, int((span.end - start_cycle) / cycles_per_char) + 1)

            # Determine character and color based on event type
            char = BLOCKS['full']
            color = EVENT_COLORS.get(span.event_type, 'white')

            for i in range(span_start, span_end):
                if 0 <= i < chart_width:
                    chars[i] = char
                    colors[i] = color

        # Build line
        label = f"{resource_id}:".ljust(label_width)

        if use_color:
            line = label
            current_color = None
            for i, (char, color) in enumerate(zip(chars, colors)):
                if color != current_color:
                    if current_color:
                        line += COLORS['reset']
                    line += COLORS.get(color, '')
                    current_color = color
                line += char
            line += COLORS['reset']
        else:
            line = label + ''.join(chars)

        # Add utilization percentage
        util_pct = utilizations[resource_id] * 100
        line += f" {util_pct:5.1f}%"

        lines.append(line + '\n')

    # Legend
    lines.append('\n' + '-' * width + '\n')
    legend = "Legend: "
    if use_color:
        for event_type, color in EVENT_COLORS.items():
            legend += f"{COLORS[color]}{BLOCKS['full']}{COLORS['reset']}={event_type}  "
    else:
        legend += f"{BLOCKS['full']}=Activity  {BLOCKS['empty']}=Idle"
    lines.append(legend + '\n')

    # Summary statistics
    lines.append('\n' + '=' * width + '\n')
    lines.append('SUMMARY STATISTICS\n')
    lines.append('-' * width + '\n')

    total_resources = len(resources)
    avg_util = sum(utilizations.values()) / total_resources if total_resources > 0 else 0
    max_util = max(utilizations.values()) if utilizations else 0
    min_util = min(utilizations.values()) if utilizations else 0

    # Find busiest resource
    busiest = max(utilizations.items(), key=lambda x: x[1]) if utilizations else ('N/A', 0)

    lines.append(f"  Total cycles:     {total_cycles:,}\n")
    lines.append(f"  Resources shown:  {total_resources}\n")
    lines.append(f"  Avg utilization:  {avg_util*100:.1f}%\n")
    lines.append(f"  Max utilization:  {max_util*100:.1f}% ({busiest[0]})\n")
    lines.append(f"  Min utilization:  {min_util*100:.1f}%\n")

    # Group by resource type
    type_utils: Dict[str, List[float]] = defaultdict(list)
    for rid, util in utilizations.items():
        rtype = rid.split('[')[0]
        type_utils[rtype].append(util)

    if len(type_utils) > 1:
        lines.append('\nBy Resource Type:\n')
        for rtype, utils in sorted(type_utils.items()):
            avg = sum(utils) / len(utils) if utils else 0
            lines.append(f"  {rtype}: {avg*100:.1f}% avg ({len(utils)} resources)\n")

    lines.append('=' * width + '\n')

    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate ASCII Gantt chart from trace files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 ascii_gantt.py tile_flow_trace.csv
    python3 ascii_gantt.py /tmp/noc_trace.csv --width 120
    python3 ascii_gantt.py trace.csv --period 0-10000 --resources L3
    python3 ascii_gantt.py trace.csv --no-color
        """
    )
    parser.add_argument('trace_file', help='Path to trace CSV file')
    parser.add_argument('--width', '-w', type=int, default=100,
                        help='Chart width in characters (default: 100)')
    parser.add_argument('--period', '-p', type=str, default=None,
                        help='Time period as START-END (e.g., 0-10000)')
    parser.add_argument('--resources', '-r', type=str, default=None,
                        help='Filter resources by prefix (e.g., L3, R, E, S)')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable ANSI color output')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output to file instead of stdout')

    args = parser.parse_args()

    # Check file exists
    if not os.path.exists(args.trace_file):
        print(f"Error: File not found: {args.trace_file}", file=sys.stderr)
        sys.exit(1)

    # Detect trace type
    trace_type = detect_trace_type(args.trace_file)
    if trace_type == 'unknown':
        print(f"Error: Unknown trace format in {args.trace_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {trace_type} trace from: {args.trace_file}", file=sys.stderr)

    # Load trace
    if trace_type == 'tile_flow':
        resources, min_cycle, max_cycle = load_tile_flow_trace(args.trace_file)
    else:
        resources, min_cycle, max_cycle = load_noc_trace(args.trace_file)

    print(f"Loaded {len(resources)} resources, cycles {min_cycle:,} - {max_cycle:,}", file=sys.stderr)

    # Parse period if specified
    start_cycle = min_cycle
    end_cycle = max_cycle
    if args.period:
        try:
            parts = args.period.split('-')
            start_cycle = int(parts[0])
            end_cycle = int(parts[1])
        except (ValueError, IndexError):
            print(f"Error: Invalid period format. Use START-END (e.g., 0-10000)", file=sys.stderr)
            sys.exit(1)

    # Generate chart
    output = render_gantt(
        resources,
        start_cycle,
        end_cycle,
        width=args.width,
        use_color=not args.no_color,
        resource_filter=args.resources
    )

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
