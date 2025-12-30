#!/usr/bin/env python3
"""
export_metrics.py - Unified metrics export for KPU simulator traces

Exports comprehensive metrics from trace files to JSON format for
programmatic consumption, dashboards, and CI/CD integration.

Usage:
    python3 export_metrics.py noc_trace.csv metrics.json
    python3 export_metrics.py tile_flow_trace.csv metrics.json --type tile_flow
    python3 export_metrics.py noc_trace.csv --stdout

Output format:
{
    "metadata": { ... },
    "summary": { ... },
    "resources": { ... },
    "periods": [ ... ],
    "hotspots": [ ... ]
}
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class TimeSpan:
    """A time interval."""
    start: int
    end: int


@dataclass
class ResourceMetrics:
    """Metrics for a single resource."""
    name: str
    resource_type: str
    total_busy_cycles: int
    utilization: float
    event_count: int
    avg_event_duration: float


@dataclass
class PeriodMetrics:
    """Metrics for a time period."""
    start_cycle: int
    end_cycle: int
    overall_utilization: float
    active_resources: int
    total_resources: int
    bytes_transferred: int
    events: int


@dataclass
class SummaryMetrics:
    """Overall summary metrics."""
    total_cycles: int
    total_events: int
    total_bytes: int
    overall_utilization: float
    peak_utilization: float
    active_resources: int
    total_resources: int
    bandwidth_gbps: float


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def calculate_utilization(intervals: List[Tuple[int, int]], period_start: int, period_end: int) -> float:
    """Calculate utilization from intervals within a period."""
    clipped = []
    for start, end in intervals:
        s = max(start, period_start)
        e = min(end, period_end)
        if e > s:
            clipped.append((s, e))

    if not clipped:
        return 0.0

    merged = merge_intervals(clipped)
    busy = sum(e - s for s, e in merged)
    return min(1.0, busy / (period_end - period_start)) if period_end > period_start else 0.0


def parse_noc_trace(filename: str) -> Dict[str, Any]:
    """
    Parse NoC trace and extract metrics.

    Returns dict with events, intervals, and derived metrics.
    """
    # First pass: determine mesh size and max cycle
    max_router_id = 0
    max_cycle = 0
    total_flits = 0

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row.get('src_router', 0))
            dst = int(row.get('dst_router', 0))
            rid = int(row.get('router_id', 0))
            cycle = int(row.get('cycle', 0))
            max_router_id = max(max_router_id, src, dst, rid)
            max_cycle = max(max_cycle, cycle)
            total_flits += 1

    num_routers = max_router_id + 1
    cols = int(math.ceil(math.sqrt(num_routers)))
    rows = cols

    # Second pass: collect intervals per resource
    router_intervals = defaultdict(list)  # router_id -> [(start, end), ...]
    east_link_intervals = defaultdict(list)  # (src, dst) -> [(start, end), ...]
    south_link_intervals = defaultdict(list)

    event_counts = defaultdict(int)
    total_bytes = 0
    seen_hops = set()

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_type = row.get('type', '')
            cycle = int(row.get('cycle', 0))

            if event_type == '0':  # INJECT
                event_counts['inject'] += 1
            elif event_type == '1':  # HOP
                router_id = int(row.get('router_id', 0))
                port = row.get('port', '')
                packet_seq = int(row.get('packet_seq', 0))
                num_flits = int(row.get('num_flits', 4))

                hop_key = (packet_seq, router_id)
                if hop_key in seen_hops:
                    continue
                seen_hops.add(hop_key)

                start = cycle - num_flits
                end = cycle

                # Determine which link was used
                if port == 'W':
                    prev_router = router_id - 1
                    east_link_intervals[(prev_router, router_id)].append((start, end))
                elif port == 'N':
                    prev_router = router_id - cols
                    south_link_intervals[(prev_router, router_id)].append((start, end))

                router_intervals[router_id].append((start, end))
                event_counts['hop'] += 1
                total_bytes += num_flits * 64  # 64 bytes per FLIT

            elif event_type == '2':  # EJECT
                event_counts['eject'] += 1

    # Calculate resource metrics
    resources = {}

    # Router metrics
    for rid in range(num_routers):
        intervals = router_intervals.get(rid, [])
        merged = merge_intervals(intervals)
        busy_cycles = sum(e - s for s, e in merged) if merged else 0
        util = busy_cycles / max_cycle if max_cycle > 0 else 0

        r = rid // cols
        c = rid % cols
        resources[f"router_{rid}"] = ResourceMetrics(
            name=f"R[{r},{c}]",
            resource_type="router",
            total_busy_cycles=busy_cycles,
            utilization=util,
            event_count=len(intervals),
            avg_event_duration=busy_cycles / len(intervals) if intervals else 0
        )

    # East link metrics
    for (src, dst), intervals in east_link_intervals.items():
        merged = merge_intervals(intervals)
        busy_cycles = sum(e - s for s, e in merged) if merged else 0
        util = busy_cycles / max_cycle if max_cycle > 0 else 0

        src_r, src_c = src // cols, src % cols
        dst_r, dst_c = dst // cols, dst % cols
        resources[f"east_link_{src}_{dst}"] = ResourceMetrics(
            name=f"[{src_r},{src_c}]→[{dst_r},{dst_c}]",
            resource_type="east_link",
            total_busy_cycles=busy_cycles,
            utilization=util,
            event_count=len(intervals),
            avg_event_duration=busy_cycles / len(intervals) if intervals else 0
        )

    # South link metrics
    for (src, dst), intervals in south_link_intervals.items():
        merged = merge_intervals(intervals)
        busy_cycles = sum(e - s for s, e in merged) if merged else 0
        util = busy_cycles / max_cycle if max_cycle > 0 else 0

        src_r, src_c = src // cols, src % cols
        dst_r, dst_c = dst // cols, dst % cols
        resources[f"south_link_{src}_{dst}"] = ResourceMetrics(
            name=f"[{src_r},{src_c}]↓[{dst_r},{dst_c}]",
            resource_type="south_link",
            total_busy_cycles=busy_cycles,
            utilization=util,
            event_count=len(intervals),
            avg_event_duration=busy_cycles / len(intervals) if intervals else 0
        )

    # Calculate period metrics
    num_periods = 10
    period_length = max(1, max_cycle // num_periods)
    periods = []

    for p in range(num_periods):
        start = p * period_length
        end = min((p + 1) * period_length, max_cycle)
        if start >= max_cycle:
            break

        # Calculate per-period metrics
        all_intervals = []
        for intervals in router_intervals.values():
            all_intervals.extend(intervals)
        for intervals in east_link_intervals.values():
            all_intervals.extend(intervals)
        for intervals in south_link_intervals.values():
            all_intervals.extend(intervals)

        period_util = calculate_utilization(all_intervals, start, end)

        # Count active resources
        active = 0
        total = len(resources)
        for res_intervals in list(router_intervals.values()) + \
                             list(east_link_intervals.values()) + \
                             list(south_link_intervals.values()):
            for s, e in res_intervals:
                if s < end and e > start:
                    active += 1
                    break

        periods.append(PeriodMetrics(
            start_cycle=start,
            end_cycle=end,
            overall_utilization=period_util,
            active_resources=active,
            total_resources=total,
            bytes_transferred=0,  # Would need more detailed tracking
            events=0
        ))

    # Calculate summary
    active_resources = sum(1 for r in resources.values() if r.utilization > 0)
    total_resources = len(resources)
    overall_util = sum(r.utilization for r in resources.values()) / total_resources if total_resources > 0 else 0
    peak_util = max((r.utilization for r in resources.values()), default=0)

    # Bandwidth: bytes / cycles, assuming 1GHz clock
    bandwidth_gbps = (total_bytes / max_cycle) * 8 if max_cycle > 0 else 0

    summary = SummaryMetrics(
        total_cycles=max_cycle,
        total_events=sum(event_counts.values()),
        total_bytes=total_bytes,
        overall_utilization=overall_util,
        peak_utilization=peak_util,
        active_resources=active_resources,
        total_resources=total_resources,
        bandwidth_gbps=bandwidth_gbps
    )

    # Identify hotspots (top 5 most utilized resources)
    sorted_resources = sorted(resources.values(), key=lambda r: r.utilization, reverse=True)
    hotspots = [{"name": r.name, "type": r.resource_type, "utilization": r.utilization}
                for r in sorted_resources[:5] if r.utilization > 0]

    return {
        "metadata": {
            "trace_type": "noc",
            "mesh_rows": rows,
            "mesh_cols": cols,
            "total_routers": num_routers
        },
        "summary": asdict(summary),
        "resources": {k: asdict(v) for k, v in resources.items()},
        "periods": [asdict(p) for p in periods],
        "hotspots": hotspots,
        "event_counts": dict(event_counts)
    }


def parse_tile_flow_trace(filename: str) -> Dict[str, Any]:
    """
    Parse tile flow trace and extract metrics.

    Returns dict with events, intervals, and derived metrics.
    """
    resource_intervals = defaultdict(list)  # resource_name -> [(start, end), ...]
    event_counts = defaultdict(int)
    max_cycle = 0
    total_bytes = 0

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cycle = int(row.get('cycle', 0))
            event_type = row.get('event', row.get('type', ''))
            resource = row.get('resource', row.get('mover_id', ''))
            duration = int(row.get('duration', 1))

            max_cycle = max(max_cycle, cycle + duration)
            event_counts[event_type] += 1

            if resource:
                resource_intervals[resource].append((cycle, cycle + duration))

            # Estimate bytes from tile size (256KB default)
            if 'transfer' in event_type.lower() or 'fetch' in event_type.lower():
                total_bytes += 256 * 1024

    # Calculate resource metrics
    resources = {}
    for name, intervals in resource_intervals.items():
        merged = merge_intervals(intervals)
        busy_cycles = sum(e - s for s, e in merged) if merged else 0
        util = busy_cycles / max_cycle if max_cycle > 0 else 0

        resources[name] = ResourceMetrics(
            name=name,
            resource_type="block_mover" if "mover" in name.lower() else "unknown",
            total_busy_cycles=busy_cycles,
            utilization=util,
            event_count=len(intervals),
            avg_event_duration=busy_cycles / len(intervals) if intervals else 0
        )

    # Calculate summary
    active_resources = sum(1 for r in resources.values() if r.utilization > 0)
    total_resources = len(resources)
    overall_util = sum(r.utilization for r in resources.values()) / total_resources if total_resources > 0 else 0
    peak_util = max((r.utilization for r in resources.values()), default=0)
    bandwidth_gbps = (total_bytes / max_cycle) * 8 if max_cycle > 0 else 0

    summary = SummaryMetrics(
        total_cycles=max_cycle,
        total_events=sum(event_counts.values()),
        total_bytes=total_bytes,
        overall_utilization=overall_util,
        peak_utilization=peak_util,
        active_resources=active_resources,
        total_resources=total_resources,
        bandwidth_gbps=bandwidth_gbps
    )

    # Hotspots
    sorted_resources = sorted(resources.values(), key=lambda r: r.utilization, reverse=True)
    hotspots = [{"name": r.name, "type": r.resource_type, "utilization": r.utilization}
                for r in sorted_resources[:5] if r.utilization > 0]

    return {
        "metadata": {
            "trace_type": "tile_flow"
        },
        "summary": asdict(summary),
        "resources": {k: asdict(v) for k, v in resources.items()},
        "periods": [],  # TODO: implement period analysis
        "hotspots": hotspots,
        "event_counts": dict(event_counts)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Export unified metrics from trace files to JSON'
    )
    parser.add_argument('trace_file', help='Input trace CSV file')
    parser.add_argument('output', nargs='?', help='Output JSON file (omit for stdout)')
    parser.add_argument('--type', choices=['noc', 'tile_flow', 'auto'], default='auto',
                       help='Trace type (default: auto-detect)')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty-print JSON output')
    parser.add_argument('--stdout', action='store_true',
                       help='Output to stdout instead of file')

    args = parser.parse_args()

    # Auto-detect trace type
    trace_type = args.type
    if trace_type == 'auto':
        with open(args.trace_file, 'r') as f:
            header = f.readline().lower()
            if 'router_id' in header or 'src_router' in header:
                trace_type = 'noc'
            else:
                trace_type = 'tile_flow'

    # Parse trace
    if trace_type == 'noc':
        metrics = parse_noc_trace(args.trace_file)
    else:
        metrics = parse_tile_flow_trace(args.trace_file)

    # Output
    indent = 2 if args.pretty else None

    if args.stdout or not args.output:
        print(json.dumps(metrics, indent=indent))
    else:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=indent)
        print(f"Metrics exported to: {args.output}")
        print(f"  Total cycles: {metrics['summary']['total_cycles']}")
        print(f"  Total events: {metrics['summary']['total_events']}")
        print(f"  Overall utilization: {metrics['summary']['overall_utilization']*100:.1f}%")
        print(f"  Peak utilization: {metrics['summary']['peak_utilization']*100:.1f}%")


if __name__ == '__main__':
    main()
