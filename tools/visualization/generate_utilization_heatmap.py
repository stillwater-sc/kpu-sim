#!/usr/bin/env python3
"""
generate_utilization_heatmap.py - Heat map visualization for large tile arrays

Generates color-coded heat maps showing resource utilization across the NoC mesh.
Designed to scale to large arrays (64x64+) where individual tile tracking becomes
impractical.

Usage:
    python3 generate_utilization_heatmap.py noc_trace.csv output.html [--rows N] [--cols N]
    python3 generate_utilization_heatmap.py noc_trace.csv output.png --png

Features:
- Color gradient: blue (idle) → green (50%) → yellow (75%) → red (saturated)
- Aggregates by time period for temporal analysis
- Router, East Link, and South Link utilization views
- Interactive HTML with hover details or static PNG export
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class HopEvent:
    """A single hop transfer event."""
    cycle: int
    src_router: int
    dst_router: int
    direction: str  # 'E' or 'S'
    packet_seq: int
    tile_name: str
    flits: int


def parse_noc_trace(filename: str) -> Tuple[List[HopEvent], int, int, int]:
    """
    Parse NoC trace CSV and extract hop events.

    Returns: (events, num_rows, num_cols, max_cycle)

    Trace format: cycle,type,router_id,port,packet_seq,tensor,m_tile,n_tile,k_tile,flit_index,num_flits,src_router,dst_router
    Type values: 0=INJECT, 1=HOP, 2=EJECT
    Port values: L=Local, N=North, S=South, E=East, W=West
    """
    import math

    # First pass: determine mesh size
    max_router_id = 0
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row.get('src_router', 0))
            dst = int(row.get('dst_router', 0))
            rid = int(row.get('router_id', 0))
            max_router_id = max(max_router_id, src, dst, rid)

    num_routers = max_router_id + 1
    cols = int(math.ceil(math.sqrt(num_routers)))
    rows = cols  # Assume square mesh

    # Second pass: parse HOP events
    events = []
    max_cycle = 0
    seen_hops = set()  # Track (packet_seq, router_id) to deduplicate per-FLIT events

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Type is numeric: 0=INJECT, 1=HOP, 2=EJECT
            event_type = row.get('type', '')

            # Handle both string and numeric types
            is_hop = (event_type == '1' or event_type == 'HOP')

            if is_hop:
                cycle = int(row['cycle'])
                src_router = int(row['src_router'])
                dst_router = int(row['dst_router'])
                router_id = int(row['router_id'])
                port = row.get('port', '')
                packet_seq = int(row.get('packet_seq', 0))
                flit_index = int(row.get('flit_index', 0))
                num_flits = int(row.get('num_flits', 4))
                tensor = int(row.get('tensor', 0))
                m_tile = int(row.get('m_tile', 0))
                n_tile = int(row.get('n_tile', 0))
                k_tile = int(row.get('k_tile', 0))

                # Only process each hop once (not per-FLIT)
                hop_key = (packet_seq, router_id)
                if hop_key in seen_hops:
                    continue
                seen_hops.add(hop_key)

                # Determine direction from port (where packet arrived FROM)
                # W=West means packet came from West, so it's traveling East
                # N=North means packet came from North, so it's traveling South
                if port == 'W':
                    direction = 'E'  # Traveling East
                    prev_router = router_id - 1  # came from West
                elif port == 'N':
                    direction = 'S'  # Traveling South
                    prev_router = router_id - cols  # came from North
                elif port == 'E':
                    direction = 'W'  # Traveling West (uncommon)
                    prev_router = router_id + 1
                elif port == 'S':
                    direction = 'N'  # Traveling North (uncommon)
                    prev_router = router_id + cols
                else:
                    continue  # Local port or unknown

                tensor_names = ['A', 'B', 'C', 'Bias', 'Workspace', 'Custom']
                tname = tensor_names[tensor] if tensor < len(tensor_names) else '?'
                tile_name = f"{tname}[{m_tile},{n_tile},{k_tile}]"

                events.append(HopEvent(
                    cycle=cycle - num_flits,  # Approximate start time
                    src_router=prev_router,
                    dst_router=router_id,
                    direction=direction,
                    packet_seq=packet_seq,
                    tile_name=tile_name,
                    flits=num_flits
                ))

                max_cycle = max(max_cycle, cycle)

    return events, rows, cols, max_cycle


def calculate_link_utilization(
    events: List[HopEvent],
    rows: int,
    cols: int,
    start_cycle: int,
    end_cycle: int
) -> Tuple[Dict[Tuple[int,int,str], float], Dict[int, float]]:
    """
    Calculate utilization for each link and router in a time window.

    Returns: (link_utilization, router_utilization)
    - link_utilization: {(src, dst, direction): fraction}
    - router_utilization: {router_id: fraction}
    """
    period_cycles = end_cycle - start_cycle
    if period_cycles <= 0:
        return {}, {}

    # Track busy intervals per link
    link_intervals = defaultdict(list)  # (src, dst, dir) -> [(start, end), ...]
    router_intervals = defaultdict(list)  # router_id -> [(start, end), ...]

    for e in events:
        hop_start = e.cycle
        hop_end = e.cycle + e.flits

        # Clip to period
        s = max(hop_start, start_cycle)
        ed = min(hop_end, end_cycle)
        if ed > s:
            link_key = (e.src_router, e.dst_router, e.direction)
            link_intervals[link_key].append((s, ed))
            router_intervals[e.src_router].append((s, ed))

    def merge_and_calculate_util(intervals: List[Tuple[int, int]]) -> float:
        if not intervals:
            return 0.0
        intervals.sort()
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        busy = sum(e - s for s, e in merged)
        return min(1.0, busy / period_cycles)

    link_util = {k: merge_and_calculate_util(v) for k, v in link_intervals.items()}
    router_util = {k: merge_and_calculate_util(v) for k, v in router_intervals.items()}

    return link_util, router_util


def utilization_to_color(util: float) -> str:
    """
    Convert utilization (0-1) to color.
    Blue (0%) → Green (50%) → Yellow (75%) → Red (100%)
    """
    if util <= 0:
        return "#4444ff"  # Blue - idle
    elif util <= 0.5:
        # Blue to Green
        t = util / 0.5
        r = int(68 * (1 - t) + 0 * t)
        g = int(68 * (1 - t) + 200 * t)
        b = int(255 * (1 - t) + 0 * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    elif util <= 0.75:
        # Green to Yellow
        t = (util - 0.5) / 0.25
        r = int(0 * (1 - t) + 255 * t)
        g = int(200 * (1 - t) + 200 * t)
        b = 0
        return f"#{r:02x}{g:02x}{b:02x}"
    else:
        # Yellow to Red
        t = (util - 0.75) / 0.25
        r = 255
        g = int(200 * (1 - t) + 0 * t)
        b = 0
        return f"#{r:02x}{g:02x}{b:02x}"


def generate_html_heatmap(
    events: List[HopEvent],
    rows: int,
    cols: int,
    max_cycle: int,
    output_file: str,
    num_time_periods: int = 10
):
    """Generate interactive HTML heat map."""

    # Calculate utilization for each time period
    period_length = max(1, max_cycle // num_time_periods)

    periods = []
    for p in range(num_time_periods):
        start = p * period_length
        end = min((p + 1) * period_length, max_cycle)
        if start >= max_cycle:
            break
        link_util, router_util = calculate_link_utilization(
            events, rows, cols, start, end
        )
        periods.append({
            'start': start,
            'end': end,
            'links': link_util,
            'routers': router_util
        })

    # Also calculate overall utilization
    overall_links, overall_routers = calculate_link_utilization(
        events, rows, cols, 0, max_cycle
    )

    # Generate JSON data first (before f-string template)
    import json

    def links_to_json(link_dict):
        """Convert link utilization dict to JSON-friendly format."""
        result = {}
        for (src, dst, direction), util in link_dict.items():
            key = f"{src}_{dst}_{direction}"
            result[key] = util
        return result

    periods_json = json.dumps([{
        'start': p['start'],
        'end': p['end'],
        'links': links_to_json(p['links']),
        'routers': p['routers']
    } for p in periods])

    overall_links_json = json.dumps(links_to_json(overall_links))
    overall_routers_json = json.dumps(overall_routers)

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>NoC Utilization Heat Map</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        h1, h2 {{ color: #00d4ff; margin-bottom: 10px; }}
        .container {{ display: flex; gap: 30px; flex-wrap: wrap; }}
        .heatmap-panel {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .mesh-grid {{
            display: inline-grid;
            gap: 2px;
            background: #333;
            padding: 10px;
            border-radius: 5px;
        }}
        .router {{
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.1s;
        }}
        .router:hover {{
            transform: scale(1.1);
            z-index: 10;
        }}
        .link-grid {{
            display: inline-grid;
            gap: 1px;
            background: #333;
            padding: 10px;
            border-radius: 5px;
        }}
        .link-cell {{
            width: 20px;
            height: 20px;
            font-size: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }}
        .controls {{
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .slider-container {{ margin: 10px 0; }}
        .slider-container label {{ display: block; margin-bottom: 5px; }}
        input[type="range"] {{ width: 300px; }}
        .legend {{
            display: flex;
            gap: 5px;
            align-items: center;
            margin-top: 10px;
        }}
        .legend-gradient {{
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, #4444ff, #00c800, #ffc800, #ff0000);
            border-radius: 3px;
        }}
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            width: 200px;
            font-size: 10px;
        }}
        .tooltip {{
            position: fixed;
            background: rgba(0,0,0,0.9);
            color: #fff;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            z-index: 1000;
        }}
        .stats {{
            background: #1a1a2e;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>NoC Utilization Heat Map</h1>
    <p>Mesh: {rows}x{cols} ({rows*cols} routers) | Simulation: {max_cycle} cycles</p>

    <div class="controls">
        <div class="slider-container">
            <label>Time Period: <span id="period-label">Overall (0-{max_cycle})</span></label>
            <input type="range" id="period-slider" min="0" max="{len(periods)}" value="0">
        </div>
        <div class="legend">
            <span>Idle</span>
            <div class="legend-gradient"></div>
            <span>Saturated</span>
        </div>
        <div class="legend-labels" style="margin-left: 25px;">
            <span>0%</span>
            <span>50%</span>
            <span>75%</span>
            <span>100%</span>
        </div>
    </div>

    <div class="container">
        <div class="heatmap-panel">
            <h2>Router Utilization</h2>
            <div class="mesh-grid" id="router-grid" style="grid-template-columns: repeat({cols}, 40px);">
            </div>
            <div class="stats" id="router-stats"></div>
        </div>

        <div class="heatmap-panel">
            <h2>East Link Utilization</h2>
            <div class="link-grid" id="east-link-grid" style="grid-template-columns: repeat({cols - 1}, 20px);">
            </div>
            <div class="stats" id="east-stats"></div>
        </div>

        <div class="heatmap-panel">
            <h2>South Link Utilization</h2>
            <div class="link-grid" id="south-link-grid" style="grid-template-columns: repeat({cols}, 20px);">
            </div>
            <div class="stats" id="south-stats"></div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
    const ROWS = {rows};
    const COLS = {cols};
    const MAX_CYCLE = {max_cycle};

    // Utilization data for each period
    const periods = {periods_json};
    const overallData = {{
        links: {overall_links_json},
        routers: {overall_routers_json}
    }};

    function utilizationToColor(util) {{
        if (util <= 0) return "#4444ff";
        else if (util <= 0.5) {{
            const t = util / 0.5;
            const r = Math.round(68 * (1 - t) + 0 * t);
            const g = Math.round(68 * (1 - t) + 200 * t);
            const b = Math.round(255 * (1 - t) + 0 * t);
            return `rgb(${{r}},${{g}},${{b}})`;
        }} else if (util <= 0.75) {{
            const t = (util - 0.5) / 0.25;
            const r = Math.round(0 * (1 - t) + 255 * t);
            const g = Math.round(200 * (1 - t) + 200 * t);
            return `rgb(${{r}},${{g}},0)`;
        }} else {{
            const t = (util - 0.75) / 0.25;
            const g = Math.round(200 * (1 - t));
            return `rgb(255,${{g}},0)`;
        }}
    }}

    function initGrids() {{
        const routerGrid = document.getElementById('router-grid');
        const eastGrid = document.getElementById('east-link-grid');
        const southGrid = document.getElementById('south-link-grid');

        // Router grid
        for (let r = 0; r < ROWS; r++) {{
            for (let c = 0; c < COLS; c++) {{
                const div = document.createElement('div');
                div.className = 'router';
                div.id = `router-${{r}}-${{c}}`;
                div.textContent = `${{r}},${{c}}`;
                div.dataset.row = r;
                div.dataset.col = c;
                routerGrid.appendChild(div);
            }}
        }}

        // East link grid (COLS-1 per row)
        for (let r = 0; r < ROWS; r++) {{
            for (let c = 0; c < COLS - 1; c++) {{
                const div = document.createElement('div');
                div.className = 'link-cell';
                div.id = `east-${{r}}-${{c}}`;
                div.textContent = '→';
                div.dataset.row = r;
                div.dataset.col = c;
                eastGrid.appendChild(div);
            }}
        }}

        // South link grid (ROWS-1 rows)
        for (let r = 0; r < ROWS - 1; r++) {{
            for (let c = 0; c < COLS; c++) {{
                const div = document.createElement('div');
                div.className = 'link-cell';
                div.id = `south-${{r}}-${{c}}`;
                div.textContent = '↓';
                div.dataset.row = r;
                div.dataset.col = c;
                southGrid.appendChild(div);
            }}
        }}
    }}

    function updateDisplay(periodIndex) {{
        const data = periodIndex === 0 ? overallData : periods[periodIndex - 1];
        const label = document.getElementById('period-label');

        if (periodIndex === 0) {{
            label.textContent = `Overall (0-${{MAX_CYCLE}})`;
        }} else {{
            const p = periods[periodIndex - 1];
            label.textContent = `Period ${{periodIndex}}: cycles ${{p.start}}-${{p.end}}`;
        }}

        // Update router colors
        let routerTotal = 0, routerActive = 0;
        for (let r = 0; r < ROWS; r++) {{
            for (let c = 0; c < COLS; c++) {{
                const id = r * COLS + c;
                const util = data.routers[id] || 0;
                const elem = document.getElementById(`router-${{r}}-${{c}}`);
                elem.style.background = utilizationToColor(util);
                elem.title = `Router [${{r}},${{c}}]: ${{(util * 100).toFixed(1)}}%`;
                routerTotal += 1;
                if (util > 0) routerActive++;
            }}
        }}
        document.getElementById('router-stats').textContent =
            `Active: ${{routerActive}}/${{routerTotal}} routers`;

        // Update east link colors
        let eastTotal = 0, eastActive = 0, eastSum = 0;
        for (let r = 0; r < ROWS; r++) {{
            for (let c = 0; c < COLS - 1; c++) {{
                const src = r * COLS + c;
                const dst = r * COLS + c + 1;
                const key = `${{src}}_${{dst}}_E`;
                const util = data.links[key] || 0;
                const elem = document.getElementById(`east-${{r}}-${{c}}`);
                elem.style.background = utilizationToColor(util);
                elem.title = `East [${{r}},${{c}}]→[${{r}},${{c+1}}]: ${{(util * 100).toFixed(1)}}%`;
                eastTotal++;
                if (util > 0) {{ eastActive++; eastSum += util; }}
            }}
        }}
        document.getElementById('east-stats').textContent =
            `Active: ${{eastActive}}/${{eastTotal}} | Avg: ${{(eastActive > 0 ? eastSum/eastActive*100 : 0).toFixed(1)}}%`;

        // Update south link colors
        let southTotal = 0, southActive = 0, southSum = 0;
        for (let r = 0; r < ROWS - 1; r++) {{
            for (let c = 0; c < COLS; c++) {{
                const src = r * COLS + c;
                const dst = (r + 1) * COLS + c;
                const key = `${{src}}_${{dst}}_S`;
                const util = data.links[key] || 0;
                const elem = document.getElementById(`south-${{r}}-${{c}}`);
                elem.style.background = utilizationToColor(util);
                elem.title = `South [${{r}},${{c}}]→[${{r+1}},${{c}}]: ${{(util * 100).toFixed(1)}}%`;
                southTotal++;
                if (util > 0) {{ southActive++; southSum += util; }}
            }}
        }}
        document.getElementById('south-stats').textContent =
            `Active: ${{southActive}}/${{southTotal}} | Avg: ${{(southActive > 0 ? southSum/southActive*100 : 0).toFixed(1)}}%`;
    }}

    // Tooltip handling
    const tooltip = document.getElementById('tooltip');
    document.addEventListener('mousemove', (e) => {{
        if (e.target.classList.contains('router') || e.target.classList.contains('link-cell')) {{
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 10) + 'px';
            tooltip.style.top = (e.clientY + 10) + 'px';
            tooltip.textContent = e.target.title;
        }} else {{
            tooltip.style.display = 'none';
        }}
    }});

    // Period slider
    document.getElementById('period-slider').addEventListener('input', (e) => {{
        updateDisplay(parseInt(e.target.value));
    }});

    // Initialize
    initGrids();
    updateDisplay(0);
    </script>
</body>
</html>
'''

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Heat map generated: {output_file}")
    print(f"  Mesh size: {rows}x{cols}")
    print(f"  Simulation cycles: {max_cycle}")
    print(f"  Time periods: {len(periods)}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate NoC utilization heat map visualization'
    )
    parser.add_argument('trace_file', help='NoC trace CSV file')
    parser.add_argument('output', help='Output file (HTML or PNG)')
    parser.add_argument('--rows', type=int, help='Override mesh rows')
    parser.add_argument('--cols', type=int, help='Override mesh columns')
    parser.add_argument('--periods', type=int, default=10,
                       help='Number of time periods (default: 10)')
    parser.add_argument('--png', action='store_true',
                       help='Generate static PNG instead of HTML')

    args = parser.parse_args()

    # Parse trace
    events, rows, cols, max_cycle = parse_noc_trace(args.trace_file)

    if args.rows:
        rows = args.rows
    if args.cols:
        cols = args.cols

    if len(events) == 0:
        print("Error: No HOP events found in trace file")
        sys.exit(1)

    print(f"Parsed {len(events)} HOP events from trace")

    if args.png:
        # PNG generation would require matplotlib
        try:
            generate_png_heatmap(events, rows, cols, max_cycle, args.output)
        except ImportError:
            print("PNG export requires matplotlib. Install with: pip install matplotlib")
            print("Falling back to HTML output...")
            output = args.output.replace('.png', '.html')
            generate_html_heatmap(events, rows, cols, max_cycle, output, args.periods)
    else:
        generate_html_heatmap(events, rows, cols, max_cycle, args.output, args.periods)


def generate_png_heatmap(events, rows, cols, max_cycle, output_file):
    """Generate static PNG heat map using matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate overall utilization
    link_util, router_util = calculate_link_utilization(
        events, rows, cols, 0, max_cycle
    )

    # Create router utilization matrix
    router_matrix = np.zeros((rows, cols))
    for router_id, util in router_util.items():
        r = router_id // cols
        c = router_id % cols
        router_matrix[r, c] = util

    # Create east link utilization matrix
    east_matrix = np.zeros((rows, cols - 1))
    for (src, dst, direction), util in link_util.items():
        if direction == 'E':
            r = src // cols
            c = src % cols
            if c < cols - 1:
                east_matrix[r, c] = util

    # Create south link utilization matrix
    south_matrix = np.zeros((rows - 1, cols))
    for (src, dst, direction), util in link_util.items():
        if direction == 'S':
            r = src // cols
            c = src % cols
            if r < rows - 1:
                south_matrix[r, c] = util

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Custom colormap: blue -> green -> yellow -> red
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#4444ff', '#00c800', '#ffc800', '#ff0000']
    cmap = LinearSegmentedColormap.from_list('utilization', colors)

    # Router utilization
    im1 = axes[0].imshow(router_matrix, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title('Router Utilization')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0], label='Utilization')

    # East link utilization
    im2 = axes[1].imshow(east_matrix, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('East Link Utilization')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[1], label='Utilization')

    # South link utilization
    im3 = axes[2].imshow(south_matrix, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title('South Link Utilization')
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[2], label='Utilization')

    plt.suptitle(f'NoC Utilization Heat Map ({rows}x{cols} mesh, {max_cycle} cycles)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"PNG heat map saved to: {output_file}")


if __name__ == '__main__':
    main()
