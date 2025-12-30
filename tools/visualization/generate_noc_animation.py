#!/usr/bin/env python3
"""
Generate NoC Animation Visualization

Takes a NoC trace CSV and generates a self-contained HTML animation
showing packets moving through the 4x4 router mesh.

Usage:
    python3 generate_noc_animation.py noc_trace.csv output.html
    python3 generate_noc_animation.py noc_trace.csv  # outputs to noc_animation.html
"""

import argparse
import json
import sys
from pathlib import Path


def load_noc_trace(filename: str) -> list:
    """Load NoC trace CSV file and return list of events."""
    events = []
    with open(filename, 'r') as f:
        headers = f.readline().strip().split(',')
        for line in f:
            values = line.strip().split(',')
            if len(values) < len(headers):
                continue
            event = {}
            for h, v in zip(headers, values):
                # Integer fields
                int_fields = ['cycle', 'router_id', 'packet_seq', 'tensor',
                              'm_tile', 'n_tile', 'k_tile', 'type',
                              'flit_index', 'num_flits', 'src_router', 'dst_router']
                if h in int_fields:
                    try:
                        event[h] = int(v)
                    except ValueError:
                        event[h] = 0
                else:
                    event[h] = v
            events.append(event)
    # Sort by cycle then by type (FLIT_ARRIVE events should come after INJECT/HOP)
    events.sort(key=lambda e: (e.get('cycle', 0), e.get('type', 0)))
    return events


def generate_noc_html(events: list, title: str = "KPU NoC Animation") -> str:
    """Generate complete HTML with embedded NoC trace data."""

    max_cycle = max(e['cycle'] for e in events) if events else 1000
    num_events = len(events)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }}
        .container {{ max-width: 1800px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 20px; }}
        h1 {{
            font-size: 2.2em;
            background: linear-gradient(90deg, #00ff88, #00aaff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .subtitle {{ color: #888; font-size: 1em; }}
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 20px;
        }}
        .noc-container {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }}
        #noc-svg {{ display: block; margin: 0 auto; }}

        /* L3 Tile styling - THE MAIN VISUAL ELEMENT */
        .l3-tile {{ cursor: pointer; transition: all 0.3s ease; }}
        .l3-tile:hover {{ filter: brightness(1.2); }}
        .l3-tile-bg {{
            fill: #1a2a4a;
            stroke: #3366aa;
            stroke-width: 2;
            rx: 8;
        }}
        .l3-tile.selected .l3-tile-bg {{
            stroke: #00ff88;
            stroke-width: 3;
            filter: drop-shadow(0 0 15px rgba(0, 255, 136, 0.5));
        }}
        .l3-tile.receiving .l3-tile-bg {{
            stroke: #ffaa00;
            stroke-width: 3;
            animation: pulse-receive 0.5s ease;
        }}
        @keyframes pulse-receive {{
            0%, 100% {{ filter: none; }}
            50% {{ filter: drop-shadow(0 0 20px rgba(255, 170, 0, 0.8)); }}
        }}
        .l3-label {{ fill: #6699cc; font-size: 11px; font-weight: bold; }}
        .l3-capacity {{ fill: #446688; font-size: 9px; }}

        /* Matrix tile blocks inside L3 */
        .matrix-tile {{
            rx: 3;
            stroke-width: 1;
            transition: all 0.2s ease;
        }}
        .matrix-tile.tensor-A {{ fill: #FF6B6B; stroke: #cc4444; }}
        .matrix-tile.tensor-B {{ fill: #4ECDC4; stroke: #2a9a92; }}
        .matrix-tile.tensor-C {{ fill: #FFE66D; stroke: #ccaa00; }}
        .matrix-tile-label {{
            font-size: 8px;
            font-weight: bold;
            fill: #000;
            pointer-events: none;
        }}

        /* Router styling - SMALL connection point */
        .router {{
            transition: all 0.3s ease;
        }}
        .router-bg {{
            fill: #2a3a5a;
            stroke: #445;
            stroke-width: 1.5;
        }}
        .router.active .router-bg {{
            fill: #3a5a3a;
            stroke: #00ff88;
            filter: drop-shadow(0 0 8px rgba(0, 255, 136, 0.6));
        }}
        .router.congested .router-bg {{
            fill: #5a3a3a;
            stroke: #ff4444;
        }}
        .router-label {{ fill: #888; font-size: 9px; }}

        /* Link styling (Router-to-Router connections) */
        .link {{
            stroke: #334455;
            stroke-width: 4;
            stroke-linecap: round;
            transition: stroke 0.3s ease;
        }}
        .link.active {{ stroke: #00ff88; }}
        .link.busy {{ stroke: #ffaa00; }}
        .link.congested {{ stroke: #ff4444; }}

        /* Local port connection (Router ↔ L3) */
        .local-port {{
            stroke: #446688;
            stroke-width: 2;
            stroke-dasharray: 4 2;
            marker-end: url(#local-port-arrow);
        }}
        .local-port.active {{
            stroke: #00ff88;
            stroke-width: 3;
            filter: drop-shadow(0 0 4px rgba(0, 255, 136, 0.5));
        }}

        /* Animated packet on link */
        .packet-in-flight {{
            transition: none;
        }}
        .packet-body {{
            stroke-width: 1.5;
        }}
        .packet-body.tensor-A {{ fill: #FF6B6B; stroke: #ff4444; }}
        .packet-body.tensor-B {{ fill: #4ECDC4; stroke: #00aaaa; }}
        .packet-body.tensor-C {{ fill: #FFE66D; stroke: #ccaa00; }}
        .packet-label {{
            font-size: 7px;
            font-weight: bold;
            fill: #000;
            pointer-events: none;
        }}

        /* Flow direction indicators */
        .flow-arrow {{
            fill: none;
            stroke: #335;
            stroke-width: 2;
            marker-end: url(#flow-arrow-head);
            opacity: 0.4;
        }}
        .flow-arrow.east {{ stroke: rgba(255, 107, 107, 0.4); }}
        .flow-arrow.south {{ stroke: rgba(78, 205, 196, 0.4); }}

        /* Controls panel */
        .controls {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            max-height: 90vh;
            overflow-y: auto;
        }}
        .control-group {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 15px;
        }}
        .control-group h3 {{
            color: #00ff88;
            margin-bottom: 12px;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .trace-info {{
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid #00ff88;
            border-radius: 8px;
            padding: 10px;
            font-size: 0.85em;
            text-align: center;
        }}
        .play-controls {{
            display: flex;
            gap: 8px;
            justify-content: center;
            margin-bottom: 12px;
        }}
        .btn {{
            background: linear-gradient(135deg, #00aa66, #006644);
            border: none;
            color: white;
            padding: 10px 18px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95em;
            transition: all 0.3s ease;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
        }}
        .btn-secondary {{ background: linear-gradient(135deg, #444, #222); }}
        .btn.active {{ background: linear-gradient(135deg, #00cc77, #008855); }}
        .speed-control {{ display: flex; align-items: center; gap: 10px; margin-top: 10px; }}
        .speed-control label {{ font-size: 0.85em; color: #888; }}
        .speed-control input {{
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            background: #222;
            border-radius: 3px;
        }}
        .speed-control input::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #00ff88;
            border-radius: 50%;
            cursor: pointer;
        }}
        .current-cycle {{
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff88;
            text-align: center;
            margin: 8px 0;
        }}
        .timeline {{
            position: relative;
            height: 50px;
            background: #111;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 8px;
            cursor: pointer;
        }}
        .timeline-progress {{
            position: absolute;
            height: 100%;
            background: linear-gradient(90deg, rgba(0, 255, 136, 0.2), rgba(0, 170, 255, 0.2));
            transition: width 0.1s ease;
        }}
        .timeline-cursor {{
            position: absolute;
            width: 2px;
            height: 100%;
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
        }}
        .timeline-events {{ position: absolute; width: 100%; height: 100%; pointer-events: none; }}
        .timeline-event {{
            position: absolute;
            width: 2px;
            height: 40%;
            bottom: 0;
            border-radius: 1px 1px 0 0;
            opacity: 0.8;
        }}
        .cycle-display {{
            display: flex;
            justify-content: space-between;
            color: #666;
            font-size: 0.8em;
        }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }}
        .stat-card {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }}
        .stat-value {{ font-size: 1.3em; font-weight: bold; color: #00ff88; }}
        .stat-label {{ font-size: 0.7em; color: #666; text-transform: uppercase; }}

        /* L3 Detail Panel */
        .l3-detail {{
            background: rgba(0, 50, 100, 0.2);
            border: 1px solid #336;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
        }}
        .l3-detail-header {{
            font-weight: bold;
            color: #6699cc;
            margin-bottom: 8px;
        }}
        .l3-tile-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }}
        .l3-tile-chip {{
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.75em;
            font-weight: bold;
        }}
        .l3-tile-chip.tensor-A {{ background: #FF6B6B; color: #000; }}
        .l3-tile-chip.tensor-B {{ background: #4ECDC4; color: #000; }}
        .l3-tile-chip.tensor-C {{ background: #FFE66D; color: #000; }}

        /* Event log */
        .event-log {{
            background: #0a0a14;
            border-radius: 8px;
            padding: 10px;
            height: 150px;
            overflow-y: auto;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.75em;
        }}
        .event-log-entry {{
            padding: 2px 6px;
            border-radius: 3px;
            margin-bottom: 2px;
            border-left: 3px solid transparent;
        }}
        .event-log-entry.INJECT {{ background: rgba(0, 255, 136, 0.15); border-left-color: #00ff88; }}
        .event-log-entry.HOP {{ background: rgba(0, 170, 255, 0.15); border-left-color: #00aaff; }}
        .event-log-entry.EJECT {{ background: rgba(255, 170, 0, 0.15); border-left-color: #ffaa00; }}
        .event-log-entry.BLOCKED {{ background: rgba(255, 68, 68, 0.15); border-left-color: #ff4444; }}

        /* Legend */
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: center;
            margin-top: 15px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.8em; }}
        .legend-color {{ width: 14px; height: 14px; border-radius: 3px; }}
        .legend-arrow {{ font-size: 1.2em; }}

        /* Filter controls */
        .filter-group {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            cursor: pointer;
            border: 1px solid #444;
            background: #222;
            color: #888;
            transition: all 0.2s ease;
        }}
        .filter-btn:hover {{ border-color: #666; color: #aaa; }}
        .filter-btn.active {{ border-color: #00ff88; color: #00ff88; background: rgba(0,255,136,0.1); }}
        .filter-btn.tensor-A.active {{ border-color: #FF6B6B; color: #FF6B6B; background: rgba(255,107,107,0.1); }}
        .filter-btn.tensor-B.active {{ border-color: #4ECDC4; color: #4ECDC4; background: rgba(78,205,196,0.1); }}
        .filter-btn.tensor-C.active {{ border-color: #FFE66D; color: #FFE66D; background: rgba(255,230,109,0.1); }}

        /* Tooltip */
        .tooltip {{
            position: fixed;
            background: rgba(0, 0, 0, 0.95);
            border: 1px solid #00ff88;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 0.85em;
            pointer-events: none;
            z-index: 1000;
            display: none;
            max-width: 250px;
        }}
        .tooltip.visible {{ display: block; }}
        .tooltip-title {{ font-weight: bold; color: #00ff88; margin-bottom: 4px; }}
        .tooltip-row {{ color: #aaa; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>KPU NoC Visualization</h1>
            <p class="subtitle">Network-on-Chip packet flow through 4x4 mesh - L3 tiles cache matrix tiles for reuse</p>
        </header>

        <div class="main-content">
            <div class="noc-container">
                <svg id="noc-svg" width="880" height="880" viewBox="0 0 880 880">
                    <defs>
                        <marker id="flow-arrow-head" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto">
                            <polygon points="0 0, 6 2.5, 0 5" fill="#446"/>
                        </marker>
                        <marker id="local-port-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                            <polygon points="0 0, 8 3, 0 6" fill="#446688"/>
                        </marker>
                        <filter id="glow-green">
                            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                            <feMerge>
                                <feMergeNode in="coloredBlur"/>
                                <feMergeNode in="SourceGraphic"/>
                            </feMerge>
                        </filter>
                    </defs>
                    <g id="flow-arrows-layer"></g>
                    <g id="links-layer"></g>
                    <g id="l3-tiles-layer"></g>
                    <g id="routers-layer"></g>
                    <g id="packets-layer"></g>
                </svg>

                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FF6B6B;"></div>
                        <span>A tiles</span>
                        <span class="legend-arrow" style="color: #FF6B6B;">→</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4ECDC4;"></div>
                        <span>B tiles</span>
                        <span class="legend-arrow" style="color: #4ECDC4;">↓</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FFE66D;"></div>
                        <span>C tiles</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #1a2a4a; border: 2px solid #3366aa;"></div>
                        <span>L3 (Lx)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #2a3a5a; border: 2px solid #445; border-radius: 50%;"></div>
                        <span>Router (Rx)</span>
                    </div>
                    <div class="legend-item">
                        <div style="width: 24px; height: 2px; background: #446688; border: 1px dashed #446688;"></div>
                        <span>Local Port</span>
                    </div>
                    <div class="legend-item">
                        <div style="width: 24px; height: 4px; background: #334455;"></div>
                        <span>NoC Link</span>
                    </div>
                </div>
            </div>

            <div class="controls">
                <div class="trace-info">
                    NoC Trace: {num_events} events, {max_cycle} cycles
                </div>

                <div class="control-group">
                    <h3>Playback</h3>
                    <div class="play-controls">
                        <button class="btn btn-secondary" id="btn-reset" title="Reset">&#8634;</button>
                        <button class="btn btn-secondary" id="btn-step-back" title="Previous Event">&#9664;</button>
                        <button class="btn" id="btn-play" title="Play/Pause">&#9654; Play</button>
                        <button class="btn btn-secondary" id="btn-step" title="Next Event">&#9654;&#9654;</button>
                    </div>
                    <div class="speed-control">
                        <label>Speed:</label>
                        <input type="range" id="speed-slider" min="1" max="50" value="5">
                        <span id="speed-value">1x</span>
                    </div>
                    <div class="speed-control">
                        <label>Step (cycles):</label>
                        <input type="range" id="step-slider" min="1" max="10" value="1">
                        <span id="step-value">1</span>
                    </div>
                </div>

                <div class="control-group">
                    <h3>Timeline</h3>
                    <div class="current-cycle">Cycle: <span id="cycle-counter">0</span></div>
                    <div class="timeline" id="timeline">
                        <div class="timeline-progress" id="timeline-progress"></div>
                        <div class="timeline-events" id="timeline-events"></div>
                        <div class="timeline-cursor" id="timeline-cursor"></div>
                    </div>
                    <div class="cycle-display">
                        <span>0</span>
                        <span>{max_cycle}</span>
                    </div>
                </div>

                <div class="control-group">
                    <h3>Filter</h3>
                    <div class="filter-group">
                        <button class="filter-btn tensor-A active" id="filter-A">A Tiles</button>
                        <button class="filter-btn tensor-B active" id="filter-B">B Tiles</button>
                        <button class="filter-btn tensor-C active" id="filter-C">C Tiles</button>
                    </div>
                </div>

                <div class="control-group">
                    <h3>Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="stat-packets">0</div>
                            <div class="stat-label">Injected</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-delivered">0</div>
                            <div class="stat-label">Delivered</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-in-flight">0</div>
                            <div class="stat-label">In Flight</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-hops">0</div>
                            <div class="stat-label">Total Hops</div>
                        </div>
                    </div>
                </div>

                <div class="control-group">
                    <h3>Selected L3</h3>
                    <div class="l3-detail" id="l3-detail">
                        <div class="l3-detail-header">Click an L3 tile to inspect</div>
                        <div class="l3-tile-list" id="l3-tile-list"></div>
                    </div>
                </div>

                <div class="control-group">
                    <h3>Event Log</h3>
                    <div class="event-log" id="event-log"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip">
        <div class="tooltip-title"></div>
        <div class="tooltip-content"></div>
    </div>

    <script>
        // Embedded NoC trace data
        const events = {json.dumps(events)};
        const maxCycle = {max_cycle};

        // Configuration - L3 tiles are the main visual element
        const MESH_SIZE = 4;
        const L3_SIZE = 130;          // L3 tiles are large (cache storage)
        const ROUTER_SIZE = 28;       // Routers are small (switching only)
        const CELL_SIZE = 200;        // Grid cell size
        const OFFSET = 70;            // Offset from edge
        const ROUTER_OFFSET = 20;     // Router offset from L3 top-left corner
        const TILE_BLOCK_SIZE = 26;   // Matrix tile block size inside L3
        const TILE_COLS = 4;          // Tiles per row in L3 grid
        const PACKET_SIZE = 16;       // Packet size on links

        // Event types
        const EVENT_INJECT = 0;
        const EVENT_HOP = 1;
        const EVENT_EJECT = 2;
        const EVENT_BLOCKED = 3;
        const EVENT_LINK_BUSY = 4;
        const EVENT_LINK_IDLE = 5;
        const EVENT_FLIT_SEND = 6;
        const EVENT_FLIT_ARRIVE = 7;

        const EVENT_NAMES = ['INJECT', 'HOP', 'EJECT', 'BLOCKED', 'LINK_BUSY', 'LINK_IDLE', 'FLIT_SEND', 'FLIT_ARRIVE'];
        const TENSOR_NAMES = {{ 0: 'A', 1: 'B', 2: 'C', 3: 'D' }};
        const TENSOR_COLORS = {{ 0: '#FF6B6B', 1: '#4ECDC4', 2: '#FFE66D', 3: '#9B59B6' }};
        const TENSOR_COLORS_LIGHT = {{ 0: '#FFAAAA', 1: '#99E6E0', 2: '#FFF2AA', 3: '#C9A0DC' }};

        // State
        let currentCycle = 0;
        let isPlaying = false;
        let playbackSpeed = 5;   // Slower default
        let stepSize = 1;
        let animationFrame = null;
        let lastFrameTime = 0;
        let selectedL3 = null;

        // Filters
        let filters = {{ A: true, B: true, C: true }};

        // Simulation state
        let l3Contents = {{}};     // l3_id -> Set of tile keys (completed tiles)
        let l3PartialTiles = {{}};  // l3_id -> Map of tileKey -> {{flitsReceived, numFlits, tensor, m, n, k}}
        let packetsInFlight = {{}}; // packet_seq -> {{src, dst, tensor, tile, currentPos}}
        let linkActivity = {{}};    // "srcId-dstId" -> {{tensor, progress, lastCycle}}
        let stats = {{ injected: 0, delivered: 0, hops: 0, flits: 0 }};

        // Initialize L3 contents
        for (let i = 0; i < 16; i++) {{
            l3Contents[i] = new Set();
            l3PartialTiles[i] = new Map();
        }}

        // Format tile label: A0.0, B11.17, etc.
        function formatTileLabel(tensor, m, n, k) {{
            const name = TENSOR_NAMES[tensor] || '?';
            switch (tensor) {{
                case 0: return `${{name}}${{m}}.${{k}}`;      // A[m,k]
                case 1: return `${{name}}${{k}}.${{n}}`;      // B[k,n]
                case 2: return `${{name}}${{m}}.${{n}}`;      // C[m,n]
                default: return `${{name}}${{m}}.${{n}}`;
            }}
        }}

        // Get L3 tile center position
        function getL3Pos(id) {{
            const row = Math.floor(id / MESH_SIZE);
            const col = id % MESH_SIZE;
            // L3 center is offset from cell origin
            return {{
                x: OFFSET + col * CELL_SIZE + L3_SIZE / 2 + 20,
                y: OFFSET + row * CELL_SIZE + L3_SIZE / 2 + 20
            }};
        }}

        // Get router position (center of router circle at top-left corner OF the L3 tile)
        function getRouterPos(id) {{
            const l3Pos = getL3Pos(id);
            // Router sits at top-left corner of L3, overlapping the corner
            return {{
                x: l3Pos.x - L3_SIZE / 2 + ROUTER_SIZE / 3,
                y: l3Pos.y - L3_SIZE / 2 + ROUTER_SIZE / 3
            }};
        }}

        // Get link endpoints
        function getLinkEndpoints(srcId, dstId) {{
            const srcPos = getRouterPos(srcId);
            const dstPos = getRouterPos(dstId);
            return {{ src: srcPos, dst: dstPos }};
        }}

        // Initialize the mesh visualization
        function initMesh() {{
            const flowArrowsLayer = document.getElementById('flow-arrows-layer');
            const linksLayer = document.getElementById('links-layer');
            const l3Layer = document.getElementById('l3-tiles-layer');
            const routersLayer = document.getElementById('routers-layer');

            flowArrowsLayer.innerHTML = '';
            linksLayer.innerHTML = '';
            l3Layer.innerHTML = '';
            routersLayer.innerHTML = '';

            // Draw flow direction indicators (background, very subtle)
            for (let row = 0; row < MESH_SIZE; row++) {{
                // East flow arrow for A tiles
                const l3y = OFFSET + row * CELL_SIZE + ROUTER_OFFSET + ROUTER_SIZE + L3_SIZE / 2;
                const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                arrow.classList.add('flow-arrow', 'east');
                arrow.setAttribute('x1', OFFSET - 10);
                arrow.setAttribute('y1', l3y);
                arrow.setAttribute('x2', OFFSET + MESH_SIZE * CELL_SIZE - 30);
                arrow.setAttribute('y2', l3y);
                flowArrowsLayer.appendChild(arrow);
            }}
            for (let col = 0; col < MESH_SIZE; col++) {{
                // South flow arrow for B tiles
                const l3x = OFFSET + col * CELL_SIZE + ROUTER_OFFSET + ROUTER_SIZE + L3_SIZE / 2;
                const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                arrow.classList.add('flow-arrow', 'south');
                arrow.setAttribute('x1', l3x);
                arrow.setAttribute('y1', OFFSET - 10);
                arrow.setAttribute('x2', l3x);
                arrow.setAttribute('y2', OFFSET + MESH_SIZE * CELL_SIZE - 30);
                flowArrowsLayer.appendChild(arrow);
            }}

            // Draw inter-router links (the NoC interconnect)
            for (let row = 0; row < MESH_SIZE; row++) {{
                for (let col = 0; col < MESH_SIZE; col++) {{
                    const id = row * MESH_SIZE + col;
                    const pos = getRouterPos(id);

                    // East link (Router to Router)
                    if (col < MESH_SIZE - 1) {{
                        const eastPos = getRouterPos(id + 1);
                        const link = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                        link.classList.add('link');
                        link.id = `link-${{id}}-${{id + 1}}`;
                        // Connect from right edge of this router to left edge of east router
                        link.setAttribute('x1', pos.x + ROUTER_SIZE / 2);
                        link.setAttribute('y1', pos.y);
                        link.setAttribute('x2', eastPos.x - ROUTER_SIZE / 2);
                        link.setAttribute('y2', eastPos.y);
                        linksLayer.appendChild(link);
                    }}

                    // South link (Router to Router)
                    if (row < MESH_SIZE - 1) {{
                        const southPos = getRouterPos(id + MESH_SIZE);
                        const link = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                        link.classList.add('link');
                        link.id = `link-${{id}}-${{id + MESH_SIZE}}`;
                        // Connect from bottom edge of this router to top edge of south router
                        link.setAttribute('x1', pos.x);
                        link.setAttribute('y1', pos.y + ROUTER_SIZE / 2);
                        link.setAttribute('x2', southPos.x);
                        link.setAttribute('y2', southPos.y - ROUTER_SIZE / 2);
                        linksLayer.appendChild(link);
                    }}
                }}
            }}

            // Draw LOCAL port connections (Router ↔ L3 tile)
            for (let row = 0; row < MESH_SIZE; row++) {{
                for (let col = 0; col < MESH_SIZE; col++) {{
                    const id = row * MESH_SIZE + col;
                    const rpos = getRouterPos(id);
                    const l3pos = getL3Pos(id);

                    // Local port connection line (diagonal from router into L3)
                    const localLink = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    localLink.classList.add('local-port');
                    localLink.id = `local-${{id}}`;
                    // From router center diagonal into L3
                    localLink.setAttribute('x1', rpos.x);
                    localLink.setAttribute('y1', rpos.y);
                    localLink.setAttribute('x2', rpos.x + 35);
                    localLink.setAttribute('y2', rpos.y + 35);
                    linksLayer.appendChild(localLink);
                }}
            }}

            // Draw L3 tiles (THE MAIN VISUAL ELEMENT)
            for (let row = 0; row < MESH_SIZE; row++) {{
                for (let col = 0; col < MESH_SIZE; col++) {{
                    const id = row * MESH_SIZE + col;
                    const pos = getL3Pos(id);

                    const l3 = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    l3.classList.add('l3-tile');
                    l3.id = `l3-${{id}}`;
                    l3.setAttribute('data-id', id);

                    // L3 background
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.classList.add('l3-tile-bg');
                    rect.setAttribute('x', pos.x - L3_SIZE / 2);
                    rect.setAttribute('y', pos.y - L3_SIZE / 2);
                    rect.setAttribute('width', L3_SIZE);
                    rect.setAttribute('height', L3_SIZE);

                    // L3 label
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.classList.add('l3-label');
                    label.setAttribute('x', pos.x);
                    label.setAttribute('y', pos.y - L3_SIZE / 2 + 14);
                    label.setAttribute('text-anchor', 'middle');
                    label.textContent = `L3[${{row}},${{col}}]`;

                    // Capacity indicator
                    const capacity = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    capacity.classList.add('l3-capacity');
                    capacity.id = `l3-capacity-${{id}}`;
                    capacity.setAttribute('x', pos.x);
                    capacity.setAttribute('y', pos.y + L3_SIZE / 2 - 6);
                    capacity.setAttribute('text-anchor', 'middle');
                    capacity.textContent = '0 tiles';

                    // Container for matrix tiles
                    const tilesContainer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    tilesContainer.id = `l3-tiles-${{id}}`;

                    l3.appendChild(rect);
                    l3.appendChild(label);
                    l3.appendChild(capacity);
                    l3.appendChild(tilesContainer);

                    // Click handler for selection
                    l3.addEventListener('click', () => selectL3(id));

                    l3Layer.appendChild(l3);
                }}
            }}

            // Draw routers (small connection points)
            for (let row = 0; row < MESH_SIZE; row++) {{
                for (let col = 0; col < MESH_SIZE; col++) {{
                    const id = row * MESH_SIZE + col;
                    const pos = getRouterPos(id);

                    const router = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    router.classList.add('router');
                    router.id = `router-${{id}}`;

                    // Router circle
                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    circle.classList.add('router-bg');
                    circle.setAttribute('cx', pos.x);
                    circle.setAttribute('cy', pos.y);
                    circle.setAttribute('r', ROUTER_SIZE / 2);

                    // Router label
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.classList.add('router-label');
                    label.setAttribute('x', pos.x);
                    label.setAttribute('y', pos.y + 3);
                    label.setAttribute('text-anchor', 'middle');
                    label.textContent = `R${{id}}`;

                    router.appendChild(circle);
                    router.appendChild(label);
                    routersLayer.appendChild(router);
                }}
            }}
        }}

        // Update L3 tile display
        function updateL3Display(l3Id) {{
            const tilesContainer = document.getElementById(`l3-tiles-${{l3Id}}`);
            const capacityLabel = document.getElementById(`l3-capacity-${{l3Id}}`);
            if (!tilesContainer) return;

            tilesContainer.innerHTML = '';

            // Combine completed tiles and partial tiles
            const completedTiles = Array.from(l3Contents[l3Id]);
            const partialTiles = Array.from(l3PartialTiles[l3Id].entries());

            // Create display list: completed tiles first, then partial tiles not in completed
            const displayTiles = [];
            completedTiles.forEach(tileKey => {{
                const [tensor, m, n, k] = tileKey.split(',').map(Number);
                displayTiles.push({{ tileKey, tensor, m, n, k, progress: 1.0 }});
            }});
            partialTiles.forEach(([tileKey, info]) => {{
                if (!l3Contents[l3Id].has(tileKey)) {{
                    const progress = info.flitsReceived / info.numFlits;
                    displayTiles.push({{
                        tileKey,
                        tensor: info.tensor,
                        m: info.m,
                        n: info.n,
                        k: info.k,
                        progress
                    }});
                }}
            }});

            capacityLabel.textContent = `${{completedTiles.length}} tiles (${{partialTiles.length}} partial)`;

            const pos = getL3Pos(l3Id);
            const startX = pos.x - L3_SIZE / 2 + 8;
            const startY = pos.y - L3_SIZE / 2 + 24;

            displayTiles.forEach((tile, index) => {{
                const {{ tileKey, tensor, m, n, k, progress }} = tile;

                // Check filter
                const tensorName = TENSOR_NAMES[tensor];
                if (!filters[tensorName]) return;

                const col = index % TILE_COLS;
                const row = Math.floor(index / TILE_COLS);
                if (row >= 3) return; // Max 3 rows visible

                const x = startX + col * (TILE_BLOCK_SIZE + 3);
                const y = startY + row * (TILE_BLOCK_SIZE + 3);

                // Background rect (light color)
                const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                bgRect.setAttribute('x', x);
                bgRect.setAttribute('y', y);
                bgRect.setAttribute('width', TILE_BLOCK_SIZE);
                bgRect.setAttribute('height', TILE_BLOCK_SIZE);
                bgRect.setAttribute('rx', 3);
                bgRect.style.fill = TENSOR_COLORS_LIGHT[tensor] || '#ddd';
                bgRect.style.stroke = TENSOR_COLORS[tensor] || '#999';
                bgRect.style.strokeWidth = '1';

                // Fill rect (shows progress from bottom up)
                const fillHeight = TILE_BLOCK_SIZE * progress;
                const fillRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                fillRect.classList.add('matrix-tile', `tensor-${{tensorName}}`);
                fillRect.setAttribute('x', x);
                fillRect.setAttribute('y', y + TILE_BLOCK_SIZE - fillHeight);
                fillRect.setAttribute('width', TILE_BLOCK_SIZE);
                fillRect.setAttribute('height', fillHeight);
                fillRect.setAttribute('rx', progress < 1 ? 0 : 3);

                // Tile label
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.classList.add('matrix-tile-label');
                label.setAttribute('x', x + TILE_BLOCK_SIZE / 2);
                label.setAttribute('y', y + TILE_BLOCK_SIZE / 2 + 3);
                label.setAttribute('text-anchor', 'middle');
                const progressPct = progress < 1 ? ` ${{Math.round(progress * 100)}}%` : '';
                label.textContent = formatTileLabel(tensor, m, n, k) + progressPct;

                tilesContainer.appendChild(bgRect);
                tilesContainer.appendChild(fillRect);
                tilesContainer.appendChild(label);
            }});

            // Update detail panel if selected
            if (selectedL3 === l3Id) {{
                updateL3DetailPanel(l3Id);
            }}
        }}

        // Select L3 for detail view
        function selectL3(id) {{
            // Deselect previous
            if (selectedL3 !== null) {{
                document.getElementById(`l3-${{selectedL3}}`)?.classList.remove('selected');
            }}

            selectedL3 = id;
            document.getElementById(`l3-${{id}}`)?.classList.add('selected');
            updateL3DetailPanel(id);
        }}

        // Update L3 detail panel
        function updateL3DetailPanel(l3Id) {{
            const header = document.querySelector('.l3-detail-header');
            const list = document.getElementById('l3-tile-list');

            const row = Math.floor(l3Id / MESH_SIZE);
            const col = l3Id % MESH_SIZE;
            header.textContent = `L3[${{row}},${{col}}] Contents:`;

            list.innerHTML = '';
            const tiles = Array.from(l3Contents[l3Id]);

            if (tiles.length === 0) {{
                list.innerHTML = '<span style="color: #666; font-size: 0.85em;">Empty</span>';
                return;
            }}

            tiles.forEach(tileKey => {{
                const [tensor, m, n, k] = tileKey.split(',').map(Number);
                const chip = document.createElement('span');
                chip.classList.add('l3-tile-chip', `tensor-${{TENSOR_NAMES[tensor]}}`);
                chip.textContent = formatTileLabel(tensor, m, n, k);
                list.appendChild(chip);
            }});
        }}

        // Show packet traveling on link
        function showPacketOnLink(packetSeq, srcRouter, dstRouter, tensor, m, n, k, progress = 0.5) {{
            const packetsLayer = document.getElementById('packets-layer');
            let packet = document.getElementById(`packet-${{packetSeq}}`);

            const endpoints = getLinkEndpoints(srcRouter, dstRouter);
            const x = endpoints.src.x + (endpoints.dst.x - endpoints.src.x) * progress;
            const y = endpoints.src.y + (endpoints.dst.y - endpoints.src.y) * progress;

            const tensorName = TENSOR_NAMES[tensor] || 'A';

            if (!packet) {{
                packet = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                packet.classList.add('packet-in-flight');
                packet.id = `packet-${{packetSeq}}`;

                const body = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                body.classList.add('packet-body', `tensor-${{tensorName}}`);
                body.setAttribute('x', -PACKET_SIZE / 2);
                body.setAttribute('y', -PACKET_SIZE / 2);
                body.setAttribute('width', PACKET_SIZE);
                body.setAttribute('height', PACKET_SIZE);
                body.setAttribute('rx', 3);

                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.classList.add('packet-label');
                label.setAttribute('x', 0);
                label.setAttribute('y', 3);
                label.setAttribute('text-anchor', 'middle');
                label.textContent = formatTileLabel(tensor, m, n, k);

                packet.appendChild(body);
                packet.appendChild(label);
                packetsLayer.appendChild(packet);
            }}

            packet.setAttribute('transform', `translate(${{x}}, ${{y}})`);
        }}

        // Remove packet from view
        function removePacket(packetSeq) {{
            const packet = document.getElementById(`packet-${{packetSeq}}`);
            if (packet) {{
                packet.remove();
            }}
        }}

        // Update link visual
        function setLinkActive(srcId, dstId, active, tensor = null) {{
            const linkId = srcId < dstId ? `link-${{srcId}}-${{dstId}}` : `link-${{dstId}}-${{srcId}}`;
            const link = document.getElementById(linkId);
            if (link) {{
                link.classList.toggle('active', active);
                if (active && tensor !== null) {{
                    link.style.stroke = TENSOR_COLORS[tensor];
                }} else {{
                    link.style.stroke = '';
                }}
            }}
        }}

        // Update router visual
        function setRouterActive(id, active) {{
            const router = document.getElementById(`router-${{id}}`);
            if (router) {{
                router.classList.toggle('active', active);
            }}
        }}

        // Flash L3 receiving
        function flashL3Receiving(id) {{
            const l3 = document.getElementById(`l3-${{id}}`);
            if (l3) {{
                l3.classList.add('receiving');
                setTimeout(() => l3.classList.remove('receiving'), 500);
            }}
        }}

        // Process a single event
        function processEvent(event) {{
            const type = event.type;
            const routerId = event.router_id;
            const packetSeq = event.packet_seq;
            const tensor = event.tensor;
            const tileKey = `${{tensor}},${{event.m_tile}},${{event.n_tile}},${{event.k_tile}}`;

            // Check filter
            const tensorName = TENSOR_NAMES[tensor];
            if (!filters[tensorName]) return;

            switch (type) {{
                case EVENT_INJECT:
                    // Packet starts at source router
                    packetsInFlight[packetSeq] = {{
                        src: routerId,
                        currentRouter: routerId,
                        tensor: tensor,
                        tile: {{ m: event.m_tile, n: event.n_tile, k: event.k_tile }}
                    }};
                    setRouterActive(routerId, true);
                    setTimeout(() => setRouterActive(routerId, false), 200);

                    // Add tile to source L3 cache (it has the tile before sending)
                    l3Contents[routerId].add(tileKey);
                    updateL3Display(routerId);

                    stats.injected++;
                    break;

                case EVENT_HOP:
                    // Packet arrived at new router
                    if (packetsInFlight[packetSeq]) {{
                        const pkt = packetsInFlight[packetSeq];
                        const prevRouter = pkt.currentRouter;

                        // Show packet moving on link
                        showPacketOnLink(packetSeq, prevRouter, routerId, tensor,
                                        event.m_tile, event.n_tile, event.k_tile, 1.0);

                        // Animate link
                        setLinkActive(prevRouter, routerId, true, tensor);
                        setTimeout(() => setLinkActive(prevRouter, routerId, false), 150);

                        // Flash router
                        setRouterActive(routerId, true);
                        setTimeout(() => setRouterActive(routerId, false), 100);

                        pkt.currentRouter = routerId;
                        stats.hops++;
                    }}
                    break;

                case EVENT_EJECT:
                    // Packet delivered to L3
                    removePacket(packetSeq);
                    delete packetsInFlight[packetSeq];

                    // Add tile to L3 cache
                    l3Contents[routerId].add(tileKey);
                    updateL3Display(routerId);
                    flashL3Receiving(routerId);

                    stats.delivered++;
                    break;

                case EVENT_BLOCKED:
                    // Show contention
                    const router = document.getElementById(`router-${{routerId}}`);
                    if (router) {{
                        router.classList.add('congested');
                        setTimeout(() => router.classList.remove('congested'), 300);
                    }}
                    break;

                case EVENT_FLIT_SEND:
                    // Show link activity for FLIT transfer
                    const srcRouter = event.src_router || 0;
                    const dstRouter = event.dst_router || 0;
                    const flitProgress = (event.flit_index || 0) / (event.num_flits || 1);

                    // Update link color based on tensor and progress
                    setLinkActive(srcRouter, dstRouter, true, tensor);

                    // Track link activity for animation
                    const linkKey = `${{srcRouter}}-${{dstRouter}}`;
                    linkActivity[linkKey] = {{
                        tensor: tensor,
                        progress: flitProgress,
                        lastCycle: event.cycle
                    }};

                    stats.flits++;
                    break;

                case EVENT_FLIT_ARRIVE:
                    // Update partial tile fill state
                    const flitIndex = event.flit_index || 0;
                    const numFlits = event.num_flits || 1;

                    // Initialize or update partial tile info
                    if (!l3PartialTiles[routerId].has(tileKey)) {{
                        l3PartialTiles[routerId].set(tileKey, {{
                            flitsReceived: 0,
                            numFlits: numFlits,
                            tensor: tensor,
                            m: event.m_tile,
                            n: event.n_tile,
                            k: event.k_tile
                        }});
                    }}

                    const tileInfo = l3PartialTiles[routerId].get(tileKey);
                    // Update flits received (use flit_index as progress indicator)
                    tileInfo.flitsReceived = Math.max(tileInfo.flitsReceived, flitIndex + 1);

                    // Update display to show progressive fill
                    updateL3Display(routerId);

                    stats.flits++;
                    break;
            }}

            updateStats();
            // Only log major events, not every FLIT
            if (type !== EVENT_FLIT_SEND && type !== EVENT_FLIT_ARRIVE) {{
                addEventToLog(event);
            }}
        }}

        // Update statistics display
        function updateStats() {{
            document.getElementById('stat-packets').textContent = stats.injected;
            document.getElementById('stat-delivered').textContent = stats.delivered;
            document.getElementById('stat-in-flight').textContent = Object.keys(packetsInFlight).length;
            document.getElementById('stat-hops').textContent = stats.hops;
        }}

        // Add event to log
        function addEventToLog(event) {{
            const log = document.getElementById('event-log');
            const entry = document.createElement('div');

            const typeName = EVENT_NAMES[event.type] || 'UNKNOWN';
            entry.classList.add('event-log-entry', typeName);

            const tileLabel = formatTileLabel(event.tensor, event.m_tile, event.n_tile, event.k_tile);
            const row = Math.floor(event.router_id / MESH_SIZE);
            const col = event.router_id % MESH_SIZE;

            let text = `[${{event.cycle}}] `;
            switch (event.type) {{
                case EVENT_INJECT:
                    text += `INJ ${{tileLabel}} @ R${{event.router_id}}`;
                    break;
                case EVENT_HOP:
                    text += `HOP ${{tileLabel}} -> R${{event.router_id}}`;
                    break;
                case EVENT_EJECT:
                    text += `DEL ${{tileLabel}} -> L3[${{row}},${{col}}]`;
                    break;
                case EVENT_BLOCKED:
                    text += `BLK ${{tileLabel}} @ R${{event.router_id}}`;
                    break;
                default:
                    text += `${{typeName}} @ R${{event.router_id}}`;
            }}

            entry.textContent = text;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;

            while (log.children.length > 100) log.removeChild(log.firstChild);
        }}

        // Timeline update
        function updateTimeline() {{
            const progress = (currentCycle / maxCycle) * 100;
            document.getElementById('timeline-progress').style.width = `${{progress}}%`;
            document.getElementById('timeline-cursor').style.left = `${{progress}}%`;
            document.getElementById('cycle-counter').textContent = currentCycle;
        }}

        // Reset state
        function resetState() {{
            currentCycle = 0;
            packetsInFlight = {{}};
            linkActivity = {{}};
            stats = {{ injected: 0, delivered: 0, hops: 0, flits: 0 }};

            for (let i = 0; i < 16; i++) {{
                l3Contents[i] = new Set();
                l3PartialTiles[i] = new Map();
                updateL3Display(i);
                setRouterActive(i, false);
            }}

            // Clear all packets
            document.getElementById('packets-layer').innerHTML = '';

            // Reset links
            document.querySelectorAll('.link').forEach(link => {{
                link.classList.remove('active', 'busy', 'congested');
                link.style.stroke = '';
            }});

            document.getElementById('event-log').innerHTML = '';
            updateStats();
            updateTimeline();
        }}

        // Animation loop
        function animate(timestamp) {{
            if (!isPlaying) return;

            const elapsed = timestamp - lastFrameTime;
            const msPerCycle = 1000 / (playbackSpeed * 2);  // Slower animation

            if (elapsed >= msPerCycle) {{
                lastFrameTime = timestamp;
                advanceCycles(stepSize);
            }}

            if (currentCycle < maxCycle) {{
                animationFrame = requestAnimationFrame(animate);
            }} else {{
                stopPlayback();
            }}
        }}

        function advanceCycles(n) {{
            const targetCycle = Math.min(currentCycle + n, maxCycle);

            for (const event of events) {{
                if (event.cycle > currentCycle && event.cycle <= targetCycle) {{
                    processEvent(event);
                }}
            }}

            currentCycle = targetCycle;
            updateTimeline();
        }}

        function stepToNextEvent() {{
            const nextEvent = events.find(e => e.cycle > currentCycle);
            if (nextEvent) {{
                advanceCycles(nextEvent.cycle - currentCycle);
            }}
        }}

        function stepBack() {{
            const prevEvents = events.filter(e => e.cycle < currentCycle);
            if (prevEvents.length > 0) {{
                const targetCycle = Math.max(0, currentCycle - stepSize);
                resetState();
                if (targetCycle > 0) {{
                    advanceCycles(targetCycle);
                }}
            }} else {{
                resetState();
            }}
        }}

        function startPlayback() {{
            isPlaying = true;
            lastFrameTime = performance.now();
            document.getElementById('btn-play').innerHTML = '&#10074;&#10074; Pause';
            document.getElementById('btn-play').classList.add('active');
            animationFrame = requestAnimationFrame(animate);
        }}

        function stopPlayback() {{
            isPlaying = false;
            document.getElementById('btn-play').innerHTML = '&#9654; Play';
            document.getElementById('btn-play').classList.remove('active');
            if (animationFrame) cancelAnimationFrame(animationFrame);
        }}

        function togglePlayback() {{
            if (isPlaying) {{
                stopPlayback();
            }} else {{
                if (currentCycle >= maxCycle) resetState();
                startPlayback();
            }}
        }}

        // Initialize timeline event markers
        function initTimelineEvents() {{
            const container = document.getElementById('timeline-events');
            container.innerHTML = '';

            for (const event of events) {{
                const marker = document.createElement('div');
                marker.classList.add('timeline-event');
                marker.style.left = `${{(event.cycle / maxCycle) * 100}}%`;

                switch (event.type) {{
                    case EVENT_INJECT:
                        marker.style.background = '#00ff88';
                        break;
                    case EVENT_HOP:
                        marker.style.background = '#00aaff';
                        marker.style.height = '25%';
                        break;
                    case EVENT_EJECT:
                        marker.style.background = '#ffaa00';
                        break;
                    case EVENT_BLOCKED:
                        marker.style.background = '#ff4444';
                        break;
                    default:
                        marker.style.background = '#666';
                }}

                container.appendChild(marker);
            }}
        }}

        // Filter toggle
        function toggleFilter(tensor) {{
            filters[tensor] = !filters[tensor];
            document.getElementById(`filter-${{tensor}}`).classList.toggle('active', filters[tensor]);

            // Refresh all L3 displays
            for (let i = 0; i < 16; i++) {{
                updateL3Display(i);
            }}
        }}

        // Initialize on load
        document.addEventListener('DOMContentLoaded', () => {{
            initMesh();
            initTimelineEvents();

            // Playback controls
            document.getElementById('btn-play').addEventListener('click', togglePlayback);
            document.getElementById('btn-step').addEventListener('click', () => {{
                stopPlayback();
                stepToNextEvent();
            }});
            document.getElementById('btn-step-back').addEventListener('click', () => {{
                stopPlayback();
                stepBack();
            }});
            document.getElementById('btn-reset').addEventListener('click', () => {{
                stopPlayback();
                resetState();
            }});

            // Speed control
            document.getElementById('speed-slider').addEventListener('input', (e) => {{
                playbackSpeed = parseInt(e.target.value, 10);
                document.getElementById('speed-value').textContent = `${{(playbackSpeed / 5).toFixed(1)}}x`;
            }});

            // Step size control
            document.getElementById('step-slider').addEventListener('input', (e) => {{
                stepSize = parseInt(e.target.value, 10);
                document.getElementById('step-value').textContent = stepSize;
            }});

            // Timeline click
            document.getElementById('timeline').addEventListener('click', (e) => {{
                stopPlayback();
                const rect = e.currentTarget.getBoundingClientRect();
                const pct = (e.clientX - rect.left) / rect.width;
                const targetCycle = Math.floor(pct * maxCycle);
                resetState();
                advanceCycles(targetCycle);
            }});

            // Filter buttons
            document.getElementById('filter-A').addEventListener('click', () => toggleFilter('A'));
            document.getElementById('filter-B').addEventListener('click', () => toggleFilter('B'));
            document.getElementById('filter-C').addEventListener('click', () => toggleFilter('C'));

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {{
                switch (e.key) {{
                    case ' ':
                        e.preventDefault();
                        togglePlayback();
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        stopPlayback();
                        stepToNextEvent();
                        break;
                    case 'ArrowLeft':
                        e.preventDefault();
                        stopPlayback();
                        stepBack();
                        break;
                    case 'r':
                    case 'R':
                        stopPlayback();
                        resetState();
                        break;
                }}
            }});
        }});
    </script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(
        description='Generate NoC animation from trace data'
    )
    parser.add_argument('trace_file', help='Path to NoC trace CSV file')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output HTML file (default: noc_animation.html)')
    parser.add_argument('--title', '-t', default='KPU NoC Animation',
                        help='Title for the visualization')

    args = parser.parse_args()

    # Check input file
    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"Error: Trace file not found: {args.trace_file}")
        sys.exit(1)

    # Load trace
    print(f"Loading NoC trace from: {args.trace_file}")
    events = load_noc_trace(args.trace_file)
    print(f"Loaded {len(events)} events")

    # Generate HTML
    html = generate_noc_html(events, args.title)

    # Determine output file
    output_file = args.output or 'noc_animation.html'

    # Write output
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Generated: {output_file}")
    print(f"Open in a browser to view the animation")


if __name__ == '__main__':
    main()
