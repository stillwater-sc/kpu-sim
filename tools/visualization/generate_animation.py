#!/usr/bin/env python3
"""
Generate Self-Contained HTML Animation

Takes a tile flow trace CSV and embeds it into the HTML visualization,
creating a single file that can be shared with investors and customers.

Usage:
    python3 generate_animation.py tile_flow_trace.csv output.html
    python3 generate_animation.py tile_flow_trace.csv  # outputs to tile_flow_animation.html
"""

import argparse
import json
import sys
from pathlib import Path


def load_csv(filename: str) -> list:
    """Load trace CSV file and return list of events."""
    events = []
    with open(filename, 'r') as f:
        headers = f.readline().strip().split(',')
        for line in f:
            values = line.strip().split(',')
            event = {}
            for h, v in zip(headers, values):
                if h in ['cycle', 'l3_id', 'tensor', 'm_tile', 'n_tile', 'k_tile', 'src_l3', 'dst_l3', 'bytes']:
                    event[h] = int(v)
                else:
                    event[h] = v
            events.append(event)
    return events


def generate_html(events: list, title: str = "KPU Tile Flow Animation") -> str:
    """Generate complete HTML with embedded trace data."""

    max_cycle = max(e['cycle'] for e in events) if events else 1000

    # The HTML template with embedded data
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
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 30px; }}
        h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #4CAF50, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .subtitle {{ color: #888; font-size: 1.1em; }}
        .main-content {{
            display: flex;
            gap: 30px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .mesh-container {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        #mesh-svg {{ display: block; }}
        .l3-tile {{ cursor: pointer; transition: all 0.3s ease; }}
        .l3-tile:hover {{ filter: brightness(1.2); }}
        .l3-tile-bg {{ fill: #2a2a4a; stroke: #444; stroke-width: 2; }}
        .l3-tile.active .l3-tile-bg {{
            stroke: #4CAF50;
            stroke-width: 3;
            filter: drop-shadow(0 0 10px rgba(76, 175, 80, 0.5));
        }}
        .l3-label {{ fill: #888; font-size: 12px; font-weight: bold; }}
        .tensor-block {{ opacity: 0.9; transition: all 0.2s ease; }}
        .tensor-block.tensor-A {{ fill: #FF6B6B; }}
        .tensor-block.tensor-B {{ fill: #4ECDC4; }}
        .tensor-block.tensor-C {{ fill: #FFE66D; }}
        .transfer-arrow {{
            stroke-width: 4;
            fill: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        .transfer-arrow.active {{
            opacity: 0.8;
            animation: pulse 0.5s ease infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ stroke-width: 4; }}
            50% {{ stroke-width: 6; }}
        }}
        .controls {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 20px;
            width: 350px;
        }}
        .control-group {{ margin-bottom: 20px; }}
        .control-group h3 {{
            color: #4CAF50;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .play-controls {{
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 15px;
        }}
        .btn {{
            background: linear-gradient(135deg, #4CAF50, #2E7D32);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
        }}
        .btn-secondary {{ background: linear-gradient(135deg, #555, #333); }}
        .speed-control {{ display: flex; align-items: center; gap: 10px; }}
        .speed-control input {{
            flex: 1;
            -webkit-appearance: none;
            height: 8px;
            background: #333;
            border-radius: 4px;
        }}
        .speed-control input::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #4CAF50;
            border-radius: 50%;
            cursor: pointer;
        }}
        .timeline {{
            position: relative;
            height: 60px;
            background: #1a1a2e;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 10px;
        }}
        .timeline-progress {{
            position: absolute;
            height: 100%;
            background: linear-gradient(90deg, rgba(76, 175, 80, 0.3), rgba(33, 150, 243, 0.3));
            transition: width 0.1s ease;
        }}
        .timeline-cursor {{
            position: absolute;
            width: 3px;
            height: 100%;
            background: #4CAF50;
            box-shadow: 0 0 10px #4CAF50;
        }}
        .timeline-events {{ position: absolute; width: 100%; height: 100%; }}
        .timeline-event {{
            position: absolute;
            width: 3px;
            height: 30%;
            bottom: 0;
            border-radius: 2px 2px 0 0;
        }}
        .cycle-display {{
            display: flex;
            justify-content: space-between;
            color: #888;
            font-size: 0.9em;
        }}
        .current-cycle {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin: 10px 0;
        }}
        .event-log {{
            background: #1a1a2e;
            border-radius: 8px;
            padding: 10px;
            height: 200px;
            overflow-y: auto;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.85em;
        }}
        .event-log-entry {{ padding: 4px 8px; border-radius: 4px; margin-bottom: 4px; }}
        .event-log-entry.L3_SEND_START {{ background: rgba(76, 175, 80, 0.2); border-left: 3px solid #4CAF50; }}
        .event-log-entry.L3_TO_L2_START {{ background: rgba(33, 150, 243, 0.2); border-left: 3px solid #2196F3; }}
        .event-log-entry.BARRIER_START {{ background: rgba(255, 152, 0, 0.2); border-left: 3px solid #FF9800; }}
        .event-log-entry.L3_RECEIVE {{ background: rgba(139, 195, 74, 0.2); border-left: 3px solid #8BC34A; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin-top: 10px; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.85em; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 4px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: #4CAF50; }}
        .stat-label {{ font-size: 0.75em; color: #888; text-transform: uppercase; }}
        .trace-info {{
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid #4CAF50;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 0.85em;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>KPU Tile Flow Visualization</h1>
            <p class="subtitle">Watch data tiles hop and forward through the L3 mesh</p>
        </header>
        <div class="main-content">
            <div class="mesh-container">
                <svg id="mesh-svg" width="520" height="520" viewBox="0 0 520 520">
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#4CAF50"/>
                        </marker>
                    </defs>
                    <g id="l3-tiles"></g>
                    <g id="transfer-arrows"></g>
                    <g id="moving-tensors"></g>
                </svg>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FF6B6B;"></div>
                        <span>Tensor A (Input)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4ECDC4;"></div>
                        <span>Tensor B (Weights)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FFE66D;"></div>
                        <span>Tensor C (Output)</span>
                    </div>
                </div>
            </div>
            <div class="controls">
                <div class="trace-info">
                    Trace: {len(events)} events, {max_cycle} cycles
                </div>
                <div class="control-group">
                    <h3>Playback</h3>
                    <div class="play-controls">
                        <button class="btn btn-secondary" id="btn-step-back">&#9664;&#9664;</button>
                        <button class="btn" id="btn-play">&#9654; Play</button>
                        <button class="btn btn-secondary" id="btn-step">&#9654;&#9654;</button>
                    </div>
                    <div class="speed-control">
                        <span>Speed:</span>
                        <input type="range" id="speed-slider" min="1" max="100" value="50">
                        <span id="speed-value">1x</span>
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
                    <h3>Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="stat-transfers">0</div>
                            <div class="stat-label">L3 Transfers</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-l2-pushes">0</div>
                            <div class="stat-label">L2 Pushes</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-barriers">0</div>
                            <div class="stat-label">Barriers</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-active">0</div>
                            <div class="stat-label">Active Tiles</div>
                        </div>
                    </div>
                </div>
                <div class="control-group">
                    <h3>Event Log</h3>
                    <div class="event-log" id="event-log"></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        // Embedded trace data
        const events = {json.dumps(events)};
        const maxCycle = {max_cycle};

        // Configuration
        const MESH_ROWS = 4, MESH_COLS = 4;
        const TILE_SIZE = 100, TILE_GAP = 30, TENSOR_SIZE = 25;

        // State
        let currentCycle = 0;
        let isPlaying = false;
        let playbackSpeed = 50;
        let animationFrame = null;
        let lastFrameTime = 0;

        const TENSOR_NAMES = {{ 0: 'A', 1: 'B', 2: 'C', 3: 'Bias' }};
        const TENSOR_COLORS = {{ 0: '#FF6B6B', 1: '#4ECDC4', 2: '#FFE66D', 3: '#9B59B6' }};

        let l3TileState = {{}};
        for (let i = 0; i < 16; i++) l3TileState[i] = {{ tensors: new Set(), active: false }};

        let stats = {{ transfers: 0, l2Pushes: 0, barriers: 0 }};

        function initMesh() {{
            const tilesGroup = document.getElementById('l3-tiles');
            tilesGroup.innerHTML = '';
            for (let row = 0; row < MESH_ROWS; row++) {{
                for (let col = 0; col < MESH_COLS; col++) {{
                    const id = row * MESH_COLS + col;
                    const x = col * (TILE_SIZE + TILE_GAP) + 10;
                    const y = row * (TILE_SIZE + TILE_GAP) + 10;
                    const tile = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    tile.classList.add('l3-tile');
                    tile.id = `l3-tile-${{id}}`;
                    tile.setAttribute('transform', `translate(${{x}}, ${{y}})`);
                    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    bg.classList.add('l3-tile-bg');
                    bg.setAttribute('width', TILE_SIZE);
                    bg.setAttribute('height', TILE_SIZE);
                    bg.setAttribute('rx', 8);
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.classList.add('l3-label');
                    label.setAttribute('x', TILE_SIZE / 2);
                    label.setAttribute('y', 15);
                    label.setAttribute('text-anchor', 'middle');
                    label.textContent = `L3[${{id}}]`;
                    const tensorGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    tensorGroup.id = `tensors-${{id}}`;
                    tensorGroup.setAttribute('transform', 'translate(10, 25)');
                    tile.appendChild(bg);
                    tile.appendChild(label);
                    tile.appendChild(tensorGroup);
                    tilesGroup.appendChild(tile);
                }}
            }}
        }}

        function getTileCenter(id) {{
            const row = Math.floor(id / MESH_COLS);
            const col = id % MESH_COLS;
            return {{
                x: col * (TILE_SIZE + TILE_GAP) + 10 + TILE_SIZE / 2,
                y: row * (TILE_SIZE + TILE_GAP) + 10 + TILE_SIZE / 2
            }};
        }}

        function updateTileTensors(tileId) {{
            const tensorGroup = document.getElementById(`tensors-${{tileId}}`);
            if (!tensorGroup) return;
            tensorGroup.innerHTML = '';
            const state = l3TileState[tileId];
            let index = 0;
            for (const tensorKey of state.tensors) {{
                const [tensor, m, n, k] = tensorKey.split(',').map(Number);
                const row = Math.floor(index / 3);
                const col = index % 3;
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.classList.add('tensor-block', `tensor-${{TENSOR_NAMES[tensor]}}`);
                rect.setAttribute('x', col * (TENSOR_SIZE + 3));
                rect.setAttribute('y', row * (TENSOR_SIZE + 3));
                rect.setAttribute('width', TENSOR_SIZE);
                rect.setAttribute('height', TENSOR_SIZE);
                rect.setAttribute('rx', 4);
                rect.setAttribute('fill', TENSOR_COLORS[tensor]);
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', col * (TENSOR_SIZE + 3) + TENSOR_SIZE / 2);
                text.setAttribute('y', row * (TENSOR_SIZE + 3) + TENSOR_SIZE / 2 + 4);
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('fill', '#000');
                text.setAttribute('font-size', '10px');
                text.setAttribute('font-weight', 'bold');
                text.textContent = TENSOR_NAMES[tensor];
                tensorGroup.appendChild(rect);
                tensorGroup.appendChild(text);
                index++;
            }}
        }}

        function showTransfer(srcId, dstId, tensor, duration = 300) {{
            const arrowsGroup = document.getElementById('transfer-arrows');
            const src = getTileCenter(srcId);
            const dst = getTileCenter(dstId);
            const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            arrow.classList.add('transfer-arrow');
            arrow.setAttribute('x1', src.x);
            arrow.setAttribute('y1', src.y);
            arrow.setAttribute('x2', dst.x);
            arrow.setAttribute('y2', dst.y);
            arrow.setAttribute('stroke', TENSOR_COLORS[tensor] || '#4CAF50');
            arrow.setAttribute('marker-end', 'url(#arrowhead)');
            arrowsGroup.appendChild(arrow);
            setTimeout(() => arrow.classList.add('active'), 10);
            setTimeout(() => {{
                arrow.classList.remove('active');
                setTimeout(() => arrow.remove(), 300);
            }}, duration);
        }}

        function processEvent(event) {{
            const tensorKey = `${{event.tensor}},${{event.m_tile}},${{event.n_tile}},${{event.k_tile}}`;
            switch (event.event_type) {{
                case 'L3_SEND_START':
                    showTransfer(event.src_l3, event.dst_l3, event.tensor);
                    l3TileState[event.dst_l3].tensors.add(tensorKey);
                    updateTileTensors(event.dst_l3);
                    stats.transfers++;
                    break;
                case 'L3_TO_L2_START':
                    l3TileState[event.l3_id].tensors.delete(tensorKey);
                    updateTileTensors(event.l3_id);
                    stats.l2Pushes++;
                    break;
                case 'BARRIER_START':
                    stats.barriers++;
                    document.getElementById(`l3-tile-${{event.l3_id}}`).classList.add('active');
                    break;
                case 'BARRIER_COMPLETE':
                    document.getElementById(`l3-tile-${{event.l3_id}}`).classList.remove('active');
                    break;
                case 'DMA_LOAD_START':
                case 'DMA_LOAD_COMPLETE':
                    l3TileState[event.l3_id].tensors.add(tensorKey);
                    updateTileTensors(event.l3_id);
                    break;
            }}
            updateStats();
            addEventToLog(event);
        }}

        function updateStats() {{
            document.getElementById('stat-transfers').textContent = stats.transfers;
            document.getElementById('stat-l2-pushes').textContent = stats.l2Pushes;
            document.getElementById('stat-barriers').textContent = stats.barriers;
            let activeTiles = 0;
            for (let i = 0; i < 16; i++) {{
                if (l3TileState[i].tensors.size > 0) activeTiles++;
            }}
            document.getElementById('stat-active').textContent = activeTiles;
        }}

        function addEventToLog(event) {{
            const log = document.getElementById('event-log');
            const entry = document.createElement('div');
            entry.classList.add('event-log-entry', event.event_type);
            const tensorName = TENSOR_NAMES[event.tensor] || '?';
            let text = `[${{event.cycle}}] L3[${{event.l3_id}}] ${{event.event_type}}`;
            if (event.event_type === 'L3_SEND_START') {{
                text = `[${{event.cycle}}] ${{tensorName}}[${{event.m_tile}},${{event.n_tile}}] L3[${{event.src_l3}}]->L3[${{event.dst_l3}}]`;
            }} else if (event.event_type === 'L3_TO_L2_START') {{
                text = `[${{event.cycle}}] ${{tensorName}}[${{event.m_tile}},${{event.n_tile}}] L3[${{event.l3_id}}]->L2`;
            }}
            entry.textContent = text;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
            while (log.children.length > 50) log.removeChild(log.firstChild);
        }}

        function updateTimeline() {{
            const progress = (currentCycle / maxCycle) * 100;
            document.getElementById('timeline-progress').style.width = `${{progress}}%`;
            document.getElementById('timeline-cursor').style.left = `${{progress}}%`;
            document.getElementById('cycle-counter').textContent = currentCycle;
        }}

        function resetState() {{
            currentCycle = 0;
            stats = {{ transfers: 0, l2Pushes: 0, barriers: 0 }};
            for (let i = 0; i < 16; i++) {{
                l3TileState[i] = {{ tensors: new Set(), active: false }};
                updateTileTensors(i);
                document.getElementById(`l3-tile-${{i}}`).classList.remove('active');
            }}
            document.getElementById('event-log').innerHTML = '';
            updateStats();
            updateTimeline();
        }}

        function animate(timestamp) {{
            if (!isPlaying) return;
            const elapsed = timestamp - lastFrameTime;
            const cyclesPerSecond = playbackSpeed * 20;
            const cyclesToAdvance = Math.floor(elapsed * cyclesPerSecond / 1000);
            if (cyclesToAdvance > 0) {{
                lastFrameTime = timestamp;
                advanceCycles(cyclesToAdvance);
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
            if (nextEvent) advanceCycles(nextEvent.cycle - currentCycle);
        }}

        function stepBack() {{
            const prevEvents = events.filter(e => e.cycle < currentCycle);
            if (prevEvents.length > 0) {{
                const targetCycle = prevEvents[prevEvents.length - 1].cycle;
                resetState();
                advanceCycles(targetCycle);
            }} else {{
                resetState();
            }}
        }}

        function startPlayback() {{
            isPlaying = true;
            lastFrameTime = performance.now();
            document.getElementById('btn-play').innerHTML = '&#10074;&#10074; Pause';
            animationFrame = requestAnimationFrame(animate);
        }}

        function stopPlayback() {{
            isPlaying = false;
            document.getElementById('btn-play').innerHTML = '&#9654; Play';
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

        function initTimelineEvents() {{
            const container = document.getElementById('timeline-events');
            container.innerHTML = '';
            for (const event of events) {{
                const marker = document.createElement('div');
                marker.classList.add('timeline-event');
                marker.style.left = `${{(event.cycle / maxCycle) * 100}}%`;
                if (event.event_type === 'L3_SEND_START') marker.style.background = '#4CAF50';
                else if (event.event_type === 'L3_TO_L2_START') marker.style.background = '#2196F3';
                else if (event.event_type === 'BARRIER_START') marker.style.background = '#FF9800';
                container.appendChild(marker);
            }}
        }}

        document.addEventListener('DOMContentLoaded', () => {{
            initMesh();
            initTimelineEvents();
            document.getElementById('btn-play').addEventListener('click', togglePlayback);
            document.getElementById('btn-step').addEventListener('click', () => {{
                stopPlayback();
                stepToNextEvent();
            }});
            document.getElementById('btn-step-back').addEventListener('click', () => {{
                stopPlayback();
                stepBack();
            }});
            document.getElementById('speed-slider').addEventListener('input', (e) => {{
                playbackSpeed = parseInt(e.target.value, 10);
                document.getElementById('speed-value').textContent = `${{(playbackSpeed / 50).toFixed(1)}}x`;
            }});
            document.getElementById('timeline').addEventListener('click', (e) => {{
                stopPlayback();
                const rect = e.currentTarget.getBoundingClientRect();
                const pct = (e.clientX - rect.left) / rect.width;
                const targetCycle = Math.floor(pct * maxCycle);
                resetState();
                advanceCycles(targetCycle);
            }});
        }});
    </script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(
        description='Generate self-contained HTML animation from trace data'
    )
    parser.add_argument('trace_file', help='Path to trace CSV file')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output HTML file (default: tile_flow_animation.html)')
    parser.add_argument('--title', '-t', default='KPU Tile Flow Animation',
                        help='Title for the visualization')

    args = parser.parse_args()

    # Check input file
    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"Error: Trace file not found: {args.trace_file}")
        sys.exit(1)

    # Load trace
    print(f"Loading trace from: {args.trace_file}")
    events = load_csv(args.trace_file)
    print(f"Loaded {len(events)} events")

    # Generate HTML
    html = generate_html(events, args.title)

    # Determine output file
    output_file = args.output or 'tile_flow_animation.html'

    # Write output
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Generated: {output_file}")
    print(f"Open in a browser to view the animation")


if __name__ == '__main__':
    main()
