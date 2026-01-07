#!/usr/bin/env python3
"""
serve-trace.py - Local HTTP server for LPDDR5 trace visualization

Serves HTML visualization tools with automatic trace file injection.
Works with swimlane, timing, blockdiagram, and bank_analyzer views.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

Usage:
    ./serve-trace.py <trace.json>                    # Serve and open browser
    ./serve-trace.py <trace.json> --port 8888        # Custom port
    ./serve-trace.py <trace.json> --no-browser       # Server only
    ./serve-trace.py <trace.json> --viewer swimlane  # Specific viewer
"""

import argparse
import http.server
import json
import os
import platform
import shutil
import socketserver
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Find project root (contains traces/ directory)
def find_project_root():
    """Find the KPU-SIM project root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "traces").is_dir() and (current / "CMakeLists.txt").is_file():
            return current
        current = current.parent
    # Fallback to current working directory
    cwd = Path.cwd()
    if (cwd / "traces").is_dir():
        return cwd
    return None

PROJECT_ROOT = find_project_root()

# Viewer configurations
VIEWERS = {
    "swimlane": {
        "file": "swimlane.html",
        "name": "Swimlane Pipeline View",
        "description": "Horizontal pipeline visualization"
    },
    "timing": {
        "file": "timing.html",
        "name": "Timing Diagram",
        "description": "Classic waveform-style DRAM timing"
    },
    "blockdiagram": {
        "file": "blockdiagram.html",
        "name": "Block Diagram Animation",
        "description": "Animated architectural view"
    },
    "bank_analyzer": {
        "file": "bank_analyzer.html",
        "name": "Bank Analyzer",
        "description": "Per-bank state analysis"
    }
}

def get_tools_dir():
    """Get the LPDDR5 visualization tools directory."""
    if PROJECT_ROOT:
        return PROJECT_ROOT / "traces" / "memory" / "lpddr5" / "tools"
    return None

def get_trace_data(trace_path):
    """Load and return trace JSON data."""
    try:
        with open(trace_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading trace file: {e}", file=sys.stderr)
        return None

class TraceHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that injects trace data into HTML viewers."""

    def __init__(self, *args, trace_path=None, trace_data=None, tools_dir=None, **kwargs):
        self.trace_path = trace_path
        self.trace_data = trace_data
        self.tools_dir = tools_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Serve trace data as JSON
        if path == "/trace.json" or path == "/data/trace.json":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(self.trace_data).encode())
            return

        # Serve viewer HTML files
        if path.startswith("/viewer/") or path in [f"/{v['file']}" for v in VIEWERS.values()]:
            viewer_name = path.replace("/viewer/", "").replace("/", "").replace(".html", "")
            if viewer_name in VIEWERS:
                viewer_file = self.tools_dir / VIEWERS[viewer_name]["file"]
                if viewer_file.exists():
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()

                    # Read and inject trace data
                    html_content = viewer_file.read_text()

                    # Inject trace data into HTML
                    # The HTML files expect to load trace.json, so we serve it at /trace.json
                    self.wfile.write(html_content.encode())
                    return

        # Serve index page
        if path == "/" or path == "/index.html":
            self.serve_index()
            return

        # Default file serving
        super().do_GET()

    def serve_index(self):
        """Serve an index page with links to all viewers."""
        trace_name = Path(self.trace_path).name if self.trace_path else "unknown"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>KPU Trace Viewer</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{ color: #00d4ff; }}
        .trace-info {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .viewers {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        .viewer-card {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            text-decoration: none;
            color: inherit;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .viewer-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.3);
        }}
        .viewer-card h3 {{
            color: #00d4ff;
            margin: 0 0 10px 0;
        }}
        .viewer-card p {{
            color: #aaa;
            margin: 0;
            font-size: 14px;
        }}
        code {{
            background: #0f0f23;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <h1>KPU Trace Viewer</h1>

    <div class="trace-info">
        <strong>Trace File:</strong> <code>{trace_name}</code>
    </div>

    <h2>Available Viewers</h2>
    <div class="viewers">
"""
        for key, viewer in VIEWERS.items():
            html += f"""
        <a href="/viewer/{key}" class="viewer-card">
            <h3>{viewer['name']}</h3>
            <p>{viewer['description']}</p>
        </a>
"""

        html += """
    </div>

    <h2 style="margin-top: 40px;">API Endpoints</h2>
    <ul>
        <li><code>/trace.json</code> - Raw trace data</li>
        <li><code>/viewer/&lt;name&gt;</code> - Specific viewer</li>
    </ul>
</body>
</html>
"""
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """Suppress default logging, use custom format."""
        pass  # Quiet by default

def create_handler(trace_path, trace_data, tools_dir):
    """Create a handler class with trace data bound."""
    def handler(*args, **kwargs):
        return TraceHandler(*args, trace_path=trace_path, trace_data=trace_data,
                           tools_dir=tools_dir, **kwargs)
    return handler

def open_browser(url, delay=0.5):
    """Open browser after a short delay."""
    def _open():
        import time
        time.sleep(delay)

        # Try to detect if we have a display
        if platform.system() == "Linux":
            display = os.environ.get("DISPLAY")
            wayland = os.environ.get("WAYLAND_DISPLAY")
            if not display and not wayland:
                print(f"No display detected. Open manually: {url}")
                return

        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser: {e}")
            print(f"Open manually: {url}")

    thread = threading.Thread(target=_open)
    thread.daemon = True
    thread.start()

def main():
    parser = argparse.ArgumentParser(
        description="Local HTTP server for LPDDR5 trace visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s traces/memory/lpddr5/single-bank/page_hits_trace.json
    %(prog)s trace.json --port 8888
    %(prog)s trace.json --no-browser
    %(prog)s trace.json --viewer swimlane
        """
    )
    parser.add_argument("trace", help="Path to trace JSON file")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--viewer", "-v", choices=list(VIEWERS.keys()),
                        help="Open specific viewer directly")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")

    args = parser.parse_args()

    # Validate trace file
    trace_path = Path(args.trace).resolve()
    if not trace_path.exists():
        print(f"Error: Trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    # Load trace data
    trace_data = get_trace_data(trace_path)
    if trace_data is None:
        sys.exit(1)

    # Find tools directory
    tools_dir = get_tools_dir()
    if not tools_dir or not tools_dir.exists():
        print(f"Error: Could not find visualization tools directory", file=sys.stderr)
        print(f"Expected: traces/memory/lpddr5/tools/", file=sys.stderr)
        sys.exit(1)

    # Verify at least one viewer exists
    available_viewers = [v for v in VIEWERS if (tools_dir / VIEWERS[v]["file"]).exists()]
    if not available_viewers:
        print(f"Error: No viewer HTML files found in {tools_dir}", file=sys.stderr)
        sys.exit(1)

    # Create server
    handler = create_handler(trace_path, trace_data, tools_dir)

    try:
        with socketserver.TCPServer((args.host, args.port), handler) as httpd:
            url = f"http://{args.host}:{args.port}"
            if args.viewer:
                url += f"/viewer/{args.viewer}"

            print(f"Serving trace: {trace_path.name}")
            print(f"Tools dir: {tools_dir}")
            print(f"Server running at: {url}")
            print(f"Available viewers: {', '.join(available_viewers)}")
            print("Press Ctrl+C to stop\n")

            if not args.no_browser:
                open_browser(url)

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\nServer stopped.")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"Error: Port {args.port} is already in use. Try --port <other>", file=sys.stderr)
        else:
            raise

if __name__ == "__main__":
    main()
