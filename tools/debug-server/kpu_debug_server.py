#!/usr/bin/env python3
"""
kpu-debug-server: serve the runs under a directory, and the viewers that read them, to a browser
(docs/plans/cycle-mflow-and-debug-server.md §3.6, step 7).

Usage:
    python3 tools/debug-server/kpu_debug_server.py --root <runs dir> [--port 8080]

It listens on 127.0.0.1 only. From a laptop, tunnel to it over SSH and open the printed URL:
    ssh -L 8080:localhost:8080 <this host>          (Windows OpenSSH, macOS, Linux)
VS Code's Remote-SSH forwards the port too (Ports panel, "Forward a Port", 8080).

Exit codes: 0 stopped (Ctrl-C); 2 bad arguments, or the port is taken.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kpu_debug.server import HOST, make_server  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--root", required=True, help="the directory of runs to serve (kpu-run --run-dir writes one per run)")
    ap.add_argument("--port", type=int, default=8080, help="the loopback port (default 8080)")
    a = ap.parse_args()
    try:
        srv = make_server(a.root, a.port)
    except (OSError, NotADirectoryError) as e:
        print(f"kpu-debug-server: {e}", file=sys.stderr)
        sys.exit(2)
    port = srv.server_address[1]
    print(f"kpu-debug-server: serving {Path(a.root).resolve()} at http://{HOST}:{port}/ (loopback only)")
    print(f"  from a laptop:  ssh -L {port}:localhost:{port} {socket.gethostname()}   then open http://localhost:{port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
