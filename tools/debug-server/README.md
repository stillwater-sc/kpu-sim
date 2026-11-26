# kpu-debug-server

Serves the runs under a directory, and the viewers that read them, to a browser. It's built for a headless Linux machine whose runs you view from a Windows or macOS laptop.

It's read-only, and listens on `127.0.0.1` only. You reach it through an SSH tunnel, which provides both the authentication and the encryption. The plan is `docs/plans/cycle-mflow-and-debug-server.md` §3.6. A full server, which can start runs and is reachable from cloud services behind OAuth, is planned separately in `docs/plans/kpu-debug-server-full.md`.

## Run it on the server

```bash
# A run directory per experiment: summary, counters, Perfetto trace, .mflow bundle, run.json
kpu-run --program mm.csp --deploy tests/program/deploy/kpu_s1.json \
        --memory-subsystem --run-dir ~/runs/2026-11-25-mm64

# Serve every run under ~/runs
python3 tools/debug-server/kpu_debug_server.py --root ~/runs            # --port 8080 by default
```

It needs Python 3.9 or later and no packages. It prints the URL and the tunnel command to use.

## Open it from a laptop

| Laptop | How |
|---|---|
| **macOS, Linux** | `ssh -L 8080:localhost:8080 <server>`, then open <http://localhost:8080> |
| **Windows 10/11** | The same command in PowerShell or Windows Terminal: OpenSSH ships with Windows. Then open <http://localhost:8080> |
| **VS Code (Remote-SSH)** | Connect to the server, open the **Ports** panel, choose **Forward a Port** and enter `8080`. VS Code then opens the forwarded address |

To keep a tunnel open in the background without a shell, use `ssh -N -L 8080:localhost:8080 <server>`. If port 8080 is taken on either end, pick another with `--port`, and use the same number on both sides of `-L`.

## What it serves

| Route | What |
|---|---|
| `/` | The runs, newest first: status, device, start time, and each artifact with links to the viewers that read it |
| `/api/runs` | The same list, as JSON |
| `/files/<path>` | A file under the root, confined to it. Range requests are honoured |
| `/view/<viewer>/` | A viewer, plus `/view/common/` for the scripts the viewers share |

**What counts as a run:**
- **A run directory:** a directory holding a `run.json` written by `kpu-run --run-dir`. Its artifacts are the files the manifest names.
- **A bare artifact:** any `.mflow` or `.tflow` bundle, or Chrome/Perfetto trace `.json`, found elsewhere under the root. Each is listed as a run of its own.
- **Not followed:** symlinks.

**Which viewers open what:**
- **`.mflow` bundles** open in the memory-side debugger (`/view/memflow/index.html?bundle=...`).
- **An LPDDR5 Chrome DRAM trace** opens in the bank analyzer, until the `.mflow` DRAM views replace it (step 9).
- **The other viewers** (tile flow, occupancy, floorplan, domain flow) are served as pages that open their own files.

## Safety

- **Loopback only:** there is no option to bind another address. It's meant to be reached only through an SSH tunnel.
- **Read-only:** GET and HEAD only; any other method gets 405.
- **Confined paths:** every path resolves inside the root, or inside a viewer's directory, after symlinks and `..` are resolved. Anything else gets 404.
- **No markup injection:** the index escapes every name it lists, so a directory name can't become markup.

## Tests

`python3 tools/debug-server/test_kpu_debug_server.py`, also run by CTest as `kpu_debug_server`.
