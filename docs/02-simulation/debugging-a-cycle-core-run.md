# Debugging a cycle-core run

How to run a program's memory traffic on the cycle core, check what it recorded, serve the
records to a browser on your laptop, and read them in the views. The plan behind this is
`docs/plans/cycle-mflow-and-debug-server.md`.

Every command below runs from the repository root, against a configured `build/` directory.

## 1. Run

Build the two tools, write a program, and run its L3 traffic on the cycle core's memory
subsystem (the DMA engine, the LPDDR5 channel and the L3 tile), with every artifact in one
run directory:

```bash
cmake --build build --target run csp-gen
mkdir -p build/runs
./build/tools/csp-gen/kpu-csp-gen --algo matmul --size 64 --tile 16 --dtype int8 \
    --target tests/program/deploy/kpu_s1.json -o build/runs/mm64.csp
./build/tools/kpu-run/kpu-run --program build/runs/mm64.csp --deploy tests/program/deploy/kpu_s1.json \
    --memory-subsystem --run-dir build/runs/mm64
```

`kpu-run` prints the run's summary: whether every slot and stored tile was right bit for bit,
whether the operational laws hold, and each component's throughput, utilization and stalls. It
exits 0 only when the data and the laws both hold. `build/runs/mm64/` then holds:

| File | What it is | Read by |
|------|------------|---------|
| `run.json` | the command, build, device, status, result and the artifacts' names | the debug server, `check_run_dir.py`, `mflow_check.py` (M8) |
| `summary.txt` | the summary `kpu-run` printed | you |
| `counters.json` | every component's and link's counters | `tests/cycle/check_trace.py` |
| `trace.json` | the cycle core's Perfetto trace: a track per component and link, each domain's clock, a lane per L3 slot | Perfetto |
| `mem.mflow/` | the memory-flow record: every burst, DRAM command, request, store-buffer change and DMA wait | `mflow_check.py`, the views |

Use `--deploy tests/program/deploy/kpu_t4.json` (and a program written for it) for the T4,
whose L3 traffic crosses its NoC.

## 2. Check

Check before you look: a check that fails names the burst, command or request to look at.

```bash
python3 tools/kpu-run/check_run_dir.py build/runs/mm64 --status ok
python3 tools/trace/mflow_check.py build/runs/mm64/mem.mflow --summary
python3 tests/cycle/check_trace.py build/runs/mm64/trace.json build/runs/mm64/counters.json
```

- **`check_run_dir.py`**: `run.json` says what ran and how it ended, and every artifact it names
  is there.
- **`mflow_check.py`**: the record against the memory side's invariants. TF10 and TF11 are the
  banks' row state and the data bus. M1–M4 are the engine's windows, the store buffer, the order
  within a request, and the bursts. M5 is DRAM timing against the device's own table, in
  controller ticks. M6 is arbitration, M7 the waits, M8 the record against `run.json`'s counts,
  and M9–M11 each command's burst, the order within a burst, and refresh. Its docstring defines
  each check; `patterns/memory/lpddr5/INVARIANTS.md` maps the LPDDR5 invariants to them.
  `--summary` adds what the run did: page outcomes, throughput against the ceiling, each bank's
  commands, and the bursts' latency.
- **`check_trace.py`**: the Perfetto trace is well formed and agrees with the counters.

Each checker exits 0 when everything holds, 1 when something is violated, and 2 when it cannot
read its input. On 1, parse the violations and fix the root cause. On 2 there are no
violations: fix the recording or the path.

The LPDDR5 patterns record the same way, with the controller driven directly and no DMA engine.
`ctest` writes and checks them:

```bash
ctest --test-dir build -R 'lpddr5_page_conflicts'
python3 tools/trace/mflow_check.py build/patterns/memory/lpddr5/page_conflicts.mflow --summary
```

## 3. Serve

The machine that ran the simulation usually has no browser. The debug server lists every run
and record under a root and serves them, with the views, to a laptop over an SSH tunnel:

```bash
python3 tools/debug-server/kpu_debug_server.py --root build/runs
```

On the laptop, run `ssh -L 8080:localhost:8080 <server>` and open `http://localhost:8080`.

- **Security:** the server listens on loopback only, so nothing else can reach it.
- **The index:** it lists each run with its status, device and artifacts, and links every
  `.mflow` record to its views. A trace gets an **Open in Perfetto** button.
- **Live updates:** the index updates as runs appear, change or go. Re-run into the same
  directory, and a view open on its record shows a bar saying the run has a newer version, with
  a button that reloads it.

`tools/debug-server/README.md` has the details.

## 4. View

A `.mflow` record opens in five views. Each reads the record alone, so the views agree with
each other and with the checker:

| View | Shows | Start here when |
|------|-------|-----------------|
| memory-side debugger (`memflow`) | requests, bursts, waits and the store buffer over time, with observations | you want the run as a whole, or M1–M4, M7 or M8 fails |
| DRAM timing diagram (`dram-timing`) | each channel's command and data buses, burst by burst, with every timing gap measured against the table | M5 fails, or a burst is slow |
| DRAM bank analyzer (`dram-bank`) | each bank's row openings, states and hit rate, conflicts | page hits are low, or TF10, M9 or M10 fails |
| request swimlanes (`dram-swim`) | each request's stages, and its bursts' stages, by engine | a request waits too long, or M3 fails |
| memory block diagram (`dram-block`) | the engines, controllers, banks and buses, cycle by cycle | you want to step through what happened |

A DRAM view's link names what it shows: a moment (`t`), a zoom, and a burst, bank or request.
Copy the link to share what you are looking at.

To look at a record without the server, pack a view and the record into one HTML file and open
it anywhere:

```bash
python3 tools/visualization/memflow/pack.py build/runs/mm64/mem.mflow -o build/runs/mm64-memflow.html
python3 tools/visualization/dram/pack.py timing build/runs/mm64/mem.mflow -o build/runs/mm64-timing.html
```

`tools/visualization/dram/pack.py` packs `timing`, `bank_analyzer`, `swimlane` or
`blockdiagram`. The cycle core's own trace, `trace.json`, opens in Perfetto: use the index's
button, or drag the file into https://ui.perfetto.dev.

## 5. When something is wrong

Start with the simplest run that shows it. Use one tile and one engine before reuse, and reuse
before a larger machine. Then find the first rung where the records diverge.

1. **The data is wrong** (`kpu-run` reports a slot or stored tile that is not bit for bit). The
   values answer to L-B. Check which tile and slot, then use the swimlanes to find the request
   that moved it.
2. **A check fails.** Its message names the burst, command or request, and its cycle. Open the
   view from the table above at that moment, then read the bursts and commands around it.
3. **It is slow but correct.** Compare `--summary` with the ceiling. The memory-side debugger's
   observations name the cause that held the engine up most (L3 credit, the window, the store
   buffer). The bank analyzer shows whether pages are hits, and the timing diagram shows which
   gap stretched a burst.
4. **It deadlocks.** `kpu-run` prints the detector's report: who waits on whom. The run ends
   `failed`, and its record up to that point is still written and opens in the views.

A new kind of bug that the checks missed gets a check of its own. The `add-invariant` skill adds
it to `mflow_check.py`, with a self-test that breaks a clean record.
