# The Cycle Core's Memory-Flow Record, and a Debug Server for Headless Runs

**Date:** 2026-11-25
**Status:** Design

Relates to #532, the cycle-based KPU prototype (`docs/plans/cycle-based-kpu-prototype.md`). This
plan comes between Phase 1 (the memory subsystem, merged) and Phase 2 (the compute subsystem,
#542). It builds on the memory-side debugger (`docs/plans/memory-side-debugger.md`), whose
`.mflow` record, checker and viewer it reuses.

## 1. Problem Statement

Phase 1 put DRAM, the DMA engine, the NoC and L3 on the cycle core. Today a run can be seen in
three ways, and none of them is enough to debug one:

| What a run writes today | What it shows | What it cannot show |
|---|---|---|
| The counters summary (`--memory-counters`) | totals, the four laws per row | when anything happened |
| The Perfetto trace (`--memory-trace`) | component states, one lane per L3 slot | bursts, DRAM commands, why an engine waited |
| The LPDDR5 Chrome trace (`--memory-dram-trace`) | the controller's commands, checked by `trace_validator.py` | which request or L3 action a command serves; DMA windows; store buffers |

The memory side already has the structure we need. `kpu-memsim` writes a `.mflow` bundle with one
row per burst, per DRAM command, per request and per store-buffer change.
`mflow_check.py` checks the bundle against TF10, TF11 and M1–M5, and
`tools/visualization/memflow/` draws it. But that pipeline runs on `MemorySideHarness`, the
`timing/` executor, not on the cycle core. A cycle-core run never produces a `.mflow`. So the
checker and the viewer never see the model we now treat as the timing authority.

There are also four older LPDDR5 views in `traces/memory/lpddr5/tools/`: `swimlane`, `timing`,
`blockdiagram` and `bank_analyzer`. `tools/trace/serve-trace.py` serves them. Each is a single
HTML page of about 1,100–1,300 lines. Each reads the Chrome trace and reconstructs what the
record would have stated outright:
- **The bank is the event's `tid`**, so every channel's and bank group's bank 3 lands on one lane.
- **Transactions are grouped by name matching** on `ACTIVATE`, `READ` and `PRECHARGE`. INV-200 and
  INV-201 exist because that grouping went wrong.
- **The page outcome is inferred** from the command sequence, not read from the controller.
- **The timing parameters are LPDDR5-6400**, hard-coded, whatever device ran.

Finally, this machine is a headless Linux server and the work is viewed from Windows and macOS
laptops. `serve-trace.py` serves one Chrome trace file to four viewers. It cannot:
- list a directory of runs;
- serve a `.mflow` bundle, which is a folder of binary columns;
- notice that a new run has landed;
- be reached safely from anything but the machine itself.

Each experiment ends with someone copying files to a laptop, or packing them into a single HTML file.

**Goal:** one pattern, from a run to a view on the laptop, which Phase 2's components will follow:

1. **Emit.** Each component reports its events to a sink that is off by default and costs nothing when off.
2. **Record.** A recorder turns the events into a `.mflow` bundle.
3. **Check.** `mflow_check.py` checks the bundle, the same as a `kpu-memsim` bundle.
4. **View.** The memflow viewer and the four refurbished views read it.
5. **Serve.** A debug server on this machine serves runs and views to any laptop.

**Non-goals:**
- NoC flit-level views (hop by hop). The `.mflow` ports table stays empty for cycle-core runs
  until a NoC table is designed.
- Phase 2 records (L2, BlockMover, streamers). §3.8 says how they will follow the pattern.
- Launching runs from the browser (§7, Q1).

## 2. Architecture

```
┌──────────────────────────────── headless Linux server ──────────────────────────────────────┐
│                                                                                              │
│  kpu-run --memory-subsystem --run-dir runs/2026-11-25-mm64/                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐       │
│  │  cycle::Simulator                                                                │       │
│  │  ┌──────────────┐  bursts  ┌──────────────┐ commands ┌──────────────┐           │       │
│  │  │  DmaEngine   │─────────►│ Lpddr5Channel│─────────►│ LPDDR5 MC    │           │       │
│  │  │ (windows,    │          │              │ observer │ (banks)      │           │       │
│  │  │  store buf)  │          └──────┬───────┘          └──────────────┘           │       │
│  │  └──────┬───────┘                 │                                              │       │
│  │         │ MemoryEvents*           │ MemoryEvents*      (null = off, no cost)     │       │
│  │         ▼                         ▼                                              │       │
│  │  ┌──────────────────────────────────────────┐                                    │       │
│  │  │  MflowRecorder (cycle/record/)            │  ps → KPU cycles; ticks kept       │       │
│  │  │  bursts · commands · requests · buffers   │                                    │       │
│  │  │  · waits (v4)                             │                                    │       │
│  │  └──────────────────┬───────────────────────┘                                    │       │
│  └─────────────────────┼────────────────────────────────────────────────────────────┘       │
│                        │ write_mflow (program/record/, the existing writer)                  │
│                        ▼                                                                     │
│   runs/2026-11-25-mm64/  run.json · summary.txt · counters.json · trace.json (Perfetto)      │
│                          mem.mflow/                                                          │
│                        │                                   │                                  │
│                        │ mflow_check.py (CTest, by hand)   │                                  │
│                        ▼                                   ▼                                  │
│   ┌───────────────────────────────────────────────────────────────────────────────┐          │
│   │  kpu-debug-server  (Python stdlib, read-only, binds 127.0.0.1)                │          │
│   │   /                  runs index (auto-refresh on new runs, SSE)               │          │
│   │   /api/runs          JSON listing from run.json                               │          │
│   │   /files/<run>/...   raw files, path-confined to --root                       │          │
│   │   /view/<viewer>     memflow · tileflow · occupancy · floorplan · domainflow  │          │
│   │                      dram/{swimlane,timing,blockdiagram,bank_analyzer}         │          │
│   │   /common/mflow.js   the shared .mflow reader                                 │          │
│   └───────────────────────────────┬───────────────────────────────────────────────┘          │
└───────────────────────────────────┼──────────────────────────────────────────────────────────┘
                                    │ ssh -L 8080:localhost:8080 server
                                    │ (Windows OpenSSH / VS Code forwarding, macOS ssh)
                                    ▼
                 ┌────────────────────────────────────────────┐
                 │  laptop browser: http://localhost:8080      │
                 │  "Open in Perfetto" → ui.perfetto.dev        │
                 │  (postMessage of the trace's bytes)         │
                 └────────────────────────────────────────────┘
```

| Component | Where | Role |
|---|---|---|
| `MemoryEvents` | `include/sw/kpu/cycle/memory/memory_events.hpp` (new) | The sink interface the memory components report to |
| `MflowRecorder` | `include/sw/kpu/cycle/record/mflow_recorder.hpp` (new) | Implements `MemoryEvents`; builds a `MemoryFlowRecord` |
| `MemoryFlowRecord` v4 | `include/sw/kpu/program/record/memory_flow_record.hpp` | Gains the cycle core's columns and a `waits` table |
| `mflow_check.py` | `tools/trace/` | Gains separate read and write windows, and checks on the waits |
| `mflow.js` | `tools/visualization/common/` (new) | One `.mflow` reader, extracted from the memflow viewer |
| DRAM views | `tools/visualization/dram/` (moved, refurbished) | swimlane, timing, blockdiagram, bank_analyzer on `mflow.js` |
| `kpu-debug-server` | `tools/debug-server/` (new; replaces `serve-trace.py`) | Serves runs and views to laptops |

## 3. Design

### 3.1 Components report events; they do not know about records

```cpp
// WRONG: the recorder polls component state every cycle.
//   It sees only end-of-cycle state: a burst posted and granted in one cycle is
//   lost, and the recorder reaches into DmaEngine's private windows.
sim.add_invariant([&](Cycle now) { rec.sample(dma.window_contents(), now); });

// WRONG: a component writes its own file, in its own format.
//   That is a fourth format beside the counters, Perfetto and the Chrome trace,
//   with no checker and no viewer.
void DmaEngine::on_retire(...) { log_ << "retire " << ticket << '\n'; }
```

```cpp
// CORRECT: a sink interface, null by default. A component calls it at the
// moment the event happens. With no sink attached the cost is one null check.
// (As built in step 1: include/sw/kpu/cycle/memory/memory_events.hpp.)
struct MemoryEvents {
    virtual void clocks(const Domain& kpu, const Domain& dram) {}
    // DMA engine: one L3 action's life (Load, Store, Fill), keyed by its ticket.
    virtual void action_started(std::uint32_t engine, Picoseconds, std::size_t ticket, const L3Action&) {}
    virtual void action_completed(std::uint32_t engine, Picoseconds, std::size_t ticket) {}
    // DMA engine: one burst, posted to DRAM, then out of the engine's window.
    virtual void burst_posted(std::uint32_t engine, Picoseconds, std::size_t ticket,
                              std::uint64_t request, std::uint64_t address, bool write) {}
    virtual void burst_retired(std::uint32_t engine, Picoseconds, std::uint64_t request) {}
    // DMA engine: the causes that held something up this cycle, a bit per DmaWait.
    virtual void dma_waits(std::uint32_t engine, Picoseconds, std::uint32_t mask) {}
    virtual void store_buffer(std::uint32_t engine, Picoseconds, std::size_t beats) {}
    // Channel: a request the controller accepted (and the id it gave it), and each
    // command, as the controller's own record, in its ticks.
    virtual void dram_accepted(std::uint32_t channel, Picoseconds, std::uint64_t controller_id,
                               std::uint64_t request, bool write, std::uint64_t address) {}
    virtual void dram_command(std::uint32_t channel, const DramCommand&) {}
};

// Inside DmaEngine:
if (events_) events_->burst_posted(engine_, domain().edge(now), c.ticket, r->id, r->address, r->write);
```

- **One attach point.** The recorder attaches through `MemoryRunOptions::events`
  (a `MemoryEvents*`, null by default). `run_memory_subsystem` hands it to every memory
  component it builds.
- **The command tap.** `Lpddr5Channel` already installs the controller's `CommandObserver`
  (it checks the data bus there). It forwards each record to `events_` from that same lambda.
  It does not install a second observer, which would replace the first.
- **Joining a command to its burst.** The channel reports each accepted request with the id the
  controller gave it (`dram_accepted`). The recorder joins the engine's request id to the
  controller's, and each command's `request_id` to its burst. The channel keeps no table of its
  own. A DramRequest's id is unique per requester only, so the recorder refuses a second engine
  until requests name their requester.
- **Waits are not exclusive.** In one cycle the engine can be held up for several causes, and
  still send a burst for another command. The events carry a bitmask per cycle. The recorder
  keeps one interval per run of consecutive cycles that a cause held. Intervals of different
  causes may overlap; one cause's never do.
- **Recording must not change the run.** A sink observes and never steers. A test runs the same
  program with and without the recorder. The end time, every counter and every L3 byte must be
  identical (§6, I2).

### 3.2 Time: the record's units, from the cycle core's picoseconds

`.mflow` times are whole executor cycles: `MemoryFlowRecord`'s `Cycle` is an integer, stored as
f64. The commands also carry exact controller ticks. The cycle core runs in picoseconds across
clock domains, and the recorder converts at one place:

```cpp
// WRONG: checking DRAM timing on KPU cycles. At 3.2 ticks a cycle, two
// commands a tick apart can land in one cycle, and M5 passes or fails by
// accident. Rounding through floating GHz also drifts from the domains' edges.
r.issue = static_cast<Cycle>(ps * kpu_ghz / 1000);

// CORRECT: the KPU cycle the event falls in, from the domain's exact edges
// (monotone, so ordered events stay ordered), for the swimlanes; the
// controller's own ticks, untouched, for timing.
r.issue      = kpu.first_after(t_ps) - 1;       // edge(n) <= t < edge(n + 1)
r.tick       = cmd.issue;                       // controller cycle, as observed
rec.ticks_per_cycle = kpu.period_ps() / dram.period_ps();   // the manifest states the ratio
```

### 3.3 Record version 4

The cycle core has facts that version 3 cannot carry. Version 4 adds them and keeps every v3
column. `read_mflow` reads v3 and v4. `kpu-memsim` keeps writing v3 until it has a reason not to.

| Change | Why |
|---|---|
| manifest `source`: `memory-side-harness` or `cycle-core` | The checker and the viewer can say which model ran |
| manifest `write_window` (0 = shared with `window`) | The cycle DMA has separate read and write windows (#555). M1 must check each against its own limit |
| `requests.kind`: Load, Store or Fill; `requests.ticket`; `requests.slot` | A request is an L3 action. A Fill has no bursts. The inspector links a request to its program action and L3 slot |
| new table `waits`: engine, reason (`DmaWait`), t_from, t_to | Why the engine stood still: slot credit, L3 credit, request credit, window, store buffer, order. This is the debugging question the counters can total but cannot place in time |
| new station kind `dmawait`, one per engine | A lane showing why the engine waited, in the pyramid like any other station |

Waits are recorded as intervals, not one row per cycle: one per run of consecutive cycles that a
cause held, so a long stall is one row (§3.1).

**Until version 4,** the recorder writes version 3:
- **`window`** is the engine's bound on reads and writes in flight together, `window +
  write_window`. That is true and weaker. The C++ test checks each window on its own.
- **Fills** are not request rows, since v3 has no kind column.
- **The waits** are kept by the recorder (`waits()`) and not written.

### 3.4 The checker

`mflow_check.py` keeps its contract (exit 0, 1 or 2) and its checks, with these changes:

| Check | Change |
|---|---|
| M1 | Loads against `window`, stores against `write_window` when it is not 0 |
| M3 | Unchanged. A Fill has no first burst, so it is exempt |
| M4 | A Fill has no bursts. A Load's or a Store's bursts are exactly the ones its bytes span (unchanged) |
| **M7** (new) | One cause's waits on an engine never overlap; different causes' may. (M6 is taken: round-robin fairness) |
| **M8** (new) | The record agrees with the counters: bursts, loads, stores and fills, and wait cycles per reason, equal `counters.json` when the run directory holds one |

A C++ test enforces M8 as well (§5). The Python check exists so that a bundle passed around on
its own can still be trusted.

### 3.5 The views

**The shared reader.** The memflow viewer parses `.mflow` in its own script: the manifest,
columns by dtype and offset, the LOD. Step 4 moves that code into
`tools/visualization/common/mflow.js` unchanged. The viewer loads it with a `<script src>` tag,
and `pack.py` inlines it when packing, so a packed page still needs no other file.

The reader exposes one data model, and the views draw it:

```js
// WRONG: each view groups Chrome events by name, and treats a tid as a bank.
//   Channel 1's bank 3 lands on channel 0's bank 3, and a PRECHARGE with no
//   READ is taken for a WRITE (INV-200).
const bank = item.tid || 0;
const type = evts.some(e => e.name.includes('READ')) ? 'READ' : 'WRITE';

// CORRECT: the record names every coordinate and outcome; a view only draws.
const run = await Mflow.open(source);             // a folder, embedded data, or a server URL
for (const c of run.commands) {
  const bank = run.bankStation(c.mc, c.channel, c.bank_group, c.bank);
  const burst = c.burst === Mflow.NONE ? null : run.bursts[c.burst];
  draw(bank, c.kind, c.issue, c.end, burst?.outcome);   // Hit | Empty | Conflict, as recorded
}
const tRCD = run.timing.tRCD;                      // the device's table, from the manifest
```

`Mflow.open(source)` accepts three sources. The first two exist today; the server makes the
third possible:
- a folder picked or dropped in the browser;
- `window.MFLOW_EMBED` (pack.py);
- `?bundle=/files/<run>/mem.mflow/`, which fetches the manifest and then each table.

**The refurbished views**, moved to `tools/visualization/dram/`. Each keeps the job it had:

| View | Keeps | Changes |
|---|---|---|
| `timing.html` | Classic waveforms: command bus, bank states, data bus | One row group per (mc, ch). Timing annotations (tRCD, tRP, tCL) from the manifest's table, in ticks |
| `bank_analyzer.html` | State and outcome breakdown per bank | Banks by (ch, bg, ba). Outcomes read, not inferred. Each bank's row-open time and conflict count |
| `swimlane.html` | Each request as a pipeline: posted → granted → ACT → CAS → data → done | Built from `bursts` and `requests`. A request links to its L3 action (ticket, slot). Waits shown on the engine's lane |
| `blockdiagram.html` | The animated architecture: engine → controller → banks → bus | Its geometry comes from the record's stations (number of engines, channels, banks), not fixed constants |

Every view gets the memflow viewer's cursor and zoom conventions, and `?t=` and `?zoom=` URL
parameters, so a link to a moment can be shared.

**The LPDDR5 patterns write `.mflow` too** (decided 2026-11-25). The executables in
`patterns/memory/lpddr5/` drive the controller directly, with no DMA engine and no harness.
`lpddr5_harness.hpp` installs the controller's `CommandObserver` and fills a
`MemoryFlowRecord`:
- `source` is `lpddr5-pattern`.
- Each pattern request is one request with its bursts.
- There are no engines, buffers, waits or ports.

`--trace <dir>.mflow` replaces the Chrome JSON output. The views then have exactly one input
format, and `mflow.js` has no Chrome adapter. What happens to the Chrome JSON and to
`trace_validator.py` is §3.7: they retire.

### 3.6 The debug server

`tools/debug-server/kpu_debug_server.py`: Python standard library only (`http.server`,
`ThreadingHTTPServer`), so it runs wherever the repo's Python runs, with no install step.

**A run is a directory.** `kpu-run --run-dir <dir>` writes every artifact it was asked for
under fixed names, plus `run.json`:

```json
{ "format": "kpu-run", "version": 1,
  "command": ["kpu-run", "--program", "mm.csp", "..."],
  "git": "09219fe", "started": "2026-11-25T21:04:11Z", "seconds": 3.2,
  "device": "kpu_s1", "status": "ok",
  "artifacts": { "summary": "summary.txt", "counters": "counters.json",
                 "perfetto": "trace.json",
                 "mflow": "mem.mflow" } }
```

The server finds runs by their `run.json`. A bare `.mflow`, `.tflow` or `*.json` trace under the
root is listed too, as a run with only that artifact. So `kpu-memsim` output and the LPDDR5
pattern traces are viewable without changing those tools.

**Routes** (all GET; the server never writes):

| Route | Returns |
|---|---|
| `/` | The runs index, newest first: status, device, time, and links to the views that fit each artifact |
| `/api/runs` | The same, as JSON |
| `/api/events` | Server-sent events: `run-added` or `run-changed` when the root changes |
| `/files/<path>` | A file under `--root`, with its MIME type. `Range` is supported for large `.bin` columns |
| `/view/<viewer>?bundle=…`, `?trace=…` | A viewer page from `tools/visualization/` |
| `/common/<file>` | Shared viewer scripts (`mflow.js`) |

**Live runs.** A thread polls the root's mtimes every second; this needs no inotify dependency
and works on any filesystem. When a run appears or a `run.json` changes, it pushes an event:
- The index adds the row.
- An open view of that run shows a "newer version — reload" bar. It does not reload itself and
  lose the cursor.

**Access.** This is the part that must not be loose:

```python
# WRONG: serves the whole filesystem to the network.
#   "/files/../../home/.ssh/id_rsa" resolves outside the root, and 0.0.0.0
#   puts it on the LAN with no authentication.
path = os.path.join(root, url_path)              # no confinement
server = ThreadingHTTPServer(("0.0.0.0", port), Handler)

# CORRECT: confine every path to the root, and bind to loopback only.
# Authentication off loopback belongs to the full server (§7, decision 1).
p = (root / url_path).resolve()
if not p.is_relative_to(root):                   # also refuses symlinks out of root
    return self.send_error(404)
server = ThreadingHTTPServer(("127.0.0.1", port), Handler)   # no --host option
```

- **Loopback plus an SSH tunnel.** `ssh -L 8080:localhost:8080 <server>`, then open
  `http://localhost:8080` on the laptop. Windows 10 and 11 ship OpenSSH, and VS Code's Remote-SSH
  forwards the port with one click. macOS has `ssh`. The tunnel provides authentication and
  encryption, so the server needs neither.
- **Headless.** The server never tries to open a browser. It prints the URLs and the tunnel
  command. An optional `systemd --user` unit (`tools/debug-server/kpu-debug-server.service`)
  keeps it up across logins.

**Perfetto.** ui.perfetto.dev runs in the laptop's browser, and from there it cannot reach the
server. The run page's "Open in Perfetto" button does this:
1. It fetches `trace.json` from the server.
2. It opens `ui.perfetto.dev`.
3. It hands over the bytes with Perfetto's documented `postMessage` protocol (PING/PONG, then
   the ArrayBuffer).

Nothing leaves the laptop except the trace, which goes into the Perfetto tab the user opened.

### 3.7 Retiring the Chrome JSON DRAM trace (decided 2026-11-25)

Three things read the DRAM Chrome trace:
- `trace_validator.py`, with INV-001–006 and INV-100–103;
- `kpu-trace-summary`;
- Perfetto, by dragging the file in.

Once the patterns and the cycle core write `.mflow`, the timing checks are duplicated:
- **Timing:** M5 already checks tRCD, tRP, tRAS, tRC, tRFCpb, tRRD_L/S, tCCD_L/S and tFAW,
  against the declared device. INV-100–103 check four of these, against a hard-coded
  LPDDR5-6400 table.
- **Transaction semantics:** INV-001–004 and INV-006 have no `.mflow` equivalent yet. Every
  command belongs to a request, a PRE belongs to the burst that opened the page, and the order
  within a request is ACT, then CAS, then PRE.
- **Turnaround:** neither checker enforces tWTR or tRTW, though INVARIANTS.md documents them.

**Decided: port, then retire.**
1. Port INV-001–004 and INV-006 to `mflow_check.py` as M9–M11, checks on each command's link
   to its burst.
2. Add tWTR and tRTW to M5.
3. Port the summary to read `.mflow`.
4. Retire the Chrome DRAM trace, `trace_validator.py`, INV-200/201 and `kpu-trace-summary`.
   Update the skills and CLAUDE.md sections that name them: the validation workflow,
   `/trace-check`, `add-invariant` and `lpddr5-trace-debug`, which move to `mflow_check.py`'s
   checks and IDs. The cycle core's `--memory-dram-trace` goes too; `--memory-mflow` replaces it.
   INVARIANTS.md keeps its definitions, each mapped to the M-check that now enforces it.

Perfetto keeps the cycle core's own component trace, which is a different file.

### 3.8 The pattern, for Phase 2

Each Phase 2 component (L2 banks, the BlockMover, the streamers, the combiner) follows the five
steps of §1:
1. **Emit:** an events interface beside the component, null by default.
2. **Record:** its own tables in its own record. For example `.cflow`, the compute-flow record,
   planned with 2a. It is not folded into `.mflow`: one record per subsystem, sharing
   `record/columnar.hpp` and the LOD.
3. **Check:** a checker with exit 0, 1 or 2 and a self-test that breaks each check.
4. **View:** a view on a shared reader. The debug server discovers the new artifact kind through
   `run.json` and links its view.
5. **The two tests:** recording changes nothing (I2), and the record agrees with the counters (I3).

## 4. Implementation Steps

Each step is one PR, one concern, with its own tests.

1. **Events and the recorder** (C++)
   - `include/sw/kpu/cycle/memory/memory_events.hpp`: the interface (§3.1).
   - `dma_engine.hpp`: call the sink at offer, credit, burst post and done, retire, wait
     intervals and store-buffer changes.
   - `lpddr5_channel.hpp`: report each accepted request with its controller id, and forward each
     command from the existing observer.
   - `memory_subsystem.hpp`: `MemoryRunOptions::events`, passed to each component.
   - `include/sw/kpu/cycle/record/mflow_recorder.hpp`: the recorder, the time conversion (§3.2)
     and `to_record()`.
   - Tests in `tests/cycle/test_cycle_mflow_recorder.cpp`:
     - I2: identical runs with and without the recorder;
     - I3: the record's totals equal `DmaCounts` and `DramCounts`;
     - a hand-built run with known bursts, commands and waits;
     - a NoC (T4) run.
2. **Record v4** (C++)
   - `memory_flow_record.hpp` and `src/program/memory_flow_record.cpp`: `source`,
     `write_window`, the request columns, the `waits` table, the `dmawait` stations.
   - `read_mflow` reads v3 and v4.
   - The round-trip test covers v4; a stored v3 bundle still reads.
3. **`kpu-run` writes it** (C++)
   - `--memory-mflow <dir>`.
   - `--run-dir <dir>`: every requested artifact under the fixed names, plus `run.json`.
   - CTests run `mflow_check.py` on a kpu_s1 matmul bundle and on a T4 NoC bundle, both passing.
4. **The checker** (Python)
   - `tools/trace/mflow_check.py`: the M1 split, the Fill exemptions, M7, M8.
   - The ports from `trace_validator.py` (§3.7): M9–M11 for INV-001–004 and INV-006, and tWTR and
     tRTW in M5. The manifest's timing table already carries tWTR_L, tWTR_S and tRTW, from both
     `dram_bridge.cpp` and the recorder.
   - `test_mflow_check.py`: a self-test that breaks each new check. Each LPDDR5 pattern bug that
     `lpddr5-trace-debug` records (wrong PRECHARGE owner, type defaulting to WRITE, the tRCD, tRP
     and tRRD violations) is rebuilt as a broken bundle that the new checks refuse.
5. **The shared reader** (JS)
   - Extract `tools/visualization/common/mflow.js` from `memflow/index.html`, with no change in
     behaviour. `pack.py` inlines it.
   - Add the v4 fields.
   - `memflow/test_viewer.py` runs the reader under Node over a v3 bundle and a v4 bundle.
6. **The memflow viewer shows v4** (JS)
   - The waits lane per engine, coloured by reason.
   - The inspector shows ticket, slot and kind.
   - A Fill appears on the engine's lane with no bursts.
7. **The debug server** (Python)
   - `tools/debug-server/kpu_debug/`: the routes, run discovery, path confinement, loopback
     only, `Range`. Run discovery, path confinement and the viewer registry are plain functions
     with no HTTP in them. The full server (`docs/plans/kpu-debug-server-full.md`) reuses them
     unchanged.
   - `README.md`: the tunnel recipe for Windows (OpenSSH, VS Code) and macOS.
   - `test_kpu_debug_server.py`: path traversal and symlinks are refused; the socket is bound
     to loopback; the listing; MIME types; `Range`.
   - Delete `tools/trace/serve-trace.py`, and update its docs (`tools/trace/README.md`,
     `patterns/memory/lpddr5/README.md`).
8. **Live updates and Perfetto** (Python, JS)
   - `/api/events` and the mtime poller; the index adds rows; the view shows the reload bar.
   - The "Open in Perfetto" button.
   - Tests: a run written while the server runs appears in `/api/runs`, and an event is sent.
9. **The refurbished views**, one PR each, in this order (each ~one file):
   - 9a `dram/timing.html`
   - 9b `dram/bank_analyzer.html`
   - 9c `dram/swimlane.html`
   - 9d `dram/blockdiagram.html`

   Each moves from `traces/memory/lpddr5/tools/` (the old file is deleted in the same PR) and is
   rebuilt on `mflow.js`. Each gets a Node data-layer test, as the other viewers have. Each gets
   a `pack.py` path, so a view can still be a single file.
10. **The LPDDR5 patterns write `.mflow`** (C++)
    - `patterns/memory/lpddr5/common/lpddr5_harness.hpp` records through the command observer
      (§3.5).
    - `--trace` writes a bundle, and each pattern's CTest runs `mflow_check.py` on it.
11. **Retire the Chrome DRAM trace** (one cleanup PR, after 4, 9 and 10)
    - Delete `trace_validator.py` and its test, `kpu-trace-summary`, and `--memory-dram-trace`
      with its CTests.
    - Delete the Chrome DRAM export path that only those used, and the committed
      `traces/memory/lpddr5/**/*_trace.json`.
    - Update CLAUDE.md's validation section and the skills (§3.7).
    - Point the MCP server's `kpu_validate_trace` (`tools/mcp/kpu_sim_server.py`) at
      `mflow_check.py`.
    - The summary moves into `mflow_check.py --summary`.
12. **Docs**: a "debugging a cycle-core run" guide in `docs/02-simulation/`: run, check, serve,
    view. Plus the CHANGELOG entry and the session log.

Steps 1–4 are the record; 5–6 the viewer; 7–8 the server; 9 the views; 10–11 the patterns
and the retirement. Steps 7 and 8 need only step 3's `run.json`, so they can run alongside 5
and 6.

## 5. Verification

```bash
# Build and the cycle tests (gcc, then clang)
cmake --build build --target test_cycle_mflow_recorder kpu-run -j8
ctest --test-dir build -R 'cycle_mflow|kpu_run' --output-on-failure

# A run directory from a real program
kpu-csp-gen --algo matmul --size 64 --tile 16 --dtype int8 \
    --target tests/program/deploy/kpu_s1.json -o mm.csp
kpu-run --program mm.csp --deploy tests/program/deploy/kpu_s1.json \
    --memory-subsystem --run-dir runs/mm64
python3 tools/trace/mflow_check.py runs/mm64/mem.mflow          # exit 0
python3 tools/trace/test_mflow_check.py                         # each check fails when broken

# The views' data layers (Node on PATH, see node-for-viewer-tests)
python3 tools/visualization/memflow/test_viewer.py runs/mm64/mem.mflow
python3 tools/visualization/dram/test_views.py runs/mm64/mem.mflow

# The server
python3 tools/debug-server/test_kpu_debug_server.py
python3 tools/debug-server/kpu_debug_server.py --root runs
#   laptop: ssh -L 8080:localhost:8080 <server>; open http://localhost:8080
```

**Expected:**
- **The kpu_s1 bundle passes** TF10, TF11 and M1–M8. Its burst count equals `counters.json`'s,
  and its data-bus occupancy agrees with the summary's ~95%.
- **The T4 bundle passes** too, and its waits show the store engine's order waits.
- **The recorder is invisible:** the same run with and without it has identical end
  picoseconds, counters and L3 bytes.
- **The server is closed to the outside:**
  - `curl localhost:8080/files/../../etc/passwd` returns 404;
  - the server listens on 127.0.0.1 only (`ss -ltn` shows no other address).
- **A new run shows up live:** started while the index is open on the laptop, it appears within
  about 2 s with no manual refresh.
- **Every view opens from the laptop:** each of the four DRAM views opens `runs/mm64` over the
  tunnel from Windows and from macOS. This is a manual check, and its outcome is recorded in
  the PR.

## 6. Key Invariants

- **I1. Same record, same check.** A cycle-core bundle and a `kpu-memsim` bundle are the same
  format. `mflow_check.py` checks both with the same code. A difference in model is carried in
  the manifest, never in a fork of the checker.
- **I2. Recording is invisible.** Attaching a sink changes no timing, counter or value. A sink
  observes and never decides.
- **I3. The record and the counters agree.** Every total in the counters can be recomputed from
  the record, and is equal (M8, and the C++ test).
- **I4. Exact DRAM time.** DRAM timing is checked on the controller's ticks, never on converted
  KPU cycles.
- **I5. The record states, the views draw.** The views infer nothing the record states. A view
  never infers a bank, an outcome or a timing parameter. If the record lacks a fact, the view
  says "unknown" rather than guessing.
- **I6. Buffer language.** No view or record uses cache terms. Banks have open rows, page
  outcomes are Hit, Empty or Conflict (DRAM page terms), and L3 slots are credited and released.
- **I7. The server is read-only and confined.** It never writes and never runs anything. Every
  path it serves resolves inside `--root` or the viewers' directory. It listens on loopback only.
- **I8. Runs are deterministic.** The same command on the same commit writes byte-identical
  `.mflow` tables, so two bundles can be diffed.

## 7. Decisions (2026-11-25)

1. **Starting runs from the browser:** wanted, together with access from cloud services through
   OAuth. That is a full server with its own plan, `docs/plans/kpu-debug-server-full.md`. The server here is the read-only,
   loopback-only first stage of it.
   - LAN token mode is dropped: the full server's authentication replaces it.
   - I7 holds for this stage.
2. **The views move now** to `tools/visualization/dram/`. No external links need to be kept.
3. **The LPDDR5 patterns write `.mflow`** (§3.5).
4. **The Chrome DRAM trace retires** once its checks are ported (§3.7, steps 4 and 11).
