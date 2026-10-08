# Memory-Side Debugger: DRAM, Controllers, DMA Engines and Buffers, Outside the NoC

**Date:** 2026-10-08
**Status:** Q1-Q5 decided 2026-10-08 (all as recommended); steps 1-3 done
**Related:**
- `docs/plans/dram-bank-model.md`: this plan carries its step 4 (per-bank record, TF10/TF11) and
  step 6 (viewer DRAM panel).
- `docs/plans/tile-flow-debugger.md` (#286): the `.tflow` debugger this mirrors.
- `docs/plans/noc-port-arbitration.md`: the port model the stubs stand in for.

## 1. Problem

The tile-flow debugger (#286) shows tiles moving through L3, the movers and the compute tiles at
L-T1. It cannot show the memory side:
- L-T1 has no banks, and DRAM there is an aggregate rate.
- The CSP tier, which does model the banks, writes no record.

Since DRAM steps 2 and 3, the CSP memory side is cycle-accurate: a hosted LPDDR5 controller per
memory controller, DMA engines with a burst window, store buffers with tickets, and NoC hooks. We
need to study, characterize and debug that subsystem **on its own**, without the L3, the movers,
the compute tiles or the NoC in the way, and see every burst and every DRAM command.

### 1.1 What the code exposes today (surveyed at `11f4029`)

| Piece | Exposes | Missing |
|---|---|---|
| `LPDDR5MemoryController` | a command trace (`enable_tracing`): ACT/RD/WR/PRE/REF with channel, bank, issue and complete cycle, and request id | the **row** (only in a ResourceTracker string); the trace is in controller cycles; no per-event hook |
| `DramBridge` | aggregate `Stats` (bursts, page hit/empty/conflict, refreshes) | never enables tracing; the controller is private; a completion gives only the tag |
| `MemoryControllerProcess`, burst path (step 3) | `BurstDone{tile, is_load, submitter}` | no events, no address, no cycle, no bank |
| `MemoryControllerProcess`, tile path | `DMA_*_START/COMPLETE` per tile | `MC_ACCESS_TYPE` only on the legacy path |
| `DMAEngineProcess` | tile-level events, `bursts_in_flight()`, store-buffer state | no per-burst events |
| `.tflow` | one `dram` station, no banks; produced only at L-T1 | a CSP producer; bank stations would also need a `build_lod` branch |

So per-bank data needs two things: an observer inside the controller, and a producer that runs the
memory side.

## 2. Architecture

```
  scenario.json --------------------------------------------------------------+
   deployment (memory.dram, dma.engines/window, noc.port...), request models, |
   port stubs, duration / stop condition                                      |
                                                                              v
  +-------------------------------- MemorySideHarness ------------------------------+
  |                                                                                 |
  |  PORT STUBS (one per attached NoC port; noc-port-arbitration §1.2)              |
  |   load side:  request model --> DMA engine (schedule_load)                      |
  |               load sink    <-- LoadRoute::inject (injection bus: 1 block/B,     |
  |                               accept rate / latency / back-pressure)            |
  |   store side: ejection model --> reserve + deliver(ticket) into the store       |
  |               buffer (output queue), at a rate; DMA writes it to DRAM           |
  |                                                                                 |
  |  DMA engines (DMAEngineProcess, window W, store buffer)                         |
  |        |  submit_burst / get_completed_burst                                    |
  |        v                                                                        |
  |  Memory controllers (MemoryControllerProcess, hosted)                           |
  |        |  DramBridge + CommandObserver (new)                                    |
  |        v                                                                        |
  |  LPDDR5MemoryController: banks, FR-FCFS, refresh                                |
  |                                                                                 |
  |  Recorder: every burst, every DRAM command, every tile request, every           |
  |            buffer and port event --> .mflow bundle                              |
  +---------------------------------------------------------------------------------+
            |                                     |
            v                                     v
   mflow_check.py (TF10, TF11, M1..)     memflow viewer page (+ characterization tab)
```

Nothing here is a new memory model. The harness wires the existing CSP processes. The stubs stand in
for everything north of the port.

## 3. Design

### 3.1 Observability (the controller and the bridge)

- **The controller gains an optional `CommandObserver`**, a callback per issued command:
  `{kind, channel, bank_group, bank, row, column, issue_tick, data_start_tick, data_end_tick,
  request_id}`. It is called where `trace_command` is called today, and also carries the row.
- **The bridge installs it and translates:**
  - controller ticks to executor cycles, through the same accumulator it already uses;
  - the request id to the burst's tag, which in turn gives the submitter, tile and ticket;
  - the controller's native address back to the spec's coordinates
    `(mc, ch, rk, bg, ba, row, col)`.
- **The MC's burst path records each burst:** submitted, first command (ACT or CAS), data
  start/end, and completed, with its decoded coordinates and page outcome (hit, empty or
  conflict).
- All of this is **opt-in** (`Config::record`). Without a recorder nothing is kept; the
  controller still makes one null check per command.

### 3.2 The harness and the port stubs

`include/sw/kpu/timing/memory_side_harness.hpp`. It is built from a `DeviceSpecification`, with
the same engine numbering and attachment as `csp_config_from` and `csp_noc_wiring`.

- **Request models (load side).** Each has a deterministic generator and a seed:
  - `stream`: contiguous tiles;
  - `strided`: a pitch;
  - `random`: uniform over a region;
  - `matrix_tiles`: GEMM tiles of a pitched matrix, after `patterns/dma/common/matrix_layouts.hpp`;
  - `replay`: the LOAD/STORE ops of a CSP schedule, by address.

  Each model has a per-port issue rate and a total.
- **Load sink**, standing in for the port's injection bus:
  - accepts one block per `block_cycles` (the spec's NoC link rate), or infinitely fast;
  - an optional accept latency;
  - optional back-pressure: the input queue depth from `noc.port.input_queue_blocks`.
- **Ejection model (store side)**, standing in for the port's output queue: blocks arrive at a
  rate, reserve a store-buffer slot (and wait if none is free), and are delivered under their
  ticket. The DMA then writes them, exactly as at L-CA.
- **No L3:** loads need no L3 credit here. The harness gives each engine a credit pool sized
  from the spec, so the credit-before-first-burst order is still exercised.

### 3.3 The record: `.mflow`, the carrier for DRAM step 4

A bundle in the `.tflow` container style (`manifest.json` plus little-endian columnar `.bin`
files, 8-byte aligned, f64 times, and an LOD pyramid), with its own format name and version.

| Table | One row per | Columns |
|---|---|---|
| `bursts` | burst | engine, mc, ch, rk, bg, ba, row, col, is_load, outcome, tile, ticket, t_submit, t_cmd, t_data0, t_data1, t_done |
| `commands` | DRAM command | mc, ch, bg, ba, row, kind (ACT/RD/WR/PRE/REF), t_issue, t_end, burst |
| `requests` | tile request | engine, port, tile, is_load, addr, bytes, t_enqueue, t_credit, t_first_burst, t_last_burst, t_retire |
| `buffers` | store-buffer change | engine, t, held, staged |
| `ports` | port-stub event | port, engine, kind (offer, accept, refuse, eject_arrive, eject_wait), t |

- **Stations:** one per bank (kind `dram_bank`), per channel data bus (`dram_bus`), per DMA engine,
  per store buffer and per port stub.
- **LOD:** `build_lod` gains branches for these kinds.
- **The record is per burst and per command, not per tile.** That is what makes TF10/TF11
  checkable. The `.tflow` record (L-T1, tiles) is unchanged.

### 3.4 The checker: `tools/trace/mflow_check.py`

Same contract as `tflow_check.py` and the LPDDR5 `trace_validator.py`: exit 0 pass, 1
violations, 2 unreadable.

| Check | Invariant |
|---|---|
| **TF10** | a bank never has two open rows: between an ACT and its PRE, every RD/WR to that bank names the ACT's row |
| **TF11** | a channel's data bus never carries two bursts at once |
| M1 | an engine never has more than W bursts in flight |
| M2 | a store buffer never holds more than its capacity |
| M3 | a request's first burst comes no earlier than its L3 credit |
| M4 | every burst completes exactly once; every request's bursts are exactly the ones its bytes span |
| M5 | DRAM timing against the **declared** device's timing table (tRCD, tRP, tRRD, tFAW, tCCD), not the LPDDR5-6400 table that `trace_validator.py` hard-codes |

A self-test breaks each check in turn, as `test_tflow_check.py` does.

### 3.5 The viewer: `tools/visualization/memflow/index.html`

A static page in the tile-flow viewer's style: one file, Canvas2D, folder load or `pack.py`
embedding, LOD swimlanes, transport and inspector.

**Panels:**
- **Banks:** one swimlane per bank, showing ACT, RD, WR, PRE and REF, coloured by command, with
  the open row as a label. Conflicts are marked.
- **Channels:** the data-bus occupancy of each channel, against the ceiling.
- **DMA engines:** each engine's bursts in flight against W, and its request lifetimes from
  credit to retirement.
- **Store buffers:** held and staged against capacity.
- **Port stubs:** offers, accepts, refusals and ejection waits.
- **Address view:** a bursts-by-bank and by-row heat map, toggled by channel, bank or row (DRAM
  step 6).
- **Inspector:** click a burst for its tile, engine, coordinates, command chain and page
  outcome. Click a request for its bursts.
- **Characterization tab:** loads a `sweep.json` (§3.6) and plots bandwidth against W, engines and
  pattern.

### 3.6 Characterization

`kpu-memsim` (new tool) runs a scenario and writes an `.mflow` bundle. With `--sweep`, it runs a
grid over engines, W, pattern and port-sink rate, and writes `sweep.json` and `sweep.csv`. The
step-3 table becomes a regression: one engine at W = 32 reaches at least 90% of the ceiling,
and 32 engines at W = 1 stay below 80%.

## 4. Implementation Steps

Each step is one PR and ends green.

1. **Observability.** (Done.)
   - `CommandObserver` in the LPDDR5 controller, carrying the row.
   - Bridge translation into executor cycles and spec coordinates.
   - Burst records on the MC's burst path.
   - Tests: the row matches the decoded address; cycles are monotone; a conflict shows
     PRE->ACT->CAS.
   - As built:
     - **Controller:** `LPDDR5MemoryController::set_command_observer`. One `CommandRecord` per
       command, called beside `trace_command`. It carries the row (a precharge carries the row
       it closes), the CAS data-bus window, and the request's activated and conflicted flags.
     - **Bridge:** `DramBridge::set_command_sink` translates into `Command`.
       - A command issues at the executor cycle the controller is being ticked to. Its end and
         data window are converted at ticks-per-cycle.
       - Coordinates come back in the spec's `(ch, bg, ba, row, col)`.
       - `tag` is the burst's submit tag. A precharge carries the tag of the burst that opened
         the row it closes: an opener's tag is kept until that precharge, because a conflict's
         precharge always comes after the opener's burst completed (found in review).
     - **MC:** `MemoryControllerProcess::Config::record`.
       - `recorded_commands()` returns every command.
       - `recorded_bursts()` returns each window burst: id, submitter, tile, address, decoded
         coordinates, submitted, first command, data window, done, and page outcome (hit,
         empty or conflict).
       - The sink is installed on first use, not in the constructor, because it captures
         `this`.
     - **Tests:**
       - three bursts (empty, hit, conflict) give the exact chain ACT 5, RD, RD, PRE 5, ACT 9,
         RD, with matching coordinates, outcomes and ordered cycles;
       - recording is off by default;
       - a precharge without its row, swapped outcomes, and unmapped request ids each fail.
2. **Harness and port stubs.**
   - `MemorySideHarness` from a `DeviceSpecification`, the request models, the load sink and the
     ejection model.
   - Tests: conservation (every request completes, every slot returns); the sink's back-pressure
     stalls injection; the step-3 table reproduces.
   (Done.)
   - **As built,** `timing/memory_side_harness.hpp` (`sw::kpu::timing::memside`):
     - **Parts:** hosted MCs with recording on, and DMA engines numbered and attached as the
       floorplan does. The stand-in L3 is one credit pool of `l3.capacity_tiles` slots.
     - **`RequestModel`:** stream, strided, random, matrix tiles and replay. `matrix_tiles`
       issues one request per tile row, the rows a pitch apart, as a 2D DMA would.
       `replay_of(schedule)` takes a schedule's LOAD or STORE tiles.
     - **Load sink:** each attached port has a per-engine input queue and an injection bus that
       carries one block per `block_cycles`, oldest head first. A full queue refuses the
       engine. A landed load frees its stand-in L3 slot `consume_latency` cycles later.
       `infinite` lands at once.
     - **Ejection model:** one ejection per engine at a time, every `eject_interval` cycles. It
       reserves a store-buffer slot (waiting while none is free), crosses the ejection bus,
       and is delivered under its ticket.
     - **Records:** each request's lifetime (offered, credit, first burst, last burst,
       retired), store-buffer occupancy on change, and port events (offer, accept, refuse,
       land, eject arrive, wait, deliver).
   - **Tests:**
     - the models give their documented addresses (matrix-tile pitch, replay counts);
     - conservation: every request retires, every credit and buffer slot returns, and the
       burst count equals the bytes spanned;
     - lifetimes are ordered;
     - a slow bus paces loads and refuses engines, while an infinite sink runs at the DRAM
       ceiling;
     - the step-3 table reproduces: 1 x 32 reaches at least 90%, 32 x 1 stays under 80%.
     - A sink that never refuses, or a consumer that never frees L3, each fails.

3. **Record (DRAM step 4).**
   - The `.mflow` writer and reader, and LOD rows for the new kinds.
   - `kpu-memsim scenario.json --out`.
   - A checked-in T4 scenario and its ctests.
   (Done.)
   - **The record,** `record/memory_flow_record.hpp` with `src/program/memory_flow_record.cpp`.
     `MemoryFlowRecord` holds the stations and five tables. `write_mflow` and `read_mflow` check
     the format, the version, and every index.
   - **Shared container code:** `record/columnar.hpp`, now used by `.tflow` as well (the
     columnar tables and the 2^53 exact-time limit).
   - **The pyramid:** `build_lod_rows`, the generic pyramid. `build_lod` (`.tflow`) is a thin
     wrapper over it, and its output is unchanged.
   - **Conversion:** `memside::to_record(harness, device)`. Each command is tied to its burst
     row, and each burst to its request.
   - **The tool:** `kpu-memsim --deploy --scenario --out`. Scenarios are strict JSON, and
     `replay_matmul` replays a matmul schedule. It prints bandwidth against the ceiling, bursts
     by outcome, commands, refusals and ejection waits. Exit codes: 0 ok, 1 unfinished, 2 bad
     input.
   - **Fixture:** `tests/memside/t4_mixed.json`, with three ctests. On the T4 it reaches 90% of
     the ceiling across 6,400 bursts.
   - **Found while building it:**
     - The record's integrity check caught step 1 dropping a pending opener's tag. FR-FCFS can
       close a row before its opener's CAS. Fixed: the precharge erases an opener's tag only once
       its burst has completed.
     - Ejection waits were logged every cycle (52,644 events in the T4 scenario). They are now
       logged once per ejection.

4. **Checker.** `mflow_check.py` with TF10, TF11 and M1-M5, plus a self-test.
5. **Viewer.** The memflow page: banks, channels, engines, buffers, ports, the address view, the
   inspector, and a smoke test.
6. **Characterization.** `--sweep`, `sweep.json` and the viewer tab, with the step-3 regression.

## 5. Decided on Review (2026-10-08)

All five went with the recommendation:
- **Q1:** a new `.mflow` bundle.
- **Q2:** a `CommandObserver` callback.
- **Q3:** all five request models.
- **Q4:** port stubs at the injection-bus rate, with input-queue back-pressure.
- **Q5:** a new memflow page.

## 6. Questions (as posed)

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Where the per-bank record lives (DRAM step 4) | (a) a new `.mflow` bundle sharing the `.tflow` container code; (b) `.tflow` v4 with optional DRAM tables | **(a).** `.tflow` is tile-granular and L-T1; per-burst and per-command data is a different grain from a different producer. Sharing the container keeps the readers and the LOD code common. The CSP executor can write both once it records (#283) |
| Q2 | How the controller is observed | (a) a `CommandObserver` callback added to the LPDDR5 controller; (b) enable its existing trace and post-process the `TraceEntry` list | **(a).** The trace lacks the row and is in controller cycles. A callback gets the row and the request at the moment of issue; unset, it costs one null check per command |
| Q3 | First request models | stream, strided, random, matrix tiles, schedule replay | **all five**; replay ties the harness to real schedules |
| Q4 | Port-stub fidelity | (a) the injection bus at the NoC link rate, with an input-queue depth from the spec; (b) infinite sink only | **(a), with infinite as an option.** The point is to see the memory side under the port's actual back-pressure |
| Q5 | The viewer | (a) a new page sharing the loader; (b) a mode in the tile-flow viewer | **(a).** Its rows are banks and commands, not tiles and stations |
