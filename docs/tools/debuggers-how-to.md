# How to Work with the KPU Debuggers

The simulator has two record-and-view debuggers. They cover the two halves of the machine:

| | Tile-flow debugger | Memory-side debugger |
|---|---|---|
| **Question it answers** | Where was every tile, and when? | What did the DRAM, controllers, DMA engines and their buffers do? |
| **Scope** | The whole program: DRAM -> L3 -> compute and back | Outside the NoC: DRAM, MCs, DMA engines, store buffers; NoC ports stubbed |
| **Producer** | `kpu-run --tflow` (L-T1, block-sequential) | `kpu-memsim --out` (CSP memory side, hosted LPDDR5 controller) |
| **Record** | `.tflow` bundle (tile residency, transits, computes) | `.mflow` bundle (bursts, DRAM commands, requests, buffers, port events) |
| **Grain** | one tile move | one DRAM burst and one DRAM command |
| **Checker** | `tools/trace/tflow_check.py` (TF1-TF9) | `tools/trace/mflow_check.py` (TF10, TF11, M1-M5) |
| **Viewer** | `tools/visualization/tileflow/` | `tools/visualization/memflow/` |
| **Plan** | `docs/plans/tile-flow-debugger.md` (#286) | `docs/plans/memory-side-debugger.md` |

Both follow the same workflow, **record -> check -> view**:
1. A tool runs the simulation and writes a bundle: a folder of a JSON manifest plus binary
   columns.
2. A checker turns invariants into exit codes.
3. A static HTML page draws the bundle. It needs no server and no build step.

Use the checker first: it says *whether* something is wrong. The viewer shows *where* and
*when*.

Paths below assume the default build tree, `build/` (`cmake --preset release && cmake --build
--preset release`).

---

## 1. Which debugger do I need?

- **A program is slow, or a tile arrives late:** start with the tile-flow debugger. It shows
  the whole pipeline at tile grain, so you can see which station starves or saturates.
- **A suspected L3 capacity or lifetime bug** (a slot freed early, a tile read before it
  arrived): use the tile-flow debugger. `tflow_check.py` TF1, TF6 and TF9 exist for exactly
  this (#279).
- **DRAM bandwidth is below the ceiling:** use the memory-side debugger. Reach for it whether
  the cause looks like page conflicts, too few bursts in flight, a store buffer that fills, or
  NoC-port back-pressure.
- **A DRAM controller, DMA window or store-buffer change:** use the memory-side debugger. Run a
  scenario before and after, and compare.
- **Choosing address layouts or tile shapes for DRAM locality:** use the memory-side debugger.
  Its `matrix_tiles` and `replay_matmul` models, together with the heat map, show the page
  behaviour of a layout.

The memory-side debugger runs the memory side **on its own**. The NoC is replaced by port stubs
that offer requests and sink loads, so the DRAM-side behaviour can be studied without the rest
of the machine's timing mixed in.

---

## 2. The tile-flow debugger

### 2.1 Record

```bash
build/tools/kpu-run/kpu-run --algo matmul --size 256 --tile 32 --level block-sequential \
    --deploy tests/program/deploy/kpu_t4.json --tflow run.tflow
```

- `--tflow` needs a level with timing (`block-sequential`); a behavioral-only run refuses it.
- `--deploy` takes the machine from a deployment spec. The fixtures are in
  `tests/program/deploy/`: `kpu_t4.json` (the reference SKU), `kpu_t16.json` and
  `kpu_t64.json`.
- `--program file.l0` runs a serialized program instead of `--algo`.

### 2.2 Check

```bash
python3 tools/trace/tflow_check.py run.tflow            # add --json or --verbose
```

Exit **0** means every invariant holds. Exit **1** means violations: read them and fix the root
cause. Exit **2** means the bundle could not be read, so there are no violations to read.

| Check | Invariant |
|---|---|
| TF1 | L3 occupancy never exceeds capacity |
| TF2 | every interval lies within 0 .. makespan |
| TF3 | no two transits overlap on one lane of a mover pool |
| TF4 | no two computes overlap on one compute tile |
| TF5 | a tile's L3 residencies never overlap |
| TF6 | every compute and transit runs while each tile it touches holds its L3 slot |
| TF7 | an op's hops chain correctly: kinds, order and endpoints |
| TF8 | the level-of-detail pyramid is the exact merge of the raw events |
| TF9 | every L3 residency was filled, by a seed, a DMA delivery or a write |

### 2.3 View

```bash
# optional: the floorplan, and the derived per-resource binding
build/tools/floorplan/kpu-floorplan --deploy tests/program/deploy/kpu_t4.json --json t4_floorplan.json
python3 tools/visualization/tileflow/bind.py run.tflow --floorplan t4_floorplan.json

# one self-contained page
python3 tools/visualization/tileflow/pack.py run.tflow --floorplan t4_floorplan.json -o run.html
```

Open `run.html`. Alternatively, open `tools/visualization/tileflow/index.html` and choose the
bundle folder (and the floorplan) with the pickers.

**What you see:**
- the floorplan at the cursor;
- diagnostics;
- an inspector;
- the DRAM address space (with a binding);
- station swimlanes, as occupancy over capacity per time bin;
- the pooled L3 occupancy against capacity.

**What it will not invent:** at L-T1, L3 and the movers are *pooled*. The record holds a
pool's occupancy, not a per-tile one, so the page draws pools as pools and hatches what is not
modelled (L2, L1). `bind.py` derives per-resource places (DRAM layout, DMA engine, home L3
tile, BlockMover, NoC route) by stated policies. Every derived row is labelled as derived: the
timing is the executor's, the place is a policy.

---

## 3. The memory-side debugger

### 3.1 Record

```bash
build/tools/kpu-memsim/kpu-memsim --deploy tests/program/deploy/kpu_t4.json \
    --scenario tests/memside/t4_mixed.json --out run.mflow
```

The tool prints:
- the makespan;
- bandwidth against the DRAM ceiling;
- bursts by page outcome (hit, empty, conflict);
- commands, refusals and ejection waits.

**Exit codes:** 0 means the run completed and was recorded. 1 means it did not finish within
`max_cycles`; it still writes what it saw, and says so. 2 means bad arguments, spec or
scenario. `--device N` picks a device of a multi-device spec.

### 3.2 Write a scenario

A scenario is strict JSON: an unknown key is refused by name, not ignored. Any number may be
given as a `"0x..."` string.

```json
{
  "window": 32,
  "ports": { "consume_latency": 16, "eject_interval": 64 },
  "streams": [
    { "port": "first", "model": { "kind": "stream", "load": true,  "base": "0x100000", "count": 64, "bytes": 4096 } },
    { "port": "first", "model": { "kind": "stream", "load": false, "base": "0x800000", "count": 32, "bytes": 4096 } },
    { "port": "first", "model": { "kind": "matrix_tiles", "load": true, "base": "0x1000000",
                                  "rows": 64, "cols": 64, "element_bytes": 4,
                                  "tile_rows": 32, "tile_cols": 32, "pitch": 8192 } }
  ]
}
```

| Key | Meaning |
|---|---|
| `window` | bursts each DMA engine keeps in flight (W; the spec's `dma.window` otherwise) |
| `store_buffer_blocks` | depth of each DMA engine's store buffer |
| `l3_slots` | the stand-in L3's credit pool |
| `max_cycles` | cut-off; past it the run exits 1 |
| `ports.infinite` | sink loads at once (no injection-bus limit) |
| `ports.block_cycles` | cycles the injection bus takes per block |
| `ports.input_queue_blocks` | depth of a port's per-engine input queue; full means refused |
| `ports.consume_latency` | cycles before a landed load frees its stand-in L3 slot |
| `ports.eject_interval` | cycles between ejections per engine (the store traffic) |
| `streams[].port` | the port index, or `"first"`; a stream's requests are dealt round-robin over the DMA engines attached to that port |
| `streams[].issue_interval` | cycles between offers per engine (0 = everything offered at cycle 0) |
| `streams[].model` | the request model, below |

**Request models** (`model.kind`; `load: true` is a DMA read toward L3, `false` a store):

| Kind | Fields | Generates |
|---|---|---|
| `stream` | `base, count, bytes` | `count` sequential requests of `bytes` |
| `strided` | `base, count, bytes, stride` | requests `stride` apart |
| `random` | `base, count, bytes, region, seed` | `bytes`-aligned requests in `[base, base + region)`, deterministic per seed |
| `matrix_tiles` | `base, rows, cols, element_bytes, tile_rows, tile_cols, pitch` | one request per tile row, rows `pitch` apart (a 2D DMA) |
| `replay_matmul` | `M, N, K, tile` | the LOAD/STORE tiles of a matmul schedule |

Fixtures are in `tests/memside/`: `t4_mixed.json` is the reference; the `bad_*.json` files are
the refusal tests.

### 3.3 Check

```bash
python3 tools/trace/mflow_check.py run.mflow            # add --json or --verbose
```

The exit codes are the same as `tflow_check.py`'s.

| Check | Invariant |
|---|---|
| TF10 | a bank has one open row: between an ACT and its PRE, every RD/WR names the ACT's row |
| TF11 | a channel's data bus never carries two bursts at once |
| M1 | an engine never has more than W bursts in flight |
| M2 | a store buffer never holds more than its capacity |
| M3 | a request's first burst comes no earlier than its credit (L3 slot or buffer slot) |
| M4 | every burst completes once; a request's bursts are exactly the ones its bytes span |
| M5 | DRAM timing (tRCD, tRP, tRRD, tFAW, tCCD) against the declared device's timing table |

### 3.4 View

```bash
python3 tools/visualization/memflow/pack.py run.mflow -o run.html
```

Open `run.html`. Alternatively, open `tools/visualization/memflow/index.html` and choose the
`.mflow` folder (or drop its files on the page). A bundle over about 12 MB is refused by
`pack.py`; view those from the folder.

**Panels, top to bottom:**
- **Summary chips:** device, makespan, bytes, bandwidth as a percentage of the ceiling, window,
  bursts, commands, page outcomes, and the DRAM timing table's provenance.
- **Bandwidth over time:** read and write bytes per bin against the dashed ceiling.
- **Observations:** computed from the record. They are *not* invariants; those are the
  checker's job.
- **Swimlanes:**

  | Lane | Shows |
  |---|---|
  | `port[k] loads in` | loads crossing the injection bus, from acceptance to landing; red ticks are refused offers |
  | `port[k] ejections` | ejections delivered; amber ticks are ejections waiting for a store-buffer slot |
  | `dma[e] bursts / W` | bursts in flight against the window (dashed line = W) |
  | `dma[e] requests` | each request from offer to retirement (blue load, orange store) |
  | `dmabuf[e] / cap` | store-buffer slots held (area) and staged (line) against capacity |
  | `mc/ch/bus` | the channel's data bus, burst by burst |
  | `mc/ch/bg/ba` | every DRAM command: ACT green, RD blue, WR orange, PRE grey, REF purple; red under an ACT means a row conflict |

  Banks start folded; click **Banks** to unfold them. Hovering a bank lane shows the open row
  at that cycle.
- **Heat map:** the bursts in view, by bank and by the rows in use, as count, conflicts or hit
  rate. It follows the zoom; click a cell for its numbers.
- **Inspector:** click a command to see its burst (coordinates, page outcome, submit, first
  command, data window, queueing delay), the burst's whole command chain, and its request.
  Click a request lane for the request and its bursts' outcomes.
- **Page outcomes by bank.**

### 3.5 Reading the memory side: common patterns

| You see | It means | Look next |
|---|---|---|
| Bandwidth well under the ceiling, `dma[e] bursts` flat below W | the engines are not offered enough work (too few requests, or a large `issue_interval`) | the request lanes, port refusals |
| `dma[e] bursts` pinned at W, bandwidth still low | the window is not the limit: DRAM is | bank lanes for conflicts, the heat map |
| Many red marks under ACTs; the heat map in conflicts mode is hot in a few banks | page conflicts: requests in one bank alternate rows | the address layout or `pitch`; the inspector's command chain shows PRE -> ACT before each CAS |
| Gaps on a channel bus while bursts wait | bank timing (tRCD/tRP/tFAW) or a refresh | bank lanes at that cycle (REF is purple) |
| `dmabuf[e]` at capacity and amber ticks on ejections | stores arrive faster than DRAM writes drain them | write bursts on the channel bus; the store buffer depth |
| Red ticks on `port[k] loads in` | the port's input queue is full: injection-bus back-pressure | `ports.block_cycles`, `input_queue_blocks` |
| A request with a long lifetime | queued behind others, or its bursts conflicted. With `issue_interval` 0 every request is offered at cycle 0, so the last ones look long by construction | click it: credit vs first burst vs retirement |

Reference numbers on the T4 (`t4_mixed.json`): 90% of the 34.1 B/cycle ceiling across 6,400
bursts. In DRAM plan step 3's measurement, one engine with W = 32 reached 95%, and 32 engines
with W = 1 reached 69%. The difference is locality, not just Little's law.

---

## 4. Controls common to both viewers

| Action | How |
|---|---|
| Zoom in | drag across the swimlanes, or wheel up over them (zooms around the pointer) |
| Zoom out | wheel down, **Zoom out**, or `-` |
| Previous view | **Back**, or Backspace |
| Whole run | **Whole run**, or `0` |
| Move the cursor | click the swimlanes, or drag the scrub bar |
| Play | **Play** sweeps the cursor across the current view |
| Fold a category | click its header |
| Inspect | click a lane item |
| Dark mode | follows the system setting |

The time unit everywhere is executor cycles.

---

## 5. Sharing a run

`pack.py` writes one HTML file with the bundle embedded. It has no external requests, so it can
be attached to an issue or a session log, or archived. Both viewers refuse a bundle whose
format version they do not know, rather than drawing it wrong. Each viewer's
`test_viewer.py` pins that version against the writer.

---

## 6. Tests that keep the debuggers honest

| ctest | Checks |
|---|---|
| `tflow_check_*`, `tflow_check_selftest` | the tile-flow checker on real runs, and that each check catches its broken case |
| `tileflow_viewer_smoke`, `tileflow_bind_t4`, `tileflow_bind_t64` | pack, version, script parse, the binding |
| `kpu_memsim_t4`, `kpu_memsim_rejects_*`, `kpu_memsim_unfinished` | the memory-side run and its refusals |
| `mflow_check_t4`, `mflow_check_selftest` | the memory-side checker on the T4 run, and that each check catches its broken case |
| `memflow_viewer_smoke` | pack, version, script parse, the page's reader on the real bundle |

Run them with `cd build && ctest -L tileflow` or `ctest -L memside`.
