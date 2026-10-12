# Domain-Flow Debugger: CSP State at L-B, Domains Moving Through the Machine (#363)

**Date:** 2026-10-11
**Status:** Held (2026-10-11). The architect ruled that L-B must model L3 placement; this plan resumes after
`docs/plans/l3-placement-at-lb.md` step 3, with slots in its machine view. §3.7's rule "L-B models a capacity,
not placement" is wrong and is superseded by ADR 0004 §5.
**Issue:** #363
**Builds on:** `driver::CspBehavioralStepper` (`include/sw/kpu/program/driver/csp_step.hpp`), the
CSP walker (`csp/lang/walk.hpp`), the columnar bundle container (`record/columnar.hpp`), and the
tile-flow, memory-side and system-schedule debuggers (`docs/plans/tile-flow-debugger.md`,
`memory-side-debugger.md`, `system-schedule-debugger.md`).

## 1. Problem Statement

A CPU or GPU state debugger shows all of the machine's concurrent state and redraws it every
tick. That does not scale to a KPU:
- one compute tile has 512 or more ALUs;
- each ALU holds an A, a B and a C element live in the same tick;
- a frame of that is tens of thousands of changing values, which nobody can read.

The premise of a CSP-driven push architecture is that the machine's granularity rises to
**domains of computation**: tiles, panels and accumulators, moving through the machine's
resources in a structured way under credits. A debugger whose frame is *domains in resources*,
and whose step is *one CSP state transition*, stays readable at any machine size. The per-ALU
state becomes a drill-down.

What exists does not show this:
- **`kpu-run --step`** prints one line per action: index, kind, tile and L3 occupancy.
  - It does not say which process acted, which residency was opened or released, or where the
    domain came from and went.
  - It does not say where in the program the action is: the source line, or the loop indices.
  - It pre-counts every action in a full walk of the program, which cannot scale: a 1M x 1M
    matmul in 32-tiles is about 3 x 10^13 actions.
- **The tile-flow debugger (`.tflow`)** draws tiles over the floorplan *in time*, from the L-T1
  run. It shows when and where, not the CSP program's state.
- **`.mflow` and `.sflow`** show DRAM commands, DMA burst arbitration and dispatch to the compute
  fabric. They are the levels below.

None of them answers the question this debugger is for: **which domains are where, which process
just moved one, and where in the program we are.**

**Why L-B first.**
- L-B is the oracle: the one place a CSP program becomes an L0 trace
  (`docs/plans/retire-l0-derivations.md`). Every finer level refines what it shows.
- L-B's transaction is a whole block move, applied atomically, which is a CSP state transition.
- L-B has no time, so the frame is pure state. Timing arrives later as an overlay from L-T1
  (step 5), joined by action index.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        kpu-run --dflow <dir>  (one L-B run)                       │
│                                                                                    │
│  ┌──────────────┐  Emission{action, op,  ┌─────────────────────┐                  │
│  │ .csp program │  l0, opens, loaded,    │ BehavioralInterpreter│  values, held(c) │
│  │  (ActionStream│  where}  ───────────► │  (L-B, the oracle)   │ ───────────┐     │
│  │   / Walker)  │   one action per step  └─────────────────────┘            │     │
│  └──────┬───────┘                                                           ▼     │
│         │ where: source line + loop indices          ┌────────────────────────────┐│
│         └──────────────────────────────────────────► │ DomainFlowRecorder         ││
│                                                       │  steps, residencies,       ││
│                                                       │  iterations, sampled tiles ││
│                                                       └─────────────┬──────────────┘│
└─────────────────────────────────────────────────────────────────────┼──────────────┘
                                                                      │ columnar.hpp
                                                                      ▼
                         run.dflow/  manifest.json  steps.bin  residency.bin
                                     where.bin  iteration.bin  values/ (opt-in)
                                                                      │
                                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│          tools/visualization/domainflow/index.html  (static; reads, never runs)    │
│                                                                                    │
│   machine view                          program view            inspector          │
│   ┌──────┐ dma  ┌──────┐ bm  ┌─────┐ str ┌────┐   csp 1.0          tile C[2,3]     │
│   │ DRAM │────►│  L3  │───►│ L2  │───►│ CF │   for j in 0..8 {    float32, 64x64  │
│   │      │◄────│ 9/17 │◄───│ 2/- │◄───│ 1  │ ►   resident B[:,j]  max |Δ| vs L-B  │
│   └──────┘store└──────┘ wb  └─────┘drain└────┘     for i ...  j=3 i=2              │
│   L3 slots by address (module:slot/bank)           step 1204 of 3968               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

The machine view is **the program's channels**, not the floorplan. At level 1 a CSP program
addresses one flat machine: one L3, its L2 and the compute fabric (`machine flat(l3 = N)`), with
one process on each edge. The floorplan belongs to the tile-flow debugger, which draws the
placed machine in time.

## 3. Design

### 3.1 What a domain is: three zoom levels

| Zoom | Domain | Identity | Lifetime |
|---|---|---|---|
| tile | one operand tile | `TileCoord` (operand, ti, tj) | as long as the operand exists |
| residency | a tile held in a channel | `Action::residency` | open (`load`, `writeback`, `inherit`) to close (`release`, `retain`) |
| working set | one loop iteration's residencies | the loop indices (`where`, §3.3) | the iteration's first action to its last |

The default frame is residencies in channels. Zooming out groups them by loop iteration
(B's column panel, A's row panel). Zooming in shows a tile's contents (§3.6).

### 3.2 The step: one CSP state transition

```cpp
// include/sw/kpu/program/record/domain_flow_record.hpp
struct DomainStep {
    std::size_t action;                 // index in program order: the join key for later levels
    csp::Action::Kind kind;             // load, move, feed, call, drain, writeback, store, release, ...
    csp::ProcessKind process;           // who acted (a Release: none)
    csp::Chan from, to;                 // csp::from_chan / to_chan of the kind
    TileCoord tile;
    std::size_t residency;              // csp::kNone when the action has none
    bool opens = false, closes = false;
    std::array<std::size_t, csp::kChannels> held{};   // occupancy of each channel AFTER the action
    std::uint32_t where;                // index into the where table (§3.3)
    std::vector<csp::Stage> context;    // the tile context the action applies (bias, activation)
};
```

Each field already exists at L-B, except `where`:
- `kind`, `tile`, `residency`, `context` and `process` are on `csp::Action`;
- `from` and `to` come from `csp::from_chan` and `to_chan`;
- `opens` and `loaded` are on the `Emission`;
- `held` is `BehavioralInterpreter::held(chan)`.

```
WRONG — the recorder re-derives the residency or the channel from the tile name
        (a second model of the program's state; it drifts from the one that ran)
CORRECT — every field is read from the Emission and the interpreter that executed the action
```

### 3.3 Where the action is in the program: the walker reports it

To step by loop iteration, every action must carry its position: the statement's source line and
the indices of its enclosing loops. The walker has both (its `Frame` stack holds each loop's
`var`, `value`, `hi` and `line`), but `Emission` does not carry them.

```cpp
// csp/lang/walk.hpp
struct Where {
    int line = 0;                                   // the statement that emitted the action
    struct Loop { std::string var; long long value, hi; int line; };
    std::vector<Loop> loops;                        // outermost first
};

struct Emission {
    Action action;
    std::optional<TileOp> op;
    std::vector<TileOp> l0;
    bool opens = false, loaded = false;
    Where where;                                    // NEW: filled by the walker as it emits
};
```

```
WRONG — the viewer (or the recorder) parses the .csp text and re-walks the loops to find where
        action N came from: a second walker, which is a second source of the program's order
CORRECT — the one walker that drives L-B reports the position it emitted from
```

The record interns `where`: consecutive actions from one statement in one iteration share an
entry, so the table grows with iterations, not with actions.

### 3.4 The record: `.dflow`, in the shared columnar container

```
run.dflow/
  manifest.json    format "dflow" version 1; level "L-B"; RunIdentity (program digest, inputs
                   digest, deployment digest); the program's text; channels with capacities
                   (as the program declares them); processes; operand table (name, shape,
                   tiling); step count; what was sampled (§3.6); what is not modelled at L-B
  steps.bin        u64 action, u8 kind, u8 process, u8 from, u8 to, u32 tile, u64 residency,
                   u8 flags (opens, closes), u64 where, u16 held[4], u32 l3_tile, u32 slot,
                   u32 bank (the action's L3 address, ADR 0004 §5)
  residency.bin    u32 tile, u8 chan, u64 open, u64 close, u32 l3_tile, u32 slot, u32 bank,
                   u8 flags (loaded, dirty, inherited, retained, pinned)
  where.bin        u32 line, u32 loop_first, u16 loop_count;  loops.bin: u32 var, i64 value,
                   i64 hi, u32 line
  iteration.bin    per (loop depth, iteration): u64 first_step, u64 last_step, u64 loads,
                   u64 stores, u16 peak[4]       -- the summary the zoomed-out view reads

  Action, residency and step indices are u64 throughout: a large operator has more than 2^32
  actions (a 1M x 1M matmul in 32-tiles has about 3 x 10^13), and the complete iteration summary
  and late residency closes must index them.
  values/          opt-in (§3.6): f32 tile contents at sampled steps
```

- It is written and read through `record/columnar.hpp`, like `.tflow` and `.mflow`: each column is
  little-endian and padded to 8 bytes, so the viewer maps it as a typed array without parsing.
- It is a new format. The case for one: neither `.tflow` (stations in time) nor `.mflow` (DRAM
  commands) has a notion of a CSP action, a residency's open and close, or a program position.
  This is raised for review (Q5).

### 3.5 Scale: lazy, bounded, summarised

- **No pre-count.** `CspBehavioralStepper` walks the whole program once to learn `size()`. The
  recorder steps lazily: the total is known only once the stream ends, and the manifest is written
  last.
- **Bounded detail, complete summary.** `--dflow-limit N` (default 1,000,000 actions, like
  `--trace-limit`) caps `steps.bin`. `iteration.bin` is written for the whole run, so a program
  too long to record step by step can still be read iteration by iteration.
  - **The limit bounds the file, not the work.** L-B runs to the end regardless, since the oracle
    is never truncated, and the iteration summary visits every action. A run is as long as L-B's
    run of the program. At 3 x 10^13 actions that is infeasible with or without a record, and a
    record cannot make it otherwise.
  - Bounding the work as well would need a different contract: stop L-B after N actions and
    write a summary marked partial, which is no longer the oracle's run. That is out of scope
    here.
  - The manifest says what was cut, and where.
- **The viewer aggregates by loop depth.** At depth 0 it shows one row per outermost iteration;
  expanding an iteration loads its steps, a range of `steps.bin`.

### 3.5a Navigation

The viewer reads a record, so it can go backwards as easily as forwards: a step back is a read of
the previous row. Nothing is executed in reverse.

- **step**, **back**;
- **run to**: the next action of a kind, the next action on a tile, the release of the selected
  residency;
- **next iteration at depth d**: the structured analogue of a breakpoint (the next `j` panel, the
  next `i` row);
- **jump to step N** and **jump to source line**.

### 3.6 Inspecting a domain's contents

Select a tile to see its contents, decoded through `tile_check`'s interpretation layer
(`ElementFormat`; float32 today). The record cannot hold every tile at every step in general: that is the
program's whole data traffic (64-tiles are 16 KiB each; a large operator moves far more than any
file holds). Instead:
- `--dflow-values <operand[ti,tj],...>`, or `all` for small programs under a byte cap, records
  the tile's contents at each step that writes it;
- the final value of every operand is always recorded, from L-B's result, so the end state can be
  inspected without opt-in.

At L-B the values *are* the oracle, so the inspector shows values only. With the L-T1 overlay
(step 5) it shows the per-tile proximity to L-B, `TileCheck`'s `max_abs` and `max_rel`.

### 3.7 Honesty rules the viewer must follow

```
WRONG — a time axis, or "cycles", on an L-B record          (L-B has no time)
CORRECT — the axis is "step (action)"; time appears only with an L-T1 overlay, labelled L-T1

WRONG — an L2 or CF capacity drawn as full or empty when the program declares none
CORRECT — drawn as "unbounded at L-B", with the count of tiles held

WRONG — residency slots drawn at fixed positions, as if L3 had addresses
CORRECT — slots are an ordering of open residencies; L-B models a capacity, not placement
          [SUPERSEDED 2026-10-11, ADR 0004 §5: L-B places every residency; draw it in its slot]

WRONG — the viewer computes a value, a residency, or a loop index
CORRECT — everything shown is read from the record the oracle's run wrote
```

## 4. Implementation Steps

Each step is one PR, reviewed and merged before the next.

1. **The walker reports where; the L-B stepper carries the step's state.**
   - `csp/lang/walk.hpp`: `Where` on `Emission`, filled by the walker as it emits.
   - `driver/csp_step.hpp`: `CspStep` gains `process`, `from`, `to`, `residency`,
     `opens`/`closes`, `held[]` and `where`. The size pre-count becomes optional: `size()` is known
     only once the stream ends, unless asked for.
   - `kpu-run --step` prints the process and the source position.
   - Tests: `tests/program/test_csp_step.cpp`, `test_csp_structured.cpp`.
     - `where` matches the loop indices of known programs (matmul `j`/`i`/`k`, LU's `k`).
     - Stepping with `where` changes no value: the result is bit-identical to `run` (same
       computation).
2. **The record.**
   - New `include/sw/kpu/program/record/domain_flow_record.hpp`: `DomainFlowRecorder`, plus
     `write_dflow` and `read_dflow` through `columnar.hpp`.
   - `kpu-run --dflow <dir> [--dflow-limit N] [--dflow-values ...]`.
   - New `tools/trace/dflow_check.py` checks the §6 invariants on a bundle.
   - Tests: `tests/program/test_domain_flow_record.cpp` (round trip; invariants; a limit that
     cuts `steps.bin` keeps `iteration.bin` whole), and a kpu-run CMake test that writes and checks
     a bundle for each CSP corpus program.
3. **The viewer.**
   - `tools/visualization/domainflow/{index.html, pack.py, README.md}`, following `tileflow/`: the
     machine view, the program view, navigation (§3.5a), and the iteration summary.
   - Reference run: matmul 512³ in 64-tiles on `kpu_s1.json` (T4 folded into one site).
   - `test_viewer.py` loads a packed bundle headless and checks that steps, residencies and the
     program position render.
4. **Tile inspection.** Values in the record (§3.6), and the inspector decoding them through
   `ElementFormat`.
5. **The L-T1 overlay.**
   - Join L-T1's records (`CspLevelOutcome::records`, each carrying its `action`) on the action
     index: per step, its start and finish, the lane, credit stalls.
   - The inspector shows proximity to L-B. The axis may then switch to time, labelled L-T1.

## 5. Verification

```bash
nice -n 19 cmake --build build -j 2 --target test_csp_step test_domain_flow_record kpu-run
cd build && nice -n 19 ctest -R "csp_step|domain_flow|kpu_run_csp_dflow" --output-on-failure

# A bundle for the reference run, then its invariants
build/tools/csp-gen/kpu-csp-gen --algo matmul --size 512 --tile 64 --l3 17 -o mm512.csp
build/tools/kpu-run/kpu-run --program mm512.csp --deploy tests/program/deploy/kpu_s1.json \
    --level behavioral --dflow /tmp/mm.dflow
python3 tools/trace/dflow_check.py /tmp/mm.dflow          # exit 0; exit 1 lists violations
```

Expected:
- `dflow_check` passes on every CSP corpus program and on the reference run.
- The record's step count equals `csp_reference(...).actions`.
- Its final values are bit-identical to `csp_reference(...).values`: the same L-B run, so the same
  computation.
- Recording does not change L-B's values or the trace: `test_csp_corpus` and `test_l0_corpus`
  pass with no regenerated file.
- The viewer's headless test renders the reference bundle.

## 6. Key Invariants

1. **The record is the oracle's run.** It is written by the L-B run that produces `csp_reference`'s
   values and trace, never by a separate walk or model.
2. **One step, one action**, in program order: `steps[i].action == i` for every recorded step.
3. **Occupancy is the residencies:** `held[c]` after step i equals the residencies in channel c
   opened at or before i and not yet closed.
4. **Every residency closes once, after it opens**, or is marked retained when the program ends.
   Nothing is read after its release (L-B refuses such a program, and the record shows the
   refusal at its step).
5. **Capacity holds:** `held[c] <= capacity(c)` wherever the program declares a capacity.
6. **The summary is complete:** `iteration.bin` covers every action, even when `steps.bin` was
   cut. Its loads and stores add up to the run's.
7. **The viewer never executes.** Every value, residency and index it shows is read from the
   record.
8. **No time at L-B.** An L-B record has no time column; time enters only with the L-T1 overlay.

## 7. Questions for review

- **Q1. File or live?**
  - **Recommended:** a file (`.dflow`), as the other three debuggers use. It is reproducible,
    shareable, and makes stepping back trivial.
  - The alternative is a live session that drives `CspBehavioralStepper` over a socket. It would
    reach programs too large to record, but it adds a server and a protocol.
- **Q2. Values in the record.**
  - **Recommended:** opt-in per tile, plus the final operands always.
  - The alternative is everything under a byte cap, which is simple but useless past small
    programs.
- **Q3. Scale.**
  - **Recommended:** a step limit plus an iteration summary that is always complete (§3.5).
  - The alternative is to sample steps (every k-th), which keeps the span but loses the
    "one step, one action" invariant.
- **Q4. The machine view's layout.**
  - **Recommended:** the program's channel graph (DRAM, L3, L2, CF, with the processes on the
    edges), with **L3 drawn as its addressed slots**. Each residency sits at its L3 address
    (module, slot, bank; ADR 0004 §5), grouped by bank, so placement, slot reuse and bank
    occupancy are visible at L-B.
  - This is the program's address space, not the T4 floorplan. The floorplan, with the modules'
    physical sites and the NoC, belongs to the tile-flow debugger and to the L-T1 overlay.
- **Q5. A new format, `.dflow`?**
  - **Recommended:** yes, in the shared columnar container. No existing bundle has actions,
    residency lifetimes or program positions.
  - The alternative is to extend `.tflow` with an L-B section. That couples the two viewers, and
    `.tflow`'s tables are time-indexed.
- **Q6. The first program.**
  - **Recommended:** matmul 512³ in 64-tiles on `kpu_s1.json` (L3 17): 3,968 actions
    (measured) in panel loops, which is readable, and the same operator as the tile-flow reference run.
  - LU, with its in-place factorization and pivot-ordered loops, is the second.
