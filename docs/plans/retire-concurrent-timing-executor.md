# Retiring ConcurrentTimingExecutor: the cycle core becomes L-CA

**Date:** 2026-11-27
**Tracking:** epic #619 (#620–#628)
**Status:** Design; re-sequenced behind `docs/plans/distributed-csp-programs.md` and the NoC rework
(2026-11-27: the NoC is a push-only, one-hop XY mesh of L3+CT nodes with a separate credit network; §7
decision 1 is answered, and track A's path runs on that mesh)
**Milestone:** v0.10.0 (tracks A, B and R0–R2); v0.11.0 (R3–R6), subject to review

## 1. Problem Statement

`ConcurrentTimingExecutor` (`include/sw/kpu/timing/`, about 12.9k lines, the 2741-line executor and its process
files) is today's L-CA. It was built on the wrong resource model:
- its resources are pools and queues, not the machine's components;
- its links carry no beats, and its clocks are not the machine's domains;
- its double-buffering speedup on the RTL's 8×8×16 is 0.96×, where the RTL shows 0.82× and the cycle core
  0.65× (`docs/analysis/cycle-prototype-evaluation.md`).

Decision 5 is taken: the cycle core (`include/sw/kpu/cycle/`) replaces it as L-CA.

The cycle core cannot be trusted as L-CA until a person can check it by hand, against the same views as the
RTL. Three gaps block that:

1. **No program small enough to trace by hand.** The smallest cycle-core runs are whole gemms. Nobody can
   follow one vector from DRAM, through the memory controller, a DMA engine and the NoC, into a named L3 tile,
   and confirm each cycle.
2. **The memory path is pinned to one L3 tile.** `plan_l3` refuses any L3 module but 0
   (`csp_l3_program.hpp:157`). `run_memory_subsystem` attaches one engine on port 0 and one L3 tile on hub 0
   (`memory_subsystem.hpp:115-121`). So a program cannot ask for "this vector, onto L3 tile 1", and the NoC
   is crossed in the most trivial way only.
3. **No shared view of that path.** The views exist on their own:
   - `.mflow` records DRAM and the DMA, but no NoC hops or L3 writes.
   - The floorplan viewer (`tools/visualization/floorplan/`, `kpu-structure` v1) draws the machine, but no
     activity.
   - The RTL's `kpu-gen-trace` v3 bundle has `hops` and `transfers` tables, but kpu-sim reads none of them.

   ADR 0005 puts every shared view in kpu-sim; this is the part of it that L-CA's validation needs.

The retirement itself is wide:
- 55 test targets in `tests/timing/`, and about 20 in `tests/program/`;
- about 20 kpu-run CLI tests;
- `kpu-memsim`, and the operator-schedule and network runners (`timing/schedule/`, `timing/graph/`, the
  m1–m3 examples, the resnet baseline);
- the `.dflow` 6.2 timed record, `--bubbles`, `--step` and the L-T1/L-CA calibration line;
- four cycle-core headers that still include `timing/`.

## 2. Architecture

```
   Track A: traceable programs        Track B: shared views                Track R: retirement
   ───────────────────────────        ─────────────────────                ───────────────────
   csp program: ONE vector             kpu-structure v1 (block ids)         R0 ADR 0006, freeze CTE
     dram[addr,+N] -> L3 tile k         ▲                 ▲                R1 move shared headers
     slot s, via DMA engine e           │ kpu-floorplan    │ esg floorplan  R2 --level cycle-accurate
            │                           │ --structure      │ (converted)       = the cycle core
            ▼                           │                  │               R3 .dflow / bubbles / step
   LPDDR5 ── MC ── DMA[e] ── port ── hub ── hub ── L3[k]   │               R4 kpu-memsim on the core
            cycle core (memory_subsystem, NocFabric)       │               R5 schedule/graph runners
            │                                              │               R6 delete timing/
            ▼                                              │
   .mflow v5: bursts, commands  +  hops, l3 writes ──► floorplan viewer: path playback
            │      (block ids)                         memflow viewer:   DRAM / DMA detail
            ▼                                          kpu-debug-server: serves both
   golden hand trace (tests/cycle/trace/*.golden) ◄─── RTL: kpu-gen-trace v3 → mflow_from_kpugen.py
```

Tracks A and B make L-CA checkable. R2 is the switch, and it waits for A's first rungs.

## 3. Design

### 3.1 Track A: one vector, to a named L3 tile

The CSP program already places every residency: `place{l3_tile, slot, bank}` (ADR 0004 §5). What is missing
is the memory path honouring `l3_tile`, and the DMA engine that carries it.

**WRONG**, a test rig that wires the path the test wants:

```cpp
auto e = noc->attach_engine(0, dma, 0);          // always port 0, always hub 0
auto t = noc->attach_l3(0, l3, 0, e.engine);     // the program's l3_tile is ignored
```

**CORRECT**, the program says where its bytes go, and the deployment says how they get there:

```cpp
// csp program (kpu-csp-gen --algo vector-load, or hand written under tests/program/csp/trace/)
//   load  x[0:256]  -> l3[1].slot[2].bank[0]   via dma[3]
//   release l3[1].slot[2]
//
// plan_l3 keeps one L3Program per L3 tile it is placed on, and the engine per action:
struct CspL3Plan {
    std::map<std::uint32_t, L3Program> per_tile;     // l3_tile -> its actions, in the program's order
    std::map<std::size_t, std::uint32_t> engine_of;  // action ticket -> DMA engine (the program's, or the
                                                     // deployment's default for that tile)
    // ...
};
// run_memory_subsystem builds one L3Tile per placed tile, one DmaEngine per engine used, and
// attaches each at the hub and port that ArrayLayout gives it. No tile or engine is implied.
```

**How the DMA engine is chosen.** A Load names its engine in the program (`via dma[e]`) or takes the
deployment's default for its L3 tile. The cycle core never picks one. A program's choice the deployment cannot
honour (engine e has no route to tile k) is refused when the program is planned, before the run starts.

**The ladder** (debug-ladder-simplest-first), each rung a test and a golden trace:

| Rung | Program | What a person checks by hand |
|---|---|---|
| A1 | One 64-byte vector, direct link, L3 tile 0 | ACT, RD, the burst returning, one beat into slot s, bank b |
| A2 | One 64-byte vector over the periphery ring, to an edge node | the ring segments it crosses, the clock crossing at the edge port, the slot's credit coming back |
| A3 | The same vector on to an interior node, forwarded by the edge node's send mover | each one-hop push, each node's transit slot and its credit |
| A4 | One 256-byte vector (4 bursts) | the 4 bursts in flight together, their row hit, beats arriving in order |
| A5 | Load, then Store back to another DRAM address | the ejection, the store buffer, the write, DRAM equal to the source |
| A6 | Two vectors from two engines to two L3 tiles | two flows sharing the rings, and their arbitration |

Each rung has a **golden hand trace**, `tests/cycle/trace/<rung>.golden`. It is a short table, one row per
event, with the cycle, the block id and the event (`dram.ch0 ACT row=12`,
`noc.hub[1] in=west out=l3 flit=0`). It is written by a person, from the timing parameters and the
topology, *before* the run. The test compares the run's record to it event by event. A difference fails the
test and names the first diverging event. The golden is never regenerated from the run. When the model is
right and the golden is wrong, the golden is corrected by hand, and the commit says why.

### 3.2 Track B: the shared view of the path

**One record, keyed by block id.** `.mflow` grows to v5 with two tables. Their rows name blocks by their
`kpu-structure` id, so the floorplan viewer can place every event:

| Table | Columns | Written by |
|---|---|---|
| `hops` | cycle, flit, packet, from_block, to_block, ring, queue_depth | `NocHub`, `NocPort` through `MemoryEvents` |
| `l3_writes` | cycle, block (L3 tile), slot, bank, offset, bytes, ticket | `L3Tile` through `MemoryEvents` |

The record's `manifest.structure` names the structure record it was run on, and `mflow_check` checks every
block id against it (a new check, M12). The RTL reaches the same record through a converter,
`tools/trace/mflow_from_kpugen.py`: kpu-gen-trace v3's `packets`, `hops` and `transfers` become `.mflow` v5.
It follows the pattern of `structure_from_floorplan.py`. kpu-sim owns the format, and esg writes it natively
later (esg#27, esg#90).

**The views.**
- **The floorplan viewer** gains a path layer:
  - it reads a `.mflow` v5 next to its structure record;
  - it draws each flit's route over the blocks;
  - it steps cycle by cycle, with a cursor shared by every view.
- **The memflow viewer** keeps the DRAM and DMA detail.
- Both open from `kpu-debug-server`; a click on a block in one selects its rows in the other.

**WRONG**, a view that knows the machine:

```js
const hubs = [[0,0],[0,1]];             // drawn from what the viewer assumes a T4 is
```

**CORRECT**, a view that draws what the record declares:

```js
const blocks = structure.blocks;        // kpu-structure v1: what this run's hardware has
for (const h of mflow.hops) drawHop(blocks[h.from_block], blocks[h.to_block], h.cycle);
```

**Both machines move to one topology (decided 2026-11-27).** The KPU is a one-hop, push-only XY mesh of
L3+CT nodes, with a separate credit network and a periphery ring on the memory controllers' clock joining
the DMA engines to the edge ports (`docs/plans/distributed-csp-programs.md`). Today's kpu-sim ring NoC and
esg's request/response mesh both differ from it. The shared viewer draws any machine from its own structure
record, so track B does not wait for either to change; a hop-by-hop comparison of the two waits until both
implement the mesh.

### 3.3 Track R: the retirement

| Step | What | Done when |
|---|---|---|
| R0 | ADR 0006: the cycle core is L-CA; CTE is frozen (fixes only, no features). CLAUDE.md's "USE THESE" list points to `cycle/`, not `timing/`. | ADR merged |
| R1 | Move out of `timing/` what the cycle core includes: `slot_pool`, `tile_descriptor`, `noc_topology` (to `cycle/` or `program/platform/`). `l2_banks` is used by `timed_record` and moves with R3. | `grep -r 'sw/kpu/timing' include/sw/kpu/cycle` is empty |
| R2 | `--level cycle-accurate` runs the cycle core: `csp_run_level`'s L-CA branch (`csp_run.hpp:475-519`) calls `run_system` (or the memory or compute subsystem, by what the program has). `csp_level_supported` states the cycle core's coverage. A program outside it is **refused at L-CA, by name**, and never falls back to CTE. | `kpu-run --level all` on the corpus runs L-CA on the cycle core or refuses it |
| R3 | L-CA's outputs from the cycle core: the `.dflow` 6.2 timed record (built from the core's tracer spans, not CTE's protocol events), `--bubbles`, `--step` (one KPU clock per step), the sweep column, the L-T1 / L-CA calibration line | the `dflow_check_timed_*_lca`, occupancy and domain-flow viewer tests pass on the core's record |
| R4 | `kpu-memsim` on the cycle core's memory subsystem: its stream, strided and random scenarios become L3 programs | `test_memory_side_harness`'s cases pass on the core, then the harness is deleted |
| R5 | The operator-schedule and network runners (`timing/schedule/`, `timing/graph/`, the m1–m3 and `examples/schedule/` examples, the resnet regression) | open decision 2 (§7) |
| R6 | Delete `include/sw/kpu/timing/`, `tests/timing/`, `src/program/dram_bridge.cpp` (once nothing hosts `MemoryControllerProcess`), and CTE's docs references; retire or rewrite the `new-csp-process` and `validate-architecture` commands | the build has no `timing/` |

**What R2 does to the tests.** CTE's tests fall in three groups, and each is handled differently:
- **Its own components and executor** (`tests/timing/`, the `test_csp_lca_*` group): deleted with R6, never
  ported. They test the wrong model.
- **Tests asserting an L-CA timing or a result through the driver** (`test_csp_double_buffer`,
  `test_lca_rates`, `kpu_run_calibration_*`, `kpu_run_csp_bubbles_lca`, …):
  - when the cycle core covers the program, the test is re-baselined on the core, and the new number is
    explained in the commit;
  - when it doesn't, the test becomes a refusal test, and the gap gets an issue.
- **Tests asserting that L-CA refuses something** (`kpu_run_csp_refuses_lca_on_t4`, `kpu_run_l0_refuses_lca`,
  `test_noc_spec`): re-pointed at the cycle core's refusals.

**The cycle core's gaps after R2** each get an issue, in the order the corpus needs them:
- multiple L3 tiles in `--system` (after track A gives the memory side them);
- the NoC in `--system`;
- COMPUTE_ACCUMULATE;
- the accumulator init stream;
- tile-context stages;
- unfused epilogues;
- WS/IS;
- typed values beyond int8 → int32;
- multiple compute tiles.

Until each one lands, L-CA refuses that kind of program. That is the honest state: CTE's numbers for those
programs came from the wrong model.

## 4. Implementation Steps

One concern per PR, in order. "Relates to" the tracking issue on every PR, except the last of each track.

1. **R0** (docs): `docs/architecture/adr/0006-cycle-core-is-l-ca.md`; CLAUDE.md's implementation-reference
   section; this plan's status.
2. **A1 + the golden format**: `tests/cycle/trace/` with `golden.py` (it reads a golden and compares it with a
   `.mflow`, naming the first divergence); `tests/program/csp/trace/a1_vector.csp`; the A1 test in
   `tests/cycle/test_cycle_trace_ladder.cpp`.
3. **Multi-tile memory path**: `csp_l3_program.hpp` (`per_tile`, `engine_of`; drop the module-0 refusal);
   `memory_subsystem.hpp` (one L3Tile per placed tile, one DmaEngine per engine used, each attached where
   `ArrayLayout` puts it); the CSP `via dma[e]` clause in the parser and `kpu-csp-gen --algo vector-load`.
   Tests: A2 and A3.
4. **`.mflow` v5**: `memory_events.hpp` (hop and L3-write events); `noc_hub.hpp`, `noc_port.hpp` and
   `l3_tile.hpp` emit them; `mflow_recorder.hpp` and `memory_flow_record.hpp` write them; `mflow_check.py`
   gets M12 (every block id is in the structure record) and a self-test. Then A4–A6.
5. **The floorplan viewer's path layer**, and the cursor it shares with the memflow viewer through
   `common/`. `test_viewer.py` cases on the A3 record.
6. **`mflow_from_kpugen.py`** and a fixture converted from esg's `build/soc_gemm/run.trace.json`, checked by
   `mflow_check` and opened in the path layer.
7. **R1**: move the shared headers.
8. **R2**: the switch. `csp_run.hpp`, `execution_level.hpp` (`level_models`), `virtual_platform.hpp`,
   `kpu_run.cpp`. Re-baselined and re-pointed tests, as in §3.3. Issues for each gap.
9. **R3**, in two PRs: first the `.dflow` timed record from the core, then `--bubbles`, `--step`, the sweep
   and the calibration line.
10. **R4**: `kpu-memsim` on the core.
11. **R5**: per decision 2.
12. **R6**: the deletion, and the docs sweep (about 70 files reference CTE; the plans and session logs keep
    their history, and the live docs are corrected).

## 5. Verification

```bash
# Track A: each rung's run matches its hand-written golden, event for event
ctest -L "cycle;trace" --output-on-failure
./build/tools/kpu-run/kpu-run --program tests/program/csp/trace/a3_far_tile.csp \
    --deploy tests/program/deploy/kpu_t4.json --memory-subsystem --run-dir /tmp/a3
python3 tools/trace/mflow_check.py /tmp/a3/memory.mflow --summary      # exit 0, M12 included
python3 tests/cycle/trace/golden.py tests/cycle/trace/a3_far_tile.golden /tmp/a3/memory.mflow

# Track B: the views open the record, and the RTL's converts into the same format
python3 tools/visualization/floorplan/test_viewer.py
python3 tools/trace/mflow_from_kpugen.py <esg>/build/soc_gemm/run.trace.json -o /tmp/rtl.mflow
python3 tools/trace/mflow_check.py /tmp/rtl.mflow
python3 tools/debug-server/kpu_debug_server.py --root /tmp      # /view/floorplan shows a3's path

# Track R: after R2, L-CA is the cycle core; after R6, timing/ is gone
./build/tools/kpu-run/kpu-run --program /tmp/mm64.csp --deploy tests/program/deploy/kpu_s1.json --level all
grep -r "ConcurrentTimingExecutor" include src tools tests examples    # empty after R6
ctest --output-on-failure                                               # gcc and clang, full suite
```

Expected after R2: `--level all` on the int8 64³ prints an L-CA makespan equal to `--system`'s 5897 clocks
(at `l3 ≥ 16`). Each program the core does not cover is refused at L-CA by name.

## 6. Key Invariants

- **One L-CA.** After R2 no code path reaches CTE from `--level cycle-accurate`, and nothing falls back to it.
- **The program places, the machine obeys.** Every Load goes to the L3 tile, slot and bank its program names,
  through the engine it names or the deployment's default. The cycle core never chooses a tile or an engine.
- **Goldens are written by hand.** A golden is never regenerated from the run that it checks.
- **Every record row names a block that exists.** Every block id in a `.mflow` v5 is in the structure
  record the run names (M12), from kpu-sim and from the RTL alike.
- **Views draw what the record declares.** No viewer hard-codes a topology.
- **Refusal over a wrong number.** A program outside the cycle core's coverage is refused at L-CA, not timed
  by another model.
- **Values still answer to L-B.** Each rung's L3 slot (and, in A5, DRAM) equals L-B's bytes, bit for bit.

## 7. Decisions

1. **The NoC (decided 2026-11-27):** a one-hop, push-only XY mesh of L3+CT nodes; see
   `docs/plans/distributed-csp-programs.md`. Its rework (#618) follows that plan's coverage report (#603), and
   this plan's tracks follow the rework.
2. **The operator-schedule and network runners (R5, open):** recommendation, delete them with R6 and the
   resnet baseline with them; network-level timing comes back through CSP programs. Confirmed with the user
   before the deletion (#627).
3. **Milestones (as filed):** tracks A and B and R0–R2 in v0.10.0, R3–R6 in v0.11.0. #458, #495, #528 and #433
   are CTE fixes that R2 makes moot.
