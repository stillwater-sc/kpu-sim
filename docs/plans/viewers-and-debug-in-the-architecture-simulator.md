# Viewers and Debug Tools Belong in the Architecture Simulator

**Status:** decided 2026-11-16: the boundary is ADR 0005, the issues moved (§4), and the
structure and floorplan record (#392) comes first.
**Repositories:** stillwater-sc/kpu-sim ("kpu-sim") and branes-ai/embodied-ai-soc-generator
("esg", the RTL generator, `kpu-gen`).

## 1. The assertion

> kpu-sim is the architecture simulator. The viewers and debug facilities are needed to debug and
> optimize the machine it models, so that the SoC generator can drive the RTL implementation of
> that machine. They should therefore be developed in kpu-sim, not in the RTL repository.

## 2. Verdict

**Correct, with one boundary.**

- Every view of **what the machine does** belongs in kpu-sim. Data movement, reuse, bubbles,
  resource occupancy, structure, NoC traffic and floorplan are views of the machine. So is the
  comparison of two runs. The views, the record formats they read and the checkers that validate
  those records are kpu-sim's, whichever side produced the run.
- The **producers of implementation truth** stay in esg:
  - the RTL run recorder and its exporters into kpu-sim's formats;
  - the instrumentation inside the hardware (the first-error capture of esg#84);
  - the implementation reports (die area, PPA).

  These describe a particular implementation, and only the RTL can produce them.

The rule, in one line: **esg produces records of the implementation in kpu-sim's formats;
kpu-sim defines those formats and checks them, and owns the views and the tools that read them.**

## 3. Why

1. **kpu-sim leads.** The architecture is decided and proven in kpu-sim, and esg implements what
   kpu-sim has shown to work (recorded 2026-11-15). The tools for deciding, proving and
   optimizing the architecture therefore live with it. Bubbles, reuse and occupancy are
   architecture questions. They are asked first of the model, at L-B, L-T1 and L-CA, at a
   fraction of the RTL's simulation time, and only then of the RTL.
2. **The formats are already kpu-sim's.**
   - `.dflow` v6 (#386), `.tflow` v6 and `kpu-tile-cmds` (#397) are defined here, with their
     checkers (`tools/trace/dflow_check.py`, `tflow_check.py`, `tile_cmds_run.py`) and golden
     samples.
   - esg already **vendors** kpu-sim's writer (esg#48: `dflow_writer.py`).
   - Its exporters are already titled "for the kpu-sim viewer" (esg#20, esg#27), or "the
     structure record of kpu-sim#392" (esg#29).
3. **The viewers already exist here.**
   - `tools/visualization/domainflow/`, `tileflow/` and `memflow/` read those records today.
   - `tools/floorplan/` (`kpu-floorplan`) draws a deployment's floorplan.
   - `kpu-run --bubbles` and `--step` are the debug views of a run (#425).
4. **Duplication has started, and each copy will drift.**
   - esg#59 to esg#61 plan a second web viewer with its own server. Its floorplan is
     `floorplan.json` v1, a second floorplan format beside `kpu-floorplan`'s.
   - esg#27 was re-ordered to draw its occupancy lanes in esg's own timeline first
     (`kpu-gen trace --svg`), before the kpu-sim export.

   Two viewers mean two renderings of one machine, which disagree in ways nobody notices until a
   bug hides between them.
5. **The viewer you need most is a comparison.** Debugging the RTL means putting its run beside
   the model's run of the same program and finding where they part (#402: two records joined
   event by event). Only one place holds both the oracle and the formats: kpu-sim. A viewer in esg
   would have to import the model's runs anyway.

## 4. The inventory: esg's pending viewer and debug issues

| esg issue | What it is | Proposed home |
|---|---|---|
| #59 viewer: `kpugen view`, a web floorplan viewer, Overview | A viewer | **Move to kpu-sim**, as a view of the shared structure and floorplan record (#392). esg keeps "write `floorplan.json` in the record's format" |
| #61 viewer: Investigate view, connectivity lenses, drill-down | A viewer | **Move to kpu-sim**, with #59 |
| #60 publish runs from headless servers to a central kpuview service | Viewer infrastructure | **Move to kpu-sim**, so kpu-sim's runs publish too. esg keeps a small publish client (or uses the viewer's packaged one) |
| #34 wavefront animations from the RTL | The RTL's teaching documentation | **Stay.** First proposed as a split. Read in full, it is the RTL's own documentation: SVGs generated from the RTL, committed in its docs, with CI checking the RTL against the predicted wavefront. It already points to kpu-sim#390 and #391 for the interactive per-clock views |
| #29 block-diagram generator from the elaborated RTL | A producer of #392 | **Stay**, as the RTL's producer of the structure record. Drawing it is kpu-sim's viewer's job |
| #27 export the resource-occupancy record | A producer of #390 | **Stay**, as the export, in #390's format. Its own timeline is redundant once the shared view exists |
| #20 tile-flow recorder; #48 `.dflow` v6 from the run recorder | Producers | **Stay** (already folded into #48) |
| #84 debug: first-error capture behind the sticky error bits | Hardware instrumentation | **Stay**: it is RTL. kpu-sim defines the error kinds it reports, as part of the tile contract, and the `.dflow` record carries an `errors` table, so the shared views show it |
| #85 perf summary and CI-gated reference runs | The RTL's CI | **Stay**, gated on kpu-sim's checkers (`dflow_check --result`) |
| #83 constrained-random differential harness | The RTL's verification | **Stay**. kpu-sim is its reference model |
| #64 `kpugen report`: die estimate and area | Implementation report | **Stay**. The numbers feed the floorplan record that kpu-sim's viewer draws |

On kpu-sim's side, the viewer work already filed is #425 (tracking), #402, #390, #391, #392, #405
and #286. The issues moved in would join #392 (floorplan, structure) and #425's viewer pane.

## 5. The plan

### Phase 0: decide the boundary

- Record §2's rule as an ADR in kpu-sim (`docs/architecture/adr/0005-...`), with a one-paragraph
  pointer in esg's `docs/`.
- Move the issues in §4: esg#59, #60 and #61. A transfer across GitHub organizations is not
  possible, so each is re-filed in kpu-sim with a link back, and the original closed with a
  pointer. esg#34 stays.

### Phase 1: the formats first

A viewer can only be shared if the records are. In order of need:
1. **The structure and floorplan record** (#392). It absorbs esg's `floorplan.json` v1 and
   `kpu-floorplan`'s JSON into one versioned schema: blocks, footprints in mm, NoC routers and
   links with their routing rule, the status/control tree, and the area breakdown. It gets a
   checker and a golden sample from each producer.
2. **The occupancy record** (#390, esg#27): L2 bank groups, Streamer, L1 streams and Combiner
   lanes, per clock.
3. **Errors** in `.dflow` v6.x: the first-error table from esg#84, and the model's refusals,
   which L-B already reports.

Each format is versioned. Its checker is pinned by esg's CI, which validates every record esg
writes.

### Phase 2: one viewer

- One web app in kpu-sim (`tools/visualization/`, called `kpuview` below), with a view switch.
  The existing views become views of the one shell:
  - domain flow, tile flow, memory flow (existing);
  - Overview and Investigate (from esg#59 and #61), reading the structure and floorplan record
    from either producer;
  - bubbles (#425), which is `kpu-run --bubbles` as a view;
  - occupancy (#390).
- **Packaged to stand alone:** static files, plus a pip-installable Python package for its
  server. esg uses it without building kpu-sim's C++.

### Phase 3: the comparison loop

- **Two records side by side,** joined event by event (#402): the RTL against L-CA on the same
  program. The view opens at the first divergence.
- **The spatial event view** (#286) and the tile-flow view of RTL and model runs (#391).

### Phase 4: publishing

- The run store and the server-sent-events service of esg#60, served by `kpuview`. Both producers
  push: `kpu-run --publish` and `kpu-gen --publish`. Runs from headless servers open on a laptop,
  RTL and model side by side.

## 6. The development workflow

1. **Contract first.** A change to the machine starts in kpu-sim:
   - the model, then its format's version and golden sample, then its checker;
   - then a `kpu-sim-sync` issue in esg naming the version to implement.

   esg's CI validates its records against the pinned checker. This is the order #397 and #386
   already followed.
2. **Conformance in CI.** esg#51 (the 4x4 int8 matmul on both sides) gates on kpu-sim's
   `dflow_check --result`: cycles to result and values. A failing gate is a divergence to open in
   the comparison view.
3. **One source for every shared description:**
   - the machine description (#396 and esg#50: `KPUArchitecture` to `DeploymentSpec`);
   - the register map (esg#52);
   - the command stream (#397 and esg#49);
   - the structure record (#392).

   A description written in two places is two descriptions.
4. **Viewer tests need no RTL.** kpu-sim's CI tests every view on checked-in sample records from
   both producers. An RTL sample is refreshed by esg's exporter when its format version changes.
5. **The same full cycle in both repositories:** `/full-cycle`, `pr-watch.sh`, CodeRabbit with
   draft auto-review. The tooling is per user and repository-agnostic.

## 7. Costs and risks

- **kpu-sim's CI grows** by the viewer's tests (Python and JavaScript, headless). They are cheap
  beside the C++ suite. They gate the viewer's paths only.
- **Contributors from the RTL side work in kpu-sim** for viewer changes. The ADR and a
  `CONTRIBUTING` note on the viewer's package keep that a small step.
- **Physical detail** (die mm, routes as wired, macro footprints) is the RTL's to produce. The
  record carries it generically, and the model's producer leaves it out or estimates it, flagged
  as such. The viewer draws what the record holds.
- **A service in a simulator repository.** It is a separate package (`tools/visualization/kpuview`)
  with its own tests, and is not built with the simulator.

## 8. Decisions

Decided on 2026-11-16:
1. The boundary rule (§2): ADR 0005.
2. The issue moves (§4): esg#59, #60 and #61 are re-filed in kpu-sim. esg#34 stays.
3. Phase 1 starts with the structure and floorplan record (#392).

Open:
4. The viewer's name and package (`kpuview`?), and the publishing service's milestone.
