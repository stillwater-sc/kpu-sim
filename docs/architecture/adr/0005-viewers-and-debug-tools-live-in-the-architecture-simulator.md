# ADR 0005: Viewers and debug tools live in the architecture simulator

| | |
|---|---|
| **Status** | **Accepted** (2026-11-16) |
| **Date** | 2026-11-16 |
| **Formalizes** | The decision of 2026-11-04 recorded in #392: "Viz and debug tools that the functional/transactional simulator and the RTL share live in kpu-sim." |
| **Repositories** | stillwater-sc/kpu-sim ("kpu-sim"); branes-ai/embodied-ai-soc-generator ("esg", the RTL generator `kpu-gen`) |
| **Context docs** | `docs/plans/viewers-and-debug-in-the-architecture-simulator.md` (the assessment and the plan) |

---

## 1. Context

kpu-sim is the architecture simulator. It decides the KPU's architecture and shows the machine
works, and esg generates the RTL that implements what kpu-sim has shown. To decide, debug and
optimize the machine, both need the same views: data movement, reuse, bubbles, resource
occupancy, structure, floorplan, NoC traffic, and the comparison of a model run with an RTL run.

By 2026-11-16 those views were starting to be built twice:
- esg planned its own web viewer with a server (esg#59 to esg#61), reading its own floorplan
  format, `floorplan.json` v1, beside kpu-sim's `kpu-floorplan`;
- esg re-ordered its occupancy export (esg#27) to draw its lanes in its own timeline first.

The record formats were already kpu-sim's (`.dflow` v6, `.tflow` v6, `kpu-tile-cmds`), with their
checkers, and esg vendors kpu-sim's `.dflow` writer (esg#48).

## 2. Decision

**esg produces records of the implementation in kpu-sim's formats. kpu-sim defines those
formats, checks them, and owns the views and the tools that read them.**

| kpu-sim owns | esg owns |
|---|---|
| The record formats, versioned, with golden samples: `.dflow`, `.tflow`, the structure and floorplan record (#392), the occupancy record (#390), `kpu-tile-cmds` | The RTL run recorder and its exporters into those formats (esg#48, #27, #29) |
| The checkers, which esg's CI pins and runs on every record it writes | Instrumentation inside the hardware, such as the first-error capture (esg#84). kpu-sim defines the error kinds the contract names |
| Every view: domain flow, tile flow, memory flow, floorplan (Overview, Investigate), bubbles, occupancy, spatial events, the comparison of two runs | Implementation reports: die area and its estimate (esg#64), the numbers a floorplan record carries |
| The run-publishing service the viewer is served from | The RTL's own documentation and its teaching animations (esg#34), and its CI and verification (esg#83, #85), with kpu-sim as the reference |

## 3. Consequences

- **One rendering of the machine.** A model run and an RTL run of the same program open in the
  same viewer, side by side, and a divergence is found there (#402).
- **The formats come first.** A view can be shared only if the record is. So the structure and
  floorplan record (#392) is the first of the shared formats to build. It absorbs esg's
  `floorplan.json` v1 and `kpu-floorplan`'s output.
- **The order of a change** is model, then format version and golden sample, then checker, then
  a `kpu-sim-sync` issue in esg naming the version to implement.
- **esg works in kpu-sim for a viewer change.** The viewer is packaged to stand alone, static
  files plus a Python package, so esg uses it without building kpu-sim's C++.
- **Physical detail is the RTL's to produce.** Footprints in mm and routes as wired are carried
  by the record when a producer has them. The model's producer leaves them out, or estimates them
  and flags the estimate. The viewer draws what the record holds.
- **Issues moved** (2026-11-16): esg#59, esg#60 and esg#61 to kpu-sim. A transfer across GitHub
  organizations is not possible, so each was re-filed here, with a link back, and the original
  closed with a pointer.

## 4. Alternatives considered

- **Viewers in esg, reading kpu-sim's records.** This keeps each tool near its main user, the RTL
  developer. Rejected: the model's runs, which decide the architecture, would then be viewed
  in the implementation's repository, and the comparison of the two needs the oracle and the
  formats in one place.
- **A separate viewer repository.** Rejected for now: the formats, the checkers and the model's
  producers change together. A third repository would add a sync step to every format change.
  The viewer is packaged so that it can move out later, if it outgrows kpu-sim.
