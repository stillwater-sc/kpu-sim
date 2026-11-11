# Roadmap to v1.0.0

**Status:** accepted and applied to the GitHub milestones, 2026-11-11
**Current release:** `v0.8.0` (tagged). `main` is about 300 commits past it.
**Replaces:** the v0.9.0, v1.0.0 and post-1.0 sections of `docs/ROADMAP.md`. It also replaces the
GitHub milestones `v0.8`, `v0.9`, `v1.0`, `csp-patterns-wave-0..5`, and `M1`-`M3`.

## 1. What v1.0.0 is

v1.0.0 simulates a **KPU built from systolic-array tiles**, programmed in the **CSP language**.
A CSP program orchestrates the KPU's resources on behalf of a Domain Flow Program:
- DMA engines, BlockMovers and Streamers;
- the L3, L2 and L1 buffers;
- the compute fabric;
- the NoC between tiles.

v1.0.0 delivers:

1. **Four working levels** that run the same program and agree on its values:
   - L-B, behavioral;
   - L-T1, block-sequential;
   - L-T2, resource-transactional;
   - L-CA, cycle-accurate.
2. **A host-side orchestration path.** A program on the host CPU calls Domain Flow Program kernels.
   Each kernel sets up the resources and runs at the level the caller chooses.
3. **Visualization** of schedules, resource occupancy and performance at every level that has
   time.

Values answer to L-B, and timing answers to L-CA (ADR 0002). The KPU RTL cross-check is how L-CA's
timing is validated. It is part of the path, not a separate product.

Outside v1.0.0, see §4:
- the CSR, RISC-V and Renode SoC interface;
- DNN model ingestion;
- the CSP movement-pattern library beyond matmul and conv2d;
- the dense linear-algebra kernels beyond the matmul/LU core.

## 2. The release train

Each minor release is one GitHub milestone, tagged when the milestone closes. Each release ships
the views of what it adds.

| Release | Theme | Exit criterion |
|---|---|---|
| **v0.9.0** | One tile, one program, three levels, first RTL cross-check | A 4×4 int8 output-stationary matmul runs at L-B, L-T1 and L-CA from one CSP program, and matches the RTL by `dflow_check --result` |
| **v0.10.0** | The cycle-accurate systolic tile | L-CA models the banked L2 and accumulator, the L1 stream buffers, the Streamer and the wavefront, for OS, WS and IS and for conv2d, and matches the RTL's tile tests |
| **v0.11.0** | L-T2, resource-transactional | L-T2 runs every program L-T1 runs, with values bit-equal to L-B and per-resource read/write/push events in the spatial viewer |
| **v0.12.0** | The multi-tile KPU | Programs distribute and broadcast across several tiles over the NoC at every level, and the NoC and occupancy appear in the shared record |
| **v0.13.0** | Host orchestration of Domain Flow Program kernels | A host program sequences several kernels through the platform at a chosen level, and the simulator returns the values it computed |
| **v1.0.0** | Stable release | The engines ADR 0001 superseded are retired. The program format is versioned with a golden corpus. A performance suite (the batched-MLP sweep) is reported and visualized at every level. The docs are complete |

## 3. Issues by release

### v0.9.0: one tile, one program, three levels, first RTL cross-check

- #394 int8 → int32 value path through CSP, L0, L-B, L-CA and the record
- #396 map the RTL generator's KPUArchitecture spec onto the DeploymentSpec
- #397 one program on both sides: lower a CSP program to the RTL's tile command stream
- #398 shared record v6.1: accumulator → L3 stores, a flat word-addressed L3
- #399 L3 placement step 6: L-CA uses the walker's L3 and L2 slots
- #285 `kpu-run`: finish its definition of done for L-B, L-T1 and L-CA (L-T2 joins in v0.11.0)

### v0.10.0: the cycle-accurate systolic tile

- #401 cycle-accurate tile model: banked L2 and accumulator, L1 stream buffers, Streamer, wavefront
- #395 dataflow selection (OS, WS, IS) in the deployment spec and at every level
- #400 conv2d (NHWC, implicit GEMM) at the CSP level, L-B and L-CA
- #406 the accumulator init stream (steps 1-2 are done; steps 3-4 depend on #395, #400 and #394)
- #408 nearest-neighbour weight loading and bubble-free weight change
- #382, #383, #384, #385, #387, #388, #389: the RTL tile tests mirrored:
  - BlockMover moves;
  - L2 and accumulator banks;
  - end to end;
  - wavefront timing;
  - credits and tokens;
  - L2 bandwidth;
  - L1 stream buffers.
- Views: #390 resource-occupancy viewer; #402 L-T1 and L-CA write `.dflow` v6 events, with a tool that joins two records

### v0.11.0: L-T2, resource-transactional

- #283 the L-T2 interpreter over the fixed §3.3 vocabulary
- #281 derive explicit CSP processes and channels from the Domain Flow Program
- #284 the backdoor as its own harness-only interface (protocol tests pose states directly)
- Views: #286 the spatial event record and viewer (concurrency and causality across the pipeline)

### v0.12.0: the multi-tile KPU

- #403 multi-tile execution: distribute and broadcast, several L3 tiles at L-CA, the NoC modelled
- #43 wire L3Interconnect and NoC routing into the base simulation
- #44 an L2 interconnect/topology (only if #403 needs it)
- Views: #405 NoC packets, hops and occupancy in the shared record; #392 the structure record; #391 the tile-flow view with memory layers as containers

### v0.13.0: host orchestration of Domain Flow Program kernels

- #305 encapsulate the program and separate the data (`.kpuld`), orchestrated from the host. The
  RISC-V and Renode part moves to the SoC track in §4.
- #257 the host API returns simulator-computed values, or says it doesn't
- #231 hardware-identical execution: ResourceManager setup and host memory

### v1.0.0: stable release

- #266 freeze and retire the superseded execution engines (ADR 0001 D6)
- #255 program-format versioning, a golden corpus, a serialize/load/execute round trip
- #144 the ProgramSerializer round trip (with #255, or retired with the engine it serves)
- #22 test fixtures on `tests/common/test_configs.hpp`
- #42 symmetric monitoring accessors on L2Layer and L3Layer
- #45, #48, #49, #50 the batched-MLP sweep: latency, throughput and efficiency, reported and
  visualized at every level (the v1.0.0 performance acceptance suite)

## 4. After v1.0.0 (milestone "Backlog (post-1.0)")

| Track | Issues |
|---|---|
| SoC interface | #404 the RTL's CSR command interface; the RISC-V and Renode part of #305 |
| CSP movement-pattern library | #88 and the wave epics #70, #73-#75, #78-#87, with their tasks #94-#98 and #110-#118 (the `wave-N` labels stay) |
| DNN models on the CSP executor | #132-#136, #262 |
| Real-DNN ingestion | #229 and its phases #230, #232-#235 |
| Dense linear algebra | #242, #243, #244, #258, #259, #260, #261 |
| L-T2 refinements | #268 statistical variance after calibration |
| Legacy program path | #256 the epilogue ops carried in a `.kpubin` program (retired with #266 unless still needed) |

#75 (the Conv2D-family patterns) carries `rtl-sync`. Its conv2d core is #400, in v0.10.0, and the
rest of the family stays in the backlog.

## 5. Closed when this plan was applied

| Issue | Why |
|---|---|
| #264 TileTransactionExecutor | Superseded. ADR 0002's L-T1 (`csp/transactional.hpp`) is the block-sequential tier. The executor was never built. |
| #253 Decision record: program format and transactional engine | Done. ADR 0001 is accepted and committed. Its remaining retirement list is #266. |

## 6. Conventions

- **Milestones are releases:** `vX.Y.0`, and nothing else. A theme or track is a label:
  - `rtl-sync`;
  - `wave-N`;
  - `csp-patterns`;
  - `dnn-milestone`;
  - the four `area:` labels: `area: rtl-reference`, `area: shared-views`, `area: soc-scale` and
    `area: hygiene` (formerly `P0`-`P3`, renamed 2026-11-11 because they read as priorities).
- **A release is tagged when its milestone closes:**
  1. A PR turns the CHANGELOG's `[Unreleased]` section into `[X.Y.0] - <date>`.
  2. Once it merges, run `scripts/release.sh X.Y.0 --push` on `main`. The script makes an
     annotated tag `vX.Y.0`, and `cmake/Version.cmake` and `setuptools-scm` read the version from it.
  3. Create the GitHub release from the tag.
- **Session logs** are named for the release in progress. Today that is `v0.9`.

## 7. Old milestone names

Closed on 2026-11-11. Issue texts in this repo and in branes-ai/embodied-ai-soc-generator may still
use the old names:

| Old milestone | Now |
|---|---|
| `v0.8` | closed; its open issues went to v0.12.0 (#43) and v1.0.0 (#22, #42) |
| `v0.9` | renamed `v0.9.0`, rescoped |
| `v1.0` | renamed `v1.0.0`, rescoped |
| `M1: first RTL cross-check` | v0.9.0 |
| `M2: tile timing parity` | v0.10.0 |
| `M3: 2x2 SoC reference` | v0.12.0 (multi-tile, NoC); the CSR command interface (#404) is in the backlog |
| `csp-patterns-wave-0..5` | Backlog (post-1.0); the `wave-N` labels keep the waves |
