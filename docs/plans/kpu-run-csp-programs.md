# `kpu-run` runs CSP programs, at every level including cycle-accurate

**Status:** Decided 2026-10-10 (revised after review, §8; Q1-Q4 as recommended, §9); steps 1-3 and 4a-4d done;
4e and 5 planned (§5, step 4; 4d's language decided 2026-10-10, 4d.2's orchestrator 2026-10-11)
**Tracks:** #283 (the L-CA half). Covers `docs/plans/csp-program-tile-sequencing.md` steps 2b and
3 (L-T1 from the program).
**Depends on:** the CSP language and its stream (ADR 0004, #343-#345), `CspDriver` (#342, #345,
#347), the virtual platform (#282)

## 1. Why

ADR 0004 §4 decided that **the CSP program is the portable program and L0 is a trace format.**
`kpu-run`, the simulator's driver, has not followed. It derives or loads an L0 program and runs
it at L-B and L-T1, and it never reaches L-CA:
- `CspDriver` runs CSP programs at L-CA, but only its own tests use it.
- L-T1 (`TileTransactionExecutor`) executes L0 ops as they become ready and *infers* residency:
  a tile takes an L3 credit at first touch and returns it after its last reader. That is not
  the program's residency. It is an execution order's accident, which is the failure ADR 0004
  was written against.
- Every program `kpu-run` runs today is derived inside the simulator (`--algo`) or is an
  unrolled trace (`--program file.l0`).

This plan makes `kpu-run` what it should be: **a simulator that takes a CSP program, which
configures its data path, and executes the operator at each level.** Producing programs is a
separate tool's job.

## 2. Roles

| Artifact or tool | Role | Not its role |
|---|---|---|
| `.csp` program | The input. Configures the data path: what is resident, the order tiles meet the fabric, what is fused on a move. Every level executes it. | — |
| `csp-gen` (new executable) | Writes `.csp` programs: the canonical schedules for matmul, the linear operator and LU, at a size, tiling and L3. Its output is ordinary program text, which can be printed, edited and fed back in. | Simulating anything |
| `kpu-run` | Validates the program and executes it at L-B, L-T1 and L-CA on a deployment. Compares every operand's values; reports each level's timing. | Generating or deriving programs |
| L0 | The value oracle (ADR 0001 D5): a program small enough to trace carries its L0 ops, and `TileProgramReference` on them gives the reference values. Also the golden corpus. | Driving execution |

## 3. What exists

| Piece | State |
|---|---|
| `lang::validate`, `lang::compile`, `lang::ActionStream` | Symbolic validation; the trace (with its L0 ops); the stream with no unrolling. |
| L-B: `csp::BehavioralInterpreter` | `begin` / `step` / `finish`; runs from the stream or the trace. |
| L-CA: `timing::CspDriver` | Trace or stream, windowed. Runs matmul and the linear operator, with tile contexts on timed vector units. Refuses LU and multi-L3 machines. |
| L-T1: `TileTransactionExecutor` | Runs L0 only; infers residency (above). |
| `kpu-run` | L0 in, `--algo` derivation inside it. `run_at(level, TileProgram, DeviceDescriptor, ...)`; L-B and L-T1. |
| `VirtualPlatform` | Holds TilePrograms; digests and run identity over them. |
| `lang::emit` | An L0 → `.csp` bridge (Belady residency at call granularity). |

## 4. Design

### 4.1 `csp-gen`

`csp-gen --algo matmul|linear|lu --size N --tile T --l3 C [schedule options] -o out.csp`

- **A library plus a thin executable.** The generators live in
  `include/sw/kpu/program/csp/gen/` as functions that build a `lang::Program` AST, printed by
  the language printer. The tool is `tools/csp-gen/`. Tests use the library directly, and
  `kpu-run` does not link it.
- **Schedules are written as structured programs with their loops,** never by unrolling and
  re-rolling:
  - **matmul:** output-stationary panel schedule. B's column panel resident per j, A's row
    panel per i, an accumulator per C tile. Schedule options choose the panel orientation and
    how much stays resident within `--l3`.
  - **linear:** matmul plus the epilogue. `--act relu|gelu|silu|atan` and
    `--place fabric|str.drain|bm.egress|unfused` write the `via` list, or the unfused calls.
  - **LU:** right-looking tile LU (getrf, laswp, trsm, gemm), with explicit residency per step.
- **The generator validates what it writes.** It runs `lang::validate` before emitting, and
  `--target spec.json` also checks placement against a machine. A schedule that does not fit
  `--l3` is refused with the validator's line-numbered reason, so the tool cannot emit an
  invalid program.
- **Schedule options are the generator's,** never deployment-spec fields: the schedule-vs-machine
  rule. The machine's L3 is an input (`--l3`, or read from `--target`). The panel and residency
  choices are options.

### 4.2 `kpu-run` takes a CSP program

`kpu-run --program op.csp --deploy machine.json [--level ...] [--window W] [--inputs ...]`

1. **Parse and validate** (`lang::validate`, against the deployment's target sites). A refusal
   is the validator's message with its line.
2. **Inputs.** A `.csp` program declares shapes, not values.
   - The default synthesizes deterministic, exactly representable values from the operand names
     (the existing `fill_inputs`).
   - `--inputs values.l0` supplies them from an L0 file's `VALUES` (the corpus's container for
     values).
3. **Reference.** If the trace is small enough (`--trace-limit`, default 1M actions), `compile`
   produces it. `TileProgramReference` over its L0 ops on the same inputs gives the reference
   values. Every level is compared against it, and against L-B, bit-exactly and on every
   operand, as today. A program too large to trace runs without the oracle, and the report
   says so.
4. **Levels**, each from the program:
   - **L-B:** `BehavioralInterpreter` from the stream.
   - **L-T1:** the program's transactions, under the program's residency (§4.3).
   - **L-CA:** `CspDriver` from the stream, on the machine `csp_config_from(device)` builds.
     `--window` is the driver's window, a schedule option (default 256).
5. **Report**, per level:
   - makespan;
   - DRAM loads, stores and bytes;
   - peak L3 (which must equal the validator's `peak_l3`);
   - compute busy;
   - vector-unit busy and bound;
   - the spec fields the level does not model.

   L-T1 and L-CA are printed side by side: the fast model against the authority. No assertion
   is made between them; a gap is a calibration finding.
6. **`--program file.l0` stays** for the trace format. It runs at L-B and at L-T1's legacy
   path, so existing fixtures keep working. L-CA is refused for it, with the reason: "an L0
   file is a trace; L-CA runs the program (generate one with csp-gen)".
7. **`--algo`, `--size`, `--tile`, `--act` leave `kpu-run`** (§9 Q1 on when). They are
   `csp-gen`'s.

### 4.3 L-T1 from the program (sequencing plan step 3)

L-T1 decomposes a CSP transaction into one tile move under L3 credits. **The credits are the
program's**:
- a Load acquires its residency's slot and a Release frees it;
- a Writeback into an open residency writes in place.

Nothing is inferred. The executor becomes an interpreter of the action stream, with the same
event engine and lumped per-hop costs as today:
- Load and Store cost the DMA hop; Move and Writeback the BlockMover hop; Feed and Drain the
  streamer hop; a Call costs its tile function's work.
- An action fires when the program's order allows it and its resource is free. Each process
  (dma, bm, str, cf) has its own queue, in program order, as at L-CA.

Values come from the same L0 kernels and tile-context element functions, so L-T1 is
bit-identical to L-B by construction.

The existing L0 path stays as `TileTransactionExecutor`'s legacy entry until the corpus and the
platform have moved (§5 step 4). Then it is retired.

### 4.4 The platform and the level contract

- **Programs:** `VirtualPlatform` holds CSP programs. A handle's program digest is the digest of
  its canonical printed text, so two spellings of one program are one program.
- **The spec:** `run_at` becomes `run_at(level, CspProgram source, inputs, DeviceSpecification,
  ...)`. Each level takes what it needs from the spec: L-T1 its descriptor, L-CA its executor
  config.
- **`level_supported(level, program, spec) -> optional<reason>`**, beside
  `level_implemented(level)`. L-CA refuses, by reason:
  - no deployment;
  - more than one L3 tile (level 2's);
  - LU, until its kernels run as functional computes at L-CA.

  `--level all` and the corpus print what was skipped and why.

### 4.5 What this plan does not do

- L-CA's `--step`, `--timeline` and `--tflow`: the L-CA record is the sequencing plan's `.sflow`
  (step 5).
- Level 2 (distribution).
- L-T2.
- LU at L-CA.

## 5. Steps

1. **`csp-gen`** (done).
   - **Library:** `include/sw/kpu/program/csp/gen/generate.hpp`, `gen::generate(Options, Target*)`.
     It writes program text with its loops, refuses a schedule that does not fit the L3 by its
     slot count, and returns only programs `lang::validate` accepts.
   - **Tool:** `tools/csp-gen` (binary `kpu-csp-gen`), with `--algo/--size/--tile`, `--l3` or
     `--target spec.json`, `--orient`, `--act`, `--place` and `-o`. Exit codes: 0 written,
     1 refused, 2 usage.
   - **Schedules:**
     - matmul: column- or row-panel, 2nt + 1 slots;
     - linear: + b[j], 2nt + 2 slots, with the epilogue `via` @ fabric, str.drain or bm.egress,
       or unfused as a second pass;
     - LU: right-looking, A resident whole, nt^2 slots, in the derivation's op order.
   - **Found on the way: the trace's L0 left out a fused epilogue.** A tile context lives on the
     moves, so its arithmetic was not an L0 op, and the trace's L0 computed `A . B` without the
     bias and activation. That L0 is the oracle step 2 depends on. The walker now records each
     stage as an L0 op (BiasAdd, Activation) where it applies, so a compiled program's L0 is a
     complete record of what it computes, fused or not.
   - **Tests:**
     - `test_csp_gen`, 6 cases. Matmul (both orientations, divisible and trailing tiles), linear
       (4 placements x 4 activations x 2 orientations) and LU (3 sizes), each at its smallest
       accepted L3:
       - the program validates, with peak L3 equal to the claimed slot count;
       - it compiles, and runs at L-B from the trace and from the stream;
       - it is bit-identical to the L0 derivation's reference;
       - its own trace's L0 reference agrees with that derivation.
     - Refusals: one slot short, placement against a target, bad options.
     - 5 CLI tests: written, from a target, refused for L3, refused for placement, bad option.
2. **`kpu-run --program file.csp` at L-B and L-CA** (done).
   - **`program/driver/csp_run.hpp`:**
     - `run_csp(CspRunRequest)` validates the program against the device's sites, builds the
       oracle, and runs each level: L-B from the stream, and L-CA through `CspDriver` on
       `csp_config_from(device)` with the window.
     - `csp_level_supported(level, program, device)` gives the reason a level cannot run, by
       name: no deployment, a multi-L3 machine, LU at L-CA, alpha != 1, a matrix not named A,
       B or C, or a program written for more L3 than the machine has. L-T1 is skipped until step
       3.
     - `csp_inputs(ast)` builds the inputs.
   - **`kpu-run`:** a `.csp` program takes `--deploy`, `--level`, `--window`, `--inputs
     values.l0`, `--trace-limit` and `--no-compare`. Any other option is refused by name; `--algo`
     and the size options point to `kpu-csp-gen`. `--level all` lists every level, with the
     reason for each one skipped. The report gives L-CA's makespan, DRAM loads, stores and
     bytes, compute busy, vector-unit busy and bound, the window, and the unmodelled spec
     fields. Values are compared bit-exactly, on every operand, against the oracle (or against
     L-B above the trace limit). An L0 file is refused at L-CA by name.
   - **`CspDriver`** reads every operand back from DRAM, not only C, so a level that writes an
     input is caught.
   - **Found on the way: the oracle and `inout` results.** An `inout` operand only needs input
     values if the program reads it before writing it. An accumulator starts from zero in the
     fabric, but L0's MatMulAccum adds onto its operand's buffer. So the unfused linear
     program's C, stored from an accumulator and then read back, made the oracle add `A . B` to
     synthesized values. L-B and L-CA agreed with each other, and both disagreed with the oracle.
     `csp_inputs` now decides by first touch: it streams until each `inout` operand is loaded (an
     input) or written (it starts at zero).
   - **Tests:**
     - `test_csp_run`, 6 cases:
       - matmul in both orientations and linear in all four placements, at L-B and L-CA:
         bit-identical to the oracle on every operand, L-CA's DRAM loads and stores equal the
         validator's, and L-B's peak L3 equals the validator's;
       - first-touch inputs;
       - LU at L-B, refused at L-CA;
       - every refusal reason;
       - the trace limit;
       - a misplaced stage.
     - 8 CLI tests:
       - matmul, unfused linear (window 16), `--inputs` from the corpus;
       - LU skipped at L-CA;
       - T4 refused, `--algo` refused, an invalid program refused, an L0 file refused at L-CA.
   - **Measured** (S1, matmul 256³ in 32-tiles, the generator's column-panel schedule):
     - 117,887 cycles;
     - 576 loads, 512 for the A panels;
     - against the step-2 Belady program's 128 loads and 43,968 cycles.

     The canonical panel schedule reloads A's row panel for every j: correct, and wasteful.
     Choosing a better schedule is the generator's job, and the comparison is now one command.
3. **L-T1 from the program** (§4.3) (done).
   - **`csp/transactional.hpp`, `TransactionalInterpreter`** (`begin`, `step`, `finish`, like the
     behavioral interpreter). It is a one-pass list schedule in program order. Every dependency
     of a CSP action points backwards, so each action's start is known when it is reached:
     - the data it reads is ready, from a scoreboard per tile and channel: the L3 copy, the
       moved and drained copies in L2, the fed operands, the accumulator chain, and RAW
       through DRAM;
     - a lane of its process is free (`DeviceDescriptor`: lanes and bytes per cycle; compute
       tiles and MACs per cycle);
     - the process issues in program order;
     - for a Load, or a Writeback that opens a residency, an L3 credit is free. The credit pool
       is the program's L3, and only the program's Releases return credits.

     A Store is the push-only store's two legs, a BlockMover ejection and then a DMA write. A
     Release waits for every reader of the residency, including stages that read a bias. An
     in-place Writeback waits for the earlier reads of its slot. Values come from
     `BehavioralInterpreter`, stepped in the same order, so they are bit-identical to L-B by
     construction.
   - **`run_csp`** runs it as L-T1, with the deployment's descriptor (or the default device).
     `kpu-run` reports:
     - the makespan;
     - DRAM traffic;
     - the L3 peak in time against the program's L3;
     - credit stalls;
     - lane-cycles per process;
     - UNCALIBRATED, and the tile contexts' vector time as unmodelled.
   - **Corrected from §6:** "peak L3 equals the validator's" holds for L-B, which executes in
     program order, but not for L-T1 or L-CA. In time, the DMA runs ahead as soon as a credit is
     free. On S1, matmul 256³ (program L3 128, live set 17) holds 117 slots at once. The
     invariant is peak <= the program's L3, and >= its live set.
   - **The L0 path's L-T1** (`TileTransactionExecutor`) is unchanged, and still runs L0 files.
     Its pinned makespans did not move, so nothing was re-baselined. It retires in step 4.
   - **Measured** (S1, matmul 256³, csp-gen's column-panel schedule):
     - L-T1: 43,840 cycles, 576 loads;
     - L-CA: 117,887 cycles.

     L-T1 is uncalibrated; the gap is the first calibration data point for the program path.
   - **Tests:**
     - `test_csp_run`, now 7 cases:
       - matmul and linear at L-B, L-T1 and L-CA, all bit-identical to the oracle;
       - L-T1: loads and stores equal the validator's, peak within [live set, program L3], a
         deterministic makespan;
       - LU at L-T1, 16 loads;
       - a streamed program with 3 slots, where every Load after the third starts no earlier
         than the Release that freed its slot, values equal to L-B.
     - CLI: LU at `--level block-sequential` on S1.
4. **The corpus, the platform, and retiring the L0 path of L-T1.** Split on 2026-10-10. The L0
   L-T1 executor turned out to underpin four working subsystems, so retiring it in one step
   would have broken them. Each consumer moves to the program first, in its own PR, and the
   executor retires last.

   | Depends on the L0 L-T1 executor or L0 platform programs | For |
   |---|---|
   | Orchestration (#305: `.kpuld`, MMIO ABI, `KpuDevice`) — moved in 4d | cross-operator residency: seeded and retained tiles, foreign-held slots |
   | Tile-flow debugger (#286: `.tflow`, viewer, checker, T4 reference run) | L-T1's per-hop records and residency intervals |
   | `kpu-run --step`, `--timeline` | L-T1's transaction cursor and timeline |
   | Characterization (`examples/characterize`) | the platform and L-T1 on L0 programs |

   - **4a: the corpus and the platform** (done).
     - **`lang::format`** (`csp/lang/format.hpp`) gives a program's canonical text: its structure
       in one layout, without comments, with constant subexpressions folded and parentheses
       only where precedence needs them. It is a fixed point and compiles to the same actions.
       Two spellings of one program format identically.
     - **`VirtualPlatform` holds CSP programs** beside its L0 ones: `load_csp(ast, inputs)`
       validates against device 0's sites; `run_csp(h, level, window)` and
       `csp_reference(h)`. The program digest is the digest of the canonical text, the
       snapshot digest that of the inputs. `RunIdentity` gains `schedule` (L-CA's window),
       compared and rendered; the class-closing test was extended for it. A CSP run is a pure
       function of (program, inputs, deployment, level, schedule) and leaves no state on the
       platform. Cross-run residency is 4d's.
     - **`driver::run_csp` is split** into `csp_reference` (the oracle) and `csp_run_level` (one
       level), which the platform calls.
     - **`kpu-run`'s `.csp` path goes through the platform:** one execution path, a run id per
       level, `--emit-l0-result` after agreement. Without `--deploy` the platform holds the
       default device, placement is not checked, and L-CA is skipped with its reason.
     - **The corpus gains three `.csp` programs:** matmul 48³/16 and LU 64/16 (moved from
       `tests/program/csp/`, with the existing inputs and results) and linear 64/16 relu (with a
       new input/result pair). `test_csp_corpus` runs each through the platform on S1 at L-B,
       L-T1 and L-CA, asserting the reason where L-CA is refused (LU). It makes the L0 corpus's
       two claims: within tolerance of the recorded result across machines, and bit-identical
       to the program's own oracle on one machine.
   - **4b: the tile-flow record from the program** (done).
     - **`.tflow` version 4:** the columns are version 3's, an op is one of the program's
       actions (`"ops": "csp-actions"`; kind = `Action::Kind`), and the residency intervals are
       the program's slots, credit to Release. Each movement action is one transit, a Store two
       (the ejection, then the DMA write), and each Call one compute. The L3 station's capacity
       is the program's L3. `record::build_csp_record(ast, L-T1 outcome, inputs, spec)` builds
       it from `TransactionalInterpreter`'s per-leg records (now carrying the action index) and
       slots; `read_tflow` reads 3 and 4.
     - `kpu-run --program file.csp --tflow dir` writes it from the L-T1 run.
     - **`tflow_check.py`** reads version 4, which must name its ops; one without `ops` is
       refused (exit 2).
       - TF6 is restated for programs: every transit into or out of L3 runs while its tile
         holds a slot. An accumulator has none, and a Feed reads L2 after the program released
         the L3 copy.
       - TF9 counts a Writeback (hop 4) as filling a slot, since a result reaches L3 that way,
         not by being computed there.
       - The other checks are unchanged.
     - **The viewer** reads 3 and 4 and names a v4 op by its action. `bind.py` binds a v4
       record unchanged.
     - **Found by the checker:** L-T1 let a tile's next Load (or opening Writeback) take a
       credit before its previous residency's Release had run, so the tile held two slots
       (TF5). A program Load now also waits for the tile's previous Release, as L-CA's DMA
       already did.
     - **Tests:**
       - `test_tile_flow_record` v4 case: matmul, unfused linear and LU. One op per action,
         slots = residencies, L3 capacity = the program's, the record's peak = L-T1's peak
         <= L3, one slot per tile at a time, a compute per Call, two legs per Store, and a
         read-back.
       - CTests: the T4 reference run from a `kpu-csp-gen` program (512³/64), and the corpus
         linear. Each is checked (all nine invariants), with the v4 self-test (clean passes;
         TF6 and TF9 fail when broken; an unlabelled v4 is exit 2; read as v3, TF6 fails), the
         viewer smoke test, and binding to the T4 floorplan.
       - The v3 path's tests are unchanged and pass.
   - **4c: stepping and the timeline from the program** (done).
     - **`--step`** walks the program at the finest of L-B and L-T1 that ran, through
       `VirtualPlatform::step_begin(CspProgramHandle, level)` (`driver/csp_step.hpp`):
       - L-B applies one action per step from the ActionStream, so values form as you step.
         Each step reports the tiles L3, L2 and the fabric hold after it.
       - L-T1 runs the program, then replays its records in start order (program order at
         one cycle). Each step reports, at its start cycle, the lanes busy per process and
         the L3 slots held. That is station occupancy, which the L0 executor could not give
         (it publishes only its peak).
       - L-CA's step is a cycle (#283) and is refused with that reason; so is `--step` on a
         run where neither L-B nor L-T1 ran.
     - **`--timeline`** writes the L-T1 records as Chrome-trace events
       (`driver/csp_timeline.hpp`): one per leg, on its process and lane, so a Store is two
       (BlockMover to the DMA buffer, DMA to DRAM). A Release takes no process and no time,
       so it is not an event; its slot is the `.tflow` record's to show.
     - **Tests:** `test_csp_step` over the corpus (matmul, linear, LU). At L-B, stepping to
       the end gives L-B's values and peak, and L3 is empty at the end. At L-T1, every action
       is replayed, starts never go backwards, lanes busy stay within each process's lanes,
       slots held stay within L-T1's peak, there is one Release per slot and two legs per
       Store. L-CA is refused. The timeline has one event per non-Release record, with its
       cycles, process, lane and bytes. Five CLI tests cover both modes, the two refusals,
       and the timeline.
   - **4d: orchestration on CSP programs.** Split in two:
     - **4d.1, the language and the levels** (done):
       - `inherit X;` and `retain X;` are statements. They lower to two new zero-time actions,
         Inherit and Retain, with no process.
       - The symbolic validator and the walker enforce the same rules:
         - an `inherit` opens the program, outside any loop, on an operand declared `in` or
           `inout`;
         - a `retain` mirrors a resident statement or names a closed accumulator, with its
           epilogue on the way (`retain y via relu @ fabric`);
         - inside loops, a `retain` names a distinct tile in every iteration;
         - a retained tile cannot be named again;
         - retained slots count against capacity to the end;
         - an inherited tile may be stored unwritten.
       - L-B: Inherit reads the operand's value into L3, and Retain leaves it there. The result
         is DRAM overlaid with the retained tiles' L3 values, and `retained()` gives those
         values.
       - L-T1: Inherit takes its credit at cycle 0 with no DMA. Retain is a zero-length record,
         and its slot is held to the makespan.
       - The `.tflow` v4 record marks inherited slots Seeded and retained ones Held. The viewer
         names the two new actions.
       - L-CA: `ConcurrentTimingExecutor::seed_l3` puts a tile's bytes into an L3 entry with a
         credit and a tag CAM entry, with no DMA. `CspDriver` seeds each Inherit (`l3_held`)
         and schedules nothing for a Retain. The result reads retained tiles from L3.
       - Fixtures: `matmul_relu_retain_32_t16.csp` (C = relu(A B), retained, never stored) and
         `bias_inherit_32_t16.csp` (C inherited, biased, stored).
         - Each matches its oracle bit for bit at L-B, L-T1 and L-CA.
         - Chained by hand at each level, the second operator's C equals the two oracles
           composed.
     - **4d.2, the orchestrator** (done): `.kpuld` operators carry `.csp` text, and `KpuDevice`
       runs chains of them, matching `retain` against `inherit`.
       - **`.kpuld` 2.0.0.** An operator carries `csp_program`, the canonical text, and
         `l0_program` becomes optional: the oracle, filled by `loadable::csp_operator` when the
         program has at most 1M actions. The reader validates the program at load. It refuses a
         1.x file with the reason ("carries L0 operators; this reader runs CSP operators"),
         after the min_consumer gate, so a file demanding a newer reader is still refused for
         that. The 1.x golden file is kept as `matmul_32_external_v1.kpuld`, to prove the
         refusal.
       - **Binding by declaration.** An operator's `inputs` bind to the operands declared `in`
         or `inout`, its `outputs` to those declared `out`, in declaration order. An operand
         whose shape or tiling differs from its tensor's is refused, so a tile index names the
         same region in every program.
       - **The device holds what programs retain.** It keeps each retained tile's values,
         because a retained tile need not be stored and DRAM may not have them. A launch
         overlays the inherited values on the program's inputs. Afterwards it writes back to
         DRAM only the tiles the program stored.
       - **The chain is checked at load**, by tensor tile, and the first operator it fails is
         marked invalid:
         - an `inherit` needs an earlier `retain` of that tile, with the same tiling, and no
           claim in between;
         - every `retain` needs a later `inherit`;
         - no operator in between may load a retained tile that was never stored;
         - two operators may not retain one tile.
       - **Decided 2026-10-11, both as recommended:**
         - **The decider issues RESERVE and LAUNCH only.** A PLACE is refused, because the
           program loads its own tiles, and a PLACE would be a DMA no program sequences. It
           returns with L-T2 (#283). Three things retire:
           - release at last read;
           - `reuse_shared_inputs`: reuse is now measured by running a chain written with
             retain/inherit against one written without;
           - the greedy-prefetch ablation with reservations off, since its wedge needs a tile
             held by the decider.
           R1-R4 and `ReserveThenLaunch` stay.
         - **The program's declared L3 is its need.** It reserves `l3` less the slots it
           inherits, which are already held. The device refuses when that does not fit beside
           the other held tiles, and never rewrites the text.
       - **MMIO ABI 2.0.**
         - `MAN_L3_SLOTS` takes `MAN_PEAK_LIVE`'s offset.
         - `MAN_LIST` (0x130) selects which list the read window shows: reads, inherits or
           retains.
       - The run identity's residency field stays empty for a CSP run: `inherit` and `retain`
         are in the program's text, which `program_digest` already covers.
       - **Measured:**
         - three GEMMs with a gap need 5 L3 slots cold and 9 warm, so holding W across the
           middle operator costs exactly its four tiles;
         - the warm chain loads 4 fewer DRAM tiles;
         - `relu_then_bias` keeps H = relu(X W) in L3 unstored, and the bias operator's
           stored H equals the two oracles composed, at L-B and L-T1.
   - **4e: retire the L0 L-T1 executor.**
     - Once 4b-4d land, `TileTransactionExecutor`, `run_at`'s L-T1 case and the L0 platform run
       path have no users.
     - L0 files then run at L-B only: they are the oracle and the trace format.
     - `examples/characterize` moves to csp-gen programs through the platform, or is retired if
       the analytical harness (`TileDag`) is all it still needs. This is decided in 4e with the
       code in front of us.
     - The L0 corpus keeps its role (format stability, the oracle), checked at L-B.
5. **`--algo` leaves `kpu-run`.**
   - Its CLI tests move to `csp-gen | kpu-run`.
   - Docs (the how-to, the L0 serialization note) and the CHANGELOG say where it went.

## 6. Verification

- **Values:** L-B = L-T1 = L-CA = the trace's L0 reference, bit for bit on one host, every operand:
  - matmul (corpus sizes and 256³);
  - linear in every placement and unfused (gelu, silu and atan within ADR tolerance across
    hosts);
  - LU at L-B and L-T1.
- **Residency is the program's at every level:**
  - peak L3 at L-B, which executes in program order, equals the validator's `peak_l3`;
  - at the timed levels (L-T1, L-CA) the DMA runs ahead when a credit is free, so their peak
    in time lies between the validator's `peak_l3` (the live set) and the program's L3
    (corrected in step 3);
  - DRAM loads at L-T1 and L-CA equal the program's Loads.
- **Determinism:** identical makespans across two runs and across CI's four platforms. A
  difference is a bug to fix, not a tolerance to add. L-CA's arbitration iterates maps keyed by
  `TileIDHash`, and step 2 audits its decision paths for iteration-order dependence.
- **Reported, not asserted:** L-T1 against L-CA makespans on S1, recorded in the session log as
  the first calibration data point.

## 7. Risks

- **Run time at L-CA.** It steps every cycle (matmul 256³ on S1 is about 44k cycles). The
  stream bounds memory; time is bounded by `max_cycles`. `--level all` on a large program is
  slow, and the report says which level took the time.
- **L-T1 timing moves.** Residency becomes the program's instead of inferred, so L-T1's makespans
  for the same operator change. That is the intended effect; the old numbers measured an
  execution order, not a program. Tests that pin L-T1 makespans are re-baselined in step 3, each
  with its old and new value in the session log.
- **The `--algo` removal touches scripts.** The CLI tests and docs are migrated in step 5. An
  `--algo` given to `kpu-run` after that prints where it went.

## 8. Review (2026-10-10)

The first draft ran L-CA from L0, lowered with Belady residency at the machine's L3. That was
the shortest path through the existing plumbing, and it contradicted ADR 0004 §4:
- the authority level would have been validated on derived residency, never on a written
  program;
- the unrolled trace would have stayed on the critical path;
- the one driver change it needed (the lowered epilogue) served the wrong input.

Revised:
- **CSP programs are the input at every level.** L0 is the oracle and the corpus.
- **L-T1 reads the program in this plan** (step 3 of the sequencing plan, pulled forward).
- **Program generation is a separate executable (`csp-gen`).** `kpu-run` is the simulator: it
  takes a CSP program, which configures its data path, and executes the operator.

## 9. Questions

- **Q1. When `--algo` leaves `kpu-run`.**
  - **Recommended:** step 5, after the corpus and the CLI tests have moved, so no step leaves the
    driver unable to run what it ran before.
  - **Alternative:** remove it in step 2 and migrate the tests then.
- **Q2. Inputs for a `.csp` program.**
  - **Recommended:** synthesized by default (`fill_inputs`, deterministic and exact), with
    `--inputs values.l0` for real data.
  - **Alternative:** a values section in `.csp` itself. I recommend against it: the program
    configures the data path, and the data is not part of it.
- **Q3. The trace limit for the oracle.**
  - **Recommended:** 1M actions by default, `--trace-limit` to change it.
  - Above the limit, the run compares levels against each other but has no L0 reference, and
    says so.
- **Q4. `csp-gen`'s schedule options.**
  - **Recommended:** one canonical schedule per operator in step 1. Panel orientation for
    matmul, and `--place` for linear, are the only options.
  - **Later:** a schedule search (the "reuse-aware schedule" deferred in the system-schedule
    plan) belongs to the generator, not the simulator.
