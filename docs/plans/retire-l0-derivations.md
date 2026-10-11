# Retire the L0 derivations: kpu-sim is the one source of an L0

**Date:** 2026-10-11
**Status:** Approved 2026-10-11 (Q1 yes, Q3 keep; Q2 decided: the L-B model is the oracle). PR 1 (#358) and PR 2 (#359) merged; PR 3 in review. PR 4 (Q4) awaits review
**Follows:** `docs/plans/kpu-run-csp-programs.md` (complete). Step 5 of that plan made the golden
corpus kpu-sim traces, and flagged this as the next step.
**Decided already (2026-10-11):** an L0 is the trace of a CSP execution (ADR 0004 §4). kpu-sim
is its only source: run a `.csp` program and take the trace. No second generator.
**Decided in review (2026-10-11):** the **L-B model is the oracle**. L-T1, L-T2 and L-CA are
transactional, refined and cycle-accurate refinements of the behavioral model, so the behavioral
model is the single source of truth for how a CSP program becomes an L0 trace of transactions and
operations, and for the values that trace computes.

## 1. Problem Statement

The C++ derivations still write L0 programs directly. They are a second L0 source beside kpu-sim,
and that is the divergence the decision above forbids:

| Derivation | Where | What it does |
|---|---|---|
| `derive_matmul_tile_program`, `derive_lu_tile_program`, `derive_linear_tile_program` | `include/sw/kpu/program/derive/` | write an L0 op list for an operator |
| `driver::derive(ProgramSpec)`, `driver::fill(TileProgram&, ProgramSpec)` | `driver/program_spec.hpp` | the spec front end, and per-operator input values |

Two more paths turn an L0 *into* a program, which inverts the trace relationship:

| Path | Where | What it does |
|---|---|---|
| `csp::lower(TileProgram, LoweringOptions)` | `csp/lower.hpp` | L0 to a CSP program, with Belady residency |
| `csp::lang::emit(TileProgram, cap, name)` | `csp/lang/emit.hpp` | L0 to `.csp` text |

The derivations no longer drive the simulator. kpu-run, the platform and the corpus moved off them
in steps 4e and 5. What still uses them:

| User | How | Category |
|---|---|---|
| `examples/characterize` | produces the L0 it characterizes (an `--algo` sweep) | production |
| `test_driver_run_at`, `test_l0_serialization`, `test_virtual_platform`, `test_tile_dependencies`, `test_tile_characterize`, `test_stream_signature` | "some valid L0 program" as a fixture | **A: fixture** |
| `test_csp_gen`, `test_csp_language`, `test_csp_structured`, `test_csp_linear`, `test_csp_linear_lca`, `test_csp_spec_oracle` | the derivation's values as a **second, independent oracle**, and `fill`'s per-operator inputs | **B: oracle, D: inputs** |
| `test_csp_program` (all of it), `test_csp_driver` (5 tests), `test_csp_linear` 209, `test_csp_language` 182 | derived L0 fed to `lower` or `emit` | **C: lowering** |
| `test_tile_program` | tests of the derivations themselves, plus the reference executor | the subject itself |

Nothing in `src/`, `tools/`, `python/` or `benchmarks/` calls a derivation (audited
2026-10-11).

## 2. Architecture

```
TODAY                                               AFTER

derive_*_tile_program ──► L0 ──► tests (A)          kpu-csp-gen / hand-written .csp
        │                  │                                 │
        │                  ├──► csp::lower ──► CSP           ▼  csp_inputs / fill_inputs (exact;
        │                  └──► lang::emit ──► .csp     ActionStream     LU strictly dominant)
        │                                                    │
driver::fill ──► inputs                                      ▼
        │                                            L-B (BehavioralInterpreter)
        └──► TileProgramReference on derived L0       ├──► values: THE ORACLE ──► L-T1, L-CA,
                 = "the derivation's values" (B)      │                          tests (B)
                                                      └──► L0 trace it executed ──► tests (A),
                                                               │                    corpus, characterize
                                                               ▼
                                                      TileProgramReference replay == L-B values
                                                      (the trace's validation, bit for bit)
```

## 3. Design

### 3.1 Fixtures come from traces

One test helper takes the L0 of a program from the program, so every fixture has one source:

```cpp
// tests/program/trace_fixtures.hpp
// The L0 of a CSP program, on the inputs kpu-run synthesizes: what `kpu-run --emit-l0` writes.
TileProgram trace_of(const csp::lang::Program& ast);            // csp_reference(...).trace
TileProgram trace_of(const csp::gen::Options& o);               // a generated program
TileProgram trace_of_file(const std::string& path);             // tests/program/csp/*.csp, corpus
```

- **WRONG:** `TileProgram p = derive(ps); fill(p, ps);`, an L0 written by C++.
- **CORRECT:** `TileProgram p = trace_of(gen::Options{.algo = "matmul", .size = 48, .tile = 16, .l3 = 16});`

Shapes kpu-csp-gen cannot write become hand-written `.csp` fixtures. The language already takes
`tile RxC`:
- rectangular tiles (7×5×6 in 3/2/4, and 6×4×8 in 2/2/4);
- a clamped trailing block (LU N=6, T=4; matmul 40³ in t16).

### 3.2 The oracle is L-B, and L-B writes the trace

Today `csp_reference` takes the trace from `csp::lang::compile` (the walker's `TraceSink`) and
replays it through `TileProgramReference`; that replay is the oracle, and L-B is one of the levels
compared against it. After this step:

- **L-B records the trace it executes.** The walker's L0 ops (feeds, calls, drains, context
  stages) travel on the `ActionStream` with the action they belong to, and
  `BehavioralInterpreter` appends them when it is asked to trace (`begin(operands, trace)`). The
  trace is what L-B ran, in the order it ran it.
- **L-B's values are the oracle.** `csp_reference` runs L-B on the stream. It needs no trace, so
  there is an oracle for every program, however large; `--trace-limit` now bounds only the trace.
- **The replay validates the trace.** When the trace is taken, `TileProgramReference` replays it
  on the same inputs and must reproduce L-B's values bit for bit, on every operand. A trace that
  does not reproduce its run is not emitted (kpu-run exits 2, naming the operand).
- **The B tests compare against L-B.** `test_csp_gen`, `test_csp_structured`, `test_csp_linear`,
  `test_csp_linear_lca` and `test_csp_spec_oracle` compare L-T1 and L-CA with L-B of the same
  program, on `csp_inputs`. The comparisons that were bit-exact stay bit-exact.

**WRONG:** keep `derive_matmul_tile_program` "for tests only", or add a test-only operator
reference. Both are a second source of truth beside the behavioral model, and the first is a
second L0 source.

What this gives up, stated: a generated program is no longer checked against an operator
computed some other way in the unit tests. That check is the corpus's: a trace is validated once
by the person who checks it in, and `test_l0_corpus` keeps kpu-sim producing it.

### 3.3 Inputs

`driver::fill`'s per-operator values are replaced by `csp_inputs` (a CSP program) and `fill_inputs`
(an L0 file):
- both are exact multiples of 1/8;
- `csp_inputs` makes an in-place-factored square operand strictly dominant (step 5).

The one bit-exact cross-check, `test_csp_spec_oracle:481`, holds on exact inputs; the matmul
fill already is. It becomes L-CA against L-B.

### 3.4 Lowering and emission retire with the derivations (recommended, Q1)

`csp::lower` and `lang::emit` exist to turn a derived L0 into a program. Once L0 is only ever a
trace, an L0→program path makes a program out of its own trace, which is backwards. kpu-csp-gen
writes the schedules, and a better schedule belongs to the generator (plan Q4 of
kpu-run-csp-programs).

What goes with them:
- `test_csp_program.cpp` (the lowering suite: Belady against an optimal-load DP, reload counts);
- `test_csp_driver`'s lowered-L0 tests at L-CA (85, 141, 161, 184, 234). Their L-CA coverage moves
  to generated programs: values, loads, the DMA credit order, refusals.
- `test_csp_linear` 209 and the `emit` refusal;
- `test_csp_language` 182.

What is lost: the proof that Belady lowering reaches the optimal load count. That is a property of
a retired path, not of the simulator.

### 3.5 `program_spec.hpp` keeps what is not a derivation

| Retires | Survives |
|---|---|
| `ProgramSpec` | argument parsing: `arg`, `has_flag`, `arg_present`, `arg_required`, `parse_dim`, `parse_double`, `parse_ints`, `parse_strs` |
| `known_algo` | input synthesis: `program_inputs`, `fill_operand`, `fill_inputs` |
| `derive` | dataflow names: `known_dataflow`, `map_for`, with a direct include of `stream_signature.hpp`; today it comes in transitively via `stream/derive/matmul_streams.hpp` |
| `fill` | the machine: `DeviceSpec`, `known_topology`, `make_deployment`, `make_device` |
| `result_operand(ProgramSpec)`; its callers name the operand | |
| the includes of `derive/*.hpp` | |

### 3.6 The stream layer stays, as the characterizer's analytical model (Q3)

`StreamProgram` and `derive_matmul_streams` no longer feed any executor; that went with L-T1 in
4e. What they still serve:
- the analytical L1 timing in `TileDag` (`examples/characterize`, `test_tile_characterize`);
- the serializer's STREAMS annotation (the map name only).

`derive_matmul_streams` reads only operand tile shapes and `MatMulAccum` op indices, so it works on a
trace unchanged. It derives an *annotation*, not an L0, so it is not a second L0 source. Whether
the systolic L1 model should survive is a separate question about characterization. It is out of
scope here.

### 3.7 `examples/characterize` takes its L0 from kpu-csp-gen

`tile_characterize --algo --sizes --tiles` keeps its interface. Each cell's L0 becomes the trace of
`csp::gen::generate(Options)`, with `l3` from the deployment's L3 (as kpu-csp-gen's `--target`
takes it). Its naive value checks (`validate_matmul`, `validate_lu`) become a comparison with
L-B's values for the same program (§3.2).

### 3.8 Functional checks: tile contents, interpreted, within a proximity bound

**Decided by the user (2026-10-11), at the start of PR 2.** A machine's functional behaviour has
two parts, and they are checked differently:

1. **The sequence of tile moves through the resources**: which tiles are loaded, moved, fed,
   drained, written back, stored and released, by which process, and in what order. This is
   structure, and it is checked **exactly**: DRAM load and store counts, credit order against
   program order, every L3 credit returned.
2. **The bits of the result.** Floating point is not associative, and a data path legitimately
   reorders reductions and converts between formats (quantization, accumulator widths). So a
   value check is made **on tile contents, through an interpretation layer**:
   - the layer reads a tile's stored bits **in the format the program intends** (float32 today;
     bf16, fp16, fp8 and the integer formats as the machine comes to carry them);
   - each element is asserted to be **within a proximity bound** of the oracle's (L-B's) value
     for the same tile;
   - a failure names the operand, the tile, the first element out of bound, and the worst
     absolute and relative error.

The layer is `include/sw/kpu/program/tile_check.hpp`:
- `ElementFormat` is the interpretation: the `DataType` the bits are meant as, and how a stored
  element decodes. Operands store float32 today, so float32 is the one format decoded; any other
  is refused by name until operands carry it, rather than checked as though it were float32.
- `Proximity` is the bound, the ADR 0001 D5 comparator per element:
  `|actual - oracle| <= atol + rtol * |oracle|`. `proximity_for(format)` gives the format's
  default (float32: atol 1e-6, rtol 1e-4); a test states a looser one where its data path
  justifies it.
- `check_tiles(actual, oracle, operand, format, proximity)` walks the operand's tile grid
  (clamped trailing tiles included) and returns a per-tile report.

**WRONG:** `CHECK(r.values.operand("C").values == lb.result().operand("C").values)`, bit
equality of whole operands as the functional bar.
**CORRECT:** `CHECK(check_tiles(r.values, lb, "C").pass)`, plus exact checks of the move
sequence.

Scope:
- **PR 2** uses the layer for every value check it writes or moves: `test_csp_driver`'s L-CA
  tests and the `test_csp_program` tests that survive the lowering.
- **PR 4** (proposed, §4 step 4, for review): the remaining value checks move to the layer.
  These are kpu-run's level comparison, `run_csp`, and the CSP and platform suites. ADR 0001 D5
  and the fidelity table in `CLAUDE.md` ("bit-identical to L-B") are amended to match.

## 4. Implementation Steps

Three PRs, so each leaves the suite green and the derivation's users strictly fewer.

1. **The L-B oracle and the fixtures** (PR 1):
   1. L-B records the trace; `csp_reference` runs L-B and validates the trace by replay (§3.2).
      kpu-run reports "oracle L-B" and the replay. The corpus must come out byte-identical, which
      shows the trace did not change source in anything but name.
   2. Add `tests/program/trace_fixtures.hpp` and the hand-written `.csp` fixtures for rectangular
      and clamped tiles (`tests/program/csp/`).
   3. Move the A users to traces:
      - `test_driver_run_at`, `test_l0_serialization`, `test_virtual_platform`;
      - `test_tile_dependencies`, `test_tile_characterize`, `test_stream_signature`.

      Literals that pin a derived op line gain the trace's `label="line N"`, for example
      `test_l0_serialization` 113.
   4. Move the B/D users to `csp_inputs` and the L-B oracle:
      - `test_csp_gen`, `test_csp_language` (except 182), `test_csp_structured`;
      - `test_csp_linear` (except 209), `test_csp_linear_lca`, `test_csp_spec_oracle`.
   5. `test_tile_program`: the kernel and executor numerics move to traces, including the
      rectangular and clamped fixtures. The derivation-shape assertions are removed with their
      subject in step 3.
2. **Retire `csp::lower` and `lang::emit`** (PR 2, Q1):
   - remove `test_csp_program.cpp` and the lowered-L0 tests in `test_csp_driver`, `test_csp_linear`
     and `test_csp_language`;
   - add `tile_check.hpp` (§3.8) and its unit test;
   - move `test_csp_driver`'s L-CA assertions onto generated and written programs. Values are
     checked per tile against L-B (§3.8). DRAM loads, stores, credit order and the refusal
     messages are checked exactly;
   - keep `test_csp_program`'s tests of the program form (process ownership, read-after-release
     refusal, L-B values, LU load and store counts) on compiled programs;
   - delete `csp/lower.hpp` and `csp/lang/emit.hpp`, and reword the comment references
     (`csp_program.hpp`, `print.hpp`, `l0_format.hpp:307`).
3. **Characterize, then delete** (PR 3):
   - move `examples/characterize` to `csp::gen` traces (§3.7);
   - delete `include/sw/kpu/program/derive/` and the `program_spec.hpp` retirements (§3.5);
   - update `CLAUDE.md`, ADR 0004 (a status note: the derivations retired), the corpus README's
     history line, the CHANGELOG and a session log.

4. **Functional checks move to the tile layer** (PR 4, proposed, Q4): kpu-run, `run_csp`, the CSP
   and platform suites; amend ADR 0001 D5 and the `CLAUDE.md` fidelity table.

## 5. Verification

- **The deletion is complete:**
  `git grep -nE "derive_(matmul|lu|linear)_tile_program|driver::derive|\bderive\(ps|csp::lower|lang::emit\(" -- ':!docs'`
  returns nothing.
- **Nothing that was checked is lost silently.**
  - Each migrated test keeps its assertion against the new source.
  - The PR description lists every assertion that is removed, with why: the lowering suite and the
    derivation-shape counts.
  - Until PR 2, assertions were not weakened from bit-exact to tolerance anywhere they were
    bit-exact; the one exception is LU, which was already compared by reconstruction and
    tolerance. From PR 2, value checks move to the tile proximity layer by decision (§3.8), and
    each PR lists the checks it moved.
- **The corpus is untouched:** `test_l0_corpus` and `test_csp_corpus` pass with no regenerated file.
- **Full suite** on all CI platforms, clang included: input conditioning is host-sensitive, as step
  5 showed.
- **`tile_characterize`'s smoke test, deploy test and bad-number test pass.** One sweep is diffed
  before and after. Its numbers move only where the trace's op order differs from the derivation's,
  and the session log records which.

## 6. Key Invariants

- Every L0 in the repository, whether fixture, corpus or characterized program, is the trace of a
  `.csp` program, compiled or run by kpu-sim.
- No code turns an L0 into a program.
- A test's oracle is L-B on the same program and inputs. It is never another L0 or another model.
- A trace leaves kpu-sim only after its replay reproduces L-B bit for bit.
- A functional value check is made per tile, through the format interpretation layer, within a
  stated proximity bound. The sequence of tile moves is checked exactly.
- Inputs are exact multiples of 1/8, and an operand factored in place is strictly dominant.

## 7. Questions

- **Q1. Retire `csp::lower` and `lang::emit` with the derivations?**
  - **Decided: yes** (§3.4). An L0→program path is the inverse of "L0 is a trace", and its
    only callers are tests of itself.
  - **Alternative:** keep `lower` as a schedule-search tool fed by traces. I recommend against it.
    It would still make a program from a trace, and the reuse-aware schedule search was placed in
    the generator.
- **Q2. The oracle for the B tests?** **Decided: L-B** (§3.2). The test-only operator reference
  was declined: the behavioral model is the single source of truth, and the refinements answer
  to it.
- **Q3. The stream layer (`StreamProgram`, `derive_matmul_streams`)?**
  - **Decided: keep it**, as the characterizer's analytical model, applied to traces (§3.6).
  - Whether the systolic L1 model should survive is a separate question.
- **Q4. How far do functional checks move to the tile layer, and does anything stay bit-exact?**
  - **Recommended:** every check of a level's values against L-B moves to the tile layer (PR 4).
    The trace's replay validation (§3.2) stays bit-exact. It is not a functional check of the
    machine. It checks that a trace records the run it came from: the same kernels in the same
    order on the same inputs. Any difference there means the trace is wrong, not that the data
    path rounded differently.
  - **For review:** whether L-T1 against L-B, which today is bit-identical by construction,
    keeps an additional bit-exact assertion as a determinism check, or is held only to the
    proximity bound.
