# Onboarding: the KPU Simulator

This guide is for a developer joining kpu-sim. It covers three things:

1. **What the simulator is for**, and the ideas that shape it.
2. **How the stack works**, from a program to a checked, timed, recorded run.
3. **A guided path through examples**, about an afternoon's work, each step saying what to look
   for.

It points to the detailed documents rather than repeating them. When this guide and a linked
document disagree, the linked document is right; please fix this guide.

---

## 1. What kpu-sim is for

The KPU (Knowledge Processing Unit) is a **domain-flow machine**: a nearest-neighbor compute fabric
fed by a hierarchy of buffers (L3, L2, L1), in which data moves by credit, not by request. kpu-sim is
its **architecture simulator**. It exists to prove that the machine works, what features would be
beneficial, which computer arithmetic would benefit execution, and how fast the engine can solve
application execution, before it is built:

- It decides the architecture. The RTL (the hardware implementation) implements what kpu-sim
  has demonstrated. kpu-sim is the experimental state space exploration tool to evaluate which
  features are useful. Once features are proven to be valuable, RTL implementation can then
  leverage the kpu-sim as a verification reference.
- It runs real programs and computes their real values at every level of detail, so a design
  change is judged both on whether the answers are still right and on what it does to time.
- It is a computer arithmetic laboratory. Number systems come from the
  [Universal](https://github.com/stillwater-sc/universal) library through one registry
  (`include/sw/kpu/program/numeric.hpp`), so a new format is a registry row, not a new code
  path.

## 2. The ideas behind it

Here are the ideas a newcomer is most likely to trip over, because they differ from how CPUs,
GPUs and most simulators work. Each one is a design rule; code that breaks one is a bug, even
when its tests pass.

### 2.1 One program, interpreted at several levels of detail

There is **one program**, a CSP program (communicating sequential processes, written in the CSP
language as a `.csp` file). Every simulation level is an **interpreter** of that same program.
The levels differ only in how finely they break up a transaction (ADR 0002):

| Level | Name | A transaction is | Speed |
|---|---|---|---|
| **L-B** | behavioral | one whole move, atomic | fastest |
| **L-T1** | block-sequential | one tile move, timed under its slot's credit | fast |
| **L-T2** | resource-transactional | per-resource reads and writes (not built yet, #283) | — |
| **L-CA** | cycle-accurate | protocol events, clock by clock | the baseline |

CSP is the *program layer*, not one of the levels. There is no "CSP model" to compare against
the others.

### 2.2 Every level computes values

Decomposition changes *when* things happen, never *what* is computed. Each modeling level must
model execution and return values to guarantee that the functionality of the machine is tested.
Spatial/Temporal machines like the KPU use timing and spatial organization to order computation.
Bugs in placement and timing will lead to incorrect execution and result values will carry that
evidence.

A set of simple rules govern the modeling:

- **Values answer to L-B**, the behavioral level, which is the oracle.
- **Timing answers to L-CA**, the cycle-accurate level.
- Every level's values are checked against L-B's, tile by tile and for every operand (an input a
  level overwrote is a bug too), within `|a − o| ≤ atol + rtol·|o|`. Bit equality is the bar
  only where the same computation is repeated: a replay, a round trip.

### 2.3 Credits up, data down: buffers, not caches

The KPU is a **credit-based dataflow** machine. A producer pushes a tile downstream only when the
consumer has given it a credit (free space); the consumer returns the credit when it is done.

- L3, L2 and L1 are **buffers, not caches**. There are no hits, misses, evictions or refetches,
  and nothing is fetched on demand. Code and trace events never use cache words.
- Components match out-of-order arrivals with a tag CAM, not by address lookup.

Read [`docs/01-architecture/kpu-execution-model.md`](01-architecture/kpu-execution-model.md)
before writing any KPU code. Its rules are mandatory.

### 2.4 Reuse is deliberate

Any reuse of data (an operand staged in L2 across a loop, a stationary operand held across `k`
steps) **must be stated in the program's data movement**. No level, executor or RTL tile may
discover reuse on its own. Reuse found on the fly hides the bubbles the program really has, and
the timing stops being the program's. When a program lacks reuse it needs, the simulator should
show the bubbles, not quietly fill them.

### 2.5 Placement is behavior

Where each tile lives (its L3 slot and bank, its L2 slot) is chosen **once**, by the walker that
expands the program, and recorded. Every level executes those addresses; no level allocates a
slot of its own (ADR 0004 §5). A pooled, anonymous L3 is never the intended model.

### 2.6 Everything streams from the array's edges

The compute fabric has only nearest-neighbour connections. Every input, including an
accumulator's initial value, a bias or the existing `C` of `C += A·B`, is streamed in from the
array's edge and propagated through it. There are no global writes into the array: a bus to every
accumulator would cost too much power and area. The design problem is pipelining those streams.

### 2.7 One record format, shared with the RTL

kpu-sim and the RTL write the same **domain-flow record** (`.dflow`,
[`docs/plans/shared-tile-flow-record.md`](plans/shared-tile-flow-record.md)): every transaction,
token and credit, with its cycle. Records from any level, or from the RTL, are checked by the
same checker, open in the same viewers, and join event by event with `dflow_diff.py`. That join
is how the RTL is held to the architecture.

### 2.8 kpu-sim writes the traces

L-B is the one place a CSP program becomes an L0 trace (the program's low-level operation
list). No C++ code writes an L0 by hand, and the golden corpus in `tests/program/corpus/` is
`kpu-run`'s output, kept only after every level agreed with it.

## 3. How the stack works

Read [`docs/02-simulation/simulator-architecture.md`](02-simulation/simulator-architecture.md)
next: it walks a program through every stage, with the code for each. In short:

```text
 kpu-csp-gen ──► .csp program ──► parse ──► validate ──┐
 (or by hand)                                          ▼
 deployment spec (.json: the machine) ──────► VirtualPlatform
                                                       │
                          walker: actions, L3/L2 slots, L0 ops
                       ┌───────────────┼────────────────┐
                       ▼               ▼                ▼
                      L-B            L-T1             L-CA
                  (the oracle)   (tile timing)   (cycle timing)
                       │               │                │
                       └─ values checked against L-B ───┘
                       │               │                │
                 L0 trace, .dflow   .dflow, .tflow,   .dflow, bubble
                                    bubble report     report, --step
```

The pieces you will touch most:

| Piece | Where |
|---|---|
| The driver, `kpu-run` | `tools/kpu-run/kpu_run.cpp` (thin: options, then the platform) |
| The program generator, `kpu-csp-gen` | `tools/csp-gen/`, `include/sw/kpu/program/csp/gen/` |
| The CSP language: parse, validate, walk, place | `include/sw/kpu/program/csp/lang/` |
| L-B and L-T1 | `include/sw/kpu/program/csp/behavioral.hpp`, `transactional.hpp` |
| L-CA: the credit model, the processes, the L2 banks | `include/sw/kpu/timing/` |
| The machines (deployment specs) | `tests/program/deploy/*.json` |
| Records, checkers and the diff | `include/sw/kpu/program/record/`, `tools/trace/` |
| Viewers | `tools/visualization/{domainflow,occupancy,tileflow,memflow}/` |

Two terms you will meet constantly:

- **The Combiner** is the tile's result-path unit between L1 and L2. It merges partial results
  and applies fused epilogues: bias, activation, scaling, format conversion. The tile has no
  separate accumulator memory and no DMA of its own
  ([`docs/01-architecture/combiner.md`](01-architecture/combiner.md)).
- **A dataflow** names the operand that stays put in the fabric: `os` (output-stationary, `C`
  stays), `ws` (weight-stationary, `B`'s tile stays), `is` (input-stationary, `A`'s tile
  stays). The choice decides where partials live and what reuse the program must state.

## 4. A guided path through examples

Every command below runs from the repository root. The paths assume the `release` preset's
build directory, `build/`.

### Step 0: build and test

```bash
cmake --preset release && cmake --build --preset release
ctest --test-dir build --output-on-failure
```

Install Node and put it on your `PATH` first: the viewers' data-layer tests run
under Node and skip silently without it. On Windows, use the `windows-msvc` presets; see
[`how-to-build.md`](../how-to-build.md). There is no `release` *test* preset; the test presets
are `default`, `unit`, `integration` and `performance`.

### Step 1: run one program at every level

```bash
build/tools/kpu-run/kpu-run --program tests/program/corpus/matmul_48x48x48_t16.csp \
    --deploy tests/program/deploy/kpu_s1.json --level all
```

Look for:
- **The program and the machine.** Open the `.csp` file: tensors, loops, `resident` and
  `release` (the program's own data movement), `acc ... in fabric` and `call gemm`. Open
  `kpu_s1.json`: the machine's compute tiles, DMA engines, movers, streamers and memories.
- **One line per level.** L-B reports no time; L-T1 and L-CA report a makespan in cycles; L-T2
  is skipped, as it is not built yet.
- **`unmodelled:` lines.** Each says what a level does not model for this machine. They are part
  of the result, not noise.
- **The calibration line.** It is L-T1's makespan over L-CA's, reported, never held to a
  tolerance: L-T1 is not yet calibrated against L-CA.
- **"values vs L-B".** Every level within bound, and the final `OK`. This is the line that must
  never regress.

### Step 2: record the run, check the records

```bash
build/tools/kpu-run/kpu-run --program tests/program/corpus/matmul_48x48x48_t16.csp \
    --deploy tests/program/deploy/kpu_s1.json --level all \
    --dflow mm.dflow --dflow-lt1 lt1.dflow --dflow-lca lca.dflow
python3 tools/trace/dflow_check.py mm.dflow
python3 tools/trace/dflow_check.py lca.dflow
```

`mm.dflow` is L-B's record: every action as a state transition, with placement. `lt1.dflow` and
`lca.dflow` are the timed levels' own records. The checker should print `OK`. Its rule codes
(G1, E2, …) are documented at the top of `dflow_check.py`.

### Step 3: look at them

```bash
python3 tools/visualization/domainflow/pack.py mm.dflow -o mm.html
python3 tools/visualization/occupancy/pack.py lca.dflow --overlay lt1.dflow -o occ.html
```

Each writes one self-contained HTML file; open it in a browser.
- **The domain-flow viewer** (`mm.html`) steps through L-B's run one action at a time: the
  machine, the program line and loop indices, and the L3 slots. Watch a `resident` fill slots,
  and a `release` return their credits.
- **The occupancy viewer** (`occ.html`) puts L-CA's run on one clock axis: a lane per process
  instance and per buffer slot, with L-T1's run overlaid. Find where the two levels' timing
  diverges.

Each viewer's `README.md` explains every panel.

### Step 4: compare two levels

```bash
python3 tools/trace/dflow_diff.py lt1.dflow lca.dflow
```

The diff joins the two records event by event on what each event *is* (kind, process, tile,
layers, program line, loop indices), not on when it happened, then reports what only one
record has and where the timing first diverges. The same tool joins a kpu-sim record with the
RTL's. Exit status 1 here is expected: the levels are meant to differ in time.

### Step 5: write your own program, and read its bubbles

```bash
build/tools/csp-gen/kpu-csp-gen --algo matmul --size 64 --tile 16 --l3 32 --dataflow os -o os.csp
build/tools/csp-gen/kpu-csp-gen --algo matmul --size 64 --tile 16 --l3 32 --dataflow ws -o ws.csp
for p in os ws; do
  build/tools/kpu-run/kpu-run --program $p.csp \
      --deploy tests/program/deploy/kpu_s1_combiner.json --level all --bubbles
done
```

`kpu-csp-gen --help` lists the operators (`matmul`, `linear`, `lu`) and the options: dataflow,
accumulator initialization, number type, epilogue placement, requantization.

Look for:
- **The programs' differences.** Diff `os.csp` against `ws.csp`: where `C`'s partials live
  (the fabric against L2 slots, merged by the Combiner), and what each stages.
- **The bubble report.** For each compute lane: how long it sat idle before each call, and
  *why*. The causes include an operand's transfer, a slot's credit, a busy process lane, a
  result's way out ahead of it, and in-order issue. Each bubble lists the chain of actions the
  report walked back through.
- **Reuse.** The report lists the reuse each operand's program states: feeds per load, feeds per
  move. A program that needs more reuse than it states shows it here as bubbles.

### Step 6: single-step

```bash
build/tools/kpu-run/kpu-run --program os.csp --deploy tests/program/deploy/kpu_s1.json \
    --level behavioral --step --step-limit 20
```

`--step` single-steps the finest level that ran. At L-B, a step is one action: its slot, its
process and edge, its program line and loop indices, and the buffers' occupancy. At
`--level cycle-accurate`, a step is one clock, with its protocol events and the transfers in
flight.

### Step 7: read a test, then change something

- `tests/program/test_csp_corpus.cpp` runs the corpus at every level and states the two claims
  the corpus makes.
- `tests/program/test_csp_bubbles.cpp` shows how a bubble's cause is asserted.
- A good first change: run a corpus program on another machine (`kpu_t4.json`, `kpu_t16.json`,
  or several `--deploy` flags at once, which is a sweep) and explain the makespans' differences
  from the bubble reports.

### About `examples/`

`examples/` (basic, behavioral, blas, …) still builds, but most of it predates the CSP program
stack. It exercises the hardware components' APIs directly, not the program path above. Use it
to learn a component's interface, not as a model of how programs run.

## 5. How we work

- **Read before you write.** Section 2's rules, the execution model, and for trace work
  `patterns/memory/lpddr5/INVARIANTS.md`.
- **Validate before you call it done.** Build and test with gcc *and* clang; run the record
  checkers on any record you produce. A crashing or failing test is a bug to fix, never one to
  skip.
- **Plans and decisions.** Designs live in `docs/plans/`, decisions in
  `docs/architecture/adr/`, and each working session's log in `docs/sessions/`.
- **The release train.** Milestones are releases toward v1.0 (`docs/ROADMAP.md`).
- **Changes land as pull requests.** Every change goes through CI and a review, with a
  `CHANGELOG.md` entry.

## 6. Where to read next

| To understand | Read |
|---|---|
| How a program runs, stage by stage | [`02-simulation/simulator-architecture.md`](02-simulation/simulator-architecture.md) |
| The credit model (mandatory) | [`01-architecture/kpu-execution-model.md`](01-architecture/kpu-execution-model.md) |
| Why the levels interpret one program | [ADR 0002](architecture/adr/0002-csp-interpretations-and-virtual-platform.md) |
| The program contract, and how values are compared | [ADR 0001](architecture/adr/0001-program-contract-and-transactional-engine.md), D5 and §7.5 |
| CSP as a language; placement as behavior | [ADR 0004](architecture/adr/0004-csp-program-is-a-language.md) |
| The language reference | [`plans/csp-language.md`](plans/csp-language.md) |
| Why the viewers live here | [ADR 0005](architecture/adr/0005-viewers-and-debug-tools-live-in-the-architecture-simulator.md) |
| The record shared with the RTL | [`plans/shared-tile-flow-record.md`](plans/shared-tile-flow-record.md) |
| The levels, and what each answers | [`02-simulation/fidelity-framework.md`](02-simulation/fidelity-framework.md) |
| The Combiner | [`01-architecture/combiner.md`](01-architecture/combiner.md) |
| The whole documentation tree | [`README.md`](README.md) |
