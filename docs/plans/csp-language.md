# The CSP Language: Writing the Tile Sequencing

**Date:** 2026-10-09
**Status:** Decided 2026-10-09 (A1-A3 with the request; Q1-Q6 answered, §7); steps 1, 1c and 2 done; step 3 (the vector processor at L-CA) next
**Decision record:** ADR 0004 (the CSP program is a written language).
**Is:** step 1b of `docs/plans/csp-program-tile-sequencing.md`, ahead of its step 2b (`kpu-run`).
**Related:**
- `include/sw/kpu/program/csp/` (the IR, lowering and L-B interpreter, #341).
- `timing/csp_driver.hpp` (L-CA, #342).
- `docs/plans/e10_fused_epilogue_pattern.md` (the fused epilogue as built).
- `docs/01-architecture/kpu-architecture.md` §2.7 (push-based fusion, SFU banks on egress).

## 1. Problem Statement

The CSP program decides the things that set reuse, and so energy:
- which tiles come into L3, and how long they stay;
- the order in which tiles meet the compute fabric;
- what is done to a tile as it moves.

Today it can only be *derived*: the C++ `derive_*` functions emit L0, and `csp::lower` decides
residency by Belady. Nobody can write a sequencing to try it. ADR 0004 changes that. This plan
defines the language.

Three drivers shape it:

| Driver | What it forces the language to express |
|---|---|
| **matmul** | loops, explicit residency (panels kept in L3), an output-stationary accumulator, tile-function calls standing in for the inner loops |
| **LU** | in-place updates, a data-dependent pivot flowing between calls, a trailing update whose working set must fit L3 |
| **linear** = matmul + bias + activation | **fusion**: *where* the epilogue runs. It can run in the compute fabric, on a streamer's drain, or on a vector processor at a BlockMover's ingress or egress, as part of the tile's **context**. Or it can run unfused, as its own pass. |

**The linear driver has nothing to build on below the language either:**
- L0 has no bias or activation ops.
- The epilogue runs only inside `execute_matmul`, as part of the compute, at no cost.
- A `VectorEngine` timing model exists (`models/temporal/datamovement/vector_engine.hpp`, the
  streamer's L1→L2 path), but nothing executes it.
- The ISA's inline-VE drain fields are serialized and never executed.
- No vector unit exists on the BlockMover path.

So this plan also builds the epilogue's operations and the vector processor that runs them.

## 2. Architecture

```
  writer: human / AI assistant / compiler (derive_*, later lower_to_csp from the DFP)
                         │  .csp text
                         ▼
   parse ──► AST ──► validate ──► lower ──► CspProgram IR (processes, channels, actions)
                                               │         ▲
                     print ◄───────────────────┘         │ (the L0 lowering, #341,
                                                         │  is another producer of the IR,
                                                         │  and emits .csp via print)
                                               ▼
                         L-B (csp::BehavioralInterpreter)   values
                         L-T1 (TileTransactionExecutor, from the IR)
                         L-CA (CspDriver → ConcurrentTimingExecutor)   values + timing
```

## 3. The Language

### 3.1 Principles

- **Structured:** loops over tile indices, C-style `{ }` blocks, no labels, no jumps (A1, Q1).
- **Implicit processes at level 1** (A2). A statement implies the actions of the dma, bm, str
  and cf processes on the flat machine.
- **Explicit placement at level 2** (A2): partition maps, owners, and NoC communication.
- **Residency is written down, and only written down** (Q2). A call reads tiles the program
  made resident in L3. A call on a non-resident tile is an error, never an implicit load.
  Residency is a property of the program, not of an execution order. That is the failure of
  writing high-performance data movement in the stored-program and SIMT abstractions, where
  what stays near the compute is an accident of the order instructions happen to run in.
- **Inner loops are tile functions.** `call gemm(...)` is a whole matmul engine invocation. The
  language never addresses elements.
- **Every statement lowers to IR actions one way,** so the printed action listing is the
  program's meaning.

### 3.2 Grammar (EBNF sketch)

```
file       := 'csp' VERSION NL program                               -- e.g. `csp 1.0` (Q3)
program    := 'program' NAME machine '{' decl* stmt* '}'
machine    := 'machine' 'flat' '(' 'l3' '=' INT ')'                      -- level 1
            | 'machine' 'grid' '(' INT 'x' INT ',' 'l3' '=' INT ')'      -- level 2
decl       := 'tensor' NAME '[' INT ',' INT ']' 'tile' INT 'x' INT io ';'
            | 'vector' NAME '[' INT ']' 'tile' INT io ';'                 -- e.g. a bias (Q6)
io         := 'in' | 'out' | 'inout'                                      -- (all in DRAM)

stmt       := for | resident | release | acc | call | store | distribute | broadcast
for        := 'for' VAR 'in' expr '..' expr '{' stmt* '}'
resident   := 'resident' tiles ';'         -- into L3; an error if already resident
release    := 'release' tiles ';'          -- the program's last use: the credit returns
acc        := 'acc' tile 'in' 'fabric' '{' stmt* '}'          -- an output-stationary accumulator
call       := 'call' FN '(' tiles ')' ( '->' | '+->' ) tile [ context ] ';'
                                           -- '->' writes, '+->' accumulates
store      := 'store' tile [ context ] ';' -- out of the fabric (or L3) to DRAM
context    := 'via' stage ( ',' stage )*   -- the tile context (§3.4)
stage      := VEOP '(' args ')' '@' place
place      := 'fabric' | 'str.drain' | 'bm.egress' | 'bm.ingress'

tiles      := tile ( ',' tile )*
tile       := NAME '[' index ( ',' index )? ']'
index      := expr | ':'                   -- ':' = the whole row or column of tiles
expr       := INT | VAR | expr ('+'|'-'|'*') expr

-- level 2 (SUMMA, block-cyclic LU)
distribute := 'distribute' NAME 'over' 'grid' map ';'    -- owner-computes
map        := 'block2d' | 'cyclic2d'
broadcast  := 'broadcast' tiles 'along' ( 'row' | 'col' ) ';'
```

Comments are `//` to the end of the line. Tensor element types beyond fp32 are deferred.

### 3.3 Meaning at level 1 (what each statement lowers to)

| Statement | IR actions (process) | Checked |
|---|---|---|
| `resident X` | Load X (dma) | X not already resident; L3 capacity at that point |
| `release X` | Release X (l3 credit) | X resident; not written-and-unstored (a written tile is stored by an explicit `store` first: nothing moves implicitly, Q2) |
| `call f(a, b) -> y` | per operand: Move (bm) and Feed (str); then Call (cf); for an in-place result: Drain (str), Writeback (bm) | operands resident; the result resident, or its `acc` open |
| `acc y in fabric { ... }` | y lives in the fabric; `+->` calls into it accumulate from zero; no L3 slot | at least one call into y; y stored after the block |
| `store y` | out of a closed `acc`: Drain, Writeback, Store, Release (an L3 slot for the writeback's moment); a resident, written tile: Store | y written and not yet stored |

The validator walks the program in order, as `csp::lower` does today. It is a static check, so
no simulation is needed to find a capacity overflow, a call on a non-resident tile, or a read
after release.

### 3.4 Tile context: fusion on the move

A **tile context** is the set of transforms bound to a tile as it crosses a boundary. Today
there are two: transpose, which the BlockMover does, and quantize, which the docs place on
moves. The linear driver adds the epilogue. A context names where each stage runs:

| `place` | Hardware | Status today |
|---|---|---|
| `fabric` | the compute fabric, on the accumulator before it leaves | built: `execute_matmul`'s bias + ReLU, at no cost |
| `str.drain` | a vector engine on the streamer's L1 → L2 drain | `VectorEngine` timing model exists, unused |
| `bm.egress` | a vector processor on the BlockMover's L2 → L3 path | new |
| `bm.ingress` | a vector processor on the BlockMover's L3 → L2 path (dequantize, scale) | new |

VE operations start with the epilogue (Q4): `add(vector)` (a broadcast bias), `scale(s)`,
`relu`, `gelu`, `silu`, and **`atan`**. Each stage costs vector-processor time
(elements / lanes x rate). That time overlaps the move it rides on, unless the vector processor
is the slower of the two.

**`atan` is the physical-mapping test case.** It needs SFU support (a transcendental unit),
which the SFU's `ActivationType` does not have today, and which a machine may place at some
sites and not others.
- Each site declares the operations it can run: `movers.vector { lanes, rate, ops: [...] }`
  (Q5).
- The validator checks every context stage against its place on the target machine. `atan @
  bm.egress` on a machine whose BlockMover vector unit lacks it is refused by name, before any
  run.
- The program must then put the stage where the hardware has it (`@ str.drain`, say), or split
  the epilogue across places.

This is the language feature `atan` forces: an operation's placement depends on the machine,
and the language makes that dependency explicit and checkable instead of discovering it at run
time.

**The linear operator, fused at the BlockMover's egress:**

```
csp 1.0
program linear machine flat(l3 = 128) {
  tensor X[256,256] tile 32x32 in;
  tensor W[256,256] tile 32x32 in;
  vector b[256]     tile 32    in;
  tensor Y[256,256] tile 32x32 out;

  for j in 0..8 {
    resident W[:, j], b[j];                       // W's column panel and b's slice stay
    for i in 0..8 {
      resident X[i, :];
      acc Y[i, j] in fabric {
        for k in 0..8 {
          call gemm(X[i, k], W[k, j]) +-> Y[i, j];
        }
      }
      store Y[i, j] via add(b[j]) @ bm.egress, relu @ bm.egress;
      release X[i, :];
    }
    release W[:, j], b[j];
  }
}
```

**The same operator in its other placements:**
- `via add(b[j]) @ fabric, relu @ fabric`: as built today, but now charged its time on the
  compute fabric.
- `@ str.drain`: on the streamer's vector engine.
- **Unfused:** `store Y[i, j]`, then later `resident Y[i, j]`,
  `call bias_act(Y[i, j], b[j]) -> Y[i, j]`, and `store Y[i, j]`. Y makes a second DRAM round
  trip.

**What the placements let the record measure:**
- DRAM bytes: unfused adds 2 x |Y|.
- Compute-fabric busy: `fabric` adds the epilogue's cycles.
- Vector-processor busy and stalls: whether the VE keeps up with the move.
- The makespan.

Values are identical in every placement. Placement changes when things happen, never what is
computed.

### 3.5 Level 2 (the SUMMA form, written)

```
csp 1.0
program summa machine grid(4x4, l3 = 64) {
  // ...declarations as in matmul...
  distribute C over grid block2d;           // owner (r,c) computes C[r*2..r*2+1, c*2..c*2+1]
  for k in 0..8 {
    broadcast A[:, k] along row;            // each grid row receives its A tiles
    broadcast B[k, :] along col;
    // ...each owner's level-1 body over its block...
  }
}
```

Level 2 is designed here so that level 1's syntax leaves room for it. It is built with the
CSP-program plan's step 4.

## 4. Implementation Steps

Each step is one PR and ends green.

1. **The language core** (matmul, LU).
   - `program/csp/lang/`: lexer, parser, AST, validator, lowering to the IR, and a printer
     (IR → `.csp`).
   - Derived L0 is emitted as `.csp` (`emit.hpp`), so a derived program can be read and edited.
   - Tests:
     - parse(print(p)) = p;
     - a hand-written matmul with explicit panels gives L-B values bit-identical to the
       reference, and the Loads it declares;
     - the validator refuses, by line and statement, a call on a non-resident tile, a capacity
       overflow, a read after release, and a store of a never-written tile;
     - hand-written LU matches the reference.
   - **At L-CA:** a hand-written program runs through `CspDriver` with values bit-identical.
   - (Done.)
   - **As built,** `include/sw/kpu/program/csp/lang/`:
     - **`parse.hpp`:** the AST, a lexer (`32x32` shapes, `0..8` ranges, dotted places such as
       `bm.egress`, `//` comments) and a recursive-descent parser. It reports syntax errors on
       their line; a missing `;` is reported on the line it should have ended.
     - **`compile.hpp`:** one walk that validates and lowers to the `CspProgram` IR. Loops are
       unrolled, `:` expands to a slice, and these are checked statically, each by line:
       - a call on a non-resident tile; capacity;
       - resident twice; a release of a non-resident tile; a release of a written, unstored
         tile;
       - a store of a never-written tile; an accumulator never stored; a tile left resident at
         the end;
       - an index out of range; an unknown function or a wrong arity.
       - Level 2, `vector` operands and `via` contexts are refused by name.
     - **Functions:** `gemm` (`+->`, `alpha`), `getrf` / `laswp` (`pivot`), `trsm_ll`,
       `trsm_ur`.
     - **`print.hpp`:** the language's IR back to text, statement for statement (flat; loops are
       not recovered). Compiling the printed text reproduces the actions, and printing that is a
       fixed point.
     - **`emit.hpp`:** derived L0 to `.csp`. It decides residency by Belady at **call
       granularity**: a call's operands are co-resident, as the language requires.
     - **IR change:** each Call carries `accumulate` (the fabric's accumulator starts at zero),
       replacing the behavioral interpreter's whole-program guess.
   - **Deviation from the plan:** `csp::lower` (#341) is *not* printed. It decides at action
     granularity, so it can release one operand between another operand's feed and load, which
     a call cannot express. Derived programs reach the language through the emitter instead.
   - **Tests:**
     - **`test_csp_language`:**
       - written matmul in two residencies: A held whole is 128 loads; A's row panel per column
         panel is 576, exactly as written; both are bit-identical to the reference;
       - written LU is bit-identical;
       - print/compile round trips, and the printed text is a fixed point;
       - emitted matmul and LU at several L3 sizes compile, stay within capacity, and compute
         the reference;
       - 14 validator and syntax refusals, by line.
     - **`test_csp_driver`:** a written matmul runs at L-CA on S1, bit-identical, with exactly
       its 32 loads.
1c. **The structured program executes; the trace is optional** (ADR 0004 §4, decided 2026-10-09).
   - **Why.** L0 is a trace format: a 1M x 1M matmul is about 3 x 10^13 tile ops. Step 1's
     compiler had the same flaw: it unrolled the language's loops into a flat action list, the
     driver enqueued every action, and the validator walked every statement. Before anything is
     built on that representation, the structured program becomes what executes.
   - **1c.1** (done), `lang/walk.hpp` and `lang/validate.hpp`:
     - **The Walker** runs the AST concretely and incrementally: one leaf statement or one block
       boundary per `step()`, emitting to a sink. Its state is bounded by the program, not the
       problem: the open residencies (at most the L3 capacity), the open accumulators, and one
       frame per enclosing block.
       - `compile()` is now the Walker draining into a `TraceSink`: the trace form, with
         identical actions, values and refusals.
       - `ActionStream` pulls actions one at a time and owns its program. Its buffer never
         holds more than one statement's actions.
     - **The symbolic validator** checks the same rules over the structure, nothing unrolled:
       - index expressions must be affine, and their ranges come from interval arithmetic over
         the loop bounds;
       - tile references are *families* (an operand plus `:` or an affine form per dimension);
         "already resident" is a possible overlap, "operand resident" is coverage, and `release`
         and `store` must mirror a resident family;
       - each loop body is checked once and must be **residency-balanced**: an iteration
         releases what it makes resident and stores what it accumulates;
       - capacity is exact per statement, from family sizes;
       - totals (loads, calls, stores) are exact for rectangular loop nests.
     - **Limit:** cross-iteration residency (double buffering: loading iteration k+1's tile
       before releasing k's) is refused by name. Such a program still runs through the walker.
       Proving it symbolically is a later step.
     - The behavioral interpreter has `begin` / `step` / `finish`, so it runs from a stream.
   - **Results:**
     - **Stream = trace,** action for action, for matmul in two forms and for LU. L-B over the
       stream computes what L-B over the trace does, which is bit-identical to the reference.
     - **Symbolic validation agrees with the trace** on peak L3, loads, calls and stores.
     - **A 1M x 1M matmul** (32 x 32 tiles, one A and one B tile resident at a time):
       - validated in about 0.05 ms;
       - peak L3 is 2;
       - 70,368,744,177,664 loads, 35,184,372,088,832 calls and 1,073,741,824 stores, exact;
       - its stream yields its first million actions with a bounded buffer;
       - nothing is unrolled, and no operand is allocated.
   - **1c.2 (done): L-CA from the stream.**
     - The driver used to seed each L3 entry with its residency's consumer count at the Load,
       which needs lookahead the stream does not have. The executor now takes the program's
       Release as an action (`schedule_release`): the BlockMover retires the entry after the
       tile's Moves issued before it. Per-tile epochs order a tile's Moves and Releases, so a
       Move issued after a Release takes the next residency's copy, never the old one.
       `TileDescriptor::l3_consumers` is replaced by `l3_held`.
     - `CspDriver` takes the trace or the stream (`ActionStream` plus input values). It hands
       actions over while the executor's backlog (`backlog()`) and its own pending Loads are
       under a window, a schedule parameter (default 256; 0 = the whole program at once).
     - Windowed issue exposed two executor assumptions that the whole program was scheduled up
       front; both are fixed.
       - An accumulating call published its result to DRAIN when it was the last call
         *scheduled* on the tile. With a window, the chain's first call can look like its last,
         and the DRAIN took a partial sum: every C tile was wrong at windows 2 to 16. A later
         call on the chain now retracts that publication, unless an earlier chain's DRAIN is
         still waiting for it.
       - The fill model charged the drain half of the fill to that same "last" call. It now also
         requires the tile's DRAIN to be scheduled.
     - **Results** (written matmul, 128^3 in 32 x 32 tiles, S1):
       - The trace path's cycles are unchanged (43,968 and 38,089 for the step-2 cases), so
         Release-based retirement times exactly as the consumer counts did.
       - The stream at windows 1, 2, 4, 16, 64 and 256 computes the reference bit for bit, with
         its 32 loads.
       - The driver holds at most the window plus one action.
       - Cycles: 24,588 / 19,675 / 18,506 / 16,008 / 12,642 / 9,985. Window 256 matches the trace
         exactly. The window bounds how far the DMA can run ahead of compute, which makes it the
         program's prefetch depth, and a schedule parameter, not a machine one.
     - **Remaining O(trace) state:** the executor's event log (and its per-tile counters, which
       are O(tiles)). L-CA of a very large program is bounded by simulated cycles first; a
       streaming event sink is a later step if it is needed.
2. **The linear operator at L-B** (done).
   - **L0, opset 1.1.0:** `BiasAdd` (a vector tile broadcast down the rows) and `Activation`
     (`act=` relu, gelu, silu, atan), with reference kernels. The element functions are shared
     by the L0 kernels and by every context stage, so fused and unfused compute the same bits.
     The transcendental forms evaluate in double and round once.
   - **Derivation:** `--algo linear` (`derive_linear_tile_program`, `ProgramSpec::act`, kpu-run
     `--act`): matmul with Feed b, BiasAdd, Activation before each Drain. It lowers and runs at
     L-B, and kpu-run's block-sequential level (L-T1) is bit-identical to it.
   - **The language:**
     - `vector` operands, indexed `b[j]`;
     - the unfused epilogue as in-place calls: `add(y, b) -> y`, `relu(y) -> y`, and so on;
     - tile contexts: `store y via op @ place, ...`, and on an in-place call's result. A stage
       becomes part of the result's Drain (`fabric`, `str.drain`) or Writeback (`bm.egress`).
       The IR's `Action::context` carries it, and the behavioral interpreter applies it to the
       tile it moves.
     - The walker and the symbolic validator check the same rules: known operations and
       places, the stages in path order, add's vector resident and tiled as the result's
       columns, an accumulator takes gemm only, and a resident tile's store (a DMA write) has
       no vector unit.
   - **Placement:** the spec gains `movers.vector { bm, str }`, each `{ lanes, rate, ops }`, and
     `lang::target_from(device)` lists each site's operations. Given a target, `compile`,
     `validate` and `ActionStream` refuse a stage where its site cannot run it. For example:
     "atan @ bm.egress: the target's BlockMover vector unit runs add, relu, gelu, silu, not
     atan".
   - **Decided in the build, for review:**
     - **The fabric's operations are fixed at {add, relu}:** what `execute_matmul` builds today
       (bias + ReLU). It is not a spec field until a fabric with more exists.
     - **`bm.ingress` is refused for a result.** It is an operand's way in, and operand contexts
       (a fused prologue: dequantize, scale) need syntax on call arguments; that is a later
       step. The "four placements" tested are `fabric`, `str.drain`, `bm.egress`, and
       unfused.
     - **The L0 → `.csp` emitter refuses the epilogue ops.** L0 does not say where the epilogue
       runs, and the emitter has no basis to choose. DFP → `.csp` should arrive with its
       placement (ADR 0004 §4).
     - **A stage's operand (add's bias) is read from L3 at L-B.** Its delivery to the site, and
       the `lanes` and `rate` time, are step 3's.
   - **Tests** (`test_csp_linear`, 8 cases):
     - the element functions against independent values;
     - L0 against a double-precision direct computation (ADR rtol);
     - the ops serialize and refuse a missing or unknown `act`;
     - the lowered L0 at L-B;
     - 12 placements (each activation at `str.drain` and `bm.egress`, relu in the fabric, two
       split placements) and unfused, bit-identical to the reference through the trace and the
       stream;
     - symbolic totals equal traced, and the printer round-trips contexts;
     - placement against the target;
     - `movers.vector` JSON round-trip and refusal;
     - refusals by line.
3. **The vector processor at L-CA.**
   - A VE stage on the BlockMover's ingress and egress, and on the streamer's drain, timed by
     a new spec field (§7 Q5). `fabric` charges the compute tile.
   - `CspDriver` carries contexts on the descriptors.
   - Measured on S1 for each placement: DRAM bytes, compute-fabric busy, VE busy and stalls,
     and the makespan. This is the fusion benefit the driver exists to show.

## 5. Verification

- **Round trip:** every program the language compiles prints and parses back to the same IR
  (`compile(print(p))` reproduces `p`'s actions, and the printed text is a fixed point). That
  includes sequential, later-stored and interleaved accumulators. Derived L0 enters through the
  emitter (`emit.hpp`), whose output compiles. `csp::lower`'s action-granular IR is not
  printable, because a call's operands must be co-resident (step 1's deviation note).
- **Values:** bit-identical to the L0 reference at L-B and L-CA for matmul, LU and linear in
  every placement; gelu and silu within tolerance.
- **The validator** finds each class of error statically and names its line.
- **Fusion, measured:**
  - unfused moves 2 x |Y| more DRAM bytes than any fused placement;
  - `fabric` adds the epilogue to compute-fabric busy;
  - `bm.egress` and `str.drain` add none, as long as the VE keeps up.

## 6. Key Invariants

- **The written program is the sequencing.** The lowering decomposes statements into actions;
  it never reorders or adds a Load or a Release.
- **Residency is declared and checked statically.** No run-time hit creates reuse a program did
  not write.
- **Contexts change when, never what.** Every placement of a fused epilogue computes the same
  values.
- **Machine parameters stay in the spec** (L3 capacity, VE lanes). Schedule choices stay in the
  program: order, residency, placement of a context.

## 7. Decisions and Questions

**Decided with the request (2026-10-09):**

| | Decision |
|---|---|
| A1 | Structured (loops, scoped residency, calls), not a flat action list |
| A2 | Processes implicit at level 1; placement explicit at level 2 |
| A3 | KPUASM is kept as the engine level; reuse to be evaluated later |
| + | Drivers: matmul, LU, and linear (matmul + bias + activation), the last for fusion and vector processors on block-move ingress and egress as part of a tile context |

**Answered (2026-10-09):**

| Q | Decision |
|---|---|
| Q1 | C-style `{ }` blocks, `;`-terminated statements, `//` comments |
| Q2 | Residency explicit only: a call on a non-resident tile is an error. Residency is a property of the program, not of an execution order, which is the failure of building high-performance data movement in the stored-program and SIMT abstractions. |
| Q3 | `.csp`, with a version identifier on the first line (`csp 1.0`) |
| Q4 | relu, gelu, silu, **and atan**. atan needs SFU support with a physical mapping dependency (a site may or may not have it), so it is the test case for placement checked against the machine (§3.4). |
| Q5 | `movers.vector { lanes, rate, ops }` per BlockMover and streamer site; `ops` lists what the site can run |
| Q6 | A bias is a `vector` operand made resident in L3, one slot for its panel, delivered with the move it rides on |

**Questions (as posed):**
- **Q1. Blocks.** Recommended: `end`-terminated keywords. They are unambiguous to generate
  (for an AI assistant especially) and survive reformatting. The alternatives are indentation
  (fragile to generate) and braces.
- **Q2. Residency: explicit only, or implicit transient loads as well?** Recommended: explicit
  only. A call on a non-resident tile is an error, because a program that hides its loads hides
  its reuse. The derived programs make every Load explicit, so they lose nothing.
- **Q3. Name and extension.** Recommended: "KPU CSP language", `.csp`, versioned by its first
  line (`csp 1.0`).
- **Q4. The activation set first.** Recommended: relu, gelu and silu. These are the SFU's
  `ActivationType` subset that the models in M2-M4 use. Sigmoid, tanh and leaky-relu follow on
  demand.
- **Q5. The vector processor as a machine parameter.** Recommended: a spec field per site,
  `movers.vector { lanes, rate }`, on BlockMovers and on streamers. A machine without it makes
  `bm.egress` and `bm.ingress` contexts invalid, refused by the validator. It is a machine
  parameter, so it belongs in the spec.
- **Q6. The bias tile.** Recommended: a `vector` operand made resident in L3 like any tile, and
  delivered to the vector processor with the move it rides on. It is one L3 slot for the whole
  panel (consumer count = its uses), the broadcast discipline that `emit_broadcast_tile` already
  uses.
