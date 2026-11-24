# Slot credits at L-CA: every L2 copy placed, every slot a state (#401, slice 2)

**Status:** done 2026-11-24 (#387 closed; PRs #496, #523–#527). The user made the decisions in §2.
One item is split out as #528: slot-keyed placed L2 entries at L-CA, so a tile can hold two copies at once.
**Issues:** #387 (mirror the RTL's tile credits and tile-ready tokens), under #401.
**RTL reference:** branes-ai/embodied-ai-soc-generator#16 and its `tests/unit/test_credits.py`,
`src/kpu_gen/tiles/systolic/controller/sync.py` (`SlotPool`), `isa/command.py` (`TileBuffers`).
**Follows:** `docs/plans/l3-placement-at-lb.md` (placement is behavior), whose step order this
plan reuses for L2; `docs/plans/banked-l2-at-lca.md` (slice 1).

## 1. The gap

The RTL orders its processes with whole-tile slot credits. Each L2 slot (and each Combiner slot,
which is an L2 slot in kpu-sim) is always in exactly one state:

| State | Entered by |
|---|---|
| FREE | a RELEASE |
| OWNED | the producer's ACQUIRE, at its start (being filled) |
| READY | the producer's SIGNAL, at its completion (the tile-ready token) |
| READING | the consumer taking the token, at its start |

A transition from the wrong state is a sticky protocol error. The Matmul and Conv2d instructions
use the slots in **ring order** (`TileBuffers.next_l2`, `next_combiner`), which is what makes two
buffers double-buffer.

L-CA has most of the protocol under other names: ACQUIRE as `CREDIT_ACQUIRED` at a Move's start,
SIGNAL as the tag-CAM insert at its completion, WAIT as the Streamer's match, RELEASE as
`CREDIT_RELEASED`. What it lacks:

- **Per-slot state.** L2 credits are a count (`CreditPool`); no slot has a state at a cycle, so no
  conservation invariant can be checked per cycle, and there is no READING.
- **Faults.** A wrong-state transition stalls rather than raising an error.
- **Placement for every copy.** Only staged copies and L2 partials have a slot the program
  placed. An unstaged operand copy takes a transient slot (`TagCAM::allocate_l2_slot`), and a
  ref-counted dedup can complete a Move onto a copy already in L2: a reuse the program did not
  state, against `reuse is deliberate`.
- **The result slot's timing.** The RTL takes a C tile's Combiner slot at its first COMPUTE;
  L-CA takes it at the drain.

## 2. Decisions (the user's, 2026-11-18)

1. **A result slot is taken at the C tile's first compute**, as the RTL does. A result always has
   a home before any compute.
2. **A and B keep separate credits.** One slot and credit per tile, as the program states them.
   The RTL's output-stationary pair (B riding on A's credit) is an encoding of the same
   dependency; the tile-command lowering (`csp/tile_commands.hpp`) already emits it.
3. **The walker places every L2 copy** (placement is behavior), and **the dedup is removed**
   (reuse is deliberate).
4. **`l2 = N` is required** of any program that moves a tile into L2, which is every program. The
   validator refuses one without it, by name. `kpu-csp-gen` always writes it, from a new
   `--l2-buffers 1|2`, and the corpus is regenerated.
5. **Acceptance for #387:** the slot-state invariants and the RTL's own double-buffering
   property (§6). The exact cycle match (907 / 744) is #401's done-when, after #389, #385 and
   #382.

## 3. Design

### 3.1 Ring placement

The walker allocates L2 slots **round-robin**: each new copy takes the next slot after the last
one allocated, skipping live slots. Lowest-free allocation, today's policy for staged copies,
cannot double-buffer. L-B runs in order, so a copy's Feed frees its slot before the next Move,
and every call would reuse the same slots; L-CA would then hold each Move for the previous
Feed's release however many slots the program declares. The ring is the RTL's `TileBuffers`
order.

### 3.2 Every copy has a slot

| Copy | Its slot's life |
|---|---|
| **An unstaged operand** (a call's Move and Feed) | A one-Feed staging: the Move writes a ring slot, the Feed reads it, and an L2Release frees it. That is the lifecycle every consumer already understands (the record, its checkers, the viewers, the trace's replay). The RTL's COMPUTE folds the release into its RELEASE bit; the tile-command lowering may fold it the same way. |
| **A staged copy** (`stage X in l2`) | Unchanged: Move writes, Feeds read and keep, L2Release frees. |
| **An L2 partial** (`acc Y in l2`) | Unchanged: reserved at the block's open, merged by each Drain, written back, then L2Release. |
| **A fabric accumulator's result** (`acc Y in fabric`) | **New:** a ring slot reserved at the C tile's first call (decision 1); the Drain writes it, the Writeback reads it, an L2Release frees it. |

### 3.3 The machine's L2 slots

`l2 = N` is each compute tile's L2 slots, shared by every copy above. `kpu-csp-gen --l2-buffers B`
sets N to what B-deep buffering needs: B × (the operand copies a call reads) plus B × (the result
tiles in flight). For an output-stationary matmul that is 2B + B = 3B; for a staged schedule, its
staged panel plus the same. The generator computes it per schedule and validates it.

### 3.4 L-T1 and L-CA execute the same slots

- **L-T1** already times staged copies under their own slot's credit. It extends that to every
  copy, and takes a result slot at its tile's first call.
- **L-CA** takes every L2 slot from the program (`set_placed_slots` covers all N), loses
  `allocate_l2_slot`'s transient slots and the dedup, and takes a result slot at its first call.
  A new **`SlotPool`** keeps each L2 slot's state (FREE, OWNED, READY, READING) and turns a
  transition from the wrong state into an error, by name. Its states are what the invariants read.

## 4. Steps

Each step is a PR, under #387, with no closing keyword until the last.

1. **The language and L-B** (PR 1):
   - `l2 = N` required, refused by name when missing; the language becomes **csp 1.1**, and a
     csp 1.0 program is refused by its version, with the reason (decided 2026-11-18);
   - ring allocation;
   - an unstaged copy placed, a one-Feed staging (Move, Feed, L2Release);
   - a fabric result reserved at its first call, and recorded as a partial with one merge;
   - L-B executes all of it as addressed storage: it refuses a write into a live slot and a read
     of the wrong one, as for L3;
   - the `.dflow` and L0 records carry the slot of every L2 event;
   - `kpu-csp-gen --l2-buffers`;
   - the corpus and fixtures regenerated.

   **The timed levels run the copies unplaced in this step.** L-T1 and L-CA already execute an
   action that carries an L2 slot as a staging, and L-CA's L2 is keyed by tile, so it cannot hold
   two copies of one tile in two slots, which ring placement produces. So until steps 2 and 3
   they see each copy without its slot (`csp::unplaced`, by the action's `copy` flag), as they
   ran it before, and skip its L2Release. The timed records do the same (step flag `kCopy`). Their
   timing is unchanged by this step; only L-B, the trace and the L-B record place the copies.
2. **L-T1** (PR 2): every copy under its own slot's credit; a result slot at its first call. With
   it, the test deferred from #476: a Move whose ready a credit set while a lane kept `l2_wait`
   at 0, asserting both. (Done, 2026-11-23.)
   - The call that opens a result's slot carries it, as `Action::result_l2`. L-T1 takes the
     credit there.
   - On more than one compute tile L-T1 runs the copies unplaced and names it in `unmodelled`:
     a level-1 program places every copy in compute tile 0's L2 (multi-tile placement is #403).
   - The records keep showing stagings only until step 3 places the copies at L-CA too. Then both
     levels' records describe the same L2.
3. **L-CA**, in three PRs (decided 2026-11-23):
   - **3a (done, 2026-11-23):** every slot placed, through L-CA's staging path, with a result's slot
     claimed at its opening call's compute. A placed transfer's credit is its slot. The dedup is off
     for CSP runs, and the legacy schedules keep it until #266. The BlockMover runs a program's
     legs in program order. A tile keeps one copy in L2 at a time: a Move waits for its earlier
     copy's L2Release, so the tile-keyed TagCAM stays valid. The records still show stagings only.
   - **3b (the pool done, 2026-11-23):** the `SlotPool` (FREE, OWNED, READY, READING) with faults and
     the per-cycle invariants (§6), driven by the L2 tag CAM, every slot FREE at the end. **The
     records (done, 2026-11-24):**
     - the `.tflow` L2 table carries the copies, with a `copy` kind (operand, result);
     - TF13 requires an operand copy to be fed once and a result never;
     - the `.dflow` timed records place them where the level did.

     Split out as #528: slot-keyed placed entries, so a tile can hold two copies at once.
   - **3c (done, 2026-11-24):** the double-buffering test (§6) on `kpu_soc_2x2`, the RTL-derived
     4×4 tile with the RTL's slot counts (Q2): `tests/program/test_csp_double_buffer.cpp`. As it carries
     over (decided 2026-11-24):
     - values are bit-identical, and L-B's, with one buffer and two;
     - two buffers cut the L2 slot waits (348 → 25 cycles), overlap more Moves with Feeds, and run
       faster;
     - no Move lands in a slot a Feed is reading.

     Two of the RTL's assertions don't carry over yet. "One buffer never overlaps a Move with a
     Feed" doesn't hold because A and B keep separate credits (decision 2). "Under 0.85× the cycles"
     doesn't hold because the run is bound by the BlockMover and DMA (0.96×). The 0.85× clause moves
     to #401's acceptance, after #415 and the RTL's mover rates.

## 5. What changes for existing programs

- Every program gains `l2 = N`; the generator writes it, and hand-written test programs are
  updated by hand.
- Timing moves where the dedup hid bubbles, and where transient slots let a Move run ahead of the
  ring. Each re-baselined number is explained in its PR, as #471 did for the banks.
- Values must not move: every level stays within L-B's bound on every program.

## 6. Verification

- **Credit conservation, every cycle:** each L2 slot in exactly one state, so free plus held equals
  N; every slot FREE at the end of a run.
- **No tile consumed before its token:** no Feed starts on a slot that is not READY; no Writeback
  before its result's Drain completes.
- **No credit taken twice:** no ACQUIRE of a slot that is not FREE; no SIGNAL or RELEASE from the
  wrong state. A run that tries one fails with the slot and the transition named.
- **Double buffering:** an 8×8×16 matmul on a 4×4 array, with `--l2-buffers 1` and `2`. As the
  RTL's test asserts: the two-buffer run overlaps a Move with a Feed, the one-buffer run never
  does, and two buffers take under 0.85× the cycles of one. Values are bit-identical between them.
- **L3 backpressure does not change values:** the same program with L3 credits scarce.
- Every level within L-B's bound, on every program and fixture; both full suites, under gcc and
  clang.

## 7. Questions for review

- **Q1.** Conv2d's multi-command tiles (an X tile filled by one LOAD per output position, the
  credit taken by the first and the token emitted by the last) need a Move that is several
  commands. That belongs with #382's strided moves, not here.
- **Q2.** A 4×4 deployment with the RTL's slot counts (4 L2 slots and 2 result slots, so `l2 = 6`)
  is added for §6's double-buffering test. Its other rates follow the RTL's `config()`.
