# Accumulator Init Stream: Bias, Accumulate and Clear (#406)

**Status:** step 1 done (language, L-B, L-T1, generator); step 2 done (L-CA feeds the init stream); steps 3-4 open
**Issue:** #406, which mirrors branes-ai/embodied-ai-soc-generator#17 (merged there in #47)
**Related:** #384 (Matmul/Conv2d end to end), #387 (credits), #394 (int8), #395 (WS/IS), #400 (conv2d)

## 1. The problem

The RTL can start an output tile's accumulation from something other than zero:
- a bias matrix D, `C = A·B + D`;
- a bias row b broadcast down the rows, `C = A·B + b`;
- the existing output, `C += A·B`.

It writes padding with `CLEAR`, which reads nothing from L3.

kpu-sim's accumulator (`acc C[i, j] in fabric { ... }`) always starts at zero. A bias exists only
as an epilogue, applied on the way out (`store C via add(b) @ str.drain`). Nothing starts an
accumulator from a value.

## 2. The model: one more edge stream

The compute fabric has nearest-neighbour links only, and everything enters it from its edges
(ADR 0002 §3.3: the fabric talks only to L1). An accumulator's initial value is no exception. It
is **another input stream of the accumulator's first call**, carried to each PE by a propagation
recurrence of its own, `c(i, j, 0) = d(i, j)`, as A and B are carried by theirs. A bias row is the
cheap case: `b[j]` enters once at the edge and propagates down its column.

So an init is an ordinary delivery: **Move** (BlockMover, L3 → L2), then **Feed** (Streamer,
L2 → L1), into the fabric, where its values become the accumulator's. There is no write path into
the accumulators, and no new action kind.

`CLEAR` is the default: an accumulator with no init starts at zero and streams nothing in, as
today.

## 3. The language

```
acc C[i, j] in fabric init D[i, j] { ... }   // a bias matrix: D tiled as C
acc C[i, j] in fabric init b[j] { ... }      // a bias row: a vector tiled as C's columns
acc C[i, j] in fabric init C[i, j] { ... }   // accumulate onto C: C inout and resident
```

The rules (the walker's, and the symbolic validator's):
- The init tile is **resident** when the block opens. It is delivered then (Move, Feed), before
  the block's first call.
- A tensor init is tiled as the accumulator: the same shape and the same tiles.
- A vector init is tiled as the accumulator's columns, the rule `add` already has.
- **Self-init** (`init C[i, j]` on `acc C[i, j]`): the accumulator takes the tile over. Its Move and
  Feed are followed by the residency's **Release**, because the value now lives in the fabric. The
  tile must not be dirty or staged.
- A staged init tile is fed from its L2 copy, like any staged operand.

## 4. The IR, the L0 trace and the values

- **`Action::seeds`** is set on the init's Feed and names the accumulator it starts. Every level
  reads it from the Feed.
- **L0, opset 1.4:** a new op, `ACC_INIT`. Its output is the accumulator tile; its input is the
  init tile, or nothing for a clear. A tensor input is copied; a vector input is broadcast down
  the rows. The walker records it with the init's Feed, which is tagged with the input port
  `South`, the edge the C stream's drain uses. The trace's L0 then carries what the program
  computes, and `TileProgramReference` stays its oracle. An accumulator without an init records
  no `ACC_INIT`, so existing traces and the golden corpus don't change.
- **L-B:** the seeded Feed puts the init's values, broadcast for a vector, into the fabric as the
  accumulator's. The first `+->` call finds it there and accumulates onto it.
- **Value order:** `D + Σₖ A·B`, accumulated in the program's k order. That is the RTL's order, not
  the epilogue's `Σₖ A·B + b`, so in fp32 an init bias and an epilogue bias are not bit-identical
  (int32, after #394, is).

## 5. Timing

- **L-T1:** the init's Move and Feed are timed as any operand's, on the movers' and Streamer's
  lanes. The accumulator's first call waits for the init's Feed, as it waits for A's and B's, so
  the init stream overlaps the first k step's operand streams instead of preceding the chain.
  The chain (fill, steady state, drain) is unchanged.
- **L-CA** (step 2): the init's Load, Move and Feed run on the executor's DMA, BlockMover and
  Streamer as any operand's do. The init's Feed carries `seeds`, and the driver names the fed tile
  as the accumulator's first call's `MatMulComputeSpec::init`. That call waits for the feed and
  starts from its values (a bias row broadcast down the rows) instead of zero. The init is not a
  k-slice: it adds no MACs and no fill. A bias matrix other than A, B and C is tagged as a vector
  operand is (MatrixID::A, tk = 1 + its index); a matrix the program computes on is still refused.
  An inout C enters DRAM from the inputs at its first Load.

## 6. Steps

1. **This PR:**
   - the `init` clause in the parser, walker, validator, formatter and printer;
   - `Action::seeds` and L0 `ACC_INIT` (opset 1.4);
   - L-B values and L-T1 timing;
   - L-CA refuses programs with an init;
   - `kpu-csp-gen --init bias|bias-row|accumulate`;
   - tests:
     - values against numpy for every form;
     - the trace replays to the same values;
     - the printer round-trips;
     - the refusals;
     - L-T1's first call waits for the init's Feed;
     - the bias is read once per output tile.
2. **L-CA (done):** the Streamer feeds the init stream and the fabric's first call starts from it.
   - tests: values for every form and both loop orders against the reference (and L-B's, bit for
     bit, through `kpu-run`); each first call starts after its init's feed has finished; a
     matrix computed on is still refused.
   - not here: the accumulator's slot credit. L-CA's executor has no accumulator slots yet; their
     credits and tokens (ACQUIRE on a C tile's first K step, SIGNAL on its last) are #387's.
3. **WS and IS** (#395), where the accumulator sits at the array's edge, and **conv2d**: the
   per-channel bias broadcast and `CLEAR` for padding (#400).
4. **int8 → int32** (#394): bit equality with the RTL's bias and accumulate.

## 7. What #406's checklist maps to

| #406 item | Step |
|---|---|
| Matmul with a bias matrix / bias row | 1 (OS); 3 (WS, IS) |
| Matmul accumulating onto C | 1 |
| Conv2d with bias and accumulate, padding | 3 |
| A padded position reads no L3 | 3 (`CLEAR`, no stream) |
| The bias is read once per output tile | 1 (L-T1: one Move per init) |
| Credits with two producers | #387 (the executor's accumulator slots) |
| The Streamer never waits on the accumulator write port | 2: there is no shared write port in this model; the init is a stream |
