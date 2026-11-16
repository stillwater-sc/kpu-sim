# The Combiner

The **Combiner** is the KPU tile's unit on the result path, between L1 and L2. It is a new
concept in the architecture, added on 2026-11-14. kpu-sim defines it and demonstrates that the
machine works with it; the KPU RTL generator then implements what kpu-sim has shown
(branes-ai/embodied-ai-soc-generator#68 replaces the RTL's accumulator with it).

**The Combiner is kpu-sim's Vector Engine, renamed and extended.** The Vector Engine already
sits on this path (it processes results during the L1 → L2 transfer) and fuses bias and
activation for output-stationary schedules. What it lacked is the merge of partials that the
weight- and input-stationary schedules need, which the TPU line of designs keeps in a
separate accumulator memory. That merge is cheap on the same path, so the Combiner takes it
on and the accumulator goes away. The rename and the new merge are tracked in #429; the rest
of the modelling in #427. systars does not have it yet either.

```text
fabric → L1 (result side) → Combiner → L2 → BlockMover → L3 → memory controller DMA → DRAM
```

## What it does

The Combiner is a **fusing, scaling, compressing and accumulating unit**. It is not a
result memory. For each result vector that leaves the fabric through L1, it can:

- **Merge intermediate partials.** The weight- and input-stationary schedules reduce across
  K blocks. The Combiner reads the partial from L2, merges the new vector into it, and
  writes it back.
- **Fuse the operators applied as results are written to memory:**
  - bias;
  - activation functions;
  - scaling up or down (requantization, with rounding and saturation);
  - block-format conversion (compression). The first block formats are the OCP Microscaling
    (MX) formats, an open specification: blocks of 32 elements sharing one 8-bit
    power-of-two scale (MXFP8, MXFP6, MXFP4, MXINT8). MXINT8 comes first, then MXFP8;
    MXFP6 and MXFP4 later.

  The activation functions run on the **SFU banks**, which are part of the Combiner: they
  are how the Vector Engine fused activations, and they come with it.
- **Write the result into L2.** If a result must spill, the BlockMover moves it through L3 to
  the memory controllers' DMA and DRAM.

Its pipeline keeps up with the fabric's result rate, one vector per array clock, so the
fused operations cost no extra pass over memory.

## What the KPU tile does not have

- **No separate accumulator.** Partials and results live in L2 slots, under L2's credits.
  Merging partials is one of the Combiner's jobs.
- **No DMA in the tile.** The tile's data movement is its CSPs, the BlockMover and the
  Streamer. Direct memory access belongs to the memory control layer, one DMA engine per
  memory controller, as in the data path of [KPU architecture](kpu-architecture.md) §1.3.

## How it relates to what kpu-sim has

| kpu-sim today | In the Combiner model |
|---|---|
| Push-based fusion, [KPU architecture](kpu-architecture.md) §2.7: fused operations inserted on the egress path, with SFU banks alongside it | The Combiner is the named unit at that insertion point, and the SFU banks are part of it. |
| The Vector Engine, in two places. The ISA path: `VectorEngine` and `BehavioralVectorEngine`, the `VE_*` instructions, `VE_ENABLE` on `STR_DRAIN_OUTPUT` and its fused drain (`docs/06-compiler/schedule-dsl.md`). The CSP levels: the place `@ str.drain`, timed on each streamer's drain from the deployment's `movers.vector.str` | **The Combiner**, renamed (#429). On the ISA path: `Combiner`, `BehavioralCombiner` and the `CMB_` prefix (`CMB_ELEMENTWISE`, `CMB_REDUCE`, `CMB_ENABLE`). In CSP programs: the place `@ combiner`, from the deployment's `combiner` block. The old spellings are accepted as deprecated aliases for one release. It keeps its operations, the softmax's elementwise ops and reductions among them, and gains the merge of partials, scaling and block-format conversion. |
| The fused epilogue: `MatMulComputeSpec { bias, activation }` applied in `execute_matmul` (`docs/plans/e10_fused_epilogue_pattern.md`) | The Combiner's bias and activation, applied as the result is written to L2. |
| The accumulator as its own banked memory with slot credits: planned for L-CA (#401, #383, #387), not built. Today only the tile command lowering (#397) names accumulator slots | Gone. Partials and results occupy L2 slots, and the Combiner's merges are L2 traffic. |
| The `systolic` block of a deployment: accumulator slots, banks, rows, latency (#396) | The Combiner's configuration. The accumulator fields retire from the model with #427; the RTL's rename of its blocks is branes-ai/embodied-ai-soc-generator#67. |
| `kpu-tile-cmds` (#397): a LOAD into the accumulator for a bias, COMPUTE accumulate or overwrite | A COMPUTE names its Combiner operation: merge or overwrite, bias, scale, activation, format. |

## The merge of partials, as built

The merge is in the CSP language and at every level (#429's merge, #430's programs):

- **The program states it.** `acc C[:, j] in l2 { ... }` is an L2 accumulator: each tile of C
  holds its partial in an L2 slot, from the machine clause's `l2 = N`. Each call `+->` into one
  of its tiles runs the gemm on a fresh accumulator in the fabric, and its Drain carries the
  Combiner's merge into the slot: **overwrite** for the tile's first K block, **accumulate** for
  the rest (a read-modify-write in L2). The store writes the slot back through L3 to DRAM.
- **The trace records it** as `MATMUL_MERGE` (L0 opset 1.5), with the merge mode and the
  partial's L2 slot. The replay checks the slot, as it checks a staging's.
- **The levels execute it.** L-B merges in the tile's number system, so int8 × int8 summed in
  int32 merges exactly. L-T1 holds the slot from the first merge's credit to its L2Release,
  serializes the read-modify-writes on it, and counts the merges' L2 bytes. L-CA takes the slot
  at its turn with an L2 credit, times each merge as one add on the Combiner
  (`combiner_lanes` × `combiner_rate`), and retires the slot after the store's Writeback has
  read it. A machine with no `combiner` block cannot run an L2 accumulator.
- **The programs:** `kpu-csp-gen --algo matmul --dataflow ws|is` writes the weight- and
  input-stationary schedules (#430). The stationary tile is staged in L2 across the loop that
  reuses it, and C's panel of partials is the L2 accumulator.

- **The epilogue on the final merge** (#427). The program states which merge is the final one:
  a call into an L2 accumulator that carries `via ... @ combiner` stages is the tile's last.
  The Combiner applies those stages to the merged value as it lands in the slot, and a later
  call into the tile is refused. `kpu-csp-gen --algo linear --dataflow ws|is --place combiner`
  writes the k loop's last block out on its own, so its calls carry the bias and activation.
  Under ws, `--place bm.egress` puts the epilogue on the panel's store instead. A fabric stage
  is refused, because it would act on one partial.

- **Block-format conversion** (#427): `mxint8 @ combiner` and `mxfp8 @ combiner` (or
  `@ bm.egress`) convert a result into an OCP Microscaling format as the last of its fused
  operations, after bias and activation (the Combiner's order: bias, scale, activation, format). Each run of 32 row-major elements
  shares an e8m0 scale.
  - Both are converted per OCP MX v1.0 in `src/program/numeric.cpp`. MXFP8's elements are
    Universal's e4m3 (saturating), and MXINT8's are int8 codes with an implicit 2^-6, ties to
    even. The block scaling (an `ilogb` exponent and an e8m0-clamped scale, applied with
    `ldexp`) is kpu-sim's until Universal's `mxblock` is fixed (stillwater-sc/universal#2083:
    v5.17.0's `mxint8` saturates each block's largest element, and its `quantize` overflows
    below a 2^-127 scale).
  - The trace records `BLOCK_CONVERT` (L0 opset 1.6), and every level computes the same bits.
  - `kpu-csp-gen --algo linear --format mxint8|mxfp8` appends it to the epilogue.
  - The result stays in fp32 words holding the values the format represents. Storing the
    block format itself (element bytes and scales) is not modelled yet.

- **Requantization** (#427): `scale(m, shift) @ combiner` takes an integer accumulator into its
  tensor's integer type, as `round(x * m / 2^shift)`: ties to even, saturated
  (`numeric::requantize`). The result tensor declares both: `tensor C[..] int8 tile .. out acc
  int32`. Its gemm accumulates in int32, and the scale is on every path into memory: a
  fabric accumulator's store, or an L2 accumulator's final merge. It is the context's only
  stage, so no int8 bias or activation yet, and it runs on the Combiner, after a bias and before
  an activation or format (the Combiner's order). The trace records `REQUANT` (opset 1.7), and an
  operand's `acc=` (container 1.4). `kpu-csp-gen --algo matmul --dtype int8 --requant m,shift`
  writes it under OS, WS and IS.
  - **The rounding mode is part of the architecture: ties to even, saturated.** It matches the
    block formats' rounding and is unbiased. kpu-sim fixes it, and the RTL implements it
    (esg#68 asks for "a scale and shift ... with rounding and saturation" and leaves the mode
    to this specification).
  - **A per-channel scale** (a vector of multipliers) is not built yet.

Not built yet:
- an integer bias or activation beside the scale, and a per-channel scale;
- an init (`acc ... in l2 init ...`);
- a weight-stationary timing of the array. L-T1 and L-CA time each merge's gemm as a
  one-call output-stationary chain.

These are #427's.

The work to model the rest of it, at L-B, L-T1 and L-CA, in the shared records and in the
contracts with the RTL, is tracked in #427.
