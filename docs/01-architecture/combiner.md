# The Combiner

The **Combiner** is the KPU tile's unit on the result path, between L1 and L2. It is a new
concept, added on 2026-11-14 for the KPU RTL generator (branes-ai/embodied-ai-soc-generator#68).

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
| The Vector Engine (`VectorEngine` at L-T1/L-CA, `BehavioralVectorEngine` at L-B, the `VE_*` instructions and `VE_ENABLE` on `STR_DRAIN_OUTPUT`) and its fused drain (`docs/06-compiler/schedule-dsl.md`) | **The Combiner**, renamed (#429), with the `CMB_` instruction prefix (`CMB_ELEMENTWISE`, `CMB_REDUCE`, `CMB_ENABLE`). It keeps its operations, the softmax's elementwise ops and reductions among them, and gains the merge of partials, scaling and block-format conversion. |
| The fused epilogue: `MatMulComputeSpec { bias, activation }` applied in `execute_matmul` (`docs/plans/e10_fused_epilogue_pattern.md`) | The Combiner's bias and activation, applied as the result is written to L2. |
| The accumulator as its own banked memory with slot credits (#401, #383, #387) | Gone. Partials and results occupy L2 slots, and the Combiner's merges are L2 traffic. |
| The `systolic` block of a deployment: accumulator slots, banks, rows, latency (#396) | The Combiner's configuration. The accumulator fields retire when the RTL drops them (branes-ai/embodied-ai-soc-generator#67). |
| `kpu-tile-cmds` (#397): a LOAD into the accumulator for a bias, COMPUTE accumulate or overwrite | A COMPUTE names its Combiner operation: merge or overwrite, bias, scale, activation, format. |

The work to model it, at L-B, L-T1 and L-CA, in the shared records and in the contracts with
the RTL, is tracked in #427.
