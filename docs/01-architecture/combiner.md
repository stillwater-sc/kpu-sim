# The Combiner

The **Combiner** is the KPU tile's unit on the result path, between L1 and L2. It is a new
concept, added on 2026-11-14 for the KPU RTL generator (branes-ai/embodied-ai-soc-generator#68).
kpu-sim does not model it yet (#427), and neither does systars. It replaces the separate
accumulator memory that kpu-sim's cycle-accurate plans and the RTL have today.

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
  - block-format conversion (compression).
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
| Push-based fusion, [KPU architecture](kpu-architecture.md) §2.7: fused operations inserted on the egress path, with SFU banks alongside it | The Combiner is the named unit at that insertion point. Open: whether the SFU banks are part of it or sit beside it. |
| The fused epilogue: `MatMulComputeSpec { bias, activation }` applied in `execute_matmul` (`docs/plans/e10_fused_epilogue_pattern.md`), and the Vector Engine's fused drain (`docs/06-compiler/schedule-dsl.md`) | The Combiner's bias and activation. Scaling and block-format conversion join them. Open: whether the Vector Engine's other operations (softmax's elementwise ops and reductions) belong to the Combiner. |
| The accumulator as its own banked memory with slot credits (#401, #383, #387) | Gone. Partials and results occupy L2 slots, and the Combiner's merges are L2 traffic. |
| The `systolic` block of a deployment: accumulator slots, banks, rows, latency (#396) | The Combiner's configuration. The accumulator fields retire when the RTL drops them (branes-ai/embodied-ai-soc-generator#67). |
| `kpu-tile-cmds` (#397): a LOAD into the accumulator for a bias, COMPUTE accumulate or overwrite | A COMPUTE names its Combiner operation: merge or overwrite, bias, scale, activation, format. |

The work to model it, at L-B, L-T1 and L-CA, in the shared records and in the contracts with
the RTL, is tracked in #427.
