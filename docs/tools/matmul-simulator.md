# Matmul simulator + tile-state tracker

`examples/schedule/matmul_simulator.cpp` runs a tiled matmul schedule
(`MatMulScheduleGenerator`) one executor cycle at a time and prints the
**horizontal tile-state log** — what tiles occupy each level of the
software-managed memory hierarchy, snapshot by snapshot. It is the matmul
companion to the [softmax simulator](softmax-simulator.md); for the general
approach, see [Tracking Schedules as Tile-State Movement](../tile-state-tracking.md).

```text
matmul_simulator [--m M] [--n N] [--k K] [--tile T]
```

`--tile` must divide `M`, `N`, and `K` evenly. Default: `C[32x32] = A[32x32] *
B[32x32]`, tile 16 (a 2x2x2 tile grid).

---

## What the matmul tile log shows

A tiled matmul `C = A x B` moves **two** operand streams and reduces over K:

- `A[ti,0,tk]` — a row-block of A (rows `ti`, K-slice `tk`).
- `B[0,tj,tk]` — a column-block of B (columns `tj`, K-slice `tk`).
- `C[ti,tj,0]` — an output block, the sum over `tk` of `A[ti,tk] * B[tk,tj]`.

The log shows the A and B tiles streaming `DRAM -> L3 -> L2 -> L1/array`
interleaved (the default `interleaved_ab` strategy alternates an A tile and a B
tile so neither operand monopolizes the buffers), each `C[ti,tj]` compute firing
once **all** its K-slices have been fed and accumulating them in the array, and
the finished `C` tiles draining back out `array -> L2 -> L3 -> DRAM`. The run
ends on the `C = A x B` host-oracle check.

## Why matmul's log looks the way it does

Reading it next to the softmax log is instructive — the two operators stress the
hierarchy differently, and the trace makes that visible.

- **Two interleaved input streams, not one.** Softmax streams a single row;
  matmul streams A row-blocks *and* B column-blocks together. In the log you see
  pairs like `A[0,0,0] A[0,0,1] B[0,0,1] B[0,1,1]` filling L3 — the
  `interleaved_ab` discipline keeping an A-burst and a B-burst simultaneously
  resident so neither starves the other (the #67 per-matrix burst share).

- **K-accumulation is a join, not a chain.** A `C[ti,tj]` compute depends on
  every A[ti,*,tk] and B[*,tj,tk] K-slice (`2 * k_tiles` feeds); it cannot fire
  until all have arrived, and its latency scales with the K-slice count. You see
  the A/B tiles for both K-slices reach the array (`*`) before `C[0,0,0]`
  appears — the reduction depth made literal.

- **Tiles are large data blocks, so the log shows movement, not content.** A
  softmax stats tile is two numbers `(m, l)` and prints its value; a matmul tile
  is a `T x T` block (256 values at tile 16), too large to summarize inline, so
  the cells show tile *identity and location*. The matmul log is a **data-motion**
  view; the softmax log is a data-motion **and content** view. Same tracker,
  different information density because the operators carry different state.

- **Reuse is visible.** An A row-tile contributes to every C tile in its row
  (all `tj`); a B column-tile to every C tile in its column (all `ti`). Watching
  A/B tiles remain resident in the array while multiple C tiles form is the
  classic matmul operand-reuse the tiling exists to exploit.

- **Blocking to the hierarchy.** L3 occupancy stays bounded by the envelope
  share as tiles stream — the blocked-linear-algebra discipline of sizing bursts
  to the credit pools (#67). A schedule that let A or B flood L3 would show it
  here as one operand crowding the left column; a wedge would show as occupancy
  that never drains. (The #61 multi-tile livelock was exactly a drain that never
  happened — invisible in cycle counts, obvious in this view.)

## How to read a band (worked example)

From `matmul_simulator` (default `C[32x32]=A*B`, tile 16, `interleaved_ab`).
Excerpted from the full trace (`...` elides cells):

```text
cyc    | L3 buffers                         | L2 banks                           | L1 / array
-------+------------------------------------+------------------------------------+-----------------------------------
38     | A[0,0,0] A[0,0,1] B[0,0,1]         | -                                  | -
74     | A[0,0,1] B[0,1,1]                  | A[0,0,0] B[0,0,1]                  | A[0,0,0]* B[0,0,1]*
159    | -                                  | -                                  | A[0,0,0]* A[0,0,1]* A[1,0,0]* A[1,0,1]* B[0,0,0]* B[0,0,1]* B[0,1,0]* B[0,1,1]*
187    | -                                  | -                                  | ... B[0,1,1]* C[0,0,0]*
248    | C[0,0,0] C[0,1,0]                  | C[1,0,0] C[1,1,0]                  | ... C[0,0,0]* C[0,1,0]* C[1,0,0]* C[1,1,0]*
```

- **cyc 38** — A and B tiles stream into L3 **interleaved**: two A K-slices and
  a B K-slice already staged. Neither operand is allowed to monopolize the
  buffers.
- **cyc 74** — the pipeline is filling: `A[0,0,0]`/`B[0,0,1]` in the array (`*`),
  their successors one boundary behind in L2, more in L3.
- **cyc 159** — all K-slices of A and B for the first outputs are resident in the
  array: `A[0,0,*]`, `A[1,0,*]`, `B[*,*,*]`. The **join** is complete, so a
  compute can fire.
- **cyc 187** — the first result `C[0,0,0]` appears — `A[0,0,0]*B[0,0,0] +
  A[0,0,1]*B[0,1,0]` accumulated over the two K-slices.
- **cyc 248** — the four output tiles `C[0,0]`, `C[0,1]`, `C[1,0]`, `C[1,1]`
  drain back out through L2/L3 toward DRAM. The run then checks `C = A x B`
  against the host reference (max abs error 0 for integer operands).

The rightmost column accumulates all resident A/B/C tiles (compute payloads
persist); read the **left** columns for the live streaming front, and note the
strategy in the header (`interleaved_ab`). Other strategies
(`output_stationary`, `blocked_ab`, `prefetch_next`) produce visibly different
streaming orders in the same view.

## Under the hood

- `MatMulScheduleGenerator` (`include/sw/kpu/timing/schedule/matmul_schedule_generator.hpp`)
  produces the movement schedule; each `C[ti,tj]` COMPUTE carries its full A/B
  K-slice dependency set.
- The simulator seeds A/B tiles from host operands, splits each COMPUTE's
  interleaved dependencies into the paired `a_tiles`/`b_tiles` of a
  `MatMulComputeSpec`, and drives the executor cycle-by-cycle so `TileTracker`
  (#165) can observe between steps — a pure observer over
  `ConcurrentTimingExecutor::tiles_at(level)`, never mutating executor state.
