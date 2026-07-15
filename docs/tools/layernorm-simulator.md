# LayerNorm simulator + tile-state tracker

`examples/schedule/layernorm_simulator.cpp` runs a LayerNorm schedule one
executor cycle at a time and prints the **horizontal tile-state log** — what
tiles occupy each level of the software-managed memory hierarchy, snapshot by
snapshot. It is a third companion to the [softmax](softmax-simulator.md) and
[matmul](matmul-simulator.md) simulators; for the general approach see
[Tracking Schedules as Tile-State Movement](../tile-state-tracking.md).

```text
layernorm_simulator [--rows R] [--len N] [--tile T]
```

Default: `1 row x 512 features`, tile 256.

---

## What the LayerNorm tile log shows

LayerNorm normalizes each row over its feature dimension:

```text
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

That is a **row-streaming reduction** (compute `mean` and `var` over the row)
followed by an **apply** pass (normalize each feature, then scale/shift by the
per-feature `gamma`/`beta`). The log shows the row streaming
`DRAM -> L3 -> L2 -> L1/array`, the stats compute producing the `(mean, var)`
tile, that tile staying **resident in the array** and feeding the apply
computes, and the normalized outputs draining back out. It ends on a host-oracle
check.

Sample (`layernorm_simulator`, default 1 row x 512, two tiles):

```text
LayerNorm simulator  —  online_row_resident  (1 row(s) x 512 features, tile 256, eps 1e-05)

cyc    | L3 buffers                         | L2 banks                           | L1 / array                        
-------+------------------------------------+------------------------------------+-----------------------------------
0      | -                                  | -                                  | -
36     | A[0,0]                             | -                                  | -
37     | A[0,0] A[0,1]                      | -                                  | -
60     | A[0,1]                             | A[0,0]                             | -
72     | A[0,1]                             | A[0,0]                             | A[0,0]*
84     | -                                  | A[0,1]                             | A[0,0]*
96     | -                                  | A[0,1]                             | A[0,0]* A[0,1]*
108    | -                                  | -                                  | A[0,0]* A[0,1]*
129    | -                                  | -                                  | A[0,0]* A[0,1]* B[0,0]=(-0.115,0.0525)*
162    | -                                  | -                                  | C[0,0]* C[0,1]*
174    | -                                  | C[0,0]                             | C[0,1]*
186    | -                                  | C[0,0] C[0,1]                      | -
199    | C[0,0]                             | C[0,1]                             | -
223    | C[0,0] C[0,1]                      | -                                  | -
250    | C[0,1]                             | -                                  | -
274    | -                                  | -                                  | -

Result: y = gamma * (x - mean)/sqrt(var + eps) + beta
  max abs error vs host oracle: 1.55641e-07  [OK]

LAYERNORM OK
```

Reading it: tiles carry their 2D `[row, feature-tile]` index. `A[0,0]`/`A[0,1]`
stream in; the **stats compute** produces `B[0,0]=(mean, var)` — here
`(-0.115, 0.0525)` — which stays **resident** and feeds the apply computes that
emit the normalized `C[0,0]`/`C[0,1]`; once the applies fire, the inputs and the
stat **retire** from the array and the outputs drain back
`array -> L2 -> L3 -> DRAM`.

## Why it looks like softmax — and where it differs

LayerNorm and softmax are the **same P3 movement pattern**: stream the row,
reduce it to a small resident statistic, keep that statistic on chip, stream the
row past it again to apply, drain the result. So the layernorm simulator reuses
the *same online row-reduction schedule* as the softmax simulator — the tile
movement is identical. What differs is the **content of the statistic** and the
**apply**:

| | statistic tile `B` | apply |
|---|---|---|
| softmax | `(max, exp-sum)` | `exp(x - max) / sum` |
| layernorm | `(mean, var)` | `gamma * (x - mean)/sqrt(var + eps) + beta` |

The tracker makes that visible: the `B` cell shows `(mean, var)` where softmax
showed `(max, sum)`, and in a batched run each row carries its own stats
(`B[0,0]=(-0.115,0.0525)`, `B[1,0]=(0.885,0.0525)` — the means differ per row).
This is the value of a content-carrying log: the same movement, told apart by
what the tiles hold. The per-feature `gamma`/`beta` are applied inside the apply
op (one slice per feature-tile); their delivery is the elementwise/broadcast
pattern and is not shown as separate tile movement here.

## Under the hood

- The movement schedule is the online row-reduction + resident-apply structure
  (`OnlineSoftmaxScheduleGenerator` — the class is softmax-named but the
  schedule is the generic P3 pattern; a dedicated LayerNorm generator is epic
  E9 / issue #78).
- `layernorm_stats` (mean + clamped population variance, the E3 VAR-moment
  semantics) and `layernorm_apply` are bound to the COMPUTEs; `(mean, var)`
  reaches the apply computes as a compute-**resident** dependency (the #155
  mechanism, no DRAM round-trip).
- `TileTracker` (#165) renders the bands from
  `ConcurrentTimingExecutor::tiles_at(level)` - a pure observer.
