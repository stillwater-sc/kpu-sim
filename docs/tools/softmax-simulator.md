# Softmax simulator + tile-state tracker

`examples/schedule/softmax_simulator.cpp` runs the online-softmax schedule
(epic E8) one executor cycle at a time and prints a **horizontal tile-state
log**: what tiles occupy each level of the software-managed memory hierarchy,
snapshot by snapshot, so a human can watch the data-movement schedule execute.

```text
softmax_simulator [--rows R] [--len N] [--tile T] [--l3 C] [--l2 C]
```

## How to read the log

Each band is one snapshot. Columns run left-to-right in the direction data
flows toward compute: **L3 | L2 | L1 / array**. A band is printed only when
occupancy changes, so successive bands read as the state-transition
progression. A tile cell shows its `TileID`; when the value plane is active it
also shows a compact content summary, and an `*` marks a tile resident in the
systolic array (compute storage).

Sample (`softmax_simulator`, default 1 row x 512, two tiles):

```text
Online softmax simulator  —  online_row_resident  (1 row(s) x 512, tile 256, envelope L3=32/L2=64)

cyc    | L3 buffers                         | L2 banks                           | L1 / array                        
-------+------------------------------------+------------------------------------+-----------------------------------
0      | -                                  | -                                  | -
36     | A[0,0,0]                           | -                                  | -
37     | A[0,0,0] A[0,1,0]                  | -                                  | -
60     | A[0,1,0]                           | A[0,0,0]                           | -
72     | A[0,1,0]                           | A[0,0,0]                           | A[0,0,0]*
84     | -                                  | A[0,1,0]                           | A[0,0,0]*
96     | -                                  | A[0,1,0]                           | A[0,0,0]* A[0,1,0]*
108    | -                                  | -                                  | A[0,0,0]* A[0,1,0]*
129    | -                                  | -                                  | A[0,0,0]* A[0,1,0]* B[0,0,0]=(-0.69,440)*
162    | -                                  | -                                  | A[0,0,0]* A[0,1,0]* B[0,0,0]=(-0.69,440)* C[0,0,0]* C[0,1,0]*
174    | -                                  | C[0,0,0]                           | A[0,0,0]* A[0,1,0]* B[0,0,0]=(-0.69,440)* C[0,0,0]* C[0,1,0]*
186    | -                                  | C[0,0,0] C[0,1,0]                  | A[0,0,0]* A[0,1,0]* B[0,0,0]=(-0.69,440)* C[0,0,0]* C[0,1,0]*
199    | C[0,0,0]                           | C[0,1,0]                           | A[0,0,0]* A[0,1,0]* B[0,0,0]=(-0.69,440)* C[0,0,0]* C[0,1,0]*
223    | C[0,0,0] C[0,1,0]                  | -                                  | A[0,0,0]* A[0,1,0]* B[0,0,0]=(-0.69,440)* C[0,0,0]* C[0,1,0]*
250    | C[0,1,0]                           | -                                  | A[0,0,0]* A[0,1,0]* B[0,0,0]=(-0.69,440)* C[0,0,0]* C[0,1,0]*
274    | -                                  | -                                  | A[0,0,0]* A[0,1,0]* B[0,0,0]=(-0.69,440)* C[0,0,0]* C[0,1,0]*

Result: each row's softmax sums to 1
  row 0: sum = 1  [OK]

SOFTMAX OK
```

Reading it: the two input tiles `A[0,0,0]`/`A[0,1,0]` stream `DRAM -> L3 -> L2 ->
L1/array`; the **stats compute** produces `B[0,0,0]=(m,l)` — here `(-0.69, 440)`,
the running max and the exp-sum normalizer — which then stays **resident in the
array** (the E8 compute-resident hand-off, no DRAM round-trip) and feeds the
**apply computes** that emit the normalized `C[0,0,0]`/`C[0,1,0]`; those drain back
`array -> L2 -> L3 -> DRAM`. The run ends on the softmax correctness check:
each row sums to 1.

These are **buffers, not caches** — the log reads as tile *arrived* / *resident*
/ credit *returned*, never hit/miss/evict.

## Under the hood

- `OnlineSoftmaxScheduleGenerator` (#156) produces the movement schedule.
- The softmax value ops (`softmax_stats`, `softmax_apply`) come from
  `FunctionalSoftmaxExecutor` (#157); the simulator binds them to the COMPUTEs
  and drives the executor cycle-by-cycle so it can observe between steps.
- `TileTracker` (`include/sw/kpu/timing/tile_tracker.hpp`, #165) renders the
  bands from `ConcurrentTimingExecutor::tiles_at(level)` — a pure observer,
  never mutating executor state.
