# Distributed CSP coverage: does the 8-mover, one-hop allocation cover P1–P9?

**Date:** 2026-11-28
**Issue:** #617 (step 6 of `docs/plans/distributed-csp-programs.md`, §3.5)
**Status:** measured. CI regenerates the block below from the generators' programs
(`kpu-csp-coverage --check`), so a change to a generator that moves a number fails until the report is
rewritten (`kpu-csp-coverage --deploy-dir tests/program/deploy --write docs/analysis/distributed-csp-coverage.md`).

## The answer

**Every pattern class has a level-2 program on the DMM mesh that the allocation covers.** The allocation
is one hop per move, eight BlockMovers per node and a periphery that carries DMA flows only. On the 1×1,
2×2 and 4×4 meshes:
- every class's program passes D0–D4;
- every program runs on L-B without deadlock (D5);
- every program gives its operator's values (D6, in each generator's tests).

No class needs the allocation changed. So nothing here blocks the NoC rework (decision 3).

Two classes fit with nothing to pay: P5, broadcast, and P8, chained-GEMM fusion. The other seven fit at a
cost, and the cost has two forms.

- **Store-and-forward.** A node pushes on a tile it does not use, so the tile can reach a node that is not
  its neighbour: P1's C going north, P6's operands going east, P9's cache going up and down its column,
  P7's routes through the diagonal. The forwarding is the price of one hop. It costs mover time and L3
  slots in the nodes it passes, and it grows with the mesh's side.
- **L3 and movers.** P2, the sliding window, is the tightest:
  - an interior node runs all 8 movers, 4 of them sends;
  - it uses 14 of its 16 L3 slots.

  P1 uses 12 of the 16, and P7 uses 11. P7's count is the one that grows with the side.

## Where the allocation would need to change

None of these is a "needs" at the sizes measured. Each is where one would come from first.

- **P7 at larger meshes.** The busiest node's L3 holds a slot for every tile routed through it, so a
  transpose's L3 cost grows with the mesh's side: 5 slots on 2×2, 11 on 4×4. Larger meshes are not
  measured here, and the generator refuses a node past its capacity. The first fix past that is a program
  change, not an allocation change: route in phases, or reuse slots under credit.
- **P2 with a stencil that does not separate.** P2 avoids corner halos by filtering in two passes, a row
  pass and a column pass. A stencil that cannot be split that way needs its diagonal neighbours' tiles,
  and on a one-hop mesh those take two hops through an edge neighbour. That costs more forwarding, at
  movers P2 already uses all of.
- **P8 with its softmax.** Attention's online softmax is a tile function between the two GEMMs. It keeps
  state per row in the node, and it is not built. It adds L2 state, not movers.
- **P9 at depth.** A head at row r is r hops from its column's edge each way, and a decode's steps are
  sequential at the engine. That is latency, not a resource the allocation lacks.

## How it is measured

`include/sw/kpu/program/csp/distributed_coverage.hpp` reads each number off the program itself, nothing
off the generator:

| Measure | What it counts |
|---|---|
| Movers (sends) | the BlockMovers a node runs, and the send movers among them: the most on one node |
| L3 slots, L2 slots | the slots the channels into a node own, against the cluster's capacity: the most on one node |
| Credits per flow | each mesh channel's slots, the fewest and the most (`-`: no mesh channel) |
| Hops | for each tile at each node it reaches, the fewest pushes from where it was loaded or made, and the most of those |
| Pushes (transit) | mesh pushes, and those out of a node that neither feeds the tile nor made it: it only forwards |
| L-B rounds | the rounds level-2 L-B takes to run it: a step count, not a timing |
| DMA KiB, busiest port | every DMA flow's bytes, each tile's elements at its type's width; the busiest edge port's |
| Busiest ring segment | the periphery's load under the ring model below, for 4, 8, 16 and 32 PHYs |

**The ring model is a hypothesis, stated so it can be argued with.**
- N PHYs sit evenly spaced around the mesh's perimeter.
- Every tensor is interleaved over all N, so each DMA flow's bytes split evenly over them.
- Each share takes the shorter way around the ring to its edge port.
- The perimeter has one unit segment per edge-node side. The busiest segment's load is the number given.

It measures how far the periphery carries data, not whether a ring is the right periphery (§3.5, the
comparison with Clos).

**The verdict**, on the 4×4 mesh:
- **needs the allocation changed:** a check fails, a node needs more than 8 movers, or it needs more than 4
  sends.
- **fits at a cost:**
  - a push only forwards; or
  - a node's channels own more than half its L3.

  The verdict gives each number.
- **fits:** everything else.

The sizes are the generators' tests' sizes, scaled with the mesh. P1 runs int8 into int32, and P3 and P6
run int32. P2, P4, P5, P7, P8 and P9 run at their defaults: P8, P4 and P9 in fp32, the rest in int32.

## The measures

<!-- coverage:begin (kpu-csp-coverage writes this block; do not edit) -->

| Class | Verdict on the 4x4 mesh |
|---|---|
| P1 GEMM | **fits at a cost:** 48 of its 240 mesh pushes only forward; 12 of a node's 16 L3 slots |
| P2 sliding window | **fits at a cost:** 24 of its 96 mesh pushes only forward; 14 of a node's 16 L3 slots |
| P3 row reduction | **fits at a cost:** 48 of its 72 mesh pushes only forward; 9 of a node's 16 L3 slots |
| P4 gather/scatter | **fits at a cost:** 36 of its 48 mesh pushes only forward |
| P5 broadcast | **fits** |
| P6 two-operand aligned | **fits at a cost:** 60 of its 72 mesh pushes only forward |
| P7 layout transform | **fits at a cost:** 76 of its 88 mesh pushes only forward; 11 of a node's 16 L3 slots |
| P8 chained-GEMM fusion | **fits** |
| P9 append | **fits at a cost:** 288 of its 360 mesh pushes only forward |

| Class | Mesh | Movers (sends) | L3 slots | L2 slots | Credits per flow | Hops | Pushes (transit) | L-B rounds | D0-D5 | DMA KiB | Busiest port KiB | Busiest ring segment KiB, 4 / 8 / 16 / 32 PHYs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 GEMM | 1x1 | 4 (1) | 10 / 16 | 8 / 8 | - | 0 | 0 (0) | 13 | pass | 6.0 | 4.0 | 1.0 / 1.5 / 1.2 / 1.1 |
| P1 GEMM | 2x2 | 6 (3) | 12 / 16 | 8 / 8 | 2-4 | 1 | 24 (0) | 17 | pass | 20.0 | 9.0 | 5.2 / 5.6 / 6.7 / 6.2 |
| P1 GEMM | 4x4 | 6 (3) | 12 / 16 | 8 / 8 | 2-4 | 3 | 240 (48) | 29 | pass | 76.0 | 17.5 | 28.1 / 24.3 / 25.1 / 27.2 |
| P2 sliding window | 1x1 | 3 (1) | 6 / 16 | 3 / 8 | - | 0 | 0 (0) | 4 | pass | 0.5 | 0.2 | 0.1 / 0.1 / 0.1 / 0.1 |
| P2 sliding window | 2x2 | 7 (3) | 12 / 16 | 7 / 8 | 2-4 | 1 | 12 (0) | 9 | pass | 2.0 | 0.5 | 0.5 / 0.5 / 0.6 / 0.5 |
| P2 sliding window | 4x4 | 8 (4) | 14 / 16 | 7 / 8 | 2-4 | 3 | 96 (24) | 21 | pass | 8.0 | 1.0 | 2.5 / 2.5 / 2.5 / 2.6 |
| P3 row reduction | 1x1 | 4 (1) | 6 / 16 | 6 / 8 | - | 0 | 0 (0) | 7 | pass | 4.1 | 4.1 | 1.0 / 1.5 / 1.3 / 1.2 |
| P3 row reduction | 2x2 | 7 (3) | 8 / 16 | 5 / 8 | 1-4 | 1 | 6 (2) | 6 | pass | 4.1 | 2.1 | 1.5 / 1.5 / 1.7 / 1.6 |
| P3 row reduction | 4x4 | 7 (3) | 9 / 16 | 5 / 8 | 1-4 | 3 | 72 (48) | 16 | pass | 32.2 | 8.1 | 14.1 / 13.1 / 12.6 / 13.3 |
| P4 gather/scatter | 1x1 | 3 (1) | 6 / 16 | 4 / 8 | - | 0 | 0 (0) | 3 | pass | 0.5 | 0.5 | 0.1 / 0.2 / 0.2 / 0.1 |
| P4 gather/scatter | 2x2 | 4 (2) | 8 / 16 | 4 / 8 | 2-4 | 1 | 4 (2) | 5 | pass | 2.0 | 1.0 | 0.5 / 0.6 / 0.8 / 0.7 |
| P4 gather/scatter | 4x4 | 4 (2) | 8 / 16 | 4 / 8 | 2-4 | 3 | 48 (36) | 10 | pass | 8.0 | 2.0 | 3.0 / 2.5 / 2.8 / 3.0 |
| P5 broadcast | 1x1 | 1 (0) | 2 / 16 | 2 / 8 | - | 0 | 0 (0) | 3 | pass | 0.1 | 0.1 | 0.0 / 0.0 / 0.0 / 0.0 |
| P5 broadcast | 2x2 | 3 (2) | 2 / 16 | 2 / 8 | 2 | 2 | 6 (0) | 3 | pass | 0.1 | 0.1 | 0.1 / 0.0 / 0.1 / 0.1 |
| P5 broadcast | 4x4 | 3 (2) | 4 / 16 | 4 / 8 | 4 | 6 | 60 (0) | 5 | pass | 0.2 | 0.2 | 0.1 / 0.1 / 0.1 / 0.1 |
| P6 two-operand aligned | 1x1 | 4 (1) | 6 / 16 | 6 / 8 | - | 0 | 0 (0) | 4 | pass | 3.0 | 2.0 | 0.5 / 0.9 / 0.7 / 0.6 |
| P6 two-operand aligned | 2x2 | 5 (2) | 8 / 16 | 6 / 8 | 2-4 | 1 | 6 (4) | 9 | pass | 12.0 | 4.0 | 3.0 / 3.2 / 3.5 / 3.3 |
| P6 two-operand aligned | 4x4 | 5 (2) | 8 / 16 | 6 / 8 | 2-4 | 3 | 72 (60) | 25 | pass | 48.0 | 8.0 | 16.0 / 15.0 / 15.5 / 15.9 |
| P7 layout transform | 1x1 | 3 (1) | 2 / 16 | 2 / 8 | - | 0 | 0 (0) | 4 | pass | 0.5 | 0.2 | 0.1 / 0.1 / 0.1 / 0.1 |
| P7 layout transform | 2x2 | 5 (3) | 5 / 16 | 2 / 8 | 1-2 | 2 | 8 (6) | 8 | pass | 2.0 | 0.5 | 0.5 / 0.5 / 0.6 / 0.5 |
| P7 layout transform | 4x4 | 5 (3) | 11 / 16 | 2 / 8 | 1-4 | 6 | 88 (76) | 24 | pass | 8.0 | 1.0 | 2.5 / 2.5 / 2.5 / 2.6 |
| P8 chained-GEMM fusion | 1x1 | 4 (1) | 6 / 16 | 4 / 8 | - | 0 | 0 (0) | 9 | pass | 2.0 | 2.0 | 0.5 / 0.8 / 0.6 / 0.6 |
| P8 chained-GEMM fusion | 2x2 | 7 (3) | 8 / 16 | 5 / 8 | 2 | 1 | 16 (0) | 10 | pass | 4.0 | 2.0 | 1.2 / 1.4 / 1.4 / 1.4 |
| P8 chained-GEMM fusion | 4x4 | 7 (3) | 8 / 16 | 5 / 8 | 2 | 3 | 96 (0) | 14 | pass | 8.0 | 2.0 | 2.8 / 2.9 / 2.9 / 3.0 |
| P9 append | 1x1 | 3 (1) | 6 / 16 | 4 / 8 | - | 0 | 0 (0) | 16 | pass | 3.8 | 2.2 | 0.6 / 1.0 / 0.8 / 0.7 |
| P9 append | 2x2 | 4 (2) | 8 / 16 | 4 / 8 | 2-4 | 1 | 30 (18) | 31 | pass | 15.0 | 7.5 | 4.9 / 4.5 / 5.0 / 4.7 |
| P9 append | 4x4 | 4 (2) | 8 / 16 | 4 / 8 | 2-4 | 3 | 360 (288) | 61 | pass | 60.0 | 15.0 | 24.8 / 22.1 / 20.8 / 22.2 |

<!-- coverage:end -->
