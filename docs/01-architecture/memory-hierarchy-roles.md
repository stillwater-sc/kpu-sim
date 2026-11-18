# The Memory Hierarchy's Roles: L3 Holds, L2 Feeds, L1 Transforms

**Status:** an architecture decision, 2026-11-18 (the architect's). It is the reference for
designing and validating the domain flow's data movement.
**Related:** [`kpu-execution-model.md`](kpu-execution-model.md) (how each level is managed:
credits, slots, banks), [`combiner.md`](combiner.md) (the result path),
[`docs/plans/blocked-linear-algebra-test-set.md`](../plans/blocked-linear-algebra-test-set.md)
(#483, the first decision taken under this rule).

The execution model says *how* each buffer level is managed. This document says *what each level
is for*, which decides where any transformation of the data may happen.

## 1. The three roles

| Level | Role | What it does to the data |
|---|---|---|
| **L3** | **Holding memory**: reuse, and double buffering to hide DRAM's latency | Holds tiles for as long as the program reuses them. Nothing is transformed. |
| **L2** | **Performance**: the massive concurrency needed to feed the compute fabric | Serves a continuous stream of read requests, with many in flight. The data must already be laid out the way the streams will read it. **No compute and no transformation happen on L2's read path.** |
| **L1** | **Stream transformation**: parallel vector in, element-wise out | Turns the vectors read from L2 into the element streams that enter the array's edges. |

## 2. Why the L2 must be pre-arranged: the arithmetic

The figures are the architect's working numbers, not a specification:

| Quantity | Value |
|---|---|
| The array's clock | 1–2 GHz, so a clock every 0.5–1 ns |
| What the array consumes | at least **two vectors per clock** |
| The L2 SRAM's clock | about 250 MHz, so 4 ns per access |
| An L2 bank's access latency, in array clocks | up to **8** at 2 GHz (4 ns / 0.5 ns), 4 at 1 GHz |

To deliver two vectors every array clock against a latency of L clocks, at least 2 · L read
requests must be in flight at once (Little's law: concurrency = rate × latency):

| Array clock | L2 latency | Requests in flight, at least |
|---|---|---|
| 2 GHz | 8 clocks | **16** |
| 1 GHz | 4 clocks | **8** |

A stream with that much concurrency has no time to transform what it reads: each request must
return data that slots straight into the stream. So the layout the streams need is created on the
way **into** L2, never on the way out.

## 3. Where transformations go

- **On the way into L2: the BlockMover.** Any rearrangement of a tile, such as a transpose or a
  layout change, happens as the BlockMover writes it from L3 into L2. L2's write path runs at the
  rate of block moves, not at the rate the fabric consumes, so it can afford the work.
- **Out of L2: the Streamer.** It reads pre-arranged rows and does nothing else. Its concurrency
  goes to keeping the requests in flight.
- **In L1:** the stream transformation from vectors to element streams, which is L1's own role.
- **The result path** (L1 → L2, through the Combiner) is the other direction, with its own rules
  ([`combiner.md`](combiner.md)).

### The first decision under this rule: transposes (#483)

GEMM with transposed operands (Aᵀ·B, A·Bᵀ) transposes the tile **in the BlockMover, as it writes
the tile into L2**. The Streamer's L2 → L1 reads have no time for it: L2's concurrency is fully
spent keeping the L1 fed. So a transposed operand reaches L2 already in the layout its stream
reads.

## 4. What this means for design and validation

These are the checks a model or a program must pass to respect the roles:

1. **Nothing on L2's read path transforms data.** A program, a level or the RTL that transposes,
   reorders or computes on a Feed (L2 → L1) is wrong. The transformation belongs to the Move
   (L3 → L2) that put the tile in L2.
2. **A layout is chosen when the tile enters L2.** In CSP terms, a transform attaches to a Move,
   at the BlockMover, never to a Feed.
3. **The fabric's feed rate is a concurrency requirement on L2.** A deployment's L2 must sustain
   at least `vectors per clock × L2 latency in array clocks` requests in flight: banks, ports and
   the Streamer's request queue together. Validation should check that bound for every
   deployment.
4. **L3 is judged on reuse and latency hiding, not bandwidth to the fabric.** Its questions are
   whether the program's reuse fits and whether double buffering covers DRAM's latency.

### Where the simulator stands

- **L-CA's banked L2** (#401, slice 1) models banks, ports and a read latency (4 clocks by
  default, or `systolic.spad_read_delay`). It charges the latency once per transfer, with the
  rows pipelined behind it, so it effectively assumes as many requests in flight as needed.
- **Not yet modelled or checked:** the number of outstanding requests, as a limit or as a
  requirement. A deployment whose L2 cannot keep `2 · L` requests in flight is not flagged today.
  This should become part of #401's later slices, and a deployment validator check.
- **Transposes** are not modelled yet (#483). When they are, they are a BlockMover transform on
  the Move into L2, under rule 2.
