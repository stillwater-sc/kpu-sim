# Tracking Schedules as Tile-State Movement

A way to *watch* a KPU schedule execute: render, snapshot by snapshot, which
tiles of state occupy each level of the software-managed memory hierarchy, laid
out horizontally in the direction data flows toward compute. This page explains
the **what**, the **why**, and the **how**.

- Tool how-to: [`docs/tools/softmax-simulator.md`](tools/softmax-simulator.md)
- Renderer: `include/sw/kpu/timing/tile_tracker.hpp`
- Observer API: `ConcurrentTimingExecutor::tiles_at(MemoryLevel)`
- Example driver: `examples/schedule/softmax_simulator.cpp`

---

## What

A KPU schedule is not a stream of instructions over a flat memory. It is a set
of **transfers of tiles of state that stage through a software-managed memory
hierarchy** — DRAM → L3 buffers → L2 banks → L1 / systolic array, and back out.
Every schedule operation moves a tile one boundary closer to (or further from)
compute, gated by credits.

Tile-state tracking makes that concrete. At each snapshot it prints one
**horizontal band**:

```text
cyc    | L3 buffers      | L2 banks        | L1 / array
```

with L3 on the left, L2 to its right, and L1/array on the far right — the same
left-to-right order data physically flows toward the compute fabric. Each cell
names a resident tile and, when the value plane is active, shows a compact
summary of its **content** (e.g. an online-softmax stats tile
`B[0,0]=(-0.69,440)` carrying its running max and exp-sum). An `*` marks a tile
resident in the array (compute storage).

Tiles carry a **2D submatrix index**. A `TileID` is a coordinate in the 3D
M/N/K tile grid `(ti, tj, tk)`, but each matrix uses only two of those axes (the
third is a placeholder pinned to 0), so a driver labels tiles with the two live
axes — matmul as `A[ti,tk]` / `B[tk,tj]` / `C[ti,tj]`, softmax as `[row, tile]`.
`A[0,1]` is thus row-block 0, K-block 1 — a genuinely different submatrix from
`A[0,0]`. (`TileTracker::Config::label` sets the convention; the raw 3-tuple is
the default.)

A band is emitted only when occupancy *changes*, so reading the bands
top-to-bottom is reading the schedule's **state-transition trace** — not a
per-cycle dump, but the sequence of "a tile arrived / became resident / its
credit was returned" events that make up the execution.

These are **buffers, not caches**. The log reads as *arrived / resident /
credit returned*, never hit/miss/evict — matching the credit-based dataflow
model (credits flow up, tiles flow down).

## Why

Credit-based dataflow is **concurrent and event-driven**: DMA, BlockMovers, and
Streamers all run at once, pushing tiles downstream only when they hold a
downstream credit. That is exactly what makes it efficient — and exactly what
makes it hard to reason about from source code or a raw timing-event list. A
"what is where, right now" view restores legibility, and it pays off in several
ways.

- **See content, not just location.** The stats tile shows its `(m, l)` pair,
  so you can confirm the *computation* is right as the data moves — the tracker
  is a correctness lens, not only a movement lens. The run ends on an
  operator-level check (softmax rows sum to 1), so the trace closes on a
  verifiable statement.

- **Diagnose movement pathologies.** A tile stuck at one level across many
  bands is a **stall**; occupancy that never advances is a **wedge**; a tile
  that never leaves L3/L2 after its consumers are done is a **credit leak**.
  These jump out of the horizontal trace in a way they never do from cycle
  counts. (The #61 multi-tile livelock was invisible precisely because no view
  showed occupancy failing to drain.)

- **Verify the schedule as designed.** The trace shows the design's invariants
  holding: **pipelining** (several rows of a batch in flight across L3/L2/L1 at
  once), the **burst bound** (L3 occupancy capped at the per-matrix envelope
  share, #67), and the **compute-resident hand-off** (a producer's tile staying
  in the array to feed a consumer with no DRAM round-trip, #155) — you watch
  the `(m, l)` tile appear and remain resident while the apply outputs form.

- **Teach the pattern.** For someone learning an operator, the trace *is* the
  explanation: online softmax is "stream the row in, reduce it to `(m, l)`,
  keep `(m, l)` on chip, stream the row past it again to normalize, drain the
  result." You can see each clause.

## How

### The value plane already tracks occupancy

As tiles move, the executor propagates their payloads across levels on the
movement completion events — `copy_payload(DRAM→L3)` on a DMA arrival,
`L3→L2` on a BlockMover move, `L2→L1→COMPUTE` on a streamer feed, and the
reverse on drain/writeback/store — and **erases** an L3/L2 payload when the
buffer's credit is released (the final TagCAM reference disappears). So the set
of tiles staged at each level is already maintained; it just needed exposing.

### The observer: `tiles_at`

```cpp
std::vector<TileID> ConcurrentTimingExecutor::tiles_at(MemoryLevel level) const;
```

returns the tiles resident at a level, **sorted by `TileID`** so a rendered log
is deterministic and diffable. It is a pure read-only observer — it never
mutates executor state. `tile_arrival_cycle_at(level, id)` gives the stage-in
cycle. (For a timing-only run with no payloads set, occupancy is empty.)

### The renderer: `TileTracker`

`TileTracker::observe(exec)` reads `tiles_at` for L3/L2/L1/COMPUTE, and appends
a band **only if occupancy changed** since the last observation. That dedupe is
what turns a cycle-by-cycle sweep into a transition log. Output is a plain
string (`log()`), deterministic, hence testable.

### Driving cycle-by-cycle

Running a schedule to completion gives you the *answer*; to see the *journey*
you step the executor one cycle at a time and observe between steps:

```cpp
TileTracker tracker;
tracker.observe(exec);                       // initial state
while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles) {
    exec.step();
    tracker.observe(exec);                   // records only real transitions
}
std::cout << tracker.log();
// The cap is a guard, not a success condition: a deadlock or a wedged
// schedule also exits the loop. Check completion before trusting the trace.
if (!exec.is_complete()) {
    std::cerr << "schedule did not complete (deadlock or max_cycles)\n";
}
```

This is exactly what `softmax_simulator` does around the E8 online-softmax
schedule. The approach is **operator-agnostic**: any schedule (matmul,
reductions, norms, attention) can be driven the same way to get the same trace —
bind the value ops to the COMPUTEs, seed the inputs, step, observe. See
[`matmul_simulator`](tools/matmul-simulator.md) for the same view over a
tiled matmul, where two operand streams (A rows, B columns) interleave and a
C tile accumulates over its K-slices.

### Reading a band (worked example)

From `softmax_simulator --rows 3 --len 512` (three softmax rows, two tiles each).
Tiles use their 2D `[row, tile]` index (the softmax driver drops the unused K
axis). Five bands, **excerpted** from the full trace:

```text
cyc    | L3 buffers                         | L2 banks                           | L1 / array
-------+------------------------------------+------------------------------------+-----------------------------------
36     | A[0,0]                             | -                                  | -
120    | A[2,1]                             | A[1,0] A[1,1] A[2,0]               | A[0,0]* A[0,1]* A[1,0]*
129    | A[2,1]                             | A[1,0] A[1,1] A[2,0]               | A[0,0]* A[0,1]* A[1,0]* B[0,0]=(-0.69,440)*
162    | -                                  | A[2,0] A[2,1]                      | A[1,0]* A[1,1]* C[0,0]* C[0,1]*
204    | -                                  | -                                  | A[2,0]* A[2,1]* C[0,0]* C[0,1]* C[1,0]* C[1,1]*
```

- **cyc 36** — the first input tile `A[0,0]` (row 0, reduction-tile 0) has
  arrived at L3.
- **cyc 120** — all three rows are in flight at once: `A[2,1]` still in L3,
  row 1's tiles in L2, row 0's tiles already in the array (`*`). That overlap is
  the schedule **pipelining** across the batch.
- **cyc 129** — the stats compute for row 0 has produced `B[0,0]=(-0.69,440)`:
  running max `-0.69`, normalizer `440`. It stays resident (its `*` persists) to
  feed row 0's apply computes — no DRAM round-trip.
- **cyc 162** — row 0 is done: its normalized outputs `C[0,0]`, `C[0,1]` have
  formed, and its inputs `A[0,*]` **and** its stat `B[0,0]` have **retired** from
  the array (no compute needs them any more). Only tiles still in play remain.
- **cyc 204** — rows 0 and 1 have produced their outputs `C[0,*]`, `C[1,*]`;
  row 2's inputs `A[2,*]` are still resident. The run ends when all three rows'
  softmax sum to 1 and the array drains empty.

### Tile liveness — the array holds only what is active

The L1/array column shows only tiles **currently in play**: a fed input while a
compute that needs it is still pending, a result until it drains, and a resident
tile (like softmax's `(m, l)`) until no pending compute holds it. When a compute
completes, its consumed inputs **retire** from the array unless another pending
compute still needs them (an A row-tile feeds several C tiles, so it lingers
until its last consumer fires); a result retires when it drains. So the array
fills as the pipeline warms, then drains to empty — you can watch a matmul's
`C` tiles leave the array and stream back out `L2 -> L3 -> DRAM` at the end.

(This retirement also fixes a real value-plane leak: the executor already frees
L3/L2 payloads on credit release, and now frees the transient L1/array copies
too, so a long simulation no longer grows memory with every tile it moves.)

### Notes and limits when reading larger traces

- **Columns truncate** at `TileTracker::Config::col_width` (default 34). Widen
  it, or use a smaller `--tile` / `--len`, for wide rows.
- The tracker only sees tiles once the **value plane** is active (payloads seeded);
  a pure timing run shows empty bands.

---

## Summary

Conceive the schedule as tiles of state migrating through a software-managed
hierarchy; expose per-level occupancy with a pure observer (`tiles_at`); render
it as horizontal, occupancy-change-deduped bands (`TileTracker`); and drive the
executor a cycle at a time so the bands form a transition trace. The result is a
human-legible, content-carrying, diffable record of *how* a credit-based
dataflow schedule actually executes — reusable for any operator.
