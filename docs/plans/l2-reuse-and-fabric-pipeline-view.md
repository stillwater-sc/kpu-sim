# L2 Reuse and the L1 + Compute-Fabric Pipeline in the Domain-Flow Debugger

**Date:** 2026-10-16
**Status:** In Progress (approved 2026-10-16: Q1-Q7 as recommended, Q3 and Q8 decided by the architect; step 1 merged #377; step 2 merged #378; step 2b merged #379; step 3 merged #380; step 4 in review)

Extends the domain-flow debugger (`docs/plans/domain-flow-debugger.md`, #363, complete) and
follows the pattern of L3 placement (`docs/plans/l3-placement-at-lb.md`). It answers the
architect's request (2026-10-16): show what L2 holds (its reuse) and what L1 and the compute
fabric (CF) hold (the operator running on the array, as fill, steady state and drain), next to
the L3 slots, so the system-level schedule can be read and debugged in one view.

## 1. Problem Statement

The domain-flow viewer draws L3 as addressed slots, but L2 and the fabric only as tile counts,
hatched "not modelled". The architect wants two more pieces of state on screen:

1. **L2 occupancy, which represents reuse.** A tile staged in L2 is fed to the fabric several
   times without coming back from L3.
2. **L1 + CF, the operator on the array.** L1 is not a reuse level: it generates the streams into
   the array. When the CF is fully pipelined, L1 holds up to three tiles at once, each partly:
   - the next operand tiles, streaming in (fill);
   - the operands of the running call (steady state);
   - the previous result, draining out (drain).

**Neither is in the model today, so the viewer cannot draw it honestly.** The page draws only
what the record holds (domain-flow plan §3.7), and the record holds only what the levels compute.

| Wanted | What the levels do today | Gap |
|--------|--------------------------|-----|
| L2 reuse | A `call` emits, per operand, Move (L3→L2) and then Feed (L2→CF), which **takes** the tile (`behavioral.hpp`, Feed `keep=false`). A tile lives in L2 from its Move to its Feed, so every use re-Moves from L3. The language has no L2 statement; `machine flat(...)` declares only `l3`. | **Reuse is zero by construction.** An L2 panel drawn today could show only transient tiles and repeated Moves. |
| L1 state | L-B has no L1 channel (`Chan {Dram, L3, L2, Cf}`); a Feed lands in `Cf`. `.tflow` has a pooled, unbounded `l1` station. | No L1 occupancy anywhere |
| Fill / steady / drain | L-B has no time; a Call is atomic. L-T1 times a Call as one leg, `macs / fabric_macs_per_cycle` (`transactional.hpp:211`), after its Feeds and before its Drain. A chain's calls run back to back, each a whole leg. | **No phases, no overlap.** A wavefront model exists (`stream/stream_signature.hpp`, `WavefrontTiming`: fill + reduce + drain), but no level uses it. The deployment carries `macs_per_cycle` but no array geometry. |

So this plan is mostly not a viewer change. Most of it is modelling that the viewer then reads.
That modelling comes in two independent parts with different owners:

- **L-T1 times the fabric pipeline (part B).** This is timing, so it belongs to the timed levels,
  never to L-B. It needs the array's geometry and a space-time mapping.
- **L2 residency becomes behavior (part A).** This is the L3 placement pattern applied one level
  down. Placement plan Q5 deferred it: "L2 and L1 slots follow the same pattern ... a later
  plan". The language says what is staged, L-B models it, and the record carries it.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          kpu-run --dflow  (one .dflow bundle)                        │
│                                                                                      │
│  ┌──────────────────────────────┐            ┌──────────────────────────────────┐    │
│  │  L-B  (oracle, step axis)    │            │  L-T1  (overlay, cycle axis)     │    │
│  │                              │            │                                  │    │
│  │  L3: addressed slots (done)  │            │  legs per action (done)          │    │
│  │  L2: staged residencies  [A] │  joined on │  slot waits (done)               │    │
│  │      (slot, reuse count)     │◄──action──►│  CF chains: in/out streams   [B] │    │
│  │  CF: open acc + its chain    │   index    │    drain north, own links       │    │
│  │      position (structural)   │            │  L1: tiles in flight, derived    │    │
│  │  NO time, NO phases          │            │      from recorded phase bounds  │    │
│  └──────────────┬───────────────┘            └──────────────┬───────────────────┘    │
│                 │ steps, residency, l2_*                    │ lt1_calls, lt1_chains  │
│                 ▼                                           ▼                        │
│        ┌──────────────────────────────────────────────────────────────────┐          │
│        │ .dflow v4/v5  (columnar)  ──►  dflow_check.py  (D*, L*, P*, S*)  │          │
│        └───────────────────────────────┬──────────────────────────────────┘          │
└────────────────────────────────────────┼─────────────────────────────────────────────┘
                                         ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ domainflow/index.html — machine view                                               │
  │                                                                                    │
  │  DRAM ──► L3 [bank 0: s0 s1 ..][bank 1: ..] ──► L2 [s0: A[0,k] x3][s1: ..] ──►     │
  │                                                                                    │
  │      L1 + CF[c]   ┌────────────── fill ──────────────┐┌── steady ──┐┌─ drain ─┐    │
  │                   │ A[1,0] ▓▓▓░░  B[0,1] ▓▓░░░       ││ A[0,3]·B.. ││ C[0,0]▓░│    │
  │                   └──────────────────────────────────┘└────────────┘└─────────┘    │
  │      step axis: the structural panel (what is fed, the acc, k of K)                │
  │      cycle axis: the three phases, with each tile's streamed fraction              │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

## 3. Design

### 3.1 Part B: L-T1 times the fabric as a pipeline

**Geometry is a machine parameter.** The deployment gains each compute tile's array shape:

```json
"fabric": { "pe_rows": 128, "pe_cols": 64 }
```

`macs_per_cycle` is ONE compute tile's rate, so validation is `pe_rows * pe_cols == macs_per_cycle`,
or the deployment is refused by name. The key is `fabric`, because `array` already names the grid
of compute tiles. A deployment without `fabric` runs as before: every call one leg, and the overlay
says so. Step 2 declares it on kpu_s1 only, `128 x 64` (decided 2026-10-16); the other deployments
gain it when their arrays are stated.

**Which calls stream (decided 2026-10-16).** A gemm in an open accumulator (`acc C in fabric`)
whose C tile fits the array. Its C tile occupies `r x c = min(tile, array)` PEs and a call takes its
reduction depth in cycles, `macs / (r*c)`, so a tile smaller than the array leaves the other PEs
idle (kpu_s1's 64x64 tiles use half its 128 rows). Every other call keeps one leg at
`macs / macs_per_cycle`, counted by why in `lt1.one_leg`:
- getrf, trsm, laswp and the vector tile functions ("not a gemm"): their array mappings are not
  designed;
- a gemm on a resident C ("in-place gemm (C preloaded)"): loading C into the PEs is a stream no
  one has designed;
- a C tile larger than the array: it would take several passes, each holding a different part
  of C, which breaks the chain's one-C-per-array rule;
- a chain opening while every compute tile holds an open accumulator.

**The mapping is output-stationary.** That is what `acc C[i,j] in fabric { for k ... }` already
means: C stays in the PEs across the chain, and A and B stream through. Other mappings
(`SpaceTimeMap::b_stationary`, ...) come later and are a schedule choice, never a deployment field
(memory: schedule vs machine parameters).

**The fabric drains through the network (architect, Q3, 2026-10-16).**
- PEs have no double-buffered result registers.
- Once a PE's reduction ends, its result is pushed into the NEWS grid in the direction opposite
  the wavefront: north, against B (Q8). It travels on the northbound links, in the grid's
  registers, which carry operands
  and results at the same time because each stream has its own spatial allocation.
- So a drain never holds up the next tile's operands. The next chain's wavefront enters right
  behind the last one, and the array stays busy. That is the target: deep pipelining at 100%
  utilization.
- This is standard output-stationary systolic scheduling. A weight-stationary array such as the
  TPU's does not drain this way, and this model must not be read as one.

**Two streams per chain on an R x S array, reduction depth K cycles (tile = array; §3.1.1):**

PE(i, j) works on a chain during `[in0 + i + j, in0 + i + j + K)`: the wavefront reaches it
`i + j` cycles after the corner, and it then does its K MACs. With `skew = (R-1) + (S-1)`:

```
in0        the chain's first operands reach PE(0,0)
front      = in0 + skew        the wavefront reaches the far PE, PE(R-1, S-1)
in1        = in0 + K           PE(0,0) finishes; its result leaves north, and the next chain's
                               first operands may enter it (each row and column stream runs
                               back to back from chain to chain)
last       = in0 + K + skew    the far PE finishes
out1       = last + E          the far PE's result leaves the array (E: §3.1.1)

fill       [in0, front)        PEs joining the chain
tail       [in1, last)         PEs leaving it, each result starting north as it finishes
steady     [front, in1)        every PE on this chain; EMPTY when K <= skew
results    [in1, out1)         on the northbound links
```

The phases are not a partition. When `K <= skew` (a short reduction, or a large array) the
chain never occupies the whole array: its fill and tail overlap during `[in1, front)`, and
`steady` is empty. The array is still busy, because each PE goes from one chain to the next
without a gap (the next chain's wavefront follows at `in1`), so utilization does not depend on
`K > skew`; only the single-chain picture does. The viewer draws the phases as overlapping
bands, never as a sequence.

**The pipeline is chain-aware.** This is the point of output-stationary and of the architect's
"three tiles":
- Within an `acc` chain, the k-steps stream back to back as one wavefront. C never leaves the
  array, so only the chain fills, once, and only the chain drains, once. A call has an injection
  interval, not phases of its own.
- Across chains on one CF, the next chain's operands start at the previous chain's `in1`. Its
  fill runs while the previous results drain north on their own links. That is the moment L1 holds
  three tiles: the next A and B streaming in, the running operands' tails, and the previous C
  streaming out.
- The one structural limit is the northbound link, and it is checked, never assumed:
  - the link out of each column's top PE carries that column's `r` results per chain;
  - so a chain whose results arrive faster than that would overtake the previous chain's;
  - its results wait in the grid's registers, which hold outputs beside inputs, so the limit
    falls on the RESULTS, never on the operands: `out1 >= previous out1 + r`, and
    `link_delay = out1 - (last + r)` is recorded;
  - delaying the operands instead would need the chain's K before its last call, and would
    contradict "a drain never stalls the next tile".
  With the matmul tiles in use (K >= r) the delay is zero.

```cpp
// WRONG: phases painted inside the existing leg, so the timing stays as it was and the panel lies
const Cycle d = macs / rate;
call.fill = {start, start + d / 10};            // invented split, no model behind it

// WRONG: a fill and a drain on every call of a chain (C leaves the array each k-step;
// that is not output-stationary, and it multiplies the chain's latency)
for (auto& call : chain) { t += fill; t += steady; t += drain; }

// WRONG: the next chain waits for the previous drain (double-buffer thinking; a TPU-style
// array). In this fabric results leave on the northbound links while operands keep entering.
next.in0 = std::max(ready, prev.out1);

// CORRECT: the streams follow from the geometry and the chain, and they ARE the timing
struct ChainStreams { Cycle in0, front, in1, last, out1, link_delay; };
ChainStreams streams_of(const Chain& ch, const ArrayShape& a, Cycle ready, Cycle prev_in1, Cycle prev_out1) {
    const Cycle K    = ch.steady_cycles();                       // sum of macs/rate over its calls
    const Cycle skew = (a.rows - 1) + (a.cols - 1);
    const Cycle E    = a.drain_hops();                           // §3.1.1
    ChainStreams s;
    s.in0        = std::max(ready, prev_in1);                    // operands never wait on a drain
    s.front      = s.in0 + skew;                                 // may lie after in1 (K <= skew)
    s.in1        = s.in0 + K;
    s.last       = s.in1 + skew;
    s.out1       = std::max(s.last + E, prev_out1 + E);          // the top link: E = r results
    s.link_delay = s.out1 - (s.last + E);
    return s;
}
// A call's own interval is its share of [in0, in1) at PE(0,0), in chain order; at PE(i, j) the
// same share runs i + j cycles later. Its Drain leg (str, L1 -> L2) is ready at out1, where it was
// ready at the Call leg's end before.
```

**The timing changes and the values do not.** L-T1's cycles move, since it is uncalibrated and
says so. Values are bit-identical to before. This is a timing-only change, which is where Q4 of
retire-l0-derivations requires bit equality. Tests that pin L-T1 cycle counts are rebaselined in
the same PR, each with the reason.

**L1 occupancy is derived from recorded bounds, never stored per cycle.**
- An operand tile is in L1 from its Feed leg's finish until its last row (A) or column (B)
  enters the array: the end of its call's share at PE(0,0), plus `R-1` (A) or `S-1` (B).
- A result tile is in L1 from `in1` (the first result arrives) to its Drain leg's finish.
- The streamed fraction of a tile at cycle t is linear within its interval. That is the model
  (`element_stride = 1`, dense streams), and the page labels it "L-T1 model, linear stream".

**Record (.dflow version 4):** two new tables, and nothing in L-B's tables changes.

```
lt1_calls.bin   action (join key), chain id, k of K, inject0, inject1   (its share of the chain)
lt1_chains.bin  chain id, cf lane, in0, front, in1, last, out1, link_delay
                (cycles; both tables empty, and meta says why, when the deployment has no array)
meta            + array_rows, array_cols, drain_edge = "north", mapping = "output-stationary"
```

#### 3.1.1 Assumptions of the stream model

- **Tile = array.** A tile larger than the array is an integer number of passes, each its own
  in and out streams, back to back on the lane. A tile that is not a multiple of the array is
  refused for the overlay by name (the run still runs; the overlay records no streams).
- **Links are unidirectional (architect, 2026-10-16).**
  - The grid is NEWS-connected, and every PE edge has two physical links, one in each
    direction.
  - In output-stationary, A travels east and B travels south, each on its own link.
  - Results travel north on the northbound links, which no operand stream uses.
  - So a drain shares a PE edge with B but never a link: the only contention is results queuing
    behind other results on a column's northbound links. That is the `link_delay` above.
- **The drain edge is north (Q8).** A result of PE(i, j) travels i hops north, plus one to
  leave the array, into L1's drain port on the north edge.
  - The farthest is PE(R-1, j), so `E = R`.
  - North is the model, not a deployment option. The record's meta says `drain_edge = "north"`,
    so a later model with another edge reads as different.

### 3.2 Part A: L2 residency is behavior

The same contract as L3 (ADR 0004 §5): the program says what is staged, the walker allocates the
address, L-B executes it and refuses what breaks it, the trace records it, and no level invents
one.

**Language:**

```
program mm machine flat(l3 = 17, l2 = 4) {          // l2: slots per compute tile's L2
  ...
  for i in 0..4 {
    stage A[i, 0..4] in l2 {                         // Move once; the Feeds below keep the copy
      for j in 0..4 {
        acc C[i, j] in fabric {
          for k in 0..4 {
            resident B[k, j];
            call gemm(A[i, k], B[k, j]) +-> C[i, j]; // A[i,k]: Feed only (staged); B: Move + Feed
            release B[k, j];
          }
        }
      }
    }                                                 // end of block: the L2 slots free (credit up)
  }
}
```

**Actions:** a `stage` emits the Move at block entry (L3 → L2, into an `L2Address{cf, slot}`).
A call on a staged tile emits only a Feed, which keeps the copy. Block exit emits `L2Release`,
which is the credit returned upstream. Unstaged operands lower exactly as today (Move + Feed,
transient), so every existing program is unchanged.

```cpp
// WRONG: L2 as a cache. Look the tile up, and Move it if absent.
if (!l2.contains(k)) emit(Move, t);             // demand fetch; a "hit" decided at run time
emit(Feed, t);

// CORRECT: the program placed it. The walker knows it is staged and where, so L-B checks it.
if (const L2Address* s = staged(t)) emit(Feed, t, *s);          // keep=true at L-B
else { emit(Move, t, place_l2_transient(t)); emit(Feed, t); }   // today's lowering
```

**L-B refuses:**
- a Feed naming an L2 slot that does not hold the tile;
- a stage into a live slot;
- staging more than `l2` slots;
- a program that ends with a staged tile.

These are the L3 checks of placement step 3, one level down.

**Trace:** the L0 opset gains `L2_STAGE` and `L2_RELEASE`, the way opset 1.2 gained `L3_*`. The
replay (`TileProgramReference`) checks them, and the compute-schedule analyses skip them.

**Record (.dflow version 5):** an `l2_*` group like L3's.
- steps gain `l2_cf`, `l2_slot`, and an opens/closes flag;
- a `l2_residency` table, each row with its Feed count (the reuse count, read by the page, never
  computed by it).

**Timing:** L-T1 holds an L2 credit per slot, the way step 5a holds L3's. L-CA follows with
placement step 6. Both are listed here for completeness and gated on the questions below.

### 3.3 The viewer

The machine view gains two panels. Every value comes from the record (domain-flow plan §3.7).

- **L2 panel.**
  - *Before part A:* a list of the tiles L2 holds at this step, and per tile its "Moves from L3"
    so far. This is honest: it shows the re-Moves that staging would remove.
  - *After part A:* L2 slots per compute tile, as L3's are drawn, each with its tile, its
    residency and its reuse count. A slot is outlined when the action touches it.
- **L1 + CF panel, per compute tile.**
  - *Step axis (L-B):* structural. It shows the call's operands and result, the open accumulator,
    its chain position (k of K), and whether the call opens or closes the chain. It shows no
    phase, because L-B has no time.
  - *Cycle axis (L-T1 overlay):* two concurrent streams per lane, not serial phases:
    - operands in, with fill, steady and tail as overlapping bands (steady may be empty);
    - results out, northbound, drawn under the next chain's fill.

    Each tile in L1 has a bar showing its streamed fraction. The ribbon gains two CF rows per lane
    (in, out) and shows the lane's utilization. Clicking a stream lands on its call or chain.
- **Navigation:** next chain start or end on this CF; next reuse of the selected L2 tile; next
  three-tile moment (a fill overlapping a drain).

```js
// WRONG: the page infers phases or reuse
const fill = (call.t1 - call.t0) * 0.1;                     // invented
const reuse = steps.filter(s => s.kind === 'feed' && s.tile === t).length;   // after part A,
                                                            // the record carries the count
// CORRECT: read the record; the only arithmetic is the documented linear fraction
const c = core.callStreams(action);                         // lt1_calls + lt1_chains, or null
const inFrac  = c ? clamp01((t - c.inject0) / Math.max(1, c.inject1 - c.inject0)) : null;
const outFrac = c ? clamp01((t - c.chain.in1) / Math.max(1, c.chain.out1 - c.chain.in1)) : null;
```

Every new string reaching HTML (tile names, CF lane names, mapping) goes through `esc()`. The
strict allow-list scan in `test_viewer.py` (domainflow and tileflow both; memory: viewer HTML
escaping) covers every new `${...}`, and a new trusted helper gets a test that it escapes.

## 4. Implementation Steps

Each step is one PR. Steps 1 and 2 need no language change. Steps 3 and 4 depend on Q1.

1. **The structural L2 and CF panels (viewer only, .dflow unchanged).** As built: L2's and the
   fabric's tiles are derived from the recorded actions and checked against L-B's held counts at
   every step (all eight test bundles agree everywhere). An L2 tile shows its Moves from L3 so far.
   The panel shows the call's operands and result and its chain position. Navigation: next chain
   start and end, and `#step=N` links. The strict escape scan is ported to the domain-flow test.
   - `tools/visualization/domainflow/index.html`:
     - the L2 tile list with Moves-from-L3 counts;
     - the L1 + CF structural panel (the call, the acc, k of K, chain open/close);
     - chain navigation;
     - the hatching keeps saying "L2 unbounded at L-B".
   - `tools/visualization/domainflow/test_viewer.py`: the data layer's L2 list and chain
     positions against Python; the strict escape scan, ported from tileflow.
   - `README.md`.
2. **L-T1 fabric pipeline and the cycle-axis panel (.dflow v4).**
   - `include/sw/kpu/program/platform/deployment_spec.hpp`: `fabric {pe_rows, pe_cols}`,
     validation against `macs_per_cycle`, and `device_view` carries it.
   - `tests/program/deploy/kpu_s1.json` gains `fabric` (128 x 64); the others when stated.
   - `include/sw/kpu/program/csp/transactional.hpp`: chains known per CF lane, `streams_of`
     per chain, each call's injection interval, and the Drain leg ready at `out1`. Feed legs are
     unchanged. The record grows `chain_streams`.
   - `include/sw/kpu/program/record/domain_flow_record.hpp`, `src/program/domain_flow_record.cpp`:
     `lt1_calls`, `lt1_chains`, meta fields, `kDflowVersion = 4`. Versions 1-3 are still read.
   - `tools/kpu-run/kpu_run.cpp`: attach the streams. The text report gains fabric
     utilization per lane (steady cycles over the lane's span) and any link delays.
   - `tools/trace/dflow_check.py` (+ self-test): P1-P5 (§6).
   - Viewer: the phase lanes, L1 bars, CF ribbon rows, three-tile navigation; versions [1..4].
   - Tests:
     - `tests/program/test_domain_flow_record.cpp`;
     - a new `tests/program/test_fabric_pipeline.cpp`: one chain fills once and drains once;
       the next chain's operands start at the previous `in1`, under its drain; a chain with
       K < r has its results wait `r - K` on the top link; back-to-back chains reach utilization
       `sum K / (last in1 - first in0) == 1`; values are bit-equal to before;
     - rebaselined L-T1 cycle pins, each with its reason.
   As built (2026-10-16):
   - The deployment key is `fabric {pe_rows, pe_cols}`; kpu_s1 declares 128 x 64.
   - A chain owns a compute tile while its calls run. It gives the array up at its Drain, or when
     the next accumulator's first call needs that compute tile: its results leave as they emerge,
     so the Drain action only moves them L1 -> L2. Calls that come back to a chain whose array was
     taken are one leg ("accumulators interleaved on one array").
   - With a declared fabric, the streamer's drain legs run on their own lanes (str lanes [n, 2n)),
     issued in order among drains only, so a drain never waits behind a feed (decided 2026-10-16).
   - Finding: every mover issues in program order, and `store C` right after its chain puts C's
     Writeback (bm) and Store (dma) ahead of the next chain's Moves and Loads. Both wait for the
     Drain, so a store-each-chain program serializes its chains: the fabric could overlap them, the
     program does not let it. `test_fabric_pipeline` pins this; a program that defers its stores
     runs back to back at full array utilization. kpu-csp-gen's matmul stores each chain, so its
     runs show no overlap until step 2b.
2b. **`store ... late` (decided 2026-10-16; its own PR).** A modifier on the store statement,
   `store C[i, j] late;`, written where the store is today. The validator checks it in place; the
   walker emits its Drain, Writeback, Store and Release after the NEXT accumulator's calls (at the
   end of the program for the last). kpu-csp-gen's matmul then writes `late`, so its drains
   overlap the next chain's fill. The loop-carried alternative (`store C[i - 1, j]` with peeled
   loops) was rejected: the symbolic validator would need loop-carried accumulator state.
   With L3 at the generator's minimum (2 panels + 1), the next A row still waits for the slots the
   current row frees, which the overlay shows as slot waits; a larger L3 removes them.
   - As built:
     - A statement that names the tile again (`resident` or `acc`) first emits the pending late
       store, and every one before it. Unfused linear depends on this: its second pass loads C
       back, so it needs the last chain's store to reach DRAM first.
     - The emitted actions carry the store's own line and loop indices.
     - The symbolic validator checks the store in place. When a program has a late store, it also
       requires a free slot at every acc open and close and at the program's end, which is where
       the writeback can land.
     - A late store refuses `via`: its epilogue's vector would need to be resident where the store
       is emitted.
     - It refuses a resident tile and a second store.
     - kpu-csp-gen writes `late` in the matmul nest: matmul and unfused linear. Fused linear does
       not, because its store carries the epilogue.
   - Finding: `late` defers by ONE chain. The block mover issues in order, so the previous chain's
     Writeback still leads the chain after next's Moves. The chains run back to back only while a
     chain's calls outlast the chain ahead's drain (skew + rows). `test_fabric_pipeline` pins both
     regimes: 8 k-steps of 32 on 64x64 pipeline fully, and 4 do not. On kpu_s1 a 512 matmul chain
     (~1,856 cycles) far outlasts its drain (~254).
   - Measured: kpu-csp-gen's 512 matmul in 64-tiles on kpu_s1 (L3 17):
     - L-T1 makespan 190,592 -> 166,014 cycles;
     - mean gap between chains 1,114 -> 724 cycles.
     - The chains still do not overlap: the one block mover is busy 147,456 of 166,014 cycles,
       moving each chain's A and B tiles out of L3 again. That is step 3's L2 reuse.
3. **L2 staging in the language and at L-B (.dflow v5, opset 1.3).**
   - `csp/lang/` (parser, AST, `walk.hpp`): `l2 =`, `stage ... in l2 { }`, `L2Address`,
     allocation, and the Feed-only lowering.
   - `csp/behavioral.hpp`: addressed L2 storage per CF, with Feed `keep=true` on a staged tile and
     the four refusals.
   - `serialize/l0_format.hpp` and the replay: `L2_STAGE`, `L2_RELEASE`.
   - The record's `l2_*` group; `dflow_check.py` S1-S4; the viewer's L2 slots with reuse counts.
   - `kpu-csp-gen --algo matmul --stage-a` emits a staged program.
   - Tests:
     - staged and unstaged programs compute identical values (bit-equal, the same arithmetic);
     - each refusal by name;
     - the replay agrees;
     - the staged matmul's Moves from L3 drop by the reuse factor.
   - As built (decisions 2026-10-16):
     - **The L3 source stays live.** A staged tile's L3 residency stays live for the whole block,
       and releasing, retaining or writing it there is refused. Staging cuts block-mover traffic,
       not L3 capacity.
     - **L-CA re-Moves from L3.** Until the executor's L2 joins placement step 6, L-CA runs each
       staged Feed as a Move from the tile's live L3 slot plus a Feed, skips the staging Move, and
       treats L2_RELEASE as nothing. The values are the same and the timing is the unstaged
       program's. It reports "L2 staging: unmapped at L-CA" and counts the staged Feeds.
     - **Index forms.** `stage` takes the existing tile forms (`A[i, :]`, `A[i, k]`), not the
       range index (`0..4`) this section's sketch used.
     - **Slot allocation.** The lowest free slot of compute tile 0's L2 (level 1 has one fabric).
     - **The CSP action.** A Move or Feed carries an `L2Address` and a staging index. The new
       action kind `L2Release` takes no process, like Release.
     - **L0 ops.** L2_STAGE is recorded after the stage Move's L3_MOVE, and L2_RELEASE at the
       block's end. `opset_for` writes 1.3.0 only when a program stages.
     - **L-T1.** A staged Feed waits for its stage Move, and L2Release is a zero-length credit
       record. L2 credits are step 4.
     - **The record.** `.dflow` v5 adds the steps' `staging`, `l2_cf`, `l2_slot`, flags `kHasL2`
       and `kStages`, the `l2_staging` table, and `machine.l2`. dflow_check adds S1-S4. The
       viewer keeps a staged copy in L2 when it is fed and shows its slot and reuse count.
     - **The trace printer.** It rebuilds `stage` blocks from the stage Moves and L2Releases.
     - **kpu-csp-gen.** `--stage-a` implies the row schedule (A's panel outer) and declares
       `l2 = tiles per panel`.
   - Measured: kpu-csp-gen's 512 matmul in 64-tiles on kpu_s1 (L3 17, row schedule, late stores):
     - Each A tile leaves L3 once per row, not once per call: 448 fewer Moves (64 stagings serve
       512 Feeds), plus 64 L2Releases, so 3,968 actions become 3,584.
     - L-T1 makespan 157,950 -> 106,622 cycles; block-mover busy 147,456 -> 90,112.
     - Each chain's calls take 960 cycles instead of 1,856.
     - Values are exact at L-B, L-T1 and L-CA. L-CA's makespan is the unstaged run's (staging is
       unmapped there).
     - Finding: the chains still barely overlap (1 of 63). The mean gap of 682 cycles is the next
       B panel's reload. The program releases B[:, j] before it makes B[:, j+1] resident, so the
       panel comes from DRAM after the chain ahead frees its slots. Prefetching it is
       cross-iteration residency, beyond the symbolic validator today.
4. **L-T1 times L2 credits; `.tflow` v6 draws L2 slots and CF phases.**
   - L-T1: an L2 slot credit per staged residency, released at `L2Release`.
   - `tile_flow_record.cpp`: L2 slot rows after the L3 slot rows, and CF phase rows.
   - The tileflow viewer: versions [3..6].
   - `tflow_check.py`: TF13 (an L2 slot never holds two residencies at once).
   - As built (2026-10-16):
     - **L-T1.** A stage Move waits for its own L2 slot's `L2Release` (`TransactionalInterpreter::
       L2Slot`), as a Load waits for its L3 slot. The copy is held from that credit to its
       `L2Release`, and its Feeds are counted. The outcome carries `l2_slots`, `peak_l2`,
       `l2_bound` and `l2_wait_cycles`, and kpu-run prints an `L2` line. A transient Move holds
       no L2 credit.
     - **`.tflow` v6**, written for every addressed CSP record, with two tables of its own rather
       than rows in `residency`, so the L3 checks (TF1, TF5, TF6, TF9, TF12) are unchanged:
       - `l2_residency`: tile, cf, slot, t0, t1, feeds;
       - `chain`: tile, station, rows, cols, calls and the five phases;
       - the manifest's `l2_slots`.

       With `l2 > 0`, the L2 station is modelled, with the program's slots as its capacity. The
       pyramid adds, after the L3 slot rows, one `l2_slot` row per slot and a `cf_in` and a
       `cf_out` row per compute tile.
     - **tflow_check.**
       - TF13: the slot within `l2_slots`, no overlap per slot, at most `l2_slots` held, each copy
         fed.
       - TF14: `front`/`last` equal `in0`/`in1` plus the skew, one operand stream per compute tile,
         and the top link exactly, as P5: `out1 = max(last + rows, previous out1 + rows)`.
       - TF8 recomputes the new rows from the raw tables.
       - New self-test: `test_tflow_check_v6.py`.
     - **The viewer** reads versions 3-6. It shows the L2 slot rows with each copy's Feeds and the
       compute-fabric rows with each chain's phase, and has an L2-reuse diagnostic.
   - Measured: the 512 staged matmul on kpu_s1, the step-3 program.
     - Makespan is unchanged at 106,622 cycles. Peak L2 is 8 of 8, over 64 stagings.
     - The L2 credit never decided a stage Move's start (`l2_bound` 0). The program releases a
       row's copies before it stages the next row, and the block mover is in order, so the next
       stage Move issues after the last Feed of the slot's previous copy anyway. The credit holds,
       and binds only when a program stages faster than it feeds.
     - The fabric rows show the step-3 finding directly: 64 chains, operand streams busy for
       61,440 cycles, a mean 682-cycle gap between chains, and one chain whose results overlap the
       next chain's operands.

L-CA's L2 credits join L3's in placement step 6. That is out of scope here.

## 5. Verification

```bash
nice -n 19 cmake --build build -j 2
cd build && ctest -j 4 --output-on-failure            # all green, rebaselines explained

# step 1: viewer only
python3 tools/visualization/domainflow/test_viewer.py

# step 2: the pipeline on the overlay
kpu-csp-gen --algo matmul --size 256 --tile 64 --l3 17 -o mm256.csp
kpu-run --program mm256.csp --deploy tests/program/deploy/kpu_s1.json \
        --level block-sequential --dflow mm256.dflow
python3 tools/trace/dflow_check.py mm256.dflow        # exit 0; P1-P5 pass
python3 tools/visualization/domainflow/pack.py mm256.dflow -o mm256.html
#   expect: 16 chains (4x4 C tiles, 4 calls each); 16 fills and 16 drains, not 64 of each;
#   on each lane every chain but the first starts at the previous in1, under its drain;
#   no link delays; values bit-equal to the run before this step (lt1_tiles unchanged)

# step 3: staging
kpu-csp-gen --algo matmul --size 256 --tile 64 --l3 17 --stage-a -o mm256s.csp
kpu-run --program mm256s.csp --deploy tests/program/deploy/kpu_s1.json --level all \
        --dflow mm256s.dflow --emit-l0 mm256s.l0
#   expect: A Moves from L3 drop from 64 to 16 (reuse 4); values bit-equal to mm256.csp;
#   the replay of the L0 passes; dflow_check S1-S4 pass
```

The pages are reviewed in a browser as packed bundles, as for #363. A visualization page is shown
for review as soon as step 1 renders.

## 6. Key Invariants

- **Every value on screen comes from the record.** The page computes two things, both stated on
  the page:
  - which tiles L2 and the fabric hold (step 1). The record counts them (`held`) but does not name
    them, so the page replays the recorded actions exactly as L-B moves tiles, and checks the
    replay against `held` at every step. On a disagreement it shows no contents;
  - the linear streamed fraction within recorded stream bounds (step 2), labelled as L-T1's model.
- **L-B has no time:** no phase, fraction or overlap is drawn on the step axis.
- **Decomposition never changes values:**
  - the pipeline changes L-T1 cycles only, and L-T1's values stay bit-equal to before;
  - staged and unstaged programs compute bit-equal results.
- **dflow_check, part B:**
  As built (`tools/trace/dflow_check.py`):
  - **P1** References: a chain's tile, cf lane and actions exist and its PEs fit the array; a call
    row names a recorded Call. With no fabric declared, both tables are empty.
  - **P2** The wavefront: `front - in0 == skew`; drained, `last - in1 == skew` and
    `out1 - last == r + link_delay`. `front` and `in1` are not ordered: `front > in1` exactly when
    `K < skew`, and then the chain has no steady phase.
  - **P3** The calls: numbered 1..calls in action order, each on the chain's cf lane, the first
    starting at `in0` and the last ending at `in1`, never overlapping. Gaps are allowed: a call
    whose operands arrive late leaves a bubble in the wavefront.
  - **P4** One chain per array: on one cf lane, operand streams `[in0, in1)` never overlap, and a
    chain's Drain starts no earlier than its `out1`. Result streams may overlap operand streams;
    that is the fabric.
  - **P5** The top link: `out1 == max(last + r, previous out1 + r)` on its lane, and the manifest's
    chain count and `link_delay_cycles` are the tables'.
- **dflow_check, part A:**
  - **S1** One residency per L2 slot at a time.
  - **S2** A Feed of a staged tile names the slot that holds it.
  - **S3** A residency's recorded reuse count equals its Feeds.
  - **S4** The run ends with every L2 slot free.
- **Credit semantics:**
  - L2 is a buffer the program stages into, and a stage's end returns the credit upstream;
  - nothing is looked up, nothing "hits", nothing is re-fetched on demand;
  - L1 is a stream buffer with no reuse.

## 7. Open Questions

- **Q1. Is L2 staging (part A) in this plan or its own? DECIDED (2026-10-16): as recommended.**
  - **Recommended:** keep it here as steps 3-4. The viewer's L2 shape depends on it, and it is
    small next to L3 placement, because it reuses that machinery (the address, the walker
    allocation, the refusals, the opset pattern).
  - Alternative: a separate `l2-placement-at-lb.md`, with this plan stopping at step 2 plus the
    honest L2 list.
- **Q2. Where does array geometry come from? DECIDED (2026-10-16): as recommended.**
  - **Recommended:** the deployment (`compute.array`), validated against `macs_per_cycle`, since
    it is a machine parameter.
  - Alternative: derive a square array from `macs_per_cycle / compute_tiles`. That is silent
    when the result is not square, so it is not recommended.
- **Q3. The overlap model. DECIDED (architect, 2026-10-16).**
  - No double-buffered result registers. Results drain into the NEWS grid opposite the
    wavefront, as soon as they emerge, and the grid's registers carry operands and results at
    once by spatial stream allocation. Drains never stall the next tile's operands, and the
    array is driven toward 100% utilization (§3.1).
  - Output-stationary systolic scheduling, not a weight-stationary (TPU-style) array.
- **Q4. Do L-T1's cycles change? DECIDED (2026-10-16): as recommended.**
  - **Recommended:** yes. The phases are the timing, and painting them inside the old leg would
    show a pipeline L-T1 does not run.
  - The cost is rebaselined cycle pins in step 2.
- **Q5. Which mappings? DECIDED (2026-10-16): as recommended.**
  - **Recommended:** output-stationary only, which is what `acc ... in fabric` means.
  - Weight-stationary and others need a program-side mapping choice: a later plan.
- **Q6. Which viewer first? DECIDED (2026-10-16): as recommended.**
  - **Recommended:** the domain-flow viewer (steps 1-3). It already draws L3 slots next to the
    program, which is the context the request names.
  - The tile-flow viewer follows in step 4.
- **Q7. The staging syntax. DECIDED (2026-10-16): as recommended.**
  - **Recommended:** a block, `stage T[...] in l2 { }`, since a block's end is the release and
    cannot be forgotten.
  - Alternative: `stage` / `unstage` statements, mirroring `resident` / `release`, which allows
    staggered lifetimes (staging the next row of A while the last columns of the current one
    run).
- **Q8. The drain edge. DECIDED (architect, 2026-10-16).**
  - North, against B, so `E = R`.
  - Links are unidirectional, two per PE edge, one each way, so the northbound result stream
    never shares a link with an operand stream (§3.1.1).
