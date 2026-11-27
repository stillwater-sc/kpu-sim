# The cycle-based KPU prototype: an evaluation (Phase 3)

**Date:** 2026-11-26
**Status:** For the user's decision (decision 5)
**Plan:** `docs/plans/cycle-based-kpu-prototype.md` §6, issue #548 (epic #532)

## 1. What exists now

The cycle core runs a CSP program from DRAM to DRAM on one simulator (`kpu-run --system`, step 3c):

```
DRAM -> DMA -> L3 tile -> BlockMover -> L2 -> Streamer -> L1 lanes -> PE array
     -> north FIFOs -> Combiner -> L2 -> BlockMover -> L3 tile -> DMA -> DRAM
```

Every rule is built as state, from the RTL's design (embodied-ai-soc-generator `b57c881`). None is
a formula. The checks:
- **Data.** DRAM ends with L-B's result, bit for bit. C reaches it from the array: nothing is
  hypothesized.
- **Invariants every cycle:**
  - the L3 slots' credits, three ways (the DMA engine, the BlockMover, the L3 tile);
  - the L2 slots' credits, three ways (the BlockMover, the Streamer, the L2);
  - each component's own checks: the L1 credit, the bank ports, the slot states, the FIFOs;
  - a north collision, a late row or a slot fault stops the run.
- **The operational laws** on every component and link, and the PE-clocks by cause adding up.
- **The deadlock detector,** with a state dump.

Phase 1 (#537–#541), Phase 2 (#542–#547) and Phase 3's steps 3a–3c (#599, #600, #601)
built it.

## 2. Where the cycles go

### 2.1 The int8 64³ on `kpu_s1` (128×64 array, 16×16 tiles)

| Run | KPU clocks | What bounds it |
|---|---|---|
| The compute subsystem alone, the L3 preloaded (`--compute-subsystem`, `l3 = 96`) | 5824 | the BlockMover and the L2 slots |
| Joined, `l3 = 96` | 5897 | the same: DRAM adds 73 clocks |
| Joined, `l3 = 32` or `l3 = 16` | 5897 | the same |
| **Joined, `l3 = 9`** (Phase 1's own program) | **6905** | **L3 capacity:** the DMA engine waits for a slot's credit 6257 cycles, and the 9 slots average 7.3 live |

**Three findings:**
1. **L3 capacity binds below 16 tiles.** At 9 tiles the run is 17% longer. From 16 up, more L3
   changes nothing.
2. **Above that, the L2 slots and the BlockMover bind.** The program stages nothing in L2, so every
   call moves both its operand tiles: 128 Moves for 64 calls. The BlockMover waits for an L2 slot's
   credit for about 1950 clocks. The Streamer waits for tiles for 1132 clocks, about 18 per command.
   Commands fire about 68 clocks apart, where the array needs 16.
   - This is the reuse the program states, and the cost of its absence (see `reuse-is-deliberate`).
     A program that stages A's row panel in L2 (`stage_a`) or keeps B stationary would move each tile
     once.
3. **The array is nearly idle: 0.54% utilization.**
   - A 16×16 tile on the 128×64 array can use at most 16·16 / (128·64) = 3.1% of the PEs, the "tile on
     a corner".
   - Within that, the commands' spacing (about 68 clocks, against 16) leaves the tile's PEs busy about
     a fifth of the time.
   - The rest of the PE-clocks: 1.6% padded (A with no B, the columns past the tile), 0.25% draining,
     97.6% waiting on L1.

The L2 banks are not the bound at this rate. They average 1.33–1.64 reads in flight, 8 at most. That
is #495's question, measured: with 4 banks of 2 read ports at 250 MHz they keep up with one A row and
one B row a clock.

### 2.2 The RTL's 8×8×16 on `kpu_soc_2x2` (4×4 array, 4×4 tiles)

| Run | One L2 buffer (`l2 = 3`) | Two (`l2 = 6`) | Two over one |
|---|---|---|---|
| The RTL (`test_credits.py`) | 907 | 744 | 0.82× |
| Today's L-CA (`ConcurrentTimingExecutor`) | 988 | 952 | 0.96× |
| The compute subsystem alone (L3 preloaded) | 1060 | 688 | 0.65× |
| **The joined cycle core** | **1107** | **723** | **0.65×** |

Double buffering pays here as it does in the RTL. Today's L-CA shows almost none of it, which is
#401's open item. The cycle core shows more than the RTL. It is slower single-buffered (1107
against 907) and faster double-buffered (723 against 744).

**The numbers are not calibrated, so they don't match the RTL's.** The differences we know of:
- **Clocks:** the core runs the array at 1 GHz and the L2 at 250 MHz, with crossings between them. The
  RTL's test counts one clock.
- **The start:** the RTL counts from the simulation's start, including its 7 CONFIG writes and START.
  The core starts with A and B in DRAM, and the DRAM fill is part of its count.
- **Movers:** the core's BlockMover moves 64-byte beats, one a clock. The RTL's mover rates are its own,
  and the deployment does not declare them.
- **The ratio:** the core's single-buffered run serializes each call's two Moves behind the previous
  call's reads. The RTL's "one buffer" overlaps differently. The ratio is the comparison to watch, not
  the counts.

## 3. Which components bound the run, in order

1. **The program's reuse.** Every call moves both operands from L3 to L2. This is the largest cost,
   and the program states it, not the machine.
2. **The L3's capacity** (below 16 tiles on the 64³).
3. **The BlockMover's rate and the L2 slots' turnover.** With 64-byte beats a 16×16 int8 tile is 4
   beats in and 16 rows out, but each Move waits for its L2 slot's credit.
4. **The tile's shape on the array:** 3.1% of the PEs at most.

DRAM, the DMA engine's windows, the L2 banks and the Combiner are far from binding: DRAM's data bus is
at 21%, the DMA's bursts fit their windows, and the Combiner holds at most one tile.

## 4. Decision 5: does the cycle core replace `ConcurrentTimingExecutor` as L-CA?

**Recommendation: yes, as the L-CA of the programs it covers, and grow it to cover the rest. Keep
today's L-CA until it does.**

**For:**
- **It is a reference.** Every rule is state, every invariant is checked every cycle, and the counters
  obey the operational laws exactly. Today's L-CA charges transfers by formula inside a transaction
  executor (plan §1). That is why it could not show double buffering paying (0.96×) while the RTL does
  (0.82×).
- **Its data is exact end to end.** C goes from the array to DRAM and is checked against L-B bit for
  bit, with nothing hypothesized.
- **It measures what L-T1 needs to be calibrated against.** L3 capacity, L2 slot turnover, bank
  concurrency and the array's idle causes are each a counter, not an assumption.

**What it doesn't cover yet** (each is a step in its own right):
- **Program scope:** an int8 OS gemm only. Missing are other number systems, WS and IS dataflows, the
  Combiner's merge of partials (COMPUTE_ACCUMULATE), epilogue stages, init streams, and LU and the other
  operators.
- **Topology:** one L3 tile on a direct link. Phase 1's NoC runs only the memory subsystem, so
  multi-tile deployments (T4, T16) are not joined.
- **Calibration:** the mover rates and the RTL's starting event are not yet matched (§2.2).
- **Speed:** the 64³ on the 128×64 array takes 0.6 s of host time, since every PE is evaluated
  every clock. That is fine for these programs, but large models need event skipping over idle PEs.

**If you decide yes, the next steps would be:**
1. The int8 OS gemm programs go to the cycle core as their L-CA (`--level cycle-accurate` runs
   `--system`).
2. The comparison with the RTL is made on a matched configuration: mover rates and starting event.
3. Grow the scope in the order the first slice needs: staged operands (reuse), then
   COMPUTE_ACCUMULATE, then the NoC join.

If you decide no, the cycle core stays a reference model beside L-CA, and L-T1's calibration (frozen)
waits on that decision.

## 5. Reproduce

```bash
cmake --build build --target run csp-gen
./build/tools/csp-gen/kpu-csp-gen --algo matmul --size 64 --tile 16 --dtype int8 --l3 9 \
    --target tests/program/deploy/kpu_s1.json -o /tmp/mm64.csp
./build/tools/kpu-run/kpu-run --program /tmp/mm64.csp --deploy tests/program/deploy/kpu_s1.json --system
./build/tools/kpu-run/kpu-run --program /tmp/mm64.csp --deploy tests/program/deploy/kpu_s1.json --level cycle-accurate
```

The RTL's 8×8×16 programs are `tests/cycle/test_cycle_system.cpp`'s `rtl_8x8x16(3)` and
`rtl_8x8x16(6)`.
