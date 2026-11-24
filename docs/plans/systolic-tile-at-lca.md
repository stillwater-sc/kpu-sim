# The systolic tile at L-CA: the deployment's array, L1 lane buffers, the Streamer's commands (#401, slice 3)

**Status:** planned 2026-11-24. The user made the decisions in §2. No step has landed.
**Issues:** #458 (L-CA reads the deployment's fabric), #389 (L1 stream buffers, a Streamer that never
bubbles, the OS north drain), #385 (the wavefront as the timing check), all under #401.
**RTL reference:** branes-ai/embodied-ai-soc-generator at `b57c881`. Its design is described in
`docs/tiles/systolic/README.md` §"The fabric, L1 and the Streamer". The sources are `controller/streamer.py`,
`memory/stream_buffer.py`, `controller/result_path.py`, `core/systolic_array.py`, `core/pe.py`
and `wavefront.py`. The tests are `test_streamer.py`, `test_stream_buffer.py`, `test_systolic_array.py`,
`test_wavefront.py` and `test_credits.py`.
**Follows:** `docs/plans/slot-credits-at-lca.md` (slice 2, #387: every L2 slot placed and in a
state), `docs/plans/banked-l2-at-lca.md` (slice 1). **Scope:** the output-stationary (OS) int8 → int32
matmul on one compute tile, which is the first slice's program. WS/IS and the Combiner's merges are not
in scope.

## 1. The gap

In the RTL, a tile computes through one **Streamer** that executes commands. An OS COMPUTE reads its
operands from L2 into one-block **L1 lane buffers** as it streams them into the array. Its results
drain north into an L2 (Combiner) slot row by row. L-CA has none of this:

| | RTL | L-CA today |
|---|---|---|
| The array | R × C PEs from the configuration | `pe_rows`/`pe_cols` not read, listed as unmapped (`csp_config_from_spec.hpp:153`) |
| L2 → L1 | a Feed is the command's reads: one A row and one B row per clock, after `spad_read_delay` | a Feed is a whole-tile transfer of `startup + bytes / rate` (`streamer_process.hpp:270-285`) |
| L1 | one buffer per lane, one block, one credit; a load without credit is `L1_ERROR` | none (`l1_depth` unused; `TILE_ARRIVED_L1` never emitted) |
| Feed → compute | the lanes stream as they load: no wait for the whole tile | compute waits for every Feed to complete (`dependencies_satisfied`, executor :2208) |
| In flight | commands back to back, one every `I = max(K, R, C)` clocks | one transfer per streamer, Feeds before Drains (`streamer_process.hpp:127-149`) |
| L2 release | `completed`: the clock the last read lands in L1, before the array has finished | at the Feed's completion, after the whole transfer |
| Compute time | the wavefront: PE(i,j) multiplies k at `s+1+i+j+k` | `fill + ceil(MACs / macs_per_cycle)`, fill = 2 × max(h, w) |
| Results | row i leaves the north FIFOs at `s+K+C+2i`; `result` when the last row is written | the Drain moves the whole tile after the compute ends |
| Drain spacing | reduction-ending commands 2R−1 apart, or the north links collide | none |

L-T1 already times an accumulator's gemm chain as an OS systolic stream on `pe_rows`/`pe_cols`
(`transactional.hpp:65-101`). Its drain rules (`out1 = last + r`, chains spaced `r` apart) are
not the RTL's, which are `s+K+C+2R−2` with a gap of 2R−1. This plan brings L-CA to the RTL, and
L-T1 with it (§4, step 3).

## 2. Decisions (the user's, 2026-11-24)

1. **One plan, three PRs.** #458 reads the array from the deployment, #389 adds the L1 lane buffers and
   the Streamer's commands, and #385 checks the wavefront against the RTL. They land in that order.
   A deployment that declares no fabric keeps today's compute timing and names it in `unmodelled`.
2. **A tile no larger than the array runs on a corner of it; a larger tile is refused.** A Call's
   h × w tile uses h A lanes and w B lanes, and the other PEs idle. At L-CA, a tile larger than the
   array (h > R or w > C) is refused by name before the run. The generator must tile to the array,
   as the RTL's L2 slot is one array-sized tile.
3. **A Call's Feeds are the load side of its command** (mine, from the RTL; the user may revisit).
   The program still states every Feed, which is its L2 → L1 movement. But L1 holds one block per
   lane, not a tile, so a Feed can't run ahead of its Call. It is executed as that Call's reads, and it
   completes at `completed`.

## 3. Design

### 3.1 The array (#458)

`csp_config_from_spec` sets `array_rows = fabric.pe_rows` and `array_cols = fabric.pe_cols`, and stops
listing them as unmapped. `kpu_soc_2x2` declares 4 × 4 directly, and its `systolic.grid` × `tile`
agrees. The read latency is `systolic.spad_read_delay` where declared (4 on `kpu_soc_2x2`), else
the L2 read latency. `CSPDriver::check_machine` refuses a Call whose tile exceeds the array (decision 2),
naming the Call's line, its tile, and the array.

With no fabric declared, today's compute model stays, and `unmodelled` says so. That applies to
`kpu_t4` and `kpu_t16`, and to any deployment without `fabric`.

### 3.2 A Call as Streamer commands (#389)

An OS Call computes an h × w tile from A (h × Kt) and B (Kt × w). As in the RTL, it runs as
`ceil(Kt / KB)` COMPUTE commands, chained over K, where `KB` is the elements in one L2 row:

- `KB = C` for int8 (the RTL: `sp_width = C × input_bits`).
- An accumulator's calls (`acc C in fabric { for k … call gemm … }`) are one chain. Only the
  last call's last command carries `last`, and only it produces results (`cmd_chain` MORE /
  CONTINUES).

Command n of a chain is accepted at clock `a_n`. With `L = max(h, w)` read slots, the RTL's timing
is:

| Event | Clock |
|---|---|
| read slot m issued (A row m if m < h, B row m if m < w) | `a + 1 + m` |
| lane m's buffer loads (in parallel, the whole block) | `a + 1 + D + m` |
| lane m streams its K elements, one per clock (the stagger is the skew) | from `s + m`, `s = a + 2 + D` |
| PE(i,j) multiplies element k | `s + 1 + i + j + k` |
| `completed`: the last read lands in L1; the L2 slots' tokens return | `a + D + L` |
| next command accepted | `≥ a + I`, `I = max(K, h, w)` |
| result row i leaves the north FIFOs into the result slot | `s + K + w + 2i` |
| `result`: the last row written; the result slot is READY | `s + K + w + 2h − 2` |

**Which events become which.**

- The Feeds' completion is `completed`. The executor's L2Release then frees the operand slots (#387),
  without waiting for the array.
- The Drain is the result path's row writes into the result slot that the opening Call claimed. Its
  Writeback then proceeds as today.
- A Combiner epilogue (`fabric_cycles`) rides the rows as today. Its timing is not changed here.

**When a command fires** (`streamer.py:408-414`; top level `top.py:387-403`):

- the previous command's interval has passed;
- the command's L2 slots are READY;
- for a reduction-ending command, the drain gap: its results leave at least 2h − 1 clocks after the
  previous tile's (`drain_wait = K + 2h − 2` at accept);
- the result slot is OWNED by its tile (#387 decision 1).

**One Streamer per compute tile** executes these commands. The row and column streamer pools no
longer serve a CSP run on a declared fabric. They remain for legacy schedules until #266, and for
deployments with no fabric.

### 3.3 The L1 lane buffers

There is one buffer per A lane (R) and per B lane (C). Each holds one block of up to `KB` elements and
carries `left`, `first` and `last`:

- **The credit** is `left ≤ 1`: the buffer is empty, or its last element leaves this clock. A block
  loaded in that clock streams on with no gap.
- **A fault, never a stall.** A load into a buffer without its credit raises an `L1Fault` naming
  the lane and the command. This is the RTL's `A_/B_LOAD_WITHOUT_CREDIT`, and it mirrors #387's
  `SlotFault`.
- **North collision.** A result launched onto a column's north link while that link carries one is
  an `L1Fault` too (`NORTH_COLLISION`). The drain gap prevents it, and the check proves that it does.

### 3.4 The L2 reads

Each read slot reads one A row and one B row. Banked L2 (`L2Banks`, slice 1) books them on the rows'
banks, A and B in different bank groups, as in the RTL. Two reads in one group in one clock are refused
(`READS_SAME_GROUP`). Pooled L2 reads one row per operand per clock. How many requests L2 sustains in
flight, and the 250 MHz bank clock, are #495, the next slice. This plan reads at the RTL's rate of
one row per operand per clock.

### 3.5 Element size

`KB` and the read count assume one element per byte lane, which the RTL has. The timed levels still
size int8 as 4 bytes (#415). The commands' clocks count elements and rows, not bytes, so they are
right for int8 now. Byte traffic (DRAM, BlockMover) stays wrong until #415.

## 4. Steps (one PR each)

1. **#458: the array from the deployment.** `array_rows/cols`, the read delay, the refusal, and
   `unmodelled` for deployments with no fabric. On a declared fabric, the OS compute time becomes the
   command chain's: from the first accept to `result`, with the Feeds still separate (§3.2's
   clocks, with `a` at the compute's start). The Feeds stay as they are. Tests:
   - a one-command 4 × 4 × 4 Call on `kpu_soc_2x2`, with `result` at `a + 2 + D + K + C + 2R − 2`;
   - the refusal;
   - a 16 × 16 tile on `kpu_s1`'s 128 × 64 array, which uses its corner.
2. **#389: L1 and the Streamer's commands.** Feeds become the commands' reads (decision 3), with the
   L1 lane buffers, their credits and `L1Fault`, `completed` and `result` as separate events, the
   drain gap and the north-collision check. These tests mirror the RTL's:
   - **no bubbles:** with K ≥ h, w, commands are accepted exactly I apart, and lane 0 is valid every
     clock across three tiles (`test_streamer.py:357-378`);
   - **drain spacing:** a gap of 2R − 1 never collides, and a forced 2R − 2 raises `NORTH_COLLISION`
     (`test_systolic_array.py:65-92`, `test_streamer.py:380-398`);
   - **early release:** `completed` comes before `result`, and the second tile's `completed` before
     the first tile's `result` (`test_streamer.py:400-412`);
   - **a chained reduction** writes only h rows, at the end (`test_streamer.py:342-355`);
   - **a load without credit** raises `L1Fault` (`test_stream_buffer.py`).

   The `.tflow` record gains the commands' events (accept, `completed`, `result`) on the compute
   tile's lane.
3. **#385: the wavefront check, and L-T1 to the RTL's rules.**
   - A per-PE trace check: PE(i,j) busy at `s+1+i+j+k`, and column 0's results at `2 + 2i`
     (`test_wavefront.py`).
   - L-T1's OS chain timing is updated to the same clocks (`out1`, and the 2R−1 gap instead of `r`),
     so that L-T1 and L-CA agree per Call on a declared fabric.
   - The first slice's L-T1/L-CA tolerance is stated in this plan and held in CI. Today the
     `kpu_run_calibration_*` tests only report it.

## 5. What changes for existing programs

- **Timing moves on declared fabrics only.** That means `kpu_s1` (128 × 64) and `kpu_soc_2x2`.
  Each PR re-baselines and explains its numbers, as #524 did. Values must not move: every level stays
  within L-B's bound on every program.
- **Pinned numbers that will move:** `test_csp_spec_oracle.cpp:529,544` (compute = `64 +
  MACs/macs_per_cycle`), and the makespans printed by `test_csp_double_buffer` and quoted in
  `test_csp_lca_place.cpp:392`. Relations that must still hold: `test_csp_double_buffer`'s (fewer
  slot waits and more overlap with two buffers, and faster), and `test_csp_driver`'s restated window
  rule.
- **Programs whose tiles exceed the array are refused at L-CA.** The corpus's 16 × 16 tiles fit both
  declared arrays.

## 6. Verification

- Values: `examples/program --placement` and `kpu-run` on the corpus and on the int8 64³ matmul, at
  L-B, L-T1 and L-CA.
- Invariants each run checks on its own (#386 D3): L1 credit conservation (no lane holds two blocks),
  no north collision, and every L2 slot FREE at the end (#387).
- **The RTL's 8×8×16 matmul on `kpu_soc_2x2`.** L-CA's cycles with one and two buffers are reported
  against the RTL's 907 / 744, not asserted. Matching them exactly is #401's done-when, after #415,
  #382 and the RTL's mover rates (the run is bound by the BlockMover). The RTL counts from simulation
  start, including its 7 CONFIG writes and START, so the plan's comparison names its starting
  event.

## 7. Open questions

- **Q1. `KB` for wider inputs.** The RTL takes `KB = min(sp_width / input_bits, sp_width /
  weight_bits)`. For int8 it is C. Wider types come with #415.
- **Q2. A tile smaller than the array.** The RTL's slot is array-sized, so it never runs a partial
  tile. On a corner (decision 2), the interval is `max(K, h, w)` and the drain gap 2h − 1, both
  from the tile's own extent. This is an extrapolation, stated here and tested on `kpu_s1`.
