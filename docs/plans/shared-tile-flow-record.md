# The Shared Tile-Flow Record: `.dflow` Version 6, for kpu-sim and the KPU RTL (#386)

**Date:** 2026-11-08
**Status:** Specified and checked. kpu-sim's L-B writes version 6. The checker accepts L-B records and
the hand-written RTL-level sample, and the domain-flow viewer opens both.
**Issue:** #386. Relates to branes-ai/embodied-ai-soc-generator#20 and stillwater-sc/systars#14.
**Builds on:** the domain-flow record (`docs/plans/domain-flow-debugger.md`, #363) and the columnar
container (`include/sw/kpu/program/record/columnar.hpp`).

## 1. Purpose

kpu-sim and the KPU systolic-tile RTL share **one record format** (#20, Q1). A kpu-sim run and an
RTL run of the same schedule are both written in it and read by the same viewer. The RTL can then
be compared, residency by residency, with kpu-sim's facsimile of the hardware.

The format is specified here, beside the existing records. The RTL generators (kpugen, systars)
write it and don't fork it. This document defines version 6 of `.dflow` and how its conformance
is checked. It adds no simulator behaviour.

The review of 2026-11-02 decided three things:

- **D1. `.dflow` is extended, not replaced.** The issue says "version 3". The format had already
  reached version 5 when this work began: version 3 added the L-T1 overlay, version 4 the fabric
  pipeline, version 5 L2 staging. So the shared record is **version 6**. Every addition is new
  columns, new tables or new manifest fields in the same columnar container. `.tflow` stays the
  L-T1 station and swimlane record.
- **D2. Events have identities both sides share.** Every event carries a kind and the attributes
  that tell it apart (§5.3). kpu-sim and the RTL can then compare notes event by event, whichever
  side wrote the record. The CSP action index is one of these identities, not the only one.
- **D3. CI compares the result, never the event trace.** CI checks two things: the cycles to the
  result, and the result's data elements (§8.2). Event vocabularies change as both sides evolve;
  a golden event trace would have to be regenerated in two repositories for every such change.
  The checker validates a record. Event sequences are a debugging tool.

## 2. The container and its versions

A bundle is a directory holding a `manifest.json` and one `.bin` file per table. This is the same
container `.tflow` and `.mflow` use:

- each column is little-endian and padded to 8 bytes;
- the manifest gives each column's `name`, `dtype` (`u8`, `u16`, `u32`, `u64`, `i64`, `f32`,
  `f64`) and byte `offset`;
- a viewer maps a column as a typed array, with no parsing.

A u64 index that names nothing is all ones (`0xFFFF…`); a u32 is `0xFFFFFFFF`, a u8 is `0xFF`.

| Version | Added |
|---|---|
| 1 | steps, tiles, residencies, positions, loops, the iteration summary (#363 step 2) |
| 2 | tile values: `snapshots`, `elements` (float32) (#363 step 4) |
| 3 | the L-T1 overlay: `lt1_legs`, `lt1_tiles` (#363 step 5) |
| 4 | the fabric pipeline: `lt1_chains`, `lt1_calls` |
| 5 | L2 staging: `l2_staging`; the steps' `staging`, `l2_cf`, `l2_slot` |
| **6** | **the shared record: level, producer, time, geometry, events, element types, domains, expected values, the result** |
| 6.1 | the RTL's accumulator → L3 store and its flat, word-addressed L3 (§5.1, §6; #398) |
| 6.2 | L2, the accumulator and the array's edge addressed by slot: kpu-sim's timed levels (§6; #402) |

Readers of version 6 read versions 1 to 5 as before. A version-1 to version-5 record is L-B's,
fp32 and dense.

**Minor revisions.** A version-6 manifest may carry `"minor"`: 0 when absent. A minor revision adds
a transaction's layer pair or a geometry form; it keeps the `layers` and `event_kinds` lists, whose
change needs a new version. A reader of 6.*n* reads every 6.*m* with *m* ≤ *n*, and refuses a
later one. A 6.0 record may use none of 6.1's additions, and a 6.1 record none of 6.2's.

**Optional columns.** A later writer may add a column to an existing table, when a record without
it still means what it did: a reader keeps the columns it knows and reads a missing optional column
as its default. The checkers type-check an optional column when it is present. They are:

| Table | Column | Type | Default | Meaning |
|---|---|---|---|---|
| `l2_staging` | `partial` | `u8` | 0 | the slot holds an L2 accumulator's partial (#429), not a staged copy: its overwrite Drain opens it, the Combiner merges every Drain into it, its Writeback reads it, its L2Release frees it, and nothing feeds it (#455) |
| `l2_staging` | `merges` | `u64` | 0 | a partial's Drains, the first included |
| `steps` | `merge` | `u8` | 0 | a Drain's merge into an L2 accumulator: 0 none, 1 overwrite, 2 accumulate |
| `steps` | `stages` | `u16` | 0 | the vector stages the action applies as it moves its tile, a bit per `csp::VeOp` (add 0, relu 1, gelu 2, silu 3, atan 4, mxint8 5, mxfp8 6, scale 7, init 8); on a merge they are the Combiner's, in the order the language fixes (#456) |
| `steps` | `init_tile` | `u32` | none | the init stage's tile (an index into `tiles`): the first merge's MERGE_INIT (#456) |
| `steps` | `dataflow` | `u8` | 0 | a Call's, or its Drain's, dataflow as the program states it (`acc ... dataflow ws`): 0 default (an L2 accumulator that states none: the deployment's `fabric.dataflow`), 1 os, 2 ws, 3 is; a fabric accumulator's is always 1 (#395) |
| `.tflow` `l2_residency` | `partial`, `merges` | `u8`, `u32` | 0 | the same, for L-T1's L2 slots |
| `.tflow` `chain`, `.dflow` `lt1_chains` | `dataflow` | `u8` | 1 | the dataflow L-T1 timed the chain by, resolved: 1 os, 2 ws, 3 is (#395) |
| `.tflow` `chain`, `.dflow` `lt1_chains` | `preload`, `streamed` | `u32` | 0 | a ws or is chain's PRELOAD clocks (R, the array's rows) and rows streamed (S): `front = in0 + R`, `in1 = front + S`, `last = in1 + W − 1`, `out1 = last + R`, its call holding the compute tile `R + max(S, R)`; checked by `tflow_check` TF14 and `dflow_check` P2/P3 |

`dflow_check` S5 and `tflow_check` TF13 check a partial. A record written before these columns had
no partials (L2 accumulators came later), so it reads as before.

Two writers produce the format:

- `src/program/domain_flow_record.cpp`, kpu-sim's C++ writer;
- `tools/trace/dflow_writer.py`, a standard-library Python writer for the RTL generators to vendor.

One checker validates it: `tools/trace/dflow_check.py`.

## 3. The manifest: what version 6 adds

Every version-5 field stays, with the same meaning. These are added:

| Field | Value |
|---|---|
| `level` | `"L-B"`, `"L-T1"`, `"L-T2"`, `"L-CA"` or `"RTL"`: the level that produced the record |
| `producer` | a non-empty string naming the tool and its revision: `"kpu-sim 0.8.0-dev292+54e0d77"`, `"kpugen <sha>"` |
| `layers` | exactly `["dram", "l3", "l2", "acc", "l1"]`: the layer enum that events index |
| `event_kinds` | exactly `["DMA", "BLOCK_MOVE", "STREAM", "COMPUTE", "DRAIN", "CREDIT", "TOKEN", "BM_ROW", "STR_ROW", "DRAIN_ROW"]` |
| `time` | `null` at L-B. At a timed level, `{"unit": "cycle", "clock": <name or null>, "cycles": N}`, where N is the cycles to result: the latest end of any event |
| `geometry` | one entry per layer (§6) |
| `operands[i].element` | `int8`, `int32`, `fp16`, `bf16`, `fp32`, `fp8_e4m3` or `fp8_e5m2` (§7.1) |
| `operands[i].domain` | `{"kind": "dense" \| "lower" \| "upper" \| "band", ...}`, with an optional padding window (§7.2) |
| `values.format` | `"per-operand"`: each operand's elements are decoded by its own `element` |
| `values.expected` | whether the record holds expected values beside the actual ones |
| `result` | `{"cycles": N \| null, "values": "match" \| "mismatch" \| "unchecked", "bound": {"atol", "rtol"} \| null}` (§8.2) |

The `layers` and `event_kinds` lists are part of the format. A producer writes them verbatim. A
reader refuses a record whose lists differ, rather than index the wrong names. Changing either
list needs a new version.

## 4. Tables: what version 6 changes and adds

### 4.1 Unchanged

`steps`, `tiles`, `residency`, `where`, `loops` and `iteration` keep their version-5 columns and
meaning: the CSP actions, the tiles they name, L3 residencies at their addresses, the program
positions, and the iteration summary.

A timed level writes the same steps, because it executes the same CSP program (ADR 0002). The steps
table has no time column at any level. Time is carried by the events.

The overlay and staging tables (`lt1_*`, `l2_staging`) remain. A producer without an L-T1 overlay
or L2 staging writes them empty.

### 4.2 Changed: the value payload

- **`elements`** becomes the payload: one column, `b` (`u8`). It holds each snapshot's elements
  back to back, row-major and little-endian, each element in its operand's type and width.
- **`snapshots`** keeps its columns. Its `first` column becomes a byte offset into the payload,
  and it gains one column:
  - `role` (`u8`): `0` for an actual value, `1` for an expected value.

  An expected snapshot sits beside an actual one of the same tile, step, channel and shape (§7.3).

The C++ record keeps fp32 elements and element offsets in memory. It converts to the payload when it
writes, and back when it reads.

### 4.3 Added: `events`

One row per event a level emits (§5). L-B writes this table empty: its steps are its events.

| Column | dtype | Meaning |
|---|---|---|
| `id` | u64 | the row index: 0, 1, 2, ... in row order |
| `parent` | u64 | for a row event (§5.4), the id of its transaction; otherwise none |
| `action` | u64 | the CSP action the event belongs to (the join to `steps`), or none |
| `where` | u64 | the program position (into `where`), which gives the loop indices; or none |
| `src_addr`, `dst_addr` | u64 | the address in the source and destination layer (§6), or none |
| `tile` | u32 | the tile (into `tiles`), or none |
| `lane` | u32 | which instance of the process (the DMA engine, the BlockMover, ...) |
| `row`, `beat` | u32 | for a row event, the row of the tile and the beat within it; otherwise none |
| `kind` | u8 | into `event_kinds` |
| `process` | u8 | into `processes` (`dma`, `bm`, `str`, `cf`), or none |
| `src`, `dst` | u8 | into `layers`, or none |
| `cycle`, `end` | u64 | **only at a timed level**: the event's first cycle and its end, with `cycle <= end` |

The `cycle` and `end` columns are present exactly when `time` is not null. That keeps the
domain-flow plan's rule that an L-B record has no time axis (§3.7 there): the checker refuses a
`cycle` or `end` column on an L-B record, and a timed record without one.

## 5. Events

### 5.1 Transactions

A transaction moves one tile from one layer to the next. Its kind fixes its layers, and each pair
of layers is one CSP action kind:

| Event kind | Layers | CSP action | Process |
|---|---|---|---|
| `DMA` | dram → l3 | `LOAD` | `dma` |
| `DMA` | l3 → dram | `STORE` | `dma` |
| `BLOCK_MOVE` | l3 → l2 | `MOVE` | `bm` |
| `BLOCK_MOVE` | l2 → l3 | `WRITEBACK` | `bm` |
| `STREAM` | l2 → l1 | `FEED` | `str` |
| `COMPUTE` | l1 → acc | `CALL` (its tile is the accumulator) | `cf` |
| `DRAIN` | acc → l2 | `DRAIN` | `str` |
| `DRAIN` | acc → l3 (6.1) | `WRITEBACK`, standing for its `DRAIN` too | `str` |

A transaction that names its action has that action's tile, and the action's L3 slot as its
address in L3 (with a word-addressed L3, its residency's word address; §6).

**The RTL's store (6.1, #398).** The RTL's STORE reads the accumulator straight into L3, without
passing through L2. The CSP program still says `DRAIN` then `WRITEBACK`. The record writes one
`DRAIN` transaction from acc to l3, joined to the `WRITEBACK` action. It also stands for the tile's
`DRAIN` step, the last one before that `WRITEBACK`, so with full joining the `DRAIN` step has no
transaction of its own.

When every transaction names its action, each recorded data step has exactly one transaction. A
producer that cannot join its events to the CSP actions leaves `action` none. The other identities
(§5.3) still apply.

### 5.2 Credits and tokens

Credits flow upstream and data flows downstream (`docs/01-architecture/kpu-execution-model.md`).
Both signals are events:

- **`TOKEN`**, a tile-ready token, goes **downstream** along a data edge. It is sent when its tile
  has arrived at the token's source. Examples: an L3 → L2 token to the BlockMover after a DMA lands
  a tile; an L2 → L1 token to the Streamer after a block move.
- **`CREDIT`** goes **upstream**, against a data edge. It is sent once its tile has left the
  credit's source, which frees the space. Examples: an L2 → L3 credit after a stream reads its
  rows; an L3 → DRAM credit when a slot is released; an acc → L1 credit after a drain.

The data edges are dram → l3, l3 → l2, l2 → l1, l1 → acc, acc → l2, l2 → l3 and l3 → dram, and from
6.1 acc → l3.

A credit or token may name its tile and the action it follows, for example a `RELEASE`. This
matches the RTL's credit and tile-ready handshakes once kpugen #16 lands.

### 5.3 Identities (D2)

An event's **identity** is its key:

> (kind, process, tile, src, src_addr, dst, dst_addr, line, loop indices, row, beat)

Here `line` is the program line of the position (`where`), and the loop indices are the
`(variable, value)` pairs it resolves to, outermost first. The line is there because two
statements in loops of one variable name have the same indices: LU's `laswp A[k, j]` and
`trsm_ll(A[k, k], A[k, j])` both move `A[k, k]` at the same `(k, j)` (#402). Both producers run
the same program text, so the line is the same in both records.
No two events in a record share a key. Among the events that name an action, no two share
`(kind, action, row, beat)` either. This is a requirement on the producer, checked by E2, not a
property the key guarantees: a program that moved the same tile twice from one statement in one
iteration, by the same process between the same layers, would give two events one key, now that
L2's address is not in it. Such a record fails E2, and `dflow_diff.py` refuses a key it cannot
join uniquely, so a collision is reported, never joined wrongly. No program the generator or the
corpus writes has one.

- **L2's and the accumulator's addresses are not part of the key** (#401, decided 2026-11-18): they
  are each producer's placement. (The occupancy viewer joins the same way, but only `dflow_diff.py`
  reports placement.) kpu-sim's banked L2 puts a tile at its slot's rows, and the RTL at
  rows its own allocator chooses, so the same transfer has different rows in the two records. L3's
  and DRAM's addresses stay in the key, since the program places them. `dflow_diff.py` compares L2
  and accumulator addresses on the matched events, and reports those that differ as *placed
  differently*. It compares an endpoint only when both records give its address, in the same form:
  slots against rows, or an address one record leaves out, is not compared, so a count of zero does
  not prove the placements agree. E2 checks uniqueness with the same key.
- The key is built from **what the event is**: which tile it moves, from where to where, in which
  iteration, and which row. It is not built from when the event happened or which engine instance
  ran it. So it is the same in a kpu-sim record and in an RTL record of the same schedule.
- `id` is local to a record. It orders rows and lets a row event name its parent. It is **not** an
  identity across records.
- `lane` is not part of the key. An RTL with two DMA engines and a model with one do the same
  transfer on different lanes.

### 5.4 Row events: sub-tile progress

Row events record how far a process has got through a tile. They are optional at every level:
the RTL has them, and L-B doesn't.

| Kind | Child of | `row` | `beat` |
|---|---|---|---|
| `BM_ROW` | `BLOCK_MOVE` | the row of the tile moved | its burst |
| `STR_ROW` | `STREAM` | the row streamed into the array's edge | its skew position |
| `DRAIN_ROW` | `DRAIN` | the output row drained | its beat |

A row event has the tile and action of its parent, and its row lies inside the tile. At a timed
level it also lies inside its parent's `[cycle, end]`. Only row events carry `row` and `beat`, and
only row events have a parent transaction.

## 6. Geometry

Every layer is declared, modelled or not. A layer the producer doesn't model is declared as not
modelled, with the reason, and a viewer draws it that way, **never as empty**. No event may move
through a layer the record declares not modelled.

| Layer | Modelled entry | Address in events |
|---|---|---|
| `dram` | `{"modelled": true}` | any (or none) |
| `l3` | `{"modelled": true, "slots": N, "banks": B}`: the machine's `l3` and `banks` | the slot, `< slots` |
| `l2` | `{"modelled": true, "banks", "rows", "row_bytes", "interleave": "row-mod-banks"}` | a transaction's: its first row; a row event's: the row it accessed, wrapping (row 0 after row `rows − 1`); both `< rows` |
| `acc` | as for L2 | the first row, `< rows` |
| `l1` | `{"modelled": true, "rows", "cols"}`: the array's edge | the edge lane, `< max(rows, cols)`, or none |

Not modelled is `{"modelled": false, "why": "<reason>"}`.

**A layer addressed by slot (6.2, #402).** L-T1 and an unbanked L-CA place tiles in L2 slots, a
staged copy or an L2 accumulator's partial, and have no rows. They declare L2, the accumulator
and L1 as `{"modelled": true, "slots": N}`, N ≥ 0. (A banked L-CA declares L2 by rows instead,
and its L2 addresses are booked rows, not slots; its accumulator stays slotted. L1 is slotted,
`slots: 0`, only when no PE array is declared; with one, any record declares L1 by the array's
edge, `{"rows", "cols"}`.)

- an event's address in such a layer is a slot, `< N`, or none. An unstaged tile's L2 address is
  none. A layer of 0 slots has no addresses: kpu-sim's accumulator (the PEs, or an L2 partial at
  its L2 slot), and its array edge when the deployment declares no PE array. With one, L1 is
  `{"rows", "cols"}` as above, every address none;
- a layer declared both ways, `slots` together with `banks`, `rows`, `row_bytes`, `interleave`
  or `cols`, is refused (G1).

A banked kpu-sim L-CA record (#401) declares L2 in the rows form instead, like the RTL: a
transaction's L2 address is its transfer's first row, and each row event's is the row it accessed,
wrapping at the L2's last row (a transfer from row 3 of a four-row L2 has rows 3, 0, 1). Each
transfer's rows are row events (BM_ROW, STR_ROW,
DRAIN_ROW: row r of the transfer, beat 0, at the clock the bank served it). A transfer that takes
more L2 rows than its tile has rows keeps its transaction but has no row events, since a row event
is a row of its tile (E3); the record's notes count those. Either way, L2 and
accumulator addresses are placement, not identity (§5.3), so records join on the rest of the key.

**A word-addressed L3 (6.1, #398).** The RTL's L3 is a flat, word-addressed memory cell per site,
placed by the host or `plan_gemm`, with no slots and no L3 credits (credits exist for its L2 and
accumulator only). Such an L3 is declared `{"modelled": true, "words": W, "word_bytes": b}`:

- the events' L3 addresses are word addresses, `< words`;
- a residency keeps one address from open to close (E5), where a slotted L3 checks the action's slot;
- no `CREDIT` frees L3 (E4). A credit frees its source, so an L2 credit sent up to L3's BlockMover
  is still an L2 credit;
- the steps keep the CSP program's slots: they are the program's placement, which the host maps to
  words. The machine's `l3` and `banks` are not checked against it (G1).

An L3 declared both ways, `words` or `word_bytes` together with `slots` or `banks`, is refused.

`row-mod-banks` is the RTL's interleave: consecutive rows walk the banks, so row *r* is in bank
*r mod banks*. A different interleave needs a new name in the format.

kpu-sim's L-B models DRAM and L3, its addressed slots. It declares the rest not modelled, because
L-B moves whole tiles:

- L2's banks and rows (L-B stages tiles in L2 slots, but not by row);
- the accumulator's banks and rows (L-B holds the accumulator as a whole tile in the fabric);
- the array's edge (L-B feeds whole tiles to the fabric).

### 6.1 A standalone tile: operands the host placed in L3 (#473)

A compute tile with no DRAM and no DMA behind it is one example: systars' `SystolicTop`, whose AXI
port is L3. An L2 or L3 testbench, or a single-tile RTL bench, is another. The host writes the
operands into L3 before the run, and leaves the results there after it. The format already says
this, with no new kind:

- **`INHERIT`** opens an L3 residency at the program's start: the tile is there before any action
  runs. It has no process and no transaction, so E5 does not ask for one. In the CSP language it is
  `inherit X`, at the top of the program (`docs/plans/csp-language.md`).
- **`RETAIN`** keeps a result in L3 past the program's end, with no store (`retain y`).
- **`geometry.dram`** is `{"modelled": false, "why": ...}`, and no event moves through DRAM (E1).

The record's steps are still the CSP program's: `INHERIT`, then `MOVE` l3 → l2 for the operands,
and so on. `tests/program/csp/standalone_tile_8_t4.csp` is such a program. Its records pass the
checker with DRAM not modelled. So does its timed record, once the events a level with a DMA sends
to it (a release's credit, a writeback's token) are absent, as they are in a standalone tile.

Writing the host's placement as a `LOAD` with no transaction is refused (E5: a `LOAD` moves a tile
once). With DRAM not modelled, E5's message points to `INHERIT`.

## 7. Values

### 7.1 Element types

| Element | Bytes | Decoded as |
|---|---|---|
| `int8`, `int32` | 1, 4 | two's complement |
| `fp32`, `fp16` | 4, 2 | IEEE 754 binary32, binary16 |
| `bf16` | 2 | the upper half of a binary32 |
| `fp8_e4m3` | 1 | OCP FP8 E4M3: bias 7, no infinity, one NaN (`S.1111.111`) |
| `fp8_e5m2` | 1 | OCP FP8 E5M2: bias 15, IEEE-style infinities and NaNs |

This extends `tile_check`'s `ElementFormat` (float32 today) to the RTL's types. The checker
(`dflow_check.decode`) and the viewer decode them the same way, and a test compares the two on
every byte pattern of the one-byte types and a spread of the wider ones.

### 7.2 Domains

A domain says which elements of an operand exist. It uses operand coordinates (r, c):

| Kind | In the domain |
|---|---|
| `dense` | every element |
| `lower` (`diagonal` d, any integer, default 0; d = −1 is strictly lower) | c − r ≤ d |
| `upper` (`diagonal` d, any integer, default 0) | r − c ≤ d |
| `band` (`lower` p, `upper` q) | r − c ≤ p and c − r ≤ q |

An optional `pad: {"rows": R, "cols": C}` marks the padding window: elements with r ≥ R or c ≥ C
are outside the domain, whatever the kind.

**Out of domain is not zero.** An element outside the domain is stored as zero bytes, so payloads
compare deterministically, and the checker refuses one that isn't. A viewer draws it as out of
domain and leaves it out of every statistic. It never draws it as a zero value. A tile wholly
outside the domain (L[0,1] of a lower-triangular L) is never loaded by a correct schedule, but its
final value is still recorded, all out of domain.

### 7.3 Snapshots, finals and expected values

The version-2 rules hold:

- every operand tile's **final** value is always recorded;
- step snapshots are opt-in, under a byte cap;
- a snapshot is the tile where its action leaves it.

Version 6 adds **expected values**, recorded beside the actual ones so that the viewer never
computes them:

- An expected snapshot has the same tile, step, channel and shape as an actual one, and at most one
  exists for each actual snapshot.
- With expected values, every final has its expected value.
- L-B writes none: it is the oracle, so its values are the expected ones.

## 8. Conformance

### 8.1 The checker

```bash
python3 tools/trace/dflow_check.py <bundle.dflow>            # exit 0 OK, 1 violations, 2 unreadable
python3 tools/trace/dflow_check.py <bundle.dflow> --result   # the result only (§8.2)
```

The checker enforces the version-1 to version-5 invariants on every record. The steps of an RTL
record are a CSP run like any other: one step per action, residencies, slots and the iteration
summary.

Version 6 adds these invariants:

| Invariant | Holds when |
|---|---|
| G1 geometry | every layer is declared; a modelled L3 is the machine's slots and banks, or (6.1) words and a word width, never both; L2 and the accumulator have banks, rows, a row width and a known interleave; L1 has rows and cols; or (6.2) L2, the accumulator or L1 has slots, and nothing of the rows form |
| T1 time | L-B has no time, no `cycle` or `end` column and no result cycles. A timed level has all three; every event has `cycle <= end`; the latest end is `time.cycles`; and `result.cycles` equals it |
| E1 references | `id` is the row; every index resolves; layers are modelled; addresses are inside their layer; a transaction moves between its kind's layers (acc → l3 from 6.1) |
| E2 identities | §5.3: no two events share a key (with the position's line), or a (kind, action, row, beat) |
| E3 progress | §5.4: a row event is a child of its kind's transaction, of the same tile and action, inside the tile, and inside its parent's time |
| E4 credits and tokens | §5.2: a token goes downstream after its tile arrived; a credit goes upstream after its tile left; no credit frees a word-addressed L3 |
| E5 the join | §5.1: a transaction's action has the matching kind, tile and L3 slot (word-addressed: its residency keeps one word address); an acc → l3 `DRAIN` also stands for its tile's `DRAIN` step; with full joining, one transaction per data step |
| X1 element types | known types; each snapshot is height × width elements of its type; its role is 0 or 1 |
| X2 domains | domains are well formed; out-of-domain elements are zero bytes |
| X3 expected values | §7.3; and `result.values` is `match` exactly when every final equals its expected value |

For X3, integer types compare bit for bit. Floating types also compare bit for bit, or within
`result.bound` (`|a − e| ≤ atol + rtol·|e|`) when the record gives a bound. A record that says
`mismatch` honestly is a valid record: it is the result that failed, not the format.

`tools/trace/test_dflow_check_v6.py` breaks each of these invariants in turn and requires the
checker to fail it. It runs on the RTL sample and on an L-B record that kpu-run wrote.

### 8.2 What CI gates on (D3)

`--result` prints the result of a record that passes every invariant (it exits 1 otherwise):

```json
{"level": "RTL", "producer": "...", "program": "trmm_8x4", "cycles": 212, "values": "match",
 "finals": {"L": {"element": "int8", "tiles": 4, "sha256": "..."}, ...}}
```

- **cycles** is the cycles to the result: null at L-B.
- **finals** is a digest of each operand's final values: SHA-256 over the actual final snapshots'
  bytes, tiles in (ti, tj) order. Out-of-domain elements are zero (X2), so the digest depends only
  on the elements in the domain.
- **values** is the record's own verdict against its expected values.

CI in either repository compares this output with a checked-in result for its own programs, and
nothing else. The output reads no event column, so a change to the events leaves it unchanged.
A test pins that: renumbering every row event's beats keeps the result byte for byte.
The digest is per element type, so an fp32 L-B run and an int8 RTL run of the same schedule are
compared through their expected values (X3), not through each other's digests.

### 8.3 Comparing two records: `dflow_diff.py` (#402)

```bash
python3 tools/trace/dflow_diff.py <a.dflow> <b.dflow> [--json] [--tolerance N] [--show K]
```

It joins two records of the same program on the identity key (§5.3). The key is spelled the same
in both: a tile by its operand name and coordinates, a process and a layer by name, a loop by its
variable's name. A layer whose addresses take different forms in the two records (6.2's slots
against the RTL's rows, 6.1's words against slots) is joined without its addresses, and the
report says which layers those are. It reports:

- the events only one record has, by kind, each row event under its transaction, and whether that
  transaction joined;
- with both records timed, the first divergence: the earliest event, in A's cycles, whose start or
  duration differs by more than `--tolerance`. Then the largest divergences of start and of duration.

It is a debugging tool, not a CI gate (D3). It exits 0 when the two join with nothing one-sided, no
divergence and nothing placed differently, 1 when they differ (a placement difference alone
included), and 2 when either cannot be read, fails the checker, or has no events (L-B's).

The RTL sample's matrix is named L. L-CA maps a program's matrices onto A, B and C, so
`dflow_rtl_sample.py --names A,B,C --program-out p.csp` writes the same sample under those names,
with its program. L-CA's record of that program joins every one of the sample's transactions. A
row event joins one with the same key under its joined transaction: L2's addresses are not in the
key, so the sample's BM_ROW row 0, beat 0 can match a banked L-CA's. The sample's other row events,
whose row or beat L-CA has no counterpart for, are one-sided, under joined transactions. The first
divergence is the first load.

## 9. Producers

| Producer | Level | Writes |
|---|---|---|
| kpu-sim `kpu-run --dflow` | L-B | version 6: `time` null; DRAM and L3 modelled, the rest declared not modelled; fp32 dense operands; no expected values; an empty `events` table with no time column; `result` with no cycles and values `unchecked` |
| kpu-sim `kpu-run --dflow-lt1`, `--dflow-lca` | L-T1, L-CA | version 6.2 (#402): L-B's steps, tiles, residencies and positions; the level's events with `cycle` and `end` (§9.2); L2, the accumulator and L1 by slot; the level's own final values, no per-step snapshots; `result` with the level's cycles and values `unchecked`; `not_modelled` naming what the level does not model |
| kpugen / systars | RTL | version 6, through `dflow_writer.py`: a cycle column, the RTL's geometry, transactions, row events, credits and tokens; its element types and domains; expected values from the golden model |
| `tools/trace/dflow_rtl_sample.py` | RTL (sample) | a hand-written stand-in for kpugen (§9.1) |

### 9.1 The RTL-level sample

The program is `C = L·B`:

- L is lower triangular, 8 × 8, int8;
- B is 8 × 4, int8;
- C is 8 × 4, int32;
- all three are in 4 × 4 tiles.

Its 33 steps are the actions kpu-run's L-B executes for the same `.csp` text, at the same L3 slots.
The steps, tiles, residencies, positions, loops and iterations equal L-B's record table for table.
L[0,1] is outside L's domain and is never loaded.

The sample also has:

- geometry for L2 (2 banks × 16 rows), the accumulator (2 × 8) and L1 (4 × 4);
- 112 events, each with a cycle;
- row progress for every block move, stream and drain;
- tokens after each landing and credits after each release;
- expected values beside every snapshot.

Its timing is invented, a fixed cost per transaction run back to back. It is a sample of the
format, not a model of the hardware.

With `--addressed` the sample is a 6.1 record, built as the RTL is: each `DRAIN` and `WRITEBACK`
pair is one acc → l3 `DRAIN`, L3 is 64 words of 4 bytes, a residency sits at its slot × 16 words,
and no credit frees L3. Its final values, and so its `--result` digests, equal the slotted
sample's.

### 9.2 kpu-sim's timed levels

A timed level executes the same CSP program as L-B, so its record keeps L-B's steps and adds its
events. L-T1's events are its legs; L-CA's are its protocol events, matched back to the actions
(`timing/csp_records.hpp`). Each data step is one transaction, joined to its action, written as the
RTL sample writes them:

| Action | Transaction | Then |
|---|---|---|
| `LOAD` | `DMA` dram → l3, at its slot | a `TOKEN` l3 → l2 |
| `MOVE` | `BLOCK_MOVE` l3 → l2, slot → its L2 slot or none | a `TOKEN` l2 → l1 |
| `FEED` | `STREAM` l2 → l1, from its L2 slot or none | unstaged, a `CREDIT` l2 → l3 |
| `CALL` | `COMPUTE` l1 → acc | |
| `DRAIN` | `DRAIN` acc → l2, into its L2 slot or none | a `CREDIT` acc → l1 and a `TOKEN` l2 → l3 |
| `WRITEBACK` | `BLOCK_MOVE` l2 → l3 | a `TOKEN` l3 → dram |
| `STORE` | `DMA` l3 → dram: its two legs, the ejection and the DRAM write, as one | |
| `RELEASE` | | a `CREDIT` l3 → dram, at the cycle the level freed the slot |
| `L2_RELEASE` | | a `CREDIT` l2 → l3 |

Two things the format can't say, which the record's `not_modelled` counts:

- **A store's slot is freed at its ejection,** into the DMA engine's buffer, before the DRAM write
  ends. The format has no layer for that buffer, so its `RELEASE`'s credit is stamped at the
  store's end (E4).
- **A tile the Combiner reads in place** (an init, a bias, a scale's vectors) never leaves L3 in a
  transaction, so its `RELEASE`'s credit has none to follow (E4) and is left out.

Row events: a banked L-CA record has each L2 transfer's rows (BM_ROW under a Move or Writeback,
STR_ROW under a Feed, DRAIN_ROW under a Drain) when its booked rows fit its tile (§6, E3). L-T1's
record and an unbanked L-CA's have none, and no level has the PE grid's rows yet (#401).

## 10. The viewer

The domain-flow viewer (`tools/visualization/domainflow/`) reads versions 1 to 6. On a version-6
record:

- the header names the level, and a timed level's cycles to result and event count;
- the inspector decodes values by element type;
- out-of-domain elements are drawn as a checker and left out of the statistics;
- when the record holds expected values, the inspector compares the tile with them.

Drawing time and row progress is #20's viewer work: the viewer reads the events and doesn't reject
them. `test_viewer.py` runs the page's data layer on the RTL sample and on L-B records, and compares
what it reads with `dflow_check`.

## 11. Not in this version

- **kpu-sim's L-T1 row events.** L-CA writes them, from its banked L2's bookings (#401). L-T1 times
  a tile move whole, so its record has none. The PE grid's rows (#401's wavefront slice) are not
  yet events either.
- **Golden event traces.** Ruled out by D3, as CI references. Event comparison between two records
  is a debugging tool, joined on the identity key (§5.3).

## 12. Verification

```bash
python3 tools/trace/dflow_rtl_sample.py -o rtl_sample.dflow
python3 tools/trace/dflow_check.py rtl_sample.dflow                     # OK
python3 tools/trace/dflow_check.py --result rtl_sample.dflow            # cycles 212, values match
build/tools/kpu-run/kpu-run --program p.csp --level behavioral --dflow lb.dflow
python3 tools/trace/dflow_rtl_sample.py --addressed -o rtl_addr.dflow  # 6.1: acc -> l3, word-addressed L3
python3 tools/trace/test_dflow_check_v6.py rtl_sample.dflow lb.dflow rtl_addr.dflow   # every invariant bites
python3 tools/trace/dflow_rtl_sample.py --names A,B,C --program-out s.csp -o rtl_abc.dflow
build/tools/kpu-run/kpu-run --program s.csp --deploy tests/program/deploy/kpu_s1.json --level all --dflow-lca lca.dflow
python3 tools/trace/dflow_diff.py lca.dflow rtl_abc.dflow                  # joins; names the first divergence
cd build && ctest -L domainflow --output-on-failure
```
