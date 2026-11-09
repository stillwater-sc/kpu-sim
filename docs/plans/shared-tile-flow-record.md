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

Readers of version 6 read versions 1 to 5 as before. A version-1 to version-5 record is L-B's,
fp32 and dense.

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

A transaction that names its action has that action's tile, and the action's L3 slot as its
address in L3.

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

The data edges are dram → l3, l3 → l2, l2 → l1, l1 → acc, acc → l2, l2 → l3 and l3 → dram.

A credit or token may name its tile and the action it follows, for example a `RELEASE`. This
matches the RTL's credit and tile-ready handshakes once kpugen #16 lands.

### 5.3 Identities (D2)

An event's **identity** is its key:

> (kind, process, tile, src, src_addr, dst, dst_addr, loop indices, row, beat)

Here the loop indices are the `(variable, value)` pairs that `where` resolves to, outermost first.
No two events in a record share a key. Among the events that name an action, no two share
`(kind, action, row, beat)` either.

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
| `l2` | `{"modelled": true, "banks", "rows", "row_bytes", "interleave": "row-mod-banks"}` | the first row, `< rows` |
| `acc` | as for L2 | the first row, `< rows` |
| `l1` | `{"modelled": true, "rows", "cols"}`: the array's edge | the edge lane, `< max(rows, cols)`, or none |

Not modelled is `{"modelled": false, "why": "<reason>"}`.

`row-mod-banks` is the RTL's interleave: consecutive rows walk the banks, so row *r* is in bank
*r mod banks*. A different interleave needs a new name in the format.

kpu-sim's L-B models DRAM and L3, its addressed slots. It declares the rest not modelled, because
L-B moves whole tiles:

- L2's banks and rows (L-B stages tiles in L2 slots, but not by row);
- the accumulator's banks and rows (L-B holds the accumulator as a whole tile in the fabric);
- the array's edge (L-B feeds whole tiles to the fabric).

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
| G1 geometry | every layer is declared; a modelled L3 is the machine's slots and banks; L2 and the accumulator have banks, rows, a row width and a known interleave; L1 has rows and cols |
| T1 time | L-B has no time, no `cycle` or `end` column and no result cycles. A timed level has all three; every event has `cycle <= end`; the latest end is `time.cycles`; and `result.cycles` equals it |
| E1 references | `id` is the row; every index resolves; layers are modelled; addresses are inside their layer; a transaction moves between its kind's layers |
| E2 identities | §5.3: no two events share a key, or a (kind, action, row, beat) |
| E3 progress | §5.4: a row event is a child of its kind's transaction, of the same tile and action, inside the tile, and inside its parent's time |
| E4 credits and tokens | §5.2: a token goes downstream after its tile arrived; a credit goes upstream after its tile left |
| E5 the join | §5.1: a transaction's action has the matching kind, tile and L3 slot; with full joining, one transaction per data step |
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

## 9. Producers

| Producer | Level | Writes |
|---|---|---|
| kpu-sim `kpu-run --dflow` | L-B | version 6: `time` null; DRAM and L3 modelled, the rest declared not modelled; fp32 dense operands; no expected values; an empty `events` table with no time column; `result` with no cycles and values `unchecked` |
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

- **kpu-sim's timed levels writing events.** L-T1 still writes its overlay (`lt1_legs`) into the L-B
  record and its stations into `.tflow`. An L-T1 or L-CA record at `level: "L-T1"` with events is
  the next producer. It needs only the event rows; the format is ready for them.
- **L-CA's slot ids.** Until `l3-placement-at-lb.md` step 6, L-CA's `slot_id` is a per-DMA-engine
  counter. Its events would carry the walker's slot only after that step.
- **Golden event traces.** Ruled out by D3, as CI references. Event comparison between two records
  is a debugging tool, joined on the identity key (§5.3).

## 12. Verification

```bash
python3 tools/trace/dflow_rtl_sample.py -o rtl_sample.dflow
python3 tools/trace/dflow_check.py rtl_sample.dflow                     # OK
python3 tools/trace/dflow_check.py --result rtl_sample.dflow            # cycles 212, values match
build/tools/kpu-run/kpu-run --program p.csp --level behavioral --dflow lb.dflow
python3 tools/trace/test_dflow_check_v6.py rtl_sample.dflow lb.dflow   # every v6 invariant bites
cd build && ctest -L domainflow --output-on-failure
```
