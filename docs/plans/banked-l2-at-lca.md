# Banked L2 at L-CA (#401, slice 1)

**Status:** planned 2026-11-17. The user chose this slice first, and made the three decisions in §2.
**Issue:** #401, mirroring #383 and #388 (RTL: esg#14, esg#24, systars#17).

## 1. The gap

L-CA's L2 is one pool of tile-sized slots (`l2_credits_`, `l2_tag_cam_`; `concurrent_timing_executor.hpp`).
Every L2 transfer takes `startup + ceil(bytes / rate)` cycles, whoever else is using L2. The
RTL's L2 is banked, with consecutive rows walking the banks, so two clients collide when they
want the same bank in the same clock. L-CA can't show that today, nor the reads and writes the
Streamer, the BlockMover and the Combiner share.

## 2. Decisions (the user's, 2026-11-17)

1. **Geometry.**
   - With a `systolic` block: `sp_banks` banks, `sp_rows_per_bank` rows each, a row of
     `grid_cols × tile_cols` input elements, and `spad_read_delay` clocks of read latency.
   - Otherwise, with `l2.banks_per_tile`: that many banks, a row of `fabric.pe_cols` elements
     (`macs_per_cycle`'s square root when no fabric is declared), the RTL's default read latency
     (4), and rows enough for the level's L2 slots. Each default is named in the level's
     `unmodelled` list.
   - With neither: today's pooled L2, unchanged.
   - In every case the bank count must be a power of two (esg#14). The deployment validator
     refuses another.
2. **Ports: esg#24's idealized scratchpad.**
   - Each bank serves 2 reads and 1 write per clock.
   - A row access whose bank is out of ports waits a clock. That's a stall the level counts and
     attributes.
   - Bank = the row's low `log2(banks)` bits (#383, esg#14). Port types from the macro
     (single-port, 1R1W; #388) come later.
3. **Rollout: re-baseline.**
   - The model applies wherever a deployment declares banks: S1, the T-series and kpu_soc_2x2.
   - Every test whose L-CA cycles move is updated. The PR records old and new cycles per test, as
     #458 did.
   - A test is pinned to the pooled L2 only when its claim is about the old model.

## 3. The model

- **Rows.** A tile of `h × w` elements of `e` bytes fills `n = ceil(h·w·e / row_bytes)` rows. An L2
  slot `s` starts at row `s · n`, mod the L2's rows. So slot `s`'s row `k` is in bank
  `(s·n + k) mod banks`.
- **A transfer is a sequence of row accesses**, one row per `ceil(row_bytes / rate)` clocks at its
  client's rate (the BlockMover's or the Streamer's bytes per cycle). Reads come first for a
  read-modify-write. Each access books a port on its bank at the earliest clock at or after its
  turn where the bank has a port of its kind free. It finishes when its last access is served
  (plus the read latency, for reads), plus the client's startup latency.
- **The port calendar** is per compute tile's L2: per clock, per bank, reads and writes booked. A
  transfer books all its accesses when it starts. Clients that start in the same clock book in
  the executor's tick order: the Streamer, then the drain and the Combiner, then the BlockMover
  (#401: "a shared write port, Streamer first"). Clocks before the current one are pruned.
- **The clients:**
  - BlockMover L3 → L2 (writes) and L2 → L3 (reads);
  - Streamer L2 → L1 feeds (reads);
  - drains L1 → L2 (writes);
  - the Combiner's merge, a read then a write per row (esg#24 requirement 5).
- **Credits are unchanged.** A slot is still a tile, under the same credit pool. Banks only time
  the transfers. Slot-state credits are #387's slice.
- **Values are unchanged.** Banks move no data of their own; the level's values stay L-B's
  within its bound.

## 4. What it reports

In this PR:

- **L-CA's outcome:** row accesses, and stalled accesses by client and by bank. `kpu-run` gets a
  line for the L2 banks.
- `level_models` reports `l2.banks_per_tile` as modelled at L-CA. Its `unmapped` notes name the
  defaults it leaves, and the parts of the `systolic` block it doesn't use.

Next, in the slice's second PR:

- **The bubble report:** a transfer's bank stalls as a wait cause.
- **The `.dflow` record (6.2):** L2 becomes `{"modelled": true, "banks", "rows", "row_bytes",
  "interleave": "row-mod-banks"}` when banked, and an event's L2 address is its first row, as in
  the RTL's form. Row events (BM_ROW, STR_ROW, DRAIN_ROW) come from the booked accesses. This is
  what #390's L2-bank lanes will draw.

## 5. Tests (mirroring #383 and #388)

- One bank: a matmul through it computes L-B's values (#383 `test_one_bank`).
- A bank count that isn't a power of two is refused (#383 `test_bank_count_must_be_a_power_of_two`).
- Row-to-bank mapping: row `i` is in bank `i mod N` (#383).
- Two reads and one write per bank per clock. A third read, or a second write, waits a clock, and
  the wait is counted.
- No Streamer stall for OS, WS and IS when only the Streamer reads (#388: two vectors per clock
  with no stall).
- A BlockMover write and a drain into the same bank in the same clock: the later one waits.
- Values bit-identical to the pooled L2's at every level (#388).
- The re-baselined characterizations, each with its old and new cycles in the PR.

## 6. Not in this slice

- Slot-state credits and the OS A/B pair (#387).
- L1 stream buffers and a bubble-free Streamer (#389).
- The per-PE wavefront (#385).
- Strided and partial-row BlockMover moves (#382).
- Macro-typed ports (#388).
- The 744 and 907 cycles of #401's done-when need all of these.
