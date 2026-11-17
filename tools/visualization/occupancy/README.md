# Resource occupancy viewer (#390)

A timed domain-flow record on one clock axis.

- **Process lanes:** one lane per process instance, with that instance's transactions: each DMA
  engine, BlockMover, Streamer and compute tile.
- **Slot lanes:** one lane per L3 and L2 slot, showing the tiles it holds, from the transaction
  that fills the slot to the credit that frees it.
- **Overlay:** a second record of the same program, joined event by event on the shared identity
  key (`docs/plans/shared-tile-flow-record.md` §5.3), as `tools/trace/dflow_diff.py` joins them.
  The first divergence is one click away.

```bash
build/tools/kpu-run/kpu-run --program p.csp --deploy d.json --level all \
    --dflow-lca lca.dflow --dflow-lt1 lt1.dflow
python3 tools/visualization/occupancy/pack.py lca.dflow --overlay lt1.dflow -o view.html
```

Or open `index.html` and choose a bundle folder, then an overlay. The records can be kpu-sim's
timed levels (`--dflow-lt1`, `--dflow-lca`), the RTL's, or `tools/trace/dflow_rtl_sample.py`'s.
An L-B record has no time, and is refused.

## What it shows

- **Process lanes.** Each transaction (DMA, BLOCK_MOVE, STREAM, COMPUTE, DRAIN) is drawn as a bar
  on its process's lane over `[cycle, end)`. A joined overlay event is drawn as an outline on the
  same lane, with the first divergence in red.
- **Slot lanes.** A tile holds an L3 or L2 slot from the end of the transaction that lands it
  there (the slot is that transaction's destination address) to the credit that frees it (the
  credit's source address). A slot not freed by the record's end is held to the end.
- **Selecting a tile** (from the menu, or by clicking one of its bars) lights its whole life:
  every transaction and slot holding of that tile.
- **The inspector** gives the clicked event's fields and, with an overlay, the joined event's
  start and duration deltas.
- **The record's own `not_modelled` notes** are listed beneath the lanes.

## Not yet

These come with the levels that model them, and the record's intervals for them:

- L2 bank lanes and bank conflicts (#401, slice 1's second PR);
- L1 stream buffers and the Streamer's sequencer (#389);
- the fabric's edges, the result buffers and the accumulator (#385, #389);
- the invariants panel (credit conservation per slot, drain collisions, bubbles).

## Tests

```bash
python3 tools/visualization/occupancy/test_viewer.py <a.dflow> <b.dflow> [more.dflow ...] --lb <lb.dflow>
```

The page's data layer runs in Node and is checked against `dflow_check`'s reading of the same
records:

- every transaction is on one lane, its process's;
- every slot holding runs from a landing to the credit that frees it, and no slot is held twice
  at once;
- a record joins itself completely;
- the overlay's join matches `dflow_diff.py`'s.

`pack.py` is tested too. The page is loaded in a headless Chromium, if Playwright has one.
