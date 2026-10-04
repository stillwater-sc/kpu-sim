# Tile-flow viewer

Step 1 of the tile-flow debugger (#286; `docs/plans/tile-flow-debugger.md`): where every tile
was, and when, on the physical floorplan and as station swimlanes.

```bash
# a run, and the floorplan of the same deployment
kpu-run --algo matmul --size 256 --tile 32 --level block-sequential \
        --deploy tests/program/deploy/kpu_t64.json --tflow run.tflow
kpu-floorplan --deploy tests/program/deploy/kpu_t64.json --json t64_floorplan.json

# check it, then look at it
python3 tools/trace/tflow_check.py run.tflow
python3 tools/visualization/tileflow/pack.py run.tflow --floorplan t64_floorplan.json -o run.html
```

Open `run.html` in a browser. It needs no server and no other file. Alternatively, open
`index.html` and choose the bundle folder (and floorplan) with the file pickers.

**Zoom (swimlanes and the occupancy strip share one time window):** drag across the swimlanes
or scroll the wheel over them to zoom in around the pointer; scroll the other way, press
**Zoom out** (or `-`) to double the window, **Back** (or Backspace) to return to the previous
view, and **Whole run** (or `0`) to see everything. Click to move the cursor.

At L-T1 the viewer draws L3 and the movers as pooled fills, because the record holds a pool
occupancy, not a per-tile one. L2 and L1 are hatched as not modelled. It does not check
invariants: that is `tflow_check.py`'s job.
