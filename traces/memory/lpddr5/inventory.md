# LPDDR5 pattern records

Each LPDDR5 pattern (`patterns/memory/lpddr5/`) writes its run as a `.mflow` bundle. Run on its
own, a pattern writes it here, under its level; under CTest it writes it to
`build/patterns/memory/lpddr5/<name>.mflow` instead. The bundles are generated, not committed
(`.gitignore`).

| Level | Pattern      | Records                                                  |
|-------|--------------|----------------------------------------------------------|
| 1     | Single Bank  | `single-bank/{page_hits,page_conflicts,mixed_rw}.mflow`   |
| 2     | Two Bank     | `two-bank/{same_group,diff_groups}.mflow`                 |
| 3     | Three Bank   | `three-bank/{same_group,mixed_groups}.mflow`              |
| 4     | Four Bank    | `four-bank/{full_group,across_groups,page_hit_burst}.mflow` |
| 5     | Dual Channel | `dual-channel/{independent,interleaved}.mflow`            |
| 6     | Bandwidth    | `bandwidth/{max_bandwidth,page_burst}.mflow`              |
| 7     | Complex      | `complex/{strided,random,tile_load,stream,multi_dma}.mflow` |

Check one, and see what it did:

```bash
python3 tools/trace/mflow_check.py traces/memory/lpddr5/dual-channel/interleaved.mflow --summary
```

Open one in the DRAM views (`tools/visualization/dram/`) through the debug server, or pack a
view with it: `python3 tools/visualization/dram/pack.py timing <bundle> -o timing.html`.

The Chrome-trace JSON these patterns used to write (`*_trace.json`) is retired
(`docs/plans/cycle-mflow-and-debug-server.md` step 11).
