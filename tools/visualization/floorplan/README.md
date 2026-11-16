# The floorplan view (Overview)

`index.html` draws a machine's floorplan from a **structure record**, `kpu-structure` v1
(`docs/plans/shared-structure-record.md`), from either producer:
- the model: `kpu-floorplan --deploy <spec.json> --structure fp.structure.json`, natively;
- the RTL generator: `kpu-gen floorplan <spec.yaml> -o floorplan.json`, converted.

It is #447, moved from branes-ai/embodied-ai-soc-generator#59 by ADR 0005: the views live in
kpu-sim, and the RTL writes records in kpu-sim's formats.

## Use

```bash
# The model writes a record directly:
kpu-floorplan --deploy tests/program/deploy/kpu_t64.json --structure t64.structure.json
# A record from the RTL generator's floorplan.json (or the model's older --json):
python3 tools/trace/structure_from_floorplan.py floorplan.json -o fp.structure.json
# One self-contained page, to open, share or attach (pack.py converts a floorplan itself too):
python3 tools/visualization/floorplan/pack.py fp.structure.json -o floorplan.html
```

Or open `index.html` and choose a record, or drop one on the page. No server, no build step, no
dependencies: it works offline.

## What it shows

- **The die**, every placed block coloured by kind, origin at the lower-left, in mm. Wheel or
  `+`/`−` to zoom, drag to pan, **Fit** to see the whole die. The pointer's position reads out in mm.
- **Layers:** blocks, labels, and each of the record's networks (NoC request and response, the
  control tree, DRAM to L3, the model's NoC ring). Each has show/hide and an opacity. Networks
  start at 45% so they read as context. Presets: blocks only, networks, everything.
- **Die size:** the die and its area, and the breakdown by category when the record has one.
  **Compare with…** a second record shows the change in die area and in each category.
- **Inspector:** a clicked block's id, kind, label, parent, class, layer and role, footprint and
  area, its sites, whether the RTL generates it, its resources, and the network nodes (routers)
  that stand for it.
- **Provenance:** who wrote the record, from what configuration, converted from what.

An estimated floorplan says so: the model's is on an illustrative pitch, and the RTL generator's
is an area model's estimate, not a layout.

## Next

- The Investigate view (#448): connectivity lenses that follow a network's `routing` from one
  block to another, and drill-down through the block tree.
- Publishing runs from remote servers to a viewer service (#449).

## Tests

`test_viewer.py` (CTest `floorplan_viewer_smoke`). `pack.py` is tested over every sample in
`tests/program/structure/`. With Playwright and its Chromium installed, each packed page also
loads headlessly, and the test checks:
- the blocks and network paths drawn;
- the inspector, the layer toggles and presets, zoom and fit, and Compare;
- that another format is refused on the page;
- that a record's strings cannot inject markup or script.
