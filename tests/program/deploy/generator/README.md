# Vendored from the KPU RTL generator

`kpu_2x2_systolic.full.yaml` is copied unchanged from branes-ai/embodied-ai-soc-generator
(`configs/kpu_2x2_systolic.full.yaml`, at 5c6a037, 2026-11-13), the expanded KPUArchitecture
`kpugen expand` writes. `tools/deploy/kpuarch_to_deployment.py` converts it to
`../kpu_soc_2x2.json` and `../kpu_soc_2x2.notes.txt`. The `kpuarch_to_deployment_selftest` ctest
fails when either drifts from the conversion.

When the generator's config changes, copy the new file here and rerun:

```bash
python3 tools/deploy/kpuarch_to_deployment.py tests/program/deploy/generator/kpu_2x2_systolic.full.yaml \
    -o tests/program/deploy/kpu_soc_2x2.json --notes tests/program/deploy/kpu_soc_2x2.notes.txt
```

The converter's `GENERATOR_DEFAULTS` (slots, banks and latencies from kpugen's `SystolicConfig`
and `tile.py`, read at the same revision) change with kpugen too.
