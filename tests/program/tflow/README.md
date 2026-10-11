# Tile-flow fixtures

- `shared_reader.l0`: the #279 program. Two ops read one tile, and the higher-indexed one
  finishes first.
- `shared_reader_v3.tflow/`: that program's L-T1 record, `.tflow` version 3. It was written by
  the L0 L-T1 executor with `--compute-tiles 2 --l3-tiles 5`, and it passes all nine
  invariants. Under the old release rule, TF9 fails it.

Version 3 is an L0 run's record. The executor that wrote it retired in kpu-run-csp-programs
step 4e, so nothing writes version 3 any more, but `read_tflow`, `tflow_check.py` and the
viewer still read it. This bundle is the version-3 input for three tests:

- `tflow_check_shared_reader`: the checker passes it.
- `tflow_check_selftest`: each invariant fails on a copy edited to break it.
- `tileflow_viewer_smoke`: the viewer packs it.

The bytes are the fixture. `.gitattributes` keeps git from transforming them, because the
manifest gives byte offsets into the columns. A CSP program's record (version 4) is written
fresh by the kpu-run tests in `tools/kpu-run/CMakeLists.txt`.
