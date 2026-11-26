# KPU Record Checkers

Python checkers for the simulator's records, standard library only. Each exits **0** when every
invariant holds, **1** when one is violated (parse the violations and fix the root cause), and
**2** when the record cannot be read (there are no violations to parse: fix the recording or the
path). Each has a self-test that breaks a clean record and sees every check fail.

| Checker | Record | Self-test |
|---------|--------|-----------|
| `mflow_check.py` | `.mflow`, the memory side: bursts, DRAM commands, requests, buffers, waits | `test_mflow_check.py` |
| `tflow_check.py` | `.tflow`, the tile flow | `test_tflow_check*.py` |
| `dflow_check.py` | `.dflow`, the domain flow (`dflow_diff.py` joins two) | `test_dflow_check*.py` |
| `structure_check.py` | the structure record (`structure_diff.py` compares two) | `test_structure_check.py` |

## mflow_check.py

The LPDDR5 patterns, `kpu-memsim --out` and `kpu-run --memory-subsystem --memory-mflow` write
`.mflow` bundles. The checker's docstring lists every check (TF10, TF11, M1–M11);
`patterns/memory/lpddr5/INVARIANTS.md` maps the LPDDR5 invariants to them.

```bash
python3 tools/trace/mflow_check.py <bundle.mflow> [--summary] [--json] [--verbose] [--run run.json]
```

`--summary` reports what the run did. It replaces `kpu-trace-summary`, which read the retired
LPDDR5 Chrome trace:

```text
$ python3 tools/trace/mflow_check.py build/patterns/memory/lpddr5/page_conflicts.mflow --summary
mflow_check: build/patterns/memory/lpddr5/page_conflicts.mflow (lpddr5, makespan 187, 4 bursts, 11 commands)
  run: 187 cycles; 4 requests (4 loads, 0 stores, 0 fills), 4 bursts (4 reads, 0 writes)
  commands: 11 (4 ACT, 4 RD, 0 WR, 3 PRE, 0 REF)
  pages: 0 hits, 1 empty, 3 conflicts; hit rate 0.0%; 1.0 RD/WR per ACT
  bytes: 256 (256 read, 0 written); 1.37 bytes/cycle; 17.1% of the 8.00 ceiling
  burst latency (cycles, posted to done): min 36.0, mean 111.0, p50 86.0, p95 186.0, max 186.0
  banks: 1 active
    mc0/ch0/bg0/ba0      ######################################## 11 commands
  checked TF10 TF11 M1 M2 M3 M4 M6 M5 M7 M9 M10 M11
  PASS
```

With `--json`, the summary is the report's `summary` object.

## Viewing records from a laptop

The debug server, `tools/debug-server/`, lists every run and record under a root and links each
`.mflow` bundle to the memory-side debugger and the DRAM views (`tools/visualization/dram/`):

```bash
python3 tools/debug-server/kpu_debug_server.py --root build/patterns/memory/lpddr5
# laptop: ssh -L 8080:localhost:8080 <server>, then open http://localhost:8080
```

It listens on loopback only and is reached through an SSH tunnel. See
`tools/debug-server/README.md`.

## See Also

- `traces/README.md` - the trace and record directory
- `patterns/memory/lpddr5/INVARIANTS.md` - the LPDDR5 invariants and the checks that enforce them
- `docs/plans/cycle-mflow-and-debug-server.md` - the `.mflow` record, its views and the server
