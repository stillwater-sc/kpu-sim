# `tile_characterize` — how to run, test, and read the results

A hands-on guide for the tile-program characterization / design-of-experiments (DoE)
harness in this directory (`tile_characterize.cpp`). For the metric *definitions* and
the modeling rationale see `docs/tools/tile-characterization.md`; for the operator /
kernel context see `docs/plans/plasma-tile-algorithms.md`.

The harness sweeps **(algorithm × size × tile-shape × compute-tiles × topology)** and,
for each cell, (1) builds the L0 tile program, (2) runs the functional reference and
**validates** it against an oracle, and (3) reports structural + first-order modeled
performance/energy metrics.

---

## 1. Build

The harness is built as part of the normal CMake build:

```console
cmake --preset release
cmake --build --preset release --target tile_characterize
```

The binary lands at `build/examples/characterize/tile_characterize`.

## 2. Run (quick start)

```console
# matmul, 128^3, 32-tiles, sweep the compute-tile count on a checkerboard
build/examples/characterize/tile_characterize \
    --algo matmul --sizes 128 --tiles 32 --compute-tiles 1,4,16,64 --topology checkerboard
```

`--help` prints every flag. The common ones:

| Flag | Meaning | Default |
|---|---|---|
| `--algo matmul\|lu` | which tile program | `matmul` |
| `--sizes N[,N...]` | square problem size(s) | `128` |
| `--tiles T[,T...]` | tile dimension(s) | `32` |
| `--compute-tiles C[,C...]` | CF tile count(s) — the **concurrency sweep** | `1,4,16` |
| `--topology single\|news\|checkerboard[,...]` | spatial layout (sets movement lanes) | `single` |
| `--macs-per-cycle` `--bytes-per-cycle` `--pj-per-mac` `--pj-per-byte` | device cost knobs | `256 / 64 / 1 / 20` |
| `--l3-tiles N` | L3 budget in tiles for feasibility (`0` = unbounded) | `0` |
| `--csv FILE` `--json FILE` | write the full metric table | — |
| `--trace FILE` | Chrome trace of the **first** cell | — |
| `--disasm` | print the tile sequence of the first cell | — |
| `--no-validate` | skip the functional oracle check | (validate on) |

Every factor flag is comma-separated → a **full-factorial** sweep (one row per cell).

## 3. Read the output

Each row is one experiment cell:

```
algo   size tile  CF topology      macs        AI    crit_path   makespan  cmp_util  bound  energy_pJ    err
matmul  128   32   1 checkerboard   2.1e+06   7.11       576.0    14976.0     0.55   mov    1.4e+07        0
matmul  128   32  16 checkerboard   2.1e+06   7.11       576.0      768.0     0.67   mov    1.4e+07        0
```

| Column | Meaning |
|---|---|
| `macs` | total multiply-accumulates (exact) |
| `AI` | arithmetic intensity = flops / byte moved (higher = more reuse) |
| `crit_path` | modeled makespan at **unlimited** compute tiles (dependency-limited floor) |
| `makespan` | list-scheduled modeled cycles on this device |
| `cmp_util` | compute-tile utilization on this device |
| `bound` | `cmp` = compute-bound, `mov` = movement-bound |
| `energy_pJ` | modeled total energy (compute + movement + leakage) |
| `err` | functional `max_err` vs. the oracle (`0` for integer matmul; ~`1e-6` for LU); `-1` if `--no-validate` |

**`err` is the correctness check** — a nonzero-beyond-tolerance `err` means the tile
program computed the wrong answer. The CSV/JSON carry the full metric set (peak live
tiles, movement bytes, lower bound, movement utilization, feasibility, …).

## 4. See the tile sequence

```console
# print the ordered tile ops (GETRF -> LASWP -> TRSM -> GEMM for LU)
build/examples/characterize/tile_characterize --algo lu --sizes 64 --tiles 32 --disasm

# write a chrome://tracing timeline (one lane per compute/movement resource)
build/examples/characterize/tile_characterize --algo lu --sizes 128 --tiles 32 \
    --compute-tiles 16 --topology checkerboard --trace lu.json
# then open chrome://tracing (or ui.perfetto.dev) and load lu.json
```

## 5. Test it

Two automated checks run under CTest (label `program` / `characterize`):

```console
# unit tests: DAG concurrency behavior + metric invariants
ctest --test-dir build -L program --output-on-failure

# just the characterization smoke test (a tiny LU sweep that must validate)
ctest --test-dir build -R tile_characterize_smoke --output-on-failure
```

- `test_tile_characterize` asserts the analytical core: matmul's makespan falls to its
  critical path as compute tiles grow, LU is far more serial, and the metrics are
  self-consistent (`makespan ≥ lower_bound`, energy sums, feasibility gating).
- `tile_characterize_smoke` runs the binary on a small LU grid and fails if the
  functional validation regresses.

To sanity-check a run yourself, confirm the `err` column is `0` (matmul) or `< 1e-4`
(LU) across the sweep.

## 6. Recipes — the machine principles to look for

```console
# (a) Concurrency ceiling: how far does makespan drop as you add compute tiles?
#     matmul scales to crit_path; LU saturates early (panel dependencies).
tile_characterize --algo lu     --sizes 128 --tiles 32 --compute-tiles 1,4,16,64 --topology checkerboard
tile_characterize --algo matmul --sizes 128 --tiles 32 --compute-tiles 1,4,16,64 --topology checkerboard

# (b) Compute vs movement bound: watch the `bound` column and AI.
tile_characterize --algo matmul --sizes 256 --tiles 32 --compute-tiles 16 \
    --bytes-per-cycle 16,256          # note: sweep by running twice; bandwidth is a single value per run

# (c) Tiling <-> reuse: bigger tiles raise AI and cut movement-bound makespan/energy.
tile_characterize --algo matmul --sizes 256 --tiles 16,32,64 --compute-tiles 16 --topology single

# (d) Feasibility: does the working set fit the L3 budget?
tile_characterize --algo matmul --sizes 256 --tiles 32 --compute-tiles 16 --l3-tiles 8
```

## 7. Caveats

Performance and energy are **first-order models** (all coefficients are CLI knobs)
because the L0 layer is timing-free. Use the numbers for **relative** comparison
across the experiment grid, not as absolute cycle/energy counts. The models sharpen
as later increments land (L1 stream timing; the driver-JIT placement pass for real
single/NEWS/checkerboard placement — `docs/plans/kpu-program-model.md` §4a) without
changing this interface. Operator coverage is currently matmul + tile-LU; the
roadmap for the rest (Cholesky/QR/pairwise-LU) is in
`docs/plans/plasma-tile-algorithms.md` (issues #242/#243/#244).
