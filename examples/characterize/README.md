# `tile_characterize` — how to run, test, and read the results

A hands-on guide for the tile-program characterization / design-of-experiments (DoE)
harness in this directory (`tile_characterize.cpp`). For the metric *definitions* and
the modeling rationale see `docs/tools/tile-characterization.md`; for the operator /
kernel context see `docs/plans/plasma-tile-algorithms.md`.

Each invocation selects **one** algorithm via `--algo`; the harness then sweeps
**(size × tile-shape × compute-tiles × topology)** and, for each cell, (1) builds the
L0 tile program, (2) runs the functional reference and **validates** it against an
oracle, and (3) reports structural + first-order modeled performance/energy metrics.
Comparing matmul vs. LU is two separate invocations.

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
| `--dataflow output-stationary\|weight-stationary\|a-stationary\|fully-streaming[,...]` | **array dataflow** (L1 space-time mapping; matmul only; aliases `os`/`ws`/`as`/`hex`) | `output-stationary` |
| `--macs-per-cycle` `--bytes-per-cycle` `--pj-per-mac` `--pj-per-byte` | device cost knobs | `256 / 64 / 1 / 20` |
| `--l3-tiles N` | L3 budget in tiles for feasibility (`0` = unbounded) | `0` |
| `--csv FILE` `--json FILE` | write the full metric table | — |
| `--trace FILE` | Chrome trace of the **first** cell | — |
| `--disasm` | print the tile sequence of the first cell | — |
| `--no-validate` | skip the functional oracle check | (validate on) |

The **list-valued** factor flags (`--sizes`, `--tiles`, `--compute-tiles`,
`--topology`, `--dataflow`) are comma-separated → a **full-factorial** sweep (one row
per cell). `--algo` selects a single algorithm per run.

When a `--dataflow` is given for matmul, the schedule is timed by the **L1 stream
signatures** (`docs/plans/l1-stream-signatures.md`) — compute ops take their systolic
wavefront latency and the C drain is stretched by its bubble — so the makespan and the
`stat`/`bub`/`network` columns become **dataflow-sensitive**.

## 3. Read the output

Each row is one experiment cell:

```text
algo   size tile  CF topology    dataflow          stat bub network            makespan  cmp_util  bound  energy_pJ    err
matmul  256   32   4 single     output-stationary C      1 Mesh2D               36864.0     0.33   mov   1.07e+08        0
matmul  256   32   4 single     weight-stationary B      0 Mesh2D               34816.0     0.35   mov   1.07e+08        0
matmul  256   32   4 single     fully-streaming   -      0 Hexagonal+overlay    34816.0     0.35   mov   1.07e+08        0
```

| Column | Meaning |
|---|---|
| `dataflow` | the array space-time mapping (which operand is held stationary) |
| `stat` | the stationary operand (`C`/`B`/`A`; `-` for fully-streaming) |
| `bub` | the C-drain **bubble** (`1` for output-stationary, `0` for the dense dataflows) |
| `network` | required fabric — `Mesh2D` or `Hexagonal+overlay` |
| `makespan` | list-scheduled cycles (systolic + dataflow-sensitive when `--dataflow` is set) |
| `cmp_util` | compute-tile utilization |
| `bound` | `cmp` = compute-bound, `mov` = movement-bound |
| `energy_pJ` | modeled total energy (compute + movement + leakage) |
| `err` | functional `max_err` vs. the oracle (`0` for integer matmul; ~`1e-6` for LU); `-1` if `--no-validate` |

**`err` is the correctness check** — a nonzero-beyond-tolerance `err` means the tile
program computed the wrong answer. The CSV/JSON carry the full metric set (macs,
arithmetic intensity, critical path, peak live tiles, movement bytes, feasibility, …).

### What is `makespan`?

**Makespan** is a scheduling-theory term for the **total elapsed time to finish all the
work** — the cycle on which the *last* tile-op completes, measured from the start. It is
**not** the sum of every op's duration (that would be everything run one-at-a-time);
because tile-ops run concurrently on the available hardware, the makespan is the
*wall-clock span* of the whole schedule. Read it as **"how many cycles does this entire
tiled operator take on this device?"** — lower is faster.

The harness computes it by (1) building the **tile-dependency DAG** (every feed/compute/
drain is a node; edges are data dependencies), (2) giving each node a duration in cycles
(the first-order lumped model, or the L1 systolic durations when `--dataflow` is set),
(3) running a greedy list scheduler that packs those ops onto the device's *finite*
resources (`--compute-tiles` compute units + movement lanes) respecting dependencies,
and (4) taking the finish time of the last op.

It sits between two reference points:

- **`crit_path`** — the makespan with *unlimited* compute tiles: the dependency-limited
  floor. You can never beat the longest chain of dependent ops.
- **`lower_bound`** — `max(crit_path, compute_work / #tiles, movement_work / #lanes)`: a
  provable floor combining the chain limit and the resource-saturation limits.

so `makespan ≥ lower_bound ≥ crit_path`, and as you add compute tiles the makespan falls
*toward* the critical path until dependencies or movement bandwidth dominate — the
"concurrency ceiling" of recipe (a). Like all timing here it is in **modeled cycles**
(first-order; for *relative* comparison across configurations, not an absolute runtime).

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
#     --bytes-per-cycle takes ONE value, so run it twice to compare bandwidths:
tile_characterize --algo matmul --sizes 256 --tiles 32 --compute-tiles 16 --bytes-per-cycle 16
tile_characterize --algo matmul --sizes 256 --tiles 32 --compute-tiles 16 --bytes-per-cycle 256

# (c) Tiling <-> reuse: bigger tiles raise AI and cut movement-bound makespan/energy.
tile_characterize --algo matmul --sizes 256 --tiles 16,32,64 --compute-tiles 16 --topology single

# (d) Feasibility: does the working set fit the L3 budget?
tile_characterize --algo matmul --sizes 256 --tiles 32 --compute-tiles 16 --l3-tiles 8

# (e) Dataflow choice: which array mapping (stationary operand + network)?
#     output-stationary is mesh-friendly but pays a drain bubble; weight/A-stationary
#     drain densely; fully-streaming (hex) is dense/contention-free but needs a hex
#     network overlay. Watch the stat / bub / network / makespan columns.
tile_characterize --algo matmul --sizes 256 --tiles 32 --compute-tiles 4 \
    --dataflow output-stationary,weight-stationary,a-stationary,fully-streaming
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
