# Tile-program characterization harness

`examples/characterize/tile_characterize` is a **design-of-experiments (DoE)**
harness that drives kpu-sim's L0 `TileProgram` layer to *validate, trace, and
characterize* linear-algebra tile programs across a grid of algorithm, size, shape,
and hardware configurations. Its purpose is to **discover the principles of the
machine** — the domain-flow analogue of how CUDA adapts warps/occupancy to problem
size and resources: given a tiled program and a device, how much concurrency can it
exploit, where does it bottleneck (compute vs. movement), and what does that cost in
time and energy?

## What it measures

For each experiment cell it (1) builds the tile program, (2) runs the L0 functional
reference and validates it against an oracle (matmul vs. naive; LU vs. `P·A=L·U`),
and (3) computes:

- **Structural (exact):** op counts (feeds/drains/computes), total MACs, bytes
  moved, **arithmetic intensity** (flops/byte), distinct tiles, and **peak live
  tiles** (an L3 footprint proxy).
- **Concurrency (from the tile-dependency DAG):** the **critical path** (makespan at
  unlimited compute tiles) and a **list-scheduled makespan** on a finite device
  (compute tiles + movement lanes), plus compute/movement utilization and whether the
  cell is **compute- or movement-bound**.
- **Energy (modeled):** compute (MACs), movement (bytes), and leakage (∝ active
  resources × makespan) in pJ.
- **Feasibility:** does the peak working set fit the device's L3 tile budget?

> Performance and energy are **first-order models** (`device_model.hpp`) until the L1
> stream layer and driver-JIT placement land (`docs/plans/kpu-program-model.md`
> §4a). Every coefficient is a CLI knob — the harness is for **relative** comparison
> across the experiment grid, not absolute cycle counts.

## Usage

```console
tile_characterize --algo lu --sizes 64,128,256 --tiles 16,32 \
                  --compute-tiles 1,4,16,64 --topology single,checkerboard \
                  --csv out.csv --json out.json --trace first.json --disasm
```

| Factor | Flag | Meaning |
|---|---|---|
| algorithm | `--algo matmul\|lu` | which tile program |
| size | `--sizes` | square problem size(s) |
| tile shape | `--tiles` | tile dimension(s) `T` |
| **concurrency** | `--compute-tiles` | CF tile count(s) — the resource sweep |
| topology | `--topology single\|news\|checkerboard` | §4a spatial layout (sets movement lanes) |
| device cost | `--macs-per-cycle` `--bytes-per-cycle` `--pj-per-mac` `--pj-per-byte` | HW knobs |
| capacity | `--l3-tiles` | L3 budget (tiles) for feasibility |

Observability:
- `--disasm` prints the **tile sequence** for the first cell (see how the program is
  ordered — GETRF → LASWP → TRSM → GEMM for LU, feed/GEMM/drain for matmul).
- `--trace FILE` writes a **Chrome trace** (`chrome://tracing` / Perfetto) of the
  scheduled tile ops, one lane per compute/movement resource.
- `--csv` / `--json` emit the full metric table for offline analysis.

## Example principles it surfaces

- **Concurrency ceiling by algorithm.** Matmul's independent output-tile chains scale
  makespan down to the critical path as compute tiles grow; LU's panel dependencies
  saturate much earlier (utilization collapses once CF exceeds the exploitable
  width) — so the right allocation differs per operator.
- **Compute vs. movement bound.** Under realistic coefficients matmul is
  *movement-bound* (matching the ResNet DRAM-bound study); the `bound` column and
  utilizations show it directly.
- **Tiling ↔ reuse.** Larger tiles raise arithmetic intensity, shrinking
  movement-bound makespan and energy — the tile-size/reuse trade-off, quantified.

## Where this goes next

The harness is built on the L0 DAG, so as later increments land it deepens without
changing its interface: L1 stream signatures replace the modeled per-op cost with
real timing; the driver-JIT placement pass (§4a) replaces the list scheduler's
homogeneous workers with actual single/NEWS/checkerboard placement; and the PLASMA
capability probes (`docs/plans/plasma-tile-algorithms.md`) add the kernels beyond
matmul/LU. The DoE grid is the instrument for discovering the allocation and layout
principles those layers must implement.
