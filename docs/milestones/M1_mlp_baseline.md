# DNN Milestone M1: MLP Baseline on the CSP Executor

**Milestone:** M1 (issue #129) — first rung of the DNN milestone ladder
(`docs/plans/csp_pattern_coverage_roadmap.md`, Section 6)
**Date:** 2026-07-11
**Demo:** `build/examples/milestones/m1_mlp_baseline`
**Status:** Demonstrate ✓ Validate ✓ Benchmark ✓

---

## What this milestone shows

Multi-layer perceptron inference executing **with real values on the
credit-based dataflow pipeline** — not a host-side reference with a timing
overlay, but actual floating-point tiles traversing
DRAM → L3 → L2 → compute → L2 → L3 → DRAM under credit/tag-CAM flow
control, with intermediate activations staying resident in compute storage
between layers. Every output element is validated against an independent
host oracle.

Two networks:

- **XOR 2-4-1** — the classic sanity network with hand-derived weights and
  exact expected outputs. Small enough to read its entire credit-dataflow
  trace event by event: the educational artifact.
- **MNIST-shape 784-128-64-10** — the canonical MNIST MLP topology with
  deterministic synthetic weights (fixed-seed LCG, reproducible on any
  platform), validated elementwise against a host reference forward pass.
  Real trained weights are a drop-in once a weight-loading path lands
  (E5/E15 scope).

## Reproduce

```bash
cmake --preset release && cmake --build --preset release --target m1_mlp_baseline
./build/examples/milestones/m1_mlp_baseline                    # benchmark table
./build/examples/milestones/m1_mlp_baseline --trace-dir traces # + Chrome traces
```

Traces open at https://ui.perfetto.dev — lanes show the memory controller,
DMA, BlockMovers, and row/column streamers; the credit acquire/release and
tile-arrival events make the dataflow protocol visible. The demo is also a
CI test (`ctest -R m1_mlp_baseline`), so this milestone cannot silently
regress.

## Measured results

Linux x86-64, gcc 13, release build, simulator commit at time of writing.
Cycles are simulated KPU cycles (1 GHz reference clock). Stall columns are
aggregate per-component-category cycles (a stalled component's wait cycles
overlap other components' useful work — high stall counts with completed
work indicate pipeline slack, not failure).

| configuration | batch | cycles | cyc/sample | DMA stalls | BM stalls | STR stalls | max abs err | check |
|---|---|---|---|---|---|---|---|---|
| XOR 2-4-1 (minimal pipeline) | 4 | 235 | 58.8 | 0 | 286 | 175 | 0.0 | PASS |
| XOR 2-4-1 (default pipeline) | 4 | 235 | 58.8 | 0 | 425 | 175 | 0.0 | PASS |
| MNIST-shape (default pipeline) | 16 | 12,028 | 751.8 | 0 | 12,211 | 8,625 | 0.0 | PASS |
| MNIST-shape (default pipeline) | 64 | 12,083 | 188.8 | 0 | 12,229 | 11,565 | 0.0 | PASS |
| MNIST-shape (default pipeline) | 256 | 24,019 | 93.8 | 0 | 24,015 | 23,325 | 0.0 | PASS |
| MNIST-shape (minimal pipeline) | 64 | 16,247 | 253.9 | 0 | 3,710 | 15,762 | 0.0 | PASS |

Pipeline configurations: *default* = 1 MC, 1 DMA, 4 BlockMovers, 2+2
streamers, 32 L3 buffers / 64 L2 banks; *minimal* = one of everything,
4 L3 / 4 L2.

## What the numbers say

- **Validation is exact on every tested output.** Max absolute error is
  0.0 across all runs — every output element equals the host oracle's
  bit-for-bit. (This establishes output equality, consistent with the CSP
  compute performing equivalent fp32 arithmetic; internal operation order
  is not separately instrumented.)
- **Batch amortization is the headline curve:** 752 → 189 → 94
  cycles/sample from batch 16 → 64 → 256. Weight movement is paid once per
  layer regardless of batch, so larger batches amortize it — the expected
  bandwidth-vs-reuse behavior, now measurable on the KPU pipeline.
- **Pipeline parallelism is visible:** the minimal pipeline pays ~34% more
  cycles (16,247 vs 12,083 at batch 64), isolating the contribution of
  parallel BlockMovers/streamers to overlap.
- **Zero DMA credit stalls everywhere:** at this scale the pipeline is
  never starved for L3 buffers; the BM/STR stall counts reflect components
  idling while waiting for upstream tiles — slack, not contention. The
  constrained-envelope stress story starts at M2 where working sets exceed
  the pools.

## Honest limitations (what M1 is not)

- **Single tile per matrix.** Each layer's activation/weight matrix moves
  as one tile (e.g., the 784×128 weight matrix is one 401 KB payload).
  Functional behavior and movement ordering are real; the *tiled*
  decomposition against 64 KB buffers — and the timing fidelity that comes
  with it — is exactly what the E5 (GEMM family) and E10 (fused epilogue)
  epics add. M1 numbers are a baseline, not a performance claim.
- **Compute latency is envelope-level.** Per-tile compute cost does not yet
  scale with the 784-deep reduction inside a single tile (K-scaling exists
  at tile granularity, per #63); tiled M2+ milestones inherit the full
  model.
- **Synthetic weights** for the MNIST-shape network (deterministic, oracle-
  validated). Classification-accuracy demos need the weight-loading path.

## Milestone ledger

| Tier | Evidence |
|---|---|
| Demonstrate | end-to-end CSP execution; Chrome traces via `--trace-dir` (XOR trace ~100 KB — readable; MNIST b64 ~46k events) |
| Validate | elementwise oracle comparison, tolerance 1e-4, measured 0.0 |
| Benchmark | table above: batch sweep 16/64/256, pipeline sweep, stall breakdown; regenerable with one command |

Next rung: **M2 ResNet-18** (#130) — needs Wave 1 (conv2d family, pooling,
GEMM completion) plus the CNN half of Wave 2 (BN fold, fused epilogue).
