# Functional CSP MLP vertical slice

The concurrent CSP timing executor can execute value-producing MLP layers while
preserving its existing credit, TagCAM, queue, dependency, and backpressure
semantics.

## What is unified

`FunctionalMLPExecutor` executes each dense layer through:

```text
DMA load -> L3 -> BlockMover -> L2 -> Streamer -> Compute
         -> Drain -> L2 -> Writeback -> L3 -> DMA store
```

Numeric tile payloads are owned by `ConcurrentTimingExecutor`. A functional
matmul consumes payloads only after every required feed occurrence completes.
The result becomes visible at the modeled compute-completion cycle, immediately
before the result tag is published for `DRAIN`.

Feed dependencies use occurrence counts rather than an "ever seen" bit. Reusing
the same tile in a later operation therefore cannot start a later compute from
an earlier feed.

## Run the proof

```bash
cmake --build build --target unified_xor_mlp
./build/examples/schedule/unified_xor_mlp
```

Expected completion markers:

```text
Numerical result: PASS
Transaction-ordered execution: PASS
```

The example deliberately configures one L3 buffer and one L2 bank. It must
produce the correct XOR outputs while reporting non-zero credit/tag stall
cycles.

## Scope boundary

This is the first honest functional/transactional vertical slice, not the final
general simulator:

- Dense FP32 matmul, optional bias, and ReLU are supported.
- Each MLP layer currently uses one logical A, B, and C tile; the matmul compute
  primitive itself accepts multiple K tiles.
- Payloads use a functional backing store. Transfers control when values become
  consumable but do not yet copy bytes through separate physical L3/L2/L1
  storage arrays.
- Conv, attention, reductions, mixed precision, and arbitrary Domain Flow
  programs still need functional compute implementations.
- `FunctionalMLPExecutor` creates a fresh CSP executor per layer. Cross-layer
  fusion and persistent on-chip activation residency remain future work.

The next extension should lower generated tiled matmul schedules into the same
functional compute specification, then validate multi-K accumulation and
multi-output-tile execution against the behavioral oracle.
