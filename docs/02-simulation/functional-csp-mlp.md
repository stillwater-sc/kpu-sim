# Functional CSP MLP vertical slice

The concurrent CSP timing executor can execute value-producing MLP layers while
preserving its existing credit, TagCAM, queue, dependency, and backpressure
semantics.

## What is unified

`FunctionalMLPExecutor` executes inputs and weights through:

```text
DMA load -> L3 -> BlockMover -> L2 -> Streamer -> Compute
                                             |
                              next layer Compute (resident)
                                             |
                         final Drain -> L2 -> L3 -> DMA store
```

DRAM, L3, L2, L1, and compute each own distinct serialized byte arrays. Bytes
are copied only in response to the corresponding CSP completion event and are
retired when the final TagCAM/credit reference disappears. A functional
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

## General Domain Flow execution

`FunctionalDomainFlowProgram` represents an arbitrary dependency DAG. Its
event-driven runner dispatches all ready branches concurrently and completes a
node only when the matching DMA, BlockMover, Streamer, compute, or store event
occurs. Nodes can use tiled matmul or a user-supplied functional tile operation,
which covers elementwise operations, activations, reductions, and new kernels
without adding another disconnected simulator.

## Current kernel coverage

- Dense FP32 matmul, optional bias, and ReLU are supported.
- Each MLP layer currently uses one logical A, B, and C tile; the matmul compute
  primitive itself accepts multiple K tiles.
- Generic compute callbacks allow arbitrary tile transformations under the same
  transactional ordering, while optimized built-in implementations for conv,
  attention, mixed precision, and other kernels remain future library work.
- Intermediate MLP activations remain in compute storage across layers; only
  the final activation takes the drain/writeback/store path.

The next extension should lower the compiler's existing `TileDataFlowGraph`
nodes into `FunctionalDomainFlowProgram` and add optimized operator-library
implementations, without changing the simulator's memory or ordering model.
