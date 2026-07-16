# M2: ResNet on the CSP Executor — a Kernel-Graph DFG design

**Status:** DRAFT (M2-T1 design) — awaiting review
**Milestone:** #130 (DNN Milestone M2: ResNet-18/50 on the CSP executor)
**Depends on (all landed this program):** E2 elementwise/broadcast (#71), E6
conv2d (#75), E7 pooling (#76), E9 batchnorm inference fold (#78), E10 fused
epilogue (#79), matmul/GEMM (#74). All five M2 capability gate cells are done.
**Template:** M1 MLP baseline (`examples/milestones/m1_mlp_baseline.cpp`,
`docs/milestones/M1_mlp_baseline.md`).

---

## 1. Goal

Build the first recognizable CNN — ResNet — **as a dataflow graph (DFG) of
operators**, executed on the credit-based CSP executor, so we exercise in one
artifact: the operators (conv, pool, folded-BN, ReLU epilogue, residual add,
global-average-pool, FC), the **fusions** (conv+BN fold, conv+bias+ReLU
epilogue), and the **layout/graph transformations** (im2col, topological
scheduling, residual-branch structure + fusible-pair analysis). M2 executes
nodes sequentially in topological order — the measured concurrency is the
tile-level pipeline overlap *within* each op, not operator-branch parallelism
(that is a named follow-on, §5). Deliver M1's three-tier DoD:

- **Demonstrate:** the graph runs end-to-end through the CSP executor; a
  `--trace-dir` exports Perfetto Chrome traces of the credit dataflow.
- **Validate:** functional equivalence vs a host oracle, per-layer and
  end-to-end, under default and constrained envelopes (tol 1e-4, as M1).
- **Benchmark:** cycles, utilization, stall breakdown, per-layer table, envelope
  sweeps.

## 2. The landscape (why a bridge is needed)

The repo has an **operator DAG** and a **value path**, and they are not yet
wired together — bridging them *is* M2:

- **`KernelGraph`** (`include/sw/kpu/kernel_graph.hpp`) is the operator DAG.
  `add_kernel(Kernel::create_conv2d/batchnorm/pool2d/global_avg_pool2d/
  elementwise/matmul, ...)`, `add_edge(prod, cons, "C", "A")`, plus the passes we
  want to exercise: `get_execution_order` (topological), `get_execution_levels`
  (the residual **skip-branch structural independence**, for analysis + viz),
  `find_fusible_pairs`, `to_dot`
  (visualization). Every ResNet kernel factory already exists. **But**
  `compile()` lowers to the ISA `DMProgram` (sequential fallback) — *not* the
  credit-dataflow CSP value path, and it does not produce oracle-validated
  numbers.

- **The CSP value path** is the schedule-generator layer this program just
  completed: `Conv2D/Pooling/BatchNorm/Elementwise/MatMulScheduleGenerator` →
  `ScheduleResult` → `ConcurrentTimingExecutor`, with the value helpers
  `conv2d_im2col.hpp`, `batchnorm_affine.hpp`, `pooling_window.hpp` and the
  functional binders. This is where real fp32 tiles flow DRAM→L3→L2→compute→…
  and where M1's oracle validation + credit/stall benchmarking live.

- **A verified ResNet BasicBlock already exists** in
  `verification/kernels/class3_multi_branch/verify_residual.cpp` — but on the
  compute-fabric path, chaining `submit_conv2d/submit_elementwise` imperatively
  and validating against composed host oracles. It is the topology + oracle
  template; M2 re-expresses it as a DFG on the CSP value path.

## 3. The decision: KernelGraph for structure, CSP schedule-generators for values

**M2 expresses ResNet as a `KernelGraph` (the DFG) and executes it through a new
graph→CSP bridge that runs each node on the schedule-generator value path,
threading activations between nodes and applying the operator fusions.** The
graph gives structure, fusion detection, execution-levels, and visualization; the
CSP path gives oracle-validated numbers and credit/stall benchmarks.

Concretely, three pieces:

1. **ResNet topology builder** — constructs the `KernelGraph`: stem
   (7×7 conv s2 → BN → ReLU → 3×3 max-pool s2), four stages of **BasicBlocks**
   (`conv3×3 → BN → ReLU → conv3×3 → BN → (+ identity/1×1-projection) → ReLU`,
   the first block of stages 2–4 stride-2 downsampling with a 1×1 projection
   skip), global-average-pool, FC. ResNet-18 first; ResNet-50 (bottleneck) is a
   follow-on. `to_dot` emits the graph for the writeup.

2. **Graph→CSP executor** (the new bridge) — walks `get_execution_order`
   (topological) and runs each kernel node **sequentially** on a shared
   `ConcurrentTimingExecutor`, seeding the node's input tiles from the producing
   node's output activation (kept resident/in-DRAM between nodes) and reading its
   output back — the same seed/binder pattern the per-op functional tests use.
   The parallelism M2 measures is the **tile-level** concurrency *within* each op
   (DMA / BlockMover / Streamer / compute overlap on the pipeline), not
   operator-graph-level branch overlap: nodes execute one at a time in topo
   order. `get_execution_levels` / `find_fusible_pairs` are used for **graph
   analysis, visualization, and the fusion decision** (§below), not to run
   independent operators concurrently — concurrent multi-node scheduling on the
   executor is a named follow-on (see §5). It applies the **fusions** as it
   lowers:
   - **conv+BN fold** (`batchnorm_affine::bn_fold`): a `conv→batchnorm` edge folds
     BN's scale/shift into the conv's weight columns + bias — one GEMM, no
     standalone BN pass (the E6-T4 / E9 fold).
   - **conv+bias+ReLU fused epilogue** (`MatMulComputeSpec::{bias, activation}`):
     a `conv→relu` (or the folded conv's bias) applies in-compute, so the
     pre-activation never round-trips to DRAM (E10).
   - **im2col layout** (`conv2d_im2col`): conv lowers to a GEMM over the unfolded
     `A_col` (the standard conv→matmul transform).
   - **residual add**: the `ElementwiseScheduleGenerator` ADD joins the block
     output with the (possibly 1×1-projected) skip tensor. The skip is a
     **data-availability** property, not concurrent execution: the skip tensor is
     produced by an upstream node and kept resident/in-DRAM until the ADD consumes
     it, so both the main-branch output and the skip are present when the ADD runs
     (both branches are earlier in the topo order). `get_execution_levels` reports
     that the two branches are *structurally* independent — the design records
     that for the graph viz and as the hook for the concurrent-scheduling
     follow-on, but M2 does not run them in parallel.

3. **Whole-network host oracle** — composes the per-op references
   (`conv2d_reference`, `batchnorm_reference`, `pool2d_reference`,
   elementwise add/ReLU, matmul) into a plain-loop forward pass over the same
   topology and weights, exactly as `verify_residual.cpp` composes its block
   oracle. The CSP output is compared elementwise (per-layer and final) at 1e-4.

## 4. What "compiler transformations" M2 tests — honestly scoped

The DFX compiler front-end (`tools/compiler/kpu-kernel-compiler`,
`DFGParser`/`DFXGenerator`) is **matmul-only today** (conv2d is an explicit
`TODO`); its tiling / dataflow-strategy / prefetch passes do not yet lower conv/
pool/BN. So M2 does **not** claim to test the full DFX compiler on a CNN. What M2
**does** exercise, end-to-end on the value path, are the transformations that are
real today:

- **operator fusions**: conv+BN fold, conv+bias+ReLU fused epilogue (both
  validated in isolation this program; M2 tests them *in composition*);
- **layout transform**: conv→im2col→GEMM;
- **graph transformations**: `KernelGraph`'s topological schedule, execution-level
  extraction (residual-branch structural independence — used for viz/analysis, not
  concurrent operator execution), and `find_fusible_pairs` (which the
  bridge consults to decide where to apply the fold/epilogue).

Extending the DFX tiling/dataflow-strategy compiler to conv/pool/BN (so the
`.dfg`→DFX→executor path compiles a whole CNN) is a **named follow-on** (the
conv2d `DFGParser` TODO + E4 layout + E5 gemm-family), tracked separately — it is
not on M2's demonstrate/validate/benchmark critical path.

## 5. Staging (each stage is a reviewable PR)

| Stage | Content | DoD tier |
|---|---|---|
| **T1 (this)** | Design + the graph→CSP bridge decision, honest compiler scope | design |
| **T2 bridge + BasicBlock** | The graph→CSP executor; a ResNet **BasicBlock** DFG (conv→BN→ReLU→conv→BN→+skip→ReLU, identity and 1×1-projection variants) run end-to-end on the CSP executor, validated vs the composed oracle | demonstrate + validate (block) |
| **T3 ResNet-18** | Full topology: stem, 4 stages of BasicBlocks, GAP, FC; whole-network oracle; `to_dot` graph | demonstrate + validate (network) |
| **T4 benchmark + writeup** | Cycles/utilization/stall table, envelope sweeps, Chrome trace, `docs/milestones/M2_resnet.md`; CI test | benchmark |

BasicBlock-first is deliberate: it is the recognizable, self-contained unit that
exercises *every* M2 concern (conv, BN fold, ReLU epilogue, residual add, and the
skip-branch DAG structure). ResNet-18 is stacking blocks + stem + GAP + FC.

## 6. Weights & scope

- **Weights:** deterministic synthetic weights (fixed-seed LCG), as M1 — the
  oracle is computed from the same weights, so validation is exact-to-fp32.
  Loading *trained* ResNet weights (ONNX/PyTorch) is the E15 runtime path,
  deferred (M1 made the same call).
- **In:** ResNet-18 inference (BasicBlock), executed **sequentially in
  topological order** on the shared CSP executor. **Follow-on:** ResNet-50
  (bottleneck block, 1×1→3×3→1×1), trained-weight loading, the full DFX conv
  compiler, and **concurrent multi-node scheduling** — running the
  execution-level-independent branches (e.g. the residual skip vs the main branch)
  concurrently on the executor rather than one node at a time. The latter is what
  would turn `get_execution_levels` from an analysis/viz aid into measured
  operator-branch parallelism; it is explicitly out of M2's scope so the demo's
  benchmark numbers describe sequential per-op execution.

## 7. Risks

- **Activation threading between nodes.** The bridge must hand a node's output
  activation to its consumer without a correctness gap (layout, tiling). Mitigated
  by reusing the exact seed/read pattern the per-op functional tests already use,
  and by per-layer oracle checks that localize any mismatch to one node.
- **fp accumulation over depth.** ResNet-18 is ~18 conv layers deep; fp32 error
  compounds. Mitigated by M1's bounded synthetic-weight scale (keeps activations
  O(1)) and a per-layer tolerance, not just end-to-end.
- **Envelope for large feature maps.** Early ResNet layers (112×112×64) are large;
  the conv im2col working set is bounded per the E6 envelope, but the demo uses
  modest input sizes (e.g. 32×32 CIFAR-style) for the block/network validation,
  with the envelope sweep documenting where generation refuses.

## 8. Deliverables & DoD mapping

- **Demonstrate:** `examples/milestones/m2_resnet.cpp` runs the BasicBlock and
  ResNet-18 DFG through the CSP executor; `--trace-dir` exports Chrome traces;
  `--dot` emits the KernelGraph.
- **Validate:** per-layer + end-to-end elementwise vs the composed host oracle at
  1e-4, default + constrained envelopes; a CI test (`ctest -R m2_resnet`).
- **Benchmark:** cycles / cyc-per-inference / DMA-BM-STR stall breakdown /
  utilization table across block variants, ResNet-18, and an envelope sweep;
  writeup `docs/milestones/M2_resnet.md`.

On approval, T2 begins with the graph→CSP bridge + the BasicBlock end-to-end.
