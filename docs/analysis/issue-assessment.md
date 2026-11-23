# Open issues: what is on the critical path to the simplest program

Assessed 2026-11-23, against main at `4d4af97`: **122 open issues**.

**The question.** What blocks kpu-sim from executing the simplest KPU program, a matmul on one
systolic tile, with L-B, L-T1 and L-CA working correctly and consistent with each other?

## Bottom line

- **Values: already consistent.** A corpus matmul (`tests/program/corpus/matmul_48x48x48_t16.csp`)
  runs at L-B, L-T1 and L-CA on `kpu_s1` and `kpu_t4`, and every operand of every level is within
  bound of L-B's (max abs 0). So does every other corpus program, the L2-staged matmul in #520,
  and an int8 → int32 64³ matmul (exact at every level, its L0 trace replayed bit-exact). #520 (`examples/program`, #503) makes this a CTest.
- **Timing: not yet consistent, by decision.** L-CA does not time the systolic array the
  deployment declares. It keeps the executor's own compute timing, and `kpu-run` says so:
  `unmodelled: fabric.pe_rows/pe_cols`. L-T1 times the array as a systolic stream. So L-T1's
  makespan is reported against L-CA's on every run, but not held to a tolerance until #401 lands
  (decided 2026-11-17, #458). The gap is large: 1366 vs 1638 cycles on the 48³ matmul, and 1512 vs
  5402 on a 64³ matmul, and 1596 vs 3949 on the int8 64³ matmul.
- **The critical path is seven issues, one piece of work:** the cycle-accurate systolic tile
  (#401), narrowed to an output-stationary int8 → int32 matmul on one tile. It covers:
  - the remaining L2 slot-credit steps (#387, steps 2 and 3);
  - L1 stream buffers and the OS north drain (#389);
  - the RTL's wavefront as the timing check (#385);
  - L2's bank concurrency, the array at 1–2 GHz fed from 250 MHz banks (#495);
  - int8 sized as 1 byte in the timed levels (#415);
  - L-CA reading the deployment's `pe_rows`/`pe_cols` (#458).
- **Not on the path: 114 of 122** (the seven, and #503 in review).

## Where 122 comes from

| Group | Count | What they are |
|---|---|---|
| Backlog (post-1.0), deferred | 52 | Mostly the July CSP-pattern epics, each split into five tasks T1–T5 (gather, layout, GEMM, …), the DNN milestones M4–M8 and the ingestion phases. A roadmap held as issues. |
| v0.10.0 | 22 | Mostly `rtl-sync`: mirroring the kpugen RTL's tests and records |
| v1.0.0 | 18 | Engine retirement, hygiene, and 9 issues filed today from the examples cleanup and the DRAM study |
| v1.1.0, v1.2.0 | 14 | Blocked linear-algebra operators (filed 2026-11) |
| v0.11.0 – v0.13.0 | 16 | L-T2, multi-tile and NoC, viewers, host and RISC-V orchestration |

Fewer than one in ten issues describes work the simplest program needs. Most of the rest records
the roadmap.

| Verdict | Count |
|---|---|
| **CP**: on the critical path | 7 |
| **Done**: in review today | 1 |
| **Simplify**: not on the path, but removes complexity | 5 |
| **RTL**: RTL parity, after the path | 12 |
| **Off**: not on the path | 48 |
| **Backlog**: deferred roadmap | 49 |

## Critical path, in order

| # | Issue | Milestone | Why |
|---|---|---|---|
| #401 | Cycle-accurate tile model: banked L2 and accumulator, L1 stream buffers, Streamer sequencing, systolic wavefront | v0.10.0 | The cycle-accurate systolic tile. L-CA times compute as fill + ceil(MACs/…), not the array the deployment declares. Narrowed to OS matmul it is: L1 stream buffers, Streamer sequencing, the wavefront, the OS north drain. |
| #387 | Mirror the RTL tile-credit and tile-ready-token tests (esg#16) | v0.10.0 | Slot credits for L2. Step 1 merged today (#496); steps 2 (L-T1) and 3 (L-CA) remain. The OS A/B pair sharing one slot is on the matmul path. |
| #389 | Tests: L1 stream-buffer credits, bubble-free Streamer and the OS north drain (match the kpugen RTL) | v0.10.0 | L1 stream buffers, a Streamer that never bubbles, and the OS north drain: the OS matmul's timing path through #401. |
| #385 | Tests: systolic stream timing matches the kpugen RTL wavefront | v0.10.0 | Defines "correct" for the tile's timing: the RTL wavefront (latency (R−1)+(C−1), last result n−1+latency after the first input). |
| #495 | L-CA: model L2's outstanding requests, and check a deployment's L2 can sustain the fabric's ingress bandwidth | v0.10.0 | L2 bank concurrency is how the design feeds the fabric: the array at 1–2 GHz, the L2 banks at 250 MHz, so 8–16 read requests in flight at ≥ 2 vectors per array clock. L-CA charges L2's latency once per transfer, so it can't show whether a deployment's L2 keeps up. Part of the first timing test (decided 2026-11-23). |
| #415 | Timed levels: size each element by its number system (int8 moves 1 byte, not 4) | v0.10.0 | The simplest program is int8 → int32 (decided 2026-11-23). Today both timed levels move int8 as 4 bytes: the int8 64³ matmul moves 98304 bytes, its fp32 size, so its traffic and timing are an fp32 program's. |
| #458 | L-CA's timing ignores the deployment's rates, fabric size and L2 banks | v0.10.0 | Open item: L-CA's fabric takes `pe_rows`/`pe_cols` (lands with #401). Until then L-T1 vs L-CA is reported, not held to a tolerance (decided 2026-11-17); on a 64×64 matmul they differ, 1512 vs 5402 cycles. |

**Done when:** the OS int8 → int32 matmul runs at L-B, L-T1 and L-CA with L-CA's fabric taken from the
deployment, values within bound of L-B (as today), L-CA's tile timing matching the RTL wavefront
(#385), L2 sustaining the fabric's ingress with the array at 1–2 GHz and the banks at 250 MHz
(#495), int8 moved as 1 byte (#415), and L-T1's makespan within a stated tolerance of L-CA's on
the corpus matmuls. This is #401's acceptance section, added 2026-11-23. Until then no issue
stated the tolerance: #458 deferred it to #401.

### Done (in review)

| # | Issue | Milestone | Why |
|---|---|---|---|
| #503 | examples: add program-stack examples (a .csp program at L-B/L-T1/L-CA, per-tile check against L-B, L3 placement) | v1.0.0 | PR #520 (`examples/program`): the simplest-program check. In review. |

## Simplify: not on the path, but the complexity you are paying for

| # | Issue | Milestone | Why |
|---|---|---|---|
| #22 | Sweep: migrate remaining test fixtures to tests/common/test_configs.hpp | v1.0.0 | Test-fixture and accessor hygiene. |
| #42 | Backfill symmetric monitoring accessors onto L2Layer/L3Layer | v1.0.0 | Test-fixture and accessor hygiene. |
| #266 | Freeze and retire the superseded execution engines (ADR 0001 D6) | v1.0.0 | Retire the overlapping engines, down to one per level. Not on the path, but the main source of the complexity in the tree. |
| #510 | compiler: L2TileScheduler models L2 and L3 as caches (LRU replacement, reloads, hit/miss rates) | v1.0.0 | Cache-model and NumPy-mock leftovers found today. |
| #512 | python: retire the legacy tools/python/stillwater_kpu package (NumPy mock) and the docs of the removed scratchpad API | v1.0.0 | Cache-model and NumPy-mock leftovers found today. |

## RTL parity: after the critical path

| # | Issue | Milestone | Why |
|---|---|---|---|
| #382 | Tests: BlockMover tile moves at any byte address, strided, partial rows (from the kpugen RTL) | v0.10.0 | BlockMover at any byte address, strided, partial rows. The simplest matmul moves aligned whole tiles. |
| #383 | Tests: L2/accumulator bank count and row-to-bank mapping (match the kpugen RTL) | v0.10.0 | Banked L2 to the RTL's geometry. A first banked L2 is in L-CA already (#401 slice 1); matching the RTL's mapping comes after the tile model. |
| #384 | Tests: Matmul and Conv2d end to end on dense layouts, output isolation, no stale accumulation (from the kpugen RTL) | v0.10.0 | Operator-level RTL parity (unaligned layouts, output isolation, conv2d). After the tile is timed. |
| #388 | Tests: banked L2 bandwidth and bank-conflict checks (match the kpugen RTL L2) | v0.10.0 | Banked L2 to the RTL's geometry. A first banked L2 is in L-CA already (#401 slice 1); matching the RTL's mapping comes after the tile model. |
| #390 | Resource-occupancy viewer: L2 banks, Streamer sequencer and L1 streams per clock, from kpu-sim and RTL traces | v0.10.0 | Shared records, viewers, the CSR interface, lowering to the RTL's command stream: RTL co-development, not simulator consistency. |
| #391 | Tile-flow view of RTL and kpu-sim runs: memory layers as containers, CSP focus cards, domain view (moved from esg#20) | v0.12.0 | Shared records, viewers, the CSR interface, lowering to the RTL's command stream: RTL co-development, not simulator consistency. |
| #392 | Structure record: one format for the hardware structure and resources of kpu-sim and the KPU RTL (block diagrams, viewer layout) | v0.10.0 | Shared records, viewers, the CSR interface, lowering to the RTL's command stream: RTL co-development, not simulator consistency. |
| #397 | One program on both sides: lower a CSP program to the RTL's tile command stream | v0.10.0 | Shared records, viewers, the CSR interface, lowering to the RTL's command stream: RTL co-development, not simulator consistency. |
| #404 | Model the RTL's CSR command interface (host CSRs, tile CMD_*, DOORBELL, STATUS) | backlog | Shared records, viewers, the CSR interface, lowering to the RTL's command stream: RTL co-development, not simulator consistency. |
| #405 | Shared record: NoC packets and hops, and resource occupancy, for the RTL's run recorder | v0.12.0 | Shared records, viewers, the CSR interface, lowering to the RTL's command stream: RTL co-development, not simulator consistency. |
| #406 | Mirror the RTL accumulator-load, bias, accumulate and CLEAR tests (esg#17) | v0.10.0 | Mirrors RTL features (accumulator LOAD/bias/CLEAR; nearest-neighbour weight loading). Not on the OS matmul path. |
| #408 | Mirror the RTL nearest-neighbour weight-loading and bubble-free weight-change tests (esg#55) | v0.10.0 | Mirrors RTL features (accumulator LOAD/bias/CLEAR; nearest-neighbour weight loading). Not on the OS matmul path. |

## Not on the path

| # | Issue | Milestone | Why |
|---|---|---|---|
| #43 | Wire L3Interconnect/NoC routing into the base simulation | v0.12.0 | Multi-tile and the NoC. The simplest program runs on one tile. |
| #44 | Add an L2 interconnect/topology to L2Layer (when needed) | v0.12.0 | Multi-tile and the NoC. The simplest program runs on one tile. |
| #45 | Epic: batched-MLP fused-operator validation suite with L3-tile sweep (latency/throughput/efficiency) | v1.0.0 | The batched-MLP sweep suite (deferred). |
| #48 | Tiling/reuse schedule configs for the 1..33 L3-tile sweep | v1.0.0 | The batched-MLP sweep suite (deferred). |
| #49 | Metrics: latency, throughput, compute efficiency for the MLP sweep | v1.0.0 | The batched-MLP sweep suite (deferred). |
| #50 | Sweep driver + validation harness for the batched-MLP suite | v1.0.0 | The batched-MLP sweep suite (deferred). |
| #144 | ProgramSerializer cannot round-trip register-file config and AUTO operand types | v1.0.0 | Program-format round trip and versioning, on the `.kpubin`/DMProgram path that #266 retires. |
| #231 | Phase 1: hardware-identical execution — ResourceManager setup + host memory | v0.13.0 | RISC-V orchestration and hardware-identical host execution. |
| #255 | Program format versioning, golden corpus, and a serialize/load/execute round-trip test | v1.0.0 | Program-format round trip and versioning, on the `.kpubin`/DMProgram path that #266 retires. |
| #257 | Host API must return simulator-computed values, or say it doesn't | v0.13.0 | The host API (`KPURuntime`) returning values: the runtime path, not the program path. |
| #268 | Statistical variance for transactional timing, after calibration | backlog | Timing variance, after calibration. |
| #281 | Derive explicit CSP processes and channels from the Domain Flow Program (ADR 0002 §5.1) | v0.11.0 | v0.11 architecture work: deriving CSP processes, the backdoor interface, the spatial event record. |
| #283 | L-T2 resource-transactional interpreter over the fixed §3.3 vocabulary | v0.11.0 | L-T2. Not needed to check L-B, L-T1 and L-CA against each other. |
| #284 | Backdoor as its own interface: global, unphysical, harness-only (ADR 0002 §3.4) | v0.11.0 | v0.11 architecture work: deriving CSP processes, the backdoor interface, the spatial event record. |
| #286 | Spatial event record and viewer: concurrency and causality across the pipeline | v0.11.0 | v0.11 architecture work: deriving CSP processes, the backdoor interface, the spatial event record. |
| #305 | Encapsulate the program, separate the data, orchestrate from RISC-V (.kpuld + Renode) | v0.13.0 | RISC-V orchestration and hardware-identical host execution. |
| #395 | Dataflow selection (output-, weight-, input-stationary) in the deployment spec and at every level | v0.10.0 | OS is the default and already flows from `fabric.dataflow`. WS/IS aren't needed for the simplest matmul. |
| #400 | conv2d (NHWC, implicit GEMM) at the CSP level, L-B and L-CA | v0.10.0 | conv2d at the CSP level. |
| #403 | Multi-tile execution: distribute and broadcast, more than one L3 tile at L-CA, the NoC modelled | v0.12.0 | Multi-tile and the NoC. The simplest program runs on one tile. |
| #416 | Number systems: more MAC pairs (bfloat16, fp16, fp8 → fp32; posit16) | backlog | More MAC number systems. |
| #417 | L3 banks in L-CA's timing, and the program's banks checked against the deployment | v0.10.0 | L3 banks are untimed by decision; the simplest matmul doesn't need them. |
| #423 | The generator's 2x2 KPU as a four-tile deployment (#396, the four-tile part) | v0.12.0 | Multi-tile and the NoC. The simplest program runs on one tile. |
| #425 | Tracking: a CSP data-movement simulator and visualizer -- what a program moves, where it reuses, where the array starves | v0.10.0 | Tracking issue for the data-movement view; its pieces exist (`--bubbles`, `.tflow`, the viewers). |
| #427 | Combiner: the KPU tile's result-path unit (merge partials; fused bias, activation, scaling, block-format conversion into L2) replaces the accumulator | v0.10.0 | The Combiner (WS/IS partial merges, fused epilogues). An OS matmul drains without it. |
| #433 | Re-accumulating a written tile in the fabric: the L0 replay, L-CA's start value and an L-CA livelock | v0.10.0 | Re-opening an accumulator on a tile already stored. The simplest matmul doesn't. |
| #447 | viewer: the floorplan view with the Overview view (local server) | v0.12.0 | Viewer features. |
| #448 | viewer: the Investigate view: connectivity lenses and drill-down into tiles | v0.12.0 | Viewer features. |
| #449 | viewer: publish runs from remote headless servers to a central kpuview service | v0.12.0 | Viewer features. |
| #479 | Epic: a blocked linear-algebra test set — GEMM schedule space and blocked decompositions | v1.2.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #480 | csp: rectangular shapes and tiles, and a range index for blocks | v1.1.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #481 | csp-gen: a GEMM schedule space — blocks per level, loop order, partial-C carriage | v1.1.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #482 | csp-gen: choose a GEMM schedule by shape — a reuse model and a search | v1.1.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #483 | GEMM with transposed operands: transpose on a move or a stream | v1.1.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #484 | TRSM as an operator: triangular solve with many right-hand sides | v1.1.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #485 | SYRK as an operator: the symmetric rank-k update | v1.1.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #486 | Blocked Cholesky, in-core and out-of-core | v1.1.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #487 | Blocked LU out-of-core: the matrix larger than L3 | v1.2.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #488 | GEMV and GER: bandwidth-bound operators | v1.2.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #489 | Batched and strided-batched GEMM | v1.2.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #490 | Blocked QR (compact WY) and TSQR | v1.2.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #491 | SYMM and TRMM: operands read from their stored half | v1.2.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #492 | Hessenberg reduction and Schur form: two-sided updates and host orchestration | v1.2.0 | Blocked linear-algebra operators and the GEMM schedule space (v1.1/v1.2). |
| #514 | docs: one page per surviving example (why, what, how), indexed from ONBOARDING | v1.0.0 | A page per example (filed today). |
| #515 | memory: per-generation LPDDR timing tables and geometry (LPDDR4/4x/5/5x/6/6x) from datasheets | v1.0.0 | The DRAM/MC/DMA study epic (filed today): its own track. The simplest matmul already runs on the hosted LPDDR5 model. |
| #516 | memory: pluggable memory-controller strategies (scheduler, page policy, queues, refresh) for A/B studies | v1.0.0 | The DRAM/MC/DMA study epic (filed today): its own track. The simplest matmul already runs on the hosted LPDDR5 model. |
| #517 | memory: DRAM + DMA validated across L-B, L-T1, L-T2 and L-CA on one workload | v1.0.0 | The DRAM/MC/DMA study epic (filed today): its own track. The simplest matmul already runs on the hosted LPDDR5 model. |
| #518 | memory: comparison harness across DRAM generations and controller strategies | v1.0.0 | The DRAM/MC/DMA study epic (filed today): its own track. The simplest matmul already runs on the hosted LPDDR5 model. |
| #519 | Epic: DRAM + memory controller + DMA as the fidelity-validation and memory-system study subsystem | v1.0.0 | The DRAM/MC/DMA study epic (filed today): its own track. The simplest matmul already runs on the hosted LPDDR5 model. |

## Backlog (post-1.0, deferred)

| # | Issue | Milestone | Why |
|---|---|---|---|
| #70 | Epic: gather/scatter CSP movement pattern (indirect addressing) | backlog | Deferred roadmap (post-1.0). |
| #73 | Epic: in-flight layout transform pattern (transpose/reshape/patchify) | backlog | Deferred roadmap (post-1.0). |
| #74 | Epic: GEMM family CSP movement pattern completion | backlog | Deferred roadmap (post-1.0). |
| #75 | Epic: Conv2D family CSP movement patterns | backlog | Deferred roadmap (post-1.0). |
| #78 | Epic: LayerNorm/RMSNorm CSP movement patterns | backlog | Deferred roadmap (post-1.0). |
| #79 | Epic: fused epilogue CSP patterns (matmul+bias+activation, conv+BN+relu) | backlog | Deferred roadmap (post-1.0). |
| #80 | Epic: fused FFN block CSP pattern (SwiGLU/GELU-MLP) | backlog | Deferred roadmap (post-1.0). |
| #81 | Epic: attention CSP patterns (chained GEMMs -> flash-style fusion) | backlog | Deferred roadmap (post-1.0). |
| #82 | Epic: KV-cache CSP movement patterns (append, paged gather) | backlog | Deferred roadmap (post-1.0). |
| #83 | Epic: RoPE & positional CSP patterns | backlog | Deferred roadmap (post-1.0). |
| #84 | Epic: embedding & logits CSP patterns | backlog | Deferred roadmap (post-1.0). |
| #85 | Epic: ViT/CNN spatial boundary CSP patterns (patchify, upsample, transposed conv) | backlog | Deferred roadmap (post-1.0). |
| #86 | Epic: JEPA masking CSP patterns (token gather/scatter) | backlog | Deferred roadmap (post-1.0). |
| #87 | Epic: model-block validation (ResNet/ViT/Llama/I-JEPA through CSP) | backlog | Deferred roadmap (post-1.0). |
| #88 | Epic: CSP data-movement pattern coverage for CV / LLM / JEPA models | backlog | Deferred roadmap (post-1.0). |
| #94 | gather-T1: pattern design & envelope analysis | backlog | Deferred roadmap (post-1.0). |
| #95 | gather-T2: ISA / executor capability closure | backlog | Deferred roadmap (post-1.0). |
| #96 | gather-T3: envelope-aware schedule generator | backlog | Deferred roadmap (post-1.0). |
| #97 | gather-T4: functional integration & oracle | backlog | Deferred roadmap (post-1.0). |
| #98 | gather-T5: regression & characterization | backlog | Deferred roadmap (post-1.0). |
| #110 | layout-T2: ISA / executor capability closure | backlog | Deferred roadmap (post-1.0). |
| #111 | layout-T3: envelope-aware schedule generator | backlog | Deferred roadmap (post-1.0). |
| #112 | layout-T4: functional integration & oracle | backlog | Deferred roadmap (post-1.0). |
| #113 | layout-T5: regression & characterization | backlog | Deferred roadmap (post-1.0). |
| #114 | gemm-T1: pattern design & envelope analysis | backlog | Deferred roadmap (post-1.0). |
| #115 | gemm-T2: ISA / executor capability closure | backlog | Deferred roadmap (post-1.0). |
| #116 | gemm-T3: envelope-aware schedule generator | backlog | Deferred roadmap (post-1.0). |
| #117 | gemm-T4: functional integration & oracle | backlog | Deferred roadmap (post-1.0). |
| #118 | gemm-T5: regression & characterization | backlog | Deferred roadmap (post-1.0). |
| #132 | DNN Milestone M4: attention head, chained -> flash-fused | backlog | Deferred roadmap (post-1.0). |
| #133 | DNN Milestone M5: Vision Transformer (DeiT-Tiny -> ViT-B/16 encoder) | backlog | Deferred roadmap (post-1.0). |
| #134 | DNN Milestone M6: YOLOv8n detection on the CSP executor | backlog | Deferred roadmap (post-1.0). |
| #135 | DNN Milestone M7: LLM decoder (GPT-2 small / TinyLlama) token-by-token decode | backlog | Deferred roadmap (post-1.0). |
| #136 | DNN Milestone M8: I-JEPA on the CSP executor | backlog | Deferred roadmap (post-1.0). |
| #229 | Epic: Real-DNN ingestion & compilation (ONNX/PyTorch → domain_flow → KPU) | backlog | Deferred roadmap (post-1.0). |
| #230 | Phase 0: KPU binary program spec + simulator reader/executor | backlog | Deferred roadmap (post-1.0). |
| #232 | Phase 2: relocate the KPU compiler backend to domain_flow | backlog | Deferred roadmap (post-1.0). |
| #233 | Phase 3: direct ONNX → DomainFlowGraph front-end | backlog | Deferred roadmap (post-1.0). |
| #234 | Phase 4: program cache + data-path personalization | backlog | Deferred roadmap (post-1.0). |
| #235 | Phase 5: weight values, numerical validation & model zoo | backlog | Deferred roadmap (post-1.0). |
| #242 | PLASMA capability P3: scalar special-functions (divide / reciprocal / sqrt) | backlog | Deferred roadmap (post-1.0). |
| #243 | PLASMA capability P5: pivoting (argmax-with-index + value-selected row swap) | backlog | Deferred roadmap (post-1.0). |
| #244 | PLASMA capability P6: multi-compute-tile residency + inter-tile reuse moves | backlog | Deferred roadmap (post-1.0). |
| #256 | Execute the bias, activation and VE ops carried in a program | backlog | Deferred roadmap (post-1.0). |
| #258 | BLAS L1 as named compositions: axpy, scal, dot, asum, nrm2 | backlog | Deferred roadmap (post-1.0). |
| #259 | Standalone TRSM plus the triangular-substitution capability (PLASMA P2) | backlog | Deferred roadmap (post-1.0). |
| #260 | POTRF / SYRK tile kernels (Cholesky) | backlog | Deferred roadmap (post-1.0). |
| #261 | float64 value path for factorization residuals | backlog | Deferred roadmap (post-1.0). |
| #262 | Host-driven LLM decode loop with state carried across tokens | backlog | Deferred roadmap (post-1.0). |

## Recommendations

1. **Do #401 as one narrow slice:** an OS int8 → int32 matmul on one tile, with acceptance
   stated up front (the wavefront from #385, L2's concurrency feeding the array from #495, the
   fabric from the deployment, L-T1 within a tolerance of L-CA). Leave WS/IS and the Combiner's
   merges, the RTL's exact bank mapping and BlockMover generality for later slices.
2. **Fold the roadmap out of the issue list.** The 52 deferred backlog issues, mostly
   five-task trees per CSP-pattern epic, could become one roadmap document plus one tracking issue
   per family. That leaves the issue list describing work that is actually scheduled. The same
   applies to the v1.1/v1.2 linear-algebra set.
3. **Keep #266 moving.** Retiring the superseded engines is the change that most reduces what a
   reader has to understand. Today's examples cleanup was mostly code written for engines that
   are being retired.
4. **Today's time.** The cleanup was slow mostly because of review rounds, not the simulator. One
   PR (#513) took five CodeRabbit rounds on a stderr-filter edge case. The simplest-program check
   itself (#520) took under an hour, because the program path already works.
