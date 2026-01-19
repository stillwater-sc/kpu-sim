# SIMT vs Domain Flow Architecture Comparison

This document provides comparison diagrams between GPU-style SIMT (Single Instruction Multiple Thread) architecture and the KPU's Domain Flow architecture.

---

## Executive Summary

| Aspect | SIMT (GPU) | Domain Flow (KPU) |
|--------|------------|-------------------|
| **Execution Model** | Control-driven | Data-driven |
| **Data Movement** | Fetch on demand | Push with credit |
| **Memory Hierarchy** | Cache-based | Buffer-based |
| **Energy Efficiency** | ~1 TOPS/W | ~10+ TOPS/W |
| **Workload Fit** | Irregular parallel | Regular tensor |

---

## Diagram 1: Execution Model Comparison

### SIMT: Control-Driven Execution

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SIMT (GPU)                                  │
│                    Control-Driven Execution                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌──────────────┐                                                 │
│    │  Instruction │◄───── Fetch instruction from I-cache            │
│    │    Fetch     │                                                 │
│    └──────┬───────┘                                                 │
│           │                                                         │
│           ▼                                                         │
│    ┌──────────────┐                                                 │
│    │   Decode &   │◄───── Decode opcode, identify operands          │
│    │   Schedule   │                                                 │
│    └──────┬───────┘                                                 │
│           │                                                         │
│           │ BROADCAST same instruction to all threads               │
│           ▼                                                         │
│    ┌──────┴──────┬──────┬──────┬──────┬──────┬──────┬──────┐       │
│    │ T0  │ T1  │ T2  │ T3  │ T4  │ T5  │ T6  │ T7  │  ...  │       │
│    │ ALU │ ALU │ ALU │ ALU │ ALU │ ALU │ ALU │ ALU │       │       │
│    └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴───────┘       │
│       │     │     │     │     │     │     │     │                   │
│       ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼                   │
│    ┌─────────────────────────────────────────────────────┐         │
│    │              Shared Memory / L1 Cache               │         │
│    │         (request-response, cache hit/miss)          │         │
│    └─────────────────────────────────────────────────────┘         │
│                                                                     │
│  BOTTLENECK: Instruction fetch/decode overhead per operation        │
│  PROBLEM: Branch divergence causes thread serialization             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Domain Flow: Data-Driven Execution

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Domain Flow (KPU)                             │
│                     Data-Driven Execution                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                         CREDITS (upstream)                          │
│                              ↑                                      │
│    ┌──────────────┐          │                                      │
│    │     DRAM     │──────────┼───────────────────┐                  │
│    │  (External)  │          │                   │                  │
│    └──────┬───────┘          │                   │ When credit      │
│           │                  │                   │ available,       │
│           │ PUSH tile ───────┘                   │ PUSH data        │
│           ▼                                      │ downstream       │
│    ┌──────────────┐                              │                  │
│    │  L3 Buffer   │◄─── Tag CAM matches tile ────┘                  │
│    │  (not cache) │                                                 │
│    └──────┬───────┘                                                 │
│           │                                                         │
│           │ PUSH tile                                               │
│           ▼                                                         │
│    ┌──────────────┐                                                 │
│    │  L2 Buffer   │◄─── Tile arrives, forward when L1 has credit    │
│    │  (not cache) │                                                 │
│    └──────┬───────┘                                                 │
│           │                                                         │
│           │ PUSH tile                                               │
│           ▼                                                         │
│    ┌──────────────┐                                                 │
│    │  L1 Streams  │◄─── Data ready, fire computation                │
│    └──────┬───────┘                                                 │
│           │                                                         │
│           ▼  DATA ARRIVES → COMPUTE FIRES                           │
│    ┌──────┴──────┬──────┬──────┬──────┬──────┬──────┬──────┐       │
│    │ PE  │ PE  │ PE  │ PE  │ PE  │ PE  │ PE  │ PE  │  ...  │       │
│    │ 0,0 │ 0,1 │ 0,2 │ 0,3 │ 1,0 │ 1,1 │ 1,2 │ 1,3 │       │       │
│    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴───────┘       │
│                      Systolic Array                                 │
│                                                                     │
│  NO instruction fetch per operation - program is implicit in data   │
│  NO branch divergence - deterministic dataflow                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 2: Memory Access Pattern Comparison

### SIMT: Request-Response (Fetch on Demand)

```
┌────────────────────────────────────────────────────────────────────┐
│                 SIMT Memory: Request-Response                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Thread              Cache                    DRAM                 │
│     │                   │                        │                  │
│     │───REQUEST────────►│                        │                  │
│     │   (load addr)     │                        │                  │
│     │                   │                        │                  │
│     │              ┌────┴────┐                   │                  │
│     │              │ HIT or  │                   │                  │
│     │              │ MISS?   │                   │                  │
│     │              └────┬────┘                   │                  │
│     │                   │                        │                  │
│     │              [if MISS]                     │                  │
│     │                   │───FETCH───────────────►│                  │
│     │                   │                        │                  │
│     │                   │◄──────DATA─────────────│                  │
│     │                   │                        │                  │
│     │◄──RESPONSE────────│                        │                  │
│     │   (data)          │                        │                  │
│     │                   │                        │                  │
│     ▼                   ▼                        ▼                  │
│                                                                     │
│   LATENCY: Variable (cache hit ~20 cycles, miss ~200+ cycles)       │
│   STALLS: Thread blocked waiting for data                           │
│   ENERGY: Repeated tag lookups, coherence traffic                   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Domain Flow: Credit-Based Push

```
┌────────────────────────────────────────────────────────────────────┐
│               Domain Flow Memory: Credit-Based Push                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Compute             L1 Buffer           L2 Buffer      DRAM      │
│     │                    │                    │            │        │
│     │                    │                    │            │        │
│     │──CREDIT───────────►│──CREDIT───────────►│──CREDIT───►│        │
│     │  (space available) │                    │            │        │
│     │                    │                    │            │        │
│     │                    │                    │◄───PUSH────│        │
│     │                    │                    │   (tile)   │        │
│     │                    │                    │            │        │
│     │                    │◄───────PUSH────────│            │        │
│     │                    │        (tile)      │            │        │
│     │                    │                    │            │        │
│     │◄───────PUSH────────│                    │            │        │
│     │       (tile)       │                    │            │        │
│     │                    │                    │            │        │
│     │  DATA ARRIVES      │                    │            │        │
│     │  ═══►FIRE◄═══      │                    │            │        │
│     ▼                    ▼                    ▼            ▼        │
│                                                                     │
│   LATENCY: Predictable (pipelined, hidden by prefetch)              │
│   STALLS: None if credits flow properly (double buffering)          │
│   ENERGY: No tag lookups, no coherence, simple push                 │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 3: Energy Breakdown Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Energy Per Operation                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SIMT (GPU)                        Domain Flow (KPU)                │
│  ───────────                       ─────────────────                │
│                                                                     │
│  ┌─────────────────┐               ┌─────────────────┐              │
│  │ Instruction     │               │                 │              │
│  │ Fetch/Decode    │ 35%           │   Eliminated    │ 0%           │
│  │ ████████████    │               │                 │              │
│  ├─────────────────┤               ├─────────────────┤              │
│  │ Register File   │               │ Register File   │              │
│  │ Access          │ 25%           │ Access          │ 15%          │
│  │ ████████        │               │ █████           │              │
│  ├─────────────────┤               ├─────────────────┤              │
│  │ Data Movement   │               │ Data Movement   │              │
│  │ (cache, NoC)    │ 30%           │ (buffers)       │ 25%          │
│  │ ██████████      │               │ ████████        │              │
│  ├─────────────────┤               ├─────────────────┤              │
│  │ Compute (MAC)   │               │ Compute (MAC)   │              │
│  │                 │ 10%           │                 │ 60%          │
│  │ ███             │               │ ████████████████│              │
│  └─────────────────┘               └─────────────────┘              │
│                                                                     │
│  Total: ~1 TOPS/W                  Total: ~10+ TOPS/W               │
│                                                                     │
│  ═══════════════════════════════════════════════════════════════    │
│  KEY INSIGHT: Domain Flow eliminates instruction overhead and       │
│  dedicates most energy to actual computation.                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 4: Side-by-Side Architecture

```
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│         SIMT (GPU)              │  │       Domain Flow (KPU)         │
├─────────────────────────────────┤  ├─────────────────────────────────┤
│                                 │  │                                 │
│  ┌─────────────────────────┐    │  │       ┌───────────────┐         │
│  │    Instruction Cache    │    │  │       │     DRAM      │         │
│  └───────────┬─────────────┘    │  │       └───────┬───────┘         │
│              │                  │  │               │                 │
│              ▼                  │  │          ┌────▼────┐            │
│  ┌─────────────────────────┐    │  │          │ CREDIT  │            │
│  │  Fetch → Decode → Issue │    │  │          └────┬────┘            │
│  └───────────┬─────────────┘    │  │               ▼                 │
│              │                  │  │  ┌─────────────────────────┐    │
│              │ Broadcast        │  │  │      L3 Buffers         │    │
│              ▼                  │  │  │   (tile staging)        │    │
│  ┌───┬───┬───┬───┬───┬───┐     │  │  └───────────┬─────────────┘    │
│  │ T │ T │ T │ T │ T │ T │     │  │              │                  │
│  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │     │  │         ┌────▼────┐             │
│  └─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┘     │  │         │  PUSH   │             │
│    │   │   │   │   │   │       │  │         └────┬────┘             │
│    ▼   ▼   ▼   ▼   ▼   ▼       │  │              ▼                  │
│  ┌─────────────────────────┐    │  │  ┌─────────────────────────┐    │
│  │   Shared Memory / L1    │    │  │  │      L2 Buffers         │    │
│  │   (cache semantics)     │    │  │  │   (block staging)       │    │
│  └───────────┬─────────────┘    │  │  └───────────┬─────────────┘    │
│              │                  │  │              │                  │
│              │ Request/Response │  │         ┌────▼────┐             │
│              ▼                  │  │         │  PUSH   │             │
│  ┌─────────────────────────┐    │  │         └────┬────┘             │
│  │        L2 Cache         │    │  │              ▼                  │
│  └───────────┬─────────────┘    │  │  ┌─────────────────────────┐    │
│              │                  │  │  │      L1 Streams         │    │
│              ▼                  │  │  │   (compute feed)        │    │
│  ┌─────────────────────────┐    │  │  └───────────┬─────────────┘    │
│  │         DRAM            │    │  │              │                  │
│  └─────────────────────────┘    │  │              ▼ Data Ready       │
│                                 │  │  ┌───┬───┬───┬───┬───┬───┐     │
│  Control flows DOWN ↓           │  │  │PE │PE │PE │PE │PE │PE │     │
│  Data flows UP ↑                │  │  │0,0│0,1│0,2│1,0│1,1│1,2│     │
│                                 │  │  └───┴───┴───┴───┴───┴───┘     │
│                                 │  │       Systolic Array           │
│                                 │  │                                 │
│                                 │  │  Credits flow UP ↑              │
│                                 │  │  Data flows DOWN ↓              │
│                                 │  │                                 │
└─────────────────────────────────┘  └─────────────────────────────────┘
```

---

## Diagram 5: Workload Efficiency Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│              Workload Efficiency: SIMT vs Domain Flow               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Efficiency                                                         │
│  100% ┤                                    ╭──────────────────      │
│       │                                 ╭──╯   Domain Flow          │
│   80% ┤                              ╭──╯      (DNN workloads)      │
│       │                           ╭──╯                              │
│   60% ┤                        ╭──╯                                 │
│       │                     ╭──╯                                    │
│   40% ┤  ╭─────────────────╯                                        │
│       │ ╱          SIMT                                             │
│   20% ┤╱           (general purpose)                                │
│       │                                                             │
│    0% ┼─────────┬─────────┬─────────┬─────────┬─────────────────    │
│       │ Sparse  │ Graph   │ Dense   │ Conv    │  Transformer        │
│       │ Ops     │ Ops     │ MatMul  │ Layers  │  Attention          │
│       │         │         │         │         │                     │
│       └─────────┴─────────┴─────────┴─────────┴─────────────────    │
│           ◄─── Irregular ───►   ◄─── Regular Tensor Ops ───►        │
│                                                                     │
│  SIMT excels at: Irregular parallelism, branch-heavy code           │
│  Domain Flow excels at: Regular tensor operations, DNNs             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Terminology Comparison

| SIMT Term | Domain Flow Term | Explanation |
|-----------|------------------|-------------|
| Thread | Processing Element (PE) | Unit of computation |
| Warp/Wavefront | Tile | Group of computations |
| Cache hit | Tile in buffer | Data available locally |
| Cache miss | Waiting for tile | Data not yet arrived |
| Eviction | Credit return | Space becomes available |
| Load instruction | N/A (implicit) | No explicit load needed |
| Branch divergence | N/A (deterministic) | All PEs follow same flow |
| Shared memory | L1 Stream buffer | Local data storage |
| Global memory | DRAM | External memory |

---

## When to Use Which

### SIMT (GPU) is Better For:
- Irregular data access patterns
- Workloads with significant branching
- General-purpose parallel computing
- Graphics rendering
- Scientific simulations with complex control flow

### Domain Flow (KPU) is Better For:
- Dense matrix operations (MatMul, Conv)
- Neural network inference
- Transformer attention
- Regular, predictable data access
- Energy-constrained edge deployment
- Batch processing with known shapes

---

## Visual Design Notes for Graphics Team

When creating publication-quality graphics from these diagrams:

1. **Color Scheme**:
   - SIMT/GPU: Use warm colors (orange/red) to suggest energy consumption
   - Domain Flow/KPU: Use cool colors (blue/green) to suggest efficiency

2. **Flow Direction**:
   - SIMT: Emphasize control flowing down, data requests going up
   - Domain Flow: Emphasize credits up, data push down (waterfall)

3. **Key Visual Elements**:
   - SIMT: Show instruction decoder as prominent, broadcasting to many threads
   - Domain Flow: Show data tiles flowing through pipeline stages

4. **Callouts**:
   - SIMT: "35% energy on instruction fetch/decode"
   - Domain Flow: "No instruction overhead - implicit program"

---

*Document created: 2026-01-19*
*For KPU Simulator v0.4.0+*
