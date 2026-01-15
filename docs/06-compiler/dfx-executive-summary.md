# Domain Flow Execution (DFX) Specification  

**A Virtual ISA for Stillwater Knowledge Processing Units (KPUs)**  
**Draft Whitepaper — Version 1.0**

---

## Executive Summary
Domain Flow Execution (DFX) is the proposed **virtual instruction set architecture (ISA)** and **intermediate representation (IR)** for the Stillwater Knowledge Processing Unit (KPU). Analogous to NVIDIA’s PTX for CUDA, DFX abstracts the **Domain Flow Architecture** into a portable, extensible execution model.  

Unlike thread‑centric ISAs, DFX encodes **flows of knowledge operands**, **buffer credits**, and **domain‑structured primitives** as first‑class citizens. It integrates **energy‑delay modeling**, **automatic operator fusion**, and **hybrid CPU/KPU/optical integration**, positioning KPUs as sustainable, future‑proof accelerators for knowledge workloads.

---

## Section 1: Introduction
DFX provides a stable, forward‑compatible abstraction of the Domain Flow Architecture, enabling compilers to target KPUs without requiring direct knowledge of hardware microarchitectural details.  

Goals:  
- **Portability** across KPU generations.  
- **Expressiveness** for domain‑flow semantics.  
- **Performance Transparency** via energy‑aware annotations.  
- **Extensibility** for hybrid and optical accelerators.  

---

## Section 2: Instruction Set Overview
DFX defines primitives across computational domains:  
- **BLAS/Tensor Algebra** (`dfx.matmul`, `dfx.tensor.contract`)  
- **Constraint Solvers** (`dfx.constraint.solve`, `dfx.constraint.project`)  
- **Spectral Methods** (`dfx.fft`, `dfx.spectral.filter`)  
- **DSP** (`dfx.convolve`, `dfx.fir`, `dfx.iir`)  
- **Model Predictive Control (MPC)** (`dfx.mpc.predict`, `dfx.mpc.optimize`)  

Operands are **domain‑structured** (tiles, tensors, spectral bases, horizons) with metadata for tiling, distribution, credits, and fusion hints.  

Buffer credits enforce flow control, while **automatic operator fusion** reduces operand movement and energy cost.

---

## Section 3: Memory and Flow Semantics
- **Result‑Stationary Scheduling**: Results remain in place, operands stream through pipelines.  
- **Buffer Credits**: Allocate, consume, release, and synchronize buffer slots.  
- **Computational Domains**: Linear Algebra, Constraint, Spectral, Signal, Control.  
- **Fusion Semantics**: Adjacent operators fused into pipelines to minimize latency and energy.  

---

## Section 4: Concurrency and Synchronization
Concurrency is modeled via **flows, not threads**.  
- **Dependencies**: Operand, domain, or credit‑based.  
- **Synchronization Primitives**: `dfx.sync.domain`, `dfx.sync.credit`, `dfx.sync.global`.  
- **Buffer Occupancy**: Explicit credit tracking prevents stalls.  
- **Fusion + Concurrency**: Fused operators collapse dependencies, reducing synchronization overhead.  

---

## Section 5: Energy‑Delay and Performance Modeling
DFX embeds sustainability directly:  
- **Annotations**: `edp=low|medium|high`, `power=watts`, `latency=cycles`, `throughput=ops/sec`.  
- **Concurrency Hints**: `concurrency=high|low`, `occupancy=n`.  
- **Sustainability Metrics**: Carbon intensity, thermal budget, efficiency index.  
- **Performance Primitives**: `dfx.profile`, `dfx.annotate`, `dfx.optimize`, `dfx.balance`.  

---

## Section 6: Instruction Encoding and Syntax
Canonical format:
```
<mnemonic> <operands> [ , <annotations> ]
```

Examples:
```
dfx.matmul   A_tile, B_tile -> C_tile, credit=+8, edp=low, concurrency=high, fuse=on
dfx.fft      signal -> spectrum, credit=+2
dfx.filter   spectrum -> filtered, fuse=on, edp=medium
dfx.ifft     filtered -> output, credit=-2
```

---

## Section 7: Toolchain Integration
DFX integrates across the stack:  
- **Compiler Front‑Ends**: Lower SURE programs into DFX IR.  
- **Optimizers**: Allocate credits, schedule domains, fuse operators.  
- **Runtime Systems**: Stream operands, manage buffers, adapt dynamically.  
- **Profilers**: Collect latency, throughput, energy, sustainability metrics.  
- **Debugging**: Flow tracing, domain checkpoints, fusion verification.  

---

## Section 8: Example End‑to‑End Compilation Flow
**Spectral MPC workload**:  
- **SURE Source** → FFT, filter, IFFT, MPC predict/optimize/update.  
- **DFX IR** → Flow primitives with credits and annotations.  
- **Optimizer** → Fusion of spectral pipeline, credit balancing.  
- **Runtime** → Operand streaming, result‑stationary scheduling.  
- **Profiler** → Reports latency, throughput, EDP, efficiency index.  

---

## Section 9: Future Extensions
DFX evolves toward hybrid and optical fabrics:  
- **Hybrid CPU/KPU Systems**: Unified memory, hybrid synchronization (`dfx.sync.hybrid`).  
- **Optical Matmul Engines**: `dfx.optical.matmul` with picosecond latency and milliwatt power.  
- **Optical Spectral Engines**: `dfx.optical.fft`, `dfx.optical.filter`.  
- **Unified Hybrid Flow Semantics**: Credits and synchronization across CPU, KPU, and optical domains.  

---

## Section 10: Comparative Positioning
- **PTX**: Thread‑centric, hides energy.  
- **LLVM IR**: Scalar, hardware‑agnostic.  
- **DSL IRs**: Domain‑specific, siloed.  
- **DFX**: Flow‑centric, energy‑aware, hybrid‑ready, multi‑domain.  

Unique contributions: flow semantics, result‑stationary scheduling, energy modeling, fusion, hybrid extensibility.

---

## Section 11: Conclusion and Roadmap
**Conclusion**: DFX is a **flow‑aware execution ecosystem**, unifying compilers, runtimes, profilers, and heterogeneous accelerators. It encodes sustainability and hybrid integration directly into the IR.  

**Roadmap**:  
- Phase 1: Reference implementation + sample workloads.  
- Phase 2: Toolchain integration + profiling tools.  
- Phase 3: Hybrid CPU/KPU systems.  
- Phase 4: Optical matmul/spectral engines.  
- Phase 5: Future domains (quantum, neuromorphic).  

**Strategic Vision**: DFX positions Stillwater KPUs as part of a **sustainable, heterogeneous compute ecosystem**, bridging knowledge flows, energy efficiency, and hybrid integration.

---

✨ This whitepaper draft presents DFX as both a **technical specification** and a **strategic framework** for the future of domain‑flow computing. It mirrors PTX’s role for CUDA but extends it into sustainability, hybridization, and multi‑domain knowledge workloads.

