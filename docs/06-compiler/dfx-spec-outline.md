# Domain Flow Execution (DFX) — Executive Briefing Outline

---

## 1. Executive Summary
- DFX = Virtual ISA / IR for Stillwater KPUs  
- Analogous to NVIDIA PTX, but **flow‑centric**  
- Encodes **knowledge flows, buffer credits, energy metrics**  
- Designed for **sustainability + hybrid CPU/KPU/optical integration**

---

## 2. Vision & Goals
- **Portability** across KPU generations  
- **Expressiveness** for domain‑flow semantics  
- **Performance Transparency** via energy annotations  
- **Extensibility** into hybrid and optical fabrics  

---

## 3. Instruction Set Overview
- **Domains**: BLAS/Tensor, Constraint Solvers, Spectral, DSP, MPC  
- **Domain‑structured operands**: tiles, tensors, spectral bases, horizons  
- **Buffer credits**: enforce flow control  
- **Automatic fusion**: reduce operand movement, latency, energy  

---

## 4. Memory & Flow Semantics
- **Result‑stationary scheduling**: results stay in place, operands stream  
- **Buffer credits**: allocate, consume, release, sync  
- **Fusion pipelines**: collapse adjacent operators into single flows  

---

## 5. Concurrency & Synchronization
- **Flow‑centric concurrency** (not threads)  
- Dependencies: operand, domain, credit  
- Synchronization primitives: `sync.domain`, `sync.credit`, `sync.global`  
- Buffer occupancy models prevent stalls  

---

## 6. Energy‑Delay & Performance Modeling
- **Annotations**: EDP, power, latency, throughput  
- **Concurrency hints**: occupancy, fusion guidance  
- **Sustainability metrics**: carbon intensity, thermal budget, efficiency index  
- **Performance primitives**: profile, annotate, optimize, balance  

---

## 7. Instruction Encoding & Syntax
- Canonical format: `<mnemonic> <operands> [ , <annotations> ]`  
- Domain‑prefixed mnemonics (e.g., `dfx.matmul`, `dfx.fft`)  
- Structured operands with tiling/distribution metadata  
- Annotations: credits, concurrency, energy, fusion  

---

## 8. Toolchain Integration
- **Compiler front‑ends**: lower SURE → DFX IR  
- **Optimizers**: allocate credits, schedule domains, fuse ops  
- **Runtime**: operand streaming, buffer management, dynamic adaptation  
- **Profilers**: latency, throughput, energy, sustainability metrics  
- **Debugging**: flow tracing, checkpoints, fusion verification  

---

## 9. End‑to‑End Example (Spectral MPC Workflow)
- **SURE Source** → FFT, filter, IFFT, MPC predict/optimize/update  
- **DFX IR** → primitives with credits + annotations  
- **Optimizer** → fusion + credit balancing  
- **Runtime** → operand streaming + synchronization  
- **Profiler** → latency, throughput, EDP, efficiency index  

---

## 10. Future Extensions
- **Hybrid CPU/KPU systems**: unified memory, hybrid sync (`sync.hybrid`)  
- **Optical matmul engines**: picosecond latency, milliwatt power  
- **Optical spectral engines**: FFT + filtering in photonic domain  
- **Unified hybrid flow semantics**: credits + sync across CPU/KPU/optical  

---

## 11. Comparative Positioning
- **PTX**: thread‑centric, hides energy  
- **LLVM IR**: scalar, hardware‑agnostic  
- **DSL IRs**: siloed, domain‑specific  
- **DFX**: flow‑centric, energy‑aware, hybrid‑ready, multi‑domain  

---

## 12. Conclusion & Roadmap
- **Conclusion**: DFX = flow‑aware execution ecosystem, unifying compilers, runtimes, profilers, accelerators  
- **Roadmap**:  
  - Phase 1: Reference implementation + sample workloads  
  - Phase 2: Toolchain integration + profiling tools  
  - Phase 3: Hybrid CPU/KPU systems  
  - Phase 4: Optical matmul/spectral engines  
  - Phase 5: Future domains (quantum, neuromorphic)  
- **Strategic Vision**: Position KPUs as part of a **sustainable, heterogeneous compute ecosystem**

