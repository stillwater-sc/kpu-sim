# DFX Executive Briefing Deck Flow

---

### Slide 1 — Title
**Domain Flow Execution (DFX): A Virtual ISA for KPUs**  
- Subtitle: *Flow‑aware, energy‑efficient, hybrid‑ready*  
- Visual: Abstract diagram of operand flows streaming through a pipeline  

---

### Slide 2 — Executive Summary
- DFX = PTX‑equivalent for Stillwater KPUs  
- Encodes flows, credits, domains, energy metrics  
- Designed for sustainability + hybrid CPU/KPU/optical integration  
- Visual: Comparison chart (PTX vs DFX)  

---

### Slide 3 — Vision & Goals
- Portability across KPU generations  
- Expressiveness for domain‑flow semantics  
- Transparency via energy annotations  
- Extensibility into hybrid/optical fabrics  
- Visual: Roadmap arrow graphic  

---

### Slide 4 — Instruction Set Overview
- Domains: BLAS, Constraint, Spectral, DSP, MPC  
- Domain‑structured operands (tiles, tensors, horizons)  
- Buffer credits enforce flow control  
- Automatic operator fusion reduces movement  
- Visual: Domain icons arranged around a central “DFX” hub  

---

### Slide 5 — Memory & Flow Semantics
- Result‑stationary scheduling  
- Buffer credits: allocate, consume, release, sync  
- Fusion pipelines collapse adjacent operators  
- Visual: Flow diagram showing credits and stationary results  

---

### Slide 6 — Concurrency & Synchronization
- Flow‑centric concurrency (not threads)  
- Dependencies: operand, domain, credit  
- Synchronization primitives: domain, credit, global  
- Buffer occupancy models prevent stalls  
- Visual: Gantt chart of operand flows  

---

### Slide 7 — Energy‑Delay & Performance Modeling
- Annotations: EDP, power, latency, throughput  
- Concurrency hints: occupancy, fusion guidance  
- Sustainability metrics: carbon intensity, efficiency index  
- Visual: Dashboard mockup with latency vs energy tradeoff  

---

### Slide 8 — Instruction Encoding & Syntax
- Canonical format: `<mnemonic> <operands> [ , <annotations> ]`  
- Domain‑prefixed mnemonics (e.g., `dfx.matmul`)  
- Structured operands with metadata  
- Visual: Code snippet styled like PTX assembly  

---

### Slide 9 — Toolchain Integration
- Compiler front‑ends lower SURE → DFX IR  
- Optimizers allocate credits, fuse ops  
- Runtime streams operands, manages buffers  
- Profilers collect latency, throughput, energy metrics  
- Visual: Layered stack diagram (SURE → DFX → Runtime → KPU)  

---

### Slide 10 — End‑to‑End Example (Spectral MPC)
- SURE source → FFT, filter, IFFT, MPC predict/optimize/update  
- DFX IR → primitives with credits + annotations  
- Optimizer → fusion + credit balancing  
- Runtime → operand streaming + synchronization  
- Profiler → latency, throughput, EDP, efficiency index  
- Visual: Workflow pipeline diagram  

---

### Slide 11 — Future Extensions
- Hybrid CPU/KPU systems (Grace Hopper‑style)  
- Optical matmul engines (picosecond latency)  
- Optical spectral engines (FFT + filtering in photonic domain)  
- Unified hybrid flow semantics across fabrics  
- Visual: Hybrid architecture diagram (CPU + KPU + Optical)  

---

### Slide 12 — Comparative Positioning
- PTX: thread‑centric, hides energy  
- LLVM IR: scalar, hardware‑agnostic  
- DSL IRs: siloed, domain‑specific  
- DFX: flow‑centric, energy‑aware, hybrid‑ready, multi‑domain  
- Visual: 2x2 positioning matrix  

---

### Slide 13 — Conclusion & Roadmap
- DFX = flow‑aware execution ecosystem  
- Roadmap:  
  - Phase 1: Reference implementation  
  - Phase 2: Toolchain integration  
  - Phase 3: Hybrid CPU/KPU systems  
  - Phase 4: Optical accelerators  
  - Phase 5: Future domains (quantum, neuromorphic)  
- Visual: Timeline graphic  

---
