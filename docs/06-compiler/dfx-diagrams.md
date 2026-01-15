# DFX Diagrams

### 1. Toolchain Integration Flow
```mermaid
flowchart LR
    A[SURE Program] --> B[Compiler Front-End]
    B --> C[DFX IR]
    C --> D[Optimizer & Scheduler]
    D --> E[Annotated DFX IR]
    E --> F[Runtime System]
    F --> G[KPU Execution]
    F --> H[Profiler & Visualization]
```
*Shows the full pipeline from source → IR → optimization → runtime → profiling.*

---

### 2. Computational Domains
```mermaid
graph TD
    DFX[DFX Core] --> LA[Linear Algebra Domain]
    DFX --> CS[Constraint Solver Domain]
    DFX --> SP[Spectral Domain]
    DFX --> DSP[Signal Processing Domain]
    DFX --> MPC[Model Predictive Control Domain]
```
*Highlights the five computational domains supported by DFX.*

---

### 3. Buffer Credit Flow Control
```mermaid
sequenceDiagram
    participant Producer
    participant Buffer
    participant Consumer

    Producer->>Buffer: Allocate credit (+n)
    Producer->>Buffer: Inject operand (consume credit)
    Consumer->>Buffer: Consume result
    Buffer->>Producer: Release credit (-n)
```
*Illustrates how credits are allocated, consumed, and released to enforce flow control.*

---

### 4. Hybrid CPU/KPU/Optical Integration
```mermaid
flowchart TB
    CPU[CPU Domain] -->|Hybrid Sync| KPU[KPU Domain]
    KPU -->|Operand Flow| Optical[Optical Accelerator]
    Optical -->|Spectral/Matmul Results| KPU
    KPU --> Output[Unified Results]
```
*Shows hybrid execution across CPU, KPU, and optical accelerators with synchronization.*

---

### 5. Roadmap Timeline
```mermaid
gantt
    title DFX Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1
    Reference Implementation :done, 2025-12-01, 2026-03-01
    section Phase 2
    Toolchain Integration :active, 2026-03-01, 2026-06-01
    section Phase 3
    Hybrid CPU/KPU Systems :2026-06-01, 2026-09-01
    section Phase 4
    Optical Accelerators :2026-09-01, 2026-12-01
    section Phase 5
    Future Domains :2027-01-01, 2027-06-01
```
*Visualizes the phased roadmap for DFX development.*

