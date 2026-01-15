# KPU Simulator Documentation

This documentation is organized by software development categories for the KPU (Knowledge Processing Unit) multi-fidelity simulator.

## Quick Start

**New to the project?** Start with these essential documents:

1. [KPU Execution Model](01-architecture/kpu-execution-model.md) - **MUST READ** - Credit-based dataflow model
2. [Simulation Fidelity Framework](02-simulation/fidelity-framework.md) - Multi-fidelity simulation design
3. [KPU Specification](01-architecture/kpu-specification.md) - Overall KPU specification

## Documentation Structure

### [01-architecture/](01-architecture/)
High-level architecture and specifications that define what the KPU is.

| Document | Description |
|----------|-------------|
| [kpu-execution-model.md](01-architecture/kpu-execution-model.md) | Credit-based dataflow execution model |
| [kpu-specification.md](01-architecture/kpu-specification.md) | Complete KPU specification |
| [kpu-architecture.md](01-architecture/kpu-architecture.md) | Overall architecture overview |
| [soc-architecture.md](01-architecture/soc-architecture.md) | SoC-level architecture |
| [product-architecture.md](01-architecture/product-architecture.md) | Product positioning |
| [spatial-computing.md](01-architecture/spatial-computing.md) | Spatial computing concepts |

### [02-simulation/](02-simulation/)
Simulation framework documentation - fidelity levels, calibration, configuration.

| Document | Description |
|----------|-------------|
| [fidelity-framework.md](02-simulation/fidelity-framework.md) | Multi-fidelity simulation design |
| [multi-fidelity-calibration.md](02-simulation/multi-fidelity-calibration.md) | Calibration workflow |
| [functional-simulation-gaps.md](02-simulation/functional-simulation-gaps.md) | Gap analysis |
| [behavioral-execution-model.md](02-simulation/behavioral-execution-model.md) | Behavioral tier execution |
| [tracing-system.md](02-simulation/tracing-system.md) | Trace generation |
| [configuration-architecture.md](02-simulation/configuration-architecture.md) | Configuration system |

### [03-memory-subsystem/](03-memory-subsystem/)
Everything about the memory hierarchy - controllers, buffers, invariants.

| Document | Description |
|----------|-------------|
| [memory-interleaving.md](03-memory-subsystem/memory-interleaving.md) | Memory interleaving design |
| [unified-address-space.md](03-memory-subsystem/unified-address-space.md) | Address space design |
| [data-orchestration.md](03-memory-subsystem/data-orchestration.md) | Data movement orchestration |

**Subdirectories:**
- [controllers/](03-memory-subsystem/controllers/) - LPDDR5 state model, pipeline, refresh control
- [l3-l2-l1/](03-memory-subsystem/l3-l2-l1/) - Buffer hierarchy, tile caching, streamer
- [invariants/](03-memory-subsystem/invariants/) - Memory timing invariants

### [04-compute-fabric/](04-compute-fabric/)
Systolic array, tiling, and dataflow strategies.

| Document | Description |
|----------|-------------|
| [systolic-array.md](04-compute-fabric/systolic-array.md) | Systolic array design |
| [tiled-matmul.md](04-compute-fabric/tiled-matmul.md) | Tiled matrix multiplication |
| [matrix-tiling-strategies.md](04-compute-fabric/matrix-tiling-strategies.md) | Tiling strategies |
| [schedule-characterization.md](04-compute-fabric/schedule-characterization.md) | Schedule analysis |
| [tensorcores-discussion.md](04-compute-fabric/tensorcores-discussion.md) | Comparison with tensor cores |

### [05-data-movement/](05-data-movement/)
DMA, BlockMover, NoC, PCIe - all data transport.

**Subdirectories:**
- [dma/](05-data-movement/dma/) - DMA engine architecture, address-based API
- [noc/](05-data-movement/noc/) - NoC router architecture, wormhole routing
- [pcie/](05-data-movement/pcie/) - PCIe arbiter, bandwidth analysis

### [06-compiler/](06-compiler/)
Compiler pipeline, DFX IR, code generation.

| Document | Description |
|----------|-------------|
| [dfx-specification.md](06-compiler/dfx-specification.md) | Domain Flow Execution specification |
| [dfg-toolchain.md](06-compiler/dfg-toolchain.md) | DFG tools |
| [graph-execution.md](06-compiler/graph-execution.md) | Graph execution |
| [implementation_plan.md](06-compiler/implementation_plan.md) | Compiler implementation plan |

### [07-runtime/](07-runtime/)
Runtime APIs, Python bindings, user guides.

| Document | Description |
|----------|-------------|
| [python-bindings.md](07-runtime/python-bindings.md) | Building and using Python bindings |
| [simulation-guide.md](07-runtime/simulation-guide.md) | How to run KPU simulations |
| [resource-management.md](07-runtime/resource-management.md) | Resource management API |

### [08-type-system/](08-type-system/)
Numeric type system, block formats, hardware type mapping.

| Document | Description |
|----------|-------------|
| [tensor_type_system.md](08-type-system/tensor_type_system.md) | Template-based tensor type system |
| [block_format_type_system.md](08-type-system/block_format_type_system.md) | Block format types (ZFP, MX) |
| [block_format_hardware_architectures.md](08-type-system/block_format_hardware_architectures.md) | Decompression hardware architectures |

### [09-virtual-platform/](09-virtual-platform/)
Evolution toward full virtual platform capability.

| Document | Description |
|----------|-------------|
| [virtual_platform_analysis.md](09-virtual-platform/virtual_platform_analysis.md) | Virtual platform gap analysis |
| [api-gaps-roadmap.md](09-virtual-platform/api-gaps-roadmap.md) | API gaps and DNN roadmap |
| [domain-flow-integration.md](09-virtual-platform/domain-flow-integration.md) | Domain flow integration |

---

## Supporting Directories

### [analysis/](analysis/)
Performance analysis, benchmarking results, bug analysis.

### [design/](design/)
Working design documents - may be incomplete or exploratory.

### [plans/](plans/)
Implementation plans for upcoming work.

### [sessions/](sessions/)
Session logs documenting what was done and when. Well organized by date.

### [reference/](reference/)
External references, competitive analysis, GPU specifications.
- [gpu-specs/](reference/gpu-specs/) - NVIDIA, AMD GPU specifications

### [project/](project/)
Project management documents.
- [milestones/](project/milestones/) - Milestone tracking
- [reports/](project/reports/) - Progress reports
- [partners/](project/partners/) - Partner information

### [archive/](archive/)
Deprecated documents kept for historical reference.

---

## Key Concepts

### Multi-Fidelity Simulation

The KPU simulator supports three fidelity tiers:

| Tier | Purpose | Speed |
|------|---------|-------|
| **BEHAVIORAL** | Functional correctness, software bring-up | ~100-1000x |
| **TRANSACTIONAL** | Architecture exploration, bottleneck ID | ~10-100x |
| **CYCLE_ACCURATE** | Performance analysis, timing validation | 1x |

### Credit-Based Dataflow

The KPU is NOT a stored-program processor. It uses credit-based dataflow:

- **Credits flow upstream** (consumer → producer)
- **Data flows downstream** (producer → consumer)
- L3, L2, L1 are **buffers**, NOT caches
- No demand-fetching - only push with credit

See [KPU Execution Model](01-architecture/kpu-execution-model.md) for details.

---

## Navigation Tips

- **Looking for architecture?** Start with `01-architecture/`
- **Setting up simulation?** See `02-simulation/` and `07-runtime/`
- **Working on memory?** Check `03-memory-subsystem/`
- **Implementing compute?** See `04-compute-fabric/`
- **Data movement questions?** Look in `05-data-movement/`
- **Compiler work?** Start with `06-compiler/`
- **Historical context?** Browse `sessions/` for session logs
