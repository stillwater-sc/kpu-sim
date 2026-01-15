# Proposed Documentation Reorganization

## Current State Analysis

The `docs/` directory currently has **~100 files** with minimal organization:
- ~70 files in the root directory
- 14 subdirectories with varying levels of organization
- Mix of architecture specs, design docs, implementation guides, analysis, session logs, and project management

### Current Subdirectories
```
docs/
├── analysis/          # 6 files - performance analysis
├── compiler/          # 2 files - compiler docs
├── design/            # 28 files - mixed design docs
├── development-notes/ # 6 files - dev notes
├── external-knowledge/# 5 files - external references
├── img/               # images
├── invariants/        # 3 files - validation invariants
├── memory/            # 2 files - memory subsystem
├── noc/               # 2 files - NoC docs
├── plans/             # 8 files - planning docs
├── pm/                # 6 files - project management
├── sessions/          # 16 files - session logs (well organized by date)
├── status/            # 1 file - status updates
└── (root)/            # ~70 files - everything else
```

---

## Proposed Structure

Organized by **software development category** with temporal context preserved via session logs.

```
docs/
│
├── README.md                           # Documentation index and navigation guide
│
├── 01-architecture/                    # High-level architecture and specifications
│   ├── kpu-specification.md            # (was: STILLWATER_KPU_SPECIFICATION.md)
│   ├── kpu-execution-model.md          # Credit-based dataflow model (MUST READ)
│   ├── kpu-architecture.md             # Overall architecture
│   ├── soc-architecture.md             # SoC-level architecture
│   ├── product-architecture.md         # Product positioning
│   └── spatial-computing.md            # Spatial computing concepts
│
├── 02-simulation/                      # Simulation framework docs
│   ├── fidelity-framework.md           # (was: SIMULATION_FIDELITY_FRAMEWORK.md)
│   ├── multi-fidelity-calibration.md   # (was: MULTI_FIDELITY_CALIBRATION_WORKFLOW.md)
│   ├── functional-simulation-gaps.md   # (was: FUNCTIONAL_SIMULATION_GAP_ANALYSIS.md)
│   ├── behavioral-execution-model.md   # (was: plans/BEHAVIORAL_EXECUTION_MODEL.md)
│   ├── tracing-system.md               # Trace generation
│   └── configuration-architecture.md   # Configuration system
│
├── 03-memory-subsystem/                # Memory hierarchy and controllers
│   ├── overview.md                     # Memory subsystem overview
│   ├── memory-interleaving.md          # (was: MEMORY_INTERLEAVING_DESIGN.md)
│   ├── unified-address-space.md        # Address space design
│   ├── data-orchestration.md           # Data movement orchestration
│   │
│   ├── controllers/                    # Memory controller designs
│   │   ├── lpddr5-state-model.md       # (was: LPDDR5_STATE_MODEL.md)
│   │   ├── lpddr5-pipeline.md          # (was: LPDDR5X_memory_pipeline.md)
│   │   └── refresh-control.md          # (was: memory/REFRESH_CONTROL_API_DESIGN.md)
│   │
│   ├── l3-l2-l1/                       # Buffer hierarchy
│   │   ├── tile-caching.md             # (was: design/tile_caching_architecture.md)
│   │   ├── l2-tile-scheduler.md        # L2 scheduling
│   │   ├── streamer.md                 # Streamer design
│   │   └── memory-orchestrator-vs-buffet.md
│   │
│   └── invariants/                     # Memory timing invariants
│       ├── lpddr5-invariants.md        # (was: invariants/lpddr5/)
│       └── lpddr5-rw-protocol.md
│
├── 04-compute-fabric/                  # Systolic array and compute
│   ├── systolic-array.md               # Systolic array design
│   ├── tiled-matmul.md                 # (was: tiled-matmul-implementation.md)
│   ├── matrix-tiling-strategies.md     # Tiling strategies
│   ├── weight-stationary-vs-output-stationary.md
│   ├── schedule-characterization.md    # Schedule analysis
│   └── tensorcores-discussion.md       # Comparison with tensor cores
│
├── 05-data-movement/                   # DMA, BlockMover, NoC
│   ├── dma/
│   │   ├── dma-engine-architecture.md  # (was: design/dma-engine-architecture.md)
│   │   ├── address-based-api.md        # (was: address-based-api-implementation-summary.md)
│   │   └── dma-quickstart.md           # (was: address-based-dma-quickstart.md)
│   │
│   ├── noc/
│   │   ├── router-architecture.md      # (was: NOC_ROUTER_ARCHITECTURE.md)
│   │   ├── wormhole-design.md          # (was: NOC_ROUTER_WORMHOLE_DESIGN.md)
│   │   ├── corrected-design.md         # (was: NOC_ROUTER_CORRECTED_DESIGN.md)
│   │   └── tracing.md                  # (was: noc/noc_tracing.md)
│   │
│   └── pcie/
│       ├── pcie-arbiter.md             # (was: pcie-arbiter-implementation-summary.md)
│       └── bandwidth-analysis.md       # (was: pcie-bandwidth-correction.md)
│
├── 06-compiler/                        # Compiler and code generation
│   ├── overview.md                     # Compiler architecture
│   ├── dfx-specification.md            # (was: design/dfx-domain-flow-execution-spec.md)
│   ├── dfg-toolchain.md                # DFG tools
│   ├── kernel-compiler.md              # Kernel compilation
│   ├── graph-execution.md              # Graph execution
│   └── implementation-plan.md          # (was: compiler/implementation_plan.md)
│
├── 07-runtime/                         # Runtime and execution
│   ├── runtime-api.md                  # Runtime API overview
│   ├── resource-management.md          # (was: resource-management-api-assessment.md)
│   ├── python-bindings.md              # (was: how-to-build-and-use-python-bindings.md)
│   └── simulation-guide.md             # (was: how-to-configure-and-run-kpu-simulations.md)
│
├── 08-type-system/                     # Type system and numeric formats
│   ├── tensor-type-system.md           # (was: tensor_type_system.md)
│   ├── block-format-type-system.md     # (was: block_format_type_system.md)
│   └── block-format-hardware.md        # (was: block_format_hardware_architectures.md)
│
├── 09-virtual-platform/                # Virtual platform evolution
│   ├── virtual-platform-analysis.md    # (was: virtual_platform_analysis.md)
│   ├── api-gaps-roadmap.md             # (was: KPU_API_GAPS_AND_ROADMAP.md)
│   └── domain-flow-integration.md      # Domain flow integration
│
├── analysis/                           # Performance analysis and benchmarking
│   ├── memory-bandwidth-dynamics.md
│   ├── memory-characterization.md
│   ├── noc-benchmarking-framework.md
│   ├── efficiency-bug-analysis.md
│   └── perception-analysis.md
│
├── design/                             # Component design documents (working docs)
│   ├── ofg-visualization.md            # OFG visualization
│   ├── systolic-array-animation.md     # Animation design
│   ├── array-design.md                 # Array design notes
│   └── ...                             # Other working design docs
│
├── plans/                              # Planning and roadmap documents
│   ├── roadmap-phase7-onwards.md       # Future roadmap
│   ├── distributed-program-execution.md
│   ├── dma-patterns-plan.md
│   ├── cycle-accurate-dma-noc.md
│   └── operand-flow-graph-analysis.md
│
├── sessions/                           # Session logs (keep as-is, well organized)
│   ├── 2025-11-23_schedule_generator_pipelining.md
│   ├── 2025-11-25_strategy_aware_scheduling.md
│   ├── ...
│   └── 2026-01-14_ofg_visualization_dataflow_fixes.md
│
├── reference/                          # External references and comparisons
│   ├── gpu-specs/
│   │   ├── nvidia-ai-accelerators.md
│   │   ├── rtx-30-series.md
│   │   ├── rtx-40-series.md
│   │   ├── rtx-50-series.md
│   │   └── amd-gpu-products.md
│   │
│   ├── dma-architecture-comparison.md
│   ├── silicon-ip-discussion.md
│   └── hw-sw-codesign-opportunities.md
│
├── project/                            # Project management
│   ├── project-plan.md
│   ├── milestones/
│   │   └── milestone-1.md
│   ├── reports/
│   │   ├── monthly-progress.md
│   │   └── ceo-flash-report.md
│   └── partners/
│       └── software-partners.md
│
└── archive/                            # Deprecated/superseded documents
    ├── development-notes/              # Old dev notes
    ├── status/                         # Old status updates
    └── superseded/                     # Documents replaced by newer versions
```

---

## Category Descriptions

### 01-architecture/
**Purpose**: Stable, high-level architecture documents that define what the KPU is.
**Audience**: New team members, partners, anyone needing the big picture.
**Update frequency**: Infrequent (major architecture changes only).

### 02-simulation/
**Purpose**: Simulation framework documentation - fidelity levels, calibration, configuration.
**Audience**: Developers working on or with the simulator.
**Update frequency**: As simulation capabilities evolve.

### 03-memory-subsystem/
**Purpose**: Everything about the memory hierarchy - controllers, buffers, invariants.
**Audience**: Memory subsystem developers, performance analysts.
**Update frequency**: Active development area.

### 04-compute-fabric/
**Purpose**: Systolic array, tiling, dataflow strategies.
**Audience**: Compute fabric developers, compiler developers.
**Update frequency**: Moderate.

### 05-data-movement/
**Purpose**: DMA, BlockMover, NoC, PCIe - all data transport.
**Audience**: Data movement developers, system integrators.
**Update frequency**: Active development area.

### 06-compiler/
**Purpose**: Compiler pipeline, DFX IR, code generation.
**Audience**: Compiler developers.
**Update frequency**: Active development area.

### 07-runtime/
**Purpose**: Runtime APIs, Python bindings, user guides.
**Audience**: Users of the simulator, application developers.
**Update frequency**: As APIs stabilize.

### 08-type-system/
**Purpose**: Numeric type system, block formats, hardware type mapping.
**Audience**: Developers working on numeric representation.
**Update frequency**: As new formats are added.

### 09-virtual-platform/
**Purpose**: Evolution toward full virtual platform capability.
**Audience**: Architects, strategic planning.
**Update frequency**: Strategic documents, updated with major milestones.

### analysis/
**Purpose**: Performance analysis, benchmarking results, bug analysis.
**Audience**: Performance engineers, debugging.
**Update frequency**: Ongoing as analysis is performed.

### design/
**Purpose**: Working design documents, may be incomplete or exploratory.
**Audience**: Active developers.
**Update frequency**: High (working documents).

### plans/
**Purpose**: Implementation plans, not yet executed.
**Audience**: Team planning.
**Update frequency**: Created before work, archived after completion.

### sessions/
**Purpose**: Session logs documenting what was done and when.
**Audience**: Historical reference, onboarding.
**Update frequency**: After each significant work session.

### reference/
**Purpose**: External references, competitive analysis, market data.
**Audience**: Anyone needing context.
**Update frequency**: As market evolves.

### project/
**Purpose**: Project management documents.
**Audience**: Management, stakeholders.
**Update frequency**: Regular reporting cadence.

### archive/
**Purpose**: Deprecated documents kept for historical reference.
**Audience**: Historical research.
**Update frequency**: Documents moved here when superseded.

---

## Migration Strategy

### Phase 1: Create Structure
```bash
mkdir -p docs/{01-architecture,02-simulation,03-memory-subsystem/controllers,03-memory-subsystem/l3-l2-l1,03-memory-subsystem/invariants,04-compute-fabric,05-data-movement/dma,05-data-movement/noc,05-data-movement/pcie,06-compiler,07-runtime,08-type-system,09-virtual-platform,reference/gpu-specs,project/milestones,project/reports,project/partners,archive/development-notes,archive/status,archive/superseded}
```

### Phase 2: Move High-Priority Documents
Start with the most important/frequently referenced documents:
1. Architecture docs → `01-architecture/`
2. Simulation framework → `02-simulation/`
3. Execution model → `01-architecture/`
4. Type system docs → `08-type-system/`

### Phase 3: Consolidate and Deduplicate
- Review documents with similar names for consolidation
- Move outdated versions to `archive/`
- Update cross-references

### Phase 4: Create README.md Index
Create a navigational index with:
- Quick links to key documents
- Document purpose descriptions
- "Start here" guidance for new users

---

## Documents to Archive/Consolidate

### Candidates for Archive
- `uml-9-23-2025.md` - Dated UML diagram (superseded?)
- `development-notes/*` - Old dev notes
- `status/*` - Status updates (historical)
- Dated documents without clear current relevance

### Candidates for Consolidation
- `NOC_ROUTER_*.md` - Three NoC router documents, consider merging
- Multiple tiling documents - `matrix-tiling-strategies.md`, `tiled-matmul-implementation.md`, `tiling_discussion.md`
- Multiple DFX documents - `dfx-*.md` files in `design/`

### Candidates for Deletion
- Backup/duplicate files (e.g., `CMakeLists_v1.txt`, `CMakeLists_v2.txt`)
- Log files (`analysis/matmul.log`)
- Non-documentation files in docs (`.xlsx`, `.pdf`, `.html`)

---

## Non-Markdown Files

Consider moving these to appropriate locations:
- `Cadence-AI-IP-Platform.pdf` → External reference or delete
- `NVIDIA-Jetson-Thor-Module.pdf` → External reference or delete
- `all_use_cases_feasibility_data_2ndset.xlsx` → Project management or delete
- `the-power-challenge-of-autonomy-infographic.html` → External reference or delete

---

## Approval Requested

Please review this proposed reorganization and provide feedback on:
1. Category structure - Does this make sense?
2. Document placement - Any files in wrong categories?
3. Archive decisions - Keep or archive specific documents?
4. Consolidation targets - Which documents should be merged?
5. Migration priority - What order to execute migration?
