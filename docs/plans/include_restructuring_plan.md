# Include Directory Restructuring Plan

## Overview

This plan restructures `include/sw/kpu/` to mirror the `src/models/` directory organization, providing consistency between source and header file locations.

## Current State

### Source Structure (Target Pattern)
```
src/models/
├── behavioral/
│   ├── compute/
│   ├── datamovement/
│   ├── memory/
│   └── noc/
├── transactional/
│   ├── compute/
│   ├── datamovement/
│   └── memory/
└── temporal/
    ├── compute/
    ├── datamovement/
    ├── memory/
    │   └── controllers/    # lpddr5_controller.cpp, ddr5_controller.cpp, etc.
    └── noc/
```

### Current Include Structure (Problem)
```
include/sw/kpu/
├── behavioral/             # Partially correct
│   ├── block_mover.hpp
│   ├── memory_model.hpp
│   ├── mlp_executor.hpp
│   ├── orchestrator.hpp
│   └── vector_engine.hpp
├── components/             # OLD - flat structure, wrong names
│   ├── lpddr5_memory_controller.hpp    # Should be: temporal/memory/controllers/lpddr5_controller.hpp
│   ├── ddr5_memory_controller.hpp
│   ├── gddr6_memory_controller.hpp
│   ├── gddr7_memory_controller.hpp
│   ├── hbm2_memory_controller.hpp
│   ├── hbm3_memory_controller.hpp
│   ├── compute/
│   │   ├── behavioral_compute_fabric.hpp
│   │   ├── transactional_compute_fabric.hpp
│   │   └── compute_fabric_interface.hpp
│   ├── dma/
│   │   ├── behavioral_dma_engine.hpp
│   │   ├── transactional_dma_engine.hpp
│   │   ├── cycle_accurate_dma_engine.hpp
│   │   └── dma_engine_interface.hpp
│   └── memory/
│       ├── behavioral_memory_controller.hpp
│       ├── transactional_memory_controller.hpp
│       ├── behavioral_l3_tile.hpp
│       ├── transactional_l3_tile.hpp
│       └── *_interface.hpp
└── noc/
    ├── behavioral_noc.hpp
    ├── wormhole_router.hpp
    ├── dataflow_noc.hpp
    └── noc_interface.hpp
```

## Target Structure

```
include/sw/kpu/
├── models/
│   ├── interfaces/                     # All interface definitions
│   │   ├── compute_fabric_interface.hpp
│   │   ├── dma_engine_interface.hpp
│   │   ├── memory_controller_interface.hpp
│   │   ├── l3_tile_interface.hpp
│   │   └── noc_interface.hpp
│   │
│   ├── behavioral/
│   │   ├── compute/
│   │   │   └── compute_fabric.hpp      # Was: behavioral_compute_fabric.hpp
│   │   ├── datamovement/
│   │   │   ├── block_mover.hpp
│   │   │   └── dma_engine.hpp          # Was: behavioral_dma_engine.hpp
│   │   ├── memory/
│   │   │   ├── memory_model.hpp
│   │   │   ├── memory_controller.hpp   # Was: behavioral_memory_controller.hpp
│   │   │   └── l3_tile.hpp             # Was: behavioral_l3_tile.hpp
│   │   ├── noc/
│   │   │   └── noc.hpp                 # Was: behavioral_noc.hpp
│   │   ├── orchestrator.hpp
│   │   ├── mlp_executor.hpp
│   │   └── vector_engine.hpp
│   │
│   ├── transactional/
│   │   ├── compute/
│   │   │   └── compute_fabric.hpp      # Was: transactional_compute_fabric.hpp
│   │   ├── datamovement/
│   │   │   └── dma_engine.hpp          # Was: transactional_dma_engine.hpp
│   │   └── memory/
│   │       ├── memory_controller.hpp   # Was: transactional_memory_controller.hpp
│   │       └── l3_tile.hpp             # Was: transactional_l3_tile.hpp
│   │
│   └── temporal/
│       ├── compute/
│       │   ├── compute_fabric.hpp      # Legacy cycle-accurate
│       │   ├── systolic_array.hpp
│       │   └── sfu.hpp
│       ├── datamovement/
│       │   ├── dma_engine.hpp          # Was: cycle_accurate_dma_engine.hpp
│       │   ├── block_mover.hpp         # Temporal block mover
│       │   ├── streamer.hpp
│       │   ├── stateful_block_mover.hpp
│       │   ├── l3_interconnect.hpp
│       │   └── vector_engine.hpp
│       ├── memory/
│       │   ├── l1_buffer.hpp
│       │   ├── l2_bank.hpp
│       │   ├── l3_tile.hpp
│       │   ├── page_buffer.hpp
│       │   ├── scratchpad.hpp
│       │   ├── storage_scheduler.hpp
│       │   └── controllers/
│       │       ├── controller_base.hpp     # Was: memory_controller.hpp
│       │       ├── lpddr5_controller.hpp   # Was: lpddr5_memory_controller.hpp
│       │       ├── ddr5_controller.hpp     # Was: ddr5_memory_controller.hpp
│       │       ├── gddr6_controller.hpp    # Was: gddr6_memory_controller.hpp
│       │       ├── gddr7_controller.hpp    # Was: gddr7_memory_controller.hpp
│       │       ├── hbm2_controller.hpp     # Was: hbm2_memory_controller.hpp
│       │       └── hbm3_controller.hpp     # Was: hbm3_memory_controller.hpp
│       └── noc/
│           ├── noc.hpp
│           ├── wormhole_router.hpp
│           ├── dataflow_noc.hpp
│           └── noc_adapters.hpp
│
├── fidelity/               # Keep as-is
├── isa/                    # Keep as-is
├── dataflow/               # Keep as-is
├── calibration/            # Keep as-is
├── config/                 # Keep as-is
└── [top-level files]       # Keep as-is
```

---

## Implementation Phases

### Phase 1: Memory Controllers (Highest Impact)

**Goal:** Rename `*_memory_controller.hpp` to `*_controller.hpp` and move to `models/temporal/memory/controllers/`

| Current Path | New Path |
|-------------|----------|
| `components/lpddr5_memory_controller.hpp` | `models/temporal/memory/controllers/lpddr5_controller.hpp` |
| `components/ddr5_memory_controller.hpp` | `models/temporal/memory/controllers/ddr5_controller.hpp` |
| `components/gddr6_memory_controller.hpp` | `models/temporal/memory/controllers/gddr6_controller.hpp` |
| `components/gddr7_memory_controller.hpp` | `models/temporal/memory/controllers/gddr7_controller.hpp` |
| `components/hbm2_memory_controller.hpp` | `models/temporal/memory/controllers/hbm2_controller.hpp` |
| `components/hbm3_memory_controller.hpp` | `models/temporal/memory/controllers/hbm3_controller.hpp` |
| `components/memory_controller.hpp` | `models/temporal/memory/controllers/controller_base.hpp` |

**Files to update:**
- `src/models/temporal/memory/controllers/*.cpp` (6 files)
- `src/models/temporal/memory/controllers/controller_factory.cpp`
- `patterns/memory/*/*.cpp` (many pattern files)

---

### Phase 2: Behavioral Headers

**Goal:** Consolidate behavioral headers under `models/behavioral/`

| Current Path | New Path |
|-------------|----------|
| `behavioral/block_mover.hpp` | `models/behavioral/datamovement/block_mover.hpp` |
| `behavioral/vector_engine.hpp` | `models/behavioral/compute/vector_engine.hpp` |
| `behavioral/memory_model.hpp` | `models/behavioral/memory/memory_model.hpp` |
| `behavioral/orchestrator.hpp` | `models/behavioral/orchestrator.hpp` |
| `behavioral/mlp_executor.hpp` | `models/behavioral/mlp_executor.hpp` |
| `components/compute/behavioral_compute_fabric.hpp` | `models/behavioral/compute/compute_fabric.hpp` |
| `components/dma/behavioral_dma_engine.hpp` | `models/behavioral/datamovement/dma_engine.hpp` |
| `components/memory/behavioral_memory_controller.hpp` | `models/behavioral/memory/memory_controller.hpp` |
| `components/memory/behavioral_l3_tile.hpp` | `models/behavioral/memory/l3_tile.hpp` |
| `noc/behavioral_noc.hpp` | `models/behavioral/noc/noc.hpp` |

**Files to update:**
- `src/models/behavioral/**/*.cpp` (9 files)
- `examples/behavioral/*.cpp`
- Various tests

---

### Phase 3: Transactional Headers

**Goal:** Consolidate transactional headers under `models/transactional/`

| Current Path | New Path |
|-------------|----------|
| `components/compute/transactional_compute_fabric.hpp` | `models/transactional/compute/compute_fabric.hpp` |
| `components/dma/transactional_dma_engine.hpp` | `models/transactional/datamovement/dma_engine.hpp` |
| `components/memory/transactional_memory_controller.hpp` | `models/transactional/memory/memory_controller.hpp` |
| `components/memory/transactional_l3_tile.hpp` | `models/transactional/memory/l3_tile.hpp` |

**Files to update:**
- `src/models/transactional/**/*.cpp` (4 files)
- Factory files in temporal/

---

### Phase 4: Temporal Headers

**Goal:** Move remaining cycle-accurate/temporal headers

| Current Path | New Path |
|-------------|----------|
| `components/compute_fabric.hpp` | `models/temporal/compute/compute_fabric.hpp` |
| `components/systolic_array.hpp` | `models/temporal/compute/systolic_array.hpp` |
| `components/sfu.hpp` | `models/temporal/compute/sfu.hpp` |
| `components/dma_engine.hpp` | `models/temporal/datamovement/dma_engine.hpp` |
| `components/dma/cycle_accurate_dma_engine.hpp` | `models/temporal/datamovement/cycle_accurate_dma_engine.hpp` |
| `components/block_mover.hpp` | `models/temporal/datamovement/block_mover.hpp` |
| `components/streamer.hpp` | `models/temporal/datamovement/streamer.hpp` |
| `components/stateful_block_mover.hpp` | `models/temporal/datamovement/stateful_block_mover.hpp` |
| `components/l3_interconnect.hpp` | `models/temporal/datamovement/l3_interconnect.hpp` |
| `components/vector_engine.hpp` | `models/temporal/datamovement/vector_engine.hpp` |
| `components/l1_buffer.hpp` | `models/temporal/memory/l1_buffer.hpp` |
| `components/l2_bank.hpp` | `models/temporal/memory/l2_bank.hpp` |
| `components/l3_tile.hpp` | `models/temporal/memory/l3_tile.hpp` |
| `components/page_buffer.hpp` | `models/temporal/memory/page_buffer.hpp` |
| `components/scratchpad.hpp` | `models/temporal/memory/scratchpad.hpp` |
| `components/storage_scheduler.hpp` | `models/temporal/memory/storage_scheduler.hpp` |
| `noc/noc.hpp` | `models/temporal/noc/noc.hpp` |
| `noc/wormhole_router.hpp` | `models/temporal/noc/wormhole_router.hpp` |
| `noc/dataflow_noc.hpp` | `models/temporal/noc/dataflow_noc.hpp` |
| `noc/noc_adapters.hpp` | `models/temporal/noc/noc_adapters.hpp` |

---

### Phase 5: Interface Consolidation

**Goal:** Move all interfaces to `models/interfaces/`

| Current Path | New Path |
|-------------|----------|
| `components/compute/compute_fabric_interface.hpp` | `models/interfaces/compute_fabric_interface.hpp` |
| `components/dma/dma_engine_interface.hpp` | `models/interfaces/dma_engine_interface.hpp` |
| `components/memory/memory_controller_interface.hpp` | `models/interfaces/memory_controller_interface.hpp` |
| `components/memory/l3_tile_interface.hpp` | `models/interfaces/l3_tile_interface.hpp` |
| `noc/noc_interface.hpp` | `models/interfaces/noc_interface.hpp` |

---

### Phase 6: Cleanup

1. Remove empty `components/` directory (after all moves complete)
2. Remove empty `noc/` directory (after all moves complete)
3. Update any remaining stale includes
4. Verify all tests pass

---

## Include Update Strategy

For each moved header, we need to:

1. **Create compatibility header** (optional, for gradual migration):
   ```cpp
   // Old location: include/sw/kpu/components/lpddr5_memory_controller.hpp
   #pragma once
   #warning "This header is deprecated. Use <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>"
   #include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>
   ```

2. **Update source files**: Change `#include` statements to new paths

3. **Update CMakeLists.txt**: Update header file lists

---

## File Count Summary

| Phase | Headers to Move | Source Files to Update | Est. Includes to Update |
|-------|-----------------|------------------------|------------------------|
| 1. Memory Controllers | 7 | ~20 | ~50 |
| 2. Behavioral | 10 | ~15 | ~30 |
| 3. Transactional | 4 | ~8 | ~15 |
| 4. Temporal | 20 | ~30 | ~80 |
| 5. Interfaces | 5 | ~10 | ~20 |
| **Total** | **46** | **~83** | **~195** |

---

## Verification

After each phase:
1. `cmake --build --preset release`
2. `ctest --preset default`
3. Commit changes

---

## Notes

- The `fidelity/`, `isa/`, `dataflow/`, `calibration/`, and `config/` directories remain unchanged
- Top-level headers (`kpu.hpp`, `kpu_simulator.hpp`, etc.) remain unchanged
- C API headers (`kpu_c_*.h`) remain unchanged
- The `sw/memory/` namespace headers are separate from `sw/kpu/` and not affected
