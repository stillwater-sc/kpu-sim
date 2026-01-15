# Include directory restructuring

Phase 1 Complete: Memory Controllers Restructured

  Summary of changes:

  1. Created new directory structure:
  include/sw/kpu/models/temporal/memory/controllers/
  2. Moved and renamed 7 header files:
  | Old Path                                | New Path                                                 |
  |-----------------------------------------|----------------------------------------------------------|
  | components/lpddr5_memory_controller.hpp | models/temporal/memory/controllers/lpddr5_controller.hpp |
  | components/ddr5_memory_controller.hpp   | models/temporal/memory/controllers/ddr5_controller.hpp   |
  | components/gddr6_memory_controller.hpp  | models/temporal/memory/controllers/gddr6_controller.hpp  |
  | components/gddr7_memory_controller.hpp  | models/temporal/memory/controllers/gddr7_controller.hpp  |
  | components/hbm2_memory_controller.hpp   | models/temporal/memory/controllers/hbm2_controller.hpp   |
  | components/hbm3_memory_controller.hpp   | models/temporal/memory/controllers/hbm3_controller.hpp   |
  | components/memory_controller.hpp        | models/temporal/memory/controllers/controller_base.hpp   |

  3. Updated ~40 source files to use new include paths
  4. Created backward-compatibility headers at old locations that forward to new locations with deprecation warnings
  5. Build and tests pass (207 targets built, behavioral tests pass)

  Remaining phases:
  - Phase 2: Behavioral headers (10 files)
  - Phase 3: Transactional headers (4 files)
  - Phase 4: Temporal compute/datamovement/noc headers (20 files)
  - Phase 5: Interface consolidation and cleanup

Phase 2 Complete: Behavioral Headers Restructured

Summary of changes:

  1. Created new directory structure:
```
  include/sw/kpu/models/behavioral/
  ├── compute/
  │   ├── compute_fabric.hpp
  │   └── vector_engine.hpp
  ├── datamovement/
  │   ├── block_mover.hpp
  │   └── dma_engine.hpp
  ├── memory/
  │   ├── l3_tile.hpp
  │   ├── memory_controller.hpp
  │   └── memory_model.hpp
  ├── noc/
  │   └── noc.hpp
  ├── mlp_executor.hpp
  └── orchestrator.hpp
```
  2. Moved and renamed 10 header files from:
    - behavioral/ → models/behavioral/
    - components/compute/behavioral_* → models/behavioral/compute/
    - components/dma/behavioral_* → models/behavioral/datamovement/
    - components/memory/behavioral_* → models/behavioral/memory/
    - noc/behavioral_* → models/behavioral/noc/
  3. Updated ~15 source files and CMakeLists.txt
  4. Created 10 backward-compatibility headers at old locations
  5. Build succeeded (235 targets), all behavioral tests pass

Remaining phases:
  - Phase 3: Transactional headers (4 files)
  - Phase 4: Temporal compute/datamovement/noc headers (20 files)
  - Phase 5: Interface consolidation and cleanup

Phase 3 Complete: Transactional Headers Restructured

Summary of changes:

  1. Created new directory structure:
```
  include/sw/kpu/models/transactional/
  ├── compute/
  │   └── compute_fabric.hpp
  ├── datamovement/
  │   └── dma_engine.hpp
  └── memory/
      ├── l3_tile.hpp
      └── memory_controller.hpp
```
  2. Moved and renamed 4 header files from:
    - components/compute/transactional_compute_fabric.hpp → models/transactional/compute/compute_fabric.hpp
    - components/dma/transactional_dma_engine.hpp → models/transactional/datamovement/dma_engine.hpp
    - components/memory/transactional_memory_controller.hpp → models/transactional/memory/memory_controller.hpp
    - components/memory/transactional_l3_tile.hpp → models/transactional/memory/l3_tile.hpp
  3. Updated 9 source files including factory files and CMakeLists.txt
  4. Created 4 backward-compatibility headers at old locations
  5. Build succeeded (226 targets), all tests pass

Remaining phases:
  - Phase 4: Temporal compute/datamovement/noc headers (~20 files)
  - Phase 5: Interface consolidation and cleanup

The include directory restructuring is now complete. Here's a summary of what was accomplished:

Include Restructuring Summary

  New Directory Structure:
```
  include/sw/kpu/models/
  ├── behavioral/
  │   ├── compute/          (compute_fabric.hpp, vector_engine.hpp)
  │   ├── datamovement/     (block_mover.hpp, dma_engine.hpp)
  │   ├── memory/           (memory_model.hpp, l3_tile.hpp, memory_controller.hpp)
  │   ├── noc/              (noc.hpp)
  │   ├── orchestrator.hpp
  │   └── mlp_executor.hpp
  ├── transactional/
  │   ├── compute/          (compute_fabric.hpp)
  │   ├── datamovement/     (dma_engine.hpp)
  │   └── memory/           (l3_tile.hpp, memory_controller.hpp)
  └── temporal/
      ├── compute/          (compute_fabric.hpp, systolic_array.hpp, sfu.hpp)
      ├── datamovement/     (dma_engine.hpp, block_mover.hpp, streamer.hpp, etc.)
      ├── memory/
      │   ├── controllers/  (lpddr5_controller.hpp, ddr5_controller.hpp, etc.)
      │   └── (l1_buffer.hpp, l2_bank.hpp, l3_tile.hpp, etc.)
      └── noc/              (noc.hpp, wormhole_router.hpp, dataflow_noc.hpp)
```

  Backward Compatibility:
  - All old paths (components/*.hpp, noc/*.hpp) now redirect to new locations
  - Deprecation warnings alert developers to update their includes
  - Build and all 63 tests pass

Key Changes:
  - Memory controllers renamed: *_memory_controller.hpp → *_controller.hpp
  - Files organized by simulation fidelity (behavioral/transactional/temporal)
  - Headers now mirror the source directory structure in src/models/

## Cleanup Complete

The include directory restructuring is complete. Here's the summary:

Removed directories:
  - include/sw/kpu/components/ (all content moved to models/)
  - include/sw/kpu/noc/ (moved to models/temporal/noc/)
  - include/sw/kpu/behavioral/ (moved to models/behavioral/)

New structure:
```
  include/sw/kpu/models/
  ├── behavioral/
  │   ├── compute/     (compute_fabric.hpp, vector_engine.hpp)
  │   ├── datamovement/(block_mover.hpp, dma_engine.hpp)
  │   ├── memory/      (memory_model.hpp, l3_tile.hpp, memory_controller.hpp)
  │   ├── noc/         (noc.hpp)
  │   ├── orchestrator.hpp
  │   └── mlp_executor.hpp
  ├── transactional/
  │   ├── compute/     (compute_fabric.hpp)
  │   ├── datamovement/(dma_engine.hpp)
  │   └── memory/      (l3_tile.hpp, memory_controller.hpp)
  ├── temporal/
  │   ├── compute/     (compute_fabric.hpp, systolic_array.hpp, sfu.hpp)
  │   ├── datamovement/(dma_engine.hpp, block_mover.hpp, streamer.hpp, etc.)
  │   ├── memory/
  │   │   ├── controllers/ (lpddr5_controller.hpp, hbm3_controller.hpp, etc.)
  │   │   └── (l1_buffer.hpp, l2_bank.hpp, l3_tile.hpp, etc.)
  │   └── noc/         (noc.hpp, wormhole_router.hpp, dataflow_noc.hpp)
  └── interfaces/      (shared interface definitions)
      ├── compute_fabric_interface.hpp
      ├── dma_engine_interface.hpp
      ├── l3_tile_interface.hpp
      ├── memory_controller_interface.hpp
      └── noc_interface.hpp
```

Results:
  - Build: 556 targets compiled successfully
  - Tests: All 63 tests passed
  - No backward-compatibility headers (clean structure)
