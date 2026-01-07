# Top Level Architecture

## How kpu_simulator.cpp Works

This file implements a cycle-accurate simulator for a KPU (a matrix-processing accelerator). Here's the architecture:

### Memory Hierarchy

  The simulator models a 6-level memory hierarchy:
```text
  Host Memory (NUMA regions)
      ↓ DMA
  External Memory Banks (KPU local DRAM)
      ↓ DMA
  L3 Tiles (on-chip, software-managed)
      ↓ BlockMover
  L2 Banks (on-chip, software-managed)
      ↓ Streamer
  L1 Buffers (compute fabric stream buffers)
      ↓
  Compute Tiles (domain flow arrays)
```

### Key Components

  | Component       | Purpose                                                              |
  |-----------------|----------------------------------------------------------------------|
  | DMA Engines     | Bulk data movement between Host <--> External <--> L3                |
  | Block Movers    | L3 <--> L2 transfers with optional transformations (transpose, etc.) |
  | Streamers       | L2↔L1 row/column streaming to feed compute                           |
  | Compute Tiles   | Domain Flow Architecture programmable systolic arrays                |
  | Address Decoder | Maps global addresses to memory regions                              |

### Constructor (lines 13-176)

Initializes all components based on Config and builds a programmable memory map:
  - Starts after a 4KB null guard zone
  - Assigns address ranges sequentially (or uses custom base addresses)
  - Attaches the address decoder to DMA engines for address translation

### Simulation Loop

  step() (lines 462-493) advances simulation by one cycle:
  1. Update cycle count on all components
  2. Process DMA transfers
  3. Process block mover transfers
  4. Update streamers
  5. Update compute tiles

  run_until_idle() (lines 495-527) steps until all components are idle.

### Data Flow Example

  run_matmul_test() (lines 561-637) demonstrates a complete operation:
  1. Write matrices A, B to external memory
  2. DMA A, B from external → L1 (scratchpad)
  3. Execute matmul on compute tile
  4. DMA result C from L1 → external
  5. Verify result

### Tracing Support

Lines 934-972 enable/disable tracing on individual components for debugging and visualization.

