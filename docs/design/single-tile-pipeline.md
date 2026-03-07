# Minimial Tile Pipeline

A minimal "single tile journey" test would be much more educational than a full matmul. Let me design this.

  Proposed: Single Tile Pipeline Demo

  Scenario: 1×1×1 matmul - just C[0,0] = A[0,0] × B[0,0]

  This shows the complete dataflow:
  DRAM → L3 → L2 → Compute → L2 → L3 → DRAM
        DMA   BM    STR      STR   BM    DMA

  Transaction Log Format (much more educational than Chrome trace):
  Cycle | Component   | Event           | Tile     | Credits      | TagCAM Action
  ------+-------------+-----------------+----------+--------------+------------------
      0 | MC0:CH0     | LOAD_START      | A[0,0,0] | L3: 32→31    | -
      0 | MC0:CH1     | LOAD_START      | B[0,0,0] | L3: 31→30    | -
      0 | L3(0,0):BM  | STALL_TAG       | A[0,0,0] | -            | lookup MISS
     50 | MC0:CH0     | LOAD_COMPLETE   | A[0,0,0] | -            | L3.insert(A[0,0,0])
     50 | L3(0,0):BM  | MOVE_START      | A[0,0,0] | L2: 64→63    | L3.match(A[0,0,0]) HIT
    ...

  Let me create this educational demo.

       1 /**
       2  * @file csp_pipeline_demo.cpp
       3  * @brief Educational demonstration of CSP credit/debit flow and tag matching
       4  *
       5  * This program demonstrates a minimal 1×1×1 matmul (single tile of each matrix)
       6  * to show the complete dataflow through the KPU memory hierarchy:
       7  *
       8  *   DRAM → L3 → L2 → Compute → L2 → L3 → DRAM
       9  *        DMA   BM    STR      STR   BM    DMA
      10  *

