# Modeling Bug

❯ why is there so little overlap in the DMAs? The DMA engines are independent, as are the A, the B, and the C matrix, so we should be able to
  concurrently DMA the A and B matrix tiles into the machine. Secondly, we are a blocked two level matmul, so we also should be able to concurrently
  feed each row and column of the checkerboard concurrently. Why are these not happening? If we look at the programs that run on the DMA, BlockMover,
  and Streamer as Communicating Sequential Processes, and we kick off their programs concurrently, and we have a DMA engine attached to each row and
  each column of the checkerboard, as a matter of fact, we should have multiple DMA engines connected to each Memory Controller, and we can have
  multiple tile requests running concurrently to overcome the latency of the page miss requests to each bank. Why are these natural concurrencies not
  happening? Can you RCA this question as this is a massive loss of performance.

This is a critical architecture question. Let me trace through why concurrency isn't happening.

● Read(src/software/isa/transactional_program_executor.cpp)

● Let me check the schedule compiler to see how instructions are ordered:

● Search(pattern: "matmul_output_stationary", path: "~/dev/stillwater/clones/kpu-sim", glob: "**/*.cpp")
  ⎿  Found 9 files (ctrl+o to expand)

● Read(src/schedules/matmul_schedule.cpp)

● Root Cause Analysis: Single Sequential Instruction Stream vs. CSP Model

The Problem

```python
  # Looking at the schedule (lines 53-68):
  sched.for_tiles("ti")
      .for_tiles("tj")
          .for_tiles("tk")
              .load(MatrixID::A)      // DMA instruction 1
              .load(MatrixID::B)      // DMA instruction 2
              .barrier()              // SYNC
              .move(MatrixID::A)      // BM instruction 1
              .move(MatrixID::B)      // BM instruction 2
              .stream_rows(MatrixID::A)
              .stream_cols(MatrixID::B)
```

This generates a single sequential instruction stream:

```text
  DMA_LOAD A[0,0,0]
  DMA_LOAD B[0,0,0]
  BARRIER
  BM_MOVE A[0,0,0]
  BM_MOVE B[0,0,0]
  STR_FEED_ROWS A
  STR_FEED_COLS B
  DMA_LOAD A[0,0,1]   <-- Next iteration, but sequentially after previous!
  ...
```

## Why Concurrency Is Lost

The timing model processes this as:

```text
  Cycle 0:   Process DMA_LOAD A[0,0,0] → schedules on channel 0, ends cycle 18
  Cycle 18:  Process DMA_LOAD B[0,0,0] → schedules on channel 1, ends cycle 36
             (Even though channel 1 was free at cycle 0!)
```

The issue is instruction-level serialization in the timing model:

  - dispatch_with_timing() is called for each instruction sequentially
  - Even though DMA channels are independent, we don't model issuing both loads at cycle 0

The KPU design is based on a Communicating Sequential Processes (CSP) Model.

CSP is fundamentally different - three concurrent processes:

  DMA Process:                    BlockMover Process:           Streamer Process:
  --------------                  -------------------           -----------------
  for all tiles:                  while running:                while running:
    if credit_l3:                   if tile_in_l3 && credit_l2:   if tile_in_l2:
      issue_load(tile)                move(tile)                    stream(tile)

These run concurrently and synchronize via:

  - Credits (downstream buffer availability)
  - Tag CAM (tile arrival detection)


## What's Missing
  ┌──────────────────────────────────────┬─────────────────────────────────────────────────┐
  │            Current Model             │                    CSP Model                    │
  ├──────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Single PC, single instruction stream │ Multiple concurrent instruction streams         │
  ├──────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Sequential dispatch                  │ Parallel issue to all ready resources           │
  ├──────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Barrier synchronization              │ Data-driven synchronization (credits + tag CAM) │
  ├──────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Resource parallelism only            │ True instruction-level parallelism              │
  └──────────────────────────────────────┴─────────────────────────────────────────────────┘

## The Fix Required

The timing model needs to be restructured to:

  1. Parse the schedule into per-component queues:
    - DMA queue: all DMA_LOAD/STORE operations
    - BM queue: all BM_MOVE/WRITEBACK operations
    - STR queue: all STR_FEED/DRAIN operations
  2. Simulate concurrent execution:
    - Each component pulls from its queue when resources AND data are ready
    - Multiple DMA channels can have multiple in-flight requests
    - Clock advances when all components have either issued or are stalled
  3. Model true credit-based flow:
    - DMA issues when: channel free AND L3 credit available
    - BM issues when: BM free AND tile in L3 (tag match) AND L2 credit available
    - STR issues when: STR free AND tile in L2 (tag match)

This is a significant architectural change from "timing overlay on sequential execution" to "discrete event simulation of concurrent dataflow."

Would you like me to design a concurrent timing model that properly captures the CSP semantics?

● I'll design a concurrent timing model based on CSP semantics with proper credit-based dataflow. Let me create a comprehensive design document.

● Write(docs/plans/concurrent_timing_model.md)
  ⎿  Wrote 891 lines to docs/plans/concurrent_timing_model.md
      1 # Concurrent Timing Model Design
      2 ## Overview
      3 This document describes a discrete-event simulation model for KPU timing that captures
      4 the natural concurrency of the credit-based dataflow architecture. The model treats
      5 DMA engines, BlockMovers, and Streamers as **Communicating Sequential Processes (CSP)**
      6 that run concurrently and synchronize via credits and tag CAM matches.
      7 ## Current Model Problems
     … +881 lines (ctrl+o to expand)

● Created design at docs/plans/concurrent_timing_model.md. Here's the summary:

Key Architectural Change

```text
  Current Model (Sequential)
  Single PC → DMA → DMA → BARRIER → BM → BM → STR → STR → ...
                   (one at a time)

  Concurrent Model (CSP)
  DMA Process 0: ──────────────────────────────────────────►
  DMA Process 1: ──────────────────────────────────────────►
  DMA Process 2: ──────────────────────────────────────────►
  DMA Process 3: ──────────────────────────────────────────►
                 └──────────┬───────────┘
                            ▼ (L3 tag CAM match)
  BM Process 0:        ────────────────────────────────────►
  BM Process 1:        ────────────────────────────────────►
                            └──────────┬───────────┘
                                       ▼ (L2 tag CAM match)
  Streamer 0:                     ─────────────────────────►
  Streamer 1:                     ─────────────────────────►
```

Core Components

  ┌───────────────────┬──────────────────────────────────────────────────────────────────────────┐
  │     Component     │                                 Purpose                                  │
  ├───────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ CreditPool        │ Models buffer availability (L3: 32 buffers, L2: 64 banks)                │
  ├───────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ TagCAM            │ Hardware tile matching for out-of-order arrival                          │
  ├───────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ WorkQueue         │ Per-component operation queues                                           │
  ├───────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ DMAEngineProcess  │ Issues loads when credit available, inserts into L3 TagCAM on completion │
  ├───────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ BlockMoverProcess │ Waits for L3 tag match + L2 credit, inserts into L2 TagCAM               │
  ├───────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ StreamerProcess   │ Waits for L2 tag match, feeds to compute                                 │
  └───────────────────┴──────────────────────────────────────────────────────────────────────────┘

  Program Distribution

  At load time, unroll the schedule and distribute to component queues:
  Sequential:  [DMA A, DMA B, BM A, BM B, STR A, STR B, ...]
                                ↓ distribute
  DMA Queue:   [A[0,0], B[0,0], A[0,1], B[0,1], ...]
  BM Queue:    [A[0,0], B[0,0], A[0,1], B[0,1], ...]
  STR Queue:   [A[0,0], B[0,0], ...]

  Synchronization (No Barriers Needed!)

```cpp
  // DMA issues when:
  if (l3_credits.acquire()) {
      issue_dma_load(tile);
  }

  // BlockMover issues when:
  if (l3_tag_cam.lookup(tile_id) && l2_credits.acquire()) {
      issue_block_move(tile);
  }

  // Streamer issues when:
  if (l2_tag_cam.lookup(tile_id)) {
      issue_stream(tile);
  }
```

Data dependencies are enforced by tag CAM matches, not explicit barriers!

Expected Performance

With 4 DMA engines loading A and B tiles concurrently:

  - Current: DMAs serialize, ~4x longer than necessary
  - Concurrent: 4 parallel DMAs, pipelined with BM/STR

Expected speedup: 4-8x for memory-bound workloads.

  Would you like me to start implementing this design?


## Live Lock Avoidance

❯ With credits being given out to upstream providers, but downstream consumers only consuming when there is a tag match, this will create opportunities
  for live lock. How are you going to plan for avoiding live lock conditions?

● Excellent question - this is a critical correctness issue. Let me analyze the livelock scenarios and design mitigations.

  Livelock Scenarios

  Scenario 1: Credit Exhaustion with Wrong Tiles
  L3 Buffers: [A[0,1], A[0,2], A[0,3], B[0,1], B[0,2], ...] (all 32 full)
  BlockMover waiting for: A[0,0] (not loaded yet!)
  DMA waiting for: L3 credit (none available)
  → LIVELOCK: Can't load A[0,0] because buffers full of "wrong" tiles

  Scenario 2: Circular Dependency
  DMA → needs L3 credit → held by tile waiting for BM
  BM → needs L2 credit → held by tile waiting for STR
  STR → waiting for different tile → can't be loaded (no L3 credit)
  → LIVELOCK: Circular wait

  Scenario 3: Out-of-Order Completion
  Issued: A[0,0], A[0,1], A[0,2] (in order)
  Completed: A[0,2], A[0,1] (out of order due to DRAM bank conflicts)
  BM waiting for: A[0,0] (oldest, but slowest)
  L3 filling up with tiles BM won't process yet
  → LIVELOCK: Head-of-line blocking

  Let me add livelock prevention to the design:

