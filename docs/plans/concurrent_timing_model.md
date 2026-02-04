# Concurrent Timing Model Design

## Overview

This document describes a discrete-event simulation model for KPU timing that captures
the natural concurrency of the credit-based dataflow architecture. The model treats
DMA engines, BlockMovers, and Streamers as **Communicating Sequential Processes (CSP)**
that run concurrently and synchronize via credits and tag CAM matches.

## Current Model Problems

The existing `TransactionalProgramExecutor` has fundamental limitations:

1. **Single instruction stream**: One PC iterates through instructions sequentially
2. **No issue parallelism**: Even independent DMAs are dispatched one at a time
3. **Timing overlay**: Timing computed post-hoc, not during simulated execution
4. **No credit modeling**: Resources assumed always available
5. **No tag CAM**: Data dependencies tracked by tile key, not by hardware matching

## Architecture

### Credit-Based Dataflow

```
                         CREDITS (upstream)
                              ↑
    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │  DRAM   │───→│   L3    │───→│   L2    │───→│   L1    │───→ Compute
    │         │    │ Buffers │    │  Banks  │    │ Buffers │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘
         ↑              ↓              ↓              ↓
       DMA           BlockMover     Streamer      Systolic
      Engines                                      Array
                         DATA (downstream)
```

### Component Processes

Each component type runs as an independent process:

```
┌────────────────────────────────────────────────────────────────┐
│                    Concurrent Timing Model                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ DMA Engine 0 │  │ DMA Engine 1 │  │ DMA Engine N │   ...   │
│  │   Process    │  │   Process    │  │   Process    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────────────────────────────────────────┐          │
│  │              L3 Buffer Pool                      │          │
│  │   Tag CAM: {tile_id → buffer_id}                │          │
│  │   Credits: available buffer count               │          │
│  └─────────────────────────────────────────────────┘          │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ BlockMover 0 │  │ BlockMover 1 │  │ BlockMover N │   ...   │
│  │   Process    │  │   Process    │  │   Process    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────────────────────────────────────────┐          │
│  │              L2 Bank Array                       │          │
│  │   Tag CAM: {tile_id → bank_id}                  │          │
│  │   Credits: available bank count                 │          │
│  └─────────────────────────────────────────────────┘          │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │  Streamer 0  │  │  Streamer 1  │   (West/North edges)      │
│  │   Process    │  │   Process    │                           │
│  └──────┬───────┘  └──────┬───────┘                           │
│         │                 │                                    │
│         ▼                 ▼                                    │
│  ┌─────────────────────────────────────────────────┐          │
│  │              Compute Fabric                      │          │
│  │   Systolic array with accumulator               │          │
│  └─────────────────────────────────────────────────┘          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Data Structures

### TileDescriptor

```cpp
struct TileDescriptor {
    TileID tile_id;           // {matrix, ti, tj, tk}
    Address dram_address;     // For DMA operations
    Size size_bytes;          // Transfer size

    // Scheduling metadata
    Cycle issue_cycle;        // When operation was issued
    Cycle complete_cycle;     // When operation completes (0 if pending)
};
```

### CreditPool

```cpp
class CreditPool {
public:
    explicit CreditPool(size_t capacity);

    // Credit operations (called by upstream producer)
    bool acquire();           // Try to get a credit, returns false if none
    size_t available() const; // How many credits available

    // Release operation (called by downstream consumer)
    void release();           // Return a credit to the pool

    // For simulation
    void reset();

private:
    size_t capacity_;
    size_t available_;
};
```

### TagCAM

```cpp
class TagCAM {
public:
    explicit TagCAM(size_t capacity);

    // Register tile arrival (called when data arrives)
    void insert(const TileID& tile_id, uint32_t slot_id, Cycle arrival_cycle);

    // Check if tile is present (called by downstream consumer)
    bool lookup(const TileID& tile_id) const;

    // Get slot and arrival time for a tile
    std::optional<std::pair<uint32_t, Cycle>> match(const TileID& tile_id) const;

    // Remove entry (called when tile is consumed)
    void invalidate(const TileID& tile_id);

    void reset();

private:
    struct Entry {
        TileID tile_id;
        uint32_t slot_id;
        Cycle arrival_cycle;
        bool valid;
    };
    std::vector<Entry> entries_;
};
```

### WorkQueue

```cpp
template<typename WorkItem>
class WorkQueue {
public:
    void enqueue(WorkItem item);
    bool empty() const;
    const WorkItem& peek() const;  // Look at front without removing
    WorkItem dequeue();            // Remove and return front
    size_t size() const;
    void reset();

private:
    std::queue<WorkItem> queue_;
};
```

## Component Processes

### DMAEngineProcess

```cpp
class DMAEngineProcess {
public:
    struct Config {
        uint32_t engine_id;
        Size bus_width_bytes = 64;
        Cycle startup_latency = 10;
        double bandwidth_gbps = 25.6;  // Per channel
        size_t queue_depth = 8;        // Outstanding requests
    };

    DMAEngineProcess(const Config& config,
                     CreditPool& l3_credits,
                     TagCAM& l3_tag_cam);

    // Schedule a tile load (adds to work queue)
    void schedule_load(const TileDescriptor& tile);
    void schedule_store(const TileDescriptor& tile);

    // Process one cycle - may issue new request or complete pending
    // Returns events that occurred this cycle
    std::vector<TimingEvent> tick(Cycle current_cycle);

    // State queries
    bool is_idle() const;
    bool has_pending_work() const;
    size_t in_flight_count() const;

    void reset();

private:
    Config config_;
    CreditPool& l3_credits_;
    TagCAM& l3_tag_cam_;

    WorkQueue<TileDescriptor> load_queue_;
    WorkQueue<TileDescriptor> store_queue_;

    // In-flight transfers (could be multiple with queue depth > 1)
    struct InFlightTransfer {
        TileDescriptor tile;
        Cycle start_cycle;
        Cycle complete_cycle;
        bool is_load;
    };
    std::vector<InFlightTransfer> in_flight_;

    Cycle compute_transfer_cycles(Size bytes) const;
    bool can_issue_load(Cycle current_cycle) const;
    bool can_issue_store(Cycle current_cycle) const;
};
```

**DMA Process Logic (per tick):**
```
1. Check for completed transfers:
   for each in_flight transfer where complete_cycle <= current_cycle:
     if is_load:
       l3_tag_cam.insert(tile_id, allocated_buffer, current_cycle)
       emit TILE_ARRIVED_L3 event
     else:
       emit TILE_STORED_DRAM event
     remove from in_flight

2. Try to issue new loads:
   while load_queue not empty AND in_flight < queue_depth:
     if l3_credits.acquire():  // Have downstream buffer space
       tile = load_queue.dequeue()
       cycles = compute_transfer_cycles(tile.size)
       in_flight.push({tile, current_cycle, current_cycle + cycles, true})
       emit DMA_LOAD_START event
     else:
       break  // Stalled on credit

3. Try to issue new stores:
   while store_queue not empty AND in_flight < queue_depth:
     // Stores don't need credits (data already in L3)
     // But need the tile to be in L3 (tag CAM match)
     tile = store_queue.peek()
     if l3_tag_cam.lookup(tile.tile_id):
       store_queue.dequeue()
       cycles = compute_transfer_cycles(tile.size)
       in_flight.push({tile, current_cycle, current_cycle + cycles, false})
       l3_tag_cam.invalidate(tile.tile_id)  // Tile leaving L3
       l3_credits.release()  // Return credit to pool
       emit DMA_STORE_START event
     else:
       break  // Waiting for tile
```

### BlockMoverProcess

```cpp
class BlockMoverProcess {
public:
    struct Config {
        uint32_t mover_id;
        Size bus_width_bytes = 64;
        Cycle startup_latency = 4;
        bool supports_transpose = true;
    };

    BlockMoverProcess(const Config& config,
                      TagCAM& l3_tag_cam,
                      CreditPool& l3_credits,  // Returns credits when consuming from L3
                      CreditPool& l2_credits,  // Acquires credits when writing to L2
                      TagCAM& l2_tag_cam);

    // Schedule tile movement
    void schedule_move(const TileDescriptor& tile, bool transpose = false);
    void schedule_writeback(const TileDescriptor& tile);

    std::vector<TimingEvent> tick(Cycle current_cycle);

    bool is_idle() const;
    bool has_pending_work() const;

    void reset();

private:
    Config config_;
    TagCAM& l3_tag_cam_;
    CreditPool& l3_credits_;
    CreditPool& l2_credits_;
    TagCAM& l2_tag_cam_;

    WorkQueue<TileDescriptor> move_queue_;      // L3 → L2
    WorkQueue<TileDescriptor> writeback_queue_; // L2 → L3

    struct InFlightTransfer {
        TileDescriptor tile;
        Cycle start_cycle;
        Cycle complete_cycle;
        bool is_move;  // true = L3→L2, false = L2→L3
    };
    std::optional<InFlightTransfer> in_flight_;  // One at a time per BlockMover

    Cycle compute_transfer_cycles(Size bytes) const;
};
```

**BlockMover Process Logic (per tick):**
```
1. Check for completed transfer:
   if in_flight.has_value() AND in_flight.complete_cycle <= current_cycle:
     if is_move (L3→L2):
       l2_tag_cam.insert(tile_id, allocated_bank, current_cycle)
       l3_tag_cam.invalidate(tile_id)  // Tile left L3
       l3_credits.release()            // Return L3 credit
       emit TILE_ARRIVED_L2 event
     else (writeback L2→L3):
       l3_tag_cam.insert(tile_id, allocated_buffer, current_cycle)
       l2_tag_cam.invalidate(tile_id)  // Tile left L2
       l2_credits.release()            // Return L2 credit
       emit TILE_ARRIVED_L3 event
     in_flight.reset()

2. If not busy, try to issue new work:
   if not in_flight.has_value():
     // Priority: moves over writebacks (keep pipeline fed)
     if move_queue not empty:
       tile = move_queue.peek()
       if l3_tag_cam.lookup(tile.tile_id) AND l2_credits.acquire():
         move_queue.dequeue()
         cycles = compute_transfer_cycles(tile.size)
         in_flight = {tile, current_cycle, current_cycle + cycles, true}
         emit BM_MOVE_START event
       // else: stalled on tag match or credit

     else if writeback_queue not empty:
       tile = writeback_queue.peek()
       if l2_tag_cam.lookup(tile.tile_id) AND l3_credits.acquire():
         writeback_queue.dequeue()
         cycles = compute_transfer_cycles(tile.size)
         in_flight = {tile, current_cycle, current_cycle + cycles, false}
         emit BM_WRITEBACK_START event
```

### StreamerProcess

```cpp
class StreamerProcess {
public:
    struct Config {
        uint32_t streamer_id;
        StreamerType type;  // ROW_STREAMER or COL_STREAMER
        Size bus_width_bytes = 64;
        Cycle startup_latency = 2;
        size_t l1_depth = 4;  // Double-buffering depth
    };

    enum class StreamerType { ROW_STREAMER, COL_STREAMER };

    StreamerProcess(const Config& config,
                    TagCAM& l2_tag_cam,
                    CreditPool& l2_credits);

    void schedule_feed(const TileDescriptor& tile);
    void schedule_drain(const TileDescriptor& tile);

    std::vector<TimingEvent> tick(Cycle current_cycle);

    bool is_idle() const;
    bool has_pending_work() const;

    void reset();

private:
    Config config_;
    TagCAM& l2_tag_cam_;
    CreditPool& l2_credits_;

    WorkQueue<TileDescriptor> feed_queue_;
    WorkQueue<TileDescriptor> drain_queue_;

    std::optional<InFlightTransfer> in_flight_;

    Cycle compute_transfer_cycles(Size bytes) const;
};
```

**Streamer Process Logic (per tick):**
```
1. Check for completed transfer:
   if in_flight.has_value() AND in_flight.complete_cycle <= current_cycle:
     if is_feed:
       l2_tag_cam.invalidate(tile_id)  // Tile consumed from L2
       l2_credits.release()            // Return L2 credit
       emit TILE_FED_TO_COMPUTE event
       // Trigger compute (systolic array fires when both A and B ready)
     else (drain):
       // Result tile goes to L2 (needs credit, already acquired)
       l2_tag_cam.insert(tile_id, bank, current_cycle)
       emit TILE_DRAINED event
     in_flight.reset()

2. If not busy, try to issue new work:
   if not in_flight.has_value():
     if feed_queue not empty:
       tile = feed_queue.peek()
       if l2_tag_cam.lookup(tile.tile_id):
         feed_queue.dequeue()
         cycles = compute_transfer_cycles(tile.size)
         in_flight = {tile, current_cycle, current_cycle + cycles, true}
         emit STR_FEED_START event

     else if drain_queue not empty:
       tile = drain_queue.peek()
       if l2_credits.acquire():  // Need space in L2 for result
         drain_queue.dequeue()
         cycles = compute_transfer_cycles(tile.size)
         in_flight = {tile, current_cycle, current_cycle + cycles, false}
         emit STR_DRAIN_START event
```

## Concurrent Executor

### ConcurrentTimingExecutor

```cpp
class ConcurrentTimingExecutor {
public:
    struct Config {
        // DMA configuration
        size_t num_dma_engines = 4;
        size_t dma_queue_depth = 8;
        double dma_bandwidth_gbps = 25.6;

        // L3 configuration
        size_t l3_buffer_count = 32;
        size_t l3_buffer_size = 64 * 1024;  // 64KB per buffer

        // BlockMover configuration
        size_t num_block_movers = 4;

        // L2 configuration
        size_t l2_bank_count = 64;
        size_t l2_bank_size = 64 * 1024;

        // Streamer configuration
        size_t num_row_streamers = 2;  // West edge
        size_t num_col_streamers = 2;  // North edge

        // Timing parameters
        double reference_clock_mhz = 1000.0;
        Cycle max_cycles = 10'000'000;
    };

    explicit ConcurrentTimingExecutor(const Config& config);

    // Load program and distribute work to component queues
    void load_program(const DMProgram& program,
                      Address a_base, Address b_base, Address c_base);

    // Run simulation to completion
    bool run();

    // Step one cycle (for debugging)
    bool step();

    // Results
    Cycle total_cycles() const { return current_cycle_; }
    const std::vector<TimingEvent>& events() const { return events_; }

    // Statistics
    struct Statistics {
        Cycle total_cycles;
        Cycle dma_busy_cycles;
        Cycle bm_busy_cycles;
        Cycle str_busy_cycles;
        Cycle compute_cycles;

        // Stall breakdown
        Cycle dma_credit_stalls;    // Waiting for L3 credit
        Cycle bm_tag_stalls;        // Waiting for tile in L3
        Cycle bm_credit_stalls;     // Waiting for L2 credit
        Cycle str_tag_stalls;       // Waiting for tile in L2

        // Throughput
        size_t tiles_loaded;
        size_t tiles_stored;
        size_t tiles_moved;
        size_t tiles_streamed;

        double dma_utilization() const;
        double bm_utilization() const;
        double str_utilization() const;
    };
    Statistics get_statistics() const;

    // Export
    void export_chrome_trace(const std::string& filename) const;

private:
    Config config_;
    Cycle current_cycle_ = 0;
    std::vector<TimingEvent> events_;

    // Credit pools
    CreditPool l3_credits_;
    CreditPool l2_credits_;

    // Tag CAMs
    TagCAM l3_tag_cam_;
    TagCAM l2_tag_cam_;

    // Component processes
    std::vector<std::unique_ptr<DMAEngineProcess>> dma_engines_;
    std::vector<std::unique_ptr<BlockMoverProcess>> block_movers_;
    std::vector<std::unique_ptr<StreamerProcess>> row_streamers_;
    std::vector<std::unique_ptr<StreamerProcess>> col_streamers_;

    // Program distribution
    void distribute_program(const DMProgram& program,
                           Address a_base, Address b_base, Address c_base);

    // Assign work to specific engines (load balancing)
    uint32_t select_dma_engine(MatrixID matrix, const TileCoord& tile) const;
    uint32_t select_block_mover(MatrixID matrix, const TileCoord& tile) const;
    uint32_t select_row_streamer(const TileCoord& tile) const;
    uint32_t select_col_streamer(const TileCoord& tile) const;

    // Check if simulation is complete
    bool is_complete() const;
};
```

### Program Distribution

The key insight is that the single instruction stream must be **distributed** to
component queues at load time, not executed sequentially:

```cpp
void ConcurrentTimingExecutor::distribute_program(
    const DMProgram& program,
    Address a_base, Address b_base, Address c_base)
{
    // Execute the program symbolically to extract all tile operations
    // This handles loops by unrolling them

    LoopState loop_state;
    AddressGenerator addr_gen;
    addr_gen.set_base(MatrixID::A, a_base);
    addr_gen.set_base(MatrixID::B, b_base);
    addr_gen.set_base(MatrixID::C, c_base);

    size_t pc = 0;
    while (pc < program.instructions.size()) {
        const auto& instr = program.instructions[pc];

        switch (instr.opcode) {
        case DMOpcode::LOOP_BEGIN: {
            const auto& ops = std::get<LoopOperands>(instr.operands);
            loop_state.begin_loop(ops.loop_id, ops.loop_count,
                                  ops.index_role, ops.loop_stride, pc);
            ++pc;
            break;
        }

        case DMOpcode::LOOP_END: {
            const auto& ops = std::get<LoopOperands>(instr.operands);
            pc = loop_state.end_loop(ops.loop_id, pc);
            break;
        }

        case DMOpcode::DMA_LOAD_TILE:
        case DMOpcode::DMA_LOAD_TILE_AUTO: {
            TileDescriptor tile = extract_tile_descriptor(instr, loop_state, addr_gen);
            uint32_t engine = select_dma_engine(tile.tile_id.matrix, tile.tile_id.coord);
            dma_engines_[engine]->schedule_load(tile);
            ++pc;
            break;
        }

        case DMOpcode::DMA_STORE_TILE:
        case DMOpcode::DMA_STORE_TILE_AUTO: {
            TileDescriptor tile = extract_tile_descriptor(instr, loop_state, addr_gen);
            uint32_t engine = select_dma_engine(tile.tile_id.matrix, tile.tile_id.coord);
            dma_engines_[engine]->schedule_store(tile);
            ++pc;
            break;
        }

        case DMOpcode::BM_MOVE_TILE:
        case DMOpcode::BM_MOVE_TILE_AUTO: {
            TileDescriptor tile = extract_tile_descriptor(instr, loop_state, addr_gen);
            uint32_t mover = select_block_mover(tile.tile_id.matrix, tile.tile_id.coord);
            block_movers_[mover]->schedule_move(tile);
            ++pc;
            break;
        }

        // ... similar for other opcodes

        case DMOpcode::HALT:
            return;  // Done distributing

        default:
            ++pc;
            break;
        }
    }
}
```

### Main Simulation Loop

```cpp
bool ConcurrentTimingExecutor::run() {
    while (!is_complete() && current_cycle_ < config_.max_cycles) {
        step();
    }
    return is_complete();
}

bool ConcurrentTimingExecutor::step() {
    // Tick all components in parallel (order doesn't matter for same cycle)

    // 1. Tick DMA engines
    for (auto& engine : dma_engines_) {
        auto events = engine->tick(current_cycle_);
        events_.insert(events_.end(), events.begin(), events.end());
    }

    // 2. Tick BlockMovers
    for (auto& mover : block_movers_) {
        auto events = mover->tick(current_cycle_);
        events_.insert(events_.end(), events.begin(), events.end());
    }

    // 3. Tick Streamers
    for (auto& streamer : row_streamers_) {
        auto events = streamer->tick(current_cycle_);
        events_.insert(events_.end(), events.begin(), events.end());
    }
    for (auto& streamer : col_streamers_) {
        auto events = streamer->tick(current_cycle_);
        events_.insert(events_.end(), events.begin(), events.end());
    }

    // 4. Advance clock
    ++current_cycle_;

    return is_complete();
}

bool ConcurrentTimingExecutor::is_complete() const {
    // Complete when all queues are empty and no in-flight work
    for (const auto& engine : dma_engines_) {
        if (!engine->is_idle() || engine->has_pending_work()) return false;
    }
    for (const auto& mover : block_movers_) {
        if (!mover->is_idle() || mover->has_pending_work()) return false;
    }
    for (const auto& streamer : row_streamers_) {
        if (!streamer->is_idle() || streamer->has_pending_work()) return false;
    }
    for (const auto& streamer : col_streamers_) {
        if (!streamer->is_idle() || streamer->has_pending_work()) return false;
    }
    return true;
}
```

## Work Assignment Strategies

### DMA Engine Selection

```cpp
uint32_t ConcurrentTimingExecutor::select_dma_engine(
    MatrixID matrix, const TileCoord& tile) const
{
    // Strategy: Round-robin across engines, or hash-based for determinism
    // Could also be: one engine per memory controller, based on address mapping

    // Option 1: Matrix-based (A uses engines 0-1, B uses 2-3)
    if (matrix == MatrixID::A) {
        return tile.ti % (config_.num_dma_engines / 2);
    } else if (matrix == MatrixID::B) {
        return (config_.num_dma_engines / 2) + (tile.tj % (config_.num_dma_engines / 2));
    } else {
        return tile.ti % config_.num_dma_engines;
    }

    // Option 2: Load-balance by queue depth
    // return engine with smallest queue
}
```

### BlockMover Selection

```cpp
uint32_t ConcurrentTimingExecutor::select_block_mover(
    MatrixID matrix, const TileCoord& tile) const
{
    // Strategy: Separate movers for ingress (L3→L2) vs egress (L2→L3)
    // Or: round-robin based on tile position

    if (matrix == MatrixID::C) {
        // C tiles use "egress" movers for writeback
        return (config_.num_block_movers / 2) + (tile.ti % (config_.num_block_movers / 2));
    } else {
        // A/B tiles use "ingress" movers
        return tile.tk % (config_.num_block_movers / 2);
    }
}
```

### Streamer Selection

```cpp
uint32_t ConcurrentTimingExecutor::select_row_streamer(const TileCoord& tile) const {
    // Row streamers feed west edge - assign by tile row
    return tile.ti % config_.num_row_streamers;
}

uint32_t ConcurrentTimingExecutor::select_col_streamer(const TileCoord& tile) const {
    // Col streamers feed north edge - assign by tile column
    return tile.tj % config_.num_col_streamers;
}
```

## Expected Concurrency Patterns

### Tile Pipeline

For a 4x4x4 tiled matmul (64 tiles of A, 64 tiles of B, 16 tiles of C):

```
Cycle   DMA0        DMA1        DMA2        DMA3        BM0         BM1         STR0        STR1
─────   ────        ────        ────        ────        ───         ───         ────        ────
0       A[0,0,0]    A[0,0,1]    B[0,0,0]    B[0,0,1]    -           -           -           -
10      A[0,0,2]    A[0,0,3]    B[0,0,2]    B[0,0,3]    -           -           -           -
18      A[0,1,0]    A[0,1,1]    B[0,1,0]    B[0,1,1]    A[0,0,0]    B[0,0,0]    -           -
26      A[0,1,2]    ...         ...         ...         A[0,0,1]    B[0,0,1]    A[0,0,0]    B[0,0,0]
...
```

### Credit Flow Example

```
Initial state:
  L3 credits: 32 (all buffers free)
  L2 credits: 64 (all banks free)

After 4 DMA loads issued:
  L3 credits: 28 (4 buffers allocated)

After 4 BM moves complete:
  L3 credits: 32 (4 credits returned)
  L2 credits: 60 (4 banks allocated)

After 4 STR feeds complete:
  L2 credits: 64 (4 credits returned)
```

## Trace Output

The concurrent model produces richer traces showing true parallelism:

```json
{"name":"DMA_LOAD","cat":"dma","ph":"X","ts":0,"dur":18,"pid":1,"tid":0,
 "args":{"tile":"A[0,0,0]","bytes":1024}},
{"name":"DMA_LOAD","cat":"dma","ph":"X","ts":0,"dur":18,"pid":1,"tid":1,
 "args":{"tile":"A[0,0,1]","bytes":1024}},
{"name":"DMA_LOAD","cat":"dma","ph":"X","ts":0,"dur":18,"pid":1,"tid":2,
 "args":{"tile":"B[0,0,0]","bytes":1024}},
{"name":"DMA_LOAD","cat":"dma","ph":"X","ts":0,"dur":18,"pid":1,"tid":3,
 "args":{"tile":"B[0,0,1]","bytes":1024}},
{"name":"BM_MOVE","cat":"block_mover","ph":"X","ts":18,"dur":8,"pid":1,"tid":100,
 "args":{"tile":"A[0,0,0]"}},
```

In Perfetto, this shows 4 DMA lanes running in parallel, followed by overlapped
BlockMover and Streamer operations.

## File Organization

```
include/sw/kpu/timing/
├── concurrent_timing_executor.hpp
├── credit_pool.hpp
├── tag_cam.hpp
├── work_queue.hpp
├── dma_engine_process.hpp
├── block_mover_process.hpp
├── streamer_process.hpp
└── tile_descriptor.hpp

src/timing/
├── concurrent_timing_executor.cpp
├── credit_pool.cpp
├── tag_cam.cpp
├── dma_engine_process.cpp
├── block_mover_process.cpp
└── streamer_process.cpp

tests/timing/
├── test_credit_pool.cpp
├── test_tag_cam.cpp
├── test_dma_engine_process.cpp
├── test_block_mover_process.cpp
├── test_streamer_process.cpp
└── test_concurrent_timing_executor.cpp
```

## Implementation Phases

### Phase 1: Core Infrastructure
- CreditPool
- TagCAM
- WorkQueue
- TileDescriptor

### Phase 2: Component Processes
- DMAEngineProcess (single engine first)
- BlockMoverProcess (single mover first)
- StreamerProcess (single streamer first)

### Phase 3: Integration
- ConcurrentTimingExecutor
- Program distribution
- Multi-engine/mover/streamer support

### Phase 4: Validation
- Compare results against sequential model
- Verify credit invariants
- Trace visualization validation

## Verification

```bash
# Build
cmake --build --preset release

# Run timing tests
ctest --preset release -R timing

# Compare sequential vs concurrent results
./build/tools/harness/timing-compare \
    --program kernels/asm/matmul_4096x1024x8192.kpuasm \
    --sequential-cycles \
    --concurrent-cycles \
    --speedup

# Export concurrent trace
./build/tools/harness/concurrent-runner \
    --program kernels/asm/matmul_64x64x64.kpuasm \
    --trace /tmp/concurrent_trace.json

# View in Perfetto
# Open https://ui.perfetto.dev and load /tmp/concurrent_trace.json
```

## Performance Expectations

For a properly pipelined matmul with:
- 4 DMA engines
- 4 BlockMovers
- 2 Row streamers, 2 Col streamers
- 32 L3 buffers, 64 L2 banks

Expected speedup over sequential model: **4-8x** depending on:
- Memory bandwidth utilization
- Credit pool sizing (enough to hide latency)
- Work distribution balance

The concurrent model should show:
- DMA channels fully utilized (4 parallel loads)
- BlockMovers overlapped with DMA (pipelining)
- Streamers overlapped with BlockMovers
- Compute overlapped with all of the above
