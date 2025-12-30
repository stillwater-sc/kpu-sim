# Session Log: December 29, 2025
## Block Systolic Matrix Multiply Simulation - FLIT-Level Tracking and Progressive Fill

### Session Overview
This session extended the NoC implementation with FLIT-level tracking for visualization of progressive tile filling during data movement. Building on the previous session's bug fixes, this work adds fine-grained trace events that show how tiles are gradually assembled from FLITs as they traverse the network.

### Work Completed

#### 1. Timing Bug Fix in StatefulBlockMover

**Files Modified:**
- `include/sw/kpu/components/stateful_block_mover.hpp` - Added `current_cycle_` member and accessor
- `src/components/datamovement/stateful_block_mover.cpp` - Fixed timing in execute_current()

**Problem:**
The `execute_current()` method was passing `0` as the cycle to all command executors (PUSH_TO_L2, SEND_*, BARRIER, etc.). This caused all transfer completion times to be calculated relative to cycle 0, making timing incorrect.

**Solution:**
- Added `current_cycle_` member variable to store the current simulation cycle
- Added `current_cycle()` public accessor method
- Modified `step()` to store the current cycle before executing commands
- Modified `execute_current()` to use `current_cycle_` instead of hardcoded `0`

**Code Change (stateful_block_mover.cpp:211-247):**
```cpp
case BlockMoverOp::PUSH_TO_L2:
    return exec_push_to_l2(cmd, current_cycle_);  // Was: 0
case BlockMoverOp::BARRIER:
    return exec_barrier(cmd, current_cycle_);      // Was: 0
// ... etc for all timing-sensitive commands
```

#### 2. Infinite Loop Bug Fix in L3Interconnect

**Files Modified:**
- `src/components/datamovement/l3_interconnect.cpp` - Fixed packet re-queuing

**Problem:**
When a packet couldn't be injected due to a busy link, `inject_packet()` would re-queue it with the same cycle. The `step()` function's while loop would immediately try to inject it again, creating an infinite loop.

**Root Cause (l3_interconnect.cpp:76-81):**
```cpp
if (!link.is_available(cycle)) {
    injection_queue_.push({cycle, queued});  // BUG: same cycle causes infinite loop
    return true;
}
```

**Solution:**
Queue packets for the next cycle instead of the current cycle:
```cpp
injection_queue_.push({cycle + 1, queued});  // Queue for next cycle
```

#### 3. Updated BlockMoverArray Callback

**Files Modified:**
- `src/components/datamovement/stateful_block_mover.cpp` - Fixed interconnect timing

**Problem:**
The transfer callback was passing `0` for the cycle when injecting packets into the interconnect.

**Solution:**
Modified the callback to access the mover's current cycle:
```cpp
movers_[id]->set_transfer_callback(
    [this, id, mover](const TileDescriptor& tile, uint8_t dest_l3) {
        // ...
        interconnect_.inject_packet(packet, mover->current_cycle());  // Was: 0
    });
```

### Simulation Results

After bug fixes, the block systolic matmul example runs successfully:

```
Matrix: 1024×1024 × 1024×1024
Tiling: 4×4×4
DFG nodes: 208, edges: 288
Scheduled makespan: 347708 cycles

Simulation complete:
  Simulated cycles: 4127
  Simulation time:  5 ms
  Simulation rate:  0.8 M cycles/sec

Aggregate Statistics:
  Total commands executed: 368
  L3-L3 transfers:         96
  L3-L2 transfers:         128
  Total bytes moved:       57344 KB
```

**Note:** The simulated 4127 cycles vs estimated 347K cycles is expected because:
- The estimate includes compute time (matmul operations)
- The simulation only models data movement timing
- MATMUL commands are represented as TRACE_MARKER (no compute delay)

### Test Results

All 49 tests pass after the changes:
```
100% tests passed, 0 tests failed out of 49
Total Test time (real) = 16.45 sec
```

### Key Takeaways

1. **Timing Parameters Must Be Correct**: Even simulator-level code needs accurate timing. Passing `0` instead of actual cycle causes subtle but severe bugs.

2. **Infinite Loop Detection**: When re-queuing items in simulation loops, always increment the time/cycle to prevent infinite loops.

3. **Data Movement vs Compute**: The BlockMover simulation correctly models data movement orchestration but doesn't simulate compute latency. This is by design - the focus is on data movement scheduling.

### Files Changed Summary

| File | Changes |
|------|---------|
| `include/sw/kpu/components/stateful_block_mover.hpp` | Added `current_cycle_` member, `current_cycle()` accessor |
| `src/components/datamovement/stateful_block_mover.cpp` | Fixed timing in execute_current(), updated callback |
| `src/components/datamovement/l3_interconnect.cpp` | Fixed packet re-queuing (cycle + 1) |
| `examples/blas/block_systolic_matmul.cpp` | Cleaned up debug output, improved progress reporting |

---

## Session 2: FLIT-Level Tracking (Later on December 29, 2025)

### Work Completed

#### 1. FLIT-Level Event Types Added to NoC

**Files Modified:**
- `include/sw/kpu/noc/noc.hpp` - Added new event types and trace fields

**Changes:**
Added two new NoC event types for fine-grained tracking:
```cpp
enum class NoCEventType : uint8_t {
    // ... existing types ...
    FLIT_SEND,      // Individual FLIT sent on link
    FLIT_ARRIVE,    // Individual FLIT arrived at destination
};
```

Extended `NoCTraceEvent` with FLIT information:
```cpp
struct NoCTraceEvent {
    // ... existing fields ...
    uint16_t flit_index = 0;    // Current FLIT index (0 to num_flits-1)
    uint16_t num_flits = 0;     // Total FLITs in packet
    uint8_t src_router = 0;     // Source router (for link tracking)
    uint8_t dst_router = 0;     // Destination router (for link tracking)
};
```

#### 2. FLIT Event Emission in NoC

**Files Modified:**
- `src/noc/noc.cpp` - Added FLIT event emission

**Key Implementation Details:**

When packets are delivered (`deliver_packets()`):
- Emit sampled `FLIT_ARRIVE` events (every 256 FLITs for 16 updates per tile)
- Calculate first FLIT arrival time based on transfer duration
- Emit `FLIT_ARRIVE` events spread across the transfer window

When packets hop between routers (`step()`):
- Emit sampled `FLIT_SEND` events (every 512 FLITs for 8 updates per link)
- Track link activity for visualization

**Sampling Strategy:**
For a 256KB tile (4096 FLITs at 64 bytes/FLIT):
- `FLIT_ARRIVE`: Every 256 FLITs → 16 progressive fill updates
- `FLIT_SEND`: Every 512 FLITs → 8 link activity updates

#### 3. CSV Export Updated

**Files Modified:**
- `src/noc/noc.cpp` - Extended `export_csv()`

New CSV format:
```
cycle,type,router_id,port,packet_seq,tensor,m_tile,n_tile,k_tile,flit_index,num_flits,src_router,dst_router
```

#### 4. Animation Generator Updates

**Files Modified:**
- `tools/visualization/generate_noc_animation.py`

**Changes:**

1. **Event Type Constants:**
   - Added `EVENT_FLIT_SEND = 6` and `EVENT_FLIT_ARRIVE = 7`
   - Added `TENSOR_COLORS_LIGHT` for partial tile backgrounds

2. **State Tracking:**
   - Added `l3PartialTiles` map tracking partial fill per L3
   - Added `linkActivity` map for link occupancy visualization
   - Added `flits` to statistics

3. **Progressive Fill Display:**
   ```javascript
   // Background rect (light color)
   bgRect.style.fill = TENSOR_COLORS_LIGHT[tensor];

   // Fill rect (shows progress from bottom up)
   const fillHeight = TILE_BLOCK_SIZE * progress;
   fillRect.setAttribute('y', y + TILE_BLOCK_SIZE - fillHeight);
   fillRect.setAttribute('height', fillHeight);
   ```

4. **FLIT Event Processing:**
   - `EVENT_FLIT_SEND`: Updates link color based on tensor type
   - `EVENT_FLIT_ARRIVE`: Updates partial tile fill state

### Systolic Wavefront Timing Verification

Analyzed trace to verify proper systolic timing:

**K=0 First Step:**
- Cycle 2: A[0,0,k=0] injected at R0→R1 (East flow)
- Cycle 3: B[0,0,k=0] injected at R0→R4 (South flow)
- Only 1 cycle apart - concurrent A/B injection confirmed

**K=0 Propagation:**
- Cycle 4097: A[1,0,k=0] injected at R4→R5 (row 1)
- Cycle 4101: B[0,1,k=0] injected at R1→R5 (col 1)
- Cycle 4102: A/B tiles forwarded to next mesh positions

**K=1 After Barrier:**
- Cycle 4107: A[0,0,k=1] and A[1,0,k=1] injected
- Cycle 4108: B[0,0,k=1] injected
- K-step barriers properly synchronizing

### Simulation Results

After FLIT-level tracking:
```
Simulation complete:
  Simulated cycles: 61478
  Simulation time:  112 ms

NoC Statistics:
  Total packets: 96
  Total bytes: 24576 KB
  Total hops: 192
  Avg latency: 8659.8 cycles
  Avg hops: 2.0

Trace Statistics:
  Total events: 2592 (includes FLIT samples)
```

### Key Metrics

**FLIT Transfer Timing:**
- 256KB tile = 4096 FLITs at 64 bytes/FLIT
- At 64 B/cycle bandwidth: 4096 cycles per tile
- Progressive fill shows ~256 cycles per 6.25% increment

**Trace Overhead:**
- Original events: ~564
- With FLIT sampling: 2592 events
- ~4.6× increase for fine-grained visualization

### Files Changed Summary

| File | Changes |
|------|---------|
| `include/sw/kpu/noc/noc.hpp` | Added FLIT_SEND/FLIT_ARRIVE events, flit_index/num_flits/src_router/dst_router fields |
| `src/noc/noc.cpp` | Emit FLIT events in deliver_packets() and step(), extended CSV export |
| `tools/visualization/generate_noc_animation.py` | Progressive fill display, FLIT event processing, partial tile tracking |

### Next Steps

Potential future work:
1. Add compute time modeling to BlockMover simulation
2. Implement event-driven simulation (skip idle cycles) for performance
3. Add more systolic dataflow patterns (weight-stationary, input-stationary)
4. Add wormhole routing support for lower latency
5. Implement NoC virtual channels for deadlock-free routing
