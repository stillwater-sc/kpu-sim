# LPDDR5X Memory Pipeline: From DRAM to L3 Tile

This document describes the memory pipeline timing from LPDDR5X external memory through the memory controller and DMA engine to the L3 tile cache, with a focus on transferring 64-byte cache lines (512-bit NoC flits).

## Overview

The KPU memory subsystem uses a pipelined architecture to move data from external LPDDR5X memory to the on-chip L3 tile cache. Understanding this pipeline is critical for accurate cycle-level simulation and performance modeling.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   LPDDR5X   │───▶│   Memory    │───▶│    DMA      │───▶│    NoC      │───▶│  L3 Tile    │
│   Channel   │    │ Controller  │    │   Engine    │    │   Router    │    │   Cache     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
   4266 MHz           250 MHz           250 MHz           250 MHz           250 MHz
   (I/O clock)       (MC clock)        (DMA clock)       (NoC clock)       (L3 clock)
```

## LPDDR5X Specifications

### Key Parameters (High-End Configuration)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Data Rate | 8533 MT/s | Megatransfers per second |
| Bus Width | 16 bits (x16) | Per channel |
| Prefetch | 16n | 16-bit prefetch architecture |
| Burst Length | BL16 (fixed) | 16 transfers per burst |
| I/O Clock | 4266 MHz | Half the data rate (DDR) |
| WCK (Write Clock) | 4266 MHz | Strobe clock for data |
| CK (Command Clock) | 533 MHz | 1/8 of I/O clock |

### Timing Parameters

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| CAS Latency | CL | ~22 CK cycles | Column address strobe latency |
| RAS to CAS | tRCD | ~18 CK cycles | Row to column delay |
| Row Precharge | tRP | ~18 CK cycles | Precharge time |
| Row Active Time | tRAS | ~42 CK cycles | Minimum row active |
| Refresh Period | tREFI | 3.9 μs | Refresh interval |

### Per-Burst Data Transfer

For a single LPDDR5X x16 channel with BL16:

```
Data per burst = 16 bits × 16 transfers = 256 bits = 32 bytes
Burst duration = 16 transfers / 8533 MT/s = 1.875 ns
```

## Fetching 64 Bytes (512-bit NoC Flit)

To transfer a 64-byte cache line from LPDDR5X to the L3 tile, we need 2 bursts from a single channel, or 1 burst each from 2 channels operating in parallel.

### Option A: Single Channel, Sequential Bursts

```
Time    │ LPDDR5X Channel
────────┼─────────────────────────────────────
0.0 ns  │ ◀──── Burst 1 (32B) ────▶
1.9 ns  │ ◀──── Burst 2 (32B) ────▶
3.8 ns  │ Complete: 64 bytes transferred
```

**Total: ~3.8 ns** (plus command overhead for non-sequential access)

### Option B: Dual Channel, Parallel Bursts (Interleaved)

```
Time    │ Channel 0          │ Channel 1
────────┼────────────────────┼────────────────────
0.0 ns  │ ◀─ Burst (32B) ─▶  │ ◀─ Burst (32B) ─▶
1.9 ns  │ Complete           │ Complete
        │      64 bytes total transferred
```

**Total: ~1.9 ns** for the data transfer phase

## Memory Controller Domain

The memory controller operates at 250-500 MHz, significantly slower than the LPDDR5X I/O interface. The controller's job is to:

1. Schedule commands to the DRAM (respecting timing constraints)
2. Assemble incoming burst data into 64-byte cache lines
3. Hand off complete cache lines to the DMA engine

### Clock Domain Summary

| Domain | Frequency | Cycle Time | Bus Width | Effective BW |
|--------|-----------|------------|-----------|--------------|
| LPDDR5X I/O | 4266 MHz | 0.234 ns | 16 bits | 17 GB/s/ch |
| Command (CK) | 533 MHz | 1.88 ns | - | - |
| Memory Controller | 250 MHz | 4.0 ns | 64 bytes | 16 GB/s |
| DMA Engine | 250 MHz | 4.0 ns | 64 bytes | 16 GB/s |
| NoC | 250 MHz | 4.0 ns | 64 bytes | 16 GB/s |
| L3 Tile | 250 MHz | 4.0 ns | 64 bytes | 16 GB/s |

### Burst Assembly in Memory Controller

The memory controller receives data from LPDDR5X at the I/O rate and assembles it into 64-byte cache lines:

```
MC Cycle │ LPDDR5X Data In    │ Assembler State     │ Output
─────────┼────────────────────┼─────────────────────┼────────────
    0    │ Burst 1 starting   │ Accumulating        │ -
    1    │ Burst 1 + 2 data   │ 64B assembled       │ 512-bit flit ready
```

For streaming (consecutive) access with row buffer hits:
- **1 MC cycle per 64B** (pipelined)

For random access (row misses):
- **~10-15 MC cycles** for first access (CAS latency)
- Then **1 MC cycle per 64B** while same row is open

## DMA Engine → NoC → L3 Pipeline

Once the memory controller has assembled a 64-byte cache line, it enters the DMA/NoC pipeline:

```
MC Cycle │ Memory Controller │ DMA Engine  │ NoC Router │ L3 Tile
─────────┼───────────────────┼─────────────┼────────────┼──────────
    0    │ 64B ready         │ -           │ -          │ -
    1    │ -                 │ Receive     │ -          │ -
    2    │ -                 │ -           │ Route      │ -
    3    │ -                 │ -           │ -          │ Write
```

### Pipeline Latency (Single 64B Transfer)

| Stage | Latency | Cumulative |
|-------|---------|------------|
| DRAM Access (CAS) | ~10 MC cycles | 10 |
| Burst Assembly | ~1 MC cycle | 11 |
| DMA Receive | 1 MC cycle | 12 |
| NoC Routing | 1 MC cycle | 13 |
| L3 Write | 1 MC cycle | 14 |
| **Total (first access)** | **~14 MC cycles** | **~56 ns** |

### Throughput (Streaming Access)

Once the pipeline is full with sequential accesses (row buffer hits):

| Metric | Value |
|--------|-------|
| Throughput | 1 flit per MC cycle |
| Bytes per cycle | 64 bytes |
| Bandwidth | 64 B × 250 MHz = **16 GB/s per channel** |

## Transferring a Compute Tile

For a 16×16 systolic array with 1-byte elements:

```
Tile size = 16 × 16 × 1 byte = 256 bytes = 4 × 64-byte flits
```

### Timing Analysis

**Streaming case (row buffer hits):**
```
Flit 0: Cycle 0-13 (first access, includes CAS latency)
Flit 1: Cycle 14 (pipelined)
Flit 2: Cycle 15 (pipelined)
Flit 3: Cycle 16 (pipelined)

Total: 17 MC cycles ≈ 68 ns
```

**Simplified model (steady-state throughput):**
```
256 bytes / 64 bytes per cycle = 4 MC cycles
```

The simulator uses the simplified model for throughput calculations, adding latency terms separately when needed.

### With 4-Byte Elements (float32)

```
Tile size = 16 × 16 × 4 bytes = 1024 bytes = 16 × 64-byte flits

Transfer time = 16 MC cycles = 64 ns (steady-state)
              = ~30 MC cycles with initial latency
```

## Multi-Channel Bandwidth

The KPU typically has 4 memory channels for aggregate bandwidth:

| Configuration | Channels | Per-Channel BW | Aggregate BW |
|---------------|----------|----------------|--------------|
| Minimum | 2 | 16 GB/s | 32 GB/s |
| **Default** | **4** | **16 GB/s** | **64 GB/s** |
| Maximum | 8 | 16 GB/s | 128 GB/s |

### Tile Layout for Channel Utilization

The tile layout policy determines which tiles map to which channels:

```
Policy: MATRIX_PARTITIONED
  - A tiles → Channels 0, 1
  - B tiles → Channels 2, 3
  - Parallel A+B loading: 2 × 16 GB/s = 32 GB/s effective

Policy: ITERATION_AWARE
  - A tiles → Even channels (0, 2)
  - B tiles → Odd channels (1, 3)
  - Zero conflicts, maximum parallelism
```

## Simulator Implementation

The simulator models this pipeline at the memory controller clock level:

```cpp
// HardwareResource::schedule_op()
Cycle cycles = (bytes + bus_width_bytes - 1) / bus_width_bytes;
// For 256 bytes with 64-byte bus: ceil(256/64) = 4 cycles
```

### ResourceConfig Defaults

```cpp
struct ResourceConfig {
    // DMA/MC domain
    double dma_clock_mhz = 250.0;           // MC clock frequency
    Size dma_bus_width_bytes = 64;          // 512-bit flit size
    double dma_bandwidth_gb_s = 16.0;       // Per channel: 64B × 250MHz

    // Multiple channels
    uint8_t num_memory_channels = 4;        // 4 × 16 GB/s = 64 GB/s total
};
```

## Summary: 64-Byte Transfer Timeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        64-Byte Cache Line Transfer                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LPDDR5X        Memory          DMA           NoC           L3              │
│  Channel        Controller      Engine        Router        Tile            │
│     │              │              │             │             │              │
│     │──BL16 (32B)──▶              │             │             │              │
│     │  (1.9 ns)    │              │             │             │              │
│     │──BL16 (32B)──▶              │             │             │              │
│     │  (1.9 ns)    │              │             │             │              │
│     │              │──assemble────▶             │             │              │
│     │              │  (1 cycle)   │             │             │              │
│     │              │              │──512b flit──▶             │              │
│     │              │              │  (1 cycle)  │             │              │
│     │              │              │             │───write────▶│              │
│     │              │              │             │  (1 cycle)  │              │
│     │              │              │             │             │              │
├──────────────────────────────────────────────────────────────────────────────┤
│  Timeline (streaming, row buffer hit):                                       │
│    DRAM burst:     ~4 ns                                                     │
│    MC assemble:    4 ns (1 MC cycle)                                         │
│    DMA + NoC + L3: 12 ns (3 MC cycles)                                       │
│    Total:          ~20 ns per 64B (pipelined: 4 ns per 64B sustained)        │
│                                                                              │
│  Timeline (random access, row miss):                                         │
│    CAS latency:    ~40 ns (10 MC cycles)                                     │
│    DRAM burst:     ~4 ns                                                     │
│    Pipeline:       12 ns                                                     │
│    Total:          ~56 ns first access                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

## References

- JEDEC LPDDR5X Standard (JESD209-5B)
- KPU Architecture Specification (`docs/STILLWATER_KPU_SPECIFICATION.md`)
- Memory Interleaving Design (`docs/MEMORY_INTERLEAVING_DESIGN.md`)
- Tile Layout Policies (`include/sw/kpu/isa/tile_layout.hpp`)
