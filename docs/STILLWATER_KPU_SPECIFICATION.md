# Stillwater Knowledge Processing Unit(TM) (KPU) Specification

**Version**: 1.0
**Date**: December 2025
**Status**: Draft

---

## 1. Overview

The Stillwater Knowledge Processing Unit (KPU) is a Domain Flow Architecture processor designed for high-performance tensor operations. Unlike traditional stored-program machines where a program counter sequences through instructions, the KPU implements a data-driven execution model where:

- **The program IS the data movement schedule** - derived from SURE (Space-time Uniform Recurrence Equation) analysis
- **The compute fabric is reactive** - Processing Elements (PEs) execute when data tokens arrive
- **Intelligence is in data orchestration** - optimal system-level schedules minimize memory traffic

This architecture is particularly suited for real-time signal processing and control, sensor fusion, linear algebra, and constraint solving workloads where regular, predictable data access patterns can be exploited for maximum efficiency.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HOST SYSTEM                                    │
│                         (DDR4/DDR5 Memory)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                              PCIe Interface
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                             STILLWATER KPU                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         EXTERNAL MEMORY                               │  │
│  │                (DDR5 / LPDDR5/6 / GDDR6 / HBM2/3/4)                   │  │
│  │                        Technology Agnostic                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                               DMA Engines                                   │
│                                    │                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                  Distributed L3 block-oriented Scratchpad             │  │
│  │                       (Software-Managed Cache Tiles)                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                               Block Movers                                  │
│                                    │                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          L2 BANK CACHE                                │  │
│  │                      (Software-Managed Banks)                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                Streamers                                    │
│                                    │                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        L1 STREAMING BUFFERS                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          COMPUTE FABRIC                               │  │
│  │                        (Systolic PE Array)                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Memory Hierarchy

### 3.1 External Memory

The KPU external memory interface is **technology-agnostic**, supporting various memory technologies without ISA changes.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Bank Count | 2 | 1-8 | Independent memory channels |
| Bank Capacity | 1 GB | 512 MB - 16 GB | Per bank |
| Bandwidth | 100 GB/s | 50-819 GB/s | Depends on technology |

**Supported Technologies**:
- DDR4 (50-100 GB/s)
- DDR5 (100-200 GB/s)
- LPDDR5 (50-200 GB/s)
- GDDR6 (100-200 GB/s)
- HBM2/3/4 (400-819 GB/s)

### 3.2 L3 Tile Cache

Software-managed cache organized as independent tiles, typically arranged in a checkerboard pattern for parallel access.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Tile Count | 32 | 2-512 | Independent tiles |
| Tile Capacity | 128 KB | 64-512 KB | Per tile |
| Total Capacity | 512 KB | 128 KB - 4 MB | Tile count × capacity |
| Access Pattern | 2D Block | - | Matrix-oriented access |

### 3.3 L2 Bank Cache

Software-managed cache organized as independent banks for high-bandwidth parallel access.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Bank Count | 8 | 4-16 | Independent banks |
| Bank Capacity | 64 KB | 32-256 KB | Per bank |
| Total Capacity | 512 KB | 128 KB - 4 MB | Bank count × capacity |
| Cache Line Size | 64 bytes | 32-128 bytes | Configurable |

### 3.4 L1 Streaming Buffers

High-bandwidth buffers that feed the systolic array with minimal latency.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Buffer Count | 4 | 2-8 | Typically: 2 for A, 1 for B, 1 for C |
| Buffer Capacity | 32 KB | 16-64 KB | Per buffer |
| Total Capacity | 128 KB | 32-512 KB | Buffer count × capacity |
| Bandwidth | 1000+ GB/s | - | On-chip, near-PE |

### 3.5 Scratchpad Memory

Additional software-managed memory for data reshuffling and staging.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Scratchpad Count | 2 | 1-4 | Independent scratchpads |
| Scratchpad Capacity | 64 KB | 32-128 KB | Per scratchpad |

---

## 4. Data Movement Architecture

### 4.1 DMA Engines

Transfer data between external memory and L3 tiles.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Engine Count | 8 | 1-32 | Independent engines, one per memory channel |
| Bandwidth | 100 GB/s | 50-200 GB/s | Per engine |
| Address Mode | Physical | - | Unified address space |

**Capabilities**:
- Contiguous block transfers
- Completion callbacks for synchronization
- Multi-cycle operation with bandwidth-aware timing

### 4.2 Block Movers

Transfer data between L3 tiles and L2 banks with optional transformations.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Mover Count | 4 | 2-8 | Independent movers |
| Bandwidth | 100 GB/s | 50-200 GB/s | Per mover |

**Transformation Types**:
| Transform | Description | Use Case |
|-----------|-------------|----------|
| IDENTITY | Direct copy | Standard tile movement |
| TRANSPOSE | Row ↔ Column swap | B matrix preparation |
| BLOCK_RESHAPE | Tiling pattern change | Format conversion |
| SHUFFLE | Custom permutation | Specialized layouts |

**Transfer Specification**:
- Source: L3 tile ID + offset
- Destination: L2 bank ID + offset
- Geometry: height × width × element_size
- Transform: One of the above types

### 4.3 Streamers

Feed data from L2 banks to L1 buffers that autonomously drive streams of elements into the fabric with systolic array timing.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Streamer Count | 8 | 4-16 | Independent streamers, one per L2 bank |
| Bandwidth | 100 GB/s | 50-200 GB/s | Per streamer |

**Stream Types**:
| Type | Direction | Description |
|------|-----------|-------------|
| ROW_STREAM | L2 → L1 | Move rows for A matrix |
| COLUMN_STREAM | L2 → L1 | Move columns for B matrix |
| DRAIN_OUTPUT | L1 → L2 | Drain C matrix results |

**Systolic Array Integration**:
- Automatic staggering for proper PE alignment
- Per-row and per-column offset tracking
- Cache line buffering for efficient L2 access

---

## 5. Compute Fabric

### 5.1 Systolic Array

The compute fabric consists of a 2D array of Processing Elements (PEs) implementing output-stationary dataflow.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Array Rows | 16 | 8-64 | PE rows |
| Array Columns | 16 | 8-64 | PE columns |
| Total PEs | 256 | 64-4096 | Rows × Columns |
| Peak Throughput | 256 MACs/cycle | - | At full utilization |

### 5.2 Processing Element (PE)

Each PE implements a MAC (Multiply-Accumulate) unit with output-stationary dataflow.

```
                    B input (from above)
                         │
                         ▼
              ┌─────────────────────┐
  A input ───►│                     │───► A output (to right)
  (from left) │    c += a × b       │
              │                     │
              └─────────────────────┘
                         │
                         ▼
                    B output (to below)
```

**PE Registers**:
| Register | Width | Description |
|----------|-------|-------------|
| a_input | 32 bits | A operand input |
| b_input | 32 bits | B operand input |
| c_accumulator | 32 bits | Output accumulator |
| a_output | 32 bits | A propagation |
| b_output | 32 bits | B propagation |

**PE Operation (per cycle)**:
```
1. c_accumulator += a_input × b_input
2. a_output ← a_input (propagate horizontally)
3. b_output ← b_input (propagate vertically)
4. Clear inputs for next cycle
```

### 5.3 Data Buses

| Bus | Direction | Purpose |
|-----|-----------|---------|
| Horizontal | Left → Right | A matrix row data |
| Vertical | Top → Bottom | B matrix column data |
| Diagonal | PE → Output | C matrix evacuation |

### 5.4 Dataflow Strategies

The KPU supports three dataflow strategies, selected at compile time based on tensor shapes:

#### Output-Stationary (Default)
- **C tiles stay in PE accumulators** throughout K-reduction
- A rows stream horizontally, B columns stream vertically
- Best for: Balanced workloads, large K dimension
- Advantage: No C writeback during accumulation

#### Weight-Stationary
- **B tiles (weights) stay in PE registers**
- A streams through, C accumulates in L2
- Best for: Large batch (M), weight reuse critical
- Advantage: Maximizes B reuse

#### Input-Stationary
- **A tiles (inputs) stay in PE registers**
- B streams through, C accumulates in L2
- Best for: Large output (N), input reuse critical
- Advantage: Maximizes A reuse

---

## 6. Data Movement ISA

### 6.1 Philosophy

In Domain Flow Architecture, the **Data Movement ISA is the primary programming interface**. The compute fabric is reactive and requires no explicit compute instructions - it executes when properly formatted data arrives.

### 6.2 Instruction Categories

#### DMA Operations (External Memory ↔ L3)
| Opcode | Description |
|--------|-------------|
| DMA_LOAD_TILE | Load tile from external memory to L3 |
| DMA_STORE_TILE | Store tile from L3 to external memory |
| DMA_PREFETCH_TILE | Prefetch tile (non-blocking hint) |

#### Block Mover Operations (L3 ↔ L2)
| Opcode | Description |
|--------|-------------|
| BM_MOVE_TILE | Move tile L3 → L2 (identity) |
| BM_TRANSPOSE_TILE | Move tile L3 → L2 with transpose |
| BM_WRITEBACK_TILE | Move tile L2 → L3 |
| BM_RESHAPE_TILE | Move with reshape transformation |

#### Streamer Operations (L2 ↔ L1)
| Opcode | Description |
|--------|-------------|
| STR_FEED_ROWS | Stream rows to systolic array (A matrix) |
| STR_FEED_COLS | Stream columns to systolic array (B matrix) |
| STR_DRAIN_OUTPUT | Drain output from systolic array (C matrix) |
| STR_BROADCAST_ROW | Broadcast row to all PE columns |
| STR_BROADCAST_COL | Broadcast column to all PE rows |

#### Synchronization Operations
| Opcode | Description |
|--------|-------------|
| BARRIER | Wait for all pending operations |
| WAIT_DMA | Wait for specific DMA completion |
| WAIT_BM | Wait for specific BlockMover completion |
| WAIT_STR | Wait for specific Streamer completion |
| SIGNAL | Signal completion token |

#### Control Operations
| Opcode | Description |
|--------|-------------|
| SET_TILE_SIZE | Configure tile dimensions |
| SET_BUFFER | Configure buffer selection |
| SET_STRIDE | Configure address strides |
| LOOP_BEGIN | Start hardware loop |
| LOOP_END | End hardware loop |
| NOP | No operation |
| HALT | End of program |

### 6.3 Instruction Format

Each instruction contains:
- **Opcode**: Operation type (8 bits)
- **Operands**: Operation-specific parameters
- **Timing hints**: Earliest/deadline cycles from SURE analysis
- **Dependencies**: Prerequisite instruction IDs
- **Debug info**: Human-readable label

### 6.4 Example: Output-Stationary MatMul

For C[M,N] = A[M,K] × B[K,N] with tiles Ti×Tj×Tk:

```
for ti = 0 to M/Ti:              // Output row tiles
  for tj = 0 to N/Tj:            // Output col tiles
    for tk = 0 to K/Tk:          // Reduction tiles
      DMA_LOAD A_tile[ti,tk]
      DMA_LOAD B_tile[tk,tj]
      BARRIER
      BM_MOVE A_tile[ti,tk] L3→L2
      BM_MOVE B_tile[tk,tj] L3→L2
      BARRIER
      STR_FEED_ROWS A_tile[ti,tk]
      STR_FEED_COLS B_tile[tk,tj]
      BARRIER
      // Compute happens reactively in PEs
    STR_DRAIN C_tile[ti,tj]
    BARRIER
    DMA_STORE C_tile[ti,tj]
    BARRIER
```

---

## 7. Programming Model

### 7.1 Compilation Flow

```
┌─────────────────┐
│  High-Level IR  │  (PyTorch, TensorFlow, ONNX)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SURE Analysis  │  Derive optimal space-time mapping
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tile Optimizer  │  Select Ti, Tj, Tk for dataflow strategy
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Schedule Builder │  Generate Data Movement ISA program
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DM Program    │  Executable data movement schedule
└─────────────────┘
```

### 7.2 Tile Optimization

The compiler selects tile sizes based on:

1. **Memory Constraints**:
   - L2 footprint: A_tile + B_tile ≤ L2 capacity
   - L1 streaming: Tiles fit in L1 buffers

2. **Dataflow Strategy**:
   - Output-stationary: Maximize K accumulation
   - Weight-stationary: Maximize B reuse
   - Input-stationary: Maximize A reuse

3. **Arithmetic Intensity**:
   - AI = (2 × Ti × Tj × Tk) / (Ti×Tk + Tk×Tj) × element_size
   - Higher AI = better compute/memory ratio

### 7.3 Performance Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Arithmetic Intensity | FLOPs / DRAM bytes | > memory bandwidth / peak compute |
| PE Utilization | Active PEs / Total PEs | > 80% |
| Memory Efficiency | Minimum traffic / Actual traffic | > 70% |

---

## 8. System Configurations

### 8.1 Reference Configurations

#### Minimal (Edge AI)
```
External Memory: 1 × 512 MB @ 68 GB/s (LPDDR5)
L3 Cache: 2 × 256 KB
L2 Cache: 4 × 128 KB
L1 Buffers: 2 × 32 KB
Compute: 1 × 8×8 systolic array
Data Movement: 1 DMA, 2 BlockMovers, 4 Streamers
Peak: 64 MACs/cycle
```

#### Standard (Workstation)
```
External Memory: 2 × 1 GB @ 100 GB/s (GDDR6)
L3 Cache: 4 × 128 KB
L2 Cache: 8 × 64 KB
L1 Buffers: 4 × 32 KB
Compute: 2 × 16×16 systolic arrays
Data Movement: 2 DMAs, 4 BlockMovers, 8 Streamers
Peak: 512 MACs/cycle
```

#### Datacenter (HBM)
```
External Memory: 4 × 8 GB @ 819 GB/s (HBM3)
L3 Cache: 8 × 512 KB
L2 Cache: 16 × 256 KB
L1 Buffers: 8 × 64 KB
Compute: 4 × 32×32 systolic arrays
Data Movement: 4 DMAs, 8 BlockMovers, 16 Streamers
Peak: 4096 MACs/cycle
```

### 8.2 Unified Address Space

All memory levels are mapped into a single address space:

```
0x0000_0000 - Host memory regions
0x1000_0000 - External memory banks
0x2000_0000 - L3 tiles
0x3000_0000 - L2 banks
0x4000_0000 - L1 buffers
0x5000_0000 - Scratchpads
```

Base addresses are configurable for custom memory layouts.

---

## 9. Tracing and Debugging

### 9.1 Trace Logger

All components support cycle-accurate tracing:
- Transaction IDs for correlation
- Start/end cycle timestamps
- Data movement sizes and addresses
- Completion callbacks

### 9.2 Performance Counters

| Counter | Description |
|---------|-------------|
| dma_bytes_transferred | Total DMA traffic |
| l3_bytes_transferred | Total L3 traffic |
| l2_bytes_transferred | Total L2 traffic |
| compute_cycles | Active compute cycles |
| stall_cycles | Pipeline stalls |
| pe_utilization | Average PE activity |

---

## 10. Future Extensions

### 10.1 Planned Features
- Sparsity support in systolic array
- Mixed-precision (INT8/FP16/BF16) operations
- Multi-chip interconnect
- Hardware loop controllers
- Prefetch engines with prediction

### 10.2 ISA Extensions
- Convolution-specific streaming patterns
- Attention mechanism support
- Reduction operations (softmax, layernorm)
- Element-wise fusion

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| SURE | Space-time Uniform Recurrence Equation - mathematical framework for optimal scheduling |
| PE | Processing Element - single MAC unit in systolic array |
| MAC | Multiply-Accumulate operation |
| Dataflow | Pattern of data movement through compute array |
| Tile | Rectangular submatrix processed as a unit |
| Output-Stationary | Dataflow where output accumulates in place |
| Weight-Stationary | Dataflow where weights remain in place |
| Input-Stationary | Dataflow where inputs remain in place |
| Arithmetic Intensity | Ratio of compute operations to memory accesses |

---

## Appendix B: Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial specification |

---

*Copyright 2025 Stillwater Supercomputing, Inc. All rights reserved.*
