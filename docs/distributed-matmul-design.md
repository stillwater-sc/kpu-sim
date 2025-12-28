# Distributed Matrix Multiplication on 4×4 Checkerboard Architecture

## 1. Problem Statement

Design and implement a distributed matrix multiplication kernel that executes on a Drone AI configuration with 16 compute tiles and 16 L3 tiles arranged in a 4×4 checkerboard pattern.

### Matrix Dimensions

| Matrix | Dimensions | Size | Description |
|--------|------------|------|-------------|
| A | 2048 × 512 | 4 MB | Input activations (M × K) |
| B | 512 × 1024 | 2 MB | Weights (K × N) |
| C | 2048 × 1024 | 8 MB | Output (M × N) |

### Constraints

1. **L3 Capacity**: 8 MB total (16 tiles × 512 KB each)
2. **Working Memory**: A (4 MB) + B (2 MB) = 6 MB in L3, leaving 2 MB for C tiles
3. **Output Destination**: Result matrix C must end up in host memory
4. **Compute Resources**: 16 compute tiles with 24×24 systolic arrays each

## 2. Hardware Configuration

### 4×4 Checkerboard Layout

```
    Col 0    Col 1    Col 2    Col 3    Col 4    Col 5    Col 6    Col 7
   ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
R0 │ L3[0]  │ CT[0]  │ L3[1]  │ CT[1]  │ L3[2]  │ CT[2]  │ L3[3]  │ CT[3]  │
   ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
R1 │ CT[4]  │ L3[4]  │ CT[5]  │ L3[5]  │ CT[6]  │ L3[6]  │ CT[7]  │ L3[7]  │
   ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
R2 │ L3[8]  │ CT[8]  │ L3[9]  │ CT[9]  │ L3[10] │ CT[10] │ L3[11] │ CT[11] │
   ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
R3 │ CT[12] │ L3[12] │ CT[13] │ L3[13] │ CT[14] │ L3[14] │ CT[15] │ L3[15] │
   └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

Legend:
  L3[n] = L3 Cache Tile n (512 KB each)
  CT[n] = Compute Tile n (24×24 systolic array)
```

Each compute tile is adjacent to 2-4 L3 tiles, enabling efficient local data access.

### Hardware Specifications

| Component | Configuration | Total Capacity |
|-----------|--------------|----------------|
| L3 Tiles | 16 × 512 KB | 8 MB |
| L2 Banks | 64 × 32 KB | 2 MB |
| L1 Buffers | 3,072 × 64 KB | 192 MB (streaming) |
| Compute Tiles | 16 × 24×24 PEs | 9,216 MACs/cycle |
| External Memory | 4 × 512 MB | 2 GB |
| Memory Bandwidth | 4 × 25 GB/s | 100 GB/s |

### L3 Memory Allocation

```
L3 Memory Map (8 MB total):
┌─────────────────────────────────────────────────────────────┐
│  L3[0-7]: Matrix A Storage (4 MB)                           │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────────────┐│
│  │ A[0]    │ A[1]    │ A[2]    │ A[3]    │ ...  A[7]       ││
│  │ 512 KB  │ 512 KB  │ 512 KB  │ 512 KB  │                 ││
│  └─────────┴─────────┴─────────┴─────────┴─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  L3[8-11]: Matrix B Storage (2 MB)                          │
│  ┌─────────┬─────────┬─────────┬─────────┐                  │
│  │ B[0]    │ B[1]    │ B[2]    │ B[3]    │                  │
│  │ 512 KB  │ 512 KB  │ 512 KB  │ 512 KB  │                  │
│  └─────────┴─────────┴─────────┴─────────┘                  │
├─────────────────────────────────────────────────────────────┤
│  L3[12-15]: C Working Tiles (2 MB)                          │
│  ┌─────────┬─────────┬─────────┬─────────┐                  │
│  │ C[w,0]  │ C[w,1]  │ C[w,2]  │ C[w,3]  │  (wave w)        │
│  │ 512 KB  │ 512 KB  │ 512 KB  │ 512 KB  │                  │
│  └─────────┴─────────┴─────────┴─────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## 3. Tiling Strategy

### Output Tiling (C Matrix)

The output matrix C (2048 × 1024) is divided into a 4×4 grid of tiles:

```
C Matrix (2048 × 1024) = 4×4 tiles
┌──────────┬──────────┬──────────┬──────────┐
│ C[0,0]   │ C[0,1]   │ C[0,2]   │ C[0,3]   │  Row 0 (M: 0-511)
│ 512×256  │ 512×256  │ 512×256  │ 512×256  │
├──────────┼──────────┼──────────┼──────────┤
│ C[1,0]   │ C[1,1]   │ C[1,2]   │ C[1,3]   │  Row 1 (M: 512-1023)
│ 512×256  │ 512×256  │ 512×256  │ 512×256  │
├──────────┼──────────┼──────────┼──────────┤
│ C[2,0]   │ C[2,1]   │ C[2,2]   │ C[2,3]   │  Row 2 (M: 1024-1535)
│ 512×256  │ 512×256  │ 512×256  │ 512×256  │
├──────────┼──────────┼──────────┼──────────┤
│ C[3,0]   │ C[3,1]   │ C[3,2]   │ C[3,3]   │  Row 3 (M: 1536-2047)
│ 512×256  │ 512×256  │ 512×256  │ 512×256  │
└──────────┴──────────┴──────────┴──────────┘

Tile size: 512 × 256 × 4 bytes = 512 KB
Total tiles: 16
```

### Input Tiling (A and B Matrices)

**Matrix A (2048 × 512)** - Row-partitioned:
```
┌──────────────────────┐
│ A[0]: rows 0-255     │ → used by CT[0-3]
├──────────────────────┤
│ A[1]: rows 256-511   │ → used by CT[0-3]
├──────────────────────┤
│ A[2]: rows 512-767   │ → used by CT[4-7]
├──────────────────────┤
│ A[3]: rows 768-1023  │ → used by CT[4-7]
├──────────────────────┤
│ A[4]: rows 1024-1279 │ → used by CT[8-11]
├──────────────────────┤
│ A[5]: rows 1280-1535 │ → used by CT[8-11]
├──────────────────────┤
│ A[6]: rows 1536-1791 │ → used by CT[12-15]
├──────────────────────┤
│ A[7]: rows 1792-2047 │ → used by CT[12-15]
└──────────────────────┘
Each tile: 256 × 512 × 4 = 512 KB
```

**Matrix B (512 × 1024)** - Column-partitioned:
```
┌────────────┬──────────────┬──────────────┬───────────────┐
│    B[0]    │      B[1]    │      B[2]    │      B[3]     │
│ cols 0-255 │ cols 256-511 │ cols 512-767 │ cols 768-1023 │
└────────────┴──────────────┴──────────────┴───────────────┘
Each tile: 512 × 256 × 4 = 512 KB
```

### Compute Tile Assignment

Each compute tile is responsible for computing specific C tiles:

| Compute Tile | C Tiles | A Tiles Needed | B Tiles Needed |
|--------------|---------|----------------|----------------|
| CT[0] | C[0,0] | A[0,1] | B[0] |
| CT[1] | C[0,1] | A[0,1] | B[1] |
| CT[2] | C[0,2] | A[0,1] | B[2] |
| CT[3] | C[0,3] | A[0,1] | B[3] |
| CT[4] | C[1,0] | A[2,3] | B[0] |
| CT[5] | C[1,1] | A[2,3] | B[1] |
| CT[6] | C[1,2] | A[2,3] | B[2] |
| CT[7] | C[1,3] | A[2,3] | B[3] |
| CT[8] | C[2,0] | A[4,5] | B[0] |
| CT[9] | C[2,1] | A[4,5] | B[1] |
| CT[10] | C[2,2] | A[4,5] | B[2] |
| CT[11] | C[2,3] | A[4,5] | B[3] |
| CT[12] | C[3,0] | A[6,7] | B[0] |
| CT[13] | C[3,1] | A[6,7] | B[1] |
| CT[14] | C[3,2] | A[6,7] | B[2] |
| CT[15] | C[3,3] | A[6,7] | B[3] |

## 4. Execution Model

### Wave-Based Execution

Since we can only buffer 4 C tiles (2 MB) at a time but have 16 C tiles total, we use **wave-based execution** with streaming:

```
Wave 0: CT[0-3]   compute C[0,0], C[0,1], C[0,2], C[0,3] → stream to host
Wave 1: CT[4-7]   compute C[1,0], C[1,1], C[1,2], C[1,3] → stream to host
Wave 2: CT[8-11]  compute C[2,0], C[2,1], C[2,2], C[2,3] → stream to host
Wave 3: CT[12-15] compute C[3,0], C[3,1], C[3,2], C[3,3] → stream to host
```

**Alternative: Pipelined Execution**

For higher utilization, overlap compute with data movement:

```
Time →
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Load A,B to L3                                                │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────┤
│ Wave 0   │ Wave 0   │ Wave 1   │ Wave 1   │ Wave 2   │ Wave 2   │ ...  │
│ Compute  │ Drain    │ Compute  │ Drain    │ Compute  │ Drain    │      │
│ CT[0-3]  │ C[0,*]   │ CT[4-7]  │ C[1,*]   │ CT[8-11] │ C[2,*]   │      │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────┘
```

### Data Flow per Wave

```
                        ┌──────────────────┐
                        │  External Memory │
                        │    (A, B src)    │
                        └────────┬─────────┘
                                 │ DMA Load (once)
                                 ▼
                        ┌──────────────────┐
                        │   L3 Cache       │
                        │ A[0-7], B[0-3]   │
                        └────────┬─────────┘
                                 │ BlockMover (per wave)
                                 ▼
              ┌──────────────────┴───────────────────┐
              ▼                                      ▼
     ┌─────────────────┐                    ┌─────────────────┐
     │   L2 Bank [k]   │                    │   L2 Bank [k+1] │
     │  A tile, B tile │                    │  A tile, B tile │
     └────────┬────────┘                    └────────┬────────┘
              │ Streamer                             │ Streamer
              ▼                                      ▼
     ┌─────────────────┐                    ┌─────────────────┐
     │   L1 Buffers    │                    │   L1 Buffers    │
     └────────┬────────┘                    └────────┬────────┘
              │ Compute                              │ Compute
              ▼                                      ▼
     ┌─────────────────┐                    ┌─────────────────┐
     │   CT[n]         │                    │   CT[n+1]       │
     │ 24×24 systolic  │                    │ 24×24 systolic  │
     └────────┬────────┘                    └────────┬────────┘
              │ Drain                                │ Drain
              ▼                                      ▼
              └──────────────────┬──────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │  L3 Working     │
                        │  C[w,0-3]       │
                        └────────┬────────┘
                                 │ DMA Drain
                                 ▼
                        ┌─────────────────┐
                        │  Host Memory    │
                        │  (C result)     │
                        └─────────────────┘
```

## 5. Algorithm Design

### Pseudocode

```cpp
// Phase 1: Load input matrices to L3
parallel_for dma_engine in [0..3]:
    dma_external_to_l3(A tiles → L3[0..7])
    dma_external_to_l3(B tiles → L3[8..11])
barrier()

// Phase 2: Wave-based computation
for wave in [0..3]:
    // Setup: Move data from L3 to L2 for this wave's compute tiles
    parallel_for ct in [wave*4 .. wave*4+3]:
        a_tile_idx = wave * 2 + (ct % 2)  // Map to A tiles
        b_tile_idx = ct % 4                // Map to B tiles

        block_move(L3[a_tile_idx] → L2[ct])
        block_move(L3[8 + b_tile_idx] → L2[ct])
    barrier()

    // Compute: All 4 compute tiles work in parallel
    parallel_for ct in [wave*4 .. wave*4+3]:
        // Stream A rows and B columns to L1
        stream_rows(L2[ct].A → L1[ct])
        stream_cols(L2[ct].B → L1[ct])

        // Execute matmul (output-stationary)
        matmul(M_tile=512, N_tile=256, K=512)

        // Drain result to L3 working tile
        drain_result(L1[ct].C → L3[12 + ct%4])
    barrier()

    // Drain: Move C tiles from L3 to host
    parallel_for dma in [0..3]:
        dma_l3_to_host(L3[12 + dma] → Host[wave*4 + dma])
    barrier()

// Phase 3: Finalize
synchronize_all()
```

### Detailed Tile Computation

For each C tile (512 × 256), the compute tile performs:

```cpp
// C[i,j] = sum_k (A[i,k] × B[k,j]) where:
//   i = wave index (0-3), determines A row tiles
//   j = ct % 4 (0-3), determines B column tile
//   k = 0 (single K tile since K=512)

// For CT[5] computing C[1,1]:
//   Uses A tiles [2,3] (rows 512-1023)
//   Uses B tile [1] (cols 256-511)
//   Accumulates: C[512:1024, 256:512] = A[512:1024, :] × B[:, 256:512]
```

### Sub-tile Iteration for 24×24 Systolic Array

Each C tile (512 × 256) is computed by a 24×24 systolic array:

```
Sub-tiling within C tile:
  M_sub_tiles = ceil(512 / 24) = 22 sub-tiles
  N_sub_tiles = ceil(256 / 24) = 11 sub-tiles
  K_sub_tiles = ceil(512 / 24) = 22 sub-tiles

  Total sub-tile iterations = 22 × 11 × 22 = 5,324 iterations per C tile
```

## 6. Performance Analysis

### Compute Requirements

```
Total FLOPs = 2 × M × N × K = 2 × 2048 × 1024 × 512 = 2,147,483,648 FLOPs ≈ 2.1 GFLOPs
```

### Theoretical Compute Time

```
16 compute tiles × 24×24 PEs × 2 ops/PE = 18,432 FLOPs/cycle
Minimum compute cycles = 2.1 GFLOPs / 18,432 = 116,508 cycles

At 1 GHz: 116.5 μs compute time (theoretical minimum)
```

### Data Movement

| Transfer | Size | Bandwidth | Time (est.) |
|----------|------|-----------|-------------|
| A: External → L3 | 4 MB | 100 GB/s | 40 μs |
| B: External → L3 | 2 MB | 100 GB/s | 20 μs |
| L3 → L2 (per wave) | 1.5 MB | ~200 GB/s | 7.5 μs |
| L2 → L1 (per wave) | 1.5 MB | ~200 GB/s | 7.5 μs |
| L1 → Host (per wave) | 2 MB | ~50 GB/s | 40 μs |
| **Total data movement** | | | ~215 μs |

### Expected Efficiency

```
Compute time: ~116.5 μs (theoretical)
Data movement: ~215 μs (estimated)
Total: ~330 μs

Compute efficiency = 116.5 / 330 = 35% (memory-bound)

Arithmetic intensity = 2.1 GFLOPs / 14 MB = 150 FLOPs/byte
(This is compute-bound for the Drone AI's 100 GB/s bandwidth)
```

### Roofline Analysis

```
Peak compute: 18.4 TFLOPs/s (16 tiles × 1152 MACs/tile × 1 GHz)
Peak bandwidth: 100 GB/s
Ridge point: 18.4 / 100 = 184 FLOPs/byte

Workload AI: 150 FLOPs/byte
Since AI (150) < Ridge (184): workload is memory-bound

Achievable performance = 100 GB/s × 150 FLOPs/byte = 15 TFLOPs/s
Efficiency vs peak = 15 / 18.4 = 81.5%
```

## 7. Implementation Plan

### File Structure

```
examples/blas/
├── CMakeLists.txt
├── big_matmul.cpp         # Main example
└── distributed_matmul.hpp # Helper utilities (optional)
```

### Implementation Steps

1. **Configuration Setup**
   - Create Drone AI config with 16 L3 tiles
   - Verify memory map and addressing

2. **Data Initialization**
   - Allocate A, B in external memory
   - Allocate C destination in host memory
   - Initialize test data

3. **Phase 1: Load to L3**
   - DMA A tiles from external to L3[0-7]
   - DMA B tiles from external to L3[8-11]
   - Parallel DMA across 4 engines

4. **Phase 2: Wave Execution** (repeat 4 times)
   - BlockMover: L3 → L2 for current wave's tiles
   - Streamer: L2 → L1 for systolic feeding
   - Compute: Execute matmul kernel
   - Drain: L1 → L2 → L3 working tiles
   - DMA: L3 working → Host

5. **Verification**
   - Compute reference result
   - Compare simulated result
   - Report accuracy

### Key Code Sections

```cpp
// 1. Configuration with 16 L3 tiles
KPUSimulator::Config config;
config.l3_tile_count = 16;
config.l3_tile_capacity_kb = 512;  // 512 KB each = 8 MB total
config.compute_tile_count = 16;
config.processor_array_rows = 24;
config.processor_array_cols = 24;
// ... other Drone AI settings

// 2. Wave execution loop
for (Size wave = 0; wave < 4; ++wave) {
    Size ct_start = wave * 4;
    Size ct_end = ct_start + 4;

    // Move A,B tiles to L2 for this wave's compute tiles
    for (Size ct = ct_start; ct < ct_end; ++ct) {
        Size a_tile = compute_a_tile_index(wave, ct);
        Size b_tile = compute_b_tile_index(ct);

        kpu.start_block_transfer(ct % block_movers,
                                 a_tile, 0,      // L3 src
                                 ct, 0,          // L2 dst
                                 ...);
    }
    barrier();

    // Compute and drain
    for (Size ct = ct_start; ct < ct_end; ++ct) {
        kpu.start_matmul(ct, ...);
    }
    barrier();

    // Stream C tiles to host
    for (Size i = 0; i < 4; ++i) {
        kpu.dma_l3_to_host(dma_engines[i],
                          L3_working_base + i * C_tile_size,
                          host_C_base + (wave * 4 + i) * C_tile_size,
                          C_tile_size);
    }
    barrier();
}
```

## 8. Test Plan

### Unit Tests

1. **Configuration Test**
   - Verify 16 L3 tiles, 16 compute tiles
   - Verify memory addressing
   - Check L3 capacity matches 8 MB

2. **Data Distribution Test**
   - Verify A tiles correctly partitioned
   - Verify B tiles correctly partitioned
   - Check tile indices map correctly

3. **Single Wave Test**
   - Execute wave 0 only
   - Verify C[0,0] through C[0,3] correct

4. **Full Execution Test**
   - Execute all 4 waves
   - Verify complete C matrix

### Validation Criteria

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Functional correctness | 100% | Compare vs reference matmul |
| Numerical accuracy | < 1e-4 relative error | Element-wise comparison |
| Wave execution | 4 waves | Cycle tracking |
| Compute tile utilization | 16 tiles active | Status queries |
| Memory residency | A,B in L3, C to host | Memory tracing |

### Test Matrix Sizes

| Size | M | N | K | Purpose |
|------|---|---|---|---------|
| Small | 512 | 256 | 128 | Debug, fast iteration |
| Medium | 1024 | 512 | 256 | Intermediate validation |
| Full | 2048 | 1024 | 512 | Target configuration |
| Large | 4096 | 2048 | 1024 | Stress test (optional) |

## 9. Success Criteria

1. **Functional**: All C elements within 1e-4 of reference
2. **Performance**: Compute efficiency > 30%
3. **Resource Usage**: All 16 compute tiles utilized
4. **Memory Management**: C tiles correctly streamed to host
5. **Scalability**: Clear path to larger matrices

## 10. Future Extensions

1. **Overlap Optimization**: Pipeline compute with drain for higher throughput
2. **Double Buffering**: Load next wave's L2 data while current wave computes
3. **Mixed Precision**: FP16/BF16 input with FP32 accumulation
4. **Kernel Fusion**: Combine with activation functions (ReLU, GELU)
5. **Batch Processing**: Multiple matrix pairs in sequence
