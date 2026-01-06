# Level 6: Complex Patterns

Real-world memory access patterns that stress test the memory controller
with realistic workloads from matrix operations, sparse data, and tile-based
computation.

## Patterns

### strided/

Tests non-sequential memory access with various stride lengths.

**Use Cases:**
- Matrix column access (stride = row width)
- Tensor dimension traversal
- FFT butterfly operations
- Image downsampling

**Test Cases:**
1. **Unit Stride** - Sequential access (baseline)
2. **Stride 2** - Every other cache line
3. **Row-Crossing Stride** - Large stride crossing DRAM rows
4. **Matrix Column** - Column access in row-major matrix
5. **Bank-Spreading** - Stride distributing across banks
6. **Power-of-2 Strides** - 2, 4, 8, 16 element strides
7. **Transpose Pattern** - Read sequential, write strided

**Performance Impact:**
```
Sequential:     ~400 cycles (baseline)
Stride 4:       ~400 cycles (same row = no penalty)
Row-crossing:   ~1800 cycles (4x slower - page conflicts)
```

### random/

Tests worst-case locality scenarios with random access patterns.

**Use Cases:**
- Hash table lookups
- Pointer chasing (linked lists)
- Graph traversal
- Sparse matrix operations

**Test Cases:**
1. **Random Single Bank** - Random rows in one bank
2. **Random Multi-Bank** - Random bank and row selection
3. **Random Optimized** - Bank-group aware random access
4. **Random vs Sequential** - Performance comparison
5. **Pointer Chasing** - Linked list traversal simulation
6. **Hash Table Access** - Uniformly distributed lookups
7. **Sparse Matrix** - Clustered non-zero element access
8. **Random Mixed R/W** - 50/50 read/write ratio

**Performance Impact:**
```
Sequential:     ~400 cycles (baseline)
Random:         ~2000+ cycles (5x slower - page conflicts)
Pointer chase:  ~700 cycles (serialized)
```

### tile-load/

Tests KPU-specific tile data movement patterns for matrix operations.

**Tile Format:**
- Standard tile: 32×32 elements = 4KB (int32)
- 64 cache lines per tile
- Row-major layout, cache-line aligned

**Test Cases:**
1. **Single Tile Load** - Basic 4KB tile from one bank
2. **Multi-Bank Tile** - Tile distributed across 4 banks
3. **Double Buffer** - Alternating tile buffers
4. **Matmul Pattern** - Load A, B tiles; write C tile
5. **Tile Streaming** - Sequential tile loading
6. **Dual-Channel Tile** - Interleaved across channels
7. **Strategy Comparison** - Single vs multi-bank vs dual-channel
8. **MLP Pattern** - Input + weight + output tiles

**Performance Results:**
```
Single bank tile:     ~1400 cycles (2.9 bytes/cycle)
Multi-bank tile:      ~900 cycles (4.5 bytes/cycle)
Dual-channel tile:    ~600 cycles (6.8 bytes/cycle)
```

## Running

```bash
# Build
cmake --build build

# Run strided patterns
./build/patterns/memory/lpddr5/lpddr5_strided

# Run random patterns
./build/patterns/memory/lpddr5/lpddr5_random

# Run tile-load patterns
./build/patterns/memory/lpddr5/lpddr5_tile_load

# With trace export
./build/patterns/memory/lpddr5/lpddr5_tile_load --trace output.json
```

## Key Insights

### Strided Access Optimization

| Stride Type | Strategy |
|-------------|----------|
| Small stride (1-4 lines) | Keep in same row = page hits |
| Row-crossing stride | Spread across banks |
| Power-of-2 stride | Use bank-spreading addressing |
| Matrix transpose | Use different banks for src/dst |

### Random Access Optimization

| Pattern | Strategy |
|---------|----------|
| Hash table | Spread buckets across banks |
| Pointer chase | Prefetch next pointer |
| Graph traversal | Sort edges by destination |
| Sparse matrix | Cluster non-zeros by row |

### Tile Loading Best Practices

| Scenario | Recommended Configuration |
|----------|--------------------------|
| Single tile | Multi-bank (4 banks across groups) |
| Matmul (A×B=C) | Each matrix in different bank |
| Streaming | Double-buffer with separate banks |
| Maximum BW | Dual-channel with interleaving |

## Bandwidth Analysis

### Tile Loading Throughput

| Configuration | Throughput | Utilization |
|---------------|------------|-------------|
| Single bank | 2.9 B/cycle | 23% |
| 4 banks (same group) | 3.8 B/cycle | 30% |
| 4 banks (across groups) | 4.5 B/cycle | 35% |
| Dual channel interleaved | 6.8 B/cycle | 53% |
| Peak theoretical | 12.8 B/cycle | 100% |

### Random Access Impact

| Pattern | Page Hit Ratio | Relative Perf |
|---------|---------------|---------------|
| Sequential | 93%+ | 1.0x |
| Strided (same row) | 93%+ | 1.0x |
| Strided (row-crossing) | 0% | 0.25x |
| Random | 0-10% | 0.2x |
| Sparse (clustered) | 50-70% | 0.5x |

## Architectural Recommendations

### For Matrix Operations

1. **Tile placement**: Align tiles to bank groups
   - A matrix: Banks 0,1,2,3 (BG0)
   - B matrix: Banks 4,5,6,7 (BG1)
   - C matrix: Banks 8,9,10,11 (BG2)

2. **Double buffering**: Use different channels
   - Current tiles: Channel 0
   - Next tiles: Channel 1

3. **Transpose**: Use different banks for input/output
   - Avoid bank conflicts during strided writes

### For Sparse/Random Workloads

1. **Hash tables**: Distribute across all 16 banks
2. **Graphs**: Sort by bank affinity when possible
3. **Sparse matrices**: Use compressed format with bank hints
4. **Pointer structures**: Add prefetch hints
