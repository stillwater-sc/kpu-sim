# Memory Interleaving Design for KPU

## Problem Statement

For a matrix multiplication C = A × B, we need to load A tiles and B tiles concurrently while potentially storing C tiles. With N memory channels, we want to maximize bandwidth by ensuring that tiles being accessed simultaneously are on different channels.

## Key Insight

The memory channel is determined by the physical address. If we have 4 channels:
- Channel 0: addresses where `(addr / interleave_size) % 4 == 0`
- Channel 1: addresses where `(addr / interleave_size) % 4 == 1`
- Channel 2: addresses where `(addr / interleave_size) % 4 == 2`
- Channel 3: addresses where `(addr / interleave_size) % 4 == 3`

The `interleave_size` is typically a cache line (64 bytes) or a larger block.

## Design Options

---

## Option 1: Matrix-Level Channel Assignment

**Concept**: Assign each matrix to a dedicated subset of channels.

```
With 4 channels:
  Channels 0,1: Matrix A
  Channel 2:    Matrix B
  Channel 3:    Matrix C

With 8 channels:
  Channels 0,1,2: Matrix A
  Channels 3,4,5: Matrix B
  Channels 6,7:   Matrix C
```

**Memory Layout**:
```
Base addresses chosen to align matrices to channel boundaries:

channel_stride = total_memory / num_channels

A_base = 0                          // Starts at channel 0
B_base = 2 * channel_stride         // Starts at channel 2
C_base = 3 * channel_stride         // Starts at channel 3
```

**Tile Address Calculation**:
```cpp
Address calculate_tile_address(MatrixID mat, Size ti, Size tj,
                                Size Ti, Size Tj, Size M, Size N, Size K) {
    Address base;
    Size rows, cols, tile_rows, tile_cols;

    switch (mat) {
        case A: base = A_base; rows = M; cols = K; tile_rows = ti; tile_cols = tk; break;
        case B: base = B_base; rows = K; cols = N; tile_rows = tk; tile_cols = tj; break;
        case C: base = C_base; rows = M; cols = N; tile_rows = ti; tile_cols = tj; break;
    }

    // Row-major tile layout
    Size tile_linear_idx = tile_rows * (cols / Tj) + tile_cols;
    Size tile_size_bytes = Ti * Tj * element_size;

    return base + tile_linear_idx * tile_size_bytes;
}
```

**Pros**:
- Simple to implement
- Guaranteed no channel conflicts between A, B, C accesses
- Predictable channel assignment

**Cons**:
- Uneven channel utilization (A may need more bandwidth than C)
- Inflexible for different workloads
- Wastes channels if one matrix is much smaller

---

## Option 2: Tile-Level Round-Robin Interleaving

**Concept**: Interleave tiles across all channels regardless of matrix. Each tile is placed on the next channel in round-robin order.

**Memory Layout**:
```
Tile size = Ti × Tj × element_size (e.g., 32×32×4 = 4KB)

All tiles laid out sequentially, channel = (tile_index % num_channels)

Global tile index assignment:
  A tiles: 0, 1, 2, ... (num_A_tiles - 1)
  B tiles: num_A_tiles, num_A_tiles + 1, ...
  C tiles: num_A_tiles + num_B_tiles, ...

Channel assignment:
  Tile 0 -> Channel 0
  Tile 1 -> Channel 1
  Tile 2 -> Channel 2
  Tile 3 -> Channel 3
  Tile 4 -> Channel 0  (wraps)
  ...
```

**Address Calculation**:
```cpp
struct TileLayout {
    Size num_a_tiles, num_b_tiles, num_c_tiles;
    Size tile_size_bytes;
    Size num_channels;
    Address base_address;

    Size get_global_tile_index(MatrixID mat, Size ti, Size tj, Size tk,
                                Size m_tiles, Size n_tiles, Size k_tiles) {
        Size local_idx;
        Size offset;

        switch (mat) {
            case A:
                local_idx = ti * k_tiles + tk;
                offset = 0;
                break;
            case B:
                local_idx = tk * n_tiles + tj;
                offset = num_a_tiles;
                break;
            case C:
                local_idx = ti * n_tiles + tj;
                offset = num_a_tiles + num_b_tiles;
                break;
        }
        return offset + local_idx;
    }

    uint8_t get_channel(Size global_tile_idx) {
        return global_tile_idx % num_channels;
    }

    Address get_address(Size global_tile_idx) {
        return base_address + global_tile_idx * tile_size_bytes;
    }
};
```

**Pros**:
- Even distribution across channels over time
- Simple round-robin assignment

**Cons**:
- A[ti,tk] and B[tk,tj] for same iteration may land on same channel (conflict!)
- No guarantee of concurrent access without conflicts

---

## Option 3: Iteration-Aware Channel Assignment (RECOMMENDED)

**Concept**: Design the layout so that tiles accessed in the same iteration are guaranteed to be on different channels.

For output-stationary matmul, each iteration accesses:
- One A tile: A[ti, tk]
- One B tile: B[tk, tj]
- (Accumulates to) C[ti, tj]

We need A[ti,tk] and B[tk,tj] to be on different channels for ALL valid (ti, tk, tj) combinations.

**Key Insight**: Use matrix ID as part of channel selection.

**Memory Layout Strategy**:
```
For 4 channels, assign:
  A tiles -> Channels 0, 2 (even channels)
  B tiles -> Channels 1, 3 (odd channels)
  C tiles -> Based on (ti + tj) % 2 to avoid conflicts during drain/store

Within each matrix, tiles are further distributed:
  A[ti, tk] -> Channel = (2 * ((ti + tk) % 2))      = 0 or 2
  B[tk, tj] -> Channel = (2 * ((tk + tj) % 2)) + 1  = 1 or 3
  C[ti, tj] -> Channel = 2 * ((ti + tj) % 2)        = 0 or 2 (but accessed at different time)
```

**Address Calculation**:
```cpp
struct IterationAwareLayout {
    Size num_channels;          // Must be even (e.g., 4, 8)
    Size tile_size_bytes;
    Address a_base, b_base, c_base;
    Size m_tiles, n_tiles, k_tiles;

    // Channels for A: 0, 2, 4, ... (even)
    // Channels for B: 1, 3, 5, ... (odd)

    uint8_t get_channel(MatrixID mat, Size ti, Size tj, Size tk) {
        Size half_channels = num_channels / 2;

        switch (mat) {
            case A: {
                // A[ti, tk] uses even channels
                Size sub_channel = (ti + tk) % half_channels;
                return 2 * sub_channel;  // 0, 2, 4, ...
            }
            case B: {
                // B[tk, tj] uses odd channels
                Size sub_channel = (tk + tj) % half_channels;
                return 2 * sub_channel + 1;  // 1, 3, 5, ...
            }
            case C: {
                // C[ti, tj] - accessed during drain, not concurrent with A/B loads
                // Can use any channel, but prefer even for locality with A
                Size sub_channel = (ti + tj) % half_channels;
                return 2 * sub_channel;
            }
        }
    }

    Address get_address(MatrixID mat, Size ti, Size tj, Size tk) {
        // Within each channel, tiles are packed sequentially
        // Address = base + channel_offset + local_tile_offset

        uint8_t channel = get_channel(mat, ti, tj, tk);
        Size local_tile_idx;
        Address base;
        Size tiles_per_channel;

        switch (mat) {
            case A:
                base = a_base;
                // Count how many A tiles come before this one on this channel
                local_tile_idx = count_a_tiles_before(ti, tk, channel);
                break;
            case B:
                base = b_base;
                local_tile_idx = count_b_tiles_before(tk, tj, channel);
                break;
            case C:
                base = c_base;
                local_tile_idx = count_c_tiles_before(ti, tj, channel);
                break;
        }

        // Each channel has its own address space region
        Size channel_stride = get_channel_capacity();
        return base + channel * channel_stride + local_tile_idx * tile_size_bytes;
    }
};
```

**Pros**:
- GUARANTEES A and B tiles are on different channels for every iteration
- Maximizes concurrent bandwidth utilization
- Works for any number of channels (as long as even)

**Cons**:
- More complex address calculation
- Requires knowing tile dimensions at layout time
- May have some internal fragmentation

---

## Option 4: Hardware-Style Address Interleaving

**Concept**: Use low-order address bits for channel selection, with tile base addresses chosen to ensure proper interleaving.

**Memory Layout**:
```
Standard hardware interleaving:
  Channel = (address / cache_line_size) % num_channels

Where cache_line_size = 64 bytes typically.

To ensure tiles land on different channels, choose tile base addresses:
  - A tiles: base addresses where (base / 64) % 4 == 0 or 2
  - B tiles: base addresses where (base / 64) % 4 == 1 or 3
```

**Tile Placement Algorithm**:
```cpp
struct HardwareInterleaveLayout {
    Size interleave_granularity = 64;  // Cache line size
    Size num_channels = 4;
    Size tile_size_bytes;

    // Compute channel from any address
    uint8_t address_to_channel(Address addr) {
        return (addr / interleave_granularity) % num_channels;
    }

    // Find next address on target channel >= min_addr
    Address find_address_on_channel(Address min_addr, uint8_t target_channel) {
        uint8_t current_channel = address_to_channel(min_addr);
        if (current_channel == target_channel) {
            return min_addr;
        }

        // Advance to next interleave boundary
        Address aligned = (min_addr / interleave_granularity + 1) * interleave_granularity;

        // Find the channel we want
        while (address_to_channel(aligned) != target_channel) {
            aligned += interleave_granularity;
        }
        return aligned;
    }

    // Layout all tiles, assigning channels based on matrix
    void layout_tiles(Address base,
                      Size m_tiles, Size n_tiles, Size k_tiles,
                      std::vector<Address>& a_addrs,
                      std::vector<Address>& b_addrs,
                      std::vector<Address>& c_addrs) {

        Address next_addr = base;

        // A tiles on even channels
        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tk = 0; tk < k_tiles; ++tk) {
                uint8_t target_channel = 2 * ((ti + tk) % (num_channels/2));
                Address addr = find_address_on_channel(next_addr, target_channel);
                a_addrs.push_back(addr);
                next_addr = addr + tile_size_bytes;
            }
        }

        // B tiles on odd channels
        for (Size tk = 0; tk < k_tiles; ++tk) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                uint8_t target_channel = 2 * ((tk + tj) % (num_channels/2)) + 1;
                Address addr = find_address_on_channel(next_addr, target_channel);
                b_addrs.push_back(addr);
                next_addr = addr + tile_size_bytes;
            }
        }

        // C tiles - can use any channel (not concurrent with A/B)
        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                c_addrs.push_back(next_addr);
                next_addr += tile_size_bytes;
            }
        }
    }
};
```

**Pros**:
- Matches real hardware behavior
- Flexible granularity
- Can verify correctness by checking addresses

**Cons**:
- Potential address space fragmentation
- Requires pre-computation of all tile addresses
- More complex than logical channel assignment

---

## Recommendation

**Option 3 (Iteration-Aware Channel Assignment)** is recommended because:

1. **Guaranteed Conflict-Free**: A and B tiles in the same iteration are always on different channels
2. **Deterministic**: Channel can be computed from (matrix, ti, tj, tk) without lookup tables
3. **Scalable**: Works for 4, 8, 16+ channels
4. **Simple Rule**: Even channels for A, odd channels for B

**Implementation Steps**:

1. Add `ChannelLayout` class to compute channel and address from tile coordinates
2. Modify `OutputStationaryProgramBuilder` to use the layout for L3 tile IDs
3. Update `ConcurrentExecutor` to use channel from instruction rather than hashing
4. Verify with debug tool that A and B operations are on different channels

---

## Address Space Layout Example

For 4 channels, 1024x1024x1024 matmul with 64x64 tiles:
- m_tiles = 16, n_tiles = 16, k_tiles = 16
- A tiles: 16 × 16 = 256 tiles
- B tiles: 16 × 16 = 256 tiles
- C tiles: 16 × 16 = 256 tiles
- Tile size: 64 × 64 × 4 = 16KB

```
Channel 0: A tiles where (ti + tk) % 2 == 0  -> 128 tiles = 2MB
Channel 1: B tiles where (tk + tj) % 2 == 0  -> 128 tiles = 2MB
Channel 2: A tiles where (ti + tk) % 2 == 1  -> 128 tiles = 2MB
Channel 3: B tiles where (tk + tj) % 2 == 1  -> 128 tiles = 2MB

C tiles: 256 tiles = 4MB (stored after computation, channel doesn't matter for load)

Total: 12MB across 4 channels, each channel holds 2-3MB
```

During any iteration (ti, tk, tj):
- A[ti,tk] is on channel 0 or 2 (even)
- B[tk,tj] is on channel 1 or 3 (odd)
- No conflicts, full bandwidth utilization!
