// patterns/dma/common/matrix_layouts.hpp
//
// Matrix layout helpers for DMA tile operations
// Handles pitch, alignment, and address computation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>

#include <sw/kpu/models/interfaces/dma_engine_interface.hpp>

namespace sw::kpu::patterns::dma {

// ============================================================================
// Matrix Layout Structure
// ============================================================================

/// Matrix layout descriptor with pitch support
///
/// Pitch (also called stride or leading dimension) is the byte distance
/// between consecutive rows. It may be larger than row_bytes() due to:
/// - Alignment requirements (e.g., 64B cache line alignment)
/// - Memory allocation padding
/// - Sub-matrix views into larger matrices
///
/// Key insight: When pitch > row_bytes(), accessing consecutive rows
/// may cause page conflicts in the memory controller.
struct MatrixLayout {
    size_t rows = 0;
    size_t cols = 0;
    size_t pitch_bytes = 0;      // Stride between row starts (may be > cols * elem_size)
    size_t element_size = 4;     // Bytes per element (4 for float)
    uint64_t base_addr = 0;

    // ========================================================================
    // Derived Properties
    // ========================================================================

    /// Actual bytes of data per row
    size_t row_bytes() const { return cols * element_size; }

    /// Total matrix size in bytes (with pitch)
    size_t total_bytes() const { return rows * pitch_bytes; }

    /// True if pitch equals row bytes (no padding)
    bool is_contiguous() const { return pitch_bytes == row_bytes(); }

    /// Padding bytes per row
    size_t padding_bytes() const {
        return pitch_bytes > row_bytes() ? pitch_bytes - row_bytes() : 0;
    }

    /// Number of elements per row that fit in one cache line
    size_t elements_per_cache_line(size_t cache_line = 64) const {
        return cache_line / element_size;
    }

    /// Number of cache lines needed for one row
    size_t cache_lines_per_row(size_t cache_line = 64) const {
        return (row_bytes() + cache_line - 1) / cache_line;
    }

    // ========================================================================
    // Address Computation
    // ========================================================================

    /// Compute byte address of element at (row, col)
    uint64_t address(size_t row, size_t col) const {
        return base_addr + row * pitch_bytes + col * element_size;
    }

    /// Compute byte address of row start
    uint64_t row_address(size_t row) const {
        return base_addr + row * pitch_bytes;
    }

    /// Get row number from byte address
    size_t row_from_address(uint64_t addr) const {
        if (addr < base_addr) return 0;
        return (addr - base_addr) / pitch_bytes;
    }

    /// Get column from byte address (within the row)
    size_t col_from_address(uint64_t addr) const {
        if (addr < base_addr) return 0;
        uint64_t offset = (addr - base_addr) % pitch_bytes;
        return offset / element_size;
    }

    // ========================================================================
    // Tile Operations
    // ========================================================================

    /// Generate DMA transfers to read a rectangular tile
    ///
    /// @param tile_row Starting row in matrix
    /// @param tile_col Starting column in matrix
    /// @param tile_height Tile height (rows)
    /// @param tile_width Tile width (columns)
    /// @param dst_base Destination base address (L3 tile)
    /// @param dst_pitch Destination pitch (0 = contiguous)
    /// @return Vector of DMA transfers (one per row typically)
    std::vector<DMATransfer> tile_read_transfers(
        size_t tile_row, size_t tile_col,
        size_t tile_height, size_t tile_width,
        uint64_t dst_base,
        size_t dst_pitch = 0) const
    {
        std::vector<DMATransfer> transfers;
        transfers.reserve(tile_height);

        size_t transfer_size = tile_width * element_size;
        size_t actual_dst_pitch = (dst_pitch > 0) ? dst_pitch : transfer_size;

        for (size_t r = 0; r < tile_height; ++r) {
            DMATransfer xfer;
            xfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            xfer.src_addr = address(tile_row + r, tile_col);
            xfer.dst_type = DMAMemoryType::L3_TILE;
            xfer.dst_addr = dst_base + r * actual_dst_pitch;
            xfer.size = static_cast<uint32_t>(transfer_size);
            xfer.stride = 0;
            xfer.count = 1;
            xfer.priority = 0;
            xfer.ordered = true;
            transfers.push_back(xfer);
        }

        return transfers;
    }

    /// Generate DMA transfers to write a rectangular tile
    std::vector<DMATransfer> tile_write_transfers(
        size_t tile_row, size_t tile_col,
        size_t tile_height, size_t tile_width,
        uint64_t src_base,
        size_t src_pitch = 0) const
    {
        std::vector<DMATransfer> transfers;
        transfers.reserve(tile_height);

        size_t transfer_size = tile_width * element_size;
        size_t actual_src_pitch = (src_pitch > 0) ? src_pitch : transfer_size;

        for (size_t r = 0; r < tile_height; ++r) {
            DMATransfer xfer;
            xfer.src_type = DMAMemoryType::L3_TILE;
            xfer.src_addr = src_base + r * actual_src_pitch;
            xfer.dst_type = DMAMemoryType::DEVICE_MEMORY;
            xfer.dst_addr = address(tile_row + r, tile_col);
            xfer.size = static_cast<uint32_t>(transfer_size);
            xfer.stride = 0;
            xfer.count = 1;
            xfer.priority = 0;
            xfer.ordered = true;
            transfers.push_back(xfer);
        }

        return transfers;
    }

    // ========================================================================
    // Page Analysis
    // ========================================================================

    /// Estimate page conflicts when reading a tile
    /// Assumes simple row-buffer mapping where different rows = different pages
    ///
    /// @param tile_height Tile height
    /// @param page_size Memory page size (default 4KB)
    /// @return Estimated number of page conflicts
    size_t estimate_page_conflicts(size_t tile_height, size_t page_size = 4096) const {
        if (tile_height <= 1) return 0;

        // If pitch >= page_size, every row is a page conflict
        if (pitch_bytes >= page_size) {
            return tile_height - 1;
        }

        // Count rows that cross page boundaries
        size_t conflicts = 0;
        uint64_t current_page = 0;  // Will be set by first row

        for (size_t r = 0; r < tile_height; ++r) {
            uint64_t row_start = row_address(r);
            uint64_t row_page = row_start / page_size;

            if (r == 0) {
                current_page = row_page;
            } else if (row_page != current_page) {
                conflicts++;
                current_page = row_page;
            }
        }

        return conflicts;
    }

    /// Calculate page hit rate for a tile
    double page_hit_rate(size_t tile_height, size_t page_size = 4096) const {
        if (tile_height <= 1) return 1.0;
        size_t conflicts = estimate_page_conflicts(tile_height, page_size);
        size_t hits = tile_height - 1 - conflicts;
        return static_cast<double>(hits) / static_cast<double>(tile_height - 1);
    }

    // ========================================================================
    // Factory Methods
    // ========================================================================

    /// Create row-major layout (pitch = row bytes, no padding)
    static MatrixLayout row_major(size_t rows, size_t cols, size_t elem_size,
                                   uint64_t base = 0) {
        MatrixLayout layout;
        layout.rows = rows;
        layout.cols = cols;
        layout.element_size = elem_size;
        layout.pitch_bytes = cols * elem_size;
        layout.base_addr = base;
        return layout;
    }

    /// Create layout with explicit pitch
    static MatrixLayout pitched(size_t rows, size_t cols, size_t pitch,
                                 size_t elem_size, uint64_t base = 0) {
        MatrixLayout layout;
        layout.rows = rows;
        layout.cols = cols;
        layout.pitch_bytes = pitch;
        layout.element_size = elem_size;
        layout.base_addr = base;
        return layout;
    }

    /// Create layout aligned to cache line boundaries
    static MatrixLayout cache_aligned(size_t rows, size_t cols, size_t elem_size,
                                       size_t cache_line = 64, uint64_t base = 0) {
        MatrixLayout layout;
        layout.rows = rows;
        layout.cols = cols;
        layout.element_size = elem_size;
        // Round pitch up to cache line boundary
        size_t row_bytes = cols * elem_size;
        layout.pitch_bytes = ((row_bytes + cache_line - 1) / cache_line) * cache_line;
        layout.base_addr = base;
        return layout;
    }

    /// Create layout aligned to page boundaries
    static MatrixLayout page_aligned(size_t rows, size_t cols, size_t elem_size,
                                      size_t page_size = 4096, uint64_t base = 0) {
        MatrixLayout layout;
        layout.rows = rows;
        layout.cols = cols;
        layout.element_size = elem_size;
        // Round pitch up to page boundary
        size_t row_bytes = cols * elem_size;
        layout.pitch_bytes = ((row_bytes + page_size - 1) / page_size) * page_size;
        layout.base_addr = base;
        return layout;
    }
};

// ============================================================================
// Column-Major Layout Helper
// ============================================================================

/// Create column-major layout (for B matrix in GEMM)
/// In column-major, consecutive elements in a column are contiguous
/// To read a row, you need strided accesses
inline MatrixLayout col_major_layout(size_t rows, size_t cols, size_t elem_size,
                                      uint64_t base = 0) {
    // In column-major: stride between elements in same row = rows * elem_size
    // This creates a transposed view
    MatrixLayout layout;
    layout.rows = cols;  // Swap for access pattern
    layout.cols = rows;
    layout.element_size = elem_size;
    layout.pitch_bytes = cols * elem_size;  // One column
    layout.base_addr = base;
    return layout;
}

// ============================================================================
// NHWC Tensor Layout Helper
// ============================================================================

/// NHWC tensor layout for conv2d patterns
/// N=batch, H=height, W=width, C=channels
struct NHWCLayout {
    size_t N, H, W, C;
    size_t element_size;
    uint64_t base_addr;

    /// Compute address of element at (n, h, w, c)
    uint64_t address(size_t n, size_t h, size_t w, size_t c) const {
        return base_addr +
               ((n * H * W * C) + (h * W * C) + (w * C) + c) * element_size;
    }

    /// Get matrix layout for a spatial slice (H×W with C channels)
    MatrixLayout spatial_slice(size_t n) const {
        MatrixLayout layout;
        layout.rows = H;
        layout.cols = W * C;  // Flatten W and C
        layout.pitch_bytes = W * C * element_size;
        layout.element_size = element_size;
        layout.base_addr = address(n, 0, 0, 0);
        return layout;
    }

    /// Create from dimensions
    static NHWCLayout create(size_t n, size_t h, size_t w, size_t c,
                              size_t elem_size = 4, uint64_t base = 0) {
        return NHWCLayout{n, h, w, c, elem_size, base};
    }
};

} // namespace sw::kpu::patterns::dma
