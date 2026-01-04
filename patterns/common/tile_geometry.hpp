#pragma once
// tile_geometry.hpp: Tile and tensor geometry definitions for pattern tests
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <cstdint>
#include <cstddef>

namespace sw::kpu::patterns {

// Tile dimensions (elements, not bytes)
constexpr size_t TILE_ROWS = 32;
constexpr size_t TILE_COLS = 32;
constexpr size_t ELEMENT_SIZE = sizeof(int32_t);  // 4 bytes

// Derived tile sizes
constexpr size_t TILE_ELEMENTS = TILE_ROWS * TILE_COLS;          // 1024 elements
constexpr size_t TILE_BYTES = TILE_ELEMENTS * ELEMENT_SIZE;      // 4096 bytes = 4KB

// NoC transfer characteristics
constexpr size_t NOC_BUS_WIDTH_BITS = 512;
constexpr size_t NOC_BUS_WIDTH_BYTES = NOC_BUS_WIDTH_BITS / 8;   // 64 bytes
constexpr size_t CYCLES_PER_TILE = TILE_BYTES / NOC_BUS_WIDTH_BYTES;  // 64 cycles

// Reference tensor dimensions (from VOCABULARY.md)
// A[2048, 512] → 64×16 block matrix of 32×32 tiles
// B[512, 1024] → 16×32 block matrix of 32×32 tiles
// C[2048, 1024] → 64×32 block matrix of 32×32 tiles

struct TensorGeometry {
    size_t rows;           // Element rows
    size_t cols;           // Element columns
    size_t block_rows;     // Number of block-rows (tiles)
    size_t block_cols;     // Number of block-columns (tiles)
    size_t total_tiles;    // Total tile count
    size_t total_bytes;    // Total size in bytes
};

constexpr TensorGeometry TENSOR_A = {
    .rows = 2048,
    .cols = 512,
    .block_rows = 2048 / TILE_ROWS,    // 64
    .block_cols = 512 / TILE_COLS,     // 16
    .total_tiles = 64 * 16,            // 1024
    .total_bytes = 2048 * 512 * ELEMENT_SIZE  // 4MB
};

constexpr TensorGeometry TENSOR_B = {
    .rows = 512,
    .cols = 1024,
    .block_rows = 512 / TILE_ROWS,     // 16
    .block_cols = 1024 / TILE_COLS,    // 32
    .total_tiles = 16 * 32,            // 512
    .total_bytes = 512 * 1024 * ELEMENT_SIZE  // 2MB
};

constexpr TensorGeometry TENSOR_C = {
    .rows = 2048,
    .cols = 1024,
    .block_rows = 2048 / TILE_ROWS,    // 64
    .block_cols = 1024 / TILE_COLS,    // 32
    .total_tiles = 64 * 32,            // 2048
    .total_bytes = 2048 * 1024 * ELEMENT_SIZE  // 8MB
};

// Block position in a tensor (block-row, block-column)
struct BlockPosition {
    size_t block_row;
    size_t block_col;

    // Compute linear tile index (row-major)
    size_t linear_index(size_t tensor_block_cols) const {
        return block_row * tensor_block_cols + block_col;
    }

    // Compute byte offset in tensor (row-major layout)
    size_t byte_offset(size_t tensor_block_cols) const {
        return linear_index(tensor_block_cols) * TILE_BYTES;
    }
};

// L3 mesh position
struct MeshPosition {
    size_t row;
    size_t col;

    size_t linear_id(size_t mesh_cols) const {
        return row * mesh_cols + col;
    }
};

constexpr size_t MESH_ROWS = 4;
constexpr size_t MESH_COLS = 4;

}  // namespace sw::kpu::patterns
