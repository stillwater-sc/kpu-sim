/**
 * @file address_generator.hpp
 * @brief Address generation for KPU AUTO addressing modes
 *
 * The AddressGenerator computes tile addresses from:
 * - Base address registers (per matrix)
 * - Stride registers (row_stride, tile_i_stride, tile_j_stride)
 * - Current loop indices (ti, tj, tk from LoopState)
 *
 * Address computation:
 *   For matrix A (indexed by ti, tk):
 *     addr = base_A + ti × stride_A_tile_i + tk × stride_A_tile_j
 *
 *   For matrix B (indexed by tk, tj):
 *     addr = base_B + tk × stride_B_tile_i + tj × stride_B_tile_j
 *
 *   For matrix C (indexed by ti, tj):
 *     addr = base_C + ti × stride_C_tile_i + tj × stride_C_tile_j
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/isa/loop_state.hpp>
#include <sw/concepts.hpp>
#include <cstdint>
#include <array>

namespace sw::kpu::isa {

/**
 * @brief Stride configuration for a matrix
 */
struct MatrixStrides {
    Size row_stride = 0;        ///< Bytes between consecutive rows
    Size tile_i_stride = 0;     ///< Bytes between consecutive row tiles
    Size tile_j_stride = 0;     ///< Bytes between consecutive column tiles

    /// Compute address offset for tile indices
    Address compute_offset(uint16_t i, uint16_t j) const {
        return static_cast<Address>(i) * tile_i_stride +
               static_cast<Address>(j) * tile_j_stride;
    }
};

/**
 * @brief Tile dimensions configuration
 */
struct TileDimensions {
    Size Ti = 16;               ///< Tile height (rows)
    Size Tj = 16;               ///< Tile width (columns)
    Size Tk = 16;               ///< Reduction tile depth
    Size element_size = 4;      ///< Element size in bytes (4 for float32)

    /// Compute tile size in bytes
    Size tile_bytes() const {
        return Ti * Tj * element_size;
    }
};

/**
 * @brief Matrix dimensions
 */
struct MatrixDimensions {
    Size rows = 0;
    Size cols = 0;
};

/**
 * @brief Address generator for AUTO addressing modes
 *
 * Computes external memory addresses and buffer offsets for
 * DMA, BlockMover, and Streamer operations using loop indices.
 */
class AddressGenerator {
public:
    AddressGenerator() = default;

    // ========================================================================
    // Configuration Methods (called by SET_* instructions)
    // ========================================================================

    /// Set external memory base address for a matrix
    void set_base(MatrixID mat, Address base) {
        base_addresses_[static_cast<size_t>(mat)] = base;
    }

    /// Set L3 base offset for a matrix
    void set_l3_base(MatrixID mat, Address offset) {
        l3_bases_[static_cast<size_t>(mat)] = offset;
    }

    /// Set L2 base offset for a matrix
    void set_l2_base(MatrixID mat, Address offset) {
        l2_bases_[static_cast<size_t>(mat)] = offset;
    }

    /// Set stride configuration for a matrix
    void set_strides(MatrixID mat, Size row_stride, Size tile_i_stride, Size tile_j_stride) {
        auto& s = strides_[static_cast<size_t>(mat)];
        s.row_stride = row_stride;
        s.tile_i_stride = tile_i_stride;
        s.tile_j_stride = tile_j_stride;
    }

    /// Set tile dimensions
    void set_tile_dim(Size Ti, Size Tj, Size Tk, Size elem_size) {
        tile_dim_.Ti = Ti;
        tile_dim_.Tj = Tj;
        tile_dim_.Tk = Tk;
        tile_dim_.element_size = elem_size;
    }

    /// Set matrix dimensions
    void set_matrix_dim(MatrixID mat, Size rows, Size cols) {
        auto& d = matrix_dims_[static_cast<size_t>(mat)];
        d.rows = rows;
        d.cols = cols;
    }

    // ========================================================================
    // Address Computation Methods
    // ========================================================================

    /**
     * @brief Compute external memory address for a matrix tile
     * @param mat Matrix identifier (A, B, or C)
     * @param loop_state Current loop state for indices
     * @return External memory address
     *
     * Address computation depends on matrix role:
     *   A[M,K]: uses (ti, tk)
     *   B[K,N]: uses (tk, tj)
     *   C[M,N]: uses (ti, tj)
     */
    Address compute_ext_addr(MatrixID mat, const LoopState& loop_state) const {
        Address base = base_addresses_[static_cast<size_t>(mat)];
        const auto& strides = strides_[static_cast<size_t>(mat)];

        uint16_t ti = loop_state.ti();
        uint16_t tj = loop_state.tj();
        uint16_t tk = loop_state.tk();

        switch (mat) {
            case MatrixID::A:
                // A[M,K] indexed by (ti, tk)
                return base + strides.compute_offset(ti, tk);

            case MatrixID::B:
                // B[K,N] indexed by (tk, tj)
                return base + strides.compute_offset(tk, tj);

            case MatrixID::C:
                // C[M,N] indexed by (ti, tj)
                return base + strides.compute_offset(ti, tj);
        }
        return base;
    }

    /**
     * @brief Compute external memory address for explicit tile coordinates
     */
    Address compute_ext_addr(MatrixID mat, TileCoord tile) const {
        Address base = base_addresses_[static_cast<size_t>(mat)];
        const auto& strides = strides_[static_cast<size_t>(mat)];

        switch (mat) {
            case MatrixID::A:
                return base + strides.compute_offset(tile.ti, tile.tk);
            case MatrixID::B:
                return base + strides.compute_offset(tile.tk, tile.tj);
            case MatrixID::C:
                return base + strides.compute_offset(tile.ti, tile.tj);
        }
        return base;
    }

    /// Compute L3 offset for a matrix tile
    Address compute_l3_offset(MatrixID mat, [[maybe_unused]] const LoopState& loop_state) const {
        // For now, use static allocation per matrix
        // Future: dynamic slot allocation based on caching policy
        return l3_bases_[static_cast<size_t>(mat)];
    }

    /// Compute L2 offset for a matrix tile
    Address compute_l2_offset(MatrixID mat, [[maybe_unused]] const LoopState& loop_state) const {
        return l2_bases_[static_cast<size_t>(mat)];
    }

    /// Get tile size in bytes
    Size tile_bytes() const {
        return tile_dim_.tile_bytes();
    }

    /// Get tile dimensions
    const TileDimensions& tile_dim() const {
        return tile_dim_;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    Address base(MatrixID mat) const {
        return base_addresses_[static_cast<size_t>(mat)];
    }

    Address l3_base(MatrixID mat) const {
        return l3_bases_[static_cast<size_t>(mat)];
    }

    Address l2_base(MatrixID mat) const {
        return l2_bases_[static_cast<size_t>(mat)];
    }

    const MatrixStrides& strides(MatrixID mat) const {
        return strides_[static_cast<size_t>(mat)];
    }

    const MatrixDimensions& matrix_dim(MatrixID mat) const {
        return matrix_dims_[static_cast<size_t>(mat)];
    }

    /// Reset all configuration
    void reset() {
        for (size_t i = 0; i < 3; ++i) {
            base_addresses_[i] = 0;
            l3_bases_[i] = 0;
            l2_bases_[i] = 0;
            strides_[i] = MatrixStrides{};
            matrix_dims_[i] = MatrixDimensions{};
        }
        tile_dim_ = TileDimensions{};
    }

private:
    // Base addresses (indexed by MatrixID)
    std::array<Address, 3> base_addresses_ = {0, 0, 0};
    std::array<Address, 3> l3_bases_ = {0, 0, 0};
    std::array<Address, 3> l2_bases_ = {0, 0, 0};

    // Stride configuration (indexed by MatrixID)
    std::array<MatrixStrides, 3> strides_;

    // Matrix dimensions (indexed by MatrixID)
    std::array<MatrixDimensions, 3> matrix_dims_;

    // Tile dimensions (shared across all matrices)
    TileDimensions tile_dim_;
};

} // namespace sw::kpu::isa
