// ============================================================================
// include/sw/kpu/behavioral/block_mover.hpp
// Behavioral (functional) BlockMover - instant L3 <-> L2 transfers
//
// The BlockMover transfers tile data between L3 (on-chip cache) and L2 (working
// memory banks) with optional transformations like transpose.
//
// In behavioral mode, transfers are instant - no timing simulation.
// ============================================================================

#pragma once

#include <sw/kpu/behavioral/memory_model.hpp>
#include <cstdint>
#include <functional>

namespace sw::kpu::behavioral {

/// Behavioral BlockMover for L3 <-> L2 transfers
///
/// This is a functional simulation that performs actual data movement:
/// - L3 -> L2: Copy tile data from L3 cache to L2 working memory
/// - L2 -> L3: Copy results from L2 back to L3 cache
/// - Optional transpose during transfer
///
/// Used for:
/// - Functional verification of data flow
/// - MLP/matmul execution in behavioral mode
class BehavioralBlockMover {
public:
    /// Transfer descriptor specifying source, destination, and parameters
    struct TransferDescriptor {
        // Source location
        MemoryRegionType src_type;      // L3_TILE or L2_BANK
        uint32_t src_region_id;         // tile_id or bank_id
        Size src_offset;                // Offset within region

        // Destination location
        MemoryRegionType dst_type;      // L3_TILE or L2_BANK
        uint32_t dst_region_id;
        Size dst_offset;

        // Transfer geometry
        uint32_t height;                // Number of rows
        uint32_t width;                 // Number of columns
        uint32_t element_size;          // Bytes per element (4 for float)

        // Strides (0 = contiguous)
        uint32_t src_stride = 0;        // Source row stride in bytes
        uint32_t dst_stride = 0;        // Destination row stride in bytes

        // Transformation
        bool transpose = false;         // Swap rows <-> columns during transfer
    };

    /// Statistics tracking
    struct Stats {
        uint64_t transfers = 0;
        uint64_t l3_to_l2_transfers = 0;
        uint64_t l2_to_l3_transfers = 0;
        Size bytes_moved = 0;
        uint64_t transpose_count = 0;

        void reset() {
            transfers = 0;
            l3_to_l2_transfers = 0;
            l2_to_l3_transfers = 0;
            bytes_moved = 0;
            transpose_count = 0;
        }
    };

    // ========================================================================
    // Constructors
    // ========================================================================

    BehavioralBlockMover() = default;
    ~BehavioralBlockMover() = default;

    // Non-copyable, movable
    BehavioralBlockMover(const BehavioralBlockMover&) = delete;
    BehavioralBlockMover& operator=(const BehavioralBlockMover&) = delete;
    BehavioralBlockMover(BehavioralBlockMover&&) = default;
    BehavioralBlockMover& operator=(BehavioralBlockMover&&) = default;

    // ========================================================================
    // Transfer Operations
    // ========================================================================

    /// Execute a transfer using a descriptor
    void transfer(const TransferDescriptor& desc, BehavioralMemoryModel* memory);

    /// Convenience: L3 tile -> L2 bank
    void l3_to_l2(
        uint32_t l3_tile_id, Size l3_offset,
        uint32_t l2_bank_id, Size l2_offset,
        uint32_t height, uint32_t width, uint32_t element_size,
        BehavioralMemoryModel* memory,
        bool transpose = false);

    /// Convenience: L2 bank -> L3 tile
    void l2_to_l3(
        uint32_t l2_bank_id, Size l2_offset,
        uint32_t l3_tile_id, Size l3_offset,
        uint32_t height, uint32_t width, uint32_t element_size,
        BehavioralMemoryModel* memory);

    /// Convenience: L3 tile -> L2 bank with strided source (for pitched matrices)
    void l3_to_l2_strided(
        uint32_t l3_tile_id, Size l3_offset, uint32_t l3_stride,
        uint32_t l2_bank_id, Size l2_offset,
        uint32_t height, uint32_t width, uint32_t element_size,
        BehavioralMemoryModel* memory,
        bool transpose = false);

    // ========================================================================
    // Statistics
    // ========================================================================

    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }

private:
    Stats stats_;

    /// Copy 2D block without transformation
    void copy_block(
        MemoryRegion* src_region, Size src_offset, uint32_t src_stride,
        MemoryRegion* dst_region, Size dst_offset, uint32_t dst_stride,
        uint32_t height, uint32_t width, uint32_t element_size);

    /// Copy 2D block with transpose (row -> col, col -> row)
    void copy_block_transpose(
        MemoryRegion* src_region, Size src_offset, uint32_t src_stride,
        MemoryRegion* dst_region, Size dst_offset, uint32_t dst_stride,
        uint32_t height, uint32_t width, uint32_t element_size);
};

} // namespace sw::kpu::behavioral
