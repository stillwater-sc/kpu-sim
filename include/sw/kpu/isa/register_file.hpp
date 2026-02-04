/**
 * @file register_file.hpp
 * @brief Unified register file for KPU ISA execution
 *
 * The ISARegisterFile combines:
 * - Loop state (8 loop counters with index role bindings)
 * - Address generator (base addresses, strides, dimensions)
 *
 * This provides a single interface for the executor to:
 * - Process SET_* configuration instructions
 * - Manage LOOP_BEGIN/LOOP_END execution
 * - Compute addresses for AUTO addressing modes
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/isa/loop_state.hpp>
#include <sw/kpu/isa/address_generator.hpp>

namespace sw::kpu::isa {

/**
 * @brief Unified register file for ISA execution
 *
 * Provides the execution context for loop machinery and
 * address generation, handling both configuration instructions
 * and runtime address computation.
 */
class ISARegisterFile {
public:
    ISARegisterFile() = default;

    // ========================================================================
    // Loop State Accessors
    // ========================================================================

    /// Get loop state (const)
    const LoopState& loops() const { return loop_state_; }

    /// Get loop state (mutable)
    LoopState& loops() { return loop_state_; }

    /// Get current tile coordinate from loop indices
    TileCoord current_tile() const { return loop_state_.current_tile(); }

    /// Get current ti index
    uint16_t ti() const { return loop_state_.ti(); }

    /// Get current tj index
    uint16_t tj() const { return loop_state_.tj(); }

    /// Get current tk index
    uint16_t tk() const { return loop_state_.tk(); }

    // ========================================================================
    // Address Generator Accessors
    // ========================================================================

    /// Get address generator (const)
    const AddressGenerator& addr() const { return addr_gen_; }

    /// Get address generator (mutable)
    AddressGenerator& addr() { return addr_gen_; }

    // ========================================================================
    // Configuration Instruction Handlers
    // ========================================================================

    /// Handle SET_BASE instruction
    void exec_set_base(const BaseAddressOperands& ops) {
        addr_gen_.set_base(ops.matrix, ops.base_addr);
    }

    /// Handle SET_L3_BASE instruction
    void exec_set_l3_base(const L3BaseOperands& ops) {
        addr_gen_.set_l3_base(ops.matrix, ops.l3_offset);
    }

    /// Handle SET_L2_BASE instruction
    void exec_set_l2_base(const L2BaseOperands& ops) {
        addr_gen_.set_l2_base(ops.matrix, ops.l2_offset);
    }

    /// Handle SET_STRIDE instruction
    void exec_set_stride(const StrideConfigOperands& ops) {
        addr_gen_.set_strides(ops.matrix, ops.row_stride,
                             ops.tile_i_stride, ops.tile_j_stride);
    }

    /// Handle SET_TILE_DIM instruction
    void exec_set_tile_dim(const TileDimOperands& ops) {
        addr_gen_.set_tile_dim(ops.Ti, ops.Tj, ops.Tk, ops.element_size);
    }

    /// Handle SET_MATRIX_DIM instruction
    void exec_set_matrix_dim(const MatrixDimOperands& ops) {
        addr_gen_.set_matrix_dim(ops.matrix, ops.rows, ops.cols);
    }

    // ========================================================================
    // Loop Control Instruction Handlers
    // ========================================================================

    /**
     * @brief Handle LOOP_BEGIN instruction
     * @param ops Loop operands
     * @param pc Current program counter
     */
    void exec_loop_begin(const LoopOperands& ops, size_t pc) {
        loop_state_.begin_loop(ops.loop_id, ops.loop_count,
                              ops.index_role, ops.loop_stride, pc);
    }

    /**
     * @brief Handle LOOP_END instruction
     * @param ops Loop operands (only loop_id used)
     * @param current_pc Current program counter
     * @return Next PC (branch back if continuing, current_pc+1 if done)
     */
    size_t exec_loop_end(const LoopOperands& ops, size_t current_pc) {
        return loop_state_.end_loop(ops.loop_id, current_pc);
    }

    // ========================================================================
    // Address Computation for AUTO Modes
    // ========================================================================

    /**
     * @brief Compute external memory address for DMA_*_AUTO
     * @param mat Matrix identifier
     * @return Computed address using current loop indices
     */
    Address compute_ext_addr(MatrixID mat) const {
        return addr_gen_.compute_ext_addr(mat, loop_state_);
    }

    /**
     * @brief Compute L3 offset for tile
     * @param mat Matrix identifier
     * @return L3 buffer offset
     */
    Address compute_l3_offset(MatrixID mat) const {
        return addr_gen_.compute_l3_offset(mat, loop_state_);
    }

    /**
     * @brief Compute L2 offset for tile
     * @param mat Matrix identifier
     * @return L2 buffer offset
     */
    Address compute_l2_offset(MatrixID mat) const {
        return addr_gen_.compute_l2_offset(mat, loop_state_);
    }

    /// Get tile dimensions
    const TileDimensions& tile_dim() const {
        return addr_gen_.tile_dim();
    }

    /// Get tile size in bytes
    Size tile_bytes() const {
        return addr_gen_.tile_bytes();
    }

    // ========================================================================
    // State Management
    // ========================================================================

    /// Reset all state
    void reset() {
        loop_state_.reset();
        addr_gen_.reset();
    }

    /// Check if any loop is active
    bool in_loop() const {
        return loop_state_.any_active();
    }

private:
    LoopState loop_state_;
    AddressGenerator addr_gen_;
};

} // namespace sw::kpu::isa
