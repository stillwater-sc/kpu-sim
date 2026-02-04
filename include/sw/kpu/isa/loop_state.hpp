/**
 * @file loop_state.hpp
 * @brief Hardware loop state management for KPU ISA
 *
 * The KPU supports hardware loops with automatic address generation.
 * This file defines the LoopState class that tracks loop counters
 * and provides tile index values for AUTO addressing modes.
 *
 * Loop Structure:
 * - 8 loop registers (LC[0..7]) for nested loops
 * - Each loop can be bound to an IndexRole (TI, TJ, TK, NONE)
 * - Current tile indices are extracted from innermost bound loops
 *
 * Example:
 *   LOOP_BEGIN 0, 256, TI    ; LC[0] drives ti, counts 0..255
 *   LOOP_BEGIN 1, 512, TJ    ; LC[1] drives tj, counts 0..511
 *   LOOP_BEGIN 2, 64, TK     ; LC[2] drives tk, counts 0..63
 *     DMA_LOAD_TILE_AUTO A, 0, BUF_0  ; addr = base_A + ti*stride_i + tk*stride_k
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <cstdint>
#include <array>
#include <stack>
#include <stdexcept>

namespace sw::kpu::isa {

/**
 * @brief State of a single hardware loop counter
 */
struct LoopCounter {
    uint16_t count = 0;         ///< Current iteration (counts up from 0)
    uint16_t limit = 0;         ///< Total iterations
    uint16_t stride = 1;        ///< Index increment per iteration
    IndexRole role = IndexRole::NONE;  ///< Which tile index this loop drives
    bool active = false;        ///< Is this loop currently executing
    size_t begin_pc = 0;        ///< PC of LOOP_BEGIN (for branch back)

    /// Reset to initial state
    void reset() {
        count = 0;
        limit = 0;
        stride = 1;
        role = IndexRole::NONE;
        active = false;
        begin_pc = 0;
    }

    /// Start a new loop
    void begin(uint16_t iterations, IndexRole idx_role, uint16_t loop_stride, size_t pc) {
        count = 0;
        limit = iterations;
        stride = loop_stride;
        role = idx_role;
        active = true;
        begin_pc = pc;
    }

    /// Increment and check if loop should continue
    /// @return true if loop should continue, false if done
    bool increment() {
        if (!active) return false;
        count += stride;
        if (count >= limit) {
            active = false;
            return false;
        }
        return true;
    }

    /// Get current index value (scaled by stride)
    uint16_t current_index() const {
        return count;
    }

    /// Check if loop has completed
    bool done() const {
        return !active || count >= limit;
    }
};

/**
 * @brief Hardware loop state manager
 *
 * Manages 8 nested hardware loop counters and provides
 * current tile indices (ti, tj, tk) for AUTO addressing.
 */
class LoopState {
public:
    static constexpr size_t MAX_LOOPS = 8;

    LoopState() {
        for (auto& lc : counters_) {
            lc.reset();
        }
    }

    /**
     * @brief Begin a hardware loop
     * @param loop_id Loop register index (0-7)
     * @param iterations Total iterations
     * @param role Index role (TI, TJ, TK, NONE)
     * @param stride Index increment per iteration
     * @param pc Current program counter
     * @throws std::out_of_range if loop_id >= MAX_LOOPS
     */
    void begin_loop(uint8_t loop_id, uint16_t iterations, IndexRole role,
                    uint16_t stride, size_t pc) {
        if (loop_id >= MAX_LOOPS) {
            throw std::out_of_range("Loop ID must be 0-7");
        }
        counters_[loop_id].begin(iterations, role, stride, pc);
        active_loops_.push(loop_id);
    }

    /**
     * @brief End a hardware loop
     * @param loop_id Loop register index
     * @param current_pc Current program counter
     * @return PC to jump to (begin_pc if continuing, current_pc+1 if done)
     */
    size_t end_loop(uint8_t loop_id, size_t current_pc) {
        if (loop_id >= MAX_LOOPS) {
            throw std::out_of_range("Loop ID must be 0-7");
        }

        auto& lc = counters_[loop_id];
        if (lc.increment()) {
            // Loop continues - branch back to begin
            return lc.begin_pc + 1;  // Skip LOOP_BEGIN, start at first body instruction
        } else {
            // Loop done - pop from active stack and continue
            if (!active_loops_.empty() && active_loops_.top() == loop_id) {
                active_loops_.pop();
            }
            return current_pc + 1;  // Continue to next instruction
        }
    }

    /**
     * @brief Get current value for a tile index role
     * @param role Which index (TI, TJ, TK)
     * @return Current index value from innermost bound loop
     *
     * Scans active loops from innermost to outermost, returning
     * the value of the first loop bound to the requested role.
     */
    uint16_t get_index(IndexRole role) const {
        if (role == IndexRole::NONE) return 0;

        // Scan loops to find one bound to this role
        // Priority: higher loop_id = more nested (inner)
        for (int i = MAX_LOOPS - 1; i >= 0; --i) {
            if (counters_[i].active && counters_[i].role == role) {
                return counters_[i].current_index();
            }
        }
        return 0;  // No loop bound to this role
    }

    /// Get current ti (output row index)
    uint16_t ti() const { return get_index(IndexRole::TI); }

    /// Get current tj (output column index)
    uint16_t tj() const { return get_index(IndexRole::TJ); }

    /// Get current tk (reduction index)
    uint16_t tk() const { return get_index(IndexRole::TK); }

    /// Get current tile coordinate
    TileCoord current_tile() const {
        return TileCoord{ti(), tj(), tk()};
    }

    /// Get loop counter state
    const LoopCounter& get_counter(uint8_t loop_id) const {
        if (loop_id >= MAX_LOOPS) {
            throw std::out_of_range("Loop ID must be 0-7");
        }
        return counters_[loop_id];
    }

    /// Check if any loop is active
    bool any_active() const {
        for (const auto& lc : counters_) {
            if (lc.active) return true;
        }
        return false;
    }

    /// Get number of active loops
    size_t active_count() const {
        size_t count = 0;
        for (const auto& lc : counters_) {
            if (lc.active) ++count;
        }
        return count;
    }

    /// Reset all loop state
    void reset() {
        for (auto& lc : counters_) {
            lc.reset();
        }
        while (!active_loops_.empty()) {
            active_loops_.pop();
        }
    }

private:
    std::array<LoopCounter, MAX_LOOPS> counters_;
    std::stack<uint8_t> active_loops_;  ///< Stack of active loop IDs (outermost to innermost)
};

} // namespace sw::kpu::isa
