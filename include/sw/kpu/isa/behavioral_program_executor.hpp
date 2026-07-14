/**
 * @file behavioral_program_executor.hpp
 * @brief Behavioral (functional) executor for Data Movement ISA programs
 *
 * Interprets a DMProgram by executing each instruction as an instant data
 * movement operation on real memory components. All transfers are memcpy;
 * barriers are no-ops. Compute fires when both A and B tiles have been
 * streamed to L1 buffers.
 *
 * This executor proves that a schedule is functionally correct: the right
 * tiles reach the right places in the right order, and the accumulated
 * result matches a reference computation.
 *
 * Memory components used (from temporal tier, holding real bytes):
 *   ExternalMemory  — DRAM (dense or sparse backing store)
 *   L3Tile          — on-chip L3 buffer slots
 *   L2Bank          — on-chip L2 working memory
 *   L1Buffer        — compute-fabric streaming buffers
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/isa/register_file.hpp>
#include <sw/kpu/models/temporal/memory/l3_tile.hpp>
#include <sw/kpu/models/temporal/memory/l2_bank.hpp>
#include <sw/kpu/models/temporal/memory/l1_buffer.hpp>
#include <sw/memory/external_memory.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

namespace sw::kpu::isa {

/**
 * @brief Behavioral executor for DMProgram instruction streams
 *
 * Executes a DMProgram using instant data movement (memcpy between memory
 * components) and a triple-loop matmul for compute. No timing, no queues,
 * no credit tracking — pure functional correctness.
 */
class BehavioralProgramExecutor {
public:
    /**
     * @brief Hardware context — references to memory components
     */
    struct HardwareContext {
        ExternalMemory& external_memory;
        std::vector<L3Tile>& l3_tiles;
        std::vector<L2Bank>& l2_banks;
        std::vector<L1Buffer>& l1_buffers;
    };

    /**
     * @brief Execution statistics
     */
    struct Statistics {
        uint64_t instructions_executed = 0;
        uint64_t dma_loads = 0;
        uint64_t dma_stores = 0;
        uint64_t bm_moves = 0;
        uint64_t bm_writebacks = 0;
        uint64_t str_feeds = 0;
        uint64_t str_drains = 0;
        uint64_t compute_invocations = 0;
        uint64_t barriers = 0;
        uint64_t loop_iterations = 0;
        uint64_t config_instructions = 0;
        Size bytes_loaded = 0;
        Size bytes_stored = 0;
    };

    explicit BehavioralProgramExecutor(HardwareContext hw);
    ~BehavioralProgramExecutor() = default;

    /**
     * @brief Load a DMProgram for execution
     *
     * @param program  The compiled schedule
     * @param a_base   Base address of matrix A in external memory
     * @param b_base   Base address of matrix B in external memory
     * @param c_base   Base address of matrix C in external memory
     */
    void load_program(const DMProgram& program,
                      Address a_base, Address b_base, Address c_base);

    /**
     * @brief Execute the loaded program to completion
     *
     * Walks every instruction in program order. Each instruction is
     * dispatched as an instant operation. Returns true if HALT reached.
     */
    bool run();

    /**
     * @brief Access execution statistics
     */
    const Statistics& statistics() const { return stats_; }

    /**
     * @brief Access external memory (for reading results)
     */
    ExternalMemory& memory() { return hw_.external_memory; }

private:
    HardwareContext hw_;
    const DMProgram* program_ = nullptr;
    Statistics stats_;
    ISARegisterFile regs_;  ///< Loop state and address generator

    // Base addresses for A, B, C in external memory
    Address a_base_ = 0;
    Address b_base_ = 0;
    Address c_base_ = 0;

    // L1 buffer layout: buffer 0 = A tile, buffer 1 = B tile
    // After both are loaded, compute fires. Result accumulates in a
    // separate region of buffer 0 (or a dedicated accumulator buffer).
    static constexpr uint8_t L1_A_BUFFER = 0;
    static constexpr uint8_t L1_B_BUFFER = 1;

    // Tracking: did we feed A and B rows/cols this iteration?
    bool a_fed_ = false;
    bool b_fed_ = false;

    // Current tile being accumulated (for output-stationary)
    uint16_t acc_ti_ = 0;
    uint16_t acc_tj_ = 0;
    bool accumulating_ = false;

    // Dispatch methods — one per opcode category
    void dispatch(const DMInstruction& instr, size_t pc);
    void dispatch_dma_load(const DMAOperands& ops);
    void dispatch_dma_store(const DMAOperands& ops);
    void dispatch_bm_move(const BlockMoverOperands& ops);
    void dispatch_bm_writeback(const BlockMoverOperands& ops);
    void dispatch_str_feed_rows(const StreamerOperands& ops);
    void dispatch_str_feed_cols(const StreamerOperands& ops);
    void dispatch_str_broadcast(const StreamerOperands& ops);
    void dispatch_str_drain(const StreamerOperands& ops);
    void dispatch_barrier();

    // AUTO addressing dispatch methods
    void dispatch_dma_load_auto(const AutoDMAOperands& ops);
    void dispatch_dma_store_auto(const AutoDMAOperands& ops);
    void dispatch_bm_move_auto(const AutoBlockMoverOperands& ops);
    void dispatch_bm_writeback_auto(const AutoBlockMoverOperands& ops);
    void dispatch_str_feed_rows_auto(const AutoStreamerOperands& ops);
    void dispatch_str_feed_cols_auto(const AutoStreamerOperands& ops);
    void dispatch_str_drain_auto(const AutoStreamerOperands& ops);

    /**
     * @brief Fire matmul compute on current L1 tile data
     *
     * Reads A tile from L1 buffer 0, B tile from L1 buffer 1,
     * accumulates C += A × B into L1 accumulator region.
     */
    void fire_compute();

    /**
     * @brief Apply a Vector Engine elementwise operation over L1 buffers
     *        (issue #100): out = op(a[, b or scalar]) elementwise, over
     *        the program's Ti x Tj tile
     */
    void execute_ve_elementwise(const VEOperands& ops);
    void execute_ve_reduce(const VEReduceOperands& ops);

    /**
     * @brief Resolve external memory address for a tile
     */
    Address resolve_address(MatrixID matrix, const TileCoord& tile) const;

    /**
     * @brief Get tile dimensions for a given matrix
     */
    void get_tile_dims(MatrixID matrix, Size& rows, Size& cols) const;
};

} // namespace sw::kpu::isa
