/**
 * @file program_executor.hpp
 * @brief Executor for Data Movement ISA programs
 *
 * The ProgramExecutor interprets Data Movement ISA instructions and
 * drives the hardware components (DMA, BlockMover, Streamer) to execute
 * the system-level schedule.
 *
 * In Domain Flow Architecture:
 * - The executor configures data movement hardware
 * - The compute fabric reacts to arriving data streams
 * - Synchronization is through barriers and completion signals
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/components/dma_engine.hpp>
#include <sw/kpu/components/block_mover.hpp>
#include <sw/kpu/components/streamer.hpp>
#include <sw/kpu/components/compute_fabric.hpp>
#include <sw/kpu/components/page_buffer.hpp>
#include <sw/kpu/components/l1_buffer.hpp>
#include <sw/kpu/components/l3_tile.hpp>
#include <sw/kpu/components/l2_bank.hpp>
#include <sw/memory/external_memory.hpp>
#include <sw/trace/trace_logger.hpp>

#include <memory>
#include <vector>
#include <queue>
#include <unordered_set>
#include <functional>

namespace sw::kpu::isa {

/**
 * @brief Execution state for tracking program progress
 */
enum class ExecutionState {
    IDLE,           // Not running
    RUNNING,        // Executing instructions
    WAITING,        // Waiting for hardware completion
    COMPLETED,      // Program finished
    ERROR           // Execution error
};

/**
 * @brief Executor for Data Movement programs
 *
 * Interprets DMProgram instructions and coordinates hardware components
 * to execute the system-level schedule.
 */
class ProgramExecutor {
public:
    /**
     * @brief Hardware context for execution
     *
     * References to all hardware components that the executor controls.
     */
    struct HardwareContext {
        // External memory (technology-agnostic)
        std::vector<ExternalMemory>* host_memory;
        std::vector<ExternalMemory>* external_memory;

        // Cache hierarchy
        std::vector<L3Tile>* l3_tiles;
        std::vector<L2Bank>* l2_banks;
        std::vector<L1Buffer>* l1_buffers;      // Compute fabric L1 buffers
        std::vector<PageBuffer>* page_buffers;   // Memory controller page buffers

        // Data movement engines
        std::vector<DMAEngine>* dma_engines;
        std::vector<BlockMover>* block_movers;
        std::vector<Streamer>* streamers;

        // Compute fabric
        ComputeFabric* compute_fabric;

        // Tracing
        trace::TraceLogger* trace_logger;
    };

    /**
     * @brief Execution statistics
     */
    struct Statistics {
        uint64_t total_cycles;
        uint64_t instructions_executed;
        uint64_t dma_operations;
        uint64_t block_mover_operations;
        uint64_t streamer_operations;
        uint64_t barriers_hit;
        uint64_t external_bytes_transferred;
        uint64_t l3_bytes_transferred;
        uint64_t l2_bytes_transferred;

        Statistics() : total_cycles(0), instructions_executed(0),
                      dma_operations(0), block_mover_operations(0),
                      streamer_operations(0), barriers_hit(0),
                      external_bytes_transferred(0), l3_bytes_transferred(0),
                      l2_bytes_transferred(0) {}
    };

    /**
     * @brief Callback for instruction completion
     */
    using CompletionCallback = std::function<void(uint32_t instruction_id)>;

public:
    explicit ProgramExecutor(HardwareContext& hw);
    ~ProgramExecutor() = default;

    /**
     * @brief Load a program for execution
     *
     * @param program The Data Movement program to execute
     * @param a_base Base address for matrix A in external memory
     * @param b_base Base address for matrix B in external memory
     * @param c_base Base address for matrix C in external memory
     */
    void load_program(const DMProgram& program,
                     Address a_base, Address b_base, Address c_base);

    /**
     * @brief Execute one cycle of the program
     *
     * Advances the program counter, issues instructions, and updates
     * hardware state. Returns true if program is still running.
     */
    bool step();

    /**
     * @brief Run program to completion
     *
     * @param max_cycles Maximum cycles to run (0 = unlimited)
     * @return true if completed normally, false if max_cycles reached
     */
    bool run(uint64_t max_cycles = 0);

    /**
     * @brief Reset executor state
     */
    void reset();

    // State queries
    ExecutionState state() const { return state_; }
    bool is_running() const { return state_ == ExecutionState::RUNNING ||
                                    state_ == ExecutionState::WAITING; }
    bool is_completed() const { return state_ == ExecutionState::COMPLETED; }

    // Statistics
    const Statistics& statistics() const { return stats_; }
    uint64_t current_cycle() const { return current_cycle_; }
    size_t program_counter() const { return pc_; }

    // Callbacks
    void set_completion_callback(CompletionCallback cb) { completion_cb_ = cb; }

private:
    HardwareContext& hw_;
    const DMProgram* program_;
    ExecutionState state_;
    Statistics stats_;

    // Program state
    size_t pc_;                     // Program counter
    uint64_t current_cycle_;        // Current simulation cycle

    // Memory base addresses
    Address a_base_;
    Address b_base_;
    Address c_base_;

    // Pending operations tracking
    std::unordered_set<uint32_t> pending_dma_;
    std::unordered_set<uint32_t> pending_bm_;
    std::unordered_set<uint32_t> pending_str_;

    // Completion callback
    CompletionCallback completion_cb_;

    // Instruction dispatch
    bool dispatch_instruction(const DMInstruction& instr);
    bool dispatch_dma(const DMInstruction& instr);
    bool dispatch_block_mover(const DMInstruction& instr);
    bool dispatch_streamer(const DMInstruction& instr);
    bool dispatch_sync(const DMInstruction& instr);

    // Hardware update
    void update_hardware();
    bool all_operations_complete() const;

    // Address resolution
    Address resolve_external_address(MatrixID matrix, const TileCoord& tile) const;
};

/**
 * @brief Utility to print program disassembly
 */
void disassemble_program(const DMProgram& program, std::ostream& out);

/**
 * @brief Utility to validate program before execution
 */
bool validate_program(const DMProgram& program, std::string& error);

} // namespace sw::kpu::isa
