/**
 * @file program_executor_interface.hpp
 * @brief Abstract interface for DMProgram executors supporting fidelity switching
 *
 * This interface enables switching between execution fidelities:
 * - BEHAVIORAL: Instant execution with correct numerical results
 * - TRANSACTIONAL: Behavioral + timing overlay with cycle estimates
 *
 * Usage:
 *   auto exec = create_program_executor(SimulationFidelity::TRANSACTIONAL, hw);
 *   exec->load_program(prog, a_base, b_base, c_base);
 *   exec->run();
 *   auto cycles = exec->total_cycles();  // Only meaningful for TRANSACTIONAL
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/models/temporal/memory/l3_tile.hpp>
#include <sw/kpu/models/temporal/memory/l2_bank.hpp>
#include <sw/kpu/models/temporal/memory/l1_buffer.hpp>
#include <sw/memory/external_memory.hpp>

#include <memory>
#include <string>
#include <vector>

namespace sw::kpu::isa {

/**
 * @brief Abstract interface for program executors
 *
 * Common interface for BehavioralProgramExecutor and TransactionalProgramExecutor.
 * Enables fidelity switching via factory function.
 */
class IProgramExecutor {
public:
    virtual ~IProgramExecutor() = default;

    /**
     * @brief Load a DMProgram for execution
     *
     * @param program  The compiled schedule
     * @param a_base   Base address of matrix A in external memory
     * @param b_base   Base address of matrix B in external memory
     * @param c_base   Base address of matrix C in external memory
     */
    virtual void load_program(const DMProgram& program,
                              Address a_base, Address b_base, Address c_base) = 0;

    /**
     * @brief Execute the loaded program to completion
     * @return True if HALT instruction reached
     */
    virtual bool run() = 0;

    /**
     * @brief Get the simulation fidelity level
     */
    virtual SimulationFidelity fidelity() const = 0;

    /**
     * @brief Get the total execution cycles (transactional only)
     * @return Cycle count, or 0 for behavioral execution
     */
    virtual Cycle total_cycles() const = 0;

    /**
     * @brief Get execution statistics as a string
     */
    virtual std::string statistics_string() const = 0;

    /**
     * @brief Export timing trace to Chrome Trace format (transactional only)
     * @param filename Output file path
     * @return True if export succeeded, false for behavioral or on error
     */
    virtual bool export_trace(const std::string& filename) const = 0;
};

/**
 * @brief Hardware context for program execution
 *
 * Contains references to all memory components needed by executors.
 */
struct HardwareContext {
    ExternalMemory& external_memory;
    std::vector<L3Tile>& l3_tiles;
    std::vector<L2Bank>& l2_banks;
    std::vector<L1Buffer>& l1_buffers;
};

/**
 * @brief Create a program executor with the specified fidelity level
 *
 * Factory function for creating executors. Supports:
 * - SimulationFidelity::BEHAVIORAL: Fast functional execution
 * - SimulationFidelity::TRANSACTIONAL: Behavioral + timing overlay
 *
 * @param fidelity  Desired simulation fidelity
 * @param hw        Hardware context with memory components
 * @return Unique pointer to executor, or nullptr if fidelity not supported
 *
 * Example:
 * ```cpp
 * auto exec = create_program_executor(SimulationFidelity::TRANSACTIONAL, hw);
 * exec->load_program(prog, a_base, b_base, c_base);
 * exec->run();
 * std::cout << "Cycles: " << exec->total_cycles() << "\n";
 * exec->export_trace("trace.json");
 * ```
 */
std::unique_ptr<IProgramExecutor> create_program_executor(
    SimulationFidelity fidelity,
    HardwareContext hw);

} // namespace sw::kpu::isa
