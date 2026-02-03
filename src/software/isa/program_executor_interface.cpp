/**
 * @file program_executor_interface.cpp
 * @brief Implementation of IProgramExecutor wrappers and factory
 */

#include <sw/kpu/isa/program_executor_interface.hpp>
#include <sw/kpu/isa/behavioral_program_executor.hpp>
#include <sw/kpu/isa/transactional_program_executor.hpp>

#include <sstream>

namespace sw::kpu::isa {

// ============================================================================
// Behavioral Executor Wrapper
// ============================================================================

class BehavioralExecutorWrapper : public IProgramExecutor {
public:
    explicit BehavioralExecutorWrapper(HardwareContext hw)
        : executor_(BehavioralProgramExecutor::HardwareContext{
              hw.external_memory, hw.l3_tiles, hw.l2_banks, hw.l1_buffers}) {}

    void load_program(const DMProgram& program,
                      Address a_base, Address b_base, Address c_base) override {
        executor_.load_program(program, a_base, b_base, c_base);
    }

    bool run() override {
        return executor_.run();
    }

    SimulationFidelity fidelity() const override {
        return SimulationFidelity::BEHAVIORAL;
    }

    Cycle total_cycles() const override {
        return 0;  // Behavioral execution is instant
    }

    std::string statistics_string() const override {
        const auto& stats = executor_.statistics();
        std::ostringstream oss;
        oss << "Behavioral Execution Statistics:\n"
            << "  Instructions executed: " << stats.instructions_executed << "\n"
            << "  DMA loads:            " << stats.dma_loads << "\n"
            << "  DMA stores:           " << stats.dma_stores << "\n"
            << "  BlockMover moves:     " << stats.bm_moves << "\n"
            << "  BlockMover writebacks:" << stats.bm_writebacks << "\n"
            << "  Streamer feeds:       " << stats.str_feeds << "\n"
            << "  Streamer drains:      " << stats.str_drains << "\n"
            << "  Compute invocations:  " << stats.compute_invocations << "\n"
            << "  Barriers:             " << stats.barriers << "\n"
            << "  Bytes loaded:         " << stats.bytes_loaded << "\n"
            << "  Bytes stored:         " << stats.bytes_stored << "\n";
        return oss.str();
    }

    bool export_trace(const std::string& /* filename */) const override {
        return false;  // No timing trace in behavioral mode
    }

private:
    BehavioralProgramExecutor executor_;
};

// ============================================================================
// Transactional Executor Wrapper
// ============================================================================

class TransactionalExecutorWrapper : public IProgramExecutor {
public:
    explicit TransactionalExecutorWrapper(HardwareContext hw)
        : executor_(BehavioralProgramExecutor::HardwareContext{
              hw.external_memory, hw.l3_tiles, hw.l2_banks, hw.l1_buffers}) {}

    void load_program(const DMProgram& program,
                      Address a_base, Address b_base, Address c_base) override {
        executor_.load_program(program, a_base, b_base, c_base);
    }

    bool run() override {
        return executor_.run();
    }

    SimulationFidelity fidelity() const override {
        return SimulationFidelity::TRANSACTIONAL;
    }

    Cycle total_cycles() const override {
        return executor_.makespan();
    }

    std::string statistics_string() const override {
        const auto& bstats = executor_.behavioral_stats();
        auto tstats = executor_.get_timing_stats();

        std::ostringstream oss;
        oss << "Transactional Execution Statistics:\n"
            << "  Total cycles:         " << tstats.total_cycles << "\n"
            << "  DMA cycles:           " << tstats.dma_cycles << "\n"
            << "  BlockMover cycles:    " << tstats.block_mover_cycles << "\n"
            << "  Streamer cycles:      " << tstats.streamer_cycles << "\n"
            << "  Compute cycles:       " << tstats.compute_cycles << "\n"
            << "\n"
            << "Behavioral Statistics:\n"
            << "  Instructions executed: " << bstats.instructions_executed << "\n"
            << "  DMA loads:            " << bstats.dma_loads << "\n"
            << "  DMA stores:           " << bstats.dma_stores << "\n"
            << "  Compute invocations:  " << bstats.compute_invocations << "\n"
            << "  Bytes loaded:         " << bstats.bytes_loaded << "\n"
            << "  Bytes stored:         " << bstats.bytes_stored << "\n";
        return oss.str();
    }

    bool export_trace(const std::string& filename) const override {
        executor_.export_chrome_trace(filename);
        return true;
    }

private:
    TransactionalProgramExecutor executor_;
};

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<IProgramExecutor> create_program_executor(
    SimulationFidelity fidelity,
    HardwareContext hw) {

    switch (fidelity) {
    case SimulationFidelity::BEHAVIORAL:
        return std::make_unique<BehavioralExecutorWrapper>(hw);

    case SimulationFidelity::TRANSACTIONAL:
        return std::make_unique<TransactionalExecutorWrapper>(hw);

    case SimulationFidelity::CYCLE_ACCURATE:
        // Cycle-accurate execution requires different infrastructure
        // (ProgramExecutor with temporal components). Not yet integrated.
        return nullptr;

    default:
        return nullptr;
    }
}

} // namespace sw::kpu::isa
