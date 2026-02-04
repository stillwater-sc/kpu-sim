/**
 * @file transactional_program_executor.cpp
 * @brief Implementation of TransactionalProgramExecutor
 */

#include <sw/kpu/isa/transactional_program_executor.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace sw::kpu::isa {

// ============================================================================
// ResourceTimeline Implementation
// ============================================================================

ResourceTimeline::ResourceTimeline(uint8_t num_dma, uint8_t num_bm,
                                   uint8_t num_str)
    : dma_available_(num_dma, 0)
    , bm_available_(num_bm, 0)
    , str_available_(num_str, 0)
    , compute_available_(0)
    , makespan_(0) {}

Cycle ResourceTimeline::schedule_dma(uint8_t channel, Cycle earliest,
                                     Cycle duration) {
    if (channel >= dma_available_.size()) channel = 0;
    Cycle start = std::max(earliest, dma_available_[channel]);
    Cycle end = start + duration;
    dma_available_[channel] = end;
    makespan_ = std::max(makespan_, end);
    return end;
}

Cycle ResourceTimeline::schedule_block_mover(uint8_t id, Cycle earliest,
                                             Cycle duration) {
    if (id >= bm_available_.size()) id = 0;
    Cycle start = std::max(earliest, bm_available_[id]);
    Cycle end = start + duration;
    bm_available_[id] = end;
    makespan_ = std::max(makespan_, end);
    return end;
}

Cycle ResourceTimeline::schedule_streamer(uint8_t id, Cycle earliest,
                                          Cycle duration) {
    if (id >= str_available_.size()) id = 0;
    Cycle start = std::max(earliest, str_available_[id]);
    Cycle end = start + duration;
    str_available_[id] = end;
    makespan_ = std::max(makespan_, end);
    return end;
}

Cycle ResourceTimeline::schedule_compute(Cycle earliest, Cycle duration) {
    Cycle start = std::max(earliest, compute_available_);
    Cycle end = start + duration;
    compute_available_ = end;
    makespan_ = std::max(makespan_, end);
    return end;
}

Cycle ResourceTimeline::dma_available(uint8_t channel) const {
    if (channel >= dma_available_.size()) return 0;
    return dma_available_[channel];
}

Cycle ResourceTimeline::block_mover_available(uint8_t id) const {
    if (id >= bm_available_.size()) return 0;
    return bm_available_[id];
}

Cycle ResourceTimeline::streamer_available(uint8_t id) const {
    if (id >= str_available_.size()) return 0;
    return str_available_[id];
}

Cycle ResourceTimeline::compute_available() const {
    return compute_available_;
}

Cycle ResourceTimeline::makespan() const {
    return makespan_;
}

void ResourceTimeline::reset() {
    std::fill(dma_available_.begin(), dma_available_.end(), 0);
    std::fill(bm_available_.begin(), bm_available_.end(), 0);
    std::fill(str_available_.begin(), str_available_.end(), 0);
    compute_available_ = 0;
    makespan_ = 0;
}

// ============================================================================
// Helper function to get opcode name
// ============================================================================

static std::string get_opcode_name(DMOpcode opcode) {
    switch (opcode) {
        case DMOpcode::DMA_LOAD_TILE: return "DMA_LOAD";
        case DMOpcode::DMA_STORE_TILE: return "DMA_STORE";
        case DMOpcode::DMA_PREFETCH_TILE: return "DMA_PREFETCH";
        case DMOpcode::DMA_LOAD_TILE_AUTO: return "DMA_LOAD_AUTO";
        case DMOpcode::DMA_STORE_TILE_AUTO: return "DMA_STORE_AUTO";
        case DMOpcode::BM_MOVE_TILE: return "BM_MOVE";
        case DMOpcode::BM_TRANSPOSE_TILE: return "BM_TRANSPOSE";
        case DMOpcode::BM_WRITEBACK_TILE: return "BM_WRITEBACK";
        case DMOpcode::BM_RESHAPE_TILE: return "BM_RESHAPE";
        case DMOpcode::BM_MOVE_TILE_AUTO: return "BM_MOVE_AUTO";
        case DMOpcode::BM_WRITEBACK_AUTO: return "BM_WRITEBACK_AUTO";
        case DMOpcode::STR_FEED_ROWS: return "STR_FEED_ROWS";
        case DMOpcode::STR_FEED_COLS: return "STR_FEED_COLS";
        case DMOpcode::STR_DRAIN_OUTPUT: return "STR_DRAIN";
        case DMOpcode::STR_BROADCAST_ROW: return "STR_BROADCAST_ROW";
        case DMOpcode::STR_BROADCAST_COL: return "STR_BROADCAST_COL";
        case DMOpcode::STR_FEED_ROWS_AUTO: return "STR_FEED_ROWS_AUTO";
        case DMOpcode::STR_FEED_COLS_AUTO: return "STR_FEED_COLS_AUTO";
        case DMOpcode::STR_DRAIN_AUTO: return "STR_DRAIN_AUTO";
        case DMOpcode::BARRIER: return "BARRIER";
        case DMOpcode::WAIT_DMA: return "WAIT_DMA";
        case DMOpcode::WAIT_BM: return "WAIT_BM";
        case DMOpcode::WAIT_STR: return "WAIT_STR";
        case DMOpcode::SIGNAL: return "SIGNAL";
        case DMOpcode::LOOP_BEGIN: return "LOOP_BEGIN";
        case DMOpcode::LOOP_END: return "LOOP_END";
        case DMOpcode::HALT: return "HALT";
        case DMOpcode::NOP: return "NOP";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// TransactionalProgramExecutor Implementation
// ============================================================================

TransactionalProgramExecutor::TransactionalProgramExecutor(
    BehavioralProgramExecutor::HardwareContext hw,
    const TimingConfig& timing)
    : behavioral_(hw)
    , timing_(timing)
    , timeline_(4, 4, 4)  // 4 DMA channels, 4 BMs, 4 streamers
{}

void TransactionalProgramExecutor::load_program(const DMProgram& program,
                                                 Address a_base, Address b_base,
                                                 Address c_base) {
    program_ = &program;
    behavioral_.load_program(program, a_base, b_base, c_base);

    // Reset timing state
    timeline_.reset();
    events_.clear();
    current_cycle_ = 0;
    last_barrier_cycle_ = 0;
    total_dma_cycles_ = 0;
    total_bm_cycles_ = 0;
    total_str_cycles_ = 0;
    total_compute_cycles_ = 0;
    total_loop_overhead_cycles_ = 0;
    total_loop_iterations_ = 0;
    timing_loop_state_.reset();

    // Extract tile dimensions from program
    if (program.Ti > 0) tile_rows_ = program.Ti;
    if (program.Tj > 0) tile_cols_ = program.Tj;
}

bool TransactionalProgramExecutor::run() {
    if (!program_) return false;

    // Phase 1: Execute behaviorally (moves real data, computes real values)
    // All behavioral operations are instant (memcpy), so we can run to completion first.
    bool halted = behavioral_.run();

    // Phase 2: Compute timing overlay
    // Iterate through instructions following actual execution path with loops.
    // Loop control is modeled with timing overhead for loop machinery.
    const auto& instructions = program_->instructions;
    size_t pc = 0;

    while (pc < instructions.size()) {
        const auto& instr = instructions[pc];

        if (instr.opcode == DMOpcode::HALT) {
            dispatch_with_timing(instr);
            break;
        }

        // Handle loop control with PC manipulation (same logic as behavioral)
        if (instr.opcode == DMOpcode::LOOP_BEGIN) {
            const auto& ops = std::get<LoopOperands>(instr.operands);

            // Add timing overhead for loop begin
            Cycle duration = timing_.loop_begin_latency;
            current_cycle_ += duration;
            total_loop_overhead_cycles_ += duration;
            total_loop_iterations_++;

            // Record event
            record_event("LOOP_BEGIN[" + std::to_string(ops.loop_id) + "]",
                        "loop", current_cycle_ - duration, current_cycle_,
                        300 + ops.loop_id, MatrixID::A, TileCoord{0, 0, 0});

            // Initialize loop counter for timing
            timing_loop_state_.begin_loop(ops.loop_id, ops.loop_count,
                                         ops.index_role, ops.loop_stride, pc);
            ++pc;
            continue;
        }

        if (instr.opcode == DMOpcode::LOOP_END) {
            const auto& ops = std::get<LoopOperands>(instr.operands);

            // Add timing overhead for loop end (check + decrement)
            Cycle duration = timing_.loop_end_latency;

            // Check if loop will continue (branch taken) or exit
            const auto& lc = timing_loop_state_.get_counter(ops.loop_id);
            bool will_continue = lc.active && (lc.count + lc.stride < lc.limit);

            if (will_continue) {
                duration += timing_.loop_branch_taken_latency;
            } else {
                duration += timing_.loop_branch_not_taken_latency;
            }

            current_cycle_ += duration;
            total_loop_overhead_cycles_ += duration;

            // Record event
            record_event("LOOP_END[" + std::to_string(ops.loop_id) + "]",
                        "loop", current_cycle_ - duration, current_cycle_,
                        300 + ops.loop_id, MatrixID::A, TileCoord{0, 0, 0});

            // Execute loop end (may branch back)
            size_t next_pc = timing_loop_state_.end_loop(ops.loop_id, pc);
            if (next_pc != pc + 1) {
                // Loop continues, branch back - count iteration
                total_loop_iterations_++;
            }
            pc = next_pc;
            continue;
        }

        dispatch_with_timing(instr);
        ++pc;
    }

    return halted;
}

void TransactionalProgramExecutor::dispatch_with_timing(
    const DMInstruction& instr) {
    const std::string& label = instr.label;
    Cycle start_cycle = current_cycle_;
    Cycle duration = 0;
    int resource_id = 0;
    std::string category;
    MatrixID matrix = MatrixID::A;
    TileCoord tile{0, 0, 0};

    switch (instr.opcode) {
    case DMOpcode::DMA_LOAD_TILE: {
        const auto& ops = std::get<DMAOperands>(instr.operands);
        matrix = ops.matrix;
        tile = ops.tile;
        Size bytes = get_tile_bytes(instr);
        duration = compute_dma_cycles(bytes);
        uint8_t channel = select_dma_channel(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.dma_available(channel));
        Cycle end = timeline_.schedule_dma(channel, start_cycle, duration);
        resource_id = channel;
        category = "dma";
        total_dma_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::DMA_STORE_TILE: {
        const auto& ops = std::get<DMAOperands>(instr.operands);
        matrix = ops.matrix;
        tile = ops.tile;
        Size bytes = get_tile_bytes(instr);
        duration = compute_dma_cycles(bytes);
        uint8_t channel = select_dma_channel(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.dma_available(channel));
        Cycle end = timeline_.schedule_dma(channel, start_cycle, duration);
        resource_id = channel;
        category = "dma";
        total_dma_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::BM_MOVE_TILE:
    case DMOpcode::BM_TRANSPOSE_TILE:
    case DMOpcode::BM_RESHAPE_TILE: {
        const auto& ops = std::get<BlockMoverOperands>(instr.operands);
        matrix = ops.matrix;
        tile = ops.tile;
        Size bytes = get_tile_bytes(instr);
        duration = compute_block_mover_cycles(bytes);
        uint8_t bm_id = select_block_mover(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.block_mover_available(bm_id));
        Cycle end = timeline_.schedule_block_mover(bm_id, start_cycle, duration);
        resource_id = 100 + bm_id;  // Offset for trace thread ID
        category = "block_mover";
        total_bm_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::BM_WRITEBACK_TILE: {
        const auto& ops = std::get<BlockMoverOperands>(instr.operands);
        matrix = ops.matrix;
        tile = ops.tile;
        Size bytes = get_tile_bytes(instr);
        duration = compute_block_mover_cycles(bytes);
        uint8_t bm_id = select_block_mover(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.block_mover_available(bm_id));
        Cycle end = timeline_.schedule_block_mover(bm_id, start_cycle, duration);
        resource_id = 100 + bm_id;
        category = "block_mover";
        total_bm_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::STR_FEED_ROWS:
    case DMOpcode::STR_FEED_COLS: {
        const auto& ops = std::get<StreamerOperands>(instr.operands);
        matrix = ops.matrix;
        tile = ops.tile;
        Size bytes = ops.height * ops.width * element_size_;
        duration = compute_streamer_cycles(bytes);
        uint8_t str_id = select_streamer(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.streamer_available(str_id));
        Cycle end = timeline_.schedule_streamer(str_id, start_cycle, duration);
        resource_id = 200 + str_id;
        category = "streamer";
        total_str_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::STR_DRAIN_OUTPUT: {
        const auto& ops = std::get<StreamerOperands>(instr.operands);
        matrix = MatrixID::C;  // Drain is always C
        tile = ops.tile;
        Size bytes = ops.height * ops.width * element_size_;
        duration = compute_streamer_cycles(bytes);
        uint8_t str_id = select_streamer(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.streamer_available(str_id));
        Cycle end = timeline_.schedule_streamer(str_id, start_cycle, duration);
        resource_id = 200 + str_id;
        category = "streamer";
        total_str_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::BARRIER: {
        // Barrier: all resources must complete before continuing
        last_barrier_cycle_ = timeline_.makespan();
        current_cycle_ = last_barrier_cycle_;
        category = "sync";
        resource_id = 999;
        duration = 0;
        break;
    }

    case DMOpcode::HALT:
        category = "control";
        resource_id = 999;
        duration = 0;
        break;

    // ========================================================================
    // AUTO addressing opcodes - use current loop state for tile coordinates
    // ========================================================================
    case DMOpcode::DMA_LOAD_TILE_AUTO: {
        const auto& ops = std::get<AutoDMAOperands>(instr.operands);
        matrix = ops.matrix;
        tile = timing_loop_state_.current_tile();
        Size bytes = tile_rows_ * tile_cols_ * element_size_;
        duration = compute_dma_cycles(bytes);
        uint8_t channel = select_dma_channel(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.dma_available(channel));
        Cycle end = timeline_.schedule_dma(channel, start_cycle, duration);
        resource_id = channel;
        category = "dma";
        total_dma_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::DMA_STORE_TILE_AUTO: {
        const auto& ops = std::get<AutoDMAOperands>(instr.operands);
        matrix = ops.matrix;
        tile = timing_loop_state_.current_tile();
        Size bytes = tile_rows_ * tile_cols_ * element_size_;
        duration = compute_dma_cycles(bytes);
        uint8_t channel = select_dma_channel(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.dma_available(channel));
        Cycle end = timeline_.schedule_dma(channel, start_cycle, duration);
        resource_id = channel;
        category = "dma";
        total_dma_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::BM_MOVE_TILE_AUTO:
    case DMOpcode::BM_WRITEBACK_AUTO: {
        const auto& ops = std::get<AutoBlockMoverOperands>(instr.operands);
        matrix = ops.matrix;
        tile = timing_loop_state_.current_tile();
        Size bytes = tile_rows_ * tile_cols_ * element_size_;
        duration = compute_block_mover_cycles(bytes);
        uint8_t bm_id = select_block_mover(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.block_mover_available(bm_id));
        Cycle end = timeline_.schedule_block_mover(bm_id, start_cycle, duration);
        resource_id = 100 + bm_id;
        category = "block_mover";
        total_bm_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::STR_FEED_ROWS_AUTO:
    case DMOpcode::STR_FEED_COLS_AUTO: {
        tile = timing_loop_state_.current_tile();
        matrix = (instr.opcode == DMOpcode::STR_FEED_ROWS_AUTO) ? MatrixID::A : MatrixID::B;
        Size bytes = tile_rows_ * tile_cols_ * element_size_;
        duration = compute_streamer_cycles(bytes);
        uint8_t str_id = select_streamer(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.streamer_available(str_id));
        Cycle end = timeline_.schedule_streamer(str_id, start_cycle, duration);
        resource_id = 200 + str_id;
        category = "streamer";
        total_str_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    case DMOpcode::STR_DRAIN_AUTO: {
        tile = timing_loop_state_.current_tile();
        matrix = MatrixID::C;
        Size bytes = tile_rows_ * tile_cols_ * element_size_;
        duration = compute_streamer_cycles(bytes);
        uint8_t str_id = select_streamer(matrix, tile);
        start_cycle = std::max(last_barrier_cycle_, timeline_.streamer_available(str_id));
        Cycle end = timeline_.schedule_streamer(str_id, start_cycle, duration);
        resource_id = 200 + str_id;
        category = "streamer";
        total_str_cycles_ += duration;
        current_cycle_ = end;
        break;
    }

    // ========================================================================
    // Configuration opcodes - minimal timing (register updates)
    // ========================================================================
    case DMOpcode::SET_BASE:
    case DMOpcode::SET_L3_BASE:
    case DMOpcode::SET_L2_BASE:
    case DMOpcode::SET_STRIDE:
    case DMOpcode::SET_TILE_DIM:
    case DMOpcode::SET_MATRIX_DIM:
    case DMOpcode::SET_TILE_SIZE:
    case DMOpcode::SET_BUFFER:
        // Configuration instructions: 1 cycle to update registers
        duration = 1;
        category = "config";
        resource_id = 999;
        current_cycle_ += duration;
        break;

    // Wait opcodes - synchronization
    case DMOpcode::WAIT_DMA:
    case DMOpcode::WAIT_BM:
    case DMOpcode::WAIT_STR:
        // Wait is like a partial barrier - wait for specific resources
        // For simplicity, treat as minimal sync overhead
        duration = 1;
        category = "sync";
        resource_id = 999;
        current_cycle_ += duration;
        break;

    case DMOpcode::SIGNAL:
        duration = 1;
        category = "sync";
        resource_id = 999;
        current_cycle_ += duration;
        break;

    case DMOpcode::NOP:
        duration = 1;
        category = "control";
        resource_id = 999;
        current_cycle_ += duration;
        break;

    // Loop opcodes are handled in run() before dispatch_with_timing()
    case DMOpcode::LOOP_BEGIN:
    case DMOpcode::LOOP_END:
        // Should not reach here - handled in run()
        break;

    default:
        // Other opcodes - minimal timing
        duration = 1;
        category = "other";
        current_cycle_ += duration;
        break;
    }

    // Record timing event
    if (duration > 0 || instr.opcode == DMOpcode::BARRIER) {
        record_event(label.empty() ? get_opcode_name(instr.opcode) : label,
                     category, start_cycle, start_cycle + duration,
                     resource_id, matrix, tile);
    }
}

Cycle TransactionalProgramExecutor::compute_dma_cycles(Size bytes) const {
    // DMA cycles = startup + transfer
    // Transfer = ceil(bytes / bus_width)
    Cycle transfer = (bytes + timing_.dma_bus_width_bytes - 1) /
                     timing_.dma_bus_width_bytes;
    return timing_.dma_startup_latency + transfer;
}

Cycle TransactionalProgramExecutor::compute_block_mover_cycles(Size bytes) const {
    Cycle transfer = (bytes + timing_.block_mover_bus_width_bytes - 1) /
                     timing_.block_mover_bus_width_bytes;
    return timing_.block_mover_startup_latency + transfer;
}

Cycle TransactionalProgramExecutor::compute_streamer_cycles(Size bytes) const {
    Cycle transfer = (bytes + timing_.streamer_bus_width_bytes - 1) /
                     timing_.streamer_bus_width_bytes;
    return timing_.streamer_startup_latency + transfer;
}

Cycle TransactionalProgramExecutor::compute_matmul_cycles(Size m, Size n,
                                                          Size k) const {
    // Systolic array matmul timing:
    // - Fill time: systolic_size cycles to fill the array
    // - Compute time: k iterations, each produces partial products
    // - Drain time: systolic_size cycles to drain
    //
    // For a 16x16 tile on a 16x16 systolic array:
    // Each k-slice: 16 cycles fill + 16 cycles drain = 32 cycles
    // Total: k * (array_size + array_size) / overlap
    //
    // With pipelining, we can overlap fill and drain:
    // Steady state: 1 cycle per MAC operation after initial fill
    //
    // Simplified model: 2 * m * n * k / (systolic_size^2) cycles
    // This represents the throughput-bound case.

    Size array_size = timing_.systolic_size;
    Size num_ops = 2 * m * n * k;  // 2 for multiply-add
    Size ops_per_cycle = array_size * array_size;  // Fully utilized array

    return (num_ops + ops_per_cycle - 1) / ops_per_cycle;
}

uint8_t TransactionalProgramExecutor::select_dma_channel(
    MatrixID /* matrix */, const TileCoord& tile) const {
    // Simple round-robin based on tile coordinates
    // More sophisticated: use TileLayout for channel assignment
    return static_cast<uint8_t>((tile.ti + tile.tj) % 4);
}

uint8_t TransactionalProgramExecutor::select_block_mover(
    MatrixID /* matrix */, const TileCoord& tile) const {
    return static_cast<uint8_t>((tile.ti + tile.tj) % 4);
}

uint8_t TransactionalProgramExecutor::select_streamer(
    MatrixID /* matrix */, const TileCoord& tile) const {
    return static_cast<uint8_t>((tile.ti + tile.tj) % 4);
}

void TransactionalProgramExecutor::record_event(
    const std::string& name, const std::string& category,
    Cycle start, Cycle end, int resource_id,
    MatrixID matrix, TileCoord tile) {
    TimingEvent event;
    event.name = name;
    event.category = category;
    event.phase = 'X';  // Complete event
    event.timestamp_us = cycles_to_us(start);
    event.duration_us = cycles_to_us(end - start);
    event.process_id = 1;
    event.thread_id = resource_id;
    event.matrix = matrix;
    event.tile = tile;
    events_.push_back(event);
}

Size TransactionalProgramExecutor::get_tile_bytes(
    const DMInstruction& instr) const {
    // Get tile size from instruction operands
    switch (instr.opcode) {
    case DMOpcode::DMA_LOAD_TILE:
    case DMOpcode::DMA_STORE_TILE: {
        const auto& ops = std::get<DMAOperands>(instr.operands);
        if (ops.size_bytes > 0) {
            return ops.size_bytes;
        }
        break;
    }
    case DMOpcode::BM_MOVE_TILE:
    case DMOpcode::BM_TRANSPOSE_TILE:
    case DMOpcode::BM_WRITEBACK_TILE:
    case DMOpcode::BM_RESHAPE_TILE: {
        const auto& ops = std::get<BlockMoverOperands>(instr.operands);
        if (ops.height > 0 && ops.width > 0) {
            return ops.height * ops.width * ops.element_size;
        }
        break;
    }
    case DMOpcode::STR_FEED_ROWS:
    case DMOpcode::STR_FEED_COLS:
    case DMOpcode::STR_DRAIN_OUTPUT: {
        const auto& ops = std::get<StreamerOperands>(instr.operands);
        if (ops.height > 0 && ops.width > 0) {
            return ops.height * ops.width * element_size_;
        }
        break;
    }
    default:
        break;
    }

    // Default: use tile dimensions from program
    return tile_rows_ * tile_cols_ * element_size_;
}

TransactionalProgramExecutor::TimingStats
TransactionalProgramExecutor::get_timing_stats() const {
    TimingStats stats;
    stats.total_cycles = timeline_.makespan();
    stats.dma_cycles = total_dma_cycles_;
    stats.block_mover_cycles = total_bm_cycles_;
    stats.streamer_cycles = total_str_cycles_;
    stats.compute_cycles = total_compute_cycles_;
    stats.loop_overhead_cycles = total_loop_overhead_cycles_;
    stats.loop_iterations = total_loop_iterations_;

    if (stats.total_cycles > 0) {
        // Utilization is (busy cycles) / (total cycles * num_resources)
        // For simplicity, assume 4 of each resource
        stats.dma_utilization = static_cast<double>(total_dma_cycles_) /
                                (stats.total_cycles * 4);
        stats.block_mover_utilization = static_cast<double>(total_bm_cycles_) /
                                        (stats.total_cycles * 4);
        stats.streamer_utilization = static_cast<double>(total_str_cycles_) /
                                     (stats.total_cycles * 4);
        stats.compute_utilization = static_cast<double>(total_compute_cycles_) /
                                    stats.total_cycles;
    }

    return stats;
}

void TransactionalProgramExecutor::export_chrome_trace(
    const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) return;

    out << "{\n";
    out << "  \"traceEvents\": [\n";

    // Emit thread/process name metadata events first
    // These give human-readable names to the numeric thread IDs
    out << "    {\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,"
        << "\"args\":{\"name\":\"KPU Transactional Executor\"}},\n";

    // DMA channels (tid 0-3)
    for (int i = 0; i < 4; ++i) {
        out << "    {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << i
            << ",\"args\":{\"name\":\"DMA Channel " << i << "\"}},\n";
    }

    // BlockMovers (tid 100-103)
    for (int i = 0; i < 4; ++i) {
        out << "    {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (100 + i)
            << ",\"args\":{\"name\":\"BlockMover " << i << "\"}},\n";
    }

    // Streamers (tid 200-203)
    for (int i = 0; i < 4; ++i) {
        out << "    {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (200 + i)
            << ",\"args\":{\"name\":\"Streamer " << i << "\"}},\n";
    }

    // Loop control (tid 300+)
    out << "    {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":300,"
        << "\"args\":{\"name\":\"Loop Control\"}},\n";

    // Sync/Control (tid 999)
    out << "    {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":999,"
        << "\"args\":{\"name\":\"Sync/Barrier\"}},\n";

    // Compute (tid 1000)
    out << "    {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":1000,"
        << "\"args\":{\"name\":\"Compute Fabric\"}}";

    // Now emit the actual trace events
    for (const auto& event : events_) {
        out << ",\n";

        out << "    {"
            << "\"name\":\"" << event.name << "\","
            << "\"cat\":\"" << event.category << "\","
            << "\"ph\":\"" << event.phase << "\","
            << "\"ts\":" << std::fixed << std::setprecision(3)
            << event.timestamp_us << ","
            << "\"dur\":" << event.duration_us << ","
            << "\"pid\":" << event.process_id << ","
            << "\"tid\":" << event.thread_id;

        // Add args with tile info
        out << ",\"args\":{"
            << "\"matrix\":\"" << static_cast<int>(event.matrix) << "\","
            << "\"tile_i\":" << event.tile.ti << ","
            << "\"tile_j\":" << event.tile.tj << ","
            << "\"tile_k\":" << event.tile.tk
            << "}}";
    }

    out << "\n  ],\n";

    // Add metadata
    out << "  \"displayTimeUnit\": \"ns\",\n";
    out << "  \"metadata\": {\n";
    out << "    \"total_cycles\": " << timeline_.makespan() << ",\n";
    out << "    \"loop_overhead_cycles\": " << total_loop_overhead_cycles_ << ",\n";
    out << "    \"loop_iterations\": " << total_loop_iterations_ << ",\n";
    out << "    \"reference_clock_mhz\": " << timing_.reference_clock_mhz << "\n";
    out << "  }\n";
    out << "}\n";
}

std::string TransactionalProgramExecutor::generate_timeline(size_t width) const {
    if (events_.empty()) return "No events recorded\n";

    std::ostringstream oss;

    Cycle total = timeline_.makespan();
    if (total == 0) total = 1;

    // Header
    oss << "Timeline (total: " << total << " cycles)\n";
    oss << std::string(width, '=') << "\n";

    // Scale factor
    double scale = static_cast<double>(width - 20) / total;

    // Group events by resource category
    std::map<std::string, std::vector<const TimingEvent*>> by_category;
    for (const auto& e : events_) {
        by_category[e.category].push_back(&e);
    }

    // Render each category
    for (const auto& [cat, evts] : by_category) {
        oss << std::left << std::setw(15) << cat << " |";

        // Create timeline bar
        std::string bar(width - 20, ' ');
        for (const auto* e : evts) {
            Cycle start_c = static_cast<Cycle>(e->timestamp_us *
                                                timing_.reference_clock_mhz);
            Cycle dur_c = static_cast<Cycle>(e->duration_us *
                                              timing_.reference_clock_mhz);
            size_t start_pos = static_cast<size_t>(start_c * scale);
            size_t len = std::max<size_t>(1, static_cast<size_t>(dur_c * scale));

            if (start_pos < bar.size()) {
                for (size_t i = 0; i < len && start_pos + i < bar.size(); ++i) {
                    bar[start_pos + i] = '#';
                }
            }
        }
        oss << bar << "|\n";
    }

    oss << std::string(width, '=') << "\n";

    // Summary stats
    auto stats = get_timing_stats();
    oss << "DMA: " << stats.dma_cycles << " cycles ("
        << std::fixed << std::setprecision(1) << (stats.dma_utilization * 100)
        << "% util)\n";
    oss << "BM:  " << stats.block_mover_cycles << " cycles ("
        << (stats.block_mover_utilization * 100) << "% util)\n";
    oss << "STR: " << stats.streamer_cycles << " cycles ("
        << (stats.streamer_utilization * 100) << "% util)\n";
    if (stats.loop_overhead_cycles > 0) {
        oss << "LOOP: " << stats.loop_overhead_cycles << " cycles ("
            << stats.loop_iterations << " iterations)\n";
    }

    return oss.str();
}

} // namespace sw::kpu::isa
