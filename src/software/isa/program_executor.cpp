/**
 * @file program_executor.cpp
 * @brief Implementation of Data Movement ISA program executor
 */

#include <sw/kpu/isa/program_executor.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace sw::kpu::isa {

ProgramExecutor::ProgramExecutor(HardwareContext& hw)
    : hw_(hw)
    , program_(nullptr)
    , state_(ExecutionState::IDLE)
    , pc_(0)
    , current_cycle_(0)
    , a_base_(0)
    , b_base_(0)
    , c_base_(0) {}

void ProgramExecutor::load_program(const DMProgram& program,
                                   Address a_base, Address b_base, Address c_base) {
    program_ = &program;
    a_base_ = a_base;
    b_base_ = b_base;
    c_base_ = c_base;

    // Reset state
    pc_ = 0;
    current_cycle_ = 0;
    state_ = ExecutionState::RUNNING;
    stats_ = Statistics();

    pending_dma_.clear();
    pending_bm_.clear();
    pending_str_.clear();

    // Set cycle counters on hardware
    for (auto& dma : *hw_.dma_engines) {
        dma.set_current_cycle(0);
    }
    for (auto& bm : *hw_.block_movers) {
        bm.set_cycle(0);
    }
    for (auto& str : *hw_.streamers) {
        str.set_cycle(0);
    }
}

bool ProgramExecutor::step() {
    if (state_ != ExecutionState::RUNNING && state_ != ExecutionState::WAITING) {
        return false;
    }

    // Update hardware components
    update_hardware();

    // If waiting for operations, check if complete
    if (state_ == ExecutionState::WAITING) {
        if (all_operations_complete()) {
            state_ = ExecutionState::RUNNING;
        } else {
            current_cycle_++;
            return true;  // Still running, waiting
        }
    }

    // Fetch and dispatch instruction
    if (pc_ < program_->instructions.size()) {
        const auto& instr = program_->instructions[pc_];

        bool dispatched = dispatch_instruction(instr);
        if (dispatched) {
            stats_.instructions_executed++;
            pc_++;

            // Check for HALT
            if (instr.opcode == DMOpcode::HALT) {
                state_ = ExecutionState::COMPLETED;
                stats_.total_cycles = current_cycle_;
                return false;
            }
        }
    } else {
        // End of program
        state_ = ExecutionState::COMPLETED;
        stats_.total_cycles = current_cycle_;
        return false;
    }

    current_cycle_++;
    return true;
}

bool ProgramExecutor::run(uint64_t max_cycles) {
    while (is_running()) {
        if (max_cycles > 0 && current_cycle_ >= max_cycles) {
            return false;  // Timeout
        }
        step();
    }
    return state_ == ExecutionState::COMPLETED;
}

void ProgramExecutor::reset() {
    program_ = nullptr;
    state_ = ExecutionState::IDLE;
    pc_ = 0;
    current_cycle_ = 0;
    stats_ = Statistics();
    pending_dma_.clear();
    pending_bm_.clear();
    pending_str_.clear();
}

bool ProgramExecutor::dispatch_instruction(const DMInstruction& instr) {
    switch (instr.opcode) {
        // DMA operations
        case DMOpcode::DMA_LOAD_TILE:
        case DMOpcode::DMA_STORE_TILE:
        case DMOpcode::DMA_PREFETCH_TILE:
            return dispatch_dma(instr);

        // Block Mover operations
        case DMOpcode::BM_MOVE_TILE:
        case DMOpcode::BM_TRANSPOSE_TILE:
        case DMOpcode::BM_WRITEBACK_TILE:
        case DMOpcode::BM_RESHAPE_TILE:
            return dispatch_block_mover(instr);

        // Streamer operations
        case DMOpcode::STR_FEED_ROWS:
        case DMOpcode::STR_FEED_COLS:
        case DMOpcode::STR_DRAIN_OUTPUT:
        case DMOpcode::STR_BROADCAST_ROW:
        case DMOpcode::STR_BROADCAST_COL:
            return dispatch_streamer(instr);

        // Synchronization
        case DMOpcode::BARRIER:
        case DMOpcode::WAIT_DMA:
        case DMOpcode::WAIT_BM:
        case DMOpcode::WAIT_STR:
        case DMOpcode::SIGNAL:
            return dispatch_sync(instr);

        // No-op and halt
        case DMOpcode::NOP:
            return true;

        case DMOpcode::HALT:
            return true;

        // Configuration (TODO: implement)
        case DMOpcode::SET_TILE_SIZE:
        case DMOpcode::SET_BUFFER:
        case DMOpcode::SET_STRIDE:
        case DMOpcode::LOOP_BEGIN:
        case DMOpcode::LOOP_END:
            // Not yet implemented
            return true;

        default:
            state_ = ExecutionState::ERROR;
            return false;
    }
}

bool ProgramExecutor::dispatch_dma(const DMInstruction& instr) {
    if (hw_.dma_engines->empty()) {
        state_ = ExecutionState::ERROR;
        return false;
    }

    const auto& ops = std::get<DMAOperands>(instr.operands);

    // Select DMA engine (round-robin or based on buffer slot)
    size_t engine_idx = static_cast<size_t>(ops.buffer) % hw_.dma_engines->size();
    auto& dma = (*hw_.dma_engines)[engine_idx];

    // Resolve external memory address
    Address ext_addr = resolve_external_address(ops.matrix, ops.tile);

    // Determine source/destination based on opcode
    if (instr.opcode == DMOpcode::DMA_LOAD_TILE) {
        // External → L3
        dma.enqueue_transfer(
            ext_addr,                                   // src: external memory
            ops.l3_offset,                              // dst: L3 tile offset
            ops.size_bytes,
            [this, id = instr.instruction_id]() {
                pending_dma_.erase(id);
                if (completion_cb_) completion_cb_(id);
            }
        );
        stats_.external_bytes_transferred += ops.size_bytes;
    } else if (instr.opcode == DMOpcode::DMA_STORE_TILE) {
        // L3 → External
        dma.enqueue_transfer(
            ops.l3_offset,                              // src: L3 tile
            ext_addr,                                   // dst: external memory
            ops.size_bytes,
            [this, id = instr.instruction_id]() {
                pending_dma_.erase(id);
                if (completion_cb_) completion_cb_(id);
            }
        );
        stats_.external_bytes_transferred += ops.size_bytes;
    }

    pending_dma_.insert(instr.instruction_id);
    stats_.dma_operations++;

    return true;
}

bool ProgramExecutor::dispatch_block_mover(const DMInstruction& instr) {
    if (hw_.block_movers->empty()) {
        state_ = ExecutionState::ERROR;
        return false;
    }

    const auto& ops = std::get<BlockMoverOperands>(instr.operands);

    // Select BlockMover based on source L3 tile
    size_t bm_idx = ops.src_l3_tile_id % hw_.block_movers->size();
    auto& bm = (*hw_.block_movers)[bm_idx];

    // Determine transform type
    BlockMover::TransformType xform = BlockMover::TransformType::IDENTITY;
    if (instr.opcode == DMOpcode::BM_TRANSPOSE_TILE) {
        xform = BlockMover::TransformType::TRANSPOSE;
    } else if (instr.opcode == DMOpcode::BM_RESHAPE_TILE) {
        xform = BlockMover::TransformType::BLOCK_RESHAPE;
    }

    bm.enqueue_block_transfer(
        ops.src_l3_tile_id,
        ops.src_offset,
        ops.dst_l2_bank_id,
        ops.dst_offset,
        ops.height,
        ops.width,
        ops.element_size,
        xform,
        [this, id = instr.instruction_id]() {
            pending_bm_.erase(id);
            if (completion_cb_) completion_cb_(id);
        }
    );

    pending_bm_.insert(instr.instruction_id);
    stats_.block_mover_operations++;
    stats_.l3_bytes_transferred += ops.height * ops.width * ops.element_size;

    return true;
}

bool ProgramExecutor::dispatch_streamer(const DMInstruction& instr) {
    if (hw_.streamers->empty()) {
        state_ = ExecutionState::ERROR;
        return false;
    }

    const auto& ops = std::get<StreamerOperands>(instr.operands);

    // Select Streamer based on L1 buffer
    size_t str_idx = ops.l1_buffer_id % hw_.streamers->size();
    auto& str = (*hw_.streamers)[str_idx];

    Streamer::StreamConfig config;
    config.l2_bank_id = ops.l2_bank_id;
    config.l1_buffer_id = ops.l1_buffer_id;
    config.l2_base_addr = ops.l2_addr;
    config.l1_base_addr = ops.l1_addr;
    config.matrix_height = ops.height;
    config.matrix_width = ops.width;
    config.element_size = 4;  // float32
    config.compute_fabric_size = ops.fabric_size;
    config.cache_line_size = 64;

    // Set direction and type based on opcode
    if (instr.opcode == DMOpcode::STR_FEED_ROWS) {
        config.direction = Streamer::StreamDirection::L2_TO_L1;
        config.stream_type = Streamer::StreamType::ROW_STREAM;
    } else if (instr.opcode == DMOpcode::STR_FEED_COLS) {
        config.direction = Streamer::StreamDirection::L2_TO_L1;
        config.stream_type = Streamer::StreamType::COLUMN_STREAM;
    } else if (instr.opcode == DMOpcode::STR_DRAIN_OUTPUT) {
        config.direction = Streamer::StreamDirection::L1_TO_L2;
        config.stream_type = Streamer::StreamType::ROW_STREAM;
    }

    config.completion_callback = [this, id = instr.instruction_id]() {
        pending_str_.erase(id);
        if (completion_cb_) completion_cb_(id);
    };

    str.enqueue_stream(config);

    pending_str_.insert(instr.instruction_id);
    stats_.streamer_operations++;
    stats_.l2_bytes_transferred += ops.height * ops.width * 4;

    return true;
}

bool ProgramExecutor::dispatch_sync(const DMInstruction& instr) {
    if (instr.opcode == DMOpcode::BARRIER) {
        // Wait for all pending operations
        if (!all_operations_complete()) {
            state_ = ExecutionState::WAITING;
        }
        stats_.barriers_hit++;
        return true;
    }

    if (instr.opcode == DMOpcode::WAIT_DMA) {
        if (!pending_dma_.empty()) {
            state_ = ExecutionState::WAITING;
        }
        return true;
    }

    if (instr.opcode == DMOpcode::WAIT_BM) {
        if (!pending_bm_.empty()) {
            state_ = ExecutionState::WAITING;
        }
        return true;
    }

    if (instr.opcode == DMOpcode::WAIT_STR) {
        if (!pending_str_.empty()) {
            state_ = ExecutionState::WAITING;
        }
        return true;
    }

    if (instr.opcode == DMOpcode::SIGNAL) {
        // Signal completion - could trigger other components
        // For now, just a no-op
        return true;
    }

    return true;
}

void ProgramExecutor::update_hardware() {
    // Update cycle on all components
    // DMA moves data between host/KPU memory and L3 tiles
    for (auto& dma : *hw_.dma_engines) {
        dma.set_current_cycle(current_cycle_);
        dma.process_transfers(*hw_.host_memory, *hw_.external_memory, *hw_.l3_tiles);
    }

    for (auto& bm : *hw_.block_movers) {
        bm.set_cycle(current_cycle_);
        bm.process_transfers(*hw_.l3_tiles, *hw_.l2_banks);
    }

    for (auto& str : *hw_.streamers) {
        str.set_cycle(current_cycle_);
        str.update(current_cycle_, *hw_.l2_banks, *hw_.l1_buffers);
    }

    // Update compute fabric (reacts to arriving data)
    if (hw_.compute_fabric) {
        hw_.compute_fabric->update(current_cycle_, *hw_.l1_buffers);
    }
}

bool ProgramExecutor::all_operations_complete() const {
    // Check if any hardware is still busy
    for (const auto& dma : *hw_.dma_engines) {
        if (dma.is_busy()) return false;
    }
    for (const auto& bm : *hw_.block_movers) {
        if (bm.is_busy()) return false;
    }
    for (const auto& str : *hw_.streamers) {
        if (str.is_busy()) return false;
    }
    return pending_dma_.empty() && pending_bm_.empty() && pending_str_.empty();
}

Address ProgramExecutor::resolve_external_address(MatrixID matrix, const TileCoord& tile) const {
    Address base = 0;
    Size row_stride = 0;
    Size ti_size = program_->Ti;
    Size tj_size = program_->Tj;
    Size tk_size = program_->Tk;

    switch (matrix) {
        case MatrixID::A:
            base = a_base_;
            // A[M,K] - tile A[ti,tk]
            row_stride = program_->K * 4;  // float32
            return base + tile.ti * ti_size * row_stride + tile.tk * tk_size * 4;

        case MatrixID::B:
            base = b_base_;
            // B[K,N] - tile B[tk,tj]
            row_stride = program_->N * 4;
            return base + tile.tk * tk_size * row_stride + tile.tj * tj_size * 4;

        case MatrixID::C:
            base = c_base_;
            // C[M,N] - tile C[ti,tj]
            row_stride = program_->N * 4;
            return base + tile.ti * ti_size * row_stride + tile.tj * tj_size * 4;

        default:
            return 0;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void disassemble_program(const DMProgram& program, std::ostream& out) {
    out << "========================================\n";
    out << "Program: " << program.name << "\n";
    out << "Dataflow: " << (program.dataflow == DMProgram::Dataflow::OUTPUT_STATIONARY ?
                           "Output-Stationary" :
                           program.dataflow == DMProgram::Dataflow::WEIGHT_STATIONARY ?
                           "Weight-Stationary" : "Input-Stationary") << "\n";
    out << "Matrix: C[" << program.M << "," << program.N << "] = "
        << "A[" << program.M << "," << program.K << "] x "
        << "B[" << program.K << "," << program.N << "]\n";
    out << "Tiling: Ti=" << program.Ti << " Tj=" << program.Tj
        << " Tk=" << program.Tk << "\n";
    out << "----------------------------------------\n";
    out << "Instructions: " << program.instructions.size() << "\n";
    out << "  DMA:    " << program.num_dma_ops() << "\n";
    out << "  BM:     " << program.num_bm_ops() << "\n";
    out << "  STR:    " << program.num_str_ops() << "\n";
    out << "  SYNC:   " << program.num_sync_ops() << "\n";
    out << "----------------------------------------\n";

    for (size_t i = 0; i < program.instructions.size(); ++i) {
        const auto& instr = program.instructions[i];
        out << std::setw(4) << i << ": " << instr.label << "\n";
    }

    out << "========================================\n";
}

bool validate_program(const DMProgram& program, std::string& error) {
    if (program.instructions.empty()) {
        error = "Program has no instructions";
        return false;
    }

    // Check that last instruction is HALT
    if (program.instructions.back().opcode != DMOpcode::HALT) {
        error = "Program does not end with HALT";
        return false;
    }

    // Check tile sizes
    if (program.Ti == 0 || program.Tj == 0 || program.Tk == 0) {
        error = "Invalid tile size (zero)";
        return false;
    }

    // Check matrix dimensions
    if (program.M == 0 || program.N == 0 || program.K == 0) {
        error = "Invalid matrix dimension (zero)";
        return false;
    }

    error = "Valid";
    return true;
}

} // namespace sw::kpu::isa
