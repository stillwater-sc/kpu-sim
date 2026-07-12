/**
 * @file behavioral_program_executor.cpp
 * @brief Behavioral (functional) executor for Data Movement ISA programs
 *
 * Each DMInstruction is dispatched as an instant memcpy between the
 * appropriate memory components. Compute fires a triple-loop matmul
 * when both A and B tiles have been streamed to L1.
 */

#include <sw/kpu/isa/behavioral_program_executor.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace sw::kpu::isa {

// ============================================================================
// Construction
// ============================================================================

BehavioralProgramExecutor::BehavioralProgramExecutor(HardwareContext hw)
    : hw_(hw) {}

// ============================================================================
// Program loading
// ============================================================================

void BehavioralProgramExecutor::load_program(
    const DMProgram& program,
    Address a_base, Address b_base, Address c_base)
{
    program_ = &program;
    a_base_ = a_base;
    b_base_ = b_base;
    c_base_ = c_base;
    stats_ = Statistics{};
    a_fed_ = false;
    b_fed_ = false;
    accumulating_ = false;
    acc_ti_ = 0;
    acc_tj_ = 0;
    regs_.reset();  // Reset ISA register file

    // Configure address generator with base addresses for AUTO modes
    regs_.addr().set_base(MatrixID::A, a_base);
    regs_.addr().set_base(MatrixID::B, b_base);
    regs_.addr().set_base(MatrixID::C, c_base);
}

// ============================================================================
// Execution
// ============================================================================

bool BehavioralProgramExecutor::run() {
    if (!program_) return false;

    const auto& instructions = program_->instructions;
    size_t pc = 0;

    while (pc < instructions.size()) {
        const auto& instr = instructions[pc];

        if (instr.opcode == DMOpcode::HALT) {
            stats_.instructions_executed++;
            return true;
        }

        // Handle loop control with PC manipulation
        if (instr.opcode == DMOpcode::LOOP_BEGIN) {
            const auto& ops = std::get<LoopOperands>(instr.operands);
            regs_.exec_loop_begin(ops, pc);
            stats_.instructions_executed++;
            stats_.loop_iterations++;
            ++pc;
            continue;
        }

        if (instr.opcode == DMOpcode::LOOP_END) {
            const auto& ops = std::get<LoopOperands>(instr.operands);
            size_t next_pc = regs_.exec_loop_end(ops, pc);
            stats_.instructions_executed++;
            if (next_pc != pc + 1) {
                // Loop continues, branch back
                stats_.loop_iterations++;
            }
            pc = next_pc;
            continue;
        }

        dispatch(instr, pc);
        stats_.instructions_executed++;
        ++pc;
    }
    return true;
}

void BehavioralProgramExecutor::dispatch(const DMInstruction& instr, [[maybe_unused]] size_t pc) {
    switch (instr.opcode) {
    case DMOpcode::DMA_LOAD_TILE:
    case DMOpcode::DMA_PREFETCH_TILE:
        dispatch_dma_load(std::get<DMAOperands>(instr.operands));
        break;

    case DMOpcode::DMA_STORE_TILE:
        dispatch_dma_store(std::get<DMAOperands>(instr.operands));
        break;

    case DMOpcode::BM_MOVE_TILE:
        dispatch_bm_move(std::get<BlockMoverOperands>(instr.operands));
        break;

    case DMOpcode::BM_WRITEBACK_TILE:
        dispatch_bm_writeback(std::get<BlockMoverOperands>(instr.operands));
        break;

    case DMOpcode::STR_FEED_ROWS:
        dispatch_str_feed_rows(std::get<StreamerOperands>(instr.operands));
        break;

    case DMOpcode::STR_FEED_COLS:
        dispatch_str_feed_cols(std::get<StreamerOperands>(instr.operands));
        break;

    case DMOpcode::STR_DRAIN_OUTPUT:
        dispatch_str_drain(std::get<StreamerOperands>(instr.operands));
        break;

    case DMOpcode::BARRIER:
    case DMOpcode::WAIT_DMA:
    case DMOpcode::WAIT_BM:
    case DMOpcode::WAIT_STR:
        dispatch_barrier();
        break;

    case DMOpcode::NOP:
    case DMOpcode::SIGNAL:
    case DMOpcode::HALT:
        // No-ops in behavioral execution
        break;

    // Transpose is a BlockMover move with transform
    case DMOpcode::BM_TRANSPOSE_TILE:
        dispatch_bm_move(std::get<BlockMoverOperands>(instr.operands));
        break;

    // ========================================================================
    // Configuration opcodes — update register file
    // ========================================================================
    case DMOpcode::SET_BASE:
        regs_.exec_set_base(std::get<BaseAddressOperands>(instr.operands));
        stats_.config_instructions++;
        break;

    case DMOpcode::SET_L3_BASE:
        regs_.exec_set_l3_base(std::get<L3BaseOperands>(instr.operands));
        stats_.config_instructions++;
        break;

    case DMOpcode::SET_L2_BASE:
        regs_.exec_set_l2_base(std::get<L2BaseOperands>(instr.operands));
        stats_.config_instructions++;
        break;

    case DMOpcode::SET_STRIDE:
        regs_.exec_set_stride(std::get<StrideConfigOperands>(instr.operands));
        stats_.config_instructions++;
        break;

    case DMOpcode::SET_TILE_DIM:
        regs_.exec_set_tile_dim(std::get<TileDimOperands>(instr.operands));
        stats_.config_instructions++;
        break;

    case DMOpcode::SET_MATRIX_DIM:
        regs_.exec_set_matrix_dim(std::get<MatrixDimOperands>(instr.operands));
        stats_.config_instructions++;
        break;

    // Legacy config (ignored for now)
    case DMOpcode::SET_TILE_SIZE:
    case DMOpcode::SET_BUFFER:
        break;

    // Loop control handled in run() — should not reach here
    case DMOpcode::LOOP_BEGIN:
    case DMOpcode::LOOP_END:
        break;

    // ========================================================================
    // AUTO addressing opcodes
    // ========================================================================
    case DMOpcode::DMA_LOAD_TILE_AUTO:
        dispatch_dma_load_auto(std::get<AutoDMAOperands>(instr.operands));
        break;

    case DMOpcode::DMA_STORE_TILE_AUTO:
        dispatch_dma_store_auto(std::get<AutoDMAOperands>(instr.operands));
        break;

    case DMOpcode::BM_MOVE_TILE_AUTO:
        dispatch_bm_move_auto(std::get<AutoBlockMoverOperands>(instr.operands));
        break;

    case DMOpcode::BM_WRITEBACK_AUTO:
        dispatch_bm_writeback_auto(std::get<AutoBlockMoverOperands>(instr.operands));
        break;

    case DMOpcode::STR_FEED_ROWS_AUTO:
        dispatch_str_feed_rows_auto(std::get<AutoStreamerOperands>(instr.operands));
        break;

    case DMOpcode::STR_FEED_COLS_AUTO:
        dispatch_str_feed_cols_auto(std::get<AutoStreamerOperands>(instr.operands));
        break;

    case DMOpcode::STR_DRAIN_AUTO:
        dispatch_str_drain_auto(std::get<AutoStreamerOperands>(instr.operands));
        break;

    // Broadcast — treat as a streamer feed for now
    case DMOpcode::STR_BROADCAST_ROW:
    case DMOpcode::STR_BROADCAST_COL:
        // Behavioral: broadcast is a no-op annotation
        // (data already in L2, compute reads it directly in softmax path)
        break;

    case DMOpcode::VE_ELEMENTWISE:
        // Functional when carrying VEOperands (issue #100); legacy
        // monostate markers remain no-ops
        if (std::holds_alternative<VEOperands>(instr.operands)) {
            execute_ve_elementwise(std::get<VEOperands>(instr.operands));
        }
        break;

    // Gather/scatter, VE_REDUCE (E3 scope), and scratch opcodes —
    // annotations, not functional in behavioral
    case DMOpcode::DMA_LOAD_GATHER:
    case DMOpcode::DMA_STORE_SCATTER:
    case DMOpcode::VE_REDUCE:
    case DMOpcode::L2_SCRATCH_WRITE:
    case DMOpcode::L2_SCRATCH_READ:
        break;

    default:
        break;
    }
}

// ============================================================================
// DMA: External Memory <-> L3
// ============================================================================

void BehavioralProgramExecutor::dispatch_dma_load(const DMAOperands& ops) {
    // External memory -> L3 tile
    // The schedule compiler generates offsets relative to 0; we add the actual base
    Address ext_addr = ops.ext_mem_addr;
    switch (ops.matrix) {
        case MatrixID::A: ext_addr += a_base_; break;
        case MatrixID::B: ext_addr += b_base_; break;
        case MatrixID::C: ext_addr += c_base_; break;
        default: break;
    }

    uint8_t l3_id = ops.l3_tile_id;
    Address l3_offset = ops.l3_offset;
    Size bytes = ops.size_bytes;

    if (bytes == 0) return;
    if (l3_id >= hw_.l3_tiles.size()) return;
    if (!program_) return;

    // Determine tile dimensions and external memory stride
    Size tile_rows = 0, tile_cols = 0, ext_stride = 0;
    Size es = sizeof(float);

    switch (ops.matrix) {
    case MatrixID::A:
        tile_rows = program_->Ti;
        tile_cols = program_->Tk;
        ext_stride = program_->K * es;  // A[M,K] row-major
        break;
    case MatrixID::B:
        tile_rows = program_->Tk;
        tile_cols = program_->Tj;
        ext_stride = program_->N * es;  // B[K,N] row-major
        break;
    case MatrixID::C:
        tile_rows = program_->Ti;
        tile_cols = program_->Tj;
        ext_stride = program_->N * es;  // C[M,N] row-major
        break;
    default:
        return;
    }

    Size row_bytes = tile_cols * es;

    // Strided transfer: read each row from external memory to contiguous L3 storage
    std::vector<uint8_t> row_buf(row_bytes);
    for (Size r = 0; r < tile_rows; ++r) {
        Address src = ext_addr + r * ext_stride;
        Address dst = l3_offset + r * row_bytes;
        hw_.external_memory.read(src, row_buf.data(), row_bytes);
        hw_.l3_tiles[l3_id].write(dst, row_buf.data(), row_bytes);
    }

    stats_.dma_loads++;
    stats_.bytes_loaded += bytes;
}

void BehavioralProgramExecutor::dispatch_dma_store(const DMAOperands& ops) {
    // L3 tile -> External memory
    // The schedule compiler generates offsets relative to 0; we add the actual base
    Address ext_addr = ops.ext_mem_addr;
    switch (ops.matrix) {
        case MatrixID::A: ext_addr += a_base_; break;
        case MatrixID::B: ext_addr += b_base_; break;
        case MatrixID::C: ext_addr += c_base_; break;
        default: break;
    }

    uint8_t l3_id = ops.l3_tile_id;
    Address l3_offset = ops.l3_offset;
    Size bytes = ops.size_bytes;

    if (bytes == 0) return;
    if (l3_id >= hw_.l3_tiles.size()) return;
    if (!program_) return;

    // Determine tile dimensions and external memory stride
    Size tile_rows = 0, tile_cols = 0, ext_stride = 0;
    Size es = sizeof(float);

    switch (ops.matrix) {
    case MatrixID::A:
        tile_rows = program_->Ti;
        tile_cols = program_->Tk;
        ext_stride = program_->K * es;  // A[M,K] row-major
        break;
    case MatrixID::B:
        tile_rows = program_->Tk;
        tile_cols = program_->Tj;
        ext_stride = program_->N * es;  // B[K,N] row-major
        break;
    case MatrixID::C:
        tile_rows = program_->Ti;
        tile_cols = program_->Tj;
        ext_stride = program_->N * es;  // C[M,N] row-major
        break;
    default:
        return;
    }

    Size row_bytes = tile_cols * es;

    // Strided transfer: write each row from contiguous L3 storage to external memory
    std::vector<uint8_t> row_buf(row_bytes);
    for (Size r = 0; r < tile_rows; ++r) {
        Address src = l3_offset + r * row_bytes;
        Address dst = ext_addr + r * ext_stride;
        hw_.l3_tiles[l3_id].read(src, row_buf.data(), row_bytes);
        hw_.external_memory.write(dst, row_buf.data(), row_bytes);
    }

    stats_.dma_stores++;
    stats_.bytes_stored += bytes;
}

// ============================================================================
// BlockMover: L3 <-> L2
// ============================================================================

void BehavioralProgramExecutor::dispatch_bm_move(const BlockMoverOperands& ops) {
    // L3 tile -> L2 bank
    uint8_t src_l3 = ops.src_l3_tile_id;
    Address src_offset = ops.src_offset;
    uint8_t dst_l2 = ops.dst_l2_bank_id;
    Address dst_offset = ops.dst_offset;
    Size bytes = ops.height * ops.width * ops.element_size;

    if (bytes == 0) return;
    if (src_l3 >= hw_.l3_tiles.size()) return;
    if (dst_l2 >= hw_.l2_banks.size()) return;

    std::vector<uint8_t> buf(bytes);
    hw_.l3_tiles[src_l3].read(src_offset, buf.data(), bytes);

    // Handle transpose
    if (ops.transform == Transform::TRANSPOSE) {
        // Transpose: swap rows and columns
        Size es = ops.element_size;
        Size h = ops.height;
        Size w = ops.width;
        std::vector<uint8_t> transposed(bytes);
        for (Size r = 0; r < h; ++r) {
            for (Size c = 0; c < w; ++c) {
                std::memcpy(
                    transposed.data() + (c * h + r) * es,
                    buf.data() + (r * w + c) * es,
                    es);
            }
        }
        hw_.l2_banks[dst_l2].write(dst_offset, transposed.data(), bytes);
    } else {
        hw_.l2_banks[dst_l2].write(dst_offset, buf.data(), bytes);
    }

    stats_.bm_moves++;
}

void BehavioralProgramExecutor::dispatch_bm_writeback(const BlockMoverOperands& ops) {
    // L2 bank -> L3 tile
    // Note: in writeback, src_l3_tile_id is actually the L2 bank (field reuse)
    //       and dst_l2_bank_id is actually the L3 tile
    uint8_t src_l2 = ops.src_l3_tile_id;
    Address src_offset = ops.src_offset;
    uint8_t dst_l3 = ops.dst_l2_bank_id;
    Address dst_offset = ops.dst_offset;
    Size bytes = ops.height * ops.width * ops.element_size;

    if (bytes == 0) return;
    if (src_l2 >= hw_.l2_banks.size()) return;
    if (dst_l3 >= hw_.l3_tiles.size()) return;

    std::vector<uint8_t> buf(bytes);
    hw_.l2_banks[src_l2].read(src_offset, buf.data(), bytes);
    hw_.l3_tiles[dst_l3].write(dst_offset, buf.data(), bytes);

    stats_.bm_writebacks++;
}

// ============================================================================
// Streamer: L2 <-> L1
// ============================================================================

void BehavioralProgramExecutor::dispatch_str_feed_rows(const StreamerOperands& ops) {
    // L2 bank -> L1 buffer (A tile, row streaming)
    uint8_t l2_id = ops.l2_bank_id;
    Address l2_addr = ops.l2_addr;
    Size bytes = ops.height * ops.width * 4;  // float32

    if (bytes == 0) return;
    if (l2_id >= hw_.l2_banks.size()) return;
    if (L1_A_BUFFER >= hw_.l1_buffers.size()) return;

    std::vector<uint8_t> buf(bytes);
    hw_.l2_banks[l2_id].read(l2_addr, buf.data(), bytes);
    hw_.l1_buffers[L1_A_BUFFER].write(0, buf.data(), bytes);

    a_fed_ = true;
    stats_.str_feeds++;

    // If both A and B are in L1, fire compute
    if (a_fed_ && b_fed_) {
        fire_compute();
        a_fed_ = false;
        b_fed_ = false;
    }
}

void BehavioralProgramExecutor::dispatch_str_feed_cols(const StreamerOperands& ops) {
    // L2 bank -> L1 buffer (B tile, column streaming)
    uint8_t l2_id = ops.l2_bank_id;
    Address l2_addr = ops.l2_addr;
    Size bytes = ops.height * ops.width * 4;  // float32

    if (bytes == 0) return;
    if (l2_id >= hw_.l2_banks.size()) return;
    if (L1_B_BUFFER >= hw_.l1_buffers.size()) return;

    std::vector<uint8_t> buf(bytes);
    hw_.l2_banks[l2_id].read(l2_addr, buf.data(), bytes);
    hw_.l1_buffers[L1_B_BUFFER].write(0, buf.data(), bytes);

    b_fed_ = true;
    stats_.str_feeds++;

    // If both A and B are in L1, fire compute
    if (a_fed_ && b_fed_) {
        fire_compute();
        a_fed_ = false;
        b_fed_ = false;
    }
}

void BehavioralProgramExecutor::dispatch_str_drain(const StreamerOperands& ops) {
    // L1 accumulator -> L2 bank (drain C tile)
    uint8_t l2_id = ops.l2_bank_id;
    Address l2_addr = ops.l2_addr;
    Size bytes = ops.height * ops.width * 4;  // float32

    if (bytes == 0) return;
    if (l2_id >= hw_.l2_banks.size()) return;
    if (L1_A_BUFFER >= hw_.l1_buffers.size()) return;

    // Read accumulated result from L1 accumulator region
    // We store the accumulator at offset = A_buffer_capacity / 2 (second half)
    Size acc_offset = hw_.l1_buffers[L1_A_BUFFER].get_capacity() / 2;
    std::vector<uint8_t> buf(bytes);
    hw_.l1_buffers[L1_A_BUFFER].read(acc_offset, buf.data(), bytes);
    hw_.l2_banks[l2_id].write(l2_addr, buf.data(), bytes);

    // Reset accumulation state for next output tile
    accumulating_ = false;

    stats_.str_drains++;
}

// ============================================================================
// Barrier
// ============================================================================

void BehavioralProgramExecutor::dispatch_barrier() {
    // All operations are instant in behavioral mode — nothing to wait for
    stats_.barriers++;
}

// ============================================================================
// Compute: C += A × B on L1 tile data
// ============================================================================

void BehavioralProgramExecutor::execute_ve_elementwise(const VEOperands& ops) {
    if (!program_) return;
    Size elems = program_->Ti * program_->Tj;
    if (elems == 0) return;
    if (ops.l1_src_a >= hw_.l1_buffers.size() ||
        ops.l1_dst >= hw_.l1_buffers.size()) {
        return;
    }

    const Size bytes = elems * sizeof(float);
    std::vector<float> a(elems);
    hw_.l1_buffers[ops.l1_src_a].read(0, a.data(), bytes);

    std::vector<float> b;
    if (ops.num_inputs == 2) {
        if (ops.l1_src_b >= hw_.l1_buffers.size()) return;
        b.resize(elems);
        hw_.l1_buffers[ops.l1_src_b].read(0, b.data(), bytes);
    }

    // IEEE semantics without domain guards (div-by-zero -> inf, sqrt of
    // negative -> NaN): the host oracle applies the same std operations,
    // so validation compares like for like
    std::vector<float> out(elems);
    for (Size i = 0; i < elems; ++i) {
        const float x = a[i];
        const float y = ops.num_inputs == 2 ? b[i] : ops.scalar;
        float r = 0.0f;
        switch (ops.op) {
            case VEOp::ADD:   r = x + y; break;
            case VEOp::SUB:   r = x - y; break;
            case VEOp::MUL:   r = x * y; break;
            case VEOp::DIV:   r = x / y; break;
            case VEOp::MAX:   r = x > y ? x : y; break;
            case VEOp::MIN:   r = x < y ? x : y; break;
            case VEOp::NEG:   r = -x; break;
            case VEOp::ABS:   r = std::fabs(x); break;
            case VEOp::SQRT:  r = std::sqrt(x); break;
            case VEOp::EXP:   r = std::exp(x); break;
            case VEOp::LOG:   r = std::log(x); break;
            case VEOp::ADD_S: r = x + ops.scalar; break;
            case VEOp::MUL_S: r = x * ops.scalar; break;
            case VEOp::POW_S: r = std::pow(x, ops.scalar); break;
        }
        out[i] = r;
    }

    hw_.l1_buffers[ops.l1_dst].write(0, out.data(), bytes);
}

void BehavioralProgramExecutor::fire_compute() {
    if (!program_) return;

    Size Ti = program_->Ti;
    Size Tj = program_->Tj;
    Size Tk = program_->Tk;

    if (Ti == 0 || Tj == 0 || Tk == 0) return;

    // Read A tile from L1 buffer 0: shape [Ti, Tk]
    Size a_bytes = Ti * Tk * sizeof(float);
    std::vector<float> a_data(Ti * Tk);
    hw_.l1_buffers[L1_A_BUFFER].read(0, a_data.data(), a_bytes);

    // Read B tile from L1 buffer 1: shape [Tk, Tj]
    Size b_bytes = Tk * Tj * sizeof(float);
    std::vector<float> b_data(Tk * Tj);
    hw_.l1_buffers[L1_B_BUFFER].read(0, b_data.data(), b_bytes);

    // Read current accumulator: shape [Ti, Tj]
    Size c_bytes = Ti * Tj * sizeof(float);
    Size acc_offset = hw_.l1_buffers[L1_A_BUFFER].get_capacity() / 2;
    std::vector<float> c_data(Ti * Tj, 0.0f);

    if (accumulating_) {
        // Continue accumulating from previous K iteration
        hw_.l1_buffers[L1_A_BUFFER].read(acc_offset, c_data.data(), c_bytes);
    }

    // Triple-loop matmul: C[Ti,Tj] += A[Ti,Tk] × B[Tk,Tj]
    for (Size i = 0; i < Ti; ++i) {
        for (Size j = 0; j < Tj; ++j) {
            float sum = c_data[i * Tj + j];
            for (Size k = 0; k < Tk; ++k) {
                sum += a_data[i * Tk + k] * b_data[k * Tj + j];
            }
            c_data[i * Tj + j] = sum;
        }
    }

    // Write back to accumulator region in L1
    hw_.l1_buffers[L1_A_BUFFER].write(acc_offset, c_data.data(), c_bytes);
    accumulating_ = true;

    stats_.compute_invocations++;
}

// ============================================================================
// Address resolution
// ============================================================================

Address BehavioralProgramExecutor::resolve_address(
    MatrixID matrix, const TileCoord& tile) const
{
    if (!program_) return 0;

    Size Ti = program_->Ti;
    Size Tj = program_->Tj;
    Size Tk = program_->Tk;
    Size es = sizeof(float);

    switch (matrix) {
    case MatrixID::A:
        // A[M,K] — tile A[ti,tk]
        return a_base_ + (tile.ti * Ti * program_->K + tile.tk * Tk) * es;
    case MatrixID::B:
        // B[K,N] — tile B[tk,tj]
        return b_base_ + (tile.tk * Tk * program_->N + tile.tj * Tj) * es;
    case MatrixID::C:
        // C[M,N] — tile C[ti,tj]
        return c_base_ + (tile.ti * Ti * program_->N + tile.tj * Tj) * es;
    default:
        return 0;
    }
}

void BehavioralProgramExecutor::get_tile_dims(
    MatrixID matrix, Size& rows, Size& cols) const
{
    if (!program_) { rows = cols = 0; return; }

    switch (matrix) {
    case MatrixID::A:
        rows = program_->Ti;
        cols = program_->Tk;
        break;
    case MatrixID::B:
        rows = program_->Tk;
        cols = program_->Tj;
        break;
    case MatrixID::C:
        rows = program_->Ti;
        cols = program_->Tj;
        break;
    default:
        rows = cols = 0;
        break;
    }
}

// ============================================================================
// AUTO Addressing: DMA
// ============================================================================

void BehavioralProgramExecutor::dispatch_dma_load_auto(const AutoDMAOperands& ops) {
    // Compute external memory address from loop indices
    Address ext_addr = regs_.compute_ext_addr(ops.matrix);

    // For now, use simplified buffer layout based on matrix
    uint8_t l3_id = ops.l3_slot;
    Address l3_offset = regs_.compute_l3_offset(ops.matrix);

    // Get tile dimensions
    Size tile_rows = 0, tile_cols = 0;
    get_tile_dims(ops.matrix, tile_rows, tile_cols);
    Size es = sizeof(float);
    Size bytes = tile_rows * tile_cols * es;

    if (bytes == 0) return;
    if (l3_id >= hw_.l3_tiles.size()) return;
    if (!program_) return;

    // Determine external memory stride based on matrix
    Size ext_stride = 0;
    switch (ops.matrix) {
    case MatrixID::A:
        ext_stride = program_->K * es;  // A[M,K] row-major
        break;
    case MatrixID::B:
        ext_stride = program_->N * es;  // B[K,N] row-major
        break;
    case MatrixID::C:
        ext_stride = program_->N * es;  // C[M,N] row-major
        break;
    default:
        return;
    }

    Size row_bytes = tile_cols * es;

    // Strided transfer: read each row from external memory to contiguous L3 storage
    std::vector<uint8_t> row_buf(row_bytes);
    for (Size r = 0; r < tile_rows; ++r) {
        Address src = ext_addr + r * ext_stride;
        Address dst = l3_offset + r * row_bytes;
        hw_.external_memory.read(src, row_buf.data(), row_bytes);
        hw_.l3_tiles[l3_id].write(dst, row_buf.data(), row_bytes);
    }

    stats_.dma_loads++;
    stats_.bytes_loaded += bytes;
}

void BehavioralProgramExecutor::dispatch_dma_store_auto(const AutoDMAOperands& ops) {
    // Compute external memory address from loop indices
    Address ext_addr = regs_.compute_ext_addr(ops.matrix);

    // Use simplified buffer layout based on matrix
    uint8_t l3_id = ops.l3_slot;
    Address l3_offset = regs_.compute_l3_offset(ops.matrix);

    // Get tile dimensions
    Size tile_rows = 0, tile_cols = 0;
    get_tile_dims(ops.matrix, tile_rows, tile_cols);
    Size es = sizeof(float);
    Size bytes = tile_rows * tile_cols * es;

    if (bytes == 0) return;
    if (l3_id >= hw_.l3_tiles.size()) return;
    if (!program_) return;

    // Determine external memory stride based on matrix
    Size ext_stride = 0;
    switch (ops.matrix) {
    case MatrixID::A:
        ext_stride = program_->K * es;  // A[M,K] row-major
        break;
    case MatrixID::B:
        ext_stride = program_->N * es;  // B[K,N] row-major
        break;
    case MatrixID::C:
        ext_stride = program_->N * es;  // C[M,N] row-major
        break;
    default:
        return;
    }

    Size row_bytes = tile_cols * es;

    // Strided transfer: write each row from contiguous L3 storage to external memory
    std::vector<uint8_t> row_buf(row_bytes);
    for (Size r = 0; r < tile_rows; ++r) {
        Address src = l3_offset + r * row_bytes;
        Address dst = ext_addr + r * ext_stride;
        hw_.l3_tiles[l3_id].read(src, row_buf.data(), row_bytes);
        hw_.external_memory.write(dst, row_buf.data(), row_bytes);
    }

    stats_.dma_stores++;
    stats_.bytes_stored += bytes;
}

// ============================================================================
// AUTO Addressing: BlockMover
// ============================================================================

void BehavioralProgramExecutor::dispatch_bm_move_auto(const AutoBlockMoverOperands& ops) {
    // L3 tile -> L2 bank
    // Compute offsets from register file
    // Note: BlockMover doesn't need L3 tile ID - it reads from wherever the tile is
    // For now, use L3 slot 0 (single buffer mode)
    uint8_t src_l3 = 0;
    Address src_offset = regs_.compute_l3_offset(ops.matrix);
    uint8_t dst_l2 = ops.l2_bank;
    Address dst_offset = regs_.compute_l2_offset(ops.matrix);

    Size tile_rows = 0, tile_cols = 0;
    get_tile_dims(ops.matrix, tile_rows, tile_cols);
    Size es = sizeof(float);
    Size bytes = tile_rows * tile_cols * es;

    if (bytes == 0) return;
    if (src_l3 >= hw_.l3_tiles.size()) return;
    if (dst_l2 >= hw_.l2_banks.size()) return;

    std::vector<uint8_t> buf(bytes);
    hw_.l3_tiles[src_l3].read(src_offset, buf.data(), bytes);
    hw_.l2_banks[dst_l2].write(dst_offset, buf.data(), bytes);

    stats_.bm_moves++;
}

void BehavioralProgramExecutor::dispatch_bm_writeback_auto(const AutoBlockMoverOperands& ops) {
    // L2 bank -> L3 tile
    uint8_t src_l2 = ops.l2_bank;
    Address src_offset = regs_.compute_l2_offset(ops.matrix);
    // For now, use L3 slot 0 (single buffer mode)
    uint8_t dst_l3 = 0;
    Address dst_offset = regs_.compute_l3_offset(ops.matrix);

    Size tile_rows = 0, tile_cols = 0;
    get_tile_dims(ops.matrix, tile_rows, tile_cols);
    Size es = sizeof(float);
    Size bytes = tile_rows * tile_cols * es;

    if (bytes == 0) return;
    if (src_l2 >= hw_.l2_banks.size()) return;
    if (dst_l3 >= hw_.l3_tiles.size()) return;

    std::vector<uint8_t> buf(bytes);
    hw_.l2_banks[src_l2].read(src_offset, buf.data(), bytes);
    hw_.l3_tiles[dst_l3].write(dst_offset, buf.data(), bytes);

    stats_.bm_writebacks++;
}

// ============================================================================
// AUTO Addressing: Streamer
// ============================================================================

void BehavioralProgramExecutor::dispatch_str_feed_rows_auto([[maybe_unused]] const AutoStreamerOperands& ops) {
    // L2 bank -> L1 buffer (A tile, row streaming)
    // For now, use L2 bank 0 and derive addresses from register file
    uint8_t l2_id = 0;
    Address l2_addr = regs_.compute_l2_offset(MatrixID::A);

    Size tile_rows = 0, tile_cols = 0;
    get_tile_dims(MatrixID::A, tile_rows, tile_cols);
    Size bytes = tile_rows * tile_cols * sizeof(float);

    if (bytes == 0) return;
    if (l2_id >= hw_.l2_banks.size()) return;
    if (L1_A_BUFFER >= hw_.l1_buffers.size()) return;

    std::vector<uint8_t> buf(bytes);
    hw_.l2_banks[l2_id].read(l2_addr, buf.data(), bytes);
    hw_.l1_buffers[L1_A_BUFFER].write(0, buf.data(), bytes);

    a_fed_ = true;
    stats_.str_feeds++;

    // If both A and B are in L1, fire compute
    if (a_fed_ && b_fed_) {
        fire_compute();
        a_fed_ = false;
        b_fed_ = false;
    }
}

void BehavioralProgramExecutor::dispatch_str_feed_cols_auto([[maybe_unused]] const AutoStreamerOperands& ops) {
    // L2 bank -> L1 buffer (B tile, column streaming)
    // For now, use L2 bank 0 and derive addresses from register file
    uint8_t l2_id = 0;
    Address l2_addr = regs_.compute_l2_offset(MatrixID::B);

    Size tile_rows = 0, tile_cols = 0;
    get_tile_dims(MatrixID::B, tile_rows, tile_cols);
    Size bytes = tile_rows * tile_cols * sizeof(float);

    if (bytes == 0) return;
    if (l2_id >= hw_.l2_banks.size()) return;
    if (L1_B_BUFFER >= hw_.l1_buffers.size()) return;

    std::vector<uint8_t> buf(bytes);
    hw_.l2_banks[l2_id].read(l2_addr, buf.data(), bytes);
    hw_.l1_buffers[L1_B_BUFFER].write(0, buf.data(), bytes);

    b_fed_ = true;
    stats_.str_feeds++;

    // If both A and B are in L1, fire compute
    if (a_fed_ && b_fed_) {
        fire_compute();
        a_fed_ = false;
        b_fed_ = false;
    }
}

void BehavioralProgramExecutor::dispatch_str_drain_auto([[maybe_unused]] const AutoStreamerOperands& ops) {
    // L1 accumulator -> L2 bank (drain C tile)
    // For now, use L2 bank 0 and derive addresses from register file
    uint8_t l2_id = 0;
    Address l2_addr = regs_.compute_l2_offset(MatrixID::C);

    Size tile_rows = 0, tile_cols = 0;
    get_tile_dims(MatrixID::C, tile_rows, tile_cols);
    Size bytes = tile_rows * tile_cols * sizeof(float);

    if (bytes == 0) return;
    if (l2_id >= hw_.l2_banks.size()) return;
    if (L1_A_BUFFER >= hw_.l1_buffers.size()) return;

    // Read accumulated result from L1 accumulator region
    Size acc_offset = hw_.l1_buffers[L1_A_BUFFER].get_capacity() / 2;
    std::vector<uint8_t> buf(bytes);
    hw_.l1_buffers[L1_A_BUFFER].read(acc_offset, buf.data(), bytes);
    hw_.l2_banks[l2_id].write(l2_addr, buf.data(), bytes);

    // Reset accumulation state for next output tile
    accumulating_ = false;

    stats_.str_drains++;
}

} // namespace sw::kpu::isa
