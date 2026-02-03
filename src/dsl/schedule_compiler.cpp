/**
 * @file schedule_compiler.cpp
 * @brief Compiles Schedule DSL into DMProgram instruction streams
 *
 * The compiler walks the schedule operation tree and emits DMInstructions.
 * It tracks tile coordinates across nested loops and assigns buffer slots.
 */

#include <sw/kpu/dsl/schedule_compiler.hpp>
#include <stdexcept>
#include <sstream>
#include <algorithm>

namespace sw::kpu::dsl {

using namespace isa;

namespace {

/**
 * @brief Compilation context tracks loop indices and emits instructions
 */
struct CompileContext {
    const Schedule& sched;
    DMProgram& prog;
    uint32_t next_id = 0;

    // Current tile indices per dimension
    uint16_t ti = 0, tj = 0, tk = 0;

    // Buffer slot alternation for double-buffering
    int buf_toggle = 0;

    BufferSlot current_buf() const {
        return (buf_toggle % 2 == 0) ? BufferSlot::BUF_0 : BufferSlot::BUF_1;
    }

    TileCoord current_tile() const {
        return TileCoord{ti, tj, tk};
    }

    Size elem_size() const { return sched.get_element_size(); }
    Size fab_size() const { return sched.get_systolic_size(); }

    void emit(DMInstruction instr) {
        instr.instruction_id = next_id++;
        prog.instructions.push_back(std::move(instr));
    }

    void emit_op(const ScheduleOp& op);
    void emit_ops(const std::vector<ScheduleOp>& ops);
    void emit_loop(const ScheduleOp& loop);
};

void CompileContext::emit_op(const ScheduleOp& op) {
    auto tile = current_tile();
    auto buf = current_buf();
    Size es = elem_size();
    Size fs = fab_size();

    switch (op.kind) {
    case ScheduleOpKind::LOAD: {
        Size tile_bytes = 0;
        Address ext_addr = 0;
        auto* t = sched.find_tensor(op.matrix);
        if (t && t->shape.size() >= 2) {
            Size rows = 0, cols = 0;
            if (op.matrix == MatrixID::A) {
                Size Ti = sched.get_tile("ti");
                Size Tk = sched.get_tile("tk");
                rows = Ti ? Ti : t->shape[0];
                cols = Tk ? Tk : t->shape[1];
                ext_addr = t->base_addr + (ti * rows * t->shape[1] + tk * cols) * es;
            } else if (op.matrix == MatrixID::B) {
                Size Tk = sched.get_tile("tk");
                Size Tj = sched.get_tile("tj");
                rows = Tk ? Tk : t->shape[0];
                cols = Tj ? Tj : t->shape[1];
                ext_addr = t->base_addr + (tk * rows * t->shape[1] + tj * cols) * es;
            } else {
                Size Ti = sched.get_tile("ti");
                Size Tj = sched.get_tile("tj");
                rows = Ti ? Ti : t->shape[0];
                cols = Tj ? Tj : t->shape[1];
                ext_addr = t->base_addr + (ti * rows * t->shape[1] + tj * cols) * es;
            }
            tile_bytes = rows * cols * es;
        }
        uint8_t l3_tile = static_cast<uint8_t>(buf);
        emit(DMInstruction::dma_load(op.matrix, tile, ext_addr, l3_tile, 0, tile_bytes));
        prog.estimates.external_mem_bytes += tile_bytes;
        prog.estimates.l3_bytes += tile_bytes;
        break;
    }

    case ScheduleOpKind::LOAD_GATHER: {
        // Im2col gather — emitted as DMA_LOAD_TILE with gather annotation in label
        Size Ti = sched.get_tile("ti");
        Size Tk = sched.get_tile("tk");
        Ti = Ti ? Ti : 64;
        Tk = Tk ? Tk : 64;
        Size tile_bytes = Ti * Tk * es;

        DMInstruction instr;
        instr.opcode = DMOpcode::DMA_LOAD_TILE;  // Future: DMA_LOAD_GATHER
        DMAOperands dops;
        dops.matrix = op.matrix;
        dops.tile = tile;
        dops.ext_mem_addr = op.im2col.input_base;
        dops.l3_tile_id = static_cast<uint8_t>(buf);
        dops.l3_offset = 0;
        dops.size_bytes = tile_bytes;
        dops.buffer = buf;
        instr.operands = dops;
        instr.label = "DMA_LOAD_GATHER im2col";
        emit(instr);
        prog.estimates.external_mem_bytes += tile_bytes;
        prog.estimates.l3_bytes += tile_bytes;
        break;
    }

    case ScheduleOpKind::STORE: {
        Size rows = 0, cols = 0;
        auto* t = sched.find_tensor(op.matrix);
        if (t && t->shape.size() >= 2) {
            Size Ti = sched.get_tile("ti");
            Size Tj = sched.get_tile("tj");
            rows = Ti ? Ti : t->shape[0];
            cols = Tj ? Tj : t->shape[1];
        }
        Size tile_bytes = rows * cols * es;
        Address ext_addr = 0;
        if (auto* t2 = sched.find_tensor(op.matrix)) {
            ext_addr = t2->base_addr + (ti * rows * (t2->shape.size() >= 2 ? t2->shape[1] : 0) + tj * cols) * es;
        }

        DMInstruction instr;
        instr.opcode = DMOpcode::DMA_STORE_TILE;
        DMAOperands dops;
        dops.matrix = op.matrix;
        dops.tile = tile;
        dops.ext_mem_addr = ext_addr;
        dops.l3_tile_id = 0;
        dops.l3_offset = 0;
        dops.size_bytes = tile_bytes;
        dops.buffer = BufferSlot::BUF_0;
        instr.operands = dops;
        instr.label = "DMA_STORE";
        emit(instr);
        prog.estimates.external_mem_bytes += tile_bytes;
        break;
    }

    case ScheduleOpKind::MOVE: {
        Size rows = 0, cols = 0;
        if (op.matrix == MatrixID::A) {
            Size Ti = sched.get_tile("ti"); Size Tk = sched.get_tile("tk");
            rows = Ti ? Ti : 16; cols = Tk ? Tk : 16;
        } else if (op.matrix == MatrixID::B) {
            Size Tk = sched.get_tile("tk"); Size Tj = sched.get_tile("tj");
            rows = Tk ? Tk : 16; cols = Tj ? Tj : 16;
        } else {
            Size Ti = sched.get_tile("ti"); Size Tj = sched.get_tile("tj");
            rows = Ti ? Ti : 16; cols = Tj ? Tj : 16;
        }
        uint8_t src_l3 = static_cast<uint8_t>(buf);
        uint8_t dst_l2 = static_cast<uint8_t>(buf);
        emit(DMInstruction::bm_move(op.matrix, tile, src_l3, 0, dst_l2, 0,
                                    rows, cols, es, op.transform));
        prog.estimates.l2_bytes += rows * cols * es;
        break;
    }

    case ScheduleOpKind::WRITEBACK: {
        Size rows = 0, cols = 0;
        Size Ti = sched.get_tile("ti"); Size Tj = sched.get_tile("tj");
        rows = Ti ? Ti : 16; cols = Tj ? Tj : 16;

        DMInstruction instr;
        instr.opcode = DMOpcode::BM_WRITEBACK_TILE;
        BlockMoverOperands bops;
        bops.matrix = op.matrix;
        bops.tile = tile;
        bops.src_l3_tile_id = 0;  // src is L2 (field naming is reused)
        bops.src_offset = 0;
        bops.dst_l2_bank_id = 0;  // dst is L3
        bops.dst_offset = 0;
        bops.height = rows;
        bops.width = cols;
        bops.element_size = es;
        bops.transform = Transform::IDENTITY;
        bops.buffer = buf;
        instr.operands = bops;
        instr.label = "BM_WRITEBACK";
        emit(instr);
        break;
    }

    case ScheduleOpKind::STREAM_ROWS: {
        Size rows = 0, cols = 0;
        if (op.matrix == MatrixID::A) {
            Size Ti = sched.get_tile("ti"); Size Tk = sched.get_tile("tk");
            rows = Ti ? Ti : 16; cols = Tk ? Tk : 16;
        } else {
            Size Tb = sched.get_tile("tb");
            rows = Tb ? Tb : 16; cols = 16;
        }
        uint8_t l2b = static_cast<uint8_t>(buf);
        emit(DMInstruction::str_feed_rows(op.matrix, tile, l2b, 0, 0, 0, rows, cols, fs));
        break;
    }

    case ScheduleOpKind::STREAM_COLS: {
        Size Tk = sched.get_tile("tk"); Size Tj = sched.get_tile("tj");
        Size rows = Tk ? Tk : 16; Size cols = Tj ? Tj : 16;
        uint8_t l2b = static_cast<uint8_t>(buf);
        emit(DMInstruction::str_feed_cols(op.matrix, tile, l2b, 1, 0, 0, rows, cols, fs));
        break;
    }

    case ScheduleOpKind::BROADCAST: {
        DMInstruction instr;
        instr.opcode = DMOpcode::STR_BROADCAST_ROW;
        StreamerOperands sops;
        sops.matrix = op.matrix;
        sops.tile = tile;
        sops.l2_bank_id = 0;
        sops.l1_buffer_id = 0;
        sops.l2_addr = 0;
        sops.l1_addr = 0;
        sops.height = 1;
        sops.width = fs;
        sops.fabric_size = fs;
        sops.buffer = buf;
        instr.operands = sops;
        instr.label = "STR_BROADCAST";
        emit(instr);
        break;
    }

    case ScheduleOpKind::COMPUTE_MATMUL:
        // In dataflow architecture, compute fires reactively on data arrival.
        // Emitted as a NOP annotation — the systolic array fires automatically.
        {
            DMInstruction instr;
            instr.opcode = DMOpcode::NOP;
            instr.label = "COMPUTE_MATMUL (reactive)";
            emit(instr);
        }
        break;

    case ScheduleOpKind::COMPUTE_ELEMENTWISE: {
        DMInstruction instr;
        instr.opcode = DMOpcode::NOP;
        std::string op_name;
        switch (op.ewise_op) {
            case ElementwiseOp::SUB: op_name = "SUB"; break;
            case ElementwiseOp::EXP: op_name = "EXP"; break;
            case ElementwiseOp::DIV: op_name = "DIV"; break;
            case ElementwiseOp::MUL: op_name = "MUL"; break;
            case ElementwiseOp::ADD: op_name = "ADD"; break;
            case ElementwiseOp::RELU: op_name = "RELU"; break;
            default: op_name = "EWISE"; break;
        }
        instr.label = "VE_ELEMENTWISE(" + op_name + ")";
        emit(instr);
        break;
    }

    case ScheduleOpKind::COMPUTE_REDUCE: {
        DMInstruction instr;
        instr.opcode = DMOpcode::NOP;
        instr.label = (op.reduce_op == ReductionOp::MAX) ? "VE_REDUCE(MAX)" :
                      (op.reduce_op == ReductionOp::SUM) ? "VE_REDUCE(SUM)" :
                                                            "VE_REDUCE(MIN)";
        emit(instr);
        break;
    }

    case ScheduleOpKind::DRAIN: {
        Size Ti = sched.get_tile("ti"); Size Tj = sched.get_tile("tj");
        Size rows = Ti ? Ti : 16; Size cols = Tj ? Tj : 16;
        emit(DMInstruction::str_drain(tile, 0, 2, 0, 0, rows, cols, fs));
        break;
    }

    case ScheduleOpKind::DRAIN_FUSED: {
        Size Ti = sched.get_tile("ti"); Size Tj = sched.get_tile("tj");
        Size rows = Ti ? Ti : 16; Size cols = Tj ? Tj : 16;
        emit(DMInstruction::str_drain(tile, 0, 2, 0, 0, rows, cols, fs,
                                      true, op.activation, true, 0));
        break;
    }

    case ScheduleOpKind::DRAIN_TO_SCRATCH: {
        Size Ti = sched.get_tile("ti"); Size Tj = sched.get_tile("tj");
        Size Tb = sched.get_tile("tb");
        Size rows = Tb ? Tb : (Ti ? Ti : 16);
        Size cols = Tj ? Tj : 16;
        auto instr = DMInstruction::str_drain(tile, 0, 2, 0, 0, rows, cols, fs);
        instr.label = "STR_DRAIN_TO_SCRATCH(" + op.scratch_name + ")";
        emit(instr);
        break;
    }

    case ScheduleOpKind::BARRIER:
        emit(DMInstruction::barrier());
        break;

    case ScheduleOpKind::WAIT_DMA:
        emit(DMInstruction::wait(0x1));
        break;

    case ScheduleOpKind::DOUBLE_BUFFER:
    case ScheduleOpKind::IF_NOT_RESIDENT:
        // Hints — no instruction emitted; affect buffer assignment strategy
        break;

    case ScheduleOpKind::FOR_TILES:
        emit_loop(op);
        break;

    case ScheduleOpKind::END_LOOP:
    case ScheduleOpKind::STORE_SCATTER:
        break;
    }
}

void CompileContext::emit_ops(const std::vector<ScheduleOp>& ops) {
    for (const auto& op : ops) {
        emit_op(op);
    }
}

void CompileContext::emit_loop(const ScheduleOp& loop) {
    const auto& dim = loop.dim_name;
    Size full_dim = sched.get_dim(dim);
    Size tile_size = sched.get_tile(dim);
    if (tile_size == 0) tile_size = full_dim;
    Size count = (full_dim > 0 && tile_size > 0) ? (full_dim + tile_size - 1) / tile_size : 1;

    // Determine which index to iterate
    for (Size idx = 0; idx < count; ++idx) {
        if (dim == "ti" || dim == "M") {
            ti = static_cast<uint16_t>(idx);
        } else if (dim == "tj" || dim == "N") {
            tj = static_cast<uint16_t>(idx);
        } else if (dim == "tk" || dim == "K") {
            tk = static_cast<uint16_t>(idx);
            buf_toggle = static_cast<int>(idx);
        } else if (dim == "tb" || dim == "B") {
            ti = static_cast<uint16_t>(idx);
        }
        emit_ops(loop.body);
    }
}

} // anonymous namespace

DMProgram compile_schedule(const Schedule& sched) {
    DMProgram prog;

    // Metadata
    prog.name = sched.name();
    prog.version = 1;
    prog.dataflow = sched.get_dataflow();

    // Dimensions from tiles
    prog.M = sched.get_dim("ti");
    prog.N = sched.get_dim("tj");
    prog.K = sched.get_dim("tk");
    prog.Ti = sched.get_tile("ti");
    prog.Tj = sched.get_tile("tj");
    prog.Tk = sched.get_tile("tk");

    // If dimensions came from batch/D convention (softmax), set from those
    if (prog.M == 0) prog.M = sched.get_dim("tb");
    if (prog.Ti == 0) prog.Ti = sched.get_tile("tb");

    // Memory map from tensors
    for (const auto& t : sched.tensors()) {
        if (t.id == MatrixID::A) prog.memory_map.a_base = t.base_addr;
        if (t.id == MatrixID::B) prog.memory_map.b_base = t.base_addr;
        if (t.id == MatrixID::C) prog.memory_map.c_base = t.base_addr;
    }

    // Compile operations
    CompileContext ctx{sched, prog};
    ctx.emit_ops(sched.operations());

    // Halt
    ctx.emit(DMInstruction::halt());

    // Compute performance estimates
    if (prog.M > 0 && prog.N > 0 && prog.K > 0) {
        Size total_flops = 2ULL * prog.M * prog.N * prog.K;
        if (prog.estimates.external_mem_bytes > 0) {
            prog.estimates.arithmetic_intensity =
                static_cast<double>(total_flops) / prog.estimates.external_mem_bytes;
        }
    }

    return prog;
}

} // namespace sw::kpu::dsl
