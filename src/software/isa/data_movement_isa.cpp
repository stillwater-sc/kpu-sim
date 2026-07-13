/**
 * @file data_movement_isa.cpp
 * @brief Implementation of Data Movement ISA for Domain Flow Architecture
 */

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace sw::kpu::isa {

// ============================================================================
// DMInstruction Static Constructors
// ============================================================================

DMInstruction DMInstruction::dma_load(MatrixID mat, TileCoord tile, Address ext_mem_addr,
                                      uint8_t l3_tile, Address l3_offset, Size bytes) {
    DMInstruction instr;
    instr.opcode = DMOpcode::DMA_LOAD_TILE;

    DMAOperands ops;
    ops.matrix = mat;
    ops.tile = tile;
    ops.ext_mem_addr = ext_mem_addr;
    ops.l3_tile_id = l3_tile;
    ops.l3_offset = l3_offset;
    ops.size_bytes = bytes;
    ops.buffer = BufferSlot::AUTO;

    instr.operands = ops;

    // Generate label using matrix-space tile coordinates:
    // A[M,K] -> tile[ti,tk], B[K,N] -> tile[tk,tj], C[M,N] -> tile[ti,tj]
    std::ostringstream oss;
    oss << "DMA_LOAD ";
    switch (mat) {
        case MatrixID::A:
            oss << "A_tile[" << tile.ti << "," << tile.tk << "]";
            break;
        case MatrixID::B:
            oss << "B_tile[" << tile.tk << "," << tile.tj << "]";
            break;
        case MatrixID::C:
            oss << "C_tile[" << tile.ti << "," << tile.tj << "]";
            break;
    }
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::bm_move(MatrixID mat, TileCoord tile,
                                     uint8_t src_l3, Address src_off,
                                     uint8_t dst_l2, Address dst_off,
                                     Size height, Size width, Size elem_size,
                                     Transform xform) {
    DMInstruction instr;
    instr.opcode = (xform == Transform::IDENTITY) ? DMOpcode::BM_MOVE_TILE :
                   (xform == Transform::TRANSPOSE) ? DMOpcode::BM_TRANSPOSE_TILE :
                   DMOpcode::BM_RESHAPE_TILE;

    BlockMoverOperands ops;
    ops.matrix = mat;
    ops.tile = tile;
    ops.src_l3_tile_id = src_l3;
    ops.src_offset = src_off;
    ops.dst_l2_bank_id = dst_l2;
    ops.dst_offset = dst_off;
    ops.height = height;
    ops.width = width;
    ops.element_size = elem_size;
    ops.transform = xform;
    ops.buffer = BufferSlot::AUTO;

    instr.operands = ops;

    // Generate label using matrix-space tile coordinates
    std::ostringstream oss;
    oss << "BM_MOVE ";
    switch (mat) {
        case MatrixID::A:
            oss << "A_tile[" << tile.ti << "," << tile.tk << "]";
            break;
        case MatrixID::B:
            oss << "B_tile[" << tile.tk << "," << tile.tj << "]";
            break;
        case MatrixID::C:
            oss << "C_tile[" << tile.ti << "," << tile.tj << "]";
            break;
    }
    oss << " L3→L2";
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::str_feed_rows(MatrixID mat, TileCoord tile,
                                           uint8_t l2_bank, uint8_t l1_buf,
                                           Address l2_addr, Address l1_addr,
                                           Size height, Size width, Size fabric_size) {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_FEED_ROWS;

    StreamerOperands ops;
    ops.matrix = mat;
    ops.tile = tile;
    ops.l2_bank_id = l2_bank;
    ops.l1_buffer_id = l1_buf;
    ops.l2_addr = l2_addr;
    ops.l1_addr = l1_addr;
    ops.height = height;
    ops.width = width;
    ops.fabric_size = fabric_size;
    ops.buffer = BufferSlot::AUTO;

    instr.operands = ops;

    // Generate label using matrix-space tile coordinates
    std::ostringstream oss;
    oss << "STR_ROWS ";
    switch (mat) {
        case MatrixID::A:
            oss << "A_tile[" << tile.ti << "," << tile.tk << "]";
            break;
        case MatrixID::B:
            oss << "B_tile[" << tile.tk << "," << tile.tj << "]";
            break;
        case MatrixID::C:
            oss << "C_tile[" << tile.ti << "," << tile.tj << "]";
            break;
    }
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::str_feed_cols(MatrixID mat, TileCoord tile,
                                           uint8_t l2_bank, uint8_t l1_buf,
                                           Address l2_addr, Address l1_addr,
                                           Size height, Size width, Size fabric_size) {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_FEED_COLS;

    StreamerOperands ops;
    ops.matrix = mat;
    ops.tile = tile;
    ops.l2_bank_id = l2_bank;
    ops.l1_buffer_id = l1_buf;
    ops.l2_addr = l2_addr;
    ops.l1_addr = l1_addr;
    ops.height = height;
    ops.width = width;
    ops.fabric_size = fabric_size;
    ops.buffer = BufferSlot::AUTO;

    instr.operands = ops;

    // Generate label using matrix-space tile coordinates
    std::ostringstream oss;
    oss << "STR_COLS ";
    switch (mat) {
        case MatrixID::A:
            oss << "A_tile[" << tile.ti << "," << tile.tk << "]";
            break;
        case MatrixID::B:
            oss << "B_tile[" << tile.tk << "," << tile.tj << "]";
            break;
        case MatrixID::C:
            oss << "C_tile[" << tile.ti << "," << tile.tj << "]";
            break;
    }
    instr.label = oss.str();

    return instr;
}

namespace {

DMInstruction make_str_broadcast(DMOpcode opcode, const char* mnemonic,
                                 MatrixID mat, TileCoord tile,
                                 uint8_t l2_bank, uint8_t l1_buf,
                                 Address l2_addr, Address l1_addr,
                                 Size height, Size width, Size fabric_size) {
    DMInstruction instr;
    instr.opcode = opcode;

    StreamerOperands ops;
    ops.matrix = mat;
    ops.tile = tile;
    ops.l2_bank_id = l2_bank;
    ops.l1_buffer_id = l1_buf;
    ops.l2_addr = l2_addr;
    ops.l1_addr = l1_addr;
    ops.height = height;
    ops.width = width;
    ops.fabric_size = fabric_size;
    ops.buffer = BufferSlot::AUTO;

    instr.operands = ops;

    std::ostringstream oss;
    oss << mnemonic << " ";
    switch (mat) {
        case MatrixID::A:
            oss << "A_tile[" << tile.ti << "," << tile.tk << "]";
            break;
        case MatrixID::B:
            oss << "B_tile[" << tile.tk << "," << tile.tj << "]";
            break;
        case MatrixID::C:
            oss << "C_tile[" << tile.ti << "," << tile.tj << "]";
            break;
    }
    instr.label = oss.str();

    return instr;
}

} // namespace

DMInstruction DMInstruction::str_broadcast_row(MatrixID mat, TileCoord tile,
                                               uint8_t l2_bank, uint8_t l1_buf,
                                               Address l2_addr, Address l1_addr,
                                               Size height, Size width, Size fabric_size) {
    return make_str_broadcast(DMOpcode::STR_BROADCAST_ROW, "STR_BCAST_ROW",
                              mat, tile, l2_bank, l1_buf, l2_addr, l1_addr,
                              height, width, fabric_size);
}

DMInstruction DMInstruction::str_broadcast_col(MatrixID mat, TileCoord tile,
                                               uint8_t l2_bank, uint8_t l1_buf,
                                               Address l2_addr, Address l1_addr,
                                               Size height, Size width, Size fabric_size) {
    return make_str_broadcast(DMOpcode::STR_BROADCAST_COL, "STR_BCAST_COL",
                              mat, tile, l2_bank, l1_buf, l2_addr, l1_addr,
                              height, width, fabric_size);
}

DMInstruction DMInstruction::str_drain(TileCoord tile,
                                       uint8_t l2_bank, uint8_t l1_buf,
                                       Address l2_addr, Address l1_addr,
                                       Size height, Size width, Size fabric_size,
                                       bool ve_enabled,
                                       ActivationType ve_activation,
                                       bool ve_bias_enabled,
                                       Address ve_bias_addr) {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_DRAIN_OUTPUT;

    StreamerOperands ops;
    ops.matrix = MatrixID::C;
    ops.tile = tile;
    ops.l2_bank_id = l2_bank;
    ops.l1_buffer_id = l1_buf;
    ops.l2_addr = l2_addr;
    ops.l1_addr = l1_addr;
    ops.height = height;
    ops.width = width;
    ops.fabric_size = fabric_size;
    ops.buffer = BufferSlot::AUTO;

    // Vector Engine configuration for fused bias+activation
    ops.ve_enabled = ve_enabled;
    ops.ve_activation = ve_activation;
    ops.ve_bias_enabled = ve_bias_enabled;
    ops.ve_bias_addr = ve_bias_addr;

    instr.operands = ops;

    // C_tile uses [ti, tj] - the output matrix tile coordinates
    std::ostringstream oss;
    oss << "STR_DRAIN C_tile[" << tile.ti << "," << tile.tj << "]";
    if (ve_enabled) {
        oss << " +VE";
        if (ve_bias_enabled) {
            oss << "+bias";
        }
        if (ve_activation != ActivationType::NONE) {
            oss << "+" << activation_type_name(ve_activation);
        }
    }
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::barrier() {
    DMInstruction instr;
    instr.opcode = DMOpcode::BARRIER;
    instr.operands = std::monostate{};
    instr.label = "BARRIER";
    return instr;
}

DMInstruction DMInstruction::wait(uint32_t op_mask) {
    DMInstruction instr;
    instr.opcode = DMOpcode::WAIT_DMA;  // Generic wait

    SyncOperands ops;
    ops.wait_mask = op_mask;
    ops.signal_id = 0;

    instr.operands = ops;

    std::ostringstream oss;
    oss << "WAIT 0x" << std::hex << op_mask;
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::signal(uint32_t signal_id) {
    DMInstruction instr;
    instr.opcode = DMOpcode::SIGNAL;

    SyncOperands ops;
    ops.wait_mask = 0;
    ops.signal_id = signal_id;

    instr.operands = ops;

    std::ostringstream oss;
    oss << "SIGNAL " << signal_id;
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::halt() {
    DMInstruction instr;
    instr.opcode = DMOpcode::HALT;
    instr.operands = std::monostate{};
    instr.label = "HALT";
    return instr;
}

// ============================================================================
// Configuration Factory Methods (for loop machinery)
// ============================================================================

DMInstruction DMInstruction::set_base(MatrixID mat, Address base_addr) {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_BASE;

    BaseAddressOperands ops;
    ops.matrix = mat;
    ops.base_addr = base_addr;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "SET_BASE " << (mat == MatrixID::A ? "A" : mat == MatrixID::B ? "B" : "C")
        << ", 0x" << std::hex << base_addr;
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::set_l3_base(MatrixID mat, Address l3_offset) {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_L3_BASE;

    L3BaseOperands ops;
    ops.matrix = mat;
    ops.l3_offset = l3_offset;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "SET_L3_BASE " << (mat == MatrixID::A ? "A" : mat == MatrixID::B ? "B" : "C")
        << ", 0x" << std::hex << l3_offset;
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::set_l2_base(MatrixID mat, Address l2_offset) {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_L2_BASE;

    L2BaseOperands ops;
    ops.matrix = mat;
    ops.l2_offset = l2_offset;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "SET_L2_BASE " << (mat == MatrixID::A ? "A" : mat == MatrixID::B ? "B" : "C")
        << ", 0x" << std::hex << l2_offset;
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::set_stride(MatrixID mat, Size row_stride,
                                        Size tile_i_stride, Size tile_j_stride) {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_STRIDE;

    StrideConfigOperands ops;
    ops.matrix = mat;
    ops.row_stride = row_stride;
    ops.tile_i_stride = tile_i_stride;
    ops.tile_j_stride = tile_j_stride;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "SET_STRIDE " << (mat == MatrixID::A ? "A" : mat == MatrixID::B ? "B" : "C")
        << ", " << row_stride << ", " << tile_i_stride << ", " << tile_j_stride;
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::set_tile_dim(Size Ti, Size Tj, Size Tk, Size elem_size) {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_TILE_DIM;

    TileDimOperands ops;
    ops.Ti = Ti;
    ops.Tj = Tj;
    ops.Tk = Tk;
    ops.element_size = elem_size;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "SET_TILE_DIM " << Ti << ", " << Tj << ", " << Tk << ", " << elem_size;
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::set_matrix_dim(MatrixID mat, Size rows, Size cols) {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_MATRIX_DIM;

    MatrixDimOperands ops;
    ops.matrix = mat;
    ops.rows = rows;
    ops.cols = cols;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "SET_MATRIX_DIM " << (mat == MatrixID::A ? "A" : mat == MatrixID::B ? "B" : "C")
        << ", " << rows << ", " << cols;
    instr.label = oss.str();

    return instr;
}

// ============================================================================
// Loop Control Factory Methods
// ============================================================================

DMInstruction DMInstruction::loop_begin(uint8_t loop_id, uint16_t count,
                                        IndexRole role, uint16_t stride) {
    DMInstruction instr;
    instr.opcode = DMOpcode::LOOP_BEGIN;

    LoopOperands ops;
    ops.loop_id = loop_id;
    ops.loop_count = count;
    ops.loop_stride = stride;
    ops.index_role = role;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "LOOP_BEGIN " << static_cast<int>(loop_id) << ", " << count;
    switch (role) {
        case IndexRole::TI: oss << ", TI"; break;
        case IndexRole::TJ: oss << ", TJ"; break;
        case IndexRole::TK: oss << ", TK"; break;
        case IndexRole::NONE: break;
    }
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::loop_end(uint8_t loop_id) {
    DMInstruction instr;
    instr.opcode = DMOpcode::LOOP_END;

    LoopOperands ops;
    ops.loop_id = loop_id;
    ops.loop_count = 0;
    ops.loop_stride = 1;
    ops.index_role = IndexRole::NONE;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "LOOP_END " << static_cast<int>(loop_id);
    instr.label = oss.str();

    return instr;
}

// ============================================================================
// AUTO Addressing Factory Methods
// ============================================================================

DMInstruction DMInstruction::dma_load_auto(MatrixID mat, uint8_t l3_slot, BufferSlot buf) {
    DMInstruction instr;
    instr.opcode = DMOpcode::DMA_LOAD_TILE_AUTO;

    AutoDMAOperands ops;
    ops.matrix = mat;
    ops.l3_slot = l3_slot;
    ops.buffer = buf;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "DMA_LOAD_TILE_AUTO "
        << (mat == MatrixID::A ? "A" : mat == MatrixID::B ? "B" : "C")
        << ", " << static_cast<int>(l3_slot)
        << ", " << (buf == BufferSlot::BUF_0 ? "BUF_0" : buf == BufferSlot::BUF_1 ? "BUF_1" : "AUTO");
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::dma_store_auto(MatrixID mat, uint8_t l3_slot, BufferSlot buf) {
    DMInstruction instr;
    instr.opcode = DMOpcode::DMA_STORE_TILE_AUTO;

    AutoDMAOperands ops;
    ops.matrix = mat;
    ops.l3_slot = l3_slot;
    ops.buffer = buf;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "DMA_STORE_TILE_AUTO "
        << (mat == MatrixID::A ? "A" : mat == MatrixID::B ? "B" : "C")
        << ", " << static_cast<int>(l3_slot)
        << ", " << (buf == BufferSlot::BUF_0 ? "BUF_0" : buf == BufferSlot::BUF_1 ? "BUF_1" : "AUTO");
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::bm_move_auto(MatrixID mat, uint8_t l2_bank,
                                          BufferSlot buf, Transform xform) {
    DMInstruction instr;
    instr.opcode = DMOpcode::BM_MOVE_TILE_AUTO;

    AutoBlockMoverOperands ops;
    ops.matrix = mat;
    ops.l2_bank = l2_bank;
    ops.buffer = buf;
    ops.transform = xform;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "BM_MOVE_TILE_AUTO "
        << (mat == MatrixID::A ? "A" : mat == MatrixID::B ? "B" : "C")
        << ", " << static_cast<int>(l2_bank)
        << ", " << (buf == BufferSlot::BUF_0 ? "BUF_0" : buf == BufferSlot::BUF_1 ? "BUF_1" : "AUTO");
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::bm_writeback_auto(MatrixID mat, uint8_t l3_slot, BufferSlot buf) {
    DMInstruction instr;
    instr.opcode = DMOpcode::BM_WRITEBACK_AUTO;

    AutoBlockMoverOperands ops;
    ops.matrix = mat;
    ops.l2_bank = l3_slot;  // Reuse l2_bank field for l3_slot on writeback
    ops.buffer = buf;
    ops.transform = Transform::IDENTITY;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "BM_WRITEBACK_AUTO "
        << (mat == MatrixID::A ? "A" : mat == MatrixID::B ? "B" : "C")
        << ", " << static_cast<int>(l3_slot)
        << ", " << (buf == BufferSlot::BUF_0 ? "BUF_0" : buf == BufferSlot::BUF_1 ? "BUF_1" : "AUTO");
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::str_feed_rows_auto(uint8_t l1_buffer, BufferSlot buf) {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_FEED_ROWS_AUTO;

    AutoStreamerOperands ops;
    ops.l1_buffer = l1_buffer;
    ops.buffer = buf;
    ops.ve_enabled = false;
    ops.ve_activation = ActivationType::NONE;
    ops.ve_bias_enabled = false;
    ops.ve_bias_addr = 0;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "STR_FEED_ROWS_AUTO " << static_cast<int>(l1_buffer)
        << ", " << (buf == BufferSlot::BUF_0 ? "BUF_0" : buf == BufferSlot::BUF_1 ? "BUF_1" : "AUTO");
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::str_feed_cols_auto(uint8_t l1_buffer, BufferSlot buf) {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_FEED_COLS_AUTO;

    AutoStreamerOperands ops;
    ops.l1_buffer = l1_buffer;
    ops.buffer = buf;
    ops.ve_enabled = false;
    ops.ve_activation = ActivationType::NONE;
    ops.ve_bias_enabled = false;
    ops.ve_bias_addr = 0;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "STR_FEED_COLS_AUTO " << static_cast<int>(l1_buffer)
        << ", " << (buf == BufferSlot::BUF_0 ? "BUF_0" : buf == BufferSlot::BUF_1 ? "BUF_1" : "AUTO");
    instr.label = oss.str();

    return instr;
}

DMInstruction DMInstruction::str_drain_auto(uint8_t l2_bank, BufferSlot buf,
                                            bool ve_enabled, ActivationType ve_act) {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_DRAIN_AUTO;

    AutoStreamerOperands ops;
    ops.l1_buffer = l2_bank;  // Reuse l1_buffer field for l2_bank on drain
    ops.buffer = buf;
    ops.ve_enabled = ve_enabled;
    ops.ve_activation = ve_act;
    ops.ve_bias_enabled = false;
    ops.ve_bias_addr = 0;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "STR_DRAIN_AUTO " << static_cast<int>(l2_bank)
        << ", " << (buf == BufferSlot::BUF_0 ? "BUF_0" : buf == BufferSlot::BUF_1 ? "BUF_1" : "AUTO");
    if (ve_enabled) {
        oss << " +VE";
    }
    instr.label = oss.str();

    return instr;
}

// ============================================================================
// Vector Engine factory methods (issue #100)
// ============================================================================

const char* to_string(VEOp op) {
    switch (op) {
        case VEOp::ADD:   return "ADD";
        case VEOp::SUB:   return "SUB";
        case VEOp::MUL:   return "MUL";
        case VEOp::DIV:   return "DIV";
        case VEOp::MAX:   return "MAX";
        case VEOp::MIN:   return "MIN";
        case VEOp::NEG:   return "NEG";
        case VEOp::ABS:   return "ABS";
        case VEOp::SQRT:  return "SQRT";
        case VEOp::EXP:   return "EXP";
        case VEOp::LOG:   return "LOG";
        case VEOp::ADD_S: return "ADD_S";
        case VEOp::MUL_S: return "MUL_S";
        case VEOp::POW_S: return "POW_S";
    }
    return "UNKNOWN";
}

bool ve_op_from_string(const std::string& name, VEOp& op) {
    static const std::pair<const char*, VEOp> kMap[] = {
        {"ADD", VEOp::ADD},     {"SUB", VEOp::SUB},   {"MUL", VEOp::MUL},
        {"DIV", VEOp::DIV},     {"MAX", VEOp::MAX},   {"MIN", VEOp::MIN},
        {"NEG", VEOp::NEG},     {"ABS", VEOp::ABS},   {"SQRT", VEOp::SQRT},
        {"EXP", VEOp::EXP},     {"LOG", VEOp::LOG},
        {"ADD_S", VEOp::ADD_S}, {"MUL_S", VEOp::MUL_S},
        {"POW_S", VEOp::POW_S},
    };
    for (const auto& [key, value] : kMap) {
        if (name == key) {
            op = value;
            return true;
        }
    }
    return false;
}

DMInstruction DMInstruction::ve_elementwise(VEOp op, uint8_t l1_src_a,
                                            uint8_t l1_src_b, uint8_t l1_dst) {
    DMInstruction instr;
    instr.opcode = DMOpcode::VE_ELEMENTWISE;

    VEOperands ops;
    ops.op = op;
    ops.num_inputs = 2;
    ops.l1_src_a = l1_src_a;
    ops.l1_src_b = l1_src_b;
    ops.l1_dst = l1_dst;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "VE_ELEMENTWISE " << to_string(op)
        << ", L1[" << static_cast<int>(l1_src_a) << "]"
        << ", L1[" << static_cast<int>(l1_src_b) << "]"
        << " -> L1[" << static_cast<int>(l1_dst) << "]";
    instr.label = oss.str();
    return instr;
}

DMInstruction DMInstruction::ve_elementwise_unary(VEOp op, uint8_t l1_src,
                                                  uint8_t l1_dst) {
    DMInstruction instr;
    instr.opcode = DMOpcode::VE_ELEMENTWISE;

    VEOperands ops;
    ops.op = op;
    ops.num_inputs = 1;
    ops.l1_src_a = l1_src;
    ops.l1_dst = l1_dst;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "VE_ELEMENTWISE " << to_string(op)
        << ", L1[" << static_cast<int>(l1_src) << "]"
        << " -> L1[" << static_cast<int>(l1_dst) << "]";
    instr.label = oss.str();
    return instr;
}

DMInstruction DMInstruction::ve_elementwise_scalar(VEOp op, float scalar,
                                                   uint8_t l1_src,
                                                   uint8_t l1_dst) {
    DMInstruction instr;
    instr.opcode = DMOpcode::VE_ELEMENTWISE;

    VEOperands ops;
    ops.op = op;
    ops.num_inputs = 1;
    ops.scalar = scalar;
    ops.l1_src_a = l1_src;
    ops.l1_dst = l1_dst;
    instr.operands = ops;

    std::ostringstream oss;
    oss << "VE_ELEMENTWISE " << to_string(op) << " " << scalar
        << ", L1[" << static_cast<int>(l1_src) << "]"
        << " -> L1[" << static_cast<int>(l1_dst) << "]";
    instr.label = oss.str();
    return instr;
}

// ============================================================================
// DMProgram Statistics
// ============================================================================

size_t DMProgram::num_dma_ops() const {
    size_t count = 0;
    for (const auto& instr : instructions) {
        if (instr.opcode == DMOpcode::DMA_LOAD_TILE ||
            instr.opcode == DMOpcode::DMA_STORE_TILE ||
            instr.opcode == DMOpcode::DMA_PREFETCH_TILE ||
            instr.opcode == DMOpcode::DMA_LOAD_GATHER ||
            instr.opcode == DMOpcode::DMA_STORE_SCATTER ||
            instr.opcode == DMOpcode::DMA_LOAD_TILE_AUTO ||
            instr.opcode == DMOpcode::DMA_STORE_TILE_AUTO) {
            ++count;
        }
    }
    return count;
}

size_t DMProgram::num_bm_ops() const {
    size_t count = 0;
    for (const auto& instr : instructions) {
        if (instr.opcode == DMOpcode::BM_MOVE_TILE ||
            instr.opcode == DMOpcode::BM_TRANSPOSE_TILE ||
            instr.opcode == DMOpcode::BM_WRITEBACK_TILE ||
            instr.opcode == DMOpcode::BM_RESHAPE_TILE ||
            instr.opcode == DMOpcode::BM_MOVE_TILE_AUTO ||
            instr.opcode == DMOpcode::BM_WRITEBACK_AUTO) {
            ++count;
        }
    }
    return count;
}

size_t DMProgram::num_str_ops() const {
    size_t count = 0;
    for (const auto& instr : instructions) {
        if (instr.opcode == DMOpcode::STR_FEED_ROWS ||
            instr.opcode == DMOpcode::STR_FEED_COLS ||
            instr.opcode == DMOpcode::STR_DRAIN_OUTPUT ||
            instr.opcode == DMOpcode::STR_BROADCAST_ROW ||
            instr.opcode == DMOpcode::STR_BROADCAST_COL ||
            instr.opcode == DMOpcode::STR_FEED_ROWS_AUTO ||
            instr.opcode == DMOpcode::STR_FEED_COLS_AUTO ||
            instr.opcode == DMOpcode::STR_DRAIN_AUTO) {
            ++count;
        }
    }
    return count;
}

size_t DMProgram::num_sync_ops() const {
    size_t count = 0;
    for (const auto& instr : instructions) {
        if (instr.opcode == DMOpcode::BARRIER ||
            instr.opcode == DMOpcode::WAIT_DMA ||
            instr.opcode == DMOpcode::WAIT_BM ||
            instr.opcode == DMOpcode::WAIT_STR ||
            instr.opcode == DMOpcode::SIGNAL) {
            ++count;
        }
    }
    return count;
}

// ============================================================================
// OutputStationaryProgramBuilder Implementation
// ============================================================================

OutputStationaryProgramBuilder::OutputStationaryProgramBuilder(const Config& config)
    : config_(config), next_instruction_id_(0) {

    // Calculate tile counts
    m_tiles_ = (config_.M + config_.Ti - 1) / config_.Ti;
    n_tiles_ = (config_.N + config_.Tj - 1) / config_.Tj;
    k_tiles_ = (config_.K + config_.Tk - 1) / config_.Tk;

    // Initialize buffer offsets
    current_l3_offset_[0] = 0;
    current_l3_offset_[1] = 0;
    current_l2_offset_[0] = 0;
    current_l2_offset_[1] = 0;

    // Initialize tile cache capacity
    tile_cache_.capacity_bytes = config_.num_l3_tiles * config_.l3_tile_capacity;
    tile_cache_.reset();
}

std::string OutputStationaryProgramBuilder::get_cache_stats() const {
    std::ostringstream oss;
    oss << "\nTile Cache Statistics:\n";
    oss << "  Hits:       " << tile_cache_.hits << "\n";
    oss << "  Misses:     " << tile_cache_.misses << "\n";
    size_t total = tile_cache_.hits + tile_cache_.misses;
    double hit_rate = total > 0 ? (100.0 * static_cast<double>(tile_cache_.hits) / static_cast<double>(total)) : 0.0;
    oss << "  Hit rate:   " << std::fixed << std::setprecision(1) << hit_rate << "%\n";
    oss << "  Bytes saved: " << (static_cast<double>(tile_cache_.bytes_saved) / 1024.0) << " KB\n";
    oss << "  Resident tiles: " << tile_cache_.resident_tiles.size() << "\n";
    return oss.str();
}

Address OutputStationaryProgramBuilder::calculate_a_tile_addr(TileCoord tile) const {
    // A is MxK, stored row-major
    // A[ti, tk] starts at row (ti * Ti), column (tk * Tk)
    Size row_start = tile.ti * config_.Ti;
    Size col_start = tile.tk * config_.Tk;
    return (row_start * config_.K + col_start) * config_.element_size;
}

Address OutputStationaryProgramBuilder::calculate_b_tile_addr(TileCoord tile) const {
    // B is KxN, stored row-major
    // B[tk, tj] starts at row (tk * Tk), column (tj * Tj)
    Size row_start = tile.tk * config_.Tk;
    Size col_start = tile.tj * config_.Tj;
    return (row_start * config_.N + col_start) * config_.element_size;
}

Address OutputStationaryProgramBuilder::calculate_c_tile_addr(TileCoord tile) const {
    // C is MxN, stored row-major
    // C[ti, tj] starts at row (ti * Ti), column (tj * Tj)
    Size row_start = tile.ti * config_.Ti;
    Size col_start = tile.tj * config_.Tj;
    return (row_start * config_.N + col_start) * config_.element_size;
}

void OutputStationaryProgramBuilder::emit_load_a_tile(DMProgram& prog, TileCoord tile, BufferSlot buf) {
    // Calculate actual tile dimensions (handle edge tiles)
    Size actual_ti = std::min(config_.Ti, config_.M - tile.ti * config_.Ti);
    Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);
    Size tile_bytes = actual_ti * actual_tk * config_.element_size;

    Address ext_addr = prog.memory_map.a_base + calculate_a_tile_addr(tile);
    uint8_t l3_tile = static_cast<uint8_t>(buf);  // Use buffer slot as L3 tile ID
    Address l3_off = current_l3_offset_[static_cast<int>(buf)];

    auto instr = DMInstruction::dma_load(MatrixID::A, tile, ext_addr, l3_tile, l3_off, tile_bytes);
    instr.instruction_id = next_instruction_id_++;
    prog.instructions.push_back(instr);

    // Track L3 allocation
    DMProgram::MemoryMap::L3Alloc alloc;
    alloc.tile_id = l3_tile;
    alloc.offset = l3_off;
    alloc.size = tile_bytes;
    alloc.matrix = MatrixID::A;
    alloc.buffer = buf;
    prog.memory_map.l3_allocations.push_back(alloc);
}

void OutputStationaryProgramBuilder::emit_load_b_tile(DMProgram& prog, TileCoord tile, BufferSlot buf) {
    Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);
    Size actual_tj = std::min(config_.Tj, config_.N - tile.tj * config_.Tj);
    Size tile_bytes = actual_tk * actual_tj * config_.element_size;

    Address ext_addr = prog.memory_map.b_base + calculate_b_tile_addr(tile);
    uint8_t l3_tile = static_cast<uint8_t>(buf);
    Address l3_off = current_l3_offset_[static_cast<int>(buf)] + config_.Ti * config_.Tk * config_.element_size;

    auto instr = DMInstruction::dma_load(MatrixID::B, tile, ext_addr, l3_tile, l3_off, tile_bytes);
    instr.instruction_id = next_instruction_id_++;
    prog.instructions.push_back(instr);
}

bool OutputStationaryProgramBuilder::try_emit_load_a_tile(DMProgram& prog, TileCoord tile, BufferSlot buf) {
    // A tiles are indexed by (ti, tk) - check if already in cache
    if (config_.enable_tile_caching &&
        tile_cache_.is_resident(MatrixID::A, tile.ti, 0, tile.tk)) {
        // Cache hit - no DMA needed
        Size actual_ti = std::min(config_.Ti, config_.M - tile.ti * config_.Ti);
        Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);
        Size tile_bytes = actual_ti * actual_tk * config_.element_size;

        tile_cache_.hits++;
        tile_cache_.bytes_saved += tile_bytes;
        return false;  // No DMA emitted
    }

    // Cache miss - emit DMA and mark as resident
    emit_load_a_tile(prog, tile, buf);

    if (config_.enable_tile_caching) {
        Size actual_ti = std::min(config_.Ti, config_.M - tile.ti * config_.Ti);
        Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);
        Size tile_bytes = actual_ti * actual_tk * config_.element_size;

        tile_cache_.misses++;
        tile_cache_.mark_resident(MatrixID::A, tile.ti, 0, tile.tk, tile_bytes);
    }

    return true;  // DMA was emitted
}

bool OutputStationaryProgramBuilder::try_emit_load_b_tile(DMProgram& prog, TileCoord tile, BufferSlot buf) {
    // B tiles are indexed by (tk, tj) - check if already in cache
    if (config_.enable_tile_caching &&
        tile_cache_.is_resident(MatrixID::B, 0, tile.tj, tile.tk)) {
        // Cache hit - no DMA needed
        Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);
        Size actual_tj = std::min(config_.Tj, config_.N - tile.tj * config_.Tj);
        Size tile_bytes = actual_tk * actual_tj * config_.element_size;

        tile_cache_.hits++;
        tile_cache_.bytes_saved += tile_bytes;
        return false;  // No DMA emitted
    }

    // Cache miss - emit DMA and mark as resident
    emit_load_b_tile(prog, tile, buf);

    if (config_.enable_tile_caching) {
        Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);
        Size actual_tj = std::min(config_.Tj, config_.N - tile.tj * config_.Tj);
        Size tile_bytes = actual_tk * actual_tj * config_.element_size;

        tile_cache_.misses++;
        tile_cache_.mark_resident(MatrixID::B, 0, tile.tj, tile.tk, tile_bytes);
    }

    return true;  // DMA was emitted
}

void OutputStationaryProgramBuilder::emit_move_a_l3_to_l2(DMProgram& prog, TileCoord tile, BufferSlot buf) {
    Size actual_ti = std::min(config_.Ti, config_.M - tile.ti * config_.Ti);
    Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);

    uint8_t src_l3 = static_cast<uint8_t>(buf);
    Address src_off = 0;  // A is at start of L3 buffer
    uint8_t dst_l2 = static_cast<uint8_t>(buf);
    Address dst_off = current_l2_offset_[static_cast<int>(buf)];

    auto instr = DMInstruction::bm_move(MatrixID::A, tile, src_l3, src_off, dst_l2, dst_off,
                                        actual_ti, actual_tk, config_.element_size);
    instr.instruction_id = next_instruction_id_++;
    prog.instructions.push_back(instr);
}

void OutputStationaryProgramBuilder::emit_move_b_l3_to_l2(DMProgram& prog, TileCoord tile, BufferSlot buf) {
    Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);
    Size actual_tj = std::min(config_.Tj, config_.N - tile.tj * config_.Tj);

    uint8_t src_l3 = static_cast<uint8_t>(buf);
    Address src_off = config_.Ti * config_.Tk * config_.element_size;  // B after A in L3
    uint8_t dst_l2 = static_cast<uint8_t>(buf);
    Address dst_off = current_l2_offset_[static_cast<int>(buf)] + config_.Ti * config_.Tk * config_.element_size;

    auto instr = DMInstruction::bm_move(MatrixID::B, tile, src_l3, src_off, dst_l2, dst_off,
                                        actual_tk, actual_tj, config_.element_size);
    instr.instruction_id = next_instruction_id_++;
    prog.instructions.push_back(instr);
}

void OutputStationaryProgramBuilder::emit_stream_a_rows(DMProgram& prog, TileCoord tile, BufferSlot buf) {
    Size actual_ti = std::min(config_.Ti, config_.M - tile.ti * config_.Ti);
    Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);

    uint8_t l2_bank = static_cast<uint8_t>(buf);
    uint8_t l1_buf = 0;
    Address l2_addr = current_l2_offset_[static_cast<int>(buf)];
    Address l1_addr = 0;

    auto instr = DMInstruction::str_feed_rows(MatrixID::A, tile, l2_bank, l1_buf,
                                              l2_addr, l1_addr,
                                              actual_ti, actual_tk, config_.systolic_size);
    instr.instruction_id = next_instruction_id_++;
    prog.instructions.push_back(instr);
}

void OutputStationaryProgramBuilder::emit_stream_b_cols(DMProgram& prog, TileCoord tile, BufferSlot buf) {
    Size actual_tk = std::min(config_.Tk, config_.K - tile.tk * config_.Tk);
    Size actual_tj = std::min(config_.Tj, config_.N - tile.tj * config_.Tj);

    uint8_t l2_bank = static_cast<uint8_t>(buf);
    uint8_t l1_buf = 1;  // B uses different L1 buffer
    Address l2_addr = current_l2_offset_[static_cast<int>(buf)] + config_.Ti * config_.Tk * config_.element_size;
    Address l1_addr = 0;

    auto instr = DMInstruction::str_feed_cols(MatrixID::B, tile, l2_bank, l1_buf,
                                              l2_addr, l1_addr,
                                              actual_tk, actual_tj, config_.systolic_size);
    instr.instruction_id = next_instruction_id_++;
    prog.instructions.push_back(instr);
}

void OutputStationaryProgramBuilder::emit_drain_c(DMProgram& prog, TileCoord tile) {
    Size actual_ti = std::min(config_.Ti, config_.M - tile.ti * config_.Ti);
    Size actual_tj = std::min(config_.Tj, config_.N - tile.tj * config_.Tj);

    uint8_t l2_bank = 0;  // C drains to bank 0
    uint8_t l1_buf = 2;   // C uses L1 buffer 2
    Address l2_addr = 0;
    Address l1_addr = 0;

    auto instr = DMInstruction::str_drain(tile, l2_bank, l1_buf, l2_addr, l1_addr,
                                          actual_ti, actual_tj, config_.systolic_size);
    instr.instruction_id = next_instruction_id_++;
    prog.instructions.push_back(instr);
}

void OutputStationaryProgramBuilder::emit_store_c_tile(DMProgram& prog, TileCoord tile) {
    Size actual_ti = std::min(config_.Ti, config_.M - tile.ti * config_.Ti);
    Size actual_tj = std::min(config_.Tj, config_.N - tile.tj * config_.Tj);
    Size tile_bytes = actual_ti * actual_tj * config_.element_size;

    Address ext_addr = prog.memory_map.c_base + calculate_c_tile_addr(tile);

    DMInstruction instr;
    instr.opcode = DMOpcode::DMA_STORE_TILE;

    DMAOperands ops;
    ops.matrix = MatrixID::C;
    ops.tile = tile;
    ops.ext_mem_addr = ext_addr;
    ops.l3_tile_id = 0;
    ops.l3_offset = 0;
    ops.size_bytes = tile_bytes;
    ops.buffer = BufferSlot::BUF_0;

    instr.operands = ops;
    instr.instruction_id = next_instruction_id_++;

    // C_tile uses [ti, tj] - the output matrix tile coordinates
    std::ostringstream oss;
    oss << "DMA_STORE C_tile[" << tile.ti << "," << tile.tj << "]";
    instr.label = oss.str();

    prog.instructions.push_back(instr);
}

void OutputStationaryProgramBuilder::emit_barrier(DMProgram& prog) {
    auto instr = DMInstruction::barrier();
    instr.instruction_id = next_instruction_id_++;
    prog.instructions.push_back(instr);
}

DMProgram OutputStationaryProgramBuilder::build() {
    DMProgram prog;

    // Set program metadata
    std::ostringstream name_oss;
    name_oss << "matmul_" << config_.M << "x" << config_.N << "x" << config_.K << "_os";
    prog.name = name_oss.str();
    prog.version = 1;

    // Set dimensions
    prog.M = config_.M;
    prog.N = config_.N;
    prog.K = config_.K;
    prog.Ti = config_.Ti;
    prog.Tj = config_.Tj;
    prog.Tk = config_.Tk;
    prog.L1_Ki = config_.L1_Ki;
    prog.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

    // Initialize memory map (base addresses set at load time)
    prog.memory_map.a_base = 0;
    prog.memory_map.b_base = 0;
    prog.memory_map.c_base = 0;

    // Initialize estimates
    prog.estimates.total_cycles = 0;
    prog.estimates.external_mem_bytes = 0;
    prog.estimates.l3_bytes = 0;
    prog.estimates.l2_bytes = 0;

    /**
     * Output-Stationary Loop Structure:
     *
     * for ti in 0..M/Ti:           // Output row tiles
     *   for tj in 0..N/Tj:         // Output col tiles
     *     // C[ti,tj] accumulates in PEs - NO WRITEBACK during K loop
     *     for tk in 0..K/Tk:       // Reduction tiles (innermost)
     *       Load A[ti,tk] from external memory to L3
     *       Load B[tk,tj] from external memory to L3
     *       Move A[ti,tk] from L3 to L2
     *       Move B[tk,tj] from L3 to L2
     *       Stream A rows to systolic array
     *       Stream B cols to systolic array
     *       // Compute: C[ti,tj] += A[ti,tk] × B[tk,tj] happens reactively
     *     // After all K tiles accumulated:
     *     Drain C[ti,tj] from PEs
     *     Store C[ti,tj] to external memory
     */

    /**
     * Pipelined Output-Stationary Execution:
     *
     * For each output tile C[ti,tj], we stream ALL k-dimension tiles continuously
     * without barriers between them. The systolic array accumulates in place.
     *
     * Key optimizations:
     * 1. Prefetch: Load/move next k-tile while current is streaming
     * 2. No barriers within K loop - continuous accumulation
     * 3. Double-buffering for overlap of data movement and compute
     * 4. Only barrier after all K tiles are streamed (before drain)
     */

    for (Size ti = 0; ti < m_tiles_; ++ti) {
        for (Size tj = 0; tj < n_tiles_; ++tj) {
            // === PHASE 1: Load first k-tile to prime the pipeline ===
            TileCoord first_tile{static_cast<uint16_t>(ti),
                                static_cast<uint16_t>(tj),
                                0};

            BufferSlot buf_0 = BufferSlot::BUF_0;
            BufferSlot buf_1 = BufferSlot::BUF_1;

            // Load first tiles to L3
            bool a_loaded = try_emit_load_a_tile(prog, first_tile, buf_0);
            bool b_loaded = try_emit_load_b_tile(prog, first_tile, buf_0);

            if (a_loaded || b_loaded) {
                emit_barrier(prog);
            }

            // Move first tiles to L2
            emit_move_a_l3_to_l2(prog, first_tile, buf_0);
            emit_move_b_l3_to_l2(prog, first_tile, buf_0);
            emit_barrier(prog);

            // === PHASE 2: Pipelined K-loop ===
            // Stream all K tiles continuously without barriers
            // Prefetch next tile while current streams

            for (Size tk = 0; tk < k_tiles_; ++tk) {
                TileCoord current_tile{static_cast<uint16_t>(ti),
                                       static_cast<uint16_t>(tj),
                                       static_cast<uint16_t>(tk)};

                BufferSlot current_buf = (tk % 2 == 0) ? buf_0 : buf_1;
                BufferSlot next_buf = (tk % 2 == 0) ? buf_1 : buf_0;

                // Prefetch next k-tile (overlapped with current streaming)
                // This issues DMA and BM commands that execute concurrently
                if (tk + 1 < k_tiles_) {
                    TileCoord next_tile{static_cast<uint16_t>(ti),
                                        static_cast<uint16_t>(tj),
                                        static_cast<uint16_t>(tk + 1)};

                    // DMA load (concurrent with streaming below)
                    bool a_prefetched = try_emit_load_a_tile(prog, next_tile, next_buf);
                    bool b_prefetched = try_emit_load_b_tile(prog, next_tile, next_buf);

                    // BM move will wait for DMA internally (no barrier needed here)
                    // The executor handles resource dependencies
                    if (a_prefetched) {
                        emit_move_a_l3_to_l2(prog, next_tile, next_buf);
                    }
                    if (b_prefetched) {
                        emit_move_b_l3_to_l2(prog, next_tile, next_buf);
                    }
                }

                // Stream current tile to systolic array
                // A rows and B columns stream concurrently, accumulating into C
                // NO BARRIER - continuous streaming across all k tiles
                emit_stream_a_rows(prog, current_tile, current_buf);
                emit_stream_b_cols(prog, current_tile, current_buf);

                // Update traffic estimates
                Size a_tile_bytes = std::min(config_.Ti, config_.M - ti * config_.Ti) *
                                   std::min(config_.Tk, config_.K - tk * config_.Tk) *
                                   config_.element_size;
                Size b_tile_bytes = std::min(config_.Tk, config_.K - tk * config_.Tk) *
                                   std::min(config_.Tj, config_.N - tj * config_.Tj) *
                                   config_.element_size;

                // Only count external mem traffic if DMA was actually issued
                if (tk == 0) {
                    if (a_loaded) prog.estimates.external_mem_bytes += a_tile_bytes;
                    if (b_loaded) prog.estimates.external_mem_bytes += b_tile_bytes;
                } else {
                    // For subsequent tiles, check cache (simplified tracking)
                    prog.estimates.external_mem_bytes += a_tile_bytes + b_tile_bytes;
                }
                prog.estimates.l3_bytes += a_tile_bytes + b_tile_bytes;
                prog.estimates.l2_bytes += a_tile_bytes + b_tile_bytes;
            }

            // === PHASE 3: Barrier after all K tiles, then drain C ===
            // This ensures all accumulation is complete before draining
            emit_barrier(prog);

            TileCoord c_tile{static_cast<uint16_t>(ti),
                            static_cast<uint16_t>(tj),
                            0};

            emit_drain_c(prog, c_tile);
            emit_store_c_tile(prog, c_tile);
            emit_barrier(prog);

            // Update C traffic
            Size c_tile_bytes = std::min(config_.Ti, config_.M - ti * config_.Ti) *
                               std::min(config_.Tj, config_.N - tj * config_.Tj) *
                               config_.element_size;
            prog.estimates.external_mem_bytes += c_tile_bytes;
        }
    }

    // Add HALT instruction
    auto halt = DMInstruction::halt();
    halt.instruction_id = next_instruction_id_++;
    prog.instructions.push_back(halt);

    // Calculate final estimates
    Size total_flops = 2 * config_.M * config_.N * config_.K;
    prog.estimates.arithmetic_intensity =
        static_cast<double>(total_flops) / static_cast<double>(prog.estimates.external_mem_bytes);

    return prog;
}

} // namespace sw::kpu::isa
