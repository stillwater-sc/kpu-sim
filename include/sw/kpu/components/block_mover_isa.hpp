// ============================================================================
// include/sw/kpu/components/block_mover_isa.hpp
// BlockMover Instruction Set Architecture for stateful tile data movement
// ============================================================================

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace sw::kpu {

// ============================================================================
// Tensor Identification
// ============================================================================

/// Identifies which tensor a tile belongs to
enum class TensorId : uint8_t {
    A = 0,          // Input matrix A
    B = 1,          // Input matrix B
    C = 2,          // Output matrix C (accumulator)
    BIAS = 3,       // Bias vector
    WORKSPACE = 4,  // Temporary workspace
    CUSTOM = 5,     // User-defined tensor
};

inline const char* to_string(TensorId id) {
    switch (id) {
        case TensorId::A: return "A";
        case TensorId::B: return "B";
        case TensorId::C: return "C";
        case TensorId::BIAS: return "BIAS";
        case TensorId::WORKSPACE: return "WORKSPACE";
        case TensorId::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Tile Descriptor
// ============================================================================

/// Describes a tensor tile's identity and location
struct TileDescriptor {
    TensorId tensor = TensorId::A;

    // Tile indices within the tensor's tiling scheme
    uint16_t m_tile = 0;    // Row tile index
    uint16_t n_tile = 0;    // Column tile index
    uint16_t k_tile = 0;    // K-dimension tile index (for A and B)

    // Memory layout
    uint32_t l3_offset = 0;     // Offset within L3 tile (bytes)
    uint32_t size = 0;          // Transfer size (bytes)

    // Tile dimensions (elements, not bytes)
    uint16_t height = 0;        // Rows in this tile
    uint16_t width = 0;         // Columns in this tile

    // Stride for 2D transfers
    uint32_t src_stride = 0;    // Source row stride (bytes)
    uint32_t dst_stride = 0;    // Destination row stride (bytes)

    bool operator==(const TileDescriptor& other) const {
        return tensor == other.tensor &&
               m_tile == other.m_tile &&
               n_tile == other.n_tile &&
               k_tile == other.k_tile;
    }

    std::string to_string() const {
        return std::string(sw::kpu::to_string(tensor)) +
               "[" + std::to_string(m_tile) +
               "," + std::to_string(n_tile) +
               "," + std::to_string(k_tile) + "]";
    }
};

// ============================================================================
// BlockMover Operations
// ============================================================================

/// BlockMover instruction opcodes
enum class BlockMoverOp : uint8_t {
    // ========== No-op ==========
    NOP = 0,

    // ========== L3 → L2 Operations (original functionality) ==========
    PUSH_TO_L2 = 1,         // Transfer tile from L3 to L2 bank

    // ========== L3 ↔ L3 Operations (inter-tile transfers) ==========
    SEND_EAST = 10,         // Send tile to L3[id + 1]
    SEND_WEST = 11,         // Send tile to L3[id - 1]
    SEND_NORTH = 12,        // Send tile to L3[id - row_size]
    SEND_SOUTH = 13,        // Send tile to L3[id + row_size]
    SEND_TO = 14,           // Send tile to arbitrary L3 (uses dest_l3_id)

    RECEIVE = 20,           // Accept incoming tile transfer
    RECEIVE_FROM = 21,      // Accept from specific source (uses src_l3_id)

    // ========== L2 → L3 Operations (drain) ==========
    PULL_FROM_L2 = 30,      // Transfer tile from L2 back to L3

    // ========== Synchronization ==========
    WAIT_TRIGGER = 40,      // Block until specified trigger(s) received
    EMIT_TRIGGER = 41,      // Signal trigger to specified destination(s)
    BARRIER = 42,           // Wait for all pending transfers to complete
    FENCE = 43,             // Memory fence (ensure ordering)

    // ========== Control Flow ==========
    LOOP_START = 50,        // Begin loop (loop_count iterations)
    LOOP_END = 51,          // End loop, decrement counter, branch if not zero
    JUMP = 52,              // Unconditional jump (relative offset)
    JUMP_IF_TRIGGER = 53,   // Conditional jump if trigger set

    // ========== Configuration ==========
    SET_L2_BANK = 60,       // Configure target L2 bank for subsequent ops
    SET_BUFFER = 61,        // Configure which L3 buffer region to use

    // ========== Debug/Profiling ==========
    TRACE_MARKER = 70,      // Emit trace event for profiling
    HALT = 71,              // Stop execution (for debugging)
};

inline const char* to_string(BlockMoverOp op) {
    switch (op) {
        case BlockMoverOp::NOP: return "NOP";
        case BlockMoverOp::PUSH_TO_L2: return "PUSH_TO_L2";
        case BlockMoverOp::SEND_EAST: return "SEND_EAST";
        case BlockMoverOp::SEND_WEST: return "SEND_WEST";
        case BlockMoverOp::SEND_NORTH: return "SEND_NORTH";
        case BlockMoverOp::SEND_SOUTH: return "SEND_SOUTH";
        case BlockMoverOp::SEND_TO: return "SEND_TO";
        case BlockMoverOp::RECEIVE: return "RECEIVE";
        case BlockMoverOp::RECEIVE_FROM: return "RECEIVE_FROM";
        case BlockMoverOp::PULL_FROM_L2: return "PULL_FROM_L2";
        case BlockMoverOp::WAIT_TRIGGER: return "WAIT_TRIGGER";
        case BlockMoverOp::EMIT_TRIGGER: return "EMIT_TRIGGER";
        case BlockMoverOp::BARRIER: return "BARRIER";
        case BlockMoverOp::FENCE: return "FENCE";
        case BlockMoverOp::LOOP_START: return "LOOP_START";
        case BlockMoverOp::LOOP_END: return "LOOP_END";
        case BlockMoverOp::JUMP: return "JUMP";
        case BlockMoverOp::JUMP_IF_TRIGGER: return "JUMP_IF_TRIGGER";
        case BlockMoverOp::SET_L2_BANK: return "SET_L2_BANK";
        case BlockMoverOp::SET_BUFFER: return "SET_BUFFER";
        case BlockMoverOp::TRACE_MARKER: return "TRACE_MARKER";
        case BlockMoverOp::HALT: return "HALT";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Trigger Channels
// ============================================================================

/// Well-known trigger channel IDs for common synchronization patterns
namespace TriggerChannel {
    constexpr uint8_t DMA_LOAD_DONE = 0;      // DMA completed loading to L3
    constexpr uint8_t A_TILE_READY = 1;       // A tile available in L3
    constexpr uint8_t B_TILE_READY = 2;       // B tile available in L3
    constexpr uint8_t C_TILE_READY = 3;       // C tile available in L3
    constexpr uint8_t COMPUTE_START = 4;      // Signal to start compute
    constexpr uint8_t COMPUTE_DONE = 5;       // Compute completed
    constexpr uint8_t DRAIN_DONE = 6;         // C tile drained to L3
    constexpr uint8_t TRANSFER_DONE = 7;      // L3↔L3 transfer completed

    // User-defined triggers: 8-15
    constexpr uint8_t USER_0 = 8;
    constexpr uint8_t USER_7 = 15;

    constexpr uint8_t NUM_CHANNELS = 16;
}

// ============================================================================
// BlockMover Command
// ============================================================================

/// A single BlockMover instruction
struct BlockMoverCommand {
    BlockMoverOp op = BlockMoverOp::NOP;

    // Tile descriptor (for data movement operations)
    TileDescriptor tile;

    // L3 addressing (for inter-L3 operations)
    uint8_t dest_l3_id = 0;     // Destination L3 tile ID (for SEND_TO)
    uint8_t src_l3_id = 0;      // Source L3 tile ID (for RECEIVE_FROM)

    // L2 addressing
    uint8_t l2_bank_id = 0;     // Target L2 bank
    uint32_t l2_offset = 0;     // Offset within L2 bank

    // Synchronization
    uint8_t trigger_id = 0;         // Which trigger channel (0-15)
    uint16_t trigger_wait_mask = 0; // Bitmask of triggers to wait for
    uint16_t trigger_dest_mask = 0; // Bitmask of L3s to send trigger to

    // Control flow
    uint16_t loop_count = 0;    // Number of loop iterations
    int16_t jump_offset = 0;    // Relative PC offset for jumps

    // Debug/trace
    uint32_t trace_id = 0;      // User-defined trace marker ID

    // Estimated timing (filled by compiler, used by simulator)
    uint16_t latency_cycles = 0;

    std::string to_string() const {
        std::string s = sw::kpu::to_string(op);
        switch (op) {
            case BlockMoverOp::PUSH_TO_L2:
            case BlockMoverOp::PULL_FROM_L2:
                s += " " + tile.to_string() + " → L2[" + std::to_string(l2_bank_id) + "]";
                break;
            case BlockMoverOp::SEND_EAST:
            case BlockMoverOp::SEND_WEST:
            case BlockMoverOp::SEND_NORTH:
            case BlockMoverOp::SEND_SOUTH:
                s += " " + tile.to_string();
                break;
            case BlockMoverOp::SEND_TO:
                s += " " + tile.to_string() + " → L3[" + std::to_string(dest_l3_id) + "]";
                break;
            case BlockMoverOp::WAIT_TRIGGER:
                s += " mask=0x" + std::to_string(trigger_wait_mask);
                break;
            case BlockMoverOp::EMIT_TRIGGER:
                s += " ch=" + std::to_string(trigger_id) + " dest=0x" + std::to_string(trigger_dest_mask);
                break;
            case BlockMoverOp::LOOP_START:
                s += " count=" + std::to_string(loop_count);
                break;
            case BlockMoverOp::LOOP_END:
            case BlockMoverOp::JUMP:
                s += " offset=" + std::to_string(jump_offset);
                break;
            default:
                break;
        }
        return s;
    }
};

// ============================================================================
// BlockMover Program
// ============================================================================

/// A complete BlockMover program (sequence of commands)
struct BlockMoverProgram {
    uint8_t l3_id = 0;                          // Which L3 tile this runs on
    std::vector<BlockMoverCommand> commands;

    // Metadata
    std::string name;
    uint32_t estimated_cycles = 0;

    size_t size() const { return commands.size(); }
    bool empty() const { return commands.empty(); }

    void append(const BlockMoverCommand& cmd) {
        commands.push_back(cmd);
    }

    void append(BlockMoverOp op) {
        BlockMoverCommand cmd;
        cmd.op = op;
        commands.push_back(cmd);
    }

    // Builder pattern helpers
    BlockMoverProgram& push_to_l2(const TileDescriptor& tile, uint8_t l2_bank, uint32_t l2_off = 0) {
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::PUSH_TO_L2;
        cmd.tile = tile;
        cmd.l2_bank_id = l2_bank;
        cmd.l2_offset = l2_off;
        commands.push_back(cmd);
        return *this;
    }

    BlockMoverProgram& send_east(const TileDescriptor& tile) {
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::SEND_EAST;
        cmd.tile = tile;
        commands.push_back(cmd);
        return *this;
    }

    BlockMoverProgram& send_south(const TileDescriptor& tile) {
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::SEND_SOUTH;
        cmd.tile = tile;
        commands.push_back(cmd);
        return *this;
    }

    BlockMoverProgram& receive() {
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::RECEIVE;
        commands.push_back(cmd);
        return *this;
    }

    BlockMoverProgram& wait_trigger(uint16_t mask) {
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::WAIT_TRIGGER;
        cmd.trigger_wait_mask = mask;
        commands.push_back(cmd);
        return *this;
    }

    BlockMoverProgram& wait_trigger(uint8_t channel) {
        return wait_trigger(static_cast<uint16_t>(1 << channel));
    }

    BlockMoverProgram& emit_trigger(uint8_t channel, uint16_t dest_mask = 0xFFFF) {
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::EMIT_TRIGGER;
        cmd.trigger_id = channel;
        cmd.trigger_dest_mask = dest_mask;
        commands.push_back(cmd);
        return *this;
    }

    BlockMoverProgram& barrier() {
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::BARRIER;
        commands.push_back(cmd);
        return *this;
    }

    BlockMoverProgram& loop_start(uint16_t count) {
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::LOOP_START;
        cmd.loop_count = count;
        commands.push_back(cmd);
        return *this;
    }

    BlockMoverProgram& loop_end() {
        BlockMoverCommand cmd;
        cmd.op = BlockMoverOp::LOOP_END;
        // jump_offset will be computed during assembly
        commands.push_back(cmd);
        return *this;
    }

    // Resolve loop offsets (must be called before execution)
    void assemble() {
        std::vector<size_t> loop_starts;
        for (size_t i = 0; i < commands.size(); ++i) {
            if (commands[i].op == BlockMoverOp::LOOP_START) {
                loop_starts.push_back(i);
            } else if (commands[i].op == BlockMoverOp::LOOP_END) {
                if (!loop_starts.empty()) {
                    size_t start = loop_starts.back();
                    loop_starts.pop_back();
                    // LOOP_END jumps back to LOOP_START
                    commands[i].jump_offset = static_cast<int16_t>(start) - static_cast<int16_t>(i);
                }
            }
        }
    }

    std::string to_string() const {
        std::string s = "BlockMoverProgram[L3=" + std::to_string(l3_id) + "] {\n";
        for (size_t i = 0; i < commands.size(); ++i) {
            s += "  " + std::to_string(i) + ": " + commands[i].to_string() + "\n";
        }
        s += "}";
        return s;
    }
};

} // namespace sw::kpu
