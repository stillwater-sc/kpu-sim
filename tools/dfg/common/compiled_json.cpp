#include "common/compiled_json.hpp"
#include <fstream>
#include <stdexcept>

namespace kpu::dfg::json {

using namespace sw::kpu::dataflow;
using namespace sw::kpu;

// ============================================================================
// BlockMoverOp conversion
// ============================================================================

std::string op_to_string(BlockMoverOp op) {
    return sw::kpu::to_string(op);
}

BlockMoverOp op_from_string(const std::string& s) {
    if (s == "NOP") return BlockMoverOp::NOP;
    if (s == "PUSH_TO_L2") return BlockMoverOp::PUSH_TO_L2;
    if (s == "SEND_EAST") return BlockMoverOp::SEND_EAST;
    if (s == "SEND_WEST") return BlockMoverOp::SEND_WEST;
    if (s == "SEND_NORTH") return BlockMoverOp::SEND_NORTH;
    if (s == "SEND_SOUTH") return BlockMoverOp::SEND_SOUTH;
    if (s == "SEND_TO") return BlockMoverOp::SEND_TO;
    if (s == "RECEIVE") return BlockMoverOp::RECEIVE;
    if (s == "RECEIVE_FROM") return BlockMoverOp::RECEIVE_FROM;
    if (s == "PULL_FROM_L2") return BlockMoverOp::PULL_FROM_L2;
    if (s == "WAIT_TRIGGER") return BlockMoverOp::WAIT_TRIGGER;
    if (s == "EMIT_TRIGGER") return BlockMoverOp::EMIT_TRIGGER;
    if (s == "BARRIER") return BlockMoverOp::BARRIER;
    if (s == "FENCE") return BlockMoverOp::FENCE;
    if (s == "WAIT_DELIVERY") return BlockMoverOp::WAIT_DELIVERY;
    if (s == "WAIT_UNTIL_CYCLE") return BlockMoverOp::WAIT_UNTIL_CYCLE;
    if (s == "LOOP_START") return BlockMoverOp::LOOP_START;
    if (s == "LOOP_END") return BlockMoverOp::LOOP_END;
    if (s == "JUMP") return BlockMoverOp::JUMP;
    if (s == "JUMP_IF_TRIGGER") return BlockMoverOp::JUMP_IF_TRIGGER;
    if (s == "SET_L2_BANK") return BlockMoverOp::SET_L2_BANK;
    if (s == "SET_BUFFER") return BlockMoverOp::SET_BUFFER;
    if (s == "TRACE_MARKER") return BlockMoverOp::TRACE_MARKER;
    if (s == "HALT") return BlockMoverOp::HALT;
    throw std::runtime_error("Unknown BlockMoverOp: " + s);
}

// ============================================================================
// BlockMoverCommand serialization
// ============================================================================

json to_json(const BlockMoverCommand& cmd) {
    json j;
    j["op"] = op_to_string(cmd.op);

    // Only include relevant fields based on operation type
    switch (cmd.op) {
        case BlockMoverOp::PUSH_TO_L2:
        case BlockMoverOp::PULL_FROM_L2:
            j["tile"] = to_json(cmd.tile);
            j["l2_bank_id"] = cmd.l2_bank_id;
            j["l2_offset"] = cmd.l2_offset;
            break;

        case BlockMoverOp::SEND_EAST:
        case BlockMoverOp::SEND_WEST:
        case BlockMoverOp::SEND_NORTH:
        case BlockMoverOp::SEND_SOUTH:
            j["tile"] = to_json(cmd.tile);
            break;

        case BlockMoverOp::SEND_TO:
            j["tile"] = to_json(cmd.tile);
            j["dest_l3_id"] = cmd.dest_l3_id;
            break;

        case BlockMoverOp::RECEIVE_FROM:
            j["tile"] = to_json(cmd.tile);
            j["src_l3_id"] = cmd.src_l3_id;
            break;

        case BlockMoverOp::WAIT_TRIGGER:
            j["trigger_wait_mask"] = cmd.trigger_wait_mask;
            break;

        case BlockMoverOp::EMIT_TRIGGER:
            j["trigger_id"] = cmd.trigger_id;
            j["trigger_dest_mask"] = cmd.trigger_dest_mask;
            break;

        case BlockMoverOp::WAIT_UNTIL_CYCLE:
            j["target_cycle"] = cmd.target_cycle;
            break;

        case BlockMoverOp::LOOP_START:
            j["loop_count"] = cmd.loop_count;
            break;

        case BlockMoverOp::LOOP_END:
        case BlockMoverOp::JUMP:
        case BlockMoverOp::JUMP_IF_TRIGGER:
            j["jump_offset"] = cmd.jump_offset;
            break;

        case BlockMoverOp::TRACE_MARKER:
            j["trace_id"] = cmd.trace_id;
            j["tile"] = to_json(cmd.tile);
            break;

        default:
            break;
    }

    if (cmd.latency_cycles > 0) {
        j["latency_cycles"] = cmd.latency_cycles;
    }

    return j;
}

BlockMoverCommand command_from_json(const json& j) {
    BlockMoverCommand cmd;
    cmd.op = op_from_string(j.value("op", "NOP"));

    if (j.contains("tile")) {
        cmd.tile = tile_from_json(j["tile"]);
    }

    cmd.dest_l3_id = j.value("dest_l3_id", uint8_t{0});
    cmd.src_l3_id = j.value("src_l3_id", uint8_t{0});
    cmd.l2_bank_id = j.value("l2_bank_id", uint8_t{0});
    cmd.l2_offset = j.value("l2_offset", uint32_t{0});
    cmd.trigger_id = j.value("trigger_id", uint8_t{0});
    cmd.trigger_wait_mask = j.value("trigger_wait_mask", uint16_t{0});
    cmd.trigger_dest_mask = j.value("trigger_dest_mask", uint16_t{0});
    cmd.loop_count = j.value("loop_count", uint16_t{0});
    cmd.jump_offset = j.value("jump_offset", int16_t{0});
    cmd.trace_id = j.value("trace_id", uint32_t{0});
    cmd.target_cycle = j.value("target_cycle", uint64_t{0});
    cmd.latency_cycles = j.value("latency_cycles", uint16_t{0});

    return cmd;
}

// ============================================================================
// BlockMoverProgram serialization
// ============================================================================

json to_json(const BlockMoverProgram& prog) {
    json j;
    j["num_commands"] = prog.size();

    json commands_array = json::array();
    for (const auto& cmd : prog.commands) {
        commands_array.push_back(to_json(cmd));
    }
    j["commands"] = commands_array;

    return j;
}

BlockMoverProgram program_from_json(const json& j) {
    BlockMoverProgram prog;

    if (j.contains("commands")) {
        for (const auto& cmd_json : j["commands"]) {
            prog.append(command_from_json(cmd_json));
        }
    }

    return prog;
}

// ============================================================================
// CompiledSchedule::Stats serialization
// ============================================================================

json to_json(const CompiledSchedule::Stats& stats) {
    return json{
        {"total_commands", stats.total_commands},
        {"total_l3_transfers", stats.total_l3_transfers},
        {"total_l2_transfers", stats.total_l2_transfers},
        {"total_dma_ops", stats.total_dma_ops},
        {"total_compute_ops", stats.total_compute_ops},
        {"total_triggers", stats.total_triggers},
        {"total_barriers", stats.total_barriers}
    };
}

CompiledSchedule::Stats stats_from_json(const json& j) {
    CompiledSchedule::Stats stats;
    stats.total_commands = j.value("total_commands", size_t{0});
    stats.total_l3_transfers = j.value("total_l3_transfers", size_t{0});
    stats.total_l2_transfers = j.value("total_l2_transfers", size_t{0});
    stats.total_dma_ops = j.value("total_dma_ops", size_t{0});
    stats.total_compute_ops = j.value("total_compute_ops", size_t{0});
    stats.total_triggers = j.value("total_triggers", size_t{0});
    stats.total_barriers = j.value("total_barriers", size_t{0});
    return stats;
}

// ============================================================================
// CompiledSchedule serialization
// ============================================================================

json to_json(const CompiledSchedule& compiled) {
    json j;
    j["version"] = "1.0";
    j["estimated_cycles"] = compiled.estimated_cycles;
    j["compute_cycles"] = compiled.compute_cycles;
    j["data_movement_cycles"] = compiled.data_movement_cycles;
    j["stats"] = to_json(compiled.stats);

    // Serialize programs (only non-empty ones)
    json programs_array = json::array();
    for (uint8_t l3_id = 0; l3_id < 16; ++l3_id) {
        const auto& prog = compiled.blockmover_programs[l3_id];
        if (!prog.empty()) {
            json prog_json = to_json(prog);
            prog_json["l3_id"] = l3_id;
            programs_array.push_back(prog_json);
        }
    }
    j["programs"] = programs_array;

    return j;
}

CompiledSchedule compiled_from_json(const json& j) {
    CompiledSchedule compiled;

    compiled.estimated_cycles = j.value("estimated_cycles", int64_t{0});
    compiled.compute_cycles = j.value("compute_cycles", int64_t{0});
    compiled.data_movement_cycles = j.value("data_movement_cycles", int64_t{0});

    if (j.contains("stats")) {
        compiled.stats = stats_from_json(j["stats"]);
    }

    if (j.contains("programs")) {
        for (const auto& prog_json : j["programs"]) {
            uint8_t l3_id = prog_json.value("l3_id", uint8_t{0});
            compiled.blockmover_programs[l3_id] = program_from_json(prog_json);
        }
    }

    return compiled;
}

// ============================================================================
// File I/O utilities
// ============================================================================

void write_compiled_json(const CompiledSchedule& compiled, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    file << to_json(compiled).dump(2);
}

CompiledSchedule read_compiled_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    json j = json::parse(file);

    // Validate this is actually a compiled schedule JSON (must have "programs" array)
    if (!j.contains("programs")) {
        throw std::runtime_error("Not a valid compiled schedule JSON: missing 'programs' field");
    }

    return compiled_from_json(j);
}

}  // namespace kpu::dfg::json
