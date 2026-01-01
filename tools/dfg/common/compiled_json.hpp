#pragma once
// CompiledSchedule JSON Serialization
// Converts CompiledSchedule (BlockMover programs) to/from JSON

#include <nlohmann/json.hpp>
#include "sw/kpu/dataflow/block_mover_compiler.hpp"
#include "common/dfg_json.hpp"

namespace kpu::dfg::json {

using json = nlohmann::json;
using namespace sw::kpu::dataflow;
using namespace sw::kpu;

// ============================================================================
// BlockMoverOp conversion
// ============================================================================
std::string op_to_string(BlockMoverOp op);
BlockMoverOp op_from_string(const std::string& s);

// ============================================================================
// BlockMoverCommand serialization
// ============================================================================
json to_json(const BlockMoverCommand& cmd);
BlockMoverCommand command_from_json(const json& j);

// ============================================================================
// BlockMoverProgram serialization
// ============================================================================
json to_json(const BlockMoverProgram& prog);
BlockMoverProgram program_from_json(const json& j);

// ============================================================================
// CompiledSchedule::Stats serialization
// ============================================================================
json to_json(const CompiledSchedule::Stats& stats);
CompiledSchedule::Stats stats_from_json(const json& j);

// ============================================================================
// CompiledSchedule serialization
// ============================================================================
json to_json(const CompiledSchedule& compiled);
CompiledSchedule compiled_from_json(const json& j);

// ============================================================================
// File I/O utilities
// ============================================================================
void write_compiled_json(const CompiledSchedule& compiled, const std::string& filename);
CompiledSchedule read_compiled_json(const std::string& filename);

}  // namespace kpu::dfg::json
