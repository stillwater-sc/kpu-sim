#pragma once
// Schedule JSON Serialization
// Converts DFGSchedule to/from JSON for tool interchange

#include <nlohmann/json.hpp>
#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"
#include "common/dfg_json.hpp"

namespace kpu::dfg::json {

using json = nlohmann::json;
using namespace sw::kpu::dataflow;

// ============================================================================
// ScheduledNode serialization
// ============================================================================
json to_json(const DFGSchedule::ScheduledNode& sn);
DFGSchedule::ScheduledNode scheduled_node_from_json(const json& j);

// ============================================================================
// DFGSchedule serialization
// ============================================================================

/// Convert schedule to JSON
/// If embed_dfg is true, the full DFG is embedded in the output
json to_json(const DFGSchedule& schedule, const TileDataFlowGraph& dfg, bool embed_dfg = true);

/// Convert JSON to schedule
/// If dfg is nullptr, the DFG is expected to be embedded in the JSON
DFGSchedule schedule_from_json(const json& j, const TileDataFlowGraph* dfg = nullptr);

// ============================================================================
// File I/O utilities
// ============================================================================
void write_schedule_json(const DFGSchedule& schedule, const TileDataFlowGraph& dfg,
                         const std::string& filename, bool embed_dfg = true);

/// Read schedule from JSON file
/// Returns both the schedule and the embedded DFG (if present)
struct ScheduleWithDFG {
    DFGSchedule schedule;
    TileDataFlowGraph dfg;
    bool has_embedded_dfg;
};
ScheduleWithDFG read_schedule_json(const std::string& filename);

}  // namespace kpu::dfg::json
