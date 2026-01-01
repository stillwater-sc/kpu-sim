#include "common/schedule_json.hpp"
#include <fstream>
#include <stdexcept>

namespace kpu::dfg::json {

using namespace sw::kpu::dataflow;

// ============================================================================
// ScheduledNode serialization
// ============================================================================

json to_json(const DFGSchedule::ScheduledNode& sn) {
    return json{
        {"node_id", sn.node_id},
        {"start_cycle", sn.start_cycle},
        {"end_cycle", sn.end_cycle},
        {"resource_id", sn.resource_id}
    };
}

DFGSchedule::ScheduledNode scheduled_node_from_json(const json& j) {
    DFGSchedule::ScheduledNode sn;
    sn.node_id = j.value("node_id", size_t{0});
    sn.start_cycle = j.value("start_cycle", int64_t{0});
    sn.end_cycle = j.value("end_cycle", int64_t{0});
    sn.resource_id = j.value("resource_id", uint8_t{0});
    return sn;
}

// ============================================================================
// DFGSchedule serialization
// ============================================================================

json to_json(const DFGSchedule& schedule, const TileDataFlowGraph& dfg, bool embed_dfg) {
    json j;
    j["version"] = "1.0";
    j["makespan"] = schedule.makespan;

    // Serialize scheduled nodes
    json nodes_array = json::array();
    for (const auto& sn : schedule.nodes) {
        nodes_array.push_back(to_json(sn));
    }
    j["schedule"] = nodes_array;

    // Serialize per-L3 schedules
    json l3_schedules_obj = json::object();
    for (const auto& [l3_id, node_ids] : schedule.l3_schedules) {
        l3_schedules_obj[std::to_string(l3_id)] = node_ids;
    }
    j["per_l3_schedule"] = l3_schedules_obj;

    // Optionally embed the DFG
    if (embed_dfg) {
        j["dfg"] = to_json(dfg);
    }

    // Add algorithm info placeholder
    j["algorithm"] = "UNKNOWN";

    return j;
}

DFGSchedule schedule_from_json(const json& j, const TileDataFlowGraph* dfg) {
    DFGSchedule schedule;

    schedule.makespan = j.value("makespan", int64_t{0});

    // Deserialize scheduled nodes
    if (j.contains("schedule")) {
        for (const auto& sn_json : j["schedule"]) {
            schedule.nodes.push_back(scheduled_node_from_json(sn_json));
        }
    }

    // Deserialize per-L3 schedules
    if (j.contains("per_l3_schedule")) {
        for (auto& [key, value] : j["per_l3_schedule"].items()) {
            uint8_t l3_id = static_cast<uint8_t>(std::stoi(key));
            schedule.l3_schedules[l3_id] = value.get<std::vector<size_t>>();
        }
    }

    return schedule;
}

// ============================================================================
// File I/O utilities
// ============================================================================

void write_schedule_json(const DFGSchedule& schedule, const TileDataFlowGraph& dfg,
                         const std::string& filename, bool embed_dfg) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    file << to_json(schedule, dfg, embed_dfg).dump(2);
}

ScheduleWithDFG read_schedule_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    json j = json::parse(file);

    // Validate this is actually a schedule JSON (has "schedule" or "makespan" key)
    if (!j.contains("schedule") && !j.contains("makespan")) {
        throw std::runtime_error("Not a valid schedule JSON: missing 'schedule' field");
    }

    ScheduleWithDFG result;

    // Check for embedded DFG
    if (j.contains("dfg")) {
        result.dfg = dfg_from_json(j["dfg"]);
        result.has_embedded_dfg = true;
        result.schedule = schedule_from_json(j, &result.dfg);
    } else {
        result.has_embedded_dfg = false;
        result.schedule = schedule_from_json(j, nullptr);
    }

    return result;
}

}  // namespace kpu::dfg::json
