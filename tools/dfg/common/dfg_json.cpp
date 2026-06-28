#include "common/dfg_json.hpp"
#include <fstream>
#include <stdexcept>

namespace kpu::dfg::json {

using namespace sw::kpu::dataflow;
using namespace sw::kpu;

// ============================================================================
// TensorId conversion
// ============================================================================

std::string tensor_id_to_string(TensorId id) {
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

TensorId tensor_id_from_string(const std::string& s) {
    if (s == "A") return TensorId::A;
    if (s == "B") return TensorId::B;
    if (s == "C") return TensorId::C;
    if (s == "BIAS") return TensorId::BIAS;
    if (s == "WORKSPACE") return TensorId::WORKSPACE;
    if (s == "CUSTOM") return TensorId::CUSTOM;
    throw std::runtime_error("Unknown tensor ID: " + s);
}

// ============================================================================
// Node type conversion
// ============================================================================

std::string node_type_to_string(DFNodeType type) {
    return sw::kpu::dataflow::to_string(type);
}

DFNodeType node_type_from_string(const std::string& s) {
    if (s == "DMA_LOAD") return DFNodeType::DMA_LOAD;
    if (s == "DMA_STORE") return DFNodeType::DMA_STORE;
    if (s == "L3_TRANSFER") return DFNodeType::L3_TRANSFER;
    if (s == "L3_TO_L2") return DFNodeType::L3_TO_L2;
    if (s == "L2_TO_L3") return DFNodeType::L2_TO_L3;
    if (s == "MATMUL") return DFNodeType::MATMUL;
    if (s == "ELEMENTWISE") return DFNodeType::ELEMENTWISE;
    if (s == "ACTIVATION") return DFNodeType::ACTIVATION;
    if (s == "REDUCTION") return DFNodeType::REDUCTION;
    if (s == "BARRIER") return DFNodeType::BARRIER;
    if (s == "TRIGGER") return DFNodeType::TRIGGER;
    throw std::runtime_error("Unknown node type: " + s);
}

// ============================================================================
// Edge type conversion
// ============================================================================

std::string edge_type_to_string(DFEdgeType type) {
    switch (type) {
        case DFEdgeType::DATA: return "DATA";
        case DFEdgeType::CONTROL: return "CONTROL";
        case DFEdgeType::ANTI: return "ANTI";
        case DFEdgeType::OUTPUT: return "OUTPUT";
        default: return "UNKNOWN";
    }
}

DFEdgeType edge_type_from_string(const std::string& s) {
    if (s == "DATA") return DFEdgeType::DATA;
    if (s == "CONTROL") return DFEdgeType::CONTROL;
    if (s == "ANTI") return DFEdgeType::ANTI;
    if (s == "OUTPUT") return DFEdgeType::OUTPUT;
    throw std::runtime_error("Unknown edge type: " + s);
}

// ============================================================================
// TileDescriptor serialization
// ============================================================================

json to_json(const TileDescriptor& tile) {
    return json{
        {"tensor", tensor_id_to_string(tile.tensor)},
        {"m_tile", tile.m_tile},
        {"n_tile", tile.n_tile},
        {"k_tile", tile.k_tile},
        {"l3_offset", tile.l3_offset},
        {"size", tile.size},
        {"height", tile.height},
        {"width", tile.width},
        {"src_stride", tile.src_stride},
        {"dst_stride", tile.dst_stride}
    };
}

TileDescriptor tile_from_json(const json& j) {
    TileDescriptor tile;
    tile.tensor = tensor_id_from_string(j.value("tensor", "A"));
    tile.m_tile = static_cast<uint16_t>(j.value("m_tile", 0));
    tile.n_tile = static_cast<uint16_t>(j.value("n_tile", 0));
    tile.k_tile = static_cast<uint16_t>(j.value("k_tile", 0));
    tile.l3_offset = j.value("l3_offset", 0);
    tile.size = j.value("size", 0);
    tile.height = static_cast<uint16_t>(j.value("height", 0));
    tile.width = static_cast<uint16_t>(j.value("width", 0));
    tile.src_stride = j.value("src_stride", 0);
    tile.dst_stride = j.value("dst_stride", 0);
    return tile;
}

// ============================================================================
// DFNode serialization
// ============================================================================

json to_json(const DFNode& node) {
    json j = {
        {"id", node.id},
        {"type", node_type_to_string(node.type)},
        {"name", node.name},
        {"tile", to_json(node.tile)},
        {"l3_id", node.l3_id},
        {"l2_bank_id", node.l2_bank_id},
        {"duration", node.duration},
        {"predecessors", node.predecessors},
        {"successors", node.successors},
        {"priority", node.priority}
    };

    // Add transfer-specific fields
    if (node.type == DFNodeType::L3_TRANSFER) {
        j["src_l3_id"] = node.src_l3_id;
        j["dst_l3_id"] = node.dst_l3_id;
    }

    // Add compute-specific fields
    if (node.type == DFNodeType::MATMUL) {
        j["M"] = node.M;
        j["N"] = node.N;
        j["K"] = node.K;
    }

    // Add scheduling info if available
    if (node.earliest_start >= 0) {
        j["earliest_start"] = node.earliest_start;
    }
    if (node.scheduled_start >= 0) {
        j["scheduled_start"] = node.scheduled_start;
    }

    return j;
}

DFNode node_from_json(const json& j) {
    DFNode node;
    node.id = j.value("id", size_t{0});
    node.type = node_type_from_string(j.value("type", "BARRIER"));
    node.name = j.value("name", "");
    node.tile = tile_from_json(j.value("tile", json::object()));
    node.l3_id = static_cast<uint8_t>(j.value("l3_id", 0));
    node.l2_bank_id = static_cast<uint8_t>(j.value("l2_bank_id", 0));
    node.duration = j.value("duration", 0);
    node.priority = j.value("priority", 0);

    // Dependencies
    if (j.contains("predecessors")) {
        node.predecessors = j["predecessors"].get<std::vector<size_t>>();
    }
    if (j.contains("successors")) {
        node.successors = j["successors"].get<std::vector<size_t>>();
    }

    // Transfer-specific
    node.src_l3_id = static_cast<uint8_t>(j.value("src_l3_id", 0));
    node.dst_l3_id = static_cast<uint8_t>(j.value("dst_l3_id", 0));

    // Compute-specific
    node.M = j.value("M", 0);
    node.N = j.value("N", 0);
    node.K = j.value("K", 0);

    // Scheduling info
    node.earliest_start = j.value("earliest_start", -1);
    node.scheduled_start = j.value("scheduled_start", -1);

    return node;
}

// ============================================================================
// DFEdge serialization
// ============================================================================

json to_json(const DFEdge& edge) {
    return json{
        {"from", edge.from_node},
        {"to", edge.to_node},
        {"type", edge_type_to_string(edge.type)},
        {"tile", to_json(edge.tile)},
        {"latency", edge.latency}
    };
}

DFEdge edge_from_json(const json& j) {
    DFEdge edge;
    edge.from_node = j.value("from", size_t{0});
    edge.to_node = j.value("to", size_t{0});
    edge.type = edge_type_from_string(j.value("type", "DATA"));
    edge.tile = tile_from_json(j.value("tile", json::object()));
    edge.latency = j.value("latency", 0);
    return edge;
}

// ============================================================================
// TimingModel serialization
// ============================================================================

json to_json(const TimingModel& timing) {
    return json{
        {"dma_bandwidth_bytes_per_cycle", timing.dma_bandwidth_bytes_per_cycle},
        {"l3_bandwidth_bytes_per_cycle", timing.l3_bandwidth_bytes_per_cycle},
        {"router_latency_cycles", timing.router_latency_cycles},
        {"link_latency_cycles", timing.link_latency_cycles},
        {"store_and_forward", timing.store_and_forward},
        {"mesh_rows", timing.mesh_rows},
        {"mesh_cols", timing.mesh_cols}
    };
}

TimingModel timing_from_json(const json& j) {
    TimingModel timing;
    timing.dma_bandwidth_bytes_per_cycle = j.value("dma_bandwidth_bytes_per_cycle", 32UL);
    timing.l3_bandwidth_bytes_per_cycle = j.value("l3_bandwidth_bytes_per_cycle", 64UL);
    timing.router_latency_cycles = j.value("router_latency_cycles", 1UL);
    timing.link_latency_cycles = j.value("link_latency_cycles", 1UL);
    timing.store_and_forward = j.value("store_and_forward", true);
    timing.mesh_rows = static_cast<uint8_t>(j.value("mesh_rows", 4));
    timing.mesh_cols = static_cast<uint8_t>(j.value("mesh_cols", 4));
    return timing;
}

// ============================================================================
// TileDataFlowGraph serialization
// ============================================================================

json to_json(const TileDataFlowGraph& dfg) {
    json j;
    j["version"] = "1.0";
    j["timing"] = to_json(dfg.timing_model());

    // Serialize nodes
    json nodes_array = json::array();
    for (const auto& node : dfg.nodes()) {
        nodes_array.push_back(to_json(node));
    }
    j["nodes"] = nodes_array;

    // Serialize edges
    json edges_array = json::array();
    for (const auto& edge : dfg.edges()) {
        edges_array.push_back(to_json(edge));
    }
    j["edges"] = edges_array;

    // Add summary stats
    j["stats"] = {
        {"num_nodes", dfg.num_nodes()},
        {"num_edges", dfg.num_edges()},
        {"is_acyclic", dfg.is_acyclic()},
        {"critical_path_length", dfg.critical_path_length()}
    };

    return j;
}

TileDataFlowGraph dfg_from_json(const json& j) {
    TileDataFlowGraph dfg;

    // Set timing model if present
    if (j.contains("timing")) {
        dfg.set_timing_model(timing_from_json(j["timing"]));
    }

    // Deserialize nodes first (add_node assigns sequential IDs)
    if (j.contains("nodes")) {
        for (const auto& node_json : j["nodes"]) {
            DFNode node = node_from_json(node_json);
            // Clear predecessors/successors since we'll reconstruct from edges
            node.predecessors.clear();
            node.successors.clear();
            dfg.add_node(node);
        }
    }

    // Deserialize edges
    if (j.contains("edges")) {
        for (const auto& edge_json : j["edges"]) {
            DFEdge edge = edge_from_json(edge_json);
            dfg.add_edge(edge.from_node, edge.to_node, edge.type);
        }
    }

    return dfg;
}

// ============================================================================
// File I/O utilities
// ============================================================================

void write_dfg_json(const TileDataFlowGraph& dfg, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    file << to_json(dfg).dump(2);
}

TileDataFlowGraph read_dfg_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    json j = json::parse(file);
    return dfg_from_json(j);
}

}  // namespace kpu::dfg::json
