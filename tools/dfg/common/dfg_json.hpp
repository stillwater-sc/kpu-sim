#pragma once
// DFG JSON Serialization
// Converts TileDataFlowGraph to/from JSON for tool interchange

#include <nlohmann/json.hpp>
#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"

namespace kpu::dfg::json {

using json = nlohmann::json;

// Import types from dataflow namespace
using sw::kpu::dataflow::TileDataFlowGraph;
using sw::kpu::dataflow::DFNode;
using sw::kpu::dataflow::DFEdge;
using sw::kpu::dataflow::DFNodeType;
using sw::kpu::dataflow::DFEdgeType;
using sw::kpu::dataflow::TimingModel;
using sw::kpu::TileDescriptor;
using sw::kpu::TensorId;

// ============================================================================
// TileDescriptor serialization
// ============================================================================
json to_json(const TileDescriptor& tile);
TileDescriptor tile_from_json(const json& j);

// ============================================================================
// DFNode serialization
// ============================================================================
json to_json(const DFNode& node);
DFNode node_from_json(const json& j);

// ============================================================================
// DFEdge serialization
// ============================================================================
json to_json(const DFEdge& edge);
DFEdge edge_from_json(const json& j);

// ============================================================================
// TimingModel serialization
// ============================================================================
json to_json(const TimingModel& timing);
TimingModel timing_from_json(const json& j);

// ============================================================================
// TileDataFlowGraph serialization
// ============================================================================
json to_json(const TileDataFlowGraph& dfg);
TileDataFlowGraph dfg_from_json(const json& j);

// ============================================================================
// Helper: Convert node type to/from string
// ============================================================================
std::string node_type_to_string(DFNodeType type);
DFNodeType node_type_from_string(const std::string& s);

// ============================================================================
// Helper: Convert edge type to/from string
// ============================================================================
std::string edge_type_to_string(DFEdgeType type);
DFEdgeType edge_type_from_string(const std::string& s);

// ============================================================================
// Helper: Convert tensor ID to/from string
// ============================================================================
std::string tensor_id_to_string(TensorId id);
TensorId tensor_id_from_string(const std::string& s);

// ============================================================================
// File I/O utilities
// ============================================================================
void write_dfg_json(const TileDataFlowGraph& dfg, const std::string& filename);
TileDataFlowGraph read_dfg_json(const std::string& filename);

}  // namespace kpu::dfg::json
