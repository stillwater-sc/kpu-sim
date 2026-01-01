// ============================================================================
// src/dataflow/tile_dataflow_graph.cpp
// Tile Data Flow Graph Implementation
// ============================================================================

#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"

#include <algorithm>
#include <iostream>
#include <queue>
#include <sstream>
#include <stack>

namespace sw::kpu::dataflow {

// ============================================================================
// TileDataFlowGraph Implementation
// ============================================================================

size_t TileDataFlowGraph::add_node(const DFNode& node) {
    size_t id = nodes_.size();
    nodes_.push_back(node);
    nodes_.back().id = id;

    // Update index
    nodes_by_l3_[node.l3_id].push_back(id);

    return id;
}

size_t TileDataFlowGraph::add_dma_load(const TileDescriptor& tile, uint8_t dest_l3) {
    DFNode node;
    node.type = DFNodeType::DMA_LOAD;
    node.tile = tile;
    node.l3_id = dest_l3;
    node.dst_l3_id = dest_l3;
    node.name = "DMA_LOAD_" + tile.to_string();

    // Use timing model for accurate duration
    node.duration = static_cast<int64_t>(timing_.dma_load_duration(tile.size));

    return add_node(node);
}

size_t TileDataFlowGraph::add_dma_store(const TileDescriptor& tile, uint8_t src_l3) {
    DFNode node;
    node.type = DFNodeType::DMA_STORE;
    node.tile = tile;
    node.l3_id = src_l3;
    node.src_l3_id = src_l3;
    node.name = "DMA_STORE_" + tile.to_string();

    // Use timing model for accurate duration
    node.duration = static_cast<int64_t>(timing_.dma_load_duration(tile.size));

    return add_node(node);
}

size_t TileDataFlowGraph::add_l3_transfer(const TileDescriptor& tile, uint8_t src_l3, uint8_t dst_l3) {
    DFNode node;
    node.type = DFNodeType::L3_TRANSFER;
    node.tile = tile;
    node.l3_id = src_l3;  // Initiated by source
    node.src_l3_id = src_l3;
    node.dst_l3_id = dst_l3;
    node.name = "L3_XFER_" + tile.to_string();

    // Use timing model for accurate duration including propagation delay
    node.duration = static_cast<int64_t>(
        timing_.l3_transfer_duration(tile.size, src_l3, dst_l3));

    return add_node(node);
}

size_t TileDataFlowGraph::add_l3_to_l2(const TileDescriptor& tile, uint8_t l3_id, uint8_t l2_bank) {
    DFNode node;
    node.type = DFNodeType::L3_TO_L2;
    node.tile = tile;
    node.l3_id = l3_id;
    node.l2_bank_id = l2_bank;
    node.name = "L3_TO_L2_" + tile.to_string();

    // Use timing model - L3→L2 uses same bandwidth as L3 link
    node.duration = static_cast<int64_t>(timing_.injection_duration(tile.size));

    return add_node(node);
}

size_t TileDataFlowGraph::add_l2_to_l3(const TileDescriptor& tile, uint8_t l3_id, uint8_t l2_bank) {
    DFNode node;
    node.type = DFNodeType::L2_TO_L3;
    node.tile = tile;
    node.l3_id = l3_id;
    node.l2_bank_id = l2_bank;
    node.name = "L2_TO_L3_" + tile.to_string();

    // Use timing model - L2→L3 uses same bandwidth as L3 link
    node.duration = static_cast<int64_t>(timing_.injection_duration(tile.size));

    return add_node(node);
}

size_t TileDataFlowGraph::add_matmul(uint8_t l3_id, uint32_t M, uint32_t N, uint32_t K,
                                      const TileDescriptor& a, const TileDescriptor& b,
                                      const TileDescriptor& c) {
    DFNode node;
    node.type = DFNodeType::MATMUL;
    node.tile = c;  // Output tile
    node.l3_id = l3_id;
    node.M = M;
    node.N = N;
    node.K = K;
    node.name = "MATMUL_" + c.to_string();

    // Compute cycles for matmul
    // For a 24×24 systolic array: throughput = 24×24 = 576 MACs/cycle
    // Total MACs = M × N × K × 2 (multiply + add)
    uint64_t macs = static_cast<uint64_t>(M) * N * K * 2;
    node.duration = static_cast<int64_t>((macs + 575) / 576);

    // Store A, B tile info (simplified - just using C tile)
    (void)a; (void)b;

    return add_node(node);
}

size_t TileDataFlowGraph::add_barrier(uint8_t l3_id) {
    DFNode node;
    node.type = DFNodeType::BARRIER;
    node.l3_id = l3_id;
    node.name = "BARRIER";
    node.duration = 1;

    return add_node(node);
}

void TileDataFlowGraph::add_edge(size_t from, size_t to, DFEdgeType type) {
    DFEdge edge;
    edge.from_node = from;
    edge.to_node = to;
    edge.type = type;
    edges_.push_back(edge);

    // Update adjacency lists
    nodes_[from].successors.push_back(to);
    nodes_[to].predecessors.push_back(from);
}

void TileDataFlowGraph::add_data_edge(size_t from, size_t to, const TileDescriptor& tile) {
    DFEdge edge;
    edge.from_node = from;
    edge.to_node = to;
    edge.type = DFEdgeType::DATA;
    edge.tile = tile;
    edges_.push_back(edge);

    nodes_[from].successors.push_back(to);
    nodes_[to].predecessors.push_back(from);
}

std::vector<size_t> TileDataFlowGraph::get_sources() const {
    std::vector<size_t> sources;
    for (const auto& node : nodes_) {
        if (node.predecessors.empty()) {
            sources.push_back(node.id);
        }
    }
    return sources;
}

std::vector<size_t> TileDataFlowGraph::get_sinks() const {
    std::vector<size_t> sinks;
    for (const auto& node : nodes_) {
        if (node.successors.empty()) {
            sinks.push_back(node.id);
        }
    }
    return sinks;
}

std::vector<size_t> TileDataFlowGraph::get_nodes_by_type(DFNodeType type) const {
    std::vector<size_t> result;
    for (const auto& node : nodes_) {
        if (node.type == type) {
            result.push_back(node.id);
        }
    }
    return result;
}

std::vector<size_t> TileDataFlowGraph::get_nodes_by_l3(uint8_t l3_id) const {
    auto it = nodes_by_l3_.find(l3_id);
    if (it != nodes_by_l3_.end()) {
        return it->second;
    }
    return {};
}

bool TileDataFlowGraph::has_cycle_from(size_t node, std::vector<int>& state) const {
    // state: 0 = unvisited, 1 = visiting, 2 = visited
    state[node] = 1;  // visiting

    for (size_t succ : nodes_[node].successors) {
        if (state[succ] == 1) {
            return true;  // Back edge found
        }
        if (state[succ] == 0 && has_cycle_from(succ, state)) {
            return true;
        }
    }

    state[node] = 2;  // visited
    return false;
}

bool TileDataFlowGraph::is_acyclic() const {
    std::vector<int> state(nodes_.size(), 0);

    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (state[i] == 0) {
            if (has_cycle_from(i, state)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<size_t> TileDataFlowGraph::topological_order() const {
    std::vector<size_t> order;
    std::vector<size_t> in_degree(nodes_.size(), 0);

    // Compute in-degrees
    for (const auto& node : nodes_) {
        for (size_t succ : node.successors) {
            in_degree[succ]++;
        }
    }

    // Initialize queue with sources
    std::queue<size_t> ready;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (in_degree[i] == 0) {
            ready.push(i);
        }
    }

    // Process
    while (!ready.empty()) {
        size_t node = ready.front();
        ready.pop();
        order.push_back(node);

        for (size_t succ : nodes_[node].successors) {
            in_degree[succ]--;
            if (in_degree[succ] == 0) {
                ready.push(succ);
            }
        }
    }

    return order;
}

void TileDataFlowGraph::compute_earliest_starts() {
    auto order = topological_order();

    for (size_t id : order) {
        DFNode& node = nodes_[id];

        int64_t earliest = 0;
        for (size_t pred : node.predecessors) {
            int64_t pred_end = nodes_[pred].earliest_start + nodes_[pred].duration;
            earliest = std::max(earliest, pred_end);
        }
        node.earliest_start = earliest;
    }
}

int64_t TileDataFlowGraph::critical_path_length() const {
    if (nodes_.empty()) return 0;

    // Need mutable copy to compute earliest starts
    TileDataFlowGraph copy = *this;
    copy.compute_earliest_starts();

    int64_t max_end = 0;
    for (const auto& node : copy.nodes_) {
        int64_t end = node.earliest_start + node.duration;
        max_end = std::max(max_end, end);
    }
    return max_end;
}

void TileDataFlowGraph::assign_critical_path_priorities() {
    // Compute ALAP (As Late As Possible) times
    // Priority = max_time - alap_time (higher priority for critical path nodes)

    auto order = topological_order();
    std::reverse(order.begin(), order.end());

    int64_t max_time = critical_path_length();

    // Compute ALAP in reverse topological order
    std::vector<int64_t> alap(nodes_.size(), max_time);

    for (size_t id : order) {
        DFNode& node = nodes_[id];

        int64_t latest = max_time;
        for (size_t succ : node.successors) {
            latest = std::min(latest, alap[succ] - nodes_[succ].duration);
        }
        alap[id] = latest - node.duration;
        node.priority = static_cast<uint32_t>(max_time - alap[id]);
    }
}

// ============================================================================
// Matmul DFG Builders
// ============================================================================

TileDataFlowGraph TileDataFlowGraph::build_matmul(
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles,
    uint8_t num_l3_tiles)
{
    // Default to output-stationary for now
    (void)num_l3_tiles;
    return build_matmul_output_stationary(M, N, K, m_tiles, n_tiles, k_tiles);
}

TileDataFlowGraph TileDataFlowGraph::build_matmul_output_stationary(
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles)
{
    TileDataFlowGraph dfg;

    uint32_t tile_m = M / m_tiles;
    uint32_t tile_n = N / n_tiles;
    uint32_t tile_k = K / k_tiles;

    // Calculate tile sizes
    uint32_t a_tile_bytes = tile_m * tile_k * sizeof(float);
    uint32_t b_tile_bytes = tile_k * tile_n * sizeof(float);
    uint32_t c_tile_bytes = tile_m * tile_n * sizeof(float);

    // Map C tiles to L3 tiles (round-robin)
    uint8_t num_l3 = 16;

    // Track node IDs for dependencies
    // load_a[m][k], load_b[k][n], compute[m][n][k], drain[m][n]
    std::vector<std::vector<size_t>> load_a(m_tiles, std::vector<size_t>(k_tiles));
    std::vector<std::vector<size_t>> load_b(k_tiles, std::vector<size_t>(n_tiles));
    std::vector<std::vector<std::vector<size_t>>> compute(
        m_tiles, std::vector<std::vector<size_t>>(n_tiles, std::vector<size_t>(k_tiles)));
    std::vector<std::vector<size_t>> drain(m_tiles, std::vector<size_t>(n_tiles));

    // Create DMA load nodes for A tiles
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t k = 0; k < k_tiles; ++k) {
            TileDescriptor tile;
            tile.tensor = TensorId::A;
            tile.m_tile = static_cast<uint16_t>(m);
            tile.k_tile = static_cast<uint16_t>(k);
            tile.size = a_tile_bytes;
            tile.height = static_cast<uint16_t>(tile_m);
            tile.width = static_cast<uint16_t>(tile_k);

            // A tiles loaded to L3s in column 0 of the mesh
            uint8_t l3_id = static_cast<uint8_t>(m % num_l3);
            load_a[m][k] = dfg.add_dma_load(tile, l3_id);
        }
    }

    // Create DMA load nodes for B tiles
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            TileDescriptor tile;
            tile.tensor = TensorId::B;
            tile.n_tile = static_cast<uint16_t>(n);
            tile.k_tile = static_cast<uint16_t>(k);
            tile.size = b_tile_bytes;
            tile.height = static_cast<uint16_t>(tile_k);
            tile.width = static_cast<uint16_t>(tile_n);

            // B tiles loaded to L3s in row 0 of the mesh
            uint8_t l3_id = static_cast<uint8_t>(n % num_l3);
            load_b[k][n] = dfg.add_dma_load(tile, l3_id);
        }
    }

    // Create compute nodes for each C tile × K step
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            uint8_t l3_id = static_cast<uint8_t>((m * n_tiles + n) % num_l3);

            for (uint32_t k = 0; k < k_tiles; ++k) {
                TileDescriptor c_tile;
                c_tile.tensor = TensorId::C;
                c_tile.m_tile = static_cast<uint16_t>(m);
                c_tile.n_tile = static_cast<uint16_t>(n);
                c_tile.k_tile = static_cast<uint16_t>(k);
                c_tile.size = c_tile_bytes;
                c_tile.height = static_cast<uint16_t>(tile_m);
                c_tile.width = static_cast<uint16_t>(tile_n);

                TileDescriptor a_tile;
                a_tile.tensor = TensorId::A;
                a_tile.m_tile = static_cast<uint16_t>(m);
                a_tile.k_tile = static_cast<uint16_t>(k);

                TileDescriptor b_tile;
                b_tile.tensor = TensorId::B;
                b_tile.n_tile = static_cast<uint16_t>(n);
                b_tile.k_tile = static_cast<uint16_t>(k);

                compute[m][n][k] = dfg.add_matmul(l3_id, tile_m, tile_n, tile_k,
                                                   a_tile, b_tile, c_tile);

                // Dependencies: need A[m,k] and B[k,n]
                dfg.add_edge(load_a[m][k], compute[m][n][k]);
                dfg.add_edge(load_b[k][n], compute[m][n][k]);

                // Chain K steps for accumulation
                if (k > 0) {
                    dfg.add_edge(compute[m][n][k-1], compute[m][n][k]);
                }
            }

            // Create drain node for final C tile
            TileDescriptor c_final;
            c_final.tensor = TensorId::C;
            c_final.m_tile = static_cast<uint16_t>(m);
            c_final.n_tile = static_cast<uint16_t>(n);
            c_final.size = c_tile_bytes;

            drain[m][n] = dfg.add_dma_store(c_final, l3_id);
            dfg.add_edge(compute[m][n][k_tiles-1], drain[m][n]);
        }
    }

    return dfg;
}

TileDataFlowGraph TileDataFlowGraph::build_matmul_a_stationary(
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles)
{
    // A-stationary: A tiles stay in place, B tiles flow through
    // This is more complex, start with output-stationary for now
    return build_matmul_output_stationary(M, N, K, m_tiles, n_tiles, k_tiles);
}

TileDataFlowGraph TileDataFlowGraph::build_matmul_b_stationary(
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles)
{
    // B-stationary: B tiles stay in place, A tiles flow through
    return build_matmul_output_stationary(M, N, K, m_tiles, n_tiles, k_tiles);
}

// ============================================================================
// Debug Output
// ============================================================================

std::string TileDataFlowGraph::to_string() const {
    std::ostringstream oss;
    oss << "TileDataFlowGraph: " << nodes_.size() << " nodes, "
        << edges_.size() << " edges\n";

    for (const auto& node : nodes_) {
        oss << "  " << node.to_string() << "\n";
        if (!node.predecessors.empty()) {
            oss << "    deps: [";
            for (size_t i = 0; i < node.predecessors.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << node.predecessors[i];
            }
            oss << "]\n";
        }
    }

    return oss.str();
}

std::string TileDataFlowGraph::to_dot() const {
    std::ostringstream oss;
    oss << "digraph TileDataFlow {\n";
    oss << "  rankdir=TB;\n";
    oss << "  node [shape=box];\n\n";

    // Nodes
    for (const auto& node : nodes_) {
        oss << "  n" << node.id << " [label=\"" << node.id << ": "
            << sw::kpu::dataflow::to_string(node.type);
        if (!node.tile.to_string().empty()) {
            oss << "\\n" << node.tile.to_string();
        }
        oss << "\\nL3[" << static_cast<int>(node.l3_id) << "]\"";

        // Color by type
        switch (node.type) {
            case DFNodeType::DMA_LOAD:
            case DFNodeType::DMA_STORE:
                oss << " style=filled fillcolor=lightblue";
                break;
            case DFNodeType::L3_TRANSFER:
                oss << " style=filled fillcolor=lightyellow";
                break;
            case DFNodeType::MATMUL:
                oss << " style=filled fillcolor=lightgreen";
                break;
            default:
                break;
        }
        oss << "];\n";
    }

    oss << "\n";

    // Edges
    for (const auto& edge : edges_) {
        oss << "  n" << edge.from_node << " -> n" << edge.to_node;
        if (edge.type == DFEdgeType::CONTROL) {
            oss << " [style=dashed]";
        }
        oss << ";\n";
    }

    oss << "}\n";
    return oss.str();
}

void TileDataFlowGraph::print_summary() const {
    std::cout << "\n=== Tile Data Flow Graph Summary ===\n";
    std::cout << "Nodes: " << nodes_.size() << "\n";
    std::cout << "Edges: " << edges_.size() << "\n";

    // Count by type
    std::map<DFNodeType, size_t> type_counts;
    for (const auto& node : nodes_) {
        type_counts[node.type]++;
    }

    std::cout << "\nNode types:\n";
    for (const auto& [type, count] : type_counts) {
        std::cout << "  " << sw::kpu::dataflow::to_string(type) << ": " << count << "\n";
    }

    // Count by L3
    std::cout << "\nNodes per L3:\n";
    for (const auto& [l3_id, node_ids] : nodes_by_l3_) {
        std::cout << "  L3[" << static_cast<int>(l3_id) << "]: " << node_ids.size() << "\n";
    }

    std::cout << "\nCritical path length: " << critical_path_length() << " cycles\n";
    std::cout << "=====================================\n";
}

// ============================================================================
// DFGSchedule Implementation
// ============================================================================

std::vector<DFGSchedule::ScheduledNode> DFGSchedule::get_l3_schedule(uint8_t l3_id) const {
    std::vector<ScheduledNode> result;
    for (const auto& sn : nodes) {
        if (sn.resource_id == l3_id) {
            result.push_back(sn);
        }
    }
    // Sort by start time
    std::sort(result.begin(), result.end(),
              [](const ScheduledNode& a, const ScheduledNode& b) {
                  return a.start_cycle < b.start_cycle;
              });
    return result;
}

bool DFGSchedule::validate(const TileDataFlowGraph& dfg) const {
    // Check that all dependencies are satisfied
    for (const auto& edge : dfg.edges()) {
        int64_t from_end = -1, to_start = -1;

        for (const auto& sn : nodes) {
            if (sn.node_id == edge.from_node) {
                from_end = sn.end_cycle;
            }
            if (sn.node_id == edge.to_node) {
                to_start = sn.start_cycle;
            }
        }

        if (from_end < 0 || to_start < 0) {
            return false;  // Node not scheduled
        }
        if (from_end > to_start) {
            return false;  // Dependency violated
        }
    }
    return true;
}

// ============================================================================
// DFGScheduler Implementation
// ============================================================================

DFGScheduler::DFGScheduler()
    : DFGScheduler(Config{})
{
}

DFGScheduler::DFGScheduler(const Config& config)
    : config_(config)
{
}

DFGSchedule DFGScheduler::schedule(const TileDataFlowGraph& dfg) {
    switch (config_.algorithm) {
        case Algorithm::ASAP:
        case Algorithm::ALAP:  // Fall through to ASAP for now
            return schedule_asap(dfg);

        case Algorithm::LIST_SCHEDULING:
        case Algorithm::CRITICAL_PATH:
            return schedule_list(dfg);

        default:
            return schedule_asap(dfg);
    }
}

DFGSchedule DFGScheduler::schedule_asap(const TileDataFlowGraph& dfg) {
    DFGSchedule schedule;

    // Make mutable copy to compute earliest starts
    TileDataFlowGraph mutable_dfg = dfg;
    mutable_dfg.compute_earliest_starts();

    for (const auto& node : mutable_dfg.nodes()) {
        DFGSchedule::ScheduledNode sn;
        sn.node_id = node.id;
        sn.start_cycle = node.earliest_start;
        sn.end_cycle = node.earliest_start + node.duration;
        sn.resource_id = node.l3_id;

        schedule.nodes.push_back(sn);
        schedule.makespan = std::max(schedule.makespan, sn.end_cycle);
    }

    return schedule;
}

DFGSchedule DFGScheduler::schedule_list(const TileDataFlowGraph& dfg) {
    // Priority-based list scheduling
    DFGSchedule schedule;

    // Make mutable copy
    TileDataFlowGraph mutable_dfg = dfg;
    mutable_dfg.compute_earliest_starts();
    mutable_dfg.assign_critical_path_priorities();

    // Track when each L3 tile becomes free
    std::vector<int64_t> l3_free_time(config_.num_l3_tiles, 0);

    // Track when each node completes
    std::vector<int64_t> node_complete(dfg.num_nodes(), -1);

    // Get nodes in priority order
    std::vector<size_t> order = mutable_dfg.topological_order();
    std::sort(order.begin(), order.end(),
              [&mutable_dfg](size_t a, size_t b) {
                  return mutable_dfg.node(a).priority > mutable_dfg.node(b).priority;
              });

    // Schedule each node
    for (size_t id : order) {
        const DFNode& node = mutable_dfg.node(id);

        // Find earliest start based on dependencies
        int64_t earliest = 0;
        for (size_t pred : node.predecessors) {
            earliest = std::max(earliest, node_complete[pred]);
        }

        // Also consider resource availability
        earliest = std::max(earliest, l3_free_time[node.l3_id]);

        // Schedule the node
        DFGSchedule::ScheduledNode sn;
        sn.node_id = id;
        sn.start_cycle = earliest;
        sn.end_cycle = earliest + node.duration;
        sn.resource_id = node.l3_id;

        schedule.nodes.push_back(sn);
        node_complete[id] = sn.end_cycle;
        l3_free_time[node.l3_id] = sn.end_cycle;
        schedule.makespan = std::max(schedule.makespan, sn.end_cycle);
    }

    return schedule;
}

int64_t DFGScheduler::estimate_makespan(const TileDataFlowGraph& dfg) const {
    return dfg.critical_path_length();
}

} // namespace sw::kpu::dataflow
