// Kernel Graph Implementation
// Multi-kernel DAG representation and compilation

#include <sw/kpu/kernel_graph.hpp>

#include <algorithm>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>

namespace sw::kpu {

// ============================================================================
// Constructors
// ============================================================================

KernelGraph::KernelGraph(std::string name)
    : name_(std::move(name)) {}

// ============================================================================
// Node Management
// ============================================================================

size_t KernelGraph::add_kernel(Kernel kernel, const std::string& name) {
    return add_kernel(std::make_unique<Kernel>(std::move(kernel)), name);
}

size_t KernelGraph::add_kernel(std::unique_ptr<Kernel> kernel, const std::string& name) {
    if (!kernel || !kernel->is_valid()) {
        throw std::invalid_argument("Cannot add invalid kernel to graph");
    }

    size_t id = next_node_id_++;
    std::string node_name = name.empty() ? kernel->name() : name;

    nodes_.emplace(id, KernelNode(id, std::move(kernel), std::move(node_name)));
    invalidate_cache();

    return id;
}

const KernelNode& KernelGraph::get_node(size_t node_id) const {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        throw std::out_of_range("Node ID " + std::to_string(node_id) + " not found");
    }
    return it->second;
}

KernelNode& KernelGraph::get_node(size_t node_id) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        throw std::out_of_range("Node ID " + std::to_string(node_id) + " not found");
    }
    return it->second;
}

const Kernel& KernelGraph::get_kernel(size_t node_id) const {
    return *get_node(node_id).kernel;
}

Kernel& KernelGraph::get_kernel(size_t node_id) {
    return *get_node(node_id).kernel;
}

bool KernelGraph::has_node(size_t node_id) const {
    return nodes_.find(node_id) != nodes_.end();
}

std::vector<size_t> KernelGraph::node_ids() const {
    std::vector<size_t> ids;
    ids.reserve(nodes_.size());
    for (const auto& [id, node] : nodes_) {
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

// ============================================================================
// Edge Management
// ============================================================================

size_t KernelGraph::add_edge(size_t from_node, size_t to_node,
                             const std::string& output_name,
                             const std::string& input_name) {
    // Validate nodes exist
    if (!has_node(from_node)) {
        throw std::invalid_argument("Source node " + std::to_string(from_node) + " not found");
    }
    if (!has_node(to_node)) {
        throw std::invalid_argument("Target node " + std::to_string(to_node) + " not found");
    }

    // Check for self-loop
    if (from_node == to_node) {
        throw std::invalid_argument("Self-loops are not allowed");
    }

    // Check for cycle
    if (would_create_cycle(from_node, to_node)) {
        throw std::invalid_argument("Edge would create a cycle in the graph");
    }

    // Calculate tensor size from producer's output
    const Kernel& producer = get_kernel(from_node);
    Size tensor_size = 0;
    for (const auto& arg : producer.arguments()) {
        if (arg.name == output_name && arg.is_output) {
            tensor_size = arg.size_bytes;
            break;
        }
    }

    size_t edge_id = edges_.size();
    edges_.emplace_back(from_node, to_node, output_name, input_name, tensor_size);

    // Update node edge lists
    nodes_.at(from_node).output_edges.push_back(edge_id);
    nodes_.at(to_node).input_edges.push_back(edge_id);

    invalidate_cache();
    return edge_id;
}

const KernelEdge& KernelGraph::get_edge(size_t edge_id) const {
    if (edge_id >= edges_.size()) {
        throw std::out_of_range("Edge ID " + std::to_string(edge_id) + " not found");
    }
    return edges_[edge_id];
}

bool KernelGraph::would_create_cycle(size_t from_node, size_t to_node) const {
    // If there's already a path from to_node to from_node, adding
    // from_node -> to_node would create a cycle
    std::unordered_set<size_t> visited;
    return has_path_dfs(to_node, from_node, visited);
}

bool KernelGraph::has_path_dfs(size_t from, size_t to,
                               std::unordered_set<size_t>& visited) const {
    if (from == to) return true;
    if (visited.count(from)) return false;

    visited.insert(from);

    auto it = nodes_.find(from);
    if (it == nodes_.end()) return false;

    for (size_t edge_id : it->second.output_edges) {
        if (has_path_dfs(edges_[edge_id].to_node, to, visited)) {
            return true;
        }
    }

    return false;
}

std::vector<size_t> KernelGraph::outgoing_edges(size_t node_id) const {
    if (!has_node(node_id)) return {};
    return get_node(node_id).output_edges;
}

std::vector<size_t> KernelGraph::incoming_edges(size_t node_id) const {
    if (!has_node(node_id)) return {};
    return get_node(node_id).input_edges;
}

// ============================================================================
// Graph Properties
// ============================================================================

bool KernelGraph::validate(std::string& error) const {
    if (nodes_.empty()) {
        error = "Graph is empty";
        return false;
    }

    // Check all edges reference valid nodes
    for (size_t i = 0; i < edges_.size(); ++i) {
        const auto& edge = edges_[i];
        if (!has_node(edge.from_node)) {
            error = "Edge " + std::to_string(i) + " references invalid source node";
            return false;
        }
        if (!has_node(edge.to_node)) {
            error = "Edge " + std::to_string(i) + " references invalid target node";
            return false;
        }
    }

    // Check for cycles by attempting topological sort
    try {
        get_execution_order();
    } catch (const std::runtime_error&) {
        error = "Graph contains cycles";
        return false;
    }

    // Check all kernels are valid
    for (const auto& [id, node] : nodes_) {
        if (!node.kernel || !node.kernel->is_valid()) {
            error = "Node " + std::to_string(id) + " has invalid kernel";
            return false;
        }
    }

    return true;
}

std::vector<size_t> KernelGraph::input_nodes() const {
    std::vector<size_t> inputs;
    for (const auto& [id, node] : nodes_) {
        if (node.input_edges.empty()) {
            inputs.push_back(id);
        }
    }
    std::sort(inputs.begin(), inputs.end());
    return inputs;
}

std::vector<size_t> KernelGraph::output_nodes() const {
    std::vector<size_t> outputs;
    for (const auto& [id, node] : nodes_) {
        if (node.output_edges.empty()) {
            outputs.push_back(id);
        }
    }
    std::sort(outputs.begin(), outputs.end());
    return outputs;
}

KernelGraphStats KernelGraph::compute_stats() const {
    KernelGraphStats stats;

    stats.num_nodes = nodes_.size();
    stats.num_edges = edges_.size();
    stats.num_input_nodes = input_nodes().size();
    stats.num_output_nodes = output_nodes().size();

    // Calculate depths and find max
    std::unordered_map<size_t, size_t> depths;
    for (const auto& [id, node] : nodes_) {
        size_t depth = calculate_node_depth(id, depths);
        stats.max_depth = std::max(stats.max_depth, depth);
    }

    // Aggregate kernel stats
    double total_intensity = 0.0;
    for (const auto& [id, node] : nodes_) {
        if (node.kernel) {
            stats.total_instructions += node.kernel->instruction_count();
            stats.total_flops += node.kernel->total_flops();
            stats.total_input_bytes += node.kernel->total_input_bytes();
            stats.total_output_bytes += node.kernel->total_output_bytes();
            total_intensity += node.kernel->arithmetic_intensity();
        }
    }

    // Calculate intermediate data (data passed between kernels)
    for (const auto& edge : edges_) {
        stats.intermediate_bytes += edge.tensor_size_bytes;
    }

    if (!nodes_.empty()) {
        stats.avg_arithmetic_intensity = total_intensity / static_cast<double>(nodes_.size());
    }

    return stats;
}

size_t KernelGraph::calculate_node_depth(size_t node_id,
                                         std::unordered_map<size_t, size_t>& depths) const {
    auto it = depths.find(node_id);
    if (it != depths.end()) return it->second;

    const auto& node = get_node(node_id);
    if (node.input_edges.empty()) {
        depths[node_id] = 0;
        return 0;
    }

    size_t max_parent_depth = 0;
    for (size_t edge_id : node.input_edges) {
        size_t parent_depth = calculate_node_depth(edges_[edge_id].from_node, depths);
        max_parent_depth = std::max(max_parent_depth, parent_depth);
    }

    depths[node_id] = max_parent_depth + 1;
    return depths[node_id];
}

// ============================================================================
// Execution Order
// ============================================================================

std::vector<size_t> KernelGraph::get_execution_order() const {
    if (cached_execution_order_.has_value()) {
        return *cached_execution_order_;
    }

    // Kahn's algorithm for topological sort
    std::unordered_map<size_t, size_t> in_degree;
    for (const auto& [id, node] : nodes_) {
        in_degree[id] = node.input_edges.size();
    }

    std::queue<size_t> ready;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) {
            ready.push(id);
        }
    }

    std::vector<size_t> order;
    order.reserve(nodes_.size());

    while (!ready.empty()) {
        size_t node_id = ready.front();
        ready.pop();
        order.push_back(node_id);

        for (size_t edge_id : get_node(node_id).output_edges) {
            size_t target = edges_[edge_id].to_node;
            if (--in_degree[target] == 0) {
                ready.push(target);
            }
        }
    }

    if (order.size() != nodes_.size()) {
        throw std::runtime_error("Graph contains a cycle - topological sort impossible");
    }

    cached_execution_order_ = order;
    return order;
}

std::vector<std::vector<size_t>> KernelGraph::get_execution_levels() const {
    std::unordered_map<size_t, size_t> depths;
    size_t max_depth = 0;

    for (const auto& [id, node] : nodes_) {
        size_t depth = calculate_node_depth(id, depths);
        max_depth = std::max(max_depth, depth);
    }

    std::vector<std::vector<size_t>> levels(max_depth + 1);
    for (const auto& [id, depth] : depths) {
        levels[depth].push_back(id);
    }

    // Sort each level for determinism
    for (auto& level : levels) {
        std::sort(level.begin(), level.end());
    }

    return levels;
}

std::vector<size_t> KernelGraph::get_critical_path() const {
    if (nodes_.empty()) return {};

    std::unordered_map<size_t, size_t> depths;
    std::unordered_map<size_t, size_t> parent_on_critical;

    // Find depths and track which parent gives max depth
    size_t max_depth = 0;
    size_t deepest_node = 0;

    for (const auto& [id, node] : nodes_) {
        size_t depth = calculate_node_depth(id, depths);
        if (depth > max_depth) {
            max_depth = depth;
            deepest_node = id;
        }
    }

    // Reconstruct critical path by backtracking
    std::vector<size_t> path;
    size_t current = deepest_node;

    while (true) {
        path.push_back(current);
        const auto& node = get_node(current);

        if (node.input_edges.empty()) break;

        // Find parent with max depth
        size_t max_parent_depth = 0;
        size_t next_node = current;
        for (size_t edge_id : node.input_edges) {
            size_t parent = edges_[edge_id].from_node;
            if (depths[parent] >= max_parent_depth) {
                max_parent_depth = depths[parent];
                next_node = parent;
            }
        }

        if (next_node == current) break;  // Safety check
        current = next_node;
    }

    std::reverse(path.begin(), path.end());
    return path;
}

// ============================================================================
// Fusion Optimization
// ============================================================================

std::vector<std::pair<size_t, size_t>> KernelGraph::find_fusible_pairs() const {
    std::vector<std::pair<size_t, size_t>> pairs;

    for (const auto& edge : edges_) {
        if (can_fuse(edge.from_node, edge.to_node)) {
            pairs.emplace_back(edge.from_node, edge.to_node);
        }
    }

    return pairs;
}

bool KernelGraph::can_fuse(size_t producer, size_t consumer) const {
    if (!has_node(producer) || !has_node(consumer)) return false;

    const auto& prod_node = get_node(producer);
    const auto& cons_node = get_node(consumer);

    // Consumer must have exactly one input from this producer
    if (cons_node.input_edges.size() != 1) return false;

    // Producer must have exactly one output going to this consumer
    bool has_edge_to_consumer = false;
    for (size_t edge_id : prod_node.output_edges) {
        if (edges_[edge_id].to_node == consumer) {
            has_edge_to_consumer = true;
        }
    }
    if (!has_edge_to_consumer) return false;

    // Check data type compatibility
    if (prod_node.kernel->dtype() != cons_node.kernel->dtype()) return false;

    // Check dimension compatibility (producer output matches consumer input)
    const auto& prod_outputs = prod_node.kernel->output_arguments();
    const auto& cons_inputs = cons_node.kernel->input_arguments();

    if (prod_outputs.empty() || cons_inputs.empty()) return false;

    // For matmul chain: producer C [M,N] should match consumer A [M,K]
    // This means M must match
    if (prod_node.kernel->M() != cons_node.kernel->M()) return false;
    if (prod_node.kernel->N() != cons_node.kernel->K()) return false;

    return true;
}

bool KernelGraph::mark_for_fusion(size_t producer, size_t consumer) {
    if (!can_fuse(producer, consumer)) return false;

    auto& prod_node = get_node(producer);
    auto& cons_node = get_node(consumer);

    prod_node.is_fused = true;
    prod_node.fused_with = consumer;
    cons_node.is_fused = true;
    cons_node.fused_with = producer;

    return true;
}

void KernelGraph::clear_fusion_marks() {
    for (auto& [id, node] : nodes_) {
        node.is_fused = false;
        node.fused_with = SIZE_MAX;
    }
}

// ============================================================================
// Compilation
// ============================================================================

KernelGraphCompileResult KernelGraph::compile(
    const KernelGraphCompileOptions& options) const {

    KernelGraphCompileResult result;

    std::string error;
    if (!validate(error)) {
        result.success = false;
        result.error_message = error;
        return result;
    }

    // No fusion requested - use sequential compilation
    if (options.fusion_strategy == FusionStrategy::NONE) {
        return compile_sequential();
    }

    // Find fusible pairs
    auto fusible = find_fusible_pairs();

    // If no fusible pairs found, fall back to sequential
    if (fusible.empty()) {
        return compile_sequential();
    }

    // Producer-consumer fusion: eliminate intermediate memory traffic
    if (options.fusion_strategy == FusionStrategy::PRODUCER_CONSUMER) {
        return compile_with_fusion(fusible, options);
    }

    // Other fusion strategies not yet implemented
    // HORIZONTAL and PIPELINE fall back to sequential for now
    return compile_sequential();
}

KernelGraphCompileResult KernelGraph::compile_sequential() const {
    KernelGraphCompileResult result;

    std::string error;
    if (!validate(error)) {
        result.success = false;
        result.error_message = error;
        return result;
    }

    result.execution_order = get_execution_order();

    // Create combined program
    result.program.name = name_.empty() ? "kernel_graph" : name_;
    result.program.version = 1;
    result.program.dataflow = isa::DMProgram::Dataflow::OUTPUT_STATIONARY;

    // Track memory allocation
    Address current_offset = 0;
    std::unordered_map<size_t, Address> node_base_addresses;

    // Compile each kernel in order
    for (size_t node_id : result.execution_order) {
        const auto& node = get_node(node_id);
        const Kernel& kernel = *node.kernel;

        node_base_addresses[node_id] = current_offset;

        // Copy instructions with offset adjustment
        for (const auto& instr : kernel.program().instructions) {
            // Skip HALT instructions except for the last kernel
            if (instr.opcode == isa::DMOpcode::HALT &&
                node_id != result.execution_order.back()) {
                continue;
            }

            result.program.instructions.push_back(instr);
        }

        // Add barrier between kernels (except after last)
        if (node_id != result.execution_order.back()) {
            result.program.instructions.push_back(isa::DMInstruction::barrier());
        }

        // Update dimensions (use first kernel's dimensions as base)
        if (result.program.M == 0) {
            result.program.M = kernel.program().M;
            result.program.N = kernel.program().N;
            result.program.K = kernel.program().K;
            result.program.Ti = kernel.program().Ti;
            result.program.Tj = kernel.program().Tj;
            result.program.Tk = kernel.program().Tk;
            result.program.L1_Ki = kernel.program().L1_Ki;
        }

        // Estimate workspace (simplified)
        result.workspace_required += kernel.total_input_bytes() + kernel.total_output_bytes();
    }

    // Aggregate estimates
    for (size_t node_id : result.execution_order) {
        const auto& kernel = *get_node(node_id).kernel;
        result.program.estimates.total_cycles += kernel.program().estimates.total_cycles;
        result.program.estimates.external_mem_bytes += kernel.program().estimates.external_mem_bytes;
        result.program.estimates.l3_bytes += kernel.program().estimates.l3_bytes;
        result.program.estimates.l2_bytes += kernel.program().estimates.l2_bytes;
    }

    // Calculate aggregate arithmetic intensity
    Size total_flops = 0;
    Size total_bytes = 0;
    for (size_t node_id : result.execution_order) {
        const auto& kernel = *get_node(node_id).kernel;
        total_flops += kernel.total_flops();
        total_bytes += kernel.total_input_bytes() + kernel.total_output_bytes();
    }
    if (total_bytes > 0) {
        result.program.estimates.arithmetic_intensity =
            static_cast<double>(total_flops) / static_cast<double>(total_bytes);
    }

    result.success = true;
    return result;
}

void KernelGraph::append_kernel_program(isa::DMProgram& target,
                                        const Kernel& kernel,
                                        Address /*base_offset*/) const {
    for (const auto& instr : kernel.program().instructions) {
        target.instructions.push_back(instr);
    }
}

// ============================================================================
// Fusion Analysis and Compilation
// ============================================================================

KernelGraph::FusionAnalysis
KernelGraph::analyze_producer_for_fusion(const Kernel& producer) const {
    FusionAnalysis analysis;
    const auto& instrs = producer.program().instructions;

    for (size_t i = 0; i < instrs.size(); ++i) {
        const auto& instr = instrs[i];

        if (instr.opcode == isa::DMOpcode::DMA_STORE_TILE) {
            if (std::holds_alternative<isa::DMAOperands>(instr.operands)) {
                const auto& ops = std::get<isa::DMAOperands>(instr.operands);
                if (ops.matrix == isa::MatrixID::C) {
                    analysis.c_store_indices.push_back(i);
                    analysis.c_l3_offset = ops.l3_offset;
                }
            }
        }
    }

    return analysis;
}

KernelGraph::FusionAnalysis
KernelGraph::analyze_consumer_for_fusion(const Kernel& consumer) const {
    FusionAnalysis analysis;
    const auto& instrs = consumer.program().instructions;

    for (size_t i = 0; i < instrs.size(); ++i) {
        const auto& instr = instrs[i];

        if (instr.opcode == isa::DMOpcode::DMA_LOAD_TILE) {
            if (std::holds_alternative<isa::DMAOperands>(instr.operands)) {
                const auto& ops = std::get<isa::DMAOperands>(instr.operands);
                if (ops.matrix == isa::MatrixID::A) {
                    analysis.a_load_indices.push_back(i);
                }
            }
        }
    }

    return analysis;
}

void KernelGraph::compile_fused_pair(isa::DMProgram& target,
                                     const KernelNode& producer,
                                     const KernelNode& consumer,
                                     Address /*base_offset*/) const {
    // Analyze both programs to find instructions to skip/modify
    auto prod_analysis = analyze_producer_for_fusion(*producer.kernel);
    auto cons_analysis = analyze_consumer_for_fusion(*consumer.kernel);

    // Build skip sets for producer (skip C stores to external memory)
    std::set<size_t> prod_skip(prod_analysis.c_store_indices.begin(),
                                prod_analysis.c_store_indices.end());

    // Build skip sets for consumer (skip A loads from external memory)
    std::set<size_t> cons_skip(cons_analysis.a_load_indices.begin(),
                                cons_analysis.a_load_indices.end());

    // === PHASE 1: Emit producer instructions ===
    // Skip C stores (keep intermediate in L3) and HALT
    const auto& prod_instrs = producer.kernel->program().instructions;
    for (size_t i = 0; i < prod_instrs.size(); ++i) {
        const auto& instr = prod_instrs[i];

        // Skip instructions in skip set
        if (prod_skip.count(i)) continue;

        // Skip HALT - we'll add one at the very end
        if (instr.opcode == isa::DMOpcode::HALT) continue;

        target.instructions.push_back(instr);
    }

    // === PHASE 2: Add minimal sync barrier ===
    // Ensures producer's output is complete before consumer starts
    target.instructions.push_back(isa::DMInstruction::barrier());

    // === PHASE 3: Emit consumer instructions ===
    // Skip A loads (data is already in L3 from producer's C)
    // Remap A's L3 source references to producer's C location
    const auto& cons_instrs = consumer.kernel->program().instructions;
    for (size_t i = 0; i < cons_instrs.size(); ++i) {
        // Skip instructions in skip set
        if (cons_skip.count(i)) continue;

        auto instr = cons_instrs[i];  // Copy for potential modification

        // Remap BM_MOVE_TILE for A to use producer's C location in L3
        if (instr.opcode == isa::DMOpcode::BM_MOVE_TILE ||
            instr.opcode == isa::DMOpcode::BM_TRANSPOSE_TILE) {
            if (std::holds_alternative<isa::BlockMoverOperands>(instr.operands)) {
                auto& ops = std::get<isa::BlockMoverOperands>(instr.operands);
                if (ops.matrix == isa::MatrixID::A) {
                    // Consumer's A is producer's C - remap L3 source
                    ops.src_offset = prod_analysis.c_l3_offset;
                    // Update label for debugging
                    instr.label = "BM_MOVE A(fused_from_C) L3→L2";
                }
            }
        }

        target.instructions.push_back(instr);
    }
}

KernelGraphCompileResult KernelGraph::compile_with_fusion(
    const std::vector<std::pair<size_t, size_t>>& fused_pairs,
    const KernelGraphCompileOptions& /*options*/) const {

    KernelGraphCompileResult result;
    result.execution_order = get_execution_order();
    result.fused_pairs = fused_pairs;

    // Create combined program
    result.program.name = name_.empty() ? "kernel_graph_fused" : name_ + "_fused";
    result.program.version = 1;
    result.program.dataflow = isa::DMProgram::Dataflow::OUTPUT_STATIONARY;

    // Track which nodes have been processed
    std::unordered_set<size_t> processed;

    // Build lookup for quick fused pair checking
    std::unordered_map<size_t, size_t> producer_to_consumer;
    for (const auto& [prod, cons] : fused_pairs) {
        producer_to_consumer[prod] = cons;
    }

    // Compile in execution order
    for (size_t node_id : result.execution_order) {
        if (processed.count(node_id)) continue;

        const auto& node = get_node(node_id);

        // Check if this node is a producer in a fused pair
        auto fuse_it = producer_to_consumer.find(node_id);
        if (fuse_it != producer_to_consumer.end()) {
            size_t consumer_id = fuse_it->second;

            // Compile as fused pair
            compile_fused_pair(result.program,
                               node,
                               get_node(consumer_id),
                               0);

            processed.insert(node_id);
            processed.insert(consumer_id);
        } else {
            // Compile as standalone kernel
            const Kernel& kernel = *node.kernel;

            for (const auto& instr : kernel.program().instructions) {
                // Skip HALT except for the last kernel
                if (instr.opcode == isa::DMOpcode::HALT &&
                    node_id != result.execution_order.back()) {
                    continue;
                }
                result.program.instructions.push_back(instr);
            }

            // Add barrier between kernels (except after last)
            if (node_id != result.execution_order.back()) {
                result.program.instructions.push_back(isa::DMInstruction::barrier());
            }

            processed.insert(node_id);
        }
    }

    // Update program dimensions from first kernel
    if (!result.execution_order.empty()) {
        const auto& first_kernel = *get_node(result.execution_order[0]).kernel;
        result.program.M = first_kernel.program().M;
        result.program.N = first_kernel.program().N;
        result.program.K = first_kernel.program().K;
        result.program.Ti = first_kernel.program().Ti;
        result.program.Tj = first_kernel.program().Tj;
        result.program.Tk = first_kernel.program().Tk;
        result.program.L1_Ki = first_kernel.program().L1_Ki;
    }

    // Update estimates with fusion savings
    update_fused_estimates(result, fused_pairs);

    result.success = true;
    return result;
}

void KernelGraph::update_fused_estimates(
    KernelGraphCompileResult& result,
    const std::vector<std::pair<size_t, size_t>>& fused_pairs) const {

    // Calculate base estimates from all kernels
    Size total_external = 0;
    Size total_flops = 0;

    for (size_t node_id : result.execution_order) {
        const auto& kernel = *get_node(node_id).kernel;
        result.program.estimates.total_cycles += kernel.program().estimates.total_cycles;
        result.program.estimates.l3_bytes += kernel.program().estimates.l3_bytes;
        result.program.estimates.l2_bytes += kernel.program().estimates.l2_bytes;
        total_external += kernel.program().estimates.external_mem_bytes;
        total_flops += kernel.total_flops();
    }

    // Calculate memory savings from fusion
    // For each fused pair: savings = 2 * intermediate_size (one store + one load)
    Size bytes_saved = 0;
    for (const auto& [prod_id, cons_id] : fused_pairs) {
        const auto& producer = *get_node(prod_id).kernel;
        // Producer's C output size = M * N * dtype_size
        Size c_size = producer.M() * producer.N() * dtype_size(producer.dtype());
        // Saved: write from producer + read from consumer
        bytes_saved += 2 * c_size;
    }

    // Apply savings
    result.program.estimates.external_mem_bytes =
        (total_external > bytes_saved) ? (total_external - bytes_saved) : 0;

    // Recalculate arithmetic intensity with reduced memory traffic
    if (result.program.estimates.external_mem_bytes > 0) {
        result.program.estimates.arithmetic_intensity =
            static_cast<double>(total_flops) / static_cast<double>(result.program.estimates.external_mem_bytes);
    }

    // Estimate workspace required
    result.workspace_required = 0;
    for (size_t node_id : result.execution_order) {
        const auto& kernel = *get_node(node_id).kernel;
        result.workspace_required += kernel.total_input_bytes() + kernel.total_output_bytes();
    }
}

// ============================================================================
// Utility
// ============================================================================

void KernelGraph::invalidate_cache() {
    cached_execution_order_.reset();
}

std::string KernelGraph::summary() const {
    std::ostringstream oss;

    oss << "=== Kernel Graph";
    if (!name_.empty()) {
        oss << ": " << name_;
    }
    oss << " ===\n";

    auto stats = compute_stats();

    oss << "Nodes: " << stats.num_nodes << "\n";
    oss << "Edges: " << stats.num_edges << "\n";
    oss << "Input nodes: " << stats.num_input_nodes << "\n";
    oss << "Output nodes: " << stats.num_output_nodes << "\n";
    oss << "Max depth: " << stats.max_depth << "\n";
    oss << "\n";

    oss << "Total instructions: " << stats.total_instructions << "\n";
    oss << "Total FLOPs: " << stats.total_flops << "\n";
    oss << "Intermediate data: " << stats.intermediate_bytes << " bytes\n";
    oss << "Avg arithmetic intensity: " << std::fixed << std::setprecision(2)
        << stats.avg_arithmetic_intensity << " FLOP/byte\n";
    oss << "\n";

    oss << "Kernels:\n";
    for (const auto& [id, node] : nodes_) {
        oss << "  [" << id << "] " << node.name << " ("
            << kernel_op_type_name(node.kernel->op_type()) << ")\n";
        oss << "       Dims: " << node.kernel->M() << "x"
            << node.kernel->N() << "x" << node.kernel->K() << "\n";
    }

    if (!edges_.empty()) {
        oss << "\nEdges:\n";
        for (size_t i = 0; i < edges_.size(); ++i) {
            const auto& edge = edges_[i];
            oss << "  [" << i << "] " << edge.from_node << "." << edge.output_name
                << " -> " << edge.to_node << "." << edge.input_name;
            if (edge.tensor_size_bytes > 0) {
                oss << " (" << edge.tensor_size_bytes << " bytes)";
            }
            oss << "\n";
        }
    }

    return oss.str();
}

std::string KernelGraph::to_dot(bool show_tensor_sizes) const {
    std::ostringstream oss;

    oss << "digraph KernelGraph {\n";
    oss << "  rankdir=TB;\n";
    oss << "  node [shape=box, style=rounded];\n\n";

    // Nodes
    for (const auto& [id, node] : nodes_) {
        oss << "  node" << id << " [label=\"" << node.name << "\\n"
            << kernel_op_type_name(node.kernel->op_type()) << "\\n"
            << node.kernel->M() << "x" << node.kernel->N() << "x"
            << node.kernel->K() << "\"];\n";
    }

    oss << "\n";

    // Edges
    for (size_t i = 0; i < edges_.size(); ++i) {
        const auto& edge = edges_[i];
        oss << "  node" << edge.from_node << " -> node" << edge.to_node;
        if (show_tensor_sizes && edge.tensor_size_bytes > 0) {
            oss << " [label=\"" << edge.output_name << "->" << edge.input_name;
            if (edge.tensor_size_bytes >= 1024 * 1024) {
                oss << " (" << (edge.tensor_size_bytes / (1024 * 1024)) << " MB)";
            } else if (edge.tensor_size_bytes >= 1024) {
                oss << " (" << (edge.tensor_size_bytes / 1024) << " KB)";
            } else {
                oss << " (" << edge.tensor_size_bytes << " B)";
            }
            oss << "\"]";
        }
        oss << ";\n";
    }

    oss << "}\n";

    return oss.str();
}

} // namespace sw::kpu
