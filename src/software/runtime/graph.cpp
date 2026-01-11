// Multi-Layer Computational Graph Implementation
// Enables execution of multi-layer neural networks

#include <sw/runtime/graph.hpp>

#include <algorithm>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <iomanip>

namespace sw::runtime {

using namespace sw::kpu;

// ============================================================================
// ComputeGraph Implementation
// ============================================================================

ComputeGraph::ComputeGraph(const std::string& name)
    : name_(name) {}

void ComputeGraph::add_input(const std::string& name,
                              const std::vector<Size>& shape,
                              DataType dtype) {
    if (finalized_) {
        throw std::runtime_error("Cannot modify finalized graph");
    }

    if (tensors_.count(name)) {
        throw std::runtime_error("Tensor '" + name + "' already exists");
    }

    TensorDescriptor tensor(name, shape, dtype);
    tensor.is_graph_input = true;
    tensor.producer_node = SIZE_MAX;  // No producer (it's an input)
    tensors_[name] = std::move(tensor);
    input_names_.push_back(name);
}

size_t ComputeGraph::add_node(const std::string& name,
                               const Kernel& kernel,
                               const std::vector<std::string>& inputs,
                               const std::vector<std::string>& outputs) {
    if (finalized_) {
        throw std::runtime_error("Cannot modify finalized graph");
    }

    if (node_name_to_index_.count(name)) {
        throw std::runtime_error("Node '" + name + "' already exists");
    }

    // Create the node
    GraphNode node(name, kernel, inputs, outputs);
    node.node_index = nodes_.size();

    // Register the node
    size_t index = nodes_.size();
    node_name_to_index_[name] = index;
    nodes_.push_back(std::move(node));

    // Create output tensors and link connectivity
    infer_tensor_shapes(index);

    return index;
}

void ComputeGraph::infer_tensor_shapes(size_t node_index) {
    const auto& node = nodes_[node_index];
    const auto& kernel = node.kernel;

    // Link input tensors to this node as a consumer
    for (const auto& input_name : node.inputs) {
        auto it = tensors_.find(input_name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Node '" + node.name +
                                     "' references unknown input tensor '" + input_name + "'");
        }
        it->second.consumer_nodes.push_back(node_index);
    }

    // Create tensors for kernel arguments
    // - "A" argument: mapped to node.inputs[0] (already exists)
    // - "C" argument: mapped to node.outputs[0] (create as output tensor)
    // - Other arguments (B, bias): created as weight tensors with node-prefixed names
    const auto& args = kernel.arguments();
    size_t output_idx = 0;

    for (const auto& arg : args) {
        if (arg.name == "A") {
            // Input activation - already linked above
            continue;
        } else if (arg.name == "C" || arg.is_output) {
            // Output activation
            if (output_idx >= node.outputs.size()) {
                throw std::runtime_error("Node '" + node.name +
                                         "' has more output arguments than output tensor names");
            }
            const std::string& output_name = node.outputs[output_idx];

            if (tensors_.count(output_name)) {
                throw std::runtime_error("Output tensor '" + output_name + "' already exists");
            }

            TensorDescriptor tensor(output_name, arg.shape, arg.dtype);
            tensor.producer_node = node_index;
            tensors_[output_name] = std::move(tensor);

            ++output_idx;
        } else {
            // Weight tensor (B, bias, etc.) - create with node-prefixed name
            std::string weight_name = node.name + "_" + arg.name;

            if (tensors_.count(weight_name)) {
                throw std::runtime_error("Weight tensor '" + weight_name + "' already exists");
            }

            TensorDescriptor tensor(weight_name, arg.shape, arg.dtype);
            tensor.producer_node = SIZE_MAX;  // No producer - it's a weight
            tensor.consumer_nodes.push_back(node_index);
            tensors_[weight_name] = std::move(tensor);
        }
    }
}

void ComputeGraph::add_output(const std::string& name) {
    if (finalized_) {
        throw std::runtime_error("Cannot modify finalized graph");
    }

    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Cannot mark unknown tensor '" + name + "' as output");
    }

    it->second.is_graph_output = true;
    output_names_.push_back(name);
}

void ComputeGraph::finalize() {
    if (finalized_) {
        return;
    }

    std::string error;
    if (!validate(error)) {
        throw std::runtime_error("Graph validation failed: " + error);
    }

    compute_dependencies();
    compute_topological_order();

    finalized_ = true;
}

void ComputeGraph::compute_dependencies() {
    for (auto& node : nodes_) {
        node.dependencies.clear();

        for (const auto& input_name : node.inputs) {
            const auto& tensor = tensors_.at(input_name);
            if (tensor.producer_node != SIZE_MAX) {
                // This tensor is produced by another node
                node.dependencies.push_back(tensor.producer_node);
            }
        }

        // Remove duplicates
        std::sort(node.dependencies.begin(), node.dependencies.end());
        node.dependencies.erase(
            std::unique(node.dependencies.begin(), node.dependencies.end()),
            node.dependencies.end());
    }
}

void ComputeGraph::compute_topological_order() {
    execution_order_.clear();

    // Compute in-degrees
    std::vector<size_t> in_degree(nodes_.size(), 0);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        in_degree[i] = nodes_[i].dependencies.size();
    }

    // Start with nodes that have no dependencies
    std::queue<size_t> ready;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (in_degree[i] == 0) {
            ready.push(i);
            nodes_[i].depth = 0;
        }
    }

    while (!ready.empty()) {
        size_t current = ready.front();
        ready.pop();
        execution_order_.push_back(current);

        // Find nodes that depend on current and decrement their in-degree
        for (size_t i = 0; i < nodes_.size(); ++i) {
            auto& deps = nodes_[i].dependencies;
            if (std::find(deps.begin(), deps.end(), current) != deps.end()) {
                --in_degree[i];
                nodes_[i].depth = std::max(nodes_[i].depth, nodes_[current].depth + 1);
                if (in_degree[i] == 0) {
                    ready.push(i);
                }
            }
        }
    }

    if (execution_order_.size() != nodes_.size()) {
        throw std::runtime_error("Graph contains a cycle");
    }
}

bool ComputeGraph::validate(std::string& error) const {
    // Check for empty graph
    if (nodes_.empty()) {
        error = "Graph has no nodes";
        return false;
    }

    // Check for inputs
    if (input_names_.empty()) {
        error = "Graph has no inputs";
        return false;
    }

    // Check for outputs
    if (output_names_.empty()) {
        error = "Graph has no outputs";
        return false;
    }

    // Check all input tensors are defined
    for (const auto& node : nodes_) {
        for (const auto& input_name : node.inputs) {
            if (!tensors_.count(input_name)) {
                error = "Node '" + node.name + "' references undefined tensor '" + input_name + "'";
                return false;
            }
        }
    }

    // Check all outputs are produced
    for (const auto& output_name : output_names_) {
        auto it = tensors_.find(output_name);
        if (it == tensors_.end()) {
            error = "Output tensor '" + output_name + "' is not defined";
            return false;
        }
        if (it->second.producer_node == SIZE_MAX && !it->second.is_graph_input) {
            error = "Output tensor '" + output_name + "' has no producer";
            return false;
        }
    }

    return true;
}

Size ComputeGraph::total_memory_bytes() const {
    Size total = 0;
    for (const auto& [name, tensor] : tensors_) {
        total += tensor.size_bytes;
    }
    return total;
}

Size ComputeGraph::peak_memory_bytes() const {
    // Simple estimation: sum of all non-input tensors that could be live at once
    // A more sophisticated analysis would track tensor lifetimes
    Size peak = 0;
    for (const auto& [name, tensor] : tensors_) {
        peak += tensor.size_bytes;
    }
    return peak;
}

const GraphNode* ComputeGraph::find_node(const std::string& name) const {
    auto it = node_name_to_index_.find(name);
    if (it == node_name_to_index_.end()) {
        return nullptr;
    }
    return &nodes_[it->second];
}

const TensorDescriptor* ComputeGraph::find_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return &it->second;
}

std::string ComputeGraph::summary() const {
    std::ostringstream ss;
    ss << "ComputeGraph: " << (name_.empty() ? "(unnamed)" : name_) << "\n";
    ss << "  Nodes: " << nodes_.size() << "\n";
    ss << "  Tensors: " << tensors_.size() << "\n";
    ss << "  Inputs: ";
    for (size_t i = 0; i < input_names_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << input_names_[i];
    }
    ss << "\n";
    ss << "  Outputs: ";
    for (size_t i = 0; i < output_names_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << output_names_[i];
    }
    ss << "\n";
    ss << "  Total memory: " << (total_memory_bytes() / 1024) << " KB\n";
    ss << "  Finalized: " << (finalized_ ? "yes" : "no") << "\n";

    if (finalized_) {
        ss << "\n  Execution order:\n";
        for (size_t i = 0; i < execution_order_.size(); ++i) {
            const auto& node = nodes_[execution_order_[i]];
            ss << "    " << (i + 1) << ". " << node.name
               << " (depth=" << node.depth << ")\n";
        }
    }

    return ss.str();
}

// ============================================================================
// GraphRunner Implementation
// ============================================================================

GraphRunner::GraphRunner(KPURuntime* runtime)
    : runtime_(runtime) {
    if (!runtime_) {
        throw std::invalid_argument("GraphRunner: runtime cannot be null");
    }
}

GraphRunner::~GraphRunner() {
    release();
}

GraphRunner::GraphRunner(GraphRunner&&) noexcept = default;
GraphRunner& GraphRunner::operator=(GraphRunner&&) noexcept = default;

void GraphRunner::set_graph(const ComputeGraph& graph) {
    if (!graph.is_finalized()) {
        throw std::invalid_argument("GraphRunner: graph must be finalized");
    }

    release();
    graph_ = std::make_unique<ComputeGraph>(graph);
}

GraphRunner::MemoryPlan GraphRunner::plan_memory() {
    MemoryPlan plan;

    if (!graph_) {
        return plan;
    }

    // Simple strategy: allocate separate buffer for each tensor
    // TODO: Implement buffer reuse for non-overlapping lifetimes
    for (const auto& [name, tensor] : graph_->tensors()) {
        size_t buffer_idx = plan.buffer_sizes.size();
        plan.tensor_to_buffer[name] = buffer_idx;
        plan.buffer_sizes.push_back(tensor.size_bytes);
    }

    return plan;
}

void GraphRunner::allocate() {
    if (!graph_) {
        throw std::runtime_error("GraphRunner: no graph set");
    }

    release();

    // Plan memory allocation
    auto plan = plan_memory();

    // Allocate buffers
    total_allocated_ = 0;
    for (Size buffer_size : plan.buffer_sizes) {
        Address addr = runtime_->malloc(buffer_size);
        if (addr == 0) {
            release();
            throw std::runtime_error("GraphRunner: failed to allocate " +
                                     std::to_string(buffer_size) + " bytes");
        }
        allocated_buffers_.push_back(addr);
        total_allocated_ += buffer_size;
    }

    // Map tensors to their buffer addresses
    for (const auto& [tensor_name, buffer_idx] : plan.tensor_to_buffer) {
        tensor_addresses_[tensor_name] = allocated_buffers_[buffer_idx];
    }

    allocated_ = true;
}

void GraphRunner::release() {
    for (Address addr : allocated_buffers_) {
        runtime_->free(addr);
    }
    allocated_buffers_.clear();
    tensor_addresses_.clear();
    total_allocated_ = 0;
    allocated_ = false;
}

void GraphRunner::set_input(const std::string& name, const void* data, Size size_bytes) {
    if (!allocated_) {
        throw std::runtime_error("GraphRunner: memory not allocated");
    }

    auto it = tensor_addresses_.find(name);
    if (it == tensor_addresses_.end()) {
        throw std::invalid_argument("GraphRunner: unknown tensor '" + name + "'");
    }

    const auto* tensor = graph_->find_tensor(name);
    if (!tensor || !tensor->is_graph_input) {
        throw std::invalid_argument("GraphRunner: '" + name + "' is not a graph input");
    }

    if (size_bytes > tensor->size_bytes) {
        throw std::invalid_argument("GraphRunner: input size exceeds tensor size");
    }

    runtime_->memcpy_h2d(it->second, data, size_bytes);
}

void GraphRunner::set_input(const std::string& name, const void* data,
                             const std::vector<Size>& shape) {
    const auto* tensor = graph_->find_tensor(name);
    if (!tensor) {
        throw std::invalid_argument("GraphRunner: unknown tensor '" + name + "'");
    }

    if (shape != tensor->shape) {
        std::ostringstream ss;
        ss << "GraphRunner: shape mismatch for '" << name << "'";
        throw std::invalid_argument(ss.str());
    }

    set_input(name, data, tensor->size_bytes);
}

void GraphRunner::get_output(const std::string& name, void* data, Size size_bytes) {
    if (!allocated_) {
        throw std::runtime_error("GraphRunner: memory not allocated");
    }

    auto it = tensor_addresses_.find(name);
    if (it == tensor_addresses_.end()) {
        throw std::invalid_argument("GraphRunner: unknown tensor '" + name + "'");
    }

    const auto* tensor = graph_->find_tensor(name);
    if (!tensor || !tensor->is_graph_output) {
        throw std::invalid_argument("GraphRunner: '" + name + "' is not a graph output");
    }

    if (size_bytes > tensor->size_bytes) {
        throw std::invalid_argument("GraphRunner: output size exceeds tensor size");
    }

    runtime_->memcpy_d2h(data, it->second, size_bytes);
}

void GraphRunner::get_output(const std::string& name, void* data) {
    const auto* tensor = graph_->find_tensor(name);
    if (!tensor) {
        throw std::invalid_argument("GraphRunner: unknown tensor '" + name + "'");
    }
    get_output(name, data, tensor->size_bytes);
}

Address GraphRunner::get_tensor_address(const std::string& name) const {
    auto it = tensor_addresses_.find(name);
    if (it == tensor_addresses_.end()) {
        return 0;
    }
    return it->second;
}

std::vector<Address> GraphRunner::build_node_arguments(size_t node_index) {
    const auto& node = graph_->node(node_index);
    const auto& kernel = node.kernel;
    const auto& args = kernel.arguments();

    std::vector<Address> addresses;
    addresses.reserve(args.size());

    // For each kernel argument, find the corresponding tensor
    // Kernel arguments are named (A, B, bias, C for MLP; A, B, C for matmul)
    // Graph node has:
    //   - inputs[0] corresponds to kernel arg "A" (activation input)
    //   - outputs[0] corresponds to kernel arg "C" (activation output)
    //   - Other args (B, bias) are weight tensors created by the graph

    for (const auto& arg : args) {
        std::string tensor_name;

        // Map kernel arguments to graph tensors
        if (arg.name == "A" && !node.inputs.empty()) {
            // Input activation - comes from graph input or previous layer
            tensor_name = node.inputs[0];
        } else if (arg.name == "C" && !node.outputs.empty()) {
            // Output activation - goes to next layer or graph output
            tensor_name = node.outputs[0];
        } else {
            // Weight tensors (B, bias) - use node-prefixed name
            tensor_name = node.name + "_" + arg.name;
        }

        Address addr = get_tensor_address(tensor_name);
        if (addr == 0) {
            throw std::runtime_error("GraphRunner: tensor '" + tensor_name +
                                     "' not allocated for argument '" + arg.name +
                                     "' in node '" + node.name + "'");
        }
        addresses.push_back(addr);
    }

    return addresses;
}

ExecutionResult GraphRunner::run_node(size_t node_index) {
    if (!allocated_) {
        return ExecutionResult(false, 0, 0.0, "Memory not allocated");
    }

    if (node_index >= graph_->num_nodes()) {
        return ExecutionResult(false, 0, 0.0, "Invalid node index");
    }

    const auto& node = graph_->node(node_index);

    // Build argument list
    std::vector<Address> args;
    try {
        args = build_node_arguments(node_index);
    } catch (const std::exception& e) {
        return ExecutionResult(false, 0, 0.0, e.what());
    }

    // Launch the kernel
    auto result = runtime_->launch(node.kernel, args);

    if (!result.success) {
        return ExecutionResult(false, 0, 0.0, result.error);
    }

    double time_ms = static_cast<double>(result.cycles) /
                     (runtime_->config().clock_ghz * 1e6);

    return ExecutionResult(true, result.cycles, time_ms);
}

GraphExecutionResult GraphRunner::run() {
    if (!graph_) {
        last_result_ = GraphExecutionResult(false, 0, 0.0, "No graph set");
        return last_result_;
    }

    if (!allocated_) {
        last_result_ = GraphExecutionResult(false, 0, 0.0, "Memory not allocated");
        return last_result_;
    }

    GraphExecutionResult result;
    result.success = true;
    result.total_cycles = 0;
    result.total_time_ms = 0.0;

    // Execute nodes in topological order
    for (size_t node_idx : graph_->execution_order()) {
        const auto& node = graph_->node(node_idx);

        auto node_result = run_node(node_idx);

        if (!node_result.success) {
            result.success = false;
            result.error = "Node '" + node.name + "' failed: " + node_result.error;
            last_result_ = result;
            return result;
        }

        // Record profile
        LayerProfile profile;
        profile.name = node.name;
        profile.node_index = node_idx;
        profile.cycles = node_result.cycles;
        profile.time_ms = node_result.time_ms;

        // Calculate input/output bytes
        profile.input_bytes = 0;
        for (const auto& input_name : node.inputs) {
            const auto* tensor = graph_->find_tensor(input_name);
            if (tensor) {
                profile.input_bytes += tensor->size_bytes;
            }
        }

        profile.output_bytes = 0;
        for (const auto& output_name : node.outputs) {
            const auto* tensor = graph_->find_tensor(output_name);
            if (tensor) {
                profile.output_bytes += tensor->size_bytes;
            }
        }

        result.layer_profiles.push_back(profile);
        result.total_cycles += node_result.cycles;
        result.total_time_ms += node_result.time_ms;
    }

    last_result_ = result;
    return result;
}

} // namespace sw::runtime
