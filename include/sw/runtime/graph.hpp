#pragma once
// Multi-Layer Computational Graph for KPU Runtime
// Enables execution of multi-layer neural networks
//
// Provides:
// - Graph construction with nodes and edges
// - Topological ordering for execution
// - Memory planning for tensor reuse
// - Per-layer profiling

#include <sw/runtime/runtime.hpp>
#include <sw/runtime/executor.hpp>
#include <sw/kpu/kernel.hpp>

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace sw::runtime {

// Forward declarations
class ComputeGraph;
class GraphRunner;

/**
 * @brief Tensor descriptor in a computational graph
 *
 * Represents an intermediate tensor (activation) flowing between nodes.
 */
struct TensorDescriptor {
    std::string name;
    std::vector<kpu::Size> shape;
    kpu::DataType dtype = kpu::DataType::FLOAT32;
    kpu::Size size_bytes = 0;

    // Graph connectivity
    size_t producer_node = SIZE_MAX;  // Node that produces this tensor (SIZE_MAX = graph input)
    std::vector<size_t> consumer_nodes;  // Nodes that consume this tensor

    // Memory allocation (filled by GraphRunner)
    kpu::Address device_address = 0;
    bool is_graph_input = false;
    bool is_graph_output = false;

    TensorDescriptor() = default;
    TensorDescriptor(const std::string& n, const std::vector<kpu::Size>& s,
                     kpu::DataType dt = kpu::DataType::FLOAT32)
        : name(n), shape(s), dtype(dt) {
        compute_size();
    }

    void compute_size() {
        size_bytes = kpu::dtype_size(dtype);
        for (auto dim : shape) {
            size_bytes *= dim;
        }
    }
};

/**
 * @brief Node in a computational graph
 *
 * Represents a single operation (kernel) in the graph.
 */
struct GraphNode {
    std::string name;
    kpu::Kernel kernel;

    // Tensor references (by name)
    std::vector<std::string> inputs;   // Input tensor names
    std::vector<std::string> outputs;  // Output tensor names

    // Computed during graph analysis
    size_t node_index = 0;
    std::vector<size_t> dependencies;  // Node indices this depends on
    size_t depth = 0;  // Depth in the graph (for scheduling)

    GraphNode() = default;
    GraphNode(const std::string& n, const kpu::Kernel& k,
              const std::vector<std::string>& ins,
              const std::vector<std::string>& outs)
        : name(n), kernel(k), inputs(ins), outputs(outs) {}
};

/**
 * @brief Multi-layer computational graph
 *
 * Represents a directed acyclic graph (DAG) of operations.
 * Supports construction, analysis, and serialization.
 *
 * Usage:
 * @code
 * ComputeGraph graph("my_mlp");
 *
 * // Define graph inputs
 * graph.add_input("input", {64, 784});
 *
 * // Add layers
 * auto layer1 = Kernel::create_mlp(64, 256, 784, ActivationType::RELU, true);
 * graph.add_node("fc1", layer1, {"input"}, {"h1"});
 *
 * auto layer2 = Kernel::create_mlp(64, 10, 256, ActivationType::NONE, true);
 * graph.add_node("fc2", layer2, {"h1"}, {"output"});
 *
 * // Mark outputs
 * graph.add_output("output");
 *
 * // Validate and analyze
 * graph.finalize();
 * @endcode
 */
class ComputeGraph {
public:
    // =========================================
    // Constructors
    // =========================================

    /**
     * @brief Construct an empty graph
     * @param name Graph name (optional)
     */
    explicit ComputeGraph(const std::string& name = "");

    ~ComputeGraph() = default;

    // Move semantics
    ComputeGraph(ComputeGraph&&) = default;
    ComputeGraph& operator=(ComputeGraph&&) = default;

    // Copy semantics
    ComputeGraph(const ComputeGraph&) = default;
    ComputeGraph& operator=(const ComputeGraph&) = default;

    // =========================================
    // Graph Construction
    // =========================================

    /**
     * @brief Add a graph input tensor
     * @param name Tensor name
     * @param shape Tensor shape
     * @param dtype Data type (default FLOAT32)
     */
    void add_input(const std::string& name,
                   const std::vector<kpu::Size>& shape,
                   kpu::DataType dtype = kpu::DataType::FLOAT32);

    /**
     * @brief Add a computation node to the graph
     * @param name Node name (must be unique)
     * @param kernel The kernel to execute
     * @param inputs Names of input tensors
     * @param outputs Names of output tensors
     * @return Node index
     *
     * Output tensor shapes are inferred from the kernel.
     */
    size_t add_node(const std::string& name,
                    const kpu::Kernel& kernel,
                    const std::vector<std::string>& inputs,
                    const std::vector<std::string>& outputs);

    /**
     * @brief Mark a tensor as a graph output
     * @param name Tensor name
     */
    void add_output(const std::string& name);

    /**
     * @brief Finalize the graph (validate and compute execution order)
     * @throws std::runtime_error if graph is invalid
     *
     * Must be called after all nodes are added and before execution.
     */
    void finalize();

    // =========================================
    // Graph Analysis
    // =========================================

    /**
     * @brief Check if graph has been finalized
     */
    bool is_finalized() const { return finalized_; }

    /**
     * @brief Get topological execution order
     * @return Vector of node indices in execution order
     */
    const std::vector<size_t>& execution_order() const { return execution_order_; }

    /**
     * @brief Get total memory required for all tensors
     */
    kpu::Size total_memory_bytes() const;

    /**
     * @brief Get peak memory usage (with optimal tensor reuse)
     */
    kpu::Size peak_memory_bytes() const;

    /**
     * @brief Get number of nodes
     */
    size_t num_nodes() const { return nodes_.size(); }

    /**
     * @brief Get number of tensors
     */
    size_t num_tensors() const { return tensors_.size(); }

    /**
     * @brief Get graph name
     */
    const std::string& name() const { return name_; }

    // =========================================
    // Accessors
    // =========================================

    /**
     * @brief Get all nodes
     */
    const std::vector<GraphNode>& nodes() const { return nodes_; }

    /**
     * @brief Get node by index
     */
    const GraphNode& node(size_t index) const { return nodes_.at(index); }

    /**
     * @brief Get node by name
     */
    const GraphNode* find_node(const std::string& name) const;

    /**
     * @brief Get all tensors
     */
    const std::unordered_map<std::string, TensorDescriptor>& tensors() const {
        return tensors_;
    }

    /**
     * @brief Get tensor by name
     */
    const TensorDescriptor* find_tensor(const std::string& name) const;

    /**
     * @brief Get graph input names
     */
    const std::vector<std::string>& input_names() const { return input_names_; }

    /**
     * @brief Get graph output names
     */
    const std::vector<std::string>& output_names() const { return output_names_; }

    // =========================================
    // Validation
    // =========================================

    /**
     * @brief Validate graph structure
     * @param error Output error message if invalid
     * @return true if valid
     */
    bool validate(std::string& error) const;

    /**
     * @brief Get human-readable summary
     */
    std::string summary() const;

private:
    std::string name_;
    std::vector<GraphNode> nodes_;
    std::unordered_map<std::string, TensorDescriptor> tensors_;
    std::unordered_map<std::string, size_t> node_name_to_index_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    std::vector<size_t> execution_order_;
    bool finalized_ = false;

    // Internal helpers
    void compute_dependencies();
    void compute_topological_order();
    void infer_tensor_shapes(size_t node_index);
};

/**
 * @brief Per-layer execution profile
 */
struct LayerProfile {
    std::string name;
    size_t node_index;
    kpu::Cycle cycles;
    double time_ms;
    kpu::Size input_bytes;
    kpu::Size output_bytes;
};

/**
 * @brief Result of graph execution
 */
struct GraphExecutionResult {
    bool success = false;
    kpu::Cycle total_cycles = 0;
    double total_time_ms = 0.0;
    std::string error;
    std::vector<LayerProfile> layer_profiles;

    GraphExecutionResult() = default;
    GraphExecutionResult(bool ok, kpu::Cycle c, double t, const std::string& err = "")
        : success(ok), total_cycles(c), total_time_ms(t), error(err) {}
};

/**
 * @brief Executor for multi-layer computational graphs
 *
 * Handles memory allocation, data transfer, and sequential execution
 * of graph nodes in topological order.
 *
 * Usage:
 * @code
 * KPUSimulator sim(config);
 * KPURuntime runtime(&sim);
 * GraphRunner runner(&runtime);
 *
 * // Set the graph
 * runner.set_graph(graph);
 *
 * // Allocate device memory
 * runner.allocate();
 *
 * // Set input data
 * runner.set_input("input", input_data.data(), input_data.size() * sizeof(float));
 *
 * // Execute all layers
 * auto result = runner.run();
 *
 * // Get output data
 * runner.get_output("output", output_data.data(), output_data.size() * sizeof(float));
 *
 * // Print per-layer timing
 * for (const auto& profile : result.layer_profiles) {
 *     std::cout << profile.name << ": " << profile.time_ms << " ms\n";
 * }
 * @endcode
 */
class GraphRunner {
public:
    // =========================================
    // Constructors
    // =========================================

    /**
     * @brief Construct runner with runtime
     * @param runtime Pointer to KPURuntime (must outlive runner)
     */
    explicit GraphRunner(KPURuntime* runtime);

    ~GraphRunner();

    // Non-copyable
    GraphRunner(const GraphRunner&) = delete;
    GraphRunner& operator=(const GraphRunner&) = delete;

    // Movable
    GraphRunner(GraphRunner&&) noexcept;
    GraphRunner& operator=(GraphRunner&&) noexcept;

    // =========================================
    // Graph Setup
    // =========================================

    /**
     * @brief Set the graph to execute
     * @param graph The computational graph (must be finalized)
     *
     * This copies the graph structure. Call allocate() after to allocate memory.
     */
    void set_graph(const ComputeGraph& graph);

    /**
     * @brief Allocate device memory for all tensors
     *
     * Uses memory planning to minimize peak memory usage by reusing
     * buffers for tensors with non-overlapping lifetimes.
     */
    void allocate();

    /**
     * @brief Check if memory has been allocated
     */
    bool is_allocated() const { return allocated_; }

    // =========================================
    // Input/Output
    // =========================================

    /**
     * @brief Set input tensor data
     * @param name Input tensor name
     * @param data Host data pointer
     * @param size_bytes Number of bytes to copy
     */
    void set_input(const std::string& name, const void* data, kpu::Size size_bytes);

    /**
     * @brief Set input tensor data with shape validation
     * @param name Input tensor name
     * @param data Host data pointer
     * @param shape Expected shape (for validation)
     */
    void set_input(const std::string& name, const void* data,
                   const std::vector<kpu::Size>& shape);

    /**
     * @brief Get output tensor data
     * @param name Output tensor name
     * @param data Host destination pointer
     * @param size_bytes Number of bytes to copy
     */
    void get_output(const std::string& name, void* data, kpu::Size size_bytes);

    /**
     * @brief Get output tensor data (copies full tensor)
     * @param name Output tensor name
     * @param data Host destination pointer
     */
    void get_output(const std::string& name, void* data);

    // =========================================
    // Execution
    // =========================================

    /**
     * @brief Execute the full graph
     * @return Execution result with per-layer profiling
     */
    GraphExecutionResult run();

    /**
     * @brief Execute a single node (for debugging)
     * @param node_index Node to execute
     * @return Execution result for this node
     */
    ExecutionResult run_node(size_t node_index);

    /**
     * @brief Get the last execution result
     */
    const GraphExecutionResult& last_result() const { return last_result_; }

    // =========================================
    // Memory Information
    // =========================================

    /**
     * @brief Get total allocated memory
     */
    kpu::Size total_allocated_bytes() const { return total_allocated_; }

    /**
     * @brief Get tensor device address
     * @param name Tensor name
     * @return Device address, or 0 if not allocated
     */
    kpu::Address get_tensor_address(const std::string& name) const;

    // =========================================
    // Cleanup
    // =========================================

    /**
     * @brief Free all allocated device memory
     */
    void release();

    /**
     * @brief Get the runtime
     */
    KPURuntime* runtime() const { return runtime_; }

    /**
     * @brief Get the graph
     */
    const ComputeGraph* graph() const { return graph_.get(); }

private:
    KPURuntime* runtime_;
    std::unique_ptr<ComputeGraph> graph_;

    // Tensor memory mapping
    std::unordered_map<std::string, kpu::Address> tensor_addresses_;
    std::vector<kpu::Address> allocated_buffers_;  // For cleanup
    kpu::Size total_allocated_ = 0;

    bool allocated_ = false;
    GraphExecutionResult last_result_;

    // Memory planning
    struct MemoryPlan {
        std::unordered_map<std::string, size_t> tensor_to_buffer;
        std::vector<kpu::Size> buffer_sizes;
    };
    MemoryPlan plan_memory();

    // Build kernel arguments for a node
    std::vector<kpu::Address> build_node_arguments(size_t node_index);
};

} // namespace sw::runtime
