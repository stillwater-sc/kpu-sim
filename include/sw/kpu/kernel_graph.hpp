#pragma once
// Kernel Graph - Multi-kernel DAG representation and compilation
// Enables execution of multiple kernels with dependencies

#include <sw/kpu/kernel.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <optional>

namespace sw::kpu {

/**
 * @brief Fusion strategy for combining multiple kernels
 */
enum class FusionStrategy : uint8_t {
    NONE = 0,           // No fusion, execute kernels separately
    PRODUCER_CONSUMER,  // Fuse producer output directly to consumer input
    HORIZONTAL,         // Fuse independent kernels for parallel execution
    PIPELINE            // Pipeline execution with overlapping data movement
};

/**
 * @brief Edge in the kernel graph representing data dependency
 */
struct KernelEdge {
    size_t from_node;           // Producer kernel index
    size_t to_node;             // Consumer kernel index
    std::string output_name;    // Output argument from producer (e.g., "C")
    std::string input_name;     // Input argument to consumer (e.g., "A")
    Size tensor_size_bytes;     // Size of transferred data

    KernelEdge() = default;
    KernelEdge(size_t from, size_t to,
               std::string out_name, std::string in_name,
               Size size = 0)
        : from_node(from), to_node(to)
        , output_name(std::move(out_name))
        , input_name(std::move(in_name))
        , tensor_size_bytes(size) {}
};

/**
 * @brief Node in the kernel graph
 */
struct KernelNode {
    size_t id;                              // Unique node ID
    std::unique_ptr<Kernel> kernel;         // The kernel
    std::string name;                       // Human-readable name
    std::vector<size_t> input_edges;        // Indices of incoming edges
    std::vector<size_t> output_edges;       // Indices of outgoing edges

    // Scheduling metadata
    int topological_order = -1;             // Order in execution sequence
    bool is_fused = false;                  // True if fused with another kernel
    size_t fused_with = SIZE_MAX;           // Node fused with (if applicable)

    KernelNode() = default;
    explicit KernelNode(size_t node_id, std::unique_ptr<Kernel> k,
                        std::string node_name = "")
        : id(node_id), kernel(std::move(k)), name(std::move(node_name)) {}

    // Move-only (kernels are heavyweight)
    KernelNode(KernelNode&&) = default;
    KernelNode& operator=(KernelNode&&) = default;
    KernelNode(const KernelNode&) = delete;
    KernelNode& operator=(const KernelNode&) = delete;
};

/**
 * @brief Statistics about a kernel graph
 */
struct KernelGraphStats {
    size_t num_nodes = 0;
    size_t num_edges = 0;
    size_t num_input_nodes = 0;     // Nodes with no incoming edges
    size_t num_output_nodes = 0;    // Nodes with no outgoing edges
    size_t max_depth = 0;           // Longest path in DAG
    size_t total_instructions = 0;
    Size total_flops = 0;
    Size total_input_bytes = 0;
    Size total_output_bytes = 0;
    Size intermediate_bytes = 0;    // Data passed between kernels
    double avg_arithmetic_intensity = 0.0;
};

/**
 * @brief Compilation options for kernel graphs
 */
struct KernelGraphCompileOptions {
    FusionStrategy fusion_strategy = FusionStrategy::PRODUCER_CONSUMER;
    bool enable_double_buffering = true;
    bool optimize_memory_allocation = true;
    bool insert_global_barriers = true;     // Barrier between unfused kernels
    Size workspace_limit = 0;               // 0 = unlimited
};

/**
 * @brief Result of kernel graph compilation
 */
struct KernelGraphCompileResult {
    isa::DMProgram program;                 // Compiled program
    std::vector<size_t> execution_order;    // Order of kernel execution
    std::vector<std::pair<size_t, size_t>> fused_pairs;  // Fused kernel pairs
    Size workspace_required = 0;            // Workspace memory needed
    bool success = false;
    std::string error_message;
};

/**
 * @brief Kernel Graph - DAG of kernels with data dependencies
 *
 * The KernelGraph class manages multiple kernels that form a directed
 * acyclic graph (DAG) where edges represent data dependencies between
 * kernel outputs and inputs.
 *
 * Features:
 * - Add kernels as nodes with unique IDs
 * - Connect kernels with typed edges (output_name -> input_name)
 * - Topological sort for valid execution order
 * - Kernel fusion optimization
 * - Compilation to single DMProgram
 *
 * Example usage:
 * @code
 * KernelGraph graph;
 *
 * // Add two matmul kernels
 * size_t k1 = graph.add_kernel(Kernel::create_matmul(M, N, K), "layer1");
 * size_t k2 = graph.add_kernel(Kernel::create_matmul(M, P, N), "layer2");
 *
 * // Connect: layer1.C -> layer2.A
 * graph.add_edge(k1, k2, "C", "A");
 *
 * // Compile to single program
 * auto result = graph.compile();
 * if (result.success) {
 *     executor.execute(result.program);
 * }
 * @endcode
 */
class KernelGraph {
public:
    // =========================================
    // Constructors
    // =========================================

    KernelGraph() = default;

    /**
     * @brief Create graph with a name
     */
    explicit KernelGraph(std::string name);

    // Move semantics (graphs can be large)
    KernelGraph(KernelGraph&&) = default;
    KernelGraph& operator=(KernelGraph&&) = default;

    // No copy (would require deep copying kernels)
    KernelGraph(const KernelGraph&) = delete;
    KernelGraph& operator=(const KernelGraph&) = delete;

    ~KernelGraph() = default;

    // =========================================
    // Node Management
    // =========================================

    /**
     * @brief Add a kernel to the graph
     * @param kernel The kernel to add (moved)
     * @param name Optional human-readable name
     * @return Node ID for the added kernel
     */
    size_t add_kernel(Kernel kernel, const std::string& name = "");

    /**
     * @brief Add a kernel by unique_ptr
     * @param kernel The kernel to add (moved)
     * @param name Optional human-readable name
     * @return Node ID for the added kernel
     */
    size_t add_kernel(std::unique_ptr<Kernel> kernel, const std::string& name = "");

    /**
     * @brief Get a kernel node by ID
     * @param node_id The node ID
     * @return Reference to the node
     * @throws std::out_of_range if node doesn't exist
     */
    const KernelNode& get_node(size_t node_id) const;
    KernelNode& get_node(size_t node_id);

    /**
     * @brief Get kernel by node ID
     */
    const Kernel& get_kernel(size_t node_id) const;
    Kernel& get_kernel(size_t node_id);

    /**
     * @brief Check if node exists
     */
    bool has_node(size_t node_id) const;

    /**
     * @brief Get number of nodes
     */
    size_t num_nodes() const { return nodes_.size(); }

    /**
     * @brief Get all node IDs
     */
    std::vector<size_t> node_ids() const;

    // =========================================
    // Edge Management
    // =========================================

    /**
     * @brief Add a data dependency edge between kernels
     * @param from_node Producer kernel node ID
     * @param to_node Consumer kernel node ID
     * @param output_name Output argument name from producer
     * @param input_name Input argument name to consumer
     * @return Edge ID
     * @throws std::invalid_argument if nodes don't exist or would create cycle
     */
    size_t add_edge(size_t from_node, size_t to_node,
                    const std::string& output_name = "C",
                    const std::string& input_name = "A");

    /**
     * @brief Get an edge by ID
     */
    const KernelEdge& get_edge(size_t edge_id) const;

    /**
     * @brief Check if adding an edge would create a cycle
     */
    bool would_create_cycle(size_t from_node, size_t to_node) const;

    /**
     * @brief Get number of edges
     */
    size_t num_edges() const { return edges_.size(); }

    /**
     * @brief Get edges from a node
     */
    std::vector<size_t> outgoing_edges(size_t node_id) const;

    /**
     * @brief Get edges to a node
     */
    std::vector<size_t> incoming_edges(size_t node_id) const;

    // =========================================
    // Graph Properties
    // =========================================

    /**
     * @brief Get graph name
     */
    const std::string& name() const { return name_; }

    /**
     * @brief Set graph name
     */
    void set_name(const std::string& name) { name_ = name; }

    /**
     * @brief Check if graph is empty
     */
    bool empty() const { return nodes_.empty(); }

    /**
     * @brief Check if graph is valid (is a DAG with valid connections)
     * @param error Output error message if invalid
     * @return true if valid
     */
    bool validate(std::string& error) const;

    /**
     * @brief Get input nodes (no incoming edges)
     */
    std::vector<size_t> input_nodes() const;

    /**
     * @brief Get output nodes (no outgoing edges)
     */
    std::vector<size_t> output_nodes() const;

    /**
     * @brief Compute graph statistics
     */
    KernelGraphStats compute_stats() const;

    // =========================================
    // Execution Order
    // =========================================

    /**
     * @brief Get topologically sorted execution order
     * @return Vector of node IDs in execution order
     * @throws std::runtime_error if graph has cycles
     *
     * Uses Kahn's algorithm for topological sort.
     */
    std::vector<size_t> get_execution_order() const;

    /**
     * @brief Get execution levels (nodes at same level can run in parallel)
     * @return Vector of vectors, each inner vector is a level
     *
     * Level 0 contains input nodes, Level 1 depends only on Level 0, etc.
     */
    std::vector<std::vector<size_t>> get_execution_levels() const;

    /**
     * @brief Get the critical path (longest path through graph)
     * @return Vector of node IDs on critical path
     */
    std::vector<size_t> get_critical_path() const;

    // =========================================
    // Fusion Optimization
    // =========================================

    /**
     * @brief Find kernels eligible for fusion
     * @return Vector of (producer, consumer) pairs that can be fused
     *
     * Two kernels can be fused if:
     * - They have a single edge connecting them
     * - The consumer has no other inputs from different nodes
     * - The output of producer matches input size of consumer
     * - Both use compatible data types
     */
    std::vector<std::pair<size_t, size_t>> find_fusible_pairs() const;

    /**
     * @brief Check if two kernels can be fused
     */
    bool can_fuse(size_t producer, size_t consumer) const;

    /**
     * @brief Mark two kernels for fusion
     * @return true if fusion was marked successfully
     */
    bool mark_for_fusion(size_t producer, size_t consumer);

    /**
     * @brief Clear all fusion marks
     */
    void clear_fusion_marks();

    // =========================================
    // Compilation
    // =========================================

    /**
     * @brief Compile the graph to a single DMProgram
     * @param options Compilation options
     * @return Compilation result with program and metadata
     */
    KernelGraphCompileResult compile(
        const KernelGraphCompileOptions& options = {}) const;

    /**
     * @brief Compile without fusion (simple concatenation)
     * @return Compilation result
     *
     * Simply concatenates kernel programs with barriers between them.
     * This is the simplest compilation strategy.
     */
    KernelGraphCompileResult compile_sequential() const;

    // =========================================
    // Iteration
    // =========================================

    /**
     * @brief Iterate over all nodes
     */
    template<typename Func>
    void for_each_node(Func&& func) const {
        for (const auto& [id, node] : nodes_) {
            func(node);
        }
    }

    /**
     * @brief Iterate over all edges
     */
    template<typename Func>
    void for_each_edge(Func&& func) const {
        for (const auto& edge : edges_) {
            func(edge);
        }
    }

    // =========================================
    // Debug and Visualization
    // =========================================

    /**
     * @brief Get human-readable summary
     */
    std::string summary() const;

    /**
     * @brief Export to DOT format for visualization
     * @param show_tensor_sizes Include tensor sizes on edges
     */
    std::string to_dot(bool show_tensor_sizes = true) const;

private:
    std::string name_;
    std::unordered_map<size_t, KernelNode> nodes_;
    std::vector<KernelEdge> edges_;
    size_t next_node_id_ = 0;

    // Cached execution order (invalidated on modifications)
    mutable std::optional<std::vector<size_t>> cached_execution_order_;

    /**
     * @brief Invalidate cached data after graph modification
     */
    void invalidate_cache();

    /**
     * @brief DFS helper for cycle detection
     */
    bool has_path_dfs(size_t from, size_t to,
                      std::unordered_set<size_t>& visited) const;

    /**
     * @brief Calculate depth of a node in the DAG
     */
    size_t calculate_node_depth(size_t node_id,
                                std::unordered_map<size_t, size_t>& depths) const;

    /**
     * @brief Compile a single kernel's program with memory offset
     */
    void append_kernel_program(isa::DMProgram& target,
                               const Kernel& kernel,
                               Address base_offset) const;

    /**
     * @brief Compile fused kernel pair
     */
    void compile_fused_pair(isa::DMProgram& target,
                            const KernelNode& producer,
                            const KernelNode& consumer,
                            Address base_offset) const;
};

} // namespace sw::kpu
