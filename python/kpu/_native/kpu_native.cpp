// python/kpu/_native/kpu_native.cpp
// pybind11 bindings for KPU simulator integration with the kpu Python package
//
// This module provides the native backend for the @kpu.compile decorator,
// enabling execution on the C++ kpu-sim library.
//
// v0.4.0: TRANSACTIONAL runtime integration
// v0.4.1: DFX parser integration

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <cmath>

// KPU Simulator includes for behavioral and transactional models
#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/fidelity/component_config.hpp>
#include <sw/kpu/models/interfaces/compute_fabric_interface.hpp>
#include <sw/kpu/models/interfaces/memory_controller_interface.hpp>
#include <sw/kpu/models/behavioral/compute/compute_fabric.hpp>
#include <sw/kpu/models/transactional/compute/compute_fabric.hpp>
#include <sw/kpu/models/transactional/memory/memory_controller.hpp>
#include <sw/kpu/stats/memory_traffic.hpp>

// v0.4.1: DFX parser for C++ DFX program representation
#include <sw/kpu/dfx/dfx_parser.hpp>

// v0.5.0: XUE Observation Architecture (C++ backend for event collection)
#include <sw/xue/event_collector.hpp>
#include <sw/xue/event_counter.hpp>
#include <sw/xue/operational_analysis.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

namespace {

// Fidelity levels matching Python constants
constexpr int FIDELITY_BEHAVIORAL = 0;
constexpr int FIDELITY_TRANSACTIONAL = 1;
constexpr int FIDELITY_CYCLE_ACCURATE = 2;

/**
 * @brief Per-level memory statistics (internal use only)
 *
 * Used internally by TRANSACTIONAL simulation. Python should use
 * kpu.get_xue_summary()['memory_hierarchy'] for per-level data.
 */
struct LevelMemoryStats {
    int64_t read_count = 0;
    int64_t write_count = 0;
    int64_t read_bytes = 0;
    int64_t write_bytes = 0;
    int64_t read_cycles = 0;
    int64_t write_cycles = 0;
    uint32_t transaction_size = 64;  // Default cache line size

    int64_t total_bytes() const { return read_bytes + write_bytes; }
    int64_t total_count() const { return read_count + write_count; }
    int64_t total_cycles() const { return read_cycles + write_cycles; }
};

/**
 * @brief Execution statistics returned to Python
 *
 * Extended for v0.4.0+ TRANSACTIONAL runtime with detailed metrics
 * from the C++ transactional simulation models.
 *
 * Note: Per-level memory hierarchy stats are available via kpu.get_xue_summary()
 * which provides detailed breakdowns from the C++ XUE EventCollector.
 */
struct NativeExecutionStats {
    // Basic timing
    int64_t cycles = 0;
    int64_t compute_cycles = 0;
    int64_t memory_cycles = 0;
    int64_t elapsed_cycles = 0;  // Wall clock cycles

    // Detailed cycle breakdown
    int64_t busy_cycles = 0;
    int64_t idle_cycles = 0;
    int64_t stall_cycles = 0;

    // Compute metrics
    int64_t matmul_flops = 0;
    int64_t total_macs = 0;
    int64_t matmul_count = 0;

    // Per-level memory stats (internal use by TRANSACTIONAL simulation)
    // Python should use kpu.get_xue_summary()['memory_hierarchy'] instead
    LevelMemoryStats dram;   // External/DRAM
    LevelMemoryStats l3;     // L3 buffer
    LevelMemoryStats l2;     // L2 buffer
    LevelMemoryStats l1;     // L1 stream

    // Memory metrics (aggregate, use get_xue_summary() for per-level breakdown)
    int64_t memory_bytes = 0;
    int64_t external_bytes = 0;

    // Memory controller stats (TRANSACTIONAL)
    int64_t memory_reads = 0;
    int64_t memory_writes = 0;
    int64_t page_hits = 0;
    int64_t page_misses = 0;
    int64_t memory_latency_cycles = 0;

    // Operation counts
    int64_t ops_executed = 0;

    // Clock frequency (must be set explicitly for TRANSACTIONAL mode)
    double clock_frequency_ghz = 0.0;

    // Performance metrics (computed using clock_frequency_ghz)
    double gflops = 0.0;
    double utilization = 0.0;
    double efficiency = 0.0;
    double memory_bandwidth_gbps = 0.0;
    double page_hit_rate = 0.0;

    py::dict to_dict() const {
        py::dict d;
        // Basic timing
        d["cycles"] = cycles;
        d["compute_cycles"] = compute_cycles;
        d["memory_cycles"] = memory_cycles;
        d["elapsed_cycles"] = elapsed_cycles;

        // Detailed breakdown
        d["busy_cycles"] = busy_cycles;
        d["idle_cycles"] = idle_cycles;
        d["stall_cycles"] = stall_cycles;

        // Compute metrics
        d["matmul_flops"] = matmul_flops;
        d["total_macs"] = total_macs;
        d["matmul_count"] = matmul_count;

        // Memory metrics
        d["memory_bytes"] = memory_bytes;
        d["external_bytes"] = external_bytes;

        // Memory controller stats
        d["memory_reads"] = memory_reads;
        d["memory_writes"] = memory_writes;
        d["page_hits"] = page_hits;
        d["page_misses"] = page_misses;
        d["memory_latency_cycles"] = memory_latency_cycles;

        // Operation counts
        d["ops_executed"] = ops_executed;

        // Clock frequency
        d["clock_frequency_ghz"] = clock_frequency_ghz;

        // Performance metrics (calculated using clock_frequency_ghz)
        d["gflops"] = gflops;
        d["utilization"] = utilization;
        d["efficiency"] = efficiency;
        d["memory_bandwidth_gbps"] = memory_bandwidth_gbps;
        d["page_hit_rate"] = page_hit_rate;

        return d;
    }
};

/**
 * @brief Native KPU runtime that executes DFX programs
 *
 * This class provides the interface between the Python kpu package
 * and execution on the KPU hardware model.
 *
 * Both BEHAVIORAL and TRANSACTIONAL modes use C++ compute fabrics:
 * - BEHAVIORAL: Uses C++ BehavioralComputeFabric for functional computation
 * - TRANSACTIONAL: Uses C++ TransactionalComputeFabric for computation + timing
 *
 * All computation is performed in C++ - no NumPy fallback for compute operations.
 * NumPy is only used for array allocation and memory management.
 *
 * IMPORTANT: For TRANSACTIONAL mode, clock_frequency_ghz must be explicitly
 * set before execution. This prevents silent use of incorrect defaults.
 */
class NativeKPURuntime {
public:
    // Sentinel value indicating clock frequency has not been set
    static constexpr double CLOCK_FREQUENCY_NOT_SET = -1.0;

    explicit NativeKPURuntime(int fidelity = FIDELITY_BEHAVIORAL)
        : fidelity_(fidelity)
        , clock_frequency_ghz_(CLOCK_FREQUENCY_NOT_SET)
        , clock_frequency_explicitly_set_(false) {
        // Initialize behavioral and transactional compute fabrics
        init_behavioral_models();
        init_transactional_models();
    }

    void init_behavioral_models() {
        // Configure behavioral compute fabric for functional correctness
        sw::kpu::ComputeFabricConfig compute_config;
        compute_config.fidelity = sw::kpu::SimulationFidelity::BEHAVIORAL;
        compute_config.array_rows = 16;
        compute_config.array_cols = 16;
        compute_config.macs_per_cycle = 256;
        compute_config.pipeline_depth = 1;  // Behavioral has no pipeline
        compute_config.enable_statistics = true;

        behavioral_compute_fabric_ = std::make_unique<sw::kpu::BehavioralComputeFabric>(compute_config, 0);
    }

    void init_transactional_models() {
        // Configure a 16x16 systolic array (256 MACs/cycle)
        sw::kpu::ComputeFabricConfig compute_config;
        compute_config.fidelity = sw::kpu::SimulationFidelity::TRANSACTIONAL;
        compute_config.array_rows = 16;
        compute_config.array_cols = 16;
        compute_config.macs_per_cycle = 256;
        compute_config.pipeline_depth = 4;
        compute_config.enable_statistics = true;

        compute_fabric_ = std::make_unique<sw::kpu::TransactionalComputeFabric>(compute_config, 0);

        // Configure memory controller (LPDDR5-like)
        sw::kpu::MemoryControllerConfig memory_config;
        memory_config.fidelity = sw::kpu::SimulationFidelity::TRANSACTIONAL;
        memory_config.technology = sw::kpu::MemoryTechnology::LPDDR5;
        memory_config.speed_mt_s = 6400;          // 6400 MT/s
        memory_config.capacity_gb = 1;
        memory_config.num_channels = 2;
        memory_config.banks_per_channel = 16;
        memory_config.queue_depth = 32;
        memory_config.enable_statistics = true;

        // Set transactional timing parameters
        memory_config.timing.mean_read_latency = 80;
        memory_config.timing.mean_write_latency = 90;
        memory_config.timing.latency_variance = 15;
        memory_config.timing.page_hit_factor = 0.6;
        memory_config.timing.page_conflict_factor = 1.4;

        memory_controller_ = std::make_unique<sw::kpu::TransactionalMemoryController>(memory_config);
    }

    void set_fidelity(int fidelity) {
        fidelity_ = fidelity;
    }

    int get_fidelity() const {
        return fidelity_;
    }

    /**
     * @brief Set the clock frequency for performance calculations
     *
     * This MUST be called before executing in TRANSACTIONAL or CYCLE_ACCURATE mode.
     * The clock frequency is used for:
     *   - GFLOPS calculation: GFLOPS = (FLOPs / cycles) * clock_ghz
     *   - Bandwidth calculation: GB/s = (bytes / cycles) * clock_ghz
     *
     * @param ghz Clock frequency in GHz (e.g., 1.0 for 1 GHz, 2.5 for 2.5 GHz)
     * @throws std::invalid_argument if ghz <= 0
     */
    void set_clock_frequency(double ghz) {
        if (ghz <= 0) {
            throw std::invalid_argument("Clock frequency must be positive (got " + std::to_string(ghz) + " GHz)");
        }
        clock_frequency_ghz_ = ghz;
        clock_frequency_explicitly_set_ = true;
    }

    double get_clock_frequency() const {
        return clock_frequency_ghz_;
    }

    bool is_clock_frequency_set() const {
        return clock_frequency_explicitly_set_;
    }

    /**
     * @brief Execute a DFX program
     *
     * @param dfx_json DFX program as Python dict (from DFXProgram.to_dict())
     * @param inputs List of numpy arrays for input tensors
     * @param mode Execution mode ("behavioral", "transactional", "cycle_accurate")
     * @return Tuple of (result numpy array, stats dict)
     *
     * @throws std::runtime_error if clock_frequency not set for TRANSACTIONAL mode
     */
    std::pair<py::array_t<float>, py::dict> execute(
        const py::dict& dfx_json,
        const std::vector<py::array_t<float>>& inputs,
        const std::string& mode = "behavioral"
    ) {
        NativeExecutionStats stats;

        // For behavioral mode, use C++ BehavioralComputeFabric for computation
        if (mode == "behavioral" || fidelity_ == FIDELITY_BEHAVIORAL) {
            return execute_behavioral(dfx_json, inputs, stats);
        }

        // For transactional/cycle-accurate mode, clock frequency MUST be set
        if (!clock_frequency_explicitly_set_) {
            throw std::runtime_error(
                "Clock frequency not set for " + mode + " mode. "
                "Call set_clock_frequency(ghz) before execution. "
                "Example: runtime.set_clock_frequency(1.0) for 1 GHz"
            );
        }

        // For transactional/cycle-accurate, use behavioral with timing estimates
        return execute_simulated(dfx_json, inputs, mode, stats);
    }

    /**
     * @brief Get runtime configuration info
     */
    py::dict get_config() const {
        py::dict config;
        config["fidelity"] = fidelity_;
        config["fidelity_name"] = fidelity_name();
        config["native_available"] = true;
        config["clock_frequency_ghz"] = clock_frequency_ghz_;
        config["clock_frequency_set"] = clock_frequency_explicitly_set_;
        return config;
    }

private:
    int fidelity_;
    double clock_frequency_ghz_;
    bool clock_frequency_explicitly_set_;
    std::unique_ptr<sw::kpu::BehavioralComputeFabric> behavioral_compute_fabric_;
    std::unique_ptr<sw::kpu::TransactionalComputeFabric> compute_fabric_;
    std::unique_ptr<sw::kpu::TransactionalMemoryController> memory_controller_;

    std::string fidelity_name() const {
        switch (fidelity_) {
            case FIDELITY_BEHAVIORAL: return "BEHAVIORAL";
            case FIDELITY_TRANSACTIONAL: return "TRANSACTIONAL";
            case FIDELITY_CYCLE_ACCURATE: return "CYCLE_ACCURATE";
            default: return "UNKNOWN";
        }
    }

    /**
     * @brief Execute using behavioral simulation (compute actual values)
     */
    std::pair<py::array_t<float>, py::dict> execute_behavioral(
        const py::dict& dfx_json,
        const std::vector<py::array_t<float>>& inputs,
        NativeExecutionStats& stats
    ) {
        // Parse DFX program
        auto ops = dfx_json["ops"].cast<py::list>();
        auto input_names = dfx_json["inputs"].cast<py::list>();
        auto output_names = dfx_json["outputs"].cast<py::list>();

        // Map tensor names to numpy arrays
        std::unordered_map<std::string, py::array_t<float>> tensors;

        // Load inputs
        for (size_t i = 0; i < inputs.size() && i < static_cast<size_t>(py::len(input_names)); ++i) {
            std::string name = input_names[i].cast<std::string>();
            tensors[name] = inputs[i];
        }

        // Execute operations in order
        for (auto op_obj : ops) {
            py::dict op = op_obj.cast<py::dict>();
            execute_op_behavioral(op, tensors, stats);
            stats.ops_executed++;
        }

        // Get output
        std::string output_name = output_names[0].cast<std::string>();
        auto result = tensors[output_name];

        return {result, stats.to_dict()};
    }

    /**
     * @brief Execute a single DFX operation using C++ BehavioralComputeFabric
     *
     * This uses the actual C++ simulation infrastructure for compute operations,
     * ensuring functional correctness is validated through the same code path
     * that will be used by higher-fidelity simulation modes.
     */
    void execute_op_behavioral(
        const py::dict& op,
        std::unordered_map<std::string, py::array_t<float>>& tensors,
        NativeExecutionStats& stats
    ) {
        std::string opcode = op["opcode"].cast<std::string>();
        auto input_names = op["inputs"].cast<py::list>();
        auto output_names = op["outputs"].cast<py::list>();

        std::string output_name = output_names[0].cast<std::string>();

        // Import numpy for array allocation
        py::module np = py::module::import("numpy");

        if (opcode == "matmul") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto A = tensors[a_name];
            auto B = tensors[b_name];

            py::buffer_info a_buf = A.request();
            py::buffer_info b_buf = B.request();

            // Get dimensions for FLOP counting
            uint32_t M = static_cast<uint32_t>(a_buf.shape[a_buf.ndim - 2]);
            uint32_t K = static_cast<uint32_t>(a_buf.shape[a_buf.ndim - 1]);
            uint32_t N = static_cast<uint32_t>(b_buf.shape[b_buf.ndim - 1]);

            // Allocate output array
            std::vector<py::ssize_t> out_shape;
            for (int i = 0; i < a_buf.ndim - 1; ++i) {
                out_shape.push_back(a_buf.shape[i]);
            }
            out_shape.push_back(b_buf.shape[b_buf.ndim - 1]);

            py::array_t<float> C = np.attr("zeros")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info c_buf = C.request();

            // Use C++ BehavioralComputeFabric for matmul
            sw::kpu::MatMulDescriptor desc;
            desc.m = M;
            desc.n = N;
            desc.k = K;

            behavioral_compute_fabric_->submit_matmul(
                desc,
                a_buf.ptr,
                b_buf.ptr,
                c_buf.ptr,
                nullptr
            );

            // Drain to complete the operation (behavioral is instant)
            behavioral_compute_fabric_->drain();

            tensors[output_name] = C;

            // Track FLOPs: 2*M*N*K (multiply-add per element)
            stats.matmul_flops += 2LL * M * N * K;

        } else if (opcode == "relu") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();
            size_t num_elements = static_cast<size_t>(x_buf.size);

            // Allocate output (same shape as input)
            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            // Use C++ BehavioralComputeFabric for elementwise ReLU
            sw::kpu::ElementwiseDescriptor desc;
            desc.op = sw::kpu::ElementwiseOp::RELU;
            desc.count = static_cast<uint32_t>(num_elements);

            behavioral_compute_fabric_->submit_elementwise(
                desc,
                x_buf.ptr,
                nullptr,  // No second operand for unary op
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "gelu") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();
            size_t num_elements = static_cast<size_t>(x_buf.size);

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            sw::kpu::ElementwiseDescriptor desc;
            desc.op = sw::kpu::ElementwiseOp::GELU;
            desc.count = static_cast<uint32_t>(num_elements);

            behavioral_compute_fabric_->submit_elementwise(
                desc,
                x_buf.ptr,
                nullptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "silu") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();
            size_t num_elements = static_cast<size_t>(x_buf.size);

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            sw::kpu::ElementwiseDescriptor desc;
            desc.op = sw::kpu::ElementwiseOp::SILU;
            desc.count = static_cast<uint32_t>(num_elements);

            behavioral_compute_fabric_->submit_elementwise(
                desc,
                x_buf.ptr,
                nullptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "sigmoid") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();
            size_t num_elements = static_cast<size_t>(x_buf.size);

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            sw::kpu::ElementwiseDescriptor desc;
            desc.op = sw::kpu::ElementwiseOp::SIGMOID;
            desc.count = static_cast<uint32_t>(num_elements);

            behavioral_compute_fabric_->submit_elementwise(
                desc,
                x_buf.ptr,
                nullptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "tanh") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();
            size_t num_elements = static_cast<size_t>(x_buf.size);

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            sw::kpu::ElementwiseDescriptor desc;
            desc.op = sw::kpu::ElementwiseOp::TANH;
            desc.count = static_cast<uint32_t>(num_elements);

            behavioral_compute_fabric_->submit_elementwise(
                desc,
                x_buf.ptr,
                nullptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "softmax") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();

            // Softmax along last axis
            // Shape: [..., dim_size]
            uint32_t dim_size = static_cast<uint32_t>(x_buf.shape[x_buf.ndim - 1]);
            uint32_t batch_size = 1;
            for (int i = 0; i < x_buf.ndim - 1; ++i) {
                batch_size *= static_cast<uint32_t>(x_buf.shape[i]);
            }

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            sw::kpu::SoftmaxDescriptor desc;
            desc.batch_size = batch_size;
            desc.dim_size = dim_size;
            desc.inner_size = 1;  // Softmax on last axis

            behavioral_compute_fabric_->submit_softmax(
                desc,
                x_buf.ptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "add") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto A = tensors[a_name];
            auto B = tensors[b_name];

            py::buffer_info a_buf = A.request();
            py::buffer_info b_buf = B.request();

            // Handle broadcasting - use numpy for shape calculation
            auto out_arr = np.attr("broadcast_arrays")(A, B);
            auto out_shape = out_arr.cast<py::list>()[0].attr("shape");
            py::array_t<float> Y = np.attr("empty")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            // For simple case (same shape), use C++ fabric
            if (a_buf.size == b_buf.size && a_buf.size == y_buf.size) {
                sw::kpu::ElementwiseDescriptor desc;
                desc.op = sw::kpu::ElementwiseOp::ADD;
                desc.count = static_cast<uint32_t>(a_buf.size);

                behavioral_compute_fabric_->submit_elementwise(
                    desc,
                    a_buf.ptr,
                    b_buf.ptr,
                    y_buf.ptr,
                    nullptr
                );
                behavioral_compute_fabric_->drain();
            } else {
                // TODO(fabric-broadcasting): C++ BehavioralComputeFabric doesn't support
                // broadcasting yet. When shapes differ, we fall back to NumPy.
                // To fix: Implement broadcast_elementwise() in compute_fabric.cpp that:
                // 1. Computes output shape via NumPy-style broadcasting rules
                // 2. Iterates with stride-aware indexing for mismatched dimensions
                // See: numpy broadcasting rules at numpy.org/doc/stable/user/basics.broadcasting.html
                Y = np.attr("add")(A, B).cast<py::array_t<float>>();
            }

            tensors[output_name] = Y;

        } else if (opcode == "sub") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto A = tensors[a_name];
            auto B = tensors[b_name];

            py::buffer_info a_buf = A.request();
            py::buffer_info b_buf = B.request();

            auto out_arr = np.attr("broadcast_arrays")(A, B);
            auto out_shape = out_arr.cast<py::list>()[0].attr("shape");
            py::array_t<float> Y = np.attr("empty")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            if (a_buf.size == b_buf.size && a_buf.size == y_buf.size) {
                sw::kpu::ElementwiseDescriptor desc;
                desc.op = sw::kpu::ElementwiseOp::SUB;
                desc.count = static_cast<uint32_t>(a_buf.size);

                behavioral_compute_fabric_->submit_elementwise(
                    desc,
                    a_buf.ptr,
                    b_buf.ptr,
                    y_buf.ptr,
                    nullptr
                );
                behavioral_compute_fabric_->drain();
            } else {
                // TODO(fabric-broadcasting): NumPy fallback for broadcasting - see add op
                Y = np.attr("subtract")(A, B).cast<py::array_t<float>>();
            }

            tensors[output_name] = Y;

        } else if (opcode == "mul") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto A = tensors[a_name];
            auto B = tensors[b_name];

            py::buffer_info a_buf = A.request();
            py::buffer_info b_buf = B.request();

            auto out_arr = np.attr("broadcast_arrays")(A, B);
            auto out_shape = out_arr.cast<py::list>()[0].attr("shape");
            py::array_t<float> Y = np.attr("empty")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            if (a_buf.size == b_buf.size && a_buf.size == y_buf.size) {
                sw::kpu::ElementwiseDescriptor desc;
                desc.op = sw::kpu::ElementwiseOp::MUL;
                desc.count = static_cast<uint32_t>(a_buf.size);

                behavioral_compute_fabric_->submit_elementwise(
                    desc,
                    a_buf.ptr,
                    b_buf.ptr,
                    y_buf.ptr,
                    nullptr
                );
                behavioral_compute_fabric_->drain();
            } else {
                // TODO(fabric-broadcasting): NumPy fallback for broadcasting - see add op
                Y = np.attr("multiply")(A, B).cast<py::array_t<float>>();
            }

            tensors[output_name] = Y;

        } else if (opcode == "div") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto A = tensors[a_name];
            auto B = tensors[b_name];

            py::buffer_info a_buf = A.request();
            py::buffer_info b_buf = B.request();

            auto out_arr = np.attr("broadcast_arrays")(A, B);
            auto out_shape = out_arr.cast<py::list>()[0].attr("shape");
            py::array_t<float> Y = np.attr("empty")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            if (a_buf.size == b_buf.size && a_buf.size == y_buf.size) {
                sw::kpu::ElementwiseDescriptor desc;
                desc.op = sw::kpu::ElementwiseOp::DIV;
                desc.count = static_cast<uint32_t>(a_buf.size);

                behavioral_compute_fabric_->submit_elementwise(
                    desc,
                    a_buf.ptr,
                    b_buf.ptr,
                    y_buf.ptr,
                    nullptr
                );
                behavioral_compute_fabric_->drain();
            } else {
                // TODO(fabric-broadcasting): NumPy fallback for broadcasting - see add op
                Y = np.attr("divide")(A, B).cast<py::array_t<float>>();
            }

            tensors[output_name] = Y;

        } else if (opcode == "neg") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();
            size_t num_elements = static_cast<size_t>(x_buf.size);

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            sw::kpu::ElementwiseDescriptor desc;
            desc.op = sw::kpu::ElementwiseOp::NEG;
            desc.count = static_cast<uint32_t>(num_elements);

            behavioral_compute_fabric_->submit_elementwise(
                desc,
                x_buf.ptr,
                nullptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "exp") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();
            size_t num_elements = static_cast<size_t>(x_buf.size);

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            sw::kpu::ElementwiseDescriptor desc;
            desc.op = sw::kpu::ElementwiseOp::EXP;
            desc.count = static_cast<uint32_t>(num_elements);

            behavioral_compute_fabric_->submit_elementwise(
                desc,
                x_buf.ptr,
                nullptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "log") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();
            size_t num_elements = static_cast<size_t>(x_buf.size);

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            sw::kpu::ElementwiseDescriptor desc;
            desc.op = sw::kpu::ElementwiseOp::LOG;
            desc.count = static_cast<uint32_t>(num_elements);

            behavioral_compute_fabric_->submit_elementwise(
                desc,
                x_buf.ptr,
                nullptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "sqrt") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::buffer_info x_buf = X.request();
            size_t num_elements = static_cast<size_t>(x_buf.size);

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            sw::kpu::ElementwiseDescriptor desc;
            desc.op = sw::kpu::ElementwiseOp::SQRT;
            desc.count = static_cast<uint32_t>(num_elements);

            behavioral_compute_fabric_->submit_elementwise(
                desc,
                x_buf.ptr,
                nullptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "conv2d") {
            // Conv2D using C++ BehavioralComputeFabric
            std::string x_name = input_names[0].cast<std::string>();
            std::string w_name = input_names[1].cast<std::string>();

            auto X = tensors[x_name];
            auto W = tensors[w_name];

            py::buffer_info x_buf = X.request();
            py::buffer_info w_buf = W.request();

            // Extract dimensions
            uint32_t batch_size = static_cast<uint32_t>(x_buf.shape[0]);
            uint32_t in_channels = static_cast<uint32_t>(x_buf.shape[1]);
            uint32_t input_h = static_cast<uint32_t>(x_buf.shape[2]);
            uint32_t input_w = static_cast<uint32_t>(x_buf.shape[3]);

            uint32_t out_channels = static_cast<uint32_t>(w_buf.shape[0]);
            uint32_t kernel_h = static_cast<uint32_t>(w_buf.shape[2]);
            uint32_t kernel_w = static_cast<uint32_t>(w_buf.shape[3]);

            // Get parameters from attrs
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            uint32_t stride_h = 1, stride_w = 1;
            uint32_t pad_h = 0, pad_w = 0;
            uint32_t dilation_h = 1, dilation_w = 1;
            uint32_t groups = 1;

            if (attrs.contains("stride")) {
                auto stride = attrs["stride"];
                if (py::isinstance<py::tuple>(stride)) {
                    auto s = stride.cast<py::tuple>();
                    stride_h = s[0].cast<uint32_t>();
                    stride_w = s[1].cast<uint32_t>();
                } else {
                    stride_h = stride_w = stride.cast<uint32_t>();
                }
            }
            if (attrs.contains("padding")) {
                auto padding = attrs["padding"];
                if (py::isinstance<py::tuple>(padding)) {
                    auto p = padding.cast<py::tuple>();
                    pad_h = p[0].cast<uint32_t>();
                    pad_w = p[1].cast<uint32_t>();
                } else {
                    pad_h = pad_w = padding.cast<uint32_t>();
                }
            }
            if (attrs.contains("dilation")) {
                auto dilation = attrs["dilation"];
                if (py::isinstance<py::tuple>(dilation)) {
                    auto d = dilation.cast<py::tuple>();
                    dilation_h = d[0].cast<uint32_t>();
                    dilation_w = d[1].cast<uint32_t>();
                } else {
                    dilation_h = dilation_w = dilation.cast<uint32_t>();
                }
            }
            if (attrs.contains("groups")) {
                groups = attrs["groups"].cast<uint32_t>();
            }

            // Compute output dimensions
            uint32_t output_h = (input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
            uint32_t output_w = (input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

            // Create output array
            std::vector<py::ssize_t> out_shape = {
                static_cast<py::ssize_t>(batch_size),
                static_cast<py::ssize_t>(out_channels),
                static_cast<py::ssize_t>(output_h),
                static_cast<py::ssize_t>(output_w)
            };
            py::array_t<float> result = np.attr("zeros")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info r_buf = result.request();

            // Get bias pointer if present
            const float* bias_ptr = nullptr;
            if (input_names.size() > 2) {
                std::string bias_name = input_names[2].cast<std::string>();
                if (tensors.count(bias_name) > 0) {
                    bias_ptr = static_cast<const float*>(tensors[bias_name].request().ptr);
                }
            }

            // Build Conv2D descriptor
            sw::kpu::Conv2DDescriptor desc;
            desc.batch_size = batch_size;
            desc.in_channels = in_channels;
            desc.in_height = input_h;
            desc.in_width = input_w;
            desc.out_channels = out_channels;
            desc.kernel_height = kernel_h;
            desc.kernel_width = kernel_w;
            desc.stride_h = stride_h;
            desc.stride_w = stride_w;
            desc.padding_h = pad_h;
            desc.padding_w = pad_w;
            desc.dilation_h = dilation_h;
            desc.dilation_w = dilation_w;
            desc.groups = groups;

            // Submit to C++ BehavioralComputeFabric
            behavioral_compute_fabric_->submit_conv2d(
                desc,
                x_buf.ptr,
                w_buf.ptr,
                bias_ptr,
                r_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = result;

            // Track conv2d FLOPs
            int64_t gemm_M = static_cast<int64_t>(batch_size) * output_h * output_w;
            int64_t gemm_K = static_cast<int64_t>(in_channels) / groups * kernel_h * kernel_w;
            int64_t gemm_N = out_channels;
            stats.matmul_flops += 2LL * gemm_M * gemm_N * gemm_K;

        } else if (opcode == "max_pool2d" || opcode == "maxpool2d" || opcode == "avg_pool2d" || opcode == "avgpool2d") {
            // Pool2D using C++ BehavioralComputeFabric
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors[x_name];

            py::buffer_info x_buf = X.request();

            uint32_t batch_size = static_cast<uint32_t>(x_buf.shape[0]);
            uint32_t channels = static_cast<uint32_t>(x_buf.shape[1]);
            uint32_t input_h = static_cast<uint32_t>(x_buf.shape[2]);
            uint32_t input_w = static_cast<uint32_t>(x_buf.shape[3]);

            // Get parameters from attrs
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            uint32_t kernel_h = 2, kernel_w = 2;
            uint32_t stride_h = 2, stride_w = 2;
            uint32_t pad_h = 0, pad_w = 0;

            if (attrs.contains("kernel_size")) {
                auto ks = attrs["kernel_size"];
                if (py::isinstance<py::tuple>(ks)) {
                    auto k = ks.cast<py::tuple>();
                    kernel_h = k[0].cast<uint32_t>();
                    kernel_w = k[1].cast<uint32_t>();
                } else {
                    kernel_h = kernel_w = ks.cast<uint32_t>();
                }
            }
            if (attrs.contains("stride")) {
                auto stride = attrs["stride"];
                if (py::isinstance<py::tuple>(stride)) {
                    auto s = stride.cast<py::tuple>();
                    stride_h = s[0].cast<uint32_t>();
                    stride_w = s[1].cast<uint32_t>();
                } else {
                    stride_h = stride_w = stride.cast<uint32_t>();
                }
            }
            if (attrs.contains("padding")) {
                auto padding = attrs["padding"];
                if (py::isinstance<py::tuple>(padding)) {
                    auto p = padding.cast<py::tuple>();
                    pad_h = p[0].cast<uint32_t>();
                    pad_w = p[1].cast<uint32_t>();
                } else {
                    pad_h = pad_w = padding.cast<uint32_t>();
                }
            }

            // Compute output dimensions
            uint32_t output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
            uint32_t output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

            // Create output array
            std::vector<py::ssize_t> out_shape = {
                static_cast<py::ssize_t>(batch_size),
                static_cast<py::ssize_t>(channels),
                static_cast<py::ssize_t>(output_h),
                static_cast<py::ssize_t>(output_w)
            };
            py::array_t<float> result = np.attr("zeros")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info r_buf = result.request();

            sw::kpu::Pool2DDescriptor desc;
            desc.pool_type = (opcode == "max_pool2d" || opcode == "maxpool2d") ? sw::kpu::Pool2DDescriptor::PoolType::MAX : sw::kpu::Pool2DDescriptor::PoolType::AVG;
            desc.batch_size = batch_size;
            desc.channels = channels;
            desc.in_height = input_h;
            desc.in_width = input_w;
            desc.kernel_height = kernel_h;
            desc.kernel_width = kernel_w;
            desc.stride_h = stride_h;
            desc.stride_w = stride_w;
            desc.padding_h = pad_h;
            desc.padding_w = pad_w;

            behavioral_compute_fabric_->submit_pool2d(
                desc,
                x_buf.ptr,
                r_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = result;

        } else if (opcode == "adaptive_avg_pool2d") {
            // Adaptive average pooling
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors[x_name];

            py::buffer_info x_buf = X.request();

            uint32_t batch_size = static_cast<uint32_t>(x_buf.shape[0]);
            uint32_t channels = static_cast<uint32_t>(x_buf.shape[1]);
            uint32_t input_h = static_cast<uint32_t>(x_buf.shape[2]);
            uint32_t input_w = static_cast<uint32_t>(x_buf.shape[3]);

            // Get target output size from attrs
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            uint32_t output_h = 1, output_w = 1;

            if (attrs.contains("output_size")) {
                auto os = attrs["output_size"];
                if (py::isinstance<py::tuple>(os)) {
                    auto o = os.cast<py::tuple>();
                    output_h = o[0].cast<uint32_t>();
                    output_w = o[1].cast<uint32_t>();
                } else {
                    output_h = output_w = os.cast<uint32_t>();
                }
            }

            // Create output array
            std::vector<py::ssize_t> out_shape = {
                static_cast<py::ssize_t>(batch_size),
                static_cast<py::ssize_t>(channels),
                static_cast<py::ssize_t>(output_h),
                static_cast<py::ssize_t>(output_w)
            };
            py::array_t<float> result = np.attr("zeros")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info r_buf = result.request();

            sw::kpu::Pool2DDescriptor desc;
            desc.pool_type = sw::kpu::Pool2DDescriptor::PoolType::ADAPTIVE_AVG;
            desc.batch_size = batch_size;
            desc.channels = channels;
            desc.in_height = input_h;
            desc.in_width = input_w;
            desc.target_out_height = output_h;
            desc.target_out_width = output_w;
            // Kernel/stride/padding will be computed by the behavioral fabric

            behavioral_compute_fabric_->submit_pool2d(
                desc,
                x_buf.ptr,
                r_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = result;

        } else if (opcode == "layer_norm" || opcode == "layernorm") {
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors[x_name];

            py::buffer_info x_buf = X.request();

            // Get weight and bias if present
            const float* weight_ptr = nullptr;
            const float* bias_ptr = nullptr;

            if (input_names.size() > 1) {
                std::string w_name = input_names[1].cast<std::string>();
                if (tensors.count(w_name) > 0) {
                    weight_ptr = static_cast<const float*>(tensors[w_name].request().ptr);
                }
            }
            if (input_names.size() > 2) {
                std::string b_name = input_names[2].cast<std::string>();
                if (tensors.count(b_name) > 0) {
                    bias_ptr = static_cast<const float*>(tensors[b_name].request().ptr);
                }
            }

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            // Normalize over the last dimension
            uint32_t normalized_size = static_cast<uint32_t>(x_buf.shape[x_buf.ndim - 1]);
            uint32_t batch_size = 1;
            for (int i = 0; i < x_buf.ndim - 1; ++i) {
                batch_size *= static_cast<uint32_t>(x_buf.shape[i]);
            }

            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            float eps = attrs.contains("eps") ? attrs["eps"].cast<float>() : 1e-5f;

            sw::kpu::LayerNormDescriptor desc;
            desc.batch_size = batch_size;
            desc.normalized_size = normalized_size;
            desc.eps = eps;

            behavioral_compute_fabric_->submit_layernorm(
                desc,
                x_buf.ptr,
                weight_ptr,
                bias_ptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "batch_norm" || opcode == "batchnorm") {
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors[x_name];

            py::buffer_info x_buf = X.request();

            // BatchNorm params: weight, bias, running_mean, running_var
            const float* weight_ptr = nullptr;
            const float* bias_ptr = nullptr;
            const float* running_mean_ptr = nullptr;
            const float* running_var_ptr = nullptr;

            if (input_names.size() > 1) {
                std::string w_name = input_names[1].cast<std::string>();
                if (tensors.count(w_name) > 0) {
                    weight_ptr = static_cast<const float*>(tensors[w_name].request().ptr);
                }
            }
            if (input_names.size() > 2) {
                std::string b_name = input_names[2].cast<std::string>();
                if (tensors.count(b_name) > 0) {
                    bias_ptr = static_cast<const float*>(tensors[b_name].request().ptr);
                }
            }
            if (input_names.size() > 3) {
                std::string rm_name = input_names[3].cast<std::string>();
                if (tensors.count(rm_name) > 0) {
                    running_mean_ptr = static_cast<const float*>(tensors[rm_name].request().ptr);
                }
            }
            if (input_names.size() > 4) {
                std::string rv_name = input_names[4].cast<std::string>();
                if (tensors.count(rv_name) > 0) {
                    running_var_ptr = static_cast<const float*>(tensors[rv_name].request().ptr);
                }
            }

            py::array_t<float> Y = np.attr("empty_like")(X).cast<py::array_t<float>>();
            py::buffer_info y_buf = Y.request();

            // Input is [N, C, H, W]
            uint32_t batch_size = static_cast<uint32_t>(x_buf.shape[0]);
            uint32_t num_features = static_cast<uint32_t>(x_buf.shape[1]);
            uint32_t spatial_size = 1;
            for (int i = 2; i < x_buf.ndim; ++i) {
                spatial_size *= static_cast<uint32_t>(x_buf.shape[i]);
            }

            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            float eps = attrs.contains("eps") ? attrs["eps"].cast<float>() : 1e-5f;
            float momentum = attrs.contains("momentum") ? attrs["momentum"].cast<float>() : 0.1f;

            sw::kpu::BatchNormDescriptor desc;
            desc.batch_size = batch_size;
            desc.num_features = num_features;
            desc.spatial_size = spatial_size;
            desc.eps = eps;
            desc.momentum = momentum;
            desc.training = false;  // Inference mode

            behavioral_compute_fabric_->submit_batchnorm(
                desc,
                x_buf.ptr,
                weight_ptr,
                bias_ptr,
                running_mean_ptr,
                running_var_ptr,
                y_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "fused_matmul_bias_relu" || opcode == "fused_matmul_relu") {
            // Fused matmul + optional bias + relu
            bool has_bias = (opcode == "fused_matmul_bias_relu");

            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();
            auto A = tensors.at(a_name);
            auto B = tensors.at(b_name);

            py::array_t<float> bias;
            if (has_bias && input_names.size() > 2) {
                std::string bias_name = input_names[2].cast<std::string>();
                bias = tensors.at(bias_name);
            }

            // Get shapes
            auto a_info = A.request();
            auto b_info = B.request();

            uint32_t M = static_cast<uint32_t>(a_info.shape[0]);
            uint32_t K = static_cast<uint32_t>(a_info.shape[1]);
            uint32_t N = static_cast<uint32_t>(b_info.shape[1]);

            // Allocate output and temp
            py::array_t<float> temp({M, N});
            py::array_t<float> Y({M, N});
            auto temp_buf = temp.request();
            auto y_buf = Y.request();

            // Step 1: MatMul
            sw::kpu::MatMulDescriptor matmul_desc;
            matmul_desc.m = M;
            matmul_desc.n = N;
            matmul_desc.k = K;

            behavioral_compute_fabric_->submit_matmul(
                matmul_desc,
                static_cast<const float*>(a_info.ptr),
                static_cast<const float*>(b_info.ptr),
                static_cast<float*>(temp_buf.ptr),
                nullptr
            );
            behavioral_compute_fabric_->drain();

            // Step 2: Add bias if present
            float* activation_input = static_cast<float*>(temp_buf.ptr);
            if (has_bias && input_names.size() > 2) {
                auto bias_buf = bias.request();
                sw::kpu::ElementwiseDescriptor add_desc;
                add_desc.op = sw::kpu::ElementwiseOp::ADD;
                add_desc.count = M * N;

                // Broadcast add (temp + bias -> temp)
                // For simplicity, do element-wise with broadcasting in-place
                float* temp_ptr = static_cast<float*>(temp_buf.ptr);
                const float* bias_ptr = static_cast<const float*>(bias_buf.ptr);
                for (uint32_t i = 0; i < M; ++i) {
                    for (uint32_t j = 0; j < N; ++j) {
                        temp_ptr[i * N + j] += bias_ptr[j];
                    }
                }

                // Record XUE event for bias add
                sw::xue::xue().record_add(M * N, 0);
            }

            // Step 3: ReLU
            sw::kpu::ElementwiseDescriptor relu_desc;
            relu_desc.op = sw::kpu::ElementwiseOp::RELU;
            relu_desc.count = M * N;

            behavioral_compute_fabric_->submit_elementwise(
                relu_desc,
                activation_input,
                nullptr,
                static_cast<float*>(y_buf.ptr),
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "fused_matmul_bias_gelu") {
            // Fused matmul + bias + gelu
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();
            auto A = tensors.at(a_name);
            auto B = tensors.at(b_name);
            py::array_t<float> bias;
            if (input_names.size() > 2) {
                std::string bias_name = input_names[2].cast<std::string>();
                bias = tensors.at(bias_name);
            }

            auto a_info = A.request();
            auto b_info = B.request();

            uint32_t M = static_cast<uint32_t>(a_info.shape[0]);
            uint32_t K = static_cast<uint32_t>(a_info.shape[1]);
            uint32_t N = static_cast<uint32_t>(b_info.shape[1]);

            py::array_t<float> temp({M, N});
            py::array_t<float> Y({M, N});
            auto temp_buf = temp.request();
            auto y_buf = Y.request();

            // MatMul
            sw::kpu::MatMulDescriptor matmul_desc;
            matmul_desc.m = M;
            matmul_desc.n = N;
            matmul_desc.k = K;

            behavioral_compute_fabric_->submit_matmul(
                matmul_desc,
                static_cast<const float*>(a_info.ptr),
                static_cast<const float*>(b_info.ptr),
                static_cast<float*>(temp_buf.ptr),
                nullptr
            );
            behavioral_compute_fabric_->drain();

            // Add bias if present
            if (input_names.size() > 2) {
                auto bias_buf = bias.request();
                float* temp_ptr = static_cast<float*>(temp_buf.ptr);
                const float* bias_ptr = static_cast<const float*>(bias_buf.ptr);
                for (uint32_t i = 0; i < M; ++i) {
                    for (uint32_t j = 0; j < N; ++j) {
                        temp_ptr[i * N + j] += bias_ptr[j];
                    }
                }
                sw::xue::xue().record_add(M * N, 0);
            }

            // GELU
            sw::kpu::ElementwiseDescriptor gelu_desc;
            gelu_desc.op = sw::kpu::ElementwiseOp::GELU;
            gelu_desc.count = M * N;

            behavioral_compute_fabric_->submit_elementwise(
                gelu_desc,
                static_cast<float*>(temp_buf.ptr),
                nullptr,
                static_cast<float*>(y_buf.ptr),
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        } else if (opcode == "fused_matmul_bias_silu") {
            // Fused matmul + bias + silu
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();
            auto A = tensors.at(a_name);
            auto B = tensors.at(b_name);
            py::array_t<float> bias;
            if (input_names.size() > 2) {
                std::string bias_name = input_names[2].cast<std::string>();
                bias = tensors.at(bias_name);
            }

            auto a_info = A.request();
            auto b_info = B.request();

            uint32_t M = static_cast<uint32_t>(a_info.shape[0]);
            uint32_t K = static_cast<uint32_t>(a_info.shape[1]);
            uint32_t N = static_cast<uint32_t>(b_info.shape[1]);

            py::array_t<float> temp({M, N});
            py::array_t<float> Y({M, N});
            auto temp_buf = temp.request();
            auto y_buf = Y.request();

            // MatMul
            sw::kpu::MatMulDescriptor matmul_desc;
            matmul_desc.m = M;
            matmul_desc.n = N;
            matmul_desc.k = K;

            behavioral_compute_fabric_->submit_matmul(
                matmul_desc,
                static_cast<const float*>(a_info.ptr),
                static_cast<const float*>(b_info.ptr),
                static_cast<float*>(temp_buf.ptr),
                nullptr
            );
            behavioral_compute_fabric_->drain();

            // Add bias if present
            if (input_names.size() > 2) {
                auto bias_buf = bias.request();
                float* temp_ptr = static_cast<float*>(temp_buf.ptr);
                const float* bias_ptr = static_cast<const float*>(bias_buf.ptr);
                for (uint32_t i = 0; i < M; ++i) {
                    for (uint32_t j = 0; j < N; ++j) {
                        temp_ptr[i * N + j] += bias_ptr[j];
                    }
                }
                sw::xue::xue().record_add(M * N, 0);
            }

            // SILU
            sw::kpu::ElementwiseDescriptor silu_desc;
            silu_desc.op = sw::kpu::ElementwiseOp::SILU;
            silu_desc.count = M * N;

            behavioral_compute_fabric_->submit_elementwise(
                silu_desc,
                static_cast<float*>(temp_buf.ptr),
                nullptr,
                static_cast<float*>(y_buf.ptr),
                nullptr
            );
            behavioral_compute_fabric_->drain();

            tensors[output_name] = Y;

        // =====================================================================
        // Shape Operations (v0.3+ benchmarks)
        // These are data reorganization ops - no compute, just memory movement
        // =====================================================================
        } else if (opcode == "reshape") {
            // Reshape: change tensor shape without copying data
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors.at(x_name);
            py::buffer_info x_buf = X.request();

            // Get target shape from attrs
            auto shape_list = attrs["shape"].cast<py::list>();
            std::vector<py::ssize_t> new_shape;
            py::ssize_t total_size = 1;
            py::ssize_t neg_one_idx = -1;

            for (size_t i = 0; i < py::len(shape_list); ++i) {
                py::ssize_t dim = shape_list[i].cast<py::ssize_t>();
                new_shape.push_back(dim);
                if (dim == -1) {
                    neg_one_idx = static_cast<py::ssize_t>(i);
                } else {
                    total_size *= dim;
                }
            }

            // Handle -1 dimension
            if (neg_one_idx >= 0 && total_size > 0) {
                new_shape[static_cast<size_t>(neg_one_idx)] = x_buf.size / total_size;
            }

            // Create reshaped view (no data copy for contiguous tensors)
            py::array_t<float> Y(new_shape);
            std::memcpy(Y.mutable_data(), X.data(), static_cast<size_t>(x_buf.size) * sizeof(float));

            // Record XUE data movement event (reshape is memory reorganization)
            sw::xue::xue().record_dma_transfer(
                static_cast<uint64_t>(x_buf.size) * sizeof(float),
                0  // No cycles for behavioral
            );

            tensors[output_name] = Y;
            stats.ops_executed++;

        } else if (opcode == "transpose") {
            // Transpose: permute tensor dimensions
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors.at(x_name);
            py::buffer_info x_buf = X.request();

            // Get axes from attrs
            auto axes_list = attrs["axes"].cast<py::list>();
            std::vector<size_t> axes;
            for (size_t i = 0; i < py::len(axes_list); ++i) {
                axes.push_back(axes_list[i].cast<size_t>());
            }

            // Build new shape and strides
            std::vector<py::ssize_t> new_shape(x_buf.ndim);
            for (size_t i = 0; i < axes.size(); ++i) {
                new_shape[i] = x_buf.shape[axes[i]];
            }

            // Allocate output and perform transpose
            py::array_t<float> Y(new_shape);
            float* y_ptr = Y.mutable_data();
            const float* x_ptr = static_cast<const float*>(x_buf.ptr);

            // Compute strides for input
            std::vector<py::ssize_t> x_strides(x_buf.ndim);
            py::ssize_t stride = 1;
            for (int i = x_buf.ndim - 1; i >= 0; --i) {
                x_strides[i] = stride;
                stride *= x_buf.shape[i];
            }

            // Compute strides for output
            std::vector<py::ssize_t> y_strides(x_buf.ndim);
            stride = 1;
            for (int i = x_buf.ndim - 1; i >= 0; --i) {
                y_strides[i] = stride;
                stride *= new_shape[i];
            }

            // Transpose via index computation
            std::vector<py::ssize_t> coords(x_buf.ndim, 0);
            for (py::ssize_t i = 0; i < x_buf.size; ++i) {
                // Compute output index
                py::ssize_t y_idx = 0;
                for (int d = 0; d < x_buf.ndim; ++d) {
                    y_idx += coords[axes[d]] * y_strides[d];
                }

                // Compute input index
                py::ssize_t x_idx = 0;
                for (int d = 0; d < x_buf.ndim; ++d) {
                    x_idx += coords[d] * x_strides[d];
                }

                y_ptr[y_idx] = x_ptr[x_idx];

                // Increment coordinates
                for (int d = x_buf.ndim - 1; d >= 0; --d) {
                    coords[d]++;
                    if (coords[d] < x_buf.shape[d]) break;
                    coords[d] = 0;
                }
            }

            // Record XUE data movement event
            sw::xue::xue().record_dma_transfer(
                static_cast<uint64_t>(x_buf.size) * sizeof(float),
                0
            );

            tensors[output_name] = Y;
            stats.ops_executed++;

        } else if (opcode == "flatten") {
            // Flatten: collapse dimensions into a single dimension
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors.at(x_name);
            py::buffer_info x_buf = X.request();

            int start_dim = attrs.contains("start_dim") ?
                attrs["start_dim"].cast<int>() : 0;
            int end_dim = attrs.contains("end_dim") ?
                attrs["end_dim"].cast<int>() : -1;

            // Normalize negative indices
            int ndim = x_buf.ndim;
            if (start_dim < 0) start_dim = ndim + start_dim;
            if (end_dim < 0) end_dim = ndim + end_dim;

            // Build new shape
            std::vector<py::ssize_t> new_shape;
            for (int d = 0; d < start_dim; ++d) {
                new_shape.push_back(x_buf.shape[d]);
            }
            py::ssize_t flat_size = 1;
            for (int d = start_dim; d <= end_dim; ++d) {
                flat_size *= x_buf.shape[d];
            }
            new_shape.push_back(flat_size);
            for (int d = end_dim + 1; d < ndim; ++d) {
                new_shape.push_back(x_buf.shape[d]);
            }

            // Reshape (no data copy for contiguous tensors)
            py::array_t<float> Y(new_shape);
            std::memcpy(Y.mutable_data(), X.data(), static_cast<size_t>(x_buf.size) * sizeof(float));

            // Record XUE data movement event
            sw::xue::xue().record_dma_transfer(
                static_cast<uint64_t>(x_buf.size) * sizeof(float),
                0
            );

            tensors[output_name] = Y;
            stats.ops_executed++;

        } else if (opcode == "concat") {
            // Concatenate: join tensors along a dimension
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            int dim = attrs.contains("dim") ? attrs["dim"].cast<int>() : 0;

            // Gather all input tensors
            std::vector<py::array_t<float>> input_arrays;
            size_t total_bytes = 0;
            for (size_t i = 0; i < py::len(input_names); ++i) {
                std::string name = input_names[i].cast<std::string>();
                input_arrays.push_back(tensors.at(name));
                py::buffer_info buf = input_arrays.back().request();
                total_bytes += static_cast<size_t>(buf.size) * sizeof(float);
            }

            if (input_arrays.empty()) {
                throw std::runtime_error("concat requires at least one input");
            }

            // Get shape info from first input
            py::buffer_info first_buf = input_arrays[0].request();
            int ndim = first_buf.ndim;
            if (dim < 0) dim = ndim + dim;

            // Compute output shape - copy from first input's shape
            std::vector<py::ssize_t> out_shape(first_buf.shape.begin(), first_buf.shape.end());
            out_shape[dim] = 0;
            for (const auto& arr : input_arrays) {
                py::buffer_info buf = arr.request();
                out_shape[dim] += buf.shape[dim];
            }

            // Allocate output
            py::array_t<float> Y(out_shape);
            float* y_ptr = Y.mutable_data();

            // Compute strides
            std::vector<py::ssize_t> strides(ndim);
            py::ssize_t stride = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                strides[d] = stride;
                stride *= out_shape[d];
            }

            // Copy each input tensor to output
            py::ssize_t offset = 0;
            for (const auto& arr : input_arrays) {
                py::buffer_info buf = arr.request();
                const float* src = static_cast<const float*>(buf.ptr);

                // For simple cases (dim=0 or contiguous), use memcpy
                if (dim == 0) {
                    std::memcpy(y_ptr + offset, src, static_cast<size_t>(buf.size) * sizeof(float));
                    offset += buf.size;
                } else {
                    // General case: copy slice by slice along concat dim
                    // Compute number of slices before concat dim
                    py::ssize_t outer_size = 1;
                    for (int d = 0; d < dim; ++d) {
                        outer_size *= buf.shape[d];
                    }
                    py::ssize_t inner_size = buf.size / outer_size;
                    py::ssize_t out_inner_size = Y.size() / outer_size;
                    py::ssize_t this_dim_size = buf.shape[dim];
                    py::ssize_t after_dim_size = inner_size / this_dim_size;

                    py::ssize_t out_offset = offset * after_dim_size;
                    for (py::ssize_t o = 0; o < outer_size; ++o) {
                        std::memcpy(
                            y_ptr + o * out_inner_size + out_offset,
                            src + o * inner_size,
                            static_cast<size_t>(inner_size) * sizeof(float)
                        );
                    }
                    offset += this_dim_size;
                }
            }

            // Record XUE data movement event
            sw::xue::xue().record_dma_transfer(
                static_cast<uint64_t>(total_bytes),
                0
            );

            tensors[output_name] = Y;
            stats.ops_executed++;

        // =====================================================================
        // Reduction Operations (v0.3+ benchmarks)
        // =====================================================================
        } else if (opcode == "sum" || opcode == "mean" || opcode == "max" || opcode == "min") {
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors.at(x_name);
            py::buffer_info x_buf = X.request();
            const float* x_ptr = static_cast<const float*>(x_buf.ptr);

            // Get reduction parameters
            bool keepdims = attrs.contains("keepdims") ?
                attrs["keepdims"].cast<bool>() : false;

            // Handle axis parameter (can be None, int, or tuple)
            std::vector<int> axes;
            bool reduce_all = false;

            if (!attrs.contains("axis") || attrs["axis"].is_none()) {
                reduce_all = true;
            } else {
                try {
                    // Try single int
                    int axis = attrs["axis"].cast<int>();
                    if (axis < 0) axis = x_buf.ndim + axis;
                    axes.push_back(axis);
                } catch (...) {
                    // Try tuple/list
                    auto axis_list = attrs["axis"].cast<py::list>();
                    for (size_t i = 0; i < py::len(axis_list); ++i) {
                        int axis = axis_list[i].cast<int>();
                        if (axis < 0) axis = x_buf.ndim + axis;
                        axes.push_back(axis);
                    }
                }
            }

            py::array_t<float> Y;

            if (reduce_all) {
                // Reduce all elements to a single value
                float result = x_ptr[0];
                if (opcode == "sum") {
                    result = 0.0f;
                    for (py::ssize_t i = 0; i < x_buf.size; ++i) {
                        result += x_ptr[i];
                    }
                } else if (opcode == "mean") {
                    result = 0.0f;
                    for (py::ssize_t i = 0; i < x_buf.size; ++i) {
                        result += x_ptr[i];
                    }
                    result /= static_cast<float>(x_buf.size);
                } else if (opcode == "max") {
                    for (py::ssize_t i = 1; i < x_buf.size; ++i) {
                        if (x_ptr[i] > result) result = x_ptr[i];
                    }
                } else if (opcode == "min") {
                    for (py::ssize_t i = 1; i < x_buf.size; ++i) {
                        if (x_ptr[i] < result) result = x_ptr[i];
                    }
                }

                if (keepdims) {
                    std::vector<py::ssize_t> out_shape(x_buf.ndim, 1);
                    Y = py::array_t<float>(out_shape);
                } else {
                    Y = py::array_t<float>({static_cast<py::ssize_t>(1)});
                }
                Y.mutable_data()[0] = result;

            } else {
                // Reduce along specific axes
                // Build output shape
                std::vector<py::ssize_t> out_shape;
                for (int d = 0; d < x_buf.ndim; ++d) {
                    bool is_reduced = std::find(axes.begin(), axes.end(), d) != axes.end();
                    if (is_reduced) {
                        if (keepdims) out_shape.push_back(1);
                    } else {
                        out_shape.push_back(x_buf.shape[d]);
                    }
                }
                if (out_shape.empty()) out_shape.push_back(1);

                Y = py::array_t<float>(out_shape);
                float* y_ptr = Y.mutable_data();
                py::ssize_t out_size = Y.size();

                // Initialize output
                if (opcode == "sum" || opcode == "mean") {
                    std::fill(y_ptr, y_ptr + out_size, 0.0f);
                } else if (opcode == "max") {
                    std::fill(y_ptr, y_ptr + out_size, -std::numeric_limits<float>::infinity());
                } else if (opcode == "min") {
                    std::fill(y_ptr, y_ptr + out_size, std::numeric_limits<float>::infinity());
                }

                // Compute input/output strides
                std::vector<py::ssize_t> x_strides(x_buf.ndim);
                py::ssize_t stride = 1;
                for (int d = x_buf.ndim - 1; d >= 0; --d) {
                    x_strides[d] = stride;
                    stride *= x_buf.shape[d];
                }

                std::vector<py::ssize_t> y_strides(out_shape.size());
                stride = 1;
                for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
                    y_strides[d] = stride;
                    stride *= out_shape[d];
                }

                // Count elements per output for mean
                py::ssize_t reduce_count = 1;
                for (int axis : axes) {
                    reduce_count *= x_buf.shape[axis];
                }

                // Iterate over input and accumulate
                std::vector<py::ssize_t> coords(x_buf.ndim, 0);
                for (py::ssize_t i = 0; i < x_buf.size; ++i) {
                    // Compute input index
                    py::ssize_t x_idx = 0;
                    for (int d = 0; d < x_buf.ndim; ++d) {
                        x_idx += coords[d] * x_strides[d];
                    }

                    // Compute output index (skip reduced dims if not keepdims)
                    py::ssize_t y_idx = 0;
                    int out_d = 0;
                    for (int d = 0; d < x_buf.ndim; ++d) {
                        bool is_reduced = std::find(axes.begin(), axes.end(), d) != axes.end();
                        if (!is_reduced || keepdims) {
                            py::ssize_t coord = is_reduced ? 0 : coords[d];
                            y_idx += coord * y_strides[out_d];
                            out_d++;
                        }
                    }

                    // Accumulate
                    float val = x_ptr[x_idx];
                    if (opcode == "sum" || opcode == "mean") {
                        y_ptr[y_idx] += val;
                    } else if (opcode == "max") {
                        if (val > y_ptr[y_idx]) y_ptr[y_idx] = val;
                    } else if (opcode == "min") {
                        if (val < y_ptr[y_idx]) y_ptr[y_idx] = val;
                    }

                    // Increment coordinates
                    for (int d = x_buf.ndim - 1; d >= 0; --d) {
                        coords[d]++;
                        if (coords[d] < x_buf.shape[d]) break;
                        coords[d] = 0;
                    }
                }

                // Finalize mean
                if (opcode == "mean") {
                    for (py::ssize_t i = 0; i < out_size; ++i) {
                        y_ptr[i] /= static_cast<float>(reduce_count);
                    }
                }
            }

            // Record XUE reduction event
            sw::xue::xue().record(sw::xue::EventType::REDUCE_SUM,
                sw::xue::EventMetadata::compute(static_cast<uint64_t>(x_buf.size)));

            tensors[output_name] = Y;
            stats.ops_executed++;

        // =====================================================================
        // Fused Convolution Operations (v0.3+ benchmarks)
        // =====================================================================
        } else if (opcode == "fused_conv2d_relu") {
            // Fused: Y = relu(conv2d(X, W) + bias)
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();

            std::string x_name = input_names[0].cast<std::string>();
            std::string w_name = input_names[1].cast<std::string>();

            auto X = tensors.at(x_name);
            auto W = tensors.at(w_name);

            py::buffer_info x_buf = X.request();
            py::buffer_info w_buf = W.request();

            // Extract dimensions
            uint32_t batch_size = static_cast<uint32_t>(x_buf.shape[0]);
            uint32_t in_channels = static_cast<uint32_t>(x_buf.shape[1]);
            uint32_t input_h = static_cast<uint32_t>(x_buf.shape[2]);
            uint32_t input_w = static_cast<uint32_t>(x_buf.shape[3]);

            uint32_t out_channels = static_cast<uint32_t>(w_buf.shape[0]);
            uint32_t kernel_h = static_cast<uint32_t>(w_buf.shape[2]);
            uint32_t kernel_w = static_cast<uint32_t>(w_buf.shape[3]);

            // Get parameters from attrs
            uint32_t stride_h = 1, stride_w = 1;
            uint32_t pad_h = 0, pad_w = 0;
            uint32_t dilation_h = 1, dilation_w = 1;
            uint32_t groups = 1;

            if (attrs.contains("stride")) {
                auto stride = attrs["stride"];
                if (py::isinstance<py::tuple>(stride)) {
                    auto s = stride.cast<py::tuple>();
                    stride_h = s[0].cast<uint32_t>();
                    stride_w = s[1].cast<uint32_t>();
                } else {
                    stride_h = stride_w = stride.cast<uint32_t>();
                }
            }
            if (attrs.contains("padding")) {
                auto padding = attrs["padding"];
                if (py::isinstance<py::tuple>(padding)) {
                    auto p = padding.cast<py::tuple>();
                    pad_h = p[0].cast<uint32_t>();
                    pad_w = p[1].cast<uint32_t>();
                } else {
                    pad_h = pad_w = padding.cast<uint32_t>();
                }
            }
            if (attrs.contains("dilation")) {
                auto dilation = attrs["dilation"];
                if (py::isinstance<py::tuple>(dilation)) {
                    auto d = dilation.cast<py::tuple>();
                    dilation_h = d[0].cast<uint32_t>();
                    dilation_w = d[1].cast<uint32_t>();
                } else {
                    dilation_h = dilation_w = dilation.cast<uint32_t>();
                }
            }
            if (attrs.contains("groups")) {
                groups = attrs["groups"].cast<uint32_t>();
            }

            // Compute output dimensions
            uint32_t output_h = (input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
            uint32_t output_w = (input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

            // Create output array
            std::vector<py::ssize_t> out_shape = {
                static_cast<py::ssize_t>(batch_size),
                static_cast<py::ssize_t>(out_channels),
                static_cast<py::ssize_t>(output_h),
                static_cast<py::ssize_t>(output_w)
            };
            py::array_t<float> conv_result = np.attr("zeros")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info r_buf = conv_result.request();

            // Get bias pointer if present
            const float* bias_ptr = nullptr;
            py::array_t<float> bias_arr;
            if (py::len(input_names) > 2) {
                std::string bias_name = input_names[2].cast<std::string>();
                if (tensors.count(bias_name) > 0) {
                    bias_arr = tensors.at(bias_name);
                    bias_ptr = static_cast<const float*>(bias_arr.request().ptr);
                }
            }

            // Build Conv2D descriptor
            sw::kpu::Conv2DDescriptor desc;
            desc.batch_size = batch_size;
            desc.in_channels = in_channels;
            desc.in_height = input_h;
            desc.in_width = input_w;
            desc.out_channels = out_channels;
            desc.kernel_height = kernel_h;
            desc.kernel_width = kernel_w;
            desc.stride_h = stride_h;
            desc.stride_w = stride_w;
            desc.padding_h = pad_h;
            desc.padding_w = pad_w;
            desc.dilation_h = dilation_h;
            desc.dilation_w = dilation_w;
            desc.groups = groups;

            // Submit Conv2D to C++ BehavioralComputeFabric
            behavioral_compute_fabric_->submit_conv2d(
                desc,
                x_buf.ptr,
                w_buf.ptr,
                bias_ptr,
                r_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            // Apply ReLU in-place
            float* result_ptr = static_cast<float*>(r_buf.ptr);
            size_t total_elements = static_cast<size_t>(batch_size) * out_channels * output_h * output_w;
            for (size_t i = 0; i < total_elements; ++i) {
                if (result_ptr[i] < 0.0f) result_ptr[i] = 0.0f;
            }

            // Record XUE events
            int64_t gemm_M = static_cast<int64_t>(batch_size) * output_h * output_w;
            int64_t gemm_K = static_cast<int64_t>(in_channels) / groups * kernel_h * kernel_w;
            int64_t gemm_N = out_channels;
            stats.matmul_flops += 2LL * gemm_M * gemm_N * gemm_K;

            sw::xue::xue().record_relu(total_elements, 0);

            tensors[output_name] = conv_result;
            stats.ops_executed++;

        } else if (opcode == "fused_conv2d_bn_relu") {
            // Fused: Y = relu(batch_norm(conv2d(X, W)))
            // inputs: [X, W, gamma, beta] (gamma/beta optional)
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();

            std::string x_name = input_names[0].cast<std::string>();
            std::string w_name = input_names[1].cast<std::string>();

            auto X = tensors.at(x_name);
            auto W = tensors.at(w_name);

            py::buffer_info x_buf = X.request();
            py::buffer_info w_buf = W.request();

            // Extract dimensions
            uint32_t batch_size = static_cast<uint32_t>(x_buf.shape[0]);
            uint32_t in_channels = static_cast<uint32_t>(x_buf.shape[1]);
            uint32_t input_h = static_cast<uint32_t>(x_buf.shape[2]);
            uint32_t input_w = static_cast<uint32_t>(x_buf.shape[3]);

            uint32_t out_channels = static_cast<uint32_t>(w_buf.shape[0]);
            uint32_t kernel_h = static_cast<uint32_t>(w_buf.shape[2]);
            uint32_t kernel_w = static_cast<uint32_t>(w_buf.shape[3]);

            // Get parameters from attrs
            uint32_t stride_h = 1, stride_w = 1;
            uint32_t pad_h = 0, pad_w = 0;
            uint32_t dilation_h = 1, dilation_w = 1;
            uint32_t groups = 1;
            float eps = 1e-5f;

            if (attrs.contains("stride")) {
                auto stride = attrs["stride"];
                if (py::isinstance<py::tuple>(stride)) {
                    auto s = stride.cast<py::tuple>();
                    stride_h = s[0].cast<uint32_t>();
                    stride_w = s[1].cast<uint32_t>();
                } else {
                    stride_h = stride_w = stride.cast<uint32_t>();
                }
            }
            if (attrs.contains("padding")) {
                auto padding = attrs["padding"];
                if (py::isinstance<py::tuple>(padding)) {
                    auto p = padding.cast<py::tuple>();
                    pad_h = p[0].cast<uint32_t>();
                    pad_w = p[1].cast<uint32_t>();
                } else {
                    pad_h = pad_w = padding.cast<uint32_t>();
                }
            }
            if (attrs.contains("dilation")) {
                auto dilation = attrs["dilation"];
                if (py::isinstance<py::tuple>(dilation)) {
                    auto d = dilation.cast<py::tuple>();
                    dilation_h = d[0].cast<uint32_t>();
                    dilation_w = d[1].cast<uint32_t>();
                } else {
                    dilation_h = dilation_w = dilation.cast<uint32_t>();
                }
            }
            if (attrs.contains("groups")) {
                groups = attrs["groups"].cast<uint32_t>();
            }
            if (attrs.contains("eps")) {
                eps = attrs["eps"].cast<float>();
            }

            // Compute output dimensions
            uint32_t output_h = (input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
            uint32_t output_w = (input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

            // Create output array
            std::vector<py::ssize_t> out_shape = {
                static_cast<py::ssize_t>(batch_size),
                static_cast<py::ssize_t>(out_channels),
                static_cast<py::ssize_t>(output_h),
                static_cast<py::ssize_t>(output_w)
            };
            py::array_t<float> conv_result = np.attr("zeros")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
            py::buffer_info r_buf = conv_result.request();

            // Build Conv2D descriptor
            sw::kpu::Conv2DDescriptor desc;
            desc.batch_size = batch_size;
            desc.in_channels = in_channels;
            desc.in_height = input_h;
            desc.in_width = input_w;
            desc.out_channels = out_channels;
            desc.kernel_height = kernel_h;
            desc.kernel_width = kernel_w;
            desc.stride_h = stride_h;
            desc.stride_w = stride_w;
            desc.padding_h = pad_h;
            desc.padding_w = pad_w;
            desc.dilation_h = dilation_h;
            desc.dilation_w = dilation_w;
            desc.groups = groups;

            // Submit Conv2D to C++ BehavioralComputeFabric (no bias for fused BN)
            behavioral_compute_fabric_->submit_conv2d(
                desc,
                x_buf.ptr,
                w_buf.ptr,
                nullptr,  // No bias - BN handles this
                r_buf.ptr,
                nullptr
            );
            behavioral_compute_fabric_->drain();

            float* result_ptr = static_cast<float*>(r_buf.ptr);
            size_t spatial_size = static_cast<size_t>(output_h) * output_w;
            size_t total_elements = static_cast<size_t>(batch_size) * out_channels * spatial_size;

            // Apply BatchNorm per channel: (x - mean) / sqrt(var + eps) * gamma + beta
            // Compute mean and variance per channel
            std::vector<float> means(out_channels, 0.0f);
            std::vector<float> vars(out_channels, 0.0f);
            size_t count_per_channel = static_cast<size_t>(batch_size) * spatial_size;

            // Compute means
            for (uint32_t n = 0; n < batch_size; ++n) {
                for (uint32_t c = 0; c < out_channels; ++c) {
                    for (size_t s = 0; s < spatial_size; ++s) {
                        size_t idx = n * out_channels * spatial_size + c * spatial_size + s;
                        means[c] += result_ptr[idx];
                    }
                }
            }
            for (uint32_t c = 0; c < out_channels; ++c) {
                means[c] /= static_cast<float>(count_per_channel);
            }

            // Compute variances
            for (uint32_t n = 0; n < batch_size; ++n) {
                for (uint32_t c = 0; c < out_channels; ++c) {
                    for (size_t s = 0; s < spatial_size; ++s) {
                        size_t idx = n * out_channels * spatial_size + c * spatial_size + s;
                        float diff = result_ptr[idx] - means[c];
                        vars[c] += diff * diff;
                    }
                }
            }
            for (uint32_t c = 0; c < out_channels; ++c) {
                vars[c] /= static_cast<float>(count_per_channel);
            }

            // Get gamma and beta if provided
            const float* gamma_ptr = nullptr;
            const float* beta_ptr = nullptr;
            py::array_t<float> gamma_arr, beta_arr;

            if (py::len(input_names) > 2) {
                std::string gamma_name = input_names[2].cast<std::string>();
                if (tensors.count(gamma_name) > 0) {
                    gamma_arr = tensors.at(gamma_name);
                    gamma_ptr = static_cast<const float*>(gamma_arr.request().ptr);
                }
            }
            if (py::len(input_names) > 3) {
                std::string beta_name = input_names[3].cast<std::string>();
                if (tensors.count(beta_name) > 0) {
                    beta_arr = tensors.at(beta_name);
                    beta_ptr = static_cast<const float*>(beta_arr.request().ptr);
                }
            }

            // Apply normalization + scale + shift + ReLU
            for (uint32_t n = 0; n < batch_size; ++n) {
                for (uint32_t c = 0; c < out_channels; ++c) {
                    float inv_std = 1.0f / std::sqrt(vars[c] + eps);
                    float gamma = gamma_ptr ? gamma_ptr[c] : 1.0f;
                    float beta = beta_ptr ? beta_ptr[c] : 0.0f;

                    for (size_t s = 0; s < spatial_size; ++s) {
                        size_t idx = n * out_channels * spatial_size + c * spatial_size + s;
                        float normalized = (result_ptr[idx] - means[c]) * inv_std;
                        float scaled = normalized * gamma + beta;
                        // ReLU
                        result_ptr[idx] = scaled > 0.0f ? scaled : 0.0f;
                    }
                }
            }

            // Record XUE events
            int64_t gemm_M = static_cast<int64_t>(batch_size) * output_h * output_w;
            int64_t gemm_K = static_cast<int64_t>(in_channels) / groups * kernel_h * kernel_w;
            int64_t gemm_N = out_channels;
            stats.matmul_flops += 2LL * gemm_M * gemm_N * gemm_K;

            // BN operations: mean, var, normalize, scale, shift = 5 ops per element
            sw::xue::xue().record_mul(total_elements * 5, 0);
            sw::xue::xue().record_relu(total_elements, 0);

            tensors[output_name] = conv_result;
            stats.ops_executed++;

        // =====================================================================
        // Transformer Operations (v0.6+ transformers)
        // =====================================================================
        } else if (opcode == "softmax") {
            // Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors.at(x_name);
            py::buffer_info x_buf = X.request();

            // Get axis (default: -1, meaning last axis)
            int axis = attrs.contains("axis") ? attrs["axis"].cast<int>() : -1;
            if (axis < 0) axis += x_buf.ndim;

            // Create output array
            std::vector<py::ssize_t> out_shape(x_buf.shape.begin(), x_buf.shape.end());
            py::array_t<float> Y(out_shape);
            float* y_ptr = Y.mutable_data();
            const float* x_ptr = static_cast<const float*>(x_buf.ptr);

            // Compute strides
            std::vector<py::ssize_t> strides(static_cast<size_t>(x_buf.ndim));
            py::ssize_t stride = 1;
            for (int d = x_buf.ndim - 1; d >= 0; --d) {
                strides[static_cast<size_t>(d)] = stride;
                stride *= x_buf.shape[static_cast<size_t>(d)];
            }

            // Size along softmax axis
            py::ssize_t axis_size = x_buf.shape[static_cast<size_t>(axis)];
            py::ssize_t outer_size = x_buf.size / axis_size;

            // Softmax computation: iterate over all slices along axis
            for (py::ssize_t outer = 0; outer < outer_size; ++outer) {
                // Compute base index for this slice
                py::ssize_t base_idx = 0;
                py::ssize_t remaining = outer;
                for (int d = x_buf.ndim - 1; d >= 0; --d) {
                    if (d == axis) continue;
                    py::ssize_t coord = remaining % x_buf.shape[static_cast<size_t>(d)];
                    remaining /= x_buf.shape[static_cast<size_t>(d)];
                    base_idx += coord * strides[static_cast<size_t>(d)];
                }

                // Find max for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (py::ssize_t i = 0; i < axis_size; ++i) {
                    float val = x_ptr[base_idx + i * strides[static_cast<size_t>(axis)]];
                    if (val > max_val) max_val = val;
                }

                // Compute exp(x - max) and sum
                float sum_exp = 0.0f;
                for (py::ssize_t i = 0; i < axis_size; ++i) {
                    py::ssize_t idx = base_idx + i * strides[static_cast<size_t>(axis)];
                    float exp_val = std::exp(x_ptr[idx] - max_val);
                    y_ptr[idx] = exp_val;
                    sum_exp += exp_val;
                }

                // Normalize
                for (py::ssize_t i = 0; i < axis_size; ++i) {
                    py::ssize_t idx = base_idx + i * strides[static_cast<size_t>(axis)];
                    y_ptr[idx] /= sum_exp;
                }
            }

            // Record XUE event: softmax involves max, exp, sum, div per element
            sw::xue::xue().record_softmax(static_cast<uint64_t>(x_buf.size), 0);

            tensors[output_name] = Y;
            stats.ops_executed++;

        } else if (opcode == "layer_norm") {
            // Layer Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();
            std::string x_name = input_names[0].cast<std::string>();
            auto X = tensors.at(x_name);
            py::buffer_info x_buf = X.request();

            // Get normalized_shape and eps
            float eps = attrs.contains("eps") ? attrs["eps"].cast<float>() : 1e-5f;

            // Determine normalization dimensions (default: last dims matching normalized_shape)
            py::ssize_t norm_size = 1;
            int norm_dims = 1;  // Default: normalize over last dimension
            if (attrs.contains("normalized_shape")) {
                auto shape_list = attrs["normalized_shape"].cast<py::list>();
                norm_dims = static_cast<int>(py::len(shape_list));
                norm_size = 1;
                for (size_t i = 0; i < py::len(shape_list); ++i) {
                    norm_size *= shape_list[i].cast<py::ssize_t>();
                }
            } else {
                norm_size = x_buf.shape[x_buf.ndim - 1];
            }

            py::ssize_t batch_size = x_buf.size / norm_size;

            // Get gamma (weight) and beta (bias) if provided
            const float* gamma_ptr = nullptr;
            const float* beta_ptr = nullptr;
            py::array_t<float> gamma_arr, beta_arr;

            if (input_names.size() > 1) {
                std::string gamma_name = input_names[1].cast<std::string>();
                if (tensors.count(gamma_name) > 0) {
                    gamma_arr = tensors.at(gamma_name);
                    gamma_ptr = static_cast<const float*>(gamma_arr.request().ptr);
                }
            }
            if (input_names.size() > 2) {
                std::string beta_name = input_names[2].cast<std::string>();
                if (tensors.count(beta_name) > 0) {
                    beta_arr = tensors.at(beta_name);
                    beta_ptr = static_cast<const float*>(beta_arr.request().ptr);
                }
            }

            // Create output array
            std::vector<py::ssize_t> out_shape(x_buf.shape.begin(), x_buf.shape.end());
            py::array_t<float> Y(out_shape);
            float* y_ptr = Y.mutable_data();
            const float* x_ptr = static_cast<const float*>(x_buf.ptr);

            // Normalize each batch element
            for (py::ssize_t b = 0; b < batch_size; ++b) {
                py::ssize_t offset = b * norm_size;

                // Compute mean
                float mean = 0.0f;
                for (py::ssize_t i = 0; i < norm_size; ++i) {
                    mean += x_ptr[offset + i];
                }
                mean /= static_cast<float>(norm_size);

                // Compute variance
                float var = 0.0f;
                for (py::ssize_t i = 0; i < norm_size; ++i) {
                    float diff = x_ptr[offset + i] - mean;
                    var += diff * diff;
                }
                var /= static_cast<float>(norm_size);

                // Normalize and apply affine transformation
                float inv_std = 1.0f / std::sqrt(var + eps);
                for (py::ssize_t i = 0; i < norm_size; ++i) {
                    float normalized = (x_ptr[offset + i] - mean) * inv_std;
                    float gamma = gamma_ptr ? gamma_ptr[i] : 1.0f;
                    float beta = beta_ptr ? beta_ptr[i] : 0.0f;
                    y_ptr[offset + i] = normalized * gamma + beta;
                }
            }

            // Record XUE event: layernorm operations
            sw::xue::xue().record_layernorm(static_cast<uint64_t>(x_buf.size), 0);

            tensors[output_name] = Y;
            stats.ops_executed++;

        } else if (opcode == "attention") {
            // Scaled Dot-Product Attention: softmax(Q @ K^T / sqrt(d_k)) @ V
            // Supports both simple attention and full multi-head with QKV projections
            py::dict attrs = op.contains("attrs") ? op["attrs"].cast<py::dict>() : py::dict();

            // Check for QKV projection mode
            bool include_qkv_projection = attrs.contains("include_qkv_projection") &&
                                          attrs["include_qkv_projection"].cast<bool>();
            bool include_output_projection = attrs.contains("include_output_projection") &&
                                             attrs["include_output_projection"].cast<bool>();

            py::ssize_t num_heads = 1;
            if (attrs.contains("num_heads")) {
                num_heads = attrs["num_heads"].cast<py::ssize_t>();
            }

            py::array_t<float> Q, K, V;
            py::ssize_t batch_size, seq_len, d_model, d_k, d_v;
            py::array_t<float> w_o;  // Output projection weights

            if (include_qkv_projection) {
                // Multi-head attention with inline QKV projections
                // Inputs: x, w_q, w_k, w_v, [w_o]
                std::string x_name = input_names[0].cast<std::string>();
                std::string wq_name = input_names[1].cast<std::string>();
                std::string wk_name = input_names[2].cast<std::string>();
                std::string wv_name = input_names[3].cast<std::string>();

                auto X = tensors.at(x_name);
                auto W_Q = tensors.at(wq_name);
                auto W_K = tensors.at(wk_name);
                auto W_V = tensors.at(wv_name);

                py::buffer_info x_buf = X.request();
                py::buffer_info wq_buf = W_Q.request();

                // X is [B, S, D]
                batch_size = x_buf.shape[0];
                seq_len = x_buf.shape[1];
                d_model = x_buf.shape[2];
                d_k = d_model / num_heads;
                d_v = d_k;

                // Compute Q = X @ W_Q, K = X @ W_K, V = X @ W_V
                // Each projection: [B, S, D] @ [D, D] = [B, S, D]
                const float* x_ptr = static_cast<const float*>(x_buf.ptr);
                const float* wq_ptr = static_cast<const float*>(wq_buf.ptr);
                const float* wk_ptr = static_cast<const float*>(W_K.request().ptr);
                const float* wv_ptr = static_cast<const float*>(W_V.request().ptr);

                // Allocate projected Q, K, V
                std::vector<py::ssize_t> proj_shape = {batch_size, seq_len, d_model};
                Q = py::array_t<float>(proj_shape);
                K = py::array_t<float>(proj_shape);
                V = py::array_t<float>(proj_shape);
                float* q_ptr = Q.mutable_data();
                float* k_ptr = K.mutable_data();
                float* v_ptr = V.mutable_data();

                // Perform projections: output[b,s,d] = sum_i(X[b,s,i] * W[i,d])
                for (py::ssize_t b = 0; b < batch_size; ++b) {
                    for (py::ssize_t s = 0; s < seq_len; ++s) {
                        for (py::ssize_t d = 0; d < d_model; ++d) {
                            float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;
                            for (py::ssize_t i = 0; i < d_model; ++i) {
                                float x_val = x_ptr[(b * seq_len + s) * d_model + i];
                                sum_q += x_val * wq_ptr[i * d_model + d];
                                sum_k += x_val * wk_ptr[i * d_model + d];
                                sum_v += x_val * wv_ptr[i * d_model + d];
                            }
                            py::ssize_t out_idx = (b * seq_len + s) * d_model + d;
                            q_ptr[out_idx] = sum_q;
                            k_ptr[out_idx] = sum_k;
                            v_ptr[out_idx] = sum_v;
                        }
                    }
                }

                // Record projection matmuls in XUE
                int64_t proj_flops = 2LL * batch_size * seq_len * d_model * d_model;
                stats.matmul_flops += 3 * proj_flops;  // Q, K, V projections
                stats.matmul_count += 3;

                // Get output projection weights if present
                if (include_output_projection && input_names.size() > 4) {
                    std::string wo_name = input_names[4].cast<std::string>();
                    w_o = tensors.at(wo_name);
                }
            } else {
                // Simple attention: Q, K, V already provided
                std::string q_name = input_names[0].cast<std::string>();
                std::string k_name = input_names[1].cast<std::string>();
                std::string v_name = input_names[2].cast<std::string>();

                Q = tensors.at(q_name);
                K = tensors.at(k_name);
                V = tensors.at(v_name);
            }

            py::buffer_info q_buf = Q.request();
            py::buffer_info k_buf = K.request();
            py::buffer_info v_buf = V.request();

            const float* q_ptr = static_cast<const float*>(q_buf.ptr);
            const float* k_ptr = static_cast<const float*>(k_buf.ptr);
            const float* v_ptr = static_cast<const float*>(v_buf.ptr);

            // Determine dimensions based on tensor rank (for non-projection case)
            // Support: 2D [seq, d], 3D [batch, seq, d], 4D [batch, heads, seq, d]
            if (!include_qkv_projection) {
                batch_size = 1;
                num_heads = 1;
            }

            if (q_buf.ndim == 4) {
                batch_size = q_buf.shape[0];
                num_heads = q_buf.shape[1];
                seq_len = q_buf.shape[2];
                d_k = q_buf.shape[3];
                d_v = v_buf.shape[3];
            } else if (q_buf.ndim == 3) {
                batch_size = q_buf.shape[0];
                seq_len = q_buf.shape[1];
                d_k = q_buf.shape[2];
                d_v = v_buf.shape[2];
            } else {  // 2D
                seq_len = q_buf.shape[0];
                d_k = q_buf.shape[1];
                d_v = v_buf.shape[1];
            }

            // Get scale factor: default is 1/sqrt(d_k)
            float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
            if (attrs.contains("scale") && !attrs["scale"].is_none()) {
                scale = attrs["scale"].cast<float>();
            }

            // For QKV projection mode, output shape is [B, S, D]
            // For simple attention, output shape matches V
            std::vector<py::ssize_t> out_shape;
            py::ssize_t head_dim = d_k;  // For QKV projection: d_k = d_model / num_heads

            if (include_qkv_projection) {
                // Output will be [B, S, D] where D = num_heads * head_dim
                out_shape = {batch_size, seq_len, d_model};
                head_dim = d_model / num_heads;
                d_k = head_dim;
                d_v = head_dim;
            } else {
                out_shape = std::vector<py::ssize_t>(v_buf.shape.begin(), v_buf.shape.end());
            }

            py::array_t<float> Y(out_shape);
            float* y_ptr = Y.mutable_data();

            // Temporary storage for scores and attention weights (per batch/head)
            size_t score_size = static_cast<size_t>(seq_len * seq_len);
            std::vector<float> scores(score_size);
            std::vector<float> attn_weights(score_size);

            // Process each batch and head
            py::ssize_t total_iterations = batch_size * num_heads;
            for (py::ssize_t iter = 0; iter < total_iterations; ++iter) {
                py::ssize_t b = iter / num_heads;
                py::ssize_t h = iter % num_heads;

                // Step 1: Compute scores = Q @ K^T * scale
                // For QKV projection: Q, K, V are [B, S, D] where D = num_heads * head_dim
                // Head h uses indices [h*head_dim, (h+1)*head_dim) for the last dimension
                for (py::ssize_t i = 0; i < seq_len; ++i) {
                    for (py::ssize_t j = 0; j < seq_len; ++j) {
                        float dot = 0.0f;
                        for (py::ssize_t k = 0; k < head_dim; ++k) {
                            py::ssize_t q_idx, k_idx;
                            if (include_qkv_projection) {
                                // [B, S, D] layout: idx = b*S*D + s*D + h*head_dim + k
                                q_idx = b * seq_len * d_model + i * d_model + h * head_dim + k;
                                k_idx = b * seq_len * d_model + j * d_model + h * head_dim + k;
                            } else if (q_buf.ndim == 4) {
                                // [B, H, S, d_k] layout
                                q_idx = (b * num_heads + h) * seq_len * d_k + i * d_k + k;
                                k_idx = (b * num_heads + h) * seq_len * d_k + j * d_k + k;
                            } else if (q_buf.ndim == 3) {
                                // [B, S, d_k] layout (single head)
                                q_idx = b * seq_len * d_k + i * d_k + k;
                                k_idx = b * seq_len * d_k + j * d_k + k;
                            } else {
                                // [S, d_k] layout
                                q_idx = i * d_k + k;
                                k_idx = j * d_k + k;
                            }
                            dot += q_ptr[q_idx] * k_ptr[k_idx];
                        }
                        scores[static_cast<size_t>(i * seq_len + j)] = dot * scale;
                    }
                }

                // Step 2: Softmax along last dimension (each row)
                for (py::ssize_t i = 0; i < seq_len; ++i) {
                    // Find max for numerical stability
                    float max_val = scores[static_cast<size_t>(i * seq_len)];
                    for (py::ssize_t j = 1; j < seq_len; ++j) {
                        float val = scores[static_cast<size_t>(i * seq_len + j)];
                        if (val > max_val) max_val = val;
                    }

                    // Compute exp(x - max) and sum
                    float sum_exp = 0.0f;
                    for (py::ssize_t j = 0; j < seq_len; ++j) {
                        size_t idx = static_cast<size_t>(i * seq_len + j);
                        float exp_val = std::exp(scores[idx] - max_val);
                        attn_weights[idx] = exp_val;
                        sum_exp += exp_val;
                    }

                    // Normalize
                    for (py::ssize_t j = 0; j < seq_len; ++j) {
                        attn_weights[static_cast<size_t>(i * seq_len + j)] /= sum_exp;
                    }
                }

                // Step 3: Compute output = attn_weights @ V
                for (py::ssize_t i = 0; i < seq_len; ++i) {
                    for (py::ssize_t j = 0; j < head_dim; ++j) {
                        float sum = 0.0f;
                        for (py::ssize_t k = 0; k < seq_len; ++k) {
                            py::ssize_t v_idx;
                            if (include_qkv_projection) {
                                // [B, S, D] layout
                                v_idx = b * seq_len * d_model + k * d_model + h * head_dim + j;
                            } else if (v_buf.ndim == 4) {
                                // [B, H, S, d_v] layout
                                v_idx = (b * num_heads + h) * seq_len * d_v + k * d_v + j;
                            } else if (v_buf.ndim == 3) {
                                // [B, S, d_v] layout
                                v_idx = b * seq_len * d_v + k * d_v + j;
                            } else {
                                // [S, d_v] layout
                                v_idx = k * d_v + j;
                            }
                            sum += attn_weights[static_cast<size_t>(i * seq_len + k)] * v_ptr[v_idx];
                        }
                        // Write to output
                        py::ssize_t out_idx;
                        if (include_qkv_projection) {
                            // [B, S, D] layout
                            out_idx = b * seq_len * d_model + i * d_model + h * head_dim + j;
                        } else if (v_buf.ndim == 4) {
                            // [B, H, S, d_v] layout
                            out_idx = (b * num_heads + h) * seq_len * d_v + i * d_v + j;
                        } else if (v_buf.ndim == 3) {
                            // [B, S, d_v] layout
                            out_idx = b * seq_len * d_v + i * d_v + j;
                        } else {
                            // [S, d_v] layout
                            out_idx = i * d_v + j;
                        }
                        y_ptr[out_idx] = sum;
                    }
                }
            }

            // Apply output projection if requested: Y = Y @ W_O
            if (include_output_projection && w_o.size() > 0) {
                py::buffer_info wo_buf = w_o.request();
                const float* wo_ptr = static_cast<const float*>(wo_buf.ptr);

                // Create new output for projection result
                py::array_t<float> Y_proj(out_shape);
                float* yp_ptr = Y_proj.mutable_data();

                // Y_proj[b, s, d] = sum_i(Y[b, s, i] * W_O[i, d])
                for (py::ssize_t b = 0; b < batch_size; ++b) {
                    for (py::ssize_t s = 0; s < seq_len; ++s) {
                        for (py::ssize_t d = 0; d < d_model; ++d) {
                            float sum = 0.0f;
                            for (py::ssize_t i = 0; i < d_model; ++i) {
                                py::ssize_t y_idx = (b * seq_len + s) * d_model + i;
                                sum += y_ptr[y_idx] * wo_ptr[i * d_model + d];
                            }
                            yp_ptr[(b * seq_len + s) * d_model + d] = sum;
                        }
                    }
                }

                // Record output projection matmul
                int64_t proj_flops = 2LL * batch_size * seq_len * d_model * d_model;
                stats.matmul_flops += proj_flops;
                stats.matmul_count += 1;

                Y = Y_proj;
                y_ptr = Y.mutable_data();
            }

            // Record XUE events for attention computation
            // Q @ K^T: 2 * batch * heads * seq_len * seq_len * d_k FLOPs
            int64_t qk_flops = 2LL * batch_size * num_heads * seq_len * seq_len * d_k;
            // Attention @ V: 2 * batch * heads * seq_len * seq_len * d_v FLOPs
            int64_t av_flops = 2LL * batch_size * num_heads * seq_len * seq_len * d_v;

            stats.matmul_flops += qk_flops + av_flops;
            stats.matmul_count += 2 * batch_size * num_heads;  // Two matmuls per attention per batch/head

            // Record XUE events directly (don't use submit_matmul which executes the matmul)
            // Q @ K^T matmul event
            sw::xue::xue().record_matmul(
                static_cast<uint32_t>(seq_len),
                static_cast<uint32_t>(seq_len),
                static_cast<uint32_t>(d_k),
                0  // cycles (behavioral mode)
            );

            // Attention @ V matmul event
            sw::xue::xue().record_matmul(
                static_cast<uint32_t>(seq_len),
                static_cast<uint32_t>(d_v),
                static_cast<uint32_t>(seq_len),
                0  // cycles (behavioral mode)
            );

            // Record softmax operations
            sw::xue::xue().record_softmax(
                static_cast<uint64_t>(batch_size * num_heads * seq_len * seq_len),
                0
            );

            tensors[output_name] = Y;
            stats.ops_executed++;

        } else {
            throw std::runtime_error("Unsupported opcode in native execution: " + opcode);
        }
    }

    /**
     * @brief Execute with transactional timing simulation
     *
     * Uses the C++ TransactionalComputeFabric for accurate throughput-based
     * timing of matmul operations. Uses TransactionalMemoryController for
     * memory traffic simulation with page hit/miss modeling.
     *
     * Other operations use behavioral execution with estimated timing.
     */
    std::pair<py::array_t<float>, py::dict> execute_simulated(
        const py::dict& dfx_json,
        const std::vector<py::array_t<float>>& inputs,
        [[maybe_unused]] const std::string& mode,
        NativeExecutionStats& stats
    ) {
        // Reset the compute fabric for this execution
        compute_fabric_->reset();
        compute_fabric_->reset_stats();

        // Reset the memory controller for this execution
        memory_controller_->reset();
        memory_controller_->reset_stats();

        // Configure transaction sizes for each memory level (XUE)
        // These represent typical transfer granularities
        stats.dram.transaction_size = 64;   // DRAM burst size (cache line)
        stats.l3.transaction_size = 256;    // L3 tile granularity
        stats.l2.transaction_size = 128;    // L2 tile granularity
        stats.l1.transaction_size = 64;     // L1 stream element size

        // Memory traffic tracker for per-level stats
        sw::kpu::stats::MemoryTraffic memory_traffic;

        // Parse DFX program
        auto ops = dfx_json["ops"].cast<py::list>();
        auto input_names = dfx_json["inputs"].cast<py::list>();
        auto output_names = dfx_json["outputs"].cast<py::list>();

        // Map tensor names to numpy arrays
        std::unordered_map<std::string, py::array_t<float>> tensors;

        // Track base address for memory simulation (simple linear allocation)
        uint64_t next_address = 0;
        std::unordered_map<std::string, uint64_t> tensor_addresses;

        // Load inputs and simulate memory reads through the hierarchy
        // Data flow: DRAM → L3 → L2 → L1 (credit-based dataflow)
        for (size_t i = 0; i < inputs.size() && i < static_cast<size_t>(py::len(input_names)); ++i) {
            std::string name = input_names[i].cast<std::string>();
            tensors[name] = inputs[i];

            // Track tensor address and size
            py::buffer_info buf = inputs[i].request();
            size_t tensor_bytes = static_cast<size_t>(buf.size) * sizeof(float);
            tensor_addresses[name] = next_address;

            // Track reads at each memory level (data flows through hierarchy)
            // DRAM reads
            memory_traffic.record_read(sw::kpu::stats::MemoryLevel::EXTERNAL, tensor_bytes);
            stats.dram.read_bytes += static_cast<int64_t>(tensor_bytes);
            stats.dram.read_count += (tensor_bytes + stats.dram.transaction_size - 1) / stats.dram.transaction_size;

            // L3 receives from DRAM, forwards to L2
            memory_traffic.record_read(sw::kpu::stats::MemoryLevel::L3, tensor_bytes);
            stats.l3.read_bytes += static_cast<int64_t>(tensor_bytes);
            stats.l3.read_count += (tensor_bytes + stats.l3.transaction_size - 1) / stats.l3.transaction_size;

            // L2 receives from L3, forwards to L1
            memory_traffic.record_read(sw::kpu::stats::MemoryLevel::L2, tensor_bytes);
            stats.l2.read_bytes += static_cast<int64_t>(tensor_bytes);
            stats.l2.read_count += (tensor_bytes + stats.l2.transaction_size - 1) / stats.l2.transaction_size;

            // L1 receives from L2, feeds to compute
            memory_traffic.record_read(sw::kpu::stats::MemoryLevel::L1, tensor_bytes);
            stats.l1.read_bytes += static_cast<int64_t>(tensor_bytes);
            stats.l1.read_count += (tensor_bytes + stats.l1.transaction_size - 1) / stats.l1.transaction_size;

            // Submit to memory controller for timing simulation
            constexpr uint32_t CACHE_LINE_SIZE = 64;  // bytes
            for (size_t offset = 0; offset < tensor_bytes; offset += CACHE_LINE_SIZE) {
                uint32_t chunk_size = std::min(static_cast<uint32_t>(CACHE_LINE_SIZE),
                                               static_cast<uint32_t>(tensor_bytes - offset));
                memory_controller_->submit_read(next_address + offset, chunk_size, nullptr);
            }

            next_address += tensor_bytes;
            stats.external_bytes += static_cast<int64_t>(tensor_bytes);
        }

        // Import numpy
        py::module np = py::module::import("numpy");

        // Execute operations with transactional timing
        for (auto op_obj : ops) {
            py::dict op = op_obj.cast<py::dict>();
            std::string opcode = op["opcode"].cast<std::string>();
            auto op_input_names = op["inputs"].cast<py::list>();
            auto op_output_names = op["outputs"].cast<py::list>();
            std::string output_name = op_output_names[0].cast<std::string>();

            if (opcode == "matmul") {
                // Use transactional compute fabric for matmul computation and timing
                std::string a_name = op_input_names[0].cast<std::string>();
                std::string b_name = op_input_names[1].cast<std::string>();

                auto A = tensors[a_name];
                auto B = tensors[b_name];

                py::buffer_info a_buf = A.request();
                py::buffer_info b_buf = B.request();

                // Get dimensions
                uint32_t M = static_cast<uint32_t>(a_buf.shape[a_buf.ndim - 2]);
                uint32_t K = static_cast<uint32_t>(a_buf.shape[a_buf.ndim - 1]);
                uint32_t N = static_cast<uint32_t>(b_buf.shape[b_buf.ndim - 1]);

                // Allocate output array
                std::vector<py::ssize_t> out_shape;
                for (int i = 0; i < a_buf.ndim - 1; ++i) {
                    out_shape.push_back(a_buf.shape[i]);
                }
                out_shape.push_back(b_buf.shape[b_buf.ndim - 1]);

                py::array_t<float> C = np.attr("zeros")(out_shape, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
                py::buffer_info c_buf = C.request();
                size_t c_bytes = static_cast<size_t>(c_buf.size) * sizeof(float);

                // Get data pointers
                float* a_ptr = static_cast<float*>(a_buf.ptr);
                float* b_ptr = static_cast<float*>(b_buf.ptr);
                float* c_ptr = static_cast<float*>(c_buf.ptr);

                // Set up matmul descriptor
                sw::kpu::MatMulDescriptor desc;
                desc.m = M;
                desc.n = N;
                desc.k = K;

                // Submit matmul to transactional fabric (computes AND times)
                compute_fabric_->submit_matmul(desc, a_ptr, b_ptr, c_ptr, nullptr);

                // Store result and allocate address
                tensors[output_name] = C;
                tensor_addresses[output_name] = next_address;

                // Drain to complete the operation
                compute_fabric_->drain();

                // Track writes through memory hierarchy (compute → L1 → L2 → L3 → DRAM)
                // L1 receives from compute
                memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L1, c_bytes);
                stats.l1.write_bytes += static_cast<int64_t>(c_bytes);
                stats.l1.write_count += (c_bytes + stats.l1.transaction_size - 1) / stats.l1.transaction_size;

                // L2 receives from L1
                memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L2, c_bytes);
                stats.l2.write_bytes += static_cast<int64_t>(c_bytes);
                stats.l2.write_count += (c_bytes + stats.l2.transaction_size - 1) / stats.l2.transaction_size;

                // L3 receives from L2
                memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L3, c_bytes);
                stats.l3.write_bytes += static_cast<int64_t>(c_bytes);
                stats.l3.write_count += (c_bytes + stats.l3.transaction_size - 1) / stats.l3.transaction_size;

                // DRAM receives from L3
                memory_traffic.record_write(sw::kpu::stats::MemoryLevel::EXTERNAL, c_bytes);
                stats.dram.write_bytes += static_cast<int64_t>(c_bytes);
                stats.dram.write_count += (c_bytes + stats.dram.transaction_size - 1) / stats.dram.transaction_size;

                // Submit to memory controller for timing simulation
                constexpr uint32_t CACHE_LINE_SIZE = 64;  // bytes
                for (size_t offset = 0; offset < c_bytes; offset += CACHE_LINE_SIZE) {
                    uint32_t chunk_size = std::min(static_cast<uint32_t>(CACHE_LINE_SIZE),
                                                   static_cast<uint32_t>(c_bytes - offset));
                    memory_controller_->submit_write(next_address + offset, nullptr, chunk_size, nullptr);
                }

                next_address += c_bytes;
                stats.memory_bytes += static_cast<int64_t>(c_bytes);

                // Track FLOPs
                stats.matmul_flops += 2LL * M * N * K;
                stats.total_macs += static_cast<int64_t>(M) * N * K;
                stats.matmul_count++;

            } else if (opcode == "linear") {
                // Linear: y = x @ W.T (+ b)
                // Weight is [out_features, in_features], needs transpose
                // Uses C++ TransactionalComputeFabric for computation and timing
                std::string x_name = op_input_names[0].cast<std::string>();
                std::string w_name = op_input_names[1].cast<std::string>();

                auto X = tensors[x_name];
                auto W = tensors[w_name];

                py::buffer_info x_buf = X.request();
                py::buffer_info w_buf = W.request();

                // Get dimensions: X is [batch, in_features], W is [out_features, in_features]
                uint32_t M = static_cast<uint32_t>(x_buf.shape[x_buf.ndim - 2]);  // batch
                uint32_t K = static_cast<uint32_t>(x_buf.shape[x_buf.ndim - 1]);  // in_features
                uint32_t N = static_cast<uint32_t>(w_buf.shape[w_buf.ndim - 2]);  // out_features

                // Transpose weight in C++: W.T has shape [in_features, out_features]
                std::vector<py::ssize_t> wt_shape = {static_cast<py::ssize_t>(K), static_cast<py::ssize_t>(N)};
                py::array_t<float> W_T(wt_shape);
                float* w_ptr = static_cast<float*>(w_buf.ptr);
                float* wt_ptr = W_T.mutable_data();
                for (uint32_t i = 0; i < N; ++i) {
                    for (uint32_t j = 0; j < K; ++j) {
                        wt_ptr[j * N + i] = w_ptr[i * K + j];
                    }
                }

                // Allocate output array
                std::vector<py::ssize_t> y_shape;
                for (int i = 0; i < x_buf.ndim - 1; ++i) {
                    y_shape.push_back(x_buf.shape[i]);
                }
                y_shape.push_back(N);

                py::array_t<float> Y(y_shape);
                py::buffer_info y_buf = Y.request();
                size_t y_bytes = static_cast<size_t>(y_buf.size) * sizeof(float);

                // Get data pointers
                float* x_ptr = static_cast<float*>(x_buf.ptr);
                float* y_ptr = static_cast<float*>(y_buf.ptr);

                // Set up matmul descriptor and execute via transactional fabric
                sw::kpu::MatMulDescriptor desc;
                desc.m = M;
                desc.n = N;
                desc.k = K;

                compute_fabric_->submit_matmul(desc, x_ptr, wt_ptr, y_ptr, nullptr);
                compute_fabric_->drain();

                // Add bias in C++ if present
                if (op_input_names.size() > 2) {
                    std::string b_name = op_input_names[2].cast<std::string>();
                    if (tensors.count(b_name) > 0) {
                        auto B = tensors[b_name];
                        py::buffer_info b_buf = B.request();
                        const float* b_ptr = static_cast<const float*>(b_buf.ptr);
                        // Add bias: y[i, j] += b[j]
                        for (uint32_t i = 0; i < M; ++i) {
                            for (uint32_t j = 0; j < N; ++j) {
                                y_ptr[i * N + j] += b_ptr[j];
                            }
                        }
                    }
                }

                tensors[output_name] = Y;
                tensor_addresses[output_name] = next_address;

                // Track writes through memory hierarchy
                memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L1, y_bytes);
                stats.l1.write_bytes += static_cast<int64_t>(y_bytes);
                stats.l1.write_count += (y_bytes + stats.l1.transaction_size - 1) / stats.l1.transaction_size;

                memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L2, y_bytes);
                stats.l2.write_bytes += static_cast<int64_t>(y_bytes);
                stats.l2.write_count += (y_bytes + stats.l2.transaction_size - 1) / stats.l2.transaction_size;

                memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L3, y_bytes);
                stats.l3.write_bytes += static_cast<int64_t>(y_bytes);
                stats.l3.write_count += (y_bytes + stats.l3.transaction_size - 1) / stats.l3.transaction_size;

                memory_traffic.record_write(sw::kpu::stats::MemoryLevel::EXTERNAL, y_bytes);
                stats.dram.write_bytes += static_cast<int64_t>(y_bytes);
                stats.dram.write_count += (y_bytes + stats.dram.transaction_size - 1) / stats.dram.transaction_size;

                // Submit to memory controller
                constexpr uint32_t CACHE_LINE_SIZE = 64;
                for (size_t offset = 0; offset < y_bytes; offset += CACHE_LINE_SIZE) {
                    uint32_t chunk_size = std::min(static_cast<uint32_t>(CACHE_LINE_SIZE),
                                                   static_cast<uint32_t>(y_bytes - offset));
                    memory_controller_->submit_write(next_address + offset, nullptr, chunk_size, nullptr);
                }

                next_address += y_bytes;
                stats.memory_bytes += static_cast<int64_t>(y_bytes);

                // Track FLOPs (matmul + optional bias add)
                stats.matmul_flops += 2LL * M * N * K;
                if (op_input_names.size() > 2) {
                    stats.matmul_flops += M * N;  // bias add
                }
                stats.total_macs += static_cast<int64_t>(M) * N * K;
                stats.matmul_count++;

            } else if (opcode == "conv2d") {
                // Conv2D - execute behaviorally and track FLOPs
                // The behavioral execution handles the computation correctly
                execute_op_behavioral(op, tensors, stats);

                // Track memory traffic for result
                if (op_output_names.size() > 0) {
                    auto result = tensors[output_name];
                    py::buffer_info result_buf = result.request();
                    size_t result_bytes = static_cast<size_t>(result_buf.size) * sizeof(float);

                    tensor_addresses[output_name] = next_address;

                    // Track writes through memory hierarchy
                    memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L1, result_bytes);
                    stats.l1.write_bytes += static_cast<int64_t>(result_bytes);
                    stats.l1.write_count += (result_bytes + stats.l1.transaction_size - 1) / stats.l1.transaction_size;

                    memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L2, result_bytes);
                    stats.l2.write_bytes += static_cast<int64_t>(result_bytes);
                    stats.l2.write_count += (result_bytes + stats.l2.transaction_size - 1) / stats.l2.transaction_size;

                    memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L3, result_bytes);
                    stats.l3.write_bytes += static_cast<int64_t>(result_bytes);
                    stats.l3.write_count += (result_bytes + stats.l3.transaction_size - 1) / stats.l3.transaction_size;

                    memory_traffic.record_write(sw::kpu::stats::MemoryLevel::EXTERNAL, result_bytes);
                    stats.dram.write_bytes += static_cast<int64_t>(result_bytes);
                    stats.dram.write_count += (result_bytes + stats.dram.transaction_size - 1) / stats.dram.transaction_size;

                    // Submit to memory controller for timing simulation
                    constexpr uint32_t CACHE_LINE_SIZE = 64;
                    for (size_t offset = 0; offset < result_bytes; offset += CACHE_LINE_SIZE) {
                        uint32_t chunk_size = std::min(static_cast<uint32_t>(CACHE_LINE_SIZE),
                                                       static_cast<uint32_t>(result_bytes - offset));
                        memory_controller_->submit_write(next_address + offset, nullptr, chunk_size, nullptr);
                    }

                    next_address += result_bytes;
                    stats.memory_bytes += static_cast<int64_t>(result_bytes);
                }

                // Note: FLOPs are already tracked in execute_op_behavioral

            } else {
                // Execute other operations behaviorally
                execute_op_behavioral(op, tensors, stats);

                // Simulate memory traffic for intermediate results
                if (op_output_names.size() > 0) {
                    auto result = tensors[output_name];
                    py::buffer_info result_buf = result.request();
                    size_t result_bytes = static_cast<size_t>(result_buf.size) * sizeof(float);

                    tensor_addresses[output_name] = next_address;

                    // Track writes through memory hierarchy for elementwise results
                    // L1 receives from compute
                    memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L1, result_bytes);
                    stats.l1.write_bytes += static_cast<int64_t>(result_bytes);
                    stats.l1.write_count += (result_bytes + stats.l1.transaction_size - 1) / stats.l1.transaction_size;

                    // L2 receives from L1
                    memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L2, result_bytes);
                    stats.l2.write_bytes += static_cast<int64_t>(result_bytes);
                    stats.l2.write_count += (result_bytes + stats.l2.transaction_size - 1) / stats.l2.transaction_size;

                    // L3 receives from L2
                    memory_traffic.record_write(sw::kpu::stats::MemoryLevel::L3, result_bytes);
                    stats.l3.write_bytes += static_cast<int64_t>(result_bytes);
                    stats.l3.write_count += (result_bytes + stats.l3.transaction_size - 1) / stats.l3.transaction_size;

                    // DRAM receives from L3
                    memory_traffic.record_write(sw::kpu::stats::MemoryLevel::EXTERNAL, result_bytes);
                    stats.dram.write_bytes += static_cast<int64_t>(result_bytes);
                    stats.dram.write_count += (result_bytes + stats.dram.transaction_size - 1) / stats.dram.transaction_size;

                    // Submit to memory controller for timing simulation
                    constexpr uint32_t CACHE_LINE_SIZE = 64;
                    for (size_t offset = 0; offset < result_bytes; offset += CACHE_LINE_SIZE) {
                        uint32_t chunk_size = std::min(static_cast<uint32_t>(CACHE_LINE_SIZE),
                                                       static_cast<uint32_t>(result_bytes - offset));
                        memory_controller_->submit_write(next_address + offset, nullptr, chunk_size, nullptr);
                    }

                    next_address += result_bytes;
                    stats.memory_bytes += static_cast<int64_t>(result_bytes);
                }
            }

            stats.ops_executed++;
        }

        // Drain memory controller to complete all pending operations
        memory_controller_->drain();

        // Collect statistics from transactional compute fabric
        const auto& fabric_stats = compute_fabric_->stats();

        stats.compute_cycles = static_cast<int64_t>(fabric_stats.total_compute_cycles);
        stats.busy_cycles = static_cast<int64_t>(fabric_stats.busy_cycles);
        stats.idle_cycles = static_cast<int64_t>(fabric_stats.idle_cycles);
        stats.stall_cycles = static_cast<int64_t>(fabric_stats.stall_cycles);

        // Collect statistics from transactional memory controller
        const auto& mem_stats = memory_controller_->stats();

        stats.memory_reads = static_cast<int64_t>(mem_stats.reads);
        stats.memory_writes = static_cast<int64_t>(mem_stats.writes);
        stats.page_hits = static_cast<int64_t>(mem_stats.page_hits);
        stats.page_misses = static_cast<int64_t>(mem_stats.page_empty + mem_stats.page_conflicts);
        stats.memory_latency_cycles = static_cast<int64_t>(mem_stats.total_latency);

        // Memory cycles from memory controller simulation
        stats.memory_cycles = stats.memory_latency_cycles;

        // Total cycles (max of compute and memory, since they can overlap)
        // For a realistic model, memory-bound workloads take longer
        stats.cycles = std::max(stats.compute_cycles, stats.memory_cycles);

        // Set elapsed cycles (T) for service rate calculations
        // This is the wall-clock execution time used to compute throughputs
        stats.elapsed_cycles = stats.cycles;

        // Estimate per-level cycle breakdown based on bandwidth hierarchy
        // Higher levels in hierarchy (closer to compute) have higher bandwidth
        // This is a simplified model; cycle-accurate would track actual latencies
        if (stats.elapsed_cycles > 0) {
            // DRAM is the bottleneck; L3/L2/L1 are faster proportionally
            double dram_fraction = 1.0;
            double l3_fraction = 0.5;   // L3 is 2x faster than DRAM
            double l2_fraction = 0.25;  // L2 is 4x faster than DRAM
            double l1_fraction = 0.125; // L1 is 8x faster than DRAM

            stats.dram.read_cycles = static_cast<int64_t>(stats.dram.read_bytes * dram_fraction / 8);
            stats.dram.write_cycles = static_cast<int64_t>(stats.dram.write_bytes * dram_fraction / 8);
            stats.l3.read_cycles = static_cast<int64_t>(stats.l3.read_bytes * l3_fraction / 8);
            stats.l3.write_cycles = static_cast<int64_t>(stats.l3.write_bytes * l3_fraction / 8);
            stats.l2.read_cycles = static_cast<int64_t>(stats.l2.read_bytes * l2_fraction / 8);
            stats.l2.write_cycles = static_cast<int64_t>(stats.l2.write_bytes * l2_fraction / 8);
            stats.l1.read_cycles = static_cast<int64_t>(stats.l1.read_bytes * l1_fraction / 8);
            stats.l1.write_cycles = static_cast<int64_t>(stats.l1.write_bytes * l1_fraction / 8);
        }

        // Store clock frequency in stats for reporting
        stats.clock_frequency_ghz = clock_frequency_ghz_;

        // Compute performance metrics using explicit clock frequency
        // GFLOPS = (FLOPs / cycles) * clock_ghz
        // At 1 GHz: 1 cycle = 1 ns, so FLOPs/cycle = GFLOPS
        // At 2 GHz: 1 cycle = 0.5 ns, so need to multiply by 2
        if (stats.cycles > 0) {
            stats.gflops = (static_cast<double>(stats.matmul_flops) / stats.cycles) * clock_frequency_ghz_;
            stats.utilization = fabric_stats.utilization();
            stats.efficiency = fabric_stats.mac_efficiency(compute_fabric_->peak_macs_per_cycle());
        }

        // Compute memory performance metrics
        uint64_t total_mem_requests = mem_stats.reads + mem_stats.writes;
        if (total_mem_requests > 0) {
            stats.page_hit_rate = mem_stats.hit_rate();
        }

        // Calculate memory bandwidth (bytes/cycle * clock_ghz = GB/s)
        if (stats.memory_cycles > 0) {
            int64_t total_bytes = stats.external_bytes + stats.memory_bytes;
            stats.memory_bandwidth_gbps = (static_cast<double>(total_bytes) / stats.memory_cycles) * clock_frequency_ghz_;
        }

        // Get output
        std::string output_name = output_names[0].cast<std::string>();
        auto result = tensors[output_name];

        return {result, stats.to_dict()};
    }
};

}  // anonymous namespace


// ============================================================================
// Python Module Definition
// ============================================================================

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native KPU simulator bindings for the kpu Python package";

    // Version
    m.attr("__version__") = "0.4.1";

    // Fidelity level constants
    m.attr("BEHAVIORAL") = FIDELITY_BEHAVIORAL;
    m.attr("TRANSACTIONAL") = FIDELITY_TRANSACTIONAL;
    m.attr("CYCLE_ACCURATE") = FIDELITY_CYCLE_ACCURATE;

    // NativeKPURuntime class
    py::class_<NativeKPURuntime>(m, "NativeRuntime",
        "Native KPU runtime for executing DFX programs")

        .def(py::init<int>(),
             py::arg("fidelity") = FIDELITY_BEHAVIORAL,
             "Create a native KPU runtime with the specified fidelity level")

        .def("set_fidelity", &NativeKPURuntime::set_fidelity,
             py::arg("fidelity"),
             "Set the simulation fidelity level")

        .def("get_fidelity", &NativeKPURuntime::get_fidelity,
             "Get the current simulation fidelity level")

        .def("set_clock_frequency", &NativeKPURuntime::set_clock_frequency,
             py::arg("ghz"),
             "Set the clock frequency in GHz.\n\n"
             "MUST be called before executing in TRANSACTIONAL or CYCLE_ACCURATE mode.\n"
             "This is used for:\n"
             "  - GFLOPS calculation: GFLOPS = (FLOPs / cycles) * clock_ghz\n"
             "  - Bandwidth calculation: GB/s = (bytes / cycles) * clock_ghz\n\n"
             "Args:\n"
             "    ghz: Clock frequency in GHz (e.g., 1.0 for 1 GHz)\n\n"
             "Raises:\n"
             "    ValueError: if ghz <= 0")

        .def("get_clock_frequency", &NativeKPURuntime::get_clock_frequency,
             "Get the clock frequency in GHz (-1.0 if not set)")

        .def("is_clock_frequency_set", &NativeKPURuntime::is_clock_frequency_set,
             "Check if clock frequency has been explicitly set")

        .def("execute", &NativeKPURuntime::execute,
             py::arg("dfx_json"),
             py::arg("inputs"),
             py::arg("mode") = "behavioral",
             "Execute a DFX program.\n\n"
             "Args:\n"
             "    dfx_json: DFX program as dict (from DFXProgram.to_dict())\n"
             "    inputs: List of numpy arrays for input tensors\n"
             "    mode: Execution mode ('behavioral', 'transactional', 'cycle_accurate')\n\n"
             "Returns:\n"
             "    Tuple of (result_array, stats_dict)")

        .def("get_config", &NativeKPURuntime::get_config,
             "Get the runtime configuration")

        .def("__repr__", [](const NativeKPURuntime& self) {
            auto config = self.get_config();
            std::string clock_str = self.is_clock_frequency_set()
                ? std::to_string(self.get_clock_frequency()) + " GHz"
                : "NOT SET";
            return "<NativeRuntime fidelity=" +
                   config["fidelity_name"].cast<std::string>() +
                   ", clock=" + clock_str + ">";
        });

    // Factory function matching Python's expected interface
    m.def("create_runtime", [](int fidelity) {
        return std::make_unique<NativeKPURuntime>(fidelity);
    }, py::arg("fidelity") = FIDELITY_BEHAVIORAL,
       "Create a native KPU runtime instance");

    // Check if native bindings are available
    m.def("is_available", []() { return true; },
          "Check if native bindings are available");

    // ========================================================================
    // v0.4.1: DFX Parser Functions
    // ========================================================================

    // Parse DFX JSON string and return program info as dict
    m.def("parse_dfx_json", [](const std::string& json_str) -> py::dict {
        try {
            sw::kpu::dfx::DFXParser parser;
            auto program = parser.parse_json(json_str);

            py::dict result;
            result["name"] = program.name;
            result["version"] = program.version;
            result["num_tensors"] = program.tensors.size();
            result["num_ops"] = program.ops.size();
            result["inputs"] = py::cast(program.inputs);
            result["outputs"] = py::cast(program.outputs);

            // Operation summary
            py::list ops_summary;
            for (const auto& op : program.ops) {
                py::dict op_info;
                op_info["opcode"] = sw::kpu::dfx::opcode_to_string(op.opcode);
                op_info["inputs"] = py::cast(op.inputs);
                op_info["outputs"] = py::cast(op.outputs);
                ops_summary.append(op_info);
            }
            result["ops"] = ops_summary;

            // Tensor summary
            py::dict tensors_info;
            for (const auto& [name, tensor] : program.tensors) {
                py::dict t_info;
                t_info["shape"] = py::cast(tensor.shape);
                t_info["dtype"] = sw::kpu::dfx::dtype_to_string(tensor.dtype);
                t_info["is_const"] = tensor.is_const;
                tensors_info[py::cast(name)] = t_info;
            }
            result["tensors"] = tensors_info;

            return result;
        } catch (const sw::kpu::dfx::ParseError& e) {
            throw std::runtime_error(std::string("DFX parse error: ") + e.what());
        }
    }, py::arg("json_str"),
       "Parse DFX JSON string and return program information.\n\n"
       "This is useful for validating DFX programs and debugging.\n\n"
       "Args:\n"
       "    json_str: JSON string containing DFX program\n\n"
       "Returns:\n"
       "    Dict with parsed program information");

    // Get DFX parser version
    m.def("dfx_parser_version", []() -> std::string {
        return "0.4.1";
    }, "Get the DFX parser version");

    // ========================================================================
    // v0.5.0: XUE Observation Architecture - C++ Backend
    // ========================================================================
    //
    // XUE Methodology: X (Throughput) → U (Utilization) → E (Efficiency)
    //
    // The Observation Architecture provides event hierarchies that aggregate
    // cleanly without logic on the datapath. Events are recorded by C++
    // simulator components (compute fabric, memory controller). Python
    // provides read-only access for operational analysis.

    // Get XUE event summary from C++ EventCollector
    m.def("get_xue_summary", []() -> py::dict {
        const auto& xue = sw::xue::EventCollector::instance();
        const auto& counter = xue.counters();

        py::dict result;

        // Aggregate metrics
        result["total_flops"] = counter.total_flops();
        result["total_bytes_moved"] = counter.total_bytes_moved();
        result["dram_bytes"] = counter.dram_bytes();
        result["arithmetic_intensity"] = counter.arithmetic_intensity();

        // Category breakdowns
        auto compute_stats = counter.get_category_stats(sw::xue::EventCategory::COMPUTE);
        auto memory_stats = counter.get_category_stats(sw::xue::EventCategory::MEMORY);
        auto datamovement_stats = counter.get_category_stats(sw::xue::EventCategory::DATA_MOVEMENT);
        auto sync_stats = counter.get_category_stats(sw::xue::EventCategory::SYNCHRONIZATION);

        // Compute category
        py::dict compute;
        compute["total_events"] = compute_stats.total_events;
        compute["total_flops"] = compute_stats.total_flops;
        compute["total_cycles"] = compute_stats.total_cycles;
        result["compute"] = compute;

        // Compute subcategories (v0.4.0: NAMED_OP for matmul, ALU_PRIMITIVE for elementwise)
        auto matmul_stats = counter.get_compute_subcategory_stats(sw::xue::ComputeSubcategory::NAMED_OP);
        auto elem_stats = counter.get_compute_subcategory_stats(sw::xue::ComputeSubcategory::ALU_PRIMITIVE);
        auto reduce_stats = counter.get_compute_subcategory_stats(sw::xue::ComputeSubcategory::REDUCTION);
        auto special_stats = counter.get_compute_subcategory_stats(sw::xue::ComputeSubcategory::SPECIAL);

        py::dict compute_breakdown;
        compute_breakdown["matmul_events"] = matmul_stats.total_events;
        compute_breakdown["matmul_flops"] = matmul_stats.total_flops;
        compute_breakdown["elementwise_events"] = elem_stats.total_events;
        compute_breakdown["elementwise_flops"] = elem_stats.total_flops;
        compute_breakdown["reduction_events"] = reduce_stats.total_events;
        compute_breakdown["reduction_flops"] = reduce_stats.total_flops;
        compute_breakdown["special_events"] = special_stats.total_events;
        compute_breakdown["special_flops"] = special_stats.total_flops;
        result["compute_breakdown"] = compute_breakdown;

        // Memory category
        py::dict memory;
        memory["total_events"] = memory_stats.total_events;
        memory["total_bytes"] = memory_stats.total_bytes;
        memory["total_cycles"] = memory_stats.total_cycles;
        result["memory"] = memory;

        // Memory subcategories (per-level hierarchy)
        auto dram_stats = counter.get_memory_subcategory_stats(sw::xue::MemorySubcategory::DRAM);
        auto l3_stats = counter.get_memory_subcategory_stats(sw::xue::MemorySubcategory::L3);
        auto l2_stats = counter.get_memory_subcategory_stats(sw::xue::MemorySubcategory::L2);
        auto l1_stats = counter.get_memory_subcategory_stats(sw::xue::MemorySubcategory::L1);

        py::dict memory_hierarchy;
        py::dict dram, l3, l2, l1;

        dram["events"] = dram_stats.total_events;
        dram["bytes"] = dram_stats.total_bytes;
        dram["cycles"] = dram_stats.total_cycles;
        memory_hierarchy["dram"] = dram;

        l3["events"] = l3_stats.total_events;
        l3["bytes"] = l3_stats.total_bytes;
        l3["cycles"] = l3_stats.total_cycles;
        memory_hierarchy["l3"] = l3;

        l2["events"] = l2_stats.total_events;
        l2["bytes"] = l2_stats.total_bytes;
        l2["cycles"] = l2_stats.total_cycles;
        memory_hierarchy["l2"] = l2;

        l1["events"] = l1_stats.total_events;
        l1["bytes"] = l1_stats.total_bytes;
        l1["cycles"] = l1_stats.total_cycles;
        memory_hierarchy["l1"] = l1;

        result["memory_hierarchy"] = memory_hierarchy;

        // Data movement
        py::dict data_movement;
        data_movement["total_events"] = datamovement_stats.total_events;
        data_movement["total_bytes"] = datamovement_stats.total_bytes;
        result["data_movement"] = data_movement;

        // Synchronization
        py::dict sync;
        sync["total_events"] = sync_stats.total_events;
        sync["stall_cycles"] = sync_stats.total_cycles;
        result["synchronization"] = sync;

        // XUE collector state
        result["enabled"] = xue.is_enabled();
        result["current_cycle"] = xue.get_cycle();

        return result;
    }, "Get XUE event summary from C++ EventCollector.\n\n"
       "Returns a dict with:\n"
       "  - total_flops: Total floating point operations\n"
       "  - total_bytes_moved: Total bytes moved through memory hierarchy\n"
       "  - dram_bytes: External memory traffic\n"
       "  - arithmetic_intensity: FLOP/byte ratio\n"
       "  - compute: Compute event category summary\n"
       "  - compute_breakdown: Per-operation-type breakdown\n"
       "  - memory: Memory event category summary\n"
       "  - memory_hierarchy: Per-level (DRAM/L3/L2/L1) breakdown\n"
       "  - data_movement: Data movement event summary\n"
       "  - synchronization: Synchronization/stall events");

    // Run operational analysis using roofline model
    m.def("get_operational_analysis", [](
            double peak_gflops,
            double dram_bandwidth_gbs,
            double clock_ghz) -> py::dict {
        // Configure hardware model
        sw::xue::HardwareModel hw;
        hw.peak_gflops = peak_gflops;
        hw.dram_bandwidth_gbs = dram_bandwidth_gbs;
        hw.clock_ghz = clock_ghz;

        // Run analysis
        sw::xue::OperationalAnalyzer analyzer(hw);
        const auto& counter = sw::xue::EventCollector::instance().counters();
        auto result = analyzer.analyze(counter);

        py::dict d;

        // Workload characteristics
        d["total_flops"] = result.total_flops;
        d["dram_bytes"] = result.dram_bytes;
        d["l3_bytes"] = result.l3_bytes;
        d["l2_bytes"] = result.l2_bytes;
        d["l1_bytes"] = result.l1_bytes;
        d["arithmetic_intensity"] = result.arithmetic_intensity;
        d["l3_arithmetic_intensity"] = result.l3_arithmetic_intensity;

        // Roofline predictions
        d["predicted_gflops"] = result.predicted_gflops;
        d["predicted_cycles"] = result.predicted_cycles;
        d["predicted_runtime_us"] = result.predicted_runtime_us;
        d["predicted_bottleneck"] = result.predicted_bottleneck;

        // Event breakdown
        d["matmul_events"] = result.matmul_events;
        d["elementwise_events"] = result.elementwise_events;
        d["reduction_events"] = result.reduction_events;
        d["memory_events"] = result.memory_events;
        d["sync_events"] = result.sync_events;

        // Hardware model info
        py::dict hw_info;
        hw_info["peak_gflops"] = hw.peak_gflops;
        hw_info["dram_bandwidth_gbs"] = hw.dram_bandwidth_gbs;
        hw_info["clock_ghz"] = hw.clock_ghz;
        hw_info["ridge_point_dram"] = hw.ridge_point_dram();
        hw_info["ridge_point_l3"] = hw.ridge_point_l3();
        hw_info["ridge_point_l2"] = hw.ridge_point_l2();
        d["hardware"] = hw_info;

        return d;
    }, py::arg("peak_gflops") = 1024.0,
       py::arg("dram_bandwidth_gbs") = 64.0,
       py::arg("clock_ghz") = 1.0,
       "Run operational analysis using roofline model.\n\n"
       "This analyzes the collected XUE events and predicts performance\n"
       "using the roofline model.\n\n"
       "Args:\n"
       "    peak_gflops: Peak compute throughput (default: 1024 for 16x16 systolic)\n"
       "    dram_bandwidth_gbs: DRAM bandwidth in GB/s (default: 64)\n"
       "    clock_ghz: Clock frequency in GHz (default: 1.0)\n\n"
       "Returns:\n"
       "    Dict with workload characteristics and performance predictions");

    // Validate operational analysis against actual simulation
    m.def("validate_operational_analysis", [](
            double actual_gflops,
            uint64_t actual_cycles,
            double peak_gflops,
            double dram_bandwidth_gbs,
            double clock_ghz) -> py::dict {
        // Configure hardware model
        sw::xue::HardwareModel hw;
        hw.peak_gflops = peak_gflops;
        hw.dram_bandwidth_gbs = dram_bandwidth_gbs;
        hw.clock_ghz = clock_ghz;

        // Run validation
        sw::xue::OperationalAnalyzer analyzer(hw);
        const auto& counter = sw::xue::EventCollector::instance().counters();
        auto result = analyzer.validate(counter, actual_gflops, actual_cycles);

        py::dict d;

        // Prediction vs actual
        d["predicted_gflops"] = result.prediction.predicted_gflops;
        d["predicted_cycles"] = result.prediction.predicted_cycles;
        d["actual_gflops"] = result.actual_gflops;
        d["actual_cycles"] = result.actual_cycles;

        // Error analysis
        d["gflops_error_percent"] = result.gflops_error_percent;
        d["cycles_error_percent"] = result.cycles_error_percent;
        d["within_10_percent"] = result.within_10_percent;

        // Efficiency
        d["roofline_efficiency"] = result.prediction.roofline_efficiency;
        d["bottleneck"] = result.prediction.predicted_bottleneck;
        d["arithmetic_intensity"] = result.prediction.arithmetic_intensity;

        return d;
    }, py::arg("actual_gflops"),
       py::arg("actual_cycles"),
       py::arg("peak_gflops") = 1024.0,
       py::arg("dram_bandwidth_gbs") = 64.0,
       py::arg("clock_ghz") = 1.0,
       "Validate operational analysis against actual simulation results.\n\n"
       "Args:\n"
       "    actual_gflops: Achieved GFLOPS from simulation\n"
       "    actual_cycles: Actual cycles from simulation\n"
       "    peak_gflops: Peak compute throughput\n"
       "    dram_bandwidth_gbs: DRAM bandwidth in GB/s\n"
       "    clock_ghz: Clock frequency in GHz\n\n"
       "Returns:\n"
       "    Dict with prediction vs actual comparison and error analysis");

    // Reset XUE event counters
    m.def("reset_xue_counters", []() {
        sw::xue::EventCollector::instance().reset();
    }, "Reset all XUE event counters.\n\n"
       "Call this before starting a new workload to get fresh event counts.");

    // Enable/disable XUE collection
    m.def("set_xue_enabled", [](bool enabled) {
        sw::xue::EventCollector::instance().set_enabled(enabled);
    }, py::arg("enabled"),
       "Enable or disable XUE event collection.\n\n"
       "When disabled, events are not recorded (zero overhead).\n"
       "This is useful for performance-critical code paths.");

    // Check if XUE is enabled
    m.def("is_xue_enabled", []() -> bool {
        return sw::xue::EventCollector::instance().is_enabled();
    }, "Check if XUE event collection is enabled.");

    // Get XUE version
    m.def("xue_version", []() -> std::string {
        return "0.5.0";
    }, "Get the XUE Observation Architecture version");

    // ========================================================================
    // STANDALONE NATIVE OPERATIONS
    // ========================================================================
    // These functions enable direct Tensor operations to route through C++
    // BehavioralComputeFabric instead of NumPy, ensuring XUE event recording
    // and proper simulation modeling.
    // ========================================================================

    // Global BehavioralComputeFabric for direct tensor operations
    static std::unique_ptr<sw::kpu::BehavioralComputeFabric> g_compute_fabric;

    auto ensure_fabric = []() -> sw::kpu::BehavioralComputeFabric& {
        if (!g_compute_fabric) {
            sw::kpu::ComputeFabricConfig config;
            config.fidelity = sw::kpu::SimulationFidelity::BEHAVIORAL;
            g_compute_fabric = std::make_unique<sw::kpu::BehavioralComputeFabric>(config, 0);
        }
        return *g_compute_fabric;
    };

    // Native matmul: C = A @ B
    m.def("native_matmul", [ensure_fabric](py::array_t<float> A, py::array_t<float> B) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();
        py::buffer_info b_buf = B.request();

        if (a_buf.ndim < 2 || b_buf.ndim < 2) {
            throw std::runtime_error("native_matmul requires 2D arrays");
        }

        // Get dimensions
        size_t M = a_buf.shape[a_buf.ndim - 2];
        size_t K = a_buf.shape[a_buf.ndim - 1];
        size_t N = b_buf.shape[b_buf.ndim - 1];

        if (static_cast<size_t>(b_buf.shape[b_buf.ndim - 2]) != K) {
            throw std::runtime_error("Matrix dimensions don't match for matmul");
        }

        // Allocate and zero-initialize output
        std::vector<py::ssize_t> c_shape = {static_cast<py::ssize_t>(M), static_cast<py::ssize_t>(N)};
        py::array_t<float> C(c_shape);
        py::buffer_info c_buf = C.request();
        std::memset(c_buf.ptr, 0, M * N * sizeof(float));

        // Execute via BehavioralComputeFabric
        sw::kpu::MatMulDescriptor desc;
        desc.m = static_cast<uint32_t>(M);
        desc.n = static_cast<uint32_t>(N);
        desc.k = static_cast<uint32_t>(K);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_matmul(desc,
                            static_cast<float*>(a_buf.ptr),
                            static_cast<float*>(b_buf.ptr),
                            static_cast<float*>(c_buf.ptr),
                            nullptr);
        fabric.drain();

        return C;
    }, py::arg("A"), py::arg("B"),
       "Native matrix multiplication via C++ BehavioralComputeFabric.\n\n"
       "Routes through C++ simulation with XUE event recording.\n\n"
       "Args:\n"
       "    A: Left matrix [M, K]\n"
       "    B: Right matrix [K, N]\n\n"
       "Returns:\n"
       "    Result matrix [M, N]");

    // Native add: C = A + B (element-wise)
    m.def("native_add", [ensure_fabric](py::array_t<float> A, py::array_t<float> B) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();
        py::buffer_info b_buf = B.request();

        // For now, require same shape (no broadcasting)
        if (a_buf.size != b_buf.size) {
            throw std::runtime_error("native_add requires arrays of same size (no broadcasting yet)");
        }

        // Allocate output with same shape as A
        py::array_t<float> C(a_buf.shape);
        py::buffer_info c_buf = C.request();

        // Execute via BehavioralComputeFabric
        sw::kpu::ElementwiseDescriptor desc;
        desc.op = sw::kpu::ElementwiseOp::ADD;
        desc.count = static_cast<uint32_t>(a_buf.size);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_elementwise(desc,
                                 static_cast<float*>(a_buf.ptr),
                                 static_cast<float*>(b_buf.ptr),
                                 static_cast<float*>(c_buf.ptr),
                                 nullptr);
        fabric.drain();

        return C;
    }, py::arg("A"), py::arg("B"),
       "Native element-wise addition via C++ BehavioralComputeFabric.");

    // Native sub: C = A - B (element-wise)
    m.def("native_sub", [ensure_fabric](py::array_t<float> A, py::array_t<float> B) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();
        py::buffer_info b_buf = B.request();

        if (a_buf.size != b_buf.size) {
            throw std::runtime_error("native_sub requires arrays of same size");
        }

        py::array_t<float> C(a_buf.shape);
        py::buffer_info c_buf = C.request();

        sw::kpu::ElementwiseDescriptor desc;
        desc.op = sw::kpu::ElementwiseOp::SUB;
        desc.count = static_cast<uint32_t>(a_buf.size);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_elementwise(desc,
                                 static_cast<float*>(a_buf.ptr),
                                 static_cast<float*>(b_buf.ptr),
                                 static_cast<float*>(c_buf.ptr),
                                 nullptr);
        fabric.drain();

        return C;
    }, py::arg("A"), py::arg("B"),
       "Native element-wise subtraction via C++ BehavioralComputeFabric.");

    // Native mul: C = A * B (element-wise)
    m.def("native_mul", [ensure_fabric](py::array_t<float> A, py::array_t<float> B) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();
        py::buffer_info b_buf = B.request();

        if (a_buf.size != b_buf.size) {
            throw std::runtime_error("native_mul requires arrays of same size");
        }

        py::array_t<float> C(a_buf.shape);
        py::buffer_info c_buf = C.request();

        sw::kpu::ElementwiseDescriptor desc;
        desc.op = sw::kpu::ElementwiseOp::MUL;
        desc.count = static_cast<uint32_t>(a_buf.size);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_elementwise(desc,
                                 static_cast<float*>(a_buf.ptr),
                                 static_cast<float*>(b_buf.ptr),
                                 static_cast<float*>(c_buf.ptr),
                                 nullptr);
        fabric.drain();

        return C;
    }, py::arg("A"), py::arg("B"),
       "Native element-wise multiplication via C++ BehavioralComputeFabric.");

    // Native div: C = A / B (element-wise)
    m.def("native_div", [ensure_fabric](py::array_t<float> A, py::array_t<float> B) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();
        py::buffer_info b_buf = B.request();

        if (a_buf.size != b_buf.size) {
            throw std::runtime_error("native_div requires arrays of same size");
        }

        py::array_t<float> C(a_buf.shape);
        py::buffer_info c_buf = C.request();

        sw::kpu::ElementwiseDescriptor desc;
        desc.op = sw::kpu::ElementwiseOp::DIV;
        desc.count = static_cast<uint32_t>(a_buf.size);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_elementwise(desc,
                                 static_cast<float*>(a_buf.ptr),
                                 static_cast<float*>(b_buf.ptr),
                                 static_cast<float*>(c_buf.ptr),
                                 nullptr);
        fabric.drain();

        return C;
    }, py::arg("A"), py::arg("B"),
       "Native element-wise division via C++ BehavioralComputeFabric.");

    // Native relu: B = max(0, A)
    m.def("native_relu", [ensure_fabric](py::array_t<float> A) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();

        py::array_t<float> B(a_buf.shape);
        py::buffer_info b_buf = B.request();

        sw::kpu::ElementwiseDescriptor desc;
        desc.op = sw::kpu::ElementwiseOp::RELU;
        desc.count = static_cast<uint32_t>(a_buf.size);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_elementwise(desc,
                                 static_cast<float*>(a_buf.ptr),
                                 nullptr,
                                 static_cast<float*>(b_buf.ptr),
                                 nullptr);
        fabric.drain();

        return B;
    }, py::arg("A"),
       "Native ReLU activation via C++ BehavioralComputeFabric.");

    // Native sigmoid: B = 1 / (1 + exp(-A))
    m.def("native_sigmoid", [ensure_fabric](py::array_t<float> A) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();

        py::array_t<float> B(a_buf.shape);
        py::buffer_info b_buf = B.request();

        sw::kpu::ElementwiseDescriptor desc;
        desc.op = sw::kpu::ElementwiseOp::SIGMOID;
        desc.count = static_cast<uint32_t>(a_buf.size);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_elementwise(desc,
                                 static_cast<float*>(a_buf.ptr),
                                 nullptr,
                                 static_cast<float*>(b_buf.ptr),
                                 nullptr);
        fabric.drain();

        return B;
    }, py::arg("A"),
       "Native sigmoid activation via C++ BehavioralComputeFabric.");

    // Native tanh
    m.def("native_tanh", [ensure_fabric](py::array_t<float> A) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();

        py::array_t<float> B(a_buf.shape);
        py::buffer_info b_buf = B.request();

        sw::kpu::ElementwiseDescriptor desc;
        desc.op = sw::kpu::ElementwiseOp::TANH;
        desc.count = static_cast<uint32_t>(a_buf.size);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_elementwise(desc,
                                 static_cast<float*>(a_buf.ptr),
                                 nullptr,
                                 static_cast<float*>(b_buf.ptr),
                                 nullptr);
        fabric.drain();

        return B;
    }, py::arg("A"),
       "Native tanh activation via C++ BehavioralComputeFabric.");

    // Native gelu
    m.def("native_gelu", [ensure_fabric](py::array_t<float> A) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();

        py::array_t<float> B(a_buf.shape);
        py::buffer_info b_buf = B.request();

        sw::kpu::ElementwiseDescriptor desc;
        desc.op = sw::kpu::ElementwiseOp::GELU;
        desc.count = static_cast<uint32_t>(a_buf.size);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_elementwise(desc,
                                 static_cast<float*>(a_buf.ptr),
                                 nullptr,
                                 static_cast<float*>(b_buf.ptr),
                                 nullptr);
        fabric.drain();

        return B;
    }, py::arg("A"),
       "Native GELU activation via C++ BehavioralComputeFabric.");

    // Native silu (swish): B = A * sigmoid(A)
    m.def("native_silu", [ensure_fabric](py::array_t<float> A) -> py::array_t<float> {
        py::buffer_info a_buf = A.request();

        py::array_t<float> B(a_buf.shape);
        py::buffer_info b_buf = B.request();

        sw::kpu::ElementwiseDescriptor desc;
        desc.op = sw::kpu::ElementwiseOp::SILU;
        desc.count = static_cast<uint32_t>(a_buf.size);
        desc.dtype = sw::kpu::DataType::FLOAT32;

        auto& fabric = ensure_fabric();
        fabric.submit_elementwise(desc,
                                 static_cast<float*>(a_buf.ptr),
                                 nullptr,
                                 static_cast<float*>(b_buf.ptr),
                                 nullptr);
        fabric.drain();

        return B;
    }, py::arg("A"),
       "Native SiLU (Swish) activation via C++ BehavioralComputeFabric.");

    // Check if native ops are available
    m.def("native_ops_available", []() -> bool {
        return true;
    }, "Check if native tensor operations are available.");
}
