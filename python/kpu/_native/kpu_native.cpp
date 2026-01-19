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

// KPU Simulator includes for transactional models
#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/fidelity/component_config.hpp>
#include <sw/kpu/models/interfaces/compute_fabric_interface.hpp>
#include <sw/kpu/models/interfaces/memory_controller_interface.hpp>
#include <sw/kpu/models/transactional/compute/compute_fabric.hpp>
#include <sw/kpu/models/transactional/memory/memory_controller.hpp>
#include <sw/kpu/stats/memory_traffic.hpp>

// v0.4.1: DFX parser for C++ DFX program representation
#include <sw/kpu/dfx/dfx_parser.hpp>

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
 * @brief Per-level memory statistics
 *
 * Tracks reads, writes, bytes, and transaction sizes for each memory level.
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

    // Service rate: bytes/cycle (effective bandwidth)
    double service_rate(int64_t elapsed_cycles) const {
        return elapsed_cycles > 0 ? static_cast<double>(total_bytes()) / elapsed_cycles : 0.0;
    }

    // Throughput: transactions/cycle
    double throughput(int64_t elapsed_cycles) const {
        return elapsed_cycles > 0 ? static_cast<double>(total_count()) / elapsed_cycles : 0.0;
    }
};

/**
 * @brief Execution statistics returned to Python
 *
 * Extended for v0.4.0+ TRANSACTIONAL runtime with detailed metrics
 * from the C++ transactional simulation models.
 *
 * XUE Event Tracking:
 *   - Per-level memory hierarchy stats (DRAM, L3, L2, L1)
 *   - Transaction sizes for service rate calculations
 *   - Elapsed cycles (T) for throughput analysis
 */
struct NativeExecutionStats {
    // Basic timing
    int64_t cycles = 0;
    int64_t compute_cycles = 0;
    int64_t memory_cycles = 0;
    int64_t elapsed_cycles = 0;  // Wall clock cycles (T) for service rates

    // Detailed cycle breakdown
    int64_t busy_cycles = 0;
    int64_t idle_cycles = 0;
    int64_t stall_cycles = 0;

    // Compute metrics
    int64_t matmul_flops = 0;
    int64_t total_macs = 0;
    int64_t matmul_count = 0;

    // Memory hierarchy statistics (XUE events)
    LevelMemoryStats dram;   // External/DRAM
    LevelMemoryStats l3;     // L3 buffer
    LevelMemoryStats l2;     // L2 buffer
    LevelMemoryStats l1;     // L1 stream

    // Legacy memory metrics (for backward compatibility)
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

    // Helper to create dict for a memory level
    py::dict level_to_dict(const LevelMemoryStats& level, const char* name) const {
        py::dict d;
        d["read_count"] = level.read_count;
        d["write_count"] = level.write_count;
        d["read_bytes"] = level.read_bytes;
        d["write_bytes"] = level.write_bytes;
        d["read_cycles"] = level.read_cycles;
        d["write_cycles"] = level.write_cycles;
        d["total_bytes"] = level.total_bytes();
        d["total_count"] = level.total_count();
        d["transaction_size"] = level.transaction_size;
        d["service_rate"] = level.service_rate(elapsed_cycles);
        d["throughput"] = level.throughput(elapsed_cycles);
        return d;
    }

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

        // Memory hierarchy stats (XUE events)
        d["dram"] = level_to_dict(dram, "dram");
        d["l3"] = level_to_dict(l3, "l3");
        d["l2"] = level_to_dict(l2, "l2");
        d["l1"] = level_to_dict(l1, "l1");

        // Legacy memory metrics
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

        // Aggregate service rates (bytes/cycle * clock_ghz = GB/s)
        d["dram_service_rate_gbps"] = dram.service_rate(elapsed_cycles) * clock_frequency_ghz;
        d["l3_service_rate_gbps"] = l3.service_rate(elapsed_cycles) * clock_frequency_ghz;
        d["l2_service_rate_gbps"] = l2.service_rate(elapsed_cycles) * clock_frequency_ghz;
        d["l1_service_rate_gbps"] = l1.service_rate(elapsed_cycles) * clock_frequency_ghz;

        return d;
    }
};

/**
 * @brief Native KPU runtime that executes DFX programs
 *
 * This class provides the interface between the Python kpu package
 * and execution on the KPU hardware model.
 *
 * For BEHAVIORAL mode, it uses NumPy for actual computation.
 * For TRANSACTIONAL mode, it uses the C++ TransactionalComputeFabric
 * for accurate throughput-based timing simulation.
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
        // Initialize transactional compute fabric with default config
        init_transactional_models();
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

        // For behavioral mode, we use pure computation via NumPy
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
     * @brief Execute a single DFX operation behaviorally using NumPy
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

        // Import numpy
        py::module np = py::module::import("numpy");

        if (opcode == "matmul") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto A = tensors[a_name];
            auto B = tensors[b_name];

            py::buffer_info a_buf = A.request();
            py::buffer_info b_buf = B.request();

            // Get dimensions for FLOP counting
            py::ssize_t M = a_buf.shape[a_buf.ndim - 2];
            py::ssize_t K = a_buf.shape[a_buf.ndim - 1];
            py::ssize_t N = b_buf.shape[b_buf.ndim - 1];

            // Compute result using numpy
            py::array_t<float> C = np.attr("matmul")(A, B).cast<py::array_t<float>>();
            tensors[output_name] = C;

            // Track FLOPs: 2*M*N*K (multiply-add per element)
            stats.matmul_flops += 2 * M * N * K;

        } else if (opcode == "relu") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::array_t<float> Y = np.attr("maximum")(X, 0.0f).cast<py::array_t<float>>();
            tensors[output_name] = Y;

        } else if (opcode == "gelu") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            double sqrt_2_pi = std::sqrt(2.0 / M_PI);
            auto x3 = np.attr("power")(X, 3);
            auto inner = np.attr("add")(X, np.attr("multiply")(0.044715, x3));
            auto tanh_arg = np.attr("multiply")(sqrt_2_pi, inner);
            auto tanh_val = np.attr("tanh")(tanh_arg);
            auto factor = np.attr("add")(1.0, tanh_val);
            auto Y = np.attr("multiply")(X, np.attr("multiply")(0.5, factor));

            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "silu") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            // SiLU: x * sigmoid(x)
            auto neg_x = np.attr("negative")(X);
            auto exp_neg_x = np.attr("exp")(neg_x);
            auto sigmoid = np.attr("divide")(1.0, np.attr("add")(1.0, exp_neg_x));
            auto Y = np.attr("multiply")(X, sigmoid);

            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "sigmoid") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            auto neg_x = np.attr("negative")(X);
            auto exp_neg_x = np.attr("exp")(neg_x);
            auto Y = np.attr("divide")(1.0, np.attr("add")(1.0, exp_neg_x));

            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "tanh") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            py::array_t<float> Y = np.attr("tanh")(X).cast<py::array_t<float>>();
            tensors[output_name] = Y;

        } else if (opcode == "softmax") {
            std::string input_name = input_names[0].cast<std::string>();
            auto X = tensors[input_name];

            // Numerically stable softmax
            auto max_x = np.attr("max")(X, py::arg("axis") = -1, py::arg("keepdims") = true);
            auto shifted = np.attr("subtract")(X, max_x);
            auto exp_x = np.attr("exp")(shifted);
            auto sum_exp = np.attr("sum")(exp_x, py::arg("axis") = -1, py::arg("keepdims") = true);
            auto Y = np.attr("divide")(exp_x, sum_exp);

            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "add") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto Y = np.attr("add")(tensors[a_name], tensors[b_name]);
            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "sub") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto Y = np.attr("subtract")(tensors[a_name], tensors[b_name]);
            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "mul") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto Y = np.attr("multiply")(tensors[a_name], tensors[b_name]);
            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "div") {
            std::string a_name = input_names[0].cast<std::string>();
            std::string b_name = input_names[1].cast<std::string>();

            auto Y = np.attr("divide")(tensors[a_name], tensors[b_name]);
            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "neg") {
            std::string input_name = input_names[0].cast<std::string>();
            auto Y = np.attr("negative")(tensors[input_name]);
            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "exp") {
            std::string input_name = input_names[0].cast<std::string>();
            auto Y = np.attr("exp")(tensors[input_name]);
            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "log") {
            std::string input_name = input_names[0].cast<std::string>();
            auto Y = np.attr("log")(tensors[input_name]);
            tensors[output_name] = Y.cast<py::array_t<float>>();

        } else if (opcode == "sqrt") {
            std::string input_name = input_names[0].cast<std::string>();
            auto Y = np.attr("sqrt")(tensors[input_name]);
            tensors[output_name] = Y.cast<py::array_t<float>>();

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
                // Use transactional compute fabric for matmul timing
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

                // Execute behavioral computation
                py::array_t<float> C = np.attr("matmul")(A, B).cast<py::array_t<float>>();
                tensors[output_name] = C;

                // Allocate address for output tensor
                py::buffer_info c_buf = C.request();
                size_t c_bytes = static_cast<size_t>(c_buf.size) * sizeof(float);
                tensor_addresses[output_name] = next_address;

                // Submit to transactional compute fabric for timing
                sw::kpu::MatMulDescriptor desc;
                desc.m = M;
                desc.n = N;
                desc.k = K;

                // Get data pointers
                float* a_ptr = static_cast<float*>(a_buf.ptr);
                float* b_ptr = static_cast<float*>(b_buf.ptr);
                float* c_ptr = static_cast<float*>(c_buf.ptr);

                // Submit matmul to transactional fabric
                compute_fabric_->submit_matmul(desc, a_ptr, b_ptr, c_ptr, nullptr);

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
}
