// python/kpu/_native/kpu_native.cpp
// pybind11 bindings for KPU simulator integration with the kpu Python package
//
// This module provides the native backend for the @kpu.compile decorator,
// enabling execution on the C++ kpu-sim library.

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
 * @brief Execution statistics returned to Python
 */
struct NativeExecutionStats {
    int64_t cycles = 0;
    int64_t compute_cycles = 0;
    int64_t memory_cycles = 0;
    int64_t matmul_flops = 0;
    int64_t memory_bytes = 0;
    int64_t ops_executed = 0;

    py::dict to_dict() const {
        py::dict d;
        d["cycles"] = cycles;
        d["compute_cycles"] = compute_cycles;
        d["memory_cycles"] = memory_cycles;
        d["matmul_flops"] = matmul_flops;
        d["memory_bytes"] = memory_bytes;
        d["ops_executed"] = ops_executed;
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
 * For TRANSACTIONAL and CYCLE_ACCURATE modes, it will integrate
 * with the C++ kpu-sim library when available.
 */
class NativeKPURuntime {
public:
    explicit NativeKPURuntime(int fidelity = FIDELITY_BEHAVIORAL)
        : fidelity_(fidelity) {
    }

    void set_fidelity(int fidelity) {
        fidelity_ = fidelity;
    }

    int get_fidelity() const {
        return fidelity_;
    }

    /**
     * @brief Execute a DFX program
     *
     * @param dfx_json DFX program as Python dict (from DFXProgram.to_dict())
     * @param inputs List of numpy arrays for input tensors
     * @param mode Execution mode ("behavioral", "transactional", "cycle_accurate")
     * @return Tuple of (result numpy array, stats dict)
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

        // For transactional/cycle-accurate, use behavioral with timing estimates
        // TODO: Integrate with C++ kpu-sim for actual timing simulation
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
        return config;
    }

private:
    int fidelity_;

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
            ssize_t M = a_buf.shape[a_buf.ndim - 2];
            ssize_t K = a_buf.shape[a_buf.ndim - 1];
            ssize_t N = b_buf.shape[b_buf.ndim - 1];

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
     * @brief Execute with timing estimates (placeholder for full simulation)
     */
    std::pair<py::array_t<float>, py::dict> execute_simulated(
        const py::dict& dfx_json,
        const std::vector<py::array_t<float>>& inputs,
        const std::string& mode,
        NativeExecutionStats& stats
    ) {
        // Execute behaviorally first to get correct results
        auto [result, _] = execute_behavioral(dfx_json, inputs, stats);

        // Add timing estimates based on FLOPs
        // Assume a 16x16 systolic array running at 1GHz, 2 ops/MAC
        int64_t peak_flops_per_cycle = 16 * 16 * 2;

        stats.compute_cycles = stats.matmul_flops / peak_flops_per_cycle;
        stats.memory_cycles = stats.compute_cycles / 4;  // Assume 25% memory overhead
        stats.cycles = stats.compute_cycles + stats.memory_cycles;

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
    m.attr("__version__") = "0.1.0";

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
            return "<NativeRuntime fidelity=" +
                   config["fidelity_name"].cast<std::string>() + ">";
        });

    // Factory function matching Python's expected interface
    m.def("create_runtime", [](int fidelity) {
        return std::make_unique<NativeKPURuntime>(fidelity);
    }, py::arg("fidelity") = FIDELITY_BEHAVIORAL,
       "Create a native KPU runtime instance");

    // Check if native bindings are available
    m.def("is_available", []() { return true; },
          "Check if native bindings are available");
}
