// Graph Executor Implementation
// High-level execution API for computational graphs

#include <sw/runtime/executor.hpp>

#include <stdexcept>
#include <sstream>

namespace sw::runtime {

using namespace sw::kpu;

// ============================================================================
// Constructor / Destructor
// ============================================================================

GraphExecutor::GraphExecutor(KPURuntime* runtime)
    : runtime_(runtime) {

    if (!runtime_) {
        throw std::invalid_argument("GraphExecutor: runtime cannot be null");
    }
}

GraphExecutor::~GraphExecutor() {
    release();
}

GraphExecutor::GraphExecutor(GraphExecutor&&) noexcept = default;
GraphExecutor& GraphExecutor::operator=(GraphExecutor&&) noexcept = default;

// ============================================================================
// Kernel Setup
// ============================================================================

void GraphExecutor::set_kernel(const Kernel& kernel) {
    // Free any existing allocations
    free_tensors();

    // Copy the kernel
    kernel_ = std::make_unique<Kernel>(kernel);

    // Allocate tensors for all arguments
    allocate_tensors();
}

void GraphExecutor::create_matmul(Size M, Size N, Size K, DataType dtype) {
    set_kernel(Kernel::create_matmul(M, N, K, dtype));
}

void GraphExecutor::create_mlp(Size M, Size N, Size K,
                                ActivationType activation,
                                bool has_bias,
                                DataType dtype) {
    set_kernel(Kernel::create_mlp(M, N, K, activation, has_bias, dtype));
}

void GraphExecutor::allocate_tensors() {
    if (!kernel_) return;

    bindings_.clear();
    arg_addresses_.clear();

    const auto& args = kernel_->arguments();
    arg_addresses_.reserve(args.size());

    for (const auto& arg : args) {
        // Create binding
        TensorBinding binding;
        binding.name = arg.name;
        binding.shape = arg.shape;
        binding.dtype = arg.dtype;
        binding.size_bytes = arg.size_bytes;

        // Allocate device memory
        Address addr = runtime_->malloc(arg.size_bytes);
        if (addr == 0) {
            // Cleanup on failure
            free_tensors();
            throw std::runtime_error("GraphExecutor: failed to allocate memory for " + arg.name);
        }

        binding.device_address = addr;
        bindings_[arg.name] = binding;
        arg_addresses_.push_back(addr);
    }
}

void GraphExecutor::free_tensors() {
    for (const auto& [name, binding] : bindings_) {
        if (binding.device_address != 0) {
            runtime_->free(binding.device_address);
        }
    }
    bindings_.clear();
    arg_addresses_.clear();
}

// ============================================================================
// Input/Output Binding
// ============================================================================

void GraphExecutor::set_input(const std::string& name, const void* data,
                               const std::vector<Size>& shape) {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        throw std::invalid_argument("GraphExecutor::set_input: unknown tensor '" + name + "'");
    }

    const auto& binding = it->second;

    // Validate shape matches
    if (shape != binding.shape) {
        std::ostringstream ss;
        ss << "GraphExecutor::set_input: shape mismatch for '" << name << "'. Expected [";
        for (size_t i = 0; i < binding.shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << binding.shape[i];
        }
        ss << "], got [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << shape[i];
        }
        ss << "]";
        throw std::invalid_argument(ss.str());
    }

    // Copy data to device
    runtime_->memcpy_h2d(binding.device_address, data, binding.size_bytes);
}

void GraphExecutor::set_input(const std::string& name, const void* data, Size size_bytes) {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        throw std::invalid_argument("GraphExecutor::set_input: unknown tensor '" + name + "'");
    }

    const auto& binding = it->second;

    if (size_bytes > binding.size_bytes) {
        throw std::invalid_argument("GraphExecutor::set_input: size exceeds allocation for '" + name + "'");
    }

    // Copy data to device
    runtime_->memcpy_h2d(binding.device_address, data, size_bytes);
}

void GraphExecutor::get_output(const std::string& name, void* data) {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        throw std::invalid_argument("GraphExecutor::get_output: unknown tensor '" + name + "'");
    }

    const auto& binding = it->second;

    // Copy data from device
    runtime_->memcpy_d2h(data, binding.device_address, binding.size_bytes);
}

void GraphExecutor::get_output(const std::string& name, void* data, Size size_bytes) {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        throw std::invalid_argument("GraphExecutor::get_output: unknown tensor '" + name + "'");
    }

    const auto& binding = it->second;

    if (size_bytes > binding.size_bytes) {
        throw std::invalid_argument("GraphExecutor::get_output: size exceeds allocation for '" + name + "'");
    }

    // Copy data from device
    runtime_->memcpy_d2h(data, binding.device_address, size_bytes);
}

const TensorBinding* GraphExecutor::get_binding(const std::string& name) const {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        return nullptr;
    }
    return &it->second;
}

const KernelArgument* GraphExecutor::find_argument(const std::string& name) const {
    if (!kernel_) return nullptr;

    for (const auto& arg : kernel_->arguments()) {
        if (arg.name == name) {
            return &arg;
        }
    }
    return nullptr;
}

// ============================================================================
// Execution
// ============================================================================

ExecutionResult GraphExecutor::execute() {
    if (!kernel_) {
        last_result_ = ExecutionResult(false, 0, 0.0, "No kernel set");
        return last_result_;
    }

    if (arg_addresses_.empty()) {
        last_result_ = ExecutionResult(false, 0, 0.0, "No tensors allocated");
        return last_result_;
    }

    // Launch the kernel
    auto launch_result = runtime_->launch(*kernel_, arg_addresses_);

    if (!launch_result.success) {
        last_result_ = ExecutionResult(false, 0, 0.0, launch_result.error);
        return last_result_;
    }

    // Convert cycles to milliseconds
    double time_ms = static_cast<double>(launch_result.cycles) /
                     (runtime_->config().clock_ghz * 1e6);

    last_result_ = ExecutionResult(true, launch_result.cycles, time_ms);
    return last_result_;
}

// ============================================================================
// Cleanup
// ============================================================================

void GraphExecutor::release() {
    free_tensors();
    kernel_.reset();
    last_result_ = ExecutionResult{};
}

} // namespace sw::runtime
