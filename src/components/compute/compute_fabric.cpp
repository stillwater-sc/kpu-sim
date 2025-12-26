#include <stdexcept>
#include <vector>

#include <sw/kpu/components/compute_fabric.hpp>
#include <sw/kpu/components/l1_buffer.hpp>

namespace sw::kpu {

// Helper function to convert ComputeType to string
static const char* compute_type_to_string(ComputeFabric::ComputeType type) {
    switch (type) {
        case ComputeFabric::ComputeType::BASIC_MATMUL: return "BASIC_MATMUL";
        case ComputeFabric::ComputeType::SYSTOLIC_ARRAY: return "SYSTOLIC_ARRAY";
        default: return "UNKNOWN";
    }
}

// ComputeFabric implementation
ComputeFabric::ComputeFabric(size_t tile_id, ComputeType type, Size systolic_rows, Size systolic_cols, double clock_freq_ghz)
    : is_computing(false), compute_start_cycle(0), tile_id(tile_id), compute_type(type),
      tracing_enabled_(false), trace_logger_(&trace::TraceLogger::instance()),
      clock_freq_ghz_(clock_freq_ghz), current_cycle_(0) {

    // Initialize systolic array if selected
    if (compute_type == ComputeType::SYSTOLIC_ARRAY) {
        systolic_array = std::make_unique<SystolicArray>(systolic_rows, systolic_cols);
    }
}

ComputeFabric::ComputeFabric(const ComputeFabric& other)
    : is_computing(other.is_computing), compute_start_cycle(other.compute_start_cycle),
      current_op(other.current_op), tile_id(other.tile_id), compute_type(other.compute_type),
      tracing_enabled_(other.tracing_enabled_), trace_logger_(other.trace_logger_),
      clock_freq_ghz_(other.clock_freq_ghz_), current_cycle_(other.current_cycle_) {

    // Deep copy systolic array if it exists
    if (other.systolic_array) {
        systolic_array = std::make_unique<SystolicArray>(*other.systolic_array);
    }
}

ComputeFabric& ComputeFabric::operator=(const ComputeFabric& other) {
    if (this != &other) {
        is_computing = other.is_computing;
        compute_start_cycle = other.compute_start_cycle;
        current_op = other.current_op;
        tile_id = other.tile_id;
        compute_type = other.compute_type;
        tracing_enabled_ = other.tracing_enabled_;
        trace_logger_ = other.trace_logger_;
        clock_freq_ghz_ = other.clock_freq_ghz_;
        current_cycle_ = other.current_cycle_;

        // Deep copy systolic array if it exists
        if (other.systolic_array) {
            systolic_array = std::make_unique<SystolicArray>(*other.systolic_array);
        } else {
            systolic_array.reset();
        }
    }
    return *this;
}

void ComputeFabric::start_matmul(const MatMulConfig& config) {
    if (is_computing) {
        throw std::runtime_error("ComputeFabric is already busy");
    }

    current_op = config;
    is_computing = true;
    compute_start_cycle = 0; // Will be set by the caller

    // Assign transaction ID for tracing
    if (tracing_enabled_ && trace_logger_) {
        current_op.transaction_id = trace_logger_->next_transaction_id();
        current_op.start_cycle = current_cycle_;

        // Log ISSUED trace
        trace::TraceEntry entry(current_cycle_, trace::ComponentType::COMPUTE_FABRIC,
                               static_cast<uint32_t>(tile_id),
                               trace::TransactionType::MATMUL, current_op.transaction_id);
        entry.clock_freq_ghz = clock_freq_ghz_;

        // Create compute payload
        trace::ComputePayload payload;
        payload.num_operations = static_cast<uint64_t>(config.m) * config.n * config.k;
        payload.m = config.m;
        payload.n = config.n;
        payload.k = config.k;
        payload.kernel_name = compute_type_to_string(compute_type);

        entry.payload = payload;
        entry.description = std::string("ComputeFabric MatMul (") +
                           std::to_string(config.m) + "x" +
                           std::to_string(config.n) + "x" +
                           std::to_string(config.k) + ", " +
                           compute_type_to_string(compute_type) + ")";

        trace_logger_->log(std::move(entry));
    }

    // Route to appropriate implementation
    if (compute_type == ComputeType::SYSTOLIC_ARRAY && systolic_array) {
        SystolicArray::MatMulConfig systolic_config;
        systolic_config.m = config.m;
        systolic_config.n = config.n;
        systolic_config.k = config.k;
        systolic_config.a_addr = config.a_addr;
        systolic_config.b_addr = config.b_addr;
        systolic_config.c_addr = config.c_addr;
        systolic_config.l1_buffer_id = config.l1_buffer_id;
        systolic_config.completion_callback = config.completion_callback;

        systolic_array->start_matmul(systolic_config);
    }
}

bool ComputeFabric::update(Cycle current_cycle, std::vector<L1Buffer>& l1_buffers) {
    if (!is_computing) {
        return false;
    }

    if (compute_start_cycle == 0) {
        compute_start_cycle = current_cycle;
    }

    bool operation_completed = false;

    // Route to appropriate implementation
    if (compute_type == ComputeType::SYSTOLIC_ARRAY && systolic_array) {
        operation_completed = systolic_array->update(current_cycle, l1_buffers);
    } else {
        // Basic matrix multiplication implementation
        Cycle required_cycles = estimate_cycles(current_op.m, current_op.n, current_op.k);

        if (current_cycle - compute_start_cycle >= required_cycles) {
            // Operation completed
            execute_matmul(l1_buffers);
            operation_completed = true;
        }
    }

    // Handle completion
    if (operation_completed) {
        // Log COMPLETED trace
        if (tracing_enabled_ && trace_logger_) {
            current_op.end_cycle = current_cycle;

            trace::TraceEntry entry(current_op.start_cycle, trace::ComponentType::COMPUTE_FABRIC,
                                   static_cast<uint32_t>(tile_id),
                                   trace::TransactionType::MATMUL, current_op.transaction_id);
            entry.complete(current_cycle);
            entry.clock_freq_ghz = clock_freq_ghz_;

            // Create compute payload
            trace::ComputePayload payload;
            payload.num_operations = static_cast<uint64_t>(current_op.m) * current_op.n * current_op.k;
            payload.m = current_op.m;
            payload.n = current_op.n;
            payload.k = current_op.k;
            payload.kernel_name = compute_type_to_string(compute_type);

            entry.payload = payload;
            entry.description = std::string("ComputeFabric MatMul completed (") +
                               std::to_string(current_op.m) + "x" +
                               std::to_string(current_op.n) + "x" +
                               std::to_string(current_op.k) + ", " +
                               compute_type_to_string(compute_type) + ")";

            trace_logger_->log(std::move(entry));
        }

        // Invoke completion callback
        if (current_op.completion_callback) {
            current_op.completion_callback();
        }

        is_computing = false;
        return true;
    }

    return false;
}

void ComputeFabric::execute_matmul(std::vector<L1Buffer>& l1_buffers) {
    if (current_op.l1_buffer_id >= l1_buffers.size()) {
        throw std::out_of_range("Invalid L1 buffer ID for matmul operation");
    }

    auto& l1_buffer = l1_buffers[current_op.l1_buffer_id];

    // Read matrices from L1 buffer
    Size a_size = current_op.m * current_op.k * sizeof(float);
    Size b_size = current_op.k * current_op.n * sizeof(float);
    Size c_size = current_op.m * current_op.n * sizeof(float);

    std::vector<float> a(current_op.m * current_op.k);
    std::vector<float> b(current_op.k * current_op.n);
    std::vector<float> c(current_op.m * current_op.n, 0.0f);

    l1_buffer.read(current_op.a_addr, a.data(), a_size);
    l1_buffer.read(current_op.b_addr, b.data(), b_size);

    // Perform matrix multiplication: C = A * B
    for (Size i = 0; i < current_op.m; ++i) {
        for (Size j = 0; j < current_op.n; ++j) {
            float sum = 0.0f;
            for (Size k = 0; k < current_op.k; ++k) {
                sum += a[i * current_op.k + k] * b[k * current_op.n + j];
            }
            c[i * current_op.n + j] = sum;
        }
    }

    // Write result back to L1 buffer
    l1_buffer.write(current_op.c_addr, c.data(), c_size);
}

Cycle ComputeFabric::estimate_cycles(Size m, Size n, Size k) const {
    if (compute_type == ComputeType::SYSTOLIC_ARRAY && systolic_array) {
        return systolic_array->estimate_cycles(m, n, k);
    } else {
        // Simplified model: assume 1 cycle per MAC operation
        return m * n * k;
    }
}

Size ComputeFabric::get_systolic_rows() const {
    if (systolic_array) {
        return systolic_array->get_rows();
    }
    return 0;
}

Size ComputeFabric::get_systolic_cols() const {
    if (systolic_array) {
        return systolic_array->get_cols();
    }
    return 0;
}

void ComputeFabric::reset() {
    is_computing = false;
    compute_start_cycle = 0;

    if (systolic_array) {
        systolic_array->reset();
    }
}

} // namespace sw::kpu