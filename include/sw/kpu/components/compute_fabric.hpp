#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <stdexcept>
#include <iomanip>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

#include <sw/concepts.hpp>
#include <sw/kpu/components/l1_buffer.hpp>
#include <sw/kpu/components/systolic_array.hpp>
#include <sw/trace/trace_logger.hpp>

namespace sw::kpu {

// Compute fabric
class KPU_API ComputeFabric {
public:
    // Configuration options for compute fabric
    enum class ComputeType {
        BASIC_MATMUL,    // Simple triple-loop matrix multiplication
        SYSTOLIC_ARRAY   // Hardware systolic array implementation
    };

    struct MatMulConfig {
        Size m, n, k; // Matrix dimensions: C[m,n] = A[m,k] * B[k,n]
        Address a_addr, b_addr, c_addr; // Addresses in L1 buffer
        size_t l1_buffer_id; // Which L1 buffer to use
        std::function<void()> completion_callback;

        // Timing and tracing
        Cycle start_cycle = 0;
        Cycle end_cycle = 0;
        uint64_t transaction_id = 0;
    };

private:
    bool is_computing;
    Cycle compute_start_cycle;
    MatMulConfig current_op;
    size_t tile_id;  // Which compute tile this fabric represents
    ComputeType compute_type;

    // Systolic array (when enabled)
    std::unique_ptr<SystolicArray> systolic_array;

    // Tracing support
    bool tracing_enabled_;
    trace::TraceLogger* trace_logger_;
    double clock_freq_ghz_;
    Cycle current_cycle_;

public:
    explicit ComputeFabric(size_t tile_id, ComputeType type = ComputeType::SYSTOLIC_ARRAY,
                          Size systolic_rows = SystolicArray::DEFAULT_ROWS,
                          Size systolic_cols = SystolicArray::DEFAULT_COLS,
                          double clock_freq_ghz = 1.0);
    ~ComputeFabric() = default;

    // Custom copy and move semantics for std::vector compatibility
    ComputeFabric(const ComputeFabric& other);
    ComputeFabric& operator=(const ComputeFabric& other);
    ComputeFabric(ComputeFabric&&) = default;
    ComputeFabric& operator=(ComputeFabric&&) = default;
    
    // Tracing control
    void enable_tracing() { tracing_enabled_ = true; }
    void disable_tracing() { tracing_enabled_ = false; }
    bool is_tracing_enabled() const { return tracing_enabled_; }

    // Cycle management for timing simulation
    void set_cycle(Cycle cycle) { current_cycle_ = cycle; }
    Cycle get_cycle() const { return current_cycle_; }

    // Compute operations
    void start_matmul(const MatMulConfig& config);
    bool update(Cycle current_cycle, std::vector<L1Buffer>& l1_buffers);
    bool is_busy() const { return is_computing; }
    void reset();
    
    // Configuration
    size_t get_tile_id() const { return tile_id; }
    ComputeType get_compute_type() const { return compute_type; }

    // Systolic array access (when available)
    SystolicArray* get_systolic_array() const { return systolic_array.get(); }
    Size get_systolic_rows() const;
    Size get_systolic_cols() const;

private:
    void execute_matmul(std::vector<L1Buffer>& l1_buffers);
    void execute_systolic_matmul(std::vector<L1Buffer>& l1_buffers);
    Cycle estimate_cycles(Size m, Size n, Size k) const;
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif