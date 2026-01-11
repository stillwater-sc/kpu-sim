#pragma once

#include <sw/kpu/models/temporal/memory/storage_scheduler.hpp>
#include <sw/kpu/models/temporal/datamovement/block_mover.hpp>
#include <sw/kpu/models/temporal/datamovement/streamer.hpp>

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

namespace sw::kpu {

// Integration layer for BlockMover to use StorageScheduler with EDDO
class KPU_API StorageSchedulerBlockMoverAdapter {
private:
    StorageScheduler* orchestrator;
    size_t adapter_id;

public:
    explicit StorageSchedulerBlockMoverAdapter(StorageScheduler* orchestrator, size_t adapter_id);

    // Enhanced block transfer with EDDO orchestration
    void orchestrated_block_transfer(size_t src_l3_tile_id, Address src_offset,
                                   size_t dst_orchestrator_bank, Address dst_offset,
                                   Size block_height, Size block_width, Size element_size,
                                   BlockMover::TransformType transform = BlockMover::TransformType::IDENTITY,
                                   std::function<void()> completion_callback = nullptr);

    // Double-buffered block movement pattern
    void double_buffered_transfer(size_t src_l3_tile_id, Address src_offset,
                                size_t primary_bank, size_t secondary_bank,
                                Size transfer_size);

    // Pipeline block movement with compute overlap
    void pipelined_transfer_compute(size_t src_l3_tile_id, Address src_offset,
                                  size_t input_bank, size_t output_bank,
                                  Size transfer_size,
                                  const std::function<void()>& compute_operation);

    bool is_busy() const;
    void reset();
};

// Integration layer for Streamer to use StorageScheduler with EDDO
class KPU_API StorageSchedulerStreamerAdapter {
private:
    StorageScheduler* orchestrator;
    size_t adapter_id;

public:
    explicit StorageSchedulerStreamerAdapter(StorageScheduler* orchestrator, size_t adapter_id);

    // EDDO-enhanced streaming configuration
    struct EDDOStreamConfig {
        // Standard streaming parameters
        size_t orchestrator_bank_id;
        size_t l1_scratchpad_id;
        Address orchestrator_base_addr;
        Address l1_base_addr;

        // Matrix dimensions for streaming
        Size matrix_height;
        Size matrix_width;
        Size element_size;
        Size systolic_array_size;

        // EDDO orchestration
        Streamer::StreamDirection direction;
        Streamer::StreamType stream_type;
        bool enable_prefetch_pipelining;
        Size prefetch_depth;

        // Completion callback
        std::function<void()> completion_callback;
    };

    // Start EDDO-orchestrated streaming
    void start_eddo_stream(const EDDOStreamConfig& config);

    // Systolic array streaming with EDDO coordination
    void orchestrated_systolic_stream(size_t input_bank, size_t weight_bank,
                                     size_t output_bank, size_t l1_scratchpad,
                                     Size matrix_dim);

    // Multi-phase streaming for complex operations
    void multi_phase_stream(const std::vector<EDDOStreamConfig>& phases);

    bool is_streaming() const;
    void abort_stream();
    void reset();
};

// High-level EDDO coordination for matrix operations
class KPU_API EDDOMatrixOrchestrator {
private:
    StorageScheduler* orchestrator;
    std::vector<StorageSchedulerBlockMoverAdapter> block_adapters;
    std::vector<StorageSchedulerStreamerAdapter> stream_adapters;
    size_t orchestrator_id;

public:
    explicit EDDOMatrixOrchestrator(StorageScheduler* orchestrator, size_t orchestrator_id);

    // Register data movement components
    void register_block_mover(BlockMover* mover);
    void register_streamer(Streamer* streamer);

    // High-level matrix operation patterns
    struct MatrixOperationConfig {
        // Matrix dimensions
        Size m, n, k; // For C = A * B, A is m×k, B is k×n, C is m×n

        // Memory layout
        Address matrix_a_addr, matrix_b_addr, matrix_c_addr;
        size_t element_size;

        // Tiling parameters
        Size tile_size_m, tile_size_n, tile_size_k;

        // Buffer allocation
        size_t num_buffer_banks;
        std::vector<size_t> input_banks;   // For A and B tiles
        std::vector<size_t> output_banks;  // For C tiles
        std::vector<size_t> temp_banks;    // For intermediate results

        // Compute integration
        std::function<void(const std::vector<float>&, const std::vector<float>&,
                          std::vector<float>&, Size, Size, Size)> compute_kernel;
    };

    // Orchestrate complete matrix multiplication with EDDO
    void orchestrate_matrix_multiply(const MatrixOperationConfig& config);

    // Orchestrate matrix transpose with streaming
    void orchestrate_matrix_transpose(Address src_addr, Address dst_addr,
                                    Size rows, Size cols, size_t element_size);

    // Orchestrate convolution operation
    void orchestrate_convolution(Address input_addr, Address kernel_addr,
                               Address output_addr, Size input_h, Size input_w,
                               Size kernel_h, Size kernel_w, Size channels);

    // Status and control
    bool is_busy() const;
    void wait_for_completion();
    void abort_all_operations();
    void reset();

    // Performance monitoring
    struct OrchestrationMetrics {
        size_t total_operations_completed;
        size_t total_bytes_moved;
        double average_bank_utilization;
        double operation_efficiency;
    };
    OrchestrationMetrics get_metrics() const;
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif