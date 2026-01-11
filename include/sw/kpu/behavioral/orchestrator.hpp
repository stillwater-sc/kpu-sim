// ============================================================================
// include/sw/kpu/behavioral/orchestrator.hpp
// Behavioral Orchestrator - coordinates data flow for MLP execution
//
// The orchestrator manages the complete data flow pipeline:
//   Host -> DMA -> L3 -> BlockMover -> L2 -> Compute -> VectorEngine
//   -> L2 -> BlockMover -> L3 -> DMA -> Host
//
// In behavioral mode, all operations are instant (no timing simulation).
// ============================================================================

#pragma once

#include <sw/kpu/behavioral/memory_model.hpp>
#include <sw/kpu/behavioral/block_mover.hpp>
#include <sw/kpu/behavioral/vector_engine.hpp>
#include <sw/kpu/components/sfu.hpp>
#include <cstdint>
#include <memory>

namespace sw::kpu::behavioral {

/// Behavioral orchestrator for MLP execution
///
/// This orchestrator coordinates the behavioral components to execute
/// complete neural network operations:
/// - Single matmul: C = A @ B
/// - MLP layer: C = activation(A @ B + bias)
/// - Tiled matmul for large matrices
///
/// The orchestrator:
/// - Manages L3/L2 buffer allocation for each operation
/// - Moves data through the memory hierarchy
/// - Invokes compute and vector operations
/// - Tracks statistics
class BehavioralOrchestrator {
public:
    /// Configuration for the orchestrator
    struct Config {
        // Memory configuration
        uint32_t num_l3_tiles = 4;
        uint32_t num_l2_banks = 8;
        Size l3_tile_capacity = 128 * 1024;  // 128 KB per tile
        Size l2_bank_capacity = 64 * 1024;   // 64 KB per bank

        // Tiling defaults
        uint32_t default_tile_m = 64;
        uint32_t default_tile_n = 64;
        uint32_t default_tile_k = 64;
    };

    /// Statistics tracking
    struct Stats {
        // Data movement
        uint64_t host_to_l3_bytes = 0;
        uint64_t l3_to_host_bytes = 0;
        uint64_t l3_to_l2_bytes = 0;
        uint64_t l2_to_l3_bytes = 0;

        // Compute
        uint64_t matmul_count = 0;
        uint64_t total_flops = 0;

        // Vector operations
        uint64_t bias_applications = 0;
        uint64_t activation_applications = 0;

        void reset() {
            host_to_l3_bytes = l3_to_host_bytes = 0;
            l3_to_l2_bytes = l2_to_l3_bytes = 0;
            matmul_count = total_flops = 0;
            bias_applications = activation_applications = 0;
        }
    };

    // ========================================================================
    // Constructors
    // ========================================================================

    /// Default constructor with default configuration
    BehavioralOrchestrator();

    /// Construct with custom configuration
    explicit BehavioralOrchestrator(const Config& config);

    ~BehavioralOrchestrator() = default;

    // Non-copyable, movable
    BehavioralOrchestrator(const BehavioralOrchestrator&) = delete;
    BehavioralOrchestrator& operator=(const BehavioralOrchestrator&) = delete;
    BehavioralOrchestrator(BehavioralOrchestrator&&) = default;
    BehavioralOrchestrator& operator=(BehavioralOrchestrator&&) = default;

    // ========================================================================
    // Matrix Multiplication
    // ========================================================================

    /// Execute matmul: C[M,N] = A[M,K] @ B[K,N]
    ///
    /// @param host_a    Host memory address of A matrix
    /// @param m         Number of rows in A and C
    /// @param k         Number of columns in A, rows in B
    /// @param host_b    Host memory address of B matrix
    /// @param n         Number of columns in B and C
    /// @param host_c    Host memory address of C matrix (output)
    /// @param accumulate If true, add to existing C; if false, overwrite
    void execute_matmul(
        Address host_a, uint32_t m, uint32_t k,
        Address host_b, uint32_t n,
        Address host_c,
        bool accumulate = false);

    /// Execute tiled matmul for large matrices
    ///
    /// Tiles the operation to fit in L3/L2 buffers.
    void execute_tiled_matmul(
        Address host_a, uint32_t m, uint32_t k,
        Address host_b, uint32_t n,
        Address host_c,
        uint32_t tile_m, uint32_t tile_n, uint32_t tile_k);

    // ========================================================================
    // MLP Layer Execution
    // ========================================================================

    /// Execute MLP layer: C = activation(A @ B + bias)
    ///
    /// @param host_input   Host address of input [batch, in_dim]
    /// @param batch        Number of input rows (batch size)
    /// @param in_dim       Input dimension (columns in input, rows in weights)
    /// @param host_weights Host address of weights [in_dim, out_dim]
    /// @param out_dim      Output dimension (columns in weights and output)
    /// @param host_bias    Host address of bias [out_dim], or 0 for no bias
    /// @param host_output  Host address of output [batch, out_dim]
    /// @param activation   Activation function to apply
    void execute_mlp_layer(
        Address host_input, uint32_t batch, uint32_t in_dim,
        Address host_weights, uint32_t out_dim,
        Address host_bias,
        Address host_output,
        ActivationType activation);

    // ========================================================================
    // Component Access
    // ========================================================================

    /// Get the memory model
    BehavioralMemoryModel& memory() { return *memory_; }
    const BehavioralMemoryModel& memory() const { return *memory_; }

    /// Get the block mover
    BehavioralBlockMover& block_mover() { return block_mover_; }
    const BehavioralBlockMover& block_mover() const { return block_mover_; }

    /// Get the vector engine
    BehavioralVectorEngine& vector_engine() { return vector_engine_; }
    const BehavioralVectorEngine& vector_engine() const { return vector_engine_; }

    // ========================================================================
    // Configuration
    // ========================================================================

    const Config& config() const { return config_; }

    // ========================================================================
    // Statistics
    // ========================================================================

    const Stats& stats() const { return stats_; }
    void reset_stats() {
        stats_.reset();
        block_mover_.reset_stats();
        vector_engine_.reset_stats();
    }

private:
    Config config_;
    Stats stats_;

    // Components
    std::unique_ptr<BehavioralMemoryModel> memory_;
    BehavioralBlockMover block_mover_;
    BehavioralVectorEngine vector_engine_;

    // L3/L2 allocation tracking (simple bump allocator, reset per operation)
    struct BufferAllocator {
        std::vector<Size> offsets;  // Current offset per tile/bank
        void reset(size_t count) {
            offsets.assign(count, 0);
        }
        Size alloc(size_t id, Size bytes) {
            Size offset = offsets[id];
            offsets[id] += bytes;
            return offset;
        }
    };
    BufferAllocator l3_alloc_;
    BufferAllocator l2_alloc_;

    void reset_allocators();

    // Helper: encode address for memory model
    Address make_l3_addr(uint32_t tile_id, Size offset);
    Address make_l2_addr(uint32_t bank_id, Size offset);

    // Helper: execute matmul using raw pointers (after data is in L2)
    void execute_matmul_l2(
        Address l2_a, uint32_t m, uint32_t k,
        Address l2_b, uint32_t n,
        Address l2_c,
        bool accumulate);
};

} // namespace sw::kpu::behavioral
