// ============================================================================
// src/models/behavioral/orchestrator.cpp
// Behavioral Orchestrator implementation
// ============================================================================

#include <sw/kpu/behavioral/orchestrator.hpp>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace sw::kpu::behavioral {

BehavioralOrchestrator::BehavioralOrchestrator()
    : BehavioralOrchestrator(Config{})
{
}

BehavioralOrchestrator::BehavioralOrchestrator(const Config& config)
    : config_(config)
    , stats_{}
    , block_mover_()
    , vector_engine_()
{
    // Create memory model with appropriate configuration
    MemoryModelConfig mem_config;
    mem_config.l3_tile_count = config_.num_l3_tiles;
    mem_config.l3_tile_capacity_bytes = config_.l3_tile_capacity;
    mem_config.l2_bank_count = config_.num_l2_banks;
    mem_config.l2_bank_capacity_bytes = config_.l2_bank_capacity;

    memory_ = std::make_unique<BehavioralMemoryModel>(mem_config);

    // Initialize allocators
    l3_alloc_.reset(config_.num_l3_tiles);
    l2_alloc_.reset(config_.num_l2_banks);
}

void BehavioralOrchestrator::reset_allocators() {
    l3_alloc_.reset(config_.num_l3_tiles);
    l2_alloc_.reset(config_.num_l2_banks);
}

Address BehavioralOrchestrator::make_l3_addr(uint32_t tile_id, Size offset) {
    return AddressEncoder::encode(MemoryRegionType::L3_TILE, tile_id, offset);
}

Address BehavioralOrchestrator::make_l2_addr(uint32_t bank_id, Size offset) {
    return AddressEncoder::encode(MemoryRegionType::L2_BANK, bank_id, offset);
}

void BehavioralOrchestrator::execute_matmul(
    Address host_a, uint32_t m, uint32_t k,
    Address host_b, uint32_t n,
    Address host_c,
    bool accumulate)
{
    // Check if matrices fit in L3/L2
    Size a_bytes = static_cast<Size>(m) * k * sizeof(float);
    Size b_bytes = static_cast<Size>(k) * n * sizeof(float);
    Size c_bytes = static_cast<Size>(m) * n * sizeof(float);

    // For now, require matrices fit in single tiles/banks
    // TODO: Add automatic tiling for large matrices
    if (a_bytes > config_.l3_tile_capacity ||
        b_bytes > config_.l3_tile_capacity ||
        c_bytes > config_.l3_tile_capacity) {
        // Use tiled execution
        execute_tiled_matmul(host_a, m, k, host_b, n, host_c,
                            config_.default_tile_m,
                            config_.default_tile_n,
                            config_.default_tile_k);
        return;
    }

    reset_allocators();

    // 1. Allocate L3 buffers
    Size l3_a_offset = l3_alloc_.alloc(0, a_bytes);
    Size l3_b_offset = l3_alloc_.alloc(1, b_bytes);
    Size l3_c_offset = l3_alloc_.alloc(2, c_bytes);

    Address l3_a = make_l3_addr(0, l3_a_offset);
    Address l3_b = make_l3_addr(1, l3_b_offset);
    Address l3_c = make_l3_addr(2, l3_c_offset);

    // 2. Copy Host -> L3 (simulating DMA)
    memory_->copy(l3_a, host_a, a_bytes);
    memory_->copy(l3_b, host_b, b_bytes);
    stats_.host_to_l3_bytes += a_bytes + b_bytes;

    // 3. Allocate L2 buffers
    Size l2_a_offset = l2_alloc_.alloc(0, a_bytes);
    Size l2_b_offset = l2_alloc_.alloc(1, b_bytes);
    Size l2_c_offset = l2_alloc_.alloc(2, c_bytes);

    Address l2_a = make_l2_addr(0, l2_a_offset);
    Address l2_b = make_l2_addr(1, l2_b_offset);
    Address l2_c = make_l2_addr(2, l2_c_offset);

    // 4. BlockMover: L3 -> L2
    block_mover_.l3_to_l2(0, l3_a_offset, 0, l2_a_offset,
                          m, k, sizeof(float), memory_.get());
    block_mover_.l3_to_l2(1, l3_b_offset, 1, l2_b_offset,
                          k, n, sizeof(float), memory_.get());
    stats_.l3_to_l2_bytes += a_bytes + b_bytes;

    // 5. Compute matmul in L2
    execute_matmul_l2(l2_a, m, k, l2_b, n, l2_c, accumulate);

    // 6. BlockMover: L2 -> L3
    block_mover_.l2_to_l3(2, l2_c_offset, 2, l3_c_offset,
                          m, n, sizeof(float), memory_.get());
    stats_.l2_to_l3_bytes += c_bytes;

    // 7. Copy L3 -> Host (simulating DMA)
    memory_->copy(host_c, l3_c, c_bytes);
    stats_.l3_to_host_bytes += c_bytes;
}

void BehavioralOrchestrator::execute_tiled_matmul(
    Address host_a, uint32_t m, uint32_t k,
    Address host_b, uint32_t n,
    Address host_c,
    uint32_t tile_m, uint32_t tile_n, uint32_t tile_k)
{
    // Initialize output to zero if not accumulating
    Size c_bytes = static_cast<Size>(m) * n * sizeof(float);
    memory_->zero(host_c, c_bytes);

    // Number of tiles in each dimension
    uint32_t num_tiles_m = (m + tile_m - 1) / tile_m;
    uint32_t num_tiles_n = (n + tile_n - 1) / tile_n;
    uint32_t num_tiles_k = (k + tile_k - 1) / tile_k;

    // Tile sizes in bytes
    Size tile_a_bytes = static_cast<Size>(tile_m) * tile_k * sizeof(float);
    Size tile_b_bytes = static_cast<Size>(tile_k) * tile_n * sizeof(float);
    Size tile_c_bytes = static_cast<Size>(tile_m) * tile_n * sizeof(float);

    // Allocate L3 tile buffers (offsets only - addresses created per row)
    reset_allocators();
    Size l3_a_tile_offset = l3_alloc_.alloc(0, tile_a_bytes);
    Size l3_b_tile_offset = l3_alloc_.alloc(1, tile_b_bytes);
    Size l3_c_tile_offset = l3_alloc_.alloc(2, tile_c_bytes);

    // Allocate L2 tile buffers
    Size l2_a_tile_offset = l2_alloc_.alloc(0, tile_a_bytes);
    Size l2_b_tile_offset = l2_alloc_.alloc(1, tile_b_bytes);
    Size l2_c_tile_offset = l2_alloc_.alloc(2, tile_c_bytes);

    Address l2_a_tile = make_l2_addr(0, l2_a_tile_offset);
    Address l2_b_tile = make_l2_addr(1, l2_b_tile_offset);
    Address l2_c_tile = make_l2_addr(2, l2_c_tile_offset);

    // Loop over output tiles (i, j)
    for (uint32_t ti = 0; ti < num_tiles_m; ++ti) {
        for (uint32_t tj = 0; tj < num_tiles_n; ++tj) {
            // Actual tile dimensions (handle edge cases)
            uint32_t actual_tile_m = std::min(tile_m, m - ti * tile_m);
            uint32_t actual_tile_n = std::min(tile_n, n - tj * tile_n);

            // Initialize C tile to zero
            memory_->zero(l2_c_tile, tile_c_bytes);

            // Accumulate over K dimension
            for (uint32_t tk = 0; tk < num_tiles_k; ++tk) {
                uint32_t actual_tile_k = std::min(tile_k, k - tk * tile_k);

                // Copy A tile from host -> L3 (strided access for submatrix)
                // A tile at (ti*tile_m, tk*tile_k) with size (actual_tile_m, actual_tile_k)
                for (uint32_t row = 0; row < actual_tile_m; ++row) {
                    Size host_row_offset = (static_cast<Size>(ti * tile_m + row) * k +
                                           tk * tile_k) * sizeof(float);
                    Size l3_row_offset = l3_a_tile_offset + row * tile_k * sizeof(float);
                    memory_->copy(make_l3_addr(0, l3_row_offset),
                                 host_a + host_row_offset,
                                 actual_tile_k * sizeof(float));
                }
                stats_.host_to_l3_bytes += static_cast<Size>(actual_tile_m) * actual_tile_k * sizeof(float);

                // Copy B tile from host -> L3
                // B tile at (tk*tile_k, tj*tile_n) with size (actual_tile_k, actual_tile_n)
                for (uint32_t row = 0; row < actual_tile_k; ++row) {
                    Size host_row_offset = (static_cast<Size>(tk * tile_k + row) * n +
                                           tj * tile_n) * sizeof(float);
                    Size l3_row_offset = l3_b_tile_offset + row * tile_n * sizeof(float);
                    memory_->copy(make_l3_addr(1, l3_row_offset),
                                 host_b + host_row_offset,
                                 actual_tile_n * sizeof(float));
                }
                stats_.host_to_l3_bytes += static_cast<Size>(actual_tile_k) * actual_tile_n * sizeof(float);

                // BlockMover: L3 -> L2
                block_mover_.l3_to_l2(0, l3_a_tile_offset, 0, l2_a_tile_offset,
                                      actual_tile_m, actual_tile_k, sizeof(float),
                                      memory_.get());
                block_mover_.l3_to_l2(1, l3_b_tile_offset, 1, l2_b_tile_offset,
                                      actual_tile_k, actual_tile_n, sizeof(float),
                                      memory_.get());
                stats_.l3_to_l2_bytes += (static_cast<Size>(actual_tile_m) * actual_tile_k +
                                         static_cast<Size>(actual_tile_k) * actual_tile_n) * sizeof(float);

                // Compute: accumulate into C tile
                execute_matmul_l2(l2_a_tile, actual_tile_m, actual_tile_k,
                                 l2_b_tile, actual_tile_n,
                                 l2_c_tile, true);  // accumulate
            }

            // BlockMover: L2 -> L3 (C tile)
            block_mover_.l2_to_l3(2, l2_c_tile_offset, 2, l3_c_tile_offset,
                                  actual_tile_m, actual_tile_n, sizeof(float),
                                  memory_.get());
            stats_.l2_to_l3_bytes += static_cast<Size>(actual_tile_m) * actual_tile_n * sizeof(float);

            // Copy C tile from L3 -> host (strided write)
            for (uint32_t row = 0; row < actual_tile_m; ++row) {
                Size host_row_offset = (static_cast<Size>(ti * tile_m + row) * n +
                                       tj * tile_n) * sizeof(float);
                Size l3_row_offset = l3_c_tile_offset + row * tile_n * sizeof(float);
                memory_->copy(host_c + host_row_offset,
                             make_l3_addr(2, l3_row_offset),
                             actual_tile_n * sizeof(float));
            }
            stats_.l3_to_host_bytes += static_cast<Size>(actual_tile_m) * actual_tile_n * sizeof(float);
        }
    }
}

void BehavioralOrchestrator::execute_mlp_layer(
    Address host_input, uint32_t batch, uint32_t in_dim,
    Address host_weights, uint32_t out_dim,
    Address host_bias,
    Address host_output,
    ActivationType activation)
{
    // First, execute the matmul: output = input @ weights
    execute_matmul(host_input, batch, in_dim,
                   host_weights, out_dim,
                   host_output, false);

    // Then apply bias and activation in-place on host output
    if (host_bias != 0 || activation != ActivationType::NONE) {
        // Get pointers to data
        float* output_ptr = memory_->get_ptr<float>(host_output);
        const float* bias_ptr = (host_bias != 0) ? memory_->get_ptr<float>(host_bias) : nullptr;

        // Apply bias + activation
        vector_engine_.process_inplace(output_ptr, batch, out_dim, bias_ptr, activation);

        if (host_bias != 0) {
            stats_.bias_applications++;
        }
        if (activation != ActivationType::NONE) {
            stats_.activation_applications++;
        }
    }
}

void BehavioralOrchestrator::execute_matmul_l2(
    Address l2_a, uint32_t m, uint32_t k,
    Address l2_b, uint32_t n,
    Address l2_c,
    bool accumulate)
{
    // Get raw pointers
    const float* a_ptr = memory_->get_ptr<float>(l2_a);
    const float* b_ptr = memory_->get_ptr<float>(l2_b);
    float* c_ptr = memory_->get_ptr<float>(l2_c);

    // Execute matmul directly (behavioral - instant)
    // C[m,n] = A[m,k] @ B[k,n]
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            float sum = accumulate ? c_ptr[i * n + j] : 0.0f;
            for (uint32_t kk = 0; kk < k; ++kk) {
                sum += a_ptr[i * k + kk] * b_ptr[kk * n + j];
            }
            c_ptr[i * n + j] = sum;
        }
    }

    stats_.matmul_count++;
    stats_.total_flops += 2ULL * m * n * k;  // 2 FLOPs per MAC
}

} // namespace sw::kpu::behavioral
