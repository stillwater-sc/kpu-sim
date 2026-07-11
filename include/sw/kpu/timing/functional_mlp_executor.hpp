// ============================================================================
// Functional MLP execution on the CSP concurrent timing model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/concurrent_timing_executor.hpp>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief Narrow functional/transactional vertical slice for MLP inference.
 *
 * Inputs and weights traverse the real CSP data path. Intermediate activations
 * remain in compute storage and are consumed directly by the next layer; only
 * the final result drains through L2/L3 and stores to DRAM.
 */
class FunctionalMLPExecutor {
public:
    using FunctionalActivation = ConcurrentTimingExecutor::FunctionalActivation;

    struct Layer {
        Size input_dim = 0;
        Size output_dim = 0;
        std::vector<float> weights;
        std::vector<float> bias;
        FunctionalActivation activation = FunctionalActivation::NONE;
        std::string name;
    };

    struct Statistics {
        Cycle total_cycles = 0;
        Cycle total_stall_cycles = 0;
        size_t layers_completed = 0;
    };

    explicit FunctionalMLPExecutor(ConcurrentTimingExecutor::Config config = {})
        : config_(std::move(config)) {}

    void add_layer(Size input_dim, Size output_dim,
                   std::vector<float> weights,
                   std::vector<float> bias = {},
                   FunctionalActivation activation = FunctionalActivation::NONE,
                   std::string name = {}) {
        if (input_dim == 0 || output_dim == 0) {
            throw std::invalid_argument("MLP layer dimensions must be non-zero");
        }
        if (!layers_.empty() && layers_.back().output_dim != input_dim) {
            throw std::invalid_argument("MLP layer dimensions are not composable");
        }
        if (weights.size() != static_cast<size_t>(input_dim) * output_dim) {
            throw std::invalid_argument("MLP weight count does not match layer dimensions");
        }
        if (!bias.empty() && bias.size() != output_dim) {
            throw std::invalid_argument("MLP bias count does not match output dimension");
        }
        layers_.push_back(Layer{input_dim, output_dim, std::move(weights),
                                std::move(bias), activation, std::move(name)});
    }

    [[nodiscard]] std::vector<float> forward(const std::vector<float>& input,
                                             Size batch_size) {
        if (layers_.empty()) throw std::runtime_error("MLP has no layers");
        if (input.size() != static_cast<size_t>(batch_size) * layers_.front().input_dim) {
            throw std::invalid_argument("MLP input count does not match batch and input dimension");
        }

        stats_ = {};
        ConcurrentTimingExecutor executor(config_);
        const Address base = 0x100000;
        auto current = make_tile(isa::MatrixID::A, batch_size,
                                 layers_.front().input_dim, base, 0);
        executor.set_tile_payload(current.tile_id,
                                  TilePayload{batch_size, layers_.front().input_dim, input});
        executor.schedule_load(current);
        executor.schedule_move(current);
        executor.schedule_feed(current);

        for (size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
            const auto& layer = layers_[layer_index];
            const Address layer_base = base + static_cast<Address>(layer_index) * 0x100000;
            auto weights = make_tile(isa::MatrixID::B, layer.input_dim, layer.output_dim,
                                     layer_base + 0x40000, static_cast<Size>(layer_index));
            auto output = make_tile(isa::MatrixID::C, batch_size, layer.output_dim,
                                    layer_base + 0x80000, static_cast<Size>(layer_index));

            executor.set_tile_payload(weights.tile_id,
                                      TilePayload{layer.input_dim, layer.output_dim,
                                                  layer.weights});
            executor.schedule_load(weights);
            executor.schedule_move(weights);
            executor.schedule_feed(weights);

            ConcurrentTimingExecutor::MatMulComputeSpec compute;
            compute.a_tiles = {current.tile_id};
            compute.b_tiles = {weights.tile_id};
            if (layer_index > 0) compute.resident_tiles = {current.tile_id};
            compute.bias = layer.bias;
            compute.activation = layer.activation;
            executor.schedule_matmul_compute(output, compute);
            current = output;
        }

        executor.schedule_drain(current);
        executor.schedule_writeback(current);
        executor.schedule_store(current);
        if (!executor.run()) throw std::runtime_error("Functional CSP MLP execution failed");

        const auto timing = executor.get_statistics();
        stats_.total_cycles = timing.total_cycles;
        stats_.total_stall_cycles = timing.dma_credit_stalls + timing.bm_tag_stalls +
                                    timing.bm_credit_stalls + timing.str_tag_stalls +
                                    timing.str_credit_stalls;
        stats_.layers_completed = layers_.size();
        events_ = executor.events();
        return executor.tile_payload_at(MemoryLevel::DRAM, current.tile_id).values;
    }

    [[nodiscard]] const Statistics& statistics() const { return stats_; }
    [[nodiscard]] const std::vector<Layer>& layers() const { return layers_; }
    [[nodiscard]] const std::vector<TimingEvent>& events() const { return events_; }

private:
    ConcurrentTimingExecutor::Config config_;
    std::vector<Layer> layers_;
    Statistics stats_;
    std::vector<TimingEvent> events_;

    [[nodiscard]] static TileDescriptor make_tile(isa::MatrixID matrix,
                                                  Size rows, Size cols,
                                                  Address address,
                                                  Size layer_index) {
        TileDescriptor tile;
        tile.tile_id = {matrix, layer_index, 0, 0};
        tile.dram_address = address;
        tile.matrix_base_address = address;
        tile.height = rows;
        tile.width = cols;
        tile.element_size = sizeof(float);
        tile.size_bytes = rows * cols * sizeof(float);
        return tile;
    }

};

} // namespace sw::kpu::timing
