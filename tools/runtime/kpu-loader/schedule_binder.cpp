/**
 * @file schedule_binder.cpp
 * @brief Implementation of schedule binder
 */

#include "schedule_binder.hpp"
#include <stdexcept>

namespace sw::kpu::runtime {

using namespace sw::kpu::compiler;

ScheduleBinder::ScheduleBinder(const KPUSimulator::Config& config)
    : config_(config)
{
}

BoundSchedule ScheduleBinder::bind(const dfx::Program& program) {
    BoundSchedule schedule;
    schedule.program = &program;
    schedule.total_cycles = 0;

    // Initialize resource stats
    schedule.resources = {};

    // Simple round-robin resource assignment for now
    size_t current_dma = 0;
    size_t current_block_mover = 0;
    size_t current_streamer = 0;
    size_t current_l3_tile = 0;
    size_t current_l2_bank = 0;
    size_t current_l1_buffer = 0;

    uint64_t current_cycle = 0;

    for (const auto& op : program.operations) {
        BoundOperation bound;
        bound.dfx_op = op.get();
        bound.start_cycle = current_cycle;

        if (auto* data_move = dynamic_cast<const dfx::DataMoveOp*>(op.get())) {
            // Determine which data movement engine to use based on levels
            dfx::MemoryLevel src_level = data_move->source.level;
            dfx::MemoryLevel dst_level = data_move->destination.level;

            if (src_level == dfx::MemoryLevel::EXTERNAL ||
                dst_level == dfx::MemoryLevel::EXTERNAL) {
                // Use DMA engine for external memory transfers
                bound.dma_engine_id = current_dma;
                current_dma = (current_dma + 1) % config_.dma_engine_count;
                schedule.resources.dma_engines_used =
                    std::max(schedule.resources.dma_engines_used, current_dma + 1);
            }
            else if ((src_level == dfx::MemoryLevel::L3 && dst_level == dfx::MemoryLevel::L2) ||
                     (src_level == dfx::MemoryLevel::L2 && dst_level == dfx::MemoryLevel::L3)) {
                // Use BlockMover for L3↔L2 transfers
                bound.block_mover_id = current_block_mover;
                current_block_mover = (current_block_mover + 1) % config_.block_mover_count;
                schedule.resources.block_movers_used =
                    std::max(schedule.resources.block_movers_used, current_block_mover + 1);
            }
            else if ((src_level == dfx::MemoryLevel::L2 && dst_level == dfx::MemoryLevel::L1) ||
                     (src_level == dfx::MemoryLevel::L1 && dst_level == dfx::MemoryLevel::L2)) {
                // Use Streamer for L2↔L1 transfers
                bound.streamer_id = current_streamer;
                current_streamer = (current_streamer + 1) % config_.streamer_count;
                schedule.resources.streamers_used =
                    std::max(schedule.resources.streamers_used, current_streamer + 1);
            }

            // Allocate memory resources
            bound.l3_tile_id = current_l3_tile;
            current_l3_tile = (current_l3_tile + 1) % config_.l3_tile_count;

            bound.l2_bank_id = current_l2_bank;
            current_l2_bank = (current_l2_bank + 1) % config_.l2_bank_count;

            bound.l1_buffer_id = current_l1_buffer;
            current_l1_buffer = (current_l1_buffer + 1) % config_.l1_buffer_count;

            // Calculate addresses
            bound.source_addr = calculate_address(data_move->source, src_level);
            bound.dest_addr = calculate_address(data_move->destination, dst_level);

            // Estimate cycles (simplified)
            size_t bytes = data_move->source.size_bytes(dfx::DataType::FLOAT32);
            bound.end_cycle = bound.start_cycle + (bytes / 64);  // Simplified: 64 bytes/cycle
        }
        else if (auto* compute = dynamic_cast<const dfx::ComputeOp*>(op.get())) {
            // Compute operations use systolic array
            // Estimate cycles based on tile size
            size_t tile_m = compute->output.tile_shape[0];
            size_t tile_n = compute->output.tile_shape[1];
            size_t tile_k = compute->inputs[0].tile_shape[1];

            // Systolic array processes (systolic_rows × systolic_cols) elements per cycle
            size_t flops = 2 * tile_m * tile_n * tile_k;
            size_t flops_per_cycle = config_.processor_array_rows * config_.processor_array_cols * 2;
            bound.end_cycle = bound.start_cycle + (flops / flops_per_cycle);
        }
        else if (auto* barrier = dynamic_cast<const dfx::BarrierOp*>(op.get())) {
            // Barriers have zero cycles
            bound.end_cycle = bound.start_cycle;
        }

        current_cycle = bound.end_cycle;
        schedule.operations.push_back(bound);
    }

    schedule.total_cycles = current_cycle;

    // Calculate throughput
    // For MATMUL: 2 * M * N * K FLOPs
    // Assuming 1 GHz clock
    size_t total_flops = 2 * program.tiling.Ti * program.tiling.num_tiles_m *
                        program.tiling.Tj * program.tiling.num_tiles_n *
                        program.tiling.Tk * program.tiling.num_tiles_k;
    double time_seconds = static_cast<double>(schedule.total_cycles) / 1e9;  // Assuming 1 GHz
    schedule.estimated_throughput = (total_flops / 1e12) / time_seconds;  // TFLOPS

    return schedule;
}

uint64_t ScheduleBinder::calculate_address(const dfx::TileSpec& tile, dfx::MemoryLevel level) {
    // Simplified address calculation
    // In a real implementation, this would use the AddressDecoder

    uint64_t base = 0;
    switch (level) {
        case dfx::MemoryLevel::EXTERNAL:
            base = config_.external_memory_base;
            break;
        case dfx::MemoryLevel::L3:
            base = config_.l3_tile_base;
            break;
        case dfx::MemoryLevel::L2:
            base = config_.l2_bank_base;
            break;
        case dfx::MemoryLevel::L1:
            base = config_.l1_buffer_base;
            break;
        case dfx::MemoryLevel::REGISTER:
            return 0;  // Registers don't have addresses
    }

    // Calculate offset based on tile indices
    size_t tile_size = 1;
    for (auto dim : tile.tile_shape) {
        tile_size *= dim;
    }
    tile_size *= 4;  // Assuming float32

    size_t linear_idx = 0;
    for (auto idx : tile.tile_indices) {
        linear_idx = linear_idx * 1024 + idx;  // Simplified linearization
    }

    return base + linear_idx * tile_size;
}

} // namespace sw::kpu::runtime
