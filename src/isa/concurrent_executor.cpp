/**
 * @file concurrent_executor.cpp
 * @brief Implementation of concurrent execution model for Data Movement ISA
 */

#include <sw/kpu/isa/concurrent_executor.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace sw::kpu::isa {

// ============================================================================
// ConcurrentExecutor Implementation
// ============================================================================

ConcurrentExecutor::ConcurrentExecutor(const ResourceConfig& config)
    : config_(config),
      compute_fabric_(ResourceType::COMPUTE_FABRIC, 0, config.compute_throughput_gflops,
                      config.systolic_size * config.systolic_size),  // Compute processes full tile per cycle
      current_cycle_(0),
      makespan_(0),
      last_barrier_cycle_(0),
      tile_layout_(nullptr)
{
    // Initialize memory channels with DMA engines
    // Each DMA engine transfers bus_width_bytes per cycle
    for (uint8_t i = 0; i < config.num_memory_channels; ++i) {
        memory_channels_.emplace_back(i, config.dma_bandwidth_gb_s,
                                      config.dma_bus_width_bytes);
    }

    // Initialize block movers (L3 → L2)
    // Each BlockMover transfers bus_width_bytes per cycle
    for (uint8_t i = 0; i < config.num_block_movers; ++i) {
        block_movers_.emplace_back(ResourceType::BLOCK_MOVER, i,
                                   config.block_mover_bandwidth_gb_s,
                                   config.block_mover_bus_width_bytes);
    }

    // Initialize streamers (L2 → L1)
    // Each Streamer transfers bus_width_bytes per cycle
    for (uint8_t i = 0; i < config.num_streamers; ++i) {
        streamers_.emplace_back(ResourceType::STREAMER, i,
                                config.streamer_bandwidth_gb_s,
                                config.streamer_bus_width_bytes);
    }
}

ConcurrentExecutor::ConcurrentExecutor(const ResourceConfig& config,
                                       std::unique_ptr<TileLayout> layout)
    : ConcurrentExecutor(config)
{
    tile_layout_ = std::move(layout);
}

void ConcurrentExecutor::set_tile_layout(std::unique_ptr<TileLayout> layout) {
    tile_layout_ = std::move(layout);
}

LayoutPolicy ConcurrentExecutor::get_layout_policy() const {
    if (tile_layout_) {
        return tile_layout_->policy();
    }
    return LayoutPolicy::MATRIX_PARTITIONED;  // Default
}

void ConcurrentExecutor::initialize_layout_for_program(const DMProgram& program) {
    // Create layout config from program dimensions
    LayoutConfig layout_config;
    layout_config.num_channels = config_.num_memory_channels;
    layout_config.num_l3_tiles = 4;  // Default
    layout_config.num_l2_banks = 8;  // Default
    // Tile size for layout: max of A tile (Ti×Tk) and B tile (Tk×Tj)
    // This is used for address allocation, not transfer size
    Size a_tile_bytes = program.Ti * program.Tk * 4;  // Assume float32
    Size b_tile_bytes = program.Tk * program.Tj * 4;
    layout_config.tile_size_bytes = std::max(a_tile_bytes, b_tile_bytes);
    layout_config.element_size = 4;

    // Calculate tile counts
    layout_config.m_tiles = (program.M + program.Ti - 1) / program.Ti;
    layout_config.n_tiles = (program.N + program.Tj - 1) / program.Tj;
    layout_config.k_tiles = (program.K + program.Tk - 1) / program.Tk;

    // Set up channel assignments for MATRIX_PARTITIONED
    // A gets half the channels, B gets the other half (or remaining)
    uint8_t half = config_.num_memory_channels / 2;
    if (half == 0) half = 1;

    layout_config.matrix_channels.a_channels.clear();
    layout_config.matrix_channels.b_channels.clear();
    layout_config.matrix_channels.c_channels.clear();

    for (uint8_t i = 0; i < half; ++i) {
        layout_config.matrix_channels.a_channels.push_back(i);
    }
    for (uint8_t i = half; i < config_.num_memory_channels; ++i) {
        layout_config.matrix_channels.b_channels.push_back(i);
    }
    // C shares channels with A (accessed at different times)
    layout_config.matrix_channels.c_channels = layout_config.matrix_channels.a_channels;

    // Create the layout - default to MATRIX_PARTITIONED for simplicity
    tile_layout_ = create_tile_layout(LayoutPolicy::MATRIX_PARTITIONED, layout_config);
}

Cycle ConcurrentExecutor::execute(const DMProgram& program) {
    // Reset state
    all_ops_.clear();
    instruction_completion_.clear();
    current_cycle_ = 0;
    makespan_ = 0;
    last_barrier_cycle_ = 0;

    // Always reinitialize tile layout for the program dimensions
    // (each program may have different tile counts)
    initialize_layout_for_program(program);

    // Reset all resources
    for (auto& mc : memory_channels_) {
        mc.dma_engine.next_available_cycle = 0;
        mc.dma_engine.completed_ops.clear();
    }
    for (auto& bm : block_movers_) {
        bm.next_available_cycle = 0;
        bm.completed_ops.clear();
    }
    for (auto& str : streamers_) {
        str.next_available_cycle = 0;
        str.completed_ops.clear();
    }
    compute_fabric_.next_available_cycle = 0;
    compute_fabric_.completed_ops.clear();

    // Schedule each instruction
    for (const auto& instr : program.instructions) {
        schedule_instruction(instr);
    }

    // Collect all operations from resources
    for (const auto& mc : memory_channels_) {
        all_ops_.insert(all_ops_.end(),
                       mc.dma_engine.completed_ops.begin(),
                       mc.dma_engine.completed_ops.end());
    }
    for (const auto& bm : block_movers_) {
        all_ops_.insert(all_ops_.end(),
                       bm.completed_ops.begin(),
                       bm.completed_ops.end());
    }
    for (const auto& str : streamers_) {
        all_ops_.insert(all_ops_.end(),
                       str.completed_ops.begin(),
                       str.completed_ops.end());
    }
    all_ops_.insert(all_ops_.end(),
                   compute_fabric_.completed_ops.begin(),
                   compute_fabric_.completed_ops.end());

    // Sort by start cycle for display
    std::sort(all_ops_.begin(), all_ops_.end(),
              [](const ScheduledOp& a, const ScheduledOp& b) {
                  return a.start_cycle < b.start_cycle;
              });

    // Calculate makespan
    for (const auto& op : all_ops_) {
        makespan_ = std::max(makespan_, op.end_cycle);
    }

    return makespan_;
}

void ConcurrentExecutor::schedule_instruction(const DMInstruction& instr) {
    Cycle earliest = get_dependency_cycle(instr);
    earliest = std::max(earliest, last_barrier_cycle_);

    Size transfer_size = get_transfer_size(instr);
    Cycle completion = 0;

    switch (instr.opcode) {
        case DMOpcode::DMA_LOAD_TILE:
        case DMOpcode::DMA_STORE_TILE:
        case DMOpcode::DMA_PREFETCH_TILE: {
            const auto& ops = std::get<DMAOperands>(instr.operands);
            uint8_t channel = select_dma_channel(ops.matrix, ops.tile);
            auto& dma = memory_channels_[channel].dma_engine;
            completion = dma.schedule_op(earliest, transfer_size,
                                        instr.instruction_id, instr.label,
                                        ops.matrix, ops.tile);
            break;
        }

        case DMOpcode::BM_MOVE_TILE:
        case DMOpcode::BM_TRANSPOSE_TILE:
        case DMOpcode::BM_WRITEBACK_TILE:
        case DMOpcode::BM_RESHAPE_TILE: {
            const auto& ops = std::get<BlockMoverOperands>(instr.operands);
            uint8_t bm_id = select_block_mover(ops.matrix, ops.tile);
            auto& bm = block_movers_[bm_id];
            completion = bm.schedule_op(earliest, transfer_size,
                                       instr.instruction_id, instr.label,
                                       ops.matrix, ops.tile);
            break;
        }

        case DMOpcode::STR_FEED_ROWS:
        case DMOpcode::STR_FEED_COLS:
        case DMOpcode::STR_DRAIN_OUTPUT:
        case DMOpcode::STR_BROADCAST_ROW:
        case DMOpcode::STR_BROADCAST_COL: {
            const auto& ops = std::get<StreamerOperands>(instr.operands);
            uint8_t str_id = select_streamer(ops.matrix, ops.tile);
            auto& str = streamers_[str_id];
            completion = str.schedule_op(earliest, transfer_size,
                                        instr.instruction_id, instr.label,
                                        ops.matrix, ops.tile);
            break;
        }

        case DMOpcode::BARRIER: {
            // Barrier waits for all in-flight operations
            Cycle barrier_time = 0;
            for (const auto& mc : memory_channels_) {
                barrier_time = std::max(barrier_time, mc.dma_engine.next_available_cycle);
            }
            for (const auto& bm : block_movers_) {
                barrier_time = std::max(barrier_time, bm.next_available_cycle);
            }
            for (const auto& str : streamers_) {
                barrier_time = std::max(barrier_time, str.next_available_cycle);
            }
            last_barrier_cycle_ = barrier_time;
            completion = barrier_time;
            break;
        }

        case DMOpcode::WAIT_DMA:
        case DMOpcode::WAIT_BM:
        case DMOpcode::WAIT_STR:
        case DMOpcode::SIGNAL:
            // These are fine-grained sync - for now treat as no-ops
            completion = earliest;
            break;

        case DMOpcode::NOP:
            completion = earliest;
            break;

        case DMOpcode::HALT:
            completion = earliest;
            break;

        default:
            completion = earliest;
            break;
    }

    instruction_completion_[instr.instruction_id] = completion;
    current_cycle_ = std::max(current_cycle_, completion);
}

HardwareResource* ConcurrentExecutor::find_available_resource(
    ResourceType type, Cycle /*at_cycle*/)
{
    HardwareResource* best = nullptr;
    Cycle earliest_available = UINT64_MAX;

    switch (type) {
        case ResourceType::DMA_ENGINE:
            for (auto& mc : memory_channels_) {
                if (mc.dma_engine.next_available_cycle < earliest_available) {
                    earliest_available = mc.dma_engine.next_available_cycle;
                    best = &mc.dma_engine;
                }
            }
            break;

        case ResourceType::BLOCK_MOVER:
            for (auto& bm : block_movers_) {
                if (bm.next_available_cycle < earliest_available) {
                    earliest_available = bm.next_available_cycle;
                    best = &bm;
                }
            }
            break;

        case ResourceType::STREAMER:
            for (auto& str : streamers_) {
                if (str.next_available_cycle < earliest_available) {
                    earliest_available = str.next_available_cycle;
                    best = &str;
                }
            }
            break;

        case ResourceType::COMPUTE_FABRIC:
            best = &compute_fabric_;
            break;
    }

    return best;
}

Size ConcurrentExecutor::get_transfer_size(const DMInstruction& instr) const {
    switch (instr.opcode) {
        case DMOpcode::DMA_LOAD_TILE:
        case DMOpcode::DMA_STORE_TILE:
        case DMOpcode::DMA_PREFETCH_TILE: {
            const auto& ops = std::get<DMAOperands>(instr.operands);
            return ops.size_bytes;
        }

        case DMOpcode::BM_MOVE_TILE:
        case DMOpcode::BM_TRANSPOSE_TILE:
        case DMOpcode::BM_WRITEBACK_TILE:
        case DMOpcode::BM_RESHAPE_TILE: {
            const auto& ops = std::get<BlockMoverOperands>(instr.operands);
            return ops.height * ops.width * ops.element_size;
        }

        case DMOpcode::STR_FEED_ROWS:
        case DMOpcode::STR_FEED_COLS:
        case DMOpcode::STR_DRAIN_OUTPUT:
        case DMOpcode::STR_BROADCAST_ROW:
        case DMOpcode::STR_BROADCAST_COL: {
            const auto& ops = std::get<StreamerOperands>(instr.operands);
            // Streaming takes fabric_size cycles to process height x width elements
            return ops.height * ops.width * 4;  // Assume 4-byte elements
        }

        default:
            return 0;
    }
}

Cycle ConcurrentExecutor::get_dependency_cycle(const DMInstruction& instr) const {
    Cycle max_dep = 0;
    for (uint32_t dep_id : instr.dependencies) {
        auto it = instruction_completion_.find(dep_id);
        if (it != instruction_completion_.end()) {
            max_dep = std::max(max_dep, it->second);
        }
    }
    return max_dep;
}

uint8_t ConcurrentExecutor::select_dma_channel(MatrixID matrix, TileCoord tile) const {
    // Use the tile layout to determine channel assignment
    if (tile_layout_) {
        return tile_layout_->get_channel(matrix, tile.ti, tile.tj, tile.tk);
    }
    // Fallback: round-robin based on tile coordinates (old behavior)
    size_t hash = static_cast<size_t>(matrix) * 1000 +
                  tile.ti * 100 + tile.tj * 10 + tile.tk;
    return static_cast<uint8_t>(hash % config_.num_memory_channels);
}

uint8_t ConcurrentExecutor::select_block_mover(MatrixID matrix, TileCoord tile) const {
    // Use the tile layout to determine L3 tile ID, then map to block mover
    if (tile_layout_) {
        auto loc = tile_layout_->get_tile_location(matrix, tile.ti, tile.tj, tile.tk);
        return loc.l3_tile_id % config_.num_block_movers;
    }
    // Fallback: distribute based on tile coordinates
    size_t idx = static_cast<size_t>(matrix) * 100 + tile.ti * 10 + tile.tk;
    return static_cast<uint8_t>(idx % config_.num_block_movers);
}

uint8_t ConcurrentExecutor::select_streamer(MatrixID matrix, TileCoord tile) const {
    // Use the tile layout to determine L2 bank ID, then map to streamer
    if (tile_layout_) {
        auto loc = tile_layout_->get_tile_location(matrix, tile.ti, tile.tj, tile.tk);
        return loc.l2_bank_id % config_.num_streamers;
    }
    // Fallback: distribute based on tile coordinates
    size_t idx = static_cast<size_t>(matrix) * 100 + tile.ti * 10 + tile.tj;
    return static_cast<uint8_t>(idx % config_.num_streamers);
}

ConcurrentExecutor::UtilizationStats ConcurrentExecutor::get_utilization() const {
    UtilizationStats stats{};
    stats.makespan = makespan_;
    stats.total_cycles = 0;

    if (makespan_ == 0) return stats;

    Cycle dma_busy = 0, bm_busy = 0, str_busy = 0, comp_busy = 0;

    for (const auto& op : all_ops_) {
        Cycle duration = op.end_cycle - op.start_cycle;
        stats.total_cycles += duration;

        switch (op.resource.type) {
            case ResourceType::DMA_ENGINE: dma_busy += duration; break;
            case ResourceType::BLOCK_MOVER: bm_busy += duration; break;
            case ResourceType::STREAMER: str_busy += duration; break;
            case ResourceType::COMPUTE_FABRIC: comp_busy += duration; break;
        }
    }

    // Utilization = busy_cycles / (makespan * num_resources)
    stats.dma_utilization = static_cast<double>(dma_busy) /
                           (makespan_ * config_.num_memory_channels);
    stats.block_mover_utilization = static_cast<double>(bm_busy) /
                                    (makespan_ * config_.num_block_movers);
    stats.streamer_utilization = static_cast<double>(str_busy) /
                                 (makespan_ * config_.num_streamers);
    stats.compute_utilization = static_cast<double>(comp_busy) / makespan_;

    return stats;
}

std::string ConcurrentExecutor::generate_timeline(size_t width) const {
    return TimelineFormatter::format_gantt(all_ops_, config_, makespan_, width);
}

std::string ConcurrentExecutor::generate_cycle_report() const {
    return TimelineFormatter::format_occupancy_table(all_ops_, config_, makespan_);
}

// ============================================================================
// TimelineFormatter Implementation
// ============================================================================

std::string TimelineFormatter::format_gantt(
    const std::vector<ScheduledOp>& ops,
    const ResourceConfig& config,
    Cycle total_cycles,
    size_t width)
{
    std::ostringstream oss;

    if (total_cycles == 0 || ops.empty()) {
        oss << "No operations to display\n";
        return oss.str();
    }

    // Calculate scale: cycles per character
    size_t chart_width = width - 20;  // Leave room for labels
    double scale = static_cast<double>(total_cycles) / chart_width;
    if (scale < 1.0) scale = 1.0;

    // Calculate timing in nanoseconds (DMA cycles @ 250 MHz = 4ns per cycle)
    double dma_cycle_ns = 1000.0 / config.dma_clock_mhz;  // ns per DMA cycle
    double total_time_ns = total_cycles * dma_cycle_ns;
    double total_time_us = total_time_ns / 1000.0;

    oss << "\n";
    oss << std::string(width, '=') << "\n";
    oss << "Resource Timeline\n";
    oss << std::string(width, '=') << "\n\n";

    // Clock domain and timing legend
    oss << "Clock Domains:\n";
    oss << "  DMA/L3:     " << std::fixed << std::setprecision(0) << config.dma_clock_mhz
        << " MHz (" << std::setprecision(1) << dma_cycle_ns << " ns/cycle), "
        << config.dma_bus_width_bytes << "-byte bus = "
        << std::setprecision(1) << config.dma_bandwidth_gb_s << " GB/s/channel\n";
    oss << "  BM/L2:      " << std::setprecision(0) << config.block_mover_clock_mhz
        << " MHz (" << std::setprecision(1) << (1000.0 / config.block_mover_clock_mhz) << " ns/cycle), "
        << config.block_mover_bus_width_bytes << "-byte bus = "
        << std::setprecision(1) << config.block_mover_bandwidth_gb_s << " GB/s/mover\n";
    oss << "  STR/L1:     " << std::setprecision(0) << config.streamer_clock_mhz
        << " MHz (" << std::setprecision(1) << (1000.0 / config.streamer_clock_mhz) << " ns/cycle), "
        << config.streamer_bus_width_bytes << "-byte bus = "
        << std::setprecision(1) << config.streamer_bandwidth_gb_s << " GB/s/streamer\n";
    oss << "  Compute:    " << std::setprecision(0) << config.compute_clock_mhz
        << " MHz (" << std::setprecision(2) << (1000.0 / config.compute_clock_mhz) << " ns/cycle), "
        << config.systolic_size << "x" << config.systolic_size << " systolic array\n\n";

    oss << "Timeline: " << total_cycles << " DMA cycles = "
        << std::setprecision(1) << total_time_ns << " ns ("
        << std::setprecision(2) << total_time_us << " µs)\n";
    oss << "Scale: 1 char = " << std::setprecision(1) << scale << " cycles = "
        << std::setprecision(1) << (scale * dma_cycle_ns) << " ns\n\n";

    // Build occupancy map for each resource
    auto render_resource = [&](ResourceType type, uint8_t index, const std::string& label) {
        oss << std::setw(12) << std::left << label << " |";

        std::vector<char> timeline(chart_width, ' ');

        // Find all ops for this resource
        for (const auto& op : ops) {
            if (op.resource.type == type && op.resource.index == index) {
                size_t start_col = static_cast<size_t>(op.start_cycle / scale);
                size_t end_col = static_cast<size_t>(op.end_cycle / scale);
                if (start_col >= chart_width) start_col = chart_width - 1;
                if (end_col >= chart_width) end_col = chart_width - 1;
                if (end_col <= start_col) end_col = start_col + 1;

                // Choose character based on matrix
                char c = '#';
                switch (op.matrix) {
                    case MatrixID::A: c = 'A'; break;
                    case MatrixID::B: c = 'B'; break;
                    case MatrixID::C: c = 'C'; break;
                }

                for (size_t col = start_col; col < end_col && col < chart_width; ++col) {
                    timeline[col] = c;
                }
            }
        }

        for (char c : timeline) {
            oss << c;
        }
        oss << "|\n";
    };

    // Render DMA engines
    oss << "DMA Engines (Ext Mem ↔ L3) @ " << std::setprecision(0) << config.dma_clock_mhz << " MHz:\n";
    for (uint8_t i = 0; i < config.num_memory_channels; ++i) {
        render_resource(ResourceType::DMA_ENGINE, i, "DMA[" + std::to_string(i) + "]");
    }
    oss << "\n";

    // Render Block Movers
    oss << "Block Movers (L3 ↔ L2) @ " << std::setprecision(0) << config.block_mover_clock_mhz << " MHz:\n";
    for (uint8_t i = 0; i < config.num_block_movers; ++i) {
        render_resource(ResourceType::BLOCK_MOVER, i, "BM[" + std::to_string(i) + "]");
    }
    oss << "\n";

    // Render Streamers
    oss << "Streamers (L2 ↔ L1) @ " << std::setprecision(0) << config.streamer_clock_mhz << " MHz:\n";
    for (uint8_t i = 0; i < config.num_streamers; ++i) {
        render_resource(ResourceType::STREAMER, i, "STR[" + std::to_string(i) + "]");
    }

    oss << "\n";
    oss << "Legend: A=Matrix A, B=Matrix B, C=Matrix C, ' '=Idle\n";

    return oss.str();
}

std::string TimelineFormatter::format_occupancy_table(
    const std::vector<ScheduledOp>& ops,
    const ResourceConfig& config,
    Cycle total_cycles)
{
    std::ostringstream oss;

    if (total_cycles == 0) {
        oss << "No execution data available\n";
        return oss.str();
    }

    // Calculate timing
    double dma_cycle_ns = 1000.0 / config.dma_clock_mhz;
    double total_time_ns = total_cycles * dma_cycle_ns;
    double total_time_us = total_time_ns / 1000.0;

    oss << "\n";
    oss << std::string(80, '=') << "\n";
    oss << "Resource Occupancy Summary\n";
    oss << std::string(80, '=') << "\n\n";

    // Show aggregate bandwidth info
    double total_dma_bw = config.num_memory_channels * config.dma_bandwidth_gb_s;
    double total_bm_bw = config.num_block_movers * config.block_mover_bandwidth_gb_s;
    double total_str_bw = config.num_streamers * config.streamer_bandwidth_gb_s;
    oss << "Aggregate Bandwidth:\n";
    oss << std::fixed;
    oss << "  External (DMA):  " << config.num_memory_channels << " ch × "
        << std::setprecision(1) << config.dma_bandwidth_gb_s << " GB/s = "
        << total_dma_bw << " GB/s\n";
    oss << "  L3→L2 (BM):      " << (int)config.num_block_movers << " movers × "
        << config.block_mover_bandwidth_gb_s << " GB/s = "
        << total_bm_bw << " GB/s\n";
    oss << "  L2→L1 (STR):     " << (int)config.num_streamers << " streamers × "
        << config.streamer_bandwidth_gb_s << " GB/s = "
        << total_str_bw << " GB/s\n\n";

    // Calculate per-resource statistics
    struct ResourceStats {
        Cycle busy_cycles = 0;
        size_t op_count = 0;
        Size bytes_moved = 0;
    };

    std::map<ResourceId, ResourceStats> stats;

    for (const auto& op : ops) {
        auto& s = stats[op.resource];
        s.busy_cycles += op.duration();
        s.op_count++;
    }

    oss << std::setw(15) << std::left << "Resource"
        << std::setw(12) << std::right << "Busy Cycles"
        << std::setw(12) << "Operations"
        << std::setw(15) << "Utilization"
        << "\n";
    oss << std::string(54, '-') << "\n";

    auto print_resource_stats = [&](ResourceType type, uint8_t count, const std::string& prefix) {
        Cycle total_busy = 0;
        size_t total_ops = 0;

        for (uint8_t i = 0; i < count; ++i) {
            ResourceId rid{type, i};
            auto it = stats.find(rid);
            if (it != stats.end()) {
                double util = static_cast<double>(it->second.busy_cycles) / total_cycles * 100.0;
                oss << std::setw(15) << std::left << (prefix + "[" + std::to_string(i) + "]")
                    << std::setw(12) << std::right << it->second.busy_cycles
                    << std::setw(12) << it->second.op_count
                    << std::setw(14) << std::fixed << std::setprecision(1) << util << "%"
                    << "\n";
                total_busy += it->second.busy_cycles;
                total_ops += it->second.op_count;
            } else {
                oss << std::setw(15) << std::left << (prefix + "[" + std::to_string(i) + "]")
                    << std::setw(12) << std::right << 0
                    << std::setw(12) << 0
                    << std::setw(14) << "0.0%"
                    << "\n";
            }
        }

        // Aggregate for this resource type
        if (count > 0) {
            double agg_util = static_cast<double>(total_busy) / (total_cycles * count) * 100.0;
            oss << std::setw(15) << std::left << ("  " + prefix + " Total")
                << std::setw(12) << std::right << total_busy
                << std::setw(12) << total_ops
                << std::setw(14) << std::fixed << std::setprecision(1) << agg_util << "%"
                << "\n";
        }
    };

    print_resource_stats(ResourceType::DMA_ENGINE, config.num_memory_channels, "DMA");
    oss << "\n";
    print_resource_stats(ResourceType::BLOCK_MOVER, config.num_block_movers, "BM");
    oss << "\n";
    print_resource_stats(ResourceType::STREAMER, config.num_streamers, "STR");

    oss << "\n" << std::string(54, '-') << "\n";
    oss << "Total execution: " << total_cycles << " DMA cycles = "
        << std::setprecision(1) << total_time_ns << " ns ("
        << std::setprecision(2) << total_time_us << " µs)\n";
    oss << "DMA cycle time: " << std::setprecision(1) << dma_cycle_ns << " ns (@ "
        << std::setprecision(0) << config.dma_clock_mhz << " MHz)\n";

    return oss.str();
}

std::string TimelineFormatter::format_cycle_view(
    const std::vector<ScheduledOp>& ops,
    const ResourceConfig& config,
    Cycle start_cycle,
    Cycle end_cycle)
{
    std::ostringstream oss;

    double dma_cycle_ns = 1000.0 / config.dma_clock_mhz;
    double start_ns = start_cycle * dma_cycle_ns;
    double end_ns = end_cycle * dma_cycle_ns;

    oss << "\n";
    oss << std::string(120, '=') << "\n";
    oss << "Cycle-by-Cycle View (DMA cycles " << start_cycle << "-" << end_cycle
        << " = " << std::fixed << std::setprecision(0) << start_ns << "-" << end_ns << " ns)\n";
    oss << "DMA @ " << config.dma_clock_mhz << " MHz, "
        << "BM/STR @ " << config.block_mover_clock_mhz << " MHz, "
        << "Compute @ " << config.compute_clock_mhz << " MHz\n";
    oss << std::string(120, '=') << "\n\n";

    // Header
    oss << std::setw(8) << "Cycle" << " |";
    for (uint8_t i = 0; i < config.num_memory_channels; ++i) {
        oss << " DMA" << static_cast<int>(i) << " |";
    }
    for (uint8_t i = 0; i < config.num_block_movers; ++i) {
        oss << " BM" << static_cast<int>(i) << "  |";
    }
    for (uint8_t i = 0; i < config.num_streamers; ++i) {
        oss << " STR" << static_cast<int>(i) << " |";
    }
    oss << "\n";

    size_t header_width = 10 +
                          config.num_memory_channels * 7 +
                          config.num_block_movers * 7 +
                          config.num_streamers * 7;
    oss << std::string(header_width, '-') << "\n";

    // For each cycle, show what each resource is doing
    for (Cycle cycle = start_cycle; cycle < end_cycle; ++cycle) {
        oss << std::setw(8) << cycle << " |";

        // Find active ops at this cycle
        auto find_active = [&](ResourceType type, uint8_t index) -> const ScheduledOp* {
            for (const auto& op : ops) {
                if (op.resource.type == type &&
                    op.resource.index == index &&
                    op.start_cycle <= cycle &&
                    op.end_cycle > cycle) {
                    return &op;
                }
            }
            return nullptr;
        };

        // DMA engines
        for (uint8_t i = 0; i < config.num_memory_channels; ++i) {
            const auto* op = find_active(ResourceType::DMA_ENGINE, i);
            if (op) {
                char mat = 'X';
                switch (op->matrix) {
                    case MatrixID::A: mat = 'A'; break;
                    case MatrixID::B: mat = 'B'; break;
                    case MatrixID::C: mat = 'C'; break;
                }
                oss << "  " << mat << "   |";
            } else {
                oss << "  -   |";
            }
        }

        // Block movers
        for (uint8_t i = 0; i < config.num_block_movers; ++i) {
            const auto* op = find_active(ResourceType::BLOCK_MOVER, i);
            if (op) {
                char mat = 'X';
                switch (op->matrix) {
                    case MatrixID::A: mat = 'A'; break;
                    case MatrixID::B: mat = 'B'; break;
                    case MatrixID::C: mat = 'C'; break;
                }
                oss << "  " << mat << "   |";
            } else {
                oss << "  -   |";
            }
        }

        // Streamers
        for (uint8_t i = 0; i < config.num_streamers; ++i) {
            const auto* op = find_active(ResourceType::STREAMER, i);
            if (op) {
                char mat = 'X';
                switch (op->matrix) {
                    case MatrixID::A: mat = 'A'; break;
                    case MatrixID::B: mat = 'B'; break;
                    case MatrixID::C: mat = 'C'; break;
                }
                oss << "  " << mat << "  |";
            } else {
                oss << "  -  |";
            }
        }

        oss << "\n";
    }

    return oss.str();
}

} // namespace sw::kpu::isa
