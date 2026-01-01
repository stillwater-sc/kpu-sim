// ============================================================================
// src/benchmark/noc_benchmark.cpp
// Network-on-Chip Benchmarking Implementation
// ============================================================================

#include <sw/benchmark/noc_benchmark.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace sw::benchmark {

// ============================================================================
// BenchmarkStats Implementation
// ============================================================================

double BenchmarkStats::variance() const {
    if (samples < 2) return 0.0;
    double m = mean();
    // Note: We don't store individual samples, so we can't compute true variance
    // This is an approximation using min/max range
    double range = static_cast<double>(max_cycles - min_cycles);
    return (range * range) / 12.0;  // Uniform distribution approximation
}

double BenchmarkStats::stddev() const {
    return std::sqrt(variance());
}

// ============================================================================
// ContentionResult Implementation
// ============================================================================

double ContentionResult::fairness_jain() const {
    if (source_bandwidths.empty()) return 1.0;

    double sum = 0.0;
    double sum_sq = 0.0;
    for (double bw : source_bandwidths) {
        sum += bw;
        sum_sq += bw * bw;
    }

    double n = static_cast<double>(source_bandwidths.size());
    if (sum_sq == 0.0) return 1.0;

    return (sum * sum) / (n * sum_sq);
}

// ============================================================================
// NoCMicrobenchmarks Implementation
// ============================================================================

NoCMicrobenchmarks::NoCMicrobenchmarks(std::unique_ptr<INoC> noc)
    : noc_(std::move(noc))
{
}

uint8_t NoCMicrobenchmarks::manhattan_distance(uint8_t src, uint8_t dst) const {
    uint8_t cols = noc_->config().cols;
    uint8_t src_row = src / cols;
    uint8_t src_col = src % cols;
    uint8_t dst_row = dst / cols;
    uint8_t dst_col = dst % cols;

    return static_cast<uint8_t>(
        std::abs(static_cast<int>(dst_row) - static_cast<int>(src_row)) +
        std::abs(static_cast<int>(dst_col) - static_cast<int>(src_col))
    );
}

LatencyResult NoCMicrobenchmarks::measure_single_flit_latency(uint8_t src, uint8_t dst) {
    return measure_tile_latency(src, dst, 64);  // Single flit = 64 bytes
}

LatencyResult NoCMicrobenchmarks::measure_tile_latency(uint8_t src, uint8_t dst,
                                                        uint32_t size) {
    LatencyResult result;
    result.name = "tile_latency";
    result.src_router = src;
    result.dst_router = dst;
    result.hop_count = manhattan_distance(src, dst);
    result.tile_size = size;

    // Warmup iterations
    for (uint32_t i = 0; i < warmup_iterations_; ++i) {
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = static_cast<uint16_t>(i);
        tile.size = size;

        uint64_t cycle = 0;
        noc_->inject_tile(src, dst, tile, cycle);
        noc_->drain(cycle);
    }

    // Measurement iterations
    for (uint32_t i = 0; i < measure_iterations_; ++i) {
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = static_cast<uint16_t>(warmup_iterations_ + i);
        tile.size = size;

        uint64_t start_cycle = 0;
        uint64_t end_cycle = 0;

        noc_->set_delivery_callback(dst, [&](const TileDescriptor&, uint8_t, uint8_t,
                                              uint64_t inject, uint64_t complete) {
            end_cycle = complete;
        });

        noc_->inject_tile(src, dst, tile, start_cycle);
        noc_->drain(start_cycle);
        end_cycle = start_cycle;  // drain updates start_cycle

        result.stats.add_sample(end_cycle);

        // Clear callback
        noc_->set_delivery_callback(dst, nullptr);
    }

    return result;
}

std::vector<LatencyResult> NoCMicrobenchmarks::sweep_latency_by_distance() {
    std::vector<LatencyResult> results;

    uint8_t rows = noc_->config().rows;
    uint8_t cols = noc_->config().cols;

    // Test different hop counts
    // Always start from router 0
    uint8_t src = 0;

    // 0 hops (same router)
    results.push_back(measure_tile_latency(src, src, 1024));

    // 1 hop (neighbor)
    if (cols > 1) {
        results.push_back(measure_tile_latency(src, 1, 1024));
    }

    // 2 hops (diagonal neighbor)
    if (rows > 1 && cols > 1) {
        uint8_t dst = cols + 1;  // Row 1, Col 1
        results.push_back(measure_tile_latency(src, dst, 1024));
    }

    // Max hops (corner to corner)
    if (rows > 1 && cols > 1) {
        uint8_t dst = static_cast<uint8_t>(rows * cols - 1);
        results.push_back(measure_tile_latency(src, dst, 1024));
    }

    return results;
}

std::vector<LatencyResult> NoCMicrobenchmarks::sweep_latency_by_size(
    uint8_t src, uint8_t dst, const std::vector<uint32_t>& sizes) {

    std::vector<LatencyResult> results;
    for (uint32_t size : sizes) {
        results.push_back(measure_tile_latency(src, dst, size));
    }
    return results;
}

ThroughputResult NoCMicrobenchmarks::measure_point_to_point_bandwidth(
    uint8_t src, uint8_t dst, uint32_t tile_size, uint32_t num_tiles) {

    ThroughputResult result;
    result.name = "point_to_point_bandwidth";
    result.config = std::to_string(src) + "->" + std::to_string(dst) +
                    " " + std::to_string(tile_size) + "B x" + std::to_string(num_tiles);

    uint64_t delivered = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor& t, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;

    // Inject all tiles
    for (uint32_t i = 0; i < num_tiles; ++i) {
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = static_cast<uint16_t>(i);
        tile.size = tile_size;

        auto inject_result = noc_->inject_tile(src, dst, tile, cycle);
        if (inject_result == InjectResult::BUSY) {
            // Step simulation to make room
            noc_->step(cycle);
            cycle++;
            i--;  // Retry this tile
        }
    }

    // Drain remaining
    noc_->drain(cycle);

    result.total_bytes = static_cast<uint64_t>(tile_size) * num_tiles;
    result.total_cycles = cycle;
    result.total_tiles = num_tiles;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

ThroughputResult NoCMicrobenchmarks::measure_aggregate_bandwidth(
    const std::vector<std::pair<uint8_t, uint8_t>>& transfers,
    uint32_t tile_size, uint32_t num_tiles) {

    ThroughputResult result;
    result.name = "aggregate_bandwidth";
    result.config = std::to_string(transfers.size()) + " flows x " +
                    std::to_string(tile_size) + "B x" + std::to_string(num_tiles);

    uint64_t delivered = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;
    uint32_t total_expected = static_cast<uint32_t>(transfers.size()) * num_tiles;

    // Inject tiles for all transfers
    std::vector<uint32_t> tiles_injected(transfers.size(), 0);

    while (delivered < total_expected) {
        bool any_injected = false;

        for (size_t t = 0; t < transfers.size(); ++t) {
            if (tiles_injected[t] < num_tiles) {
                TileDescriptor tile;
                tile.tensor = TensorId::A;
                tile.m_tile = static_cast<uint16_t>(t);
                tile.n_tile = static_cast<uint16_t>(tiles_injected[t]);
                tile.size = tile_size;

                auto inject_result = noc_->inject_tile(
                    transfers[t].first, transfers[t].second, tile, cycle);

                if (inject_result == InjectResult::SUCCESS) {
                    tiles_injected[t]++;
                    any_injected = true;
                }
            }
        }

        noc_->step(cycle);
        cycle++;

        // Safety limit
        if (cycle > 1000000) break;
    }

    result.total_bytes = static_cast<uint64_t>(tile_size) * total_expected;
    result.total_cycles = cycle;
    result.total_tiles = total_expected;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

ThroughputResult NoCMicrobenchmarks::measure_bisection_bandwidth(uint32_t tile_size) {
    uint8_t rows = noc_->config().rows;
    uint8_t cols = noc_->config().cols;

    // Create transfers from left half to right half
    std::vector<std::pair<uint8_t, uint8_t>> transfers;

    uint8_t mid_col = cols / 2;
    for (uint8_t row = 0; row < rows; ++row) {
        for (uint8_t col = 0; col < mid_col; ++col) {
            uint8_t src = row * cols + col;
            uint8_t dst = row * cols + (cols - 1 - col);
            transfers.push_back({src, dst});
        }
    }

    auto result = measure_aggregate_bandwidth(transfers, tile_size, 10);
    result.name = "bisection_bandwidth";
    return result;
}

ContentionResult NoCMicrobenchmarks::measure_hot_spot(
    const std::vector<uint8_t>& sources, uint8_t dst, uint32_t tile_size) {

    ContentionResult result;
    result.name = "hot_spot";
    result.config = std::to_string(sources.size()) + " sources -> router " +
                    std::to_string(dst);
    result.num_sources = static_cast<uint32_t>(sources.size());
    result.num_destinations = 1;

    std::vector<uint64_t> per_source_bytes(sources.size(), 0);
    uint32_t num_tiles = 20;

    noc_->set_global_delivery_callback([&](const TileDescriptor& t, uint8_t src, uint8_t,
                                            uint64_t, uint64_t) {
        for (size_t i = 0; i < sources.size(); ++i) {
            if (sources[i] == src) {
                per_source_bytes[i] += t.size;
                break;
            }
        }
    });

    uint64_t cycle = 0;
    std::vector<uint32_t> tiles_injected(sources.size(), 0);
    uint32_t total_expected = static_cast<uint32_t>(sources.size()) * num_tiles;
    uint64_t delivered = 0;

    while (delivered < total_expected) {
        for (size_t i = 0; i < sources.size(); ++i) {
            if (tiles_injected[i] < num_tiles) {
                TileDescriptor tile;
                tile.tensor = TensorId::A;
                tile.m_tile = static_cast<uint16_t>(i);
                tile.n_tile = static_cast<uint16_t>(tiles_injected[i]);
                tile.size = tile_size;

                if (noc_->inject_tile(sources[i], dst, tile, cycle) == InjectResult::SUCCESS) {
                    tiles_injected[i]++;
                }
            }
        }

        noc_->step(cycle);

        // Count delivered
        delivered = 0;
        for (auto bytes : per_source_bytes) {
            delivered += bytes / tile_size;
        }

        cycle++;
        if (cycle > 1000000) break;
    }

    result.total_cycles = cycle;

    // Calculate per-source bandwidth
    for (size_t i = 0; i < sources.size(); ++i) {
        double bw = cycle > 0 ?
            static_cast<double>(per_source_bytes[i]) / cycle : 0.0;
        result.source_bandwidths.push_back(bw);
    }

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

ContentionResult NoCMicrobenchmarks::measure_center_congestion(uint32_t tile_size) {
    uint8_t rows = noc_->config().rows;
    uint8_t cols = noc_->config().cols;

    // All corner routers send to center
    std::vector<uint8_t> sources;
    sources.push_back(0);                          // Top-left
    sources.push_back(cols - 1);                   // Top-right
    sources.push_back((rows - 1) * cols);          // Bottom-left
    sources.push_back(rows * cols - 1);            // Bottom-right

    uint8_t center = (rows / 2) * cols + (cols / 2);

    auto result = measure_hot_spot(sources, center, tile_size);
    result.name = "center_congestion";
    return result;
}

// ============================================================================
// NoCPatternBenchmarks Implementation
// ============================================================================

NoCPatternBenchmarks::NoCPatternBenchmarks(std::unique_ptr<INoC> noc)
    : noc_(std::move(noc))
{
}

PatternResult NoCPatternBenchmarks::measure_broadcast_all(uint8_t src, uint32_t tile_size) {
    PatternResult result;
    result.pattern_name = "broadcast_all";
    result.config = "src=" + std::to_string(src) + " size=" + std::to_string(tile_size);

    uint8_t num_routers = noc_->num_routers();
    uint64_t delivered = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;

    // Send same tile to all destinations
    for (uint8_t dst = 0; dst < num_routers; ++dst) {
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = 0;
        tile.n_tile = dst;
        tile.size = tile_size;

        while (noc_->inject_tile(src, dst, tile, cycle) != InjectResult::SUCCESS) {
            noc_->step(cycle);
            cycle++;
        }
    }

    noc_->drain(cycle);

    result.total_cycles = cycle;
    result.total_bytes = static_cast<uint64_t>(tile_size) * num_routers;
    result.num_transfers = num_routers;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

PatternResult NoCPatternBenchmarks::measure_broadcast_row(uint8_t src, uint8_t row,
                                                           uint32_t tile_size) {
    PatternResult result;
    result.pattern_name = "broadcast_row";
    result.config = "src=" + std::to_string(src) + " row=" + std::to_string(row);

    uint8_t cols = noc_->config().cols;
    uint64_t delivered = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;

    for (uint8_t col = 0; col < cols; ++col) {
        uint8_t dst = row * cols + col;
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = 0;
        tile.n_tile = col;
        tile.size = tile_size;

        while (noc_->inject_tile(src, dst, tile, cycle) != InjectResult::SUCCESS) {
            noc_->step(cycle);
            cycle++;
        }
    }

    noc_->drain(cycle);

    result.total_cycles = cycle;
    result.total_bytes = static_cast<uint64_t>(tile_size) * cols;
    result.num_transfers = cols;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

PatternResult NoCPatternBenchmarks::measure_broadcast_col(uint8_t src, uint8_t col,
                                                           uint32_t tile_size) {
    PatternResult result;
    result.pattern_name = "broadcast_col";
    result.config = "src=" + std::to_string(src) + " col=" + std::to_string(col);

    uint8_t rows = noc_->config().rows;
    uint8_t cols = noc_->config().cols;
    uint64_t delivered = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;

    for (uint8_t row = 0; row < rows; ++row) {
        uint8_t dst = row * cols + col;
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = row;
        tile.n_tile = 0;
        tile.size = tile_size;

        while (noc_->inject_tile(src, dst, tile, cycle) != InjectResult::SUCCESS) {
            noc_->step(cycle);
            cycle++;
        }
    }

    noc_->drain(cycle);

    result.total_cycles = cycle;
    result.total_bytes = static_cast<uint64_t>(tile_size) * rows;
    result.num_transfers = rows;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

PatternResult NoCPatternBenchmarks::measure_reduce_all(uint8_t dst, uint32_t tile_size) {
    PatternResult result;
    result.pattern_name = "reduce_all";
    result.config = "dst=" + std::to_string(dst) + " size=" + std::to_string(tile_size);

    uint8_t num_routers = noc_->num_routers();
    uint64_t delivered = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;

    // All routers send to destination
    for (uint8_t src = 0; src < num_routers; ++src) {
        TileDescriptor tile;
        tile.tensor = TensorId::C;
        tile.m_tile = src;
        tile.n_tile = 0;
        tile.size = tile_size;

        while (noc_->inject_tile(src, dst, tile, cycle) != InjectResult::SUCCESS) {
            noc_->step(cycle);
            cycle++;
        }
    }

    noc_->drain(cycle);

    result.total_cycles = cycle;
    result.total_bytes = static_cast<uint64_t>(tile_size) * num_routers;
    result.num_transfers = num_routers;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

PatternResult NoCPatternBenchmarks::measure_reduce_tree(uint8_t dst, uint32_t tile_size) {
    // Tree reduction: pairs reduce, then pairs of pairs, etc.
    // For simplicity, simulate as sequential reduce (actual tree would need
    // intermediate compute which we don't model here)
    return measure_reduce_all(dst, tile_size);
}

PatternResult NoCPatternBenchmarks::measure_allreduce(uint32_t tile_size) {
    PatternResult result;
    result.pattern_name = "allreduce";
    result.config = "size=" + std::to_string(tile_size);

    uint8_t num_routers = noc_->num_routers();

    // Phase 1: Reduce to router 0
    auto reduce_result = measure_reduce_all(0, tile_size);

    // Phase 2: Broadcast from router 0
    auto broadcast_result = measure_broadcast_all(0, tile_size);

    result.total_cycles = reduce_result.total_cycles + broadcast_result.total_cycles;
    result.total_bytes = reduce_result.total_bytes + broadcast_result.total_bytes;
    result.num_transfers = reduce_result.num_transfers + broadcast_result.num_transfers;

    return result;
}

PatternResult NoCPatternBenchmarks::measure_east_flow(uint8_t start_col, uint32_t tile_size) {
    PatternResult result;
    result.pattern_name = "east_flow";
    result.config = "start_col=" + std::to_string(start_col);

    uint8_t rows = noc_->config().rows;
    uint8_t cols = noc_->config().cols;
    uint64_t delivered = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;
    uint32_t total_transfers = 0;

    // Each row: tile flows from start_col to end
    for (uint8_t row = 0; row < rows; ++row) {
        for (uint8_t col = start_col; col < cols - 1; ++col) {
            uint8_t src = row * cols + col;
            uint8_t dst = row * cols + col + 1;

            TileDescriptor tile;
            tile.tensor = TensorId::A;
            tile.m_tile = row;
            tile.n_tile = col;
            tile.size = tile_size;

            while (noc_->inject_tile(src, dst, tile, cycle) != InjectResult::SUCCESS) {
                noc_->step(cycle);
                cycle++;
            }
            total_transfers++;
        }
    }

    noc_->drain(cycle);

    result.total_cycles = cycle;
    result.total_bytes = static_cast<uint64_t>(tile_size) * total_transfers;
    result.num_transfers = total_transfers;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

PatternResult NoCPatternBenchmarks::measure_south_flow(uint8_t start_row, uint32_t tile_size) {
    PatternResult result;
    result.pattern_name = "south_flow";
    result.config = "start_row=" + std::to_string(start_row);

    uint8_t rows = noc_->config().rows;
    uint8_t cols = noc_->config().cols;
    uint64_t delivered = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;
    uint32_t total_transfers = 0;

    // Each column: tile flows from start_row to end
    for (uint8_t col = 0; col < cols; ++col) {
        for (uint8_t row = start_row; row < rows - 1; ++row) {
            uint8_t src = row * cols + col;
            uint8_t dst = (row + 1) * cols + col;

            TileDescriptor tile;
            tile.tensor = TensorId::B;
            tile.m_tile = row;
            tile.n_tile = col;
            tile.size = tile_size;

            while (noc_->inject_tile(src, dst, tile, cycle) != InjectResult::SUCCESS) {
                noc_->step(cycle);
                cycle++;
            }
            total_transfers++;
        }
    }

    noc_->drain(cycle);

    result.total_cycles = cycle;
    result.total_bytes = static_cast<uint64_t>(tile_size) * total_transfers;
    result.num_transfers = total_transfers;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

PatternResult NoCPatternBenchmarks::measure_systolic_flow(uint32_t a_tile_size,
                                                           uint32_t b_tile_size) {
    PatternResult result;
    result.pattern_name = "systolic_flow";
    result.config = "a=" + std::to_string(a_tile_size) + " b=" + std::to_string(b_tile_size);

    // Combined east and south flow
    auto east = measure_east_flow(0, a_tile_size);
    auto south = measure_south_flow(0, b_tile_size);

    // Take max cycles (parallel execution)
    result.total_cycles = std::max(east.total_cycles, south.total_cycles);
    result.total_bytes = east.total_bytes + south.total_bytes;
    result.num_transfers = east.num_transfers + south.num_transfers;

    return result;
}

PatternResult NoCPatternBenchmarks::measure_all_to_all(uint32_t tile_size) {
    PatternResult result;
    result.pattern_name = "all_to_all";
    result.config = "size=" + std::to_string(tile_size);

    uint8_t num_routers = noc_->num_routers();
    uint64_t delivered = 0;
    uint32_t total_expected = static_cast<uint32_t>(num_routers) * num_routers;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;

    // Each router sends to every other router
    std::vector<std::vector<bool>> sent(num_routers, std::vector<bool>(num_routers, false));

    while (delivered < total_expected) {
        for (uint8_t src = 0; src < num_routers; ++src) {
            for (uint8_t dst = 0; dst < num_routers; ++dst) {
                if (!sent[src][dst]) {
                    TileDescriptor tile;
                    tile.tensor = TensorId::A;
                    tile.m_tile = src;
                    tile.n_tile = dst;
                    tile.size = tile_size;

                    if (noc_->inject_tile(src, dst, tile, cycle) == InjectResult::SUCCESS) {
                        sent[src][dst] = true;
                    }
                }
            }
        }

        noc_->step(cycle);
        cycle++;

        if (cycle > 10000000) break;  // Safety limit
    }

    noc_->drain(cycle);

    result.total_cycles = cycle;
    result.total_bytes = static_cast<uint64_t>(tile_size) * total_expected;
    result.num_transfers = total_expected;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

PatternResult NoCPatternBenchmarks::measure_permutation(const std::vector<uint8_t>& mapping,
                                                         uint32_t tile_size) {
    PatternResult result;
    result.pattern_name = "permutation";
    result.config = "size=" + std::to_string(tile_size);

    uint64_t delivered = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    uint64_t cycle = 0;

    for (uint8_t src = 0; src < mapping.size(); ++src) {
        uint8_t dst = mapping[src];
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = src;
        tile.size = tile_size;

        while (noc_->inject_tile(src, dst, tile, cycle) != InjectResult::SUCCESS) {
            noc_->step(cycle);
            cycle++;
        }
    }

    noc_->drain(cycle);

    result.total_cycles = cycle;
    result.total_bytes = static_cast<uint64_t>(tile_size) * mapping.size();
    result.num_transfers = static_cast<uint32_t>(mapping.size());

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

// ============================================================================
// NoCOperatorBenchmarks Implementation
// ============================================================================

NoCOperatorBenchmarks::NoCOperatorBenchmarks(std::unique_ptr<INoC> noc)
    : noc_(std::move(noc)), config_(noc_->config())
{
}

OperatorResult NoCOperatorBenchmarks::benchmark_matmul(uint32_t M, uint32_t N, uint32_t K,
                                                        uint32_t Ti, uint32_t Tj, uint32_t Tk) {
    OperatorResult result;
    result.operator_name = "matmul";
    result.config = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) +
                    " tiles=" + std::to_string(Ti) + "x" + std::to_string(Tj) + "x" + std::to_string(Tk);

    // Calculate tile counts
    uint32_t m_tiles = (M + Ti - 1) / Ti;
    uint32_t n_tiles = (N + Tj - 1) / Tj;
    uint32_t k_tiles = (K + Tk - 1) / Tk;

    // Tile sizes in bytes (assuming float32)
    uint32_t a_tile_bytes = Ti * Tk * 4;
    uint32_t b_tile_bytes = Tk * Tj * 4;
    uint32_t c_tile_bytes = Ti * Tj * 4;

    // Total FLOPs
    result.total_flops = 2ULL * M * N * K;

    uint8_t rows = config_.rows;
    uint8_t cols = config_.cols;

    uint64_t delivered = 0;
    uint64_t cycle = 0;

    noc_->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                            uint64_t, uint64_t) {
        delivered++;
    });

    // Simulate systolic data flow for output-stationary matmul
    // A tiles flow east, B tiles flow south

    // For each K iteration
    for (uint32_t k = 0; k < k_tiles; ++k) {
        // Inject A tiles into leftmost column
        for (uint32_t m = 0; m < std::min(m_tiles, static_cast<uint32_t>(rows)); ++m) {
            TileDescriptor tile;
            tile.tensor = TensorId::A;
            tile.m_tile = static_cast<uint16_t>(m);
            tile.k_tile = static_cast<uint16_t>(k);
            tile.size = a_tile_bytes;

            uint8_t src = static_cast<uint8_t>(m * cols);  // Leftmost column
            uint8_t dst = src;  // Start at same position

            while (noc_->inject_tile(src, dst, tile, cycle) != InjectResult::SUCCESS) {
                noc_->step(cycle);
                cycle++;
            }
            result.noc_tiles_transferred++;
        }

        // Inject B tiles into topmost row
        for (uint32_t n = 0; n < std::min(n_tiles, static_cast<uint32_t>(cols)); ++n) {
            TileDescriptor tile;
            tile.tensor = TensorId::B;
            tile.k_tile = static_cast<uint16_t>(k);
            tile.n_tile = static_cast<uint16_t>(n);
            tile.size = b_tile_bytes;

            uint8_t src = static_cast<uint8_t>(n);  // Topmost row
            uint8_t dst = src;

            while (noc_->inject_tile(src, dst, tile, cycle) != InjectResult::SUCCESS) {
                noc_->step(cycle);
                cycle++;
            }
            result.noc_tiles_transferred++;
        }

        // Step simulation to allow tiles to flow
        for (int step = 0; step < static_cast<int>(rows + cols); ++step) {
            noc_->step(cycle);
            cycle++;
        }
    }

    noc_->drain(cycle);

    result.total_cycles = cycle;
    result.noc_bytes_transferred = static_cast<uint64_t>(result.noc_tiles_transferred) *
                                   ((a_tile_bytes + b_tile_bytes) / 2);

    // Estimate compute cycles (systolic array throughput)
    uint64_t systolic_ops = 24ULL * 24 * 2;  // 24x24 systolic, 2 ops/MAC
    result.compute_cycles = result.total_flops / systolic_ops;

    result.noc_wait_cycles = result.total_cycles > result.compute_cycles ?
                             result.total_cycles - result.compute_cycles : 0;

    noc_->set_global_delivery_callback(nullptr);

    return result;
}

NoCOperatorBenchmarks::AttentionResult NoCOperatorBenchmarks::benchmark_attention(
    uint32_t batch, uint32_t seq_len, uint32_t d_model, uint32_t num_heads) {

    AttentionResult result;
    uint32_t d_k = d_model / num_heads;
    uint32_t d_v = d_k;

    // Use matmul benchmark for each phase
    // Note: This is a simplified model - actual attention has more complex data flow

    // Phase 1: QKV Projections (3 parallel matmuls)
    // Q = X @ W_Q: [batch*seq_len, d_model] x [d_model, d_k*heads]
    result.qkv_projection = benchmark_matmul(batch * seq_len, d_model, d_model, 64, 64, 64);
    result.qkv_projection.operator_name = "qkv_projection";
    result.qkv_projection.total_flops *= 3;  // Q, K, V

    // Phase 2: Q @ K^T: [batch*heads, seq_len, d_k] x [batch*heads, d_k, seq_len]
    // For each head: [seq_len, d_k] x [d_k, seq_len] = [seq_len, seq_len]
    result.qk_matmul = benchmark_matmul(seq_len, seq_len, d_k, 64, 64, 64);
    result.qk_matmul.operator_name = "qk_matmul";
    result.qk_matmul.total_flops *= batch * num_heads;

    // Phase 3: Softmax (reduction-heavy, less NoC)
    result.softmax = benchmark_softmax(batch * num_heads * seq_len, seq_len);
    result.softmax.operator_name = "attention_softmax";

    // Phase 4: Attention @ V: [batch*heads, seq_len, seq_len] x [batch*heads, seq_len, d_v]
    result.attn_v_matmul = benchmark_matmul(seq_len, d_v, seq_len, 64, 64, 64);
    result.attn_v_matmul.operator_name = "attn_v_matmul";
    result.attn_v_matmul.total_flops *= batch * num_heads;

    // Phase 5: Output projection
    result.output_projection = benchmark_matmul(batch * seq_len, d_model, d_model, 64, 64, 64);
    result.output_projection.operator_name = "output_projection";

    // Total
    result.total.operator_name = "attention_total";
    result.total.config = "batch=" + std::to_string(batch) +
                          " seq=" + std::to_string(seq_len) +
                          " d=" + std::to_string(d_model) +
                          " heads=" + std::to_string(num_heads);
    result.total.total_cycles = result.qkv_projection.total_cycles +
                                 result.qk_matmul.total_cycles +
                                 result.softmax.total_cycles +
                                 result.attn_v_matmul.total_cycles +
                                 result.output_projection.total_cycles;
    result.total.total_flops = result.qkv_projection.total_flops +
                                result.qk_matmul.total_flops +
                                result.softmax.total_flops +
                                result.attn_v_matmul.total_flops +
                                result.output_projection.total_flops;
    result.total.noc_bytes_transferred = result.qkv_projection.noc_bytes_transferred +
                                          result.qk_matmul.noc_bytes_transferred +
                                          result.softmax.noc_bytes_transferred +
                                          result.attn_v_matmul.noc_bytes_transferred +
                                          result.output_projection.noc_bytes_transferred;

    return result;
}

OperatorResult NoCOperatorBenchmarks::benchmark_softmax(uint32_t rows, uint32_t cols) {
    OperatorResult result;
    result.operator_name = "softmax";
    result.config = std::to_string(rows) + "x" + std::to_string(cols);

    // Softmax requires:
    // 1. Max reduction per row
    // 2. Exp of (x - max)
    // 3. Sum reduction per row
    // 4. Division

    // FLOPs: rows * (cols comparisons + cols exp + cols additions + cols divisions)
    result.total_flops = static_cast<uint64_t>(rows) * cols * 4;

    // Data movement: Each row needs to be gathered for reduction, then scattered
    uint32_t row_bytes = cols * 4;  // float32

    NoCPatternBenchmarks patterns(create_noc(noc_->type(), config_));

    // Simulate row-wise reduction pattern
    auto reduce_result = patterns.measure_reduce_all(0, row_bytes);

    // Scale by number of rows
    result.total_cycles = reduce_result.total_cycles * rows / config_.rows / config_.cols;
    result.noc_bytes_transferred = static_cast<uint64_t>(row_bytes) * rows * 2;  // reduce + broadcast
    result.noc_tiles_transferred = rows * 2;

    result.compute_cycles = result.total_flops / (24 * 24);  // Approximate
    result.noc_wait_cycles = result.total_cycles > result.compute_cycles ?
                             result.total_cycles - result.compute_cycles : 0;

    return result;
}

OperatorResult NoCOperatorBenchmarks::benchmark_batchnorm(uint32_t batch,
                                                           uint32_t channels,
                                                           uint32_t HW) {
    OperatorResult result;
    result.operator_name = "batchnorm";
    result.config = "b=" + std::to_string(batch) +
                    " c=" + std::to_string(channels) +
                    " hw=" + std::to_string(HW);

    // BatchNorm: reduce across batch dimension for mean/var, then normalize
    // FLOPs per element: ~8 (mean, var, normalize, scale, shift)
    result.total_flops = static_cast<uint64_t>(batch) * channels * HW * 8;

    // Need all-reduce pattern for mean and variance
    uint32_t channel_bytes = batch * HW * 4;  // All spatial locations for one channel

    NoCPatternBenchmarks patterns(create_noc(noc_->type(), config_));
    auto allreduce_result = patterns.measure_allreduce(channel_bytes);

    // Scale by number of channels
    result.total_cycles = allreduce_result.total_cycles * channels /
                          (config_.rows * config_.cols);
    result.noc_bytes_transferred = allreduce_result.total_bytes * channels;
    result.noc_tiles_transferred = allreduce_result.num_transfers * channels;

    result.compute_cycles = result.total_flops / (24 * 24);
    result.noc_wait_cycles = result.total_cycles > result.compute_cycles ?
                             result.total_cycles - result.compute_cycles : 0;

    return result;
}

OperatorResult NoCOperatorBenchmarks::benchmark_layernorm(uint32_t batch, uint32_t features) {
    OperatorResult result;
    result.operator_name = "layernorm";
    result.config = "b=" + std::to_string(batch) + " f=" + std::to_string(features);

    // LayerNorm: normalize across features (local to each sample)
    // Less NoC traffic than BatchNorm since it's per-sample
    result.total_flops = static_cast<uint64_t>(batch) * features * 8;

    // Each sample is normalized independently - mostly local computation
    uint32_t sample_bytes = features * 4;

    // Minimal NoC traffic (local reduction)
    result.total_cycles = batch * (sample_bytes / 64);  // Approximate
    result.noc_bytes_transferred = 0;  // Local operation
    result.noc_tiles_transferred = 0;

    result.compute_cycles = result.total_flops / (24 * 24);
    result.noc_wait_cycles = 0;

    return result;
}

OperatorResult NoCOperatorBenchmarks::benchmark_pooling(uint32_t H, uint32_t W,
                                                         uint32_t pool_size) {
    OperatorResult result;
    result.operator_name = "pooling";
    result.config = std::to_string(H) + "x" + std::to_string(W) +
                    " pool=" + std::to_string(pool_size);

    // Pooling: local reduction within pool windows
    uint32_t out_H = H / pool_size;
    uint32_t out_W = W / pool_size;

    result.total_flops = static_cast<uint64_t>(out_H) * out_W * pool_size * pool_size;

    // Pooling is largely local - minimal NoC traffic
    result.total_cycles = out_H * out_W;
    result.noc_bytes_transferred = 0;
    result.noc_tiles_transferred = 0;
    result.compute_cycles = result.total_flops / (24 * 24);
    result.noc_wait_cycles = 0;

    return result;
}

OperatorResult NoCOperatorBenchmarks::benchmark_conv2d(uint32_t batch, uint32_t in_channels,
                                                        uint32_t out_channels,
                                                        uint32_t H, uint32_t W,
                                                        uint32_t kH, uint32_t kW) {
    OperatorResult result;
    result.operator_name = "conv2d";
    result.config = "b=" + std::to_string(batch) +
                    " in=" + std::to_string(in_channels) +
                    " out=" + std::to_string(out_channels) +
                    " hw=" + std::to_string(H) + "x" + std::to_string(W) +
                    " k=" + std::to_string(kH) + "x" + std::to_string(kW);

    // Conv2D via im2col: Convert to matmul
    // im2col matrix: [batch * out_H * out_W, in_channels * kH * kW]
    // filter matrix: [in_channels * kH * kW, out_channels]
    uint32_t out_H = H - kH + 1;
    uint32_t out_W = W - kW + 1;

    uint32_t M = batch * out_H * out_W;
    uint32_t K = in_channels * kH * kW;
    uint32_t N = out_channels;

    // Delegate to matmul benchmark
    result = benchmark_matmul(M, N, K, 64, 64, 64);
    result.operator_name = "conv2d";
    result.config = "b=" + std::to_string(batch) +
                    " in=" + std::to_string(in_channels) +
                    " out=" + std::to_string(out_channels);

    return result;
}

// ============================================================================
// NoCBenchmarkHarness Implementation
// ============================================================================

NoCBenchmarkHarness::NoCBenchmarkHarness(const NoCBenchmarkConfig& config)
    : config_(config)
{
}

NoCBenchmarkHarness::MicrobenchmarkSuite NoCBenchmarkHarness::run_microbenchmarks() {
    MicrobenchmarkSuite suite;

    auto noc = create_noc(config_.noc_type, config_.noc_config);
    NoCMicrobenchmarks micro(std::move(noc));
    micro.set_iterations(config_.warmup_iterations, config_.measurement_iterations);

    // Latency by distance
    suite.latency_by_distance = micro.sweep_latency_by_distance();

    // Latency by size
    noc = create_noc(config_.noc_type, config_.noc_config);
    NoCMicrobenchmarks micro2(std::move(noc));
    suite.latency_by_size = micro2.sweep_latency_by_size(0, 1, config_.tile_sizes);

    // Throughput
    for (uint32_t size : config_.tile_sizes) {
        noc = create_noc(config_.noc_type, config_.noc_config);
        NoCMicrobenchmarks micro3(std::move(noc));
        suite.throughput.push_back(
            micro3.measure_point_to_point_bandwidth(0, 1, size, 100));
    }

    // Contention
    noc = create_noc(config_.noc_type, config_.noc_config);
    NoCMicrobenchmarks micro4(std::move(noc));
    suite.contention.push_back(micro4.measure_center_congestion(1024));

    return suite;
}

NoCBenchmarkHarness::PatternSuite NoCBenchmarkHarness::run_pattern_benchmarks() {
    PatternSuite suite;

    // Broadcast patterns
    for (uint32_t size : {1024u, 4096u, 16384u}) {
        auto noc = create_noc(config_.noc_type, config_.noc_config);
        NoCPatternBenchmarks patterns(std::move(noc));
        suite.broadcast.push_back(patterns.measure_broadcast_all(0, size));
    }

    // Reduce patterns
    for (uint32_t size : {1024u, 4096u, 16384u}) {
        auto noc = create_noc(config_.noc_type, config_.noc_config);
        NoCPatternBenchmarks patterns(std::move(noc));
        suite.reduce.push_back(patterns.measure_reduce_all(0, size));
    }

    // Systolic patterns
    {
        auto noc = create_noc(config_.noc_type, config_.noc_config);
        NoCPatternBenchmarks patterns(std::move(noc));
        suite.systolic.push_back(patterns.measure_systolic_flow(4096, 4096));
    }

    // All-to-all
    for (uint32_t size : {256u, 1024u}) {
        auto noc = create_noc(config_.noc_type, config_.noc_config);
        NoCPatternBenchmarks patterns(std::move(noc));
        suite.all_to_all.push_back(patterns.measure_all_to_all(size));
    }

    return suite;
}

NoCBenchmarkHarness::OperatorSuite NoCBenchmarkHarness::run_operator_benchmarks() {
    OperatorSuite suite;

    // MatMul benchmarks
    for (uint32_t size : config_.matmul_sizes) {
        auto noc = create_noc(config_.noc_type, config_.noc_config);
        NoCOperatorBenchmarks ops(std::move(noc));
        suite.matmul.push_back(ops.benchmark_matmul(size, size, size, 64, 64, 64));
    }

    // Attention benchmarks
    for (uint32_t seq : config_.seq_lengths) {
        auto noc = create_noc(config_.noc_type, config_.noc_config);
        NoCOperatorBenchmarks ops(std::move(noc));
        suite.attention.push_back(
            ops.benchmark_attention(1, seq, config_.d_model, config_.num_heads));
    }

    // Reduction ops
    {
        auto noc = create_noc(config_.noc_type, config_.noc_config);
        NoCOperatorBenchmarks ops(std::move(noc));
        suite.reduction_ops.push_back(ops.benchmark_softmax(1024, 1024));
        suite.reduction_ops.push_back(ops.benchmark_batchnorm(32, 256, 1024));
        suite.reduction_ops.push_back(ops.benchmark_layernorm(32, 768));
    }

    return suite;
}

NoCBenchmarkHarness::FullResults NoCBenchmarkHarness::run_all() {
    FullResults results;
    results.noc_type = config_.noc_type;
    results.noc_config = config_.noc_config;

    results.micro = run_microbenchmarks();
    results.pattern = run_pattern_benchmarks();
    results.operators = run_operator_benchmarks();

    return results;
}

NoCBenchmarkHarness::ComparisonReport NoCBenchmarkHarness::compare(
    const FullResults& baseline, const FullResults& candidate) {

    ComparisonReport report;
    report.baseline_type = baseline.noc_type;
    report.candidate_type = candidate.noc_type;

    // Compare latency results
    for (size_t i = 0; i < std::min(baseline.micro.latency_by_distance.size(),
                                     candidate.micro.latency_by_distance.size()); ++i) {
        const auto& b = baseline.micro.latency_by_distance[i];
        const auto& c = candidate.micro.latency_by_distance[i];

        ComparisonReport::Comparison comp;
        comp.benchmark_name = "latency_" + std::to_string(b.hop_count) + "_hops";
        comp.metric = "cycles";
        comp.baseline_value = b.stats.mean();
        comp.candidate_value = c.stats.mean();
        report.comparisons.push_back(comp);
    }

    // Compare matmul results
    for (size_t i = 0; i < std::min(baseline.operators.matmul.size(),
                                     candidate.operators.matmul.size()); ++i) {
        const auto& b = baseline.operators.matmul[i];
        const auto& c = candidate.operators.matmul[i];

        ComparisonReport::Comparison comp;
        comp.benchmark_name = "matmul_" + b.config;
        comp.metric = "cycles";
        comp.baseline_value = static_cast<double>(b.total_cycles);
        comp.candidate_value = static_cast<double>(c.total_cycles);
        report.comparisons.push_back(comp);
    }

    return report;
}

std::string NoCBenchmarkHarness::ComparisonReport::to_markdown() const {
    std::ostringstream ss;

    ss << "# NoC Comparison Report\n\n";
    ss << "Baseline: " << to_string(baseline_type) << "\n";
    ss << "Candidate: " << to_string(candidate_type) << "\n\n";

    ss << "| Benchmark | Metric | Baseline | Candidate | Speedup |\n";
    ss << "|-----------|--------|----------|-----------|--------|\n";

    for (const auto& c : comparisons) {
        ss << "| " << c.benchmark_name
           << " | " << c.metric
           << " | " << std::fixed << std::setprecision(2) << c.baseline_value
           << " | " << c.candidate_value
           << " | " << c.speedup() << "x |\n";
    }

    return ss.str();
}

std::string NoCBenchmarkHarness::ComparisonReport::to_csv() const {
    std::ostringstream ss;

    ss << "benchmark,metric,baseline,candidate,speedup\n";

    for (const auto& c : comparisons) {
        ss << c.benchmark_name << ","
           << c.metric << ","
           << c.baseline_value << ","
           << c.candidate_value << ","
           << c.speedup() << "\n";
    }

    return ss.str();
}

std::string NoCBenchmarkHarness::ComparisonReport::to_json() const {
    std::ostringstream ss;

    ss << "{\n";
    ss << "  \"baseline\": \"" << to_string(baseline_type) << "\",\n";
    ss << "  \"candidate\": \"" << to_string(candidate_type) << "\",\n";
    ss << "  \"comparisons\": [\n";

    for (size_t i = 0; i < comparisons.size(); ++i) {
        const auto& c = comparisons[i];
        ss << "    {\n";
        ss << "      \"benchmark\": \"" << c.benchmark_name << "\",\n";
        ss << "      \"metric\": \"" << c.metric << "\",\n";
        ss << "      \"baseline\": " << c.baseline_value << ",\n";
        ss << "      \"candidate\": " << c.candidate_value << ",\n";
        ss << "      \"speedup\": " << c.speedup() << "\n";
        ss << "    }" << (i < comparisons.size() - 1 ? "," : "") << "\n";
    }

    ss << "  ]\n";
    ss << "}\n";

    return ss.str();
}

// ============================================================================
// NoCBenchmarkReporter Implementation
// ============================================================================

std::string NoCBenchmarkReporter::generate_summary(const NoCBenchmarkHarness::FullResults& results) {
    std::ostringstream ss;

    ss << "=== NoC Benchmark Summary ===\n\n";
    ss << "NoC Type: " << to_string(results.noc_type) << "\n";
    ss << "Mesh Size: " << static_cast<int>(results.noc_config.rows) << "x"
       << static_cast<int>(results.noc_config.cols) << "\n\n";

    // Microbenchmarks summary
    ss << "--- Latency ---\n";
    for (const auto& r : results.micro.latency_by_distance) {
        ss << "  " << static_cast<int>(r.hop_count) << " hops: "
           << std::fixed << std::setprecision(1) << r.stats.mean() << " cycles\n";
    }

    ss << "\n--- Throughput ---\n";
    for (const auto& r : results.micro.throughput) {
        ss << "  " << r.config << ": "
           << std::fixed << std::setprecision(2) << r.bytes_per_cycle() << " B/cycle\n";
    }

    ss << "\n--- Operators ---\n";
    for (const auto& r : results.operators.matmul) {
        double gflops = r.total_cycles > 0 ?
            static_cast<double>(r.total_flops) / r.total_cycles : 0.0;
        ss << "  MatMul " << r.config << ": "
           << r.total_cycles << " cycles, "
           << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
    }

    return ss.str();
}

std::string NoCBenchmarkReporter::generate_markdown_report(
    const NoCBenchmarkHarness::FullResults& results) {

    std::ostringstream ss;

    ss << "# NoC Benchmark Report\n\n";
    ss << "## Configuration\n\n";
    ss << "- NoC Type: " << to_string(results.noc_type) << "\n";
    ss << "- Mesh Size: " << static_cast<int>(results.noc_config.rows) << "x"
       << static_cast<int>(results.noc_config.cols) << "\n";
    ss << "- Buffer Depth: " << results.noc_config.buffer_depth << "\n\n";

    // Latency table
    ss << "## Latency Benchmarks\n\n";
    ss << "| Hops | Tile Size | Mean (cycles) | Min | Max |\n";
    ss << "|------|-----------|---------------|-----|-----|\n";
    for (const auto& r : results.micro.latency_by_distance) {
        ss << "| " << static_cast<int>(r.hop_count)
           << " | " << r.tile_size
           << " | " << std::fixed << std::setprecision(1) << r.stats.mean()
           << " | " << r.stats.min_cycles
           << " | " << r.stats.max_cycles << " |\n";
    }

    // Throughput table
    ss << "\n## Throughput Benchmarks\n\n";
    ss << "| Config | Bytes/Cycle | Tiles/Cycle |\n";
    ss << "|--------|-------------|-------------|\n";
    for (const auto& r : results.micro.throughput) {
        ss << "| " << r.config
           << " | " << std::fixed << std::setprecision(2) << r.bytes_per_cycle()
           << " | " << r.tiles_per_cycle() << " |\n";
    }

    // Pattern benchmarks
    ss << "\n## Pattern Benchmarks\n\n";
    ss << "| Pattern | Config | Cycles | Bytes | BW (B/cycle) |\n";
    ss << "|---------|--------|--------|-------|-------------|\n";
    for (const auto& r : results.pattern.broadcast) {
        ss << "| " << r.pattern_name
           << " | " << r.config
           << " | " << r.total_cycles
           << " | " << r.total_bytes
           << " | " << std::fixed << std::setprecision(2) << r.aggregate_bandwidth() << " |\n";
    }
    for (const auto& r : results.pattern.reduce) {
        ss << "| " << r.pattern_name
           << " | " << r.config
           << " | " << r.total_cycles
           << " | " << r.total_bytes
           << " | " << std::fixed << std::setprecision(2) << r.aggregate_bandwidth() << " |\n";
    }

    // Operator benchmarks
    ss << "\n## Operator Benchmarks\n\n";
    ss << "| Operator | Config | Cycles | GFLOPS | NoC Bound? |\n";
    ss << "|----------|--------|--------|--------|------------|\n";
    for (const auto& r : results.operators.matmul) {
        double gflops = r.total_cycles > 0 ?
            static_cast<double>(r.total_flops) / r.total_cycles : 0.0;
        ss << "| " << r.operator_name
           << " | " << r.config
           << " | " << r.total_cycles
           << " | " << std::fixed << std::setprecision(2) << gflops
           << " | " << (r.is_noc_bound() ? "Yes" : "No") << " |\n";
    }

    return ss.str();
}

std::string NoCBenchmarkReporter::generate_csv(const NoCBenchmarkHarness::FullResults& results) {
    std::ostringstream ss;

    ss << "category,benchmark,config,cycles,bytes,flops,bandwidth\n";

    for (const auto& r : results.micro.latency_by_distance) {
        ss << "latency," << r.name << "," << static_cast<int>(r.hop_count) << "_hops,"
           << r.stats.mean() << "," << r.tile_size << ",0," << r.bytes_per_cycle() << "\n";
    }

    for (const auto& r : results.micro.throughput) {
        ss << "throughput," << r.name << "," << r.config << ","
           << r.total_cycles << "," << r.total_bytes << ",0," << r.bytes_per_cycle() << "\n";
    }

    for (const auto& r : results.operators.matmul) {
        double bw = r.total_cycles > 0 ?
            static_cast<double>(r.noc_bytes_transferred) / r.total_cycles : 0.0;
        ss << "operator," << r.operator_name << "," << r.config << ","
           << r.total_cycles << "," << r.noc_bytes_transferred << ","
           << r.total_flops << "," << bw << "\n";
    }

    return ss.str();
}

std::string NoCBenchmarkReporter::generate_json(const NoCBenchmarkHarness::FullResults& results) {
    std::ostringstream ss;

    ss << "{\n";
    ss << "  \"noc_type\": \"" << to_string(results.noc_type) << "\",\n";
    ss << "  \"mesh_rows\": " << static_cast<int>(results.noc_config.rows) << ",\n";
    ss << "  \"mesh_cols\": " << static_cast<int>(results.noc_config.cols) << ",\n";

    ss << "  \"latency\": [\n";
    for (size_t i = 0; i < results.micro.latency_by_distance.size(); ++i) {
        const auto& r = results.micro.latency_by_distance[i];
        ss << "    {\"hops\": " << static_cast<int>(r.hop_count)
           << ", \"mean\": " << r.stats.mean()
           << ", \"min\": " << r.stats.min_cycles
           << ", \"max\": " << r.stats.max_cycles << "}"
           << (i < results.micro.latency_by_distance.size() - 1 ? "," : "") << "\n";
    }
    ss << "  ],\n";

    ss << "  \"operators\": [\n";
    for (size_t i = 0; i < results.operators.matmul.size(); ++i) {
        const auto& r = results.operators.matmul[i];
        ss << "    {\"name\": \"" << r.operator_name << "\""
           << ", \"config\": \"" << r.config << "\""
           << ", \"cycles\": " << r.total_cycles
           << ", \"flops\": " << r.total_flops << "}"
           << (i < results.operators.matmul.size() - 1 ? "," : "") << "\n";
    }
    ss << "  ]\n";

    ss << "}\n";

    return ss.str();
}

std::string NoCBenchmarkReporter::generate_latency_heatmap_data(
    const std::vector<LatencyResult>& results, uint8_t mesh_size) {

    std::ostringstream ss;

    ss << "# Latency heatmap data\n";
    ss << "# src dst latency_cycles\n";

    for (const auto& r : results) {
        ss << static_cast<int>(r.src_router) << " "
           << static_cast<int>(r.dst_router) << " "
           << r.stats.mean() << "\n";
    }

    return ss.str();
}

std::string NoCBenchmarkReporter::generate_gnuplot_script(
    const NoCBenchmarkHarness::FullResults& results,
    const std::string& output_prefix) {

    std::ostringstream ss;

    ss << "# Gnuplot script for NoC benchmarks\n";
    ss << "set terminal pngcairo size 800,600\n\n";

    // Latency vs hops plot
    ss << "set output '" << output_prefix << "_latency.png'\n";
    ss << "set xlabel 'Hop Count'\n";
    ss << "set ylabel 'Latency (cycles)'\n";
    ss << "set title 'NoC Latency vs Distance'\n";
    ss << "plot '-' with linespoints title '" << to_string(results.noc_type) << "'\n";
    for (const auto& r : results.micro.latency_by_distance) {
        ss << static_cast<int>(r.hop_count) << " " << r.stats.mean() << "\n";
    }
    ss << "e\n\n";

    // Throughput vs size plot
    ss << "set output '" << output_prefix << "_throughput.png'\n";
    ss << "set xlabel 'Tile Size (bytes)'\n";
    ss << "set ylabel 'Bandwidth (bytes/cycle)'\n";
    ss << "set title 'NoC Throughput vs Tile Size'\n";
    ss << "set logscale x 2\n";
    ss << "plot '-' with linespoints title 'Throughput'\n";
    for (const auto& r : results.micro.throughput) {
        // Parse tile size from config
        ss << r.total_bytes / r.total_tiles << " " << r.bytes_per_cycle() << "\n";
    }
    ss << "e\n";

    return ss.str();
}

} // namespace sw::benchmark
