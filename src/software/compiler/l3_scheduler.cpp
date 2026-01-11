/**
 * @file l3_scheduler.cpp
 * @brief Implementation of L3 cache scheduler and DRAM overfetch analysis
 */

#include <sw/compiler/l3_scheduler.hpp>
#include <sw/compiler/schedule_characterizer.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

namespace sw::kpu::compiler {

// ============================================================================
// L3CacheSimulator Implementation
// ============================================================================

L3CacheSimulator::L3CacheSimulator(const L3Config& config)
    : config_(config)
    , current_occupancy_(0)
    , peak_occupancy_(0)
    , num_evictions_(0)
    , access_counter_(0)
{
    cache_contents_.reserve(1024); // Pre-allocate for performance
}

void L3CacheSimulator::reset() {
    cache_contents_.clear();
    current_occupancy_ = 0;
    peak_occupancy_ = 0;
    num_evictions_ = 0;
    access_counter_ = 0;
}

bool L3CacheSimulator::is_cached(const TensorRegion& region) const {
    // Check if this exact region is in cache
    for (const auto& cached : cache_contents_) {
        if (cached.type == region.type &&
            cached.offset == region.offset &&
            cached.size == region.size) {
            return true;
        }
    }
    return false;
}

void L3CacheSimulator::evict_to_make_space(Size needed_space) {
    // Simple LRU: evict oldest (smallest access_time) until we have space
    while (current_occupancy_ + needed_space > config_.l3_capacity && !cache_contents_.empty()) {
        // Find LRU entry (minimum access_time)
        auto lru_it = std::min_element(cache_contents_.begin(), cache_contents_.end(),
            [](const TensorRegion& a, const TensorRegion& b) {
                return a.access_time < b.access_time;
            });

        // Evict it
        current_occupancy_ -= lru_it->size;
        cache_contents_.erase(lru_it);
        num_evictions_++;
    }
}

bool L3CacheSimulator::load(const TensorRegion& region) {
    access_counter_++;

    // Check if already cached (HIT)
    if (is_cached(region)) {
        // Update access time for LRU
        for (auto& cached : cache_contents_) {
            if (cached.type == region.type &&
                cached.offset == region.offset &&
                cached.size == region.size) {
                cached.access_time = access_counter_;
                break;
            }
        }
        return true; // HIT
    }

    // MISS - need to fetch from DRAM
    // Make space if needed
    if (current_occupancy_ + region.size > config_.l3_capacity) {
        evict_to_make_space(region.size);
    }

    // Add to cache (if it fits)
    if (region.size <= config_.l3_capacity) {
        TensorRegion new_region = region;
        new_region.access_time = access_counter_;
        cache_contents_.push_back(new_region);
        current_occupancy_ += region.size;

        // Update peak
        if (current_occupancy_ > peak_occupancy_) {
            peak_occupancy_ = current_occupancy_;
        }
    }
    // Note: If region is larger than L3, we don't cache it (streaming access)

    return false; // MISS
}

// ============================================================================
// L3Scheduler Implementation
// ============================================================================

L3Scheduler::L3Scheduler(const L3Config& config)
    : config_(config)
{}

Size L3Scheduler::calculate_ideal_dram_traffic(
    const TensorShape& shape,
    Size element_size)
{
    // Ideal: Load each tensor exactly once
    // A[M,K] + B[K,N] + C[M,N] (read) + C[M,N] (write)
    Size size_a = shape.M * shape.K * element_size;
    Size size_b = shape.K * shape.N * element_size;
    Size size_c = shape.M * shape.N * element_size;

    // Read A, B, C (initial) and write C (final)
    return size_a + size_b + size_c + size_c;
}

TensorRegion L3Scheduler::get_tile_region(
    TensorRegion::TensorType tensor_type,
    Size tile_i, Size tile_j, Size tile_k,
    const std::tuple<Size, Size, Size>& tile_dims,
    Size element_size,
    Size access_time) const
{
    auto [Ti, Tj, Tk] = tile_dims;

    Size offset = 0;
    Size size = 0;

    switch (tensor_type) {
        case TensorRegion::A:
            // A[M, K] - tile A[i*Ti : (i+1)*Ti, k*Tk : (k+1)*Tk]
            offset = (tile_i * Ti * Tk + tile_k * Tk) * element_size;
            size = Ti * Tk * element_size;
            break;

        case TensorRegion::B:
            // B[K, N] - tile B[k*Tk : (k+1)*Tk, j*Tj : (j+1)*Tj]
            offset = (tile_k * Tk * Tj + tile_j * Tj) * element_size;
            size = Tk * Tj * element_size;
            break;

        case TensorRegion::C:
            // C[M, N] - tile C[i*Ti : (i+1)*Ti, j*Tj : (j+1)*Tj]
            offset = (tile_i * Ti * Tj + tile_j * Tj) * element_size;
            size = Ti * Tj * element_size;
            break;
    }

    return TensorRegion(tensor_type, offset, size, access_time);
}

L3Schedule L3Scheduler::simulate_l2_execution(
    const TensorShape& shape,
    const L2TileScheduler::L2Schedule& l2_schedule,
    Size element_size,
    L3CacheSimulator& simulator)
{
    L3Schedule result;
    result.config = config_;

    // Calculate ideal tensor sizes
    result.tensor_a_size = shape.M * shape.K * element_size;
    result.tensor_b_size = shape.K * shape.N * element_size;
    result.tensor_c_size = shape.M * shape.N * element_size;

    // Get tile dimensions from L2 schedule
    Size Ti = l2_schedule.config.Ti;
    Size Tj = l2_schedule.config.Tj;
    Size Tk = l2_schedule.config.Tk;
    auto tile_dims = std::make_tuple(Ti, Tj, Tk);

    // Calculate number of tiles in each dimension
    Size num_tiles_i = (shape.M + Ti - 1) / Ti;
    Size num_tiles_j = (shape.N + Tj - 1) / Tj;
    Size num_tiles_k = (shape.K + Tk - 1) / Tk;

    // Track bytes fetched per tensor
    Size bytes_fetched_a = 0;
    Size bytes_fetched_b = 0;
    Size bytes_fetched_c = 0;

    Size access_time = 0;

    // Simulate the L2 tile execution order using the strategy from L2 schedule
    // This respects the dataflow strategy (WS, IS, OS)

    // Track which C tiles we've loaded to avoid double-loading
    std::set<std::pair<Size, Size>> loaded_c_tiles;

    // Helper lambda to process a single (ti, tj, tk) compute operation
    auto process_compute = [&](Size i, Size j, Size k) {
        // Load C tile once per unique (i,j)
        if (loaded_c_tiles.find({i, j}) == loaded_c_tiles.end()) {
            TensorRegion c_tile = get_tile_region(
                TensorRegion::C, i, j, 0, tile_dims, element_size, access_time++);

            bool c_hit = simulator.load(c_tile);
            result.l2_tile_loads_total++;
            if (c_hit) {
                result.l2_tile_loads_hit_l3++;
            } else {
                result.l2_tile_loads_miss_l3++;
                bytes_fetched_c += c_tile.size;
            }
            loaded_c_tiles.insert({i, j});
        }

        // Load A tile A[i, k]
        TensorRegion a_tile = get_tile_region(
            TensorRegion::A, i, 0, k, tile_dims, element_size, access_time++);

        bool a_hit = simulator.load(a_tile);
        result.l2_tile_loads_total++;
        if (a_hit) {
            result.l2_tile_loads_hit_l3++;
        } else {
            result.l2_tile_loads_miss_l3++;
            bytes_fetched_a += a_tile.size;
        }

        // Load B tile B[k, j]
        TensorRegion b_tile = get_tile_region(
            TensorRegion::B, 0, j, k, tile_dims, element_size, access_time++);

        bool b_hit = simulator.load(b_tile);
        result.l2_tile_loads_total++;
        if (b_hit) {
            result.l2_tile_loads_hit_l3++;
        } else {
            result.l2_tile_loads_miss_l3++;
            bytes_fetched_b += b_tile.size;
        }

        // Compute: C[i,j] += A[i,k] * B[k,j]
        result.l3_to_l2_bytes += a_tile.size + b_tile.size;
    };

    // Execute according to the L2 schedule's strategy
    switch (l2_schedule.strategy) {
        case L2TileScheduler::SchedulingStrategy::WEIGHT_STATIONARY:
            // tk → ti → tj: Keep B tiles resident
            for (Size k = 0; k < num_tiles_k; ++k) {
                for (Size i = 0; i < num_tiles_i; ++i) {
                    for (Size j = 0; j < num_tiles_j; ++j) {
                        process_compute(i, j, k);
                    }
                }
            }
            break;

        case L2TileScheduler::SchedulingStrategy::INPUT_STATIONARY:
            // tk → tj → ti: Keep A tiles resident
            for (Size k = 0; k < num_tiles_k; ++k) {
                for (Size j = 0; j < num_tiles_j; ++j) {
                    for (Size i = 0; i < num_tiles_i; ++i) {
                        process_compute(i, j, k);
                    }
                }
            }
            break;

        case L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY:
        default:
            // ti → tj → tk: Keep C tiles resident (output-stationary)
            for (Size i = 0; i < num_tiles_i; ++i) {
                for (Size j = 0; j < num_tiles_j; ++j) {
                    for (Size k = 0; k < num_tiles_k; ++k) {
                        process_compute(i, j, k);
                    }
                }
            }
            break;
    }

    // Store fetched bytes
    result.tensor_a_fetched = bytes_fetched_a;
    result.tensor_b_fetched = bytes_fetched_b;
    result.tensor_c_fetched = bytes_fetched_c;

    // Calculate overfetch factors
    result.overfetch_factor_a = static_cast<double>(bytes_fetched_a) / result.tensor_a_size;
    result.overfetch_factor_b = static_cast<double>(bytes_fetched_b) / result.tensor_b_size;
    result.overfetch_factor_c = static_cast<double>(bytes_fetched_c) / result.tensor_c_size;

    Size total_ideal = result.tensor_a_size + result.tensor_b_size + result.tensor_c_size;
    Size total_fetched = bytes_fetched_a + bytes_fetched_b + bytes_fetched_c;
    result.overfetch_factor_total = static_cast<double>(total_fetched) / total_ideal;

    // DRAM traffic
    result.dram_to_l3_bytes = total_fetched;
    result.total_dram_reads = total_fetched;
    result.total_dram_writes = result.tensor_c_size; // Write C back

    // L3 utilization
    result.l3_capacity_used_peak = simulator.get_peak_occupancy();
    result.l3_utilization = static_cast<double>(result.l3_capacity_used_peak) / config_.l3_capacity;
    result.num_l3_evictions = simulator.get_num_evictions();

    // Hit rate
    if (result.l2_tile_loads_total > 0) {
        result.l3_hit_rate = static_cast<double>(result.l2_tile_loads_hit_l3) / result.l2_tile_loads_total;
    }

    return result;
}

L3Schedule L3Scheduler::schedule_l3(
    const TensorShape& shape,
    const L2TileScheduler::L2Schedule& l2_schedule,
    Size element_size)
{
    // Create cache simulator
    L3CacheSimulator simulator(config_);

    // Simulate L2 schedule execution
    return simulate_l2_execution(shape, l2_schedule, element_size, simulator);
}

std::map<Size, L3Schedule> L3Scheduler::sweep_l3_size(
    const TensorShape& shape,
    const L2TileScheduler::L2Schedule& l2_schedule,
    const std::vector<Size>& l3_sizes,
    Size element_size)
{
    std::map<Size, L3Schedule> results;

    for (Size l3_size : l3_sizes) {
        // Create temporary config with this L3 size
        L3Config temp_config = config_;
        temp_config.l3_capacity = l3_size;

        // Create scheduler with this config
        L3Scheduler temp_scheduler(temp_config);

        // Schedule and store result
        results[l3_size] = temp_scheduler.schedule_l3(shape, l2_schedule, element_size);
    }

    return results;
}

} // namespace sw::kpu::compiler
