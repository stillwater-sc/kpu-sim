/**
 * @file l2_tile_scheduler.cpp
 * @brief Implementation of L2 tile scheduler for KPU matrix multiplication
 */

#include <sw/compiler/l2_tile_scheduler.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace sw::kpu::compiler {

L2TileScheduler::L2TileScheduler(
    const TileOptimizer::MemoryHierarchy& mem,
    Size systolic_size)
    : memory_(mem)
    , systolic_size_(systolic_size)
    , policy_(ReplacementPolicy::LRU)
    , strategy_(SchedulingStrategy::OUTPUT_STATIONARY)
{
}

L2TileScheduler::L2Schedule L2TileScheduler::generate_schedule(
    Size M, Size N, Size K,
    const TileOptimizer::TileConfig& config,
    ReplacementPolicy policy,
    SchedulingStrategy strategy)
{
    L2Schedule schedule;

    // Set basic parameters
    schedule.M = M;
    schedule.N = N;
    schedule.K = K;
    schedule.config = config;
    schedule.strategy = strategy;

    // Calculate tile grid dimensions
    schedule.num_tile_rows_A = (M + config.Ti - 1) / config.Ti;
    schedule.num_tile_cols_A = (K + config.Tk - 1) / config.Tk;
    schedule.num_tile_rows_B = (K + config.Tk - 1) / config.Tk;
    schedule.num_tile_cols_B = (N + config.Tj - 1) / config.Tj;
    schedule.num_tile_rows_C = (M + config.Ti - 1) / config.Ti;
    schedule.num_tile_cols_C = (N + config.Tj - 1) / config.Tj;

    // L2 configuration
    schedule.num_l2_banks = memory_.L2_bank_count;
    schedule.l2_bank_size = memory_.L2_size;
    schedule.l2_total_capacity = schedule.num_l2_banks * schedule.l2_bank_size;

    // Calculate how many tiles can fit in L2
    Size avg_tile_size = (config.Ti * config.Tk + // A tile
                         config.Tk * config.Tj +  // B tile
                         config.Ti * config.Tj)   // C tile
                         * memory_.element_size / 3; // Average
    schedule.max_l2_slots = schedule.l2_total_capacity / avg_tile_size;

    // Enforce user constraint: assume at least 128 tiles
    schedule.max_l2_slots = std::max(schedule.max_l2_slots, Size(128));

    // Store policy and strategy
    policy_ = policy;
    strategy_ = strategy;

    // Allocate initial tiles
    allocate_initial_tiles(schedule);

    // Generate load sequence
    generate_load_sequence(schedule);

    // Calculate statistics
    calculate_reuse_stats(schedule);

    return schedule;
}

void L2TileScheduler::allocate_initial_tiles(L2Schedule& schedule) {
    // Determine initial tile set to load
    // Strategy: Load as many unique A and B tiles as possible
    // C tiles accumulate in the systolic array (output-stationary)

    std::set<TileID> initial_tiles;

    // For output-stationary: we want to minimize reloads by loading
    // tiles that will be reused the most

    // Note: num_a_tiles and num_b_tiles calculated but reserved for future capacity planning
    // Size num_a_tiles = schedule.num_tile_rows_A * schedule.num_tile_cols_A;
    // Size num_b_tiles = schedule.num_tile_rows_B * schedule.num_tile_cols_B;

    // Add A tiles
    for (Size ti = 0; ti < schedule.num_tile_rows_A; ++ti) {
        for (Size tk = 0; tk < schedule.num_tile_cols_A; ++tk) {
            initial_tiles.insert(TileID{'A', ti, tk});
        }
    }

    // Add B tiles
    for (Size tk = 0; tk < schedule.num_tile_rows_B; ++tk) {
        for (Size tj = 0; tj < schedule.num_tile_cols_B; ++tj) {
            initial_tiles.insert(TileID{'B', tk, tj});
        }
    }

    // Note: C tiles stay in systolic array PEs (output-stationary)
    // We may load them into L2 for accumulation if K > Tk

    // Allocate L2 slots for these tiles
    schedule.slots.clear();
    size_t slot_idx = 0;

    for (const auto& tile_id : initial_tiles) {
        if (slot_idx >= schedule.max_l2_slots) {
            break; // L2 full
        }

        // Calculate tile size
        Size tile_bytes = calculate_tile_bytes(tile_id, schedule);

        // Round-robin across L2 banks
        size_t bank_id = slot_idx % schedule.num_l2_banks;
        Address offset = (slot_idx / schedule.num_l2_banks) *
                        (schedule.l2_bank_size / (schedule.max_l2_slots / schedule.num_l2_banks));

        L2Slot slot(bank_id, offset, tile_bytes);
        slot.tile = tile_id;

        schedule.slots.push_back(slot);
        slot_idx++;
    }
}

void L2TileScheduler::generate_load_sequence(L2Schedule& schedule) {
    schedule.load_sequence.clear();

    // Generate compute order
    auto compute_order = generate_compute_order(schedule);

    Size time_step = 0;

    // Track L3 cache state (simple model: assume L3 holds recent tiles)
    std::set<TileID> l3_cache;
    // Reserved for future L3 capacity-aware scheduling
    // Size l3_capacity_tiles = (memory_.L3_size * memory_.L3_tile_count) /
    //                         (schedule.config.Ti * schedule.config.Tk * memory_.element_size);

    // Process each compute operation
    for (const auto& [ti, tj, tk] : compute_order) {
        // Determine required tiles
        auto required = get_required_tiles(ti, tj, tk);

        // Check and load A tile
        if (!is_resident(schedule, required.a_tile)) {
            auto slot_idx = find_or_allocate_slot(schedule, required.a_tile, time_step);

            if (!slot_idx) {
                // Need to evict
                auto evicted_slot = evict_tile(schedule, time_step);
                if (evicted_slot) {
                    slot_idx = evicted_slot;
                    // Allocate in the evicted slot
                    schedule.slots[*slot_idx].tile = required.a_tile;
                    schedule.slots[*slot_idx].size_bytes = calculate_tile_bytes(required.a_tile, schedule);
                } else {
                    std::cerr << "ERROR: Cannot evict tile for " << required.a_tile.to_string() << std::endl;
                    continue;
                }
            }

            // Record tile load
            TileLoad load;
            load.tile_id = required.a_tile;
            load.slot_index = *slot_idx;
            load.time_step = time_step;
            load.compute_ti = ti;
            load.compute_tj = tj;
            load.compute_tk = tk;

            // Check if in L3
            bool in_l3 = is_in_l3(required.a_tile, schedule);
            load.from_dram = !in_l3;
            load.type = (schedule.tile_load_count[required.a_tile] == 0)
                       ? TileLoad::Type::INITIAL_LOAD
                       : TileLoad::Type::RELOAD;

            schedule.load_sequence.push_back(load);
            schedule.tile_load_count[required.a_tile]++;

            // Update L3 state
            update_l3_state(required.a_tile, schedule);
        }

        // Update last access time
        tile_last_access_[required.a_tile] = time_step;

        // Check and load B tile
        if (!is_resident(schedule, required.b_tile)) {
            auto slot_idx = find_or_allocate_slot(schedule, required.b_tile, time_step);

            if (!slot_idx) {
                // Need to evict
                auto evicted_slot = evict_tile(schedule, time_step);
                if (evicted_slot) {
                    slot_idx = evicted_slot;
                    schedule.slots[*slot_idx].tile = required.b_tile;
                    schedule.slots[*slot_idx].size_bytes = calculate_tile_bytes(required.b_tile, schedule);
                }
            }

            // Record tile load
            TileLoad load;
            load.tile_id = required.b_tile;
            load.slot_index = *slot_idx;
            load.time_step = time_step;
            load.compute_ti = ti;
            load.compute_tj = tj;
            load.compute_tk = tk;

            bool in_l3 = is_in_l3(required.b_tile, schedule);
            load.from_dram = !in_l3;
            load.type = (schedule.tile_load_count[required.b_tile] == 0)
                       ? TileLoad::Type::INITIAL_LOAD
                       : TileLoad::Type::RELOAD;

            schedule.load_sequence.push_back(load);
            schedule.tile_load_count[required.b_tile]++;

            update_l3_state(required.b_tile, schedule);
        }

        tile_last_access_[required.b_tile] = time_step;

        // Track tile access
        schedule.tile_access_count[required.a_tile]++;
        schedule.tile_access_count[required.b_tile]++;
        schedule.tile_access_count[required.c_tile]++;

        time_step++;
    }

    // Calculate statistics
    schedule.total_loads = schedule.load_sequence.size();
    schedule.initial_loads = 0;
    schedule.reloads = 0;
    schedule.l3_hits = 0;
    schedule.l3_misses = 0;
    schedule.total_bytes_loaded = 0;

    for (const auto& load : schedule.load_sequence) {
        if (load.type == TileLoad::Type::INITIAL_LOAD) {
            schedule.initial_loads++;
        } else if (load.type == TileLoad::Type::RELOAD) {
            schedule.reloads++;
        }

        if (load.from_dram) {
            schedule.l3_misses++;
        } else {
            schedule.l3_hits++;
        }

        schedule.total_bytes_loaded += calculate_tile_bytes(load.tile_id, schedule);
    }

    // Calculate hit rates
    Size total_accesses = 0;
    for (const auto& [tile, count] : schedule.tile_access_count) {
        total_accesses += count;
    }

    Size l2_hits = total_accesses - schedule.total_loads;
    schedule.l2_hit_rate = total_accesses > 0
                          ? (100.0 * l2_hits) / total_accesses
                          : 0.0;

    schedule.l3_hit_rate = schedule.total_loads > 0
                          ? (100.0 * schedule.l3_hits) / schedule.total_loads
                          : 0.0;
}

std::optional<size_t> L2TileScheduler::find_or_allocate_slot(
    L2Schedule& schedule,
    const TileID& tile_id,
    Size time_step)
{
    (void)time_step;  // Reserved for time-aware scheduling

    // Check if tile is already resident
    auto slot_idx = get_slot_index(schedule, tile_id);
    if (slot_idx) {
        return slot_idx;
    }

    // Look for a free slot
    for (size_t i = 0; i < schedule.slots.size(); ++i) {
        if (schedule.slots[i].is_free()) {
            return i;
        }
    }

    // No free slot found
    return std::nullopt;
}

std::optional<size_t> L2TileScheduler::evict_tile(
    L2Schedule& schedule,
    Size time_step)
{
    return select_victim(schedule, time_step);
}

bool L2TileScheduler::is_resident(const L2Schedule& schedule, const TileID& tile_id) const {
    return get_slot_index(schedule, tile_id).has_value();
}

std::optional<size_t> L2TileScheduler::get_slot_index(
    const L2Schedule& schedule,
    const TileID& tile_id) const
{
    for (size_t i = 0; i < schedule.slots.size(); ++i) {
        if (schedule.slots[i].tile && *schedule.slots[i].tile == tile_id) {
            return i;
        }
    }
    return std::nullopt;
}

void L2TileScheduler::calculate_reuse_stats(L2Schedule& schedule) const {
    (void)schedule;  // Stats already calculated in generate_load_sequence
}

Size L2TileScheduler::calculate_tile_bytes(
    const TileID& tile_id,
    const L2Schedule& schedule) const
{
    Size rows = 0, cols = 0;

    if (tile_id.matrix == 'A') {
        rows = get_tile_dimension(schedule.M, tile_id.row_idx, schedule.config.Ti);
        cols = get_tile_dimension(schedule.K, tile_id.col_idx, schedule.config.Tk);
    } else if (tile_id.matrix == 'B') {
        rows = get_tile_dimension(schedule.K, tile_id.row_idx, schedule.config.Tk);
        cols = get_tile_dimension(schedule.N, tile_id.col_idx, schedule.config.Tj);
    } else { // 'C'
        rows = get_tile_dimension(schedule.M, tile_id.row_idx, schedule.config.Ti);
        cols = get_tile_dimension(schedule.N, tile_id.col_idx, schedule.config.Tj);
    }

    return rows * cols * memory_.element_size;
}

std::vector<std::tuple<Size, Size, Size>> L2TileScheduler::generate_compute_order(
    const L2Schedule& schedule) const
{
    std::vector<std::tuple<Size, Size, Size>> order;

    switch (strategy_) {
        case SchedulingStrategy::WEIGHT_STATIONARY:
            // Weight-stationary: tk → ti → tj
            // Keep B tiles resident across output tiles
            // Best for: WIDE and DEEP matrices (large B)
            for (Size tk = 0; tk < schedule.num_tile_cols_A; ++tk) {
                for (Size ti = 0; ti < schedule.num_tile_rows_C; ++ti) {
                    for (Size tj = 0; tj < schedule.num_tile_cols_C; ++tj) {
                        order.push_back(std::make_tuple(ti, tj, tk));
                    }
                }
            }
            break;

        case SchedulingStrategy::INPUT_STATIONARY:
            // Input-stationary: tk → tj → ti
            // Keep A tiles resident across output tiles
            // Best for: TALL matrices (large A)
            for (Size tk = 0; tk < schedule.num_tile_cols_A; ++tk) {
                for (Size tj = 0; tj < schedule.num_tile_cols_C; ++tj) {
                    for (Size ti = 0; ti < schedule.num_tile_rows_C; ++ti) {
                        order.push_back(std::make_tuple(ti, tj, tk));
                    }
                }
            }
            break;

        case SchedulingStrategy::OUTPUT_STATIONARY:
        default:
            // Output-stationary: ti → tj → tk
            // Keep C tiles resident, accumulate across K
            // Best for: Small C, or when both A and B fit in L3
            for (Size ti = 0; ti < schedule.num_tile_rows_C; ++ti) {
                for (Size tj = 0; tj < schedule.num_tile_cols_C; ++tj) {
                    for (Size tk = 0; tk < schedule.num_tile_cols_A; ++tk) {
                        order.push_back(std::make_tuple(ti, tj, tk));
                    }
                }
            }
            break;
    }

    return order;
}

std::optional<size_t> L2TileScheduler::select_victim(
    const L2Schedule& schedule,
    Size time_step) const
{
    switch (policy_) {
        case ReplacementPolicy::LRU:
            return select_lru_victim(schedule);
        case ReplacementPolicy::OPTIMAL:
            return select_optimal_victim(schedule, time_step);
        case ReplacementPolicy::FIFO:
        case ReplacementPolicy::MANUAL:
        default:
            // Fall back to LRU
            return select_lru_victim(schedule);
    }
}

std::optional<size_t> L2TileScheduler::select_lru_victim(
    const L2Schedule& schedule) const
{
    Size oldest_time = std::numeric_limits<Size>::max();
    std::optional<size_t> victim_idx;

    for (size_t i = 0; i < schedule.slots.size(); ++i) {
        if (schedule.slots[i].tile) {
            auto it = tile_last_access_.find(*schedule.slots[i].tile);
            Size last_access = (it != tile_last_access_.end()) ? it->second : 0;

            if (last_access < oldest_time) {
                oldest_time = last_access;
                victim_idx = i;
            }
        }
    }

    return victim_idx;
}

std::optional<size_t> L2TileScheduler::select_optimal_victim(
    const L2Schedule& schedule,
    Size time_step) const
{
    (void)time_step;  // Reserved for Belady's optimal algorithm

    // Belady's optimal: evict the tile that will be used furthest in the future
    // This requires future knowledge, which we have since we know the compute order

    // For now, use LRU as a placeholder
    // TODO: Implement optimal lookahead
    return select_lru_victim(schedule);
}

bool L2TileScheduler::is_in_l3(const TileID& tile_id, const L2Schedule& schedule) const {
    // Simple L3 model: assume tiles stay in L3 after first load
    // In reality, L3 also has capacity constraints
    return schedule.tile_load_count.find(tile_id) != schedule.tile_load_count.end() &&
           schedule.tile_load_count.at(tile_id) > 0;
}

void L2TileScheduler::update_l3_state(const TileID& tile_id, L2Schedule& schedule) {
    (void)tile_id;    // Reserved for L3 capacity tracking
    (void)schedule;   // Reserved for L3 eviction modeling
    // Mark tile as loaded into L3
    // In a more sophisticated model, we would track L3 capacity and evictions
}

void L2TileScheduler::print_schedule(const L2Schedule& schedule, bool verbose) const {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              L2 TILE SCHEDULE SUMMARY                              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Matrix dimensions
    std::cout << "Matrix Dimensions:\n";
    std::cout << "  M = " << schedule.M << ", N = " << schedule.N << ", K = " << schedule.K << "\n";
    std::cout << "\n";

    // Tile configuration
    std::cout << "Tile Configuration:\n";
    std::cout << "  Ti = " << schedule.config.Ti
              << ", Tj = " << schedule.config.Tj
              << ", Tk = " << schedule.config.Tk << "\n";
    std::cout << "\n";

    // Tile grid
    std::cout << "Tile Grid:\n";
    std::cout << "  A: " << schedule.num_tile_rows_A << " x " << schedule.num_tile_cols_A
              << " = " << (schedule.num_tile_rows_A * schedule.num_tile_cols_A) << " tiles\n";
    std::cout << "  B: " << schedule.num_tile_rows_B << " x " << schedule.num_tile_cols_B
              << " = " << (schedule.num_tile_rows_B * schedule.num_tile_cols_B) << " tiles\n";
    std::cout << "  C: " << schedule.num_tile_rows_C << " x " << schedule.num_tile_cols_C
              << " = " << (schedule.num_tile_rows_C * schedule.num_tile_cols_C) << " tiles\n";
    std::cout << "\n";

    // L2 configuration
    std::cout << "L2 Configuration:\n";
    std::cout << "  Banks: " << schedule.num_l2_banks << "\n";
    std::cout << "  Bank Size: " << (schedule.l2_bank_size / 1024) << " KB\n";
    std::cout << "  Total Capacity: " << (schedule.l2_total_capacity / 1024) << " KB\n";
    std::cout << "  Max Slots: " << schedule.max_l2_slots << " tiles\n";
    std::cout << "  Allocated Slots: " << schedule.slots.size() << " tiles\n";
    std::cout << "\n";

    // Load statistics
    std::cout << "Load Statistics:\n";
    std::cout << "  Total Loads: " << schedule.total_loads << "\n";
    std::cout << "  Initial Loads: " << schedule.initial_loads << "\n";
    std::cout << "  Reloads: " << schedule.reloads << "\n";
    std::cout << "  L3 Hits: " << schedule.l3_hits << "\n";
    std::cout << "  L3 Misses (DRAM): " << schedule.l3_misses << "\n";
    std::cout << "\n";

    std::cout << "Hit Rates:\n";
    std::cout << "  L2 Hit Rate: " << std::fixed << std::setprecision(2)
              << schedule.l2_hit_rate << "%\n";
    std::cout << "  L3 Hit Rate: " << schedule.l3_hit_rate << "%\n";
    std::cout << "\n";

    std::cout << "Data Movement:\n";
    std::cout << "  Total Bytes Loaded (L3 -> L2): "
              << (schedule.total_bytes_loaded / 1024.0 / 1024.0) << " MB\n";
    std::cout << "\n";

    if (verbose) {
        print_l2_state(schedule);
        print_load_sequence(schedule, 50);
        print_reuse_stats(schedule);
    }
}

void L2TileScheduler::print_l2_state(const L2Schedule& schedule) const {
    std::cout << "L2 Slot Allocations:\n";
    std::cout << "  ┌──────┬────────┬──────────┬────────────┬────────────┐\n";
    std::cout << "  │ Slot │  Bank  │  Offset  │    Size    │    Tile    │\n";
    std::cout << "  ├──────┼────────┼──────────┼────────────┼────────────┤\n";

    for (size_t i = 0; i < std::min(schedule.slots.size(), Size(20)); ++i) {
        const auto& slot = schedule.slots[i];
        std::cout << "  │ " << std::setw(4) << i << " │ "
                  << std::setw(6) << slot.bank_id << " │ "
                  << std::setw(8) << slot.offset << " │ "
                  << std::setw(8) << (slot.size_bytes / 1024) << "KB │ ";

        if (slot.tile) {
            std::cout << std::setw(10) << slot.tile->to_string();
        } else {
            std::cout << std::setw(10) << "[FREE]";
        }
        std::cout << " │\n";
    }

    if (schedule.slots.size() > 200) {
        std::cout << "  │  ... │   ...  │   ...    │    ...     │    ...     │\n";
        std::cout << "  │      │        │          │            │ (+" << (schedule.slots.size() - 200)
                  << " more) │\n";
    }

    std::cout << "  └──────┴────────┴──────────┴────────────┴────────────┘\n";
    std::cout << "\n";
}

void L2TileScheduler::print_load_sequence(const L2Schedule& schedule, Size max_entries) const {
    std::cout << "Tile Load Sequence:\n";

    if (schedule.load_sequence.empty()) {
        std::cout << "  (No loads recorded)\n\n";
        return;
    }

    std::cout << "  ┌──────┬──────────┬────────────┬──────────┬────────────────┐\n";
    std::cout << "  │ Time │   Type   │    Tile    │   Slot   │     Source     │\n";
    std::cout << "  ├──────┼──────────┼────────────┼──────────┼────────────────┤\n";

    for (size_t i = 0; i < std::min(schedule.load_sequence.size(), max_entries); ++i) {
        const auto& load = schedule.load_sequence[i];

        std::string type_str;
        switch (load.type) {
            case TileLoad::Type::INITIAL_LOAD: type_str = "LOAD"; break;
            case TileLoad::Type::RELOAD: type_str = "RELOAD"; break;
            case TileLoad::Type::PREFETCH: type_str = "PREFETCH"; break;
        }

        std::string source_str = load.from_dram ? "DRAM" : "L3";

        std::cout << "  │ " << std::setw(4) << load.time_step << " │ "
                  << std::setw(8) << type_str << " │ "
                  << std::setw(10) << load.tile_id.to_string() << " │ "
                  << std::setw(8) << load.slot_index << " │ "
                  << std::setw(14) << source_str << " │\n";
    }

    if (schedule.load_sequence.size() > max_entries) {
        std::cout << "  │  ... │   ...    │    ...     │   ...    │      ...       │\n";
        std::cout << "  │      │          │            │          │ (+" << (schedule.load_sequence.size() - max_entries)
                  << " more)  │\n";
    }

    std::cout << "  └──────┴──────────┴────────────┴──────────┴────────────────┘\n";
    std::cout << "\n";
}

void L2TileScheduler::print_reuse_stats(const L2Schedule& schedule) const {
    std::cout << "Tile Reuse Statistics:\n";

    if (schedule.tile_access_count.empty()) {
        std::cout << "  (No reuse data)\n\n";
        return;
    }

    // Compute summary stats
    Size total_accesses = 0;
    Size total_loads = 0;
    Size min_reuse = std::numeric_limits<Size>::max();
    Size max_reuse = 0;

    for (const auto& [tile_id, access_count] : schedule.tile_access_count) {
        total_accesses += access_count;
        Size load_count = schedule.tile_load_count.count(tile_id)
                         ? schedule.tile_load_count.at(tile_id)
                         : 0;
        total_loads += load_count;

        Size reuse = access_count - load_count;
        min_reuse = std::min(min_reuse, reuse);
        max_reuse = std::max(max_reuse, reuse);
    }

    double avg_reuse = schedule.tile_access_count.size() > 0
                      ? (double)(total_accesses - total_loads) / schedule.tile_access_count.size()
                      : 0.0;

    std::cout << "  Total Accesses: " << total_accesses << "\n";
    std::cout << "  Total Loads: " << total_loads << "\n";
    std::cout << "  Total Reuses: " << (total_accesses - total_loads) << "\n";
    std::cout << "  Average Reuse: " << std::fixed << std::setprecision(2) << avg_reuse << "\n";
    std::cout << "  Min Reuse: " << min_reuse << "\n";
    std::cout << "  Max Reuse: " << max_reuse << "\n";
    std::cout << "\n";
}

std::string L2TileScheduler::export_json(const L2Schedule& schedule) const {
    std::ostringstream oss;

    oss << "{\n";
    oss << "  \"matrix_dimensions\": {\n";
    oss << "    \"M\": " << schedule.M << ",\n";
    oss << "    \"N\": " << schedule.N << ",\n";
    oss << "    \"K\": " << schedule.K << "\n";
    oss << "  },\n";

    oss << "  \"tile_config\": {\n";
    oss << "    \"Ti\": " << schedule.config.Ti << ",\n";
    oss << "    \"Tj\": " << schedule.config.Tj << ",\n";
    oss << "    \"Tk\": " << schedule.config.Tk << "\n";
    oss << "  },\n";

    oss << "  \"statistics\": {\n";
    oss << "    \"total_loads\": " << schedule.total_loads << ",\n";
    oss << "    \"initial_loads\": " << schedule.initial_loads << ",\n";
    oss << "    \"reloads\": " << schedule.reloads << ",\n";
    oss << "    \"l3_hits\": " << schedule.l3_hits << ",\n";
    oss << "    \"l3_misses\": " << schedule.l3_misses << ",\n";
    oss << "    \"l2_hit_rate\": " << schedule.l2_hit_rate << ",\n";
    oss << "    \"l3_hit_rate\": " << schedule.l3_hit_rate << "\n";
    oss << "  }\n";

    oss << "}\n";

    return oss.str();
}

void L2TileScheduler::visualize_l2_timeline(const L2Schedule& schedule) const {
    (void)schedule;  // Reserved for timeline visualization implementation
    std::cout << "L2 Timeline Visualization:\n";
    std::cout << "  (Not yet implemented)\n";
    std::cout << "\n";
}

} // namespace sw::kpu::compiler
