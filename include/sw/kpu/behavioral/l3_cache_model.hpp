// ============================================================================
// include/sw/kpu/behavioral/l3_cache_model.hpp
// L3 Cache Model with LRU Replacement for Tile Reuse Tracking
// ============================================================================
//
// Models the L3 on-chip cache for tiled matrix operations, tracking:
// - Cache hits: tile already resident
// - Cache misses: first-time load
// - Refetches: tile was evicted and reloaded
//
// ============================================================================

#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <functional>
#include <string>

namespace sw::kpu::behavioral {

/// Operand type for cache tracking
enum class CacheOperandType : uint8_t {
    A = 0,
    B = 1,
    C = 2,
    D = 3
};

inline const char* to_string(CacheOperandType type) {
    switch (type) {
        case CacheOperandType::A: return "A";
        case CacheOperandType::B: return "B";
        case CacheOperandType::C: return "C";
        case CacheOperandType::D: return "D";
        default: return "?";
    }
}

/// Replacement policy for the cache
enum class ReplacementPolicy : uint8_t {
    LRU,        // Least Recently Used
    FIFO,       // First In, First Out
    RANDOM      // Random eviction
};

/// Result of a cache access
enum class CacheAccessResult : uint8_t {
    HIT,        // Tile already in cache
    MISS,       // First-time load
    REFETCH     // Tile was evicted and is being reloaded
};

inline const char* to_string(CacheAccessResult result) {
    switch (result) {
        case CacheAccessResult::HIT: return "HIT";
        case CacheAccessResult::MISS: return "MISS";
        case CacheAccessResult::REFETCH: return "REFETCH";
        default: return "?";
    }
}

/// Key for identifying a tile in the cache
struct TileKey {
    CacheOperandType operand;
    uint16_t tile_i;
    uint16_t tile_j;

    bool operator==(const TileKey& other) const {
        return operand == other.operand &&
               tile_i == other.tile_i &&
               tile_j == other.tile_j;
    }

    std::string to_string() const {
        return std::string(sw::kpu::behavioral::to_string(operand)) +
               "[" + std::to_string(tile_i) + "," + std::to_string(tile_j) + "]";
    }
};

/// Hash function for TileKey
struct TileKeyHash {
    std::size_t operator()(const TileKey& key) const {
        std::size_t h1 = static_cast<std::size_t>(key.operand);
        std::size_t h2 = static_cast<std::size_t>(key.tile_i);
        std::size_t h3 = static_cast<std::size_t>(key.tile_j);
        return h1 ^ (h2 << 8) ^ (h3 << 16);
    }
};

/// Entry stored in the cache
struct CacheEntry {
    TileKey key;
    uint64_t load_cycle;        // When first loaded
    uint64_t last_access_cycle; // When last accessed
    uint32_t access_count;      // How many times accessed
    bool dirty;                 // Modified since load
};

/// Statistics for cache performance
struct L3CacheStats {
    // Overall statistics
    uint64_t total_accesses = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;        // First-time loads
    uint64_t refetches = 0;     // Re-loads after eviction
    uint64_t evictions = 0;

    // Per-operand breakdown
    uint64_t hits_by_operand[4] = {0};
    uint64_t misses_by_operand[4] = {0};
    uint64_t refetches_by_operand[4] = {0};
    uint64_t evictions_by_operand[4] = {0};

    // Derived metrics
    double hit_rate() const {
        return total_accesses > 0 ?
            static_cast<double>(hits) / total_accesses : 0.0;
    }

    double reuse_efficiency() const {
        uint64_t loads = misses + refetches;
        return loads > 0 ?
            static_cast<double>(hits) / (hits + loads) : 0.0;
    }

    double bandwidth_saved_percent() const {
        // How much bandwidth saved vs naive (no caching)
        uint64_t naive_loads = total_accesses;
        uint64_t actual_loads = misses + refetches;
        return naive_loads > 0 ?
            100.0 * (1.0 - static_cast<double>(actual_loads) / naive_loads) : 0.0;
    }

    double average_reuse() const {
        uint64_t loads = misses + refetches;
        return loads > 0 ?
            static_cast<double>(hits) / loads : 0.0;
    }

    double refetch_ratio() const {
        uint64_t loads = misses + refetches;
        return loads > 0 ?
            static_cast<double>(refetches) / loads : 0.0;
    }
};

/// Cache event for visualization
struct L3CacheEvent {
    uint64_t cycle;
    TileKey tile;
    CacheAccessResult result;
    uint32_t occupancy;         // Tiles currently in cache
    uint32_t access_count;      // Access count for this tile
    TileKey evicted_tile;       // Tile evicted (if any)
    bool had_eviction;

    L3CacheEvent() : cycle(0), occupancy(0), access_count(0), had_eviction(false) {}
};

/// Configuration for L3 cache (defined outside class to avoid default param issues)
struct L3CacheConfig {
    uint32_t capacity_tiles = 16;   // Total tiles that fit in L3
    ReplacementPolicy policy = ReplacementPolicy::LRU;

    // Optional: partition capacity by operand type
    bool use_partitioning = false;
    uint32_t a_partition = 4;       // Tiles reserved for A
    uint32_t b_partition = 8;       // Tiles reserved for B
    uint32_t c_partition = 4;       // Tiles reserved for C/D
};

// ============================================================================
// L3 Cache Model
// ============================================================================

/// Models L3 cache behavior with configurable capacity and replacement policy
class L3CacheModel {
public:
    using Config = L3CacheConfig;

    explicit L3CacheModel(const Config& config = Config{})
        : config_(config) {
        reset();
    }

    /// Reset the cache to empty state
    void reset() {
        cache_.clear();
        lru_list_.clear();
        ever_loaded_.clear();
        stats_ = L3CacheStats{};
        events_.clear();
    }

    /// Access a tile in the cache
    /// @return Result of the access (HIT, MISS, or REFETCH)
    CacheAccessResult access(const TileKey& key, uint64_t cycle) {
        stats_.total_accesses++;

        auto op_idx = static_cast<size_t>(key.operand);

        // Check if tile is in cache
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // HIT: update access time and move to front of LRU
            it->second.last_access_cycle = cycle;
            it->second.access_count++;
            update_lru(key);

            stats_.hits++;
            stats_.hits_by_operand[op_idx]++;

            record_event(cycle, key, CacheAccessResult::HIT,
                         it->second.access_count, TileKey{}, false);

            return CacheAccessResult::HIT;
        }

        // Cache miss - need to load
        bool is_refetch = ever_loaded_.count(key) > 0;
        CacheAccessResult result = is_refetch ?
            CacheAccessResult::REFETCH : CacheAccessResult::MISS;

        // Evict if necessary
        TileKey evicted_key{};
        bool had_eviction = false;
        if (cache_.size() >= config_.capacity_tiles) {
            evicted_key = evict_one();
            had_eviction = true;
        }

        // Insert new entry
        CacheEntry entry;
        entry.key = key;
        entry.load_cycle = cycle;
        entry.last_access_cycle = cycle;
        entry.access_count = 1;
        entry.dirty = false;

        cache_[key] = entry;
        lru_list_.push_front(key);
        ever_loaded_.insert(key);

        // Update statistics
        if (is_refetch) {
            stats_.refetches++;
            stats_.refetches_by_operand[op_idx]++;
        } else {
            stats_.misses++;
            stats_.misses_by_operand[op_idx]++;
        }

        record_event(cycle, key, result, 1, evicted_key, had_eviction);

        return result;
    }

    /// Check if a tile is currently in the cache
    bool contains(const TileKey& key) const {
        return cache_.count(key) > 0;
    }

    /// Get current cache occupancy (number of tiles)
    uint32_t occupancy() const {
        return static_cast<uint32_t>(cache_.size());
    }

    /// Get cache capacity
    uint32_t capacity() const {
        return config_.capacity_tiles;
    }

    /// Get statistics
    const L3CacheStats& stats() const { return stats_; }

    /// Get recorded events for visualization
    const std::vector<L3CacheEvent>& events() const { return events_; }

    /// Get list of currently resident tiles
    std::vector<TileKey> resident_tiles() const {
        std::vector<TileKey> tiles;
        tiles.reserve(cache_.size());
        for (const auto& [key, entry] : cache_) {
            tiles.push_back(key);
        }
        return tiles;
    }

    /// Get entry for a tile (if resident)
    const CacheEntry* get_entry(const TileKey& key) const {
        auto it = cache_.find(key);
        return it != cache_.end() ? &it->second : nullptr;
    }

    /// Get configuration
    const Config& config() const { return config_; }

private:
    Config config_;
    std::unordered_map<TileKey, CacheEntry, TileKeyHash> cache_;
    std::list<TileKey> lru_list_;  // Front = most recently used
    std::unordered_set<TileKey, TileKeyHash> ever_loaded_;
    L3CacheStats stats_;
    std::vector<L3CacheEvent> events_;

    void update_lru(const TileKey& key) {
        if (config_.policy != ReplacementPolicy::LRU) return;

        // Remove from current position and add to front
        lru_list_.remove(key);
        lru_list_.push_front(key);
    }

    TileKey evict_one() {
        TileKey victim;

        switch (config_.policy) {
            case ReplacementPolicy::LRU:
                // Evict least recently used (back of list)
                victim = lru_list_.back();
                lru_list_.pop_back();
                break;

            case ReplacementPolicy::FIFO:
                // Evict oldest (back of list, but list not reordered on access)
                victim = lru_list_.back();
                lru_list_.pop_back();
                break;

            case ReplacementPolicy::RANDOM:
                // Pick random entry
                if (!cache_.empty()) {
                    auto it = cache_.begin();
                    std::advance(it, rand() % cache_.size());
                    victim = it->first;
                    lru_list_.remove(victim);
                }
                break;
        }

        auto op_idx = static_cast<size_t>(victim.operand);
        stats_.evictions++;
        stats_.evictions_by_operand[op_idx]++;

        cache_.erase(victim);
        return victim;
    }

    void record_event(uint64_t cycle, const TileKey& tile,
                     CacheAccessResult result, uint32_t access_count,
                     const TileKey& evicted, bool had_eviction) {
        L3CacheEvent event;
        event.cycle = cycle;
        event.tile = tile;
        event.result = result;
        event.occupancy = static_cast<uint32_t>(cache_.size());
        event.access_count = access_count;
        event.evicted_tile = evicted;
        event.had_eviction = had_eviction;
        events_.push_back(event);
    }
};

} // namespace sw::kpu::behavioral
