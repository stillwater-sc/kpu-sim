/**
 * @file tile_cache.hpp
 * @brief Software simulation of L3 tile cache for reuse tracking
 *
 * This implements Phase 1 of the tile caching architecture: software-only
 * tracking of which tiles are resident in L3 cache. This allows the program
 * builder to skip redundant DMA loads and accurately model tile reuse.
 *
 * Key features:
 * - Track tile residency by (matrix, ti, tj, tk) key
 * - LRU eviction when capacity is exceeded
 * - Reference counting for tiles in active use
 * - Statistics collection for hit/miss rates
 *
 * See docs/TILE_CACHING_ARCHITECTURE.md for full design documentation.
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <map>
#include <list>
#include <optional>
#include <string>
#include <sstream>
#include <iomanip>

namespace sw::kpu::isa {

/**
 * @brief Key for identifying a tile in the cache
 */
struct TileKey {
    MatrixID matrix;
    uint16_t ti;
    uint16_t tj;
    uint16_t tk;

    bool operator==(const TileKey& other) const {
        return matrix == other.matrix && ti == other.ti &&
               tj == other.tj && tk == other.tk;
    }

    bool operator<(const TileKey& other) const {
        if (matrix != other.matrix) return matrix < other.matrix;
        if (ti != other.ti) return ti < other.ti;
        if (tj != other.tj) return tj < other.tj;
        return tk < other.tk;
    }

    std::string to_string() const {
        std::ostringstream oss;
        char mat = (matrix == MatrixID::A) ? 'A' :
                   (matrix == MatrixID::B) ? 'B' : 'C';
        oss << mat << "[" << ti << "," << tj << "," << tk << "]";
        return oss.str();
    }
};

/**
 * @brief Entry in the tile cache
 */
struct TileCacheEntry {
    TileKey key;
    Size size_bytes;
    uint8_t refcount;       // Active references (0 = evictable)
    bool locked;            // Cannot be evicted even if refcount=0
    Cycle load_cycle;       // When tile was loaded
    Cycle last_access_cycle;// For LRU
};

/**
 * @brief Statistics for tile cache performance
 */
struct TileCacheStats {
    size_t hits = 0;           // Tile found in cache
    size_t misses = 0;         // Tile not found, DMA required
    size_t evictions = 0;      // Tiles evicted to make room
    size_t writebacks = 0;     // Dirty tiles written back (C tiles)
    Size bytes_loaded = 0;     // Total bytes loaded from external memory
    Size bytes_saved = 0;      // Bytes saved by cache hits

    double hit_rate() const {
        size_t total = hits + misses;
        return total > 0 ? static_cast<double>(hits) / total : 0.0;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "Tile Cache Statistics:\n";
        oss << "  Hits:       " << hits << "\n";
        oss << "  Misses:     " << misses << "\n";
        oss << "  Hit rate:   " << std::fixed << std::setprecision(1)
            << (hit_rate() * 100) << "%\n";
        oss << "  Evictions:  " << evictions << "\n";
        oss << "  Writebacks: " << writebacks << "\n";
        oss << "  Bytes loaded: " << (bytes_loaded / 1024.0) << " KB\n";
        oss << "  Bytes saved:  " << (bytes_saved / 1024.0) << " KB\n";
        return oss.str();
    }
};

/**
 * @brief Software simulation of L3 tile cache
 *
 * Models tile residency in L3 with LRU eviction and reference counting.
 * Used by the program builder to determine when DMA loads can be skipped.
 */
class TileCache {
public:
    /**
     * @brief Configuration for the tile cache
     */
    struct Config {
        Size total_capacity_bytes;  // Total L3 capacity
        Size num_l3_tiles;          // Number of physical L3 tiles
        Size tile_capacity_bytes;   // Per-tile capacity

        Config() : total_capacity_bytes(512 * 1024),
                   num_l3_tiles(4),
                   tile_capacity_bytes(128 * 1024) {}
    };

    explicit TileCache(const Config& config = Config());

    /**
     * @brief Reset cache to empty state
     */
    void reset();

    /**
     * @brief Check if a tile is resident in cache
     * @param key Tile identifier
     * @return true if tile is in cache and valid
     */
    bool is_resident(const TileKey& key) const;

    /**
     * @brief Lookup a tile, updating access time if found
     * @param key Tile identifier
     * @param current_cycle Current simulation cycle (for LRU)
     * @return Cache entry if found, nullopt if miss
     */
    std::optional<TileCacheEntry> lookup(const TileKey& key, Cycle current_cycle);

    /**
     * @brief Allocate space for a tile (may trigger eviction)
     * @param key Tile identifier
     * @param size_bytes Size of the tile data
     * @param current_cycle Current simulation cycle
     * @param lock If true, tile cannot be evicted until released
     * @return true if allocation succeeded, false if no space available
     */
    bool allocate(const TileKey& key, Size size_bytes, Cycle current_cycle,
                  bool lock = false);

    /**
     * @brief Acquire a reference to a tile (increment refcount)
     * @param key Tile identifier
     * @return true if tile exists and was acquired
     */
    bool acquire(const TileKey& key);

    /**
     * @brief Release a reference to a tile (decrement refcount)
     * @param key Tile identifier
     * @return true if tile exists and was released
     */
    bool release(const TileKey& key);

    /**
     * @brief Unlock a tile (allow eviction when refcount=0)
     * @param key Tile identifier
     */
    void unlock(const TileKey& key);

    /**
     * @brief Invalidate a tile (remove from cache)
     * @param key Tile identifier
     * @return true if tile was in cache and removed
     */
    bool invalidate(const TileKey& key);

    /**
     * @brief Get current cache statistics
     */
    const TileCacheStats& stats() const { return stats_; }

    /**
     * @brief Get current cache utilization
     * @return Fraction of capacity used (0.0 to 1.0)
     */
    double utilization() const {
        return static_cast<double>(used_bytes_) / config_.total_capacity_bytes;
    }

    /**
     * @brief Get number of tiles currently in cache
     */
    size_t size() const { return entries_.size(); }

    /**
     * @brief Check if cache has room for a tile of given size
     */
    bool can_allocate(Size size_bytes) const;

    /**
     * @brief Generate summary string
     */
    std::string summary() const;

private:
    Config config_;
    std::map<TileKey, TileCacheEntry> entries_;
    std::list<TileKey> lru_order_;  // Front = most recent, back = least recent
    Size used_bytes_ = 0;
    TileCacheStats stats_;

    /**
     * @brief Evict tiles until size_bytes are available
     * @return true if space was freed, false if cannot evict enough
     */
    bool evict_for_space(Size size_bytes);

    /**
     * @brief Select a tile for eviction (LRU with refcount=0)
     * @return Key of tile to evict, or nullopt if none available
     */
    std::optional<TileKey> select_victim() const;

    /**
     * @brief Move tile to front of LRU list
     */
    void touch_lru(const TileKey& key);

    /**
     * @brief Remove tile from LRU list
     */
    void remove_from_lru(const TileKey& key);
};

/**
 * @brief Tile cache aware program builder helper
 *
 * Wraps a TileCache to provide convenient methods for the program builder.
 */
class TileCacheTracker {
public:
    explicit TileCacheTracker(const TileCache::Config& config = TileCache::Config());

    /**
     * @brief Check if tile needs to be loaded (not in cache)
     * @param matrix Matrix identifier
     * @param tile Tile coordinates
     * @param size_bytes Tile size
     * @param current_cycle Current cycle
     * @return true if DMA load is required
     */
    bool needs_load(MatrixID matrix, TileCoord tile, Size size_bytes,
                    Cycle current_cycle);

    /**
     * @brief Mark tile as loaded (after DMA completes)
     */
    void mark_loaded(MatrixID matrix, TileCoord tile, Size size_bytes,
                     Cycle current_cycle, bool lock = false);

    /**
     * @brief Release tile (no longer needed for current computation)
     */
    void release_tile(MatrixID matrix, TileCoord tile);

    /**
     * @brief Get underlying cache for statistics
     */
    const TileCache& cache() const { return cache_; }
    TileCache& cache() { return cache_; }

    /**
     * @brief Reset tracker state
     */
    void reset() { cache_.reset(); }

private:
    TileCache cache_;

    TileKey make_key(MatrixID matrix, TileCoord tile) const {
        return TileKey{matrix, tile.ti, tile.tj, tile.tk};
    }
};

} // namespace sw::kpu::isa
