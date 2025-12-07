/**
 * @file tile_cache.cpp
 * @brief Implementation of software tile cache simulation
 */

#include <sw/kpu/isa/tile_cache.hpp>
#include <algorithm>
#include <iomanip>

namespace sw::kpu::isa {

// ============================================================================
// TileCache Implementation
// ============================================================================

TileCache::TileCache(const Config& config)
    : config_(config), used_bytes_(0)
{
}

void TileCache::reset() {
    entries_.clear();
    lru_order_.clear();
    used_bytes_ = 0;
    stats_ = TileCacheStats{};
}

bool TileCache::is_resident(const TileKey& key) const {
    return entries_.find(key) != entries_.end();
}

std::optional<TileCacheEntry> TileCache::lookup(const TileKey& key, Cycle current_cycle) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        stats_.misses++;
        return std::nullopt;
    }

    // Update access time and LRU
    it->second.last_access_cycle = current_cycle;
    touch_lru(key);

    stats_.hits++;
    stats_.bytes_saved += it->second.size_bytes;

    return it->second;
}

bool TileCache::allocate(const TileKey& key, Size size_bytes, Cycle current_cycle,
                         bool lock) {
    // Check if already in cache
    if (is_resident(key)) {
        // Just update metadata
        auto& entry = entries_[key];
        entry.last_access_cycle = current_cycle;
        entry.refcount++;
        if (lock) entry.locked = true;
        touch_lru(key);
        return true;
    }

    // Need to allocate new space
    if (!can_allocate(size_bytes)) {
        // Try to evict
        if (!evict_for_space(size_bytes)) {
            return false;  // Cannot make room
        }
    }

    // Create new entry
    TileCacheEntry entry;
    entry.key = key;
    entry.size_bytes = size_bytes;
    entry.refcount = 1;
    entry.locked = lock;
    entry.load_cycle = current_cycle;
    entry.last_access_cycle = current_cycle;

    entries_[key] = entry;
    lru_order_.push_front(key);
    used_bytes_ += size_bytes;

    stats_.bytes_loaded += size_bytes;

    return true;
}

bool TileCache::acquire(const TileKey& key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }

    it->second.refcount++;
    return true;
}

bool TileCache::release(const TileKey& key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }

    if (it->second.refcount > 0) {
        it->second.refcount--;
    }

    // Clear lock when refcount reaches 0
    if (it->second.refcount == 0) {
        it->second.locked = false;
    }

    return true;
}

void TileCache::unlock(const TileKey& key) {
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        it->second.locked = false;
    }
}

bool TileCache::invalidate(const TileKey& key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }

    // Track writeback for C tiles
    if (key.matrix == MatrixID::C) {
        stats_.writebacks++;
    }

    used_bytes_ -= it->second.size_bytes;
    remove_from_lru(key);
    entries_.erase(it);

    return true;
}

bool TileCache::can_allocate(Size size_bytes) const {
    return (used_bytes_ + size_bytes) <= config_.total_capacity_bytes;
}

bool TileCache::evict_for_space(Size size_bytes) {
    while (!can_allocate(size_bytes)) {
        auto victim = select_victim();
        if (!victim) {
            return false;  // No evictable tiles
        }

        stats_.evictions++;

        // Track writeback for C tiles
        auto it = entries_.find(*victim);
        if (it != entries_.end() && victim->matrix == MatrixID::C) {
            stats_.writebacks++;
        }

        used_bytes_ -= entries_[*victim].size_bytes;
        remove_from_lru(*victim);
        entries_.erase(*victim);
    }

    return true;
}

std::optional<TileKey> TileCache::select_victim() const {
    // Walk LRU from back (least recently used) to front
    for (auto it = lru_order_.rbegin(); it != lru_order_.rend(); ++it) {
        auto entry_it = entries_.find(*it);
        if (entry_it == entries_.end()) continue;

        const auto& entry = entry_it->second;

        // Skip if in use or locked
        if (entry.refcount > 0) continue;
        if (entry.locked) continue;

        return *it;
    }

    return std::nullopt;  // No evictable tiles
}

void TileCache::touch_lru(const TileKey& key) {
    // Remove from current position
    lru_order_.remove(key);
    // Add to front (most recently used)
    lru_order_.push_front(key);
}

void TileCache::remove_from_lru(const TileKey& key) {
    lru_order_.remove(key);
}

std::string TileCache::summary() const {
    std::ostringstream oss;

    oss << "\nL3 Tile Cache Status:\n";
    oss << "  Capacity:    " << (config_.total_capacity_bytes / 1024) << " KB\n";
    oss << "  Used:        " << (used_bytes_ / 1024) << " KB ("
        << std::fixed << std::setprecision(1) << (utilization() * 100) << "%)\n";
    oss << "  Tiles:       " << entries_.size() << "\n";

    oss << "\n" << stats_.to_string();

    if (!entries_.empty()) {
        oss << "\nResident tiles:\n";
        for (const auto& [key, entry] : entries_) {
            oss << "  " << key.to_string()
                << " size=" << entry.size_bytes
                << " refcount=" << (int)entry.refcount
                << (entry.locked ? " LOCKED" : "")
                << "\n";
        }
    }

    return oss.str();
}

// ============================================================================
// TileCacheTracker Implementation
// ============================================================================

TileCacheTracker::TileCacheTracker(const TileCache::Config& config)
    : cache_(config)
{
}

bool TileCacheTracker::needs_load(MatrixID matrix, TileCoord tile, Size size_bytes,
                                   Cycle current_cycle) {
    TileKey key = make_key(matrix, tile);

    // Check if in cache
    auto result = cache_.lookup(key, current_cycle);
    if (result) {
        // Cache hit - acquire reference, no load needed
        cache_.acquire(key);
        return false;
    }

    // Cache miss - will need to load
    return true;
}

void TileCacheTracker::mark_loaded(MatrixID matrix, TileCoord tile, Size size_bytes,
                                    Cycle current_cycle, bool lock) {
    TileKey key = make_key(matrix, tile);
    cache_.allocate(key, size_bytes, current_cycle, lock);
}

void TileCacheTracker::release_tile(MatrixID matrix, TileCoord tile) {
    TileKey key = make_key(matrix, tile);
    cache_.release(key);
}

} // namespace sw::kpu::isa
