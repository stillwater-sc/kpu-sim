// ============================================================================
// include/sw/kpu/timing/tile_tracker.hpp
// Human-readable tile-state tracker: horizontal L3 | L2 | L1/array occupancy
// bands appended as a progression log (issue #165)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/tile_descriptor.hpp>

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief Renders the tiles staged in the software-managed memory hierarchy
 *        as a horizontal, per-snapshot text band, appended to a log
 *
 * We conceptualize the data-movement schedule as transfers of tiles of state
 * staging in the memory hierarchy. This tracker makes that legible: at each
 * snapshot it prints one band with L3 on the left, L2 to its right, and
 * L1/array on the far right - the direction data flows toward compute - and
 * successive bands read top-to-bottom as the state-transition progression.
 * Each tile cell shows its TileID and, where the value plane is active, a
 * compact content summary (e.g. the (m, l) pair of an online-softmax stats
 * tile), so the reader sees content, not just location.
 *
 * A pure observer over ConcurrentTimingExecutor::tiles_at (#165) - it never
 * mutates executor state. `observe()` dedupes: it appends a band only when
 * occupancy changed since the last observation, so the log tracks
 * transitions rather than idle cycles. Output is deterministic (tiles_at is
 * sorted), hence diffable in tests.
 *
 * These are BUFFERS, not caches: the log reads as "tile arrived / resident /
 * credit returned", never hit/miss/evict.
 */
class TileTracker {
public:
    struct Config {
        std::size_t col_width = 34;   ///< width of each level column
        std::size_t max_values = 3;   ///< values shown per tile cell (else elided)
        bool show_compute_in_l1 = true;  ///< fold COMPUTE (array) into the L1 column
    };

    TileTracker() = default;
    explicit TileTracker(Config config) : config_(config) {}

    /// Column header + rule. Emitted automatically before the first band.
    [[nodiscard]] std::string header() const {
        std::ostringstream ss;
        ss << pad("cyc", 6) << " | " << pad("L3 buffers", config_.col_width)
           << " | " << pad("L2 banks", config_.col_width)
           << " | " << pad("L1 / array", config_.col_width) << "\n";
        ss << rule();
        return ss.str();
    }

    /**
     * @brief Append a band iff occupancy changed since the last observe
     * @return true if a band was appended
     */
    bool observe(const ConcurrentTimingExecutor& exec) {
        std::string sig = signature(exec);
        if (sig == last_signature_) return false;
        last_signature_ = sig;
        snapshot(exec);
        return true;
    }

    /// Append a band for the current state unconditionally.
    void snapshot(const ConcurrentTimingExecutor& exec) {
        if (!header_emitted_) { log_ += header(); header_emitted_ = true; }
        std::ostringstream ss;
        ss << pad(std::to_string(exec.current_cycle()), 6) << " | "
           << pad(render_level(exec, MemoryLevel::L3), config_.col_width) << " | "
           << pad(render_level(exec, MemoryLevel::L2), config_.col_width) << " | "
           << render_l1_array(exec) << "\n";
        log_ += ss.str();
    }

    [[nodiscard]] const std::string& log() const { return log_; }
    void clear() { log_.clear(); last_signature_.clear(); header_emitted_ = false; }

private:
    Config config_;
    std::string log_;
    std::string last_signature_;
    bool header_emitted_ = false;

    static std::string pad(const std::string& s, std::size_t w) {
        if (s.size() >= w) return s.substr(0, w);
        return s + std::string(w - s.size(), ' ');
    }
    std::string rule() const {
        auto dashes = [](std::size_t n) { return std::string(n, '-'); };
        return dashes(6) + "-+-" + dashes(config_.col_width) + "-+-" +
               dashes(config_.col_width) + "-+-" + dashes(config_.col_width) + "\n";
    }

    /// One cell: "A[0,2]" plus "=(v0,v1,...)" when a small payload is present.
    std::string cell(const ConcurrentTimingExecutor& exec, MemoryLevel level,
                     const TileID& id) const {
        std::string c = id.to_string();
        if (exec.has_tile_payload_at(level, id)) {
            const auto& p = exec.tile_payload_at(level, id);
            if (!p.values.empty() && p.values.size() <= config_.max_values) {
                std::ostringstream v;
                v << "=(";
                for (std::size_t i = 0; i < p.values.size(); ++i) {
                    if (i) v << ",";
                    v << format_value(p.values[i]);
                }
                v << ")";
                c += v.str();
            }
        }
        return c;
    }

    std::string render_level(const ConcurrentTimingExecutor& exec,
                             MemoryLevel level) const {
        const auto ids = exec.tiles_at(level);
        if (ids.empty()) return "-";
        std::string out;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i) out += " ";
            out += cell(exec, level, ids[i]);
        }
        return out;
    }

    /// The rightmost column: L1 tiles plus COMPUTE (array) tiles marked '*',
    /// rendered in a single TileID-sorted sequence. A tile resident at both
    /// L1 and COMPUTE is shown once, as in-array (its furthest state).
    std::string render_l1_array(const ConcurrentTimingExecutor& exec) const {
        const auto l1 = exec.tiles_at(MemoryLevel::L1);
        const auto compute = config_.show_compute_in_l1
            ? exec.tiles_at(MemoryLevel::COMPUTE) : std::vector<TileID>{};
        auto in_compute = [&](const TileID& id) {
            for (const auto& c : compute) if (c == id) return true;
            return false;
        };
        // Merge into one sorted list: L1-only tiles + all compute tiles.
        struct Entry { TileID id; bool in_array; };
        std::vector<Entry> merged;
        for (const auto& id : l1) if (!in_compute(id)) merged.push_back({id, false});
        for (const auto& id : compute) merged.push_back({id, true});
        std::sort(merged.begin(), merged.end(),
                  [](const Entry& a, const Entry& b) { return a.id < b.id; });

        std::string out;
        for (const auto& e : merged) {
            if (!out.empty()) out += " ";
            out += cell(exec, e.in_array ? MemoryLevel::COMPUTE : MemoryLevel::L1, e.id);
            if (e.in_array) out += "*";
        }
        return out.empty() ? "-" : out;
    }

    static std::string format_value(float v) {
        std::ostringstream ss;
        ss.precision(3);
        ss << v;
        return ss.str();
    }

    /// Occupancy fingerprint for dedupe (levels x tile ids x arrival cycle).
    std::string signature(const ConcurrentTimingExecutor& exec) const {
        std::ostringstream ss;
        for (MemoryLevel level : {MemoryLevel::L3, MemoryLevel::L2,
                                  MemoryLevel::L1, MemoryLevel::COMPUTE}) {
            ss << static_cast<int>(level) << ":";
            for (const auto& id : exec.tiles_at(level)) ss << id.to_string() << ",";
            ss << ";";
        }
        return ss.str();
    }
};

} // namespace sw::kpu::timing
