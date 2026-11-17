// ============================================================================
// include/sw/kpu/timing/l2_banks.hpp
// L-CA's banked L2 (#401 slice 1; docs/plans/banked-l2-at-lca.md): a compute tile's L2 as banks
// of rows, row i in bank i mod N (#383, esg#14: N a power of two, the bank the row's low bits),
// each bank serving `read_ports` reads and `write_ports` writes per clock (esg#24's idealized
// scratchpad: 2 and 1), reads `read_latency` clocks late.
//
// A transfer is a sequence of row accesses, one row per `cycles_per_row` clocks at its client's
// rate. It books them all when it starts, in a per-compute-tile port calendar: row k at the first
// clock at or after its turn where its bank has a port of its kind free; a clock it waits is a
// stall, counted by client and by bank. It finishes when its last row is served (plus the read
// latency for a read). A merge is a read-modify-write: each row takes a read and a write port of
// its bank in one clock. Clients starting in one clock book in the executor's tick order.
//
// Credits are not banks: an L2 slot is still a tile under its credit pool. Banks only time the
// transfers, and move no data of their own.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/timing/process_interface.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::timing {

class L2Banks {
public:
    struct Geometry {
        std::uint32_t banks = 1;
        std::uint32_t rows_per_bank = 0;    // 0: as many as the slots need
        std::uint32_t row_bytes = 256;
        Cycle read_latency = 4;
        std::uint32_t read_ports = 2, write_ports = 1;
    };
    enum class Access : std::uint8_t { Read, Write, ReadModifyWrite };
    enum class Client : std::uint8_t { BlockMover, Streamer, Drain, Combiner };
    static constexpr std::size_t kClients = 4;
    static const char* name(Client c) {
        switch (c) {
            case Client::BlockMover: return "BlockMover";
            case Client::Streamer: return "Streamer";
            case Client::Drain: return "drain";
            case Client::Combiner: return "Combiner";
        }
        return "?";
    }

    struct Stats {
        std::uint64_t accesses = 0;                         // row accesses booked
        std::array<std::uint64_t, kClients> stalls{};       // clocks a row waited for its bank, by client
        std::vector<std::uint64_t> bank_stalls;             // ... by bank
        std::uint64_t stalls_total() const {
            std::uint64_t n = 0;
            for (auto s : stalls) n += s;
            return n;
        }
    };

    static bool power_of_two(std::uint64_t v) { return v && !(v & (v - 1)); }

    L2Banks(Geometry g, std::size_t compute_tiles) : g_(g), cal_(std::max<std::size_t>(1, compute_tiles)) {
        if (!power_of_two(g_.banks))
            throw std::invalid_argument("L2Banks: " + std::to_string(g_.banks) +
                                        " banks; the RTL requires a power of two (esg#14)");
        if (g_.row_bytes == 0) throw std::invalid_argument("L2Banks: a row of 0 bytes");
        if (g_.read_ports == 0 || g_.write_ports == 0) throw std::invalid_argument("L2Banks: a bank with no port");
        stats_.bank_stalls.assign(g_.banks, 0);
    }

    const Geometry& geometry() const { return g_; }
    const Stats& stats() const { return stats_; }

    // The rows a tile of `bytes` fills, and the bank of row `r` (its low bits).
    std::uint64_t rows_of(std::uint64_t bytes) const { return std::max<std::uint64_t>(1, (bytes + g_.row_bytes - 1) / g_.row_bytes); }
    std::uint32_t bank_of(std::uint64_t row) const { return static_cast<std::uint32_t>(row & (g_.banks - 1)); }
    // An L2 slot's first row: slots of `rows` rows laid end to end, wrapping at the L2's rows.
    std::uint64_t first_row(std::uint32_t slot, std::uint64_t rows) const {
        const std::uint64_t r = std::uint64_t(slot) * rows;
        return g_.rows_per_bank ? r % (std::uint64_t(g_.rows_per_bank) * g_.banks) : r;
    }
    // The clocks one row takes at `bytes_per_cycle`.
    Cycle cycles_per_row(double bytes_per_cycle) const {
        return std::max<Cycle>(1, static_cast<Cycle>(std::ceil(g_.row_bytes / bytes_per_cycle)));
    }

    // Book a transfer of `bytes` at L2 slot `slot` of compute tile `cf`, starting at `start`, one
    // row per `per_row` clocks. Returns the clocks from `start` to its last row served (with the
    // read latency for a read or a merge).
    Cycle book(std::size_t cf, std::uint32_t slot, std::uint64_t bytes, Access kind, Client who, Cycle start,
               Cycle per_row) {
        auto& cal = cal_.at(cf < cal_.size() ? cf : 0);
        // Clocks before this one can no longer be booked: forget them.
        if (start > kKeep) cal.erase(cal.begin(), cal.lower_bound(start - kKeep));
        const std::uint64_t n = rows_of(bytes), r0 = first_row(slot, n);
        const bool reads = kind != Access::Write, writes = kind != Access::Read;
        Cycle at = start, last = start;
        for (std::uint64_t k = 0; k < n; ++k) {
            const std::uint32_t b = bank_of(r0 + k);
            Cycle t = std::max(at, start + k * per_row);
            for (;; ++t) {
                auto& use = cal[t];
                if (use.size() < g_.banks) use.assign(g_.banks, Use{});
                Use& u = use[b];
                if ((!reads || u.reads < g_.read_ports) && (!writes || u.writes < g_.write_ports)) {
                    u.reads += reads;
                    u.writes += writes;
                    break;
                }
                ++stats_.stalls[static_cast<std::size_t>(who)];
                ++stats_.bank_stalls[b];
            }
            ++stats_.accesses;
            last = t;
            at = t + per_row;
        }
        return (last + per_row - start) + (reads ? g_.read_latency : 0);
    }

private:
    struct Use { std::uint8_t reads = 0, writes = 0; };
    static constexpr Cycle kKeep = 1u << 16;           // clocks of history a late booking may still see
    Geometry g_;
    std::vector<std::map<Cycle, std::vector<Use>>> cal_;
    Stats stats_;
};

}  // namespace sw::kpu::timing
