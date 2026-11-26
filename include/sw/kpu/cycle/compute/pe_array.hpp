#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/pe_array.hpp
// The PE array on the cycle core (#544; docs/plans/cycle-based-kpu-prototype.md §5.1, step 2c): R x C
// registered MAC processing elements, output-stationary, as the RTL builds them
// (embodied-ai-soc-generator b57c881, tiles/systolic/core/pe.py and systolic_array.py).
//
// EVERY PE HAS AN INPUT REGISTER ON EACH OF ITS LINKS, so a value moves one PE a clock:
//   - west -> east: A, with its markers `valid`, `first` (the reduction's first product) and `last`;
//   - north -> south: B;
//   - south -> north: results, with `valid`.
// A PE computes on its registers. While A is valid it accumulates c += a * b, and `first` overwrites
// (c = a * b), so a reduction needs no clear. On `last` it LAUNCHES c + a * b north, driving its
// north output with it in place of the result passing from the south; a launch while a result is
// passing is a NORTH COLLISION, an L1Fault (the RTL's out_collision; the Streamer's drain rule
// prevents it, step 2d). The array has no skew registers: its callers, the L1 lanes, stagger their
// streams (l1_lanes.hpp).
//
// THE EDGES. Each clock the array takes one element from each row's west link and each column's
// north link (an element presented at clock t is in the edge PE's register at t + 1), and drives
// each column's result, if any, onto its north output link.
//
// Nothing here is a formula: the RTL's clocks (PE(i,j) multiplies element k at s + 1 + i + j + k,
// and column j carries row i's result at s + K + j + 2i) emerge from the registers, and the tests
// measure them from the array's log of what each PE did (macs(), launches()).
//
// A DRIVES THE PE, as in the RTL: it computes while A is valid, whatever B holds. A tile smaller than
// the array runs on its corner, so A flows east past the tile's last column into PEs whose B lane is
// idle: they multiply by 0 and launch a PADDED result, which leaves the north edge like any other
// (the RTL's result buffers drop the columns past the tile, step 2d) and collides like any other.
// So A and B must meet per reduction: every product of a reduction has its B, of the same command
// and element, or none has (a padded column). A skew off by a clock does neither, and is a defect of
// the caller's, raised by name.
//
// COUNTERS. The array is busy in a clock any PE holds an operand or a result; an item is a result,
// arriving when it launches and completing when it leaves the north edge.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/compute/l1_lanes.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/trace.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

// One result leaving the array's north edge: PE(row, col)'s reduction for `command`.
struct ArrayResult {
    std::uint64_t id = 0;
    std::uint32_t row = 0, col = 0;
    std::uint32_t command = 0;
    std::int32_t value = 0;
    bool padded = false;   // a column past the tile's: no B in any product
};

class PEArray : public Component {
public:
    struct Config {
        std::uint32_t rows = 4;            // R
        std::uint32_t cols = 4;            // C
        std::uint32_t element_bytes = 1;   // A's and B's, signed (int8: 1)
        bool log = true;                   // keep macs() and launches()

        // From a device's declared fabric: pe_rows x pe_cols.
        static Config from(const sw::kpu::program::platform::DeviceSpecification& d) {
            if (!d.fabric.pe_rows || !d.fabric.pe_cols)
                throw std::invalid_argument("cycle::PEArray: device " + d.name + " declares no fabric.pe_rows/pe_cols");
            Config c;
            c.rows = static_cast<std::uint32_t>(*d.fabric.pe_rows);
            c.cols = static_cast<std::uint32_t>(*d.fabric.pe_cols);
            return c;
        }
    };
    // What a PE did, for the tests and the trace.
    struct Mac {
        Cycle at;
        std::uint32_t row, col, command, k;
    };
    struct Launch {
        Cycle at;          // the clock the PE launched
        Cycle left;        // the clock column `col` drove it out of the north edge
        std::uint32_t row, col, command;
        std::int32_t value;
        bool padded;
    };

    PEArray(std::string name, Domain& d, Config cfg) : Component(std::move(name), d), config_(cfg) {
        if (cfg.rows == 0 || cfg.cols == 0)
            throw std::invalid_argument("cycle::PEArray " + this->name() + ": the rows and columns must be positive");
        if (cfg.element_bytes != 1 && cfg.element_bytes != 2 && cfg.element_bytes != 4)
            throw std::invalid_argument("cycle::PEArray " + this->name() + ": an element is 1, 2 or 4 bytes");
        clear();
    }

    // west: one link per row (A); north_in: one per column (B); north_out: one per column (results).
    void connect(std::vector<Link<L1Element>*> west, std::vector<Link<L1Element>*> north_in,
                 std::vector<Link<ArrayResult>*> north_out) {
        if (west.size() != config_.rows || north_in.size() != config_.cols || north_out.size() != config_.cols)
            throw std::invalid_argument("cycle::PEArray " + name() + ": " + std::to_string(west.size()) + " west, " +
                                        std::to_string(north_in.size()) + " north and " +
                                        std::to_string(north_out.size()) + " result links, for " +
                                        std::to_string(config_.rows) + " x " + std::to_string(config_.cols) + " PEs");
        west_ = std::move(west);
        north_in_ = std::move(north_in);
        north_out_ = std::move(north_out);
    }

    // Perfetto lanes, one per PE, with a span for each multiply (opt-in: R x C lanes).
    void trace_lanes(Tracer& t) {
        tracer_ = &t;
        lanes_.clear();
        for (std::uint32_t i = 0; i < config_.rows; ++i)
            for (std::uint32_t j = 0; j < config_.cols; ++j)
                lanes_.push_back(t.add_lane(name() + " PE(" + std::to_string(i) + "," + std::to_string(j) + ")",
                                            domain().name()));
    }

    void evaluate(Cycle now) override {
        if (west_.empty()) throw std::logic_error(where(now) + ": not connected");
        const std::uint32_t R = config_.rows, C = config_.cols;
        // The edges' inputs this clock.
        std::vector<Operand> west(R), north(C);
        for (std::uint32_t i = 0; i < R; ++i)
            if (west_[i]->can_pop(now)) west[i] = operand(west_[i]->pop(now));
        for (std::uint32_t j = 0; j < C; ++j)
            if (north_in_[j]->can_pop(now)) north[j] = operand(north_in_[j]->pop(now));

        // Each PE on its registers: its product, its accumulator, and what it drives north.
        std::vector<Result> up(std::size_t(R) * C);   // each PE's north output this clock
        bool busy = false;
        for (std::uint32_t i = 0; i < R; ++i)
            for (std::uint32_t j = 0; j < C; ++j) {
                PE& p = pe(i, j);
                busy = busy || p.a.valid || p.b.valid || p.c_up.valid;
                Result out = p.c_up;
                if (p.a.valid) {
                    // The operands meet: every product of the reduction with its B, or none.
                    const bool padded = p.a.first ? !p.b.valid : p.padded;
                    if (padded == p.b.valid || (p.b.valid && (p.a.k != p.b.k || p.a.command != p.b.command)))
                        throw std::logic_error(where(now) + ": the operands do not meet at PE(" + std::to_string(i) +
                                               "," + std::to_string(j) + "): " + describe(p.a, "A") + ", " +
                                               describe(p.b, "B") + (padded == p.b.valid ? ", in a reduction "
                                                   + std::string(padded ? "with no B before" : "with its B before")
                                                   : std::string()));
                    p.padded = padded;
                    const std::int64_t product = p.b.valid ? std::int64_t(p.a.v) * p.b.v : 0;
                    const std::int32_t total = wrap(p.a.first ? product : std::int64_t(p.c) + product);
                    if (config_.log) macs_.push_back({now, i, j, p.a.command, p.a.k});
                    ++mac_count_[std::size_t(i) * C + j];
                    if (tracer_)
                        tracer_->span(lanes_[std::size_t(i) * C + j], "k=" + std::to_string(p.a.k), domain().edge(now),
                                      domain().edge(now + 1));
                    if (p.a.last) {
                        if (p.c_up.valid)
                            throw L1Fault(where(now) + ": a north collision at PE(" + std::to_string(i) + "," +
                                          std::to_string(j) + "): its result for command " +
                                          std::to_string(p.a.command) + " launched while PE(" +
                                          std::to_string(p.c_up.row) + "," + std::to_string(j) + ")'s for command " +
                                          std::to_string(p.c_up.command) + " passed it");
                        out = Result{true, total, i, p.a.command, padded, launches_.size()};
                        launches_.push_back({now, 0, i, j, p.a.command, total, padded});
                        counters().accept();
                    } else {
                        p.c = total;
                    }
                }
                up[std::size_t(i) * C + j] = out;
            }

        // The north edge: row 0's outputs leave.
        for (std::uint32_t j = 0; j < C; ++j) {
            const Result& r = up[j];
            if (!r.valid) continue;
            if (!north_out_[j]->can_push(now))
                throw L1Fault(where(now) + ": column " + std::to_string(j) + "'s result buffer did not take PE(" +
                              std::to_string(r.row) + "," + std::to_string(j) + ")'s result (the array cannot wait)");
            north_out_[j]->push(now, ArrayResult{++id_, r.row, j, r.command, r.value, r.padded});
            launches_[r.launch].left = now;
            counters().complete();
        }

        // Every register latches its input: A from the west, B from the north, results from the south.
        std::vector<PE> next = pes_;
        for (std::uint32_t i = 0; i < R; ++i)
            for (std::uint32_t j = 0; j < C; ++j) {
                PE& n = next[std::size_t(i) * C + j];
                n.a = j == 0 ? west[i] : pe(i, j - 1).a;
                n.b = i == 0 ? north[j] : pe(i - 1, j).b;
                n.c_up = i + 1 == R ? Result{} : up[std::size_t(i + 1) * C + j];
            }
        pes_ = std::move(next);
        if (busy) counters().busy();
    }

    [[nodiscard]] bool idle() const override {
        for (const PE& p : pes_)
            if (p.a.valid || p.b.valid || p.c_up.valid) return false;
        return true;
    }

    void reset() override { clear(); }

    [[nodiscard]] std::string describe() const override {
        std::size_t a = 0, results = 0;
        for (const PE& p : pes_) {
            a += p.a.valid;
            results += p.c_up.valid;
        }
        return std::to_string(a) + " PEs with operands, " + std::to_string(results) + " results moving north, " +
               std::to_string(launches_.size()) + " launched";
    }

    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] const std::vector<Mac>& macs() const { return macs_; }
    [[nodiscard]] const std::vector<Launch>& launches() const { return launches_; }
    // PE(i, j)'s multiplies over the run: its utilization's numerator.
    [[nodiscard]] std::uint64_t macs_at(std::uint32_t i, std::uint32_t j) const {
        return mac_count_.at(std::size_t(i) * config_.cols + j);
    }

private:
    struct Operand {
        bool valid = false;
        std::int32_t v = 0;
        std::uint32_t k = 0, command = 0;
        bool first = false, last = false;
    };
    struct Result {
        bool valid = false;
        std::int32_t value = 0;
        std::uint32_t row = 0, command = 0;
        bool padded = false;
        std::size_t launch = 0;   // its entry in launches_
    };
    struct PE {
        Operand a, b;
        Result c_up;
        std::int32_t c = 0;   // the accumulator
        bool padded = false;  // its reduction's products have no B (a column past the tile)
    };

    [[nodiscard]] std::string where(Cycle now) const { return "cycle::PEArray " + name() + " at cycle " + std::to_string(now); }
    PE& pe(std::uint32_t i, std::uint32_t j) { return pes_[std::size_t(i) * config_.cols + j]; }
    static std::int32_t wrap(std::int64_t v) { return static_cast<std::int32_t>(static_cast<std::uint32_t>(v)); }
    [[nodiscard]] Operand operand(const L1Element& e) const {
        const std::uint32_t bits = 8 * config_.element_bytes;
        std::int64_t v = e.bits;
        if (bits < 32 && (e.bits >> (bits - 1)) & 1u) v -= std::int64_t(1) << bits;   // sign-extend
        return {true, static_cast<std::int32_t>(v), e.k, e.command, e.first, e.last};
    }
    static std::string describe(const Operand& o, const char* side) {
        if (!o.valid) return std::string(side) + " none";
        return std::string(side) + " of command " + std::to_string(o.command) + " element " + std::to_string(o.k);
    }

    void clear() {
        pes_.assign(std::size_t(config_.rows) * config_.cols, PE{});
        mac_count_.assign(pes_.size(), 0);
        macs_.clear();
        launches_.clear();
        id_ = 0;
    }

    Config config_;
    std::vector<PE> pes_;   // row-major
    std::vector<std::uint64_t> mac_count_;
    std::vector<Mac> macs_;
    std::vector<Launch> launches_;
    std::uint64_t id_ = 0;
    std::vector<Link<L1Element>*> west_, north_in_;
    std::vector<Link<ArrayResult>*> north_out_;
    Tracer* tracer_ = nullptr;
    std::vector<std::uint32_t> lanes_;
};

}  // namespace sw::kpu::cycle
