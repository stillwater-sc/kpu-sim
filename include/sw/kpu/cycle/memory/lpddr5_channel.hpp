#pragma once
// ============================================================================
// include/sw/kpu/cycle/memory/lpddr5_channel.hpp
// One LPDDR5 channel on the cycle core (#537; docs/plans/cycle-based-kpu-prototype.md §3.4, §4.1).
//
// PORTED, NOT REWRITTEN. The channel hosts the repo's bank-level controller,
// LPDDR5MemoryController (models/temporal/memory/controllers): bank groups, tRRD/tCCD/tFAW,
// per-bank refresh, the data bus, FR-FCFS, and its own invariant checker. The component adds only
// what the cycle core needs around it:
//
//   clock      the component lives in the DRAM domain (LPDDR5-6400: the controller's 3.2 GHz,
//              312.5 ps) and ticks the controller exactly once per cycle of it, so the
//              controller's cycle is the domain's cycle
//   links      requests arrive on a link (`in`), responses leave on one (`out`), both with
//              credits. A request is a read or a write of `size` bytes at `address`; its response
//              carries the read's bytes, or acknowledges the write, with the request's id and tag
//   values     the controller times requests and holds no memory, so the channel keeps the
//              bytes: a sparse store, 64-byte lines, zero where nothing was written. A write is
//              applied, and a read's bytes taken, when the channel accepts the request, so values
//              follow the order requests arrive in, whatever order the controller serves them in.
//              The response leaves when the controller completes the request
//   bounds     at most `queue depth` requests are outstanding (accepted, not yet responded to),
//              the controller's queue and the responses waiting for a credit together, so a
//              stalled consumer backs the channel up into its input link: no loss, no growth
//
// Each cycle, in this order: (1) the oldest completed response is pushed, if `out` has a credit;
// (2) one landed request is accepted, if the bound allows; (3) the controller ticks once, and the
// requests it completes become responses, pushed from the next cycle.
//
// COUNTERS. The component's operational-analysis counters (counters.hpp): a request is an arrival
// when accepted and a completion when its response is pushed; the channel is stalled (no credit)
// in a cycle a response waits for `out`'s credit, busy while anything is outstanding, idle
// otherwise. dram() adds the channel's own: reads, writes, the page outcomes (hit, empty,
// conflict), refreshes, and the data bus's busy cycles (the cycles so far a CAS's burst was on
// it), for its utilization.
//
// CHECKS, every cycle (Simulator::set_checks): the controller's invariants (it checks them as it
// ticks), the data bus never carrying two bursts at once, and the outstanding bound. Any failure
// stops the run as an InvariantViolation naming the channel and the cycle.
//
// EVENTS (memory_events.hpp): with a sink attached (set_events), each request the controller
// accepts, under the id the controller gives it, and each command the controller issues, as the
// controller's own record (from the observer that also checks the data bus).
//
// TRACE. The controller's own command trace, in memory: enable_controller_trace() before the run
// (a test compares it with a bare controller's). The run's DRAM record is the .mflow
// (record/mflow_recorder.hpp); the component's state is on the cycle core's trace as any other's.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/memory_events.hpp>
#include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

struct DramRequest {
    std::uint64_t id = 0;               // unique per requester, for tracing and matching
    bool write = false;
    std::uint64_t address = 0;
    std::uint32_t size = 64;            // bytes; a write's bytes.size() when it has bytes
    std::vector<std::uint8_t> bytes;    // a write's data
    std::uint64_t tag = 0;              // the requester's: the tile, slot or burst it belongs to
};

struct DramResponse {
    std::uint64_t id = 0;               // the request's
    bool write = false;
    std::uint64_t address = 0;
    std::vector<std::uint8_t> bytes;    // a read's data; empty for a write's acknowledgement
    std::uint64_t tag = 0;
    Cycle accepted = 0, completed = 0;  // the channel's cycles: accepted, and the controller done
};

// The channel's own counters, beside the operational-analysis ones.
struct DramCounts {
    std::uint64_t reads = 0, writes = 0;
    std::uint64_t page_hits = 0, page_empty = 0, page_conflicts = 0;
    std::uint64_t refreshes = 0;
    std::uint64_t data_bus_busy = 0;    // cycles the data bus carried a burst
    std::uint64_t cycles = 0;           // the channel's cycles
    [[nodiscard]] double data_bus_utilization() const {
        return cycles ? static_cast<double>(data_bus_busy) / static_cast<double>(cycles) : 0.0;
    }
};

class Lpddr5Channel : public Component {
public:
    using Controller = lpddr5::LPDDR5MemoryController;

    // One channel: the controller's configuration must have num_channels = 1 (a device with two
    // channels has two Lpddr5Channel components).
    Lpddr5Channel(std::string name, Domain& dram, const Controller::Config& config)
        : Component(std::move(name), dram), config_(config) {
        if (config_.num_channels != 1)
            throw std::invalid_argument("cycle::Lpddr5Channel " + this->name() + ": one channel per component (num_channels " +
                                        std::to_string(config_.num_channels) + ")");
        if (config_.queue_depth == 0)
            throw std::invalid_argument("cycle::Lpddr5Channel " + this->name() + ": queue depth must be at least 1");
        // The controller holds 16 banks a channel: a wider bank field would decode a bank past them.
        if (config_.bank_bits > 4 || config_.banks_per_channel > 16)
            throw std::invalid_argument("cycle::Lpddr5Channel " + this->name() + ": the controller models 16 banks a channel (bank_bits " +
                                        std::to_string(config_.bank_bits) + ", banks " + std::to_string(config_.banks_per_channel) + ")");
        build();
    }

    void connect(Link<DramRequest>& in, Link<DramResponse>& out) {
        in_ = &in;
        out_ = &out;
    }

    // The sink this channel reports to, as channel `channel` (null: none, the default).
    void set_events(MemoryEvents* events, std::uint32_t channel = 0) {
        events_ = events;
        channel_ = channel;
    }

    // Bytes in DRAM before the run (and again after reset()): the program's inputs.
    void preload(std::uint64_t address, const std::vector<std::uint8_t>& bytes) {
        require_range(address, bytes.size(), "preload");
        write_bytes(image_, address, bytes.data(), bytes.size());
        write_bytes(store_, address, bytes.data(), bytes.size());
    }
    // Bytes in DRAM now.
    [[nodiscard]] std::vector<std::uint8_t> peek(std::uint64_t address, std::size_t size) const {
        require_range(address, size, "peek");
        std::vector<std::uint8_t> v(size);
        read_bytes(store_, address, v.data(), size);
        return v;
    }

    void evaluate(Cycle now) override {
        if (!in_ || !out_) throw std::logic_error("cycle::Lpddr5Channel " + name() + ": not connected");
        if (mc_->current_cycle() != now)
            throw std::logic_error("cycle::Lpddr5Channel " + name() + ": the controller is at cycle " +
                                   std::to_string(mc_->current_cycle()) + ", the domain at " + std::to_string(now));
        // 1. The oldest completed response, if there is a credit for it.
        bool blocked = false;
        if (!ready_.empty()) {
            if (out_->can_push(now)) {
                out_->push(now, std::move(ready_.front()));
                ready_.pop_front();
                --outstanding_;
                counters().complete();
            } else {
                blocked = true;
            }
        }
        // 2. One landed request, within the bound.
        if (in_->can_pop(now) && outstanding_ < config_.queue_depth && mc_->can_accept()) accept(now, in_->pop(now));
        // 3. The controller's cycle; what it completes is pushed from the next cycle.
        mc_->tick();
        ++dram_.cycles;
        count_bus(now);
        if (blocked) counters().stall(Stall::NoCredit);
        else if (outstanding_ > 0) counters().busy();
    }

    [[nodiscard]] bool idle() const override { return outstanding_ == 0; }

    void check(Cycle now) const override {
        if (mc_->has_violations()) {
            const auto& v = mc_->lpddr5_violations().front();
            throw std::logic_error("cycle::Lpddr5Channel " + name() + " at cycle " + std::to_string(now) +
                                   ": the controller's invariant " + v.invariant_id + " (controller cycle " +
                                   std::to_string(v.cycle) + ", bank " + std::to_string(v.bank) + "): " + v.message);
        }
        if (!bus_error_.empty())
            throw std::logic_error("cycle::Lpddr5Channel " + name() + " at cycle " + std::to_string(now) + ": " + bus_error_);
        if (outstanding_ > config_.queue_depth)
            throw std::logic_error("cycle::Lpddr5Channel " + name() + " at cycle " + std::to_string(now) + ": " +
                                   std::to_string(outstanding_) + " outstanding, past the queue depth " +
                                   std::to_string(config_.queue_depth));
    }

    void reset() override {
        build();
        store_ = image_;
        ready_.clear();
        inflight_.clear();
        outstanding_ = 0;
        dram_ = {};
        bus_end_ = 0;
        bus_windows_.clear();
        bus_error_.clear();
    }

    [[nodiscard]] std::string describe() const override {
        return std::to_string(outstanding_) + " outstanding (" + std::to_string(mc_->pending_count()) +
               " in the controller, " + std::to_string(ready_.size()) + " responses waiting), page hits " +
               std::to_string(mc_->lpddr5_stats().page_hits) + ", refreshes " + std::to_string(mc_->lpddr5_stats().refreshes);
    }

    // The channel's own counters (DramCounts).
    [[nodiscard]] DramCounts dram() const {
        DramCounts d = dram_;
        const auto& s = mc_->lpddr5_stats();
        d.reads = s.reads;
        d.writes = s.writes;
        d.page_hits = s.page_hits;
        d.page_empty = s.page_empty;
        d.page_conflicts = s.page_conflicts;
        d.refreshes = s.refreshes;
        return d;
    }
    [[nodiscard]] std::size_t outstanding() const { return outstanding_; }
    [[nodiscard]] const Controller& controller() const { return *mc_; }

    // The controller's command trace, kept in memory (controller().trace_entries()). Enable before
    // the run: reset() keeps it.
    void enable_controller_trace(bool on) {
        trace_ = on;
        mc_->enable_tracing(on);
    }

private:
    static constexpr std::size_t kLine = 64;
    using Store = std::unordered_map<std::uint64_t, std::array<std::uint8_t, kLine>>;

    void build() {
        mc_ = std::make_unique<Controller>(config_);
        mc_->enable_invariants(true);
        mc_->enable_tracing(trace_);
        // Each CAS's data-bus window [data_start, data_end), reported when the CAS issues: no two at
        // once, and its cycles counted busy as they pass (count_bus), not ahead of time.
        mc_->set_command_observer([this](const Controller::CommandRecord& r) {
            using K = Controller::CommandRecord::Kind;
            if (events_) events_->dram_command(channel_, r);
            if (r.kind != K::Read && r.kind != K::Write) return;
            if (r.data_start < bus_end_ && bus_error_.empty())
                bus_error_ = "a burst on the data bus at controller cycle " + std::to_string(r.data_start) +
                             " overlaps the one ending at " + std::to_string(bus_end_);
            bus_end_ = std::max(bus_end_, r.data_end);
            bus_windows_.emplace_back(r.data_start, r.data_end);
        });
    }
    // Cycle c of the controller: busy if a burst's window covers it. Windows do not overlap and
    // arrive in the order they start (checked above), so only the oldest can cover c.
    void count_bus(Cycle c) {
        while (!bus_windows_.empty() && bus_windows_.front().second <= c) bus_windows_.pop_front();
        if (!bus_windows_.empty() && bus_windows_.front().first <= c) ++dram_.data_bus_busy;
    }
    // [address, address + size) must not wrap past 2^64 (a one-byte access at the top is fine).
    void require_range(std::uint64_t address, std::size_t size, const char* what) const {
        if (size > 0 && address > ~std::uint64_t{0} - (size - 1))
            throw std::invalid_argument("cycle::Lpddr5Channel " + name() + ": " + what + " of " + std::to_string(size) +
                                        " bytes at " + std::to_string(address) + " wraps past the top of the address space");
    }

    void accept(Cycle now, DramRequest q) {
        require_range(q.address, q.write && !q.bytes.empty() ? q.bytes.size() : q.size, q.write ? "a write" : "a read");
        counters().accept();
        ++outstanding_;
        DramResponse r;
        r.id = q.id;
        r.write = q.write;
        r.address = q.address;
        r.tag = q.tag;
        r.accepted = now;
        std::optional<std::uint64_t> id;
        const std::uint64_t key = next_key_++;
        auto done = [this, key] {
            auto it = inflight_.find(key);
            it->second.completed = mc_->current_cycle();
            ready_.push_back(std::move(it->second));
            inflight_.erase(it);
        };
        if (q.write) {
            const std::uint32_t size = q.bytes.empty() ? q.size : static_cast<std::uint32_t>(q.bytes.size());
            q.bytes.resize(size);
            write_bytes(store_, q.address, q.bytes.data(), q.bytes.size());
            inflight_.emplace(key, std::move(r));
            id = mc_->submit_write(q.address, q.bytes.data(), size, done);
        } else {
            r.bytes.resize(q.size);
            read_bytes(store_, q.address, r.bytes.data(), q.size);
            inflight_.emplace(key, std::move(r));
            id = mc_->submit_read(q.address, q.size, done);
        }
        if (!id) throw std::logic_error("cycle::Lpddr5Channel " + name() + ": the controller refused a request it could accept");
        if (events_) events_->dram_accepted(channel_, domain().edge(now), *id, q.id, q.write, q.address);
    }

    static void write_bytes(Store& s, std::uint64_t address, const std::uint8_t* p, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t a = address + i;
            auto [it, fresh] = s.try_emplace(a / kLine);
            if (fresh) it->second.fill(0);
            it->second[a % kLine] = p[i];
        }
    }
    static void read_bytes(const Store& s, std::uint64_t address, std::uint8_t* p, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t a = address + i;
            const auto it = s.find(a / kLine);
            p[i] = it == s.end() ? 0 : it->second[a % kLine];
        }
    }

    Controller::Config config_;
    std::unique_ptr<Controller> mc_;
    Link<DramRequest>* in_ = nullptr;
    Link<DramResponse>* out_ = nullptr;
    Store store_, image_;
    std::unordered_map<std::uint64_t, DramResponse> inflight_;   // in the controller, by key
    std::deque<DramResponse> ready_;                             // completed, waiting for a credit
    std::uint64_t next_key_ = 0;
    std::size_t outstanding_ = 0;
    DramCounts dram_;
    std::uint64_t bus_end_ = 0;
    std::deque<std::pair<std::uint64_t, std::uint64_t>> bus_windows_;   // bursts not yet past
    std::string bus_error_;
    bool trace_ = false;
    MemoryEvents* events_ = nullptr;
    std::uint32_t channel_ = 0;
};

}  // namespace sw::kpu::cycle
