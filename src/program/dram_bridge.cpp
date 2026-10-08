// ============================================================================
// src/program/dram_bridge.cpp
// The hosted LPDDR5 controller (docs/plans/dram-bank-model.md step 2). See the header.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/timing/dram_bridge.hpp>

#include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>

#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>
#include <sstream>

namespace sw::kpu::timing {

using program::platform::DramCoord;
using program::platform::DramMapError;

DramHosting DramHosting::of(const program::platform::DeviceSpecification& d) {
    DramHosting h{program::platform::DramAddressMap::of(d), d.memory.dram->technology,
                  d.memory.dram->data_rate_mtps, d.memory.dram->channel_width_bits};
    const std::string who = "dram bridge: device \"" + d.name + "\": ";
    if (h.technology != "lpddr5" && h.technology != "lpddr5x")
        throw DramMapError(who + "the hosted controller is LPDDR5; '" + h.technology +
                           "' needs its own controller");
    if (h.map.channels() > 2)
        throw DramMapError(who + "an LPDDR5 controller drives 1 or 2 channels, not " +
                           std::to_string(h.map.channels()));
    if (h.map.ranks() != 1)
        throw DramMapError(who + "the hosted controller models one rank, not " +
                           std::to_string(h.map.ranks()));
    if (h.map.bank_groups() != 4 || h.map.banks_per_group() != 4)
        throw DramMapError(who + "the hosted controller has 4 bank groups of 4 banks");
    if (h.map.rows() > (std::uint64_t{1} << 31))
        throw DramMapError(who + "more rows than the controller's 31-bit row field");
    const unsigned beats = h.map.burst_bytes() * 8 / h.channel_width_bits;
    if (beats != 16 && beats != 32)
        throw DramMapError(who + "a " + std::to_string(h.map.burst_bytes()) + "-byte burst on a x" +
                           std::to_string(h.channel_width_bits) + " channel is BL" +
                           std::to_string(beats) + "; LPDDR5 has BL16 and BL32");
    return h;
}

namespace {
using lpddr5::TimingParams;

// The controller's table is LPDDR5-6400, counted in its 3.2 GHz clock. A faster part keeps
// the nanosecond-fixed parameters fixed in nanoseconds, so their cycle counts grow with the
// clock; the burst-relative ones (burst length, CAS-to-CAS) stay in cycles. Rounded up: a
// constraint rounded down would let the model run faster than the part.
TimingParams scaled(unsigned data_rate_mtps) {
    TimingParams t;
    if (data_rate_mtps == 6400) return t;
    const double f = static_cast<double>(data_rate_mtps) / 6400.0;
    auto up = [&](std::uint32_t& p) { p = static_cast<std::uint32_t>(std::ceil(p * f)); };
    for (std::uint32_t* p : {&t.tRCD, &t.tRP, &t.tRAS, &t.tCL, &t.tWL, &t.tWR, &t.tRTP,
                             &t.tRRD_L, &t.tRRD_S, &t.tWTR_L, &t.tWTR_S, &t.tRTW,
                             &t.tRFCpb, &t.tRFCab, &t.tFAW})
        up(*p);
    t.tRC = std::max<std::uint32_t>(static_cast<std::uint32_t>(std::ceil(t.tRC * f)), t.tRAS + t.tRP);
    // The refresh INTERVAL rounds down: refreshing late is the optimistic error.
    t.tREFIpb = static_cast<std::uint32_t>(std::floor(t.tREFIpb * f));
    return t;
}

unsigned bits_for(std::uint64_t n) {   // n is a power of two
    unsigned b = 0;
    while ((std::uint64_t{1} << b) < n) ++b;
    return b;
}
} // namespace

struct DramBridge::Impl {
    lpddr5::LPDDR5MemoryController::Config cfg;
    std::unique_ptr<lpddr5::LPDDR5MemoryController> mc;
    std::vector<std::uint64_t> done;    // filled by completion callbacks during a tick
    double owed = 0.0;
    Cycle last = 0;
    std::uint64_t bursts = 0, misrouted = 0;
    std::function<void(const Command&)> sink;
    std::unordered_map<std::uint64_t, std::uint64_t> tag_of;   // controller request id -> tag
    // Requests that opened a row (issued an ACT). The precharge that closes the row names its
    // opener, and comes after the opener's burst has completed, so an opener's tag is kept until
    // that precharge is observed; every other request's is dropped when its burst completes.
    std::unordered_set<std::uint64_t> openers;
    // Openers whose burst has completed: their tag is dropped at the precharge. An opener still
    // pending when its row is closed (FR-FCFS may close a row for a starved request before the
    // opener's CAS) keeps its tag: it will open the row again and its commands still name it.
    std::unordered_set<std::uint64_t> completed_openers;

    std::uint64_t native(const DramCoord& c, unsigned flat_bank) const {
        // The controller's own layout: [row | bank | col | channel | 64-byte offset].
        std::uint64_t a = c.row;
        a = (a << cfg.bank_bits) | flat_bank;
        a = (a << cfg.col_bits) | c.col;
        if (cfg.num_channels > 1) a = (a << 1) | c.channel;
        return a << 6;
    }
};

DramBridge::DramBridge(const DramHosting& h, unsigned controller_id, double executor_clock_ghz,
                       std::uint32_t queue_depth)
    : impl_(std::make_unique<Impl>()), hosting_(h), controller_id_(controller_id),
      // DDR: the controller's clock is half the data rate.
      ticks_per_cycle_((h.data_rate_mtps / 2.0) / (executor_clock_ghz * 1000.0)) {
    auto& c = impl_->cfg;
    c.num_channels = static_cast<std::uint8_t>(h.map.channels());
    c.banks_per_channel = 16;
    c.bank_groups = 4;
    const unsigned beats = h.map.burst_bytes() * 8 / h.channel_width_bits;
    c.burst_length = beats == 32 ? lpddr5::BurstLength::BL32 : lpddr5::BurstLength::BL16;
    c.queue_depth = queue_depth;
    c.timing = scaled(h.data_rate_mtps);
    c.row_bits = std::max(1u, bits_for(h.map.rows()));
    c.col_bits = std::max(1u, bits_for(h.map.bursts_per_page()));
    c.bank_bits = 4;
    c.channel_bits = c.num_channels > 1 ? 1 : 0;
    impl_->mc = std::make_unique<lpddr5::LPDDR5MemoryController>(c);
    impl_->mc->set_command_observer([this](const lpddr5::LPDDR5MemoryController::CommandRecord& r) {
        if (!impl_->sink) return;
        using K = lpddr5::LPDDR5MemoryController::CommandRecord::Kind;
        // The controller is being ticked up to executor cycle `last`: the command issues then,
        // and its later points are that many ticks on, at ticks_per_cycle_.
        const Cycle now = impl_->last;
        auto at = [&](std::uint64_t tick) -> Cycle {
            if (tick <= r.issue) return now;
            return now + static_cast<Cycle>(
                             std::ceil(static_cast<double>(tick - r.issue) / ticks_per_cycle_));
        };
        Command cmd;
        cmd.kind = r.kind == K::Activate    ? Command::Kind::Activate
                 : r.kind == K::Read        ? Command::Kind::Read
                 : r.kind == K::Write       ? Command::Kind::Write
                 : r.kind == K::Precharge   ? Command::Kind::Precharge
                                            : Command::Kind::Refresh;
        cmd.mc = controller_id_;
        cmd.channel = r.channel;
        cmd.bank_group = r.bank / hosting_.map.banks_per_group();
        cmd.bank = r.bank % hosting_.map.banks_per_group();
        cmd.row = r.row;
        cmd.col = r.col;
        cmd.issue = now;
        cmd.end = at(r.end);
        if (cmd.kind == Command::Kind::Read || cmd.kind == Command::Kind::Write) {
            cmd.data_start = at(r.data_start);
            cmd.data_end = at(r.data_end);
        }
        if (r.request_id) {
            auto it = impl_->tag_of.find(r.request_id);
            if (it != impl_->tag_of.end()) cmd.tag = it->second;
            if (r.kind == K::Activate) {
                impl_->openers.insert(r.request_id);
            } else if (r.kind == K::Precharge) {
                // The row is closed. Its opener's tag goes only if that burst has completed.
                impl_->openers.erase(r.request_id);
                if (impl_->completed_openers.erase(r.request_id)) impl_->tag_of.erase(r.request_id);
            }
        }
        cmd.activated = r.activated;
        cmd.conflicted = r.conflicted;
        impl_->sink(cmd);
    });

    std::ostringstream n;
    n << h.technology << "-" << h.data_rate_mtps << ": "
      << (h.data_rate_mtps == 6400 ? "the controller's LPDDR5-6400 table"
                                   : "timing DERIVED from the LPDDR5-6400 table by data rate "
                                     "(ns-fixed parameters scaled, burst-relative kept), not a "
                                     "datasheet table")
      << "; FR-FCFS scheduling; " << ticks_per_cycle_ << " controller ticks per executor cycle";
    timing_note_ = n.str();
}

DramBridge::~DramBridge() = default;

bool DramBridge::submit(std::uint64_t address, bool is_load, std::uint64_t tag) {
    const DramCoord c = hosting_.map.decode(address);       // throws past the top of memory
    if (!impl_->mc->can_accept()) return false;
    if (c.mc != controller_id_) ++impl_->misrouted;
    const std::uint64_t a = impl_->native(c, hosting_.map.flat_bank(c));
    auto& done = impl_->done;
    // With a command sink, map the controller's request id to the tag so commands name their
    // burst. The id is known only after submit, so the callback reads it from a shared slot.
    Impl* impl = impl_->sink ? impl_.get() : nullptr;
    auto slot = std::make_shared<std::uint64_t>(0);
    auto cb = [&done, tag, impl, slot] {
        done.push_back(tag);
        // An opener's tag outlives its burst: the precharge that closes its row names it.
        if (!impl) return;
        if (impl->openers.count(*slot)) impl->completed_openers.insert(*slot);
        else impl->tag_of.erase(*slot);
    };
    const auto id = is_load ? impl_->mc->submit_read(a, hosting_.map.burst_bytes(), cb)
                            : impl_->mc->submit_write(a, nullptr, hosting_.map.burst_bytes(), cb);
    if (!id) return false;
    *slot = *id;
    if (impl) impl->tag_of[*id] = tag;
    ++impl_->bursts;
    return true;
}

void DramBridge::advance(Cycle now, std::vector<std::uint64_t>& completed) {
    if (now > impl_->last) {
        impl_->owed += static_cast<double>(now - impl_->last) * ticks_per_cycle_;
        impl_->last = now;
    }
    for (; impl_->owed >= 1.0; impl_->owed -= 1.0) impl_->mc->tick();
    completed.insert(completed.end(), impl_->done.begin(), impl_->done.end());
    impl_->done.clear();
}

bool DramBridge::busy() const { return impl_->mc->has_pending(); }

void DramBridge::reset() {
    impl_->mc->reset();
    impl_->done.clear();
    impl_->owed = 0.0;
    impl_->last = 0;
    impl_->bursts = impl_->misrouted = 0;
    impl_->tag_of.clear();
    impl_->openers.clear();
    impl_->completed_openers.clear();
}

void DramBridge::set_command_sink(std::function<void(const Command&)> sink) {
    impl_->sink = std::move(sink);
}

DramBridge::Stats DramBridge::stats() const {
    const auto& s = impl_->mc->lpddr5_stats();
    Stats out;
    out.bursts = impl_->bursts;
    out.page_hits = s.page_hits;
    out.page_empty = s.page_empty;
    out.page_conflicts = s.page_conflicts;
    out.refreshes = s.refreshes;
    out.stall_ticks = s.stall_cycles;
    out.misrouted = impl_->misrouted;
    out.violations = impl_->mc->lpddr5_violations().size();
    return out;
}

} // namespace sw::kpu::timing
