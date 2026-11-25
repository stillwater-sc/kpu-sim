#pragma once
// ============================================================================
// include/sw/kpu/cycle/record/mflow_recorder.hpp
// The cycle core's memory-flow recorder (docs/plans/cycle-mflow-and-debug-server.md §3.1-§3.2,
// step 1): a MemoryEvents sink that turns what the DMA engine and the LPDDR5 channel report into a
// MemoryFlowRecord, the .mflow record kpu-memsim writes (program/record/memory_flow_record.hpp),
// so tools/trace/mflow_check.py and tools/visualization/memflow read a cycle-core run as they read
// a memory-side one.
//
//   MflowRecorder rec(MflowSetup{"kpu_s1", window, 0, store_buffer_beats, lpddr5_one_channel()});
//   MemoryRunOptions o; o.events = &rec;
//   const MemoryRun r = run_memory_subsystem(plan, &device, o);
//   write_mflow(rec.record(r.end), "run.mflow");
//
// WHAT IS RECORDED
//   bursts    one per DRAM request a DMA engine sends: posted (the engine sends it), submitted
//             (the channel accepts it into the controller), its first command, its data window,
//             and done (it leaves the engine's window: a read's beat pushed to L3, a write's
//             acknowledgment taken). Its bank, row, column and page outcome are its CAS's
//   commands  every command the controller issues, the burst it serves (a PRECHARGE: the burst that
//             opened the page; a REFRESH: none), in KPU cycles and in the controller's own ticks
//   requests  one per Load, Store and Fill, with its kind, ticket and slot: offered (the action
//             before it started; 0 for the first), credit (it started: a Load or a Fill holds its
//             slot's credit), its first and last bursts (a Fill has none: both are its start), and
//             retired (complete)
//   buffers   the store buffer's occupancy each time it changes (held = staged: a beat is in the
//             buffer only once it has been ejected into it)
//   windows   the engine's two: `window` its reads in flight, `write_window` its writes
//   waits     the causes that held a DMA engine up, as intervals [from, to) of KPU cycles, one per
//             run of consecutive cycles a cause held. Several causes can hold in one cycle, so two
//             causes' intervals may overlap; one cause's never do
//
// TIME. The record's times are whole KPU cycles: an event at time t is in the KPU cycle n with
// edge(n) <= t < edge(n + 1). That is monotone, so ordered events stay ordered. A command's points
// are also kept exactly, in the controller's ticks (k_issue, ...): DRAM timing is checked on those
// (I4), never on the rounded cycles.
//
// ONE ENGINE A CHANNEL. A DramRequest's id is unique per requester, and the channel reports
// accepted requests by that id alone; the recorder therefore refuses a second engine until requests
// name their requester.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>
#include <sw/kpu/cycle/memory/memory_events.hpp>
#include <sw/kpu/cycle/time.hpp>
#include <sw/kpu/program/record/memory_flow_record.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

// What the record's manifest says about the machine, known before the run.
struct MflowSetup {
    std::string device = "cycle";
    std::size_t window = 8;                    // a DMA engine's reads in flight
    std::size_t write_window = 0;              // its writes in flight (0 = `window`)
    std::size_t store_buffer_beats = 4;
    Lpddr5Channel::Controller::Config dram;    // the channel's controller (lpddr5_one_channel())
};

// The record's wait reasons are the engine's DmaWait, in its order.
static_assert(static_cast<int>(program::record::MemoryFlowRecord::WaitReason::Order) == static_cast<int>(DmaWait::Order) &&
                  program::record::MemoryFlowRecord::kWaitReasons == kDmaWaits,
              "MemoryFlowRecord::WaitReason must list DmaWait's causes in its order");

class MflowRecorder : public MemoryEvents {
public:
    using Record = program::record::MemoryFlowRecord;

    struct Wait {
        std::uint32_t engine = 0;
        DmaWait why = DmaWait::SlotCredit;
        Cycle from = 0, to = 0;                // KPU cycles, [from, to)
    };
    struct Action {
        std::uint32_t engine = 0;
        std::size_t ticket = 0;
        L3Action::Kind kind = L3Action::Kind::Load;
        std::uint32_t slot = 0;
        std::uint64_t address = 0;
        std::uint32_t bytes = 0;
        Cycle offered = 0, started = 0;
        std::optional<Cycle> completed, first_burst, last_burst;
    };

    explicit MflowRecorder(MflowSetup setup) : setup_(std::move(setup)) {
        if (setup_.write_window == 0) setup_.write_window = setup_.window;
        if (setup_.dram.bank_groups == 0 || setup_.dram.banks_per_channel % setup_.dram.bank_groups != 0)
            throw std::invalid_argument("MflowRecorder: " + std::to_string(setup_.dram.banks_per_channel) +
                                        " banks do not divide into " + std::to_string(setup_.dram.bank_groups) +
                                        " bank groups");
    }

    // ---- MemoryEvents ----
    // Copies of the clocks, not references: the domains belong to the run's Simulator, which is
    // gone by the time record() is called after run_memory_subsystem returns. (Only the period
    // and phase are used: edge() and first_after() do not depend on a domain's running state.)
    void clocks(const Domain& kpu, const Domain& dram) override {
        kpu_.emplace(kpu);
        dram_.emplace(dram);
    }

    void action_started(std::uint32_t engine, Picoseconds t, std::size_t ticket, const L3Action& a) override {
        const Cycle c = kpu_cycle(t);
        Action x;
        x.engine = engine;
        x.ticket = ticket;
        x.kind = a.kind;
        x.slot = a.slot;
        x.address = a.dram_address;
        x.bytes = a.bytes;
        const auto last = last_started_.find(engine);
        x.offered = last == last_started_.end() ? 0 : last->second;
        x.started = c;
        last_started_[engine] = c;
        action_of_[ticket] = actions_.size();
        actions_.push_back(x);
    }
    void action_completed(std::uint32_t, Picoseconds t, std::size_t ticket) override {
        action(ticket).completed = kpu_cycle(t);
    }
    void burst_posted(std::uint32_t engine, Picoseconds t, std::size_t ticket, std::uint64_t request,
                      std::uint64_t, bool write) override {
        if (!engine_) engine_ = engine;
        else if (*engine_ != engine)
            throw std::logic_error("MflowRecorder: bursts from engines " + std::to_string(*engine_) + " and " +
                                   std::to_string(engine) + "; one engine a channel until requests name their requester");
        const Cycle c = kpu_cycle(t);
        Record::Burst b;
        b.engine = engine;
        b.is_load = !write;
        b.posted = c;
        if (!by_request_.emplace(request, bursts_.size()).second)
            throw std::logic_error("MflowRecorder: engine " + std::to_string(engine) + " posted request " +
                                   std::to_string(request) + " twice");
        bursts_.push_back(b);
        burst_ticket_.push_back(ticket);
        first_command_.push_back(false);
        Action& a = action(ticket);
        if (!a.first_burst) a.first_burst = c;
        a.last_burst = c;
    }
    void burst_retired(std::uint32_t, Picoseconds t, std::uint64_t request) override {
        burst(request).done = kpu_cycle(t);
    }
    void dma_waits(std::uint32_t engine, Picoseconds t, std::uint32_t mask) override {
        const Cycle c = kpu_cycle(t);
        auto& open = open_waits_[engine];
        for (std::size_t w = 0; w < kDmaWaits; ++w) {
            if (!(mask & (1u << w))) continue;
            if (open[w] && waits_[*open[w]].to == c) {
                waits_[*open[w]].to = c + 1;
            } else {
                open[w] = waits_.size();
                waits_.push_back({engine, static_cast<DmaWait>(w), c, c + 1});
            }
        }
    }
    void store_buffer(std::uint32_t engine, Picoseconds t, std::size_t beats) override {
        const auto n = static_cast<std::uint32_t>(beats);
        buffers_.push_back({engine, kpu_cycle(t), n, n});
    }
    void dram_accepted(std::uint32_t, Picoseconds t, std::uint64_t controller_id, std::uint64_t request, bool,
                       std::uint64_t) override {
        const auto it = by_request_.find(request);
        if (it == by_request_.end())
            throw std::logic_error("MflowRecorder: the channel accepted request " + std::to_string(request) +
                                   ", which no engine posted");
        bursts_[it->second].submitted = kpu_cycle(t);
        by_controller_[controller_id] = it->second;
    }
    void dram_command(std::uint32_t channel, const DramCommand& r) override {
        using K = DramCommand::Kind;
        using CK = Record::CommandKind;
        const std::uint32_t per_group = setup_.dram.banks_per_channel / setup_.dram.bank_groups;
        Record::Command x;
        x.mc = 0;
        x.channel = static_cast<std::uint8_t>(channel);
        x.bank_group = static_cast<std::uint8_t>(r.bank / per_group);
        x.bank = static_cast<std::uint8_t>(r.bank % per_group);
        x.row = r.row;
        x.kind = r.kind == K::Activate ? CK::Activate
               : r.kind == K::Read     ? CK::Read
               : r.kind == K::Write    ? CK::Write
               : r.kind == K::Precharge ? CK::Precharge
                                        : CK::Refresh;
        x.issue = at(r.issue);
        x.end = at(r.end);
        x.tick = r.issue;
        x.tick_end = r.end;
        const bool cas = r.kind == K::Read || r.kind == K::Write;
        if (cas) {
            x.data_start = at(r.data_start);
            x.data_end = at(r.data_end);
            x.tick_data_start = r.data_start;
            x.tick_data_end = r.data_end;
        }
        if (r.request_id) {
            const auto it = by_controller_.find(r.request_id);
            if (it != by_controller_.end()) {
                x.burst = static_cast<std::uint32_t>(it->second);
                Record::Burst& b = bursts_[it->second];
                if (!first_command_[it->second]) {
                    first_command_[it->second] = true;
                    b.first_command = x.issue;
                }
                if (cas) {
                    b.channel = x.channel;
                    b.bank_group = x.bank_group;
                    b.bank = x.bank;
                    b.row = r.row;
                    b.col = r.col;
                    b.data_start = x.data_start;
                    b.data_end = x.data_end;
                    b.outcome = r.conflicted ? Record::Outcome::Conflict
                              : r.activated  ? Record::Outcome::Empty
                                             : Record::Outcome::Hit;
                }
            }
        }
        commands_.push_back(x);
    }

    // ---- results ----
    // The record of the run that ended at `end` (MemoryRun::end, Simulator::now()).
    [[nodiscard]] Record record(Picoseconds end) const {
        require_clocks();
        Record rec;
        rec.source = "cycle-core";
        rec.device = setup_.device;
        rec.makespan = kpu_cycle(end);
        rec.window = static_cast<std::uint32_t>(setup_.window);
        rec.write_window = static_cast<std::uint32_t>(setup_.write_window);
        rec.burst_bytes = kL3BeatBytes;
        rec.arbitration = "fixed";   // one engine a channel: its requests in the order it sends them
        rec.grant_quantum = 1;
        rec.ticks_per_cycle = kpu_->period_ps() / dram_->period_ps();
        rec.dram_timing = timing_table();
        const std::uint32_t burst_ticks = burst_length_ticks();
        rec.ceiling_bytes_per_cycle =
            burst_ticks ? static_cast<double>(kL3BeatBytes) * rec.ticks_per_cycle / burst_ticks : 0.0;
        std::ostringstream note;
        note << "LPDDR5-6400: the controller's table; FR-FCFS scheduling; " << rec.ticks_per_cycle
             << " controller ticks per KPU cycle; recorded on the cycle core (one channel, one DMA engine)";
        rec.timing_note = note.str();

        // Stations: banks, the data bus, the engine, its store buffer, its waits (memory_flow_record.hpp's order).
        const std::uint32_t per_group = setup_.dram.banks_per_channel / setup_.dram.bank_groups;
        const std::string mc = setup_.device + "/mc[0]/ch[0]";
        for (std::uint32_t g = 0; g < setup_.dram.bank_groups; ++g)
            for (std::uint32_t b = 0; b < per_group; ++b)
                rec.stations.push_back({mc + "/bg[" + std::to_string(g) + "]/ba[" + std::to_string(b) + "]", "dram_bank", 1});
        rec.stations.push_back({mc + "/bus", "dram_bus", 1});
        const std::uint32_t engine = engine_.value_or(0);
        rec.stations.push_back({setup_.device + "/dma[" + std::to_string(engine) + "]", "dma", setup_.window + setup_.write_window});
        rec.stations.push_back({setup_.device + "/dmabuf[" + std::to_string(engine) + "]", "dmabuf", setup_.store_buffer_beats});
        rec.stations.push_back({setup_.device + "/dmawait[" + std::to_string(engine) + "]", "dmawait", Record::kWaitReasons});

        // Requests: the Loads, Stores and Fills, in the order they started.
        std::unordered_map<std::size_t, std::uint32_t> request_of;   // ticket -> request row
        for (const Action& a : actions_) {
            if (a.kind == L3Action::Kind::Release) continue;   // never an engine's action
            request_of[a.ticket] = static_cast<std::uint32_t>(rec.requests.size());
            Record::Request q;
            q.engine = a.engine;
            q.port = 0;
            q.is_load = a.kind != L3Action::Kind::Store;   // a Fill lands in L3, as a Load does
            q.kind = a.kind == L3Action::Kind::Load    ? Record::RequestKind::Load
                   : a.kind == L3Action::Kind::Store   ? Record::RequestKind::Store
                                                       : Record::RequestKind::Fill;
            q.ticket = static_cast<std::uint32_t>(a.ticket);
            q.slot = a.slot;
            q.address = a.address;
            q.bytes = a.bytes;
            q.offered = a.offered;
            q.credit = a.started;
            q.first_burst = a.first_burst.value_or(a.started);
            q.last_burst = a.last_burst.value_or(a.started);
            q.retired = a.completed.value_or(0);   // 0: the run ended before it completed
            rec.requests.push_back(q);
        }
        rec.bursts = bursts_;
        for (std::size_t i = 0; i < rec.bursts.size(); ++i) {
            const auto it = request_of.find(burst_ticket_[i]);
            rec.bursts[i].request = it == request_of.end() ? program::record::kNone : it->second;
        }
        rec.commands = commands_;
        rec.buffers = buffers_;
        for (const Wait& w : waits_)
            rec.waits.push_back({w.engine, static_cast<Record::WaitReason>(w.why), w.from, w.to});
        return rec;
    }

    [[nodiscard]] const std::vector<Wait>& waits() const { return waits_; }
    [[nodiscard]] const std::vector<Action>& actions() const { return actions_; }
    [[nodiscard]] const MflowSetup& setup() const { return setup_; }

private:
    void require_clocks() const {
        if (!kpu_ || !dram_) throw std::logic_error("MflowRecorder: no clocks (attach it before the run)");
    }
    // The KPU cycle t falls in: edge(n) <= t < edge(n + 1).
    [[nodiscard]] Cycle kpu_cycle(Picoseconds t) const {
        require_clocks();
        const Cycle n = kpu_->first_after(t);
        return n == 0 ? 0 : n - 1;
    }
    // A controller tick, as the KPU cycle it falls in.
    [[nodiscard]] Cycle at(std::uint64_t tick) const { return kpu_cycle(dram_->edge(tick)); }

    Action& action(std::size_t ticket) {
        const auto it = action_of_.find(ticket);
        if (it == action_of_.end())
            throw std::logic_error("MflowRecorder: an event of action " + std::to_string(ticket) + ", which never started");
        return actions_[it->second];
    }
    Record::Burst& burst(std::uint64_t request) {
        const auto it = by_request_.find(request);
        if (it == by_request_.end())
            throw std::logic_error("MflowRecorder: an event of request " + std::to_string(request) + ", never posted");
        return bursts_[it->second];
    }

    [[nodiscard]] std::uint32_t burst_length_ticks() const {
        const auto& t = setup_.dram.timing;
        return setup_.dram.burst_length == lpddr5::BurstLength::BL32 ? t.tBurst_BL32 : t.tBurst_BL16;
    }
    // The table the controller ran (as DramBridge::timing_table: the same names, for mflow_check M5).
    [[nodiscard]] std::vector<std::pair<std::string, std::uint32_t>> timing_table() const {
        const auto& t = setup_.dram.timing;
        return {{"tRCD", t.tRCD},     {"tRP", t.tRP},       {"tRAS", t.tRAS},     {"tRC", t.tRC},
                {"tCL", t.tCL},       {"tWL", t.tWL},       {"tWR", t.tWR},       {"tRTP", t.tRTP},
                {"tRRD_L", t.tRRD_L}, {"tRRD_S", t.tRRD_S}, {"tCCD_L", t.tCCD_L}, {"tCCD_S", t.tCCD_S},
                {"tWTR_L", t.tWTR_L}, {"tWTR_S", t.tWTR_S}, {"tRTW", t.tRTW},     {"tBurst", burst_length_ticks()},
                {"tRFCpb", t.tRFCpb}, {"tRFCab", t.tRFCab}, {"tREFIpb", t.tREFIpb}, {"tFAW", t.tFAW}};
    }

    MflowSetup setup_;
    std::optional<Domain> kpu_, dram_;
    std::optional<std::uint32_t> engine_;
    std::vector<Record::Burst> bursts_;
    std::vector<std::size_t> burst_ticket_;
    std::vector<bool> first_command_;                          // a burst's first command is recorded
    std::unordered_map<std::uint64_t, std::size_t> by_request_;     // DramRequest id -> burst
    std::unordered_map<std::uint64_t, std::size_t> by_controller_;  // controller id -> burst
    std::vector<Record::Command> commands_;
    std::vector<Record::BufferSample> buffers_;
    std::vector<Action> actions_;
    std::unordered_map<std::size_t, std::size_t> action_of_;   // ticket -> action
    std::unordered_map<std::uint32_t, Cycle> last_started_;    // engine -> its last action's start
    std::vector<Wait> waits_;
    std::unordered_map<std::uint32_t, std::array<std::optional<std::size_t>, kDmaWaits>> open_waits_;
};

}  // namespace sw::kpu::cycle
