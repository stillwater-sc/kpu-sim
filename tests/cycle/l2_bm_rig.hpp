#pragma once
// ============================================================================
// tests/cycle/l2_bm_rig.hpp
// L3 -> BlockMover -> L2 and back, for step 2a's tests (#542): the BlockMover in the KPU's domain
// (1 GHz by default), the L2 in its own (250 MHz by default), a crossing each way between them; and
// a stand-in for L3 in the KPU's domain.
//
// THE STAND-IN L3 holds bytes at addresses, loaded directly (plan §5.2: "the L3s start with A and B
// in the slots the program places them in ... or the same state loaded directly"). It serves the
// BlockMover's row reads in order, one beat of the bus a clock, and takes its writeback one beat a
// clock. Phase 1's L3Tile gets the BlockMover's ports at the join (step 3), where their contention
// with the DMA engine's can be measured; until then this stand-in is the BlockMover's L3.
//
// The rig registers check_l2_slot_credits as an invariant, so it holds every cycle of every run.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/compute/block_mover.hpp>
#include <sw/kpu/cycle/compute/l2.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::cycle::test {

class L3Standin : public Component {
public:
    L3Standin(std::string name, Domain& d, std::uint32_t bus_bytes) : Component(std::move(name), d), bus_(bus_bytes) {}

    void connect(Link<L3RowRead>& reads, Link<Beat>& data, Link<Beat>& writes) {
        reads_ = &reads;
        data_ = &data;
        writes_ = &writes;
    }
    void load(std::uint64_t address, const std::vector<std::uint8_t>& bytes) {
        for (std::size_t i = 0; i < bytes.size(); ++i) mem_[address + i] = bytes[i];
    }
    [[nodiscard]] std::vector<std::uint8_t> peek(std::uint64_t address, std::size_t n) const {
        std::vector<std::uint8_t> v(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const auto it = mem_.find(address + i);
            if (it != mem_.end()) v[i] = it->second;
        }
        return v;
    }

    void evaluate(Cycle now) override {
        bool moved = false;
        if (!current_ && reads_->can_pop(now)) {
            current_ = reads_->pop(now);
            sent_ = 0;
        }
        if (current_ && data_->can_push(now)) {
            Beat b;
            b.id = ++id_;
            b.address = current_->address + sent_;
            b.bytes = peek(b.address, bus_);
            b.tag = current_->tag;
            b.first = sent_ == 0;
            sent_ += bus_;
            b.last = sent_ >= current_->bytes;
            data_->push(now, std::move(b));
            if (sent_ >= current_->bytes) current_.reset();
            moved = true;
        }
        if (writes_->can_pop(now)) {
            const Beat b = writes_->pop(now);
            load(b.address, b.bytes);
            ++beats_written_;
            moved = true;
        }
        if (moved) counters().busy();
    }
    [[nodiscard]] bool idle() const override { return !current_; }
    [[nodiscard]] std::uint64_t beats_written() const { return beats_written_; }

private:
    std::uint32_t bus_;
    std::map<std::uint64_t, std::uint8_t> mem_;
    std::optional<L3RowRead> current_;
    std::uint32_t sent_ = 0;
    std::uint64_t id_ = 0, beats_written_ = 0;
    Link<L3RowRead>* reads_ = nullptr;
    Link<Beat>* data_ = nullptr;
    Link<Beat>* writes_ = nullptr;
};

struct L2BmRig {
    struct Options {
        L2::Config l2{};
        std::uint32_t bus_bytes = 64;
        std::uint32_t row_buffers = 2;
        std::uint32_t read_window = 8;
        double kpu_ghz = 1.0;
        double l2_ghz = 0.25;
        std::size_t l3_link = 2;     // the L3 links' credits
        std::size_t l2_link = 4;     // the crossings' entries
        Cycle credit_sync = 2;       // the edges of the mover's clock before it sees a slot credit
        Tracer* tracer = nullptr;
    };

    Simulator sim;
    Domain& kpu;
    Domain& l2d;
    L3Standin& l3;
    BlockMover& bm;
    L2& l2;
    Link<L3RowRead>& l3_reads;
    Link<Beat>& l3_data;
    Link<Beat>& l3_writes;
    Link<L2RowWrite>& l2_writes;
    Link<L2RowRead>& l2_reads;
    Link<L2RowData>& l2_data;
    Link<SlotCredit>& credits;

    L2BmRig(const L2Program& program, const Options& o)
        : kpu(traced(sim, o.tracer).add_domain(Domain::from_ghz("kpu", o.kpu_ghz))),
          l2d(sim.add_domain(Domain::from_ghz("l2", o.l2_ghz))),
          l3(sim.add<L3Standin>("l3", kpu, o.bus_bytes)),
          bm(sim.add<BlockMover>("bm", kpu,
                                 BlockMover::Config{o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, o.bus_bytes,
                                                    o.row_buffers, o.read_window},
                                 program)),
          l2(sim.add<L2>("l2", l2d, o.l2, program)),
          l3_reads(sim.link<L3RowRead>("bm->l3 reads", bm, l3, o.l3_link)),
          l3_data(sim.link<Beat>("l3->bm", l3, bm, o.l3_link)),
          l3_writes(sim.link<Beat>("bm->l3 writes", bm, l3, o.l3_link)),
          l2_writes(sim.crossing<L2RowWrite>("bm->l2 writes", bm, l2, o.l2_link)),
          l2_reads(sim.crossing<L2RowRead>("bm->l2 reads", bm, l2, o.l2_link)),
          l2_data(sim.crossing<L2RowData>("l2->bm", l2, bm, o.l2_link)),
          credits(sim.crossing<SlotCredit>("l2 credits", l2, bm, o.l2.slots, o.credit_sync, 2)) {
        l3.connect(l3_reads, l3_data, l3_writes);
        bm.connect(l3_reads, l3_data, l3_writes, l2_writes, l2_reads, l2_data, credits);
        l2.connect(l2_writes, l2_reads, l2_data, credits);
        sim.add_invariant("l2 slot credits", [this] { check_l2_slot_credits(bm, l2); });
    }

private:
    static Simulator& traced(Simulator& s, Tracer* t) {
        if (t) s.attach(*t);
        return s;
    }
};

}  // namespace sw::kpu::cycle::test
