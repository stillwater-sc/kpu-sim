#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/l3_standin.hpp
// A stand-in for L3 on the BlockMover's side (#542, step 2a; moved from the tests for kpu-run's
// compute subsystem, #547): bytes at addresses, loaded directly (docs/plans/cycle-based-kpu-
// prototype.md §5.2: "the L3s start with A and B in the slots the program places them in ... or the
// same state loaded directly"). It serves the BlockMover's row reads in order, one beat of the bus a
// clock, and takes its writeback one beat a clock. Phase 1's L3Tile takes its place at the join
// (step 3), where the BlockMover's ports contend with the DMA engine's.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

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

}  // namespace sw::kpu::cycle
