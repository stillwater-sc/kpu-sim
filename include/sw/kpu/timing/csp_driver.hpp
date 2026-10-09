// ============================================================================
// include/sw/kpu/timing/csp_driver.hpp
// L-CA from the CSP program (docs/plans/csp-program-tile-sequencing.md §3.4, step 2).
//
// The cycle-level executor used to run a ScheduleResult from a hand-written generator -- a
// second source of tile sequencing beside the CSP program. The driver removes it: each process
// of the CSP program gets its own actions, in program order, and the executor's own semantics
// are then CSP -- a process takes its next action when its input is present (tag CAM) and its
// output has a credit, a Load decomposes into DRAM bursts as it begins, a Call runs on the
// compute fabric for its tile function's MAC time.
//
//   dma   Load  -> schedule_load   on the dma process's lanes (the device's DMA engines; which
//                                  lane is the executor's binding), in program order: a Load is
//                                  handed over only once the one before
//                                  it holds its L3 credit, so credits are taken in the program's
//                                  order (no hold-and-wait). Its residency's consumer count
//                                  rides along (TileDescriptor::l3_consumers): the L3 entry is
//                                  seeded with it and frees after the last consumer.
//         Store -> schedule_store  (the BlockMover ejects into the engine's store buffer, the
//                                  engine writes DRAM -- the executor's push-only store)
//   bm    Move  -> schedule_move;  Writeback -> schedule_writeback
//   str   Feed  -> schedule_feed;  Drain -> schedule_drain
//   cf    Call  -> schedule_matmul_compute (values computed at completion)
//   Release     -> nothing to hand over: the slot frees when its last consumer completes
//
// This step runs explicit (matmul) programs: operands A, B and C. LU at L-CA needs its kernels
// as functional computes and in-place residencies; that is the next increment.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/timing/concurrent_timing_executor.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::timing {

class CspDriver {
public:
    struct Result {
        bool completed = false;
        bool livelock = false;
        Cycle cycles = 0;
        std::size_t dram_loads = 0;         // DMA_LOAD_COMPLETE events: DRAM reads of whole tiles
        std::size_t dram_stores = 0;
        program::TileProgram values;        // the source program's operands, as DRAM holds them
    };

    CspDriver(ConcurrentTimingExecutor& exec, const program::csp::CspProgram& p) : exec_(exec), p_(p) {
        using program::csp::Action;
        for (const auto& name : p.source.operand_order())
            if (name != "A" && name != "B" && name != "C")
                throw std::invalid_argument("CspDriver: operand '" + name + "': L-CA runs matmul programs "
                                            "(operands A, B, C) in this step");
        for (const auto& op : p.source.ops())
            if (op.kind != program::TileOpKind::Feed && op.kind != program::TileOpKind::Drain &&
                op.kind != program::TileOpKind::MatMulAccum)
                throw std::invalid_argument(std::string("CspDriver: L0 op ") + program::to_string(op.kind) +
                                            ": L-CA runs matmul programs in this step");
        // The program's channel capacities are its target's (decision Q2): a program that needs
        // more L3 than the machine has, or was lowered for an unbounded one, is not this
        // machine's program. Level 1 is one flat L3: a machine with several is level 2's.
        if (exec.l3_tiles() != 1)
            throw std::invalid_argument("CspDriver: the machine has " + std::to_string(exec.l3_tiles()) +
                                        " L3 tiles; a level-1 program addresses one (level 2 distributes)");
        const std::size_t l3 = p.channel(program::csp::Chan::L3).capacity;
        if (l3 == 0 || l3 > exec.config().l3_buffer_count)
            throw std::invalid_argument("CspDriver: the program was lowered for an L3 of " +
                                        (l3 ? std::to_string(l3) + " tiles" : std::string("unbounded size")) +
                                        "; the machine's holds " + std::to_string(exec.config().l3_buffer_count));
        // DRAM layout: each operand's tiles row-major, operands one after another.
        Address base = 0x100000;
        for (const auto& name : p.source.operand_order()) {
            const auto& t = p.source.operand(name);
            base_[name] = base;
            const Address bytes = static_cast<Address>(t.rows) * t.cols * 4;
            base = (base + bytes + 0xFFFF) & ~Address{0xFFFF};
        }
    }

    Result run() {
        using program::csp::Action;
        exec_.reset();
        // DRAM holds every input tile; the program's loads read them.
        for (const auto& name : p_.source.operand_order()) {
            if (name == "C") continue;
            const auto& t = p_.source.operand(name);
            for (program::Dim ti = 0; ti < t.n_tile_rows(); ++ti)
                for (program::Dim tj = 0; tj < t.n_tile_cols(); ++tj)
                    exec_.set_tile_payload(id(name, ti, tj), payload(t, ti, tj));
        }
        // Every process's actions in program order. Loads wait in the dma process's queue.
        for (const Action& a : p_.actions) {
            const TileDescriptor d = descriptor(a.tile);
            switch (a.kind) {
                case Action::Kind::Load: {
                    TileDescriptor l = d;
                    l.l3_consumers = static_cast<uint32_t>(p_.residencies.at(a.residency).consumers);
                    loads_.push_back(l);
                    break;
                }
                case Action::Kind::Move:
                    exec_.schedule_move(d, d.tile_id.matrix == isa::MatrixID::B);
                    break;
                case Action::Kind::Feed:      exec_.schedule_feed(d); break;
                case Action::Kind::Drain:     exec_.schedule_drain(d); break;
                case Action::Kind::Writeback: exec_.schedule_writeback(d); break;
                case Action::Kind::Store:     exec_.schedule_store(d); break;
                case Action::Kind::Call: {
                    const auto& op = p_.source.ops().at(a.l0_op);
                    ConcurrentTimingExecutor::MatMulComputeSpec spec;
                    spec.a_tiles = {id(op.inputs.at(0))};
                    spec.b_tiles = {id(op.inputs.at(1))};
                    spec.accumulate = true;         // one k-slice; C stays in the fabric
                    exec_.schedule_matmul_compute(descriptor(op.outputs.at(0)), spec);
                    break;
                }
                case Action::Kind::Release: break;
            }
        }

        Result r;
        const Cycle max = exec_.config().max_cycles;
        std::size_t next_load = 0, seen = 0;
        bool waiting = false;           // a handed-over Load has not yet taken its credit
        TileID waiting_for{};
        while (exec_.current_cycle() < max) {
            // The dma process, in order: hand over the next Load once the last holds its credit.
            const auto& ev = exec_.events();
            for (; seen < ev.size(); ++seen)
                if (waiting && ev[seen].type == EventType::CREDIT_ACQUIRED &&
                    ev[seen].component_name.rfind("DMA", 0) == 0 && ev[seen].tile_id == waiting_for)
                    waiting = false;
            if (!waiting && next_load < loads_.size()) {
                const TileDescriptor& l = loads_[next_load];
                exec_.schedule_load(l);     // the engine (a lane of the dma process): the executor's
                waiting = true;
                waiting_for = l.tile_id;
                ++next_load;
            }
            if (next_load == loads_.size() && !waiting && exec_.is_complete()) break;
            exec_.step();
            if (exec_.livelock_detected()) { r.livelock = true; break; }
        }
        r.cycles = exec_.current_cycle();
        r.completed = next_load == loads_.size() && exec_.is_complete();
        for (const auto& e : exec_.events()) {
            r.dram_loads += e.type == EventType::DMA_LOAD_COMPLETE;
            r.dram_stores += e.type == EventType::DMA_STORE_COMPLETE;
        }
        r.values = p_.source;
        if (r.completed) {
            auto& C = r.values.operand("C");
            for (program::Dim ti = 0; ti < C.n_tile_rows(); ++ti)
                for (program::Dim tj = 0; tj < C.n_tile_cols(); ++tj) {
                    const auto& v = exec_.tile_payload_at(MemoryLevel::DRAM, id("C", ti, tj)).values;
                    std::size_t n = 0;
                    for (program::Dim row = C.row_begin(ti); row < C.row_end(ti); ++row)
                        for (program::Dim col = C.col_begin(tj); col < C.col_end(tj); ++col) C.at(row, col) = v.at(n++);
                }
        }
        return r;
    }

private:
    ConcurrentTimingExecutor& exec_;
    const program::csp::CspProgram& p_;
    std::map<std::string, Address> base_;
    std::deque<TileDescriptor> loads_;

    static TileID id(const std::string& operand, program::Dim ti, program::Dim tj) {
        TileID t;
        t.matrix = operand == "A" ? isa::MatrixID::A : operand == "B" ? isa::MatrixID::B : isa::MatrixID::C;
        t.ti = ti;
        t.tj = tj;
        return t;
    }
    static TileID id(const program::TileCoord& c) { return id(c.operand, c.ti, c.tj); }

    TileDescriptor descriptor(const program::TileCoord& c) const {
        const auto& t = p_.source.operand(c.operand);
        TileDescriptor d;
        d.tile_id = id(c);
        d.height = t.row_end(c.ti) - t.row_begin(c.ti);
        d.width = t.col_end(c.tj) - t.col_begin(c.tj);
        d.element_size = 4;
        d.size_bytes = d.height * d.width * 4;
        d.matrix_base_address = base_.at(c.operand);
        d.dram_address = base_.at(c.operand) +
                         (static_cast<Address>(c.ti) * t.n_tile_cols() + c.tj) * t.tile_rows * t.tile_cols * 4;
        return d;
    }

    static TilePayload payload(const program::TensorOperand& t, program::Dim ti, program::Dim tj) {
        TilePayload p;
        p.rows = t.row_end(ti) - t.row_begin(ti);
        p.cols = t.col_end(tj) - t.col_begin(tj);
        for (program::Dim r = t.row_begin(ti); r < t.row_end(ti); ++r)
            for (program::Dim c = t.col_begin(tj); c < t.col_end(tj); ++c) p.values.push_back(t.at(r, c));
        return p;
    }
};

}  // namespace sw::kpu::timing
