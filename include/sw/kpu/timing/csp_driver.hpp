// ============================================================================
// include/sw/kpu/timing/csp_driver.hpp
// L-CA from the CSP program (docs/plans/csp-program-tile-sequencing.md §3.4, step 2; from the
// structured program, docs/plans/csp-language.md step 1c.2).
//
// The cycle-level executor used to run a ScheduleResult from a hand-written generator -- a
// second source of tile sequencing beside the CSP program. The driver removes it: each process
// of the CSP program gets its own actions, in program order, and the executor's own semantics
// are then CSP -- a process takes its next action when its input is present (tag CAM) and its
// output has a credit, a Load decomposes into DRAM bursts as it begins, a Call runs on the
// compute fabric for its tile function's MAC time.
//
// The actions come from either form of the program:
//   - the TRACE (a CspProgram): the flat action list, for small programs and the record;
//   - the STREAM (lang::ActionStream): pulled from the structured program as it runs, never
//     unrolled. The driver hands the executor actions while its backlog (actions handed over
//     and not yet taken) is under a window, so what it holds is bounded by the window, not by
//     the program. Nothing needs lookahead: an L3 entry is retired by the program's Release.
//
//   dma   Load  -> schedule_load   on the dma process's lanes (the device's DMA engines; which
//                                  lane is the executor's binding), in program order: a Load is
//                                  handed over only once the one before it holds its L3 credit,
//                                  so credits are taken in the program's order (no
//                                  hold-and-wait). The entry is l3_held: Moves out of it do not
//                                  consume it.
//         Store -> schedule_store  (the BlockMover ejects into the engine's store buffer, the
//                                  engine writes DRAM -- the executor's push-only store)
//   bm    Move  -> schedule_move;  Writeback -> schedule_writeback
//   str   Feed  -> schedule_feed;  Drain -> schedule_drain
//   cf    Call  -> schedule_matmul_compute (values computed at completion)
//   Release     -> schedule_release, for a residency a Load opened: the BlockMover retires the
//                  entry after the Moves the program issued before the Release. A result's
//                  residency (opened by a Writeback) is retired by its Store's ejection.
//
// This step runs explicit (matmul) programs: operands A, B and C, gemm with alpha 1. LU at
// L-CA needs its kernels as functional computes and in-place residencies; that is a later
// increment.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>
#include <sw/kpu/timing/concurrent_timing_executor.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <set>
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
        std::size_t actions = 0;            // the program's actions handed to the executor
        std::size_t peak_held = 0;          // most actions the driver and executor held at once
        program::TileProgram values;        // the source program's operands, as DRAM holds them
    };

    // window = 0: hand over the whole program at once (the trace's default, as step 2 did).
    static constexpr std::size_t kWholeProgram = 0;

    // The trace form: its operands carry the values.
    CspDriver(ConcurrentTimingExecutor& exec, const program::csp::CspProgram& p,
              std::size_t window = kWholeProgram)
        : exec_(exec), trace_(&p), inputs_(p.source), window_(window) {
        for (const auto& op : p.source.ops())
            if (op.kind != program::TileOpKind::Feed && op.kind != program::TileOpKind::Drain)
                check_call(op);
        check_machine(p.channel(program::csp::Chan::L3).capacity);
    }

    // The structured form: actions pulled from the program's stream; `inputs` supplies the
    // values of the operands it declares (the stream allocates none).
    CspDriver(ConcurrentTimingExecutor& exec, program::csp::lang::ActionStream& s,
              const program::TileProgram& inputs, std::size_t window = 256)
        : exec_(exec), stream_(&s), inputs_(inputs), window_(window) {
        const auto& declared = s.operands();
        for (const auto& name : declared.operand_order()) {
            if (!inputs.has_operand(name))
                throw std::invalid_argument("CspDriver: the program declares '" + name +
                                            "'; the inputs do not supply it");
            const auto& d = declared.operand(name);
            const auto& v = inputs.operand(name);
            if (d.rows != v.rows || d.cols != v.cols || d.tile_rows != v.tile_rows || d.tile_cols != v.tile_cols)
                throw std::invalid_argument("CspDriver: the inputs' '" + name + "' is not the shape the program declares");
        }
        check_machine(s.capacity());
    }

    Result run() {
        exec_.reset();
        // DRAM holds every input tile; the program's loads read them.
        for (const auto& name : inputs_.operand_order()) {
            if (name == "C") continue;
            const auto& t = inputs_.operand(name);
            for (program::Dim ti = 0; ti < t.n_tile_rows(); ++ti)
                for (program::Dim tj = 0; tj < t.n_tile_cols(); ++tj)
                    exec_.set_tile_payload(id(name, ti, tj), payload(t, ti, tj));
        }
        next_ = 0;
        loads_.clear();
        loaded_.clear();

        Result r;
        const Cycle max = exec_.config().max_cycles;
        std::size_t seen = 0;
        bool ended = false;
        bool waiting = false;           // a handed-over Load has not yet taken its credit
        TileID waiting_for{};
        auto done = [&] { return ended && loads_.empty() && !waiting && exec_.is_complete(); };
        while (exec_.current_cycle() < max) {
            // Pull the program's next actions while the backlog is under the window. Every
            // action's inputs come earlier in program order, so they are already handed over.
            while (!ended && (window_ == kWholeProgram || exec_.backlog() + loads_.size() < window_)) {
                if (!pull()) ended = true;
                else ++r.actions;
            }
            r.peak_held = std::max(r.peak_held, exec_.backlog() + loads_.size());
            // The dma process, in order: hand over the next Load once the last holds its credit.
            const auto& ev = exec_.events();
            for (; seen < ev.size(); ++seen)
                if (waiting && ev[seen].type == EventType::CREDIT_ACQUIRED &&
                    ev[seen].component_name.rfind("DMA", 0) == 0 && ev[seen].tile_id == waiting_for)
                    waiting = false;
            if (!waiting && !loads_.empty()) {
                const TileDescriptor l = loads_.front();
                loads_.pop_front();
                exec_.schedule_load(l);     // the engine (a lane of the dma process): the executor's
                waiting = true;
                waiting_for = l.tile_id;
            }
            if (done()) break;
            exec_.step();
            if (exec_.livelock_detected()) { r.livelock = true; break; }
        }
        r.cycles = exec_.current_cycle();
        r.completed = done();
        for (const auto& e : exec_.events()) {
            r.dram_loads += e.type == EventType::DMA_LOAD_COMPLETE;
            r.dram_stores += e.type == EventType::DMA_STORE_COMPLETE;
        }
        r.values = inputs_;
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
    const program::csp::CspProgram* trace_ = nullptr;
    program::csp::lang::ActionStream* stream_ = nullptr;
    const program::TileProgram& inputs_;
    std::size_t window_;
    std::map<std::string, Address> base_;
    std::size_t next_ = 0;                      // the trace's next action
    std::deque<TileDescriptor> loads_;          // the dma process's Loads not yet handed over
    std::set<TileID> loaded_;                   // tiles whose open residency a Load opened

    void check_machine(std::size_t l3) {
        for (const auto& name : inputs_.operand_order())
            if (name != "A" && name != "B" && name != "C")
                throw std::invalid_argument("CspDriver: operand '" + name + "': L-CA runs matmul programs "
                                            "(operands A, B, C) in this step");
        // The program's channel capacities are its target's (decision Q2): a program that needs
        // more L3 than the machine has, or was lowered for an unbounded one, is not this
        // machine's program. Level 1 is one flat L3: a machine with several is level 2's.
        if (exec_.l3_tiles() != 1)
            throw std::invalid_argument("CspDriver: the machine has " + std::to_string(exec_.l3_tiles()) +
                                        " L3 tiles; a level-1 program addresses one (level 2 distributes)");
        if (l3 == 0 || l3 > exec_.config().l3_buffer_count)
            throw std::invalid_argument("CspDriver: the program was lowered for an L3 of " +
                                        (l3 ? std::to_string(l3) + " tiles" : std::string("unbounded size")) +
                                        "; the machine's holds " + std::to_string(exec_.config().l3_buffer_count));
        // DRAM layout: each operand's tiles row-major, operands one after another.
        Address base = 0x100000;
        for (const auto& name : inputs_.operand_order()) {
            const auto& t = inputs_.operand(name);
            base_[name] = base;
            const Address bytes = static_cast<Address>(t.rows) * t.cols * 4;
            base = (base + bytes + 0xFFFF) & ~Address{0xFFFF};
        }
    }

    static void check_call(const program::TileOp& op) {
        if (op.kind != program::TileOpKind::MatMulAccum)
            throw std::invalid_argument(std::string("CspDriver: L0 op ") + program::to_string(op.kind) +
                                        ": L-CA runs matmul programs in this step");
        if (op.alpha != 1.0f)
            throw std::invalid_argument("CspDriver: gemm with alpha " + std::to_string(op.alpha) +
                                        ": the fabric's matmul computes alpha 1 in this step");
    }

    // The program's next action, handed to its process. false: the program has ended.
    bool pull() {
        using program::csp::Action;
        Action a;
        std::optional<program::TileOp> op;
        if (trace_) {
            if (next_ == trace_->actions.size()) return false;
            a = trace_->actions[next_++];
            if (a.kind == Action::Kind::Call) op = trace_->source.ops().at(a.l0_op);
        } else {
            auto e = stream_->next();
            if (!e) return false;
            a = e->action;
            op = std::move(e->op);
            if (a.kind == Action::Kind::Call) {
                if (!op) throw std::logic_error("CspDriver: the stream's Call carries no tile function");
                check_call(*op);
            }
        }
        TileDescriptor d = descriptor(a.tile);
        switch (a.kind) {
            case Action::Kind::Load:
                d.l3_held = true;
                loaded_.insert(d.tile_id);
                loads_.push_back(d);
                break;
            case Action::Kind::Move:
                d.l3_held = loaded_.count(d.tile_id) != 0;
                exec_.schedule_move(d, d.tile_id.matrix == isa::MatrixID::B);
                break;
            case Action::Kind::Feed:      exec_.schedule_feed(d); break;
            case Action::Kind::Drain:     exec_.schedule_drain(d); break;
            case Action::Kind::Writeback: exec_.schedule_writeback(d); break;
            case Action::Kind::Store:     exec_.schedule_store(d); break;
            case Action::Kind::Call: {
                ConcurrentTimingExecutor::MatMulComputeSpec spec;
                spec.a_tiles = {id(op->inputs.at(0))};
                spec.b_tiles = {id(op->inputs.at(1))};
                spec.accumulate = true;         // one k-slice; C stays in the fabric
                exec_.schedule_matmul_compute(descriptor(op->outputs.at(0)), spec);
                break;
            }
            case Action::Kind::Release:
                if (loaded_.erase(d.tile_id)) exec_.schedule_release(d);
                break;
        }
        return true;
    }

    static TileID id(const std::string& operand, program::Dim ti, program::Dim tj) {
        TileID t;
        t.matrix = operand == "A" ? isa::MatrixID::A : operand == "B" ? isa::MatrixID::B : isa::MatrixID::C;
        t.ti = ti;
        t.tj = tj;
        return t;
    }
    static TileID id(const program::TileCoord& c) { return id(c.operand, c.ti, c.tj); }

    TileDescriptor descriptor(const program::TileCoord& c) const {
        const auto& t = inputs_.operand(c.operand);
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
