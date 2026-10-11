// ============================================================================
// src/program/kpu_device.cpp
// The machine side of the call ABI (#305 increment 3), running CSP operators
// (kpu-run-csp-programs step 4d.2). See the header for R1-R4, the chain checks, and the
// deadlock-freedom argument they carry.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/orchestration/kpu_device.hpp>

#include <sw/kpu/program/csp/lang/walk.hpp>

#include <exception>
#include <sstream>
#include <stdexcept>

namespace sw::kpu::orchestration {

namespace lang = program::csp::lang;
using Kind = program::csp::Action::Kind;

// ---- the data plane ---------------------------------------------------------
void TensorStore::declare(const loadable::TensorRef& t) {
    std::uint64_t n = 1;
    for (std::uint64_t d : t.shape) n *= d;
    values_[t.name].assign(static_cast<std::size_t>(n), 0.0f);
    shapes_[t.name] = t.shape;
}

bool TensorStore::has(const std::string& name) const { return values_.count(name) != 0; }

std::vector<float>& TensorStore::values(const std::string& name) {
    const auto it = values_.find(name);
    if (it == values_.end())
        throw std::invalid_argument("tensor store: no tensor \"" + name + "\"");
    return it->second;
}

const std::vector<float>& TensorStore::values(const std::string& name) const {
    const auto it = values_.find(name);
    if (it == values_.end())
        throw std::invalid_argument("tensor store: no tensor \"" + name + "\"");
    return it->second;
}

namespace {

std::size_t elements(const program::TensorOperand& o) {
    return static_cast<std::size_t>(o.rows) * static_cast<std::size_t>(o.cols);
}

} // namespace

void TensorStore::load_into(program::TileProgram& prog, const std::string& operand,
                            const std::string& tensor) const {
    if (!prog.has_operand(operand)) return;
    const std::vector<float>& src = values(tensor);
    auto& dst = prog.operand(operand);
    // SIZES MUST AGREE. A truncating copy would let the loadable and the program disagree
    // about a tensor while the run reported success -- which is the silent wrong answer this
    // whole layer is built to avoid.
    if (src.size() != elements(dst))
        throw std::invalid_argument(
            "tensor store: tensor \"" + tensor + "\" holds " + std::to_string(src.size()) +
            " values, operand \"" + operand + "\" holds " + std::to_string(elements(dst)));
    dst.values = src;
}

void TensorStore::store_tile(const program::TileProgram& prog, const program::TileCoord& tile,
                             const std::string& tensor) {
    const auto& src = prog.operand(tile.operand);
    std::vector<float>& dst = values(tensor);
    if (src.values.size() != dst.size())
        throw std::invalid_argument(
            "tensor store: tensor \"" + tensor + "\" holds " + std::to_string(dst.size()) +
            " values, operand \"" + tile.operand + "\" holds " +
            std::to_string(src.values.size()));
    for (program::Dim r = src.row_begin(tile.ti); r < src.row_end(tile.ti); ++r)
        for (program::Dim c = src.col_begin(tile.tj); c < src.col_end(tile.tj); ++c) {
            const std::size_t at = static_cast<std::size_t>(r) * src.cols + c;
            dst[at] = src.values[at];
        }
}

std::map<std::string, std::string> operand_binding(const lang::Program& ast,
                                                   const loadable::Operator& op,
                                                   const loadable::Loadable& l) {
    // DECLARATION ORDER. `in` and `inout` operands are the operator's inputs -- an inout one
    // is read, so it needs a tensor to read -- and `out` operands its outputs.
    std::vector<const lang::Decl*> ins, outs;
    for (const lang::Decl& d : ast.decls) (d.io == "out" ? outs : ins).push_back(&d);

    if (ins.size() != op.inputs.size())
        throw std::invalid_argument(
            "operator \"" + op.name + "\": its program declares " + std::to_string(ins.size()) +
            " input operand(s) (in or inout), the loadable names " +
            std::to_string(op.inputs.size()) + " input tensor(s)");
    if (outs.size() != op.outputs.size())
        throw std::invalid_argument(
            "operator \"" + op.name + "\": its program declares " + std::to_string(outs.size()) +
            " output operand(s), the loadable names " + std::to_string(op.outputs.size()) +
            " output tensor(s)");

    auto render = [](const std::vector<std::uint64_t>& v) {
        std::string out;
        for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "x" : "") + std::to_string(v[i]);
        return out;
    };
    std::map<std::string, std::string> out;
    auto bind = [&](const lang::Decl& d, const std::string& tensor) {
        const loadable::TensorRef* t = nullptr;
        for (const loadable::TensorRef& r : l.tensors)
            if (r.name == tensor) t = &r;
        if (!t)
            throw std::invalid_argument("operator \"" + op.name + "\" names tensor \"" + tensor +
                                        "\", which the loadable does not declare");
        const auto rows = static_cast<std::uint64_t>(d.rows), cols = static_cast<std::uint64_t>(d.cols);
        const auto trows = static_cast<std::uint64_t>(d.tile_rows);
        const auto tcols = static_cast<std::uint64_t>(d.tile_cols);
        const std::vector<std::uint64_t> shape = d.is_vector ? std::vector<std::uint64_t>{rows}
                                                             : std::vector<std::uint64_t>{rows, cols};
        const std::vector<std::uint64_t> tile = d.is_vector ? std::vector<std::uint64_t>{trows}
                                                            : std::vector<std::uint64_t>{trows, tcols};
        if (t->shape != shape)
            throw std::invalid_argument("operator \"" + op.name + "\": operand " + d.name + " is " +
                                        render(shape) + ", tensor \"" + tensor + "\" is " +
                                        render(t->shape));
        // THE TILING TOO, when the tensor declares one: residency across operators names a
        // tile by its index, so two programs tiling one tensor differently would name
        // different regions with the same index.
        if (!t->tile_shape.empty() && t->tile_shape != tile)
            throw std::invalid_argument("operator \"" + op.name + "\": operand " + d.name +
                                        " is tiled " + render(tile) + ", tensor \"" + tensor +
                                        "\" is tiled " + render(t->tile_shape));
        out[d.name] = tensor;
    };
    for (std::size_t i = 0; i < ins.size(); ++i) bind(*ins[i], op.inputs[i]);
    for (std::size_t i = 0; i < outs.size(); ++i) bind(*outs[i], op.outputs[i]);
    return out;
}

// ---- the device -------------------------------------------------------------
KpuDevice::KpuDevice(const loadable::Loadable& l, program::platform::VirtualPlatform& platform,
                     TensorStore& tensors, ExecutionLevel level)
    : l_(l), platform_(platform), tensors_(tensors), level_(level) {
    capacity_ = static_cast<std::uint32_t>(platform_.deployment().device_view().l3_tiles);

    for (const loadable::TensorRef& t : l_.tensors)
        if (!tensors_.has(t.name)) tensors_.declare(t);

    // THE MANIFESTS, derived once, here, from the CSP programs -- so the orchestrator never
    // parses one (plan Q2). A program that does not parse, walk or bind is not an error at
    // load: the operator is marked invalid, and the orchestrator refuses when it reaches it.
    for (std::size_t i = 0; i < l_.operators.size(); ++i) {
        const loadable::Operator& lop = l_.operators[i];
        Op o;
        o.manifest.index = static_cast<std::uint32_t>(i);
        try {
            o.ast = lang::parse(lop.csp_program);
            o.binding = operand_binding(o.ast, lop, l_);
            o.manifest.l3_slots = static_cast<std::uint32_t>(o.ast.l3);

            // One walk of the program, in its own order, for what crosses its boundary: the
            // tiles it brings into L3 (Load, Inherit), the ones it hands on (Retain), and the
            // ones it writes to DRAM (Store). In both vocabularies -- the tensor tile is what
            // residency across operators is keyed by, the operand tile what this program's
            // values are indexed by.
            std::set<std::string> seen_read, seen_store;
            auto bound = [&](const program::TileCoord& c) {
                return BoundTile{TileRef{o.binding.at(c.operand), c.ti, c.tj}, c};
            };
            lang::ActionStream s(o.ast);
            while (auto e = s.next()) {
                const auto& a = e->action;
                switch (a.kind) {
                    case Kind::Load:
                    case Kind::Inherit: {
                        const BoundTile b = bound(a.tile);
                        if (seen_read.insert(b.tensor_tile.key()).second)
                            o.manifest.reads.push_back(b.tensor_tile);
                        if (a.kind == Kind::Inherit) {
                            o.inherits.push_back(b);
                            o.manifest.inherits.push_back(b.tensor_tile);
                        }
                        break;
                    }
                    case Kind::Retain: {
                        const BoundTile b = bound(a.tile);
                        o.retains.push_back(b);
                        o.manifest.retains.push_back(b.tensor_tile);
                        break;
                    }
                    case Kind::Store: {
                        const BoundTile b = bound(a.tile);
                        if (seen_store.insert(b.tensor_tile.key()).second) o.stores.push_back(b);
                        break;
                    }
                    default: break;
                }
            }
        } catch (const std::exception& e) {
            o.manifest.error = "operator \"" + lop.name + "\": " + e.what();
            ops_.push_back(std::move(o));
            continue;
        }
        o.manifest.valid = true;
        ops_.push_back(std::move(o));
    }
    check_chain();
}

// THE CHAIN: every inherit has a retainer, every retain a claimant, and nothing reads a stale
// DRAM copy of a tile held in between. See the header.
void KpuDevice::check_chain() {
    struct Holder {
        std::size_t op;
        bool dirty;                         // DRAM does not hold its value
        program::TileCoord operand_tile;    // in the retaining program's vocabulary
    };
    std::map<std::string, Holder> held;     // by tensor key
    auto invalidate = [&](std::size_t k, const std::string& why) {
        if (!ops_[k].manifest.valid) return;
        ops_[k].manifest.valid = false;
        ops_[k].manifest.error = "operator \"" + l_.operators[k].name + "\" " + why;
    };
    auto decl_of = [&](std::size_t k, const std::string& operand) -> const lang::Decl& {
        for (const lang::Decl& d : ops_[k].ast.decls)
            if (d.name == operand) return d;
        throw std::logic_error("no declaration of " + operand);
    };

    std::size_t k = 0;
    for (; k < ops_.size() && ops_[k].manifest.valid; ++k) {
        Op& o = ops_[k];
        // What this program writes and stores, by operand tile key.
        std::set<std::string> written, stored;
        std::map<std::string, TileRef> loaded;          // by tensor key
        lang::ActionStream s(o.ast);
        while (auto e = s.next()) {
            const auto& a = e->action;
            if (a.kind == Kind::Writeback) written.insert(program::tile_key(a.tile));
            if (a.kind == Kind::Call && e->op)
                for (const auto& c : e->op->outputs) written.insert(program::tile_key(c));
            if (a.kind == Kind::Store) stored.insert(program::tile_key(a.tile));
            if (a.kind == Kind::Load) {
                const TileRef t{o.binding.at(a.tile.operand), a.tile.ti, a.tile.tj};
                loaded.emplace(t.key(), t);
            }
        }

        std::map<std::string, bool> inherited_dirty;
        for (const BoundTile& b : o.inherits) {
            const std::string key = b.tensor_tile.key();
            const auto it = held.find(key);
            if (it == held.end()) {
                invalidate(k, "inherits " + b.tensor_tile.str() + " (its " + b.operand_tile.to_string() +
                                  "), which no earlier operator retains");
                break;
            }
            const lang::Decl& mine = decl_of(k, b.operand_tile.operand);
            const lang::Decl& theirs = decl_of(it->second.op, it->second.operand_tile.operand);
            if (mine.tile_rows != theirs.tile_rows || mine.tile_cols != theirs.tile_cols) {
                invalidate(k, "inherits " + b.tensor_tile.str() + " tiled " + std::to_string(mine.tile_rows) +
                                  "x" + std::to_string(mine.tile_cols) + ", which operator \"" +
                                  l_.operators[it->second.op].name + "\" retains tiled " +
                                  std::to_string(theirs.tile_rows) + "x" + std::to_string(theirs.tile_cols));
                break;
            }
            inherited_dirty[key] = it->second.dirty;
            held.erase(it);
        }
        if (!o.manifest.valid) break;

        for (const auto& [key, h] : held)
            if (h.dirty && loaded.count(key)) {
                invalidate(k, "loads " + loaded.at(key).str() + " from DRAM, which operator \"" +
                                  l_.operators[h.op].name +
                                  "\" retains and has not stored: DRAM does not hold its value");
                break;
            }
        if (!o.manifest.valid) break;

        for (const BoundTile& b : o.retains) {
            const std::string key = b.tensor_tile.key();
            const auto it = held.find(key);
            if (it != held.end()) {
                invalidate(k, "retains " + b.tensor_tile.str() + ", which operator \"" +
                                  l_.operators[it->second.op].name + "\" already retains");
                break;
            }
            const std::string okey = program::tile_key(b.operand_tile);
            const bool was_dirty = inherited_dirty.count(key) && inherited_dirty.at(key);
            const bool dirty = !stored.count(okey) && (written.count(okey) || was_dirty);
            held.emplace(key, Holder{k, dirty, b.operand_tile});
        }
        if (!o.manifest.valid) break;
    }
    // A retained tile nothing claims. Only when the chain was checked to its end: an invalid
    // operator stops the run before the claimant would have been reached anyway.
    if (k == ops_.size())
        for (const auto& [key, h] : held) {
            const Op& o = ops_[h.op];
            for (const BoundTile& b : o.retains)
                if (b.tensor_tile.key() == key) {
                    invalidate(h.op, "retains " + b.tensor_tile.str() +
                                         ", which no later operator inherits: a retained tile need "
                                         "not be stored, so a later operator must claim it");
                    break;
                }
        }
}

bool KpuDevice::pop_completion(Completion& out) {
    if (completions_.empty()) return false;
    out = std::move(completions_.front());
    completions_.pop_front();
    return true;
}

StatusSnapshot KpuDevice::status() const {
    StatusSnapshot s;
    s.l3_capacity = capacity_;
    s.held = static_cast<std::uint32_t>(held_.size());
    s.operators = operator_count();
    for (const Op& o : ops_) {
        if (o.reserved && !o.completed) s.reserved += o.reservation;
        if (o.completed) ++s.completed;
    }
    return s;
}

std::uint32_t KpuDevice::free_slots() const {
    const StatusSnapshot s = status();
    return s.credits_free();
}

long KpuDevice::earliest_uncompleted() const {
    for (std::size_t i = 0; i < ops_.size(); ++i)
        if (!ops_[i].completed) return static_cast<long>(i);
    return -1;
}

bool KpuDevice::find_op(const std::string& name, std::uint32_t& out) const {
    for (std::size_t i = 0; i < l_.operators.size(); ++i)
        if (l_.operators[i].name == name) {
            out = static_cast<std::uint32_t>(i);
            return true;
        }
    return false;
}

void KpuDevice::refuse(const Descriptor& d, CompletionStatus st, RefusalCause cause,
                       std::string why, std::uint32_t needed, std::uint32_t available,
                       std::uint32_t blocking) {
    Completion c;
    c.descriptor_id = d.id;
    c.status = st;
    c.cause = cause;
    c.diagnosis = std::move(why);
    c.needed = needed;
    c.available = available;
    c.capacity = capacity_;
    c.blocking_op = blocking;
    post(std::move(c));
}

void KpuDevice::submit(const Descriptor& d) {
    switch (d.kind) {
        case DescriptorKind::Reserve: do_reserve(d); return;
        case DescriptorKind::Place:   do_place(d);   return;
        case DescriptorKind::Release: do_release(d); return;
        case DescriptorKind::Launch:  do_launch(d);  return;
        case DescriptorKind::Fence: {
            // Every descriptor is serviced when it is submitted, so everything before a FENCE
            // has completed by the time the FENCE is seen.
            Completion c;
            c.descriptor_id = d.id;
            post(std::move(c));
            return;
        }
        case DescriptorKind::Configure:
            refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
                   "CONFIGURE \"" + d.target +
                       "\": no programmable compute tile is modelled before #305 increment 5");
            return;
    }
    // A kind byte outside the vocabulary -- possible on the wire, where decode cannot know what
    // a guest meant. EVERY descriptor gets exactly one completion, so it is refused by name
    // rather than dropped, which would surface as a misleading "no completion" protocol error.
    refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
           "descriptor " + std::to_string(d.id) + ": kind " +
               std::to_string(static_cast<unsigned>(d.kind)) + " is not in the vocabulary");
}

void KpuDevice::do_reserve(const Descriptor& d) {
    std::uint32_t j = 0;
    if (!find_op(d.target, j)) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               "RESERVE for \"" + d.target + "\": the loadable has no such operator");
        return;
    }
    Op& o = ops_[j];
    if (!o.manifest.valid) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               o.manifest.error);
        return;
    }
    if (o.completed || o.reserved) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               "operator \"" + d.target + "\" " +
                   (o.completed ? "has already completed" : "already holds a reservation"));
        return;
    }
    // R1. A later operator never holds slots an earlier one still needs.
    for (std::uint32_t i = 0; i < j; ++i)
        if (!ops_[i].completed && !ops_[i].reserved) {
            refuse(d, CompletionStatus::RefusedInsufficientCredit,
                   RefusalCause::ReservationOutOfOrder,
                   "operator \"" + d.target + "\" asked to reserve before operator \"" +
                       l_.operators[i].name +
                       "\", which holds no reservation; reservations are granted in operator order",
                   d.slots, free_slots(), i);
            return;
        }
    // R2. All or none, and never queued.
    const std::uint32_t free = free_slots();
    if (capacity_ != 0 && d.slots > free) {
        const StatusSnapshot s = status();
        std::ostringstream why;
        why << "operator \"" << d.target << "\" needs " << d.slots
            << " L3 slot(s) reserved, " << free << " available of " << capacity_ << " ("
            << s.held << " held, " << s.reserved << " reserved by other operators)";
        refuse(d, CompletionStatus::RefusedInsufficientCredit, RefusalCause::InsufficientCredit,
               why.str(), d.slots, free);
        return;
    }
    o.reserved = true;
    o.reservation = d.slots;
    Completion c;
    c.descriptor_id = d.id;
    post(std::move(c));
}

void KpuDevice::do_place(const Descriptor& d) {
    // R3. A program loads its own tiles; a PLACE would be a DMA that no program sequences.
    refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
           "PLACE of " + d.tile.str() + " for \"" + d.target +
               "\": an operator's CSP program loads its own tiles, so a PLACE would be a DMA no "
               "program sequences (PLACE returns with L-T2, #283)");
}

void KpuDevice::do_release(const Descriptor& d) {
    const std::string key = d.tile.key();
    if (d.flags & kReleaseAtLastRead) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               "RELEASE of " + d.tile.str() +
                   " at last read: an operator's CSP program releases the tiles it reads itself");
        return;
    }
    if (!held_.count(key)) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               "RELEASE of " + d.tile.str() + ": no credit is held for it");
        return;
    }
    // Dropping a retained tile before its claimant runs: the claimant's LAUNCH then refuses,
    // naming the tile it inherits and nobody holds.
    held_.erase(key);
    held_refs_.erase(key);
    held_values_.erase(key);
    Completion c;
    c.descriptor_id = d.id;
    c.released = {d.tile};
    post(std::move(c));
}

void KpuDevice::do_launch(const Descriptor& d) {
    std::uint32_t j = 0;
    if (!find_op(d.target, j)) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               "LAUNCH of \"" + d.target + "\": the loadable has no such operator");
        return;
    }
    Op& o = ops_[j];
    if (!o.manifest.valid) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               o.manifest.error);
        return;
    }
    // R4: operators launch in order. Increment 4's guest may be wrong about that; the device
    // is where it is caught.
    const long e = earliest_uncompleted();
    if (static_cast<long>(j) != e) {
        const std::uint32_t blocking = e < 0 ? kNoOperator : static_cast<std::uint32_t>(e);
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::ReservationOutOfOrder,
               "LAUNCH of \"" + d.target + "\" " +
                   (e < 0 ? std::string("after every operator completed")
                          : "before operator \"" + l_.operators[blocking].name +
                                "\" completed; operators launch in order"),
               0, 0, blocking);
        return;
    }
    if (!o.reserved) {
        refuse(d, CompletionStatus::RefusedInsufficientCredit, RefusalCause::NoReservation,
               "LAUNCH of \"" + d.target + "\", which holds no reservation");
        return;
    }
    // R4: what it inherits is held. The chain check proved an earlier operator retains it; a
    // RELEASE since then is what can make this fail.
    for (const BoundTile& b : o.inherits)
        if (!held_.count(b.tensor_tile.key())) {
            refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
                   "operator \"" + d.target + "\" inherits " + b.tensor_tile.str() +
                       ", which is not held: it was released after the operator that retained it");
            return;
        }
    // R4: the reservation must be a SUFFICIENT bound for what this run will do. The device
    // checks it rather than trusting the orchestrator's arithmetic: a granted reservation that
    // ends in an in-run refusal is exactly the failure increment 3 removed.
    const std::uint32_t need = o.manifest.bound();
    if (need > o.reservation) {
        refuse(d, CompletionStatus::RefusedInsufficientCredit, RefusalCause::ReservationExceeded,
               "operator \"" + d.target + "\" needs " + std::to_string(need) +
                   " L3 slot(s) to run (its program's L3 is " + std::to_string(o.manifest.l3_slots) +
                   ", " + std::to_string(o.inherits.size()) + " of them inherited and held), but reserved " +
                   std::to_string(o.reservation),
               need, o.reservation);
        return;
    }

    CspLevelOutcome outcome;
    try {
        // THE INPUTS: DRAM, as the tensors hold it, with each inherited tile's L3 value laid
        // over its operand -- an inherited tile enters the run from L3, and its value there need
        // not be in DRAM (a retained tile need not be stored).
        program::TileProgram inputs = lang::declared_operands(o.ast);
        for (const auto& [operand, tensor] : o.binding) tensors_.load_into(inputs, operand, tensor);
        for (const BoundTile& b : o.inherits) {
            auto& op = inputs.operand(b.operand_tile.operand);
            const std::vector<float>& v = held_values_.at(b.tensor_tile.key());
            std::size_t at = 0;
            for (program::Dim r = op.row_begin(b.operand_tile.ti); r < op.row_end(b.operand_tile.ti); ++r)
                for (program::Dim c = op.col_begin(b.operand_tile.tj); c < op.col_end(b.operand_tile.tj); ++c)
                    op.at(r, c) = v.at(at++);
        }
        const auto handle = platform_.load_csp(o.ast, std::move(inputs));
        auto run = platform_.run_csp(handle, level_);
        if (run.outcome.skipped) throw std::runtime_error(*run.outcome.skipped);
        outcome = std::move(run.outcome);
    } catch (const std::exception& ex) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               "operator \"" + d.target + "\": " + ex.what());
        return;
    }

    // ---- the data plane after the run ---------------------------------------
    // DRAM changes only where the program STORED. The level's result is DRAM with the retained
    // tiles' L3 values laid over it, so copying whole operands back would put a retained,
    // unstored tile's value in DRAM -- a write the program never made.
    for (const BoundTile& b : o.stores) tensors_.store_tile(outcome.values, b.operand_tile, b.tensor_tile.tensor);
    // ...and the ledger: what it inherited is consumed, what it retained is held, with its value.
    for (const BoundTile& b : o.inherits) {
        held_.erase(b.tensor_tile.key());
        held_refs_.erase(b.tensor_tile.key());
        held_values_.erase(b.tensor_tile.key());
    }
    for (const BoundTile& b : o.retains) {
        const auto& op = outcome.values.operand(b.operand_tile.operand);
        std::vector<float> v;
        for (program::Dim r = op.row_begin(b.operand_tile.ti); r < op.row_end(b.operand_tile.ti); ++r)
            for (program::Dim c = op.col_begin(b.operand_tile.tj); c < op.col_end(b.operand_tile.tj); ++c)
                v.push_back(op.at(r, c));
        const std::string key = b.tensor_tile.key();
        held_.insert(key);
        held_refs_.emplace(key, b.tensor_tile);
        held_values_[key] = std::move(v);
    }
    o.completed = true;
    o.reserved = false;
    o.reservation = 0;

    Completion c;
    c.descriptor_id = d.id;
    c.timed = outcome.has_timing;
    c.cycles = outcome.makespan;
    post(std::move(c));
    outcomes_.push_back(std::move(outcome));
}

// ---- the direct transport ---------------------------------------------------
void DirectPort::submit(const Descriptor& d) { dev_.submit(d); }
bool DirectPort::poll_completion(Completion& out) { return dev_.pop_completion(out); }
StatusSnapshot DirectPort::read_status() { return dev_.status(); }
bool DirectPort::is_resident(const TileRef& t) { return dev_.is_resident(t); }
std::uint32_t DirectPort::operator_count() { return dev_.operator_count(); }
OperatorManifest DirectPort::manifest(std::uint32_t op) { return dev_.manifest(op); }
std::vector<program::platform::ResourceName> DirectPort::inventory() {
    return program::platform::ResourceMap(dev_.deployment()).enumerate();
}

} // namespace sw::kpu::orchestration
