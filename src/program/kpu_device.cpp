// ============================================================================
// src/program/kpu_device.cpp
// The machine side of the call ABI (#305 increment 3). See the header for R1-R4 and the
// deadlock-freedom argument they carry.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/orchestration/kpu_device.hpp>

#include <sw/kpu/program/characterize/characterization.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <exception>
#include <sstream>
#include <stdexcept>

namespace sw::kpu::orchestration {

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

void TensorStore::load_into(program::TileProgram& prog, const std::string& operand,
                            const std::string& tensor) const {
    if (!prog.has_operand(operand)) return;
    const std::vector<float>& src = values(tensor);
    auto& dst = prog.operand(operand).values;
    // SHAPES MUST AGREE. A truncating copy would let the loadable and the L0 program
    // disagree about a tensor while the run reported success -- which is the silent
    // wrong answer this whole layer is built to avoid.
    if (src.size() != dst.size())
        throw std::invalid_argument(
            "tensor store: tensor \"" + tensor + "\" holds " + std::to_string(src.size()) +
            " values, operand \"" + operand + "\" holds " + std::to_string(dst.size()));
    dst = src;
}

void TensorStore::store_from(const program::TileProgram& prog, const std::string& operand,
                             const std::string& tensor) {
    if (!prog.has_operand(operand)) return;
    const auto& src = prog.operand(operand).values;
    std::vector<float>& dst = values(tensor);
    if (src.size() != dst.size())
        throw std::invalid_argument(
            "tensor store: tensor \"" + tensor + "\" holds " + std::to_string(dst.size()) +
            " values, operand \"" + operand + "\" holds " + std::to_string(src.size()));
    dst = src;
}

std::map<std::string, std::string> operand_binding(const program::TileProgram& prog,
                                                   const loadable::Operator& op) {
    // The operands the program READS, in first-appearance order -- the same rule
    // driver::program_inputs uses, and for the same reason: "read at all", not "read before
    // written", because an in-place kernel reads and writes one operand and still needs an
    // input.
    std::vector<std::string> reads;
    std::set<std::string> seen;
    for (const program::TileOp& o : prog.ops())
        for (const program::TileCoord& c : o.inputs)
            if (seen.insert(c.operand).second) reads.push_back(c.operand);
    // ...and the ones it only writes.
    std::vector<std::string> writes;
    std::set<std::string> written;
    for (const program::TileOp& o : prog.ops())
        for (const program::TileCoord& c : o.outputs)
            if (!seen.count(c.operand) && written.insert(c.operand).second)
                writes.push_back(c.operand);

    if (reads.size() != op.inputs.size())
        throw std::invalid_argument(
            "operator \"" + op.name + "\": its program reads " +
            std::to_string(reads.size()) + " operand(s), the loadable names " +
            std::to_string(op.inputs.size()) + " input tensor(s)");
    if (writes.size() != op.outputs.size())
        throw std::invalid_argument(
            "operator \"" + op.name + "\": its program writes " +
            std::to_string(writes.size()) + " operand(s), the loadable names " +
            std::to_string(op.outputs.size()) + " output tensor(s)");

    std::map<std::string, std::string> out;
    for (std::size_t i = 0; i < reads.size(); ++i) out[reads[i]] = op.inputs[i];
    for (std::size_t i = 0; i < writes.size(); ++i) out[writes[i]] = op.outputs[i];
    return out;
}

// ---- the device -------------------------------------------------------------
KpuDevice::KpuDevice(const loadable::Loadable& l, program::platform::VirtualPlatform& platform,
                     TensorStore& tensors, ExecutionLevel level, DeviceOptions dopt)
    : l_(l), platform_(platform), tensors_(tensors), level_(level), dopt_(dopt) {
    capacity_ = static_cast<std::uint32_t>(platform_.deployment().device_view().l3_tiles);

    for (const loadable::TensorRef& t : l_.tensors)
        if (!tensors_.has(t.name)) tensors_.declare(t);

    // THE MANIFESTS, derived once, here, from the L0 programs -- so the orchestrator never
    // parses L0 (plan Q2). A program that does not parse or bind is not an error at load: the
    // operator is marked invalid, and the orchestrator refuses when it reaches it, exactly
    // where increment 2 did.
    for (std::size_t i = 0; i < l_.operators.size(); ++i) {
        const loadable::Operator& lop = l_.operators[i];
        Op o;
        o.manifest.index = static_cast<std::uint32_t>(i);
        try {
            o.prog = program::serialize::from_string(lop.l0_program);
            o.binding = operand_binding(o.prog, lop);
        } catch (const std::exception& e) {
            o.manifest.error = e.what();
            ops_.push_back(std::move(o));
            continue;
        }

        // Every tile the program READS, in first-appearance order, IN BOTH VOCABULARIES.
        // Reads, not writes: a tile the operator produces is not something to place -- the
        // launch creates it.
        //
        //   tensor key    the MACHINE's name -- residency persists across operators, so two
        //                 operators reading one tensor must agree on it
        //   operand key   the KERNEL's name -- what the executor compares against, equal to
        //                 the tensor key only by coincidence
        //
        // Handing the executor a tensor key was a live bug in increment 2; the device is now
        // the only place the translation happens.
        std::set<std::string> seen_operand, seen_tensor;
        for (const program::TileOp& op : o.prog.ops())
            for (const program::TileCoord& c : op.inputs) {
                const auto it = o.binding.find(c.operand);
                if (it == o.binding.end()) continue;
                const std::string okey = program::tile_key(c);
                if (!seen_operand.insert(okey).second) continue;
                const TileRef t{it->second, c.ti, c.tj};
                o.reads.push_back(ReadTile{t, okey});
                if (seen_tensor.insert(t.key()).second) o.manifest.reads.push_back(t);
            }
        o.manifest.peak_live_tiles =
            static_cast<std::uint32_t>(program::characterize::peak_live_tiles(o.prog));
        o.manifest.valid = true;
        ops_.push_back(std::move(o));
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

std::uint32_t KpuDevice::occupied_ablation() const {
    std::size_t n = held_.size();
    for (const Op& o : ops_)
        if (!o.completed) n += o.pending.size();
    return static_cast<std::uint32_t>(n);
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
            // has completed by the time the FENCE is seen -- except a RELEASE at last read,
            // which waits on a LAUNCH by design and is not what a FENCE orders against.
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
    if (dopt_.enforce_reservations) {
        // R1. A later operator never holds slots an earlier one still needs.
        for (std::uint32_t i = 0; i < j; ++i)
            if (!ops_[i].completed && !ops_[i].reserved) {
                refuse(d, CompletionStatus::RefusedInsufficientCredit,
                       RefusalCause::ReservationOutOfOrder,
                       "operator \"" + d.target + "\" asked to reserve before operator \"" +
                           l_.operators[i].name +
                           "\", which holds no reservation; reservations are granted in "
                           "operator order",
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
            refuse(d, CompletionStatus::RefusedInsufficientCredit,
                   RefusalCause::InsufficientCredit, why.str(), d.slots, free);
            return;
        }
    }
    o.reserved = true;
    o.reservation = d.slots;
    Completion c;
    c.descriptor_id = d.id;
    post(std::move(c));
}

void KpuDevice::do_place(const Descriptor& d) {
    std::uint32_t j = 0;
    if (!find_op(d.target, j)) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               "PLACE of " + d.tile.str() + " for \"" + d.target +
                   "\": the loadable has no such operator");
        return;
    }
    Op& o = ops_[j];
    const std::string key = d.tile.key();
    if (dopt_.enforce_reservations) {
        // R3.
        if (!o.reserved || o.completed) {
            refuse(d, CompletionStatus::RefusedInsufficientCredit, RefusalCause::NoReservation,
                   "PLACE of " + d.tile.str() + " for operator \"" + d.target +
                       "\", which holds no reservation");
            return;
        }
        // A PREFETCH -- placing for an operator while an earlier one has not launched --
        // occupies real slots during that earlier run, so it must fit in the reservation.
        // Placing for the NEXT operator to launch does not: at L-T1 its tiles flow through
        // its reserved slots inside its own run.
        if (static_cast<long>(j) != earliest_uncompleted() && !o.pending.count(key) &&
            o.pending.size() + 1 > o.reservation) {
            refuse(d, CompletionStatus::RefusedInsufficientCredit,
                   RefusalCause::ReservationExceeded,
                   "PLACE of " + d.tile.str() + " ahead of its turn would hold " +
                       std::to_string(o.pending.size() + 1) + " slot(s) for operator \"" +
                       d.target + "\", which reserved " + std::to_string(o.reservation),
                   static_cast<std::uint32_t>(o.pending.size() + 1), o.reservation);
            return;
        }
    } else if (capacity_ != 0 && !held_.count(key) && !o.pending.count(key) &&
               occupied_ablation() + 1 > capacity_) {
        // THE ABLATION: one slot per PLACE, nothing ordered. When it runs out the diagnosis
        // still names who holds the credits -- the latest OTHER operator with placed tiles --
        // because "it did not fit" sends the reader nowhere.
        std::uint32_t blocking = kNoOperator;
        for (std::size_t i = ops_.size(); i-- > 0;)
            if (i != j && !ops_[i].completed && !ops_[i].pending.empty()) {
                blocking = static_cast<std::uint32_t>(i);
                break;
            }
        std::ostringstream why;
        why << "PLACE of " << d.tile.str() << " for operator \"" << d.target
            << "\" needs 1 L3 slot, 0 available of " << capacity_;
        if (blocking != kNoOperator)
            why << ": " << ops_[blocking].pending.size() << " held by operator \""
                << l_.operators[blocking].name << "\", which cannot launch before \""
                << d.target << "\" completes";
        refuse(d, CompletionStatus::RefusedInsufficientCredit, RefusalCause::InsufficientCredit,
               why.str(), 1, 0, blocking);
        return;
    }
    o.pending.insert(key);
    o.pending_refs.emplace(key, d.tile);
    // A PLACE has no completion cycle of its own at L-T1 (#305 §6.2): the executor decides
    // when the leg happens, inside the launch. Reporting a number here would be inventing one.
    Completion c;
    c.descriptor_id = d.id;
    post(std::move(c));
}

void KpuDevice::do_release(const Descriptor& d) {
    const std::string key = d.tile.key();
    bool known = held_.count(key) != 0;
    for (const Op& o : ops_)
        if (!o.completed && o.pending.count(key)) known = true;
    if (!known) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported,
               "RELEASE of " + d.tile.str() + ": no credit is held for it");
        return;
    }
    if (d.flags & kReleaseAtLastRead) {
        // Deferred: the completion is posted after the next LAUNCH, because that is when the
        // credit comes back. See kReleaseAtLastRead.
        pending_release_.emplace(d.id, d.tile);
        return;
    }
    held_.erase(key);
    held_refs_.erase(key);
    for (Op& o : ops_)
        if (!o.completed) {
            o.pending.erase(key);
            o.pending_refs.erase(key);
        }
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

    // What this launch must leave behind: every held or placed tile it reads that is not
    // released at last read.
    std::set<std::string> releasing;
    for (const auto& [id, t] : pending_release_) releasing.insert(t.key());
    std::set<std::string> retained_tensor;
    for (const TileRef& t : o.manifest.reads) {
        const std::string k = t.key();
        if ((held_.count(k) || o.pending.count(k)) && !releasing.count(k))
            retained_tensor.insert(k);
    }

    if (dopt_.enforce_reservations) {
        if (!o.reserved) {
            refuse(d, CompletionStatus::RefusedInsufficientCredit, RefusalCause::NoReservation,
                   "LAUNCH of \"" + d.target + "\", which holds no reservation");
            return;
        }
        // R4: the reservation must be a SUFFICIENT bound for what this run will do. The device
        // checks it rather than trusting the orchestrator's arithmetic: a granted reservation
        // that ends in an in-run refusal is exactly the failure this increment removes.
        const std::uint32_t need = o.manifest.bound(retained_tensor.size());
        if (need > o.reservation) {
            refuse(d, CompletionStatus::RefusedInsufficientCredit,
                   RefusalCause::ReservationExceeded,
                   "operator \"" + d.target + "\" needs " + std::to_string(need) +
                       " L3 slot(s) to run (" + std::to_string(o.manifest.peak_live_tiles) +
                       " live + " + std::to_string(retained_tensor.size()) +
                       " retained), but reserved " + std::to_string(o.reservation),
                   need, o.reservation);
            return;
        }
    }

    // ---- what the executor is told, in the executor's spelling --------------
    // SEEDED: held before this launch, so they skip the DMA leg. A tile PLACEd for this
    // operator -- prefetched or not -- is NOT seeded: at L-T1 a PLACE has no leg of its own,
    // so the run that first reads it is where its DMA leg is charged (plan §3.5).
    std::set<std::string> seeded, retained, named;
    for (const ReadTile& r : o.reads) {
        const std::string k = r.tensor_tile.key();
        named.insert(k);
        if (held_.count(k)) seeded.insert(r.operand_key);
        if (retained_tensor.count(k)) retained.insert(r.operand_key);
    }
    // SLOTS THIS PROGRAM CANNOT NAME: held tiles it does not read, and every other operator's
    // claim -- its whole reservation, or under the ablation its placed tiles.
    std::size_t foreign = 0;
    for (const std::string& k : held_)
        if (!named.count(k)) ++foreign;
    for (std::size_t i = 0; i < ops_.size(); ++i) {
        if (i == j || ops_[i].completed) continue;
        foreign += dopt_.enforce_reservations
                       ? (ops_[i].reserved ? ops_[i].reservation : 0)
                       : ops_[i].pending.size();
    }

    const auto dev = platform_.deployment().device_view();
    RunOutcome outcome;
    try {
        program::TileProgram prog = o.prog;
        for (const auto& [operand, tensor] : o.binding) tensors_.load_into(prog, operand, tensor);
        const auto handle = platform_.load_program(std::move(prog));
        const auto snapshot = platform_.snapshot();
        const auto run = platform_.run(handle, level_, snapshot,
                                       program::Placement::single(dev.compute_tiles), nullptr,
                                       seeded, retained, foreign);
        outcome = run.outcome;
        for (const auto& [operand, tensor] : o.binding)
            tensors_.store_from(platform_.program(handle), operand, tensor);
    } catch (const std::exception& ex) {
        refuse(d, CompletionStatus::RefusedUnsupported, RefusalCause::Unsupported, ex.what());
        return;
    }

    Completion c;
    c.descriptor_id = d.id;
    c.timed = outcome.has_timing;
    c.cycles = outcome.makespan;
    post(std::move(c));
    outcomes_.push_back(std::move(outcome));

    // ---- the ledger after the run -------------------------------------------
    for (const TileRef& t : o.manifest.reads)
        if (retained_tensor.count(t.key())) {
            held_.insert(t.key());
            held_refs_.emplace(t.key(), t);
        }
    o.completed = true;
    o.reserved = false;
    o.reservation = 0;
    o.pending.clear();
    o.pending_refs.clear();

    // The credits released at last read came back inside this launch; their completions are
    // posted now, in descriptor order.
    for (const auto& [id, t] : pending_release_) {
        held_.erase(t.key());
        held_refs_.erase(t.key());
        Completion r;
        r.descriptor_id = id;
        r.released = {t};
        post(std::move(r));
    }
    pending_release_.clear();
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
