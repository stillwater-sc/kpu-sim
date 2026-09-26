// ============================================================================
// src/program/orchestrator.cpp
// The deciding orchestrator (#305 increment 2). See the header for what it may and may
// not touch, and for why program-order acquisition is inherited rather than invented.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/orchestration/orchestrator.hpp>

#include <sw/kpu/program/platform/digest.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <set>
#include <sstream>
#include <stdexcept>

namespace sw::kpu::orchestration {

// ---- descriptor and completion rendering ------------------------------------
const char* to_string(DescriptorKind k) {
    switch (k) {
        case DescriptorKind::Place:     return "PLACE";
        case DescriptorKind::Release:   return "RELEASE";
        case DescriptorKind::Configure: return "CONFIGURE";
        case DescriptorKind::Launch:    return "LAUNCH";
        case DescriptorKind::Fence:     return "FENCE";
    }
    return "?";
}

const char* to_string(CompletionStatus s) {
    switch (s) {
        case CompletionStatus::Done:                      return "done";
        case CompletionStatus::RefusedInsufficientCredit: return "refused:credit";
        case CompletionStatus::RefusedUnsupported:        return "refused:unsupported";
    }
    return "?";
}

std::string Descriptor::str() const {
    std::string out = std::to_string(id) + " " + to_string(kind);
    switch (kind) {
        case DescriptorKind::Place:
            out += " " + tile.str() + " " + program::to_string(leg) + " -> " +
                   program::platform::format(resource);
            break;
        case DescriptorKind::Release:
            out += " " + tile.str();
            break;
        case DescriptorKind::Configure:
        case DescriptorKind::Launch:
            out += " " + target + " cf[" + std::to_string(compute_tile) + "]";
            break;
        case DescriptorKind::Fence:
            out += " after " + std::to_string(wait_for);
            break;
    }
    return out;
}

std::string Completion::str() const {
    std::string out = std::to_string(descriptor_id) + " " + to_string(status);
    // "not timed" and "zero cycles" are different claims, and the rendering keeps them
    // apart for the same reason RunOutcome::has_timing does.
    out += timed ? " cycles=" + std::to_string(cycles) : " cycles=not-modelled";
    for (const TileRef& t : released) out += " released:" + t.str();
    if (!diagnosis.empty()) out += " (" + diagnosis + ")";
    return out;
}

std::string DescriptorTrace::canonical_bytes() const {
    std::string out;
    for (const Descriptor& d : issued) out += d.str() + "\n";
    out += "--\n";
    for (const Completion& c : completions) out += c.str() + "\n";
    return out;
}

std::string DescriptorTrace::digest() const {
    return program::platform::digest_of(canonical_bytes());
}

std::size_t OrchestrationResult::dma_transfers() const {
    std::size_t n = 0;
    for (const RunOutcome& r : per_operator) {
        if (!r.stats) continue;
        const auto it = r.stats->hop_transfers.find(program::Hop::DmaDramToL3);
        if (it != r.stats->hop_transfers.end()) n += it->second;
    }
    return n;
}

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

namespace {

// Every tile an operator's program READS, in first-appearance order. Reads, not writes: a
// tile the operator produces is not something to place -- it is something the launch
// creates. Same distinction as `program_inputs` draws for operands, one level down.
// NAMED BY TENSOR, not by operand. Residency is a fact about the machine and persists
// across operators, so two operators reading one tensor must produce the SAME tile key --
// which they do not if the key is built from the kernel's local operand name.
std::vector<TileRef> tiles_read(const program::TileProgram& prog,
                                const std::map<std::string, std::string>& binding) {
    std::vector<TileRef> out;
    std::set<std::string> seen;
    for (const program::TileOp& op : prog.ops())
        for (const program::TileCoord& c : op.inputs) {
            const auto it = binding.find(c.operand);
            if (it == binding.end()) continue;
            TileRef t{it->second, c.ti, c.tj};
            if (seen.insert(t.key()).second) out.push_back(t);
        }
    return out;
}

// Which tensors a later operator still reads. What the orchestrator needs in order to know
// what is worth keeping, and it is the ONLY lookahead it does -- deciding on the basis of
// the program it was given, not of anything it measured.
std::set<std::string> tensors_read_after(const loadable::Loadable& l, std::size_t from) {
    std::set<std::string> out;
    for (std::size_t i = from; i < l.operators.size(); ++i)
        for (const std::string& t : l.operators[i].inputs) out.insert(t);
    return out;
}

} // namespace

// ---- the orchestrator -------------------------------------------------------
OrchestrationResult orchestrate(const loadable::Loadable& l,
                                program::platform::VirtualPlatform& platform,
                                TensorStore& tensors,
                                const OrchestratorOptions& opt) {
    OrchestrationResult result;
    std::uint64_t next_id = 1;

    for (const loadable::TensorRef& t : l.tensors)
        if (!tensors.has(t.name)) tensors.declare(t);

    const auto& spec = platform.deployment();
    const program::characterize::DeviceDescriptor dev = spec.device_view();
    const std::size_t l3_capacity = dev.l3_tiles;

    // What the ORCHESTRATOR believes is resident. Its own bookkeeping, not a peek into the
    // executor: it placed these tiles and it owns their lifetime, which is exactly the
    // contract TileExecutionRequest::initially_resident states from the other side.
    std::set<std::string> resident;
    std::map<std::string, TileRef> resident_refs;

    auto issue = [&](Descriptor d) {
        d.id = next_id++;
        result.trace.issued.push_back(d);
        return d.id;
    };
    auto complete = [&](std::uint64_t id, CompletionStatus st, bool timed, program::Cycle cyc,
                        std::vector<TileRef> released = {}, std::string why = {}) {
        Completion c;
        c.descriptor_id = id;
        c.status = st;
        c.timed = timed;
        c.cycles = cyc;
        c.released = std::move(released);
        c.diagnosis = std::move(why);
        result.trace.completions.push_back(c);
    };

    for (std::size_t i = 0; i < l.operators.size(); ++i) {
        const loadable::Operator& op = l.operators[i];
        result.operator_names.push_back(op.name);

        program::TileProgram prog = program::serialize::from_string(op.l0_program);
        std::map<std::string, std::string> binding;
        try {
            binding = operand_binding(prog, op);
        } catch (const std::exception& e) {
            result.refused = true;
            result.diagnosis = e.what();
            return result;
        }
        const std::vector<TileRef> needed = tiles_read(prog, binding);

        // ---- decide what to place ------------------------------------------
        std::vector<TileRef> to_place;
        for (const TileRef& t : needed)
            if (!resident.count(t.key())) to_place.push_back(t);

        // ---- decide what to give back, BEFORE asking for more --------------
        // A tile no remaining operator reads is dead weight. Releasing first is what lets a
        // bounded machine run a chain longer than its L3: asking for credit before
        // returning what is finished would refuse a run that fits.
        const std::set<std::string> still_wanted = tensors_read_after(l, i);
        std::vector<TileRef> to_release;
        for (const auto& [key, ref] : resident_refs)
            if (!still_wanted.count(ref.tensor)) to_release.push_back(ref);
        for (const TileRef& t : to_release) {
            Descriptor d;
            d.kind = DescriptorKind::Release;
            d.tile = t;
            const std::uint64_t id = issue(d);
            resident.erase(t.key());
            resident_refs.erase(t.key());
            // A RELEASE has no latency of its own at this level -- returning a credit is
            // bookkeeping, not a move -- and saying "not modelled" beats reporting 0.
            complete(id, CompletionStatus::Done, /*timed=*/false, 0, {t});
        }

        // ---- can the machine take them? ------------------------------------
        StatusView status(spec, resident, l3_capacity);
        if (!status.can_place(to_place.size())) {
            // THE ABI SAYS NO, and the orchestrator reports a decision point rather than
            // hanging. This is the case a static schedule never meets, because the compiler
            // proved its schedule fits; a runtime allocator meets it and must survive it.
            Descriptor d;
            d.kind = DescriptorKind::Place;
            d.tile = to_place.empty() ? TileRef{} : to_place.front();
            const std::uint64_t id = issue(d);
            std::ostringstream why;
            why << "operator \"" << op.name << "\" needs " << to_place.size()
                << " new L3 slot(s), " << status.credits_available() << " available of "
                << l3_capacity << " (" << status.resident_count() << " resident)";
            complete(id, CompletionStatus::RefusedInsufficientCredit, false, 0, {}, why.str());
            result.refused = true;
            result.diagnosis = why.str();
            return result;
        }

        for (const TileRef& t : to_place) {
            Descriptor d;
            d.kind = DescriptorKind::Place;
            d.tile = t;
            // ONE LEG. At L-T1 the orchestrator orders the DMA leg only; the BlockMover and
            // Streamer legs are the executor's, scheduled across the run under credits.
            // See the header on why the vocabulary is wider than what this level uses.
            d.leg = program::Hop::DmaDramToL3;
            d.resource = program::platform::ResourceName{spec.device(0).name,
                                                        program::platform::ResourceKind::L3Tile,
                                                        {0},
                                                        0};
            const std::uint64_t id = issue(d);
            if (opt.reuse_shared_inputs) {
                resident.insert(t.key());
                resident_refs.emplace(t.key(), t);
            }
            // A PLACE has no completion cycle of its own at L-T1 (#305 §6.2): the executor
            // decides when the leg happens, inside the launch. Reporting a number here would
            // be inventing one.
            complete(id, CompletionStatus::Done, /*timed=*/false, 0);
        }

        // ---- launch --------------------------------------------------------
        for (const auto& [operand, tensor] : binding)
            tensors.load_into(prog, operand, tensor);

        Descriptor launch;
        launch.kind = DescriptorKind::Launch;
        launch.target = op.name;
        const std::uint64_t launch_id = issue(launch);

        const auto handle = platform.load_program(prog);
        const auto snapshot = platform.snapshot();
        RunOutcome outcome;
        try {
            const auto run = platform.run(handle, opt.level, snapshot,
                                          program::Placement::single(dev.compute_tiles),
                                          nullptr, resident);
            outcome = run.outcome;
        } catch (const std::exception& e) {
            complete(launch_id, CompletionStatus::RefusedUnsupported, false, 0, {}, e.what());
            result.refused = true;
            result.diagnosis = e.what();
            return result;
        }
        complete(launch_id, CompletionStatus::Done, outcome.has_timing, outcome.makespan);

        for (const auto& [operand, tensor] : binding)
            tensors.store_from(platform.program(handle), operand, tensor);

        for (const std::string& u : outcome.unmodelled_inputs) result.unmodelled.push_back(u);
        result.per_operator.push_back(outcome);
    }

    return result;
}

} // namespace sw::kpu::orchestration
