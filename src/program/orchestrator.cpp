// ============================================================================
// src/program/orchestrator.cpp
// The deciding orchestrator (#305 increments 2-3). See the header for what it may and may
// not touch, and kpu_device.hpp for the reservation rules it decides under.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/orchestration/orchestrator.hpp>

#include <sw/kpu/orchestration/mmio.hpp>

#include <sw/kpu/program/platform/digest.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <set>

namespace sw::kpu::orchestration {

// ---- descriptor and completion rendering ------------------------------------
const char* to_string(DescriptorKind k) {
    switch (k) {
        case DescriptorKind::Place:     return "PLACE";
        case DescriptorKind::Release:   return "RELEASE";
        case DescriptorKind::Configure: return "CONFIGURE";
        case DescriptorKind::Launch:    return "LAUNCH";
        case DescriptorKind::Fence:     return "FENCE";
        case DescriptorKind::Reserve:   return "RESERVE";
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

const char* to_string(RefusalCause c) {
    switch (c) {
        case RefusalCause::None:                  return "none";
        case RefusalCause::InsufficientCredit:    return "insufficient-credit";
        case RefusalCause::ReservationOutOfOrder: return "out-of-order";
        case RefusalCause::NoReservation:         return "no-reservation";
        case RefusalCause::ReservationExceeded:   return "reservation-exceeded";
        case RefusalCause::Unsupported:           return "unsupported";
    }
    return "?";
}

const char* to_string(AllocationPolicy p) {
    switch (p) {
        case AllocationPolicy::ProgramOrder:      return "program-order";
        case AllocationPolicy::ReserveThenLaunch: return "reserve-then-launch";
        case AllocationPolicy::GreedyPrefetch:    return "greedy-prefetch";
    }
    return "?";
}

std::string Descriptor::str() const {
    std::string out = std::to_string(id) + " " + to_string(kind);
    switch (kind) {
        case DescriptorKind::Place:
            out += " " + tile.str() + " " + program::to_string(leg) + " -> " +
                   program::platform::format(resource);
            if (!target.empty()) out += " for " + target;
            break;
        case DescriptorKind::Release:
            out += " " + tile.str();
            if (flags & kReleaseAtLastRead) out += " at-last-read";
            break;
        case DescriptorKind::Reserve:
            out += " " + target + " slots=" + std::to_string(slots);
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
    if (cause != RefusalCause::None) out += std::string(" cause=") + orchestration::to_string(cause);
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


namespace {

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

// ---- the decider ------------------------------------------------------------
OrchestrationResult run_orchestrator(const loadable::Loadable& l, KpuPort& port,
                                     const OrchestratorOptions& opt) {
    OrchestrationResult result;
    std::uint64_t next_id = 1;
    const std::size_t n = l.operators.size();

    if (port.operator_count() != n) {
        result.refused = true;
        result.diagnosis = "the device holds " + std::to_string(port.operator_count()) +
                           " operator(s), the loadable names " + std::to_string(n);
        return result;
    }

    // What the ORCHESTRATOR has decided to hold across launches, by tensor key. The device
    // keeps the authoritative ledger; this is the decider's memory of its own decisions, which
    // is what it plans with. The two agreeing is checked by every test that measures reuse.
    std::set<std::string> held;
    std::vector<bool> reserved(n, false);
    std::vector<std::set<std::string>> placed(n);      // PLACEd ahead of the operator's turn

    auto issue = [&](Descriptor d) {
        d.id = next_id++;
        result.trace.issued.push_back(d);
        port.submit(d);
        return d.id;
    };
    // Collect everything the device has posted, in the order it posted it. Completion order is
    // NOT issue order: a RELEASE at last read completes after the LAUNCH it governs.
    auto drain = [&] {
        Completion c;
        while (port.poll_completion(c)) result.trace.completions.push_back(std::move(c));
    };
    // The completion for `id`. The device never blocks, so if it is not there after a drain it
    // never will be -- and the honest response is a refusal naming the protocol bug, not a loop.
    auto await = [&](std::uint64_t id, Completion& out) {
        drain();
        for (auto it = result.trace.completions.rbegin(); it != result.trace.completions.rend();
             ++it)
            if (it->descriptor_id == id) {
                out = *it;
                return true;
            }
        out = Completion{};
        out.descriptor_id = id;
        out.status = CompletionStatus::RefusedUnsupported;
        out.cause = RefusalCause::Unsupported;
        out.diagnosis = "descriptor " + std::to_string(id) +
                        " has no completion; the device never blocks, so this is a protocol "
                        "error, not a wait";
        return false;
    };
    auto stop = [&](const Completion& c) {
        result.refused = true;
        result.diagnosis = c.diagnosis;
        return result;
    };

    // WHERE A PLACE LANDS, from the device's own inventory rather than a spelling the
    // orchestrator assumes. The first L3 tile: at L-T1 the executor does not distinguish L3
    // tiles, and naming one is the ABI's job, not a placement decision this level can make.
    program::platform::ResourceName l3_target;
    for (const auto& r : port.inventory())
        if (r.kind == program::platform::ResourceKind::L3Tile) {
            l3_target = r;
            break;
        }

    auto place = [&](std::size_t op, const TileRef& t, Completion& c) {
        Descriptor d;
        d.kind = DescriptorKind::Place;
        d.tile = t;
        // ONE LEG. At L-T1 the orchestrator orders the DMA leg only; the BlockMover and
        // Streamer legs are the executor's, scheduled across the run under credits.
        d.leg = program::Hop::DmaDramToL3;
        d.resource = l3_target;
        d.target = l.operators[op].name;
        return await(issue(d), c) && c.status == CompletionStatus::Done;
    };
    auto reserve = [&](std::size_t op, std::uint32_t slots, Completion& c) {
        Descriptor d;
        d.kind = DescriptorKind::Reserve;
        d.target = l.operators[op].name;
        d.slots = slots;
        return await(issue(d), c) && c.status == CompletionStatus::Done;
    };
    // How many of an operator's read tiles it will keep past its own launch.
    auto retained_count = [&](const OperatorManifest& m, std::size_t op) {
        if (!opt.reuse_shared_inputs) return std::size_t{0};
        const std::set<std::string> later = tensors_read_after(l, op + 1);
        std::size_t k = 0;
        for (const TileRef& t : m.reads)
            if (later.count(t.tensor)) ++k;
        return k;
    };

    for (std::size_t k = 0; k < n; ++k) {
        const loadable::Operator& op = l.operators[k];
        result.operator_names.push_back(op.name);

        const OperatorManifest m = port.manifest(static_cast<std::uint32_t>(k));
        if (!m.valid) {
            result.refused = true;
            result.diagnosis = m.error;
            return result;
        }

        // WHAT A LATER OPERATOR STILL WANTS -- from k+1 on, so this operator's own reads do not
        // count. It decides what is worth keeping past this launch, and therefore what is
        // worth paying an L3 slot for during it.
        const std::set<std::string> wanted_later = tensors_read_after(l, k + 1);
        auto keep = [&](const TileRef& t) {
            return opt.reuse_shared_inputs && wanted_later.count(t.tensor) != 0;
        };

        // ---- one ledger, one spelling ----------------------------------------
        // The device's credit ledger is authoritative; `held` is the orchestrator's memory of
        // its own decisions. Read the status surface and refuse if they disagree, rather than
        // plan against a machine that is not there -- the "two spellings of one fact" failure
        // increment 2 kept meeting, caught at the point it would start to matter.
        const StatusSnapshot status = port.read_status();
        if (status.held != held.size() || status.completed != k) {
            result.refused = true;
            result.diagnosis = "before operator \"" + op.name + "\": the device holds " +
                               std::to_string(status.held) + " tile(s) and has completed " +
                               std::to_string(status.completed) +
                               " operator(s); the orchestrator decided " +
                               std::to_string(held.size()) + " and " + std::to_string(k);
            return result;
        }

        Completion c;

        // ---- reserve: all or none, in operator order (R1, R2) ----------------
        // THE ABI SAYS NO, and the orchestrator reports a decision point rather than hanging.
        // The reservation is the SUFFICIENT bound (plan §3.4), so a granted one completes.
        if (opt.policy != AllocationPolicy::GreedyPrefetch && !reserved[k]) {
            if (!reserve(k, m.bound(retained_count(m, k)), c)) return stop(c);
            reserved[k] = true;
        }

        // ---- place AHEAD: the freedom program-order acquisition forbids -------
        // Only with a granted reservation for k+1 (or under the test-only greedy ablation,
        // which has none and is how the wedge is shown). Tiles k also reads are left to their
        // turn: k places or keeps them itself, and one tile is one slot.
        if (opt.policy != AllocationPolicy::ProgramOrder && k + 1 < n) {
            const OperatorManifest next = port.manifest(static_cast<std::uint32_t>(k + 1));
            if (next.valid) {
                const std::uint32_t next_bound = next.bound(retained_count(next, k + 1));
                bool granted = opt.policy == AllocationPolicy::GreedyPrefetch;
                if (opt.policy == AllocationPolicy::ReserveThenLaunch && !reserved[k + 1]) {
                    if (reserve(k + 1, next_bound, c)) {
                        reserved[k + 1] = true;
                        granted = true;
                    } else if (c.cause != RefusalCause::InsufficientCredit) {
                        return stop(c);
                    }
                    // InsufficientCredit: the DECISION POINT. The machine cannot take k+1 yet;
                    // run k without prefetching, and k+1 reserves on its own turn.
                }
                if (granted) {
                    std::set<std::string> mine;
                    for (const TileRef& t : m.reads) mine.insert(t.key());
                    // NEVER AHEAD OF ITS PRODUCER. A tile of a tensor operator k is about to
                    // WRITE does not exist yet in its final form; placing it before k runs
                    // would be a read-after-write hazard. At L-T1 it would not change a value
                    // (the leg is charged inside k+1's run), which is exactly why it has to be
                    // excluded by rule rather than caught by a value comparison.
                    const std::set<std::string> produced(op.outputs.begin(), op.outputs.end());
                    for (const TileRef& t : next.reads) {
                        const std::string key = t.key();
                        if (held.count(key) || mine.count(key) || placed[k + 1].count(key) ||
                            produced.count(t.tensor))
                            continue;
                        if (opt.policy == AllocationPolicy::ReserveThenLaunch &&
                            placed[k + 1].size() >= next_bound)
                            break;                       // stay inside the reservation (R3)
                        if (!place(k + 1, t, c)) return stop(c);
                        placed[k + 1].insert(key);
                    }
                }
            }
        }

        // ---- place this operator's tiles --------------------------------------
        for (const TileRef& t : m.reads) {
            const std::string key = t.key();
            if (held.count(key) || placed[k].count(key)) continue;
            if (!place(k, t, c)) return stop(c);
        }

        // ---- release at last read: decided BEFORE the launch it governs -------
        // The device needs this at launch time, to know which credits come back at each
        // tile's last reader. Holding a tile nobody will read again buys nothing and costs a
        // slot for the whole run, so everything not kept for a later operator is released.
        for (const TileRef& t : m.reads) {
            if (keep(t)) continue;
            Descriptor d;
            d.kind = DescriptorKind::Release;
            d.tile = t;
            d.flags = kReleaseAtLastRead;
            issue(d);                                    // completes after the launch
        }

        // ---- launch ------------------------------------------------------------
        Descriptor launch;
        launch.kind = DescriptorKind::Launch;
        launch.target = op.name;
        if (!await(issue(launch), c) || c.status != CompletionStatus::Done) return stop(c);
        drain();

        for (const TileRef& t : m.reads) {
            if (keep(t)) held.insert(t.key());
            else held.erase(t.key());
        }
    }

    drain();
    return result;
}

// ---- the end-to-end entry point --------------------------------------------
OrchestrationResult orchestrate(const loadable::Loadable& l,
                                program::platform::VirtualPlatform& platform,
                                TensorStore& tensors,
                                const OrchestratorOptions& opt) {
    KpuDevice device(l, platform, tensors, opt.level, opt.device);
    OrchestrationResult result;
    if (opt.transport == Transport::Mmio) {
        MmioSystem system(device);
        result = run_orchestrator(l, system.port(), opt);
        result.bus_log_digest = system.bus().log_digest();
        result.bus_accesses = system.bus().log().size();
    } else {
        DirectPort port(device);
        result = run_orchestrator(l, port, opt);
    }
    // What the launches did is the DEVICE's evidence, recorded beside the trace rather than
    // read through the ABI -- the orchestrator never needed it to decide.
    result.per_operator = device.outcomes();
    for (const RunOutcome& o : result.per_operator)
        for (const std::string& u : o.unmodelled_inputs) result.unmodelled.push_back(u);
    return result;
}

} // namespace sw::kpu::orchestration
