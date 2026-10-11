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
    for (const auto& r : per_operator) n += static_cast<std::size_t>(r.dram_loads);
    return n;
}


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

    // What the ORCHESTRATOR expects held across launches, by tensor key: what the programs it
    // launched retained and nothing since inherited. The device keeps the authoritative ledger;
    // this is the decider's model of it, from the manifests, which is what it plans with.
    std::set<std::string> held;
    std::vector<bool> reserved(n, false);

    auto issue = [&](Descriptor d) {
        d.id = next_id++;
        result.trace.issued.push_back(d);
        port.submit(d);
        return d.id;
    };
    // Collect everything the device has posted, in the order it posted it.
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
    auto reserve = [&](std::size_t op, std::uint32_t slots, Completion& c) {
        Descriptor d;
        d.kind = DescriptorKind::Reserve;
        d.target = l.operators[op].name;
        d.slots = slots;
        return await(issue(d), c) && c.status == CompletionStatus::Done;
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

        // ---- one ledger, one spelling ----------------------------------------
        // The device's credit ledger is authoritative; `held` is the orchestrator's model of it.
        // Read the status surface and refuse if they disagree, rather than plan against a
        // machine that is not there.
        const StatusSnapshot status = port.read_status();
        if (status.held != held.size() || status.completed != k) {
            result.refused = true;
            result.diagnosis = "before operator \"" + op.name + "\": the device holds " +
                               std::to_string(status.held) + " tile(s) and has completed " +
                               std::to_string(status.completed) +
                               " operator(s); the orchestrator expected " +
                               std::to_string(held.size()) + " and " + std::to_string(k);
            return result;
        }

        Completion c;

        // ---- reserve: all or none, in operator order (R1, R2) ----------------
        // THE ABI SAYS NO, and the orchestrator reports a decision point rather than hanging.
        // The reservation is the SUFFICIENT bound -- the program's L3, less the slots it
        // inherits, which are already held -- so a granted one completes.
        if (!reserved[k]) {
            if (!reserve(k, m.bound(), c)) return stop(c);
            reserved[k] = true;
        }

        // ---- reserve AHEAD: whether k+1 fits is decided while k runs -----------
        if (opt.policy == AllocationPolicy::ReserveThenLaunch && k + 1 < n && !reserved[k + 1]) {
            const OperatorManifest next = port.manifest(static_cast<std::uint32_t>(k + 1));
            if (next.valid) {
                if (reserve(k + 1, next.bound(), c)) {
                    reserved[k + 1] = true;
                } else if (c.cause != RefusalCause::InsufficientCredit) {
                    return stop(c);
                }
                // InsufficientCredit: the DECISION POINT. The machine cannot take k+1 beside k
                // yet; run k, and k+1 reserves on its own turn.
            }
        }

        // ---- launch ------------------------------------------------------------
        Descriptor launch;
        launch.kind = DescriptorKind::Launch;
        launch.target = op.name;
        if (!await(issue(launch), c) || c.status != CompletionStatus::Done) return stop(c);
        drain();

        for (const TileRef& t : m.inherits) held.erase(t.key());
        for (const TileRef& t : m.retains) held.insert(t.key());
    }

    drain();
    return result;
}

// ---- the end-to-end entry point --------------------------------------------
OrchestrationResult orchestrate(const loadable::Loadable& l,
                                program::platform::VirtualPlatform& platform,
                                TensorStore& tensors,
                                const OrchestratorOptions& opt) {
    KpuDevice device(l, platform, tensors, opt.level);
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
    for (const auto& o : result.per_operator)
        for (const std::string& u : o.unmodelled) result.unmodelled.push_back(u);
    return result;
}

} // namespace sw::kpu::orchestration
