// ============================================================================
// include/sw/kpu/program/driver/timeline_trace.hpp
// Map an L-T1 run's timeline onto sw::trace::TraceEntry, so the existing viewers
// read it.
//
// Design note §D5: reuse the trace FORMAT; the mapping into it is work, because
// nothing serialized TileOpRecord or HopRecord before this. ChromeTraceExporter
// consumes TraceEntry, and ResourceTrackerExporter consumes ResourceTrack —
// neither consumes the executor's own records.
//
// ONE EVENT PER HOP, not one per op. A transfer's legs run on different processes
// at different times, and collapsing them into a single span would hide exactly
// what the movement chain exists to show: which process a tile is waiting on.
// The op's own span is recoverable as the first leg's start to the last leg's
// finish, so nothing is lost by not emitting it.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/tile_transaction_executor.hpp>
#include <sw/kpu/program/tile_work.hpp>
#include <sw/trace/trace_entry.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::driver {

// Which component performs a leg. The hop names its governing CSP process, and the
// trace vocabulary has one entry per process for three of the four, so those are a
// rename rather than a modelling decision.
//
// THE NoC HAS NO TRACE IDENTITY YET, AND THIS REFUSES TO INVENT ONE. Mapping an
// L3->L3 leg onto BLOCK_MOVER would put NoC lane 0 and BlockMover lane 0 on the same
// track, misreporting the occupancy of both — and picking a component for a link the
// vocabulary does not model is a decision about that vocabulary, not about this
// mapping. No program emits this leg today: `build_chain` has no path to it, and the
// L3<->L3 hop stays unused until multi-compute-tile execution lands (#244). So the
// guard is unreachable, and it is here to make sure whoever makes the leg reachable
// has to give it a real identity rather than inherit a wrong one.
inline sw::trace::ComponentType component_of(Hop h) {
    switch (mover_of(h)) {
        case Mover::Dma:        return sw::trace::ComponentType::DMA_ENGINE;
        case Mover::BlockMover: return sw::trace::ComponentType::BLOCK_MOVER;
        case Mover::Streamer:   return sw::trace::ComponentType::STREAMER;
        case Mover::Noc:        break;
    }
    throw std::invalid_argument(
        std::string("timeline_trace: ") + to_string(h) +
        " has no trace component: the NoC needs its own identity in sw::trace::"
        "ComponentType before an L3->L3 leg can be traced (#244)");
}

// Movement is a transfer; compute is a matmul. READ/WRITE would have to pick a
// direction per leg, and a leg is one transfer that both reads and writes.
inline sw::trace::TransactionType transaction_of(Hop) {
    return sw::trace::TransactionType::TRANSFER;
}

inline std::string tile_label(const TileOp& op) {
    if (!op.inputs.empty()) return op.inputs[0].to_string();
    if (!op.outputs.empty()) return op.outputs[0].to_string();
    return "?";
}

// ----------------------------------------------------------------------------
// Convert a run's timeline into trace entries. `element_bytes` comes from the
// device, so the byte payloads match what the executor actually charged for.
// ----------------------------------------------------------------------------
inline std::vector<sw::trace::TraceEntry> to_trace_entries(
        const TileProgram& prog, const std::vector<TileOpRecord>& timeline,
        double element_bytes) {
    std::vector<sw::trace::TraceEntry> out;
    const auto& ops = prog.ops();
    std::uint64_t id = 0;

    for (const TileOpRecord& rec : timeline) {
        if (rec.op_index >= ops.size()) continue;
        const TileOp& op = ops[rec.op_index];
        const TileWork w = tile_work_of(prog, op, element_bytes);

        if (rec.hops.empty()) {
            // A compute op, or movement a residency made free. Zero-work ops are
            // emitted too: "it happened and cost nothing" is a fact worth seeing,
            // and dropping them would make the op indices in the trace skip.
            sw::trace::TraceEntry e(rec.start,
                                    w.is_compute ? sw::trace::ComponentType::COMPUTE_FABRIC
                                                 : component_of(Hop::DmaDramToL3),
                                    rec.resource_id,
                                    w.is_compute ? sw::trace::TransactionType::MATMUL
                                                 : sw::trace::TransactionType::TRANSFER,
                                    id++);
            if (w.is_compute) {
                sw::trace::ComputePayload p{};
                p.num_operations = static_cast<std::uint64_t>(w.macs);
                p.kernel_name = to_string(op.kind);
                e.payload = p;
            }
            e.description = std::string(to_string(op.kind)) + " " + tile_label(op) +
                            " [op " + std::to_string(rec.op_index) + "]" +
                            (rec.zero_work ? " (zero work)" : "");
            e.complete(rec.finish);
            out.push_back(std::move(e));
            continue;
        }

        for (const HopRecord& hr : rec.hops) {
            sw::trace::TraceEntry e(hr.start, component_of(hr.hop), hr.lane,
                                    transaction_of(hr.hop), id++);
            sw::trace::DMAPayload p{};
            p.bytes_transferred = static_cast<std::uint64_t>(w.bytes);
            e.payload = p;
            e.description = std::string(to_string(hr.hop)) + " " + tile_label(op) +
                            " [op " + std::to_string(rec.op_index) + "]";
            e.complete(hr.finish);
            out.push_back(std::move(e));
        }
    }
    return out;
}

} // namespace sw::kpu::program::driver
