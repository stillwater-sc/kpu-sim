// ============================================================================
// include/sw/kpu/program/driver/csp_timeline.hpp
// A CSP program's L-T1 run as Chrome-trace intervals (docs/plans/kpu-run-csp-programs.md
// step 4c).
//
// ONE EVENT PER RECORD: a leg of an action, on its process and lane, from its start to its
// finish -- so a Store is two events (the BlockMover's ejection into a DMA buffer, then the
// DMA's write), as it is two legs. A Release takes no process and no time; it is not an
// interval, so it is not an event (the slot it frees is the .tflow record's to show).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/transactional.hpp>
#include <sw/kpu/program/tile_program.hpp>
#include <sw/trace/trace_entry.hpp>

#include <map>
#include <string>
#include <vector>

namespace sw::kpu::program::driver {

inline sw::trace::ComponentType component_of(csp::TransactionalInterpreter::Proc p) {
    using P = csp::TransactionalInterpreter::Proc;
    switch (p) {
        case P::Dma: return sw::trace::ComponentType::DMA_ENGINE;
        case P::Bm:  return sw::trace::ComponentType::BLOCK_MOVER;
        case P::Str: return sw::trace::ComponentType::STREAMER;
        case P::Cf:  return sw::trace::ComponentType::COMPUTE_FABRIC;
    }
    return sw::trace::ComponentType::COMPUTE_FABRIC;
}

// `operands` gives the tiles' shapes (bytes per tile), `element_bytes` the device's element size.
inline std::vector<sw::trace::TraceEntry> csp_trace_entries(
        const std::vector<csp::TransactionalInterpreter::Record>& records, const TileProgram& operands,
        double element_bytes) {
    using P = csp::TransactionalInterpreter::Proc;
    std::vector<sw::trace::TraceEntry> out;
    std::map<std::size_t, std::size_t> legs;         // a Store's legs, in record order
    std::uint64_t id = 0;
    for (const auto& r : records) {
        const std::size_t leg = legs[r.action]++;
        if (!csp::TransactionalInterpreter::has_process(r.kind)) continue;
        const bool compute = r.proc == P::Cf;
        sw::trace::TraceEntry e(r.start, component_of(r.proc), static_cast<std::uint32_t>(r.lane),
                                compute ? sw::trace::TransactionType::MATMUL : sw::trace::TransactionType::TRANSFER,
                                id++);
        if (compute) {
            sw::trace::ComputePayload p{};
            p.kernel_name = "call";
            e.payload = p;
        } else {
            const TensorOperand& o = operands.operand(r.tile.operand);
            const double elems = double(o.row_end(r.tile.ti) - o.row_begin(r.tile.ti)) *
                                 double(o.col_end(r.tile.tj) - o.col_begin(r.tile.tj));
            sw::trace::DMAPayload p{};
            p.bytes_transferred = static_cast<std::uint64_t>(elems * element_bytes);
            e.payload = p;
        }
        e.description = std::string(csp::to_string(r.kind)) + " " + r.tile.to_string() + " [action " +
                        std::to_string(r.action) + "]" +
                        (r.kind == csp::Action::Kind::Store ? (leg == 0 ? " l3 -> dma buffer" : " dma buffer -> dram")
                                                            : "");
        e.complete(r.finish);
        out.push_back(std::move(e));
    }
    return out;
}

}  // namespace sw::kpu::program::driver
