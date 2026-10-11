// ============================================================================
// include/sw/kpu/program/hop.hpp
// The legs of the movement chain and the processes that own them.
//
// These outlived the L0 L-T1 executor they were written for (retired in
// kpu-run-csp-programs step 4e): the tile-flow record's `hop` and `mover` columns and
// the orchestration ABI's PLACE leg name a leg by this enum, and the values are part of
// both formats.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <cstdint>

namespace sw::kpu::program {

using Cycle = std::uint64_t;

// One leg of the movement chain, named by its governing CSP process and the memories it
// connects (design note §6.1).
//
//   DMA         DRAM <-> L3
//   BlockMover  L3 <-> L2, and L3 -> L3 across the NoC (reuse)
//   Streamer    L2 <-> L1
//
// HOPS DO NOT COLLAPSE. Each leg is a distinct process over a distinct physical pathway,
// so one stage standing for two describes a machine that cannot be built. A span always
// contains all of its hops; residency changes only where the chain STARTS. There is
// deliberately no `Collapsed` member and no single-pool mode — an earlier revision had
// both, which was the premise of this increment and it was wrong.
//
// The fabric is not a hop endpoint: it reads only L1, and the L1 stream buffers push
// elements into it, which is not a mover and owns no lanes (ADR 0002 §3.3).
//
// EVERY HOP IS A PUSH, a block write by the process that owns it (the KPU is a push machine;
// docs/plans/noc-port-arbitration.md §1.2). There is no DMA read: a result leaves the machine
// as a BlockMover writing it from L3 into a DMA engine's buffer -- the port's ejection bus --
// and the DMA engine then writes the buffer to DRAM. The values are explicit because they are
// the record's `hop` column (.tflow version 3).
enum class Hop : int {
    DmaDramToL3 = 0,             // 1. inbound: a DMA engine pushes the block into L3 (injection)
    BlockMoverL3ToL2 = 1,        // 2. inbound: BlockMover, may restructure/reshape
    StreamerL2ToL1 = 2,          // 3. inbound: Streamer writes an L1 stream buffer
    StreamerL1ToL2 = 3,          // 5. outbound: Streamer writes results back to L2
    BlockMoverL2ToL3 = 4,        // 6. outbound: BlockMover writes them into L3
    BlockMoverL3ToDmaBuffer = 5, // 7. outbound: BlockMover pushes the block out of the machine
                                 //    into a DMA engine buffer (ejection)
    DmaBufferToDram = 6,         // 8. outbound: the DMA engine writes its buffer to DRAM
    BlockMoverL3ToL3 = 7,        // reuse: across the NoC
};

inline const char* to_string(Hop h) {
    switch (h) {
        case Hop::DmaDramToL3:      return "dma:dram->l3";
        case Hop::BlockMoverL3ToL2: return "bm:l3->l2";
        case Hop::StreamerL2ToL1:   return "str:l2->l1";
        case Hop::StreamerL1ToL2:   return "str:l1->l2";
        case Hop::BlockMoverL2ToL3: return "bm:l2->l3";
        case Hop::BlockMoverL3ToDmaBuffer: return "bm:l3->dmabuf";
        case Hop::DmaBufferToDram:  return "dma:dmabuf->dram";
        case Hop::BlockMoverL3ToL3: return "bm:l3->l3";
    }
    return "?";
}

// The process that owns the lanes. Inbound and outbound legs of one process SHARE its
// pool: there is one set of BlockMovers, not one per direction.
enum class Mover { Dma, BlockMover, Streamer, Noc };

inline const char* to_string(Mover m) {
    switch (m) {
        case Mover::Dma:        return "dma";
        case Mover::BlockMover: return "block-mover";
        case Mover::Streamer:   return "streamer";
        case Mover::Noc:        return "noc";
    }
    return "?";
}

inline Mover mover_of(Hop h) {
    switch (h) {
        case Hop::DmaDramToL3:
        case Hop::DmaBufferToDram:  return Mover::Dma;
        case Hop::BlockMoverL3ToL2:
        case Hop::BlockMoverL2ToL3:
        case Hop::BlockMoverL3ToDmaBuffer: return Mover::BlockMover;
        case Hop::StreamerL2ToL1:
        case Hop::StreamerL1ToL2:   return Mover::Streamer;
        case Hop::BlockMoverL3ToL3: return Mover::Noc;
    }
    return Mover::Dma;
}

} // namespace sw::kpu::program
