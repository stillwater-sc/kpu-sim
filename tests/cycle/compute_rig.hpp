#pragma once
// ============================================================================
// tests/cycle/compute_rig.hpp
// The compute subsystem from L3 to L3, for the tests of steps 2d-2f (#545-#547): the library's
// ComputeMachine (compute_subsystem.hpp), the same machine kpu-run --compute-subsystem builds:
// L3 -> BlockMover -> L2 -> Streamer -> L1 lanes -> PE array -> north FIFOs -> Combiner -> L2 ->
// BlockMover -> L3, with the three-way check_l2_slot_credits as an invariant.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include "l2_bm_rig.hpp"

#include <sw/kpu/cycle/compute/block_mover.hpp>
#include <sw/kpu/cycle/compute/combiner.hpp>
#include <sw/kpu/cycle/compute/compute_subsystem.hpp>
#include <sw/kpu/cycle/compute/l1_lanes.hpp>
#include <sw/kpu/cycle/compute/l2.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/compute/pe_array.hpp>
#include <sw/kpu/cycle/compute/streamer.hpp>
#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sw::kpu::cycle::test {

using ComputeRig = sw::kpu::cycle::ComputeMachine;

}  // namespace sw::kpu::cycle::test
