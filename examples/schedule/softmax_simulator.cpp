// ============================================================================
// examples/schedule/softmax_simulator.cpp
// Standalone online-softmax simulator with a horizontal tile-state debug log
// (issue #165, deliverable B).
//
// Runs the E8 online-softmax schedule (OnlineSoftmaxScheduleGenerator #156 +
// the value ops from FunctionalSoftmaxExecutor #157) one executor cycle at a
// time, and after each cycle asks the TileTracker (#165) to record any change
// in what tiles occupy L3 / L2 / L1 / array. The result is a human-readable
// progression: watch the row stream DRAM->L3->L2->L1, the stats compute
// produce (m, l), that (m, l) stay resident and feed the apply computes, and
// the normalized C tiles drain back out. Ends on the row-sums-to-1 check.
//
// Usage: softmax_simulator [--rows R] [--len N] [--tile T] [--l3 C] [--l2 C]
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/tile_tracker.hpp>
#include <sw/kpu/timing/schedule/online_softmax_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/functional_softmax_executor.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;

namespace {

// Enqueue one schedule operation on the executor, binding the softmax value
// ops to the COMPUTEs: stats (matrix B) -> (m, l); apply (matrix C) ->
// exp(x - m)/l reading the resident (m, l). This is the same binding
// FunctionalSoftmaxExecutor uses, done inline so we can drive the executor
// cycle-by-cycle and observe between steps.
void enqueue(ConcurrentTimingExecutor& exec, const ScheduleOperation& op,
             Size row_elems) {
    switch (op.type) {
        case ScheduleOpType::LOAD:      exec.schedule_load(op.tile, op.engine_id); break;
        case ScheduleOpType::MOVE:      exec.schedule_move(op.tile, op.transpose, op.mover_id); break;
        case ScheduleOpType::FEED:      exec.schedule_feed(op.tile, op.streamer_id); break;
        case ScheduleOpType::DRAIN:     exec.schedule_drain(op.tile, op.streamer_id); break;
        case ScheduleOpType::WRITEBACK: exec.schedule_writeback(op.tile, op.mover_id); break;
        case ScheduleOpType::STORE:     exec.schedule_store(op.tile, op.engine_id); break;
        case ScheduleOpType::COMPUTE: {
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            spec.input_tiles = op.dependency_tiles;
            for (const auto& r : op.resident_tiles) spec.input_tiles.push_back(r);
            spec.resident_tiles = op.resident_tiles;
            if (op.tile.tile_id.matrix == MatrixID::B) {
                spec.operation = softmax_stats;
            } else {
                spec.operation = [row_elems](const std::vector<TilePayload>& in) {
                    return softmax_apply(in, row_elems);
                };
            }
            exec.schedule_functional_compute(op.tile, spec);
            break;
        }
    }
}

long arg(int argc, char** argv, const char* flag, long def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atol(argv[i + 1]);
    return def;
}

} // namespace

int main(int argc, char** argv) {
    OnlineSoftmaxScheduleGenerator::Config cfg;
    cfg.num_rows       = static_cast<Size>(arg(argc, argv, "--rows", 1));
    cfg.reduction_elems = static_cast<Size>(arg(argc, argv, "--len", 512));
    cfg.tile_elems     = static_cast<Size>(arg(argc, argv, "--tile", 256));
    cfg.l3_buffer_count = static_cast<Size>(arg(argc, argv, "--l3", 32));
    cfg.l2_bank_count  = static_cast<Size>(arg(argc, argv, "--l2", 64));
    cfg.in_base = 0x100000; cfg.stat_base = 0x200000; cfg.out_base = 0x300000;

    auto schedule = OnlineSoftmaxScheduleGenerator(cfg).generate();
    if (!schedule.valid) {
        std::cerr << "schedule refused: " << schedule.error_message << "\n";
        return 1;
    }

    // Deterministic input: a gentle per-row ramp
    const Size n = cfg.num_rows * cfg.reduction_elems;
    std::vector<float> data(n);
    for (Size r = 0; r < cfg.num_rows; ++r)
        for (Size i = 0; i < cfg.reduction_elems; ++i)
            data[r * cfg.reduction_elems + i] =
                static_cast<float>(r) - 1.0f + 0.01f * static_cast<float>(i % 32);

    std::cout << "Online softmax simulator  —  " << schedule.metadata.strategy
              << "  (" << cfg.num_rows << " row(s) x " << cfg.reduction_elems
              << ", tile " << cfg.tile_elems << ", envelope L3=" << cfg.l3_buffer_count
              << "/L2=" << cfg.l2_bank_count << ")\n\n";

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.l3_buffer_count = cfg.l3_buffer_count;
    ecfg.l2_bank_count = cfg.l2_bank_count;
    ecfg.max_cycles = 2'000'000;
    ConcurrentTimingExecutor exec(ecfg);

    // Seed A input tiles from the data (matrix A LOADs only)
    const Size te = cfg.tile_elems;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        if (op.tile.tile_id.matrix != MatrixID::A) continue;
        const Size row = op.tile.tile_id.ti, t = op.tile.tile_id.tj;
        const Size off = row * cfg.reduction_elems + t * te;
        const Size elems = op.tile.height;
        exec.set_tile_payload(op.tile.tile_id,
                              TilePayload{elems, 1,
                                          std::vector<float>(data.begin() + off,
                                                             data.begin() + off + elems)});
    }
    for (const auto& op : schedule.operations) enqueue(exec, op, cfg.reduction_elems);

    // Drive one cycle at a time, tracking every occupancy transition
    TileTracker tracker;
    tracker.observe(exec);
    while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles) {
        exec.step();
        tracker.observe(exec);
    }
    std::cout << tracker.log() << "\n";

    if (!exec.is_complete()) {
        std::cerr << "did not complete within max_cycles\n";
        return 1;
    }

    // Correctness: each output row sums to 1 (softmax is a distribution)
    std::cout << "Result: each row's softmax sums to 1\n";
    bool ok = true;
    for (Size r = 0; r < cfg.num_rows; ++r) {
        double sum = 0.0;
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::STORE) continue;
            if (op.tile.tile_id.matrix != MatrixID::C) continue;
            if (op.tile.tile_id.ti != r) continue;
            const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, op.tile.tile_id);
            for (float v : p.values) sum += v;
        }
        const bool row_ok = std::abs(sum - 1.0) < 1e-3;
        ok = ok && row_ok;
        std::cout << "  row " << r << ": sum = " << sum
                  << (row_ok ? "  [OK]" : "  [FAIL]") << "\n";
    }
    std::cout << "\n" << (ok ? "SOFTMAX OK" : "SOFTMAX MISMATCH") << "\n";
    return ok ? 0 : 1;
}
