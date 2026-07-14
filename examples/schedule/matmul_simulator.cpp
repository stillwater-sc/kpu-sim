// ============================================================================
// examples/schedule/matmul_simulator.cpp
// Standalone matmul simulator with a horizontal tile-state debug log
// (companion to softmax_simulator; see docs/tile-state-tracking.md).
//
// Runs a tiled matmul schedule (MatMulScheduleGenerator) one executor cycle at
// a time and, after each cycle, records any change in what tiles occupy
// L3 / L2 / L1 / array via the TileTracker. Watch A tiles (rows) and B tiles
// (columns) stream DRAM->L3->L2->L1/array, a C[ti,tj] compute accumulate over
// the K-slices in the array, and the C result drain back out. Ends on the
// C = A x B host-oracle check.
//
// Usage: matmul_simulator [--m M] [--n N] [--k K] [--tile T]
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/tile_tracker.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;

namespace {

// Enqueue one schedule operation; a COMPUTE for C[ti,tj] becomes a
// value-producing MatMulComputeSpec by splitting its interleaved A/B K-slice
// dependencies into the paired a_tiles / b_tiles the executor accumulates.
void enqueue(ConcurrentTimingExecutor& exec, const ScheduleOperation& op) {
    switch (op.type) {
        case ScheduleOpType::LOAD:      exec.schedule_load(op.tile, op.engine_id); break;
        case ScheduleOpType::MOVE:      exec.schedule_move(op.tile, op.transpose, op.mover_id); break;
        case ScheduleOpType::FEED:      exec.schedule_feed(op.tile, op.streamer_id); break;
        case ScheduleOpType::DRAIN:     exec.schedule_drain(op.tile, op.streamer_id); break;
        case ScheduleOpType::WRITEBACK: exec.schedule_writeback(op.tile, op.mover_id); break;
        case ScheduleOpType::STORE:     exec.schedule_store(op.tile, op.engine_id); break;
        case ScheduleOpType::COMPUTE: {
            ConcurrentTimingExecutor::MatMulComputeSpec spec;
            for (const auto& dep : op.dependency_tiles) {
                if (dep.matrix == MatrixID::A) spec.a_tiles.push_back(dep);
                else                            spec.b_tiles.push_back(dep);
            }
            exec.schedule_matmul_compute(op.tile, spec);
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
    struct Opt { const char* flag; long def; };
    long vals[4];
    const Opt opts[4] = {{"--m", 32}, {"--n", 32}, {"--k", 32}, {"--tile", 16}};
    for (int i = 0; i < 4; ++i) {
        vals[i] = arg(argc, argv, opts[i].flag, opts[i].def);
        if (vals[i] <= 0) {
            std::cerr << "error: " << opts[i].flag << " must be a positive integer\n"
                      << "usage: matmul_simulator [--m M] [--n N] [--k K] [--tile T]\n";
            return 2;
        }
    }
    const Size M = vals[0], N = vals[1], K = vals[2], T = vals[3];
    if (M % T || N % T || K % T) {
        std::cerr << "error: --tile must divide --m, --n and --k evenly\n";
        return 2;
    }

    try {
    MatMulScheduleGenerator::Config cfg;
    cfg.M = M; cfg.N = N; cfg.K = K;
    cfg.Ti = cfg.Tj = cfg.Tk = T;
    cfg.a_base = 0x100000; cfg.b_base = 0x400000; cfg.c_base = 0x700000;
    auto schedule = MatMulScheduleGenerator(cfg).generate();
    if (!schedule.valid) {
        std::cerr << "schedule refused: " << schedule.error_message << "\n";
        return 1;
    }

    // Host operands (deterministic, non-trivial) and the reference C = A x B
    std::vector<double> A(M * K), B(K * N), C(M * N, 0.0);
    for (Size i = 0; i < M; ++i)
        for (Size k = 0; k < K; ++k) A[i * K + k] = ((i + k) % 4) + 1;
    for (Size k = 0; k < K; ++k)
        for (Size j = 0; j < N; ++j) B[k * N + j] = ((k * 2 + j) % 3) + 1;
    for (Size i = 0; i < M; ++i)
        for (Size k = 0; k < K; ++k)
            for (Size j = 0; j < N; ++j) C[i * N + j] += A[i * K + k] * B[k * N + j];

    std::cout << "Matmul simulator  —  " << schedule.metadata.strategy
              << "  (C[" << M << "x" << N << "] = A[" << M << "x" << K
              << "] * B[" << K << "x" << N << "], tile " << T << ")\n\n";

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.max_cycles = 2'000'000;
    ConcurrentTimingExecutor exec(ecfg);

    // Seed the A and B input tiles from the host operands. A[ti,0,tk] is the
    // [T x T] block A(ti,tk); B[0,tj,tk] is the block B(tk,tj).
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        std::vector<float> block(T * T);
        if (id.matrix == MatrixID::A) {
            for (Size r = 0; r < T; ++r)
                for (Size c = 0; c < T; ++c)
                    block[r * T + c] = static_cast<float>(A[(id.ti * T + r) * K + id.tk * T + c]);
        } else {  // B[0, tj, tk] -> B block (tk, tj)
            for (Size r = 0; r < T; ++r)
                for (Size c = 0; c < T; ++c)
                    block[r * T + c] = static_cast<float>(B[(id.tk * T + r) * N + id.tj * T + c]);
        }
        exec.set_tile_payload(id, TilePayload{T, T, std::move(block)});
    }
    for (const auto& op : schedule.operations) enqueue(exec, op);

    // Drive one cycle at a time, tracking every occupancy transition
    TileTracker tracker;
    tracker.observe(exec);
    while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles) {
        exec.step();
        tracker.observe(exec);
    }
    std::cout << tracker.log() << "\n";

    if (!exec.is_complete()) {
        std::cerr << "schedule did not complete (deadlock or max_cycles)\n";
        return 1;
    }

    // Correctness: each C tile matches the host reference C = A x B
    std::cout << "Result: C = A x B\n";
    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < T; ++r)
            for (Size c = 0; c < T; ++c) {
                const double got = p.values[r * T + c];
                const double want = C[(id.ti * T + r) * N + id.tj * T + c];
                max_err = std::max(max_err, std::abs(got - want));
            }
    }
    const bool ok = max_err < 1e-3;
    std::cout << "  max abs error vs host oracle: " << max_err
              << (ok ? "  [OK]" : "  [FAIL]") << "\n";
    std::cout << "\n" << (ok ? "MATMUL OK" : "MATMUL MISMATCH") << "\n";
    return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
