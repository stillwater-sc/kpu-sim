// ============================================================================
// examples/schedule/layernorm_simulator.cpp
// Standalone LayerNorm simulator with a horizontal tile-state debug log
// (companion to softmax_simulator; see docs/tile-state-tracking.md).
//
// LayerNorm over the feature dimension: y = gamma * (x - mean)/sqrt(var + eps)
//   + beta, per row. Like softmax it is a row-streaming reduction whose (mean,
// var) statistic is handed to the apply computes as a compute-RESIDENT
// dependency (the E8 mechanism, no DRAM round-trip) - the SAME P3 movement
// pattern, so it reuses the online row-reduction schedule; only the value ops
// differ (VAR-moment stats + an affine normalize instead of softmax's
// max/exp-sum). Runs one executor cycle at a time and tracks every occupancy
// transition. Ends on a host-oracle check.
//
// Usage: layernorm_simulator [--rows R] [--len N] [--tile T]
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/tile_tracker.hpp>
#include <sw/kpu/timing/schedule/online_softmax_schedule_generator.hpp>

#include <algorithm>
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

constexpr float kEps = 1e-5f;

// Stats op: reduce a row's tiles to (mean, var) - population variance,
// clamped >= 0 (the E3 VAR-moment semantics). Result rides lanes [mean, var].
TilePayload layernorm_stats(const std::vector<TilePayload>& inputs) {
    double sum = 0.0, sumsq = 0.0;
    std::size_t n = 0;
    for (const auto& tile : inputs)
        for (float v : tile.values) { sum += v; sumsq += static_cast<double>(v) * v; ++n; }
    const double mean = n ? sum / n : 0.0;
    const double var = n ? std::max(0.0, sumsq / n - mean * mean) : 0.0;
    return TilePayload{2, 1, {static_cast<float>(mean), static_cast<float>(var)}};
}

// Apply op: y = gamma * (x - mean)/sqrt(var + eps) + beta, elementwise, with
// per-feature gamma/beta slices for this tile. inputs = [x (fed), (mean,var)
// (resident)].
TilePayload layernorm_apply(const std::vector<TilePayload>& inputs,
                            const std::vector<float>& gamma,
                            const std::vector<float>& beta) {
    const auto& x = inputs.at(0);
    const float mean = inputs.at(1).values.at(0);
    const float var = inputs.at(1).values.at(1);
    const float inv = 1.0f / std::sqrt(var + kEps);
    TilePayload out{x.rows, x.cols, std::vector<float>(x.values.size())};
    for (std::size_t i = 0; i < x.values.size(); ++i)
        out.values[i] = gamma[i] * (x.values[i] - mean) * inv + beta[i];
    return out;
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
    const Opt opts[4] = {{"--rows", 1}, {"--len", 512}, {"--tile", 256},
                         {"--max-cycles", 2'000'000}};
    for (int i = 0; i < 4; ++i) {
        vals[i] = arg(argc, argv, opts[i].flag, opts[i].def);
        if (vals[i] <= 0) {
            std::cerr << "error: " << opts[i].flag << " must be a positive integer\n"
                      << "usage: layernorm_simulator [--rows R] [--len N] [--tile T]"
                         " [--max-cycles C]\n";
            return 2;
        }
    }

    try {
    OnlineSoftmaxScheduleGenerator::Config cfg;   // shared row-reduction movement
    cfg.num_rows = static_cast<Size>(vals[0]);
    cfg.reduction_elems = static_cast<Size>(vals[1]);
    cfg.tile_elems = static_cast<Size>(vals[2]);
    cfg.in_base = 0x100000; cfg.stat_base = 0x200000; cfg.out_base = 0x300000;

    auto schedule = OnlineSoftmaxScheduleGenerator(cfg).generate();
    if (!schedule.valid) {
        std::cerr << "schedule refused: " << schedule.error_message << "\n";
        return 1;
    }

    const Size R = cfg.num_rows, F = cfg.reduction_elems, T = cfg.tile_elems;

    // Deterministic input + affine parameters (gamma/beta are per-feature)
    std::vector<float> data(R * F), gamma(F), beta(F);
    for (Size r = 0; r < R; ++r)
        for (Size i = 0; i < F; ++i)
            data[r * F + i] = static_cast<float>(r) - 0.5f + 0.02f * static_cast<float>(i % 40);
    for (Size i = 0; i < F; ++i) {
        gamma[i] = 1.0f + 0.001f * static_cast<float>(i % 16);
        beta[i]  = 0.01f * static_cast<float>(i % 8);
    }

    std::cout << "LayerNorm simulator  —  " << schedule.metadata.strategy
              << "  (" << R << " row(s) x " << F << " features, tile " << T
              << ", eps " << kEps << ")\n\n";

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.max_cycles = static_cast<Cycle>(vals[3]);   // --max-cycles
    ConcurrentTimingExecutor exec(ecfg);

    // Seed the input row tiles (matrix A) from the data
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        if (op.tile.tile_id.matrix != MatrixID::A) continue;
        const Size row = op.tile.tile_id.ti, t = op.tile.tile_id.tj;
        const Size off = row * F + t * T;
        const Size elems = op.tile.height;
        exec.set_tile_payload(op.tile.tile_id,
                              TilePayload{elems, 1,
                                          std::vector<float>(data.begin() + off,
                                                             data.begin() + off + elems)});
    }

    // Enqueue, binding the layernorm value ops. The apply for output tile
    // C[row, t] uses the gamma/beta slice for feature block t.
    for (const auto& op : schedule.operations) {
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
                    spec.operation = layernorm_stats;
                } else {
                    const Size t = op.tile.tile_id.tj;   // feature block
                    const Size off = t * T;
                    const Size len = std::min(T, F - off);
                    std::vector<float> g(gamma.begin() + off, gamma.begin() + off + len);
                    std::vector<float> b(beta.begin() + off, beta.begin() + off + len);
                    spec.operation = [g, b](const std::vector<TilePayload>& in) {
                        return layernorm_apply(in, g, b);
                    };
                }
                exec.schedule_functional_compute(op.tile, spec);
                break;
            }
        }
    }

    // Drive one cycle at a time, tracking every occupancy transition.
    // Softmax/layernorm tiles index (row, feature-tile) in (ti, tj).
    TileTracker::Config tcfg;
    tcfg.label = [](const TileID& id) {
        const char* m = id.matrix == MatrixID::A ? "A"
                      : id.matrix == MatrixID::B ? "B" : "C";
        return std::string(m) + "[" + std::to_string(id.ti) + "," +
               std::to_string(id.tj) + "]";
    };
    TileTracker tracker(tcfg);
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

    // Correctness: every output element matches the host LayerNorm reference.
    // Fail closed - a NaN, a missing tile, or a duplicated store must not pass.
    std::cout << "Result: y = gamma * (x - mean)/sqrt(var + eps) + beta\n";
    double max_err = 0.0;
    bool finite_ok = true;
    std::vector<int> covered(static_cast<std::size_t>(R) * F, 0);  // times each element was stored
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        if (id.matrix != MatrixID::C) continue;
        const Size r = id.ti, off = id.tj * T;
        // host mean/var for this row
        double sum = 0.0, sumsq = 0.0;
        for (Size i = 0; i < F; ++i) { sum += data[r * F + i]; sumsq += static_cast<double>(data[r*F+i]) * data[r*F+i]; }
        const double mean = sum / F, var = std::max(0.0, sumsq / F - mean * mean);
        const double inv = 1.0 / std::sqrt(var + kEps);
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (std::size_t i = 0; i < p.values.size(); ++i) {
            if (!std::isfinite(p.values[i])) finite_ok = false;
            const double want = gamma[off + i] * (data[r * F + off + i] - mean) * inv + beta[off + i];
            max_err = std::max(max_err, std::abs(static_cast<double>(p.values[i]) - want));
            ++covered[r * F + off + i];
        }
    }
    // Every output element must be stored exactly once
    bool coverage_ok = true;
    for (int c : covered) if (c != 1) { coverage_ok = false; break; }

    const bool ok = finite_ok && coverage_ok && max_err < 1e-3;
    std::cout << "  max abs error vs host oracle: " << max_err
              << (finite_ok ? "" : "  [non-finite output]")
              << (coverage_ok ? "" : "  [missing/duplicate output tile]")
              << (ok ? "  [OK]" : "  [FAIL]") << "\n";
    std::cout << "\n" << (ok ? "LAYERNORM OK" : "LAYERNORM MISMATCH") << "\n";
    return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
