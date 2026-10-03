// ============================================================================
// tests/program/orchestration_fixtures.hpp
// Loadables and helpers shared by the orchestration suites (#305 increments 2-3).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/orchestration/orchestrator.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace orchestration_fixtures {

using namespace sw::kpu::orchestration;
using sw::kpu::loadable::Loadable;
using sw::kpu::loadable::Operator;
using sw::kpu::loadable::TensorRef;
using sw::kpu::program::TileProgram;
using sw::kpu::program::driver::DeviceSpec;
using sw::kpu::program::driver::ExecutionLevel;
using sw::kpu::program::platform::VirtualPlatform;


inline std::string l0_matmul(unsigned n, unsigned t) {
    sw::kpu::program::driver::ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = n;
    ps.tile = t;
    return sw::kpu::program::serialize::to_string(sw::kpu::program::driver::derive(ps));
}

// TWO GEMMS SHARING A WEIGHT TENSOR. The shared input is what makes statefulness observable,
// and it is also the real case: a weight tensor read by layer after layer. The first operator's
// OUTPUT would not serve — after a writeback it is in DRAM, so keeping it "resident" would mean
// re-fetching it, which saves nothing.
//
// NO TENSOR IS NAMED LIKE AN OPERAND, deliberately. The kernel's operands are A/B/C whatever
// the model calls them, so a fixture whose tensors are also A/B/C cannot tell a tensor-keyed
// bug from an operand-keyed one: both spellings agree by coincidence, and a review finding
// about exactly that confusion reproduced as "all tests pass". X/W/H/Y are disjoint from
// A/B/C, so any place the two vocabularies are mixed up now shows up as a wrong answer or a
// missed reuse.
inline Loadable two_gemms_sharing_weights() {
    Loadable l;
    l.name = "gemm-gemm-shared-weights";

    auto tensor = [](const char* name, std::uint64_t addr, bool is_input) {
        TensorRef t;
        t.name = name;
        t.shape = {32, 32};
        t.tile_shape = {16, 16};
        t.device_address = addr;
        t.size_bytes = 32 * 32 * 4;
        if (is_input) {
            t.source_uri = "weights.bin";
            t.source_offset = addr;
            t.source_length = 32 * 32 * 4;
            t.content_digest = "declared00000000";
        }
        return t;
    };
    // X = activations in, W = the shared weights, H = the hidden result, Y = the output.
    l.tensors = {tensor("X", 0x1000, true), tensor("W", 0x2000, true),
                 tensor("H", 0x3000, false), tensor("Y", 0x4000, false)};

    Operator first;
    first.name = "gemm0";
    first.l0_program = l0_matmul(32, 16);
    first.inputs = {"X", "W"};        // -> the program's operands A, B
    first.outputs = {"H"};            // -> its operand C

    // The second GEMM consumes the first's output and reads W AGAIN. Its L0 program uses the
    // same operand names -- A/B/C, because that is what the kernel calls them -- and the
    // binding maps those onto different TENSORS. Without that indirection both operators
    // would read and write the same tensor, the second would accumulate onto the first's
    // result, and the hidden result would come out doubled. Which is exactly what happened.
    Operator second;
    second.name = "gemm1";
    second.l0_program = l0_matmul(32, 16);
    second.inputs = {"H", "W"};       // H from gemm0, and W shared -> operands A, B
    second.outputs = {"Y"};

    l.operators = {first, second};
    l.profile.min_compute_tiles = 1;
    return l;
}

// THREE OPERATORS WITH A GAP: W is read by the first and the THIRD, and not by the second.
// That gap is the case a two-operator chain cannot produce -- during the middle run the
// orchestrator holds four W tiles that the middle program never mentions, so they have no
// operand name and therefore no tile key in its vocabulary. They still occupy L3.
inline Loadable three_gemms_with_a_gap() {
    Loadable l;
    l.name = "gemm-gemm-gemm-reuse-across-a-gap";

    auto tensor = [](const char* name, std::uint64_t addr, bool is_input) {
        TensorRef t;
        t.name = name;
        t.shape = {32, 32};
        t.tile_shape = {16, 16};
        t.device_address = addr;
        t.size_bytes = 32 * 32 * 4;
        if (is_input) {
            t.source_uri = "weights.bin";
            t.source_offset = addr;
            t.source_length = 32 * 32 * 4;
            t.content_digest = "declared00000000";
        }
        return t;
    };
    l.tensors = {tensor("X", 0x1000, true), tensor("W", 0x2000, true),
                 tensor("V", 0x5000, true), tensor("H", 0x3000, false),
                 tensor("G", 0x6000, false), tensor("Y", 0x4000, false)};

    auto gemm = [](const char* name, std::vector<std::string> in, std::vector<std::string> out) {
        Operator o;
        o.name = name;
        o.l0_program = l0_matmul(32, 16);
        o.inputs = std::move(in);
        o.outputs = std::move(out);
        return o;
    };
    l.operators = {gemm("gemm0", {"X", "W"}, {"H"}),
                   gemm("gemm1", {"H", "V"}, {"G"}),     // does not mention W
                   gemm("gemm2", {"G", "W"}, {"Y"})};    // wants it again
    l.profile.min_compute_tiles = 1;
    return l;
}

inline void fill_inputs(TensorStore& s) {
    auto& a = s.values("X");
    auto& b = s.values("W");
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(i % 7) - 3.0f;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(i % 5) - 2.0f;
    if (s.has("V")) {
        auto& v = s.values("V");
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = float(i % 3) - 1.0f;
    }
}

inline VirtualPlatform fresh(std::uint32_t l3_tiles = 0) {
    DeviceSpec ds;
    ds.l3_tiles = l3_tiles;
    return VirtualPlatform(sw::kpu::program::driver::make_deployment(ds));
}

inline bool bit_identical(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    return true;
}

} // namespace orchestration_fixtures
