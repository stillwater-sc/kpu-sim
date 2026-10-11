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
#include <fstream>
#include <sstream>
#include <stdexcept>
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


// THE OPERATORS ARE CSP PROGRAMS (kpu-run-csp-programs step 4d.2): a 32x32x32 GEMM in 16x16
// tiles, column-panel order (what kpu-csp-gen writes), in three forms that differ only in what
// happens to B across operators:
//
//   cold     loads B per column panel and releases it          L3 5  (B 2 + A 2 + C 1)
//   retain   loads all of B first, and RETAINS it at the end    L3 7  (B 4 + A 2 + C 1)
//   inherit  INHERITS all of B, and releases it at the end      L3 7, 4 of them held already
//
// Every form loads A per (j, i): 8 tiles. Cold and retain also load B's 4; inherit does not.
inline std::string gemm_csp(const char* name, const char* b_mode) {
    const std::string mode = b_mode;
    const bool cold = mode == "cold";
    std::string t = "csp 1.0\nprogram " + std::string(name) + " machine flat(l3 = " +
                    (cold ? "5" : "7") + ") {\n"
                    "  tensor A[32,32] tile 16x16 in;\n"
                    "  tensor B[32,32] tile 16x16 in;\n"
                    "  tensor C[32,32] tile 16x16 out;\n";
    if (mode == "retain") t += "  resident B[:, :];\n";
    if (mode == "inherit") t += "  inherit B[:, :];\n";
    t += "  for j in 0..2 {\n";
    if (cold) t += "    resident B[:, j];\n";
    t += "    for i in 0..2 {\n"
         "      resident A[i, :];\n"
         "      acc C[i, j] in fabric {\n"
         "        for k in 0..2 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n"
         "      }\n"
         "      store C[i, j];\n"
         "      release A[i, :];\n"
         "    }\n";
    if (cold) t += "    release B[:, j];\n";
    t += "  }\n";
    if (mode == "retain") t += "  retain B[:, :];\n";
    if (mode == "inherit") t += "  release B[:, :];\n";
    return t + "}\n";
}

inline Operator gemm(const char* name, const char* b_mode, std::vector<std::string> in,
                     std::vector<std::string> out) {
    return sw::kpu::loadable::csp_operator(name, gemm_csp(name, b_mode), std::move(in), std::move(out));
}

inline TensorRef tensor(const char* name, std::uint64_t addr, bool is_input,
                        std::vector<std::uint64_t> shape = {32, 32},
                        std::vector<std::uint64_t> tile = {16, 16}) {
    TensorRef t;
    t.name = name;
    t.shape = std::move(shape);
    t.tile_shape = std::move(tile);
    std::uint64_t n = 4;
    for (std::uint64_t d : t.shape) n *= d;
    t.device_address = addr;
    t.size_bytes = n;
    if (is_input) {
        t.source_uri = "weights.bin";
        t.source_offset = addr;
        t.source_length = n;
        t.content_digest = "declared00000000";
    }
    return t;
}

// TWO GEMMS SHARING A WEIGHT TENSOR. The shared input is what makes statefulness observable,
// and it is also the real case: a weight tensor read by layer after layer. `warm` writes the
// sharing into the programs -- the first retains W, the second inherits it -- and cold does
// not, so the two loadables differ in residency alone.
//
// NO TENSOR IS NAMED LIKE AN OPERAND, deliberately. The kernel's operands are A/B/C whatever
// the model calls them, so a fixture whose tensors are also A/B/C cannot tell a tensor-keyed
// bug from an operand-keyed one: both spellings agree by coincidence, and a review finding
// about exactly that confusion reproduced as "all tests pass". X/W/H/Y are disjoint from
// A/B/C, so any place the two vocabularies are mixed up now shows up as a wrong answer or a
// missed reuse.
inline Loadable two_gemms_sharing_weights(bool warm = true) {
    Loadable l;
    l.name = warm ? "gemm-gemm-shared-weights" : "gemm-gemm-cold";
    // X = activations in, W = the shared weights, H = the hidden result, Y = the output.
    l.tensors = {tensor("X", 0x1000, true), tensor("W", 0x2000, true),
                 tensor("H", 0x3000, false), tensor("Y", 0x4000, false)};
    // The second GEMM consumes the first's output and reads W AGAIN. Both programs call their
    // operands A/B/C, and the binding maps those onto different TENSORS. Without that
    // indirection both operators would read and write the same tensor, the second would
    // accumulate onto the first's result, and the hidden result would come out doubled. Which
    // is exactly what happened once.
    l.operators = {gemm("gemm0", warm ? "retain" : "cold", {"X", "W"}, {"H"}),
                   gemm("gemm1", warm ? "inherit" : "cold", {"H", "W"}, {"Y"})};
    l.profile.min_compute_tiles = 1;
    return l;
}

// THREE OPERATORS WITH A GAP: W is read by the first and the THIRD, and not by the second.
// That gap is the case a two-operator chain cannot produce -- during the middle run the device
// holds four W tiles that the middle program never mentions. They still occupy L3, so the
// middle operator runs in what is left.
inline Loadable three_gemms_with_a_gap(bool warm = true) {
    Loadable l;
    l.name = warm ? "gemm-gemm-gemm-reuse-across-a-gap" : "gemm-gemm-gemm-cold";
    l.tensors = {tensor("X", 0x1000, true), tensor("W", 0x2000, true),
                 tensor("V", 0x5000, true), tensor("H", 0x3000, false),
                 tensor("G", 0x6000, false), tensor("Y", 0x4000, false)};
    l.operators = {gemm("gemm0", warm ? "retain" : "cold", {"X", "W"}, {"H"}),
                   gemm("gemm1", "cold", {"H", "V"}, {"G"}),               // does not mention W
                   gemm("gemm2", warm ? "inherit" : "cold", {"G", "W"}, {"Y"})};   // wants it again
    l.profile.min_compute_tiles = 1;
    return l;
}

inline std::string read_csp(const std::string& name) {
    std::ifstream in("tests/program/csp/" + name, std::ios::binary);
    if (!in) throw std::runtime_error("no fixture tests/program/csp/" + name);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

// A RESULT KEPT IN L3 AND NEVER STORED BY ITS PRODUCER: H = relu(X W) is retained by the
// first operator (tests/program/csp/matmul_relu_retain_32_t16.csp) and inherited by the second
// (bias_inherit_32_t16.csp), which adds a bias in place and stores it -- the store the chain
// owes. H's DRAM copy is untouched between the two.
inline Loadable relu_then_bias() {
    Loadable l;
    l.name = "matmul-relu-then-bias";
    l.tensors = {tensor("X", 0x1000, true), tensor("W", 0x2000, true),
                 tensor("bias", 0x5000, true, {32}, {16}), tensor("H", 0x3000, false)};
    l.operators = {
        sw::kpu::loadable::csp_operator("matmul_relu", read_csp("matmul_relu_retain_32_t16.csp"),
                                        {"X", "W"}, {"H"}),
        // H is inout here: the program reads it (inherits it) and stores it. No `out` operand.
        sw::kpu::loadable::csp_operator("bias", read_csp("bias_inherit_32_t16.csp"), {"H", "bias"}, {})};
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
    if (s.has("bias")) {
        auto& v = s.values("bias");
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.5f * float(i % 4) - 0.75f;
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
