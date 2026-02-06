// ============================================================================
// examples/schedule/csp_schedule_demo.cpp
// Demonstrate CSP-style schedule generation for DNN operations
//
// This example shows how to generate concurrent, livelock-safe schedules
// for common DNN operations using the credit-based dataflow model.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/conv2d_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/softmax_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/layernorm_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/batchnorm_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_validator.hpp>

#include <iostream>
#include <iomanip>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

// ============================================================================
// Helper Functions
// ============================================================================

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void print_schedule_info(const ScheduleResult& schedule, const std::string& desc) {
    std::cout << "\n" << desc << std::endl;
    std::cout << "  Name: " << schedule.metadata.name << std::endl;
    std::cout << "  Strategy: " << schedule.metadata.strategy << std::endl;
    std::cout << "  Total operations: " << schedule.size() << std::endl;
    std::cout << "  Operation breakdown:" << std::endl;
    std::cout << "    LOAD:      " << std::setw(6) << schedule.count_ops(ScheduleOpType::LOAD) << std::endl;
    std::cout << "    MOVE:      " << std::setw(6) << schedule.count_ops(ScheduleOpType::MOVE) << std::endl;
    std::cout << "    FEED:      " << std::setw(6) << schedule.count_ops(ScheduleOpType::FEED) << std::endl;
    std::cout << "    DRAIN:     " << std::setw(6) << schedule.count_ops(ScheduleOpType::DRAIN) << std::endl;
    std::cout << "    WRITEBACK: " << std::setw(6) << schedule.count_ops(ScheduleOpType::WRITEBACK) << std::endl;
    std::cout << "    STORE:     " << std::setw(6) << schedule.count_ops(ScheduleOpType::STORE) << std::endl;

    auto analysis = ScheduleAnalysis::analyze(schedule);
    std::cout << "  Livelock analysis:" << std::endl;
    std::cout << "    Max consecutive A: " << analysis.max_consecutive_a << std::endl;
    std::cout << "    Max consecutive B: " << analysis.max_consecutive_b << std::endl;
    std::cout << "    Interleaved: " << (analysis.is_interleaved ? "YES" : "NO") << std::endl;

    auto validation = validate_schedule(schedule);
    std::cout << "  Validation: " << (validation.valid ? "PASSED" : "FAILED");
    if (validation.count_warnings() > 0) {
        std::cout << " (" << validation.count_warnings() << " warnings)";
    }
    std::cout << std::endl;
}

// ============================================================================
// MatMul Demonstration
// ============================================================================

void demo_matmul() {
    print_header("MatMul CSP Schedules");

    // Small MatMul
    {
        MatMulScheduleGenerator::Config config;
        config.M = 64; config.N = 64; config.K = 64;
        config.Ti = 16; config.Tj = 16; config.Tk = 16;
        config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

        MatMulScheduleGenerator gen(config);
        print_schedule_info(gen.generate(), "Small MatMul: C[64,64] = A[64,64] x B[64,64]");
    }

    // Large MatMul (like transformer FFN)
    {
        MatMulScheduleGenerator::Config config;
        config.M = 512; config.N = 2048; config.K = 512;
        config.Ti = 16; config.Tj = 16; config.Tk = 16;
        config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

        MatMulScheduleGenerator gen(config);
        print_schedule_info(gen.generate(), "Large MatMul (FFN): C[512,2048] = A[512,512] x B[512,2048]");
    }

    // Compare strategies
    std::cout << "\n  Strategy comparison for C[128,128] = A[128,64] x B[64,128]:" << std::endl;

    MatMulScheduleGenerator::Config base_config;
    base_config.M = 128; base_config.N = 128; base_config.K = 64;

    base_config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;
    auto interleaved = MatMulScheduleGenerator(base_config).generate();

    base_config.strategy = MatMulScheduleGenerator::Strategy::OUTPUT_STATIONARY;
    auto output_stat = MatMulScheduleGenerator(base_config).generate();

    base_config.strategy = MatMulScheduleGenerator::Strategy::BLOCKED_AB;
    auto blocked = MatMulScheduleGenerator(base_config).generate();

    std::cout << "    INTERLEAVED_AB:    " << interleaved.size() << " ops, "
              << "max_consec_A=" << ScheduleAnalysis::analyze(interleaved).max_consecutive_a
              << ", livelock-safe=" << (is_livelock_safe(interleaved) ? "YES" : "NO") << std::endl;

    std::cout << "    OUTPUT_STATIONARY: " << output_stat.size() << " ops, "
              << "max_consec_A=" << ScheduleAnalysis::analyze(output_stat).max_consecutive_a
              << ", livelock-safe=" << (is_livelock_safe(output_stat) ? "YES" : "NO") << std::endl;

    std::cout << "    BLOCKED_AB:        " << blocked.size() << " ops, "
              << "max_consec_A=" << ScheduleAnalysis::analyze(blocked).max_consecutive_a
              << ", livelock-safe=" << (is_livelock_safe(blocked) ? "YES" : "NO") << std::endl;
}

// ============================================================================
// Conv2D Demonstration
// ============================================================================

void demo_conv2d() {
    print_header("Conv2D CSP Schedules (im2col)");

    // ResNet conv layer
    {
        Conv2DScheduleGenerator::Config config;
        config.N = 1;
        config.H_in = 56; config.W_in = 56; config.C_in = 64;
        config.C_out = 64;
        config.Kh = 3; config.Kw = 3;
        config.stride_h = 1; config.stride_w = 1;
        config.padding_h = 1; config.padding_w = 1;
        config.Ti = 16; config.Tj = 16; config.Tk = 16;

        Conv2DScheduleGenerator gen(config);
        auto schedule = gen.generate();

        std::cout << "\n  ResNet-50 Layer: 56x56x64 -> 56x56x64 (3x3)" << std::endl;
        std::cout << "    Output shape: " << config.H_out() << "x" << config.W_out() << "x" << config.C_out << std::endl;
        std::cout << "    Im2col matrix: [" << config.M() << " x " << config.K() << "] x [" << config.K() << " x " << config.C_out << "]" << std::endl;
        std::cout << "    Total operations: " << schedule.size() << std::endl;
        std::cout << "    Livelock-safe: " << (is_livelock_safe(schedule) ? "YES" : "NO") << std::endl;
    }

    // VGG-style conv
    {
        Conv2DScheduleGenerator::Config config;
        config.N = 1;
        config.H_in = 224; config.W_in = 224; config.C_in = 3;
        config.C_out = 64;
        config.Kh = 3; config.Kw = 3;
        config.stride_h = 1; config.stride_w = 1;
        config.padding_h = 1; config.padding_w = 1;
        config.Ti = 16; config.Tj = 16; config.Tk = 16;

        Conv2DScheduleGenerator gen(config);
        auto schedule = gen.generate();

        std::cout << "\n  VGG-style first layer: 224x224x3 -> 224x224x64 (3x3)" << std::endl;
        std::cout << "    Output shape: " << config.H_out() << "x" << config.W_out() << "x" << config.C_out << std::endl;
        std::cout << "    Total operations: " << schedule.size() << std::endl;
        std::cout << "    Livelock-safe: " << (is_livelock_safe(schedule) ? "YES" : "NO") << std::endl;
    }
}

// ============================================================================
// Softmax Demonstration
// ============================================================================

void demo_softmax() {
    print_header("Softmax CSP Schedule (multi-pass)");

    SoftmaxScheduleGenerator::Config config;
    config.batch_size = 32;
    config.reduction_dim = 1024;
    config.Ti = 16; config.Tj = 16;

    SoftmaxScheduleGenerator gen(config);
    auto schedule = gen.generate();

    std::cout << "\n  Transformer attention softmax: [32 x 1024]" << std::endl;
    std::cout << "    Multi-pass algorithm:" << std::endl;
    std::cout << "      Pass 1: Find max (numerical stability)" << std::endl;
    std::cout << "      Pass 2: Compute exp(x - max)" << std::endl;
    std::cout << "      Pass 3: Sum exponentials" << std::endl;
    std::cout << "      Pass 4: Normalize (divide by sum)" << std::endl;
    std::cout << "    Total operations: " << schedule.size() << std::endl;
    std::cout << "    Validation: " << (validate_schedule(schedule).valid ? "PASSED" : "FAILED") << std::endl;
}

// ============================================================================
// LayerNorm Demonstration
// ============================================================================

void demo_layernorm() {
    print_header("LayerNorm CSP Schedule (multi-pass)");

    LayerNormScheduleGenerator::Config config;
    config.batch_size = 32;
    config.sequence_length = 512;
    config.hidden_size = 768;
    config.Ti = 16; config.Tj = 16;
    config.affine = true;

    LayerNormScheduleGenerator gen(config);
    auto schedule = gen.generate();

    std::cout << "\n  BERT LayerNorm: [32 x 512 x 768]" << std::endl;
    std::cout << "    Normalized over hidden_size=768" << std::endl;
    std::cout << "    Multi-pass algorithm:" << std::endl;
    std::cout << "      Pass 1: Compute mean" << std::endl;
    std::cout << "      Pass 2: Compute variance" << std::endl;
    std::cout << "      Pass 3: Normalize + affine (gamma, beta)" << std::endl;
    std::cout << "    Total operations: " << schedule.size() << std::endl;
    std::cout << "    Validation: " << (validate_schedule(schedule).valid ? "PASSED" : "FAILED") << std::endl;
}

// ============================================================================
// BatchNorm Demonstration
// ============================================================================

void demo_batchnorm() {
    print_header("BatchNorm CSP Schedule");

    // Inference mode
    {
        BatchNormScheduleGenerator::Config config;
        config.N = 32;
        config.C = 64;
        config.H = 56; config.W = 56;
        config.Ti = 16;
        config.training = false;

        BatchNormScheduleGenerator gen(config);
        auto schedule = gen.generate();

        std::cout << "\n  ResNet BatchNorm (inference): [32 x 64 x 56 x 56]" << std::endl;
        std::cout << "    Uses pre-computed running_mean and running_var" << std::endl;
        std::cout << "    Single-pass: (x - mean) * rsqrt(var + eps) * gamma + beta" << std::endl;
        std::cout << "    Total operations: " << schedule.size() << std::endl;
        std::cout << "    Validation: " << (validate_schedule(schedule).valid ? "PASSED" : "FAILED") << std::endl;
    }

    // Training mode
    {
        BatchNormScheduleGenerator::Config config;
        config.N = 32;
        config.C = 64;
        config.H = 56; config.W = 56;
        config.Ti = 16;
        config.training = true;

        BatchNormScheduleGenerator gen(config);
        auto schedule = gen.generate();

        std::cout << "\n  ResNet BatchNorm (training): [32 x 64 x 56 x 56]" << std::endl;
        std::cout << "    Computes batch statistics on-the-fly" << std::endl;
        std::cout << "    Multi-pass: mean, variance, normalize" << std::endl;
        std::cout << "    Total operations: " << schedule.size() << std::endl;
        std::cout << "    Validation: " << (validate_schedule(schedule).valid ? "PASSED" : "FAILED") << std::endl;
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     CSP-Style Concurrent Timing Schedules for DNN Operations         ║\n";
    std::cout << "║                                                                      ║\n";
    std::cout << "║     Features:                                                        ║\n";
    std::cout << "║     - Credit-based dataflow (no cache semantics)                     ║\n";
    std::cout << "║     - Livelock-safe interleaved scheduling                           ║\n";
    std::cout << "║     - Static validation before execution                             ║\n";
    std::cout << "║     - Multi-pass support for reduction operations                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    demo_matmul();
    demo_conv2d();
    demo_softmax();
    demo_layernorm();
    demo_batchnorm();

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "All CSP schedules generated and validated successfully!" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    return 0;
}
