#pragma once
// ============================================================================
// Type Dispatch Mechanism for KPU Compute Fabric
// ============================================================================
//
// This header provides the runtime-to-compile-time type dispatch mechanism
// for the BehavioralComputeFabric. It bridges the gap between:
//   - Runtime: DataType enum from descriptor (user-specified)
//   - Compile-time: Template instantiation of typed kernels
//
// DESIGN PHILOSOPHY:
// ──────────────────
// The dispatch mechanism follows a "switch-dispatch" pattern that:
// 1. Maps runtime DataType to compile-time type parameters
// 2. Invokes the appropriate template kernel instantiation
// 3. Handles accumulator type selection via ScalarTraits
//
// This approach ensures:
// - Type safety at compile time (kernel templates verify types)
// - Runtime flexibility (user can specify any supported dtype)
// - Optimal code generation (compiler optimizes each instantiation)
//
// SUPPORTED TYPE PATHS:
// ─────────────────────
// ┌────────────────┬─────────────────────┬────────────────────────────────────┐
// │ DataType       │ Scalar Type         │ Accumulator Type                   │
// ├────────────────┼─────────────────────┼────────────────────────────────────┤
// │ FLOAT32        │ float               │ float                              │
// │ FLOAT16        │ fp16_t (Universal)  │ float                              │
// │ BFLOAT16       │ bf16_t (Universal)  │ float                              │
// │ FP8_E4M3       │ fp8e4m3_t           │ float                              │
// │ FP8_E5M2       │ fp8e5m2_t           │ float                              │
// │ FP8_E3M4       │ fp8e3m4_t           │ float                              │
// │ FP8_E2M5       │ fp8e2m5_t           │ float                              │
// │ FP4            │ fp4_t               │ float                              │
// │ INT8           │ int8_t              │ int32_t                            │
// │ INT4           │ int8_t (unpacked)*  │ int32_t                            │
// └────────────────┴─────────────────────┴────────────────────────────────────┘
//
// *INT4 is stored packed (2 per byte) but unpacked to int8_t for compute.
//  The Streamer handles pack/unpack. See compute_fabric_interface.hpp for details.
//
// USAGE IN BEHAVIORAL COMPUTE FABRIC:
// ───────────────────────────────────
//   void submit_matmul(const MatMulDescriptor& desc, ...) {
//       dispatch_matmul(desc.dtype, desc.m, desc.n, desc.k,
//                       a_data, b_data, c_data, desc.accumulate);
//   }
//
// ============================================================================

#include <cstdint>
#include <stdexcept>
#include <sw/kpu/data_types.hpp>
#include <sw/kpu/quantization/scalar_traits.hpp>
#include <sw/kpu/quantization/universal_types.hpp>
#include <sw/kpu/quantization/kernels.hpp>

namespace sw::kpu {

// ============================================================================
// Type-Specific Execution Functions
// ============================================================================
// These functions execute matmul with specific types. They are called by
// the dispatch mechanism below.

namespace detail {

/// Execute matmul with explicit scalar and accumulator types
template<typename Scalar, typename Accumulator>
void execute_typed_matmul(
    uint32_t m, uint32_t n, uint32_t k,
    const void* a_data, const void* b_data, void* c_data,
    bool accumulate)
{
    using namespace kernels;

    // Create tensor views with appropriate types
    auto a_ptr = static_cast<const Scalar*>(a_data);
    auto b_ptr = static_cast<const Scalar*>(b_data);
    auto c_ptr = static_cast<Accumulator*>(c_data);

    TensorView<const Scalar> A(const_cast<Scalar*>(a_ptr), m, k);
    TensorView<const Scalar> B(const_cast<Scalar*>(b_ptr), k, n);
    TensorView<Accumulator> C(c_ptr, m, n);

    Accumulator alpha = Accumulator(1);
    Accumulator beta = accumulate ? Accumulator(1) : Accumulator(0);

    matmul<Scalar, Accumulator>(A, B, C, alpha, beta);
}

/// Execute ReLU with explicit scalar type
template<typename Scalar>
void execute_typed_relu(const void* input, void* output, uint64_t count) {
    kernels::relu(static_cast<const Scalar*>(input),
                  static_cast<Scalar*>(output),
                  static_cast<size_t>(count));
}

/// Execute GELU with explicit scalar type
template<typename Scalar>
void execute_typed_gelu(const void* input, void* output, uint64_t count) {
    kernels::gelu(static_cast<const Scalar*>(input),
                  static_cast<Scalar*>(output),
                  static_cast<size_t>(count));
}

/// Execute SiLU with explicit scalar type
template<typename Scalar>
void execute_typed_silu(const void* input, void* output, uint64_t count) {
    kernels::silu(static_cast<const Scalar*>(input),
                  static_cast<Scalar*>(output),
                  static_cast<size_t>(count));
}

/// Execute elementwise add with explicit scalar type
template<typename Scalar>
void execute_typed_add(const void* a, const void* b, void* output, uint64_t count) {
    kernels::add(static_cast<const Scalar*>(a),
                 static_cast<const Scalar*>(b),
                 static_cast<Scalar*>(output),
                 static_cast<size_t>(count));
}

/// Execute elementwise mul with explicit scalar type
template<typename Scalar>
void execute_typed_mul(const void* a, const void* b, void* output, uint64_t count) {
    kernels::mul(static_cast<const Scalar*>(a),
                 static_cast<const Scalar*>(b),
                 static_cast<Scalar*>(output),
                 static_cast<size_t>(count));
}

/// Execute softmax with explicit scalar type
template<typename Scalar>
void execute_typed_softmax(const void* input, void* output,
                           uint32_t batch_size, uint32_t dim_size) {
    kernels::softmax(static_cast<const Scalar*>(input),
                     static_cast<Scalar*>(output),
                     batch_size, dim_size);
}

/// Execute layer normalization with explicit scalar type
template<typename Scalar>
void execute_typed_layernorm(
    const void* input, void* output,
    const void* gamma, const void* beta,
    uint32_t batch_size, uint32_t dim, float eps)
{
    kernels::layer_norm(static_cast<const Scalar*>(input),
                        static_cast<Scalar*>(output),
                        static_cast<const Scalar*>(gamma),
                        static_cast<const Scalar*>(beta),
                        batch_size, dim, eps);
}

} // namespace detail

// ============================================================================
// Runtime Type Dispatch Functions
// ============================================================================
// These functions dispatch from runtime DataType to compile-time templates.

/// Dispatch matmul based on DataType
///
/// This function bridges the runtime DataType enum to compile-time template
/// instantiations. It handles:
/// 1. Native types (float, int8_t)
/// 2. Universal library types (fp16_t, bf16_t, fp8e4m3_t, etc.)
/// 3. Automatic accumulator selection via ScalarTraits
///
/// @param dtype The data type from the MatMulDescriptor
/// @param m, n, k Matrix dimensions
/// @param a_data, b_data, c_data Pointers to matrix data
/// @param accumulate If true, adds to existing C; otherwise overwrites
///
/// @throws std::invalid_argument if dtype is not supported
inline void dispatch_matmul(
    DataType dtype,
    uint32_t m, uint32_t n, uint32_t k,
    const void* a_data, const void* b_data, void* c_data,
    bool accumulate)
{
    // ==========================================================================
    // TYPE DISPATCH DESIGN
    // ==========================================================================
    // This switch statement maps runtime DataType to compile-time templates.
    //
    // For each supported type:
    // 1. The Scalar type determines input data interpretation
    // 2. The Accumulator type (from ScalarTraits) determines output precision
    //
    // The Universal library provides software emulation for:
    // - fp16_t: IEEE 754 half-precision (5 exp bits, 10 mantissa bits)
    // - bf16_t: Brain float (8 exp bits, 7 mantissa bits)
    // - fp8e4m3_t: NVIDIA FP8 (4 exp, 3 mantissa)
    // - fp8e5m2_t: IEEE-like FP8 (5 exp, 2 mantissa)
    // - fp8e3m4_t, fp8e2m5_t: Alternative FP8 formats
    // - fp4_t: Ultra-low precision (1 exp, 2 mantissa)
    //
    // For quantized integer types (INT8, INT4), the accumulator is INT32
    // to prevent overflow during MAC operations.
    // ==========================================================================

    switch (dtype) {
        // ======================================================================
        // NATIVE TYPES
        // ======================================================================
        case DataType::FLOAT32:
            detail::execute_typed_matmul<float, float>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        // ======================================================================
        // UNIVERSAL LIBRARY TYPES - 16-bit floats
        // ======================================================================
        case DataType::FLOAT16:
            detail::execute_typed_matmul<fp16_t, float>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        case DataType::BFLOAT16:
            detail::execute_typed_matmul<bf16_t, float>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        // ======================================================================
        // UNIVERSAL LIBRARY TYPES - 8-bit floats (FP8 variants)
        // ======================================================================
        case DataType::FP8_E4M3:
            detail::execute_typed_matmul<fp8e4m3_t, float>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        case DataType::FP8_E5M2:
            detail::execute_typed_matmul<fp8e5m2_t, float>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        case DataType::FP8_E3M4:
            detail::execute_typed_matmul<fp8e3m4_t, float>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        case DataType::FP8_E2M5:
            detail::execute_typed_matmul<fp8e2m5_t, float>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        // ======================================================================
        // UNIVERSAL LIBRARY TYPES - 4-bit float
        // ======================================================================
        case DataType::FP4:
            detail::execute_typed_matmul<fp4_t, float>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        // ======================================================================
        // INTEGER QUANTIZED TYPES
        // ======================================================================
        case DataType::INT8:
            // INT8 inputs with INT32 accumulator to prevent overflow
            detail::execute_typed_matmul<int8_t, int32_t>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        case DataType::INT4:
            // INT4 is stored packed but computed unpacked as int8_t
            // The Streamer unpacks INT4 pairs before compute.
            // Here we assume data is already unpacked to int8_t values.
            detail::execute_typed_matmul<int8_t, int32_t>(
                m, n, k, a_data, b_data, c_data, accumulate);
            break;

        // ======================================================================
        // UNSUPPORTED TYPES
        // ======================================================================
        case DataType::INT16:
        case DataType::INT32:
        case DataType::UINT8:
        default:
            throw std::invalid_argument(
                "Unsupported DataType for matmul dispatch: " + dtype_name(dtype));
    }
}

/// Dispatch elementwise ReLU based on DataType
inline void dispatch_relu(DataType dtype, const void* input, void* output, uint64_t count) {
    switch (dtype) {
        case DataType::FLOAT32:
            detail::execute_typed_relu<float>(input, output, count);
            break;
        case DataType::FLOAT16:
            detail::execute_typed_relu<fp16_t>(input, output, count);
            break;
        case DataType::BFLOAT16:
            detail::execute_typed_relu<bf16_t>(input, output, count);
            break;
        case DataType::FP8_E4M3:
            detail::execute_typed_relu<fp8e4m3_t>(input, output, count);
            break;
        case DataType::FP8_E5M2:
            detail::execute_typed_relu<fp8e5m2_t>(input, output, count);
            break;
        case DataType::FP8_E3M4:
            detail::execute_typed_relu<fp8e3m4_t>(input, output, count);
            break;
        case DataType::FP8_E2M5:
            detail::execute_typed_relu<fp8e2m5_t>(input, output, count);
            break;
        case DataType::FP4:
            detail::execute_typed_relu<fp4_t>(input, output, count);
            break;
        default:
            throw std::invalid_argument(
                "Unsupported DataType for ReLU dispatch: " + dtype_name(dtype));
    }
}

/// Dispatch elementwise GELU based on DataType
inline void dispatch_gelu(DataType dtype, const void* input, void* output, uint64_t count) {
    switch (dtype) {
        case DataType::FLOAT32:
            detail::execute_typed_gelu<float>(input, output, count);
            break;
        case DataType::FLOAT16:
            detail::execute_typed_gelu<fp16_t>(input, output, count);
            break;
        case DataType::BFLOAT16:
            detail::execute_typed_gelu<bf16_t>(input, output, count);
            break;
        case DataType::FP8_E4M3:
            detail::execute_typed_gelu<fp8e4m3_t>(input, output, count);
            break;
        case DataType::FP8_E5M2:
            detail::execute_typed_gelu<fp8e5m2_t>(input, output, count);
            break;
        default:
            throw std::invalid_argument(
                "Unsupported DataType for GELU dispatch: " + dtype_name(dtype));
    }
}

/// Dispatch elementwise SiLU based on DataType
inline void dispatch_silu(DataType dtype, const void* input, void* output, uint64_t count) {
    switch (dtype) {
        case DataType::FLOAT32:
            detail::execute_typed_silu<float>(input, output, count);
            break;
        case DataType::FLOAT16:
            detail::execute_typed_silu<fp16_t>(input, output, count);
            break;
        case DataType::BFLOAT16:
            detail::execute_typed_silu<bf16_t>(input, output, count);
            break;
        case DataType::FP8_E4M3:
            detail::execute_typed_silu<fp8e4m3_t>(input, output, count);
            break;
        case DataType::FP8_E5M2:
            detail::execute_typed_silu<fp8e5m2_t>(input, output, count);
            break;
        default:
            throw std::invalid_argument(
                "Unsupported DataType for SiLU dispatch: " + dtype_name(dtype));
    }
}

/// Dispatch elementwise add based on DataType
inline void dispatch_add(DataType dtype, const void* a, const void* b, void* output, uint64_t count) {
    switch (dtype) {
        case DataType::FLOAT32:
            detail::execute_typed_add<float>(a, b, output, count);
            break;
        case DataType::FLOAT16:
            detail::execute_typed_add<fp16_t>(a, b, output, count);
            break;
        case DataType::BFLOAT16:
            detail::execute_typed_add<bf16_t>(a, b, output, count);
            break;
        case DataType::FP8_E4M3:
            detail::execute_typed_add<fp8e4m3_t>(a, b, output, count);
            break;
        case DataType::INT8:
            detail::execute_typed_add<int8_t>(a, b, output, count);
            break;
        default:
            throw std::invalid_argument(
                "Unsupported DataType for add dispatch: " + dtype_name(dtype));
    }
}

/// Dispatch elementwise mul based on DataType
inline void dispatch_mul(DataType dtype, const void* a, const void* b, void* output, uint64_t count) {
    switch (dtype) {
        case DataType::FLOAT32:
            detail::execute_typed_mul<float>(a, b, output, count);
            break;
        case DataType::FLOAT16:
            detail::execute_typed_mul<fp16_t>(a, b, output, count);
            break;
        case DataType::BFLOAT16:
            detail::execute_typed_mul<bf16_t>(a, b, output, count);
            break;
        case DataType::FP8_E4M3:
            detail::execute_typed_mul<fp8e4m3_t>(a, b, output, count);
            break;
        case DataType::INT8:
            detail::execute_typed_mul<int8_t>(a, b, output, count);
            break;
        default:
            throw std::invalid_argument(
                "Unsupported DataType for mul dispatch: " + dtype_name(dtype));
    }
}

/// Dispatch softmax based on DataType
inline void dispatch_softmax(DataType dtype, const void* input, void* output,
                             uint32_t batch_size, uint32_t dim_size) {
    switch (dtype) {
        case DataType::FLOAT32:
            detail::execute_typed_softmax<float>(input, output, batch_size, dim_size);
            break;
        case DataType::FLOAT16:
            detail::execute_typed_softmax<fp16_t>(input, output, batch_size, dim_size);
            break;
        case DataType::BFLOAT16:
            detail::execute_typed_softmax<bf16_t>(input, output, batch_size, dim_size);
            break;
        default:
            throw std::invalid_argument(
                "Unsupported DataType for softmax dispatch: " + dtype_name(dtype));
    }
}

/// Dispatch layer normalization based on DataType
inline void dispatch_layernorm(
    DataType dtype,
    const void* input, void* output,
    const void* gamma, const void* beta,
    uint32_t batch_size, uint32_t dim, float eps)
{
    switch (dtype) {
        case DataType::FLOAT32:
            detail::execute_typed_layernorm<float>(
                input, output, gamma, beta, batch_size, dim, eps);
            break;
        case DataType::FLOAT16:
            detail::execute_typed_layernorm<fp16_t>(
                input, output, gamma, beta, batch_size, dim, eps);
            break;
        case DataType::BFLOAT16:
            detail::execute_typed_layernorm<bf16_t>(
                input, output, gamma, beta, batch_size, dim, eps);
            break;
        default:
            throw std::invalid_argument(
                "Unsupported DataType for layernorm dispatch: " + dtype_name(dtype));
    }
}

} // namespace sw::kpu
