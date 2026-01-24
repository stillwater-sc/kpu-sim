#pragma once
// Quantization support for KPU simulator
// Umbrella header for all quantization components

// Core type traits and concepts
#include <sw/kpu/quantization/scalar_traits.hpp>

// Universal library types (FP16, BF16, FP8, FP4)
// Requires Universal library to be available
#if __has_include(<universal/number/bfloat16/bfloat16.hpp>)
#include <sw/kpu/quantization/universal_types.hpp>
#define KPU_HAS_UNIVERSAL 1
#else
#define KPU_HAS_UNIVERSAL 0
#endif

// Packed storage types (INT4)
#include <sw/kpu/quantization/packed_types.hpp>

// Quantization parameters (scale, zero_point)
#include <sw/kpu/quantization/quant_params.hpp>

// Template kernels
#include <sw/kpu/quantization/kernels.hpp>
