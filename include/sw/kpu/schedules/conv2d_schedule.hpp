/**
 * @file conv2d_schedule.hpp
 * @brief Conv2D schedule using im2col + output-stationary matmul
 */

#pragma once

#include <sw/kpu/dsl/schedule.hpp>

namespace sw::kpu::schedules {

/**
 * @brief Create a conv2d schedule using im2col lowering.
 *
 * Lowers convolution to matmul:
 *   A_col[N*Oh*Ow, Ci*Kh*Kw] = im2col(input)
 *   B_w  [Ci*Kh*Kw, Co]       = reshape(weights)
 *   C_out[N*Oh*Ow, Co]        = A_col x B_w + bias
 *
 * Uses DMA gather for im2col and fused drain with bias+ReLU.
 */
dsl::Schedule conv2d_im2col(
    Size batch, Size Ci, Size H, Size W,
    Size Co, Size Kh, Size Kw,
    Size stride = 1, Size padding = 0,
    Size Ti = 64, Size Tj = 64);

} // namespace sw::kpu::schedules
