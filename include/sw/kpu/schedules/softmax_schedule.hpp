/**
 * @file softmax_schedule.hpp
 * @brief Softmax schedule: y[b,i] = exp(x[b,i] - max) / sum(exp(x[b,i] - max))
 *
 * Three-pass reduction + elementwise kernel using the Vector Engine.
 */

#pragma once

#include <sw/kpu/dsl/schedule.hpp>

namespace sw::kpu::schedules {

/**
 * @brief Create a softmax schedule (multi-pass VE kernel).
 *
 * Pass 1: Find max per row (reduction over D)
 * Pass 2: Compute exp(x - max) (elementwise over D)
 * Pass 3: Sum reduction + divide (reduction then elementwise)
 *
 * @param B   Batch dimension
 * @param D   Softmax dimension
 * @param Tb  Batch tile size
 */
dsl::Schedule softmax(Size B, Size D, Size Tb = 32);

} // namespace sw::kpu::schedules
