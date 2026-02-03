/**
 * @file matmul_schedule.hpp
 * @brief Output-stationary MatMul schedule: C[M,N] += A[M,K] x B[K,N]
 */

#pragma once

#include <sw/kpu/dsl/schedule.hpp>

namespace sw::kpu::schedules {

/**
 * @brief Create an output-stationary matmul schedule.
 *
 * Loop order: for ti, for tj, for tk (output outer, reduction inner)
 * C stays in PE accumulators across K iterations.
 *
 * @param M    Number of rows in A and C
 * @param N    Number of columns in B and C
 * @param K    Shared dimension (columns of A, rows of B)
 * @param Ti   Tile size for M dimension
 * @param Tj   Tile size for N dimension
 * @param Tk   Tile size for K dimension
 */
dsl::Schedule matmul_output_stationary(
    Size M, Size N, Size K,
    Size Ti = 128, Size Tj = 128, Size Tk = 64);

} // namespace sw::kpu::schedules
