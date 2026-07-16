// ============================================================================
// include/sw/kpu/timing/schedule/pooling_window.hpp
// Host-side pooling window-patchify + reduce + oracle (E7-T2, #192)
//
// Pooling reduces each channel over a spatial window (see
// docs/plans/e7_pooling_pattern.md) - no cross-channel mixing:
//
//   max-pool: y[n,c,ho,wo] = max_{kh,kw} x[n,c, ho*s+kh-p, wo*s+kw-p]
//   avg-pool: y[n,c,ho,wo] = mean over the VALID (non-padded) taps
//   gap:      y[n,c]       = mean_{h,w} x[n,c,h,w]   (global average)
//
// The M2 realization is a per-channel im2col window unfold [Hout*Wout, Kh*Kw]
// reduced along the window axis by the existing VE_REDUCE (MAX / MEAN). This
// header provides the window patchify (with the pooling-specific fill: -inf for
// MAX so padding never wins, 0 + a valid-count for AVG so the mean excludes
// padding, matching ref_pool2d) and direct pooling reference oracles. Pooling is
// a Vector-Engine reduce, not a matmul - no new executor kernel.
//
// Tensors are row-major NCHW fp32: x[((n*C + c)*H + h)*W + w].
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>  // Size, TilePayload

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sw::kpu::timing::schedule {

enum class PoolType { MAX, AVG };

/**
 * @brief Pooling geometry (NCHW), with the conv floor for the output extents.
 */
struct Pool2DGeometry {
    Size N = 1, C = 1, H = 1, W = 1;
    Size Kh = 1, Kw = 1;
    Size stride_h = 1, stride_w = 1;
    Size pad_h = 0, pad_w = 0;

    [[nodiscard]] Size H_out() const {
        if (H + 2 * pad_h < Kh) return 0;
        return (H + 2 * pad_h - Kh) / stride_h + 1;  // floor
    }
    [[nodiscard]] Size W_out() const {
        if (W + 2 * pad_w < Kw) return 0;
        return (W + 2 * pad_w - Kw) / stride_w + 1;  // floor
    }
    [[nodiscard]] Size out_spatial() const { return H_out() * W_out(); }
    [[nodiscard]] Size window() const { return Kh * Kw; }
    [[nodiscard]] std::size_t elems() const {
        return static_cast<std::size_t>(N) * C * H * W;
    }
    [[nodiscard]] std::size_t out_elems() const {
        return static_cast<std::size_t>(N) * C * H_out() * W_out();
    }
    [[nodiscard]] bool valid() const {
        return N && C && H && W && Kh && Kw && stride_h && stride_w &&
               H_out() && W_out();
    }
};

/**
 * @brief One channel's im2col window matrix plus per-row valid-tap counts.
 *
 * `rows` is [N*H_out*W_out, Kh*Kw] row-major; each row is one output position's
 * window. Padded taps are filled with -inf (MAX) or 0 (AVG). `counts[m]` is the
 * number of non-padded taps in row m (used by AVG to divide, so the mean
 * excludes padding).
 */
struct PoolWindow {
    std::vector<float> rows;
    std::vector<Size> counts;
};

/**
 * @brief Materialize channel `c`'s pooling windows from an NCHW input.
 */
[[nodiscard]] inline PoolWindow
pool_window_channel(const std::vector<float>& input, const Pool2DGeometry& geom,
                    Size n, Size c, PoolType type) {
    if (!geom.valid()) throw std::invalid_argument("pool_window_channel: invalid geometry");
    if (input.size() != geom.elems())
        throw std::invalid_argument("pool_window_channel: input size does not match geometry");
    if (n >= geom.N || c >= geom.C)
        throw std::invalid_argument("pool_window_channel: n/c out of range");

    const Size Hout = geom.H_out(), Wout = geom.W_out(), K = geom.window();
    const Size M = Hout * Wout;
    const float pad_fill = (type == PoolType::MAX)
        ? -std::numeric_limits<float>::infinity() : 0.0f;

    PoolWindow out;
    out.rows.assign(static_cast<std::size_t>(M) * K, pad_fill);
    out.counts.assign(M, 0);

    const std::size_t plane = (static_cast<std::size_t>(n) * geom.C + c) *
                              geom.H * geom.W;
    for (Size ho = 0; ho < Hout; ++ho)
        for (Size wo = 0; wo < Wout; ++wo) {
            const Size m = ho * Wout + wo;
            Size valid = 0;
            for (Size kh = 0; kh < geom.Kh; ++kh) {
                const long ih = static_cast<long>(ho * geom.stride_h + kh) -
                                static_cast<long>(geom.pad_h);
                const bool h_in = ih >= 0 && ih < static_cast<long>(geom.H);
                for (Size kw = 0; kw < geom.Kw; ++kw) {
                    const long iw = static_cast<long>(wo * geom.stride_w + kw) -
                                    static_cast<long>(geom.pad_w);
                    const bool w_in = iw >= 0 && iw < static_cast<long>(geom.W);
                    const Size k = kh * geom.Kw + kw;
                    if (h_in && w_in) {
                        out.rows[static_cast<std::size_t>(m) * K + k] =
                            input[plane + static_cast<std::size_t>(ih) * geom.W +
                                  static_cast<std::size_t>(iw)];
                        ++valid;
                    }
                }
            }
            out.counts[m] = valid;
        }
    return out;
}

/**
 * @brief Reduce one window row to its pooled value (the VE_REDUCE the executor
 *        applies): MAX -> max element, AVG -> sum / valid-count.
 */
[[nodiscard]] inline float
reduce_window_row(const float* row, Size window, Size valid_count, PoolType type) {
    if (type == PoolType::MAX) {
        float m = -std::numeric_limits<float>::infinity();
        for (Size k = 0; k < window; ++k) m = std::max(m, row[k]);
        return m;
    }
    float sum = 0.0f;
    for (Size k = 0; k < window; ++k) sum += row[k];  // padded taps are 0
    return valid_count > 0 ? sum / static_cast<float>(valid_count) : 0.0f;
}

/**
 * @brief Direct pooling reference (the functional oracle), NCHW output
 *        [N, C, H_out, W_out]. AVG excludes padding (matches ref_pool2d).
 */
[[nodiscard]] inline std::vector<float>
pool2d_reference(const std::vector<float>& input, const Pool2DGeometry& geom,
                 PoolType type) {
    if (!geom.valid()) throw std::invalid_argument("pool2d_reference: invalid geometry");
    if (input.size() != geom.elems())
        throw std::invalid_argument("pool2d_reference: input size does not match geometry");

    const Size Hout = geom.H_out(), Wout = geom.W_out();
    std::vector<float> y(geom.out_elems());
    for (Size n = 0; n < geom.N; ++n)
        for (Size c = 0; c < geom.C; ++c) {
            const std::size_t plane = (static_cast<std::size_t>(n) * geom.C + c) *
                                      geom.H * geom.W;
            for (Size ho = 0; ho < Hout; ++ho)
                for (Size wo = 0; wo < Wout; ++wo) {
                    float acc = (type == PoolType::MAX)
                        ? -std::numeric_limits<float>::infinity() : 0.0f;
                    Size cnt = 0;
                    for (Size kh = 0; kh < geom.Kh; ++kh) {
                        const long ih = static_cast<long>(ho * geom.stride_h + kh) -
                                        static_cast<long>(geom.pad_h);
                        if (ih < 0 || ih >= static_cast<long>(geom.H)) continue;
                        for (Size kw = 0; kw < geom.Kw; ++kw) {
                            const long iw = static_cast<long>(wo * geom.stride_w + kw) -
                                            static_cast<long>(geom.pad_w);
                            if (iw < 0 || iw >= static_cast<long>(geom.W)) continue;
                            const float v = input[plane + static_cast<std::size_t>(ih) * geom.W +
                                                  static_cast<std::size_t>(iw)];
                            if (type == PoolType::MAX) acc = std::max(acc, v);
                            else                        acc += v;
                            ++cnt;
                        }
                    }
                    if (type == PoolType::AVG && cnt > 0) acc /= static_cast<float>(cnt);
                    y[((static_cast<std::size_t>(n) * geom.C + c) * Hout + ho) * Wout + wo] = acc;
                }
        }
    return y;
}

/**
 * @brief Global average pool: mean over the whole H*W plane per channel.
 *        Output is [N, C] row-major (the [N,C,1,1] tensor flattened).
 */
[[nodiscard]] inline std::vector<float>
global_avg_pool_reference(const std::vector<float>& input, const Pool2DGeometry& geom) {
    if (!geom.valid()) throw std::invalid_argument("global_avg_pool_reference: invalid geometry");
    if (input.size() != geom.elems())
        throw std::invalid_argument("global_avg_pool_reference: input size does not match geometry");

    const std::size_t plane_sz = static_cast<std::size_t>(geom.H) * geom.W;
    std::vector<float> y(static_cast<std::size_t>(geom.N) * geom.C);
    for (Size n = 0; n < geom.N; ++n)
        for (Size c = 0; c < geom.C; ++c) {
            const std::size_t base = (static_cast<std::size_t>(n) * geom.C + c) * plane_sz;
            float sum = 0.0f;
            for (std::size_t i = 0; i < plane_sz; ++i) sum += input[base + i];
            y[static_cast<std::size_t>(n) * geom.C + c] =
                sum / static_cast<float>(plane_sz);
        }
    return y;
}

} // namespace sw::kpu::timing::schedule
