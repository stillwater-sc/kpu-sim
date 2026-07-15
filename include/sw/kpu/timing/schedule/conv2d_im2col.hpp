// ============================================================================
// include/sw/kpu/timing/schedule/conv2d_im2col.hpp
// Host-side im2col patchify helper for the conv2d functional tier (E6-T2, #120)
//
// Conv2D is lowered to a single GEMM C_out = A_col @ B_w (see
// docs/plans/e6_conv2d_pattern.md), reusing the value-producing matmul path.
// The only conv-specific host-side capability the functional tier needs is the
// im2col patchify: materialize the A_col operand from an NCHW input tensor
// (writing explicit zeros for padded positions), and reshape the filter to the
// B_w operand. The GEMM itself — K-accumulation, per-output-channel bias, and
// ReLU — is already provided by ConcurrentTimingExecutor::MatMulComputeSpec.
//
// Orientation (matches src/schedules/conv2d_schedule.cpp and the timing
// Conv2DScheduleGenerator):
//   A = A_col : [M = N*Hout*Wout, K = Cin*Kh*Kw]   (im2col rows)
//   B = B_w   : [K,               Ncols = Cout]    (reshaped weights)
//   C = C_out : [M,               Cout]            (reshapes to y[n,co,ho,wo])
//
// Scope: groups = 1, dilation = 1 (the M2 ResNet subset), matching the
// generator's K() = Cin*Kh*Kw. Depthwise/grouped/dilated conv are E6/M3
// follow-ons (see the T1 design). Tensors are row-major fp32:
//   input  : NCHW,   x[((n*Cin + ci)*Hin + h)*Win + w]
//   filter : [Cout, Cin, Kh, Kw], f[((co*Cin + ci)*Kh + kh)*Kw + kw]
// The K index ordering is shared by im2col and the weight reshape:
//   k = (ci*Kh + kh)*Kw + kw
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>  // Size, TilePayload

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sw::kpu::timing::schedule {

namespace detail {

/**
 * @brief Map an output position + filter tap to its (padded) input coordinate.
 *
 * Computes the signed source coordinate `out*stride + k - pad` and reports
 * whether it lands inside `[0, extent)`. On an in-bounds hit `coord` is set;
 * an out-of-bounds coordinate is a padding position (returns false, `coord`
 * untouched). The signed intermediate is `std::int64_t` (not `long`, which is
 * only 32-bit on LLP64/Windows) so the range is independent of `Size`'s width.
 * Shared by im2col_nchw and conv2d_reference so the two stay in lock-step when
 * dilation/grouped conv extend this logic (E6/M3 follow-ons).
 */
[[nodiscard]] inline bool padded_coord(Size out, Size stride, Size k, Size pad,
                                       Size extent, Size& coord) {
    const std::int64_t c = static_cast<std::int64_t>(out * stride + k) -
                           static_cast<std::int64_t>(pad);
    if (c < 0 || c >= static_cast<std::int64_t>(extent)) return false;
    coord = static_cast<Size>(c);
    return true;
}

} // namespace detail

/**
 * @brief Conv2D geometry (groups = 1, dilation = 1) and its GEMM lowering.
 *
 * Output spatial extents use an explicit floor (integer division on the
 * non-negative padded extent), so no divisibility of the padded input by the
 * stride is assumed. If the padded input is smaller than the kernel the output
 * extent is 0 (an invalid, empty convolution) rather than an unsigned wrap.
 */
struct Conv2DGeometry {
    Size N = 1;        ///< batch
    Size C_in = 1;     ///< input channels
    Size H_in = 1;     ///< input height
    Size W_in = 1;     ///< input width
    Size C_out = 1;    ///< output channels
    Size Kh = 1;       ///< kernel height
    Size Kw = 1;       ///< kernel width
    Size stride_h = 1;
    Size stride_w = 1;
    Size pad_h = 0;
    Size pad_w = 0;

    [[nodiscard]] Size H_out() const {
        if (H_in + 2 * pad_h < Kh) return 0;
        return (H_in + 2 * pad_h - Kh) / stride_h + 1;  // floor
    }
    [[nodiscard]] Size W_out() const {
        if (W_in + 2 * pad_w < Kw) return 0;
        return (W_in + 2 * pad_w - Kw) / stride_w + 1;  // floor
    }

    /// GEMM M axis: batch x output spatial positions (A_col rows).
    [[nodiscard]] Size M() const { return N * H_out() * W_out(); }
    /// GEMM K axis: the receptive field (A_col cols / B_w rows).
    [[nodiscard]] Size K() const { return C_in * Kh * Kw; }
    /// GEMM N axis: output channels (B_w cols).
    [[nodiscard]] Size Ncols() const { return C_out; }

    /// Element count of a dense input tensor in NCHW layout.
    [[nodiscard]] std::size_t input_elems() const {
        return static_cast<std::size_t>(N) * C_in * H_in * W_in;
    }
    /// Element count of a dense filter tensor [Cout, Cin, Kh, Kw].
    [[nodiscard]] std::size_t filter_elems() const {
        return static_cast<std::size_t>(C_out) * C_in * Kh * Kw;
    }

    [[nodiscard]] bool valid() const {
        return N && C_in && H_in && W_in && C_out && Kh && Kw &&
               stride_h && stride_w && H_out() && W_out();
    }
};

/**
 * @brief Materialize the im2col A_col matrix from an NCHW input tensor.
 *
 * A_col has shape [M, K] with row m = (n*H_out + ho)*W_out + wo and column
 * k = (ci*Kh + kh)*Kw + kw. Each entry is the input element covered by that
 * (output position, filter tap); positions falling in the padding region
 * contribute an explicit 0, so the downstream GEMM needs no special path.
 *
 * @param input NCHW fp32 tensor, size geom.input_elems()
 * @return row-major [M, K] fp32 matrix
 */
[[nodiscard]] inline std::vector<float>
im2col_nchw(const std::vector<float>& input, const Conv2DGeometry& geom) {
    if (!geom.valid()) throw std::invalid_argument("im2col_nchw: invalid geometry");
    if (input.size() != geom.input_elems())
        throw std::invalid_argument("im2col_nchw: input size does not match geometry");

    const Size Hout = geom.H_out(), Wout = geom.W_out();
    const Size K = geom.K();
    const Size M = geom.M();
    std::vector<float> a_col(static_cast<std::size_t>(M) * K, 0.0f);

    for (Size n = 0; n < geom.N; ++n) {
        for (Size ho = 0; ho < Hout; ++ho) {
            for (Size wo = 0; wo < Wout; ++wo) {
                const Size m = (n * Hout + ho) * Wout + wo;
                for (Size ci = 0; ci < geom.C_in; ++ci) {
                    for (Size kh = 0; kh < geom.Kh; ++kh) {
                        Size ih = 0;  // padding positions leave a_col at 0
                        const bool h_in = detail::padded_coord(
                            ho, geom.stride_h, kh, geom.pad_h, geom.H_in, ih);
                        for (Size kw = 0; kw < geom.Kw; ++kw) {
                            Size iw = 0;
                            const bool w_in = detail::padded_coord(
                                wo, geom.stride_w, kw, geom.pad_w, geom.W_in, iw);
                            const Size k = (ci * geom.Kh + kh) * geom.Kw + kw;
                            if (h_in && w_in) {
                                const std::size_t src =
                                    ((static_cast<std::size_t>(n) * geom.C_in + ci) *
                                         geom.H_in + ih) * geom.W_in + iw;
                                a_col[static_cast<std::size_t>(m) * K + k] = input[src];
                            }
                            // else: padded position stays 0.0f
                        }
                    }
                }
            }
        }
    }
    return a_col;
}

/**
 * @brief Reshape a [Cout, Cin, Kh, Kw] filter to the B_w GEMM operand [K, Cout].
 *
 * B_w[k, co] = filter[co, ci, kh, kw] with the same k = (ci*Kh + kh)*Kw + kw
 * ordering used by im2col_nchw, so A_col @ B_w reproduces the convolution.
 *
 * @return row-major [K, Cout] fp32 matrix
 */
[[nodiscard]] inline std::vector<float>
filter_to_bw_nchw(const std::vector<float>& filter, const Conv2DGeometry& geom) {
    if (!geom.valid()) throw std::invalid_argument("filter_to_bw_nchw: invalid geometry");
    if (filter.size() != geom.filter_elems())
        throw std::invalid_argument("filter_to_bw_nchw: filter size does not match geometry");

    const Size K = geom.K();
    std::vector<float> b_w(static_cast<std::size_t>(K) * geom.C_out, 0.0f);
    for (Size co = 0; co < geom.C_out; ++co) {
        for (Size ci = 0; ci < geom.C_in; ++ci) {
            for (Size kh = 0; kh < geom.Kh; ++kh) {
                for (Size kw = 0; kw < geom.Kw; ++kw) {
                    const Size k = (ci * geom.Kh + kh) * geom.Kw + kw;
                    const std::size_t src =
                        ((static_cast<std::size_t>(co) * geom.C_in + ci) * geom.Kh + kh) *
                            geom.Kw + kw;
                    b_w[static_cast<std::size_t>(k) * geom.C_out + co] = filter[src];
                }
            }
        }
    }
    return b_w;
}

/**
 * @brief Direct-convolution host reference (the functional oracle).
 *
 * y[n, co, ho, wo] = (bias ? bias[co] : 0) +
 *     sum_{ci,kh,kw} x[n, ci, ho*sh + kh - ph, wo*sw + kw - pw] * f[co, ci, kh, kw]
 * with an optional ReLU. Computed directly (no im2col) so it independently
 * checks the A_col @ B_w lowering. Output is NCHW: [N, Cout, Hout, Wout].
 *
 * @param bias size Cout, or empty for no bias
 * @param relu apply max(0, .) after bias
 * @return row-major NCHW output, size N*Cout*Hout*Wout
 */
[[nodiscard]] inline std::vector<float>
conv2d_reference(const std::vector<float>& input, const std::vector<float>& filter,
                 const std::vector<float>& bias, const Conv2DGeometry& geom,
                 bool relu = false) {
    if (!geom.valid()) throw std::invalid_argument("conv2d_reference: invalid geometry");
    if (input.size() != geom.input_elems())
        throw std::invalid_argument("conv2d_reference: input size does not match geometry");
    if (filter.size() != geom.filter_elems())
        throw std::invalid_argument("conv2d_reference: filter size does not match geometry");
    if (!bias.empty() && bias.size() != static_cast<std::size_t>(geom.C_out))
        throw std::invalid_argument("conv2d_reference: bias size must be C_out or empty");

    const Size Hout = geom.H_out(), Wout = geom.W_out();
    std::vector<float> y(
        static_cast<std::size_t>(geom.N) * geom.C_out * Hout * Wout, 0.0f);

    for (Size n = 0; n < geom.N; ++n) {
        for (Size co = 0; co < geom.C_out; ++co) {
            for (Size ho = 0; ho < Hout; ++ho) {
                for (Size wo = 0; wo < Wout; ++wo) {
                    float acc = bias.empty() ? 0.0f : bias[co];
                    for (Size ci = 0; ci < geom.C_in; ++ci) {
                        for (Size kh = 0; kh < geom.Kh; ++kh) {
                            Size ih = 0;
                            if (!detail::padded_coord(ho, geom.stride_h, kh,
                                                      geom.pad_h, geom.H_in, ih))
                                continue;
                            for (Size kw = 0; kw < geom.Kw; ++kw) {
                                Size iw = 0;
                                if (!detail::padded_coord(wo, geom.stride_w, kw,
                                                          geom.pad_w, geom.W_in, iw))
                                    continue;
                                const std::size_t xs =
                                    ((static_cast<std::size_t>(n) * geom.C_in + ci) *
                                         geom.H_in + ih) * geom.W_in + iw;
                                const std::size_t fs =
                                    ((static_cast<std::size_t>(co) * geom.C_in + ci) *
                                         geom.Kh + kh) * geom.Kw + kw;
                                acc += input[xs] * filter[fs];
                            }
                        }
                    }
                    if (relu && acc < 0.0f) acc = 0.0f;
                    const std::size_t ys =
                        ((static_cast<std::size_t>(n) * geom.C_out + co) * Hout + ho) *
                            Wout + wo;
                    y[ys] = acc;
                }
            }
        }
    }
    return y;
}

} // namespace sw::kpu::timing::schedule
