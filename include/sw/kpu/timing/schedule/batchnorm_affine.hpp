// ============================================================================
// include/sw/kpu/timing/schedule/batchnorm_affine.hpp
// Host-side BatchNorm-inference scale/shift fold + oracle (E9-T2, #179)
//
// BatchNorm at inference is a per-channel affine with precomputed running
// statistics (see docs/plans/e9_batchnorm_pattern.md):
//
//   y[n,c,h,w] = gamma[c] * (x[n,c,h,w] - mean[c]) / sqrt(var[c] + eps) + beta[c]
//
// which folds exactly into a single scale/shift pair per channel:
//
//   scale[c] = gamma[c] / sqrt(var[c] + eps)
//   shift[c] = beta[c] - mean[c] * scale[c]
//   =>  y[n,c,h,w] = x[n,c,h,w] * scale[c] + shift[c]
//
// So BN inference is a per-channel broadcast-affine — a Vector Engine op on the
// existing FunctionalComputeSpec value path, no new compute kernel. This header
// provides the host-side fold (the T3 generator loads scale/shift instead of the
// four raw params, halving resident params to 2C+1) and a direct BN reference
// oracle. Tensors are row-major NCHW fp32: x[((n*C + c)*H + h)*W + w].
//
// Scope: inference only (the M2 path). Training (which computes batch mean/var,
// a P3 reduction) is an E9 follow-on. This is the same fold conv2d T4 applies
// for conv+BN folding.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>  // Size, TilePayload

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sw::kpu::timing::schedule {

/**
 * @brief BatchNorm tensor geometry (NCHW).
 */
struct BatchNormGeometry {
    Size N = 1;  ///< batch
    Size C = 1;  ///< channels
    Size H = 1;  ///< height
    Size W = 1;  ///< width

    [[nodiscard]] Size spatial() const { return H * W; }
    [[nodiscard]] std::size_t elems() const {
        return static_cast<std::size_t>(N) * C * H * W;
    }
    [[nodiscard]] bool valid() const { return N && C && H && W; }
};

/**
 * @brief The folded per-channel affine: y = x * scale[c] + shift[c].
 */
struct BatchNormAffine {
    std::vector<float> scale;  ///< size C
    std::vector<float> shift;  ///< size C
};

/**
 * @brief Fold the four per-channel BN params into a scale/shift affine.
 *
 * scale[c] = gamma[c] / sqrt(var[c] + eps)
 * shift[c] = beta[c]  - mean[c] * scale[c]
 *
 * eps guards the variance (as in the E3/E8 clamped-variance discipline); a
 * non-positive var+eps is a malformed input and throws rather than producing a
 * NaN/Inf scale.
 *
 * @param gamma,beta,mean,var per-channel vectors, each size C
 * @param eps  numerical-stability epsilon (> 0)
 */
[[nodiscard]] inline BatchNormAffine
bn_fold(const std::vector<float>& gamma, const std::vector<float>& beta,
        const std::vector<float>& mean, const std::vector<float>& var,
        float eps) {
    const std::size_t C = gamma.size();
    if (C == 0)
        throw std::invalid_argument("bn_fold: C must be non-zero");
    if (beta.size() != C || mean.size() != C || var.size() != C)
        throw std::invalid_argument("bn_fold: gamma/beta/mean/var sizes must match");
    if (!(eps > 0.0f))
        throw std::invalid_argument("bn_fold: eps must be > 0");

    BatchNormAffine a;
    a.scale.resize(C);
    a.shift.resize(C);
    for (std::size_t c = 0; c < C; ++c) {
        const float denom = var[c] + eps;
        if (!(denom > 0.0f))
            throw std::invalid_argument("bn_fold: var[c] + eps must be > 0");
        a.scale[c] = gamma[c] / std::sqrt(denom);
        a.shift[c] = beta[c] - mean[c] * a.scale[c];
    }
    return a;
}

/**
 * @brief Apply a folded per-channel affine to an NCHW input (the fast path).
 *
 * y[n,c,h,w] = x[n,c,h,w] * scale[c] + shift[c].
 *
 * @return row-major NCHW output, size geom.elems()
 */
[[nodiscard]] inline std::vector<float>
batchnorm_apply(const std::vector<float>& input, const BatchNormAffine& affine,
                const BatchNormGeometry& geom) {
    if (!geom.valid()) throw std::invalid_argument("batchnorm_apply: invalid geometry");
    if (input.size() != geom.elems())
        throw std::invalid_argument("batchnorm_apply: input size does not match geometry");
    if (affine.scale.size() != static_cast<std::size_t>(geom.C) ||
        affine.shift.size() != static_cast<std::size_t>(geom.C))
        throw std::invalid_argument("batchnorm_apply: scale/shift size must be C");

    const Size spatial = geom.spatial();
    std::vector<float> y(geom.elems());
    for (Size n = 0; n < geom.N; ++n)
        for (Size c = 0; c < geom.C; ++c) {
            const float sc = affine.scale[c], sh = affine.shift[c];
            const std::size_t base =
                (static_cast<std::size_t>(n) * geom.C + c) * spatial;
            for (Size s = 0; s < spatial; ++s)
                y[base + s] = input[base + s] * sc + sh;
        }
    return y;
}

/**
 * @brief Direct BatchNorm-inference reference (the functional oracle).
 *
 * Computes y = gamma*(x - mean)/sqrt(var + eps) + beta directly from the four
 * raw params (not via the fold), so it independently checks the fold + apply.
 *
 * @return row-major NCHW output, size geom.elems()
 */
[[nodiscard]] inline std::vector<float>
batchnorm_reference(const std::vector<float>& input, const std::vector<float>& gamma,
                    const std::vector<float>& beta, const std::vector<float>& mean,
                    const std::vector<float>& var, float eps,
                    const BatchNormGeometry& geom) {
    if (!geom.valid()) throw std::invalid_argument("batchnorm_reference: invalid geometry");
    if (input.size() != geom.elems())
        throw std::invalid_argument("batchnorm_reference: input size does not match geometry");
    const std::size_t C = static_cast<std::size_t>(geom.C);
    if (gamma.size() != C || beta.size() != C || mean.size() != C || var.size() != C)
        throw std::invalid_argument("batchnorm_reference: param sizes must be C");
    if (!(eps > 0.0f))
        throw std::invalid_argument("batchnorm_reference: eps must be > 0");

    const Size spatial = geom.spatial();
    std::vector<float> y(geom.elems());
    for (Size n = 0; n < geom.N; ++n)
        for (Size c = 0; c < geom.C; ++c) {
            const float denom = var[c] + eps;
            if (!(denom > 0.0f))
                throw std::invalid_argument("batchnorm_reference: var[c] + eps must be > 0");
            const float inv = 1.0f / std::sqrt(denom);
            const std::size_t base =
                (static_cast<std::size_t>(n) * geom.C + c) * spatial;
            for (Size s = 0; s < spatial; ++s)
                y[base + s] = gamma[c] * (input[base + s] - mean[c]) * inv + beta[c];
        }
    return y;
}

} // namespace sw::kpu::timing::schedule
