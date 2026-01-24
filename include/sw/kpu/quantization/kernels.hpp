#pragma once
// Template kernels for quantized operations
// All kernels are parameterized by scalar and accumulator types

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <sw/kpu/quantization/scalar_traits.hpp>

namespace sw::kpu::kernels {

// =============================================================================
// TensorView - Lightweight view into tensor data
// =============================================================================

/// Tensor view for kernel operations
/// Does not own data - just provides structured access
template<typename Scalar>
struct TensorView {
    Scalar* data;
    size_t rows;
    size_t cols;
    size_t stride;  // Row stride in elements

    /// Construct a tensor view
    TensorView(Scalar* d, size_t r, size_t c)
        : data(d), rows(r), cols(c), stride(c) {}

    TensorView(Scalar* d, size_t r, size_t c, size_t s)
        : data(d), rows(r), cols(c), stride(s) {}

    /// Element access (row-major)
    Scalar& operator()(size_t i, size_t j) {
        return data[i * stride + j];
    }

    const Scalar& operator()(size_t i, size_t j) const {
        return data[i * stride + j];
    }

    /// Linear access
    Scalar& operator[](size_t idx) { return data[idx]; }
    const Scalar& operator[](size_t idx) const { return data[idx]; }

    /// Total number of elements
    size_t size() const { return rows * cols; }
};

// =============================================================================
// Matrix Multiplication Kernels
// =============================================================================

/**
 * @brief Matrix multiplication with explicit accumulator type
 *
 * Computes: C = alpha * A @ B + beta * C
 *
 * Template parameters enable energy/cost optimization:
 * - Same-type: matmul<int8_t, int32_t>(A, B, C) for INT8 with INT32 accumulator
 * - Full precision: matmul<float, float>(A, B, C) for FP32
 * - Mixed precision via explicit accumulator choice
 *
 * @tparam Scalar Input element type (A and B)
 * @tparam Accumulator Output/accumulator type (C)
 * @param A Left matrix [M x K]
 * @param B Right matrix [K x N]
 * @param C Output matrix [M x N]
 * @param alpha Scale factor for A @ B
 * @param beta Scale factor for existing C
 */
template<typename Scalar, typename Accumulator>
void matmul(
    const TensorView<const Scalar>& A,  // [M x K]
    const TensorView<const Scalar>& B,  // [K x N]
    TensorView<Accumulator>& C,         // [M x N]
    Accumulator alpha = Accumulator(1),
    Accumulator beta = Accumulator(0)
) {
    const size_t M = A.rows;
    const size_t K = A.cols;
    const size_t N = B.cols;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Accumulator sum = beta * C(i, j);
            for (size_t k = 0; k < K; ++k) {
                // Explicit conversion to accumulator type
                sum += static_cast<Accumulator>(A(i, k)) *
                       static_cast<Accumulator>(B(k, j));
            }
            C(i, j) = alpha * sum;
        }
    }
}

/**
 * @brief Mixed-precision matrix multiplication
 *
 * Computes: C = alpha * A @ B + beta * C
 * Where A and B can have different types (e.g., FP32 activations, INT8 weights)
 *
 * Useful for:
 * - Weight-only quantization: FP32 activations, INT8/INT4 weights
 * - Asymmetric quantization: Different precision for different operands
 *
 * @tparam ScalarA Left matrix element type
 * @tparam ScalarB Right matrix element type
 * @tparam Accumulator Output/accumulator type
 */
template<typename ScalarA, typename ScalarB, typename Accumulator>
void matmul_mixed(
    const TensorView<const ScalarA>& A,  // [M x K]
    const TensorView<const ScalarB>& B,  // [K x N]
    TensorView<Accumulator>& C,          // [M x N]
    Accumulator alpha = Accumulator(1),
    Accumulator beta = Accumulator(0)
) {
    const size_t M = A.rows;
    const size_t K = A.cols;
    const size_t N = B.cols;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Accumulator sum = beta * C(i, j);
            for (size_t k = 0; k < K; ++k) {
                sum += static_cast<Accumulator>(A(i, k)) *
                       static_cast<Accumulator>(B(k, j));
            }
            C(i, j) = alpha * sum;
        }
    }
}

// =============================================================================
// Elementwise Operations
// =============================================================================

/// Elementwise addition: C = A + B
template<typename Scalar>
void add(
    const Scalar* A,
    const Scalar* B,
    Scalar* C,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        C[i] = A[i] + B[i];
    }
}

/// Elementwise multiplication: C = A * B (Hadamard product)
template<typename Scalar>
void mul(
    const Scalar* A,
    const Scalar* B,
    Scalar* C,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        C[i] = A[i] * B[i];
    }
}

/// Scalar multiplication: Y = alpha * X
template<typename Scalar>
void scale(
    const Scalar* X,
    Scalar* Y,
    Scalar alpha,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        Y[i] = alpha * X[i];
    }
}

/// In-place scalar multiplication: X *= alpha
template<typename Scalar>
void scale_inplace(
    Scalar* X,
    Scalar alpha,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        X[i] = alpha * X[i];
    }
}

// =============================================================================
// Activation Functions
// =============================================================================

/// ReLU activation: Y = max(0, X)
template<typename Scalar>
void relu(const Scalar* X, Scalar* Y, size_t count) {
    const Scalar zero = Scalar(0);
    for (size_t i = 0; i < count; ++i) {
        Y[i] = (X[i] > zero) ? X[i] : zero;
    }
}

/// In-place ReLU activation
template<typename Scalar>
void relu_inplace(Scalar* X, size_t count) {
    const Scalar zero = Scalar(0);
    for (size_t i = 0; i < count; ++i) {
        if (X[i] < zero) {
            X[i] = zero;
        }
    }
}

/// Leaky ReLU: Y = max(alpha * X, X)
template<typename Scalar>
void leaky_relu(const Scalar* X, Scalar* Y, Scalar alpha, size_t count) {
    const Scalar zero = Scalar(0);
    for (size_t i = 0; i < count; ++i) {
        Y[i] = (X[i] > zero) ? X[i] : alpha * X[i];
    }
}

/// GELU approximation: Y = 0.5 * X * (1 + tanh(sqrt(2/pi) * (X + 0.044715 * X^3)))
template<typename Scalar>
void gelu(const Scalar* X, Scalar* Y, size_t count) {
    using Acc = typename ScalarTraits<Scalar>::accumulator_type;
    constexpr Acc sqrt_2_pi = Acc(0.7978845608028654);  // sqrt(2/pi)
    constexpr Acc coeff = Acc(0.044715);

    for (size_t i = 0; i < count; ++i) {
        Acc x = static_cast<Acc>(X[i]);
        Acc inner = sqrt_2_pi * (x + coeff * x * x * x);
        Acc result = Acc(0.5) * x * (Acc(1) + std::tanh(inner));
        Y[i] = static_cast<Scalar>(result);
    }
}

/// SiLU (Swish): Y = X * sigmoid(X)
template<typename Scalar>
void silu(const Scalar* X, Scalar* Y, size_t count) {
    using Acc = typename ScalarTraits<Scalar>::accumulator_type;
    for (size_t i = 0; i < count; ++i) {
        Acc x = static_cast<Acc>(X[i]);
        Acc sigmoid_x = Acc(1) / (Acc(1) + std::exp(-x));
        Y[i] = static_cast<Scalar>(x * sigmoid_x);
    }
}

// =============================================================================
// Softmax
// =============================================================================

/**
 * @brief Softmax with automatic accumulator promotion
 *
 * Computes numerically stable softmax:
 * Y[i] = exp(X[i] - max(X)) / sum(exp(X - max(X)))
 *
 * @tparam Scalar Input/output element type
 * @param X Input tensor [batch_size x dim]
 * @param Y Output tensor [batch_size x dim]
 * @param batch_size Number of independent softmax operations
 * @param dim Size of each softmax dimension
 */
template<typename Scalar>
void softmax(
    const Scalar* X,
    Scalar* Y,
    size_t batch_size,
    size_t dim
) {
    using Acc = typename ScalarTraits<Scalar>::accumulator_type;

    for (size_t b = 0; b < batch_size; ++b) {
        const Scalar* x_batch = X + b * dim;
        Scalar* y_batch = Y + b * dim;

        // Find max for numerical stability
        Acc max_val = static_cast<Acc>(x_batch[0]);
        for (size_t i = 1; i < dim; ++i) {
            Acc val = static_cast<Acc>(x_batch[i]);
            if (val > max_val) max_val = val;
        }

        // Compute exp and sum
        Acc sum = Acc(0);
        for (size_t i = 0; i < dim; ++i) {
            Acc exp_val = std::exp(static_cast<Acc>(x_batch[i]) - max_val);
            y_batch[i] = static_cast<Scalar>(exp_val);
            sum += exp_val;
        }

        // Normalize
        Acc inv_sum = Acc(1) / sum;
        for (size_t i = 0; i < dim; ++i) {
            y_batch[i] = static_cast<Scalar>(static_cast<Acc>(y_batch[i]) * inv_sum);
        }
    }
}

// =============================================================================
// Layer Normalization
// =============================================================================

/**
 * @brief Layer normalization
 *
 * Computes: Y = (X - mean) / sqrt(var + eps) * gamma + beta
 *
 * @tparam Scalar Input/output element type
 * @param X Input tensor [batch_size x dim]
 * @param Y Output tensor [batch_size x dim]
 * @param gamma Scale parameter [dim]
 * @param beta Bias parameter [dim]
 * @param batch_size Number of samples
 * @param dim Normalization dimension
 * @param eps Epsilon for numerical stability
 */
template<typename Scalar>
void layer_norm(
    const Scalar* X,
    Scalar* Y,
    const Scalar* gamma,
    const Scalar* beta,
    size_t batch_size,
    size_t dim,
    float eps = 1e-5f
) {
    using Acc = typename ScalarTraits<Scalar>::accumulator_type;

    for (size_t b = 0; b < batch_size; ++b) {
        const Scalar* x_batch = X + b * dim;
        Scalar* y_batch = Y + b * dim;

        // Compute mean
        Acc mean = Acc(0);
        for (size_t i = 0; i < dim; ++i) {
            mean += static_cast<Acc>(x_batch[i]);
        }
        mean /= static_cast<Acc>(dim);

        // Compute variance
        Acc var = Acc(0);
        for (size_t i = 0; i < dim; ++i) {
            Acc diff = static_cast<Acc>(x_batch[i]) - mean;
            var += diff * diff;
        }
        var /= static_cast<Acc>(dim);

        // Normalize and apply affine transform
        Acc inv_std = Acc(1) / std::sqrt(var + static_cast<Acc>(eps));
        for (size_t i = 0; i < dim; ++i) {
            Acc normalized = (static_cast<Acc>(x_batch[i]) - mean) * inv_std;
            Acc result = normalized * static_cast<Acc>(gamma[i]) +
                         static_cast<Acc>(beta[i]);
            y_batch[i] = static_cast<Scalar>(result);
        }
    }
}

// =============================================================================
// Bias Addition (Fused Operations)
// =============================================================================

/// Add bias to each row: Y[i,j] = X[i,j] + bias[j]
template<typename Scalar>
void add_bias(
    const Scalar* X,
    const Scalar* bias,
    Scalar* Y,
    size_t batch_size,
    size_t dim
) {
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < dim; ++i) {
            Y[b * dim + i] = X[b * dim + i] + bias[i];
        }
    }
}

/// In-place bias addition
template<typename Scalar>
void add_bias_inplace(
    Scalar* X,
    const Scalar* bias,
    size_t batch_size,
    size_t dim
) {
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < dim; ++i) {
            X[b * dim + i] = X[b * dim + i] + bias[i];
        }
    }
}

} // namespace sw::kpu::kernels
