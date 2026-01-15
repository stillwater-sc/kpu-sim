# KPU Tensor Type System Design

## Overview

This document describes the design of `kpu::Tensor` and the expression template system for the KPU SDK. Following the Eigen/Trilinos pattern, we parameterize on scalar type at compile time rather than using runtime `dtype` like Python/NumPy.

## Design Philosophy

### Why Templates Over Runtime dtype

```cpp
// BAD: Python-style runtime dispatch
class Tensor {
    DataType dtype_;  // Runtime enum
    void* data_;      // Type-erased storage

    Tensor matmul(const Tensor& other) {
        switch (dtype_) {
            case FLOAT32: return matmul_impl<float>(...);
            case FLOAT16: return matmul_impl<half>(...);
            // Every operation needs this dispatch
        }
    }
};

// GOOD: Template on scalar type
template<typename Scalar>
class Tensor {
    Scalar* data_;
};

template<typename Scalar>
Tensor<Scalar> matmul(const Tensor<Scalar>& A, const Tensor<Scalar>& B);
```

Benefits of template approach:
- **Type safety**: Errors caught at compile time
- **Zero overhead**: No runtime dispatch
- **User extensibility**: Support custom types via trait specialization
- **Expression templates**: Enable lazy evaluation and fusion

---

## Layer 1: Scalar Type Traits

```cpp
namespace kpu {

/**
 * @brief Type traits for scalar types
 *
 * Users can specialize this for custom number systems (posits, etc.)
 */
template<typename Scalar>
struct ScalarTraits {
    static constexpr bool is_supported = false;
    static constexpr HardwareType hw_type = HardwareType::UNSUPPORTED;
    static constexpr size_t size = sizeof(Scalar);
    static constexpr size_t alignment = alignof(Scalar);

    // For mixed-precision: what type does hardware accumulate in?
    using accumulator_type = Scalar;

    // Conversion to/from hardware format
    static void to_hardware(const Scalar* src, void* dst, size_t n);
    static void from_hardware(const void* src, Scalar* dst, size_t n);
};

// Specialization for float
template<>
struct ScalarTraits<float> {
    static constexpr bool is_supported = true;
    static constexpr HardwareType hw_type = HardwareType::FLOAT32;
    static constexpr size_t size = 4;
    static constexpr size_t alignment = 4;
    using accumulator_type = float;

    static void to_hardware(const float* src, void* dst, size_t n) {
        std::memcpy(dst, src, n * sizeof(float));
    }
    static void from_hardware(const void* src, float* dst, size_t n) {
        std::memcpy(dst, src, n * sizeof(float));
    }
};

// Specialization for double
template<>
struct ScalarTraits<double> {
    static constexpr bool is_supported = true;
    static constexpr HardwareType hw_type = HardwareType::FLOAT64;
    static constexpr size_t size = 8;
    static constexpr size_t alignment = 8;
    using accumulator_type = double;
    // ...
};

// Specialization for half precision
template<>
struct ScalarTraits<half_t> {
    static constexpr bool is_supported = true;
    static constexpr HardwareType hw_type = HardwareType::FLOAT16;
    static constexpr size_t size = 2;
    static constexpr size_t alignment = 2;
    using accumulator_type = float;  // Accumulate in FP32 for accuracy
    // ...
};

} // namespace kpu
```

### User-Defined Scalar Types

Users can add support for custom number systems:

```cpp
#include <universal/number/posit/posit.hpp>

namespace kpu {

template<>
struct ScalarTraits<sw::universal::posit<32,2>> {
    using posit_t = sw::universal::posit<32,2>;

    static constexpr bool is_supported = true;
    static constexpr HardwareType hw_type = HardwareType::POSIT32;
    static constexpr size_t size = 4;
    static constexpr size_t alignment = 4;

    // Posits can use quire for exact dot products
    using accumulator_type = sw::universal::quire<32,2>;

    static void to_hardware(const posit_t* src, void* dst, size_t n) {
        auto* out = static_cast<uint32_t*>(dst);
        for (size_t i = 0; i < n; ++i) {
            out[i] = src[i].bits();
        }
    }

    static void from_hardware(const void* src, posit_t* dst, size_t n) {
        auto* in = static_cast<const uint32_t*>(src);
        for (size_t i = 0; i < n; ++i) {
            dst[i].set_raw_bits(in[i]);
        }
    }
};

} // namespace kpu
```

---

## Layer 2: Core Tensor Class

```cpp
namespace kpu {

// Forward declarations
template<typename Scalar> class Tensor;
template<typename Expr> class TensorExpr;

/**
 * @brief Concrete tensor storage
 */
template<typename Scalar>
class Tensor : public TensorExpr<Tensor<Scalar>> {
    static_assert(ScalarTraits<Scalar>::is_supported,
                  "Scalar type must be supported by KPU hardware");
public:
    using scalar_type = Scalar;
    using traits = ScalarTraits<Scalar>;
    using shape_type = std::vector<size_t>;

    // ========================================
    // Construction
    // ========================================

    Tensor() = default;
    explicit Tensor(shape_type shape);
    Tensor(shape_type shape, Scalar fill_value);

    // Factory methods
    static Tensor zeros(shape_type shape);
    static Tensor ones(shape_type shape);
    static Tensor from_data(const Scalar* data, shape_type shape);
    static Tensor randn(shape_type shape);  // Random normal
    static Tensor from_file(const std::string& path, shape_type shape);

    // ========================================
    // Properties
    // ========================================

    const shape_type& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t numel() const;
    size_t size_bytes() const { return numel() * sizeof(Scalar); }

    // ========================================
    // Data Access (host side)
    // ========================================

    Scalar* data() { return data_.get(); }
    const Scalar* data() const { return data_.get(); }

    // Element access (for debugging, not performance-critical)
    Scalar& operator()(std::initializer_list<size_t> indices);
    const Scalar& operator()(std::initializer_list<size_t> indices) const;

    // ========================================
    // Device Memory
    // ========================================

    bool is_on_device() const { return device_address_ != 0; }
    Address device_address() const { return device_address_; }
    void to_device(ExecutionContext& ctx);
    void to_host(ExecutionContext& ctx);

    // ========================================
    // Expression Template Support
    // ========================================

    // Concrete tensors evaluate to themselves
    void eval_into(Tensor& dst, ExecutionContext& ctx) const {
        dst = *this;
    }

private:
    shape_type shape_;
    std::unique_ptr<Scalar[]> data_;  // Host storage
    Address device_address_ = 0;       // Device address (0 = not on device)
};

} // namespace kpu
```

---

## Layer 3: Expression Templates

Expression templates allow us to capture computation graphs at compile time.

### Base Expression Class (CRTP)

```cpp
namespace kpu {

/**
 * @brief CRTP base for all tensor expressions
 *
 * The Derived type contains the expression tree structure.
 */
template<typename Derived>
class TensorExpr {
public:
    // Access derived class
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    // Evaluate expression into concrete tensor
    template<typename Scalar>
    Tensor<Scalar> eval(ExecutionContext& ctx) const;

    // Get scalar type of this expression
    using scalar_type = typename Derived::scalar_type;
};

} // namespace kpu
```

### Matrix Multiplication Expression

```cpp
namespace kpu {

/**
 * @brief Expression representing A @ B
 */
template<typename LHS, typename RHS>
class MatMulExpr : public TensorExpr<MatMulExpr<LHS, RHS>> {
public:
    using lhs_scalar = typename LHS::scalar_type;
    using rhs_scalar = typename RHS::scalar_type;
    using scalar_type = std::common_type_t<lhs_scalar, rhs_scalar>;

    MatMulExpr(const LHS& lhs, const RHS& rhs)
        : lhs_(lhs), rhs_(rhs) {}

    const LHS& lhs() const { return lhs_; }
    const RHS& rhs() const { return rhs_; }

    // Compile-time shape inference
    std::vector<size_t> shape() const {
        auto lhs_shape = lhs_.derived().shape();
        auto rhs_shape = rhs_.derived().shape();
        return {lhs_shape[0], rhs_shape[1]};
    }

    // Evaluate: builds KPU computation graph
    void eval_into(Tensor<scalar_type>& dst, ExecutionContext& ctx) const;

private:
    const LHS& lhs_;
    const RHS& rhs_;
};

} // namespace kpu
```

### Unary Expression

```cpp
namespace kpu {

/**
 * @brief Expression representing unary op(x)
 */
template<typename Operand, typename UnaryOp>
class UnaryExpr : public TensorExpr<UnaryExpr<Operand, UnaryOp>> {
public:
    using scalar_type = typename Operand::scalar_type;

    UnaryExpr(const Operand& operand, UnaryOp op)
        : operand_(operand), op_(op) {}

    const Operand& operand() const { return operand_; }
    UnaryOp op() const { return op_; }

    std::vector<size_t> shape() const {
        return operand_.derived().shape();
    }

    void eval_into(Tensor<scalar_type>& dst, ExecutionContext& ctx) const;

private:
    const Operand& operand_;
    UnaryOp op_;
};

// Unary operation tags
struct ReluOp { static constexpr ActivationType type = ActivationType::RELU; };
struct GeluOp { static constexpr ActivationType type = ActivationType::GELU; };
struct SigmoidOp { static constexpr ActivationType type = ActivationType::SIGMOID; };
struct TanhOp { static constexpr ActivationType type = ActivationType::TANH; };

} // namespace kpu
```

### Binary Expression

```cpp
namespace kpu {

/**
 * @brief Expression representing binary op(a, b)
 */
template<typename LHS, typename RHS, typename BinaryOp>
class BinaryExpr : public TensorExpr<BinaryExpr<LHS, RHS, BinaryOp>> {
public:
    using scalar_type = std::common_type_t<
        typename LHS::scalar_type,
        typename RHS::scalar_type
    >;

    BinaryExpr(const LHS& lhs, const RHS& rhs, BinaryOp op)
        : lhs_(lhs), rhs_(rhs), op_(op) {}

    void eval_into(Tensor<scalar_type>& dst, ExecutionContext& ctx) const;

private:
    const LHS& lhs_;
    const RHS& rhs_;
    BinaryOp op_;
};

// Binary operation tags
struct AddOp {};
struct SubOp {};
struct MulOp {};
struct DivOp {};

} // namespace kpu
```

---

## Layer 4: Operation Functions

```cpp
namespace kpu {

// ========================================
// Matrix Operations
// ========================================

template<typename LHS, typename RHS>
auto matmul(const TensorExpr<LHS>& lhs, const TensorExpr<RHS>& rhs) {
    return MatMulExpr<LHS, RHS>(lhs.derived(), rhs.derived());
}

// ========================================
// Activation Functions
// ========================================

template<typename Operand>
auto relu(const TensorExpr<Operand>& x) {
    return UnaryExpr<Operand, ReluOp>(x.derived(), ReluOp{});
}

template<typename Operand>
auto gelu(const TensorExpr<Operand>& x) {
    return UnaryExpr<Operand, GeluOp>(x.derived(), GeluOp{});
}

template<typename Operand>
auto sigmoid(const TensorExpr<Operand>& x) {
    return UnaryExpr<Operand, SigmoidOp>(x.derived(), SigmoidOp{});
}

template<typename Operand>
auto tanh(const TensorExpr<Operand>& x) {
    return UnaryExpr<Operand, TanhOp>(x.derived(), TanhOp{});
}

// ========================================
// Elementwise Operations
// ========================================

template<typename LHS, typename RHS>
auto operator+(const TensorExpr<LHS>& lhs, const TensorExpr<RHS>& rhs) {
    return BinaryExpr<LHS, RHS, AddOp>(lhs.derived(), rhs.derived(), AddOp{});
}

template<typename LHS, typename RHS>
auto operator-(const TensorExpr<LHS>& lhs, const TensorExpr<RHS>& rhs) {
    return BinaryExpr<LHS, RHS, SubOp>(lhs.derived(), rhs.derived(), SubOp{});
}

template<typename LHS, typename RHS>
auto operator*(const TensorExpr<LHS>& lhs, const TensorExpr<RHS>& rhs) {
    return BinaryExpr<LHS, RHS, MulOp>(lhs.derived(), rhs.derived(), MulOp{});
}

} // namespace kpu
```

---

## Layer 5: Graph Building from Expressions

Expression templates give us a compile-time AST. We walk this tree to build the runtime computation graph.

```cpp
namespace kpu {

/**
 * @brief Converts expression template tree to runtime graph
 */
class GraphBuilder {
public:
    explicit GraphBuilder(ExecutionContext& ctx) : ctx_(ctx) {}

    template<typename Expr>
    NodeId build(const TensorExpr<Expr>& expr) {
        return build_impl(expr.derived());
    }

private:
    // Concrete tensor: leaf node
    template<typename Scalar>
    NodeId build_impl(const Tensor<Scalar>& tensor) {
        return graph_.add_input(
            tensor.device_address(),
            tensor.shape(),
            ScalarTraits<Scalar>::hw_type
        );
    }

    // MatMul expression
    template<typename LHS, typename RHS>
    NodeId build_impl(const MatMulExpr<LHS, RHS>& expr) {
        NodeId lhs_id = build_impl(expr.lhs().derived());
        NodeId rhs_id = build_impl(expr.rhs().derived());
        return graph_.add_matmul(lhs_id, rhs_id);
    }

    // Unary expression
    template<typename Operand, typename Op>
    NodeId build_impl(const UnaryExpr<Operand, Op>& expr) {
        NodeId input_id = build_impl(expr.operand().derived());
        return graph_.add_activation(input_id, Op::type);
    }

    // Binary expression
    template<typename LHS, typename RHS, typename Op>
    NodeId build_impl(const BinaryExpr<LHS, RHS, Op>& expr) {
        NodeId lhs_id = build_impl(expr.lhs().derived());
        NodeId rhs_id = build_impl(expr.rhs().derived());
        return graph_.add_elementwise(lhs_id, rhs_id, Op{});
    }

    ExecutionContext& ctx_;
    ComputeGraph graph_;
};

} // namespace kpu
```

---

## Layer 6: Mixed Precision Support

Templates naturally handle mixed precision:

```cpp
namespace kpu {

/**
 * @brief Traits for mixed-precision operations
 */
template<typename A, typename B>
struct MixedPrecisionTraits {
    // Default: use common type
    using result_type = std::common_type_t<A, B>;
    using accumulator_type = result_type;
};

// Half inputs, float accumulator
template<>
struct MixedPrecisionTraits<half_t, half_t> {
    using result_type = half_t;
    using accumulator_type = float;
};

// Int8 inputs, int32 accumulator (quantized inference)
template<>
struct MixedPrecisionTraits<int8_t, int8_t> {
    using result_type = int8_t;
    using accumulator_type = int32_t;
};

// Mixed half/float
template<>
struct MixedPrecisionTraits<half_t, float> {
    using result_type = float;
    using accumulator_type = float;
};

} // namespace kpu
```

---

## Layer 7: Eager vs Lazy Execution

```cpp
namespace kpu {

enum class ExecutionMode {
    LAZY,   // Build graph, execute on sync (default)
    EAGER   // Execute immediately
};

// Global execution mode (thread-local)
inline ExecutionMode& execution_mode() {
    thread_local ExecutionMode mode = ExecutionMode::LAZY;
    return mode;
}

void set_execution_mode(ExecutionMode mode) {
    execution_mode() = mode;
}

// In expression evaluation:
template<typename Derived>
template<typename Scalar>
Tensor<Scalar> TensorExpr<Derived>::eval(ExecutionContext& ctx) const {
    Tensor<Scalar> result(derived().shape());
    result.to_device(ctx);

    if (execution_mode() == ExecutionMode::EAGER) {
        // Execute immediately
        derived().eval_into(result, ctx);
        ctx.synchronize();
    } else {
        // Add to deferred graph, execute on to_host() or explicit sync
        ctx.add_deferred(derived(), result);
    }

    return result;
}

} // namespace kpu
```

---

## Layer 8: Python Interop (Type Erasure)

For Python bindings, we need a type-erased wrapper:

```cpp
namespace kpu {

/**
 * @brief Type-erased tensor for Python/C interop
 */
class AnyTensor {
public:
    template<typename Scalar>
    AnyTensor(Tensor<Scalar>&& t)
        : storage_(std::move(t)) {}

    DataType dtype() const {
        return std::visit([](const auto& t) {
            using T = std::decay_t<decltype(t)>;
            using Scalar = typename T::scalar_type;
            return to_dtype<Scalar>();
        }, storage_);
    }

    template<typename Scalar>
    Tensor<Scalar>& as() {
        return std::get<Tensor<Scalar>>(storage_);
    }

    template<typename Scalar>
    const Tensor<Scalar>& as() const {
        return std::get<Tensor<Scalar>>(storage_);
    }

private:
    std::variant<
        Tensor<float>,
        Tensor<double>,
        Tensor<half_t>,
        Tensor<int8_t>,
        Tensor<int32_t>
    > storage_;
};

} // namespace kpu
```

---

## Complete Example

```cpp
#include <kpu/tensor.hpp>
#include <universal/number/posit/posit.hpp>

using namespace sw::universal;

int main() {
    kpu::Device device;
    kpu::ExecutionContext ctx(device);

    // ========================================
    // Float32 computation
    // ========================================

    kpu::Tensor<float> A = kpu::Tensor<float>::from_file("A.bin", {1024, 512});
    kpu::Tensor<float> B = kpu::Tensor<float>::from_file("B.bin", {512, 256});

    // Expression template captures computation (no execution yet)
    auto expr = kpu::relu(kpu::matmul(A, B));

    // Evaluate: builds graph, compiles to DFX, executes on KPU
    kpu::Tensor<float> C = expr.eval(ctx);

    std::cout << "C[0,0] = " << C({0, 0}) << "\n";

    // ========================================
    // Posit computation (same code, different type)
    // ========================================

    kpu::Tensor<posit<32,2>> P = kpu::Tensor<posit<32,2>>::ones({128, 128});
    kpu::Tensor<posit<32,2>> Q = kpu::Tensor<posit<32,2>>::ones({128, 128});

    // Type-safe, compiles to posit hardware ops
    auto R = kpu::matmul(P, Q).eval(ctx);

    // ========================================
    // Mixed precision
    // ========================================

    kpu::Tensor<half_t> X = kpu::Tensor<half_t>::randn({512, 512});
    kpu::Tensor<half_t> Y = kpu::Tensor<half_t>::randn({512, 512});

    // Accumulates in float32 internally (per MixedPrecisionTraits)
    auto Z = kpu::matmul(X, Y).eval(ctx);

    // ========================================
    // Complex expression (all lazy)
    // ========================================

    auto h1 = kpu::relu(kpu::matmul(A, B));
    auto h2 = kpu::gelu(kpu::matmul(h1, B));
    auto out = h1 + h2;  // Residual connection

    // Single eval builds entire graph
    kpu::Tensor<float> result = out.eval(ctx);

    return 0;
}
```

---

## Design Tradeoffs Summary

| Aspect | Template Approach | Runtime dtype |
|--------|------------------|---------------|
| Type safety | Compile-time | Runtime errors |
| Performance | Zero overhead | Dispatch overhead |
| Custom types | User specializes traits | Modify library |
| Code size | Larger (instantiation) | Smaller |
| Python interop | Needs type erasure layer | Natural fit |
| Expression fusion | Via templates | Runtime graph |

---

## Related Documents

- `BLOCK_FORMAT_TYPE_SYSTEM.md` - Block-compressed formats (ZFP, MX)
- `VIRTUAL_PLATFORM_ANALYSIS.md` - Overall virtual platform architecture
- `kpu-execution-model.md` - Credit-based dataflow model

---

*Document created: 2026-01-15*
