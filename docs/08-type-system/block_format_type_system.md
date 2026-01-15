# Block Format Type System for KPU

## Overview

This document describes the type system design for block-compressed formats (ZFP, Micro Exponents, etc.) in the KPU SDK. Block formats differ fundamentally from scalar types because they operate on **groups of values**, requiring decompression somewhere in the memory hierarchy before computation.

## Scalar vs Block Types

| Property | Scalar (float32, posit32) | Block (ZFP, MX) |
|----------|--------------------------|-----------------|
| Unit of operation | Single value | Block (4x4, 32 elements, etc.) |
| Memory representation | 1:1 with compute | Compressed |
| Compute representation | Same as memory | Decompressed |
| Alignment requirements | Element size | Block size |
| Conversion | Identity or simple cast | Encode/decode pipeline |

## Template-Based Design Philosophy

Following the Eigen/Trilinos pattern, we parameterize on scalar/block type rather than using runtime `dtype`:

```cpp
// BAD: Python-style runtime dispatch
class Tensor {
    DataType dtype_;  // Runtime enum
    void* data_;      // Type-erased storage
};

// GOOD: Template on scalar type
template<typename Scalar>
class Tensor {
    Scalar* data_;
};

// GOOD: Template on block format
template<typename BlockFormat>
class BlockTensor {
    uint8_t* compressed_data_;
};
```

Benefits:
- Compile-time type safety
- Zero runtime dispatch overhead
- User-extensible via trait specialization
- Expression template optimization possible

---

## Core Type Traits

### ScalarTraits (for scalar types)

```cpp
namespace kpu {

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
template<> struct ScalarTraits<float> {
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

// Specialization for posit<32,2> (user-defined type)
template<>
struct ScalarTraits<sw::universal::posit<32,2>> {
    using posit_t = sw::universal::posit<32,2>;

    static constexpr bool is_supported = true;
    static constexpr HardwareType hw_type = HardwareType::POSIT32;
    static constexpr size_t size = 4;
    static constexpr size_t alignment = 4;

    // Posits accumulate in quire for exact dot products
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

### BlockTraits (for block formats)

```cpp
namespace kpu {

template<typename BlockFormat>
struct BlockTraits {
    static constexpr bool is_block_format = false;
};

} // namespace kpu
```

---

## ZFP Block Format

### Overview

ZFP is a lossy compression format for floating-point arrays:
- Operates on 4^d blocks (4 for 1D, 4x4 for 2D, 4x4x4 for 3D)
- Compression pipeline: block gather -> decorrelating transform -> embedded coding
- Configurable: fixed-rate, fixed-precision, or fixed-accuracy modes
- Typical compression ratios: 2-10x

### Type Definition

```cpp
namespace kpu {

template<typename Scalar, size_t Dims, size_t Rate>
struct ZFP {
    using scalar_type = Scalar;
    static constexpr size_t dimensions = Dims;
    static constexpr size_t bits_per_value = Rate;
    static constexpr size_t block_side = 4;  // ZFP always uses 4^d blocks
    static constexpr size_t block_size = detail::pow(4, Dims);  // 4, 16, 64, 256
};

// Common configurations
using ZFP_1D_16 = ZFP<float, 1, 16>;  // 1D, 16 bits/value, ~2x compression
using ZFP_2D_16 = ZFP<float, 2, 16>;  // 2D, 16 bits/value, ~2x compression
using ZFP_2D_8  = ZFP<float, 2, 8>;   // 2D, 8 bits/value, ~4x compression
using ZFP_3D_16 = ZFP<float, 3, 16>;  // 3D, 16 bits/value, ~2x compression

} // namespace kpu
```

### BlockTraits Specialization

```cpp
namespace kpu {

template<typename Scalar, size_t Dims, size_t Rate>
struct BlockTraits<ZFP<Scalar, Dims, Rate>> {
    using format = ZFP<Scalar, Dims, Rate>;
    using scalar_type = Scalar;
    using compute_type = Scalar;  // ZFP decompresses to original scalar type

    static constexpr bool is_block_format = true;
    static constexpr size_t block_elements = format::block_size;
    static constexpr size_t block_dimensions = Dims;

    // Block shape: {4} for 1D, {4,4} for 2D, {4,4,4} for 3D
    static constexpr std::array<size_t, Dims> block_shape =
        detail::make_filled_array<Dims>(size_t{4});

    // Compressed size per block in bytes
    static constexpr size_t compressed_block_bytes =
        (block_elements * Rate + 7) / 8;

    // Decompressed size per block in bytes
    static constexpr size_t decompressed_block_bytes =
        block_elements * sizeof(Scalar);

    // Compression ratio
    static constexpr float compression_ratio =
        static_cast<float>(decompressed_block_bytes) / compressed_block_bytes;

    // Hardware mapping
    static constexpr HardwareType storage_hw_type = HardwareType::ZFP_BLOCK;
    static constexpr HardwareType compute_hw_type =
        ScalarTraits<Scalar>::hw_type;

    // Decompression complexity (cycles per block, approximate)
    static constexpr size_t decompress_cycles =
        block_elements * 2;  // Transform + bit unpacking
};

} // namespace kpu
```

---

## Micro Exponents (MX) Block Format

### Overview

MX formats use a shared exponent across a block of narrow mantissas:
- Developed by Microsoft, AMD, Intel for AI workloads
- Block of 32 elements shares one 8-bit exponent
- Elements have narrow mantissas (2-7 bits depending on variant)
- Much simpler than ZFP: just scale/unscale operations
- Lower compression ratios but faster encode/decode

### Type Definition

```cpp
namespace kpu {

template<size_t MantissaBits>
struct MicroExponent {
    static constexpr size_t mantissa_bits = MantissaBits;
    static constexpr size_t exponent_bits = 8;  // Shared exponent is always 8 bits
    static constexpr size_t block_size = 32;    // MX uses 32-element blocks
};

// Standard MX configurations
using MXFP8 = MicroExponent<7>;  // 1 sign + 7 mantissa = 8 bits, ~2x compression
using MXFP6 = MicroExponent<5>;  // 1 sign + 5 mantissa = 6 bits, ~2.7x compression
using MXFP4 = MicroExponent<3>;  // 1 sign + 3 mantissa = 4 bits, ~4x compression

} // namespace kpu
```

### BlockTraits Specialization

```cpp
namespace kpu {

template<size_t MantissaBits>
struct BlockTraits<MicroExponent<MantissaBits>> {
    using format = MicroExponent<MantissaBits>;
    using scalar_type = float;     // MX is defined relative to float
    using compute_type = float;    // Compute in float after scaling

    static constexpr bool is_block_format = true;
    static constexpr size_t block_elements = 32;
    static constexpr size_t block_dimensions = 1;  // MX is 1D blocks

    static constexpr std::array<size_t, 1> block_shape = {32};

    // Compressed: 32 elements * (1 sign + mantissa_bits) + 8 bits shared exponent
    static constexpr size_t compressed_block_bits =
        block_elements * (1 + MantissaBits) + 8;
    static constexpr size_t compressed_block_bytes =
        (compressed_block_bits + 7) / 8;

    // Decompressed: 32 * float32
    static constexpr size_t decompressed_block_bytes = 32 * sizeof(float);

    static constexpr float compression_ratio =
        static_cast<float>(decompressed_block_bytes) / compressed_block_bytes;

    static constexpr HardwareType storage_hw_type =
        (MantissaBits == 7) ? HardwareType::MXFP8 :
        (MantissaBits == 5) ? HardwareType::MXFP6 :
        (MantissaBits == 3) ? HardwareType::MXFP4 :
        HardwareType::UNSUPPORTED;

    static constexpr HardwareType compute_hw_type = HardwareType::FLOAT32;

    // Decompression is very fast: just multiply by shared scale
    static constexpr size_t decompress_cycles = block_elements;
};

} // namespace kpu
```

---

## Decompression Point Architecture

### The Key Question

Where in the memory hierarchy does decompression happen?

```
External Memory (Compressed)
       | DMA
       v
L3 Buffers (???)
       | BlockMover
       v
L2 Banks (???)
       | Streamer
       v
L1 Streams (???)
       |
       v
Compute Fabric (Must be computable format)
```

### Options

| Option | L3 | L2 | L1 | Pros | Cons |
|--------|----|----|----| -----|------|
| Decompress at DMA | Dense | Dense | Dense | Simple | No bandwidth benefit |
| Decompress at BlockMover | Compressed | Dense | Dense | Good balance | BlockMover complexity |
| Decompress at Streamer | Compressed | Compressed | Dense | Max bandwidth | Streamer complexity |
| Native block compute | Compressed | Compressed | Compressed | Max efficiency | Complex MACs |

### Configuration

```cpp
namespace kpu {

enum class DecompressionPoint {
    AT_DMA,          // Decompress when data enters L3
    AT_BLOCK_MOVER,  // Decompress during L3->L2 transfer
    AT_STREAMER,     // Decompress during L2->L1 streaming
    NATIVE_COMPUTE   // Hardware computes directly on compressed format
};

template<typename BlockFormat>
struct BlockFormatConfig {
    DecompressionPoint decompression_point = DecompressionPoint::AT_BLOCK_MOVER;

    // Derived from decompression_point
    bool l3_compressed() const {
        return decompression_point != DecompressionPoint::AT_DMA;
    }

    bool l2_compressed() const {
        return decompression_point == DecompressionPoint::AT_STREAMER ||
               decompression_point == DecompressionPoint::NATIVE_COMPUTE;
    }

    // L1 is always decompressed unless native compute
    bool l1_compressed() const {
        return decompression_point == DecompressionPoint::NATIVE_COMPUTE;
    }
};

} // namespace kpu
```

---

## BlockTensor Class

```cpp
namespace kpu {

template<typename BlockFormat>
class BlockTensor : public TensorExpr<BlockTensor<BlockFormat>> {
public:
    using traits = BlockTraits<BlockFormat>;
    using scalar_type = typename traits::scalar_type;
    using compute_type = typename traits::compute_type;

    static_assert(traits::is_block_format,
                  "BlockFormat must be a block-compressed format");

    // Construction - shape must be block-aligned
    explicit BlockTensor(std::vector<size_t> shape);

    // Create from uncompressed data (compresses on construction)
    static BlockTensor from_dense(const Tensor<scalar_type>& dense);

    // Properties
    const std::vector<size_t>& shape() const { return shape_; }
    size_t num_blocks() const;
    std::vector<size_t> block_grid_shape() const;

    // Memory sizes
    size_t compressed_bytes() const {
        return num_blocks() * traits::compressed_block_bytes;
    }
    size_t decompressed_bytes() const {
        return num_blocks() * traits::decompressed_block_bytes;
    }
    float compression_ratio() const {
        return static_cast<float>(decompressed_bytes()) / compressed_bytes();
    }

    // Access compressed data
    const void* compressed_data() const { return compressed_data_.get(); }
    void* compressed_data() { return compressed_data_.get(); }

    // Decompress entire tensor
    Tensor<compute_type> decompress() const;

    // Device memory
    Address compressed_device_address() const { return compressed_device_addr_; }
    void to_device(ExecutionContext& ctx);

private:
    std::vector<size_t> shape_;
    std::unique_ptr<uint8_t[]> compressed_data_;
    Address compressed_device_addr_ = 0;

    void validate_alignment(const std::vector<size_t>& shape);
    void allocate_compressed();
};

} // namespace kpu
```

---

## Hardware Decompression Unit

```cpp
namespace kpu {

template<typename BlockFormat>
class DecompressionUnit {
public:
    using traits = BlockTraits<BlockFormat>;

    struct Stats {
        size_t blocks_decompressed = 0;
        size_t compressed_bytes_in = 0;
        size_t decompressed_bytes_out = 0;
        Cycle total_cycles = 0;
    };

    // Decompress a single block
    Cycle decompress_block(const void* compressed, void* decompressed);

    // Decompress a tile (multiple blocks)
    Cycle decompress_tile(const void* compressed, void* decompressed,
                          const std::vector<size_t>& tile_shape);

    const Stats& stats() const { return stats_; }

private:
    Stats stats_;
};

// Specialization for ZFP
template<typename Scalar, size_t Dims, size_t Rate>
class DecompressionUnit<ZFP<Scalar, Dims, Rate>> {
    // ZFP decompression: bit unpacking + inverse transform
    void decompress_block_impl(const void* in, void* out) {
        // 1. Unpack bits to fixed-point values
        // 2. Apply inverse decorrelating transform
        // 3. Convert to floating-point
    }
};

// Specialization for MX
template<size_t MantissaBits>
class DecompressionUnit<MicroExponent<MantissaBits>> {
    // MX decompression: scale by shared exponent
    void decompress_block_impl(const void* in, void* out) {
        // 1. Extract 8-bit shared exponent
        // 2. Extract narrow mantissas
        // 3. Multiply each by 2^exponent
    }
};

} // namespace kpu
```

---

## Expression Templates for Block Tensors

```cpp
namespace kpu {

template<typename LhsFormat, typename RhsFormat>
class BlockMatMulExpr : public TensorExpr<BlockMatMulExpr<LhsFormat, RhsFormat>> {
public:
    using lhs_traits = BlockTraits<LhsFormat>;
    using rhs_traits = BlockTraits<RhsFormat>;

    // Compute type is the common decompressed type
    using scalar_type = std::common_type_t<
        typename lhs_traits::compute_type,
        typename rhs_traits::compute_type
    >;

    BlockMatMulExpr(const BlockTensor<LhsFormat>& lhs,
                    const BlockTensor<RhsFormat>& rhs);

    std::vector<size_t> shape() const;

    // Evaluate into dense tensor
    void eval_into(Tensor<scalar_type>& dst, ExecutionContext& ctx) const;

    // Evaluate into block tensor (recompress output)
    template<typename OutputFormat>
    void eval_into(BlockTensor<OutputFormat>& dst, ExecutionContext& ctx) const;

private:
    const BlockTensor<LhsFormat>& lhs_;
    const BlockTensor<RhsFormat>& rhs_;
};

// Operation functions
template<typename LhsFormat, typename RhsFormat>
auto matmul(const BlockTensor<LhsFormat>& lhs, const BlockTensor<RhsFormat>& rhs) {
    return BlockMatMulExpr<LhsFormat, RhsFormat>(lhs, rhs);
}

// Mixed: block tensor * dense tensor
template<typename BlockFormat, typename Scalar>
auto matmul(const BlockTensor<BlockFormat>& lhs, const Tensor<Scalar>& rhs);

} // namespace kpu
```

---

## Tile Scheduling with Block Alignment

The compiler must respect block boundaries when selecting tile sizes:

```cpp
namespace kpu::compiler {

template<typename BlockFormat>
class BlockAwareTileOptimizer {
public:
    using traits = BlockTraits<BlockFormat>;

    TileConfig optimize(size_t M, size_t N, size_t K,
                        const MemoryHierarchy& mem,
                        const BlockFormatConfig<BlockFormat>& config) {
        // Start with standard optimization
        TileConfig cfg = base_optimizer_.optimize(M, N, K, mem);

        // Round up to block boundaries
        cfg.Ti = align_up(cfg.Ti, block_side<0>());
        cfg.Tj = align_up(cfg.Tj, block_side<1>());
        cfg.Tk = align_up(cfg.Tk, block_side<0>());

        // Verify tiles fit in memory (accounting for compression)
        size_t l3_bytes = config.l3_compressed() ?
            compressed_tile_bytes(cfg) : decompressed_tile_bytes(cfg);

        if (l3_bytes > mem.l3_capacity) {
            cfg = reduce_and_align(cfg, mem, config);
        }

        return cfg;
    }

private:
    template<size_t Dim>
    static constexpr size_t block_side() {
        if constexpr (Dim < traits::block_dimensions) {
            return traits::block_shape[Dim];
        } else {
            return 1;
        }
    }

    static size_t align_up(size_t value, size_t alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }
};

} // namespace kpu::compiler
```

---

## Bandwidth Modeling

Block formats change bandwidth calculations:

```cpp
namespace kpu::compiler {

template<typename BlockFormat>
class CompressedBandwidthModel {
public:
    using traits = BlockTraits<BlockFormat>;

    struct Estimate {
        double external_raw_gbps;        // Raw external bandwidth
        double external_effective_gbps;  // After accounting for compression
        double l3_l2_gbps;
        double l2_l1_gbps;
        double arithmetic_intensity;     // FLOPs per byte from external memory
        std::string bottleneck;          // "MEMORY_BOUND" or "COMPUTE_BOUND"
    };

    Estimate estimate(size_t M, size_t N, size_t K,
                      const BlockFormatConfig<BlockFormat>& config) {
        Estimate result;

        // External sees compressed data
        size_t a_bytes = compressed_matrix_bytes(M, K);
        size_t b_bytes = compressed_matrix_bytes(K, N);
        size_t c_bytes = M * N * sizeof(typename traits::compute_type);
        size_t external_bytes = a_bytes + b_bytes + c_bytes;

        // Compute sees decompressed data
        size_t flops = 2ULL * M * N * K;

        result.arithmetic_intensity =
            static_cast<double>(flops) / external_bytes;

        // Effective bandwidth is amplified by compression ratio
        result.external_effective_gbps =
            result.external_raw_gbps * traits::compression_ratio;

        return result;
    }

private:
    size_t compressed_matrix_bytes(size_t rows, size_t cols) {
        size_t elements = rows * cols;
        size_t blocks = (elements + traits::block_elements - 1) /
                        traits::block_elements;
        return blocks * traits::compressed_block_bytes;
    }
};

} // namespace kpu::compiler
```

---

## Data Flow Pipeline

```
+------------------------------------------------------------------+
|                    Host Memory (Dense Float32)                    |
+------------------------------------------------------------------+
                              |
                              v Compress (BlockTensor::from_dense)
+------------------------------------------------------------------+
|                    Host Memory (ZFP Compressed)                   |
|                    ~2x smaller than dense                         |
+------------------------------------------------------------------+
                              |
                              v DMA (compressed bytes)
+------------------------------------------------------------------+
|                    L3 Buffers (ZFP Compressed)                    |
|                    ~2x bandwidth amplification                    |
+------------------------------------------------------------------+
                              |
                              v BlockMover + Decompress
+------------------------------------------------------------------+
|                    L2 Banks (Dense Float32)                       |
|                    Ready for streaming                            |
+------------------------------------------------------------------+
                              |
                              v Streamer (dense values)
+------------------------------------------------------------------+
|                    L1 Streams (Dense Float32)                     |
+------------------------------------------------------------------+
                              |
                              v Feed to PEs
+------------------------------------------------------------------+
|                    Systolic Array (Float32 MACs)                  |
+------------------------------------------------------------------+
```

---

## Complete Example

```cpp
#include <kpu/tensor.hpp>
#include <kpu/block_tensor.hpp>

int main() {
    kpu::Device device;
    kpu::ExecutionContext ctx(device);

    // Configure ZFP: 2D blocks, 16 bits per value (~2x compression)
    using ZFP16 = kpu::ZFP<float, 2, 16>;

    // Configure decompression point
    kpu::BlockFormatConfig<ZFP16> config;
    config.decompression_point = kpu::DecompressionPoint::AT_BLOCK_MOVER;
    ctx.set_block_config(config);

    // Create dense tensors
    kpu::Tensor<float> A_dense = kpu::Tensor<float>::randn({1024, 512});
    kpu::Tensor<float> B_dense = kpu::Tensor<float>::randn({512, 256});

    // Compress to ZFP (shapes must be 4-aligned)
    kpu::BlockTensor<ZFP16> A = kpu::BlockTensor<ZFP16>::from_dense(A_dense);
    kpu::BlockTensor<ZFP16> B = kpu::BlockTensor<ZFP16>::from_dense(B_dense);

    std::cout << "A compression ratio: " << A.compression_ratio() << "x\n";

    // MatMul on compressed tensors
    auto C_expr = kpu::matmul(A, B);
    kpu::Tensor<float> C = C_expr.eval(ctx);

    // Validate against dense computation
    auto C_ref = kpu::matmul(A_dense, B_dense).eval(ctx);
    float max_error = kpu::max_abs_diff(C, C_ref);
    std::cout << "Max error vs dense: " << max_error << "\n";

    return 0;
}
```

---

## ZFP vs MX Tradeoffs

| Aspect | ZFP | Micro Exponents (MX) |
|--------|-----|----------------------|
| **Block size** | 4^d (16 for 2D) | 32 (1D) |
| **Compression ratio** | 2-10x (configurable) | 2-4x (fixed by variant) |
| **Encode complexity** | High (transform + coding) | Low (find max, scale) |
| **Decode complexity** | High | Very low |
| **Error distribution** | Smooth (transform-based) | Block-correlated |
| **Best for** | Large arrays, high compression | Inference, bandwidth savings |
| **Hardware support** | Needs decoder unit | Simple multiply |

---

## Future Extensions

### Custom Block Formats

Users can define their own block formats by specializing `BlockTraits`:

```cpp
// User-defined block format
struct MyBlockFormat {
    static constexpr size_t block_size = 64;
    // ...
};

template<>
struct kpu::BlockTraits<MyBlockFormat> {
    static constexpr bool is_block_format = true;
    static constexpr size_t block_elements = 64;
    // ... full specialization
};
```

### Native Block Compute

Future hardware could compute directly on compressed formats:

```cpp
// Hypothetical: native MX compute
template<>
struct BlockTraits<MXFP4> {
    // ...
    static constexpr bool supports_native_compute = true;
    static constexpr HardwareType compute_hw_type = HardwareType::MXFP4_NATIVE;
};
```

---

*Document created: 2026-01-15*
*Related: VIRTUAL_PLATFORM_ANALYSIS.md, kpu-execution-model.md*
