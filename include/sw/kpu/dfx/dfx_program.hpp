/**
 * @file dfx_program.hpp
 * @brief High-level DFX Program representation matching Python kpu package format
 *
 * This file defines C++ structures that mirror the Python DFX format from
 * python/kpu/dfx_emitter.py. This enables parsing DFX JSON from Python and
 * executing it directly in the C++ simulator.
 *
 * The Python DFX format is operation-level (matmul, conv2d, relu, etc.),
 * while the lower-level sw::kpu::compiler::dfx format is tile-level.
 * This high-level format is used by the native Python bindings.
 *
 * @note For tile-level DFX representation, see include/sw/compiler/dfx/dfx.hpp
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <optional>

namespace sw::kpu::dfx {

// ============================================================================
// Data Types (matching Python DFXDataType)
// ============================================================================

/**
 * @brief Element data types supported by high-level DFX
 */
enum class DataType {
    FLOAT32,    ///< f32 - 32-bit floating point
    FLOAT16,    ///< f16 - 16-bit floating point
    BFLOAT16,   ///< bf16 - Brain floating point 16
    INT32,      ///< i32 - 32-bit signed integer
    INT16,      ///< i16 - 16-bit signed integer
    INT8,       ///< i8 - 8-bit signed integer
    UINT8,      ///< u8 - 8-bit unsigned integer
    BOOL        ///< bool - Boolean type
};

/**
 * @brief Convert string to DataType
 */
DataType string_to_dtype(const std::string& s);

/**
 * @brief Convert DataType to string
 */
std::string dtype_to_string(DataType dtype);

/**
 * @brief Get size in bytes for a data type
 */
size_t dtype_size(DataType dtype);

// ============================================================================
// Memory Levels (matching Python DFXMemLevel)
// ============================================================================

/**
 * @brief Memory hierarchy levels
 */
enum class MemLevel {
    EXTERNAL = 0,   ///< DRAM / Host memory
    L3 = 1,         ///< L3 SRAM (shared across tiles)
    L2 = 2,         ///< L2 SRAM (per-tile)
    L1 = 3,         ///< L1 register file (per-PE)
    REGISTER = 4    ///< Accumulator registers
};

// ============================================================================
// Operation Codes (matching Python DFXOpCode)
// ============================================================================

/**
 * @brief DFX operation codes
 *
 * These map directly to the Python DFXOpCode enum values.
 */
enum class OpCode {
    // Data movement
    LOAD,
    STORE,
    PREFETCH,
    COPY,

    // Compute - matrix
    MATMUL,
    CONV2D,

    // Compute - activation
    RELU,
    GELU,
    SILU,
    SIGMOID,
    TANH,
    SOFTMAX,

    // Compute - normalization
    LAYER_NORM,
    BATCH_NORM,

    // Compute - elementwise
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    EXP,
    LOG,
    SQRT,

    // Compute - reduction
    SUM,
    MEAN,
    MAX,
    MIN,

    // Compute - pooling
    MAXPOOL2D,
    AVGPOOL2D,
    ADAPTIVE_AVGPOOL2D,

    // Shape operations
    RESHAPE,
    TRANSPOSE,
    CONCAT,
    FLATTEN,

    // Control
    BARRIER,
    NOP,

    // Unknown
    UNKNOWN
};

/**
 * @brief Convert string to OpCode
 */
OpCode string_to_opcode(const std::string& s);

/**
 * @brief Convert OpCode to string
 */
std::string opcode_to_string(OpCode op);

// ============================================================================
// Attribute Value Type
// ============================================================================

/**
 * @brief Variant type for operation attributes
 *
 * Attributes can be integers, floats, booleans, strings, or lists of integers.
 */
using AttrValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>
>;

/**
 * @brief Attribute map type
 */
using AttrMap = std::unordered_map<std::string, AttrValue>;

// ============================================================================
// Tensor Descriptor
// ============================================================================

/**
 * @brief Tensor descriptor in DFX program
 *
 * Represents a named tensor with shape, type, and memory location.
 * Matches Python DFXTensor.
 */
struct Tensor {
    std::string name;               ///< Unique tensor name
    std::vector<int64_t> shape;     ///< Tensor dimensions
    DataType dtype = DataType::FLOAT32;
    MemLevel memory_level = MemLevel::EXTERNAL;
    bool is_const = false;          ///< True for weights that don't change

    /**
     * @brief Calculate total number of elements
     */
    int64_t num_elements() const {
        int64_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * @brief Calculate total size in bytes
     */
    size_t size_bytes() const {
        return static_cast<size_t>(num_elements()) * dtype_size(dtype);
    }
};

// ============================================================================
// Operation
// ============================================================================

/**
 * @brief A single operation in the DFX program
 *
 * Matches Python DFXOp structure.
 */
struct Op {
    OpCode opcode = OpCode::UNKNOWN;
    std::vector<std::string> inputs;    ///< Input tensor names
    std::vector<std::string> outputs;   ///< Output tensor names
    AttrMap attrs;                       ///< Operation attributes

    /**
     * @brief Get attribute value with type checking
     */
    template<typename T>
    std::optional<T> get_attr(const std::string& key) const {
        auto it = attrs.find(key);
        if (it == attrs.end()) {
            return std::nullopt;
        }
        if (auto* val = std::get_if<T>(&it->second)) {
            return *val;
        }
        return std::nullopt;
    }

    /**
     * @brief Get integer attribute with default
     */
    int64_t get_int(const std::string& key, int64_t default_val = 0) const {
        return get_attr<int64_t>(key).value_or(default_val);
    }

    /**
     * @brief Get float attribute with default
     */
    double get_float(const std::string& key, double default_val = 0.0) const {
        return get_attr<double>(key).value_or(default_val);
    }

    /**
     * @brief Get bool attribute with default
     */
    bool get_bool(const std::string& key, bool default_val = false) const {
        return get_attr<bool>(key).value_or(default_val);
    }

    /**
     * @brief Get shape/list attribute
     */
    std::vector<int64_t> get_shape(const std::string& key) const {
        auto opt = get_attr<std::vector<int64_t>>(key);
        return opt.value_or(std::vector<int64_t>{});
    }
};

// ============================================================================
// Program Metadata
// ============================================================================

/**
 * @brief Metadata about the DFX program
 */
struct Metadata {
    int64_t num_ops = 0;
    int64_t num_tensors = 0;
    int64_t total_matmul_flops = 0;
    std::unordered_map<std::string, int64_t> op_counts;
};

// ============================================================================
// DFX Program
// ============================================================================

/**
 * @brief Complete DFX program
 *
 * Contains tensor descriptors and operations in execution order.
 * Matches Python DFXProgram structure.
 */
struct Program {
    std::string name;                                   ///< Program name
    std::string version = "1.0";                        ///< DFX format version
    std::unordered_map<std::string, Tensor> tensors;    ///< Tensor descriptors
    std::vector<Op> ops;                                ///< Operations in order
    std::vector<std::string> inputs;                    ///< Input tensor names
    std::vector<std::string> outputs;                   ///< Output tensor names
    Metadata metadata;                                  ///< Program metadata

    /**
     * @brief Find tensor by name
     */
    const Tensor* find_tensor(const std::string& name) const {
        auto it = tensors.find(name);
        return (it != tensors.end()) ? &it->second : nullptr;
    }

    /**
     * @brief Check if program is valid
     */
    bool is_valid() const {
        return !name.empty() && !ops.empty() && !inputs.empty() && !outputs.empty();
    }

    /**
     * @brief Get human-readable summary
     */
    std::string summary() const;
};

} // namespace sw::kpu::dfx
