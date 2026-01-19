/**
 * @file dfx_parser.cpp
 * @brief DFX JSON parser implementation
 */

#include <sw/kpu/dfx/dfx_parser.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include <algorithm>

namespace sw::kpu::dfx {

using json = nlohmann::json;

// ============================================================================
// Data Type Conversion
// ============================================================================

DataType string_to_dtype(const std::string& s) {
    static const std::unordered_map<std::string, DataType> dtype_map = {
        {"f32", DataType::FLOAT32},
        {"f16", DataType::FLOAT16},
        {"bf16", DataType::BFLOAT16},
        {"i32", DataType::INT32},
        {"i16", DataType::INT16},
        {"i8", DataType::INT8},
        {"u8", DataType::UINT8},
        {"bool", DataType::BOOL},
    };

    auto it = dtype_map.find(s);
    return (it != dtype_map.end()) ? it->second : DataType::FLOAT32;
}

std::string dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "f32";
        case DataType::FLOAT16: return "f16";
        case DataType::BFLOAT16: return "bf16";
        case DataType::INT32: return "i32";
        case DataType::INT16: return "i16";
        case DataType::INT8: return "i8";
        case DataType::UINT8: return "u8";
        case DataType::BOOL: return "bool";
    }
    return "f32";
}

size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
        case DataType::INT32:
            return 4;
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
        case DataType::INT16:
            return 2;
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
            return 1;
    }
    return 4;
}

// ============================================================================
// OpCode Conversion
// ============================================================================

OpCode string_to_opcode(const std::string& s) {
    static const std::unordered_map<std::string, OpCode> opcode_map = {
        // Data movement
        {"load", OpCode::LOAD},
        {"store", OpCode::STORE},
        {"prefetch", OpCode::PREFETCH},
        {"copy", OpCode::COPY},
        // Compute - matrix
        {"matmul", OpCode::MATMUL},
        {"conv2d", OpCode::CONV2D},
        // Compute - activation
        {"relu", OpCode::RELU},
        {"gelu", OpCode::GELU},
        {"silu", OpCode::SILU},
        {"sigmoid", OpCode::SIGMOID},
        {"tanh", OpCode::TANH},
        {"softmax", OpCode::SOFTMAX},
        // Compute - normalization
        {"layer_norm", OpCode::LAYER_NORM},
        {"batch_norm", OpCode::BATCH_NORM},
        // Compute - elementwise
        {"add", OpCode::ADD},
        {"sub", OpCode::SUB},
        {"mul", OpCode::MUL},
        {"div", OpCode::DIV},
        {"neg", OpCode::NEG},
        {"exp", OpCode::EXP},
        {"log", OpCode::LOG},
        {"sqrt", OpCode::SQRT},
        // Compute - reduction
        {"sum", OpCode::SUM},
        {"mean", OpCode::MEAN},
        {"max", OpCode::MAX},
        {"min", OpCode::MIN},
        // Compute - pooling
        {"maxpool2d", OpCode::MAXPOOL2D},
        {"avgpool2d", OpCode::AVGPOOL2D},
        {"adaptive_avgpool2d", OpCode::ADAPTIVE_AVGPOOL2D},
        // Shape operations
        {"reshape", OpCode::RESHAPE},
        {"transpose", OpCode::TRANSPOSE},
        {"concat", OpCode::CONCAT},
        {"flatten", OpCode::FLATTEN},
        // Control
        {"barrier", OpCode::BARRIER},
        {"nop", OpCode::NOP},
    };

    auto it = opcode_map.find(s);
    return (it != opcode_map.end()) ? it->second : OpCode::UNKNOWN;
}

std::string opcode_to_string(OpCode op) {
    switch (op) {
        case OpCode::LOAD: return "load";
        case OpCode::STORE: return "store";
        case OpCode::PREFETCH: return "prefetch";
        case OpCode::COPY: return "copy";
        case OpCode::MATMUL: return "matmul";
        case OpCode::CONV2D: return "conv2d";
        case OpCode::RELU: return "relu";
        case OpCode::GELU: return "gelu";
        case OpCode::SILU: return "silu";
        case OpCode::SIGMOID: return "sigmoid";
        case OpCode::TANH: return "tanh";
        case OpCode::SOFTMAX: return "softmax";
        case OpCode::LAYER_NORM: return "layer_norm";
        case OpCode::BATCH_NORM: return "batch_norm";
        case OpCode::ADD: return "add";
        case OpCode::SUB: return "sub";
        case OpCode::MUL: return "mul";
        case OpCode::DIV: return "div";
        case OpCode::NEG: return "neg";
        case OpCode::EXP: return "exp";
        case OpCode::LOG: return "log";
        case OpCode::SQRT: return "sqrt";
        case OpCode::SUM: return "sum";
        case OpCode::MEAN: return "mean";
        case OpCode::MAX: return "max";
        case OpCode::MIN: return "min";
        case OpCode::MAXPOOL2D: return "maxpool2d";
        case OpCode::AVGPOOL2D: return "avgpool2d";
        case OpCode::ADAPTIVE_AVGPOOL2D: return "adaptive_avgpool2d";
        case OpCode::RESHAPE: return "reshape";
        case OpCode::TRANSPOSE: return "transpose";
        case OpCode::CONCAT: return "concat";
        case OpCode::FLATTEN: return "flatten";
        case OpCode::BARRIER: return "barrier";
        case OpCode::NOP: return "nop";
        case OpCode::UNKNOWN: return "unknown";
    }
    return "unknown";
}

// ============================================================================
// Program Summary
// ============================================================================

std::string Program::summary() const {
    std::ostringstream oss;
    oss << "DFXProgram '" << name << "'\n";
    oss << "  Version: " << version << "\n";
    oss << "  Tensors: " << tensors.size() << "\n";
    oss << "  Operations: " << ops.size() << "\n";
    oss << "  Inputs: [";
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << inputs[i];
    }
    oss << "]\n";
    oss << "  Outputs: [";
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << outputs[i];
    }
    oss << "]\n\n";
    oss << "  Operations:\n";
    for (size_t i = 0; i < ops.size(); ++i) {
        const auto& op = ops[i];
        oss << "    [" << i << "] " << opcode_to_string(op.opcode) << "(";
        for (size_t j = 0; j < op.inputs.size(); ++j) {
            if (j > 0) oss << ", ";
            oss << op.inputs[j];
        }
        oss << ") -> (";
        for (size_t j = 0; j < op.outputs.size(); ++j) {
            if (j > 0) oss << ", ";
            oss << op.outputs[j];
        }
        oss << ")\n";
    }
    return oss.str();
}

// ============================================================================
// Parser Implementation
// ============================================================================

namespace {

/**
 * @brief Parse attribute value from JSON
 */
AttrValue parse_attr_value_impl(const json& j) {
    if (j.is_number_integer()) {
        return static_cast<int64_t>(j.get<int64_t>());
    } else if (j.is_number_float()) {
        return j.get<double>();
    } else if (j.is_boolean()) {
        return j.get<bool>();
    } else if (j.is_string()) {
        return j.get<std::string>();
    } else if (j.is_array()) {
        std::vector<int64_t> vec;
        for (const auto& elem : j) {
            if (elem.is_number()) {
                vec.push_back(static_cast<int64_t>(elem.get<int64_t>()));
            }
        }
        return vec;
    }
    return static_cast<int64_t>(0);
}

/**
 * @brief Parse tensor from JSON
 */
Tensor parse_tensor_impl(const json& j) {
    Tensor tensor;
    tensor.name = j.value("name", "");
    tensor.dtype = string_to_dtype(j.value("dtype", "f32"));
    tensor.memory_level = static_cast<MemLevel>(j.value("memory_level", 0));
    tensor.is_const = j.value("is_const", false);

    if (j.contains("shape") && j["shape"].is_array()) {
        for (const auto& dim : j["shape"]) {
            tensor.shape.push_back(dim.get<int64_t>());
        }
    }

    return tensor;
}

/**
 * @brief Parse operation from JSON
 */
Op parse_op_impl(const json& j) {
    Op op;
    op.opcode = string_to_opcode(j.value("opcode", "unknown"));

    if (j.contains("inputs") && j["inputs"].is_array()) {
        for (const auto& input : j["inputs"]) {
            op.inputs.push_back(input.get<std::string>());
        }
    }

    if (j.contains("outputs") && j["outputs"].is_array()) {
        for (const auto& output : j["outputs"]) {
            op.outputs.push_back(output.get<std::string>());
        }
    }

    if (j.contains("attrs") && j["attrs"].is_object()) {
        for (auto& [key, val] : j["attrs"].items()) {
            op.attrs[key] = parse_attr_value_impl(val);
        }
    }

    return op;
}

/**
 * @brief Parse metadata from JSON
 */
Metadata parse_metadata_impl(const json& j) {
    Metadata meta;
    meta.num_ops = j.value("num_ops", static_cast<int64_t>(0));
    meta.num_tensors = j.value("num_tensors", static_cast<int64_t>(0));
    meta.total_matmul_flops = j.value("total_matmul_flops", static_cast<int64_t>(0));

    if (j.contains("op_counts") && j["op_counts"].is_object()) {
        for (auto& [key, val] : j["op_counts"].items()) {
            meta.op_counts[key] = val.get<int64_t>();
        }
    }

    return meta;
}

} // anonymous namespace

// ============================================================================
// DFXParser Methods
// ============================================================================

Program DFXParser::parse_json(const std::string& json_str) {
    try {
        json j = json::parse(json_str);
        return parse_json_object(j);
    } catch (const json::parse_error& e) {
        throw ParseError("JSON parse error: " + std::string(e.what()));
    } catch (const json::type_error& e) {
        throw ParseError("JSON type error: " + std::string(e.what()));
    }
}

Program DFXParser::parse_json_object(const nlohmann::json& j) {
    Program prog;

    // Parse basic fields
    prog.name = j.value("name", "unnamed");
    prog.version = j.value("version", "1.0");

    // Parse inputs and outputs
    if (j.contains("inputs") && j["inputs"].is_array()) {
        for (const auto& input : j["inputs"]) {
            prog.inputs.push_back(input.get<std::string>());
        }
    }

    if (j.contains("outputs") && j["outputs"].is_array()) {
        for (const auto& output : j["outputs"]) {
            prog.outputs.push_back(output.get<std::string>());
        }
    }

    // Parse tensors
    if (j.contains("tensors") && j["tensors"].is_object()) {
        for (auto& [name, tensor_json] : j["tensors"].items()) {
            prog.tensors[name] = parse_tensor_impl(tensor_json);
        }
    }

    // Parse operations
    if (j.contains("ops") && j["ops"].is_array()) {
        for (const auto& op_json : j["ops"]) {
            prog.ops.push_back(parse_op_impl(op_json));
        }
    }

    // Parse metadata
    if (j.contains("metadata") && j["metadata"].is_object()) {
        prog.metadata = parse_metadata_impl(j["metadata"]);
    }

    return prog;
}

} // namespace sw::kpu::dfx
