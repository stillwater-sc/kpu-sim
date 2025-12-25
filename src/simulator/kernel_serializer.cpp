// Kernel Serializer Implementation
// Binary and JSON serialization for Kernel objects

#include <sw/kpu/kernel_serializer.hpp>
#include <sw/kpu/isa/program_serializer.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace sw::kpu {

using json = nlohmann::json;

// ============================================================================
// Binary Serialization
// ============================================================================

std::vector<uint8_t> KernelSerializer::serialize(const Kernel& kernel) const {
    std::vector<uint8_t> buffer;

    // Magic and version
    write_value(buffer, KERNEL_MAGIC);
    write_value(buffer, KERNEL_VERSION);

    // Basic metadata
    write_string(buffer, kernel.name());
    write_value(buffer, static_cast<uint8_t>(kernel.op_type()));
    write_value(buffer, static_cast<uint8_t>(kernel.dtype()));

    // Dimensions
    write_value(buffer, kernel.M());
    write_value(buffer, kernel.N());
    write_value(buffer, kernel.K());

    // Tile sizes
    write_value(buffer, kernel.Ti());
    write_value(buffer, kernel.Tj());
    write_value(buffer, kernel.Tk());
    write_value(buffer, kernel.program().L1_Ki);

    // MLP-specific
    write_value(buffer, static_cast<uint8_t>(kernel.has_bias() ? 1 : 0));
    write_value(buffer, static_cast<uint8_t>(kernel.activation()));

    // Arguments
    const auto& args = kernel.arguments();
    write_value(buffer, static_cast<uint32_t>(args.size()));

    for (const auto& arg : args) {
        write_string(buffer, arg.name);
        write_value(buffer, static_cast<uint8_t>(arg.dtype));
        write_value(buffer, static_cast<uint8_t>(arg.is_output ? 1 : 0));
        write_value(buffer, static_cast<uint8_t>(arg.shape.size()));
        for (Size dim : arg.shape) {
            write_value(buffer, dim);
        }
        write_value(buffer, arg.size_bytes);
    }

    // Serialize the program
    std::vector<uint8_t> program_data = program_serializer_.serialize(kernel.program());
    write_value(buffer, static_cast<uint32_t>(program_data.size()));
    buffer.insert(buffer.end(), program_data.begin(), program_data.end());

    return buffer;
}

Kernel KernelSerializer::deserialize(const std::vector<uint8_t>& data) const {
    if (data.size() < 8) {
        throw isa::SerializationError("Data too small to be a valid kernel");
    }

    size_t offset = 0;

    // Check magic
    uint32_t magic = read_value<uint32_t>(data, offset);
    if (magic != KERNEL_MAGIC) {
        throw isa::SerializationError("Invalid magic number - not a KPU kernel file");
    }

    // Check version
    uint32_t version = read_value<uint32_t>(data, offset);
    if (version > KERNEL_VERSION) {
        throw isa::SerializationError("Unsupported kernel version: " + std::to_string(version));
    }

    // Basic metadata
    std::string name = read_string(data, offset);
    KernelOpType op_type = static_cast<KernelOpType>(read_value<uint8_t>(data, offset));
    DataType dtype = static_cast<DataType>(read_value<uint8_t>(data, offset));

    // Dimensions
    Size M = read_value<Size>(data, offset);
    Size N = read_value<Size>(data, offset);
    Size K = read_value<Size>(data, offset);

    // Tile sizes
    Size Ti = read_value<Size>(data, offset);
    Size Tj = read_value<Size>(data, offset);
    Size Tk = read_value<Size>(data, offset);
    Size L1_Ki = read_value<Size>(data, offset);

    // MLP-specific
    bool has_bias = read_value<uint8_t>(data, offset) != 0;
    ActivationType activation = static_cast<ActivationType>(read_value<uint8_t>(data, offset));

    // Arguments
    uint32_t num_args = read_value<uint32_t>(data, offset);
    std::vector<KernelArgument> arguments;
    arguments.reserve(num_args);

    for (uint32_t i = 0; i < num_args; ++i) {
        KernelArgument arg;
        arg.name = read_string(data, offset);
        arg.dtype = static_cast<DataType>(read_value<uint8_t>(data, offset));
        arg.is_output = read_value<uint8_t>(data, offset) != 0;
        uint8_t num_dims = read_value<uint8_t>(data, offset);
        arg.shape.resize(num_dims);
        for (uint8_t d = 0; d < num_dims; ++d) {
            arg.shape[d] = read_value<Size>(data, offset);
        }
        arg.size_bytes = read_value<Size>(data, offset);
        arguments.push_back(std::move(arg));
    }

    // Deserialize the program
    uint32_t program_size = read_value<uint32_t>(data, offset);
    if (offset + program_size > data.size()) {
        throw isa::SerializationError("Unexpected end of data reading program");
    }

    std::vector<uint8_t> program_data(data.begin() + offset, data.begin() + offset + program_size);
    isa::DMProgram program = program_serializer_.deserialize(program_data);

    // Reconstruct the kernel
    // Use the appropriate factory based on op_type
    Kernel kernel;
    switch (op_type) {
        case KernelOpType::MATMUL:
            kernel = Kernel::create_matmul(M, N, K, dtype);
            break;
        case KernelOpType::MLP:
            kernel = Kernel::create_mlp(M, N, K, activation, has_bias, dtype);
            break;
        default:
            // For other types, create a basic matmul and override
            kernel = Kernel::create_matmul(M, N, K, dtype);
            break;
    }

    // Override the program with the loaded one
    kernel.set_program(std::move(program));

    return kernel;
}

void KernelSerializer::save(const Kernel& kernel, const std::string& path) const {
    std::vector<uint8_t> data = serialize(kernel);

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw isa::SerializationError("Failed to open file for writing: " + path);
    }

    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    if (!file) {
        throw isa::SerializationError("Failed to write to file: " + path);
    }
}

Kernel KernelSerializer::load(const std::string& path) const {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw isa::SerializationError("Failed to open file for reading: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw isa::SerializationError("Failed to read from file: " + path);
    }

    return deserialize(data);
}

// ============================================================================
// JSON Serialization
// ============================================================================

std::string KernelSerializer::to_json(const Kernel& kernel, bool pretty) const {
    json j;

    // Metadata
    j["name"] = kernel.name();
    j["op_type"] = kernel_op_type_name(kernel.op_type());
    j["dtype"] = dtype_name(kernel.dtype());

    // Dimensions
    j["dimensions"] = json{
        {"M", kernel.M()},
        {"N", kernel.N()},
        {"K", kernel.K()}
    };

    // Tiles
    j["tiles"] = json{
        {"Ti", kernel.Ti()},
        {"Tj", kernel.Tj()},
        {"Tk", kernel.Tk()},
        {"L1_Ki", kernel.program().L1_Ki}
    };

    // MLP-specific
    j["has_bias"] = kernel.has_bias();
    j["activation"] = activation_type_name(kernel.activation());

    // Arguments
    j["arguments"] = json::array();
    for (const auto& arg : kernel.arguments()) {
        json arg_j;
        arg_j["name"] = arg.name;
        arg_j["dtype"] = dtype_name(arg.dtype);
        arg_j["is_output"] = arg.is_output;
        arg_j["shape"] = arg.shape;
        arg_j["size_bytes"] = arg.size_bytes;
        j["arguments"].push_back(arg_j);
    }

    // Statistics
    j["stats"] = json{
        {"instruction_count", kernel.instruction_count()},
        {"total_flops", kernel.total_flops()},
        {"total_input_bytes", kernel.total_input_bytes()},
        {"total_output_bytes", kernel.total_output_bytes()},
        {"arithmetic_intensity", kernel.arithmetic_intensity()}
    };

    // Embed the program as nested JSON
    j["program"] = json::parse(program_serializer_.to_json(kernel.program(), false));

    return pretty ? j.dump(2) : j.dump();
}

Kernel KernelSerializer::from_json(const std::string& json_str) const {
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::parse_error& e) {
        throw isa::SerializationError("JSON parse error: " + std::string(e.what()));
    }

    // Parse basic info
    std::string name = j.at("name").get<std::string>();
    std::string op_type_str = j.at("op_type").get<std::string>();
    std::string dtype_str = j.at("dtype").get<std::string>();

    // Parse dimensions
    Size M = j.at("dimensions").at("M").get<Size>();
    Size N = j.at("dimensions").at("N").get<Size>();
    Size K = j.at("dimensions").at("K").get<Size>();

    // Parse MLP-specific
    bool has_bias = j.at("has_bias").get<bool>();
    std::string activation_str = j.at("activation").get<std::string>();

    // Convert strings to enums
    DataType dtype = DataType::FLOAT32;  // Default
    if (dtype_str == "float16" || dtype_str == "FLOAT16") dtype = DataType::FLOAT16;
    else if (dtype_str == "bfloat16" || dtype_str == "BFLOAT16") dtype = DataType::BFLOAT16;
    else if (dtype_str == "int8" || dtype_str == "INT8") dtype = DataType::INT8;
    else if (dtype_str == "int4" || dtype_str == "INT4") dtype = DataType::INT4;

    ActivationType activation = ActivationType::NONE;
    if (activation_str == "relu" || activation_str == "RELU") activation = ActivationType::RELU;
    else if (activation_str == "gelu" || activation_str == "GELU") activation = ActivationType::GELU;
    else if (activation_str == "sigmoid" || activation_str == "SIGMOID") activation = ActivationType::SIGMOID;
    else if (activation_str == "tanh" || activation_str == "TANH") activation = ActivationType::TANH;
    else if (activation_str == "silu" || activation_str == "SILU") activation = ActivationType::SILU;

    // Create kernel based on op type
    Kernel kernel;
    if (op_type_str == "mlp" || op_type_str == "MLP") {
        kernel = Kernel::create_mlp(M, N, K, activation, has_bias, dtype);
    } else {
        kernel = Kernel::create_matmul(M, N, K, dtype);
    }

    // Parse and set the program
    std::string program_json = j.at("program").dump();
    isa::DMProgram program = program_serializer_.from_json(program_json);
    kernel.set_program(std::move(program));

    return kernel;
}

void KernelSerializer::save_json(const Kernel& kernel, const std::string& path, bool pretty) const {
    std::string json_str = to_json(kernel, pretty);

    std::ofstream file(path);
    if (!file) {
        throw isa::SerializationError("Failed to open file for writing: " + path);
    }
    file << json_str;
}

Kernel KernelSerializer::load_json(const std::string& path) const {
    std::ifstream file(path);
    if (!file) {
        throw isa::SerializationError("Failed to open file for reading: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return from_json(buffer.str());
}

// ============================================================================
// Utilities
// ============================================================================

bool KernelSerializer::validate(const std::vector<uint8_t>& data) const {
    if (data.size() < 8) return false;

    uint32_t magic;
    std::memcpy(&magic, data.data(), sizeof(magic));
    if (magic != KERNEL_MAGIC) return false;

    uint32_t version;
    std::memcpy(&version, data.data() + 4, sizeof(version));
    if (version > KERNEL_VERSION) return false;

    return true;
}

std::string KernelSerializer::detect_format(const std::string& path) {
    size_t dot = path.rfind('.');
    if (dot == std::string::npos) return "binary";

    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".kpukernel" || ext == ".bin") return "binary";
    if (ext == ".json") return "json";

    return "binary";
}

Kernel KernelSerializer::load_auto(const std::string& path) const {
    std::string format = detect_format(path);
    if (format == "json") {
        return load_json(path);
    }
    return load(path);
}

void KernelSerializer::save_auto(const Kernel& kernel, const std::string& path) const {
    std::string format = detect_format(path);
    if (format == "json") {
        save_json(kernel, path);
    } else {
        save(kernel, path);
    }
}

// ============================================================================
// String Helpers
// ============================================================================

void KernelSerializer::write_string(std::vector<uint8_t>& buffer, const std::string& str) const {
    write_value(buffer, static_cast<uint16_t>(str.size()));
    buffer.insert(buffer.end(), str.begin(), str.end());
}

std::string KernelSerializer::read_string(const std::vector<uint8_t>& data, size_t& offset) const {
    uint16_t len = read_value<uint16_t>(data, offset);
    if (offset + len > data.size()) {
        throw isa::SerializationError("Unexpected end of data reading string");
    }
    std::string str(data.begin() + offset, data.begin() + offset + len);
    offset += len;
    return str;
}

} // namespace sw::kpu
