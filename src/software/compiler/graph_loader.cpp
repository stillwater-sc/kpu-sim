#include <sw/compiler/graph_loader.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>

// For JSON parsing
#ifdef KPU_HAS_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

// For domain_flow parsing
#ifdef KPU_HAS_DOMAIN_FLOW
#include <dfa/dfa.hpp>
// Note: We may need util/data_file.hpp for some utilities, but keeping it minimal for now
#endif

namespace sw::kpu::compiler {

// ============================================================================
// ComputationalGraph Implementation
// ============================================================================

bool ComputationalGraph::validate() const {
    // Check all operator inputs reference valid tensors
    for (const auto& op : operators) {
        for (const auto& input : op->input_tensors) {
            if (tensors.find(input) == tensors.end()) {
                std::cerr << "Error: Operator '" << op->name
                         << "' references undefined input tensor '" << input << "'\n";
                return false;
            }
        }
        for (const auto& output : op->output_tensors) {
            if (tensors.find(output) == tensors.end()) {
                std::cerr << "Error: Operator '" << op->name
                         << "' references undefined output tensor '" << output << "'\n";
                return false;
            }
        }
    }
    return true;
}

std::vector<Operator*> ComputationalGraph::get_execution_order() const {
    // Simple topological sort
    // TODO: Implement proper topological sort based on data dependencies
    std::vector<Operator*> order;
    for (const auto& op : operators) {
        order.push_back(op.get());
    }
    return order;
}

void ComputationalGraph::print() const {
    std::cout << "Graph: " << name << "\n";
    std::cout << "Tensors (" << tensors.size() << "):\n";
    for (const auto& [name, desc] : tensors) {
        std::cout << "  " << name << ": [";
        for (size_t i = 0; i < desc.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << desc.shape[i];
        }
        std::cout << "] " << desc.dtype << "\n";
    }

    std::cout << "Operators (" << operators.size() << "):\n";
    for (const auto& op : operators) {
        std::cout << "  " << op->name << " (type=" << static_cast<int>(op->type) << ")\n";
        std::cout << "    inputs: ";
        for (const auto& in : op->input_tensors) std::cout << in << " ";
        std::cout << "\n    outputs: ";
        for (const auto& out : op->output_tensors) std::cout << out << " ";
        std::cout << "\n";
    }
}

// ============================================================================
// JSONGraphLoader Implementation
// ============================================================================

std::unique_ptr<ComputationalGraph> JSONGraphLoader::load(const std::string& filepath) {
#ifdef KPU_HAS_JSON
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    json j;
    file >> j;

    auto graph = std::make_unique<ComputationalGraph>(j["name"]);

    // Parse tensors
    for (const auto& [name, tensor_json] : j["tensors"].items()) {
        TensorDescriptor desc;
        desc.name = name;
        desc.shape = tensor_json["shape"].get<std::vector<size_t>>();
        desc.dtype = tensor_json["dtype"].get<std::string>();
        desc.layout = tensor_json.value("layout", "row_major");

        // Calculate size in bytes
        size_t element_size = 4; // Default to float32
        if (desc.dtype == "float32" || desc.dtype == "int32") element_size = 4;
        else if (desc.dtype == "float16") element_size = 2;
        else if (desc.dtype == "int8") element_size = 1;

        desc.size_bytes = element_size;
        for (size_t dim : desc.shape) {
            desc.size_bytes *= dim;
        }

        graph->add_tensor(desc);
    }

    // Parse operators
    for (const auto& op_json : j["operators"]) {
        std::string type_str = op_json["type"];
        OperatorType type = OperatorType::UNKNOWN;

        // Convert string to OperatorType
        if (type_str == "MATMUL" || type_str == "GEMM") type = OperatorType::MATMUL;
        else if (type_str == "LINEAR" || type_str == "FC") type = OperatorType::LINEAR;
        else if (type_str == "CONV2D") type = OperatorType::CONV2D;
        else if (type_str == "CONV2D_DEPTHWISE") type = OperatorType::CONV2D_DEPTHWISE;
        else if (type_str == "CONV3D") type = OperatorType::CONV3D;
        else if (type_str == "RELU") type = OperatorType::RELU;
        else if (type_str == "GELU") type = OperatorType::GELU;
        else if (type_str == "SIGMOID") type = OperatorType::SIGMOID;
        else if (type_str == "ELEMENTWISE_ADD" || type_str == "ADD") type = OperatorType::ELEMENTWISE_ADD;
        else if (type_str == "ELEMENTWISE_MUL" || type_str == "MUL") type = OperatorType::ELEMENTWISE_MUL;
        else if (type_str == "ELEMENTWISE_SUB" || type_str == "SUB") type = OperatorType::ELEMENTWISE_SUB;
        else if (type_str == "ELEMENTWISE_DIV" || type_str == "DIV") type = OperatorType::ELEMENTWISE_DIV;
        else if (type_str == "MAXPOOL" || type_str == "MAXPOOL2D") type = OperatorType::MAXPOOL;
        else if (type_str == "AVGPOOL" || type_str == "AVGPOOL2D") type = OperatorType::AVGPOOL;
        else if (type_str == "RESHAPE") type = OperatorType::RESHAPE;
        else if (type_str == "TRANSPOSE") type = OperatorType::TRANSPOSE;
        else if (type_str == "CONCAT") type = OperatorType::CONCAT;
        else if (type_str == "SPLIT") type = OperatorType::SPLIT;
        else if (type_str == "PAD") type = OperatorType::PAD;
        else if (type_str == "GATHER") type = OperatorType::GATHER;
        else if (type_str == "CAST") type = OperatorType::CAST;
        else if (type_str == "CLAMP") type = OperatorType::CLAMP;
        else if (type_str == "ABS") type = OperatorType::ABS;
        else if (type_str == "NEGATE") type = OperatorType::NEGATE;
        else if (type_str == "EXP") type = OperatorType::EXP;
        else if (type_str == "RECIPROCAL") type = OperatorType::RECIPROCAL;
        else if (type_str == "REDUCE_SUM") type = OperatorType::REDUCE_SUM;
        else if (type_str == "REDUCE_MAX") type = OperatorType::REDUCE_MAX;
        else if (type_str == "REDUCE_MIN") type = OperatorType::REDUCE_MIN;
        else if (type_str == "REDUCE_PROD") type = OperatorType::REDUCE_PROD;
        else if (type_str == "LAYERNORM") type = OperatorType::LAYERNORM;
        else if (type_str == "BATCHNORM") type = OperatorType::BATCHNORM;
        else if (type_str == "SOFTMAX") type = OperatorType::SOFTMAX;
        else if (type_str == "ATTENTION") type = OperatorType::ATTENTION;

        auto* op = graph->add_operator(op_json["name"], type);

        // Add inputs
        for (const auto& input : op_json["inputs"]) {
            op->add_input(input);
        }

        // Add outputs
        for (const auto& output : op_json["outputs"]) {
            op->add_output(output);
        }

        // Add attributes
        if (op_json.contains("attributes")) {
            for (const auto& [key, value] : op_json["attributes"].items()) {
                op->set_attribute(key, value.dump());
            }
        }
    }

    if (!graph->validate()) {
        throw std::runtime_error("Graph validation failed");
    }

    return graph;
#else
    throw std::runtime_error("JSON support not available (nlohmann_json not found)");
#endif
}

// ============================================================================
// DomainFlowGraphLoader Implementation
// ============================================================================

#ifdef KPU_HAS_DOMAIN_FLOW

std::unique_ptr<ComputationalGraph> DomainFlowGraphLoader::load(const std::string& filepath) {
    // Use domain_flow's native serialization format (no LLVM/MLIR)
    auto graph = std::make_unique<ComputationalGraph>("domain_flow_graph");

    std::cout << "DomainFlowGraphLoader::load() - Using native serialization format\n";
    std::cout << "  File: " << filepath << "\n";

    // Parse domain_flow native format
    if (!parse_domain_flow_format(filepath, graph.get())) {
        throw std::runtime_error("Failed to parse domain_flow format: " + filepath);
    }

    return graph;
}

bool DomainFlowGraphLoader::parse_domain_flow_format(const std::string& filepath,
                                                      ComputationalGraph* graph) {
    try {
        // Create and load domain_flow graph
        sw::dfa::DomainFlowGraph dfg("loaded_graph");
        dfg.load(filepath);

        // Set graph name
        graph->name = dfg.getName();

        // Convert tensors
        // Discover tensors through operators
        std::unordered_set<std::string> tensor_names;

        // Convert operators - iterate using nodes() which returns a map
        for (const auto& [node_id, df_node] : dfg.nodes()) {
            // Create KPU operator
            std::string op_name = df_node.getName();

            // Get operator type (returns DomainFlowOperator enum)
            auto df_op_type = df_node.getOperatorType();
            // Convert domain_flow enum to KPU OperatorType
            OperatorType op_type = OperatorType::UNKNOWN;

            // Map DomainFlowOperator to KPU OperatorType
            using DFOp = sw::dfa::DomainFlowOperator;
            switch (df_op_type) {
                case DFOp::MATMUL:
                    op_type = OperatorType::MATMUL;
                    break;
                case DFOp::CONV2D:
                    op_type = OperatorType::CONV2D;
                    break;
                case DFOp::DEPTHWISE_CONV2D:
                    op_type = OperatorType::CONV2D_DEPTHWISE;
                    break;
                case DFOp::ADD:
                    op_type = OperatorType::ELEMENTWISE_ADD;
                    break;
                case DFOp::MUL:
                    op_type = OperatorType::ELEMENTWISE_MUL;
                    break;
                case DFOp::RELU:
                    op_type = OperatorType::RELU;
                    break;
                case DFOp::SIGMOID:
                    op_type = OperatorType::SIGMOID;
                    break;
                case DFOp::LINEAR:
                case DFOp::FC:
                    op_type = OperatorType::LINEAR;
                    break;
                case DFOp::CONV3D:
                    op_type = OperatorType::CONV3D;
                    break;
                case DFOp::SUB:
                    op_type = OperatorType::ELEMENTWISE_SUB;
                    break;
                case DFOp::DIV:
                    op_type = OperatorType::ELEMENTWISE_DIV;
                    break;
                case DFOp::PAD:
                    op_type = OperatorType::PAD;
                    break;
                case DFOp::GATHER:
                    op_type = OperatorType::GATHER;
                    break;
                case DFOp::CAST:
                    op_type = OperatorType::CAST;
                    break;
                case DFOp::CLAMP:
                    op_type = OperatorType::CLAMP;
                    break;
                case DFOp::ABS:
                    op_type = OperatorType::ABS;
                    break;
                case DFOp::NEGATE:
                    op_type = OperatorType::NEGATE;
                    break;
                case DFOp::EXP:
                    op_type = OperatorType::EXP;
                    break;
                case DFOp::RECIPROCAL:
                    op_type = OperatorType::RECIPROCAL;
                    break;
                case DFOp::REDUCE_SUM:
                    op_type = OperatorType::REDUCE_SUM;
                    break;
                case DFOp::REDUCE_MAX:
                    op_type = OperatorType::REDUCE_MAX;
                    break;
                case DFOp::REDUCE_MIN:
                    op_type = OperatorType::REDUCE_MIN;
                    break;
                case DFOp::REDUCE_PROD:
                    op_type = OperatorType::REDUCE_PROD;
                    break;
                case DFOp::MAXPOOL2D:
                    op_type = OperatorType::MAXPOOL;
                    break;
                case DFOp::AVGPOOL2D:
                    op_type = OperatorType::AVGPOOL;
                    break;
                case DFOp::RESHAPE:
                    op_type = OperatorType::RESHAPE;
                    break;
                case DFOp::TRANSPOSE:
                    op_type = OperatorType::TRANSPOSE;
                    break;
                case DFOp::CONCAT:
                    op_type = OperatorType::CONCAT;
                    break;
                // Add more mappings as needed
                default:
                    op_type = OperatorType::UNKNOWN;
                    break;
            }

            auto* kpu_op = graph->add_operator(op_name, op_type);

            // Add inputs (operands)
            for (size_t j = 0; j < df_node.getNrInputs(); ++j) {
                // Get operand type/name from the map
                std::string input_name = df_node.getOperandType(j);
                if (!input_name.empty()) {
                    kpu_op->add_input(input_name);
                    tensor_names.insert(input_name);
                }
            }

            // Add outputs (results)
            for (size_t k = 0; k < df_node.getNrOutputs(); ++k) {
                // Get result value/name from the map
                std::string output_name = df_node.getResultValue(k);
                if (!output_name.empty()) {
                    kpu_op->add_output(output_name);
                    tensor_names.insert(output_name);
                }
            }

            // Copy attributes
            const auto& attributes = df_node.getAttributes();
            for (const auto& [attr_name, attr_value] : attributes) {
                kpu_op->set_attribute(attr_name, attr_value);
            }
        }

        // Create tensor descriptors for discovered tensors
        for (const auto& tensor_name : tensor_names) {
            // TODO: Get actual tensor metadata from domain_flow
            // For now, create placeholder descriptors
            TensorDescriptor desc;
            desc.name = tensor_name;
            desc.dtype = "float32";  // Default
            desc.layout = "row_major";  // Default

            // Try to parse shape from tensor name if encoded
            // Otherwise use placeholder
            desc.shape = {1};  // Placeholder
            desc.size_bytes = 4;  // Placeholder

            graph->add_tensor(desc);
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing domain_flow graph: " << e.what() << std::endl;
        return false;
    }
}

OperatorType DomainFlowGraphLoader::convert_operator_type(const std::string& df_op_name) {
    // Map domain_flow operator names to KPU OperatorType
    if (df_op_name == "matmul" || df_op_name == "gemm") return OperatorType::MATMUL;
    if (df_op_name == "conv2d") return OperatorType::CONV2D;
    if (df_op_name == "relu") return OperatorType::RELU;
    if (df_op_name == "gelu") return OperatorType::GELU;
    // Add more mappings as domain_flow format is defined

    return OperatorType::UNKNOWN;
}

#endif // KPU_HAS_DOMAIN_FLOW

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<GraphLoader> create_graph_loader(const std::string& filepath) {
    // Determine loader based on file extension
    std::string ext;
    size_t dot_pos = filepath.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = filepath.substr(dot_pos);
    }

#ifdef KPU_HAS_DOMAIN_FLOW
    // domain_flow native serialization format (.dfg)
    if (ext == ".dfg") {
        return std::make_unique<DomainFlowGraphLoader>();
    }
#endif

    // JSON fallback format
    if (ext == ".json") {
        return std::make_unique<JSONGraphLoader>();
    }

    throw std::runtime_error("Unsupported file format: " + ext);
}

std::unique_ptr<ComputationalGraph> load_graph(const std::string& filepath) {
    auto loader = create_graph_loader(filepath);
    return loader->load(filepath);
}

} // namespace sw::kpu::compiler
