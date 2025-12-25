// Program Serializer Implementation
// Binary and JSON serialization for DMProgram

#include <sw/kpu/isa/program_serializer.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace sw::kpu::isa {

using json = nlohmann::json;

// ============================================================================
// Binary Serialization
// ============================================================================

std::vector<uint8_t> ProgramSerializer::serialize(const DMProgram& program) const {
    std::vector<uint8_t> buffer;
    buffer.reserve(serialized_size(program));

    write_header(buffer, program);
    write_instructions(buffer, program);
    write_memory_map(buffer, program);
    write_estimates(buffer, program);

    return buffer;
}

DMProgram ProgramSerializer::deserialize(const std::vector<uint8_t>& data) const {
    if (data.size() < 8) {
        throw SerializationError("Data too small to be a valid program");
    }

    DMProgram program;
    size_t offset = 0;

    offset = read_header(data, offset, program);
    offset = read_instructions(data, offset, program);
    offset = read_memory_map(data, offset, program);
    offset = read_estimates(data, offset, program);

    return program;
}

void ProgramSerializer::save(const DMProgram& program, const std::string& path) const {
    std::vector<uint8_t> data = serialize(program);

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw SerializationError("Failed to open file for writing: " + path);
    }

    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    if (!file) {
        throw SerializationError("Failed to write to file: " + path);
    }
}

DMProgram ProgramSerializer::load(const std::string& path) const {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw SerializationError("Failed to open file for reading: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw SerializationError("Failed to read from file: " + path);
    }

    return deserialize(data);
}

// ============================================================================
// Header Serialization
// ============================================================================

void ProgramSerializer::write_header(std::vector<uint8_t>& buffer, const DMProgram& program) const {
    // Magic and version
    write_value(buffer, DMPROGRAM_MAGIC);
    write_value(buffer, DMPROGRAM_VERSION);

    // Program name
    write_string(buffer, program.name);

    // Matrix dimensions
    write_value(buffer, program.M);
    write_value(buffer, program.N);
    write_value(buffer, program.K);

    // Tile dimensions
    write_value(buffer, program.Ti);
    write_value(buffer, program.Tj);
    write_value(buffer, program.Tk);
    write_value(buffer, program.L1_Ki);

    // Dataflow
    write_value(buffer, static_cast<uint8_t>(program.dataflow));

    // Number of instructions
    write_value(buffer, static_cast<uint32_t>(program.instructions.size()));
}

size_t ProgramSerializer::read_header(const std::vector<uint8_t>& data, size_t offset, DMProgram& program) const {
    // Check magic
    uint32_t magic = read_value<uint32_t>(data, offset);
    if (magic != DMPROGRAM_MAGIC) {
        throw SerializationError("Invalid magic number - not a KPU program file");
    }

    // Check version
    uint32_t version = read_value<uint32_t>(data, offset);
    if (version > DMPROGRAM_VERSION) {
        throw SerializationError("Unsupported program version: " + std::to_string(version));
    }
    program.version = version;

    // Program name
    program.name = read_string(data, offset, 256);

    // Matrix dimensions
    program.M = read_value<Size>(data, offset);
    program.N = read_value<Size>(data, offset);
    program.K = read_value<Size>(data, offset);

    // Tile dimensions
    program.Ti = read_value<Size>(data, offset);
    program.Tj = read_value<Size>(data, offset);
    program.Tk = read_value<Size>(data, offset);
    program.L1_Ki = read_value<Size>(data, offset);

    // Dataflow
    program.dataflow = static_cast<DMProgram::Dataflow>(read_value<uint8_t>(data, offset));

    // Number of instructions (read but don't use - we'll read them in read_instructions)
    /*uint32_t num_instr =*/ read_value<uint32_t>(data, offset);

    return offset;
}

// ============================================================================
// Instruction Serialization
// ============================================================================

void ProgramSerializer::write_instructions(std::vector<uint8_t>& buffer, const DMProgram& program) const {
    for (const auto& instr : program.instructions) {
        write_instruction(buffer, instr);
    }
}

void ProgramSerializer::write_instruction(std::vector<uint8_t>& buffer, const DMInstruction& instr) const {
    // Opcode
    write_value(buffer, static_cast<uint8_t>(instr.opcode));

    // Operand type indicator
    uint8_t operand_type = static_cast<uint8_t>(instr.operands.index());
    write_value(buffer, operand_type);

    // Timing
    write_value(buffer, instr.earliest_cycle);
    write_value(buffer, instr.deadline_cycle);
    write_value(buffer, instr.instruction_id);

    // Dependencies
    write_value(buffer, static_cast<uint32_t>(instr.dependencies.size()));
    for (uint32_t dep : instr.dependencies) {
        write_value(buffer, dep);
    }

    // Label
    write_string(buffer, instr.label);

    // Operands based on type
    std::visit([this, &buffer](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            // No operands
        } else if constexpr (std::is_same_v<T, DMAOperands>) {
            write_value(buffer, static_cast<uint8_t>(arg.matrix));
            write_value(buffer, arg.tile.ti);
            write_value(buffer, arg.tile.tj);
            write_value(buffer, arg.tile.tk);
            write_value(buffer, arg.ext_mem_addr);
            write_value(buffer, arg.l3_tile_id);
            write_value(buffer, arg.l3_offset);
            write_value(buffer, arg.size_bytes);
            write_value(buffer, static_cast<uint8_t>(arg.buffer));
        } else if constexpr (std::is_same_v<T, BlockMoverOperands>) {
            write_value(buffer, static_cast<uint8_t>(arg.matrix));
            write_value(buffer, arg.tile.ti);
            write_value(buffer, arg.tile.tj);
            write_value(buffer, arg.tile.tk);
            write_value(buffer, arg.src_l3_tile_id);
            write_value(buffer, arg.src_offset);
            write_value(buffer, arg.dst_l2_bank_id);
            write_value(buffer, arg.dst_offset);
            write_value(buffer, arg.height);
            write_value(buffer, arg.width);
            write_value(buffer, arg.element_size);
            write_value(buffer, static_cast<uint8_t>(arg.transform));
            write_value(buffer, static_cast<uint8_t>(arg.buffer));
        } else if constexpr (std::is_same_v<T, StreamerOperands>) {
            write_value(buffer, static_cast<uint8_t>(arg.matrix));
            write_value(buffer, arg.tile.ti);
            write_value(buffer, arg.tile.tj);
            write_value(buffer, arg.tile.tk);
            write_value(buffer, arg.l2_bank_id);
            write_value(buffer, arg.l1_buffer_id);
            write_value(buffer, arg.l2_addr);
            write_value(buffer, arg.l1_addr);
            write_value(buffer, arg.height);
            write_value(buffer, arg.width);
            write_value(buffer, arg.fabric_size);
            write_value(buffer, static_cast<uint8_t>(arg.buffer));
            // VE fields
            write_value(buffer, static_cast<uint8_t>(arg.ve_enabled ? 1 : 0));
            write_value(buffer, static_cast<uint8_t>(arg.ve_activation));
            write_value(buffer, static_cast<uint8_t>(arg.ve_bias_enabled ? 1 : 0));
            write_value(buffer, arg.ve_bias_addr);
        } else if constexpr (std::is_same_v<T, SyncOperands>) {
            write_value(buffer, arg.wait_mask);
            write_value(buffer, arg.signal_id);
        } else if constexpr (std::is_same_v<T, LoopOperands>) {
            write_value(buffer, arg.loop_count);
            write_value(buffer, arg.loop_id);
            write_value(buffer, arg.loop_stride);
        } else if constexpr (std::is_same_v<T, ConfigOperands>) {
            write_value(buffer, arg.Ti);
            write_value(buffer, arg.Tj);
            write_value(buffer, arg.Tk);
            write_value(buffer, arg.L1_Ki);
            write_value(buffer, arg.buffer_id);
            write_value(buffer, arg.stride_m);
            write_value(buffer, arg.stride_n);
            write_value(buffer, arg.stride_k);
        }
    }, instr.operands);
}

size_t ProgramSerializer::read_instructions(const std::vector<uint8_t>& data, size_t offset, DMProgram& program) const {
    // We need to know how many instructions - it was stored in header
    // Seek back to read it (or we could pass it as a parameter)
    size_t temp_offset = 8;  // After magic and version
    // Skip name
    uint32_t name_len = read_value<uint32_t>(data, temp_offset);
    temp_offset += name_len;
    // Skip M, N, K, Ti, Tj, Tk, L1_Ki, dataflow
    temp_offset += 7 * sizeof(Size) + 1;
    uint32_t num_instr = read_value<uint32_t>(data, temp_offset);

    program.instructions.clear();
    program.instructions.reserve(num_instr);

    for (uint32_t i = 0; i < num_instr; ++i) {
        DMInstruction instr;
        offset = read_instruction(data, offset, instr);
        program.instructions.push_back(std::move(instr));
    }

    return offset;
}

size_t ProgramSerializer::read_instruction(const std::vector<uint8_t>& data, size_t offset, DMInstruction& instr) const {
    // Opcode
    instr.opcode = static_cast<DMOpcode>(read_value<uint8_t>(data, offset));

    // Operand type
    uint8_t operand_type = read_value<uint8_t>(data, offset);

    // Timing
    instr.earliest_cycle = read_value<uint32_t>(data, offset);
    instr.deadline_cycle = read_value<uint32_t>(data, offset);
    instr.instruction_id = read_value<uint32_t>(data, offset);

    // Dependencies
    uint32_t num_deps = read_value<uint32_t>(data, offset);
    instr.dependencies.resize(num_deps);
    for (uint32_t i = 0; i < num_deps; ++i) {
        instr.dependencies[i] = read_value<uint32_t>(data, offset);
    }

    // Label
    instr.label = read_string(data, offset, 256);

    // Operands based on type
    switch (operand_type) {
        case 0: // monostate
            instr.operands = std::monostate{};
            break;
        case 1: { // DMAOperands
            DMAOperands ops;
            ops.matrix = static_cast<MatrixID>(read_value<uint8_t>(data, offset));
            ops.tile.ti = read_value<uint16_t>(data, offset);
            ops.tile.tj = read_value<uint16_t>(data, offset);
            ops.tile.tk = read_value<uint16_t>(data, offset);
            ops.ext_mem_addr = read_value<Address>(data, offset);
            ops.l3_tile_id = read_value<uint8_t>(data, offset);
            ops.l3_offset = read_value<Address>(data, offset);
            ops.size_bytes = read_value<Size>(data, offset);
            ops.buffer = static_cast<BufferSlot>(read_value<uint8_t>(data, offset));
            instr.operands = ops;
            break;
        }
        case 2: { // BlockMoverOperands
            BlockMoverOperands ops;
            ops.matrix = static_cast<MatrixID>(read_value<uint8_t>(data, offset));
            ops.tile.ti = read_value<uint16_t>(data, offset);
            ops.tile.tj = read_value<uint16_t>(data, offset);
            ops.tile.tk = read_value<uint16_t>(data, offset);
            ops.src_l3_tile_id = read_value<uint8_t>(data, offset);
            ops.src_offset = read_value<Address>(data, offset);
            ops.dst_l2_bank_id = read_value<uint8_t>(data, offset);
            ops.dst_offset = read_value<Address>(data, offset);
            ops.height = read_value<Size>(data, offset);
            ops.width = read_value<Size>(data, offset);
            ops.element_size = read_value<Size>(data, offset);
            ops.transform = static_cast<Transform>(read_value<uint8_t>(data, offset));
            ops.buffer = static_cast<BufferSlot>(read_value<uint8_t>(data, offset));
            instr.operands = ops;
            break;
        }
        case 3: { // StreamerOperands
            StreamerOperands ops;
            ops.matrix = static_cast<MatrixID>(read_value<uint8_t>(data, offset));
            ops.tile.ti = read_value<uint16_t>(data, offset);
            ops.tile.tj = read_value<uint16_t>(data, offset);
            ops.tile.tk = read_value<uint16_t>(data, offset);
            ops.l2_bank_id = read_value<uint8_t>(data, offset);
            ops.l1_buffer_id = read_value<uint8_t>(data, offset);
            ops.l2_addr = read_value<Address>(data, offset);
            ops.l1_addr = read_value<Address>(data, offset);
            ops.height = read_value<Size>(data, offset);
            ops.width = read_value<Size>(data, offset);
            ops.fabric_size = read_value<Size>(data, offset);
            ops.buffer = static_cast<BufferSlot>(read_value<uint8_t>(data, offset));
            // VE fields
            ops.ve_enabled = read_value<uint8_t>(data, offset) != 0;
            ops.ve_activation = static_cast<ActivationType>(read_value<uint8_t>(data, offset));
            ops.ve_bias_enabled = read_value<uint8_t>(data, offset) != 0;
            ops.ve_bias_addr = read_value<Address>(data, offset);
            instr.operands = ops;
            break;
        }
        case 4: { // SyncOperands
            SyncOperands ops;
            ops.wait_mask = read_value<uint32_t>(data, offset);
            ops.signal_id = read_value<uint32_t>(data, offset);
            instr.operands = ops;
            break;
        }
        case 5: { // LoopOperands
            LoopOperands ops;
            ops.loop_count = read_value<uint16_t>(data, offset);
            ops.loop_id = read_value<uint8_t>(data, offset);
            ops.loop_stride = read_value<uint16_t>(data, offset);
            instr.operands = ops;
            break;
        }
        case 6: { // ConfigOperands
            ConfigOperands ops;
            ops.Ti = read_value<Size>(data, offset);
            ops.Tj = read_value<Size>(data, offset);
            ops.Tk = read_value<Size>(data, offset);
            ops.L1_Ki = read_value<Size>(data, offset);
            ops.buffer_id = read_value<uint8_t>(data, offset);
            ops.stride_m = read_value<Size>(data, offset);
            ops.stride_n = read_value<Size>(data, offset);
            ops.stride_k = read_value<Size>(data, offset);
            instr.operands = ops;
            break;
        }
        default:
            throw SerializationError("Unknown operand type: " + std::to_string(operand_type));
    }

    return offset;
}

// ============================================================================
// Memory Map Serialization
// ============================================================================

void ProgramSerializer::write_memory_map(std::vector<uint8_t>& buffer, const DMProgram& program) const {
    // Base addresses
    write_value(buffer, program.memory_map.a_base);
    write_value(buffer, program.memory_map.b_base);
    write_value(buffer, program.memory_map.c_base);

    // L3 allocations
    write_value(buffer, static_cast<uint32_t>(program.memory_map.l3_allocations.size()));
    for (const auto& alloc : program.memory_map.l3_allocations) {
        write_value(buffer, alloc.tile_id);
        write_value(buffer, alloc.offset);
        write_value(buffer, alloc.size);
        write_value(buffer, static_cast<uint8_t>(alloc.matrix));
        write_value(buffer, static_cast<uint8_t>(alloc.buffer));
    }

    // L2 allocations
    write_value(buffer, static_cast<uint32_t>(program.memory_map.l2_allocations.size()));
    for (const auto& alloc : program.memory_map.l2_allocations) {
        write_value(buffer, alloc.bank_id);
        write_value(buffer, alloc.offset);
        write_value(buffer, alloc.size);
        write_value(buffer, static_cast<uint8_t>(alloc.matrix));
        write_value(buffer, static_cast<uint8_t>(alloc.buffer));
    }
}

size_t ProgramSerializer::read_memory_map(const std::vector<uint8_t>& data, size_t offset, DMProgram& program) const {
    // Base addresses
    program.memory_map.a_base = read_value<Address>(data, offset);
    program.memory_map.b_base = read_value<Address>(data, offset);
    program.memory_map.c_base = read_value<Address>(data, offset);

    // L3 allocations
    uint32_t num_l3 = read_value<uint32_t>(data, offset);
    program.memory_map.l3_allocations.resize(num_l3);
    for (uint32_t i = 0; i < num_l3; ++i) {
        auto& alloc = program.memory_map.l3_allocations[i];
        alloc.tile_id = read_value<uint8_t>(data, offset);
        alloc.offset = read_value<Address>(data, offset);
        alloc.size = read_value<Size>(data, offset);
        alloc.matrix = static_cast<MatrixID>(read_value<uint8_t>(data, offset));
        alloc.buffer = static_cast<BufferSlot>(read_value<uint8_t>(data, offset));
    }

    // L2 allocations
    uint32_t num_l2 = read_value<uint32_t>(data, offset);
    program.memory_map.l2_allocations.resize(num_l2);
    for (uint32_t i = 0; i < num_l2; ++i) {
        auto& alloc = program.memory_map.l2_allocations[i];
        alloc.bank_id = read_value<uint8_t>(data, offset);
        alloc.offset = read_value<Address>(data, offset);
        alloc.size = read_value<Size>(data, offset);
        alloc.matrix = static_cast<MatrixID>(read_value<uint8_t>(data, offset));
        alloc.buffer = static_cast<BufferSlot>(read_value<uint8_t>(data, offset));
    }

    return offset;
}

// ============================================================================
// Estimates Serialization
// ============================================================================

void ProgramSerializer::write_estimates(std::vector<uint8_t>& buffer, const DMProgram& program) const {
    write_value(buffer, program.estimates.total_cycles);
    write_value(buffer, program.estimates.external_mem_bytes);
    write_value(buffer, program.estimates.l3_bytes);
    write_value(buffer, program.estimates.l2_bytes);
    write_value(buffer, program.estimates.arithmetic_intensity);
    write_value(buffer, program.estimates.estimated_gflops);
}

size_t ProgramSerializer::read_estimates(const std::vector<uint8_t>& data, size_t offset, DMProgram& program) const {
    program.estimates.total_cycles = read_value<uint64_t>(data, offset);
    program.estimates.external_mem_bytes = read_value<uint64_t>(data, offset);
    program.estimates.l3_bytes = read_value<uint64_t>(data, offset);
    program.estimates.l2_bytes = read_value<uint64_t>(data, offset);
    program.estimates.arithmetic_intensity = read_value<double>(data, offset);
    program.estimates.estimated_gflops = read_value<double>(data, offset);
    return offset;
}

// ============================================================================
// String Helpers
// ============================================================================

void ProgramSerializer::write_string(std::vector<uint8_t>& buffer, const std::string& str) const {
    write_value(buffer, static_cast<uint32_t>(str.size()));
    buffer.insert(buffer.end(), str.begin(), str.end());
}

std::string ProgramSerializer::read_string(const std::vector<uint8_t>& data, size_t& offset, size_t max_len) const {
    uint32_t len = read_value<uint32_t>(data, offset);
    if (len > max_len) {
        throw SerializationError("String too long: " + std::to_string(len));
    }
    if (offset + len > data.size()) {
        throw SerializationError("Unexpected end of data reading string");
    }
    std::string str(data.begin() + offset, data.begin() + offset + len);
    offset += len;
    return str;
}

void ProgramSerializer::write_bytes(std::vector<uint8_t>& buffer, const void* data, size_t size) const {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    buffer.insert(buffer.end(), bytes, bytes + size);
}

void ProgramSerializer::read_bytes(const std::vector<uint8_t>& data, size_t& offset, void* out, size_t size) const {
    if (offset + size > data.size()) {
        throw SerializationError("Unexpected end of data reading bytes");
    }
    std::memcpy(out, data.data() + offset, size);
    offset += size;
}

// ============================================================================
// JSON Serialization
// ============================================================================

// Helper functions for JSON conversion
static json tile_coord_to_json(const TileCoord& coord) {
    return json{{"ti", coord.ti}, {"tj", coord.tj}, {"tk", coord.tk}};
}

static TileCoord tile_coord_from_json(const json& j) {
    return TileCoord{
        j.at("ti").get<uint16_t>(),
        j.at("tj").get<uint16_t>(),
        j.at("tk").get<uint16_t>()
    };
}

static const char* opcode_to_string(DMOpcode op) {
    switch (op) {
        case DMOpcode::DMA_LOAD_TILE: return "DMA_LOAD_TILE";
        case DMOpcode::DMA_STORE_TILE: return "DMA_STORE_TILE";
        case DMOpcode::DMA_PREFETCH_TILE: return "DMA_PREFETCH_TILE";
        case DMOpcode::BM_MOVE_TILE: return "BM_MOVE_TILE";
        case DMOpcode::BM_TRANSPOSE_TILE: return "BM_TRANSPOSE_TILE";
        case DMOpcode::BM_WRITEBACK_TILE: return "BM_WRITEBACK_TILE";
        case DMOpcode::BM_RESHAPE_TILE: return "BM_RESHAPE_TILE";
        case DMOpcode::STR_FEED_ROWS: return "STR_FEED_ROWS";
        case DMOpcode::STR_FEED_COLS: return "STR_FEED_COLS";
        case DMOpcode::STR_DRAIN_OUTPUT: return "STR_DRAIN_OUTPUT";
        case DMOpcode::STR_BROADCAST_ROW: return "STR_BROADCAST_ROW";
        case DMOpcode::STR_BROADCAST_COL: return "STR_BROADCAST_COL";
        case DMOpcode::BARRIER: return "BARRIER";
        case DMOpcode::WAIT_DMA: return "WAIT_DMA";
        case DMOpcode::WAIT_BM: return "WAIT_BM";
        case DMOpcode::WAIT_STR: return "WAIT_STR";
        case DMOpcode::SIGNAL: return "SIGNAL";
        case DMOpcode::SET_TILE_SIZE: return "SET_TILE_SIZE";
        case DMOpcode::SET_BUFFER: return "SET_BUFFER";
        case DMOpcode::SET_STRIDE: return "SET_STRIDE";
        case DMOpcode::LOOP_BEGIN: return "LOOP_BEGIN";
        case DMOpcode::LOOP_END: return "LOOP_END";
        case DMOpcode::NOP: return "NOP";
        case DMOpcode::HALT: return "HALT";
        default: return "UNKNOWN";
    }
}

static DMOpcode opcode_from_string(const std::string& str) {
    static const std::unordered_map<std::string, DMOpcode> map = {
        {"DMA_LOAD_TILE", DMOpcode::DMA_LOAD_TILE},
        {"DMA_STORE_TILE", DMOpcode::DMA_STORE_TILE},
        {"DMA_PREFETCH_TILE", DMOpcode::DMA_PREFETCH_TILE},
        {"BM_MOVE_TILE", DMOpcode::BM_MOVE_TILE},
        {"BM_TRANSPOSE_TILE", DMOpcode::BM_TRANSPOSE_TILE},
        {"BM_WRITEBACK_TILE", DMOpcode::BM_WRITEBACK_TILE},
        {"BM_RESHAPE_TILE", DMOpcode::BM_RESHAPE_TILE},
        {"STR_FEED_ROWS", DMOpcode::STR_FEED_ROWS},
        {"STR_FEED_COLS", DMOpcode::STR_FEED_COLS},
        {"STR_DRAIN_OUTPUT", DMOpcode::STR_DRAIN_OUTPUT},
        {"STR_BROADCAST_ROW", DMOpcode::STR_BROADCAST_ROW},
        {"STR_BROADCAST_COL", DMOpcode::STR_BROADCAST_COL},
        {"BARRIER", DMOpcode::BARRIER},
        {"WAIT_DMA", DMOpcode::WAIT_DMA},
        {"WAIT_BM", DMOpcode::WAIT_BM},
        {"WAIT_STR", DMOpcode::WAIT_STR},
        {"SIGNAL", DMOpcode::SIGNAL},
        {"SET_TILE_SIZE", DMOpcode::SET_TILE_SIZE},
        {"SET_BUFFER", DMOpcode::SET_BUFFER},
        {"SET_STRIDE", DMOpcode::SET_STRIDE},
        {"LOOP_BEGIN", DMOpcode::LOOP_BEGIN},
        {"LOOP_END", DMOpcode::LOOP_END},
        {"NOP", DMOpcode::NOP},
        {"HALT", DMOpcode::HALT}
    };
    auto it = map.find(str);
    if (it == map.end()) {
        throw SerializationError("Unknown opcode: " + str);
    }
    return it->second;
}

static json instruction_to_json(const DMInstruction& instr) {
    json j;
    j["opcode"] = opcode_to_string(instr.opcode);
    j["earliest_cycle"] = instr.earliest_cycle;
    j["deadline_cycle"] = instr.deadline_cycle;
    j["instruction_id"] = instr.instruction_id;
    j["dependencies"] = instr.dependencies;
    j["label"] = instr.label;

    std::visit([&j](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            j["operands"] = nullptr;
        } else if constexpr (std::is_same_v<T, DMAOperands>) {
            j["operands"] = json{
                {"type", "DMA"},
                {"matrix", static_cast<int>(arg.matrix)},
                {"tile", tile_coord_to_json(arg.tile)},
                {"ext_mem_addr", arg.ext_mem_addr},
                {"l3_tile_id", arg.l3_tile_id},
                {"l3_offset", arg.l3_offset},
                {"size_bytes", arg.size_bytes},
                {"buffer", static_cast<int>(arg.buffer)}
            };
        } else if constexpr (std::is_same_v<T, BlockMoverOperands>) {
            j["operands"] = json{
                {"type", "BlockMover"},
                {"matrix", static_cast<int>(arg.matrix)},
                {"tile", tile_coord_to_json(arg.tile)},
                {"src_l3_tile_id", arg.src_l3_tile_id},
                {"src_offset", arg.src_offset},
                {"dst_l2_bank_id", arg.dst_l2_bank_id},
                {"dst_offset", arg.dst_offset},
                {"height", arg.height},
                {"width", arg.width},
                {"element_size", arg.element_size},
                {"transform", static_cast<int>(arg.transform)},
                {"buffer", static_cast<int>(arg.buffer)}
            };
        } else if constexpr (std::is_same_v<T, StreamerOperands>) {
            j["operands"] = json{
                {"type", "Streamer"},
                {"matrix", static_cast<int>(arg.matrix)},
                {"tile", tile_coord_to_json(arg.tile)},
                {"l2_bank_id", arg.l2_bank_id},
                {"l1_buffer_id", arg.l1_buffer_id},
                {"l2_addr", arg.l2_addr},
                {"l1_addr", arg.l1_addr},
                {"height", arg.height},
                {"width", arg.width},
                {"fabric_size", arg.fabric_size},
                {"buffer", static_cast<int>(arg.buffer)},
                {"ve_enabled", arg.ve_enabled},
                {"ve_activation", static_cast<int>(arg.ve_activation)},
                {"ve_bias_enabled", arg.ve_bias_enabled},
                {"ve_bias_addr", arg.ve_bias_addr}
            };
        } else if constexpr (std::is_same_v<T, SyncOperands>) {
            j["operands"] = json{
                {"type", "Sync"},
                {"wait_mask", arg.wait_mask},
                {"signal_id", arg.signal_id}
            };
        } else if constexpr (std::is_same_v<T, LoopOperands>) {
            j["operands"] = json{
                {"type", "Loop"},
                {"loop_count", arg.loop_count},
                {"loop_id", arg.loop_id},
                {"loop_stride", arg.loop_stride}
            };
        } else if constexpr (std::is_same_v<T, ConfigOperands>) {
            j["operands"] = json{
                {"type", "Config"},
                {"Ti", arg.Ti},
                {"Tj", arg.Tj},
                {"Tk", arg.Tk},
                {"L1_Ki", arg.L1_Ki},
                {"buffer_id", arg.buffer_id},
                {"stride_m", arg.stride_m},
                {"stride_n", arg.stride_n},
                {"stride_k", arg.stride_k}
            };
        }
    }, instr.operands);

    return j;
}

static DMInstruction instruction_from_json(const json& j) {
    DMInstruction instr;
    instr.opcode = opcode_from_string(j.at("opcode").get<std::string>());
    instr.earliest_cycle = j.at("earliest_cycle").get<uint32_t>();
    instr.deadline_cycle = j.at("deadline_cycle").get<uint32_t>();
    instr.instruction_id = j.at("instruction_id").get<uint32_t>();
    instr.dependencies = j.at("dependencies").get<std::vector<uint32_t>>();
    instr.label = j.at("label").get<std::string>();

    const auto& ops = j.at("operands");
    if (ops.is_null()) {
        instr.operands = std::monostate{};
    } else {
        std::string type = ops.at("type").get<std::string>();
        if (type == "DMA") {
            DMAOperands o;
            o.matrix = static_cast<MatrixID>(ops.at("matrix").get<int>());
            o.tile = tile_coord_from_json(ops.at("tile"));
            o.ext_mem_addr = ops.at("ext_mem_addr").get<Address>();
            o.l3_tile_id = ops.at("l3_tile_id").get<uint8_t>();
            o.l3_offset = ops.at("l3_offset").get<Address>();
            o.size_bytes = ops.at("size_bytes").get<Size>();
            o.buffer = static_cast<BufferSlot>(ops.at("buffer").get<int>());
            instr.operands = o;
        } else if (type == "BlockMover") {
            BlockMoverOperands o;
            o.matrix = static_cast<MatrixID>(ops.at("matrix").get<int>());
            o.tile = tile_coord_from_json(ops.at("tile"));
            o.src_l3_tile_id = ops.at("src_l3_tile_id").get<uint8_t>();
            o.src_offset = ops.at("src_offset").get<Address>();
            o.dst_l2_bank_id = ops.at("dst_l2_bank_id").get<uint8_t>();
            o.dst_offset = ops.at("dst_offset").get<Address>();
            o.height = ops.at("height").get<Size>();
            o.width = ops.at("width").get<Size>();
            o.element_size = ops.at("element_size").get<Size>();
            o.transform = static_cast<Transform>(ops.at("transform").get<int>());
            o.buffer = static_cast<BufferSlot>(ops.at("buffer").get<int>());
            instr.operands = o;
        } else if (type == "Streamer") {
            StreamerOperands o;
            o.matrix = static_cast<MatrixID>(ops.at("matrix").get<int>());
            o.tile = tile_coord_from_json(ops.at("tile"));
            o.l2_bank_id = ops.at("l2_bank_id").get<uint8_t>();
            o.l1_buffer_id = ops.at("l1_buffer_id").get<uint8_t>();
            o.l2_addr = ops.at("l2_addr").get<Address>();
            o.l1_addr = ops.at("l1_addr").get<Address>();
            o.height = ops.at("height").get<Size>();
            o.width = ops.at("width").get<Size>();
            o.fabric_size = ops.at("fabric_size").get<Size>();
            o.buffer = static_cast<BufferSlot>(ops.at("buffer").get<int>());
            o.ve_enabled = ops.at("ve_enabled").get<bool>();
            o.ve_activation = static_cast<ActivationType>(ops.at("ve_activation").get<int>());
            o.ve_bias_enabled = ops.at("ve_bias_enabled").get<bool>();
            o.ve_bias_addr = ops.at("ve_bias_addr").get<Address>();
            instr.operands = o;
        } else if (type == "Sync") {
            SyncOperands o;
            o.wait_mask = ops.at("wait_mask").get<uint32_t>();
            o.signal_id = ops.at("signal_id").get<uint32_t>();
            instr.operands = o;
        } else if (type == "Loop") {
            LoopOperands o;
            o.loop_count = ops.at("loop_count").get<uint16_t>();
            o.loop_id = ops.at("loop_id").get<uint8_t>();
            o.loop_stride = ops.at("loop_stride").get<uint16_t>();
            instr.operands = o;
        } else if (type == "Config") {
            ConfigOperands o;
            o.Ti = ops.at("Ti").get<Size>();
            o.Tj = ops.at("Tj").get<Size>();
            o.Tk = ops.at("Tk").get<Size>();
            o.L1_Ki = ops.at("L1_Ki").get<Size>();
            o.buffer_id = ops.at("buffer_id").get<uint8_t>();
            o.stride_m = ops.at("stride_m").get<Size>();
            o.stride_n = ops.at("stride_n").get<Size>();
            o.stride_k = ops.at("stride_k").get<Size>();
            instr.operands = o;
        } else {
            throw SerializationError("Unknown operand type: " + type);
        }
    }

    return instr;
}

std::string ProgramSerializer::to_json(const DMProgram& program, bool pretty) const {
    json j;

    // Header
    j["name"] = program.name;
    j["version"] = program.version;
    j["dimensions"] = json{{"M", program.M}, {"N", program.N}, {"K", program.K}};
    j["tiles"] = json{{"Ti", program.Ti}, {"Tj", program.Tj}, {"Tk", program.Tk}, {"L1_Ki", program.L1_Ki}};

    const char* dataflow_str;
    switch (program.dataflow) {
        case DMProgram::Dataflow::OUTPUT_STATIONARY: dataflow_str = "OUTPUT_STATIONARY"; break;
        case DMProgram::Dataflow::WEIGHT_STATIONARY: dataflow_str = "WEIGHT_STATIONARY"; break;
        case DMProgram::Dataflow::INPUT_STATIONARY: dataflow_str = "INPUT_STATIONARY"; break;
        default: dataflow_str = "UNKNOWN";
    }
    j["dataflow"] = dataflow_str;

    // Instructions
    j["instructions"] = json::array();
    for (const auto& instr : program.instructions) {
        j["instructions"].push_back(instruction_to_json(instr));
    }

    // Memory map
    j["memory_map"] = json{
        {"a_base", program.memory_map.a_base},
        {"b_base", program.memory_map.b_base},
        {"c_base", program.memory_map.c_base}
    };

    j["memory_map"]["l3_allocations"] = json::array();
    for (const auto& alloc : program.memory_map.l3_allocations) {
        j["memory_map"]["l3_allocations"].push_back(json{
            {"tile_id", alloc.tile_id},
            {"offset", alloc.offset},
            {"size", alloc.size},
            {"matrix", static_cast<int>(alloc.matrix)},
            {"buffer", static_cast<int>(alloc.buffer)}
        });
    }

    j["memory_map"]["l2_allocations"] = json::array();
    for (const auto& alloc : program.memory_map.l2_allocations) {
        j["memory_map"]["l2_allocations"].push_back(json{
            {"bank_id", alloc.bank_id},
            {"offset", alloc.offset},
            {"size", alloc.size},
            {"matrix", static_cast<int>(alloc.matrix)},
            {"buffer", static_cast<int>(alloc.buffer)}
        });
    }

    // Estimates
    j["estimates"] = json{
        {"total_cycles", program.estimates.total_cycles},
        {"external_mem_bytes", program.estimates.external_mem_bytes},
        {"l3_bytes", program.estimates.l3_bytes},
        {"l2_bytes", program.estimates.l2_bytes},
        {"arithmetic_intensity", program.estimates.arithmetic_intensity},
        {"estimated_gflops", program.estimates.estimated_gflops}
    };

    return pretty ? j.dump(2) : j.dump();
}

DMProgram ProgramSerializer::from_json(const std::string& json_str) const {
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::parse_error& e) {
        throw SerializationError("JSON parse error: " + std::string(e.what()));
    }

    DMProgram program;

    // Header
    program.name = j.at("name").get<std::string>();
    program.version = j.at("version").get<uint32_t>();
    program.M = j.at("dimensions").at("M").get<Size>();
    program.N = j.at("dimensions").at("N").get<Size>();
    program.K = j.at("dimensions").at("K").get<Size>();
    program.Ti = j.at("tiles").at("Ti").get<Size>();
    program.Tj = j.at("tiles").at("Tj").get<Size>();
    program.Tk = j.at("tiles").at("Tk").get<Size>();
    program.L1_Ki = j.at("tiles").at("L1_Ki").get<Size>();

    std::string dataflow_str = j.at("dataflow").get<std::string>();
    if (dataflow_str == "OUTPUT_STATIONARY") {
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;
    } else if (dataflow_str == "WEIGHT_STATIONARY") {
        program.dataflow = DMProgram::Dataflow::WEIGHT_STATIONARY;
    } else if (dataflow_str == "INPUT_STATIONARY") {
        program.dataflow = DMProgram::Dataflow::INPUT_STATIONARY;
    } else {
        throw SerializationError("Unknown dataflow: " + dataflow_str);
    }

    // Instructions
    for (const auto& instr_j : j.at("instructions")) {
        program.instructions.push_back(instruction_from_json(instr_j));
    }

    // Memory map
    const auto& mm = j.at("memory_map");
    program.memory_map.a_base = mm.at("a_base").get<Address>();
    program.memory_map.b_base = mm.at("b_base").get<Address>();
    program.memory_map.c_base = mm.at("c_base").get<Address>();

    for (const auto& alloc_j : mm.at("l3_allocations")) {
        DMProgram::MemoryMap::L3Alloc alloc;
        alloc.tile_id = alloc_j.at("tile_id").get<uint8_t>();
        alloc.offset = alloc_j.at("offset").get<Address>();
        alloc.size = alloc_j.at("size").get<Size>();
        alloc.matrix = static_cast<MatrixID>(alloc_j.at("matrix").get<int>());
        alloc.buffer = static_cast<BufferSlot>(alloc_j.at("buffer").get<int>());
        program.memory_map.l3_allocations.push_back(alloc);
    }

    for (const auto& alloc_j : mm.at("l2_allocations")) {
        DMProgram::MemoryMap::L2Alloc alloc;
        alloc.bank_id = alloc_j.at("bank_id").get<uint8_t>();
        alloc.offset = alloc_j.at("offset").get<Address>();
        alloc.size = alloc_j.at("size").get<Size>();
        alloc.matrix = static_cast<MatrixID>(alloc_j.at("matrix").get<int>());
        alloc.buffer = static_cast<BufferSlot>(alloc_j.at("buffer").get<int>());
        program.memory_map.l2_allocations.push_back(alloc);
    }

    // Estimates
    const auto& est = j.at("estimates");
    program.estimates.total_cycles = est.at("total_cycles").get<uint64_t>();
    program.estimates.external_mem_bytes = est.at("external_mem_bytes").get<uint64_t>();
    program.estimates.l3_bytes = est.at("l3_bytes").get<uint64_t>();
    program.estimates.l2_bytes = est.at("l2_bytes").get<uint64_t>();
    program.estimates.arithmetic_intensity = est.at("arithmetic_intensity").get<double>();
    program.estimates.estimated_gflops = est.at("estimated_gflops").get<double>();

    return program;
}

void ProgramSerializer::save_json(const DMProgram& program, const std::string& path, bool pretty) const {
    std::string json_str = to_json(program, pretty);

    std::ofstream file(path);
    if (!file) {
        throw SerializationError("Failed to open file for writing: " + path);
    }
    file << json_str;
}

DMProgram ProgramSerializer::load_json(const std::string& path) const {
    std::ifstream file(path);
    if (!file) {
        throw SerializationError("Failed to open file for reading: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return from_json(buffer.str());
}

// ============================================================================
// Utilities
// ============================================================================

size_t ProgramSerializer::serialized_size(const DMProgram& program) const {
    size_t size = 0;

    // Header (approximate)
    size += 8;  // magic + version
    size += 4 + program.name.size();  // name
    size += 7 * sizeof(Size) + 1;  // dimensions, tiles, dataflow
    size += 4;  // num_instructions

    // Instructions (approximate)
    for (const auto& instr : program.instructions) {
        size += 2;  // opcode + operand_type
        size += 12;  // timing
        size += 4 + instr.dependencies.size() * 4;  // deps
        size += 2 + instr.label.size();  // label
        size += 100;  // operands (approximate)
    }

    // Memory map (approximate)
    size += 3 * sizeof(Address);  // bases
    size += 4 + program.memory_map.l3_allocations.size() * 32;
    size += 4 + program.memory_map.l2_allocations.size() * 32;

    // Estimates
    size += 6 * 8;  // 4 uint64 + 2 double

    return size;
}

bool ProgramSerializer::validate(const std::vector<uint8_t>& data) const {
    if (data.size() < 8) return false;

    uint32_t magic;
    std::memcpy(&magic, data.data(), sizeof(magic));
    if (magic != DMPROGRAM_MAGIC) return false;

    uint32_t version;
    std::memcpy(&version, data.data() + 4, sizeof(version));
    if (version > DMPROGRAM_VERSION) return false;

    return true;
}

std::string ProgramSerializer::detect_format(const std::string& path) {
    size_t dot = path.rfind('.');
    if (dot == std::string::npos) return "binary";

    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".kpubin" || ext == ".bin") return "binary";
    if (ext == ".kpujson" || ext == ".json") return "json";

    return "binary";
}

} // namespace sw::kpu::isa
