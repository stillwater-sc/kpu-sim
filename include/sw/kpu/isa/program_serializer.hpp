#pragma once
// Program Serializer for Data Movement ISA
// Enables saving/loading compiled DMProgram binary and JSON formats

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/concepts.hpp>

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <fstream>

namespace sw::kpu::isa {

/**
 * @brief Binary format magic number and version
 */
constexpr uint32_t DMPROGRAM_MAGIC = 0x4B505544;  // "KPUD" in little-endian
constexpr uint32_t DMPROGRAM_VERSION = 1;

/**
 * @brief Error thrown during serialization/deserialization
 */
class SerializationError : public std::runtime_error {
public:
    explicit SerializationError(const std::string& msg)
        : std::runtime_error("Serialization error: " + msg) {}
};

/**
 * @brief Program Serializer for DMProgram
 *
 * Provides serialization to/from:
 * - Binary format (.kpubin) - compact, fast loading
 * - JSON format (.kpujson) - human-readable, for debugging
 *
 * Binary Format Layout:
 * ```
 * [Header]
 *   magic:        4 bytes (0x4B505544 "KPUD")
 *   version:      4 bytes
 *   name_len:     4 bytes
 *   name:         name_len bytes
 *   M, N, K:      3 * 8 bytes
 *   Ti, Tj, Tk:   3 * 8 bytes
 *   L1_Ki:        8 bytes
 *   dataflow:     1 byte
 *   num_instr:    4 bytes
 *
 * [Instructions]
 *   For each instruction:
 *     opcode:     1 byte
 *     operand_type: 1 byte
 *     earliest_cycle: 4 bytes
 *     deadline_cycle: 4 bytes
 *     instruction_id: 4 bytes
 *     num_deps:   4 bytes
 *     deps:       num_deps * 4 bytes
 *     label_len:  2 bytes
 *     label:      label_len bytes
 *     operands:   variable (depends on opcode)
 *
 * [Memory Map]
 *   a_base, b_base, c_base: 3 * 8 bytes
 *   num_l3_allocs: 4 bytes
 *   l3_allocs: ...
 *   num_l2_allocs: 4 bytes
 *   l2_allocs: ...
 *
 * [Estimates]
 *   total_cycles: 8 bytes
 *   external_mem_bytes: 8 bytes
 *   l3_bytes: 8 bytes
 *   l2_bytes: 8 bytes
 *   arithmetic_intensity: 8 bytes
 *   estimated_gflops: 8 bytes
 * ```
 *
 * Usage:
 * @code
 * // Save program to binary file
 * ProgramSerializer serializer;
 * serializer.save(program, "matmul.kpubin");
 *
 * // Load program from binary file
 * DMProgram loaded = serializer.load("matmul.kpubin");
 *
 * // Convert to/from JSON
 * std::string json = serializer.to_json(program);
 * DMProgram from_json = serializer.from_json(json);
 * @endcode
 */
class ProgramSerializer {
public:
    ProgramSerializer() = default;

    // =========================================
    // Binary Serialization
    // =========================================

    /**
     * @brief Serialize a program to binary format
     * @param program The program to serialize
     * @return Binary data as vector of bytes
     */
    std::vector<uint8_t> serialize(const DMProgram& program) const;

    /**
     * @brief Deserialize a program from binary format
     * @param data Binary data
     * @return The deserialized program
     * @throws SerializationError if data is invalid
     */
    DMProgram deserialize(const std::vector<uint8_t>& data) const;

    /**
     * @brief Save a program to a binary file
     * @param program The program to save
     * @param path File path (typically .kpubin extension)
     * @throws SerializationError on I/O error
     */
    void save(const DMProgram& program, const std::string& path) const;

    /**
     * @brief Load a program from a binary file
     * @param path File path
     * @return The loaded program
     * @throws SerializationError on I/O or format error
     */
    DMProgram load(const std::string& path) const;

    // =========================================
    // JSON Serialization
    // =========================================

    /**
     * @brief Convert a program to JSON string
     * @param program The program to convert
     * @param pretty If true, format with indentation (default true)
     * @return JSON string representation
     */
    std::string to_json(const DMProgram& program, bool pretty = true) const;

    /**
     * @brief Parse a program from JSON string
     * @param json JSON string
     * @return The parsed program
     * @throws SerializationError if JSON is invalid
     */
    DMProgram from_json(const std::string& json) const;

    /**
     * @brief Save a program to a JSON file
     * @param program The program to save
     * @param path File path (typically .kpujson extension)
     * @param pretty If true, format with indentation
     */
    void save_json(const DMProgram& program, const std::string& path,
                   bool pretty = true) const;

    /**
     * @brief Load a program from a JSON file
     * @param path File path
     * @return The loaded program
     */
    DMProgram load_json(const std::string& path) const;

    // =========================================
    // Utilities
    // =========================================

    /**
     * @brief Get the size of a serialized program
     * @param program The program
     * @return Size in bytes
     */
    size_t serialized_size(const DMProgram& program) const;

    /**
     * @brief Validate binary data without fully deserializing
     * @param data Binary data
     * @return true if data appears to be a valid program
     */
    bool validate(const std::vector<uint8_t>& data) const;

    /**
     * @brief Get file format from path extension
     * @param path File path
     * @return "binary" for .kpubin, "json" for .kpujson/.json
     */
    static std::string detect_format(const std::string& path);

private:
    // Binary serialization helpers
    void write_header(std::vector<uint8_t>& buffer, const DMProgram& program) const;
    void write_instructions(std::vector<uint8_t>& buffer, const DMProgram& program) const;
    void write_instruction(std::vector<uint8_t>& buffer, const DMInstruction& instr) const;
    void write_memory_map(std::vector<uint8_t>& buffer, const DMProgram& program) const;
    void write_estimates(std::vector<uint8_t>& buffer, const DMProgram& program) const;

    size_t read_header(const std::vector<uint8_t>& data, size_t offset, DMProgram& program) const;
    size_t read_instructions(const std::vector<uint8_t>& data, size_t offset, DMProgram& program) const;
    size_t read_instruction(const std::vector<uint8_t>& data, size_t offset, DMInstruction& instr) const;
    size_t read_memory_map(const std::vector<uint8_t>& data, size_t offset, DMProgram& program) const;
    size_t read_estimates(const std::vector<uint8_t>& data, size_t offset, DMProgram& program) const;

    // Primitive write helpers
    template<typename T>
    void write_value(std::vector<uint8_t>& buffer, T value) const;
    void write_string(std::vector<uint8_t>& buffer, const std::string& str) const;
    void write_bytes(std::vector<uint8_t>& buffer, const void* data, size_t size) const;

    // Primitive read helpers
    template<typename T>
    T read_value(const std::vector<uint8_t>& data, size_t& offset) const;
    std::string read_string(const std::vector<uint8_t>& data, size_t& offset, size_t max_len = 65535) const;
    void read_bytes(const std::vector<uint8_t>& data, size_t& offset, void* out, size_t size) const;
};

// ============================================================================
// Inline Template Implementations
// ============================================================================

template<typename T>
void ProgramSerializer::write_value(std::vector<uint8_t>& buffer, T value) const {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

template<typename T>
T ProgramSerializer::read_value(const std::vector<uint8_t>& data, size_t& offset) const {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
    if (offset + sizeof(T) > data.size()) {
        throw SerializationError("Unexpected end of data reading value");
    }
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

} // namespace sw::kpu::isa
