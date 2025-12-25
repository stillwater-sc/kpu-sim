#pragma once
// Kernel Serializer
// Enables saving/loading complete Kernel objects with metadata

#include <sw/kpu/kernel.hpp>
#include <sw/kpu/isa/program_serializer.hpp>

#include <vector>
#include <string>
#include <cstdint>

namespace sw::kpu {

/**
 * @brief Binary format magic number and version for kernels
 */
constexpr uint32_t KERNEL_MAGIC = 0x4B50554B;  // "KPUK" in little-endian
constexpr uint32_t KERNEL_VERSION = 1;

/**
 * @brief Kernel Serializer
 *
 * Serializes complete Kernel objects including:
 * - Kernel metadata (name, op type, dimensions)
 * - Arguments (names, shapes, data types)
 * - Compilation options (tile sizes, dataflow)
 * - The underlying DMProgram
 *
 * Binary Format Layout:
 * ```
 * [Kernel Header]
 *   magic:        4 bytes (0x4B50554B "KPUK")
 *   version:      4 bytes
 *   name_len:     4 bytes
 *   name:         name_len bytes
 *   op_type:      1 byte
 *   dtype:        1 byte
 *   M, N, K:      3 * 8 bytes
 *   Ti, Tj, Tk:   3 * 8 bytes
 *   L1_Ki:        8 bytes
 *   has_bias:     1 byte
 *   activation:   1 byte
 *   num_args:     4 bytes
 *
 * [Arguments]
 *   For each argument:
 *     name_len:   2 bytes
 *     name:       name_len bytes
 *     dtype:      1 byte
 *     is_output:  1 byte
 *     num_dims:   1 byte
 *     shape:      num_dims * 8 bytes
 *     size_bytes: 8 bytes
 *
 * [DMProgram]
 *   program_size: 4 bytes
 *   program_data: program_size bytes (from ProgramSerializer)
 * ```
 *
 * Usage:
 * @code
 * // Save kernel to file
 * KernelSerializer serializer;
 * Kernel kernel = Kernel::create_matmul(1024, 1024, 1024);
 * serializer.save(kernel, "matmul.kpukernel");
 *
 * // Load kernel from file
 * Kernel loaded = serializer.load("matmul.kpukernel");
 *
 * // JSON format
 * std::string json = serializer.to_json(kernel);
 * Kernel from_json = serializer.from_json(json);
 * @endcode
 */
class KernelSerializer {
public:
    KernelSerializer() = default;

    // =========================================
    // Binary Serialization
    // =========================================

    /**
     * @brief Serialize a kernel to binary format
     * @param kernel The kernel to serialize
     * @return Binary data as vector of bytes
     */
    std::vector<uint8_t> serialize(const Kernel& kernel) const;

    /**
     * @brief Deserialize a kernel from binary format
     * @param data Binary data
     * @return The deserialized kernel
     * @throws isa::SerializationError if data is invalid
     */
    Kernel deserialize(const std::vector<uint8_t>& data) const;

    /**
     * @brief Save a kernel to a binary file
     * @param kernel The kernel to save
     * @param path File path (typically .kpukernel extension)
     * @throws isa::SerializationError on I/O error
     */
    void save(const Kernel& kernel, const std::string& path) const;

    /**
     * @brief Load a kernel from a binary file
     * @param path File path
     * @return The loaded kernel
     * @throws isa::SerializationError on I/O or format error
     */
    Kernel load(const std::string& path) const;

    // =========================================
    // JSON Serialization
    // =========================================

    /**
     * @brief Convert a kernel to JSON string
     * @param kernel The kernel to convert
     * @param pretty If true, format with indentation (default true)
     * @return JSON string representation
     */
    std::string to_json(const Kernel& kernel, bool pretty = true) const;

    /**
     * @brief Parse a kernel from JSON string
     * @param json JSON string
     * @return The parsed kernel
     * @throws isa::SerializationError if JSON is invalid
     */
    Kernel from_json(const std::string& json) const;

    /**
     * @brief Save a kernel to a JSON file
     * @param kernel The kernel to save
     * @param path File path (typically .json extension)
     * @param pretty If true, format with indentation
     */
    void save_json(const Kernel& kernel, const std::string& path,
                   bool pretty = true) const;

    /**
     * @brief Load a kernel from a JSON file
     * @param path File path
     * @return The loaded kernel
     */
    Kernel load_json(const std::string& path) const;

    // =========================================
    // Utilities
    // =========================================

    /**
     * @brief Validate binary data without fully deserializing
     * @param data Binary data
     * @return true if data appears to be a valid kernel
     */
    bool validate(const std::vector<uint8_t>& data) const;

    /**
     * @brief Get file format from path extension
     * @param path File path
     * @return "binary" for .kpukernel, "json" for .json
     */
    static std::string detect_format(const std::string& path);

    /**
     * @brief Auto-detect format and load
     * @param path File path
     * @return The loaded kernel
     */
    Kernel load_auto(const std::string& path) const;

    /**
     * @brief Auto-detect format and save
     * @param kernel The kernel to save
     * @param path File path
     */
    void save_auto(const Kernel& kernel, const std::string& path) const;

private:
    isa::ProgramSerializer program_serializer_;

    // Binary helpers
    template<typename T>
    void write_value(std::vector<uint8_t>& buffer, T value) const;
    void write_string(std::vector<uint8_t>& buffer, const std::string& str) const;

    template<typename T>
    T read_value(const std::vector<uint8_t>& data, size_t& offset) const;
    std::string read_string(const std::vector<uint8_t>& data, size_t& offset) const;
};

// ============================================================================
// Inline Template Implementations
// ============================================================================

template<typename T>
void KernelSerializer::write_value(std::vector<uint8_t>& buffer, T value) const {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

template<typename T>
T KernelSerializer::read_value(const std::vector<uint8_t>& data, size_t& offset) const {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
    if (offset + sizeof(T) > data.size()) {
        throw isa::SerializationError("Unexpected end of data reading value");
    }
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

} // namespace sw::kpu
