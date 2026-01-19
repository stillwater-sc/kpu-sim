/**
 * @file dfx_parser.hpp
 * @brief DFX JSON parser for Python kpu package format
 *
 * Parses DFX programs from JSON strings or pybind11 dictionaries
 * into C++ Program structures for execution in the native runtime.
 */

#pragma once

#include "dfx_program.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <stdexcept>

namespace sw::kpu::dfx {

/**
 * @brief Exception for DFX parsing errors
 */
class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief DFX program parser
 *
 * Parses DFX programs from various formats into the C++ Program structure.
 *
 * Example usage:
 * @code
 *   DFXParser parser;
 *   Program prog = parser.parse_json(json_string);
 * @endcode
 */
class DFXParser {
public:
    DFXParser() = default;
    ~DFXParser() = default;

    /**
     * @brief Parse DFX program from JSON string
     *
     * @param json_str JSON string containing DFX program
     * @return Parsed Program structure
     * @throws ParseError if parsing fails
     */
    Program parse_json(const std::string& json_str);

    /**
     * @brief Parse DFX program from nlohmann::json object
     *
     * @param j JSON object
     * @return Parsed Program structure
     * @throws ParseError if parsing fails
     */
    Program parse_json_object(const nlohmann::json& j);
};

/**
 * @brief Convenience function to parse DFX from JSON string
 */
inline Program parse_dfx_json(const std::string& json_str) {
    DFXParser parser;
    return parser.parse_json(json_str);
}

} // namespace sw::kpu::dfx
