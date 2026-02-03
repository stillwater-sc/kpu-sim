#pragma once
/**
 * @file assembler.hpp
 * @brief KPU Assembly Language Assembler
 *
 * Parses .kpuasm text files and produces DMProgram objects that can be
 * serialized to .kpubin binary format for execution on simulators.
 */

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/isa/program_serializer.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <variant>
#include <stdexcept>

namespace sw::kpu::isa {

/**
 * @brief Error thrown during assembly
 */
class AssemblyError : public std::runtime_error {
public:
    AssemblyError(const std::string& filename, size_t line, const std::string& msg)
        : std::runtime_error(filename + ":" + std::to_string(line) + ": error: " + msg),
          filename_(filename), line_(line), message_(msg) {}

    const std::string& filename() const { return filename_; }
    size_t line() const { return line_; }
    const std::string& message() const { return message_; }

private:
    std::string filename_;
    size_t line_;
    std::string message_;
};

/**
 * @brief Assembly warning (non-fatal)
 */
struct AssemblyWarning {
    std::string filename;
    size_t line;
    std::string message;

    std::string to_string() const {
        return filename + ":" + std::to_string(line) + ": warning: " + message;
    }
};

/**
 * @brief Token types for lexer
 */
enum class TokenType {
    IDENTIFIER,     // Opcode, directive, label
    NUMBER,         // Decimal or hex number
    STRING,         // Quoted string
    LPAREN,         // (
    RPAREN,         // )
    COMMA,          // ,
    COLON,          // :
    EQUALS,         // =
    NEWLINE,        // End of line
    END_OF_FILE
};

/**
 * @brief A single token
 */
struct Token {
    TokenType type;
    std::string text;
    size_t line;
    size_t column;

    // Numeric value (if type is NUMBER)
    int64_t int_value = 0;
};

/**
 * @brief Lexer for KPUASM
 */
class Lexer {
public:
    explicit Lexer(const std::string& source, const std::string& filename = "<input>");

    Token next_token();
    Token peek_token();
    bool at_end() const;
    size_t current_line() const { return line_; }

private:
    std::string source_;
    std::string filename_;
    size_t pos_ = 0;
    size_t line_ = 1;
    size_t column_ = 1;
    std::optional<Token> peeked_;

    char current() const;
    char advance();
    void skip_whitespace();
    void skip_comment();
    Token read_identifier();
    Token read_number();
    Token read_string();
    Token make_token(TokenType type, const std::string& text = "");
};

/**
 * @brief KPU Assembler
 *
 * Parses KPUASM source and produces DMProgram objects.
 *
 * Usage:
 * @code
 * Assembler assembler;
 * DMProgram program = assembler.assemble_file("matmul.kpuasm");
 *
 * // Or from string
 * DMProgram program = assembler.assemble(source_code, "matmul.kpuasm");
 *
 * // Check for warnings
 * for (const auto& warn : assembler.warnings()) {
 *     std::cerr << warn.to_string() << "\n";
 * }
 *
 * // Save to binary
 * ProgramSerializer serializer;
 * serializer.save(program, "matmul.kpubin");
 * @endcode
 */
class Assembler {
public:
    Assembler() = default;

    /**
     * @brief Assemble source code into a DMProgram
     * @param source KPUASM source code
     * @param filename Filename for error messages
     * @return Assembled program
     * @throws AssemblyError on syntax or semantic error
     */
    DMProgram assemble(const std::string& source, const std::string& filename = "<input>");

    /**
     * @brief Assemble a file into a DMProgram
     * @param path Path to .kpuasm file
     * @return Assembled program
     * @throws AssemblyError on syntax or semantic error
     * @throws std::runtime_error on I/O error
     */
    DMProgram assemble_file(const std::string& path);

    /**
     * @brief Get warnings generated during assembly
     */
    const std::vector<AssemblyWarning>& warnings() const { return warnings_; }

    /**
     * @brief Clear warnings
     */
    void clear_warnings() { warnings_.clear(); }

private:
    std::vector<AssemblyWarning> warnings_;
    std::unordered_map<std::string, size_t> labels_;  // label -> instruction index
    std::string current_file_;

    // Parser state
    Lexer* lexer_ = nullptr;
    Token current_token_;

    // Program being built
    DMProgram program_;
    uint32_t next_instruction_id_ = 0;

    // Parsing helpers
    void advance();
    bool check(TokenType type) const;
    bool match(TokenType type);
    void expect(TokenType type, const std::string& what);
    Token consume(TokenType type, const std::string& what);

    void error(const std::string& msg);
    void warning(const std::string& msg);

    // Directive parsing
    void parse_directive();
    void parse_name_directive();
    void parse_version_directive();
    void parse_dimensions_directive();
    void parse_tiling_directive();
    void parse_l1_ki_directive();
    void parse_dataflow_directive();
    void parse_base_directive(const std::string& which);

    // Instruction parsing
    void parse_instruction();
    DMInstruction parse_dma_load_tile();
    DMInstruction parse_dma_store_tile();
    DMInstruction parse_dma_prefetch_tile();
    DMInstruction parse_dma_load_gather();
    DMInstruction parse_dma_store_scatter();
    DMInstruction parse_bm_move_tile();
    DMInstruction parse_bm_transpose_tile();
    DMInstruction parse_bm_writeback_tile();
    DMInstruction parse_bm_reshape_tile();
    DMInstruction parse_str_feed_rows();
    DMInstruction parse_str_feed_cols();
    DMInstruction parse_str_drain_output();
    DMInstruction parse_str_broadcast_row();
    DMInstruction parse_str_broadcast_col();
    DMInstruction parse_ve_elementwise();
    DMInstruction parse_ve_reduce();
    DMInstruction parse_barrier();
    DMInstruction parse_wait_dma();
    DMInstruction parse_wait_bm();
    DMInstruction parse_wait_str();
    DMInstruction parse_signal();
    DMInstruction parse_set_tile_size();
    DMInstruction parse_set_buffer();
    DMInstruction parse_set_stride();
    DMInstruction parse_loop_begin();
    DMInstruction parse_loop_end();
    DMInstruction parse_l2_scratch_write();
    DMInstruction parse_l2_scratch_read();
    DMInstruction parse_nop();
    DMInstruction parse_halt();

    // Operand parsing
    MatrixID parse_matrix_id();
    TileCoord parse_tile_coord();
    BufferSlot parse_buffer_slot();
    Transform parse_transform();
    ActivationType parse_activation_type();
    int64_t parse_number();
    Address parse_address();
    std::string parse_string();
};

} // namespace sw::kpu::isa
