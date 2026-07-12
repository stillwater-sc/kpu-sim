/**
 * @file assembler.cpp
 * @brief KPU Assembly Language Assembler Implementation
 */

#include <sw/kpu/isa/assembler.hpp>

#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>

namespace sw::kpu::isa {

// ============================================================================
// Lexer Implementation
// ============================================================================

Lexer::Lexer(const std::string& source, const std::string& filename)
    : source_(source), filename_(filename) {}

char Lexer::current() const {
    if (pos_ >= source_.size()) return '\0';
    return source_[pos_];
}

char Lexer::advance() {
    char c = current();
    pos_++;
    if (c == '\n') {
        line_++;
        column_ = 1;
    } else {
        column_++;
    }
    return c;
}

void Lexer::skip_whitespace() {
    while (pos_ < source_.size()) {
        char c = current();
        if (c == ' ' || c == '\t' || c == '\r') {
            advance();
        } else if (c == ';' || c == '#') {
            skip_comment();
        } else {
            break;
        }
    }
}

void Lexer::skip_comment() {
    // Skip to end of line
    while (pos_ < source_.size() && current() != '\n') {
        advance();
    }
}

Token Lexer::make_token(TokenType type, const std::string& text) {
    return Token{type, text, line_, column_};
}

Token Lexer::read_identifier() {
    size_t start_line = line_;
    size_t start_col = column_;
    std::string text;

    while (pos_ < source_.size()) {
        char c = current();
        if (std::isalnum(c) || c == '_' || c == '.') {
            text += c;
            advance();
        } else {
            break;
        }
    }

    return Token{TokenType::IDENTIFIER, text, start_line, start_col};
}

Token Lexer::read_number() {
    size_t start_line = line_;
    size_t start_col = column_;
    std::string text;
    int64_t value = 0;
    bool is_hex = false;
    bool is_negative = false;

    // Check for negative sign
    if (current() == '-') {
        is_negative = true;
        text += advance();
    }

    // Check for hex prefix
    if (current() == '0' && pos_ + 1 < source_.size() &&
        (source_[pos_ + 1] == 'x' || source_[pos_ + 1] == 'X')) {
        text += advance();  // '0'
        text += advance();  // 'x'
        is_hex = true;
    }

    // Read digits
    while (pos_ < source_.size()) {
        char c = current();
        if (is_hex) {
            if (std::isxdigit(c)) {
                text += c;
                if (c >= '0' && c <= '9') {
                    value = value * 16 + (c - '0');
                } else if (c >= 'a' && c <= 'f') {
                    value = value * 16 + (c - 'a' + 10);
                } else {
                    value = value * 16 + (c - 'A' + 10);
                }
                advance();
            } else {
                break;
            }
        } else {
            if (std::isdigit(c)) {
                text += c;
                value = value * 10 + (c - '0');
                advance();
            } else {
                break;
            }
        }
    }

    if (is_negative) {
        value = -value;
    }

    Token tok{TokenType::NUMBER, text, start_line, start_col};
    tok.int_value = value;
    return tok;
}

Token Lexer::read_string() {
    size_t start_line = line_;
    size_t start_col = column_;
    std::string text;

    advance();  // Skip opening quote

    while (pos_ < source_.size() && current() != '"') {
        if (current() == '\n') {
            throw std::runtime_error("Unterminated string at line " + std::to_string(line_));
        }
        if (current() == '\\' && pos_ + 1 < source_.size()) {
            advance();  // Skip backslash
            char c = advance();
            switch (c) {
                case 'n': text += '\n'; break;
                case 't': text += '\t'; break;
                case 'r': text += '\r'; break;
                case '\\': text += '\\'; break;
                case '"': text += '"'; break;
                default: text += c; break;
            }
        } else {
            text += advance();
        }
    }

    if (current() != '"') {
        throw std::runtime_error("Unterminated string at line " + std::to_string(line_));
    }
    advance();  // Skip closing quote

    return Token{TokenType::STRING, text, start_line, start_col};
}

Token Lexer::next_token() {
    if (peeked_) {
        Token tok = *peeked_;
        peeked_.reset();
        return tok;
    }

    skip_whitespace();

    if (pos_ >= source_.size()) {
        return make_token(TokenType::END_OF_FILE);
    }

    char c = current();

    // Newline
    if (c == '\n') {
        advance();
        return make_token(TokenType::NEWLINE);
    }

    // Single-character tokens
    if (c == '(') { advance(); return make_token(TokenType::LPAREN, "("); }
    if (c == ')') { advance(); return make_token(TokenType::RPAREN, ")"); }
    if (c == ',') { advance(); return make_token(TokenType::COMMA, ","); }
    if (c == ':') { advance(); return make_token(TokenType::COLON, ":"); }
    if (c == '=') { advance(); return make_token(TokenType::EQUALS, "="); }

    // String
    if (c == '"') {
        return read_string();
    }

    // Number (including negative)
    if (std::isdigit(c) || (c == '-' && pos_ + 1 < source_.size() && std::isdigit(source_[pos_ + 1]))) {
        return read_number();
    }

    // Identifier (includes directives starting with '.')
    if (std::isalpha(c) || c == '_' || c == '.') {
        return read_identifier();
    }

    throw std::runtime_error("Unexpected character '" + std::string(1, c) +
                             "' at line " + std::to_string(line_));
}

Token Lexer::peek_token() {
    if (!peeked_) {
        peeked_ = next_token();
    }
    return *peeked_;
}

bool Lexer::at_end() const {
    return pos_ >= source_.size();
}

// ============================================================================
// Assembler Implementation
// ============================================================================

DMProgram Assembler::assemble(const std::string& source, const std::string& filename) {
    warnings_.clear();
    labels_.clear();
    current_file_ = filename;
    program_ = DMProgram{};
    next_instruction_id_ = 0;

    Lexer lexer(source, filename);
    lexer_ = &lexer;
    current_token_ = lexer_->next_token();

    // Parse the program
    while (!check(TokenType::END_OF_FILE)) {
        // Skip empty lines
        while (match(TokenType::NEWLINE)) {}

        if (check(TokenType::END_OF_FILE)) break;

        // Check for label
        if (check(TokenType::IDENTIFIER)) {
            Token id = current_token_;
            advance();
            if (match(TokenType::COLON)) {
                // It's a label
                labels_[id.text] = program_.instructions.size();
                continue;
            }

            // It's an instruction or directive
            // Put the token back conceptually by processing it
            if (id.text[0] == '.') {
                // Directive - we need to handle this specially since we already consumed the token
                if (id.text == ".name") { parse_name_directive(); }
                else if (id.text == ".version") { parse_version_directive(); }
                else if (id.text == ".dimensions") { parse_dimensions_directive(); }
                else if (id.text == ".tiling") { parse_tiling_directive(); }
                else if (id.text == ".l1_ki") { parse_l1_ki_directive(); }
                else if (id.text == ".dataflow") { parse_dataflow_directive(); }
                else if (id.text == ".a_base") { parse_base_directive("a"); }
                else if (id.text == ".b_base") { parse_base_directive("b"); }
                else if (id.text == ".c_base") { parse_base_directive("c"); }
                else { error("unknown directive '" + id.text + "'"); }
            } else {
                // Instruction - parse based on opcode
                DMInstruction instr;
                if (id.text == "DMA_LOAD_TILE") { instr = parse_dma_load_tile(); }
                else if (id.text == "DMA_STORE_TILE") { instr = parse_dma_store_tile(); }
                else if (id.text == "DMA_PREFETCH_TILE") { instr = parse_dma_prefetch_tile(); }
                else if (id.text == "DMA_LOAD_GATHER") { instr = parse_dma_load_gather(); }
                else if (id.text == "DMA_STORE_SCATTER") { instr = parse_dma_store_scatter(); }
                else if (id.text == "BM_MOVE_TILE") { instr = parse_bm_move_tile(); }
                else if (id.text == "BM_TRANSPOSE_TILE") { instr = parse_bm_transpose_tile(); }
                else if (id.text == "BM_WRITEBACK_TILE") { instr = parse_bm_writeback_tile(); }
                else if (id.text == "BM_RESHAPE_TILE") { instr = parse_bm_reshape_tile(); }
                else if (id.text == "STR_FEED_ROWS") { instr = parse_str_feed_rows(); }
                else if (id.text == "STR_FEED_COLS") { instr = parse_str_feed_cols(); }
                else if (id.text == "STR_DRAIN_OUTPUT") { instr = parse_str_drain_output(); }
                else if (id.text == "STR_BROADCAST_ROW") { instr = parse_str_broadcast_row(); }
                else if (id.text == "STR_BROADCAST_COL") { instr = parse_str_broadcast_col(); }
                else if (id.text == "VE_ELEMENTWISE") { instr = parse_ve_elementwise(); }
                else if (id.text == "VE_REDUCE") { instr = parse_ve_reduce(); }
                else if (id.text == "BARRIER") { instr = parse_barrier(); }
                else if (id.text == "WAIT_DMA") { instr = parse_wait_dma(); }
                else if (id.text == "WAIT_BM") { instr = parse_wait_bm(); }
                else if (id.text == "WAIT_STR") { instr = parse_wait_str(); }
                else if (id.text == "SIGNAL") { instr = parse_signal(); }
                else if (id.text == "SET_TILE_SIZE") { instr = parse_set_tile_size(); }
                else if (id.text == "SET_BUFFER") { instr = parse_set_buffer(); }
                else if (id.text == "SET_BASE") { instr = parse_set_base(); }
                else if (id.text == "SET_L3_BASE") { instr = parse_set_l3_base(); }
                else if (id.text == "SET_L2_BASE") { instr = parse_set_l2_base(); }
                else if (id.text == "SET_STRIDE") { instr = parse_set_stride_config(); }
                else if (id.text == "SET_TILE_DIM") { instr = parse_set_tile_dim(); }
                else if (id.text == "SET_MATRIX_DIM") { instr = parse_set_matrix_dim(); }
                else if (id.text == "LOOP_BEGIN") { instr = parse_loop_begin(); }
                else if (id.text == "LOOP_END") { instr = parse_loop_end(); }
                // AUTO addressing opcodes
                else if (id.text == "DMA_LOAD_TILE_AUTO") { instr = parse_dma_load_tile_auto(); }
                else if (id.text == "DMA_STORE_TILE_AUTO") { instr = parse_dma_store_tile_auto(); }
                else if (id.text == "BM_MOVE_TILE_AUTO") { instr = parse_bm_move_tile_auto(); }
                else if (id.text == "BM_WRITEBACK_AUTO") { instr = parse_bm_writeback_auto(); }
                else if (id.text == "STR_FEED_ROWS_AUTO") { instr = parse_str_feed_rows_auto(); }
                else if (id.text == "STR_FEED_COLS_AUTO") { instr = parse_str_feed_cols_auto(); }
                else if (id.text == "STR_DRAIN_AUTO") { instr = parse_str_drain_auto(); }
                else if (id.text == "L2_SCRATCH_WRITE") { instr = parse_l2_scratch_write(); }
                else if (id.text == "L2_SCRATCH_READ") { instr = parse_l2_scratch_read(); }
                else if (id.text == "NOP") { instr = parse_nop(); }
                else if (id.text == "HALT") { instr = parse_halt(); }
                else { error("unknown opcode '" + id.text + "'"); }

                instr.instruction_id = next_instruction_id_++;
                program_.instructions.push_back(instr);
            }
        } else {
            error("expected label, directive, or instruction");
        }

        // Expect newline or EOF after instruction/directive
        if (!check(TokenType::END_OF_FILE) && !check(TokenType::NEWLINE)) {
            error("expected end of line");
        }
    }

    lexer_ = nullptr;
    return program_;
}

DMProgram Assembler::assemble_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return assemble(buffer.str(), path);
}

void Assembler::advance() {
    current_token_ = lexer_->next_token();
}

bool Assembler::check(TokenType type) const {
    return current_token_.type == type;
}

bool Assembler::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

void Assembler::expect(TokenType type, const std::string& what) {
    if (!check(type)) {
        error("expected " + what);
    }
}

Token Assembler::consume(TokenType type, const std::string& what) {
    expect(type, what);
    Token tok = current_token_;
    advance();
    return tok;
}

void Assembler::error(const std::string& msg) {
    throw AssemblyError(current_file_, current_token_.line, msg);
}

void Assembler::warning(const std::string& msg) {
    warnings_.push_back(AssemblyWarning{current_file_, current_token_.line, msg});
}

// ============================================================================
// Directive Parsing
// ============================================================================

void Assembler::parse_name_directive() {
    program_.name = consume(TokenType::STRING, "program name string").text;
}

void Assembler::parse_version_directive() {
    program_.version = static_cast<uint32_t>(consume(TokenType::NUMBER, "version number").int_value);
}

void Assembler::parse_dimensions_directive() {
    // Parse M=value, N=value, K=value
    while (!check(TokenType::NEWLINE) && !check(TokenType::END_OF_FILE)) {
        Token id = consume(TokenType::IDENTIFIER, "dimension name (M, N, or K)");
        consume(TokenType::EQUALS, "'='");
        int64_t value = consume(TokenType::NUMBER, "dimension value").int_value;

        if (id.text == "M") program_.M = static_cast<Size>(value);
        else if (id.text == "N") program_.N = static_cast<Size>(value);
        else if (id.text == "K") program_.K = static_cast<Size>(value);
        else warning("unknown dimension '" + id.text + "'");

        match(TokenType::COMMA);  // Optional comma
    }
}

void Assembler::parse_tiling_directive() {
    // Parse Ti=value, Tj=value, Tk=value
    while (!check(TokenType::NEWLINE) && !check(TokenType::END_OF_FILE)) {
        Token id = consume(TokenType::IDENTIFIER, "tile dimension name (Ti, Tj, or Tk)");
        consume(TokenType::EQUALS, "'='");
        int64_t value = consume(TokenType::NUMBER, "tile dimension value").int_value;

        if (id.text == "Ti") program_.Ti = static_cast<Size>(value);
        else if (id.text == "Tj") program_.Tj = static_cast<Size>(value);
        else if (id.text == "Tk") program_.Tk = static_cast<Size>(value);
        else warning("unknown tile dimension '" + id.text + "'");

        match(TokenType::COMMA);  // Optional comma
    }
}

void Assembler::parse_l1_ki_directive() {
    program_.L1_Ki = static_cast<Size>(consume(TokenType::NUMBER, "L1_Ki value").int_value);
}

void Assembler::parse_dataflow_directive() {
    Token df = consume(TokenType::IDENTIFIER, "dataflow type");
    if (df.text == "output_stationary") {
        program_.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;
    } else if (df.text == "weight_stationary") {
        program_.dataflow = DMProgram::Dataflow::WEIGHT_STATIONARY;
    } else if (df.text == "input_stationary") {
        program_.dataflow = DMProgram::Dataflow::INPUT_STATIONARY;
    } else {
        error("unknown dataflow type '" + df.text + "'");
    }
}

void Assembler::parse_base_directive(const std::string& which) {
    Address addr = parse_address();
    if (which == "a") program_.memory_map.a_base = addr;
    else if (which == "b") program_.memory_map.b_base = addr;
    else if (which == "c") program_.memory_map.c_base = addr;
}

// ============================================================================
// Operand Parsing
// ============================================================================

MatrixID Assembler::parse_matrix_id() {
    Token tok = consume(TokenType::IDENTIFIER, "matrix ID (A, B, or C)");
    if (tok.text == "A") return MatrixID::A;
    if (tok.text == "B") return MatrixID::B;
    if (tok.text == "C") return MatrixID::C;
    error("invalid matrix ID '" + tok.text + "' (expected A, B, or C)");
    return MatrixID::A;  // Unreachable
}

TileCoord Assembler::parse_tile_coord() {
    consume(TokenType::LPAREN, "'('");
    int64_t ti = consume(TokenType::NUMBER, "ti value").int_value;
    match(TokenType::COMMA);
    int64_t tj = consume(TokenType::NUMBER, "tj value").int_value;
    match(TokenType::COMMA);
    int64_t tk = consume(TokenType::NUMBER, "tk value").int_value;
    consume(TokenType::RPAREN, "')'");

    return TileCoord{
        static_cast<uint16_t>(ti),
        static_cast<uint16_t>(tj),
        static_cast<uint16_t>(tk)
    };
}

BufferSlot Assembler::parse_buffer_slot() {
    Token tok = consume(TokenType::IDENTIFIER, "buffer slot (BUF_0, BUF_1, or AUTO)");
    if (tok.text == "BUF_0") return BufferSlot::BUF_0;
    if (tok.text == "BUF_1") return BufferSlot::BUF_1;
    if (tok.text == "AUTO") return BufferSlot::AUTO;
    error("invalid buffer slot '" + tok.text + "'");
    return BufferSlot::BUF_0;  // Unreachable
}

Transform Assembler::parse_transform() {
    Token tok = consume(TokenType::IDENTIFIER, "transform type");
    if (tok.text == "IDENTITY") return Transform::IDENTITY;
    if (tok.text == "TRANSPOSE") return Transform::TRANSPOSE;
    if (tok.text == "RESHAPE") return Transform::RESHAPE;
    if (tok.text == "SHUFFLE") return Transform::SHUFFLE;
    error("invalid transform type '" + tok.text + "'");
    return Transform::IDENTITY;  // Unreachable
}

ActivationType Assembler::parse_activation_type() {
    Token tok = consume(TokenType::IDENTIFIER, "activation type");
    if (tok.text == "NONE") return ActivationType::NONE;
    if (tok.text == "RELU") return ActivationType::RELU;
    if (tok.text == "GELU") return ActivationType::GELU;
    if (tok.text == "SIGMOID") return ActivationType::SIGMOID;
    if (tok.text == "TANH") return ActivationType::TANH;
    if (tok.text == "SILU" || tok.text == "SWISH") return ActivationType::SILU;
    if (tok.text == "SOFTPLUS") return ActivationType::SOFTPLUS;
    if (tok.text == "LEAKY_RELU") return ActivationType::LEAKY_RELU;
    error("invalid activation type '" + tok.text + "'");
    return ActivationType::NONE;  // Unreachable
}

int64_t Assembler::parse_number() {
    return consume(TokenType::NUMBER, "number").int_value;
}

Address Assembler::parse_address() {
    return static_cast<Address>(consume(TokenType::NUMBER, "address").int_value);
}

std::string Assembler::parse_string() {
    return consume(TokenType::STRING, "string").text;
}

// ============================================================================
// Instruction Parsing
// ============================================================================

DMInstruction Assembler::parse_dma_load_tile() {
    DMInstruction instr;
    instr.opcode = DMOpcode::DMA_LOAD_TILE;

    DMAOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.tile = parse_tile_coord();
    match(TokenType::COMMA);
    ops.ext_mem_addr = parse_address();
    match(TokenType::COMMA);
    ops.l3_tile_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.l3_offset = parse_address();
    match(TokenType::COMMA);
    ops.size_bytes = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();

    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_dma_store_tile() {
    DMInstruction instr;
    instr.opcode = DMOpcode::DMA_STORE_TILE;

    DMAOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.tile = parse_tile_coord();
    match(TokenType::COMMA);
    ops.ext_mem_addr = parse_address();
    match(TokenType::COMMA);
    ops.l3_tile_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.l3_offset = parse_address();
    match(TokenType::COMMA);
    ops.size_bytes = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();

    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_dma_prefetch_tile() {
    DMInstruction instr;
    instr.opcode = DMOpcode::DMA_PREFETCH_TILE;

    DMAOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.tile = parse_tile_coord();
    match(TokenType::COMMA);
    ops.ext_mem_addr = parse_address();
    match(TokenType::COMMA);
    ops.l3_tile_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.l3_offset = parse_address();
    match(TokenType::COMMA);
    ops.size_bytes = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();

    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_dma_load_gather() {
    // Same as DMA_LOAD_TILE for now (gather parameters handled separately)
    DMInstruction instr = parse_dma_load_tile();
    instr.opcode = DMOpcode::DMA_LOAD_GATHER;
    return instr;
}

DMInstruction Assembler::parse_dma_store_scatter() {
    DMInstruction instr = parse_dma_store_tile();
    instr.opcode = DMOpcode::DMA_STORE_SCATTER;
    return instr;
}

DMInstruction Assembler::parse_bm_move_tile() {
    DMInstruction instr;
    instr.opcode = DMOpcode::BM_MOVE_TILE;

    BlockMoverOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.tile = parse_tile_coord();
    match(TokenType::COMMA);
    ops.src_l3_tile_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.src_offset = parse_address();
    match(TokenType::COMMA);
    ops.dst_l2_bank_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.dst_offset = parse_address();
    match(TokenType::COMMA);
    ops.height = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.width = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.element_size = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();
    ops.transform = Transform::IDENTITY;

    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_bm_transpose_tile() {
    DMInstruction instr = parse_bm_move_tile();
    instr.opcode = DMOpcode::BM_TRANSPOSE_TILE;
    std::get<BlockMoverOperands>(instr.operands).transform = Transform::TRANSPOSE;
    return instr;
}

DMInstruction Assembler::parse_bm_writeback_tile() {
    DMInstruction instr = parse_bm_move_tile();
    instr.opcode = DMOpcode::BM_WRITEBACK_TILE;
    return instr;
}

DMInstruction Assembler::parse_bm_reshape_tile() {
    DMInstruction instr = parse_bm_move_tile();
    instr.opcode = DMOpcode::BM_RESHAPE_TILE;
    std::get<BlockMoverOperands>(instr.operands).transform = Transform::RESHAPE;
    return instr;
}

DMInstruction Assembler::parse_str_feed_rows() {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_FEED_ROWS;

    StreamerOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.tile = parse_tile_coord();
    match(TokenType::COMMA);
    ops.l2_bank_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.l1_buffer_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.l2_addr = parse_address();
    match(TokenType::COMMA);
    ops.l1_addr = parse_address();
    match(TokenType::COMMA);
    ops.height = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.width = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.fabric_size = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();

    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_str_feed_cols() {
    DMInstruction instr = parse_str_feed_rows();
    instr.opcode = DMOpcode::STR_FEED_COLS;
    return instr;
}

DMInstruction Assembler::parse_str_drain_output() {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_DRAIN_OUTPUT;

    StreamerOperands ops;
    ops.matrix = MatrixID::C;  // Drain is always for C
    ops.tile = parse_tile_coord();
    match(TokenType::COMMA);
    ops.l2_bank_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.l1_buffer_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.l2_addr = parse_address();
    match(TokenType::COMMA);
    ops.l1_addr = parse_address();
    match(TokenType::COMMA);
    ops.height = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.width = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.fabric_size = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();

    // Check for optional VE options
    if (match(TokenType::COMMA)) {
        Token tok = consume(TokenType::IDENTIFIER, "VE_ENABLE or end of operands");
        if (tok.text == "VE_ENABLE") {
            ops.ve_enabled = true;
            ops.ve_activation = parse_activation_type();

            // Check for optional BIAS
            if (match(TokenType::COMMA)) {
                Token bias_tok = consume(TokenType::IDENTIFIER, "BIAS");
                if (bias_tok.text == "BIAS") {
                    ops.ve_bias_enabled = true;
                    ops.ve_bias_addr = parse_address();
                } else {
                    error("expected 'BIAS' after VE_ENABLE activation");
                }
            }
        } else {
            error("expected 'VE_ENABLE'");
        }
    }

    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_str_broadcast_row() {
    DMInstruction instr = parse_str_feed_rows();
    instr.opcode = DMOpcode::STR_BROADCAST_ROW;
    return instr;
}

DMInstruction Assembler::parse_str_broadcast_col() {
    DMInstruction instr = parse_str_feed_rows();
    instr.opcode = DMOpcode::STR_BROADCAST_COL;
    return instr;
}

DMInstruction Assembler::parse_ve_elementwise() {
    // Syntax (issue #100):
    //   binary:  VE_ELEMENTWISE <OP>, <l1_src_a>, <l1_src_b>, <l1_dst>
    //   unary:   VE_ELEMENTWISE <OP>, <l1_src>, <l1_dst>
    //   scalar:  VE_ELEMENTWISE <OP_S> <scalar>, <l1_src>, <l1_dst>
    DMInstruction instr;
    instr.opcode = DMOpcode::VE_ELEMENTWISE;

    Token op_tok = consume(TokenType::IDENTIFIER, "elementwise operation");
    VEOp op;
    if (!ve_op_from_string(op_tok.text, op)) {
        error("Unknown VE elementwise operation: " + op_tok.text);
    }

    VEOperands ops;
    ops.op = op;

    const bool scalar_form =
        op == VEOp::ADD_S || op == VEOp::MUL_S || op == VEOp::POW_S;
    const bool unary_form =
        scalar_form || op == VEOp::NEG || op == VEOp::ABS ||
        op == VEOp::SQRT || op == VEOp::EXP || op == VEOp::LOG;

    if (scalar_form) {
        ops.scalar = static_cast<float>(parse_number());
    }
    match(TokenType::COMMA);
    ops.l1_src_a = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    if (unary_form) {
        ops.num_inputs = 1;
        ops.l1_dst = static_cast<uint8_t>(parse_number());
    } else {
        ops.num_inputs = 2;
        ops.l1_src_b = static_cast<uint8_t>(parse_number());
        match(TokenType::COMMA);
        ops.l1_dst = static_cast<uint8_t>(parse_number());
    }

    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_ve_reduce() {
    DMInstruction instr;
    instr.opcode = DMOpcode::VE_REDUCE;
    consume(TokenType::IDENTIFIER, "reduction operation");
    instr.operands = std::monostate{};
    return instr;
}

DMInstruction Assembler::parse_barrier() {
    DMInstruction instr;
    instr.opcode = DMOpcode::BARRIER;
    instr.operands = std::monostate{};
    return instr;
}

DMInstruction Assembler::parse_wait_dma() {
    DMInstruction instr;
    instr.opcode = DMOpcode::WAIT_DMA;
    SyncOperands ops;
    ops.wait_mask = static_cast<uint32_t>(parse_number());
    ops.signal_id = 0;
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_wait_bm() {
    DMInstruction instr;
    instr.opcode = DMOpcode::WAIT_BM;
    SyncOperands ops;
    ops.wait_mask = static_cast<uint32_t>(parse_number());
    ops.signal_id = 0;
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_wait_str() {
    DMInstruction instr;
    instr.opcode = DMOpcode::WAIT_STR;
    SyncOperands ops;
    ops.wait_mask = static_cast<uint32_t>(parse_number());
    ops.signal_id = 0;
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_signal() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SIGNAL;
    SyncOperands ops;
    ops.wait_mask = 0;
    ops.signal_id = static_cast<uint32_t>(parse_number());
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_set_tile_size() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_TILE_SIZE;
    ConfigOperands ops{};
    ops.Ti = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.Tj = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.Tk = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.L1_Ki = static_cast<Size>(parse_number());
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_set_buffer() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_BUFFER;
    ConfigOperands ops{};
    ops.buffer_id = static_cast<uint8_t>(parse_number());
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_set_stride() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_STRIDE;
    ConfigOperands ops{};
    ops.stride_m = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.stride_n = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.stride_k = static_cast<Size>(parse_number());
    instr.operands = ops;
    return instr;
}

IndexRole Assembler::parse_index_role() {
    Token tok = consume(TokenType::IDENTIFIER, "index role (TI, TJ, TK, or NONE)");
    if (tok.text == "TI") return IndexRole::TI;
    if (tok.text == "TJ") return IndexRole::TJ;
    if (tok.text == "TK") return IndexRole::TK;
    if (tok.text == "NONE") return IndexRole::NONE;
    error("invalid index role '" + tok.text + "' (expected TI, TJ, TK, or NONE)");
    return IndexRole::NONE;  // Unreachable
}

DMInstruction Assembler::parse_loop_begin() {
    DMInstruction instr;
    instr.opcode = DMOpcode::LOOP_BEGIN;
    LoopOperands ops;
    ops.loop_id = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.loop_count = static_cast<uint16_t>(parse_number());

    // Check for optional index role and stride
    if (match(TokenType::COMMA)) {
        // Could be either index_role or loop_stride (for backward compatibility)
        if (check(TokenType::IDENTIFIER)) {
            // It's an index role
            ops.index_role = parse_index_role();
            ops.loop_stride = 1;  // Default stride
        } else {
            // It's a stride (old format)
            ops.loop_stride = static_cast<uint16_t>(parse_number());
            ops.index_role = IndexRole::NONE;
        }
    } else {
        ops.loop_stride = 1;
        ops.index_role = IndexRole::NONE;
    }

    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_loop_end() {
    DMInstruction instr;
    instr.opcode = DMOpcode::LOOP_END;
    LoopOperands ops;
    ops.loop_id = static_cast<uint8_t>(parse_number());
    ops.loop_count = 0;
    ops.loop_stride = 1;
    ops.index_role = IndexRole::NONE;
    instr.operands = ops;
    return instr;
}

// ============================================================================
// Configuration Instruction Parsing (new ISA extensions)
// ============================================================================

DMInstruction Assembler::parse_set_base() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_BASE;
    BaseAddressOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.base_addr = parse_address();
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_set_l3_base() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_L3_BASE;
    L3BaseOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.l3_offset = parse_address();
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_set_l2_base() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_L2_BASE;
    L2BaseOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.l2_offset = parse_address();
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_set_stride_config() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_STRIDE;
    StrideConfigOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.row_stride = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.tile_i_stride = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.tile_j_stride = static_cast<Size>(parse_number());
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_set_tile_dim() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_TILE_DIM;
    TileDimOperands ops;
    ops.Ti = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.Tj = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.Tk = static_cast<Size>(parse_number());
    // Element size is optional, default to 4 (float32)
    if (match(TokenType::COMMA)) {
        ops.element_size = static_cast<Size>(parse_number());
    } else {
        ops.element_size = 4;
    }
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_set_matrix_dim() {
    DMInstruction instr;
    instr.opcode = DMOpcode::SET_MATRIX_DIM;
    MatrixDimOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.rows = static_cast<Size>(parse_number());
    match(TokenType::COMMA);
    ops.cols = static_cast<Size>(parse_number());
    instr.operands = ops;
    return instr;
}

// ============================================================================
// AUTO Addressing Instruction Parsing
// ============================================================================

DMInstruction Assembler::parse_dma_load_tile_auto() {
    DMInstruction instr;
    instr.opcode = DMOpcode::DMA_LOAD_TILE_AUTO;
    AutoDMAOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.l3_slot = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_dma_store_tile_auto() {
    DMInstruction instr;
    instr.opcode = DMOpcode::DMA_STORE_TILE_AUTO;
    AutoDMAOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.l3_slot = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_bm_move_tile_auto() {
    DMInstruction instr;
    instr.opcode = DMOpcode::BM_MOVE_TILE_AUTO;
    AutoBlockMoverOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.l2_bank = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();
    ops.transform = Transform::IDENTITY;
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_bm_writeback_auto() {
    DMInstruction instr;
    instr.opcode = DMOpcode::BM_WRITEBACK_AUTO;
    AutoBlockMoverOperands ops;
    ops.matrix = parse_matrix_id();
    match(TokenType::COMMA);
    ops.l2_bank = static_cast<uint8_t>(parse_number());  // l3_slot stored in l2_bank
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();
    ops.transform = Transform::IDENTITY;
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_str_feed_rows_auto() {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_FEED_ROWS_AUTO;
    AutoStreamerOperands ops;
    ops.l1_buffer = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();
    ops.ve_enabled = false;
    ops.ve_activation = ActivationType::NONE;
    ops.ve_bias_enabled = false;
    ops.ve_bias_addr = 0;
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_str_feed_cols_auto() {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_FEED_COLS_AUTO;
    AutoStreamerOperands ops;
    ops.l1_buffer = static_cast<uint8_t>(parse_number());
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();
    ops.ve_enabled = false;
    ops.ve_activation = ActivationType::NONE;
    ops.ve_bias_enabled = false;
    ops.ve_bias_addr = 0;
    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_str_drain_auto() {
    DMInstruction instr;
    instr.opcode = DMOpcode::STR_DRAIN_AUTO;
    AutoStreamerOperands ops;
    ops.l1_buffer = static_cast<uint8_t>(parse_number());  // l2_bank stored in l1_buffer
    match(TokenType::COMMA);
    ops.buffer = parse_buffer_slot();
    ops.ve_enabled = false;
    ops.ve_activation = ActivationType::NONE;
    ops.ve_bias_enabled = false;
    ops.ve_bias_addr = 0;

    // Check for optional VE configuration
    if (match(TokenType::COMMA)) {
        Token tok = consume(TokenType::IDENTIFIER, "VE_ENABLE");
        if (tok.text == "VE_ENABLE") {
            ops.ve_enabled = true;
            ops.ve_activation = parse_activation_type();
        } else {
            error("expected 'VE_ENABLE'");
        }
    }

    instr.operands = ops;
    return instr;
}

DMInstruction Assembler::parse_l2_scratch_write() {
    DMInstruction instr;
    instr.opcode = DMOpcode::L2_SCRATCH_WRITE;
    instr.operands = std::monostate{};
    return instr;
}

DMInstruction Assembler::parse_l2_scratch_read() {
    DMInstruction instr;
    instr.opcode = DMOpcode::L2_SCRATCH_READ;
    instr.operands = std::monostate{};
    return instr;
}

DMInstruction Assembler::parse_nop() {
    DMInstruction instr;
    instr.opcode = DMOpcode::NOP;
    instr.operands = std::monostate{};
    return instr;
}

DMInstruction Assembler::parse_halt() {
    DMInstruction instr;
    instr.opcode = DMOpcode::HALT;
    instr.operands = std::monostate{};
    return instr;
}

} // namespace sw::kpu::isa
