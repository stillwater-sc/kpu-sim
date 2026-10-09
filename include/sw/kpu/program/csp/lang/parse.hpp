// ============================================================================
// include/sw/kpu/program/csp/lang/parse.hpp
// The CSP language: its AST and parser (docs/plans/csp-language.md §3; ADR 0004).
//
//   csp 1.0
//   program matmul machine flat(l3 = 128) {
//     tensor A[256,256] tile 32x32 in;
//     ...
//     for j in 0..8 {
//       resident B[:, j];
//       ...
//       acc C[i, j] in fabric {
//         for k in 0..8 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }
//       }
//       store C[i, j];
//     }
//   }
//
// C-style blocks, ';'-terminated statements, '//' comments. The parser builds the AST and
// reports syntax errors by line; meaning (residency, capacity, operands present) is the
// compiler's (compile.hpp).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::lang {

inline constexpr const char* kVersion = "1.0";

class SyntaxError : public std::runtime_error {
public:
    SyntaxError(int line, const std::string& what)
        : std::runtime_error("csp line " + std::to_string(line) + ": " + what), line_(line) {}
    int line() const { return line_; }
private:
    int line_;
};

// ---- AST --------------------------------------------------------------------------------------

struct Expr {
    enum class Kind : std::uint8_t { Int, Var, Bin } kind = Kind::Int;
    long long value = 0;
    std::string name;
    char op = 0;                              // Bin: '+', '-', '*'
    std::vector<Expr> args;                   // Bin: two operands
};

struct Index {
    bool all = false;                         // ':' -- every tile along this dimension
    Expr expr;
};

struct TileRef {
    std::string name;
    std::vector<Index> index;                 // one (a vector) or two (a tensor)
    int line = 0;
};

struct Stage {                                // a tile-context stage: op(args) @ place
    std::string op;
    std::vector<TileRef> args;
    std::string place;
    int line = 0;
};

struct Stmt {
    enum class Kind : std::uint8_t { For, Resident, Release, Acc, Call, Store, Distribute, Broadcast };
    Kind kind = Kind::Call;
    int line = 0;
    // For
    std::string var;
    Expr lo, hi;
    std::vector<Stmt> body;                   // For, Acc
    // Resident, Release, Broadcast; Call: the arguments
    std::vector<TileRef> tiles;
    // Call
    std::string fn;
    bool accumulate = false;                  // '+->'
    TileRef out;                              // Call: the result; Acc, Store: the tile
    std::optional<Expr> pivot;
    std::optional<double> alpha;
    std::vector<Stage> context;               // Call, Store: 'via' stages
    // Distribute / Broadcast
    std::string map, along;
};

struct Decl {
    bool is_vector = false;
    std::string name;
    long long rows = 0, cols = 0, tile_rows = 0, tile_cols = 0;
    std::string io;                           // in | out | inout
    int line = 0;
};

struct Program {
    std::string version;
    std::string name;
    std::string machine;                      // flat | grid
    long long l3 = 0;
    long long grid_rows = 0, grid_cols = 0;
    std::vector<Decl> decls;
    std::vector<Stmt> body;
};

// ---- lexer ------------------------------------------------------------------------------------

struct Token {
    enum class Kind : std::uint8_t { Ident, Int, Num, Sym, End } kind = Kind::End;
    std::string text;
    int line = 0;
};

inline std::vector<Token> lex(const std::string& src) {
    std::vector<Token> out;
    int line = 1;
    std::size_t i = 0;
    auto push = [&](Token::Kind k, std::string t) { out.push_back({k, std::move(t), line}); };
    while (i < src.size()) {
        const char c = src[i];
        if (c == '\n') { ++line; ++i; continue; }
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (c == '/' && i + 1 < src.size() && src[i + 1] == '/') {
            while (i < src.size() && src[i] != '\n') ++i;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            std::size_t j = i;
            while (j < src.size() && std::isdigit(static_cast<unsigned char>(src[j]))) ++j;
            // A decimal ("1.0", "0.5", "1e-07", "9.99999975e-05"), but not a range ("0..8").
            bool real = false;
            if (j + 1 < src.size() && src[j] == '.' && std::isdigit(static_cast<unsigned char>(src[j + 1]))) {
                ++j;
                while (j < src.size() && std::isdigit(static_cast<unsigned char>(src[j]))) ++j;
                real = true;
            }
            if (j < src.size() && (src[j] == 'e' || src[j] == 'E')) {
                std::size_t k = j + 1;
                if (k < src.size() && (src[k] == '+' || src[k] == '-')) ++k;
                if (k < src.size() && std::isdigit(static_cast<unsigned char>(src[k]))) {
                    while (k < src.size() && std::isdigit(static_cast<unsigned char>(src[k]))) ++k;
                    j = k;
                    real = true;
                }
            }
            if (real) {
                push(Token::Kind::Num, src.substr(i, j - i));
                i = j;
                continue;
            }
            push(Token::Kind::Int, src.substr(i, j - i));
            i = j;
            // A shape "32x32": the 'x' is a separator, not the start of a name.
            if (i + 1 < src.size() && src[i] == 'x' && std::isdigit(static_cast<unsigned char>(src[i + 1]))) {
                push(Token::Kind::Sym, "x");
                ++i;
            }
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::size_t j = i;
            // A name may hold single dots ("bm.egress"); ".." starts a range ("i..k").
            while (j < src.size() && (std::isalnum(static_cast<unsigned char>(src[j])) || src[j] == '_' ||
                                      (src[j] == '.' && !(j + 1 < src.size() && src[j + 1] == '.'))))
                ++j;
            while (j > i && src[j - 1] == '.') --j;
            push(Token::Kind::Ident, src.substr(i, j - i));
            i = j;
            continue;
        }
        for (const char* s : {"+->", "->", ".."}) {
            const std::string sym = s;
            if (src.compare(i, sym.size(), sym) == 0) {
                push(Token::Kind::Sym, sym);
                i += sym.size();
                goto next;
            }
        }
        if (std::string("{}()[],;:=@+-*").find(c) != std::string::npos) {
            push(Token::Kind::Sym, std::string(1, c));
            ++i;
            continue;
        }
        throw SyntaxError(line, std::string("unexpected character '") + c + "'");
    next:;
    }
    push(Token::Kind::End, "end of file");
    return out;
}

// ---- parser -----------------------------------------------------------------------------------

class Parser {
public:
    explicit Parser(const std::string& src) : t_(lex(src)) {}

    Program parse() {
        Program p;
        expect_word("csp");
        p.version = take_version();
        if (p.version != kVersion)
            throw SyntaxError(line(), "this reader reads csp " + std::string(kVersion) + ", not " + p.version);
        expect_word("program");
        p.name = ident("a program name");
        expect_word("machine");
        p.machine = ident("flat or grid");
        expect("(");
        if (p.machine == "flat") {
            expect_word("l3");
            expect("=");
            p.l3 = integer("the L3 capacity");
        } else if (p.machine == "grid") {
            p.grid_rows = integer("the grid's rows");
            expect("x");
            p.grid_cols = integer("the grid's columns");
            expect(",");
            expect_word("l3");
            expect("=");
            p.l3 = integer("the L3 capacity");
        } else {
            throw SyntaxError(line(), "machine '" + p.machine + "': flat or grid");
        }
        expect(")");
        expect("{");
        while (word("tensor") || word("vector")) p.decls.push_back(decl());
        while (!sym("}")) p.body.push_back(stmt());
        expect("}");
        if (peek().kind != Token::Kind::End) throw SyntaxError(line(), "text after the program");
        return p;
    }

private:
    std::vector<Token> t_;
    std::size_t at_ = 0;

    const Token& peek() const { return t_[at_]; }
    int line() const { return peek().line; }
    bool sym(const char* s) const { return peek().kind == Token::Kind::Sym && peek().text == s; }
    bool word(const char* s) const { return peek().kind == Token::Kind::Ident && peek().text == s; }
    Token take() { return t_[at_ < t_.size() - 1 ? at_++ : at_]; }
    void expect(const char* s) {
        // A missing terminator belongs to the line it should have ended, not the next one.
        if (!sym(s))
            throw SyntaxError(at_ > 0 ? t_[at_ - 1].line : line(),
                              std::string("expected '") + s + "', found '" + peek().text + "'");
        take();
    }
    void expect_word(const char* s) {
        if (!word(s)) throw SyntaxError(line(), std::string("expected '") + s + "', found '" + peek().text + "'");
        take();
    }
    std::string ident(const char* what) {
        if (peek().kind != Token::Kind::Ident) throw SyntaxError(line(), std::string("expected ") + what + ", found '" + peek().text + "'");
        return take().text;
    }
    long long integer(const char* what) {
        if (peek().kind != Token::Kind::Int) throw SyntaxError(line(), std::string("expected ") + what + ", found '" + peek().text + "'");
        return std::stoll(take().text);
    }
    std::string take_version() {
        if (peek().kind != Token::Kind::Num) throw SyntaxError(line(), "expected a version such as 1.0");
        return take().text;
    }

    Decl decl() {
        Decl d;
        d.line = line();
        d.is_vector = take().text == "vector";
        d.name = ident("an operand name");
        expect("[");
        d.rows = integer("a size");
        if (!d.is_vector) {
            expect(",");
            d.cols = integer("a size");
        } else {
            d.cols = 1;
        }
        expect("]");
        expect_word("tile");
        d.tile_rows = integer("a tile size");
        if (!d.is_vector) {
            expect("x");
            d.tile_cols = integer("a tile size");
        } else {
            d.tile_cols = 1;
        }
        d.io = ident("in, out or inout");
        if (d.io != "in" && d.io != "out" && d.io != "inout")
            throw SyntaxError(d.line, "'" + d.io + "': in, out or inout");
        expect(";");
        return d;
    }

    Expr primary() {
        if (sym("(")) {
            take();
            Expr e = expr();
            expect(")");
            return e;
        }
        Expr e;
        if (peek().kind == Token::Kind::Int) {
            e.kind = Expr::Kind::Int;
            e.value = std::stoll(take().text);
        } else if (peek().kind == Token::Kind::Ident) {
            e.kind = Expr::Kind::Var;
            e.name = take().text;
        } else {
            throw SyntaxError(line(), "expected an index expression, found '" + peek().text + "'");
        }
        return e;
    }
    Expr term() {
        Expr e = primary();
        while (sym("*")) {
            take();
            Expr b;
            b.kind = Expr::Kind::Bin;
            b.op = '*';
            b.args = {e, primary()};
            e = b;
        }
        return e;
    }
    Expr expr() {
        Expr e = term();
        while (sym("+") || sym("-")) {
            const char op = take().text[0];
            Expr b;
            b.kind = Expr::Kind::Bin;
            b.op = op;
            b.args = {e, term()};
            e = b;
        }
        return e;
    }

    TileRef tile() {
        TileRef r;
        r.line = line();
        r.name = ident("an operand name");
        expect("[");
        do {
            if (sym(",")) take();
            Index ix;
            if (sym(":")) {
                take();
                ix.all = true;
            } else {
                ix.expr = expr();
            }
            r.index.push_back(ix);
        } while (sym(","));
        expect("]");
        return r;
    }
    std::vector<TileRef> tiles() {
        std::vector<TileRef> v{tile()};
        while (sym(",")) {
            take();
            v.push_back(tile());
        }
        return v;
    }

    std::vector<Stage> context() {
        std::vector<Stage> v;
        if (!word("via")) return v;
        take();
        do {
            if (sym(",")) take();
            Stage s;
            s.line = line();
            s.op = ident("a vector operation");
            if (sym("(")) {
                take();
                if (!sym(")")) s.args = tiles();
                expect(")");
            }
            expect("@");
            s.place = ident("a place: fabric, str.drain, bm.egress or bm.ingress");
            v.push_back(s);
        } while (sym(","));
        return v;
    }

    double number() {
        bool neg = false;
        if (sym("-")) { take(); neg = true; }
        if (peek().kind != Token::Kind::Int && peek().kind != Token::Kind::Num)
            throw SyntaxError(line(), "expected a number, found '" + peek().text + "'");
        const double v = std::stod(take().text);
        return neg ? -v : v;
    }

    std::vector<Stmt> block() {
        expect("{");
        std::vector<Stmt> v;
        while (!sym("}")) {
            if (peek().kind == Token::Kind::End) throw SyntaxError(line(), "unclosed block");
            v.push_back(stmt());
        }
        expect("}");
        return v;
    }

    Stmt stmt() {
        Stmt s;
        s.line = line();
        const std::string kw = ident("a statement");
        if (kw == "for") {
            s.kind = Stmt::Kind::For;
            s.var = ident("a loop variable");
            expect_word("in");
            s.lo = expr();
            expect("..");
            s.hi = expr();
            s.body = block();
            return s;
        }
        if (kw == "resident" || kw == "release") {
            s.kind = kw == "resident" ? Stmt::Kind::Resident : Stmt::Kind::Release;
            s.tiles = tiles();
            expect(";");
            return s;
        }
        if (kw == "acc") {
            s.kind = Stmt::Kind::Acc;
            s.out = tile();
            expect_word("in");
            expect_word("fabric");
            s.body = block();
            return s;
        }
        if (kw == "call") {
            s.kind = Stmt::Kind::Call;
            s.fn = ident("a tile function");
            expect("(");
            if (!sym(")")) s.tiles = tiles();
            expect(")");
            if (sym("+->")) s.accumulate = true;
            else if (!sym("->")) throw SyntaxError(line(), "expected '->' or '+->' after the call");
            take();
            s.out = tile();
            for (;;) {
                if (word("pivot")) { take(); s.pivot = expr(); continue; }
                if (word("alpha")) { take(); s.alpha = number(); continue; }
                break;
            }
            s.context = context();
            expect(";");
            return s;
        }
        if (kw == "store") {
            s.kind = Stmt::Kind::Store;
            s.out = tile();
            s.context = context();
            expect(";");
            return s;
        }
        if (kw == "distribute") {
            s.kind = Stmt::Kind::Distribute;
            s.tiles = {TileRef{ident("an operand"), {}, s.line}};
            expect_word("over");
            expect_word("grid");
            s.map = ident("block2d or cyclic2d");
            expect(";");
            return s;
        }
        if (kw == "broadcast") {
            s.kind = Stmt::Kind::Broadcast;
            s.tiles = tiles();
            expect_word("along");
            s.along = ident("row or col");
            expect(";");
            return s;
        }
        throw SyntaxError(s.line, "unknown statement '" + kw + "'");
    }
};

inline Program parse(const std::string& source) { return Parser(source).parse(); }

}  // namespace sw::kpu::program::csp::lang
