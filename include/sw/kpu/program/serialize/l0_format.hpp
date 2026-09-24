// ============================================================================
// include/sw/kpu/program/serialize/l0_format.hpp
// The L0 portable program, as a file (#265 increment 1).
//
// ADR 0001 D1 makes the L0 TileProgram the portable program, so this is the
// format that has to survive: a program written today must still load and
// execute identically later, or the "portable" claim is decoration.
//
// TEXT, LINE-ORIENTED, one record per line, deliberately (design note §5). L0
// files are op lists, not weights, so being greppable, diffable and reviewable in
// a pull request is worth more than parse speed for a format whose consumer then
// runs a simulation orders of magnitude longer. A golden corpus whose rot shows up
// in a diff is a corpus that cannot rot quietly -- which is exactly how
// kernels/bin/*.kpubin died: opcodes renumbered with no version bump, and those
// files now abort with "std::get: wrong index for variant". A crash is not a
// diagnostic.
//
// Op fields are KEYED rather than positional (kind=..., in=..., out=...), which
// is what makes the add-only rule of §4/R5 cheap: a new optional attribute is a
// new key that old readers ignore, and no existing field moves.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/version.hpp>

#include <cctype>
#include <cstdint>
#include <istream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::serialize {

// ----------------------------------------------------------------------------
// Versions (design note §4, requirements R1/R3/R4)
//
// Three axes, because they answer different questions and move at different
// rates: the CONTAINER's structure, the OP SET's semantics, and who produced the
// file. Adding an op is not the same kind of change as adding a record.
// ----------------------------------------------------------------------------
struct Version {
    unsigned major = 0, minor = 0, patch = 0;

    std::string str() const {
        return std::to_string(major) + "." + std::to_string(minor) + "." +
               std::to_string(patch);
    }
    // Ordering is on (major, minor, patch), so a reader can compare its own
    // support against a file's demand without string games.
    bool operator<(const Version& o) const {
        if (major != o.major) return major < o.major;
        if (minor != o.minor) return minor < o.minor;
        return patch < o.patch;
    }
    bool operator==(const Version& o) const {
        return major == o.major && minor == o.minor && patch == o.patch;
    }
    bool operator<=(const Version& o) const { return *this < o || *this == o; }
};

inline constexpr const char* kMagic = "KPUL0";

// What this build writes, and what it can read.
inline Version format_version()  { return {1, 0, 0}; }   // container structure
inline Version opset_version()   { return {1, 0, 0}; }   // the TileOpKind surface
inline Version reader_version()  { return {1, 0, 0}; }   // what THIS reader supports

// R1c: who produced the file, at ITS OWN version. Reusing the format version here
// would make the field useless for the thing it exists for -- identifying a
// producer that wrote bad files (R4's bad_producers list needs a version that
// tracks the build, not one that tracks the schema).
inline Version producer_version() {
    return {static_cast<unsigned>(sw::kpu::VERSION_MAJOR),
            static_cast<unsigned>(sw::kpu::VERSION_MINOR),
            static_cast<unsigned>(sw::kpu::VERSION_PATCH)};
}

// ----------------------------------------------------------------------------
// One error type, so a caller can distinguish "this file is not for me" from
// "this file is broken" without parsing a message.
// ----------------------------------------------------------------------------
class FormatError : public std::runtime_error {
public:
    enum class Cause {
        NotAnL0File,        // missing or wrong magic
        MalformedPreamble,  // a required preamble record missing or unparseable
        UnsupportedVersion, // the file demands a newer reader (R4)
        UnknownOp,          // an op this build does not implement (R8: refuse)
        MalformedRecord,    // a record missing a required field
        Truncated,          // ended before END
    };
    FormatError(Cause c, const std::string& what)
        : std::runtime_error(what), cause_(c) {}
    Cause cause() const { return cause_; }
private:
    Cause cause_;
};

// ----------------------------------------------------------------------------
// Writing
// ----------------------------------------------------------------------------
namespace detail {

// Quote a record field. CONTROL CHARACTERS MUST BE ENCODED, not merely escaped: the
// reader is getline()-based, so a literal newline inside a value splits one record into
// two, and a value containing a line reading END would terminate the read early and
// silently drop every op after it. Escaping only " and \ left that reachable.
inline std::string quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    out += '"';
    return out;
}

// Reserved characters in the OPERAND component of a coordinate, percent-encoded. A
// TensorOperand name is a free string, and coord() used to write it raw -- so a name
// containing a space, ':' or ';' produced a file write_l0() could emit and read_l0()
// could not parse. Encoding is narrow on purpose: only what the coordinate grammar and
// the field splitter treat as structure.
inline std::string encode_operand(const std::string& s) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        const bool reserved = c == '%' || c == ':' || c == ';' || c == ',' || c == '"' ||
                              c == '\\' || c <= ' ' || c == 0x7f;
        if (reserved) {
            out += '%';
            out += hex[(c >> 4) & 0xF];
            out += hex[c & 0xF];
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

inline std::string coord(const TileCoord& c) {
    return encode_operand(c.operand) + ":" + std::to_string(c.ti) + "," +
           std::to_string(c.tj);
}

// Enough digits to round-trip a float exactly, and independent of the caller's locale.
inline std::string exact_float(float f) {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::setprecision(std::numeric_limits<float>::max_digits10) << f;
    return os.str();
}

inline float parse_float(const std::string& s, const std::string& where) {
    std::istringstream is(s);
    is.imbue(std::locale::classic());       // never the caller's decimal separator
    float v = 0.0f;
    is >> v;
    if (!is || !is.eof())
        throw FormatError(FormatError::Cause::MalformedRecord,
                          "l0: " + where + ": '" + s + "' is not a number");
    return v;
}

inline std::string coord_list(const std::vector<TileCoord>& v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ";";
        out += coord(v[i]);
    }
    return out;
}

} // namespace detail

// Write `prog`'s structure. Values are increment 2; `VALUES none` says so IN THE
// FILE, so a reader knows it holds a kernel rather than a test case whose inputs
// happened to be zero (design note §3).
inline void write_l0(std::ostream& os, const TileProgram& prog) {
    os << kMagic << " " << format_version().str() << "\n";
    os << "MIN_CONSUMER " << format_version().str() << "\n";
    os << "OPSET tile " << opset_version().str() << "\n";
    os << "PRODUCER kpu-sim " << producer_version().str() << "\n";
    os << "PROGRAM " << detail::quote(prog.name()) << "\n";
    os << "VALUES none\n";

    for (const std::string& key : prog.operand_order()) {
        const TensorOperand& op = prog.operand(key);
        // Logical shape AND tile shape, because ragged trailing tiles are derived
        // from both and a file that dropped either would not round-trip.
        os << "OPERAND " << detail::quote(op.name) << " rows=" << op.rows
           << " cols=" << op.cols << " tile_rows=" << op.tile_rows
           << " tile_cols=" << op.tile_cols << "\n";
    }

    for (const TileOp& op : prog.ops()) {
        os << "OP kind=" << to_string(op.kind);
        if (!op.inputs.empty())  os << " in=" << detail::coord_list(op.inputs);
        if (!op.outputs.empty()) os << " out=" << detail::coord_list(op.outputs);
        if (op.kind == TileOpKind::Feed || op.kind == TileOpKind::Drain) {
            os << " port_kind=" << (op.port_kind == PortKind::Input ? "input" : "output");
            if (!op.port.empty()) os << " port=" << detail::quote(op.port);
        }
        // alpha is written only when it is not the default, so a diff of two programs
        // shows what actually differs rather than what is merely present.
        //
        // WRITTEN WITH max_digits10 THROUGH A CLASSIC-LOCALE STREAM. The default 6
        // significant digits do not round-trip a float: 1.0000001f writes as "1" and
        // reads back as 1.0f, and the reloaded program then computes a DIFFERENT
        // RESULT -- which breaks the bit-identical claim this whole format rests on. The
        // existing tests only used alpha=-1, which is exact, so they could not see it.
        // The classic locale matters for the same reason: an imbued locale can write
        // "-0,5", which the reader then rejects.
        if (op.kind == TileOpKind::MatMulAccum && op.alpha != 1.0f)
            os << " alpha=" << detail::exact_float(op.alpha);
        if (op.pivot_slot >= 0) os << " pivot=" << op.pivot_slot;
        if (!op.label.empty()) os << " label=" << detail::quote(op.label);
        os << "\n";
    }
    os << "END\n";
}

// ----------------------------------------------------------------------------
// Reading
// ----------------------------------------------------------------------------
namespace detail {

inline std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

inline Version parse_version(const std::string& s, const std::string& where) {
    Version v;
    char dot1 = 0, dot2 = 0;
    std::istringstream is(s);
    if (!(is >> v.major) || !(is >> dot1) || dot1 != '.' || !(is >> v.minor) ||
        !(is >> dot2) || dot2 != '.' || !(is >> v.patch))
        throw FormatError(FormatError::Cause::MalformedPreamble,
                          "l0: " + where + ": '" + s + "' is not MAJOR.MINOR.PATCH");
    return v;
}

// Split a record into its leading keyword and its keyed fields. Quoted values may
// contain spaces, which is why this is not a plain istringstream loop.
inline std::string split_fields(const std::string& line,
                                std::map<std::string, std::string>& fields) {
    std::size_t i = 0;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    const std::size_t kw_begin = i;
    while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    const std::string keyword = line.substr(kw_begin, i - kw_begin);

    while (i < line.size()) {
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (i >= line.size()) break;
        const std::size_t key_begin = i;
        while (i < line.size() && line[i] != '=' &&
               !std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        const std::string key = line.substr(key_begin, i - key_begin);
        std::string value;
        if (i < line.size() && line[i] == '=') {
            ++i;
            if (i < line.size() && line[i] == '"') {
                ++i;
                while (i < line.size() && line[i] != '"') {
                    // Decode symmetrically with quote(). Stripping the backslash and
                    // keeping the next character would turn \n into a literal 'n'.
                    if (line[i] == '\\' && i + 1 < line.size()) {
                        switch (line[i + 1]) {
                            case 'n': value += '\n'; break;
                            case 'r': value += '\r'; break;
                            case 't': value += '\t'; break;
                            default:  value += line[i + 1]; break;   // " and backslash
                        }
                        i += 2;
                        continue;
                    }
                    value += line[i++];
                }
                if (i < line.size()) ++i;         // closing quote
            } else {
                const std::size_t v_begin = i;
                while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i])))
                    ++i;
                value = line.substr(v_begin, i - v_begin);
            }
        }
        if (!key.empty()) fields[key] = value;
    }
    return keyword;
}

inline const std::string& require(const std::map<std::string, std::string>& f,
                                  const std::string& key, const std::string& where) {
    const auto it = f.find(key);
    if (it == f.end())
        throw FormatError(FormatError::Cause::MalformedRecord,
                          "l0: " + where + ": missing required field '" + key + "'");
    return it->second;
}

// BOUNDED, because std::stoul alone is not enough for a file format. It accepts a sign
// and leading whitespace, so "-1" becomes ULONG_MAX, the cast to Dim becomes UINT32_MAX,
// and TensorOperand then tries to allocate rows*cols floats -- an oversized allocation or
// a length_error thrown from OUTSIDE this function, so a caller catching FormatError would
// not catch it. A value above Dim's range would also be silently truncated by the cast.
//
// This is the second time today that a bare std::stoul has been the bug (the driver CLI
// was the first), because I wrote a fresh parser instead of reusing a checked one.
inline unsigned long long to_bounded(const std::string& s, unsigned long long max_value,
                                     const std::string& where) {
    if (s.empty() || !std::isdigit(static_cast<unsigned char>(s[0])))
        throw FormatError(FormatError::Cause::MalformedRecord,
                          "l0: " + where + ": '" + s +
                          "' is not a non-negative integer (no sign, no leading space)");
    try {
        std::size_t used = 0;
        const unsigned long long v = std::stoull(s, &used);
        if (used != s.size())
            throw FormatError(FormatError::Cause::MalformedRecord,
                              "l0: " + where + ": '" + s + "' has trailing characters");
        if (v > max_value)
            throw FormatError(FormatError::Cause::MalformedRecord,
                              "l0: " + where + ": " + s + " exceeds the maximum " +
                              std::to_string(max_value) + " this field can hold");
        return v;
    } catch (const FormatError&) {
        throw;
    } catch (const std::exception&) {
        throw FormatError(FormatError::Cause::MalformedRecord,
                          "l0: " + where + ": '" + s + "' is not representable");
    }
}

inline Dim to_dim(const std::string& s, const std::string& where) {
    return static_cast<Dim>(
        to_bounded(s, static_cast<unsigned long long>(std::numeric_limits<Dim>::max()), where));
}

// Percent-decode the operand component, symmetrically with encode_operand().
inline std::string decode_operand(const std::string& s, const std::string& where) {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '%') { out += s[i]; continue; }
        if (i + 2 >= s.size())
            throw FormatError(FormatError::Cause::MalformedRecord,
                              "l0: " + where + ": truncated percent escape in '" + s + "'");
        const auto nib = [&](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            throw FormatError(FormatError::Cause::MalformedRecord,
                              "l0: " + where + ": bad percent escape in '" + s + "'");
        };
        out += static_cast<char>((nib(s[i + 1]) << 4) | nib(s[i + 2]));
        i += 2;
    }
    return out;
}

inline TileCoord parse_coord(const std::string& s, const std::string& where) {
    const auto colon = s.find(':');
    const auto comma = s.find(',', colon == std::string::npos ? 0 : colon);
    if (colon == std::string::npos || comma == std::string::npos)
        throw FormatError(FormatError::Cause::MalformedRecord,
                          "l0: " + where + ": '" + s + "' is not operand:ti,tj");
    TileCoord c;
    c.operand = decode_operand(s.substr(0, colon), where);
    c.ti = to_dim(s.substr(colon + 1, comma - colon - 1), where);
    c.tj = to_dim(s.substr(comma + 1), where);
    return c;
}

inline std::vector<TileCoord> parse_coords(const std::string& s, const std::string& where) {
    std::vector<TileCoord> out;
    std::size_t begin = 0;
    while (begin <= s.size()) {
        const auto sep = s.find(';', begin);
        const std::string tok = s.substr(begin, sep == std::string::npos ? std::string::npos
                                                                        : sep - begin);
        if (!tok.empty()) out.push_back(parse_coord(tok, where));
        if (sep == std::string::npos) break;
        begin = sep + 1;
    }
    return out;
}

// R8: an unknown OP is refused. Executing a program containing an op this build
// does not implement would compute the wrong answer silently, which is strictly
// worse than failing to load.
inline TileOpKind parse_kind(const std::string& s) {
    for (TileOpKind k : {TileOpKind::Feed, TileOpKind::Drain, TileOpKind::MatMulAccum,
                         TileOpKind::LuDiagFactor, TileOpKind::PivotApply,
                         TileOpKind::TrsmLowerLeft, TileOpKind::TrsmUpperRight})
        if (s == to_string(k)) return k;
    throw FormatError(FormatError::Cause::UnknownOp,
                      "l0: unknown op '" + s + "': this build implements opset tile " +
                      opset_version().str() + ", so it cannot execute this program");
}

} // namespace detail

// Read a program. Throws FormatError, never a variant or bad_alloc surprise.
inline TileProgram read_l0(std::istream& is) {
    std::string line;
    // ---- preamble -----------------------------------------------------------
    if (!std::getline(is, line))
        throw FormatError(FormatError::Cause::Truncated, "l0: empty input");
    {
        std::istringstream first(detail::trim(line));
        std::string magic, ver;
        first >> magic >> ver;
        if (magic != kMagic)
            throw FormatError(FormatError::Cause::NotAnL0File,
                              "l0: expected magic '" + std::string(kMagic) + "', got '" +
                              magic + "'");
        const Version file_format = detail::parse_version(ver, "format version");
        // A MAJOR bump means the container changed shape, so an older reader must
        // not try. Minor/patch are additive by the R5 rule.
        if (reader_version().major < file_format.major)
            throw FormatError(FormatError::Cause::UnsupportedVersion,
                              "l0: file format " + file_format.str() +
                              " is newer than this reader (" + reader_version().str() + ")");
    }

    std::string name;
    bool has_values = false, saw_min_consumer = false, saw_end = false;
    TileProgram prog;
    std::vector<TileOp> ops;
    std::vector<TensorOperand> operands;

    while (std::getline(is, line)) {
        const std::string rec = detail::trim(line);
        if (rec.empty() || rec[0] == '#') continue;

        std::map<std::string, std::string> f;
        const std::string kw = detail::split_fields(rec, f);

        if (kw == "MIN_CONSUMER") {
            // R4, the requirement with teeth: the FILE says which reader it needs,
            // and an older one refuses cleanly instead of mis-parsing.
            std::istringstream v(rec.substr(kw.size()));
            std::string s;
            v >> s;
            const Version need = detail::parse_version(s, "MIN_CONSUMER");
            if (!(need <= reader_version()))
                throw FormatError(FormatError::Cause::UnsupportedVersion,
                                  "l0: file requires a reader >= " + need.str() +
                                  "; this build is " + reader_version().str());
            saw_min_consumer = true;
        } else if (kw == "PROGRAM") {
            // The name is one quoted positional token. Routed through the same keyed
            // parser so quoting and escaping behave identically everywhere -- a program
            // named `a"b` must round-trip like any other string.
            std::map<std::string, std::string> kv;
            detail::split_fields("PROGRAM name=" + detail::trim(rec.substr(kw.size())), kv);
            name = kv.count("name") ? kv["name"] : "";
        } else if (kw == "VALUES") {
            const std::string rest = detail::trim(rec.substr(kw.size()));
            has_values = (rest == "inline");
        } else if (kw == "OPERAND") {
            // The name is the first positional token, quoted; the rest are keyed.
            const std::string rest = detail::trim(rec.substr(kw.size()));
            std::map<std::string, std::string> kv;
            detail::split_fields("OPERAND name=" + rest, kv);
            const std::string op_name = kv.count("name") ? kv["name"] : "";
            if (op_name.empty())
                throw FormatError(FormatError::Cause::MalformedRecord,
                                  "l0: OPERAND: missing name");
            operands.emplace_back(op_name,
                                  detail::to_dim(detail::require(kv, "rows", "OPERAND"), "OPERAND"),
                                  detail::to_dim(detail::require(kv, "cols", "OPERAND"), "OPERAND"),
                                  detail::to_dim(detail::require(kv, "tile_rows", "OPERAND"), "OPERAND"),
                                  detail::to_dim(detail::require(kv, "tile_cols", "OPERAND"), "OPERAND"));
        } else if (kw == "OP") {
            TileOp op;
            op.kind = detail::parse_kind(detail::require(f, "kind", "OP"));
            if (f.count("in"))  op.inputs = detail::parse_coords(f["in"], "OP in");
            if (f.count("out")) op.outputs = detail::parse_coords(f["out"], "OP out");
            if (f.count("port_kind"))
                op.port_kind = (f["port_kind"] == "output") ? PortKind::Output
                                                            : PortKind::Input;
            if (f.count("port")) op.port = f["port"];
            if (f.count("alpha")) op.alpha = detail::parse_float(f["alpha"], "OP alpha");
            if (f.count("pivot"))
                op.pivot_slot = static_cast<int>(detail::to_bounded(
                    f["pivot"],
                    static_cast<unsigned long long>(std::numeric_limits<int>::max()),
                    "OP pivot"));
            if (f.count("label")) op.label = f["label"];
            // R8, the other half: an unknown OPTIONAL FIELD is ignored, so a minor
            // producer bump stays readable instead of failing on a key that carries
            // no semantics this build needs.
            ops.push_back(std::move(op));
        } else if (kw == "END") {
            saw_end = true;
            break;
        }
        // OPSET / PRODUCER and any future preamble record: recorded by the writer,
        // not required by this reader beyond the gates above.
    }

    if (!saw_min_consumer)
        throw FormatError(FormatError::Cause::MalformedPreamble,
                          "l0: missing MIN_CONSUMER; refusing rather than guessing which "
                          "reader this file was written for");
    if (!saw_end)
        throw FormatError(FormatError::Cause::Truncated,
                          "l0: input ended before END: the file is truncated, and a "
                          "partial program would execute a partial answer");

    // ---- validate against the program's own registry ------------------------
    // Without this, a malformed file does not fail here -- it fails LATER, in a worse
    // place. A duplicate OPERAND throws std::invalid_argument from add_operand(), which a
    // caller catching FormatError does not catch. An op naming an undeclared operand
    // throws only at execution. And a ti/tj beyond the operand's tile grid INDEXES
    // TensorOperand::values PAST ITS END -- undefined behaviour, not a diagnostic, from a
    // file a loader accepted.
    TileProgram out(name);
    for (TensorOperand& o : operands) {
        if (out.has_operand(o.name))
            throw FormatError(FormatError::Cause::MalformedRecord,
                              "l0: OPERAND \"" + o.name + "\" declared twice");
        try {
            out.add_operand(std::move(o));
        } catch (const std::exception& e) {
            // Includes an allocation the file asks for but the process cannot provide:
            // that is a property of the file, so it is reported as one.
            throw FormatError(FormatError::Cause::MalformedRecord,
                              std::string("l0: OPERAND rejected: ") + e.what());
        }
    }

    auto check_coord = [&out](const TileCoord& c, const std::string& where) {
        if (!out.has_operand(c.operand))
            throw FormatError(FormatError::Cause::MalformedRecord,
                              "l0: " + where + ": tile " + c.to_string() +
                              " names operand \"" + c.operand +
                              "\", which the file never declares");
        const TensorOperand& op = out.operand(c.operand);
        if (c.ti >= op.n_tile_rows() || c.tj >= op.n_tile_cols())
            throw FormatError(FormatError::Cause::MalformedRecord,
                              "l0: " + where + ": tile " + c.to_string() +
                              " is outside operand \"" + c.operand + "\"'s " +
                              std::to_string(op.n_tile_rows()) + "x" +
                              std::to_string(op.n_tile_cols()) + " tile grid");
    };
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const std::string where = "OP " + std::to_string(i) + " (" +
                                  std::string(to_string(ops[i].kind)) + ")";
        for (const TileCoord& c : ops[i].inputs)  check_coord(c, where + " in");
        for (const TileCoord& c : ops[i].outputs) check_coord(c, where + " out");
    }

    for (TileOp& o : ops) out.push(std::move(o));
    (void)has_values;          // increment 2
    return out;
}

// Convenience round-trip helpers, so callers do not each reinvent the streams.
inline std::string to_string(const TileProgram& prog) {
    std::ostringstream os;
    write_l0(os, prog);
    return os.str();
}

inline TileProgram from_string(const std::string& text) {
    std::istringstream is(text);
    return read_l0(is);
}

} // namespace sw::kpu::program::serialize
