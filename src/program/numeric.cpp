// ============================================================================
// src/program/numeric.cpp
// The number-system registry and the typed MACs, on stillwater-sc/universal
// (include/sw/kpu/program/numeric.hpp; docs/plans/typed-value-path.md).
//
// The one translation unit that includes Universal: the templates are instantiated here, once,
// and every level calls them through numeric.hpp's function pointers -- so L-B, the L0 replay,
// L-T1 and L-CA compute the same bits for every number system.
//
// Adding a number system: a Codec for its Universal type (word <-> the type) and a row in
// kSystems. Adding a MAC pair: a row in kMacs.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/numeric.hpp>

#include <universal/number/integer/integer.hpp>
#include <universal/number/mxfloat/mxfloat.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sw::kpu::program::numeric {
namespace {

namespace su = sw::universal;

// A Universal type <-> its word. An integer<N>'s bits are its blocks, least significant first.
template <class T>
struct Codec;

template <unsigned N, class Bt>
struct Codec<su::integer<N, Bt>> {
    using T = su::integer<N, Bt>;
    static_assert(N <= 32, "a word holds 32 bits (docs/plans/typed-value-path.md §3)");
    static T from_word(Word w) {
        T t;
        t.setbits(static_cast<std::uint64_t>(w));
        return t;
    }
    static Word to_word(const T& t) {
        constexpr unsigned bb = sizeof(Bt) * 8;
        Word w = 0;
        for (unsigned i = 0; i * bb < N; ++i) w |= static_cast<Word>(t.block(i)) << (i * bb);
        if constexpr (N < 32) w &= (Word{1} << N) - 1;   // constexpr: no shift by 32 to instantiate
        return w;
    }
    static double decode(Word w) { return static_cast<double>(static_cast<long long>(from_word(w))); }
    static Word encode(double v) {
        // The integer part modulo 2^N, two's complement, as a hardware register of N bits holds
        // it. fmod first, exactly, so the cast to long long is in range for any finite value.
        if (!std::isfinite(v))
            throw std::invalid_argument("numeric: " + std::to_string(v) + " has no integer<" + std::to_string(N) +
                                        "> encoding");
        const long long i = static_cast<long long>(std::fmod(std::trunc(v), std::ldexp(1.0, static_cast<int>(N))));
        return to_word(T(i));
    }
};

// fp32: the word is the float. Native arithmetic, so an fp32 program's bits are unchanged.
struct Fp32 {
    static double decode(Word w) { return static_cast<double>(float_of(w)); }
    static Word encode(double v) { return word_of(static_cast<float>(v)); }
};

using int8_t_u = su::integer<8, std::uint8_t>;
using int32_t_u = su::integer<32, std::uint32_t>;

struct System {
    NumberSystem info;
    double (*decode)(Word);
    Word (*encode)(double);
};

const std::vector<System>& table() {
    static const std::vector<System> t = {
        {{"fp32", 32, Kind::Real, "float (IEEE 754 binary32)"}, &Fp32::decode, &Fp32::encode},
        {{"int8", 8, Kind::Integer, "integer<8>"}, &Codec<int8_t_u>::decode, &Codec<int8_t_u>::encode},
        {{"int32", 32, Kind::Integer, "integer<32>"}, &Codec<int32_t_u>::decode, &Codec<int32_t_u>::encode},
    };
    return t;
}

const System& system(const std::string& name) {
    for (const System& s : table())
        if (s.info.name == name) return s;
    throw std::invalid_argument("numeric: '" + name + "' is not a registered number system (" + names() + ")");
}

// c += a . b, the product and the sum in Acc; In is converted to Acc first (int8 -> int32).
template <class InC, class AccC>
void mac_typed(const Word* a, const Word* b, Word* c, std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    using Acc = decltype(AccC::from_word(0));
    std::vector<Acc> bk(static_cast<std::size_t>(k) * n);
    for (std::size_t i = 0; i < bk.size(); ++i) bk[i] = Acc(static_cast<long long>(InC::from_word(b[i])));
    for (std::uint32_t i = 0; i < m; ++i)
        for (std::uint32_t j = 0; j < n; ++j) {
            Acc acc = AccC::from_word(c[static_cast<std::size_t>(i) * n + j]);
            for (std::uint32_t l = 0; l < k; ++l) {
                const Acc av(static_cast<long long>(InC::from_word(a[static_cast<std::size_t>(i) * k + l])));
                acc += av * bk[static_cast<std::size_t>(l) * n + j];
            }
            c[static_cast<std::size_t>(i) * n + j] = AccC::to_word(acc);
        }
}

// fp32: the reference kernel's order (tile_kernels.hpp): the k sum in a local float, then added
// onto c -- so a typed call on fp32 operands computes the bits the untyped kernel does.
void mac_fp32(const Word* a, const Word* b, Word* c, std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    for (std::uint32_t i = 0; i < m; ++i)
        for (std::uint32_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (std::uint32_t l = 0; l < k; ++l)
                acc += float_of(a[static_cast<std::size_t>(i) * k + l]) * float_of(b[static_cast<std::size_t>(l) * n + j]);
            Word& w = c[static_cast<std::size_t>(i) * n + j];
            w = word_of(float_of(w) + acc);
        }
}

struct MacRow { const char* in; const char* acc; Mac fn; };

const std::vector<MacRow>& macs() {
    static const std::vector<MacRow> t = {
        {"fp32", "fp32", &mac_fp32},
        {"int8", "int32", &mac_typed<Codec<int8_t_u>, Codec<int32_t_u>>},
    };
    return t;
}

}  // namespace

const std::vector<NumberSystem>& systems() {
    static const std::vector<NumberSystem> s = [] {
        std::vector<NumberSystem> v;
        for (const System& x : table()) v.push_back(x.info);
        return v;
    }();
    return s;
}

const NumberSystem* find(const std::string& name) {
    for (const NumberSystem& s : systems())
        if (s.name == name) return &s;
    return nullptr;
}

const NumberSystem& get(const std::string& name) { return system(name).info; }

std::string names() {
    std::string out;
    for (const NumberSystem& s : systems()) out += (out.empty() ? "" : ", ") + s.name;
    return out;
}

double decode(const std::string& type, Word w) { return system(type).decode(w); }
Word encode(const std::string& type, double v) { return system(type).encode(v); }

Mac mac(const std::string& in, const std::string& acc) {
    for (const MacRow& r : macs())
        if (in == r.in && acc == r.acc) return r.fn;
    return nullptr;
}

std::vector<std::pair<std::string, std::string>> mac_pairs() {
    std::vector<std::pair<std::string, std::string>> out;
    for (const MacRow& r : macs()) out.emplace_back(r.in, r.acc);
    return out;
}

std::string mac_names() {
    std::string out;
    for (const MacRow& r : macs()) out += (out.empty() ? "" : ", ") + std::string(r.in) + " -> " + r.acc;
    return out;
}

// ---- the OCP Microscaling block formats (#427) ----
namespace {

// The OCP MX v1.0 conversion of one format, block by block. A block's shared scale is the e8m0
// power of two X = 2^(floor(log2 amax) - emax), clamped to e8m0's [2^-127, 2^127], where emax is
// the element type's largest exponent; each element is x / X in the element type, read back times
// X. floor(log2) is ilogb, exact for every finite value (a rounded log2 is not near powers of
// two). The scaling is ldexp, which never forms 1 / X: Universal v5.17.0's mxblock::quantize does,
// and overflows to inf below X = 2^-127 (stillwater-sc/universal#2083). A non-finite amax makes
// the scale NaN, and the block reads back as NaN, as Universal's blocks do.
template <class Element>
void convert_blocks(float* v, std::size_t n) {
    for (std::size_t at = 0; at < n; at += kBlockSize) {
        const std::size_t len = std::min(kBlockSize, n - at);
        float amax = 0.0f;
        for (std::size_t i = 0; i < len; ++i) amax = std::max(amax, std::fabs(v[at + i]));
        if (!std::isfinite(amax)) {
            for (std::size_t i = 0; i < len; ++i) v[at + i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        if (amax == 0.0f) {
            for (std::size_t i = 0; i < len; ++i) v[at + i] = 0.0f;
            continue;
        }
        const int x = std::clamp(std::ilogb(amax) - Element::emax, -127, 127);   // the e8m0 scale
        for (std::size_t i = 0; i < len; ++i) v[at + i] = std::ldexp(Element::round(std::ldexp(v[at + i], -x)), x);
    }
}

// MXFP8's element: Universal's e4m3, saturating (emax 8). Its conversion from float rounds to
// nearest even and clamps to +-448.
struct Fp8Element {
    static constexpr int emax = 8;
    static float round(float s) { return static_cast<float>(su::e4m3_saturating(s)); }
};

// MXINT8's element, per OCP MX v1.0 (until Universal's mxint8 is fixed, universal#2083: v5.17.0
// saturates each block's largest element and rounds ties away from zero): an int8 code with an
// implicit scale of 2^-6, so its largest exponent is 0 and its values are code / 64 in
// [-2, 127/64]. The code is rounded to nearest, ties to even (the default rounding mode), and
// clamped to [-128, 127].
struct Int8Element {
    static constexpr int emax = 0;
    static float round(float s) {
        return std::ldexp(std::clamp(std::nearbyint(std::ldexp(s, 6)), -128.0f, 127.0f), -6);
    }
};

struct BlockFormat {
    const char* name;
    void (*convert)(float*, std::size_t);
    std::size_t element_bytes;
};

const std::vector<BlockFormat>& block_formats() {
    static const std::vector<BlockFormat> f = {
        {"mxint8", &convert_blocks<Int8Element>, 1},     // Universal's mxint8, once universal#2083 is fixed
        {"mxfp8", &convert_blocks<Fp8Element>, 1},
    };
    return f;
}

const BlockFormat& block_format(const std::string& name) {
    for (const BlockFormat& f : block_formats())
        if (name == f.name) return f;
    throw std::invalid_argument("numeric: '" + name + "' is not a block format (" + block_format_names() + ")");
}

}  // namespace

bool is_block_format(const std::string& name) {
    for (const BlockFormat& f : block_formats())
        if (name == f.name) return true;
    return false;
}

std::string block_format_names() {
    std::string s;
    for (const BlockFormat& f : block_formats()) s += (s.empty() ? "" : ", ") + std::string(f.name);
    return s;
}

void block_convert(const std::string& format, float* v, std::size_t n) { block_format(format).convert(v, n); }

std::size_t block_bytes(const std::string& format, std::size_t n) {
    return n * block_format(format).element_bytes + (n + kBlockSize - 1) / kBlockSize;
}

}  // namespace sw::kpu::program::numeric
