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

#include <cmath>
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
        return N == 32 ? w : (w & ((Word{1} << N) - 1));
    }
    static double decode(Word w) { return static_cast<double>(static_cast<long long>(from_word(w))); }
    static Word encode(double v) {
        // The integer part, modulo 2^64 into the type's constructor, which keeps the low N bits:
        // two's complement, as a hardware register of N bits holds it.
        const long long i = static_cast<long long>(std::trunc(v));
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

}  // namespace sw::kpu::program::numeric
