// ============================================================================
// tools/csp-gen/csp_gen.cpp
// kpu-csp-gen -- write a CSP program (docs/plans/kpu-run-csp-programs.md step 1).
//
// The simulator (kpu-run) executes CSP programs; this tool writes them: the canonical
// schedule of an operator at a size, tiling and L3, as .csp text with its loops. The output
// is an ordinary program -- read it, edit it, run it.
//
//   kpu-csp-gen --algo matmul --size 256 --tile 32 --l3 128 -o matmul.csp
//   kpu-csp-gen --algo linear --size 128 --tile 32 --target s1.json --act atan --place combiner
//
// Exit codes: 0 written; 1 the schedule or the language refused it; 2 a usage error.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cctype>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>

namespace {

namespace gen = sw::kpu::program::csp::gen;
namespace lang = sw::kpu::program::csp::lang;
namespace platform = sw::kpu::program::platform;

const char* kUsage =
R"(kpu-csp-gen -- write a CSP program: an operator's canonical schedule, as .csp text.

  --algo <matmul|linear|lu>     the operator                         (required)
  --size <N>                    square: M = N = K = N (N for LU)     (required)
  --tile <T>                    T x T tiles                          (required)
  --l3 <tiles>                  the machine's L3 capacity, in tiles
  --target <spec.json>          a deployment: its L3 (device 0), and its vector units, which
                                a linear epilogue's placement is checked against
                                (one of --l3, --target is required; --l3 overrides the spec's)
  --orient <col|row>            matmul, linear: which panel is outer   (default col)
  --stage-a                     matmul: stage A's row panel in L2 for the loop over B's
                                panels (the row schedule; declares l2 = tiles per panel)
  --init <none|bias|bias-row|accumulate>
                                matmul: where each accumulator starts, streamed into the
                                fabric before its first call: bias C = A.B + D, bias-row
                                C = A.B + b (b broadcast down the rows), accumulate C += A.B
                                (default none: from zero)
  --dtype <fp32|int8>           matmul (and a requantized linear): A's and B's number system;
                                C (and D or b) is in its registered accumulator's: int8 -> int32
                                (default fp32)
  --l2-buffers <1|2>            the L2 slots the program declares (l2 = N): enough for this many
                                rounds of what one call holds at once, its operands' copies and its
                                result's slot, on top of what its blocks hold. 1 is single
                                buffering, 2 double                    (default 2)
  --dataflow <os|ws|is>         matmul, linear: the stationary operand (default os). os holds C in
                                the fabric; ws holds B's tile, staged in L2 across the i loop;
                                is holds A's, across the j loop. ws and is keep C's partials
                                in L2 slots, merged by the Combiner (they declare
                                l2 = tiles per panel + 1, take no --orient or --stage-a, and of
                                --init take bias and bias-row, which the Combiner adds on each
                                tile's first merge). A linear epilogue under ws or is runs @ combiner on
                                each tile's final merge, @ bm.egress (ws), or unfused
  --act <relu|gelu|silu|atan>   linear: the activation               (default relu)
  --place <fabric|combiner|bm.egress|unfused>
                                linear: where the epilogue runs      (default fabric)
  --requant <m,shift>           matmul or linear with an integer --dtype: requantize the result
                                into the dtype on the Combiner, round(x * m / 2^shift), ties to
                                even, saturated; C is declared `<dtype> ... out acc <accumulator>`.
                                A linear's epilogue is then add(b[j]), scale, relu, all @ combiner
                                (--place combiner): b is in the accumulator's type. A matmul's
                                --init bias or bias-row (int32) is added before the scale, for
                                every dataflow; --init accumulate is refused
  --requant-channels            with --requant: per channel, each column's multiplier and shift
                                from the int32 vectors qm and qs, scale(qm[j], qs[j]); their
                                values come with the inputs (--inputs), shifts in 0..62
  --format <none|mxint8|mxfp8>  linear: convert the result into an OCP Microscaling block format,
                                after the activation, at --place (combiner | bm.egress)
                                (default none)
  -o <file>                     write here                           (default: stdout)

The program is validated before it is written; a schedule that does not fit the L3, or a
stage the target cannot run, is refused with the reason.
)";

int usage(const std::string& why) {
    std::cerr << "kpu-csp-gen: " << why << "\n\n" << kUsage;
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--help" || k == "-h") {
            std::cout << kUsage;
            return 0;
        }
        if (k.rfind("-", 0) != 0) return usage("unexpected argument '" + k + "'");
        if (k == "--stage-a" || k == "--requant-channels") {   // the options without a value
            if (a.count(k)) return usage(k + " given twice");
            a[k] = "";
            continue;
        }
        // No option takes a value that begins with '-': the next token being another option
        // means this one has no value (`-o --help` must not write a file named "--help").
        // --requant's multiplier may be negative: `--requant -2,1`.
        const bool signed_value = k == "--requant" && i + 1 < argc && argv[i + 1][0] == '-' &&
                                  std::isdigit(static_cast<unsigned char>(argv[i + 1][1]));
        if (i + 1 >= argc || (argv[i + 1][0] == '-' && !signed_value)) return usage(k + ": missing value");
        if (a.count(k)) return usage(k + " given twice");
        a[k] = argv[++i];
    }
    for (const auto& [k, v] : a)
        if (k != "--algo" && k != "--size" && k != "--tile" && k != "--l3" && k != "--target" && k != "--orient" &&
            k != "--act" && k != "--place" && k != "--stage-a" && k != "--init" && k != "--dtype" &&
            k != "--dataflow" && k != "--format" && k != "--requant" && k != "--requant-channels" && k != "--l2-buffers" && k != "-o")
            return usage("unknown option '" + k + "'");
    for (const char* k : {"--algo", "--size", "--tile"})
        if (!a.count(k)) return usage(std::string(k) + " is required");
    if (!a.count("--l3") && !a.count("--target")) return usage("--l3 or --target is required");

    gen::Options o;
    o.algo = a["--algo"];
    auto number = [&](const char* k, std::uint64_t& out) {
        const std::string& s = a[k];
        if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos || s.size() > 9) return false;
        out = std::stoull(s);
        return out > 0;
    };
    std::uint64_t size = 0, tile = 0, l3 = 0;
    if (!number("--size", size)) return usage("--size: a positive integer");
    if (!number("--tile", tile)) return usage("--tile: a positive integer");
    o.size = static_cast<sw::kpu::program::Dim>(size);
    o.tile = static_cast<sw::kpu::program::Dim>(tile);
    if (a.count("--orient")) o.orient = a["--orient"];
    if (a.count("--stage-a")) {
        if (o.algo != "matmul") return usage("--stage-a is matmul's; --algo is '" + o.algo + "'");
        o.stage_a = true;
        if (!a.count("--orient")) o.orient = "row";   // A's panel outer: the schedule staging needs
    }
    if (a.count("--place")) o.place = a["--place"] == "str.drain" ? "combiner" : a["--place"];   // str.drain: deprecated (#429)
    if (a.count("--init")) {
        if (o.algo != "matmul") return usage("--init is matmul's; --algo is '" + o.algo + "'");
        o.init = a["--init"];
    }
    if (a.count("--dtype")) {
        if (o.algo != "matmul" && o.algo != "linear") return usage("--dtype is matmul's and linear's; --algo is '" + o.algo + "'");
        o.dtype = a["--dtype"];
    }
    if (a.count("--requant")) {
        if (o.algo != "matmul" && o.algo != "linear") return usage("--requant is matmul's and linear's; --algo is '" + o.algo + "'");
        const std::string& v = a["--requant"];
        const auto comma = v.find(',');
        try {
            if (comma == std::string::npos) throw std::invalid_argument("no comma");
            std::size_t used = 0;
            o.requant_m = std::stoll(v.substr(0, comma), &used);
            if (used != comma) throw std::invalid_argument("m");
            const std::string sh = v.substr(comma + 1);
            o.requant_shift = std::stoi(sh, &used);
            if (used != sh.size()) throw std::invalid_argument("shift");
        } catch (const std::exception&) {
            return usage("--requant '" + v + "': m,shift (integers)");
        }
        o.requant = true;
    }
    if (a.count("--requant-channels")) o.requant_channels = true;
    if (a.count("--format")) {
        if (o.algo != "linear") return usage("--format is linear's; --algo is '" + o.algo + "'");
        o.format = a["--format"];
    }
    if (a.count("--l2-buffers")) {
        const std::string v = a["--l2-buffers"];
        if (v != "1" && v != "2") return usage("--l2-buffers is 1 (single buffering) or 2 (double)");
        o.l2_buffers = std::stoi(v);
    }
    if (a.count("--dataflow")) {
        if (o.algo != "matmul" && o.algo != "linear")
            return usage("--dataflow is matmul's and linear's; --algo is '" + o.algo + "'");
        o.dataflow = a["--dataflow"];
    }
    if (a.count("--act") && !sw::kpu::program::parse_activation(a["--act"], o.act))
        return usage("--act '" + a["--act"] + "' (relu | gelu | silu | atan)");
    if (o.algo != "linear" && (a.count("--act") || a.count("--place")))
        return usage("--act and --place are the linear operator's; --algo is '" + o.algo + "'");
    if (o.algo == "lu" && a.count("--orient")) return usage("--orient is matmul's and linear's");

    std::optional<lang::Target> target;
    try {
        if (a.count("--target")) {
            const platform::DeploymentSpec spec = platform::read_spec_file(a["--target"]);
            if (spec.devices.size() != 1)
                return usage("--target: the deployment has " + std::to_string(spec.devices.size()) +
                             " devices; a level-1 program targets one");
            const auto& d = spec.devices.front();
            o.l3 = d.l3.capacity_tiles;
            if (o.l3 == 0 && !a.count("--l3"))
                return usage("--target: the deployment's L3 is unbounded (capacity_tiles 0); give --l3");
            target = lang::target_from(d);
        }
        if (a.count("--l3")) {
            if (!number("--l3", l3)) return usage("--l3: a positive integer");
            o.l3 = static_cast<std::size_t>(l3);
        }
    } catch (const std::exception& e) {
        return usage(e.what());
    }

    std::string text;
    try {
        text = gen::generate(o, target ? &*target : nullptr);
    } catch (const gen::GenError& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "kpu-csp-gen: the generated program was refused: " << e.what() << "\n";
        return 1;
    }
    if (!a.count("-o")) {
        std::cout << text;
        return 0;
    }
    std::ofstream f(a["-o"]);
    if (!(f << text)) {
        std::cerr << "kpu-csp-gen: cannot write '" << a["-o"] << "'\n";
        return 2;
    }
    std::cerr << "kpu-csp-gen: wrote " << a["-o"] << "\n";
    return 0;
}
