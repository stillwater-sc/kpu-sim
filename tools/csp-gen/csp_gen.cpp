// ============================================================================
// tools/csp-gen/csp_gen.cpp
// kpu-csp-gen -- write a CSP program (docs/plans/kpu-run-csp-programs.md step 1).
//
// The simulator (kpu-run) executes CSP programs; this tool writes them: the canonical
// schedule of an operator at a size, tiling and L3, as .csp text with its loops. The output
// is an ordinary program -- read it, edit it, run it.
//
//   kpu-csp-gen --algo matmul --size 256 --tile 32 --l3 128 -o matmul.csp
//   kpu-csp-gen --algo linear --size 128 --tile 32 --target s1.json --act atan --place str.drain
//
// Exit codes: 0 written; 1 the schedule or the language refused it; 2 a usage error.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

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
  --act <relu|gelu|silu|atan>   linear: the activation               (default relu)
  --place <fabric|str.drain|bm.egress|unfused>
                                linear: where the epilogue runs      (default fabric)
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
        // No option takes a value that begins with '-': the next token being another option
        // means this one has no value (`-o --help` must not write a file named "--help").
        if (i + 1 >= argc || argv[i + 1][0] == '-') return usage(k + ": missing value");
        if (a.count(k)) return usage(k + " given twice");
        a[k] = argv[++i];
    }
    for (const auto& [k, v] : a)
        if (k != "--algo" && k != "--size" && k != "--tile" && k != "--l3" && k != "--target" && k != "--orient" &&
            k != "--act" && k != "--place" && k != "-o")
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
    if (a.count("--place")) o.place = a["--place"];
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
