// ============================================================================
// tools/csp-check/kpu_csp_check.cpp
// kpu-csp-check -- check a CSP program against the machine it is written for, before any simulation
// (#607; docs/plans/distributed-csp-programs.md §3.3, §4 step 4).
//
//   kpu-csp-check --distributed PROGRAM.dcsp --deploy SPEC.json [--device NAME] [--run]
//
// Reads a level-2 program (dcsp 1), checks its structure and D0-D4 against the deployment's mesh
// (csp/distributed_check.hpp), and with --run also runs it on L-B (distributed_behavioral.hpp) from
// deterministic inputs, which is D5: a deadlock names every stuck process.
//
// Exit codes: 0 every check passed; 1 a violation (each printed with its rule); 2 the program or
// the deployment could not be read.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/distributed_behavioral.hpp>
#include <sw/kpu/program/csp/distributed_check.hpp>
#include <sw/kpu/program/csp/lang/distributed_text.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using namespace sw::kpu::program;

namespace {

void usage(std::ostream& o) {
    o << "usage: kpu-csp-check --distributed PROGRAM.dcsp --deploy SPEC.json [--device NAME] [--run]\n"
         "  Checks a level-2 program's structure and D0-D4 against the deployment's mesh;\n"
         "  --run also runs it on L-B from deterministic inputs (D5, liveness).\n"
         "  Exit: 0 passed, 1 a violation, 2 unreadable input.\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string program, deploy, device;
    bool distributed = false, run = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](std::string& into) {
            if (i + 1 >= argc) {
                std::cerr << "kpu-csp-check: " << a << " needs a value\n";
                return false;
            }
            into = argv[++i];
            return true;
        };
        if (a == "--help" || a == "-h") { usage(std::cout); return 0; }
        else if (a == "--distributed") { distributed = true; if (!value(program)) return 2; }
        else if (a == "--deploy") { if (!value(deploy)) return 2; }
        else if (a == "--device") { if (!value(device)) return 2; }
        else if (a == "--run") run = true;
        else {
            std::cerr << "kpu-csp-check: unknown argument '" << a << "'\n";
            usage(std::cerr);
            return 2;
        }
    }
    if (!distributed || deploy.empty()) {
        std::cerr << "kpu-csp-check: --distributed PROGRAM and --deploy SPEC are required\n";
        usage(std::cerr);
        return 2;
    }

    csp::DistributedProgram p;
    platform::DeploymentSpec spec;
    try {
        std::ifstream in(program, std::ios::binary);
        if (!in) {
            std::cerr << "kpu-csp-check: cannot read '" << program << "'\n";
            return 2;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        p = csp::lang::parse_distributed(ss.str());
        spec = platform::read_spec_file(deploy);
    } catch (const std::exception& e) {
        std::cerr << "kpu-csp-check: " << e.what() << "\n";
        return 2;
    }
    std::size_t di = 0;
    if (!device.empty()) {
        di = spec.index_of(device);
        if (di == spec.devices.size()) {
            std::cerr << "kpu-csp-check: no device named '" << device << "' in " << deploy << "\n";
            return 2;
        }
    }
    const platform::DeviceSpecification& d = spec.device(static_cast<Dim>(di));

    std::cout << "kpu-csp-check: " << p.name << " on " << d.name << " (" << p.rows << "x" << p.cols << " mesh, "
              << p.processes.size() << " processes, " << p.channels.size() << " channels, " << p.actions.size()
              << " actions)\n";
    const auto v = csp::check_distributed(p, d);
    for (const auto& x : v) std::cout << "  " << x.rule << "  " << x.message << "\n";
    if (!v.empty()) {
        std::cout << "kpu-csp-check: " << v.size() << " violation(s)\n";
        return 1;
    }
    std::cout << "  D0-D4 pass\n";
    if (run) {
        try {
            csp::DistributedBehavioral lb;
            const auto sum = lb.run(p, csp::distributed_inputs(p));
            std::cout << "  D5 pass: L-B ran " << sum.actions << " actions (" << sum.calls << " calls) in " << sum.rounds
                      << " rounds, at most " << sum.peak_live << " slots live\n";
        } catch (const csp::DistributedDeadlock& e) {
            std::cout << "  D5  " << e.what() << "\n";
            return 1;
        } catch (const std::exception& e) {
            std::cout << "  D5  " << e.what() << "\n";
            return 1;
        }
    }
    return 0;
}
