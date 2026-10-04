// ============================================================================
// tools/floorplan/kpu_floorplan.cpp
// kpu-floorplan: the physical shape of a deployment, as a summary, a JSON floorplan and an
// SVG (#286 step 1; docs/plans/tile-flow-debugger.md §3.1).
//
//   kpu-floorplan --deploy <spec.json> [--device <name>] [--json <out>] [--svg <out>]
//   kpu-floorplan --deploy <spec.json> --import <floorplan.json>     validate a floorplan
//
// Exit codes: 0 done; 1 the deployment has no floorplan, or an imported one does not fit
// it (the reason is printed); 2 usage error.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/platform/floorplan.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using namespace sw::kpu::program::platform;
using sw::kpu::program::Dim;

namespace {

void usage(std::ostream& o) {
    o << "usage: kpu-floorplan --deploy <spec.json> [--device <name>] [--json <out.json>]\n"
         "                     [--svg <out.svg>] [--import <floorplan.json>]\n"
         "\n"
         "  --deploy <file>   the deployment spec to lay out (required)\n"
         "  --device <name>   which device of a multi-device spec (default: the first)\n"
         "  --json <file>     write the floorplan as JSON (the format --import reads)\n"
         "  --svg <file>      write the floorplan as SVG\n"
         "  --import <file>   read a floorplan and validate it against the deployment,\n"
         "                    instead of generating one\n";
}

bool write_file(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
    return static_cast<bool>(out);
}

} // namespace

int main(int argc, char** argv) {
    std::string deploy, device, json_out, svg_out, import;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](std::string& into) {
            if (i + 1 >= argc) {
                std::cerr << "kpu-floorplan: " << a << " needs a value\n";
                return false;
            }
            into = argv[++i];
            return true;
        };
        if (a == "--help" || a == "-h") { usage(std::cout); return 0; }
        else if (a == "--deploy") { if (!value(deploy)) return 2; }
        else if (a == "--device") { if (!value(device)) return 2; }
        else if (a == "--json") { if (!value(json_out)) return 2; }
        else if (a == "--svg") { if (!value(svg_out)) return 2; }
        else if (a == "--import") { if (!value(import)) return 2; }
        else {
            std::cerr << "kpu-floorplan: unknown argument '" << a << "'\n";
            usage(std::cerr);
            return 2;
        }
    }
    if (deploy.empty()) {
        std::cerr << "kpu-floorplan: --deploy is required\n";
        usage(std::cerr);
        return 2;
    }

    DeploymentSpec spec;
    try {
        spec = read_spec_file(deploy);
    } catch (const std::exception& e) {
        std::cerr << "kpu-floorplan: " << e.what() << "\n";
        return 2;
    }
    Dim di = 0;
    if (!device.empty()) {
        const std::size_t i = spec.index_of(device);
        if (i == spec.devices.size()) {
            std::cerr << "kpu-floorplan: no device named '" << device << "' in " << deploy << "\n";
            return 2;
        }
        di = static_cast<Dim>(i);
    }

    SocFloorplan fp;
    try {
        if (!import.empty()) {
            std::ifstream in(import, std::ios::binary);
            if (!in) {
                std::cerr << "kpu-floorplan: cannot read '" << import << "'\n";
                return 2;
            }
            std::ostringstream ss;
            ss << in.rdbuf();
            fp = floorplan_from_json(ss.str(), spec);
        } else {
            fp = generate_floorplan(spec, di);
        }
    } catch (const FloorplanError& e) {
        std::cerr << "kpu-floorplan: " << e.what() << "\n";
        return 1;
    }

    // ---- the summary
    const DeviceSpecification& d = spec.device(static_cast<Dim>(spec.index_of(fp.device)));
    std::map<BlockKind, std::size_t> kinds;
    std::function<void(const std::vector<FloorplanBlock>&)> walk = [&](const auto& bs) {
        for (const FloorplanBlock& b : bs) {
            ++kinds[b.kind];
            walk(b.children);
        }
    };
    walk(fp.blocks);
    std::cout << "kpu-floorplan: " << fp.device << " (" << fp.source << ")\n";
    if (const auto L = ArrayLayout::of(d)) {
        std::cout << "  array        " << L->rows() << "x" << L->cols() << ": " << L->l3_count()
                  << " L3 tiles, " << L->cf_count() << " compute tiles, " << L->block_movers().size()
                  << " BlockMovers\n";
        if (L->has_noc()) {
            Dim rows = 0, cols = 0;
            for (const NocLoop& l : L->loops()) (l.axis == NocLoop::Axis::Row ? rows : cols)++;
            std::cout << "  noc          folded 2D torus: " << rows << " row loops x " << cols
                      << " column loops, " << L->links().size() << " wires, " << L->ports().size()
                      << " fold-end ports\n";
        } else {
            std::cout << "  noc          none: " << L->noc_reason() << "\n";
        }
    }
    std::cout << "  blocks       " << fp.block_count() << ":";
    for (const auto& [k, n] : kinds) std::cout << " " << to_string(k) << "=" << n;
    std::cout << "\n";
    std::size_t attached = 0;
    for (const NocLink& l : fp.noc) attached += l.kind == NocLink::Kind::Attach;
    std::cout << "  dma ports    " << attached << " DMA attachment(s) (first pass)\n";
    std::cout << "  die          " << fp.die.w_um << " x " << fp.die.h_um << " um (illustrative)\n";
    std::cout << "  digest       " << fp.digest() << "\n";
    for (const std::string& n : layout_notes(d)) std::cout << "  note         " << n << "\n";

    if (!json_out.empty()) {
        if (!write_file(json_out, floorplan_to_json(fp))) {
            std::cerr << "kpu-floorplan: cannot write '" << json_out << "'\n";
            return 2;
        }
        std::cout << "  wrote        " << json_out << "\n";
    }
    if (!svg_out.empty()) {
        if (!write_file(svg_out, floorplan_to_svg(fp))) {
            std::cerr << "kpu-floorplan: cannot write '" << svg_out << "'\n";
            return 2;
        }
        std::cout << "  wrote        " << svg_out << "\n";
    }
    return 0;
}
