#include <iostream>
#include <iomanip>
#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>
#include <sw/kpu/models/temporal/datamovement/stateful_block_mover.hpp>
#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"
#include "sw/kpu/dataflow/block_mover_compiler.hpp"

// Debug deadlock with small test

using namespace sw::kpu;
using namespace sw::kpu::dataflow;

int main() {
    TileDataFlowGraph dfg;

    // Small test: just 2x2 with k=1
    uint32_t m_tiles = 2, n_tiles = 2, k_tiles = 1;
    uint8_t mesh_cols = 4, mesh_rows = 4;
    uint32_t tile_bytes = 64;  // Small tiles for fast test

    std::vector<std::vector<size_t>> load_a(m_tiles, std::vector<size_t>(k_tiles));
    std::vector<std::vector<size_t>> load_b(k_tiles, std::vector<size_t>(n_tiles));
    std::vector<std::vector<std::vector<size_t>>> a_at_col(
        m_tiles, std::vector<std::vector<size_t>>(k_tiles, std::vector<size_t>(mesh_cols)));
    std::vector<std::vector<std::vector<size_t>>> b_at_row(
        k_tiles, std::vector<std::vector<size_t>>(n_tiles, std::vector<size_t>(mesh_rows)));

    // DMA loads for A (column 0)
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t k = 0; k < k_tiles; ++k) {
            TileDescriptor tile;
            tile.tensor = TensorId::A;
            tile.m_tile = m;
            tile.k_tile = k;
            tile.size = tile_bytes;
            uint8_t l3_id = (m % mesh_rows) * mesh_cols;
            load_a[m][k] = dfg.add_dma_load(tile, l3_id);
            a_at_col[m][k][0] = load_a[m][k];
        }
    }

    // DMA loads for B (row 0)
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            TileDescriptor tile;
            tile.tensor = TensorId::B;
            tile.n_tile = n;
            tile.k_tile = k;
            tile.size = tile_bytes;
            uint8_t l3_id = n % mesh_cols;
            load_b[k][n] = dfg.add_dma_load(tile, l3_id);
            b_at_row[k][n][0] = load_b[k][n];
        }
    }

    // A flows East (just one hop: 0->1)
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t k = 0; k < k_tiles; ++k) {
            TileDescriptor tile;
            tile.tensor = TensorId::A;
            tile.m_tile = m;
            tile.k_tile = k;
            tile.size = tile_bytes;

            uint8_t row = m % mesh_rows;
            uint8_t src_l3 = row * mesh_cols;
            uint8_t dst_l3 = row * mesh_cols + 1;
            size_t transfer = dfg.add_l3_transfer(tile, src_l3, dst_l3);
            dfg.add_edge(a_at_col[m][k][0], transfer);
            a_at_col[m][k][1] = transfer;
        }
    }

    // B flows South (just one hop: 0->4, 1->5)
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            TileDescriptor tile;
            tile.tensor = TensorId::B;
            tile.n_tile = n;
            tile.k_tile = k;
            tile.size = tile_bytes;

            uint8_t col = n % mesh_cols;
            uint8_t src_l3 = col;
            uint8_t dst_l3 = mesh_cols + col;
            size_t transfer = dfg.add_l3_transfer(tile, src_l3, dst_l3);
            dfg.add_edge(b_at_row[k][n][0], transfer);
            b_at_row[k][n][1] = transfer;
        }
    }

    std::cout << "DFG: " << dfg.num_nodes() << " nodes, " << dfg.num_edges() << " edges\n";

    // Schedule and compile
    DFGScheduler scheduler;
    DFGSchedule schedule = scheduler.schedule(dfg);
    BlockMoverCompiler compiler;
    CompiledSchedule compiled = compiler.compile(dfg, schedule);

    // Print programs
    std::cout << "\nPrograms:\n";
    for (uint8_t l3 = 0; l3 < 16; ++l3) {
        const auto& prog = compiled.program(l3);
        if (!prog.empty()) {
            std::cout << "L3[" << (int)l3 << "]: ";
            for (const auto& cmd : prog.commands) {
                std::cout << to_string(cmd.op) << " ";
            }
            std::cout << "\n";
        }
    }

    // Create and run BlockMoverArray
    BlockMoverArray::Config array_config;
    array_config.rows = mesh_rows;
    array_config.cols = mesh_cols;
    BlockMoverArray array(array_config);

    std::vector<BlockMoverProgram> programs;
    for (uint8_t l3 = 0; l3 < 16; ++l3) {
        if (!compiled.program(l3).empty()) {
            programs.push_back(compiled.program(l3));
        }
    }
    array.load_programs(programs);

    std::cout << "\nSimulating...\n";
    uint64_t cycle = 0;
    uint64_t max_cycles = 1000;

    while (!array.all_idle() && cycle < max_cycles) {
        if (cycle < 20 || cycle % 100 == 0) {
            std::cout << "Cycle " << cycle << ": ";
            for (uint8_t l3 = 0; l3 < 16; ++l3) {
                const auto& mover = array[l3];
                if (mover.state() != BlockMoverState::IDLE) {
                    std::cout << "[" << (int)l3 << "]=" << to_string(mover.state());
                    if (mover.current_command()) {
                        std::cout << "(" << to_string(mover.current_command()->op) << ")";
                    }
                    std::cout << " ";
                }
            }
            std::cout << "\n";
        }
        array.step(cycle);
        cycle++;
    }

    std::cout << "\nFinal: cycle=" << cycle << ", idle=" << array.all_idle() << "\n";

    return 0;
}
/*
   g++ -std=c++20 -Iinclude -o ./examples/test/debug_deadlock ./examples/test/debug_deadlock.cpp \
       -Lbuild/src/dataflow -Lbuild/src/isa -Lbuild/src/components/datamovement \
       -Lbuild/src/components/memory -Lbuild/src/components/compute -Lbuild/src/trace \
       -Lbuild/src/noc -Lbuild/_deps/fmt-build -Lbuild/_deps/spdlog-build \
       -lkpu_dataflow -lkpu_isa -lkpu_datamovement_components -lkpu_memory_components \
       -lkpu_compute_components -lkpu_trace -lkpu_noc -lfmt -lspdlog -lpthread 2>&1 && \
   LD_LIBRARY_PATH=build/src/dataflow:build/src/isa:build/src/components/datamovement:build/src/components/memory:build/src/components/compute:build/src/trace:build/src/noc:build/_deps/fmt-build:build/_deps/spdlog-build ./examples/test/debug_deadlock 2>&1

*/

