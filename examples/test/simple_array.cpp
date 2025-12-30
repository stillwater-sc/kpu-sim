#include <iostream>
#include <iomanip>
#include "sw/kpu/components/block_mover_isa.hpp"
#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"
#include "sw/kpu/dataflow/block_mover_compiler.hpp"

/*
      g++ -std=c++20 -Iinclude -o /tmp/debug_programs2 /tmp/debug_programs2.cpp \
          -Lbuild/src/dataflow -Lbuild/src/isa -Lbuild/src/components/datamovement \
          -Lbuild/src/components/memory -Lbuild/src/components/compute -Lbuild/src/trace \
          -Lbuild/src/noc -Lbuild/_deps/fmt-build -Lbuild/_deps/spdlog-build \
          -lkpu_dataflow -lkpu_isa -lkpu_datamovement_components -lkpu_memory_components \
          -lkpu_compute_components -lkpu_trace -lkpu_noc -lfmt -lspdlog -lpthread 2>&1 && \
      LD_LIBRARY_PATH=build/src/dataflow:build/src/isa:build/src/components/datamovement:build/src/components/memory:build/src/components/compute:build/src/trace:build/src/noc:build/_deps/fmt-build:build/_deps/spdlog-build /tmp/debug_programs2 2>&1)
 */
 
using namespace sw::kpu;
using namespace sw::kpu::dataflow;

int main() {
    TileDataFlowGraph dfg;

    uint32_t m_tiles = 4, n_tiles = 4, k_tiles = 4;
    uint8_t mesh_cols = 4, mesh_rows = 4;
    uint32_t tile_bytes = 256 * 256 * 4;

    // Track nodes
    std::vector<std::vector<size_t>> load_a(m_tiles, std::vector<size_t>(k_tiles));
    std::vector<std::vector<size_t>> load_b(k_tiles, std::vector<size_t>(n_tiles));
    std::vector<std::vector<std::vector<size_t>>> a_at_col(
        m_tiles, std::vector<std::vector<size_t>>(k_tiles, std::vector<size_t>(mesh_cols)));
    std::vector<std::vector<std::vector<size_t>>> b_at_row(
        k_tiles, std::vector<std::vector<size_t>>(n_tiles, std::vector<size_t>(mesh_rows)));

    // DMA loads for A tiles (column 0)
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

    // DMA loads for B tiles (row 0)
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

    // A flows East
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t k = 0; k < k_tiles; ++k) {
            TileDescriptor tile;
            tile.tensor = TensorId::A;
            tile.m_tile = m;
            tile.k_tile = k;
            tile.size = tile_bytes;

            uint8_t row = m % mesh_rows;
            for (uint8_t col = 1; col < mesh_cols; ++col) {
                uint8_t src_l3 = row * mesh_cols + (col - 1);
                uint8_t dst_l3 = row * mesh_cols + col;
                size_t transfer = dfg.add_l3_transfer(tile, src_l3, dst_l3);
                dfg.add_edge(a_at_col[m][k][col - 1], transfer);
                a_at_col[m][k][col] = transfer;
            }
        }
    }

    // B flows South
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            TileDescriptor tile;
            tile.tensor = TensorId::B;
            tile.n_tile = n;
            tile.k_tile = k;
            tile.size = tile_bytes;

            uint8_t col = n % mesh_cols;
            for (uint8_t row = 1; row < mesh_rows; ++row) {
                uint8_t src_l3 = (row - 1) * mesh_cols + col;
                uint8_t dst_l3 = row * mesh_cols + col;
                size_t transfer = dfg.add_l3_transfer(tile, src_l3, dst_l3);
                dfg.add_edge(b_at_row[k][n][row - 1], transfer);
                b_at_row[k][n][row] = transfer;
            }
        }
    }

    // Schedule and compile
    DFGScheduler scheduler;
    DFGSchedule schedule = scheduler.schedule(dfg);
    BlockMoverCompiler compiler;
    CompiledSchedule compiled = compiler.compile(dfg, schedule);

    // Print first 5 commands of each L3 program
    std::cout << "\nFirst 5 commands per L3:\n";
    for (uint8_t l3 = 0; l3 < 16; ++l3) {
        const auto& prog = compiled.program(l3);
        if (!prog.empty()) {
            std::cout << "L3[" << std::setw(2) << (int)l3 << "]: ";
            int count = 0;
            for (const auto& cmd : prog.commands) {
                if (count++ < 5) {
                    std::cout << to_string(cmd.op);
                    if (count < 5) std::cout << ", ";
                }
            }
            std::cout << "\n";
        }
    }

    return 0;
}

