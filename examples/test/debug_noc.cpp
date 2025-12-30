#include <iostream>
#include <iomanip>
#include "sw/kpu/components/block_mover_isa.hpp"
#include "sw/kpu/components/stateful_block_mover.hpp"
#include "sw/kpu/noc/noc.hpp"

using namespace sw::kpu;

int main() {
    // Create a simple NoC and test injection/delivery
    noc::NoCConfig config{
        noc::NoCTopology::MESH_2D,
        4, 4,  // rows, cols
        noc::RoutingAlgorithm::XY,
        noc::FlowControl::STORE_AND_FORWARD,
        1,  // router_latency
        1,  // link_latency
        64,  // link_bandwidth
        64,  // injection_bandwidth
        32,  // dma_bandwidth
        8,   // input_buffer_flits
        4,   // output_buffer_flits
        64,  // flit_size_bytes
        4096,  // max_packet_flits
        1,   // num_virtual_channels
        true,  // dma_at_edges
        1    // dma_ports_per_edge
    };

    noc::NoC noc(config);

    // Set up delivery callback
    bool delivered = false;
    noc.set_l3_delivery_callback(1, [&](const noc::NoCPacket& pkt, uint64_t cycle) {
        std::cout << "DELIVERED at cycle " << cycle << ": packet seq=" << pkt.sequence_num
                    << " from router " << (int)pkt.src_router << " to router " << (int)pkt.dst_router << "\n";
        delivered = true;
    });

    // Create a tile and inject
    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.k_tile = 0;
    tile.size = 64;  // Small tile

    std::cout << "Injecting packet from router 0 to router 1...\n";
    auto result = noc.inject_l3_to_l3(0, 1, tile, 0);
    std::cout << "Injection result: " << (result == noc::TransferResult::SUCCESS ? "SUCCESS" : "FAILED") << "\n";

    // Step the NoC
    std::cout << "\nStepping NoC:\n";
    for (uint64_t cycle = 0; cycle < 20 && !delivered; cycle++) {
        noc.step(cycle);

        // Check if packet is in flight
        std::cout << "Cycle " << cycle << ": is_idle=" << noc.is_idle() << "\n";
    }

    if (!delivered) {
        std::cout << "\nPacket NOT delivered after 20 cycles!\n";
    }

    // Print stats
    const auto& stats = noc.stats();
    std::cout << "\nNoC Stats:\n";
    std::cout << "  total_packets: " << stats.total_packets << "\n";
    std::cout << "  total_hops: " << stats.total_hops << "\n";
    std::cout << "  total_latency_cycles: " << stats.total_latency_cycles << "\n";

    return 0;
}

/*
g++ -std=c++20 -Iinclude -o ./examples/test/debug_noc ./examples/test/debug_noc.cpp \
    -Lbuild/src/noc -Lbuild/src/trace -Lbuild/src/components/memory \
    -Lbuild/_deps/fmt-build -Lbuild/_deps/spdlog-build \
    -lkpu_noc -lkpu_trace -lkpu_memory_components -lfmt -lspdlog -lpthread 2>&1 && \
LD_LIBRARY_PATH=build/src/noc:build/src/trace:build/src/components/memory:build/_deps/fmt-build:build/_deps/spdlog-build ./examples/test/debug_noc 2>&1

Debug NoC delivery
*/
