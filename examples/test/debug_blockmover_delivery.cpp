#include <iostream>
#include <iomanip>
#include "sw/kpu/components/stateful_block_mover.hpp"
#include "sw/kpu/noc/noc.hpp"

using namespace sw::kpu;

int main() {
    // Create BlockMoverArray
    BlockMoverArray::Config config;
    config.rows = 4;
    config.cols = 4;
    BlockMoverArray array(config);

    // Track deliveries
    int deliveries = 0;

    // Override delivery callback to log
    array.noc().set_l3_delivery_callback(1, [&](const noc::NoCPacket& pkt, uint64_t cycle) {
        std::cout << "NoC DELIVERY at cycle " << cycle << ": router 0 -> router 1\n";
        deliveries++;
        // Still need to forward to original callback
        L3TransferPacket l3_pkt;
        l3_pkt.src_l3_id = pkt.src_router;
        l3_pkt.dst_l3_id = pkt.dst_router;
        l3_pkt.tile = pkt.tile;
        l3_pkt.arrival_cycle = cycle;
        array[1].receive_packet(l3_pkt);
    });

    // Create simple program: L3[0] sends to L3[1], L3[1] receives
    BlockMoverProgram prog0;
    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.k_tile = 0;
    tile.size = 64;

    BlockMoverCommand send_cmd;
    send_cmd.op = BlockMoverOp::SEND_EAST;
    send_cmd.tile = tile;
    prog0.l3_id = 0;
    prog0.append(send_cmd);

    BlockMoverProgram prog1;
    BlockMoverCommand recv_cmd;
    recv_cmd.op = BlockMoverOp::RECEIVE_FROM;
    recv_cmd.tile = tile;
    recv_cmd.src_l3_id = 0;
    prog1.l3_id = 1;
    prog1.append(recv_cmd);

    // Load programs
    array.load_programs({prog0, prog1});

    std::cout << "Starting simulation...\n";
    std::cout << "L3[0] state: " << to_string(array[0].state()) << "\n";
    std::cout << "L3[1] state: " << to_string(array[1].state()) << "\n";

    for (uint64_t cycle = 0; cycle < 20; cycle++) {
        array.step(cycle);

        std::cout << "Cycle " << cycle << ": L3[0]=" << to_string(array[0].state());
        if (array[0].current_command()) {
            std::cout << "(" << to_string(array[0].current_command()->op) << ")";
        }
        std::cout << ", L3[1]=" << to_string(array[1].state());
        if (array[1].current_command()) {
            std::cout << "(" << to_string(array[1].current_command()->op) << ")";
        }
        std::cout << ", noc_idle=" << array.noc().is_idle();
        std::cout << ", recv_queue=" << array[1].receive_queue_size();
        std::cout << "\n";

        if (array.all_idle()) {
            std::cout << "All idle at cycle " << cycle << "\n";
            break;
        }
    }

    std::cout << "\nTotal deliveries: " << deliveries << "\n";
    std::cout << "NoC stats: packets=" << array.noc().stats().total_packets << "\n";

    return 0;
}

/*
Debug BlockMoverArray delivery
g++ -std=c++20 -Iinclude -o ./examples/test/debug_blockmover_delivery ./examples/test/debug_blockmover_delivery.cpp \
    -Lbuild/src/dataflow -Lbuild/src/isa -Lbuild/src/components/datamovement \
    -Lbuild/src/components/memory -Lbuild/src/components/compute -Lbuild/src/trace \
    -Lbuild/src/noc -Lbuild/_deps/fmt-build -Lbuild/_deps/spdlog-build \
    -lkpu_dataflow -lkpu_isa -lkpu_datamovement_components -lkpu_memory_components \
    -lkpu_compute_components -lkpu_trace -lkpu_noc -lfmt -lspdlog -lpthread 2>&1 && \
LD_LIBRARY_PATH=build/src/dataflow:build/src/isa:build/src/components/datamovement:build/src/components/memory:build/src/components/compute:build/src/trace:build/src/noc:build/_deps/fmt-build:build/_deps/spdlog-build ./examples/test/debug_blockmover_delivery 2>&1

*/
