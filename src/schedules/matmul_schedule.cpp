/**
 * @file matmul_schedule.cpp
 * @brief Output-stationary MatMul schedule: C[M,N] += A[M,K] x B[K,N]
 */

#include <sw/kpu/schedules/matmul_schedule.hpp>

namespace sw::kpu::schedules {

using namespace dsl;
using isa::MatrixID;
using isa::DMProgram;

Schedule matmul_output_stationary(Size M, Size N, Size K,
                                  Size Ti, Size Tj, Size Tk) {
    Schedule sched("matmul_os_" + std::to_string(M) + "x" +
                   std::to_string(N) + "x" + std::to_string(K));

    // Declare tensors
    sched.tensor(Tensor{"A", MatrixID::A, {M, K}, 0, DataType::FP32});
    sched.tensor(Tensor{"B", MatrixID::B, {K, N}, 0, DataType::FP32});
    sched.tensor(Tensor{"C", MatrixID::C, {M, N}, 0, DataType::FP32});

    // Tiling
    sched.tile("ti", Ti);
    sched.tile("tj", Tj);
    sched.tile("tk", Tk);

    // Dataflow
    sched.dataflow(DMProgram::Dataflow::OUTPUT_STATIONARY);

    /**
     * Output-Stationary Schedule:
     *   for ti in 0..M/Ti:
     *     for tj in 0..N/Tj:
     *       for tk in 0..K/Tk:
     *         load(A[ti,tk])          // DMA: DRAM -> L3
     *         load(B[tk,tj])          // DMA: DRAM -> L3
     *         barrier()               // wait for DMA
     *         move(A[ti,tk])          // BM: L3 -> L2
     *         move(B[tk,tj])          // BM: L3 -> L2
     *         stream_rows(A)          // STR: L2 -> L1 west edge
     *         stream_cols(B)          // STR: L2 -> L1 north edge
     *         // PE: C += A x B (reactive)
     *       end
     *       drain(C[ti,tj])           // STR: accumulator -> L2
     *       writeback(C[ti,tj])       // BM: L2 -> L3
     *       store(C[ti,tj])           // DMA: L3 -> DRAM
     *     end
     *   end
     */

    sched.for_tiles("ti")
        .for_tiles("tj")
            .for_tiles("tk")
                .load(MatrixID::A)
                .load(MatrixID::B)
                .barrier()
                .move(MatrixID::A)
                .move(MatrixID::B)
                .stream_rows(MatrixID::A)
                .stream_cols(MatrixID::B)
            .end()
            .drain()
            .writeback(MatrixID::C)
            .store(MatrixID::C)
        .end()
    .end();

    return sched;
}

} // namespace sw::kpu::schedules
