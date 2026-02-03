/**
 * @file conv2d_schedule.cpp
 * @brief Conv2D schedule using im2col lowering + output-stationary matmul
 */

#include <sw/kpu/schedules/conv2d_schedule.hpp>

namespace sw::kpu::schedules {

using namespace dsl;
using isa::MatrixID;
using isa::DMProgram;

Schedule conv2d_im2col(Size batch, Size Ci, Size H, Size W,
                       Size Co, Size Kh, Size Kw,
                       Size stride, Size padding,
                       Size Ti, Size Tj) {
    // Compute output spatial dimensions
    Size Oh = (H + 2 * padding - Kh) / stride + 1;
    Size Ow = (W + 2 * padding - Kw) / stride + 1;

    // Lowered dimensions:
    //   M = batch * Oh * Ow   (spatial output)
    //   N = Co                (output channels)
    //   K = Ci * Kh * Kw      (receptive field)
    Size M = batch * Oh * Ow;
    Size N = Co;
    Size K = Ci * Kh * Kw;

    Schedule sched("conv2d_im2col_" + std::to_string(batch) + "x" +
                   std::to_string(Ci) + "x" + std::to_string(H) + "x" +
                   std::to_string(W) + "_" + std::to_string(Co) + "x" +
                   std::to_string(Kh) + "x" + std::to_string(Kw));

    // Tensors in lowered matmul form
    sched.tensor(Tensor{"A_col", MatrixID::A, {M, K}, 0, DataType::FP32});
    sched.tensor(Tensor{"B_w",   MatrixID::B, {K, N}, 0, DataType::FP32});
    sched.tensor(Tensor{"C_out", MatrixID::C, {M, N}, 0, DataType::FP32});

    // Tiling: full receptive field as Tk
    Size Tk = K;
    sched.tile("ti", Ti);
    sched.tile("tj", Tj);
    sched.tile("tk", Tk);

    sched.dataflow(DMProgram::Dataflow::OUTPUT_STATIONARY);

    /**
     * Conv2D Schedule (im2col):
     *   for ti in 0..M/Ti:           // spatial output tiles
     *     for tj in 0..Co/Tj:        // output channel tiles
     *       for tk in 0..K/Tk:       // input channel tiles (usually 1)
     *         load_gather(A[ti,tk])   // DMA: im2col gather from DRAM -> L3
     *         load(B[tk,tj])          // DMA: weight tile (contiguous)
     *         barrier()
     *         move(A[ti,tk])          // BM: L3 -> L2
     *         move(B[tk,tj])          // BM: L3 -> L2
     *         stream_rows(A)          // STR: patches to west edge
     *         stream_cols(B)          // STR: weights to north edge
     *       end
     *       drain_fused(C[ti,tj], bias, RELU)
     *       writeback(C[ti,tj])
     *       store(C[ti,tj])
     *     end
     *   end
     */

    Im2ColParams im2col_params;
    im2col_params.input_base = 0;
    im2col_params.H = H;
    im2col_params.W = W;
    im2col_params.Ci = Ci;
    im2col_params.Kh = Kh;
    im2col_params.Kw = Kw;
    im2col_params.stride_h = stride;
    im2col_params.stride_w = stride;
    im2col_params.pad_h = padding;
    im2col_params.pad_w = padding;

    sched.for_tiles("ti")
        .for_tiles("tj")
            .for_tiles("tk")
                .load_gather(MatrixID::A, im2col_params)
                .load(MatrixID::B)
                .barrier()
                .move(MatrixID::A)
                .move(MatrixID::B)
                .stream_rows(MatrixID::A)
                .stream_cols(MatrixID::B)
            .end()
            .drain_fused(ActivationType::RELU)
            .writeback(MatrixID::C)
            .store(MatrixID::C)
        .end()
    .end();

    return sched;
}

} // namespace sw::kpu::schedules
