/**
 * @file softmax_schedule.cpp
 * @brief Softmax schedule: y[b,i] = exp(x[b,i] - max) / sum(exp(x[b,i] - max))
 *
 * Three-pass reduction + elementwise kernel using the Vector Engine.
 * No systolic array — all compute via VE.
 */

#include <sw/kpu/schedules/softmax_schedule.hpp>

namespace sw::kpu::schedules {

using namespace dsl;
using isa::MatrixID;
using isa::DMProgram;

Schedule softmax(Size B, Size D, Size Tb) {
    Schedule sched("softmax_" + std::to_string(B) + "x" + std::to_string(D));

    // X is input, Y is output. Both use MatrixID::A / MatrixID::C convention.
    sched.tensor(Tensor{"X", MatrixID::A, {B, D}, 0, DataType::FP32});
    sched.tensor(Tensor{"Y", MatrixID::C, {B, D}, 0, DataType::FP32});

    sched.tile("tb", Tb);
    sched.dataflow(DMProgram::Dataflow::OUTPUT_STATIONARY);

    /**
     * Softmax Schedule (3-pass VE kernel):
     *
     * for tb in 0..B/Tb:
     *   // PASS 1: Find max per row
     *   load(X[tb])
     *   move(X[tb])
     *   stream_rows(X[tb])
     *   compute_reduce(MAX)
     *   drain_to_scratch(max_buf)
     *
     *   // PASS 2: exp(x - max)
     *   broadcast(max_buf)
     *   stream_rows(X[tb])          // re-stream from L2
     *   compute_elementwise(SUB)
     *   compute_elementwise(EXP)
     *   drain()                     // exp values -> L2
     *
     *   // PASS 3: sum + divide
     *   stream_rows(exp_buf)
     *   compute_reduce(SUM)
     *   drain_to_scratch(sum_buf)
     *   broadcast(sum_buf)
     *   stream_rows(exp_buf)
     *   compute_elementwise(DIV)
     *   drain()                     // final Y -> L2
     *
     *   writeback(Y[tb])
     *   store(Y[tb])
     * end
     */

    sched.for_tiles("tb")
        // PASS 1: Find max per row (reduction over D)
        .load(MatrixID::A)
        .barrier()
        .move(MatrixID::A)
        .stream_rows(MatrixID::A)
        .compute_reduce(ReductionOp::MAX)
        .drain_to_scratch("max_buf")

        // PASS 2: exp(x - max)
        .broadcast(MatrixID::A)        // broadcast max values
        .stream_rows(MatrixID::A)      // re-stream input from L2
        .compute_elementwise(ElementwiseOp::SUB)
        .compute_elementwise(ElementwiseOp::EXP)
        .drain()                        // exp values -> L2

        // PASS 3: sum + divide
        .stream_rows(MatrixID::C)      // stream exp values
        .compute_reduce(ReductionOp::SUM)
        .drain_to_scratch("sum_buf")
        .broadcast(MatrixID::C)        // broadcast sum
        .stream_rows(MatrixID::C)      // re-stream exp values
        .compute_elementwise(ElementwiseOp::DIV)
        .drain()                        // final output -> L2

        .writeback(MatrixID::C)
        .store(MatrixID::C)
    .end();

    return sched;
}

} // namespace sw::kpu::schedules
