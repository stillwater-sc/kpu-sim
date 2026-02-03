/**
 * @file schedule.hpp
 * @brief Schedule DSL for declaring KPU system-level data movement schedules
 *
 * A Schedule is a declarative C++ description that compiles to a DMProgram.
 * It describes tensor declarations, tiling, loop nesting, data movement,
 * compute operations, and buffer management.
 *
 * Three prototypical schedules are supported:
 * - MatMul (output-stationary): C[M,N] += A[M,K] x B[K,N]
 * - Conv2D (im2col + matmul): output = im2col(input) x weights + bias
 * - Softmax (multi-pass reduction): y = exp(x - max) / sum(exp(x - max))
 */

#pragma once

#include <sw/concepts.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace sw::kpu::dsl {

using isa::MatrixID;
using isa::Transform;
using isa::DMProgram;
using isa::DMOpcode;
using isa::BufferSlot;
using sw::kpu::ActivationType;

// ============================================================================
// Tensor Declaration
// ============================================================================

enum class DataType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    FP8  = 4,
};

inline Size datatype_size(DataType dt) {
    switch (dt) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::INT8: return 1;
        case DataType::FP8:  return 1;
        default: return 4;
    }
}

struct Tensor {
    std::string name;
    MatrixID id;
    std::vector<Size> shape;
    Address base_addr = 0;
    DataType dtype = DataType::FP32;
};

// ============================================================================
// Elementwise and Reduction Operations (for VE)
// ============================================================================

enum class ElementwiseOp : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    EXP = 4,
    LOG = 5,
    SQRT = 6,
    RELU = 7,
    GELU = 8,
    SIGMOID = 9,
};

enum class ReductionOp : uint8_t {
    SUM = 0,
    MAX = 1,
    MIN = 2,
};

// ============================================================================
// Schedule Operations (IR nodes before compilation)
// ============================================================================

enum class ScheduleOpKind : uint8_t {
    // Data movement
    LOAD,               // DRAM -> L3
    STORE,              // L3 -> DRAM
    MOVE,               // L3 -> L2
    WRITEBACK,          // L2 -> L3
    STREAM_ROWS,        // L2 -> L1 (A-side, west edge)
    STREAM_COLS,        // L2 -> L1 (B-side, north edge)
    BROADCAST,          // L2 -> all PEs

    // Compute
    COMPUTE_MATMUL,     // systolic C += A x B
    COMPUTE_ELEMENTWISE,// VE elementwise op
    COMPUTE_REDUCE,     // VE reduction op

    // Drain
    DRAIN,              // accumulator -> L2
    DRAIN_FUSED,        // drain + activation via VE
    DRAIN_TO_SCRATCH,   // drain to L2 scratch region

    // Synchronization
    BARRIER,
    WAIT_DMA,

    // Double-buffering
    DOUBLE_BUFFER,

    // Conditional
    IF_NOT_RESIDENT,

    // Loop control
    FOR_TILES,          // loop over tile dimension
    END_LOOP,

    // Gather/scatter (for im2col)
    LOAD_GATHER,        // strided gather from DRAM -> L3
    STORE_SCATTER,      // strided scatter from L3 -> DRAM
};

/**
 * @brief Im2Col gather parameters for conv2d
 */
struct Im2ColParams {
    Address input_base = 0;
    Size H = 0, W = 0, Ci = 0;
    Size Kh = 0, Kw = 0;
    Size stride_h = 1, stride_w = 1;
    Size pad_h = 0, pad_w = 0;
    Size spatial_offset = 0;
};

/**
 * @brief A single operation in the schedule IR
 */
struct ScheduleOp {
    ScheduleOpKind kind;
    MatrixID matrix = MatrixID::A;
    Transform transform = Transform::IDENTITY;
    ElementwiseOp ewise_op = ElementwiseOp::ADD;
    ReductionOp reduce_op = ReductionOp::SUM;
    ActivationType activation = ActivationType::NONE;
    std::string dim_name;           // for FOR_TILES
    Im2ColParams im2col;            // for LOAD_GATHER
    std::string scratch_name;       // for DRAIN_TO_SCRATCH

    // Nested operations (for loop bodies)
    std::vector<ScheduleOp> body;
};

// ============================================================================
// LoopScope — fluent builder for loop bodies
// ============================================================================

class Schedule;  // forward

class LoopScope {
public:
    LoopScope(std::vector<ScheduleOp>& target, Schedule& sched,
              std::vector<std::vector<ScheduleOp>*> parent_chain = {})
        : ops_(target), sched_(sched), parent_chain_(std::move(parent_chain)) {}

    LoopScope for_tiles(std::string dim);

    LoopScope load(MatrixID mat);
    LoopScope load_gather(MatrixID mat, const Im2ColParams& params);
    LoopScope store(MatrixID mat);
    LoopScope move(MatrixID mat, Transform xf = Transform::IDENTITY);
    LoopScope writeback(MatrixID mat);
    LoopScope stream_rows(MatrixID mat);
    LoopScope stream_cols(MatrixID mat);
    LoopScope broadcast(MatrixID mat);

    LoopScope compute_matmul();
    LoopScope compute_elementwise(ElementwiseOp op);
    LoopScope compute_reduce(ReductionOp op);

    LoopScope drain();
    LoopScope drain_fused(ActivationType act);
    LoopScope drain_to_scratch(const std::string& name);

    LoopScope barrier();
    LoopScope wait_dma();

    LoopScope double_buffer(MatrixID mat);
    LoopScope if_not_resident(MatrixID mat);

    /**
     * @brief End this loop scope and return to the parent scope.
     * Returns a LoopScope referencing the parent's operation list.
     */
    LoopScope end();

private:
    std::vector<ScheduleOp>& ops_;
    Schedule& sched_;
    std::vector<std::vector<ScheduleOp>*> parent_chain_;  // stack of parent ops ptrs
};

// ============================================================================
// Schedule — top-level builder
// ============================================================================

class Schedule {
public:
    explicit Schedule(std::string name);

    // Tensor declarations
    Schedule& tensor(const Tensor& t);

    // Tiling configuration
    Schedule& tile(std::string dim, Size size);

    // Dataflow strategy
    Schedule& dataflow(DMProgram::Dataflow df);

    // Hardware config
    Schedule& systolic_size(Size s);
    Schedule& l3_buffers(uint8_t count);
    Schedule& l2_banks(uint8_t count);
    Schedule& element_size(Size s);

    // Loop nesting (returns LoopScope for chaining)
    LoopScope for_tiles(std::string dim);

    // Compile to DMProgram
    DMProgram compile() const;

    // Accessors
    const std::string& name() const { return name_; }
    const std::vector<Tensor>& tensors() const { return tensors_; }
    const std::vector<ScheduleOp>& operations() const { return ops_; }

    Size get_tile(const std::string& dim) const;
    Size get_dim(const std::string& dim) const;
    DMProgram::Dataflow get_dataflow() const { return dataflow_; }
    Size get_systolic_size() const { return systolic_size_; }
    Size get_element_size() const { return element_size_; }
    uint8_t get_l3_buffers() const { return l3_buffers_; }
    uint8_t get_l2_banks() const { return l2_banks_; }

    // Find tensor by MatrixID
    const Tensor* find_tensor(MatrixID id) const;

private:
    std::string name_;
    std::vector<Tensor> tensors_;
    std::vector<std::pair<std::string, Size>> tiles_;
    DMProgram::Dataflow dataflow_ = DMProgram::Dataflow::OUTPUT_STATIONARY;
    Size systolic_size_ = 16;
    Size element_size_ = 4;
    uint8_t l3_buffers_ = 4;
    uint8_t l2_banks_ = 8;

    // Top-level operations (loops and ops)
    std::vector<ScheduleOp> ops_;

    friend class LoopScope;
};

} // namespace sw::kpu::dsl
