/**
 * @file schedule.cpp
 * @brief Implementation of the Schedule DSL builder
 */

#include <sw/kpu/dsl/schedule.hpp>
#include <stdexcept>
#include <algorithm>

namespace sw::kpu::dsl {

// ============================================================================
// Schedule implementation
// ============================================================================

Schedule::Schedule(std::string name) : name_(std::move(name)) {}

Schedule& Schedule::tensor(const Tensor& t) {
    tensors_.push_back(t);
    return *this;
}

Schedule& Schedule::tile(std::string dim, Size size) {
    tiles_.emplace_back(std::move(dim), size);
    return *this;
}

Schedule& Schedule::dataflow(DMProgram::Dataflow df) {
    dataflow_ = df;
    return *this;
}

Schedule& Schedule::systolic_size(Size s) {
    systolic_size_ = s;
    return *this;
}

Schedule& Schedule::l3_buffers(uint8_t count) {
    l3_buffers_ = count;
    return *this;
}

Schedule& Schedule::l2_banks(uint8_t count) {
    l2_banks_ = count;
    return *this;
}

Schedule& Schedule::element_size(Size s) {
    element_size_ = s;
    return *this;
}

Schedule& Schedule::layout_policy(LayoutPolicy policy) {
    layout_policy_ = policy;
    return *this;
}

Schedule& Schedule::num_channels(uint8_t count) {
    num_channels_ = count;
    return *this;
}

LoopScope Schedule::for_tiles(std::string dim) {
    ScheduleOp loop_op;
    loop_op.kind = ScheduleOpKind::FOR_TILES;
    loop_op.dim_name = dim;
    ops_.push_back(std::move(loop_op));
    return LoopScope(ops_.back().body, *this, {&ops_});
}

Size Schedule::get_tile(const std::string& dim) const {
    for (const auto& [d, s] : tiles_) {
        if (d == dim) return s;
    }
    return 0;
}

Size Schedule::get_dim(const std::string& dim) const {
    // Infer dimension from tensor shapes and tile names
    // Convention: "M" or "ti" -> first dim of A/C
    //             "N" or "tj" -> second dim of C / second dim of B
    //             "K" or "tk" -> second dim of A / first dim of B
    //             "B" or "tb" -> batch dimension
    //             "D"         -> softmax dimension
    if (dim == "ti" || dim == "M") {
        for (const auto& t : tensors_) {
            if (t.id == MatrixID::A && t.shape.size() >= 1) return t.shape[0];
        }
    }
    if (dim == "tj" || dim == "N") {
        for (const auto& t : tensors_) {
            if (t.id == MatrixID::C && t.shape.size() >= 2) return t.shape[1];
            if (t.id == MatrixID::B && t.shape.size() >= 2) return t.shape[1];
        }
    }
    if (dim == "tk" || dim == "K") {
        for (const auto& t : tensors_) {
            if (t.id == MatrixID::A && t.shape.size() >= 2) return t.shape[1];
        }
    }
    if (dim == "tb" || dim == "B") {
        for (const auto& t : tensors_) {
            if (t.id == MatrixID::A && t.shape.size() >= 1) return t.shape[0];
        }
    }
    return 0;
}

const Tensor* Schedule::find_tensor(MatrixID id) const {
    for (const auto& t : tensors_) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

// ============================================================================
// LoopScope implementation
// ============================================================================

LoopScope LoopScope::for_tiles(std::string dim) {
    ScheduleOp loop_op;
    loop_op.kind = ScheduleOpKind::FOR_TILES;
    loop_op.dim_name = std::move(dim);
    ops_.push_back(std::move(loop_op));
    // Child's parent chain = our parent chain + our ops
    auto child_chain = parent_chain_;
    child_chain.push_back(&ops_);
    return LoopScope(ops_.back().body, sched_, std::move(child_chain));
}

LoopScope LoopScope::load(MatrixID mat) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::LOAD;
    op.matrix = mat;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::load_gather(MatrixID mat, const Im2ColParams& params) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::LOAD_GATHER;
    op.matrix = mat;
    op.im2col = params;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::store(MatrixID mat) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::STORE;
    op.matrix = mat;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::move(MatrixID mat, Transform xf) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::MOVE;
    op.matrix = mat;
    op.transform = xf;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::writeback(MatrixID mat) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::WRITEBACK;
    op.matrix = mat;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::stream_rows(MatrixID mat) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::STREAM_ROWS;
    op.matrix = mat;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::stream_cols(MatrixID mat) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::STREAM_COLS;
    op.matrix = mat;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::broadcast(MatrixID mat) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::BROADCAST;
    op.matrix = mat;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::compute_matmul() {
    ScheduleOp op;
    op.kind = ScheduleOpKind::COMPUTE_MATMUL;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::compute_elementwise(ElementwiseOp eop) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::COMPUTE_ELEMENTWISE;
    op.ewise_op = eop;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::compute_reduce(ReductionOp rop) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::COMPUTE_REDUCE;
    op.reduce_op = rop;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::drain() {
    ScheduleOp op;
    op.kind = ScheduleOpKind::DRAIN;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::drain_fused(ActivationType act) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::DRAIN_FUSED;
    op.activation = act;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::drain_to_scratch(const std::string& name) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::DRAIN_TO_SCRATCH;
    op.scratch_name = name;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::barrier() {
    ScheduleOp op;
    op.kind = ScheduleOpKind::BARRIER;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::wait_dma() {
    ScheduleOp op;
    op.kind = ScheduleOpKind::WAIT_DMA;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::double_buffer(MatrixID mat) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::DOUBLE_BUFFER;
    op.matrix = mat;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::if_not_resident(MatrixID mat) {
    ScheduleOp op;
    op.kind = ScheduleOpKind::IF_NOT_RESIDENT;
    op.matrix = mat;
    ops_.push_back(std::move(op));
    return *this;
}

LoopScope LoopScope::end() {
    if (!parent_chain_.empty()) {
        auto* parent = parent_chain_.back();
        auto new_chain = std::vector<std::vector<ScheduleOp>*>(
            parent_chain_.begin(), parent_chain_.end() - 1);
        return LoopScope(*parent, sched_, std::move(new_chain));
    }
    return LoopScope(ops_, sched_, {});
}

} // namespace sw::kpu::dsl
