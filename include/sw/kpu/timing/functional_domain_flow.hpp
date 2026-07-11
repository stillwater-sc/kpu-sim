// ============================================================================
// Event-driven functional Domain Flow execution on the CSP timing simulator
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/concurrent_timing_executor.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sw::kpu::timing {

class FunctionalDomainFlowProgram {
public:
    enum class Operation {
        LOAD, MOVE, FEED, MATMUL, COMPUTE, DRAIN, WRITEBACK, STORE, BARRIER
    };

    struct Node {
        Operation operation = Operation::BARRIER;
        TileDescriptor tile;
        ConcurrentTimingExecutor::MatMulComputeSpec matmul;
        ConcurrentTimingExecutor::FunctionalComputeSpec compute;
        std::vector<size_t> predecessors;

        Node() = default;
        Node(Operation op, TileDescriptor descriptor,
             ConcurrentTimingExecutor::MatMulComputeSpec matmul_spec,
             std::vector<size_t> dependencies)
            : operation(op), tile(std::move(descriptor)), matmul(std::move(matmul_spec)),
              predecessors(std::move(dependencies)) {}
    };

    size_t add(Node node) {
        for (size_t predecessor : node.predecessors) {
            if (predecessor >= nodes_.size()) {
                throw std::invalid_argument("Domain Flow predecessor must already exist");
            }
        }
        nodes_.push_back(std::move(node));
        return nodes_.size() - 1;
    }

    [[nodiscard]] const std::vector<Node>& nodes() const { return nodes_; }
    [[nodiscard]] bool empty() const { return nodes_.empty(); }

private:
    std::vector<Node> nodes_;
};

/**
 * Dispatches every ready node in a Domain Flow DAG and marks it complete only
 * when its corresponding CSP completion event occurs. Independent branches
 * remain concurrent; dependency edges provide the global orchestration.
 */
class FunctionalDomainFlowExecutor {
public:
    explicit FunctionalDomainFlowExecutor(ConcurrentTimingExecutor::Config config = {})
        : executor_(std::move(config)) {}

    void set_tile_payload(const TileID& id, TilePayload payload) {
        executor_.set_tile_payload(id, std::move(payload));
    }

    bool run(const FunctionalDomainFlowProgram& program) {
        if (program.empty()) return true;
        std::vector<bool> scheduled(program.nodes().size(), false);
        std::vector<bool> completed(program.nodes().size(), false);
        size_t completed_count = 0;
        size_t event_cursor = executor_.events().size();

        while (completed_count < program.nodes().size() &&
               executor_.current_cycle() < executor_.config().max_cycles) {
            for (size_t i = 0; i < program.nodes().size(); ++i) {
                if (scheduled[i]) continue;
                const auto& node = program.nodes()[i];
                const bool ready = std::all_of(node.predecessors.begin(), node.predecessors.end(),
                    [&completed](size_t predecessor) { return completed[predecessor]; });
                if (!ready) continue;
                scheduled[i] = true;
                if (node.operation == FunctionalDomainFlowProgram::Operation::BARRIER) {
                    completed[i] = true;
                    ++completed_count;
                } else {
                    dispatch(node);
                }
            }

            if (completed_count == program.nodes().size()) break;
            executor_.step();
            const auto& events = executor_.events();
            for (; event_cursor < events.size(); ++event_cursor) {
                for (size_t i = 0; i < program.nodes().size(); ++i) {
                    if (!scheduled[i] || completed[i]) continue;
                    if (matches_completion(program.nodes()[i], events[event_cursor])) {
                        completed[i] = true;
                        ++completed_count;
                    }
                }
            }
        }
        return completed_count == program.nodes().size() && executor_.is_complete();
    }

    [[nodiscard]] ConcurrentTimingExecutor& executor() { return executor_; }
    [[nodiscard]] const ConcurrentTimingExecutor& executor() const { return executor_; }

private:
    ConcurrentTimingExecutor executor_;

    void dispatch(const FunctionalDomainFlowProgram::Node& node) {
        using Op = FunctionalDomainFlowProgram::Operation;
        switch (node.operation) {
            case Op::LOAD: executor_.schedule_load(node.tile); break;
            case Op::MOVE: executor_.schedule_move(node.tile); break;
            case Op::FEED: executor_.schedule_feed(node.tile); break;
            case Op::MATMUL: executor_.schedule_matmul_compute(node.tile, node.matmul); break;
            case Op::COMPUTE: executor_.schedule_functional_compute(node.tile, node.compute); break;
            case Op::DRAIN: executor_.schedule_drain(node.tile); break;
            case Op::WRITEBACK: executor_.schedule_writeback(node.tile); break;
            case Op::STORE: executor_.schedule_store(node.tile); break;
            case Op::BARRIER: break;
        }
    }

    [[nodiscard]] static bool matches_completion(const FunctionalDomainFlowProgram::Node& node,
                                                  const TimingEvent& event) {
        if (event.tile_id != node.tile.tile_id) return false;
        using Op = FunctionalDomainFlowProgram::Operation;
        switch (node.operation) {
            case Op::LOAD: return event.type == EventType::TILE_ARRIVED_L3;
            case Op::MOVE: return event.type == EventType::BM_MOVE_COMPLETE;
            case Op::FEED: return event.type == EventType::TILE_FED_TO_COMPUTE;
            case Op::MATMUL:
            case Op::COMPUTE: return event.type == EventType::COMPUTE_COMPLETE;
            case Op::DRAIN: return event.type == EventType::TILE_DRAINED;
            case Op::WRITEBACK: return event.type == EventType::BM_WRITEBACK_COMPLETE;
            case Op::STORE: return event.type == EventType::DMA_STORE_COMPLETE;
            case Op::BARRIER: return false;
        }
        return false;
    }
};

} // namespace sw::kpu::timing
