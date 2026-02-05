// ============================================================================
// include/sw/kpu/timing/work_queue.hpp
// Work queue for CSP process scheduling
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>

#include <algorithm>
#include <deque>
#include <functional>
#include <optional>
#include <queue>
#include <stdexcept>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief Simple FIFO work queue for component processes
 *
 * Each CSP process (DMA, BlockMover, Streamer) has a work queue containing
 * items to process. The queue supports:
 * - FIFO ordering (default)
 * - Peek without remove (for checking if work can proceed)
 * - Size limits (for bounded lookahead)
 *
 * @tparam WorkItem Type of work item (typically TileDescriptor)
 */
template<typename WorkItem>
class WorkQueue {
public:
    /**
     * @brief Construct an unbounded work queue
     */
    WorkQueue() : max_size_(0) {}  // 0 means unbounded

    /**
     * @brief Construct a bounded work queue
     * @param max_size Maximum queue depth (for bounded lookahead)
     */
    explicit WorkQueue(size_t max_size) : max_size_(max_size) {}

    /**
     * @brief Add an item to the back of the queue
     * @param item Work item to enqueue
     * @return true if enqueued, false if queue is at max capacity
     */
    bool enqueue(const WorkItem& item) {
        if (max_size_ > 0 && queue_.size() >= max_size_) {
            return false;
        }
        queue_.push_back(item);
        return true;
    }

    /**
     * @brief Add an item to the back of the queue (move version)
     */
    bool enqueue(WorkItem&& item) {
        if (max_size_ > 0 && queue_.size() >= max_size_) {
            return false;
        }
        queue_.push_back(std::move(item));
        return true;
    }

    /**
     * @brief Check if queue is empty
     */
    [[nodiscard]] bool empty() const {
        return queue_.empty();
    }

    /**
     * @brief Check if queue is at capacity
     */
    [[nodiscard]] bool full() const {
        return max_size_ > 0 && queue_.size() >= max_size_;
    }

    /**
     * @brief Get number of items in queue
     */
    [[nodiscard]] size_t size() const {
        return queue_.size();
    }

    /**
     * @brief Get maximum capacity (0 = unbounded)
     */
    [[nodiscard]] size_t max_size() const {
        return max_size_;
    }

    /**
     * @brief Look at front item without removing
     * @return Reference to front item
     * @throws std::out_of_range if queue is empty
     */
    [[nodiscard]] const WorkItem& peek() const {
        if (queue_.empty()) {
            throw std::out_of_range("WorkQueue::peek() on empty queue");
        }
        return queue_.front();
    }

    /**
     * @brief Look at front item without removing (optional version)
     * @return Front item if exists, std::nullopt otherwise
     */
    [[nodiscard]] std::optional<WorkItem> try_peek() const {
        if (queue_.empty()) {
            return std::nullopt;
        }
        return queue_.front();
    }

    /**
     * @brief Remove and return front item
     * @return The front item
     * @throws std::out_of_range if queue is empty
     */
    WorkItem dequeue() {
        if (queue_.empty()) {
            throw std::out_of_range("WorkQueue::dequeue() on empty queue");
        }
        WorkItem item = std::move(queue_.front());
        queue_.pop_front();
        return item;
    }

    /**
     * @brief Remove and return front item (optional version)
     * @return Front item if exists, std::nullopt otherwise
     */
    std::optional<WorkItem> try_dequeue() {
        if (queue_.empty()) {
            return std::nullopt;
        }
        WorkItem item = std::move(queue_.front());
        queue_.pop_front();
        return item;
    }

    /**
     * @brief Clear all items from queue
     */
    void reset() {
        queue_.clear();
    }

    /**
     * @brief Get all items (for debugging/inspection)
     */
    [[nodiscard]] std::vector<WorkItem> items() const {
        return std::vector<WorkItem>(queue_.begin(), queue_.end());
    }

    /**
     * @brief Find first item matching predicate
     * @param pred Predicate function
     * @return Matching item if found, std::nullopt otherwise
     */
    template<typename Predicate>
    [[nodiscard]] std::optional<WorkItem> find_if(Predicate pred) const {
        for (const auto& item : queue_) {
            if (pred(item)) {
                return item;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Remove first item matching predicate
     * @param pred Predicate function
     * @return true if item was found and removed
     */
    template<typename Predicate>
    bool remove_if(Predicate pred) {
        auto it = std::find_if(queue_.begin(), queue_.end(), pred);
        if (it != queue_.end()) {
            queue_.erase(it);
            return true;
        }
        return false;
    }

    /**
     * @brief Find and remove first item matching predicate (work-conserving scan)
     *
     * This enables work-conserving scheduling: instead of only checking the head
     * of the queue, scan the entire queue for any item that's ready to process.
     *
     * @param pred Predicate function returning true for items ready to process
     * @return The first matching item if found, std::nullopt otherwise
     */
    template<typename Predicate>
    std::optional<WorkItem> dequeue_if(Predicate pred) {
        auto it = std::find_if(queue_.begin(), queue_.end(), pred);
        if (it != queue_.end()) {
            WorkItem item = std::move(*it);
            queue_.erase(it);
            return item;
        }
        return std::nullopt;
    }

    /**
     * @brief Get item at specific index (for work-conserving scan)
     * @param index Position in queue
     * @return Reference to item at position
     * @throws std::out_of_range if index is invalid
     */
    [[nodiscard]] const WorkItem& at(size_t index) const {
        if (index >= queue_.size()) {
            throw std::out_of_range("WorkQueue::at() index out of range");
        }
        return queue_[index];
    }

    /**
     * @brief Remove item at specific index
     * @param index Position to remove
     * @return The removed item
     * @throws std::out_of_range if index is invalid
     */
    WorkItem remove_at(size_t index) {
        if (index >= queue_.size()) {
            throw std::out_of_range("WorkQueue::remove_at() index out of range");
        }
        WorkItem item = std::move(queue_[index]);
        queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(index));
        return item;
    }

private:
    std::deque<WorkItem> queue_;
    size_t max_size_;
};

// ============================================================================
// PriorityWorkQueue - For priority aging and urgency-based scheduling
// ============================================================================

/**
 * @brief Work queue with priority ordering for aging/urgency
 *
 * Items are ordered by priority (higher priority processed first).
 * Supports dynamic priority updates for age-based prioritization.
 *
 * @tparam WorkItem Type of work item
 * @tparam Priority Type of priority value (default size_t)
 */
template<typename WorkItem, typename Priority = size_t>
class PriorityWorkQueue {
public:
    struct PrioritizedItem {
        WorkItem item;
        Priority priority;

        bool operator<(const PrioritizedItem& other) const {
            // Lower priority number = lower priority (for std::priority_queue max-heap)
            return priority < other.priority;
        }
    };

    /**
     * @brief Construct an unbounded priority queue
     */
    PriorityWorkQueue() : max_size_(0) {}

    /**
     * @brief Construct a bounded priority queue
     */
    explicit PriorityWorkQueue(size_t max_size) : max_size_(max_size) {}

    /**
     * @brief Add item with priority
     * @param item Work item
     * @param priority Priority value (higher = more urgent)
     * @return true if enqueued, false if at capacity
     */
    bool enqueue(const WorkItem& item, Priority priority) {
        if (max_size_ > 0 && heap_.size() >= max_size_) {
            return false;
        }
        heap_.push({item, priority});
        return true;
    }

    /**
     * @brief Check if empty
     */
    [[nodiscard]] bool empty() const {
        return heap_.empty();
    }

    /**
     * @brief Check if at capacity
     */
    [[nodiscard]] bool full() const {
        return max_size_ > 0 && heap_.size() >= max_size_;
    }

    /**
     * @brief Get number of items
     */
    [[nodiscard]] size_t size() const {
        return heap_.size();
    }

    /**
     * @brief Peek at highest priority item
     * @return Highest priority item if exists, std::nullopt otherwise
     */
    [[nodiscard]] std::optional<PrioritizedItem> try_peek() const {
        if (heap_.empty()) {
            return std::nullopt;
        }
        return heap_.top();
    }

    /**
     * @brief Remove and return highest priority item
     * @return Item if exists, std::nullopt otherwise
     */
    std::optional<WorkItem> try_dequeue() {
        if (heap_.empty()) {
            return std::nullopt;
        }
        WorkItem item = heap_.top().item;
        heap_.pop();
        return item;
    }

    /**
     * @brief Clear all items
     */
    void reset() {
        while (!heap_.empty()) {
            heap_.pop();
        }
    }

private:
    std::priority_queue<PrioritizedItem> heap_;
    size_t max_size_;
};

// ============================================================================
// TileWorkQueue - Specialized queue for tile operations
// ============================================================================

/**
 * @brief Specialized work queue for tile operations with matrix partitioning
 *
 * Supports work-conserving behavior by allowing out-of-order processing
 * when a specific tile can't proceed.
 */
class TileWorkQueue {
public:
    /**
     * @brief Construct tile work queue
     * @param max_per_matrix Maximum items per matrix type (for livelock prevention)
     */
    explicit TileWorkQueue(size_t max_per_matrix = 0)
        : max_per_matrix_(max_per_matrix) {}

    /**
     * @brief Add a tile to the queue
     * @param tile Tile descriptor
     * @return true if added, false if partition is full
     */
    bool enqueue(const TileDescriptor& tile) {
        auto matrix = tile.tile_id.matrix;
        auto& queue = get_queue(matrix);

        if (max_per_matrix_ > 0 && queue.size() >= max_per_matrix_) {
            return false;
        }

        queue.push_back(tile);
        return true;
    }

    /**
     * @brief Get total number of items across all partitions
     */
    [[nodiscard]] size_t total_size() const {
        return queue_a_.size() + queue_b_.size() + queue_c_.size();
    }

    /**
     * @brief Check if all partitions are empty
     */
    [[nodiscard]] bool empty() const {
        return queue_a_.empty() && queue_b_.empty() && queue_c_.empty();
    }

    /**
     * @brief Get number of items for a matrix
     */
    [[nodiscard]] size_t size(isa::MatrixID matrix) const {
        switch (matrix) {
            case isa::MatrixID::A: return queue_a_.size();
            case isa::MatrixID::B: return queue_b_.size();
            case isa::MatrixID::C: return queue_c_.size();
        }
        return 0;
    }

    /**
     * @brief Peek at front of a partition
     */
    [[nodiscard]] std::optional<TileDescriptor> peek(isa::MatrixID matrix) const {
        const auto& queue = get_queue_const(matrix);
        if (queue.empty()) {
            return std::nullopt;
        }
        return queue.front();
    }

    /**
     * @brief Dequeue from a partition
     */
    std::optional<TileDescriptor> dequeue(isa::MatrixID matrix) {
        auto& queue = get_queue(matrix);
        if (queue.empty()) {
            return std::nullopt;
        }
        TileDescriptor item = std::move(queue.front());
        queue.pop_front();
        return item;
    }

    /**
     * @brief Find any ready tile (work-conserving)
     * @param can_proceed Predicate to check if tile can be processed
     * @return First processable tile, or std::nullopt
     */
    template<typename Predicate>
    std::optional<TileDescriptor> find_ready(Predicate can_proceed) {
        // Check each partition for a ready tile
        for (auto& tile : queue_a_) {
            if (can_proceed(tile)) {
                return tile;
            }
        }
        for (auto& tile : queue_b_) {
            if (can_proceed(tile)) {
                return tile;
            }
        }
        for (auto& tile : queue_c_) {
            if (can_proceed(tile)) {
                return tile;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Find and remove the oldest ready tile (priority aging)
     * @param can_proceed Predicate to check if tile can be processed
     * @param current_cycle Current simulation cycle (for age calculation)
     * @return Oldest processable tile, or std::nullopt
     */
    template<typename Predicate>
    std::optional<TileDescriptor> dequeue_oldest_ready(Predicate can_proceed, [[maybe_unused]] Cycle current_cycle) {
        TileDescriptor* oldest = nullptr;
        std::deque<TileDescriptor>* oldest_queue = nullptr;
        size_t oldest_idx = 0;

        auto check_queue = [&](std::deque<TileDescriptor>& queue) {
            for (size_t i = 0; i < queue.size(); ++i) {
                auto& tile = queue[i];
                if (can_proceed(tile)) {
                    if (!oldest || tile.enqueue_cycle < oldest->enqueue_cycle) {
                        oldest = &tile;
                        oldest_queue = &queue;
                        oldest_idx = i;
                    }
                }
            }
        };

        check_queue(queue_a_);
        check_queue(queue_b_);
        check_queue(queue_c_);

        if (oldest && oldest_queue) {
            TileDescriptor result = *oldest;
            oldest_queue->erase(oldest_queue->begin() + oldest_idx);
            return result;
        }
        return std::nullopt;
    }

    /**
     * @brief Clear all partitions
     */
    void reset() {
        queue_a_.clear();
        queue_b_.clear();
        queue_c_.clear();
    }

private:
    std::deque<TileDescriptor>& get_queue(isa::MatrixID matrix) {
        switch (matrix) {
            case isa::MatrixID::A: return queue_a_;
            case isa::MatrixID::B: return queue_b_;
            case isa::MatrixID::C: return queue_c_;
        }
        return queue_a_;  // Fallback
    }

    const std::deque<TileDescriptor>& get_queue_const(isa::MatrixID matrix) const {
        switch (matrix) {
            case isa::MatrixID::A: return queue_a_;
            case isa::MatrixID::B: return queue_b_;
            case isa::MatrixID::C: return queue_c_;
        }
        return queue_a_;  // Fallback
    }

    std::deque<TileDescriptor> queue_a_;
    std::deque<TileDescriptor> queue_b_;
    std::deque<TileDescriptor> queue_c_;
    size_t max_per_matrix_;
};

} // namespace sw::kpu::timing
