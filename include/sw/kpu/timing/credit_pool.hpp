// ============================================================================
// include/sw/kpu/timing/credit_pool.hpp
// Credit pool for credit-based dataflow synchronization
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <cstddef>
#include <stdexcept>

namespace sw::kpu::timing {

/**
 * @brief Credit pool for credit-based dataflow synchronization
 *
 * In the KPU's credit-based dataflow model:
 * - Upstream producers must acquire a credit before pushing data downstream
 * - Downstream consumers release credits when done with data
 * - Credits represent available buffer/bank slots at the downstream level
 *
 * Example flow:
 * 1. DMA engine acquires L3 credit (checks if L3 buffer available)
 * 2. DMA pushes tile to L3 buffer
 * 3. BlockMover consumes tile from L3, releases L3 credit
 *
 * Invariants:
 * - available_ <= capacity_ (never more credits than capacity)
 * - available_ >= 0 (implicitly, since size_t is unsigned)
 * - acquire() decrements available_, release() increments it
 */
class CreditPool {
public:
    /**
     * @brief Construct a credit pool with given capacity
     * @param capacity Maximum number of credits (e.g., number of L3 buffers)
     * @throws std::invalid_argument if capacity is 0
     */
    explicit CreditPool(size_t capacity)
        : capacity_(capacity), available_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("CreditPool capacity must be > 0");
        }
    }

    /**
     * @brief Try to acquire a credit
     * @return true if credit acquired, false if no credits available
     *
     * Called by upstream producer before pushing data downstream.
     * Non-blocking: returns immediately with result.
     */
    bool acquire() {
        if (available_ > 0) {
            --available_;
            return true;
        }
        return false;
    }

    /**
     * @brief Acquire a credit, blocking until one is available
     * @note In discrete event simulation, this should not be called directly.
     *       Use acquire() and handle the false case by rescheduling.
     * @throws std::runtime_error always (not supported in DES)
     */
    void acquire_blocking() {
        throw std::runtime_error(
            "acquire_blocking() not supported in discrete event simulation. "
            "Use acquire() and reschedule if false.");
    }

    /**
     * @brief Get number of available credits
     * @return Number of credits that can be acquired
     */
    [[nodiscard]] size_t available() const {
        return available_;
    }

    /**
     * @brief Get pool capacity
     * @return Maximum number of credits
     */
    [[nodiscard]] size_t capacity() const {
        return capacity_;
    }

    /**
     * @brief Check if any credits are available
     * @return true if at least one credit is available
     */
    [[nodiscard]] bool has_credit() const {
        return available_ > 0;
    }

    /**
     * @brief Check if pool is empty (no credits available)
     * @return true if no credits available
     */
    [[nodiscard]] bool empty() const {
        return available_ == 0;
    }

    /**
     * @brief Check if pool is full (all credits available)
     * @return true if all credits available (no outstanding acquisitions)
     */
    [[nodiscard]] bool full() const {
        return available_ == capacity_;
    }

    /**
     * @brief Release a credit back to the pool
     * @throws std::overflow_error if releasing more credits than capacity
     *
     * Called by downstream consumer when done with data.
     */
    void release() {
        if (available_ >= capacity_) {
            throw std::overflow_error(
                "CreditPool::release() called but pool is already full. "
                "This indicates a double-release bug.");
        }
        ++available_;
    }

    /**
     * @brief Reset the pool to full capacity
     *
     * Used at simulation start or when resetting state.
     */
    void reset() {
        available_ = capacity_;
    }

    /**
     * @brief Get number of outstanding (acquired but not released) credits
     * @return Number of credits currently in use
     */
    [[nodiscard]] size_t outstanding() const {
        return capacity_ - available_;
    }

private:
    size_t capacity_;   ///< Maximum number of credits
    size_t available_;  ///< Currently available credits
};

// ============================================================================
// PartitionedCreditPool - Credits partitioned by matrix type
// ============================================================================

/**
 * @brief Credit pool partitioned by matrix type to prevent livelock
 *
 * In matrix multiplication, we have three matrix types: A, B, and C.
 * If all credit pools are shared, a pathological case can arise where
 * all buffers are filled with A tiles waiting for B tiles (or vice versa),
 * causing livelock.
 *
 * PartitionedCreditPool reserves a portion of credits for each matrix type,
 * ensuring progress is always possible.
 *
 * Partition strategy (configurable):
 * - Default: Equal partition (capacity/3 per matrix, remainder to C)
 * - Custom: User-specified per-matrix allocation
 */
class PartitionedCreditPool {
public:
    /**
     * @brief Construct with equal partition across A, B, C matrices
     * @param total_capacity Total number of credits to partition
     * @throws std::invalid_argument if total_capacity < 3
     */
    explicit PartitionedCreditPool(size_t total_capacity)
        : total_capacity_(total_capacity) {
        if (total_capacity < 3) {
            throw std::invalid_argument(
                "PartitionedCreditPool requires capacity >= 3 for partitioning");
        }

        // Equal partition: divide evenly, give remainder to C (output matrix)
        size_t per_matrix = total_capacity / 3;
        size_t remainder = total_capacity % 3;

        capacity_a_ = per_matrix;
        capacity_b_ = per_matrix;
        capacity_c_ = per_matrix + remainder;

        available_a_ = capacity_a_;
        available_b_ = capacity_b_;
        available_c_ = capacity_c_;
    }

    /**
     * @brief Construct with custom partition
     * @param capacity_a Credits reserved for matrix A
     * @param capacity_b Credits reserved for matrix B
     * @param capacity_c Credits reserved for matrix C
     * @throws std::invalid_argument if any capacity is 0
     */
    PartitionedCreditPool(size_t capacity_a, size_t capacity_b, size_t capacity_c)
        : total_capacity_(capacity_a + capacity_b + capacity_c),
          capacity_a_(capacity_a), capacity_b_(capacity_b), capacity_c_(capacity_c),
          available_a_(capacity_a), available_b_(capacity_b), available_c_(capacity_c) {
        if (capacity_a == 0 || capacity_b == 0 || capacity_c == 0) {
            throw std::invalid_argument(
                "PartitionedCreditPool requires non-zero capacity for each matrix");
        }
    }

    /**
     * @brief Matrix identifier for partition selection
     */
    enum class Matrix { A, B, C };

    /**
     * @brief Try to acquire a credit for specified matrix
     * @param matrix Which matrix partition to acquire from
     * @return true if credit acquired, false if partition empty
     */
    bool acquire(Matrix matrix) {
        switch (matrix) {
            case Matrix::A:
                if (available_a_ > 0) { --available_a_; return true; }
                break;
            case Matrix::B:
                if (available_b_ > 0) { --available_b_; return true; }
                break;
            case Matrix::C:
                if (available_c_ > 0) { --available_c_; return true; }
                break;
        }
        return false;
    }

    /**
     * @brief Release a credit to specified matrix partition
     * @param matrix Which matrix partition to release to
     * @throws std::overflow_error if partition is already full
     */
    void release(Matrix matrix) {
        switch (matrix) {
            case Matrix::A:
                if (available_a_ >= capacity_a_) {
                    throw std::overflow_error("Matrix A partition full");
                }
                ++available_a_;
                break;
            case Matrix::B:
                if (available_b_ >= capacity_b_) {
                    throw std::overflow_error("Matrix B partition full");
                }
                ++available_b_;
                break;
            case Matrix::C:
                if (available_c_ >= capacity_c_) {
                    throw std::overflow_error("Matrix C partition full");
                }
                ++available_c_;
                break;
        }
    }

    /**
     * @brief Get available credits for a matrix
     */
    [[nodiscard]] size_t available(Matrix matrix) const {
        switch (matrix) {
            case Matrix::A: return available_a_;
            case Matrix::B: return available_b_;
            case Matrix::C: return available_c_;
        }
        return 0;
    }

    /**
     * @brief Get capacity for a matrix
     */
    [[nodiscard]] size_t capacity(Matrix matrix) const {
        switch (matrix) {
            case Matrix::A: return capacity_a_;
            case Matrix::B: return capacity_b_;
            case Matrix::C: return capacity_c_;
        }
        return 0;
    }

    /**
     * @brief Get total available credits across all partitions
     */
    [[nodiscard]] size_t total_available() const {
        return available_a_ + available_b_ + available_c_;
    }

    /**
     * @brief Get total capacity across all partitions
     */
    [[nodiscard]] size_t total_capacity() const {
        return total_capacity_;
    }

    /**
     * @brief Reset all partitions to full capacity
     */
    void reset() {
        available_a_ = capacity_a_;
        available_b_ = capacity_b_;
        available_c_ = capacity_c_;
    }

private:
    size_t total_capacity_;
    size_t capacity_a_, capacity_b_, capacity_c_;
    size_t available_a_, available_b_, available_c_;
};

} // namespace sw::kpu::timing
