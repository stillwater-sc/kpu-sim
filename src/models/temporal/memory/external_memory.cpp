#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>

#include <sw/memory/external_memory.hpp>

namespace sw::kpu {

// ============================================================================
// Helper functions
// ============================================================================

void ExternalMemory::initialize_backend(const Config& config) {
    capacity_ = config.capacity_mb * 1024ULL * 1024ULL;
    bandwidth_bytes_per_cycle_ = config.bandwidth_gbps * 1000000000ULL / 8ULL / 1000000000ULL;
    last_access_cycle_ = 0;

    // Determine which backend to use
    BackendType selected_backend = config.backend;

    if (config.auto_backend) {
        // Auto-select based on size
        if (config.capacity_mb >= SPARSE_THRESHOLD_MB) {
            selected_backend = BackendType::Sparse;
        } else {
            selected_backend = BackendType::Dense;
        }
    }

    active_backend_ = selected_backend;

    // Initialize the selected backend
    if (active_backend_ == BackendType::Sparse) {
        // Use sparse memory-mapped backend
        memory::SparseMemory::Config sparse_config(capacity_);
        sparse_config.zero_on_access = true;
        sparse_config.track_pages = true;
        sparse_config.thread_safe = true;

        sparse_storage_ = std::make_unique<memory::SparseMemory>(sparse_config);

        // Log the selection (optional, for debugging)
        #ifdef KPU_DEBUG_MEMORY
        std::cout << "ExternalMemory: Using SPARSE backend for "
                  << config.capacity_mb << " MB ("
                  << (capacity_ / (1024.0 * 1024.0 * 1024.0)) << " GB)\n";
        #endif
    } else {
        // Use dense vector-based backend (original behavior)
        dense_storage_.resize(capacity_);
        std::fill(dense_storage_.begin(), dense_storage_.end(), static_cast<uint8_t>(0));

        #ifdef KPU_DEBUG_MEMORY
        std::cout << "ExternalMemory: Using DENSE backend for "
                  << config.capacity_mb << " MB\n";
        #endif
    }
}

// ============================================================================
// Constructors
// ============================================================================

ExternalMemory::ExternalMemory(Size capacity_mb, Size bandwidth_gbps)
    : active_backend_(BackendType::Dense)
    , capacity_(0)
    , bandwidth_bytes_per_cycle_(0)
    , last_access_cycle_(0) {

    Config config(capacity_mb, bandwidth_gbps);
    initialize_backend(config);
}

ExternalMemory::ExternalMemory(const Config& config)
    : active_backend_(BackendType::Dense)
    , capacity_(0)
    , bandwidth_bytes_per_cycle_(0)
    , last_access_cycle_(0) {

    initialize_backend(config);
}

// ============================================================================
// Core operations
// ============================================================================

void ExternalMemory::read(Address addr, void* data, Size size) {
    if (addr + size > capacity_) {
        throw std::out_of_range("Memory read out of bounds");
    }

    if (active_backend_ == BackendType::Sparse) {
        sparse_storage_->read(addr, data, size);
    } else {
        std::memcpy(data, dense_storage_.data() + addr, size);
    }

    last_access_cycle_++;
}

void ExternalMemory::write(Address addr, const void* data, Size size) {
    if (addr + size > capacity_) {
        throw std::out_of_range("Memory write out of bounds");
    }

    if (active_backend_ == BackendType::Sparse) {
        sparse_storage_->write(addr, data, size);
    } else {
        std::memcpy(dense_storage_.data() + addr, data, size);
    }

    last_access_cycle_++;
}

bool ExternalMemory::is_ready() const {
    // Simplified: assume always ready for now
    return true;
}

void ExternalMemory::reset() {
    if (active_backend_ == BackendType::Sparse) {
        sparse_storage_->clear();
    } else {
        std::fill(dense_storage_.begin(), dense_storage_.end(), static_cast<uint8_t>(0));
    }

    last_access_cycle_ = 0;
}

// ============================================================================
// Statistics and information
// ============================================================================

ExternalMemory::Stats ExternalMemory::get_stats() const {
    Stats stats;
    stats.capacity_bytes = capacity_;
    stats.bandwidth_bps = bandwidth_bytes_per_cycle_ * 1000000000ULL;  // Assuming 1GHz
    stats.last_access_cycle = last_access_cycle_;
    stats.backend = active_backend_;

    if (active_backend_ == BackendType::Sparse) {
        memory::SparseMemory::Stats sparse_stats = sparse_storage_->get_stats();
        stats.resident_bytes = sparse_stats.resident_size;
        stats.utilization = sparse_stats.utilization;
    } else {
        // Dense backend: all memory is resident
        stats.resident_bytes = capacity_;
        stats.utilization = 1.0;
    }

    return stats;
}

const char* ExternalMemory::get_backend_name() const {
    switch (active_backend_) {
        case BackendType::Dense:
            return "Dense (std::vector)";
        case BackendType::Sparse:
            return "Sparse (memory-mapped)";
        default:
            return "Unknown";
    }
}

} // namespace sw::kpu