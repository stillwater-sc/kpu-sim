// Platform-specific includes MUST come before namespace declaration
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#endif

#include <sw/memory/sparse_memory.hpp>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace sw::kpu::memory {

// ============================================================================
// Constructor
// ============================================================================

SparseMemory::SparseMemory(const Config& config)
    : config_(config) {

    if (config.virtual_size == 0) {
        throw std::invalid_argument("Virtual size must be greater than zero");
    }

    // Create the memory mapping (reserves virtual address space)
    MemoryMap::Config map_config(config.virtual_size);
    map_config.populate = false;  // Sparse allocation - don't commit pages yet
    map_config.page_size = config.page_size;

    map_ = std::make_unique<MemoryMap>(map_config);

    // Initialize thread safety if requested
    if (config_.thread_safe) {
        mutex_ = std::make_unique<std::mutex>();
    }

    // Reserve space for page tracking
    if (config_.track_pages) {
        Size estimated_pages = config.virtual_size / page_size();
        // Reserve buckets to avoid rehashing (use small fraction for sparse)
        accessed_pages_.reserve(std::min(estimated_pages / 100, Size(10000)));
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

Size SparseMemory::get_page_index(Address addr) const {
    return addr / page_size();
}

void SparseMemory::ensure_page_committed(Address addr) {
    // On Windows with MEM_RESERVE, we need to explicitly commit pages
    // On Unix with MAP_NORESERVE, pages are committed automatically on fault

#ifdef _WIN32
    // Calculate page-aligned address
    Size psize = page_size();
    Address page_addr = (addr / psize) * psize;

    // Try to commit the page
    // Note: This is handled automatically by the OS page fault handler,
    // but we can do it explicitly for better control
    void* base = map_->data();
    void* page_ptr = static_cast<char*>(base) + page_addr;

    // VirtualAlloc with MEM_COMMIT on an already-reserved region
    // This is a no-op if the page is already committed
    void* result = VirtualAlloc(
        page_ptr,
        psize,
        MEM_COMMIT,
        PAGE_READWRITE
    );

    if (result == nullptr) {
        DWORD error = GetLastError();
        throw std::runtime_error(
            "Failed to commit memory page at offset " +
            std::to_string(page_addr) + ", error: " + std::to_string(error)
        );
    }

    // Zero the page if requested
    if (config_.zero_on_access) {
        Size page_idx = get_page_index(addr);
        if (accessed_pages_.find(page_idx) == accessed_pages_.end()) {
            std::memset(page_ptr, 0, psize);
        }
    }
#else
    // On Unix systems, the OS automatically handles committing pages on the first
    // access (page fault). We just need to handle zeroing the page if requested.
    Size psize = page_size();
    void* base = map_->data();

    if (base == nullptr) {
        throw std::runtime_error("MemoryMap base pointer is null");
    }

    // Check if this is the first access to this page
    Size page_idx = get_page_index(addr);
    bool first_access = !config_.track_pages ||
                        (accessed_pages_.find(page_idx) == accessed_pages_.end());

    if (first_access && config_.zero_on_access) {
        // If this is the first time we're accessing this page and we need to zero it,
        // we can just write zeros to the whole page. This will trigger the page
        // fault and commit the page in a single, simple operation.
        Address page_addr = (addr / psize) * psize;
        void* page_ptr = static_cast<char*>(base) + page_addr;
        std::memset(page_ptr, 0, psize);
    }
    // If we don't need to zero the page, the first read/write access by the caller
    // will trigger the page fault and commit the page automatically. No extra
    // work is needed here.
#endif
}

void SparseMemory::track_page_access(Address addr) {
    if (!config_.track_pages) {
        return;
    }

    Size page_idx = get_page_index(addr);
    accessed_pages_.insert(page_idx);
}

// ============================================================================
// Public interface
// ============================================================================

void SparseMemory::read(Address addr, void* data, Size size) {
    if (!is_valid_range(addr, size)) {
        throw std::out_of_range("Read address out of bounds");
    }

    // Lock if thread-safe
    std::unique_lock<std::mutex> lock;
    if (mutex_) {
        lock = std::unique_lock<std::mutex>(*mutex_);
    }

    // Ensure all pages in the range are committed
    Size psize = page_size();
    Address start_page = (addr / psize) * psize;
    Address end_addr = addr + size;

    for (Address page_addr = start_page; page_addr < end_addr; page_addr += psize) {
        ensure_page_committed(page_addr);
        track_page_access(page_addr);
    }

    // Perform the read
    void* src = static_cast<char*>(map_->data()) + addr;
    std::memcpy(data, src, size);
}

void SparseMemory::write(Address addr, const void* data, Size size) {
    if (!is_valid_range(addr, size)) {
        throw std::out_of_range("Write address out of bounds");
    }

    // Lock if thread-safe
    std::unique_lock<std::mutex> lock;
    if (mutex_) {
        lock = std::unique_lock<std::mutex>(*mutex_);
    }

    // Ensure all pages in the range are committed
    Size psize = page_size();
    Address start_page = (addr / psize) * psize;
    Address end_addr = addr + size;

    for (Address page_addr = start_page; page_addr < end_addr; page_addr += psize) {
        ensure_page_committed(page_addr);
        track_page_access(page_addr);
    }

    // Perform the write
    void* dst = static_cast<char*>(map_->data()) + addr;
    std::memcpy(dst, data, size);
}

void* SparseMemory::get_pointer(Address addr) {
    if (!is_valid_address(addr)) {
        throw std::out_of_range("Address out of bounds");
    }

    // Lock if thread-safe
    std::unique_lock<std::mutex> lock;
    if (mutex_) {
        lock = std::unique_lock<std::mutex>(*mutex_);
    }

    // IMPORTANT: For sparse memory, we must ensure the page containing this
    // address is committed before returning a pointer to it. Otherwise,
    // accessing the pointer will cause access violations on Windows or
    // bus errors on Linux with MAP_NORESERVE.
    Size psize = page_size();
    Address page_addr = (addr / psize) * psize;
    ensure_page_committed(page_addr);
    track_page_access(page_addr);

    return static_cast<char*>(map_->data()) + addr;
}

const void* SparseMemory::get_pointer(Address addr) const {
    if (!is_valid_address(addr)) {
        throw std::out_of_range("Address out of bounds");
    }

    // For const access, we assume the page has already been committed
    // by a previous non-const operation. This is reasonable since you
    // typically write before reading.
    //
    // Note: If this assumption is violated, accessing the returned pointer
    // may cause access violations. Use read() instead for guaranteed safety.
    return static_cast<const char*>(map_->data()) + addr;
}

SparseMemory::Stats SparseMemory::get_stats() const {
    Stats stats;

    // Lock if thread-safe
    std::unique_lock<std::mutex> lock;
    if (mutex_) {
        lock = std::unique_lock<std::mutex>(*mutex_);
    }

    stats.virtual_size = config_.virtual_size;
    stats.page_size = page_size();

    if (config_.track_pages) {
        stats.accessed_pages = accessed_pages_.size();
    }

    // Get resident size from the underlying memory map
    MemoryMap::Stats map_stats = map_->get_stats();
    stats.resident_size = map_stats.resident_size;

    // Calculate utilization
    if (stats.virtual_size > 0) {
        stats.utilization = static_cast<double>(stats.resident_size) /
                          static_cast<double>(stats.virtual_size);
    }

    return stats;
}

Size SparseMemory::page_size() const {
    return map_->page_size();
}

void SparseMemory::prefault(Address addr, Size size) {
    if (!is_valid_range(addr, size)) {
        throw std::out_of_range("Prefault range out of bounds");
    }

    // Lock if thread-safe
    std::unique_lock<std::mutex> lock;
    if (mutex_) {
        lock = std::unique_lock<std::mutex>(*mutex_);
    }

    // Ensure all pages in the range are committed
    Size psize = page_size();
    Address start_page = (addr / psize) * psize;
    Address end_addr = addr + size;

    for (Address page_addr = start_page; page_addr < end_addr; page_addr += psize) {
        ensure_page_committed(page_addr);
        track_page_access(page_addr);
    }

    // Use the underlying memory map's prefault functionality
    map_->prefault(addr, size);
}

void SparseMemory::clear() {
    // Lock if thread-safe
    std::unique_lock<std::mutex> lock;
    if (mutex_) {
        lock = std::unique_lock<std::mutex>(*mutex_);
    }

    // For sparse memory, we only need to zero the pages that were actually accessed
    if (config_.track_pages && !accessed_pages_.empty()) {
        Size psize = page_size();
        for (Size page_idx : accessed_pages_) {
            Address page_addr = page_idx * psize;
            if (page_addr < config_.virtual_size) {
                Size clear_size = std::min<Size>(psize, config_.virtual_size - page_addr);
                std::memset(static_cast<char*>(map_->data()) + page_addr, 0, clear_size);
            }
        }
    } else if (!config_.track_pages) {
        // If we're not tracking pages, we need to zero all virtual space
        // This is expensive and should be avoided for very large sparse memory
        // For very large sizes, consider recreating the mapping instead
        if (config_.virtual_size > 1024ULL * 1024 * 1024) {  // > 1GB
            // For large sparse memory, just recreate the mapping
            // This is much faster than zeroing 100s of GB
            MemoryMap::Config map_config(config_.virtual_size);
            map_config.populate = false;
            map_config.page_size = config_.page_size;
            map_ = std::make_unique<MemoryMap>(map_config);
        } else {
            // For smaller sizes, zeroing is OK
            std::memset(map_->data(), 0, config_.virtual_size);
        }
    }

    // Clear the accessed pages tracking
    accessed_pages_.clear();
}

void SparseMemory::advise(Address addr, Size size, MemoryMap::Advice advice) {
    if (!is_valid_range(addr, size)) {
        throw std::out_of_range("Advise range out of bounds");
    }

    map_->advise(addr, size, advice);
}

} // namespace sw::kpu::memory
