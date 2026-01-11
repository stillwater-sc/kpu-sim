// Platform-specific includes MUST come before namespace declaration
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
    #include <psapi.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <cerrno>
    #ifdef __APPLE__
        #include <mach/mach.h>
        #include <mach/vm_statistics.h>
    #endif
#endif

#include <sw/memory/memory_map.hpp>
#include <cstring>
#include <algorithm>
#include <vector>

namespace sw::kpu::memory {

// ============================================================================
// Platform-specific utility functions
// ============================================================================

#ifdef _WIN32
// Windows implementation

static DWORD get_protection_flags(bool read_only) {
    return read_only ? PAGE_READONLY : PAGE_READWRITE;
}

Size MemoryMap::get_system_page_size() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

bool MemoryMap::has_huge_pages() {
    // Check for large page support
    SIZE_T large_page_min = GetLargePageMinimum();
    return large_page_min > 0;
}

#else
// Unix/Linux/MacOS implementation

static int get_protection_flags(bool read_only) {
    return read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
}

Size MemoryMap::get_system_page_size() {
    return static_cast<Size>(sysconf(_SC_PAGESIZE));
}

bool MemoryMap::has_huge_pages() {
#ifdef __linux__
    // Check for transparent huge pages
    return access("/sys/kernel/mm/transparent_hugepage/enabled", F_OK) == 0;
#elif defined(__APPLE__)
    // macOS has superpage support but it's automatic
    return true;
#else
    return false;
#endif
}

#endif

// ============================================================================
// MemoryMap implementation
// ============================================================================

void MemoryMap::platform_init() {
    page_size_ = get_system_page_size();
#ifdef _WIN32
    file_mapping_handle_ = nullptr;
#else
    fd_ = -1;
#endif
}

void* MemoryMap::platform_map(const Config& config) {
    if (config.size == 0) {
        throw std::invalid_argument("Cannot create zero-size memory mapping");
    }

#ifdef _WIN32
    // ========================================================================
    // Windows implementation using VirtualAlloc for anonymous mapping
    // ========================================================================

    // For Windows:
    // - config.populate = true: Use MEM_RESERVE | MEM_COMMIT (allocate all upfront)
    // - config.populate = false: Use MEM_RESERVE only (sparse, for SparseMemory wrapper)
    //
    // NOTE: If using MEM_RESERVE only, you MUST call VirtualAlloc(MEM_COMMIT) on
    // specific pages before accessing them, or you'll get access violations.
    // The SparseMemory class handles this in ensure_page_committed().
    DWORD alloc_type = config.populate ? (MEM_RESERVE | MEM_COMMIT) : MEM_RESERVE;
    DWORD protect = get_protection_flags(config.read_only);

    void* ptr = VirtualAlloc(
        nullptr,              // Let system choose address
        config.size,          // Size of region
        alloc_type,           // Reserve or reserve+commit
        protect               // Protection flags
    );

    if (ptr == nullptr) {
        DWORD error = GetLastError();
        throw std::runtime_error(
            "VirtualAlloc failed: error code " + std::to_string(error)
        );
    }

    return ptr;

#else
    // ========================================================================
    // Unix/Linux/MacOS implementation using mmap for anonymous mapping
    // ========================================================================

    int prot = get_protection_flags(config.read_only);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;

    // For sparse allocation on some systems, MAP_NORESERVE can be used
    // to avoid committing swap space. However, this can lead to SIGBUS
    // errors if the system runs out of memory. For greater stability,
    // we will not use MAP_NORESERVE by default. The OS will still use
    // demand paging to allocate physical pages on first access.
    if (!config.populate) {
        // flags |= MAP_NORESERVE; // Disabled for stability
    }

#ifdef __APPLE__
    // macOS: Use VM_FLAGS_SUPERPAGE_SIZE_ANY for potential huge page support
    // This is a hint; the kernel will decide
    if (!config.populate && config.size >= 2 * 1024 * 1024) {
        // Only for large allocations
        flags |= MAP_NORESERVE;
    }
#endif

#ifdef __linux__
    // Linux: Use MAP_POPULATE to prefault pages if requested
    if (config.populate) {
        flags |= MAP_POPULATE;
    }

    // For very large allocations, hint that we want huge pages
    if (config.size >= 2 * 1024 * 1024) {  // >= 2MB
#ifdef MAP_HUGETLB
        // Try with huge pages first, fall back if it fails
        int huge_flags = flags | MAP_HUGETLB;
        void* ptr = mmap(nullptr, config.size, prot, huge_flags, -1, 0);
        if (ptr != MAP_FAILED) {
            return ptr;
        }
        // Fall through to regular mmap
#endif
    }
#endif

    void* ptr = mmap(
        nullptr,              // Let system choose address
        config.size,          // Size of region
        prot,                 // Protection flags
        flags,                // Mapping flags
        -1,                   // Anonymous mapping (no fd)
        0                     // Offset (not used for anonymous)
    );

    if (ptr == MAP_FAILED) {
        throw std::runtime_error(
            "mmap failed: " + std::string(std::strerror(errno))
        );
    }

    return ptr;
#endif
}

void MemoryMap::platform_unmap() {
    if (!is_mapped_ || base_ptr_ == nullptr) {
        return;
    }

#ifdef _WIN32
    // Windows: Use VirtualFree to release the memory
    if (!VirtualFree(base_ptr_, 0, MEM_RELEASE)) {
        // Log error but don't throw in destructor
        DWORD error = GetLastError();
        (void)error; // Suppress unused variable warning
    }

    if (file_mapping_handle_ != nullptr) {
        CloseHandle(file_mapping_handle_);
        file_mapping_handle_ = nullptr;
    }
#else
    // Unix: Use munmap
    if (munmap(base_ptr_, size_) != 0) {
        // Log error but don't throw in destructor
        int error = errno;
        (void)error; // Suppress unused variable warning
    }

    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
#endif

    base_ptr_ = nullptr;
    is_mapped_ = false;
}

Size MemoryMap::platform_get_resident_size() const {
    if (!is_mapped_) {
        return 0;
    }

#ifdef _WIN32
    // Windows: Use QueryWorkingSetEx to get resident pages
    // This is more complex and requires iterating through pages
    // For simplicity, we'll use VirtualQuery

    Size resident = 0;
    Size offset = 0;

    while (offset < size_) {
        MEMORY_BASIC_INFORMATION mbi;
        SIZE_T result = VirtualQuery(
            static_cast<char*>(base_ptr_) + offset,
            &mbi,
            sizeof(mbi)
        );

        if (result == 0) {
            break;
        }

        // Count committed pages
        if (mbi.State == MEM_COMMIT) {
            resident += mbi.RegionSize;
        }

        offset += mbi.RegionSize;
    }

    return resident;

#elif defined(__linux__)
    // Linux: Use mincore to check which pages are resident
    // Note: On WSL and some systems, mincore may not work correctly with
    // large sparse MAP_NORESERVE mappings. We'll limit our check to smaller sizes.

    Size num_pages = (size_ + page_size_ - 1) / page_size_;

    // For very large mappings (> 256MB), don't bother with mincore
    // as it can be slow and may not work reliably with MAP_NORESERVE
    const Size MAX_SIZE_FOR_MINCORE = 256ULL * 1024 * 1024;  // 256MB

    if (size_ > MAX_SIZE_FOR_MINCORE) {
        // For very large mappings, just return 0
        // The caller should use page tracking instead of resident size queries
        return 0;
    }

    std::vector<unsigned char> vec;
    try {
        vec.resize(num_pages);
    } catch (const std::bad_alloc&) {
        // Can't allocate vector for page tracking
        return 0;
    }

    int result = mincore(base_ptr_, size_, vec.data());
    if (result == 0) {
        Size resident = 0;
        for (Size i = 0; i < num_pages; ++i) {
            if (vec[i] & 1) {  // Page is resident
                resident += page_size_;
            }
        }
        return resident;
    }

    // mincore failed - this can happen with MAP_NORESERVE on some systems
    // Return 0 rather than throwing
    return 0;

#elif defined(__APPLE__)
    // macOS: Use mincore similar to Linux
    Size num_pages = (size_ + page_size_ - 1) / page_size_;
    std::vector<char> vec(num_pages);

    if (mincore(base_ptr_, size_, vec.data()) == 0) {
        Size resident = 0;
        for (Size i = 0; i < num_pages; ++i) {
            if (vec[i] & MINCORE_INCORE) {  // Page is resident
                resident += page_size_;
            }
        }
        return resident;
    }

    return 0;  // Failed to query
#else
    return 0;  // Not implemented for this platform
#endif
}

// ============================================================================
// Constructor and destructor
// ============================================================================

MemoryMap::MemoryMap(const Config& config)
    : base_ptr_(nullptr)
    , size_(config.size)
    , is_mapped_(false)
    , page_size_(0) {

    platform_init();
    base_ptr_ = platform_map(config);
    is_mapped_ = true;
}

MemoryMap::~MemoryMap() {
    platform_unmap();
}

// ============================================================================
// Move semantics
// ============================================================================

MemoryMap::MemoryMap(MemoryMap&& other) noexcept
    : base_ptr_(other.base_ptr_)
    , size_(other.size_)
    , is_mapped_(other.is_mapped_)
    , page_size_(other.page_size_)
#ifdef _WIN32
    , file_mapping_handle_(other.file_mapping_handle_)
#else
    , fd_(other.fd_)
#endif
{
    other.base_ptr_ = nullptr;
    other.is_mapped_ = false;
#ifdef _WIN32
    other.file_mapping_handle_ = nullptr;
#else
    other.fd_ = -1;
#endif
}

MemoryMap& MemoryMap::operator=(MemoryMap&& other) noexcept {
    if (this != &other) {
        platform_unmap();

        base_ptr_ = other.base_ptr_;
        size_ = other.size_;
        is_mapped_ = other.is_mapped_;
        page_size_ = other.page_size_;
#ifdef _WIN32
        file_mapping_handle_ = other.file_mapping_handle_;
        other.file_mapping_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif

        other.base_ptr_ = nullptr;
        other.is_mapped_ = false;
    }
    return *this;
}

// ============================================================================
// Public interface methods
// ============================================================================

MemoryMap::Stats MemoryMap::get_stats() const {
    Stats stats;
    stats.virtual_size = size_;
    stats.resident_size = platform_get_resident_size();
    stats.page_size = page_size_;
    stats.page_faults = 0;  // Not easily available cross-platform
    return stats;
}

void MemoryMap::advise(Size offset, Size length, Advice advice) {
    if (!is_mapped_) {
        throw std::runtime_error("Cannot advise unmapped memory");
    }

    if (offset + length > size_) {
        throw std::out_of_range("Advice region out of bounds");
    }

#ifdef _WIN32
    // Windows: Limited support for memory advice
    // We can use PrefetchVirtualMemory for WillNeed
    if (advice == Advice::WillNeed) {
        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = static_cast<char*>(base_ptr_) + offset;
        entry.NumberOfBytes = length;

        // PrefetchVirtualMemory is Windows 8+ only
        #if defined(NTDDI_VERSION) && NTDDI_VERSION >= NTDDI_WIN8
        PrefetchVirtualMemory(
            GetCurrentProcess(),
            1,
            &entry,
            0
        );
        #endif
    }
    // Other advice types not well supported on Windows

#else
    // Unix: Use madvise
    int posix_advice;
    switch (advice) {
        case Advice::Normal:
            posix_advice = MADV_NORMAL;
            break;
        case Advice::Sequential:
            posix_advice = MADV_SEQUENTIAL;
            break;
        case Advice::Random:
            posix_advice = MADV_RANDOM;
            break;
        case Advice::WillNeed:
            posix_advice = MADV_WILLNEED;
            break;
        case Advice::DontNeed:
            posix_advice = MADV_DONTNEED;
            break;
        default:
            posix_advice = MADV_NORMAL;
    }

    madvise(
        static_cast<char*>(base_ptr_) + offset,
        length,
        posix_advice
    );
#endif
}

void MemoryMap::prefault(Size offset, Size length) {
    if (!is_mapped_) {
        throw std::runtime_error("Cannot prefault unmapped memory");
    }

    if (offset + length > size_) {
        throw std::out_of_range("Prefault region out of bounds");
    }

#ifdef _WIN32
    // Windows: Commit the pages explicitly
    void* addr = static_cast<char*>(base_ptr_) + offset;
    void* result = VirtualAlloc(
        addr,
        length,
        MEM_COMMIT,
        PAGE_READWRITE
    );

    if (result == nullptr) {
        throw std::runtime_error("Failed to commit pages");
    }

    // Touch each page to ensure it's faulted in
    volatile char* ptr = static_cast<char*>(result);
    for (Size i = 0; i < length; i += page_size_) {
        ptr[i] = ptr[i];  // Read to fault in the page
    }

#else
    // Unix: Touch each page to force page faults
    // Alternatively, use mlock to lock pages in memory
    volatile char* ptr = static_cast<char*>(base_ptr_) + offset;
    for (Size i = 0; i < length; i += page_size_) {
        ptr[i] = ptr[i];  // Read to fault in the page
    }

    // Optionally lock pages in memory (requires privileges)
    // mlock(static_cast<char*>(base_ptr_) + offset, length);
#endif
}

void MemoryMap::sync(Size offset, Size length, bool async) {
    if (!is_mapped_) {
        throw std::runtime_error("Cannot sync unmapped memory");
    }

    if (offset + length > size_) {
        throw std::out_of_range("Sync region out of bounds");
    }

#ifdef _WIN32
    // Windows: Use FlushViewOfFile for mapped files
    // For anonymous mappings (VirtualAlloc), this is a no-op
    // since there's no backing store
    (void)async;  // Not used for anonymous mappings

#else
    // Unix: Use msync for file-backed mappings
    if (fd_ != -1) {
        int flags = async ? MS_ASYNC : MS_SYNC;
        msync(
            static_cast<char*>(base_ptr_) + offset,
            length,
            flags
        );
    }
#endif
}

} // namespace sw::kpu::memory
