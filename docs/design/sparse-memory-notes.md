# Sparse Memory design notes

## Why This Works

  - std::vector can hold move-only types (types that can be moved but not copied)
  - By explicitly making both classes movable, std::vector can use move semantics when resizing or reordering
  elements
  - The = delete declarations prevent accidental copying
  - The = default declarations let the compiler generate efficient move operations

  Best Practices Followed

  1. Rule of Five: When you define/delete any of the five special member functions (destructor, copy constructor,
  copy assignment, move constructor, move assignment), you should consider all five.
  2. Move-only types: Types that manage unique resources should be move-only, not copyable.
  3. Explicit declarations: Even when = default, being explicit about copy/move semantics makes the intent clear.

## Summary: Windows Memory Mapping Segfault Fix

  Problem

  Tests were crashing with segfaults on Windows when trying to directly access memory-mapped regions.

  Root Cause

  On Windows, when using VirtualAlloc with MEM_RESERVE only (without MEM_COMMIT), the memory pages are reserved but
  not committed. Attempting to read or write to reserved-but-not-committed memory causes an access violation
  (segfault).

  Unlike Unix systems where mmap with MAP_NORESERVE allows the page fault handler to automatically commit pages on
  first access, Windows requires explicit commitment via VirtualAlloc(..., MEM_COMMIT, ...).

  Key Differences: Windows vs. Unix

  | Aspect               | Windows (VirtualAlloc)                | Unix (mmap)                 |
  |----------------------|---------------------------------------|-----------------------------|
  | Reserve only         | MEM_RESERVE                           | MAP_NORESERVE               |
  | Auto-commit on fault | No - access violation                 | Yes - kernel commits page   |
  | Explicit commit      | Required via MEM_COMMIT               | Not needed                  |
  | Mixed usage          | Can combine: MEM_RESERVE | MEM_COMMIT | Single mmap call            |

  Solution Implemented

  1. Config Flag: populate

  Added populate flag to control memory commitment:
  // Memory committed immediately (for direct access)
  config.populate = true;   // Windows: MEM_RESERVE | MEM_COMMIT

  // Memory reserved only (for sparse/on-demand)
  config.populate = false;  // Windows: MEM_RESERVE only

  2. Fixed Tests

  Updated all tests that directly access memory to use populate = true:
  // Before (segfault on Windows)
  MemoryMap::Config config(1024 * 1024);
  MemoryMap map(config);
  std::memcpy(map.data(), &value, sizeof(value));  // Access violation!

  // After (works on Windows)
  MemoryMap::Config config(1024 * 1024);
  config.populate = true;  // Commit memory for direct access
  MemoryMap map(config);
  std::memcpy(map.data(), &value, sizeof(value));  // Works!

  3. SparseMemory Handles Commitment

  For sparse memory (large allocations), SparseMemory::ensure_page_committed() explicitly commits pages on-demand:
  #ifdef _WIN32
      void* result = VirtualAlloc(
          page_ptr,
          page_size,
          MEM_COMMIT,      // Commit this specific page
          PAGE_READWRITE
      );
  #endif

  Architecture

  ┌─────────────────────────────────────────────────────┐
  │                             User Code                                   │
  └──────────────────┬──────────────────────────────────┘
                            │
       ┌──────────────┴──────────────┐
       │                                        │
       ▼                                       ▼
  ┌────────────────┐    ┌──────────────────────┐
  │      MemoryMap       │    │        SparseMemory          │
  │      (populate=T)    │    │        (populate=F)          │
  ├────────────────┤    ├──────────────────────┤
  │     Direct access    │    │     ensure_page_committed    │
  │     Tests commit     │    │      commits on-demand       │
  │     all upfront      │    │          per page            │
  └────────────────┘    └──────────────────────┘
           │                                │
           └───────────┬───────────┘
                           ▼
              ┌─────────────────┐
              │     VirtualAlloc      │
              │     (Windows API)     │
              └─────────────────┘
