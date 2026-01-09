#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/memory/memory_map.hpp>
#include <cstring>
#include <vector>
#include <thread>

using namespace sw::kpu::memory;
using namespace sw::kpu;

TEST_CASE("MemoryMap basic functionality", "[memory][memmap]") {
    SECTION("Create and destroy mapping") {
        INFO("Testing basic memory map creation with 4KB allocation");
        MemoryMap::Config config(4096);  // 4KB
        config.populate = true;  // Commit memory for direct access
        MemoryMap map(config);

        INFO("Allocation size: 4096 bytes");
        INFO("Page size: " << map.page_size() << " bytes");

        REQUIRE(map.is_mapped());
        REQUIRE(map.size() == 4096);
        REQUIRE(map.data() != nullptr);
    }

    SECTION("Get system page size") {
        Size page_size = MemoryMap::get_system_page_size();

        INFO("System page size: " << page_size << " bytes");
        INFO("Common page sizes: 4KB (x86), 16KB (Apple Silicon), 64KB (some ARM)");

        REQUIRE(page_size > 0);
        // Typical page sizes are 4KB, 8KB, 16KB
        REQUIRE(page_size >= 4096);
    }

    SECTION("Write and read simple data") {
        Size size = 1024 * 1024;  // 1MB
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory for direct access
        MemoryMap map(config);

        INFO("Allocation size: " << size << " bytes (1MB)");
        INFO("Testing write/read of 32-bit value at offset 0");

        // Write some data
        uint32_t test_value = 0xDEADBEEF;
        std::memcpy(map.data(), &test_value, sizeof(test_value));

        // Read it back
        uint32_t read_value = 0;
        std::memcpy(&read_value, map.data(), sizeof(read_value));

        INFO("Written: 0x" << std::hex << test_value);
        INFO("Read back: 0x" << std::hex << read_value);

        REQUIRE(read_value == test_value);
    }

    SECTION("Write and read at different offsets") {
        Size size = 1024 * 1024;  // 1MB
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory for direct access
        MemoryMap map(config);

        INFO("Allocation size: " << size << " bytes (1MB)");
        INFO("Testing write/read of 64-bit patterns at 10 offsets (1KB apart)");

        // Write pattern at different offsets
        for (Size offset = 0; offset < 10; ++offset) {
            uint64_t pattern = 0x0123456789ABCDEF + offset;
            std::memcpy(static_cast<char*>(map.data()) + offset * 1024,
                       &pattern, sizeof(pattern));
        }

        // Read back and verify
        for (Size offset = 0; offset < 10; ++offset) {
            uint64_t expected = 0x0123456789ABCDEF + offset;
            uint64_t actual = 0;
            std::memcpy(&actual,
                       static_cast<char*>(map.data()) + offset * 1024,
                       sizeof(actual));

            INFO("Offset " << offset << ": byte offset " << (offset * 1024));
            INFO("Expected: 0x" << std::hex << expected);
            INFO("Actual: 0x" << std::hex << actual);

            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("MemoryMap sparse allocation", "[memory][memmap][sparse]") {
    // NOTE: MAP_NORESERVE sparse allocation with on-demand page faulting does not
    // work reliably on WSL. The prefault() operation (which tries to touch pages
    // to trigger page faults) causes bus errors on WSL.
    //
    // This test works on:
    // - Native Windows (using VirtualAlloc with MEM_RESERVE + explicit MEM_COMMIT in prefault)
    // - Native Linux (using mmap with on-demand paging)
    //
    // It should NOT run on WSL due to known bus errors with MAP_NORESERVE.

#ifdef __linux__
    // Check if we're running under WSL by looking for "microsoft" in uname
    std::string uname_output;
    {
        FILE* pipe = popen("uname -r", "r");
        if (pipe) {
            char buffer[128];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                uname_output += buffer;
            }
            pclose(pipe);
        }
    }

    // Skip if running under WSL
    if (uname_output.find("microsoft") != std::string::npos ||
        uname_output.find("Microsoft") != std::string::npos ||
        uname_output.find("WSL") != std::string::npos) {
        WARN("Sparse MemoryMap tests skipped on WSL - use SparseMemory wrapper instead");
        return;
    }
#endif

    SECTION("Large virtual size with minimal physical usage") {
        // Request 1GB virtual space - this demonstrates sparse allocation well:
        // - Virtual reservation: 1GB (trivial on 64-bit systems)
        // - Physical commitment: Only the pages we actually touch (~16-64KB depending on page size)
        // - This is the key benefit: massive virtual address space with minimal RAM usage
        Size virtual_size = 1ULL * 1024ULL * 1024ULL * 1024ULL;  // 1GB
        MemoryMap::Config config(virtual_size);
        config.populate = false;  // Sparse allocation

        INFO("Testing sparse allocation: 1GB virtual, minimal physical");
        INFO("Virtual size: " << virtual_size << " bytes (1GB)");

        MemoryMap map(config);

        Size page_size = map.page_size();
        Size total_pages = virtual_size / page_size;

        INFO("Page size: " << page_size << " bytes");
        INFO("Total pages in virtual space: " << total_pages);

        REQUIRE(map.is_mapped());
        REQUIRE(map.size() == virtual_size);

        // Write to a few pages scattered across the address space
        // We're touching maybe 4-16 pages out of 262,144 pages (for 4KB pages)
        // Physical memory used: ~16-64KB out of 1GB reserved
        std::vector<Size> test_offsets = {
            0,
            page_size * 10,
            page_size * 100,
            page_size * 1000,
            page_size * 10000,
            page_size * 100000
        };

        INFO("Testing " << test_offsets.size() << " scattered write locations");
        INFO("Physical memory touched: ~" << (test_offsets.size() * page_size / 1024) << " KB");

        // IMPORTANT: For sparse (non-populated) memory, we MUST prefault pages
        // before writing to them
        for (Size offset : test_offsets) {
            if (offset + sizeof(uint64_t) > virtual_size) continue;
            // Prefault the page containing this offset
            map.prefault(offset, page_size);
        }

        for (Size offset : test_offsets) {
            if (offset + sizeof(uint64_t) > virtual_size) continue;
            uint64_t marker = 0xFEEDFACE00000000ULL + offset;
            std::memcpy(static_cast<char*>(map.data()) + offset,
                       &marker, sizeof(marker));
        }

        // Verify the writes
        for (Size offset : test_offsets) {
            if (offset + sizeof(uint64_t) > virtual_size) continue;
            uint64_t expected = 0xFEEDFACE00000000ULL + offset;
            uint64_t actual = 0;
            std::memcpy(&actual,
                       static_cast<char*>(map.data()) + offset,
                       sizeof(actual));

            INFO("Offset: " << offset << " (page " << (offset / page_size) << ")");
            INFO("Expected: 0x" << std::hex << expected);
            INFO("Actual: 0x" << std::hex << actual);

            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("MemoryMap move semantics", "[memory][memmap]") {
    SECTION("Move constructor") {
        INFO("Testing move constructor: data should transfer, source becomes invalid");

        MemoryMap::Config config(4096);
        config.populate = true;  // Commit memory for direct access
        MemoryMap map1(config);

        // Write data to first map
        uint32_t test_val = 0xCAFEBABE;
        std::memcpy(map1.data(), &test_val, sizeof(test_val));

        void* original_ptr = map1.data();
        INFO("Original pointer: " << original_ptr);
        INFO("Test value: 0x" << std::hex << test_val);

        // Move to second map
        MemoryMap map2(std::move(map1));

        INFO("After move - map2 pointer: " << map2.data());
        INFO("After move - map1.is_mapped(): " << map1.is_mapped());

        REQUIRE(map2.is_mapped());
        REQUIRE(!map1.is_mapped());

        // Verify data is preserved
        uint32_t read_val = 0;
        std::memcpy(&read_val, map2.data(), sizeof(read_val));

        INFO("Read value from moved map: 0x" << std::hex << read_val);

        REQUIRE(read_val == test_val);
    }

    SECTION("Move assignment") {
        INFO("Testing move assignment: target releases old memory, takes source");

        MemoryMap::Config config1(4096);
        config1.populate = true;  // Commit memory for direct access
        MemoryMap map1(config1);

        uint32_t test_val1 = 0xDEADBEEF;
        std::memcpy(map1.data(), &test_val1, sizeof(test_val1));

        MemoryMap::Config config2(8192);
        config2.populate = true;  // Commit memory for direct access
        MemoryMap map2(config2);

        INFO("map1 size before move: " << map1.size());
        INFO("map2 size before move: " << map2.size());

        // Move assign
        map2 = std::move(map1);

        INFO("map2 size after move: " << map2.size());
        INFO("map1.is_mapped() after move: " << map1.is_mapped());

        REQUIRE(map2.is_mapped());
        REQUIRE(!map1.is_mapped());
        REQUIRE(map2.size() == 4096);

        // Verify data
        uint32_t read_val = 0;
        std::memcpy(&read_val, map2.data(), sizeof(read_val));
        REQUIRE(read_val == test_val1);
    }
}

TEST_CASE("MemoryMap statistics", "[memory][memmap][stats]") {
    SECTION("Get statistics") {
        Size size = 1024 * 1024;  // 1MB
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory
        MemoryMap map(config);

        MemoryMap::Stats stats = map.get_stats();

        INFO("Requested size: " << size << " bytes (1MB)");
        INFO("Stats - virtual_size: " << stats.virtual_size << " bytes");
        INFO("Stats - resident_size: " << stats.resident_size << " bytes");
        INFO("Stats - page_size: " << stats.page_size << " bytes");
        INFO("Stats - page_faults: " << stats.page_faults);

        REQUIRE(stats.virtual_size == size);
        REQUIRE(stats.page_size > 0);
        REQUIRE(stats.resident_size <= size);
    }
}

TEST_CASE("MemoryMap advice and prefaulting", "[memory][memmap][advice]") {
    SECTION("Prefault pages") {
        Size size = 4 * 1024 * 1024;  // 4MB (enough for 100 pages even with 16KB pages)
        MemoryMap::Config config(size);
        config.populate = false;  // Sparse

        MemoryMap map(config);

        // Prefault first 100 pages (or fewer if that exceeds size)
        Size page_size = map.page_size();
        Size num_pages = size / page_size;
        Size pages_to_prefault = std::min(static_cast<Size>(100), num_pages);
        Size prefault_size = pages_to_prefault * page_size;

        INFO("Allocation size: " << size << " bytes (" << (size / 1024 / 1024) << " MB)");
        INFO("System page size: " << page_size << " bytes");
        INFO("Total pages in allocation: " << num_pages);
        INFO("Pages to prefault: " << pages_to_prefault);
        INFO("Prefault size: " << prefault_size << " bytes");

        REQUIRE_NOTHROW(map.prefault(0, prefault_size));
    }

    SECTION("Memory advice hints") {
        Size size = 1024 * 1024;  // 1MB
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory
        MemoryMap map(config);

        INFO("Testing madvise() hints on 1MB mapping");
        INFO("These hints help the OS optimize paging behavior");

        // These should not throw
        INFO("Advice::Sequential - hint for sequential access pattern");
        REQUIRE_NOTHROW(map.advise(0, size, MemoryMap::Advice::Sequential));

        INFO("Advice::Random - hint for random access pattern");
        REQUIRE_NOTHROW(map.advise(0, size, MemoryMap::Advice::Random));

        INFO("Advice::WillNeed - hint to prefetch pages");
        REQUIRE_NOTHROW(map.advise(0, size, MemoryMap::Advice::WillNeed));
    }
}

TEST_CASE("MemoryMap error handling", "[memory][memmap][error]") {
    SECTION("Zero size mapping") {
        INFO("Testing that zero-size allocation throws std::invalid_argument");
        MemoryMap::Config config(0);
        REQUIRE_THROWS_AS(MemoryMap(config), std::invalid_argument);
    }

    SECTION("Out of range advice") {
        Size size = 4096;
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory
        MemoryMap map(config);

        INFO("Allocation size: " << size << " bytes");
        INFO("Testing advise() with length " << (size + 1) << " (exceeds allocation)");

        // Advice beyond mapping size should throw
        REQUIRE_THROWS_AS(map.advise(0, size + 1, MemoryMap::Advice::Normal),
                         std::out_of_range);
    }
}

TEST_CASE("MemoryMap concurrent access", "[memory][memmap][concurrent]") {
    SECTION("Multiple threads writing to different pages") {
        // Use larger size to accommodate systems with 16KB pages (Apple Silicon)
        Size size = 4 * 1024 * 1024;  // 4MB
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory for direct access
        MemoryMap map(config);

        Size page_size = map.page_size();
        const int num_threads = 4;
        const int writes_per_thread = 10;

        // Space threads far enough apart to avoid overlap
        // Each thread needs writes_per_thread * page_size bytes
        // Use 16 * page_size spacing to be safe (at least 256KB with 16KB pages)
        Size thread_spacing = page_size * 16;
        Size space_per_thread = writes_per_thread * page_size;

        INFO("=== Concurrent Write Test Configuration ===");
        INFO("Allocation size: " << size << " bytes (" << (size / 1024 / 1024) << " MB)");
        INFO("System page size: " << page_size << " bytes");
        INFO("Number of threads: " << num_threads);
        INFO("Writes per thread: " << writes_per_thread);
        INFO("Space needed per thread: " << space_per_thread << " bytes");
        INFO("Thread spacing: " << thread_spacing << " bytes");
        INFO("Total space needed: " << (num_threads * thread_spacing) << " bytes");

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&map, page_size, thread_spacing, i]() {
                // Each thread writes to its own set of pages
                Size base_offset = i * thread_spacing;
                for (int j = 0; j < writes_per_thread; ++j) {
                    uint64_t value = (static_cast<uint64_t>(i) << 32) | j;
                    Size offset = base_offset + j * page_size;
                    if (offset + sizeof(value) <= map.size()) {
                        std::memcpy(static_cast<char*>(map.data()) + offset,
                                   &value, sizeof(value));
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        INFO("All threads completed, verifying writes...");

        // Verify all writes
        for (int i = 0; i < num_threads; ++i) {
            Size base_offset = i * thread_spacing;
            for (int j = 0; j < writes_per_thread; ++j) {
                uint64_t expected = (static_cast<uint64_t>(i) << 32) | j;
                uint64_t actual = 0;
                Size offset = base_offset + j * page_size;
                if (offset + sizeof(actual) <= size) {
                    std::memcpy(&actual,
                               static_cast<char*>(map.data()) + offset,
                               sizeof(actual));

                    INFO("Thread " << i << ", write " << j);
                    INFO("Offset: " << offset << " bytes (page " << (offset / page_size) << ")");
                    INFO("Expected: 0x" << std::hex << expected << " (thread=" << std::dec << i << ", j=" << j << ")");
                    INFO("Actual: 0x" << std::hex << actual);

                    REQUIRE(actual == expected);
                }
            }
        }
    }
}
