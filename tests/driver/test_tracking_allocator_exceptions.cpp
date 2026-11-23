// TrackingAllocator's strong exception guarantee (#497 part 2, review of #506).
//
// try_allocate must leave the allocator exactly as it was when it throws: an allocation of a
// map node, or the copy of a long label, can throw std::bad_alloc. Before the fix the tail path
// inserted its alignment padding and then threw on the allocation's own insertion, leaving a
// free block past next_free_; the next calls could then hand out one address twice.
//
// This file replaces the global operator new so a test can make the N-th allocation fail. It is
// its own test executable, because the replacement applies to the whole program.

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/allocator.hpp>

#include <cstdlib>
#include <new>
#include <set>
#include <string>

namespace {

// -1: never fail. N >= 0: the allocation after N more succeed throws std::bad_alloc.
thread_local long fail_countdown = -1;

}  // namespace

void* operator new(std::size_t size) {
    if (fail_countdown == 0) {
        fail_countdown = -1;
        throw std::bad_alloc();
    }
    if (fail_countdown > 0) --fail_countdown;
    if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
    throw std::bad_alloc();
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace sw::kpu;

namespace {

// A label long enough to defeat the small-string optimisation, so copying it allocates.
const std::string kLongLabel(200, 'x');

// Run `alloc` with the N-th allocation failing, for N = 0, 1, 2, ... until a run succeeds.
// After every failed run the allocator must be unchanged: consistent, with the same
// allocations and the same byte count.
template <typename Alloc>
void fail_each_allocation_in_turn(TrackingAllocator& allocator, Alloc alloc) {
    for (long n = 0; n < 16; ++n) {
        const auto before = allocator.get_all_allocations();
        const Size bytes_before = allocator.get_allocated_bytes();

        fail_countdown = n;
        bool threw = false;
        try {
            alloc();
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        fail_countdown = -1;

        INFO("failing allocation " << n);
        REQUIRE(allocator.is_consistent());
        if (!threw) return;  // every allocation the call makes has now been failed once

        const auto after = allocator.get_all_allocations();
        REQUIRE(after.size() == before.size());
        for (size_t i = 0; i < after.size(); ++i) {
            REQUIRE(after[i].address == before[i].address);
            REQUIRE(after[i].size == before[i].size);
        }
        REQUIRE(allocator.get_allocated_bytes() == bytes_before);
    }
    FAIL("the allocation never succeeded");
}

// No two live allocations overlap.
void require_disjoint(const TrackingAllocator& allocator) {
    Address end = 0;
    bool first = true;
    for (const auto& a : allocator.get_all_allocations()) {
        if (!first) REQUIRE(a.address >= end);
        end = a.address + a.size;
        first = false;
    }
}

}  // namespace

TEST_CASE("TrackingAllocator: a throwing tail allocation leaves no stray padding",
          "[allocator][exceptions]") {
    TrackingAllocator allocator(0, 8);
    REQUIRE(allocator.try_allocate(1, 1) == Address{0});

    // The tail path with padding: [1, 4) is padding, the allocation is [4, 5).
    fail_each_allocation_in_turn(allocator, [&] { allocator.try_allocate(1, 4, kLongLabel); });

    // The reviewer's sequence: these must never return an address that is already live.
    allocator.try_allocate(1, 1);
    allocator.try_allocate(3, 1);
    REQUIRE(allocator.is_consistent());
    require_disjoint(allocator);
}

TEST_CASE("TrackingAllocator: a throwing reuse of a free block leaves it whole",
          "[allocator][exceptions]") {
    TrackingAllocator allocator(0x2000, 8192);
    allocator.allocate(64, 64);                    // 0x2000
    Address a = allocator.allocate(256, 64);       // 0x2040
    allocator.allocate(64, 64);                    // 0x2140, keeps a off the tail
    REQUIRE(allocator.deallocate(a));

    // Reuse with a prefix and a suffix: [0x2040, 0x2080) free, [0x2080, 0x20c0) taken,
    // [0x20c0, 0x2140) free.
    fail_each_allocation_in_turn(allocator, [&] { allocator.try_allocate(64, 128, kLongLabel); });
    REQUIRE(allocator.is_allocated(0x2080));
    require_disjoint(allocator);

    // Reuse at the block's own start, splitting off a suffix.
    REQUIRE(allocator.deallocate(0x2080));
    fail_each_allocation_in_turn(allocator, [&] { allocator.try_allocate(32, 64, kLongLabel); });
    REQUIRE(allocator.is_allocated(0x2040));
    require_disjoint(allocator);
}
