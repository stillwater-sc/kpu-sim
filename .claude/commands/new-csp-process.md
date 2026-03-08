Generate a new CSP process component: $ARGUMENTS

Follow the established KPU CSP process pattern:

1. Read these reference implementations first:
   - include/sw/kpu/timing/dma_engine_process.hpp (submit/poll with MC)
   - include/sw/kpu/timing/block_mover_process.hpp (tag match + credit)
   - include/sw/kpu/timing/streamer_process.hpp (feed/drain with compute)

2. Generate the header in include/sw/kpu/timing/ with:
   - Config struct with display_name() method
   - Constructor taking credit pools and tag CAMs as references (not owned)
   - IProcess interface: tick(), is_idle(), has_pending_work(), id(), name(), reset()
   - Schedule methods for the component's operations
   - Private state machine with clear state transitions
   - Statistics accessors
   - is_complete() method

3. Generate a test file in tests/timing/ with:
   - Construction and default state tests
   - Single-operation happy path tests
   - Multi-operation sequencing tests
   - Stall/backpressure tests (no credits, no tag match)
   - Credit conservation checks (initial == final after full cycle)

4. Update tests/timing/CMakeLists.txt to include the new test.

5. Build and verify: cmake --build --preset release

CRITICAL RULES (from CLAUDE.md):
- Use credit-based push semantics, NEVER fetch-on-demand
- Use buffer/credit terminology, NEVER cache terminology
- Component must WAIT for downstream credit before pushing
- Component must RETURN credit upstream after consuming
- Tag CAM insert on arrival, match before consume, invalidate after move
