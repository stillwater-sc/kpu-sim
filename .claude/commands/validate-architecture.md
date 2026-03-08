Audit the codebase for credit-based dataflow violations.

Search for these anti-patterns in include/sw/kpu/timing/ and tests/timing/:

1. **Cache terminology** (FORBIDDEN):
   Search for: cache, hit, miss, evict, LRU, refetch
   Exclude legitimate uses: row_hit, row_miss, tag_cam, cache_model (deprecated files)
   Each remaining occurrence is a violation.

2. **Fetch-on-demand patterns**:
   Search for: request.*response, poll.*data, fetch, demand
   Check if any component pulls data instead of waiting for push with credit.

3. **Missing credit checks**:
   For each process's tick() method, verify:
   - Producer checks credit.acquire() before pushing downstream
   - Consumer calls credit.release() after consuming

4. **Missing tag operations**:
   - insert() after tile arrives at a memory level
   - match() before consuming from a level
   - invalidate() after tile is moved/consumed

5. **Tick ordering**:
   Verify ConcurrentTimingExecutor ticks in order: MC → DMA → BlockMover → Streamer

Report findings as:
| File:Line | Violation Type | Severity | Suggested Fix |
|-----------|---------------|----------|---------------|

If no violations found, report "Architecture audit CLEAN."
