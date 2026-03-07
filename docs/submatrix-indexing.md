# Sub-matrix indexing

The tile indexing and address calculation.

TileID Semantics: A[ti, tj, tk]

For matrix A[M,K] in matmul C = A × B:
  - ti = tile row index (which row of C this contributes to)
  - tj = always 0 for A (unused dimension)
  - tk = tile column index in K dimension (reduction axis)

For a 64×64 matrix A with 16×16 tiles:
  ┌──────────┬─────────────────┬──────────────────────────────┐
  │ Tile ID  │ Matrix Elements │           Meaning            │
  ├──────────┼─────────────────┼──────────────────────────────┤
  │ A[0,0,0] │ A[0:16, 0:16]   │ First 16 rows, columns 0-15  │
  ├──────────┼─────────────────┼──────────────────────────────┤
  │ A[0,0,1] │ A[0:16, 16:32]  │ First 16 rows, columns 16-31 │
  ├──────────┼─────────────────┼──────────────────────────────┤
  │ A[0,0,2] │ A[0:16, 32:48]  │ First 16 rows, columns 32-47 │
  ├──────────┼─────────────────┼──────────────────────────────┤
  │ A[0,0,3] │ A[0:16, 48:64]  │ First 16 rows, columns 48-63 │
  ├──────────┼─────────────────┼──────────────────────────────┤
  │ A[1,0,0] │ A[16:32, 0:16]  │ Second 16 rows, columns 0-15 │
  └──────────┴─────────────────┴──────────────────────────────┘

## Base vs Addr

From matmul_schedule_generator.hpp:434:

  // A[M,K]: row-major, tile at (ti, tk)
  return config_.a_base + (ti * config_.k_tiles() + tk) * tile_bytes;

  ┌───────┬─────────────────────────────────────┬─────────────────┐
  │ Field │               Meaning               │     Example     │
  ├───────┼─────────────────────────────────────┼─────────────────┤
  │ base  │ Start of entire matrix A in DRAM    │ 0x1000          │
  ├───────┼─────────────────────────────────────┼─────────────────┤
  │ addr  │ Start of this specific tile in DRAM │ Varies per tile │
  └───────┴─────────────────────────────────────┴─────────────────┘

Calculation for 64×64 FP32 matrix with 16×16 tiles:
  - k_tiles = 64/16 = 4
  - tile_bytes = 16 × 16 × 4 = 1024 bytes = 0x400
  ┌──────────┬────────────────────────────┬────────┐
  │   Tile   │        Calculation         │  addr  │
  ├──────────┼────────────────────────────┼────────┤
  │ A[0,0,0] │ 0x1000 + (0×4 + 0) × 0x400 │ 0x1000 │
  ├──────────┼────────────────────────────┼────────┤
  │ A[0,0,1] │ 0x1000 + (0×4 + 1) × 0x400 │ 0x1400 │
  ├──────────┼────────────────────────────┼────────┤
  │ A[0,0,2] │ 0x1000 + (0×4 + 2) × 0x400 │ 0x1800 │
  ├──────────┼────────────────────────────┼────────┤
  │ A[0,0,3] │ 0x1000 + (0×4 + 3) × 0x400 │ 0x1C00 │
  ├──────────┼────────────────────────────┼────────┤
  │ A[1,0,0] │ 0x1000 + (1×4 + 0) × 0x400 │ 0x2000 │
  └──────────┴────────────────────────────┴────────┘

So base identifies which matrix (A, B, or C), while addr identifies the exact DRAM location of that tile within the matrix.
