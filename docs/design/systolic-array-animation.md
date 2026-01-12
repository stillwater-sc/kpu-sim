# Systolic Array Animation

  Analysis: Complete Dataflow Animation for All Three Schedules

  Phase Structure Comparison
  ┌──────────┬───────────────────────────────┬────────────────────────────────┬────────────────────────────────┐
  │  Phase   │   C-Stationary (S=[0,0,1])    │    A-Stationary (S=[0,1,0])    │    B-Stationary (S=[1,0,0])    │
  ├──────────┼───────────────────────────────┼────────────────────────────────┼────────────────────────────────┤
  │ PE Array │ M×N (4×4)                     │ M×K (4×2)                      │ K×N (2×4)                      │
  ├──────────┼───────────────────────────────┼────────────────────────────────┼────────────────────────────────┤
  │ Load     │ None (streaming)              │ A enters W→E with row skew     │ B enters N→S with col skew     │
  ├──────────┼───────────────────────────────┼────────────────────────────────┼────────────────────────────────┤
  │ Compute  │ A flows E, B flows S, C stays │ B flows S, C flows E           │ A flows E, C flows S           │
  ├──────────┼───────────────────────────────┼────────────────────────────────┼────────────────────────────────┤
  │ Unload   │ C must be drained explicitly  │ C streams out E edge naturally │ C streams out S edge naturally │
  └──────────┴───────────────────────────────┴────────────────────────────────┴────────────────────────────────┘
  Detailed Timing Analysis

  A-Stationary (M×K = 4×2 array):

  Load Phase: A[i,k] → PE(i,k)
  - A[:,k] enters from West with row skew (row i has i delay registers)
  - Cycle 0: A[:,0] enters skew buffer
  - Cycle 1: A[:,1] enters skew buffer
  - A[0,0] arrives at PE(0,0) at t=0
  - A[1,0] arrives at PE(1,0) at t=1 (1 cycle delay)
  - A[0,1] arrives at PE(0,1) at t=1
  - A[3,1] arrives at PE(3,1) at t=4
  - Load complete at cycle: K + (M-1) - 1 = 2 + 3 - 1 = 4

  Compute Phase: B streams, C flows East
  - After load, B[k,j] streams in from North with column skew
  - Column k has k delay registers (same skew structure as load)
  - C partial sums flow West→East, exit from column K-1

  Unload Phase: C exits right edge
  - C[i,j] exits PE(i,K-1) with timing dependent on when B[K-1,j] arrived
  - C emerges with diagonal skew - needs de-skew buffer to align
  - De-skew buffer: row i needs (M-1-i) delay registers to align

  B-Stationary (K×N = 2×4 array):

  Load Phase: B[k,j] → PE(k,j)
  - B[k,:] enters from North with column skew (col j has j delay registers)
  - Cycle 0: B[0,:] enters skew buffer
  - Cycle 1: B[1,:] enters skew buffer
  - B[0,0] arrives at PE(0,0) at t=0
  - B[0,1] arrives at PE(0,1) at t=1 (1 cycle delay)
  - B[1,3] arrives at PE(1,3) at t=4
  - Load complete at cycle: K + (N-1) - 1 = 2 + 3 - 1 = 4

  Compute Phase: A streams, C flows South
  - After load, A[i,k] streams in from West with row skew
  - Row k has k delay registers
  - C partial sums flow North→South, exit from row K-1

  Unload Phase: C exits bottom edge
  - C[i,j] exits PE(K-1,j) with timing dependent on when A[i,K-1] arrived
  - C emerges with diagonal skew - needs de-skew buffer to align
  - De-skew buffer: column j needs (N-1-j) delay registers to align

  C-Stationary (M×N = 4×4 array):

  Load Phase: None

  Compute Phase: A and B stream, C accumulates
  - A and B both stream with their respective skew buffers
  - C accumulates in place in each PE
  - Compute complete when last EOS reaches PE(M-1,N-1)

  Unload Phase: C must be drained
  - C is "trapped" in PEs - no natural exit path
  - Options:
    1. Row drain: Read C row by row (N cycles per row, M×N total)
    2. Column drain: Read C column by column
    3. Diagonal drain: Use reverse skew to create wavefront exit
  - For fair comparison, should show diagonal drain with skew buffer

  Proposed Visualization Structure

  ┌─────────────────────────────────────────────────────────────┐
  │                    B SRAM / Input                           │
  ├─────────────────────────────────────────────────────────────┤
  │                    B Skew Buffer (input)                    │
  │                         ↓ ↓ ↓ ↓                             │
  ├──────────┬──────────────────────────────┬───────────────────┤
  │ A SRAM   │                              │ C Skew Buffer     │
  │    ↓     │                              │ (output for A-st) │
  │ A Skew   │      PE Array                │       ↓           │
  │ Buffer   │   (size varies by schedule)  │    C Output       │
  │    →     │                              │                   │
  ├──────────┴──────────────────────────────┴───────────────────┤
  │                    C Skew Buffer (output for B-stationary)  │
  │                              ↓                              │
  │                         C Output                            │
  └─────────────────────────────────────────────────────────────┘

  Solution Options

  Option 1: Separate visualizations
  - Keep three separate animation modes
  - Each shows the appropriate PE array size
  - Add skew buffers for C output where needed
  - Pros: Cleaner, each schedule is self-contained
  - Cons: Harder to compare side-by-side

  Option 2: Unified layout with dynamic sizing
  - Single layout that adapts PE array size per schedule
  - Always show all skew buffer positions (some empty depending on schedule)
  - Pros: Easier to compare, consistent structure
  - Cons: More complex layout logic

  Option 3: Side-by-side comparison mode
  - Show all three running simultaneously
  - Smaller PE arrays, synchronized timing
  - Pros: Direct visual comparison
  - Cons: May be visually crowded

  My Recommendation

  I recommend Option 2 with the following implementation:

  1. Dynamic PE array: Render M×K for A-stationary, K×N for B-stationary, M×N for C-stationary
  2. Three-phase animation for all schedules:
    - Load phase (animated for A/B-stationary, skipped for C-stationary)
    - Compute phase (all three)
    - Unload phase (C drains for C-stationary, C streams with de-skew for A/B-stationary)
  3. Visual elements:
    - Input skew buffers (A on left, B on top)
    - Output skew buffers (C on right for A-stationary, C on bottom for B-stationary)
    - Phase indicator showing current phase and progress
  4. Cycle counter showing total cycles for fair comparison:
    - A-stationary: Load(K+M-1) + Compute(N+K+M-1) + natural drain
    - B-stationary: Load(K+N-1) + Compute(M+K+N-1) + natural drain
    - C-stationary: Compute(K+M+N-1) + Drain(M+N-1)

