  Weight-Stationary vs Output-Stationary Trade-off:

  | Aspect            | Output-Stationary (Current) | Weight-Stationary (To Implement) |
  |-------------------|-----------------------------|----------------------------------|
  | What stays in PEs | Output accumulators (C)     | Weights (B)                      |
  | Energy benefit    | C reuse across K tiles      | B reuse across M tiles           |
  | Best workload     | Deep accumulation (large K) | Batch processing (large M)       |
  | Loop order        | ti → tj → tk                | tk → tj → ti                     |

  📋 4-Week Implementation Plan

  Week 1: Tile Optimizer for WS
  - New constraint: Tk × Tj ≤ PE_register_capacity
  - Different objective: Maximize B reuse = (M/Ti) × (K/Tk)
  - New method: TileOptimizer::optimize_weight_stationary()

  Week 2: L2 Scheduler for WS
  - Loop order inversion: Weight tiles outer, input rows inner
  - L2 allocation change: Store A+C tiles (not A+B)
  - B tiles stay in PE registers (loaded once per outer loop)
  - New method: L2TileScheduler::generate_schedule_ws()

  Week 3: Performance Models
  - Energy: WS saves B reads but adds C writes
  - Latency: WS amortizes B load across M iterations
  - Strategy-specific calculation functions

  Week 4: Integration
  - Dispatch on strategy in ScheduleCharacterizer
  - Run characterization with all 3 strategies
  - Validate performance patterns match theory

  🔬 Expected Results

  WS Should Win (examples):
  - BERT projections [128, 768, 768]: Load weights once, stream 128 batches
  - CNN convolutions: Weight sharing across spatial dimensions
  - Any large M: Batch >> Hidden dimensions

  OS Should Win (examples):
  - Deep MLPs [1, 4096, 4096]: Single batch, deep accumulation
  - Large K: Accumulation-heavy workloads
  - Attention K >> M, N

  ✅ Success Criteria

  1. WS produces measurably different energy/latency from OS
  2. Performance differences match dataflow theory
  3. Pareto frontier shows both WS and OS as optimal for different shapes
  4. 10-20% of workloads favor WS over OS
  5. Energy model shows B read savings vs C write costs

