# ResNet Study Results

All 8 PRs from this session are merged (GH Issues #220–#227), main is at 7a992ad, every branch cleaned up (local + remote), and the session decision log is on main.

Timing suite: 59/59 green, including the new resnet_regression ctest passing on all three platforms.

---
## ResNet-on-KPU Benchmarking Study — Results

A research guide (docs/benchmarking/resnet-benchmarking-guide.md) and a six-item instrumentation roadmap that turned the M2 ResNet demo into a performance
instrument:

|   #    |                          Deliverable                          |                 How to run                 |
|--------|---------------------------------------------------------------|--------------------------------------------|
| 1 / 1b | Movement-fabric utilization — directly measured active cycles | m2_resnet                                  |
| 2      | Compute FLOP efficiency + roofline position                   | m2_resnet                                  |
| 3      | Buffer-occupancy timeline (TileTracker)                       | m2_resnet --occupancy                      |
| 4      | Representative-scale offline run (channel growth + [2,2,2,2]) | m2_resnet --full                           |
| 5      | Concurrency-headroom (branch-overlap critical path)           | m2_resnet                                  |
| 6      | JSON export + CI regression guard                             | m2_resnet --json / resnet_regression ctest |

The findings — five lenses, one answer

|      Lens      |                   Measurement                    |                Verdict                 |
|----------------|--------------------------------------------------|----------------------------------------|
| Utilization    | DMA 84.6% busy; BlockMover 14.1%, Streamer 10.7% | Movers starve behind DMA               |
| Compute        | AI 5.25 FLOP/byte (< 8 ridge); peak eff 21.9%    | Memory-bound                           |
| Occupancy      | peak L3 4/32, L2 2/64                            | Buffers not the limit                  |
| Scale (--full) | AI → 7.6, eff → 37%, DMA 84.5%                   | Holds — moves toward ridge, not across |
| Concurrency    | branch overlap upper bound 1.02×                 | Not a lever (ResNet is ~a chain)       |

Conclusion: ResNet on this KPU is DRAM-bandwidth-bound — and nothing else binds. The compute array (~22–37% of peak) and the on-chip movers/buffers all sit idle waiting on the DRAM→L3 DMA. The single architectural lever that would help is DRAM bandwidth / more DMA engines; on-chip buffering, compute, and concurrent branch scheduling would each move the needle by little to nothing for this workload.

## Two corrections along the way (documented in the session log)

1. I first reported "~20–40% utilization, BlockMover is the bottleneck" — an artifact of an instrumentation gap and a weak busy = total − stall heuristic. CodeRabbit surfaced the gap; the direct-measurement counter (1b) then reversed the diagnosis to DMA-bound. Lesson: don't quote a finding before the metric is validated — which is exactly why item 6's regression guard now locks the validated numbers.
2. Nearly shipped a floored fractional metric (1b); CodeRabbit caught it, fixed with a non-divisible regression test.

## Now guarded

The conclusion is no longer a one-time observation — resnet_regression diffs every metric against the committed deterministic baseline on every PR across all platforms, so any code change that moves ResNet's cycles/utilization/compute numbers fails CI until the baseline is deliberately regenerated.

## Open follow-ons (noted, not blocking)

True full-resolution ResNet (needs a native large-K/grouped-conv schedule to be tractable), a per-layer arithmetic-intensity breakdown, and real concurrent multi-op execution — the last only if a non-ResNet workload with genuine branch parallelism ever justifies it.
