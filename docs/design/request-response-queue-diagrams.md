# Request/Response Resource Diagrams

Based on my research into PCIe, NVMe, AXI, DDR memory controllers, and network processor architectures, here are 5 different visualization options for a unified request/response front-end block:

## Option 1: NVMe-Style Queue Pair

  Based on https://blog.westerndigital.com/nvme-queues-explained/ where Submission Queue and Completion Queue are tightly paired.
```text
  ┌─────────────────────────────────────────────────────────────────┐
  │                    REQUEST/RESPONSE FRONT-END                   │
  ├─────────────────────────────────────────────────────────────────┤
  │  ┌─────────────────────┐       ┌─────────────────────┐          │
  │  │  SUBMISSION QUEUE   │       │  COMPLETION QUEUE   │          │
  │  │  (Request FIFO)     │       │  (Response Buffer)  │          │
  │  ├─────────────────────┤       ├─────────────────────┤          │
  │  │ ┌───┬───┬───┬───┐   │  TAG  │   ┌───┬───┬───┬───┐ │          │
  │  │ │R0 │R1 │R2 │R3 │───┼──────►│   │   │C0 │   │   │ │          │
  │  │ └───┴───┴───┴───┘   │ MATCH │   └───┴───┴───┴───┘ │          │
  │  │      TAIL──►HEAD    │       │   HEAD◄──TAIL       │          │
  │  └─────────────────────┘       └─────────────────────┘          │
  │           │                              ▲                      │
  │           ▼ TO SCHEDULER                 │ FROM DATA BUS        │
  └─────────────────────────────────────────────────────────────────┘

  Pros: Clean separation, familiar to hardware engineers, shows circular buffer semantics
  Cons: Doesn't visually show the tag-to-slot mapping as clearly
```

## Option 2: PCIe Split Transaction Tracker

  Based on https://www.intel.com/content/www/us/en/docs/programmable/683647/18-0/transaction-layer.html with pending transaction tracking.

```text
  ┌─────────────────────────────────────────────────────────────────┐
  │                    REQUEST/RESPONSE FRONT-END                   │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  INPUT ──►┌─────┐    ┌─────────────────────┐    ┌─────┐         │
  │           │ REQ │    │  PENDING TXN TABLE  │    │ RSP │◄── DATA │
  │           │FIFO │    │  ┌────┬────┬────┐   │    │ BUF │    BUS  │
  │           ├─────┤    │  │Tag │Addr│Stat│   │    ├─────┤         │
  │           │ R0  │───►│  │ 0  │0x40│PEND│◄──┼────│ --- │         │
  │           │ R1  │    │  │ 1  │0x80│WAIT│   │    │ D1  │         │
  │           │ R2  │    │  │ 2  │0xC0│DONE│───┼───►│ D2  │──► OUT  │
  │           │ ... │    │  └────┴────┴────┘   │    │ ... │         │
  │           └─────┘    └─────────────────────┘    └─────┘         │
  │               │              ▲    │                ▲            │
  │               ▼              │    ▼                │            │
  │           TO SCHED      TAG ALLOC    TAG FREE   DATA IN         │
  └─────────────────────────────────────────────────────────────────┘

  Pros: Shows tag lifecycle clearly (allocate → pending → complete → free)
  Cons: More complex, three distinct structures to track
```

## Option 3: Unified Reorder Buffer (ROB) Style

  Based on https://docs.boom-core.org/en/latest/sections/reorder-buffer.html from processor design.

```text
  ┌─────────────────────────────────────────────────────────────────┐
  │                    REQUEST/RESPONSE FRONT-END                   │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │    ┌─────────────────────────────────────────────────────┐      │
  │    │             REORDER BUFFER (Circular)               │      │
  │    ├─────┬─────────────┬────────┬─────────┬──────────────┤      │
  │    │ Tag │  Request    │ State  │ Bank    │  Response    │      │
  │    ├─────┼─────────────┼────────┼─────────┼──────────────┤      │
  │    │  0  │ R0: Rd 0x40 │████████│  B0     │ [D0 D1 D2..] │ ◄─┐  │
  │    │  1  │ R1: Rd 0x80 │▓▓▓▓░░░░│  B0     │ [...........]│   │  │
  │    │  2  │ R2: Rd 0xC0 │░░░░░░░░│  B0     │ [...........]│   │  │
  │    │  3  │ ----------- │        │  --     │ [...........]│   │  │
  │    └─────┴─────────────┴────────┴─────────┴──────────────┘   │  │
  │       ▲                                                      │  │
  │       │ HEAD                                          TAIL ──┘  │
  │       ▼                                                         │
  │    RETIRE ◄── Complete in order                                 │
  └─────────────────────────────────────────────────────────────────┘

      State: ████ = Complete, ▓▓▓▓ = Data Arriving, ░░░░ = Pending

  Pros: Single unified structure, shows in-order retirement, progress visualization
  Cons: May be harder to see separate request/response flows
```

## Option 4: AXI-Style Dual-Channel with Scoreboard

  Based on https://developer.arm.com/documentation/102202/latest/Transfer-behavior-and-transaction-ordering with ID matching.

```text
  ┌─────────────────────────────────────────────────────────────────┐
  │                    REQUEST/RESPONSE FRONT-END                   │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  REQUEST CHANNEL (AR/AW) ──────────────────────────────────►    │
  │  ┌─────┬─────┬─────┬─────┐                                      │
  │  │ R0  │ R1  │ R2  │ R3  │  ════════════════════►  TO DRAM      │
  │  │ID:0 │ID:1 │ID:2 │ID:3 │                                      │
  │  └──┬──┴──┬──┴──┬──┴──┬──┘                                      │
  │     │     │     │     │     OUTSTANDING ID SCOREBOARD           │
  │     │     │     │     │    ┌────┬────┬────┬────┐                │
  │     └─────┼─────┼─────┼───►│ 0  │ 1  │ 2  │ 3  │                │
  │           └─────┼─────┼───►│ ●  │ ●  │ ○  │ ○  │                │
  │                 └─────┼───►│WAIT│WAIT│FREE│FREE│                │
  │                       └───►└────┴────┴────┴────┘                │
  │                                 │     │                         │
  │  RESPONSE CHANNEL (R/B) ◄───────┴─────┴──────────────────────   │
  │  ┌─────┬─────┬─────┬─────┐                                      │
  │  │ D0  │ D1  │     │     │  ◄════════════════════  FROM DRAM    │
  │  │ID:0 │ID:1 │     │     │   (out-of-order OK)                  │
  │  └─────┴─────┴─────┴─────┘                                      │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

      ● = Outstanding (waiting)    ○ = Free slot

  Pros: Clear bidirectional flow, explicit ID tracking, shows out-of-order capability
  Cons: Two separate channel visualizations may look disconnected
```

## Option 5: Paired Slot Buffer (Tag-Matched FIFO)

  Hybrid design combining request queue with preallocated response slots.

```text
  ┌─────────────────────────────────────────────────────────────────┐
  │                    REQUEST/RESPONSE FRONT-END                   │
  │                                                                 │
  │   INPUT ──►  ┌─────────────────────────────────────┐  ──► OUT   │
  │              │         TRANSACTION BUFFER          │            │
  │              ├──────────────┬──────────────────────┤            │
  │              │   REQUEST    │      RESPONSE        │            │
  │              │    SLOT      │       SLOT           │            │
  │              ├──────────────┼──────────────────────┤            │
  │         ┌───►│ R0: Rd 0x40  │  ████ D0 READY  ████ │───► RETIRE │
  │         │    ├──────────────┼──────────────────────┤            │
  │         │    │ R1: Rd 0x80  │  ░░░░ -- WAIT   ░░░░ │◄─── DATA   │
  │         │    ├──────────────┼──────────────────────┤     BUS    │
  │ ALLOC ──┤    │ R2: Rd 0xC0  │  ░░░░ -- WAIT   ░░░░ │◄───        │
  │         │    ├──────────────┼──────────────────────┤            │
  │         └───►│ -- (empty)   │  ---- FREE --------- │            │
  │              └──────────────┴──────────────────────┘            │
  │                     │                    ▲                      │
  │                     ▼                    │                      │
  │              TO SCHEDULER         FROM DATA BUS                 │
  └─────────────────────────────────────────────────────────────────┘

      ████ = Data present    ░░░░ = Allocated, waiting

  Pros: Tightly coupled request/response in same row, clear preallocation concept, intuitive left-to-right/right-to-left flow
  Cons: Fixed slot assignment may not show out-of-order returns as clearly
```

  ---
## Summary Comparison

  | Option              | Structure                      | Best For                    | Complexity |
  |---------------------|--------------------------------|-----------------------------|------------|
  | 1. NVMe Queue Pair  | Dual circular buffers          | Storage-style interfaces    | Medium     |
  | 2. PCIe Split Txn   | 3-part (FIFO + Table + Buffer) | Bridge/interconnect         | High       |
  | 3. ROB Style        | Single unified buffer          | Processor-like ordering     | Low        |
  | 4. AXI Dual-Channel | Scoreboard + channels          | Bus protocol visualization  | Medium     |
  | 5. Paired Slot      | Matched request/response rows  | Memory controller front-end | Low        |

