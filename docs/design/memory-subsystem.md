# Memory Subsystem

Based on my research of memory controller IP (Synopsys, Cadence) and LPDDR5 device architectures, here are three diagram proposals for your review:


## Proposal 1: Controller-Centric View (Emphasizes Request-to-Command Translation)

```text
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                              MEMORY CONTROLLER                                              │
  │  ┌──────────────┐    ┌──────────────────────────────────────┐    ┌───────────────────────┐  │
  │  │ AXI SLAVE    │    │         COMMAND GENERATION           │    │    DDR-PHY            │  │
  │  │              │    │  ┌─────────┐  ┌─────────┐            │    │                       │  │
  │  │ ┌──────────┐ │    │  │ READ    │  │ WRITE   │            │    │  CA Bus ──────────────┼──┼─→
  │  │ │ Request  │─┼────┼─→│ CMD     │  │ CMD     │  ┌──────┐  │    │                       │  │
  │  │ │ Queue    │ │    │  │ Pool    │  │ Pool    │  │SCHED │  │    │                       │  │
  │  │ │ R0 R1 R2 │ │    │  │ ┌─────┐ │  │ ┌─────┐ │  │ULER/ │──┼────┼─>  CMD Generator      │  │
  │  │ │ W0 W1    │ │    │  │ │ACT  │ │  │ │ACT  │ │  │ARBIT │  │    │                       │  │
  │  │ └──────────┘ │    │  │ │RD   │ │  │ │WR   │ │  │ER    │  │    │                       │  │
  │  │              │    │  │ │PRE  │ │  │ │PRE  │ │  └──────┘  │    │                       │  │
  │  │ ┌──────────┐ │    │  │ └─────┘ │  │ └─────┘ │            │    │  DQ Bus <-────────────┼──┼─→
  │  │ │ Response │←┼────┼──│─────────│──│─────────│────────────┼────┼──                     │  │
  │  │ │ Buffer   │ │    │  └─────────┘  └─────────┘            │    │  Read Data Buffer     │  │
  │  │ │ R0 R1 R2 │ │    │                                      │    │  Write Data Buffer    │  │
  │  │ └──────────┘ │    │                                      │    │                       │  │
  │  └──────────────┘    └──────────────────────────────────────┘    └───────────────────────┘  │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                       LPDDR5 DEVICE                                         │
  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │
  │  │  Bank Group 0          Bank Group 1          Bank Group 2          Bank Group 3       │  │
  │  │  ┌───┬───┬───┬───┐    ┌───┬───┬───┬───┐    ┌───┬───┬───┬───┐    ┌───┬───┬───┬───┐     │  │
  │  │  │B0 │B1 │B2 │B3 │    │B4 │B5 │B6 │B7 │    │B8 │B9 │B10│B11│    │B12│B13│B14│B15│     │  │
  │  │  └─┬─┴─┬─┴─┬─┴─┬─┘    └─┬─┴─┬─┴─┬─┴─┬─┘    └─┬─┴─┬─┴─┬─┴─┬─┘    └─┬─┴─┬─┴─┬─┴─┬─┘     │  │
  │  │    │   │   │   │        │   │   │   │        │   │   │   │        │   │   │   │       │  │
  │  │    └───┴───┴───┴────────┴───┴───┴───┴────────┴───┴───┴───┴────────┴───┴───┴───┘       │  │
  │  │                              Global I/O (x16/x32)                                     │  │
  │  └───────────────────────────────────────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

  Key Features:
  - Shows clear separation between requests (R0, R1, W0) and commands (ACT, RD, WR, PRE)
  - Read/Write command pools hold pending DRAM commands generated from user requests
  - Scheduler/Arbiter decides which command to issue based on timing constraints
  - Response buffer matches returning data to original request IDs


## Proposal 2: Pipeline Stage View (Emphasizes Request Flow Through Stages)

```text

                           MEMORY CONTROLLER                                 LPDDR5 DEVICE
  ┌────────────────────────────────────────────────────────────────┐   ┌────────────────────────┐
  │                                                                │   │                        │
  │  REQUEST                                                       │   │   ┌────────────────┐   │
  │  STAGE        DECODE         SCHEDULE        ISSUE             │   │   │ Bank Array     │   │
  │                                                                │   │   │ ┌──┬──┬──┬──┐  │   │
  │  ┌─────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │   │   │ │0 │1 │2 │3 │  │   │
  │  │ R0  │───→│ Addr    │───→│ Timing  │───→│ CMD     │─── ACT ──┼───┼──→│ ├──┼──┼──┼──┤  │   │
  │  │ R1  │    │ Decode  │    │ Check   │    │ Issue   │─── RD ───┼───┼──→│ │4 │5 │6 │7 │  │   │
  │  │ R2  │    │ BG/Bank │    │ tRCD    │    │         │─── WR ───┼───┼──→│ ├──┼──┼──┼──┤  │   │
  │  │ W0  │    │ Row/Col │    │ tCAS    │    │         │─── PRE ──┼───┼──→│ │8 │9 │10│11│  │   │
  │  │ W1  │    │         │    │ tRP     │    │         │          │   │   │ ├──┼──┼──┼──┤  │   │
  │  └─────┘    └─────────┘    └─────────┘    └─────────┘          │   │   │ │12│13│14│15│  │   │
  │     │                          │              │                │   │   │ └──┴──┴──┴──┘  │   │
  │     │                          │              │                │   │   └───────┬────────┘   │
  │     │                          │              │                │   │           │            │
  │     │                          │              │                │   │   ┌───────┴────────┐   │
  │     │                          │              │                │   │   │ Sense Amps     │   │
  │     │                          │              │                │   │   │ (Row Buffer)   │   │
  │     │                          │              │                │   │   └───────┬────────┘   │
  │     │                          │              │                │   │           │            │
  │  RESPONSE                                                      │   │   ┌───────┴────────┐   │
  │  STAGE       REORDER         DATA RX                           │   │   │ Column Mux     │   │
  │                                                                │   │   └───────┬────────┘   │
  │  ┌─────┐    ┌─────────┐    ┌─────────┐                         │   │           │            │
  │  │ R0  │←───│ Match   │←───│ PHY     │←────────────────────────┼───┼───────────┘            │
  │  │ R1  │    │ to ID   │    │ Capture │          DATA           │   │                        │
  │  │ R2  │    │         │    │         │                         │   │                        │
  │  └─────┘    └─────────┘    └─────────┘                         │   │                        │
  │                                                                │   │                        │
  └────────────────────────────────────────────────────────────────┘   └────────────────────────┘
```

  Key Features:
  - Clear pipeline stages: Request → Decode → Schedule → Issue → DRAM → Data Return → Reorder
  - Shows timing checks (tRCD, tCAS, tRP) in scheduling stage
  - One request (R0) may generate multiple commands (ACT + RD)
  - LPDDR5 device shows internal path: Bank → Sense Amp → Column Mux → DQ


## Proposal 3: Transaction Lifecycle View (Emphasizes Request State Machine)

```text`
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   REQUEST LIFECYCLE                                          │
  │                                                                                              │
  │    QUEUED        DECODED       CMD_PENDING      DATA_PHASE         COMPLETE                  │
  │   ┌──────┐      ┌──────┐        ┌──────┐         ┌──────┐          ┌──────┐                  │
  │   │  R0  │─────→│  R0  │───────→│  R0  │────────→│  R0  │─────────→│  R0  │                  │
  │   │  R1  │      │  R1  │        │  R1  │         │      │          │      │                  │
  │   │  W0  │      └──────┘        └──────┘         └──────┘          └──────┘                  │
  │   └──────┘         │               │                 ▲                                       │
  │                    │               │                 │                                       │
  │                    ▼               ▼                 │                                       │
  │              ┌───────────────────────────────────────┴───────────────────┐                   │
  │              │              COMMAND GENERATION & TRACKING                │                   │
  │              │  ┌─────────────────────────────────────────────────────┐  │                   │
  │              │  │  Request R0 @ Bank0, Row 0x100, Col 0x40            │  │                   │
  │              │  │    └─→ ACT(Bank0, Row 0x100) ───→ [ISSUED]          │  │                   │
  │              │  │    └─→ RD(Bank0, Col 0x40)  ───→ [WAITING tRCD]     │  │                   │
  │              │  └─────────────────────────────────────────────────────┘  │                   │
  │              └───────────────────────────────────────────────────────────┘                   │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                       ┌────────────────────────┼────────────────────────┐
                       │                        │                        │
                       ▼                        ▼                        ▼
  ┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────────────┐
  │   CA BUS (Commands)     │   │   DQ BUS (Data)         │   │   LPDDR5 BANK STATE             │
  │                         │   │                         │   │                                 │
  │   ───ACT────→           │   │   ←─────────────────    │   │   IDLE ─→ ACTIVE ─→ READ_BURST  │
  │   ───RD─────→           │   │       Read Data         │   │     ▲                    │      │
  │   ───WR─────→           │   │   ─────────────────→    │   │     └─────PRE────────────┘      │
  │   ───PRE────→           │   │       Write Data        │   │                                 │
  │                         │   │                         │   │   Row Buffer: [Row 0x100]       │
  └─────────────────────────┘   └─────────────────────────┘   └─────────────────────────────────┘
```

Key Features:
  - Shows request state machine: QUEUED → DECODED → CMD_PENDING → DATA_PHASE → COMPLETE
  - Explicitly shows one request (R0) generates multiple commands (ACT + RD)
  - Bank state machine: IDLE → ACTIVE → READ_BURST → (PRE) → IDLE
  - Separates CA bus (commands) from DQ bus (data)


Summary Comparison

  | Aspect                     | Proposal 1              | Proposal 2      | Proposal 3        |
  |----------------------------|-------------------------|-----------------|-------------------|
  | Focus                      | Controller architecture | Pipeline stages | Request lifecycle |
  | Request/Command separation | Clear (pools)           | Clear (stages)  | Very explicit     |
  | Animation complexity       | Medium                  | Low             | High              |
  | Educational value          | Architecture            | Timing flow     | State tracking    |
  | Matches industry diagrams  | Synopsys/Cadence style  | Textbook style  | Debug tool style  |


