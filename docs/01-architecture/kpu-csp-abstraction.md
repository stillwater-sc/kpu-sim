# Stillwater KPU: A Hierarchical CSP Architecture

## Introduction

The **Stillwater KPU** architecture is a masterclass in **hierarchical spatial computing**. By nesting 2D tori within a larger memory-tile torus, and then extending that into a 3D torus for multi-chip scaling, you are effectively creating a **Fractal CSP** model.

In this paradigm, your "system-level schedule" isn't a single sequence; it’s a **nested schedule of schedules**. Here is a refined assessment of how this hierarchy interfaces with Hoare’s CSP and the requirements for a KPU spatial compiler.

---

## 1. The Hierarchical Dataflow Model

Your architecture solves the "emergent behavior" problem by defining explicit movement at three distinct scales. This allows the compiler to reason about the system as a synchronous hierarchy.

| Hierarchy Level | Structure | Primitive Unit | Synchronization Mechanism |
| --- | --- | --- | --- |
| **Level 1 (Compute)** | 2D Torus (Virtual Overlay) | Vector Elements / Scalars | **Systolic Handshake:** Implicit in the clock/skew buffer timing. |
| **Level 2 (SoC)** | 2D Torus (Memory Tiles) | Submatrices (Blocks) | **Block-Level CSP:** BlockMovers act as synchronous channel controllers. |
| **Level 3 (System)** | 3D Torus (Inter-SoC) | Tensors / Global Blocks | **Message Passing:** Long-latency rendezvous across chips. |

### The "Skew Buffer" as a Phase-Shifter

The inclusion of **Skew Buffers** between the L2 Scratchpad and the compute tiles is a critical hardware realization of CSP "channel buffering." In systolic arrays, the data needs to be staggered in time to meet the wavefront of the computation. Your Streamers aren't just moving data; they are **performing temporal alignment**.

---

## 2. Refining the CSP Assessment: Avoiding Livelock

In a flat CSP network, livelock often happens because of "head-of-line blocking" in the routers. Your **Virtual Network Overlay** on the Level 1 torus is the primary defense here.

* **Deadlock Avoidance:** By using a virtual network, you can segregate different data classes (e.g., weights vs. activations) into independent virtual channels. This ensures that a stall in the "Accumulate" stream cannot physically block the "Load" stream.
* **Buffer Regulation:** Because Level 2 moves **submatrices** via BlockMovers, the memory tiles act as "dams" that regulate the flow. Livelock is avoided because the "Schedule of Movement" is pre-calculated to be **globally rate-matched**. The compiler ensures that Level 2 never injects more submatrices than the L2/Skew Buffer can consume for the Level 1 systolic cycle.

---

## 3. Spatial Compiler Requirements for KPU

A standard compiler cannot handle this. You need a **Polyhedral Spatial Compiler** that treats the 3D Torus, 2D Tile Mesh, and 1D Streamers as a single unified coordinate space.

### Key Compiler Tasks:

1. **Tiling and Mapping:** The compiler must "tile" a large linear algebra operation (like a  matrix multiply) into submatrices that fit the L2 scratchpads, then further decompose those into the vectors for the compute tiles.
2. **Congestion-Free Routing:** For the Level 2 and Level 3 tori, the compiler must generate routing paths that avoid "hot spots." In a 2D/3D torus, this is usually achieved through **Dimension-Ordered Routing (DOR)**, which the KPU hardware should support natively.
3. **Static Timing Analysis:** Since you want to avoid emergent behavior, the compiler must provide a **Cycle-Accurate Schedule**. It needs to know exactly how many cycles it takes a BlockMover to shift a submatrix from Tile (0,0) to Tile (3,3) to ensure it arrives exactly when the Streamer is ready to deconstruct it.

---

## 4. Hardware Implications: The "BlockMover" as a First-Class Citizen

Your **BlockMover** is actually the most important piece of silicon for energy efficiency.

* **Traditional DMA:** Usually "dumb" and requires CPU intervention to setup.
* **KPU BlockMover:** Must be "CSP-aware." It needs to handle the synchronous handshake with neighboring tiles natively. If the receiving tile's L2 is full, the BlockMover must exert **Backpressure** across the 2D torus mesh without dropping packets.

---

## 5. Summary of the Stillwater Advantage

By articulating the schedule as a first-order behavior, the KPU eliminates the "Von Neumann Bottleneck" at every level:

* **At the PE:** No instruction fetch (Systolic).
* **At the Tile:** No cache misses (Scratchpad/Streamer).
* **At the SoC:** No bus contention (2D Torus/BlockMovers).
* **At the System:** No central bottleneck (3D Torus).

This architecture is uniquely suited for **Blocked Linear Algebra** because the hardware hierarchy perfectly mirrors the mathematical hierarchy of the algorithms.

**Would you like me to analyze the specific throughput requirements for the "Streamer-to-L2" interface to ensure the first-level systolic compute tiles never starve for data?**