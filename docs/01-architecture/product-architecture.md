# Product Architecture

## Branes.ai Embodied AI Design System (KPU-Centric)

The document outlines:

1.  **A unique strategic offering** (Roadmap Engine).
2.  **A custom, efficiency-focused hardware core** (KPU).
3.  **A full co-design software stack** (MIA, PulseOpt, KCG, KAC, DST).
4.  **Critical non-functional requirements** (SAE, Security, Robustness).
5.  **A focused external communication tool** ($\text{OT-Bench}$).
6.  **A concrete development schedule** ($\text{MVP-1}$).

## Introduction
The Branes.ai system is a full-stack **Design, Optimization, and Deployment Suite** that leverages the **Stillwater Supercomputing KPU IP** to deliver superior efficiency for embodied AI. The system integrates a KPU-specific compiler and runtime, enabling deployment on **Custom ASICs** and **FPGA-based semi-custom designs**.

Integrating a proprietary **Knowledge Processing Unit (KPU)** as the target architecture fundamentally shifts the entire product strategy from an optimization layer over vendor hardware (like Jetson) to a comprehensive **Full-Stack AI Hardware/Software Solution**.

Branes.ai is a **Strategic Design Partner** that provides objective, pre-silicon architectural guidance. This significantly differentiates our value proposition against vertically integrated competitors, such as, NVIDIA, Xilinx, or Synopsys.

The core value of the Branes.AI product is this foundational module focused on **Architectural Exploration and Product Roadmapping**, which leverages the core capabilities of the Analysis (MIA) and Optimization (PulseOpt) components.

### 1. Design System Goals and Requirements

| Category | Description | Target Metric |
| :--- | :--- | :--- |
| **Compute Engine** | Utilize the proprietary KPU IP as the differentiating energy-efficient compute target. | **EDP** Reduction, Operator Coverage, Resource Scheduling efficiency |
| **Performance** | Achieve high sustained throughput for embodied AI workloads. | $\ge 90\%$ (of KPU peak TOPS) sustained |
| **Efficiency** | Maximize TOPS/Watt via architectural and software co-design. | $\mathbf{\ge 50 \times}$ improvement over Jetson baseline (e.g., 50 TOPS at 2W) |
| **Flexibility** | Support various target designs (ASIC and FPGA) and model architectures. | Support **ASIC tape-out flow** and **FPGA bitstream generation** |
| **Target Customers** | High-volume robotics and drone companies requiring custom silicon efficiency. | Successful tape-out/bitstream for 2 pilot customers |

***

## 1. Strategic Differentiation: Architectural Exploration and Product Roadmapping (Roadmap Engine)

This module is the customer-facing front-end of the Branes.ai offering, providing objective, data-driven analysis to inform the customer's hardware roadmap across various $\text{SKUs}$ and timelines. This is enabled by Branes.ai's ability to model the efficiency of diverse compute architectures, not just the KPU.

### 1.1. Application Profiling & Application-Architecture Mapping

* **Input:** Customer's end-to-end $\text{AI}$ model graph and $\text{SKU}$-specific requirements (e.g., Target Latency, Power Budget, Bill of Materials ($\text{BOM}$) Cost).
* **Architecture Modeling Library:** A comprehensive library of $\text{ASIC}$/$\text{IP}$ architectural models (e.g., $\text{CPU}$ core types, $\text{GPU}$ execution units, $\text{TPU}$ systolic arrays, $\text{KPU}$ parameters, $\text{CGRA}$, $\text{DSP}$) that accurately predict performance and power given a specific $\text{AI}$ workload.
* **Energy-Delay Product (EDP) Metric:** Calculation of the EDP ($\text{EDP} = \text{Energy} \times \text{Delay}$) as the primary metric for comparing architectural choices.

### 1.2. Multi-Architecture Pareto Optimization

* **Pareto Graph Generation:** Automatically generate $\text{Pareto}$ Frontiers plotting:
    * **$\text{EDP}$ vs. $\text{Cost}$ ($\text{BOM}$):** For different $\text{SKUs}$ today (low-cost $\text{CPU}$ vs. high-end $\text{KPU}$ ASIC).
    * **$\text{TOPS}/\text{Watt}$ vs. $\text{Area}$ (Silicon/FPGA):** For custom $\text{IP}$ designs.
* **Roadmap Simulation:** Project the performance and efficiency of different architectures (e.g., next-gen $\text{CPU}$ vs. $\text{KPU}$ v2) over a 3-5 year timeline, helping the customer identify inflection points where the KPU becomes the necessary solution.

### 1.3. Output and Deliverables

* **Architectural Guidance Report:** Detailed analysis for each $\text{SKU}$, recommending the optimal processor choice ($\text{CPU}$, $\text{GPU}$, $\text{KPU}$, $\text{FPGA}$, etc.) that meets the power and latency constraints at the lowest cost.
* **KPU Parameter Selection:** If the KPU is selected, this module provides the initial constraints for the **KPU Configuration Generator ($\text{KCG}$)** in the main workflow.

---

## 2. Architectural Components (The Branes.ai KPU Stack)

The architecture is broken into two major domains: the **Software Stack** and the **KPU Hardware Abstraction & Configuration Layer (HACL)**.

### 2.1. Software Stack: Design and Optimization

This stack prepares the model for the KPU's unique architecture.

* **Model Ingestion and Analysis (MIA)**
* **The Optimization Engine (PulseOpt)**

#### 2.1.1. Model Ingestion and Analysis (MIA)

* **Input:** 
  1. PyTorch/TensorFlow Model Checkpoint (.pth, .pb)
  2. Model Graph (ONNX)
  3. StableHLO flatbuffer
* **Key Functions:** 
  1. Graph Parsing (to DomainFlow-IR)
  2. Architecture Fingerprinting
  3. Establishing the target $\text{Power}$ and $\text{TOPS}_{sustained}$ goals

**Architecture Fingerprinting** is the process of conducting a deep, quantitative analysis of a customer's neural network to extract a detailed, immutable "signature" that defines its computational and data-flow requirements.

Architecture Fingerprinting is about understanding the deep, unique characteristics of a customer's model *before* optimization. It's a metaphor drawn from different scientific endeavors, specifically material science, but here, it specifically refers to the process within the **Model Ingestion and Analysis (MIA)** module to generate high-level abstract information about the compute graph.

This "fingerprint" is the necessary input for the optimization and architectural exploration phases.

### What the "Architecture Fingerprint" Captures

The output of the MIA's "fingerprinting" process is a structured data set that details the following:

1.  **Computational Hotspots and Dependencies:**
    * Which layers consume the most cycles ($\text{TOPS}$)? (e.g., specific $\text{Convolution}$ layers, $\text{Attention}$ blocks).
    * What are the precise arithmetic types ($\text{FP32}$, $\text{FP16}$, $\text{INT8}$, etc.) required for each layer to maintain target accuracy?
    * Detailed breakdown of operation types: $\text{GEMM}$ (General Matrix Multiply), $\text{Convolution}$, $\text{Element-wise}$ operations, $\text{Recurrence}$ ($\text{RNN/LSTM}$), $\text{Normalization}$ ($\text{Batch/Layer/Weights/etc.}$), $\text{SoftMax}$.

2.  **Memory Access and Data Flow Signature:**
    * **Data Locality/Reuse:** The required size and nature of on-chip memory (e.g., scratchpads, caches) needed to avoid costly off-chip DRAM access. This is the **most critical** factor for energy efficiency.
    * **Inter-Layer Data Volume:** The volume, shape, and format of tensors passed between layers, which dictates the necessary memory bandwidth and bus utilization.

3.  **Graph Topology and Structure:**
    * **Parallelism Profile:** The maximum level of instruction-level and operator-level parallelism inherent in the graph. This informs how many KPU cores are necessary.
    * **Custom/Non-Standard Operations:** Identification of unique or specialized layers that will require custom microcode generation (KAC) rather than standard $\text{KPU}$ primitives.

**Output of fingerprinting**

1.  **Objective Architectural Guidance (Roadmap Engine):** The fingerprint allows the **Roadmap Engine** to accurately simulate the model's behavior on drastically different architectures ($\text{CPU}$, $\text{GPU}$, $\text{TPU}$, $\text{KPU}$, etc.) without running the full code on each. It provides the core data for generating the Energy-Delay-Product (EDP) Pareto graphs.
2.  **Targeted Optimization (PulseOpt):** Instead of applying a generic optimization script, the PulseOpt uses the fingerprint to apply **KPU-aware techniques** (e.g., pruning, mixed-precision quantization) only where they yield the maximum efficiency gain with minimum accuracy loss.
3.  **KPU Configuration (KCG):** The fingerprint is the direct input used by the **KPU Configuration Generator ($\text{KCG}$)** to determine the optimal physical parameters of the KPU core—the number of processing elements, the size of the on-chip memory buffers, and the clock speed—to perfectly match the application's unique needs.

In essence, **Architecture Fingerprinting** is the step that translates the abstract software model into the precise hardware requirements, and is fundamental to Branes.ai's co-design philosophy.



#### 2.1.2. The Optimization Engine (PulseOpt)

* **Core Focus:** Transforming the model to leverage the KPU's specific computational primitives (e.g., memory access patterns, sparsity handling).
* **Key Functions:**
    * **KPU-Aware Quantization:** Automated search for optimal bit-width tailored to the KPU's data path widths.
    * **Data Flow Pruning:** Optimize the model graph to align with the KPU's internal data movement and processing units, minimizing pipeline stalls.
    * **Computational Graph Mapping:** High-level optimization to map the model's operations onto the KPU's parallel execution units.

#### 2.2. Hardware and Compiler Stack: KPU Implementation

This stack is responsible for compiling the optimized model into a hardware configuration and executable runtime.

##### 2.2.1. The KPU Configuration Generator (KCG - *New Component*)

This module handles the parameterized nature of the KPU IP, defining the final silicon/FPGA configuration based on the target application's constraints (area, power, performance).

* **Input:** Customer constraints (ASIC Area/Power Budget, FPGA Size) and PulseOpt's model profile.
* **Key Functions:**
    * **Parameter Search:** Determines the optimal number of KPU cores, on-chip memory size, and external memory interface parameters.
    * **HACL Output:** Generates the **Hardware Configuration File ($\text{HCF}$)** which is used to synthesize the KPU core.

##### 2.2.2. The KPU-Specific Compiler (KAC)

This replaces the generic HAC and is deeply customized for the KPU instruction set architecture (ISA).

* **Key Functions:**
    * **Kernel Code Generation:** Translates the PulseOpt's KPU-IR into **KPU Microcode/ISA instructions**.
    * **Memory Tiling/Scheduling:** Generates a highly efficient execution schedule that manages data movement between external memory, the KPU's on-chip memory, and registers. This is where most of the efficiency gain is realized.
    * **Custom Runtime Generation:** Packages the microcode and memory schedule into a lean, deployable library.

##### 2.2.3. The Deployment & Synthesis Toolkit (DST)

* **ASIC Flow:** Takes the KCG's HCF and outputs RTL (Register-Transfer Level) for $\text{ASIC}$ synthesis and tape-out.
* **FPGA Flow:** Takes the KCG's HCF and outputs $\text{HDL}$ (Hardware Description Language) and a wrapper for the target $\text{FPGA}$ family's toolchain (e.g., Vivado) to generate the bitstream.
*(The rest of the stack remains focused on the KPU, but now the Roadmap Engine feeds the KCG.)*

### 2.2. Hardware and Compiler Stack: KPU Implementation
* **The KPU Configuration Generator (KCG):** *Now accepts input directly from the **Roadmap Engine** to define the optimal KPU core parameters.*
* **The KPU-Specific Compiler (KAC)**
* **The Deployment & Synthesis Toolkit (DST)**

---

## 3. Engineering Team Structure and Focus

* **New Role:** A **Strategic Modeling Team** is needed to maintain and validate the Architecture Modeling Library.

| Team | Focus Area | Key Deliverables |
| :--- | :--- | :--- |
| **Strategic Modeling (AI/Systems Architects)** | **Roadmap Engine**, Architecture Modeling Library | $\text{Pareto}$ Graphs, $\text{EDP}$ Metric Validation, $\text{SKU}$ Roadmapping Reports. |
| **Model & Core IP (ML/Compiler Engineers)** | MIA, PulseOpt, KAC | KPU-IR Definition, Optimization Algorithms, KPU Microcode Generation. |
| **Hardware & Systems (ASIC/FPGA Engineers)** | KCG, DST, Compliance | KPU Parameterization Flow, RTL/HDL Generation, FPGA Bitstream Integration, Timing Closure, ISO 26262/SAE Readiness. |
| **Runtime & Integration (Software/Robotics Engineers)** | Deployment Runtime, Customer Interface, Benchmarking | KPU Runtime Library (**C++**), Tooling for deployment to $\text{ASIC}$/$\text{FPGA}$ dev boards, Customer Pilot Support, "Brane" Metric Implementation. |

***

## 4. Critical Non-Functional Requirements & Compliance

These requirements are essential for market acceptance, particularly in the target embodied AI and robotics markets.

### 4.1. Compliance & Certification (SAE Focus)

Achieving certification readiness is a mandatory step for high-value autonomous systems.

| Component | Requirement | Ownership |
| :--- | :--- | :--- |
| **Safety Architecture** | Compliance with **ISO 26262** (Automotive Functional Safety) standards for the KPU RTL and compiler flow. | Hardware & Systems Team |
| **SAE Readiness** | Design of the KPU and Runtime to support the partitioning and isolation required for $\text{ASIL}$ (Automotive Safety Integrity Level) decomposition ($\text{ASIL B/C}$ targets). | Hardware & Systems Team |
| **Runtime Integrity** | Runtime to include self-checking and diagnostic capabilities, outputting necessary metrics for $\text{SAE}$ L3-L5 system integrity reporting. | Runtime & Integration Team |

### 4.2. Robustness and Reliability

The system must operate reliably under harsh, real-world conditions (e.g., thermal fluctuations, electromagnetic interference, sustained high load).

* **Thermal Management Hooks:** The KAC and DRM must expose $\text{APIs}$ for the host system to monitor and dynamically adjust KPU frequency/voltage based on temperature readings, ensuring sustained performance, not just peak.
* **Error Correction & Resilience:** Implementation of $\text{ECC}$ (Error Correcting Code) on critical on-chip memory blocks (in the KPU RTL) and robust data path checking mechanisms to prevent silent data corruption.
* **Continuous Stress Testing:** Dedicated environment for running $24/7$ load tests on $\text{FPGA}$ prototypes to identify thermal/load-related failures before ASIC tape-out.

#### 4.3. Security and Integrity

Protecting the model IP and the integrity of the execution is paramount.

* **IP Protection:** Model microcode and critical compiler artifacts must be encrypted or obfuscated before deployment. The KPU architecture should support **Secure Boot** integration to verify the authenticity of the deployed microcode.
* **Hardware Isolation:** Mechanisms within the KPU to isolate execution of different workloads, preventing malicious or compromised tasks from interfering with safety-critical ones (e.g., the control loop).
* **Data Integrity:** Secure memory interfaces and side-channel attack mitigation strategies for the KPU IP, particularly if deployed in systems that handle sensitive user data.

---

## 5. Branes.ai Embodied AI Benchmark & Partner Ecosystem

To communicate value and drive adoption, Branes.ai will establish a purposeful, transparent benchmarking system focused on critical autonomy capabilities.

### 5.1. Benchmark Architecture (The "Brane" Metric)

The benchmark will focus on **Sustained System Efficiency**, moving beyond theoretical peak $\text{TOPS}$.

* **Metric Definition (The Brane):** Introduce a clear, composite metric (e.g., $\text{TOPS}_{sustained} / \text{Watt} \times \text{Latency}_{p99}$), called the "Brane" score, to reflect real-world embodied $\text{AI}$ performance.
* **Workload Definition:** Define a suite of representative embodied $\text{AI}$ workloads (e.g., LiDAR processing, SLAM fusion, $\text{End-to-End}$ control) that are common across the target customer base.
* **Transparency:** Publish a detailed methodology for testing, including temperature, input data rate, and full system power draw measurements, to directly counter the "shell game" tactics used by competitors.

The benchmark will continue to use the composite $\text{EDP}$-based metric ($\text{TOPS}_{sustained} / \text{Watt} \times \text{Latency}_{p99}$), called the "**Brane**" score, to reflect real-world, sustained system efficiency.

### 5.2. The Object Tracker Benchmark (OT-Bench)

The core Branes.ai benchmark will be the **Object Tracker Benchmark ($\text{OT-Bench}$)**, structured around the increasing complexity and efficiency demands of autonomous perception.

| Benchmark Level | Complexity | Key Technologies | Example Use Cases |
| :--- | :--- | :--- | :--- |
| **OT-Bench 2D** | Static Sensors, 2D Bounding Box Tracking. | Camera-only video processing, $\text{CNN}$-based detection. | Home Security, Sports Analysis, Law Enforcement, Public Safety, Warfighter, Battlefield. |
| **OT-Bench 3D** | Moving Sensors, 6 $\text{DoF}$ (Degrees of Freedom) Tracking, Sensor Fusion. | Camera, $\text{LiDAR}$, and $\text{RADAR}$ sensor fusion, $\text{SLAM}$, 3D Bounding Boxes. | Autonomous Cars/Trucks, Logistics $\text{UAVs}$ (Drones), Industrial Robotics. |

### 5.3. Collaboration & Partner Engagement

* **Partner Program:** Establish a program for $\text{AI}$ research groups, robotics integrators, and academic labs to gain early access to the KPU/Compiler stack and run their own models on the benchmark.
* **Benchmark Steering Committee:** Form a committee (including early customers and partners) to govern the benchmark's definition, ensuring its relevance and impartiality.
* **Public Scoreboard:** Maintain an online, audited scoreboard tracking the performance of the Branes.ai solution against commodity hardware (e.g., Jetson Orin reference designs) using the transparent "Brane" metric.

***
## 6. Phased Development Timeline and MVP Definitions

The development timeline is structured around sequential **Minimum Viable Product ($\text{MVP}$)** demonstrations, with an immediate focus on demonstrating strategic value and foundational hardware-software interfaces.

| Phase | Goal | Key Deliverables | Target Deadline |
| :--- | :--- | :--- | :--- |
| **MVP-1 (Strategic Foundation)** | Demonstrate architectural guidance and define the hardware-software interface. | **Product Roadmapping Functionality:** Full **Roadmap Engine** $\text{Pareto}$ graph generation demo. | End of October 2025 |
| | | **KPU Resource API (Draft 1.0):** First official draft of the $\text{API}$ specification for the runtime and compiler to target. | End of October 2025 |
| | | **OT-Bench 2D (Alpha):** Release of the first working $\text{OT-Bench}$ prototype demonstrating tracking on **2D** use cases. | End of October 2025 |
| **MVP-2 (Execution Proof)** | Demonstrate initial end-to-end KPU execution and optimization gains. | Full optimization of one $\text{OT-Bench 2D}$ model via the $\text{PulseOpt}$ and $\text{KAC}$. | TBD (e.g., Q1 2026) |
| | | First functional **KPU FPGA Prototype** running compiled microcode. | TBD (e.g., Q1 2026) |
| **MVP-3 (Roadmap Ready)** | Demonstrate multi-sensor capability and compliance readiness. | **OT-Bench 3D (Beta):** Full implementation of the 3D sensor fusion benchmark. | TBD (e.g., Q2 2026) |
| | | Initial **SAE/ISO 26262** compliance documentation for KPU $\text{RTL}$ and compiler. | TBD (e.g., Q2 2026) |

***

## Appendix A: Design Explorer Metrics

Given that the Design Explorer component centers on exploring different system architecture with respect to their energy-efficiency and performance, tracking development metrics should reflect both architectural differentiation and energy-delay optimization maturity. 

Here are three high-impact metrics we could use:

 1. **EDP Reduction Across Benchmarks**
    - **What it measures**: Percentage reduction in energy-delay product (EDP) compared to baseline compute engines (e.g., CPU, GPU, TPU) for representative workloads.
    - **Why it matters**: Directly quantifies the value of the KPU IP as an energy-efficient differentiator.
    - **How to track**: Use standardized matrix ops (matmul, LU, QR) across varying shapes and sparsity levels. Annotate with operand streaming and tiling strategies.

 2. **Operator Coverage and Optimization Depth**
    - **What it measures**: Fraction of supported loop nest patterns (e.g., tiled matmul, pipelined LU) that are fully optimized for KPU-specific memory movement, operand injection, and concurrency.
    - **Why it matters**: Reflects how deeply the design system exploits the KPU’s architectural strengths.
    - **How to track**: Maintain a registry of loop nest templates and annotate each with optimization status (e.g., tiling validated, buffer occupancy modeled, EDP synthesized).

 3. **Resource-constrained Scheduling Fidelity**
     - **What it measures**: Accuracy and completeness of synthesized compute subgraphs and flow control models (e.g., credit-based buffering, operand latency) for KPU pipelines.
    - **Why it matters**: Ensures that concurrency, dependencies, and energy-delay tradeoffs are realistically captured and guide scheduling.
    - **How to track**: Compare synthesized charts against cycle-accurate models. Score fidelity based on buffer occupancy, stall prediction, and throughput alignment.


### What Is Scheduling Fidelity?

In the context of **compute resource allocation**, Scheduling Fidelity refers to how accurately the generated subgraph models:

- **Temporal scheduling of operations** (e.g., operand injection, compute, reduction, data movement)
- **Resource occupancy** (e.g., buffer usage, ALU availability, streaming ports)
- **Flow control dynamics** (e.g., credit-based stalls, backpressure, injection latency)
- **Dependency resolution** (e.g., operand readiness, result forwarding)

These metrics need to **respect architectural constraints** and **predict real-world behavior** under varying workloads.

### Refined Metric 3: Resource-Constrained Scheduling Fidelity

| Submetric                        | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| **Occupancy Accuracy**           | How closely buffer usage and ALU allocation match RTL or cycle-accurate models. |
| **Stall Prediction Validity**    | Whether predicted stalls (due to flow control or operand unavailability) occur in simulation. |
| **Dependency Resolution Timing** | Accuracy of modeled operand readiness and result forwarding delays.         |
| **Throughput Alignment**         | Whether the synthesized schedule achieves expected throughput under constraints. |

### How to Measure It

- Compare synthesized Gantt charts against:
  - Cycle-accurate architectural models that ultimately will need RTL simulations validation/characterization
  - Empirical traces from FPGA/ASIC prototypes
- Score fidelity based on:
  - % deviation in buffer occupancy over time
  - % match in stall cycles
  - Alignment of critical path timing
  - Operator throughput vs. theoretical peak

Perfect, Theodore. Let’s synthesize a robust CSV schema and visualization strategy that supports **resource-constrained scheduling fidelity** for KPU pipelines. The goal is to track how well your synthesized Gantt charts reflect real architectural behavior—buffer occupancy, stalls, operand readiness, and throughput.

---

## CSV Schema: Resource-Constrained Scheduling Fidelity

This schema is designed to be **loop-nest aware**, **stream-aware**, and **cycle-resolved**, enabling granular validation against cycle-accurate models and ultimately RTL simulation or FPGA traces.

### CSV Columns

| Column Name                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `loop_id`                   | Unique identifier for the loop nest or kernel                               |
| `stream_id`                 | Operand stream identifier (e.g., A, B, C, partial sum)                      |
| `cycle`                     | Simulation cycle index                                                      |
| `stage`                     | Pipeline stage (e.g., inject, compute, reduce, forward)                     |
| `buffer_occupancy`          | Number of entries in buffer at this cycle                                   |
| `stall_flag`                | Boolean (1/0) indicating if stall occurred due to flow control              |
| `stall_reason`              | Textual reason (e.g., credit exhaustion, operand unavailable)               |
| `bubble_flag`                | Boolean (1/0) indicating if bubble occurred due to flow control              |
| `bubble_reason`              | Textual reason (e.g., stream not ready, operand unavailable)               |
| `alu_active`                | Boolean (1/0) indicating ALU activity                                        |
| `operand_ready`             | Boolean (1/0) indicating operand availability                               |
| `result_forwarded`          | Boolean (1/0) indicating result forwarding occurred                          |
| `throughput`                | Cumulative instructions completed up to this cycle                          |
| `edp_estimate`              | Estimated energy-delay product at this cycle (optional, if modeled)         |

Perfect adjustment, Theodore. In a push-architecture like the KPU, **bubbles**—idle cycles due to underutilized resources or unmet operand readiness—are far more telling than stalls. Let’s incorporate that into the fidelity metric and synthesize a CSV schema and Python visualization that reflect this architectural nuance.

---

## 📊 Updated CSV Schema: Resource-Constrained Scheduling Fidelity (KPU-Aware)

Here’s the refined schema with **bubble diagnostics** added:

| Column Name           | Description                                                                 |
|------------------------|------------------------------------------------------------------------------|
| `loop_id`              | Unique identifier for the loop nest or kernel                               |
| `stream_id`            | Operand stream identifier (e.g., A, B, C, partial sum)                      |
| `cycle`                | Simulation cycle index                                                      |
| `stage`                | Pipeline stage (e.g., inject, compute, reduce, forward)                     |
| `buffer_occupancy`     | Number of entries in buffer at this cycle                                   |
| `bubble_flag`          | Boolean (1/0) indicating if a bubble occurred                               |
| `bubble_reason`        | Textual reason (e.g., operand not ready, ALU idle, partial sum delay)       |
| `alu_active`           | Boolean (1/0) indicating ALU activity                                        |
| `operand_ready`        | Boolean (1/0) indicating operand availability                               |
| `result_forwarded`     | Boolean (1/0) indicating result forwarding occurred                          |
| `throughput`           | Cumulative instructions completed up to this cycle                          |
| `edp_estimate`         | Estimated energy-delay product at this cycle (optional)                     |


### Sample CSV Snippet

```csv
loop_id,stream_id,cycle,stage,buffer_occupancy,bubble_flag,bubble_reason,alu_active,operand_ready,result_forwarded,throughput,edp_estimate
matmul_128x128,A,0,inject,4,0,,0,0,0,0,0.0
matmul_128x128,A,1,inject,5,1,operand not ready,0,0,0,0,0.1
matmul_128x128,A,2,compute,6,0,,1,1,0,1,0.2
matmul_128x128,B,2,inject,3,1,ALU idle,0,1,0,1,0.2
matmul_128x128,C,3,reduce,2,0,,1,1,1,2,0.3
```

## Python Visualization Code

Here’s a Python script using `matplotlib` and `seaborn` to visualize:

1. **Bubble Timeline**
2. **Buffer Occupancy Heatmap**
3. **Throughput vs. EDP Curve**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("kpu_schedule_fidelity.csv")

# Bubble Timeline
plt.figure(figsize=(12, 4))
bubble_cycles = df[df['bubble_flag'] == 1]
sns.scatterplot(data=bubble_cycles, x='cycle', y='stream_id', hue='bubble_reason', style='stage', s=100)
plt.title("Bubble Timeline by Stream and Reason")
plt.xlabel("Cycle")
plt.ylabel("Stream ID")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Buffer Occupancy Heatmap
pivot = df.pivot_table(index='stream_id', columns='cycle', values='buffer_occupancy', aggfunc='mean')
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="YlGnBu", annot=False)
plt.title("Buffer Occupancy Over Time")
plt.xlabel("Cycle")
plt.ylabel("Stream ID")
plt.tight_layout()
plt.show()

# Throughput vs. EDP
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='cycle', y='throughput', label='Throughput')
sns.lineplot(data=df, x='cycle', y='edp_estimate', label='EDP Estimate')
plt.title("Throughput vs. Energy-Delay Product")
plt.xlabel("Cycle")
plt.ylabel("Metric Value")
plt.legend()
plt.tight_layout()
plt.show()
```

---

### Usage Modes

- **Per-loop analysis**: Filter by `loop_id` to validate scheduling fidelity for each kernel.
- **Stream diagnostics**: Track `stream_id` to analyze operand injection and flow control.
- **Stall profiling**: Aggregate `stall_flag` and `stall_reason` to identify bottlenecks.
- **Throughput tracking**: Use `throughput` to compare against theoretical peak.

---

## Visualization Strategy

Here’s how to turn the CSV into actionable insights:

### 1. **Resource Scheduling Graph Overlay with Stall/Bubble Annotations**
- **X-axis**: Cycle
- **Y-axis**: Pipeline stages per stream
- **Overlay**: Color-coded bars for active stages, with stall cycles marked (e.g., red for credit stalls, orange for operand stalls)

### 2. **Buffer Occupancy Heatmap**
- **X-axis**: Cycle
- **Y-axis**: Buffer ID or stream
- **Color**: Occupancy level (e.g., gradient from green to red)

### 3. **Stall/Bubble Cause Histogram**
- Bar chart showing frequency of each `stall_reason` across all loops and streams
- Bar chart showing frequency of each `bubble_reason` across tiles

### 4. **Throughput vs. EDP Curve**
- Line chart comparing cumulative `throughput` and `edp_estimate` over time
- Useful for identifying diminishing returns or energy-delay tradeoffs

---

## Appendix B: Normalization Operators

### Common Normalization Techniques in DNNs

| Normalization Type       | Normalizes Over             | Typical Use Case                            |
|--------------------------|-----------------------------|---------------------------------------------|
| **Batch Normalization**  | Batch dimension             | CNNs, MLPs; stabilizes training             |
| **Layer Normalization**  | Features within a layer     | Transformers, RNNs; independent of batch    |
| **Instance Normalization** | Each individual sample     | Style transfer, image generation            |
| **Group Normalization**  | Groups of channels/features | Small batch sizes, vision tasks             |
| **Weight Normalization** | Model weights               | Optimization stability                      |
| **Spectral Normalization** | Singular values of weights | GANs; controls Lipschitz constant           |

---

### Quick Highlights

- **Layer Normalization**: Normalizes across the feature dimension for each sample. Crucial in NLP models like Transformers where batch statistics are less meaningful.
  
- **Instance Normalization**: Normalizes each sample independently, often used in style transfer to preserve content while adjusting style.

- **Group Normalization**: Divides channels into groups and normalizes within each group. It’s effective when batch sizes are small and batch statistics are unreliable.

- **Weight Normalization**: Reparameterizes weight vectors to decouple magnitude and direction, improving optimization dynamics.

- **Spectral Normalization**: Constrains the spectral norm (largest singular value) of weight matrices, often used in GANs to stabilize training by controlling the Lipschitz constant.

---

### Choosing the Right Normalization

- **Large batches, convolutional nets** → BatchNorm
- **Small batches or variable-length sequences** → LayerNorm or GroupNorm
- **Generative models (GANs, style transfer)** → InstanceNorm or SpectralNorm
- **Custom optimization behavior** → WeightNorm

---

When modeling operand injection or loop nests for dynamic algorithms, normalization layers can influence memory movement and scheduling. For example, BatchNorm introduces additional dependencies across batches, while LayerNorm is more localized and easier to parallelize.