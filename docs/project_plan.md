# Project Plan

By leveraging **Agentic AI** for a 3x productivity boost, your 9-person internal team effectively operates with the output of a ~25–30 person traditional engineering organization. This allows you to maintain a rapid cadence from FPGA to Production in 24–36 months.

Below is the synthesized project management plan, resource allocation, and estimated cost structure for the next three years.

---

## 1. High-Level Project Schedule (3-Year Roadmap)

The schedule is front-loaded with verification and software-hardware co-design to ensure the production silicon is "right the first time."

| Phase | Timeline | Major Milestones | Key Deliverables |
| --- | --- | --- | --- |
| **Phase 1: FPGA & RTL** | Months 1–6 | SoC Architecture Freeze; RTL Functional Completion. | FPGA Verification Vehicle; Functional Simulator. |
| **Phase 2: MPW** | Months 6–9 | Synthesis & Timing Closure; MPW Tape-out. | MPW Silicon (Early Functional Validation). |
| **Phase 3: Test Chip** | Months 9–12 | Logic Fixes; Physical Design Refinement; Test Chip Tape-out. | Test Chip (Full IP Validation, Initial Bring-up). |
| **Phase 4: Production** | Months 12–24 | Final Software Stack Integration; Mass Production Tape-out. | Production Silicon; BIOS/Drivers/Compiler Alpha. |
| **Phase 5: Post-Silicon** | Months 24–36 | Silicon Bring-up; Qualification; Customer Sampling. | Mass Production Ramp; Commercial Driver/Compiler Release. |

---

## 2. Resource Allocation & Team Structure

Your team relies on highly specialized SMEs. The **Agentic AI** multiplier is most critical in **Logic QA** and **Physical Design**, where automated agents can handle regression testing and P&R (Place and Route) iterations.

### Internal Team (The "Lean Core")

* **1 Architect:** SoC design, functional simulation, and performance modeling.
* **1 SME (Memory/DMA):** HBM/DDR controller integration and data movement.
* **1 SME (ALUs/SFUs):** Implementation of the regular AI compute fabric.
* **1 SME (AI Kernels):** Hardware-aware kernel optimization (GEMM, Convolutions).
* **1 SME (NoC/PCIe):** On-chip interconnect and host-to-device communication.
* **3 Logic QA SMEs:** Leveraging AI agents for 24/7 autonomous test generation and bug triaging.
* **1 Physical Design SME:** Synthesis, floorplanning, and GDSII generation.

### Outsourced/Partner Resources

* **System Engineering:** Board design, thermal management, and power delivery.
* **BIOS/Bring-up:** Low-level firmware and initial silicon "heartbeat" testing.
* **Driver/Compiler:** Porting LLVM/MLIR to the custom hardware ISA.

---

## 3. Estimated Resource Cost (3-Year Total)

*Note: Estimates assume a mid-range process node (e.g., 16nm/12nm) and US-based SME salaries. Costs are significantly higher for 7nm/5nm due to mask set expenses.*

| Category | Estimated Cost (3 Years) | Notes |
| --- | --- | --- |
| **Internal Salaries** | **$8.5M – $10.0M** | 9 SMEs at ~$320k avg. total comp (including AI tool overhead). |
| **EDA Tool Licensing** | **$4.0M – $6.0M** | Synopsys/Cadence/Siemens suites (Startup pricing models). |
| **Tape-out Costs** | **$7.0M – $12.0M** | Combined: MPW ($500k), Test Chip ($2M), Production Mask ($5M+). |
| **Outsourced Partners** | **$3.5M – $5.0M** | System, BIOS, and Compiler development contracts. |
| **Infrastructure/Cloud** | **$1.5M – $2.5M** | Compute for Agentic AI, EDA simulations, and FPGA hosting. |
| **Total Estimated Budget** | **$24.5M – $35.5M** | **~60% lower than traditional peers** due to AI efficiency. |

---

## 4. Strategic Management Considerations

### The Agentic AI Advantage

To maintain the **3x productivity gain**, your management strategy must focus on:

1. **AI-Generated Verification:** Using agents to write SystemVerilog assertions and UVM sequences.
2. **Automated Documentation:** Keeping the hardware spec and the functional simulator in perfect sync using LLM-based translation.
3. **Human-in-the-Loop PD:** The PD SME acts as a "commander" for AI agents that run dozens of floorplan iterations overnight to find the optimal PPA (Power, Performance, Area).

### Risk Mitigation

* **The "Outsourcing" Gap:** Hardware/Software co-design is the biggest risk. Ensure the **Compiler SME** (Partner) has access to the **Functional Simulator** by Month 3.
* **Regularity vs. Overhead:** While the AI accelerator is "very regular," the **NoC and PCIe** often become the bottleneck. Prioritize the NoC SME’s work early in the FPGA phase.

---

To achieve a 3x productivity boost in Logic QA, your 3 SMEs act as "orchestrators" rather than "authors." They manage a fleet of AI agents that handle the heavy lifting of UVM (Universal Verification Methodology) boilerplate, stimulus generation, and debug triaging.

The following breakdown outlines how these agents are deployed across your 24-month timeline.

---

## Logic QA Month-by-Month Roadmap (AI-Augmented)

### Phase 1: FPGA & RTL (Months 1–6)

*Focus: Speed-of-light testbench creation and sanity checking.*

* **Month 1–2: Spec-to-UVM Agents.** SMEs feed architectural specs into agents to generate UVM environments (drivers, monitors, scoreboards). While the SMEs refine the memory model, agents generate 80% of the boilerplate code.
* **Month 3–4: Autonomous Fuzzing.** Agents generate random AI instruction streams (GEMMs, activations) to stress-test the ALU and SFU pipelines.
* **Month 5–6: FPGA Vehicle Support.** Agents assist in translating RTL assertions to FPGA-synthesizable checkers to ensure the FPGA vehicle matches the simulator's behavior.

### Phase 2: MPW & Test Chip (Months 7–12)

*Focus: Deep corner-case hunting and regression management.*

* **Month 7–8: Coverage-Driven Agents.** AI monitors functional coverage holes. Instead of a human writing directed tests, the agent analyzes which state-machine transitions are missed and automatically generates the stimulus to hit them.
* **Month 9: MPW Sign-off.** Agents run 24/7 regressions. When a failure occurs, an **Auto-Triage Agent** clusters the logs, identifies the root cause in the RTL, and suggests a fix to the Design SME.
* **Month 10–12: Formal Verification Agents.** Using formal tools to prove the absence of deadlocks in the NoC and PCIe interface. AI agents assist in writing complex SVA (SystemVerilog Assertions).

### Phase 3: The Road to Production (Months 13–24)

*Focus: Silicon-accurate verification and performance "golden" models.*

* **Month 13–18: Software-in-the-Loop (SiL).** Agents bridge the gap between the outsourced compiler and the RTL. They run compiled AI kernels (e.g., ResNet, Transformers) through the simulator to verify that the hardware executes the graph correctly.
* **Month 19–21: Gate-Level Simulation (GLS).** This is traditionally slow. AI agents optimize the simulation profile, focusing compute resources on high-risk timing paths identified by the PD SME.
* **Month 22–24: Final Production Tape-out.** A final "sanity agent" performs cross-checks between the architectural Golden Model, the RTL, and the final Netlist to ensure no regressions were introduced during physical optimization.

---

## How the "3x Productivity" is Realized

| Task | Traditional Effort | Agentic AI Effort | AI Role |
| --- | --- | --- | --- |
| **UVM Environment Setup** | 4–6 weeks | 1 week | Code generation from specs. |
| **Bug Triaging** | 2–4 hours / bug | 10 mins / bug | Automated log analysis & root-cause. |
| **Coverage Closure** | Months of manual tuning | Weeks | Genetic algorithms for stimulus gen. |
| **Documentation** | Continuous manual effort | Automated | Auto-generating test plans from RTL. |

---

## The "Agentic QA Stack"

To make this work, your 3 QA SMEs will need to maintain a specialized infrastructure:

1. **Vector Database:** Containing all previous bug reports and RTL versions so agents can "learn" your chip's specific quirks.
2. **Continuous Integration (CI) Agents:** Agents that don't just run tests, but automatically re-run failed tests with increased logging and debug hooks enabled.
3. **Natural Language Interfaces:** Allowing the Architect to ask, *"Did we ever test a PCIe read-retry during an HBM refresh?"* and having an agent immediately generate that specific scenario.

---

## Value-creation Milestones

This investment roadmap breaks down your development into five distinct financial and technical milestones. By leveraging **Agentic AI** with a 9-person SME core, you are essentially achieving the output of a Series B-scale organization on a "seed-plus" budget.

Below is the cost-to-deliverable breakdown and the 3-year ROI assessment tailored for an investor presentation.

---

## 1. Project Roadmap & Milestone Costs

*Estimated costs assume a 12nm/16nm FinFET process node (industry standard for specialized AI accelerators).*

| Milestone | Deliverable | Timeline | Est. Direct Cost | Primary Resource Focus |
| --- | --- | --- | --- | --- |
| **M1: fsim** | Functional C++ Simulator | Months 1–3 | **$0.4M** | Architect + AI Kernel SME |
| **M2: FPGA** | Validated RTL & Prototype | Months 3–6 | **$1.8M** | Design SMEs + Logic QA |
| **M3: MPW** | Core IP Silicon (Standard Cell) | Months 6–9 | **$1.2M** | PD SME + Foundry Broker |
| **M4: Test Chip** | I/O, PCB, & BSP Validation | Months 9–12 | **$4.5M** | All SMEs + Outsourced System/BIOS |
| **M5: Production** | Market-Ready Silicon | Months 12–24 | **$12.0M+** | Full Mask Set + Compiler Partner |

---

## 2. Detailed Deliverable Breakdown

### Milestone 1: fsim (The Software Enabler)

The **Architect** uses Agentic AI to auto-generate C++ class structures from the ISA (Instruction Set Architecture) spec.

* **Cost Drivers:** Architect salary + High-performance compute for simulation.
* **Deliverable:** A bit-accurate, cycle-approximate model.
* **Investor Value:** Enables the **Compiler/Driver partners** to begin work 18 months before silicon, significantly reducing "time-to-market" for the software stack.

### Milestone 2: FPGA Prototype (Functional Confidence)

The **Design SMEs** implement the RTL.

* **Cost Drivers:** High-end FPGA boards (e.g., AMD Virtex UltraScale+), EDA prototyping licenses, and Logic QA (3 SMEs using AI agents for regression).
* **Deliverable:** Hardware running at 1/10th or 1/100th clock speed but executing real AI kernels.
* **Investor Value:** De-risks the logic. Proves that the "straightforward" ALUs actually work with the AI kernels.

### Milestone 3: MPW (Standard Cell Proof)

A "smallish installation" of your core IP (e.g., 1–4 tiles instead of the full 64-tile array).

* **Cost Drivers:** Foundry shuttle fees ($300k–$500k) and Physical Design (PD) SME labor.
* **Deliverable:** A handful of dies that prove the standard cell library, timing closure flow, and power density.
* **Investor Value:** De-risks the physical implementation. Proves the design can be manufactured on the chosen process node.

### Milestone 4: Test Chip (System Integration)

This is a "full-system" chip but at lower volumes. It includes the PCIe and NoC interfaces.

* **Cost Drivers:** Higher fabrication fees, **OSAT** (Outsourced Semiconductor Assembly and Test) for packaging, and **Outsourced System Engineering** for the PCB/Board Support Package (BSP).
* **Deliverable:** A PCIe plug-in card that "heartbeats" in a Linux server.
* **Investor Value:** Validates the I/O, thermal profile, and BIOS. This is the "Point of No Return" for production investment.

---

## 3. 3-Year Resource Allocation (9 SMEs + AI)

To maintain the **3x productivity boost**, the team is structured as a **"Human-Agent Swarm."**

* **Year 1 (The Build):** Heavy focus on RTL and Logic QA. **Cost: ~$8.5M** (including EDA and FPGA hardware).
* **Year 2 (The Silicon):** Shift toward Physical Design and Hardware/Software integration (Outsourced BIOS/Compiler). **Cost: ~$15.0M** (includes Test Chip and Production Mask Set).
* **Year 3 (The Scale):** Post-silicon bring-up and customer sampling. **Cost: ~$6.0M** (Support, Qual, and Yield Optimization).

---

## 4. ROI & Investor Value Proposition

| Metric | Traditional Lean Team (25-30 People) | Your Team (9 SMEs + Agentic AI) |
| --- | --- | --- |
| **Total R&D OpEx (3yrs)** | $45M – $60M | **$25M – $30M** |
| **Time to Market** | 30 – 36 Months | **24 Months** |
| **Silicon Success Rate** | 85% (First Pass) | **95%+** (Due to AI-led verification) |

### Investor Takeaway

By keeping a **flat, SME-led organization** and using AI to handle the "junior engineer" tasks (testbench writing, log triaging, floorplan iterations), you are reducing your **Burn Rate by ~50%** while maintaining an aggressive production timeline.

**The ROI is realized through "Massive Cost Avoidance":** You are reaching the Production milestone with a significantly lower CapEx requirement, allowing for higher equity retention for founders and early investors.
