\# TensorCore discussion



I am reading the NVIDIA docs but there is no explanation how the TensorCores actually work. Are they systolic arrays, are they fully unrolled dot product circuits, are they a 2D vector engine, etc. Can you help me understand what the TensorCore is and how it would for example compare to a Cerebras, SambaNova, or Hailo compute tile?



The NVIDIA documentation tends to focus on the programming interface rather than the underlying hardware implementation. 



\## \*\*How NVIDIA TensorCores Actually Work\*\*



The internals of TensorCores are not publicly disclosed by NVIDIA, but they are commonly assumed to be systolic arrays, similar to Google TPUs, though there is no consensus in the microbenchmarking literature. The circuits which implement matrix multiplication in both Google TPU and NVIDIA TensorCore adopt a systolic algorithm for matrix multiplication - a network of processing elements (PEs) that rhythmically compute and pass data through the system.



\*\*TensorCore Architecture Characteristics:\*\*



1\. \*\*Matrix Multiply-Accumulate (MMA) Units\*\*: TensorCores execute instructions like HMMA16.16816.F32 which calculates D = AB + C for matrices A, B, C, and D, where the MMA stands for "Matrix Multiply and Accumulate."



2\. \*\*CISC-like Behavior\*\*: It's helpful to think of TensorCores as a form of complex instruction set computer (CISC) hardware, since they perform many operations with a single instruction fetch, dramatically reducing power requirements.



3\. \*\*Warp-Level Parallelism\*\*: Different generations use different thread participation models - Volta requires 8 threads (quadpair) to participate in an MMA operation, while Ampere requires a full warp of 32 threads, simplifying the programming model.



4\. \*\*Specialized Memory Layouts\*\*: TensorCores require specific data layouts in memory that match their internal organization, with operations like ldmatrix loading data from shared memory to registers in layouts optimized for TensorCore computation.



\## \*\*Comparison with Other Architectures\*\*



\### \*\*Cerebras WSE (Wafer-Scale Engine)\*\*



\*\*Architecture\*\*: Cerebras uses a fine-grained dataflow architecture with 850,000 AI-optimized compute cores capable of individually ignoring zeros regardless of pattern, enabling automatic sparsity harvesting.



\*\*Key Differences from TensorCores\*\*:

\- \*\*Scale\*\*: WSE-3 has 900,000 cores on a single wafer compared to TensorCores which are just specialized units within GPU SMs

\- \*\*Memory\*\*: 44 GB of SRAM on-chip (880× more than H100), allowing models up to ~40 billion parameters to fit entirely on-chip with single-cycle access

\- \*\*Dataflow\*\*: WSE's dataflow architecture eliminates memory latency and bandwidth bottlenecks by leveraging localized communication between processing elements



\### \*\*SambaNova RDU (Reconfigurable Dataflow Unit)\*\*



\*\*Architecture\*\*: Uses a Reconfigurable Dataflow Architecture (RDA) that creates custom processing pipelines allowing data to flow through the complete computation graph, minimizing data movement.



\*\*Key Differences from TensorCores\*\*:

\- \*\*Programming Model\*\*: RDA contrasts with GPU's kernel-by-kernel programming model by creating dataflow patterns for each kernel's unique requirements

\- \*\*Tiled Design\*\*: RDU consists of a tiled array of reconfigurable processing and memory units connected through a high-speed, 3D on-chip switching fabric

\- \*\*Memory Architecture\*\*: Three-tiered memory architecture with terabytes of addressable memory, designed specifically for the largest generative AI models



\### \*\*Hailo-8 Neural Processor\*\*



\*\*Architecture\*\*: Uses a proprietary structure-driven dataflow architecture with distributed memory fabric and extremely efficient computational elements that can be applied variably as needed.



\*\*Key Differences from TensorCores\*\*:

\- \*\*Edge Focus\*\*: Self-contained RAM without external DRAM requirements, significantly decreasing latency and reducing power consumption for edge applications

\- \*\*Block Organization\*\*: Comprised of three types of blocks (control, memory, and compute) assigned to various layers of the neural network graph, with all processing happening on-chip

\- \*\*Power Efficiency\*\*: Delivers 26 TOPS at only 2.5W typical power consumption, optimized for edge deployment



\## \*\*Fundamental Architectural Differences\*\*



\### \*\*TensorCores (NVIDIA)\*\*

\- \*\*Hybrid Architecture\*\*: Specialized units within general-purpose GPU SMs

\- \*\*Memory Model\*\*: Hierarchical (registers → shared memory → L2 → HBM)

\- \*\*Programming\*\*: CUDA with explicit kernel launches and memory management

\- \*\*Strengths\*\*: Flexibility, mature ecosystem, can handle diverse workloads beyond AI



\### \*\*Dataflow Architectures (Cerebras, SambaNova, Hailo)\*\*

\- \*\*Pure Dataflow\*\*: Data streams through custom processing pipelines without traditional instruction fetch/decode overhead

\- \*\*Memory Model\*\*: Large on-chip memories (GB-scale SRAM) with direct access patterns

\- \*\*Programming\*\*: Compiler-driven with automatic dataflow graph generation from high-level frameworks

\- \*\*Strengths\*\*: Higher efficiency for AI workloads, lower power per operation, simplified programming for AI



The key insight is that while both approaches can implement systolic algorithms, TensorCores are embedded within a general-purpose GPU architecture, whereas the dataflow processors are purpose-built from the ground up for AI workloads. This fundamental difference explains why dataflow architectures often achieve better power efficiency and can handle larger models with fewer external dependencies, while TensorCores offer more flexibility and can leverage the broader GPU compute ecosystem.



\# Hailo's energy efficiency



\## \*\*How Hailo Achieves Impressive Power Efficiency\*\*



Hailo's power efficiency (delivering 26 TOPS at just 2.5W, or ~10.4 TOPS/W) comes from several fundamental architectural innovations that eliminate the major sources of energy waste in traditional computing:



\### \*\*1. Elimination of Instruction Overhead\*\*



The most significant efficiency gain comes from abandoning the Von Neumann architecture entirely:



In traditional CPU/GPU ISA-based systems, less than 10% of consumed energy is spent on executing the instruction's real function (e.g., "add"). The rest goes to instruction fetch, decode, branch prediction, and other overhead.



Hailo's dataflow architecture consumes only a fraction of the energy by eliminating instruction processing entirely - it expends additional energy only on data access, resulting in much smaller per-instruction overhead.



\### \*\*2. Domain-Specific Dataflow Architecture\*\*



Deep Learning neural networks require highly repetitive computing operations based on the simple MAC (Multiply and Accumulate) operation. This dataflow architecture with simple repetitive operations is highly power efficient compared to general-purpose architectures since there is no need for instructions and the compute engine is highly simplified.



\*\*Key architectural elements:\*\*



\- Innovative control scheme combining hardware and software to reach very low joules/operation with high flexibility

\- Distributed memory fabric with purpose-built pipeline elements that allow very low-power memory access in neural network processing

\- Extremely efficient computational elements that can be applied variably, as needed

\- Dataflow-oriented interconnect that adapts to the neural network structure and allows high resource utilization



\### \*\*3. Self-Contained Memory Architecture\*\*



A key reason for the performance improvement is that RAM is self-contained without the need for external DRAM like other solutions. This decreases latency significantly and reduces power consumption.



Hailo-8 includes a large amount of on-chip memory that is tightly integrated with the compute elements to enable efficient data processing and reduce data movement, which helps further improve power efficiency. Unlike other AI accelerators, it does not require additional external memory for efficient operation.



\### \*\*4. Spatial Resource Allocation and Data Locality\*\*



Memory, control and compute blocks allocated to each layer are mapped onto the chip as close to each other as possible, and subsequent layers are fitted in close proximity. The idea is to minimize the distance the data has to travel.



"We are not doing any tiling, we are not doing any compression, we are not doing any sparsity, none of the techniques that traditional computation architectures need to do to overcome the bandwidth issue," because the architecture itself solves the fundamental data movement problem.



\### \*\*5. Three-Block Architecture Design\*\*



The Hailo-8 chip is comprised of three types of blocks - control, memory, and compute - which are assigned to various layers of the neural network graph. All processing takes place internally within the chip.



\*\*Efficiency benefits:\*\*

\- \*\*No external memory bandwidth bottlenecks\*\*

\- \*\*Minimal data movement between processing stages\*\*

\- \*\*No kernel launch overhead\*\* (unlike GPUs that require kernel-by-kernel execution)

\- \*\*Optimal resource utilization\*\* based on actual neural network requirements



\### \*\*6. Hardware-Software Co-Design\*\*



The Hailo Dataflow Compiler is co-designed with the hardware architecture, enabling efficient deployment of neural network models. During compilation, it decomposes each network layer into necessary computational elements and maps the resource graph to physical resources, creating a customized data pipeline for maximum performance and efficiency.



\### \*\*7. Comparison with Traditional Architectures\*\*



\*\*Energy breakdown comparison:\*\*

\- \*\*Traditional ISA (CPU/GPU)\*\*: <10% on actual computation, >90% on instruction overhead, memory hierarchy management, and control logic

\- \*\*Hailo Dataflow\*\*: Most energy goes directly to computation and essential data access, with minimal control overhead



\*\*Why GPUs are still inefficient for this use case:\*\*

While GPUs were a step towards specialization for graphics applications, they achieve greater efficiency than CPUs by dividing instruction overhead between large amounts of data executions. However, they are still ISA-based machines and hit the same evolutionary wall regarding power efficiency.



\### \*\*8. Real-World Power Efficiency\*\*



Unlike theoretical TOPS/W calculations that divide maximum theoretical compute capacity by maximum power consumption (which can be misleading), Hailo's power efficiency is measured under real workloads and maintains consistency as workloads vary and scale.



The result is that Hailo-8 achieves about 20 times the power efficiency of comparable NVIDIA solutions while delivering similar or better performance in a much smaller form factor.



\## \*\*Key Architectural Insight\*\*



The fundamental insight is that neural networks have very predictable, repetitive dataflow patterns. By building hardware that directly implements these patterns without the overhead of general-purpose instruction processing, Hailo achieves dramatic power reductions. The architecture essentially becomes a "hardwired" neural network processor where data flows through dedicated pathways rather than being managed by a general-purpose control unit.



This is why domain-specific dataflow architectures represent the next stage of compute evolution - as data processing needs grow larger, previously neglected power efficiency becomes critical and guides away from flexible but power-hungry ISA architectures toward specialized solutions.



\# Hailo's On-Chip Execution Model constraint



\## \*\*Hailo Memory Architecture: Two Different Approaches\*\*



\### \*\*Hailo-8 Series: Memory-Constrained but Highly Efficient\*\*



The Hailo-8 "does not require external memory" and is designed for edge AI workloads, which means you're correct - it can only run models that fit entirely in its on-chip memory along with all intermediate results.



\*\*What Hailo-8 CAN run:\*\*

\- Computer vision models (object detection, segmentation, classification)

\- Small transformer models for vision tasks

\- Real-time video analytics at 1080p/4K

\- Edge AI inference workloads with limited memory footprint



\*\*What Hailo-8 CANNOT run:\*\*

\- Large language models requiring GB of memory

\- Large transformer models (GPT-style with billions of parameters)

\- Any model where weights + activations exceed on-chip memory



\### \*\*Hailo-10H: The Solution for Large Models\*\*



Recognizing this exact limitation, Hailo developed the Hailo-10H specifically for larger models:



The Hailo-10H includes a direct DDR interface, allowing it to scale for large models such as LLMs, VLMs, Stable Diffusion, and more



The Hailo-10H M.2 module comes with 8GB of LPDDR4 on-module memory and is available with 4GB or 8GB LPDDR4/4X configurations



When running LLMs on the Hailo-10H, the entire LLM pipeline is offloaded to the accelerator while minimizing impact on host processor utilization and DRAM capacity



\## \*\*Performance Examples on Hailo-10H:\*\*



The Hailo-10H can run Llama2-7B LLM at up to 10 tokens per second or Stable Diffusion 2.1 image generation at 5 seconds per image, both while drawing under 5W of power



It achieves first-token latency of under 1 second and over 10 tokens per second on various 2B language and vision-language models



\## \*\*The Trade-off: Efficiency vs. Model Size\*\*



Your observation highlights a fundamental trade-off in AI accelerator design:



1\. \*\*Hailo-8 approach\*\*: Maximum power efficiency by eliminating external memory access, but limited to smaller models that fit on-chip

2\. \*\*Hailo-10H approach\*\*: Adds external memory interface to support larger models, with some efficiency trade-offs but still much more efficient than GPUs



Unlike traditional edge accelerators focused primarily on vision tasks, the Hailo-10H's architecture includes a direct DDR interface for scaling larger models, which addresses one of the key bottlenecks in LLM and VLM inference at the edge



\## \*\*Community Recognition of This Limitation\*\*



The Hailo community has indeed recognized this limitation. Multiple community posts ask about running LLMs on Hailo-8, with users wondering if the Hailo-8L can assist in running local quantized PyTorch models - essentially asking the same question you raised.



So you're completely correct: the original Hailo-8's "no external memory" approach means it cannot run large transformer models. Hailo-10H was specifically designed to address this limitation while maintaining much of the power efficiency benefits of their dataflow architecture.

