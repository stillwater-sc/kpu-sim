# KPU Compute Fabric classifications

Based on the Stillwater Supercomputing Domain Flow methodology for systolic arrays, here are five different naming conventions for classifying the array topology by the **SURE**, the timing vector **τ**, and the projection transformation **S**.

***

### 1. The Triplet Designation

This is the most direct and descriptive convention. It explicitly names the three core components that define the systolic array.

* **Format:** `(SURE, τ, S)`

* **Example:** `(MatrixMultiply, [1, 1], [1, 0])`

* **Utility:** This convention is useful for formal documentation and academic papers where precision and clarity are paramount. It leaves no ambiguity about the defining parameters.

***

### 2. The Domain-Scheduled Designator

This convention emphasizes the relationship between the computational domain, its timing, and the final physical realization. It’s a more abstract, high-level approach.

* **Format:** `DomainFlow(SURE | τ | S)`

* **Example:** `DomainFlow(FFT | [2, 1, 1] | [1, 1, 0])`

* **Utility:** This naming scheme is suitable for a software or compiler environment where the components are "pipe-lined" in a descriptive name, highlighting the flow from an abstract problem (the SURE) to a physical solution (S).

***

### 3. The Functional-Spacetime Signature

This convention combines the functional purpose of the array with its geometric properties in spacetime. It uses a concise alphanumeric signature to categorize the design.

* **Format:** `[Function]_[SURE-ID]_[τ-Vector-Hash]_[S-Matrix-Hash]`

* **Example:** `MM_sure_a_tau_b_S_c`

* **Utility:** This is ideal for a database or library of pre-synthesized designs where each unique combination of parameters needs a distinct, searchable identifier. The hashes ensure a compact, unique representation.

***

### 4. The Computational-Kinematic Identifier

Drawing an analogy from physics, this convention frames the design in terms of its computational "motion" and the "space" it inhabits.

* **Format:** `[SURE-Name]_kine_τ[τ-Vector]_proj_S[S-Matrix]`

* **Example:** `Matrix_Mult_kine_τ[1,1]_proj_S[1,0]`

* **Utility:** This method is a more expressive form of the triplet designation. It is particularly effective for presenting designs to hardware engineers or researchers familiar with the physical concepts of motion and projection, as it ties the mathematical concepts to a more intuitive physical interpretation.

***

### 5. The Topological-Morphological Code

This naming convention focuses on the final topological structure of the systolic array, which is a direct consequence of the three defining parameters.

* **Format:** `[Topology]_[Morphology]_[SURE-ID]_[τ-ID]`

* **Example:** `Linear_Array_MM_1_T_11`

* **Utility:** This is useful for high-level classification where the primary concern is the resulting physical layout of the array (e.g., a 1D linear array vs. a 2D mesh), rather than the specific mathematical vectors that produced it. It is less precise but provides a quick way to group similar physical designs.



