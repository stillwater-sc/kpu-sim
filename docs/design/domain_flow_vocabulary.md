# Domain Flow Architecture Vocabulary

This document defines the vocabulary used throughout the KPU simulator, documentation, and code.
These distinctions are critical for correctly understanding systolic array behavior.

---

## Fundamental Distinction: Memory Layout vs Execution Sequencing

### Memory Layout (Static)
Describes how tensors are organized in memory. **No reduction dimension exists here.**

| Tensor | Element Shape | Element Type | Size | Block Structure | Tile Size |
|--------|---------------|--------------|------|-----------------|-----------|
| A | [2048, 512] | int32 | 4MB | 64×16 block matrix | 32×32 |
| B | [512, 1024] | int32 | 2MB | 16×32 block matrix | 32×32 |
| C | [2048, 1024] | int32 | 8MB | 64×32 block matrix | 32×32 |

- **A** is a 64×16 block matrix of 32×32 tiles (64 block-rows, 16 block-columns)
- **B** is a 16×32 block matrix of 32×32 tiles (16 block-rows, 32 block-columns)
- **C** is a 64×32 block matrix of 32×32 tiles (64 block-rows, 32 block-columns)

### Execution Sequencing (Dynamic)
The reduction dimension "k" appears ONLY when describing the execution sequence
of a block matrix multiply operation. It is NOT a property of the tensor memory layout.

---

## Systolic Array vs Stored-Program Machine

### Stored-Program Machines (CPU, GPU)
- Explicit k-loop in code: `for k in range(K): C[i,j] += A[i,k] * B[k,j]`
- Dynamic resource contention requires arbitration
- Reduction dimension visible in program structure

### Systolic Arrays (Domain Flow Architecture)
- **No explicit k-loop** - reduction happens implicitly through spatial dataflow
- **Choreographed data movement** - no dynamic resource contention
- Data flows through the array; accumulation is a natural consequence of dataflow
- Reduction dimension implicit in the spatial arrangement and timing

---

## Tile Vocabulary

### Correct Usage

| Term | Meaning |
|------|---------|
| Input tile | A tile from tensor A or B being fed into computation |
| Output tile | A tile from tensor C produced by computation |
| Block position A{i,j} | Tile at block-row i, block-column j of tensor A |
| Block position B{i,j} | Tile at block-row i, block-column j of tensor B |
| Block position C{i,j} | Tile at block-row i, block-column j of tensor C |

### Avoid These Terms
- ~~k-tile~~ (conflates memory layout with execution)
- ~~row-tile / col-tile / k-tile~~ (ambiguous, suggests loop structure)

### Preferred Descriptions

**Memory layout:**
> "A is organized as a 64×16 block matrix of 32×32 tiles"

**Execution sequence:**
> "To compute C{0,0}, we sequence through block-column 0..15 of A's row 0,
> paired with block-row 0..15 of B's column 0"

---

## Computation Description

When describing the computation of an output tile C{i,j}:

**Correct:**
> C{i,j} is computed by streaming A's block-row i and B's block-column j
> through the systolic array. The partial products accumulate spatially
> as data flows through the compute elements.

**Avoid:**
> ~~C{i,j} = Σ(k=0..K) A{i,k} × B{k,j}~~ (implies explicit loop iteration)

---

## Data Movement Paths

For a 4×4 L3 mesh computing C = A × B:

| Tensor | Entry Edge | Flow Direction | Purpose |
|--------|------------|----------------|---------|
| A | WEST | Flows EAST | Input tiles for left operand |
| B | NORTH | Flows SOUTH | Input tiles for right operand |
| C | SOUTH/EAST | Flows to edge | Output tiles to memory |

---

## Summary

1. **Tensors have memory layout** - described as block matrices of tiles
2. **Execution has sequencing** - the order in which tiles are processed
3. **Reduction is implicit** - it happens through dataflow, not loops
4. **No dynamic contention** - all data movement is choreographed

This vocabulary must be used consistently in:
- Documentation (README files, design docs)
- Code comments and variable names
- Test descriptions and trace event naming
- Visualization labels and legends
