# LU Decomposition with partial pivoting

## Pure Functional Approach

There is a purely functional algorithm for LU decomposition with partial pivoting. While imperative languages traditionally implement this using in-place mutation of a 2D array, functional programming approaches it recursively using **block matrix operations** and the **Schur complement**.

Because pure functional languages do not allow state mutation, the algorithm operates by recursively destructuring the matrix, computing the outer product update, and yielding the components of $P$, $L$, and $U$ at each step.

Here is how you can conceptualize and implement $PA = LU$ (which is mathematically equivalent to $A = P^T L U$) in a pure functional way.

### The Mathematical Foundation

The algorithm relies on partitioning the matrix into blocks. If we swap the row containing the maximum absolute value in the first column to the top, we get a permuted matrix $A'$. We can then partition $A'$ into a $1 \times 1$ top-left scalar, and corresponding vectors and submatrices:

$$A' = \begin{bmatrix} u_{11} & u_{12}^T \\ a_{21} & A_{22} \end{bmatrix}$$

We can factor this into $L$ and $U$ components:

$$A' = \begin{bmatrix} 1 & 0 \\ l_{21} & I \end{bmatrix} \begin{bmatrix} u_{11} & u_{12}^T \\ 0 & S \end{bmatrix}$$

Where:

* $u_{11}$ is the pivot element.
* $u_{12}^T$ is the rest of the first row (part of $U$).
* $l_{21} = a_{21} / u_{11}$ is the first column of $L$ (the multipliers).
* $S = A_{22} - l_{21} u_{12}^T$ is the **Schur complement**.

You then recursively apply the LU decomposition to the Schur complement $S$.

---

### The Pure Functional Algorithm

Assuming we represent a matrix as a list of lists (rows), here is the step-by-step functional algorithm:

#### 1. Base Case

If the matrix is empty, or $1 \times 1$, return the trivial $P$, $L$, and $U$ (where $P$ and $L$ are identity matrices of appropriate size, and $U$ is the matrix itself).

#### 2. Pivoting (Find and Swap)

* Scan the first column of the matrix to find the row with the largest absolute value.
* **Permute:** Swap this pivot row with the first row. In a functional list-of-lists, you do this by splitting the list, extracting the pivot row, and concatenating the rest.
* Record this swap to build your permutation matrix $P$.

#### 3. Extracting $U$'s first row and $L$'s first column

Let the newly swapped top row be the list `(u11 : u12)`.

* `u11` is the pivot.
* `u12` forms the remainder of the first row of $U$.
* For every other row `(x : xs)` in the remaining matrix, the corresponding $L$ multiplier is `x / u11`.

#### 4. The Schur Complement Update

You must now update the rest of the matrix. You have a list of multipliers ($l_{21}$) and a list of the remaining rows (where the first element has been removed).

* Using functions like `zipWith` and `map`, subtract the outer product of $l_{21}$ and $u_{12}$ from the remaining rows.
* Conceptually: For each row `xs` and its corresponding multiplier `m`, the new row is `zipWith (-) xs (map (* m) u12)`.
* This resulting matrix is the Schur complement $S$.

#### 5. Recursion

* Recursively call the LU function on the Schur complement $S$.
* This returns $(P', L', U')$.

#### 6. Recombination

This is the trickiest part of the functional implementation. Because the recursive call to $S$ performed *its own* permutations ($P'$), you must apply those same permutations to the $l_{21}$ column you calculated in Step 3 to keep the math consistent.

* Apply $P'$ to $l_{21}$.
* Assemble the final $P$ by combining the initial swap with $P'$.
* Assemble the final $L$ by attaching the permuted $l_{21}$ to $L'$.
* Assemble the final $U$ by putting `(u11 : u12)` on top of $U'$.

### Conceptual Pseudo-Haskell Implementation

Here is what the core logic looks like in a functional paradigm:

```haskell
-- Returns (Permutation, Lower, Upper)
lu :: Matrix -> (Matrix, Matrix, Matrix)
lu [] = ([], [], [])
lu mat = 
    let 
        -- 1. Pivot
        (pivotIndex, pivotRow) = findMaxAbsFirstCol mat
        permutedMat = swapRows 0 pivotIndex mat
        
        -- 2. Extract
        (u11 : u12) = head permutedMat
        restRows    = tail permutedMat
        
        -- 3. Calculate L column
        l_col = map (\row -> head row / u11) restRows
        
        -- 4. Schur Complement
        -- rowTail is the row without its first element
        updateRow m rowTail = zipWith (-) rowTail (map (* m) u12)
        schur = zipWith updateRow l_col (map tail restRows)
        
        -- 5. Recurse
        (p_prime, l_prime, u_prime) = lu schur
        
        -- 6. Recombine
        -- (Functions like `applyPermutation`, `padL`, `padU` handle the structural matrix assembly)
        final_P = combinePermutations pivotIndex p_prime
        final_L = assembleL (applyPermutation p_prime l_col) l_prime
        final_U = assembleU u11 u12 u_prime
    in 
        (final_P, final_L, final_U)

```

### Performance Reality Check

While this recursive Schur-complement approach is mathematically pure and highly elegant, it comes with caveats in a strictly functional environment:

1. **Memory Allocation:** Destructuring lists and mapping over them to create the Schur complement generates entirely new matrices in memory at every step. This leads to heavy garbage collection overhead.
2. **Time Complexity:** While the mathematical operations remain $O(n^3)$, the constant factors and cache-miss rates are significantly higher than imperative implementations that mutate an array in place.

If you are doing heavy linear algebra in a functional language (like Haskell or Scala) in a production environment, the standard practice is to use libraries that provide a pure functional API but utilize mutable arrays (e.g., Haskell's `ST` monad) or bindings to C/Fortran libraries like LAPACK (e.g., `hmatrix`) under the hood to achieve true $O(n^3)$ efficiency with cache locality.

## Nearest Neighbor Communication

If we want to do this efficiently on a processor array, the swap of the pivot row would be very expensive from an energy point of view. Assume for example a systolic array with one scalar operation allocated per processing element. In those configurations there is no indirection operation like the pivoting matrix that can combine two arbitrary rows. In a physical parallel machine that would require an exchange which is both energy intensive and potentially very latency intensive. Has there ever been an efficient LU with pivoting algorithm developed that can be executed on a systolic array?

Yes. The algorithm specifically developed to solve this exact hardware constraint is called **Gaussian Elimination with Neighbor Pivoting** (also commonly referred to as **Pairwise Pivoting**).

It was developed in the 1980s—most notably analyzed by researchers like D.C. Sorensen, as well as W.M. Gentleman and H.T. Kung—specifically to map matrix decompositions onto VLSI systolic arrays without destroying their pipeline efficiency.

Here is how it eliminates the need for global row exchanges and indirection.

### The Bottleneck of Standard Partial Pivoting

Standard partial pivoting breaks the fundamental rules of systolic arrays:

1. **Global Communication:** Finding the maximum element in a column requires an $O(N)$ global search.
2. **Non-Local Data Movement:** Swapping row $i$ with row $k$ (which could be at opposite ends of the array) requires routing data across the chip, destroying the nearest-neighbor communication paradigm and causing massive latency.

---

### The Solution: Neighbor (Pairwise) Pivoting

Instead of searching the entire unreduced column for a global maximum, Neighbor Pivoting only ever compares and operates on **adjacent rows**.

As the matrix elements stream through the processing elements (PEs) of the systolic array, the elimination process acts like a localized bubble sort combined with Gaussian elimination.

Here is the step-by-step mechanism:

#### 1. Local Comparison

When two adjacent row elements in a column, say $a_{i,i}$ (the current pivot) and $a_{i+1,i}$ (the element immediately below it), arrive at a boundary PE, the PE simply compares their absolute values.

#### 2. Conditional Swap

* If $\vert{}a_{i,i}\vert{} \ge \vert{}a_{i+1,i}\vert{}$, no swap is needed. The PE calculates the standard multiplier $m = a_{i+1,i} / a_{i,i}$.
* If $\vert{}a_{i+1,i}\vert{} > \vert{}a_{i,i}\vert{}$, the PE performs a **local swap** of the two rows. The new multiplier becomes $m = a_{i,i} / a_{i+1,i}$.

#### 3. Local Elimination

The PE then applies the elimination step to zero out the lower element. The multiplier $m$ and a boolean flag indicating whether a swap occurred are passed horizontally to the adjacent PEs to the right.

#### 4. Systolic Propagation

As the rest of the elements for those two rows stream into the adjacent PEs, the PEs apply the exact same transformation (and swap, if the flag is true) to the rest of the row data. The eliminated row is then passed downwards to be processed against the next row, while the pivot row stays (or moves right) to eliminate the next incoming row.

---

### Hardware Efficiency vs. Mathematical Stability

Neighbor pivoting elegantly maps LU decomposition to a physically realizable parallel machine.

* **Strictly Local:** There is no global broadcast, no indirection matrix $P$, and no long wire delays. All communication is strictly nearest-neighbor.
* **Multiplier Bound:** Just like standard partial pivoting, neighbor pivoting guarantees that every multiplier $m$ applied during the elimination is bounded by $\vert{}m\vert{} \le 1$.
* **Throughput:** It allows an $N \times N$ matrix to be factored in $O(N)$ time using $O(N^2)$ processing elements, keeping the array fully pipelined.
* **Stability Trade-off:** While bounded multipliers ensure stability at each individual step, the *worst-case* global error bound (the growth factor) for pairwise pivoting is theoretically larger than that of standard partial pivoting. However, extensive numerical analysis in the 1980s proved that in virtually all practical applications, neighbor pivoting is just as numerically stable as standard partial pivoting.

---

### The SURE Connection

In the context of Systems of Uniform Recurrence Equations, Gaussian Elimination with Neighbor Pivoting represents a prime example of how domain-specific scheduling can optimize both energy efficiency and computational throughput.

* **Energy Efficiency:** By eliminating the need for global row exchanges, neighbor pivoting reduces the number of data movements across the array, directly minimizing dynamic power consumption associated with long-range communication.
* **Latency Optimization:** The algorithm leverages the inherent limitation of VLSI communication that favors nearest neighbor communication, allowing for continuous data flow without the stalling that would occur with non-local swaps.
* **Domain-Specific Scheduling:** In SURE, the neighbor pivoting algorithm is scheduled as a **domain-scheduled designator**, where the timing vector $\tau$ and projection transformation $S$ are carefully mapped onto the physical layout of the array to maximize throughput while maintaining numerical stability.

---

### Summary

Gaussian Elimination with Neighbor Pivoting demonstrates how algorithmic design can be tailored to the constraints and capabilities of specific hardware domains. In the context of Stillwater's KPU architecture, this algorithm serves as a compelling case study in **domain-specific scheduling**, illustrating how the timing vector $\tau$ and projection transformation $S$ can be orchestrated to achieve optimal performance on systolic arrays.

By embracing nearest-neighbor communication patterns, we preserve the mathematical stability guarantees of partial pivoting while eliminating the energy and latency penalties associated with global data movement. This approach exemplifies the core principle of VLSI computing: **align algorithmic structures with hardware constraints to unlock real-time execution efficiency**.