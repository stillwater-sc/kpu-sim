# Fused batched-MLP layer: SURE and domain-flow derivation

**Status:** Design (issue #46, epic #45)
**Operator:** `Y = activation( X · W + b )` — fused matmul + bias + activation
**Depends on:** `branes-ai/domain_flow` (`sw::dfa`, `KPU_HAS_DOMAIN_FLOW`)

## 1. Operator

A batched MLP layer over a batch of `B` samples:

```
X : [B, K]   (batch B, in-features K)
W : [K, N]   (weights)
b : [N]      (bias, broadcast over the batch)
Y : [B, N]   Y[i,j] = activation( Σ_k X[i,k]·W[k,j] + b[j] )
activation ∈ { ReLU, GELU, SiLU, Sigmoid, … }
```

We model this as a **single fused operator**, i.e. a single System of Uniform
Recurrence Equations (SURE) over one iteration domain — *not* three operators
(matmul, bias-add, activation) chained by inter-operator edges.

## 2. Why "fused" must merge the domains (not annotate three of them)

In a SURE an operator **is** a domain of computation: an index space plus a
system of recurrence equations whose dependence vectors are uniform over that
domain. Composing three separate operators yields **three domains joined by
inter-domain edges**; each edge means the producer's result is a *tensor that
leaves its domain and is communicated* to the consumer's domain.

**Unfused** (three domains, two communicated intermediates):

| Domain | Index space | Reads | Emits (leaves the domain) |
|--------|-------------|-------|---------------------------|
| `D₁` matmul | `(i,j,k)` | `X`, `W` | `A[i,j] = C(i,j,K-1)` |
| `D₂` bias  | `(i,j)`   | `A`, `b` | `Z[i,j] = A[i,j] + b[j]` |
| `D₃` act   | `(i,j)`   | `Z`      | `Y[i,j] = act(Z[i,j])` |

Marking these three nodes "fused" with metadata changes **none** of the
recurrence equations: the domains are still separate, the edges still imply
communicating `A` and `Z`, and the local recurrence that should connect the
accumulation result to the epilogue does not exist in the representation. Fusion
genuinely *changes the recurrence equations*, so a flag cannot express it.

**Fused** merges `D₂` and `D₃` onto `D₁`'s terminal face and rewrites the
epilogue as **boundary recurrences inside the single domain**, so neither `A`
nor `Z` is ever materialized or communicated.

## 3. The fused SURE

Single iteration domain:

```
D = { (i, j, k) ∈ ℤ³ : 0 ≤ i < B, 0 ≤ j < N, 0 ≤ k < K }
```

Bias/activation add **no iteration dimensions** — they are pointwise on the
`(i,j)` output face — so the fused domain is exactly the matmul domain.

### 3.1 Recurrence equations

**Input propagation (uniform broadcast):**

```
X(i, j, k) = X(i, j-1, k)        seeded at j = 0 from input X[i,k]     dep (0,1,0)
W(i, j, k) = W(i-1, j, k)        seeded at i = 0 from input W[k,j]     dep (1,0,0)
```

`X` is independent of `j` (reused across output columns); `W` is independent of
`i` (reused across the batch). The seeds inject the external tensors at the
`j=0` / `i=0` faces.

**Accumulation (reduction along k):**

```
C(i, j, k) = C(i, j, k-1) + X(i, j, k) · W(i, j, k)
C(i, j, -1) = 0                                                        dep (0,0,1)
```

**Epilogue on the terminal face k = K-1 (the fusion):**

```
Y(i, j) = activation( C(i, j, K-1) + b(j) )
```

`C` at the terminal face is consumed **in place** by the bias-add and
activation. The dependence epilogue→accumulation is **intra-domain** (a boundary
dependence within `D`), not an inter-domain edge. `b(j)` enters at the terminal
face (broadcast over `i`, `k`). The only value that leaves `D` is `Y(i,j)`.

### 3.2 Dependence summary

| Variable | Recurrence | Dependence vector | Meaning |
|----------|------------|-------------------|---------|
| `X` | `X(i,j,k)=X(i,j-1,k)` | `(0,1,0)` | input reuse across columns |
| `W` | `W(i,j,k)=W(i-1,j,k)` | `(1,0,0)` | weight reuse across batch |
| `C` | `C(i,j,k)=C(i,j,k-1)+X·W` | `(0,0,1)` | accumulation |
| `Y` | `Y(i,j)=act(C(i,j,K-1)+b(j))` | boundary (k=K-1) | fused epilogue, in place |

This is the energy-optimal transformation: the bias and activation cost no extra
data movement because they ride the accumulation's exit at the terminal face;
the matmul's intermediate `A` and the bias output `Z` never exist as tensors.

## 4. Schedule (space–time mapping)

`domain_flow` does **not** synthesize schedules (`generateSchedule()` is a stub);
we supply an explicit linear schedule τ. For the **output-stationary** dataflow
the simulator already uses:

- **Time** advances along the accumulation axis `k`: `τ = (0, 0, 1)` (or a skewed
  variant for the systolic fill/drain), so `time(i,j,k) = τ · (i,j,k) = k`.
- **Space** = the `(i,j)` output plane, mapped onto the systolic array; each PE
  owns one output `Y(i,j)` and accumulates over `k` in time. This is precisely
  why "output-stationary": the stationary operand is the accumulator `C(i,j,·)`.
- The epilogue fires once per PE at `k = K-1` (the fused tail), reusing the SFU
  for the activation.

Automated schedule **synthesis** (deriving τ from the dependence cone) is out of
scope for #46 and is filed separately — see §7.

## 5. Mapping to `sw::dfa` (domain_flow) primitives

Built with the standalone SURE primitives, so **no edit to domain_flow's
operator enum is required** (the enum only drives shape-based auto-elaboration,
which we bypass by constructing the recurrence system explicitly):

| Concept | domain_flow type | Header |
|---------|------------------|--------|
| Iteration domain `D` | `ConstraintSet<int>` → `IndexSpace<int>` | `dfa/constraint_set.hpp`, `dfa/index_space.hpp` |
| Domain of computation | `DomainOfComputation<int>` (node `.doc`) | `dfa/domain_of_computation.hpp` |
| Recurrence variables `X,W,C,b,Y` | `RecurrenceVariable` | `dfa/recurrence_var.hpp` |
| Dependence maps | `AffineMap<int>` (`f(x)=Ax+c`) | `dfa/affine_map.hpp` |
| Schedule τ / wavefronts | `ScheduleVector<int>`, `Schedule<int>` | `dfa/schedule.hpp` |
| Node / graph | `DomainFlowNode`, `DomainFlowGraph` | `dfa/domain_flow_node.hpp`, `dfa/domain_flow_graph.hpp` |

Construction sketch (one fused node, one domain):

1. Build `ConstraintSet` for `0≤i<B, 0≤j<N, 0≤k<K`; `IndexSpace::enumerate()`.
2. Create `RecurrenceVariable`s `X,W,C,b,Y`; attach dependences via `AffineMap`:
   `X.dependsOn(X, shift(0,1,0))`, `W.dependsOn(W, shift(1,0,0))`,
   `C.dependsOn(C, shift(0,0,1))`, plus the terminal-face epilogue for `Y`.
3. Attach the recurrence system + index space to a `DomainFlowNode`'s
   `DomainOfComputation`; tag the node (attribute `fused = matmul_bias_<act>`),
   map `b` and `Y` confluences to the `k=K-1` face.
4. `applyLinearSchedule(τ)` with `τ=(0,0,1)` to generate wavefronts; sanity-check
   latency/speed-of-light.

### Open verification item

If domain_flow's `DomainOfComputation` / `RecurrenceVariable` API turns out to
force everything through `elaborate(opType)` and cannot cleanly carry a custom
boundary recurrence, the minimal fallback is a small **upstream** addition to
`branes-ai/domain_flow` (a `FUSED_MATMUL_BIAS_ACT` elaboration). This would be
raised as an explicit decision, not a silent fork.

## 6. Bridge into kpu-sim

The derived fused domain feeds the simulator's existing IR:

- `ComputationalGraph` (`include/sw/compiler/graph_loader.hpp`) — single fused op
  node carrying shapes + activation kind.
- `TileDataFlowGraph` (`include/sw/kpu/dataflow/tile_dataflow_graph.hpp`) — the
  tiled lowering target consumed by #47.

Validation for #46: build the fused-MLP `DomainFlowGraph`, derive the domain +
index space, apply τ, and confirm the **single-tile** case reproduces the
reference `Y = act(X·W + b)` (numerically, against `compute_harness`).

## 7. Scope boundaries

- **#46 (this):** the fused SURE + derived domain-flow program (domains,
  dependence vectors, explicit output-stationary schedule) + bridge + single-tile
  validation.
- **#47:** lower the fused domain to a tiled `DMProgram` (matmul epilogue).
- **#48:** multi-tile tiling/reuse configs (1…33 L3 tiles).
- **Separate/new issue:** automated polyhedral schedule **synthesis** (domain_flow
  `generateSchedule()` is currently a stub).
- **Possible upstream:** a fused-operator auto-elaboration in `branes-ai/domain_flow`
  (only if the manual-construction path proves insufficient — see §5).
