# Productive AI Assistant Code Generation Methodology

Question: I am seeing the AI Code Assistant produce code without comprehension and synthesize subtle bugs. When I point that out, the AI Assistant produces a correction, again without comprehesion and synthesizes a new set of subtle bugs. This cycle repeats never solving the problem, and sometimes revisiting old proven incorrect code. How can I avoid this productivity sink? This is all modern C++ code to model a dynamic computer system.

---
## Claude.ai answer

This is a well-known phenomenon often called the **"Stochastic Loop"** or "AI Hallucination Cycle." Because LLMs predict the next most likely token based on patterns rather than executing a logical mental model of the system, they can struggle with the strict structural requirements of modern C++ (C++20/23), especially in complex domains like dynamic system modeling.

To break this cycle and turn the assistant back into a productivity tool, you need to shift from "conversational debugging" to **"constrained architectural prompting."**

### Stop the "Apology-Correction" Loop

When you tell an AI "That's wrong," its training pushes it to provide a *different* answer, not necessarily a *correct* one. This often leads it to revert to deprecated patterns or "fix" a bug by introducing a more obscure one.

* **The Fix:** Do not ask it to "fix" the code in the same thread once the loop starts. **Reset the context.** Start a new prompt that includes the flawed code and explicitly lists the constraints it violated.
* **The Rule:** If it fails twice on the same logic, stop. It has likely "lost the thread" of the scope.

### Use "Chain of Thought" with Pseudo-Code First

LLMs often fail in C++ because they try to manage syntax, memory safety (RAII), and logic simultaneously.

* **The Strategy:** Ask the AI to write the **algorithm in Markdown bullet points or pseudo-code first.**
* **Why it works:** You can audit the logic *before* any code is generated. If the logic in the bullet points is flawed, the code will be too. Correct the logic in English first, then ask: *"Now, implement this validated logic using C++20 standard library features."*

### Define the "System of Truth" (The Header)

In dynamic system modeling, bugs often arise from unclear ownership or state transitions. AI tends to guess these.

* **The Strategy:** Provide the Assistant with a "Ground Truth" block.
* **Example:** > "Here is my `SystemState` struct and my interface for `IDynamicComponent`. Using **only** these definitions and `std::variant` for state handling, implement the `Update` loop. Do not introduce new member variables."

### Constraint-Based Prompting

Modern C++ is vast. If you don't specify the "how," the AI might mix C-style pointers with `std::shared_ptr`, or use legacy loops instead of Ranges.

* **Add a "Strict Constraints" section to your prompt:**
* "Use `std::span` for buffer passing to avoid copies."
* "Use `consteval` or `constexpr` where possible for system Constants."
* "No manual memory management; use RAII."
* "If a state transition is invalid, throw a custom `SystemException`."

### The "Rubber Duck" Prompting Method

Instead of asking for code, ask the AI to **critique your current implementation** or explain why a specific bug might be occurring in a "Memory Safety" context.

* **Try this:** "Analyze this C++ code for potential race conditions or iterator invalidation in the `std::vector` of components. Do not rewrite the code yet; just list the risks."

#### Comparison of Workflow Styles

| Traditional Prompting (Sink) | Structural Prompting (Productive) |
| --- | --- |
| "Fix the bug in this update loop." | "Explain the logic of this loop, then identify where the state transition fails." |
| "Rewrite this to be faster." | "Apply `std::execution::par` to this algorithm, ensuring no data races on the `Results` vector." |
| Continuous chatting in one thread. | Fresh threads for new components to avoid "context drift." |

#### Example of a "Circuit Breaker" Prompt

If you are stuck in a loop, use a prompt like this to reset:

> "We are stuck in a loop. Let's start over. I am modeling a dynamic system with the following requirements: [List 1, 2, 3].
> **Step 1:** Describe the mathematical logic for the state update.
> **Step 2:** I will verify that logic.
> **Step 3:** Once verified, you will provide a C++20 implementation using `std::optional` for error handling."


---
## ChatGPT Answer

You are describing a well-known failure mode of code-generation LLMs: *local syntactic repair without a global semantic model*. In C++—especially modern, template-heavy, stateful, or performance-sensitive C++—this can become a severe productivity sink if not managed deliberately.

Below are **practical, discipline-level strategies** that experienced teams use to avoid exactly the loop you describe.

### Stop Asking for “Code”; Ask for **Models and Invariants First**

LLMs are weak at *maintaining* a consistent conceptual model across iterations unless you force one explicitly.

**Before any code generation**, require the model to state:

* System boundaries
* State variables and ownership rules
* Invariants (must always be true)
* Temporal constraints (what may change, when)
* Failure modes

**Example prompt pattern:**

> “Do not write code yet.
> First, describe the system as a state machine with invariants.
> Explicitly list ownership, lifetime, and concurrency assumptions.”

Only once that model is agreed do you permit code.

This prevents the model from “patching symptoms” because it now has commitments it must preserve.

### Force a **One-Way Refinement Pipeline**

The failure you observe happens because the AI Assistant is allowed to *rewrite freely*.

Instead, enforce this rule:

> **Each iteration may only refine or constrain previous code, never replace it wholesale.**

Concrete techniques:

* Require diffs instead of full rewrites
* Require justification for every change
* Require explicit mapping from bug → violated invariant → fix

**Prompt constraint:**

> “You may only modify lines directly related to the stated invariant violation.
> All other behavior must remain identical.”

This sharply reduces regression churn.

### Make the Model Prove It Understands the Bug

When you point out a bug, do **not** ask for a fix immediately.

First require a *formal explanation*:

* What assumption was incorrect?
* What invariant was violated?
* Why the original code *appeared* correct
* Why the proposed fix prevents recurrence

If the explanation is shallow or circular, the fix will be wrong.

**Red flag:** phrases like *“this should fix”*, *“likely caused by”*, or *“to be safe”*.

### Use “Executable Specifications” as the Primary Interface

Instead of iterating on implementation code, iterate on:

* Property-based tests
* Compile-time assertions
* Contracts (`[[expects]]`, `[[ensures]]`, static_asserts)
* Deterministic simulation harnesses

Have the AI Assistant **only** add or refine tests until they capture the behavior.

Then:

> “Modify the implementation until all tests pass. No new tests may be added.”

This flips the power dynamic: the model must now conform to a specification rather than invent logic.

### Partition the System Along Cognitive Fault Lines

LLMs are particularly bad at **cross-cutting concerns**, which your “dynamic computer system” almost certainly has.

Partition aggressively:

* Deterministic core vs orchestration
* State transition logic vs resource management
* Policy vs mechanism
* Compile-time structure vs runtime behavior

Only let the AI Assistant operate on **one partition at a time**, with others treated as black boxes.

This mirrors how FPGA-based systems are validated: tight contracts, narrow interfaces, zero ambiguity.

### Lock Down “Proven Correct” Code Explicitly

If the model keeps revisiting known-bad solutions, it is because nothing tells it *they are forbidden*.

You must say so explicitly:

> “The following approaches are incorrect and must never be reintroduced:
>
> 1. …
> 2. …
> 3. …
>    Reintroducing them is a hard failure.”

LLMs do not infer historical rejection unless you state it.

### Know When **Not** to Use AI Assistants

For systems like yours (dynamic, stateful, performance-critical C++), the AI Assistant is best used for:

* Scaffolding
* Boilerplate
* Interface sketches
* Test generation
* Refactoring under tight constraints

It is **not** reliable for:

* Emergent system behavior
* Non-local invariants
* Subtle lifetime or aliasing rules
* Concurrency correctness
* Performance-driven designs

If you find yourself debugging the AI Assistant more than you would a junior engineer, you are past the breakeven point.

### A Mental Reframe That Helps

Treat the AI Assistant as:

> A very fast, tireless **syntax-level assistant**, not a reasoning peer.

You provide:

* Architecture
* Invariants
* Constraints
* Acceptance criteria

the AI Assistant provides:

* Textual completion under those constraints

When roles are reversed, the failure mode you describe is inevitable.


### Bottom Line

You are not doing anything wrong—your observation is accurate.
The solution is **process discipline**, not better prompting verbosity.

---

# Model Invariants

Below is a **practical, engineering-grade method** for designing a *formal invariant checklist* suitable for a **dynamic, stateful modern C++ system**—especially one that resembles a “computer system” rather than a pure algorithm. This is written to be *used*, not admired.

The goal is to create a checklist that:

* Prevents AI Assistant/LLMs from inventing semantics
* Prevents regressions across iterations
* Encodes system understanding in a form you can mechanically enforce

---

## 1. What an Invariant Checklist Actually Is (and Is Not)

An invariant checklist is **not** documentation and **not** prose.

It is a **contractual control surface** between:

* Your mental model
* The codebase
* Any automated assistants

Every item must be:

* Binary (true / false)
* Locally checkable
* Attached to a scope (type, module, phase, thread)

If an invariant cannot be mechanically checked or logically falsified, it does not belong on the checklist.

---

## 2. The Six Invariant Classes You Should Always Define

For dynamic systems, invariants fall into **six non-overlapping classes**. Skipping any of these guarantees bugs.

### 1. **Structural Invariants** (Shape of the System)

These define what *exists*.

Examples:

* Exactly one `Scheduler` exists per system instance
* Each `Node` belongs to exactly one `Cluster`
* Graphs are acyclic / DAG / allowed cycles with constraints

**Checklist form:**

* [ ] Cardinality constraints
* [ ] Containment rules
* [ ] Graph properties

**C++ enforcement tools:**

* `static_assert`
* Private constructors
* Non-copyable / non-movable types
* Factory-only creation

---

### 2. **Ownership & Lifetime Invariants** (The #1 Source of Subtle Bugs)

These define *who is allowed to destroy what, and when*.

Examples:

* All `Device` objects outlive any `Task` referencing them
* No shared mutable ownership of system state
* References never escape the owning subsystem

**Checklist form:**

* [ ] Single owner vs shared owner
* [ ] Borrowed vs owning references
* [ ] Destruction order constraints

**C++ enforcement tools:**

* `unique_ptr` as default
* `observer_ptr` / raw pointer only for non-owning
* Explicit lifetime comments tied to types
* `= delete` copy/move where unsafe

---

### 3. **State Invariants** (Legal Configurations)

These define which combinations of state are valid.

Examples:

* A `Core` cannot be `Running` unless `PowerState == On`
* A `Task` cannot be both `Scheduled` and `Completed`
* Buffer size must be nonzero when DMA is enabled

**Checklist form:**

* [ ] Valid states
* [ ] Forbidden state combinations
* [ ] Initial and terminal states

**C++ enforcement tools:**

* `enum class` + exhaustive `switch`
* `std::variant` for mutually exclusive state
* `constexpr` state validation functions
* Debug-mode `assert` + release-mode assumptions

---

### 4. **Transition Invariants** (How State Changes)

These define *what is allowed to change*, and *what must not*.

Examples:

* State transitions are monotonic
* No transition occurs while the system is paused
* Only the scheduler thread may transition tasks to `Running`

**Checklist form:**

* [ ] Allowed transitions
* [ ] Transition preconditions
* [ ] Postconditions

**C++ enforcement tools:**

* Transition tables
* Centralized state mutation APIs
* `[[nodiscard]]` results for transitions
* RAII guards for critical sections

---

### 5. **Temporal & Ordering Invariants** (When Things Happen)

These define ordering constraints that are *not* visible in static structure.

Examples:

* Initialization must complete before scheduling begins
* Shutdown must drain tasks before resource release
* Clock ticks are strictly monotonic

**Checklist form:**

* [ ] Happens-before relationships
* [ ] Phase boundaries
* [ ] No reentrancy zones

**C++ enforcement tools:**

* Explicit system phases
* Barriers / latches
* Thread annotations (even if informal)
* Compile-time phase tagging types

---

### 6. **Concurrency & Isolation Invariants**

Even if your system is “mostly single-threaded,” you must define these.

Examples:

* No shared mutable state across threads
* All shared state protected by a specific mutex
* Lock acquisition order is total and fixed

**Checklist form:**

* [ ] Thread ownership
* [ ] Synchronization primitives
* [ ] Lock ordering

**C++ enforcement tools:**

* `std::mutex` ownership comments
* Lock hierarchy documentation
* Thread-local storage
* TSAN + assertions

---

## 3. Turning the Checklist into a Living Artifact

Here is the critical step most people miss.

### The Checklist Must Exist in Three Forms Simultaneously

1. **Human-readable table** (for reasoning)
2. **Code-adjacent assertions** (for enforcement)
3. **Promptable artifact** (for AI Assistant)

Example checklist item:

> **Invariant O-3:**
> A `Task` may hold only non-owning references to `Device` objects.
> `Device` lifetime strictly dominates `Task` lifetime.

Then enforce it as:

* Type design (`observer_ptr<Device>`)
* Static analysis
* A comment the AI Assistant cannot ignore

---

## 4. How to Use the Checklist with an AI Assistant (This Matters)

When prompting:

> “You must preserve all invariants listed below.
> For each modification, state which invariants are affected and how they remain satisfied.”

When debugging:

> “Which invariant is violated?
> If none are violated, the bug report is incomplete.”

When an AI Assistant proposes a fix:

> “Map each change to the invariant it enforces.
> If a change cannot be mapped, remove it.”

This forces *semantic grounding*.

---

## 5. A Minimal Example (Condensed)

**Checklist excerpt:**

* S-1: System has exactly one active scheduler
* L-2: Scheduler owns all tasks
* ST-1: Task states are mutually exclusive
* T-3: Tasks cannot transition while system is paused

**Code consequences:**

* Scheduler constructed via factory
* `std::variant<Task::Queued, Task::Running, Task::Completed>`
* All transitions through `Scheduler::transition_task`
* Pause implemented as a guard object

The AI Assistant can now only operate *inside the box*.

---

## 6. Why This Works (And Why Your Experience Is Predictable)

LLMs do not reason globally; they pattern-match locally.

An invariant checklist:

* Shrinks the search space
* Prevents regression
* Converts “understanding” into constraints
* Makes bugs *classifiable*

Once bugs map cleanly to violated invariants, the infinite repair loop disappears.

---

## 7. Next Step (Optional but Powerful)

If you want, the next productive step is:

* Take one problematic module
* Extract **10–15 invariants**
* Encode them as assertions and type constraints
* Use an AI Assistant only after they exist

# Invariant Concept for LPDDR5

LPDDR5 is an excellent first module because the specification is already *constraint-heavy*, which makes it ideal for invariant-driven design.

Below is a **formal invariant checklist framework** specifically adapted to an **LPDDR5 memory + controller** pairing, written so it can be used to both (a) guide C++ design and (b) constrain the AI Assistant.

I will assume:

* You are modeling behaviorally (not cycle-accurate RTL), but
* You care about correctness of command legality, ordering, timing windows, and state.

Where appropriate, I will note how to scale this toward cycle accuracy later.

---

## 1. First Principle: Treat the LPDDR5 Spec as a State Machine, Not a Feature List

The LPDDR5 specification is often read as:

> “Here are commands, timings, and parameters.”

For invariant design, it must be reframed as:

> “Here is a **set of mutually exclusive states**, **legal transitions**, and **forbidden sequences**.”

Your invariant checklist should *mirror the spec’s implicit automaton*.

---

## 2. Top-Level Module Boundary

Before listing invariants, fix the boundary explicitly:

**Module under consideration:**

* LPDDR5 Device Model
* Memory Controller (MC)
* Interface between them (command bus + timing model)

**Explicit exclusions (for now):**

* PHY training details
* Signal integrity
* Bit-level timing

This matters because invariants only apply *within* a declared scope.

---

## 3. LPDDR5 Invariant Classes Applied

### A. Structural Invariants (What Exists)

These prevent illegal topologies.

**S-1: Controller–Device Cardinality**

* Exactly *N* LPDDR5 devices are attached to one controller
* Each device has a unique logical rank/channel identity

**S-2: Command Serialization**

* Each device observes a **single total order** of commands
* No parallel command streams per device

**S-3: Timing Model Ownership**

* Timing state is owned by the controller, not the device
* Device is passive: it validates commands against timing state

**Enforcement (C++):**

* `MemoryController` owns `TimingState`
* `LPDDR5Device` exposes `validate(Command, TimingState)`
* No device-side mutation of timing

---

### B. Ownership & Lifetime Invariants

LPDDR5 bugs often come from “who tracks what”.

**L-1: Controller Dominates Device Lifetime**

* Devices cannot outlive the controller
* No command references a destroyed device

**L-2: Timing State is Single-Owner**

* Timing counters are not shared or aliased
* No external mutation

**L-3: Command Objects Are Ephemeral**

* Commands are value types, not persistent state
* No device stores past command objects

**Enforcement:**

* `unique_ptr<LPDDR5Device>`
* Commands passed by value or `const&`
* No storing of command references

---

### C. State Invariants (Legal Device States)

This is the *core* of LPDDR5 correctness.

Define an explicit device state enum:

Examples (simplified):

* `Reset`
* `Init`
* `Idle`
* `Active`
* `PowerDown`
* `SelfRefresh`

**ST-1: State Exclusivity**

* Device is in exactly one state at all times

**ST-2: Command–State Compatibility**

* Each command is legal only in a defined subset of states

  * e.g. ACTIVATE illegal in `PowerDown`
  * READ/WRITE illegal unless `Active`

**ST-3: Bank-Level vs Device-Level State Separation**

* Bank states are independent
* Device-level state gates bank transitions

**Enforcement:**

* `enum class DeviceState`
* Bank state tracked via `std::array<BankState, N>`
* Central `validate_state(command)` function
* Exhaustive `switch` with `default = delete`

---

### D. Transition Invariants (What May Change, and How)

LPDDR5 has **very strict transition legality**.

**TR-1: Transitions Are Command-Driven**

* Device state changes *only* as a result of valid commands
* No spontaneous transitions

**TR-2: Single Transition per Command**

* One command causes at most one state transition
* No cascading implicit transitions

**TR-3: Forbidden Backward Transitions**

* Certain transitions are irreversible without reset

  * e.g. `Init → Reset` forbidden

**TR-4: Bank and Device Transitions Are Ordered**

* Device-level transitions gate bank-level transitions

**Enforcement:**

* Transition table (data-driven)
* No direct state mutation outside transition handler
* `[[nodiscard]] TransitionResult`

---

### E. Temporal / Timing Invariants (This Is Where Most Bugs Live)

This is where the AI Assistant typically hallucinates.

**T-1: All Timing Constraints Are Enforced Centrally**

* No ad-hoc delays
* No scattered “if enough cycles passed” logic

**T-2: Timing Windows Are Monotonic**

* Counters only increase
* No reset except explicit reset event

**T-3: Command Legality Requires Timing Clearance**

* A command is illegal if *any* required timing window is still open

  * tRCD, tRP, tRAS, tWR, etc.

**T-4: Timing Is Device-Relative**

* All timing measured relative to last relevant command *per device / per bank*

**Enforcement:**

* TimingState contains named counters
* Each command declares required satisfied counters
* `validate_timing(command, timing_state)` is pure

---

### F. Concurrency & Ordering Invariants

Even in a single-threaded model, these must be explicit.

**C-1: Deterministic Evaluation Order**

* Given the same command sequence, behavior is deterministic

**C-2: No Reentrancy During Command Evaluation**

* Device validation cannot issue commands or mutate controller state

**C-3: Controller Is the Sole Authority**

* Device cannot reject a command *after* acceptance

**Enforcement:**

* Command evaluation is a two-phase process:

  1. Validate
  2. Apply
* No callbacks into controller during validation

---

## 4. A Concrete Invariant Checklist (Condensed Example)

This is what you actually hand to the AI Assistant.

```
LPDDR5 Controller Invariants (Excerpt)

S-1: Each LPDDR5 device observes a single total order of commands.
L-2: Timing state is exclusively owned by the controller.
ST-1: Device is always in exactly one DeviceState.
ST-2: Commands are only legal in explicitly allowed DeviceStates.
TR-1: All state transitions are command-driven.
T-1: All timing constraints are enforced centrally.
T-3: A command is illegal if any required timing window is open.
C-2: Command validation is non-reentrant and side-effect free.
```

Then you say:

> “You must preserve all invariants above.
> For each change, explicitly state which invariant it relies on.”

---

## 5. How This Prevents the AI Assistant Failure Mode You Observed

Without invariants:

* the AI Assistant “fixes” a timing bug by skipping a check
* Then reintroduces a forbidden transition
* Then breaks bank/device separation

With invariants:

* Every fix must map to **one violated invariant**
* New bugs show up immediately as invariant violations
* the AI Assistant cannot legally invent shortcuts

The infinite loop collapses into a bounded correction process.

---

## 6. Recommended Next Step

Before writing more code:

1. Write a **complete DeviceState + BankState table**
2. Write a **command × state legality matrix**
3. Write a **timing constraint table** (names only, no numbers yet)

Only after those exist should the AI Assistant be allowed to generate implementation code.

If you want, next we can:

* Design the **command legality matrix**
* Design the **timing constraint abstraction**
* Show how to encode these invariants into modern C++ types so they are compiler-enforced


