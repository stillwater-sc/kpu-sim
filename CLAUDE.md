# Claude Code Integration Guidelines

This document provides guidance for Claude Code when working on the KPU-SIM project.

## Repository Purpose: Multi-Fidelity Simulation

**READ THIS FIRST** - The KPU simulator is a **multi-fidelity simulation environment**
that supports three tiers of modeling abstraction:

### Simulation Fidelity Tiers

| Tier | Purpose | Speed | Computes Values? |
|------|---------|-------|------------------|
| **BEHAVIORAL** | Functional correctness, software bring-up | ~100-1000x | **YES** |
| **TRANSACTIONAL** | Architecture exploration, bottleneck ID | ~10-100x | Statistical |
| **CYCLE_ACCURATE** | Performance analysis, timing validation | 1x (baseline) | Via integration |

### The Multi-Fidelity Philosophy

The progression works as follows:

1. **Cycle-Accurate First**: Model subsystems from first principles to capture emergent
   behavior with high confidence (e.g., DRAM timing, bank conflicts, page dynamics)

2. **Characterize Statistics**: Extract latency, concurrency, and resource occupation
   statistics from cycle-accurate simulation

3. **Build Transactional Models**: Use collected statistics to create faster models
   with queue-based contention and aggregate latencies

4. **Create Behavioral Models**: Highest abstraction with functional correctness,
   enabling software development and validation

### Key Use Cases by Fidelity

**BEHAVIORAL (Functional Simulation)**:
- Execute high-level operators (e.g., eigenvalue solver, MLP, convolution)
- Validate application software correctness
- Software/firmware bring-up before hardware
- CI/CD pipeline testing
- **Components compute actual values and propagate results**

**TRANSACTIONAL (Performance Estimation)**:
- Early architecture design space exploration
- Workload characterization
- Power/performance estimation
- Identify bottlenecks without full timing detail

**CYCLE_ACCURATE (Timing Validation)**:
- Precise performance analysis
- Protocol compliance verification
- Trace generation for visualization
- Hardware/software co-design
- Invariant-based validation

### Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/SIMULATION_FIDELITY_FRAMEWORK.md` | Full multi-fidelity design (READ THIS) |
| `include/sw/kpu/fidelity/simulation_fidelity.hpp` | Fidelity enums and types |
| `include/sw/kpu/fidelity/component_config.hpp` | Per-component configuration |

### Component Fidelity Support

Each component can operate at its own fidelity level:

| Component | BEHAVIORAL | TRANSACTIONAL | CYCLE_ACCURATE |
|-----------|------------|---------------|----------------|
| Memory Controller | Fixed latency | Queue model | Full DRAM FSM |
| DMA Engine | Instant transfer | Bandwidth model | Channel arbitration |
| Compute Fabric | **Actual compute** | Throughput model | Pipeline stages |
| NoC | Zero latency | Hop count model | Wormhole routing |
| L3/L2/L1 Memory | Direct access | Bank contention | Port arbitration |

### Configuring Fidelity

```cpp
#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/fidelity/component_config.hpp>

// Configure behavioral simulation for functional testing
ComponentConfig config;
config.memory_fidelity = SimulationFidelity::BEHAVIORAL;
config.compute_fidelity = SimulationFidelity::BEHAVIORAL;  // Computes actual values!

// Configure cycle-accurate for performance analysis
config.memory_fidelity = SimulationFidelity::CYCLE_ACCURATE;
config.verification_level = VerificationLevel::INVARIANTS;
```

---

## Validation Requirements

This section covers validation for **cycle-accurate** simulation, particularly
memory controller traces and timing invariants.

### Core Principle: Validate Before Declaring Complete

**Never declare code generation complete without validation.**

Claude Code must follow this workflow for any code that produces artifacts:

```
1. Generate code
2. Run validation tools
3. If validation fails:
   a. Parse error output
   b. Identify root cause
   c. Fix the issue
   d. Return to step 2
4. Only declare complete when validation passes
```

## Project Structure

```
kpu-sim/
├── CLAUDE.md                              # This file - read first!
│
├── docs/
│   ├── SIMULATION_FIDELITY_FRAMEWORK.md   # Multi-fidelity design (MUST READ)
│   ├── KPU_API_GAPS_AND_ROADMAP.md        # API gaps and DNN roadmap
│   └── sessions/                          # Session logs and changelogs
│
├── include/sw/kpu/
│   ├── fidelity/
│   │   ├── simulation_fidelity.hpp        # Fidelity enums (BEHAVIORAL, etc.)
│   │   └── component_config.hpp           # Per-component configuration
│   ├── components/
│   │   ├── compute_fabric.hpp             # Compute with actual matmul
│   │   ├── behavioral_compute_fabric.hpp  # Behavioral compute model
│   │   └── memory/
│   │       ├── memory_controller_interface.hpp  # MC interface
│   │       └── behavioral_memory_controller.hpp # Behavioral MC
│   └── kpu_simulator.hpp                  # Main simulator class
│
├── patterns/
│   └── memory/
│       └── lpddr5/
│           ├── INVARIANTS.md              # Trace invariants (cycle-accurate)
│           └── common/
│               ├── trace_validator.py     # Standalone trace validator
│               └── lpddr5_harness.hpp     # C++ test harness
│
├── traces/
│   └── memory/
│       └── lpddr5/                        # Generated trace files
│
└── examples/
    └── mlp/                               # MLP examples (XOR, MNIST, etc.)
```

## Validation Tools

### 1. Trace Validator (Python)

**Location:** `patterns/memory/lpddr5/common/trace_validator.py`

**Usage:**
```bash
python3 patterns/memory/lpddr5/common/trace_validator.py <trace_file.json>
```

**Exit Codes:**
- `0` - All invariants pass
- `1` - One or more invariants violated
- `2` - Error reading/parsing trace file

**When to Run:**
- After generating any trace file
- After modifying trace generation code
- After modifying memory controller behavior

**Output Parsing:**
The validator produces structured output that Claude Code should parse:

```json
{
  "status": "FAILED",
  "violations": [
    {
      "invariant": "INV-001",
      "message": "txn_id=0 has no data operation",
      "fix_hint": "PRECHARGE should have same txn_id as READ/WRITE"
    }
  ]
}
```

Use `--json` flag for machine-readable output:
```bash
python3 trace_validator.py trace.json --json
```

### 2. C++ Test Harness

**Location:** `patterns/memory/lpddr5/common/lpddr5_harness.hpp`

The harness provides:
- `verify_no_violations()` - Check for invariant violations during simulation
- `verify_stats()` - Verify expected statistics
- Runtime invariant checking in the memory controller

### 3. Build Verification

Always run builds and tests after code changes:
```bash
cmake --preset release && cmake --build --preset release
ctest --preset release
```

## Invariant Documentation

**Primary Location:** `patterns/memory/lpddr5/INVARIANTS.md`

### Key Invariants to Enforce

| ID | Description | Severity |
|----|-------------|----------|
| INV-001 | Every txn_id must have exactly ONE data operation | ERROR |
| INV-002 | ACTIVATE/PRECHARGE must belong to valid transactions | ERROR |
| INV-003 | Commands must be temporally ordered correctly | ERROR |
| INV-100 | tRCD constraint (ACT to READ/WRITE) | WARNING |
| INV-101 | tRP constraint (PRE to ACT) | ERROR |

### Adding New Invariants

When adding new invariants:

1. Document in `INVARIANTS.md` with:
   - Unique ID (INV-XXX)
   - Description
   - Rationale
   - Validation logic
   - Failure example
   - Fix hint

2. Implement in `trace_validator.py`:
   ```python
   def _check_inv_xxx_name(self):
       """INV-XXX: Description."""
       # Validation logic
       if violation_detected:
           self.violations.append(Violation(
               invariant='INV-XXX',
               severity=Severity.ERROR,
               message="...",
               fix_hint="..."
           ))
   ```

3. Add to validator's `validate()` method

## Development Workflow

### For Trace Generation Code

```
1. Modify C++ trace generation code
2. Rebuild: cmake --build --preset release
3. Run pattern to generate trace:
   ./build/patterns/memory/lpddr5/single-bank/page-conflicts
4. Validate trace:
   python3 patterns/memory/lpddr5/common/trace_validator.py \
     traces/memory/lpddr5/single-bank/page_conflicts_trace.json
5. If failed:
   - Read violation messages
   - Trace to C++ code causing issue
   - Fix and repeat from step 2
6. When passed: commit changes
```

### For Visualization Code

```
1. Modify HTML visualization code
2. Run validator on all traces:
   for f in traces/memory/lpddr5/single-bank/*.json; do
     python3 patterns/memory/lpddr5/common/trace_validator.py "$f"
   done
3. Test visualization in browser
4. When passed: commit changes
```

## Common Bugs and Fixes

### Bug: PRECHARGE has wrong txn_id

**Symptom:** Validator reports INV-001/INV-002 violation for txn_id=0

**Root Cause:** PRECHARGE assigned sentinel txn_id instead of original request's txn_id

**Fix:** Track which request opened the page, use that txn_id for PRECHARGE

### Bug: Request type defaults to WRITE

**Symptom:** Requests labeled as WRITE when they should be READ

**Root Cause:** Type detection logic: `type = isRead ? 'READ' : 'WRITE'`

**Fix:** Explicit check:
```javascript
if (!hasRead && !hasWrite) continue; // Skip - not a request
type = hasRead ? 'READ' : 'WRITE';
```

### Bug: Timing constraint violation

**Symptom:** tRCD, tRP, tRRD violations

**Root Cause:** Commands issued without respecting timing parameters

**Fix:** Add timing checks before issuing commands, respect parameters in TIMING

## Session Logging

After significant work, create a session log:

**Location:** `docs/sessions/YYYY-MM-DD_description.md`

Include:
- What was done
- What bugs were found
- What invariants were added/modified
- Validation results

## Questions to Ask

Before generating code, ask:

1. "What fidelity level is this code targeting?"
   - BEHAVIORAL: Must compute actual values
   - TRANSACTIONAL: Statistical timing models
   - CYCLE_ACCURATE: Full protocol state machines

2. "What invariants apply to this code?"

3. "What validation tools should I run?"

4. "What are the expected outputs?"

After generating code, ask:

1. "Did all validations pass?"

2. "Are there any warnings I should address?"

3. "Is the code covered by existing invariants?"

4. "Does the behavioral tier still compute correct functional results?"

## Remember

**This is a multi-fidelity simulator, not just a timing model.**

Key principles:
- BEHAVIORAL tier must compute actual values for software validation
- Read `docs/SIMULATION_FIDELITY_FRAMEWORK.md` before making architectural changes
- Read INVARIANTS.md before generating trace-related code
- Run validators after every change
- Parse and act on validation failures
- Add new invariants when bugs are discovered
- Document changes in session logs

**Correct code is more valuable than fast code.**
