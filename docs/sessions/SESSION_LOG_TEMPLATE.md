# Session Log Template

Use this template for every development session to maintain governance and accountability.

## Required Sections

### Header
```markdown
# [Version] [Feature Name] Decision Log

**Date:** YYYY-MM-DD
**Version:** vX.Y
**Status:** In Progress | Complete | Blocked
**Tests:** X/Y passing
```

### 1. Summary
One paragraph describing what was accomplished or attempted.

### 2. Scope
Table of features/operations with implementation status and test counts.

### 3. Technical Decisions
For each significant decision:
- **Decision N: [Title]**
- **Choice:** What was decided
- **Alternatives Considered:** What else was evaluated
- **Rationale:** Why this choice was made
- **Files Modified:** List of affected files

### 4. Issues Encountered
Document all bugs, crashes, and unexpected behavior:
- Symptom
- Root cause
- Fix applied

### 5. Wrong Decisions (CRITICAL)
**This section is mandatory.** Document any incorrect decisions made during the session:
- What was the wrong decision?
- Why was it wrong?
- What was the correction?
- What is the lesson learned?

If no wrong decisions were made, explicitly state: "No wrong decisions identified this session."

### 6. Verification
```bash
# Commands to verify the work
```

### 7. Next Steps
What remains to be done.

---

## Governance Rules

1. **Every session must produce a decision log** - No exceptions
2. **Wrong decisions must be documented** - This is not optional
3. **Crashes and test failures must be debugged** - Never skip failing tests
4. **Commit messages reference the decision log** - Traceability
5. **Decision logs are version controlled** - Part of the codebase

## File Naming Convention

```
docs/sessions/YYYY-MM-DD_vX.Y_feature_name.md
```

Examples:
- `2026-01-23_v0.6_transformer_ops.md`
- `2026-01-24_v0.7_quantization.md`
- `2026-01-25_v0.8_model_execution.md`
