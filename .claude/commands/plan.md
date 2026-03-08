Create a design plan for: $ARGUMENTS

1. Create a new file in docs/plans/ with a descriptive kebab-case filename.

2. Follow this structure:

   # Title
   **Date:** YYYY-MM-DD
   **Status:** Design | In Progress | Implemented

   ## 1. Problem Statement
   What problem does this solve? What's wrong with the current approach?

   ## 2. Architecture
   ASCII diagram showing component relationships.
   Use the established style from docs/plans/dma_csp.md.

   ## 3. Design
   Key data structures and APIs with C++ code snippets.
   Show both WRONG (what to avoid) and CORRECT (what to do) patterns.

   ## 4. Implementation Steps
   Numbered steps with specific files to modify.

   ## 5. Verification
   Specific test commands and expected outcomes.

   ## 6. Key Invariants
   What must always be true after this change?

3. Do NOT implement the plan. Just create the document for review.
   The user will review and then ask for implementation.
