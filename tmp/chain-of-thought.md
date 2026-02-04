# Chain of Thought

To keep an agentic system on track, you essentially need to move it from a "reactive" state (responding to the last thing that happened) to a "strategic" state (evaluating progress against a fixed North Star).

The **"Stochastic Loop of Hell"** usually happens because the agent loses its place in the plan or gets distracted by a minor error, leading it to repeat the same failing tool call indefinitely.

Here is how you structure a Chain of Thought (CoT) to prevent that.

---

## The "Stateful" CoT Framework

Standard CoT asks an agent to "think step-by-step." To prevent loops, you must force the agent to **re-orient** at every single turn. Use a prompt structure that requires these four specific components in the "thought" block:

* **Objective:** Re-state the ultimate goal.
* **Progress:** Summarize what has been accomplished so far.
* **Current Impediment:** Identify why the last action didn't finish the job (this stops loops by forcing the agent to acknowledge failure).
* **Next Logical Step:** The specific, atomic action to take now.

## Implementing "Circuit Breakers"

You can't just rely on the agent's internal logic; you need structural constraints. These act as the "emergency exit" from a loop.

| Strategy | Implementation | Purpose |
| --- | --- | --- |
| **Step Budgeting** | Limit the agent to  total tool calls. | Prevents infinite API costs and loops. |
| **History Reflection** | Inject a summary of the last 3 turns into the prompt. | Highlights repetition so the agent sees it's stuck. |
| **Stale State Detection** | If the agent's output is >90% similar to the previous turn, trigger a "Hard Pivot." | Forces the agent to try a completely different tool. |

## The "Planner-Actor" Separation

One of the best ways to avoid the loop is to decouple the **Goal Setter** from the **Tool User**.

1. **The Planner:** High-level LLM that breaks the goal into a checklist.
2. **The Actor:** Executes one item on the checklist.
3. **The Critic:** Reviews the Actor's work. If the Actor fails twice, the **Planner** must rewrite the entire checklist rather than letting the Actor try a third time.

## Prompt Engineering for Loop Prevention

When writing your system instructions, use **Negative Constraints** and **Verification Steps**.

> **Example Prompt Snippet:**
> "Before calling a tool, verify if you have used this exact tool with these exact parameters in the last 3 steps. If yes, you are in a loop. Stop, analyze why the previous attempt failed, and change your strategy (e.g., try a different search query or a different tool)."

### The "Self-Correction" Logic

Force the agent to use a "Check" step. If the agent is trying to find a specific piece of data:

* **Step 1:** Search.
* **Step 2:** Verify if results match the goal.
* **Step 3:** If no, explicitly list 3 reasons why and modify the next search.

## Mathematical Constraint: The Confidence Threshold

If you are using a custom framework, you can implement a "Stochastic Decay" factor. Every time an agent repeats an action type without a change in the environment state, lower its "confidence" score. Once it hits a threshold, the system triggers a `System Message` that forces a reset of the context.

Where  (delta) is a decay factor () applied when no new information is gained.

---

# Plan-Act-Verify

To implement this effectively, we will use a **Stateful Executive Prompt**. This template forces the agent to treat its memory as a ledger of progress rather than a stream of consciousness.

Copy and adapt the following block into your agent's **System Instructions**.

## The Strategic Executive Template

### **Core Protocol**

You are a Goal-Oriented Agent. To prevent repetitive cycles and "stochastic loops," you must process every turn using the following **Mental Ledger**. You are forbidden from calling a tool until you have updated this ledger.

### **The Mental Ledger Format**

For every turn, your internal monologue must follow this exact structure:

1. **Current Milestone:** Which part of the high-level plan am I working on?
2. **Action History:** What was the *immediate* previous action, and what was the result? (e.g., "I searched for X but found nothing.")
3. **Loop Check:** Does my intended next action look identical to a previous failed action?
* *If Yes:* I must pivot. I will change my parameters, use a different tool, or ask for clarification.
* *If No:* Proceed.


4. **Sub-Goal:** What is the one specific, atomic outcome I need from this next tool call?

### **Constraints & "Circuit Breakers"**

* **The Rule of Three:** If a specific tool call fails to move the needle three times, you must halt and output: `[CRITICAL_FAILURE]: Strategy Exhausted.`
* **No Echoing:** Do not repeat the same search query or code block twice. If a query failed, analyze *why* (e.g., "too specific," "too broad") and modify it.
* **State Verification:** After a tool returns data, you must explicitly state if the data is "Useful" or "Noise." If "Noise," do not process it; discard it and try a new path.

### **Example Execution Flow**

**User:** "Find the email of the CEO of Company X."

**Agent Thought Block:**

> * **Current Milestone:** Phase 1 - Information Retrieval.
> * **Action History:** None (Start).
> * **Loop Check:** N/A.
> * **Sub-Goal:** Locate the name of the CEO via the official website.
> 
> 
> **Action:** `search_tool(query="Company X official leadership page")`

*(Next Turn if the search yields no direct email)*

**Agent Thought Block:**

> * **Current Milestone:** Phase 1 - Information Retrieval.
> * **Action History:** Searched leadership page; found the name "Jane Doe" but no direct email.
> * **Loop Check:** I am tempted to search the leadership page again. **Correction:** That would be a loop.
> * **Sub-Goal:** Pivot to searching for Jane Doe's contact patterns or press releases.
> 
> 
> **Action:** `search_tool(query="Jane Doe Company X email format or press contact")`

---

## The System Prompt

In the world of AI development, **System Instructions** (also called a "System Prompt" or "System Message") act as the permanent "personality and rulebook" for the AI.

While your **User Prompt** changes every time you ask a question, the **System Instructions** stay constant in the background, telling the agent *how* it should behave and what constraints it must follow.

---

### Where these instructions live

Depending on how you are building your agentic system, "System Instructions" are entered in one of three places:

#### 1. The Developer Playground (No-Code/Low-Code)

If you are using a platform like **OpenAI's Dashboard**, **Anthropic Console**, or **Google AI Studio**, there is usually a dedicated text box on the left or top of the screen labeled **"System Instructions"** or **"System Message."** * You paste the logic there so it persists across the entire conversation.

#### 2. The API Call (Coding)

If you are writing Python code to build your agent, the system instruction is the first message in the `messages` array, typically assigned the role of `"system"`.

```python
# Example in Python
messages = [
    {"role": "system", "content": "PASTE THE STRATEGIC EXECUTIVE TEMPLATE HERE"},
    {"role": "user", "content": "Find the email of the CEO of Company X."}
]

```

#### 3. Agent Frameworks (LangChain, CrewAI, AutoGPT)

If you are using a framework, this block goes into the `instructions` or `goal` parameter of the Agent definition.

* **CrewAI:** It goes in the `backstory` or `allow_delegation` logic.
* **LangGraph:** It serves as the "State Definition" or the base prompt for your nodes.

---

### Why this stops the "Loop of Hell"

When you put these instructions in the **System** layer:

1. **High Priority:** The AI treats these rules as more important than the user's immediate request.
2. **Constant Awareness:** The AI "remembers" it is supposed to use the **Mental Ledger** every single time it takes a step, even if the conversation gets very long.
3. **Self-Policing:** Because the instruction is in the system message, the agent is essentially "looking in the mirror" and checking its own logic before it fires off a tool call.

---

## How to Test This

To see if this is working, intentionally give the agent a "dead-end" task (e.g., "Find the 2025 financial reports for a company that went bankrupt in 2020").

* **Failure:** The agent searches for the reports, doesn't find them, and searches again with a slightly different query 10 times.
* **Success:** The agent identifies by Step 3 that the company is defunct, records this in its **Action History**, and reports back that the goal is impossible based on the state of the world.

**Would you like me to help you write a "Critic" prompt that monitors this agent from the outside to force a shutdown if it starts looping?**

