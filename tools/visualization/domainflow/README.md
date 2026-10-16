# Domain-flow viewer

Step 3 of the domain-flow debugger (#363; `docs/plans/domain-flow-debugger.md`). It shows a CSP
program's L-B run as domains moving through the program's channels, one CSP action per step.

```bash
# a program, its L-B run recorded as a .dflow bundle
kpu-csp-gen --algo matmul --size 512 --tile 64 --l3 17 -o mm512.csp
kpu-run --program mm512.csp --deploy tests/program/deploy/kpu_s1.json \
        --level block-sequential --dflow mm512.dflow --dflow-values "C[2,3],A[2,0],B[0,3]"

# check it, then look at it
python3 tools/trace/dflow_check.py mm512.dflow
python3 tools/visualization/domainflow/pack.py mm512.dflow -o mm512.html
```

Open `mm512.html` in a browser. It needs no server and no other file. Alternatively, open
`index.html` and choose the bundle folder with the file picker.

**What it shows:**
- **Machine view:** DRAM, L3, L2 and the fabric, with the process on each edge. The edge the
  current action moves a tile along is lit. L3 is drawn as its addressed slots, grouped by bank,
  each showing the tile it holds and the residency. The slot the action touches is outlined; a
  slot this action released is dashed. L2 and the fabric show their tile counts and are labelled
  "no bound at L-B", since a level-1 program declares only L3.
- **Program view:** the source, with the current line highlighted and every enclosing loop's
  index. Click a line to run to its next action.
- **Ribbon:** every loop iteration at every depth across the whole run, and L3 occupancy over the
  recorded steps. Click to jump.
- **Inspector:** the current action's fields; select a slot to list every tile it held.
- **Tile inspection (step 4):** the current step's tile, or the tile in a selected slot, as a
  heatmap (diverging about zero; a value that is not finite is marked), with its min, max, mean
  and a readout of any element in tile and operand coordinates. "At this step" shows the value the
  latest recorded action left it with, and in which channel; "final" shows its value when the run
  ended. Every operand's final values are always recorded; a tile's value at each action only when
  `--dflow-values` selected it (`C[2,3],A[2,0]`, an operand name, or `all`), under
  `--dflow-values-cap` (64 MiB). The elements are float32, the format the operands carry; the page
  refuses a bundle naming another format rather than misread it.

- **The L-T1 overlay (step 5):** when L-T1 ran in the same `kpu-run` (`--level block-sequential`
  or `all`), the record carries its run joined on the action index. The inspector lists the
  current action's legs (process, lane, start and finish in L-T1 cycles), says when a Load's L3
  slot decided its start and what the wait cost, and shows the tile's proximity to L-B under L-T1
  (max |a - o|, max relative error, within bound or not). The axis can switch to L-T1 cycles: the
  ribbon then draws every leg on its process's row, marks slot waits, and a click lands on the
  action running at that cycle (on an idle cycle, it stays where it is). Everything in the overlay is labelled L-T1 and uncalibrated.

- **L2 and the compute fabric (L2 reuse / fabric plan, step 1):** L2 and the fabric list their
  tiles after each action. An L2 tile moved from L3 more than once so far is marked `×n`: at L-B
  every use of a tile is a new Move, because the language does not yet stage tiles in L2, so the
  marks show the re-Moves that staging will remove. Under the flow, the **L1 + compute fabric**
  panel shows the operands fed for the next call, the call (its result, operands, and its place in
  its chain, k of K, with a bar), and the results in the array (open accumulators). A chain is the
  calls on one result while it stays in the fabric, up to its Drain. The panel is structural: L-B
  has no time, so it shows no fill, steady state or drain. Those are timing and arrive with the
  L-T1 pipeline overlay (step 2). The record counts L2's and the fabric's tiles but does not name
  them; the page derives which tiles they are from the recorded actions, exactly as L-B moves them,
  and checks the derivation against L-B's counts at every step. If they ever disagree, the page
  says so and shows no contents rather than wrong ones.

**Navigation:** step, back, play (Space, the arrow keys, Home, End), the slider, the next
iteration at any depth, the next action of a kind or on the same tile, the release of the current
residency, the next slot-bound Load (with an overlay), the next chain start or end, and jump to a
step. A link to `page.html#step=123` opens at that step.

- **The L-T1 pipeline (step 2, `.dflow` version 4):** when L-T1 ran on a deployment that declares
  its PE array (`fabric`), the fabric panel adds L-T1's pipeline at the current action's start
  cycle, per compute tile:
  - the operands streaming into the array, and how far, at PE(0,0);
  - every chain on the array and its phase: fill, steady, tail, or fill+tail for a chain shorter
    than the array's skew;
  - the results draining north, and how far.

  The bars are linear within each recorded interval (the model's dense stream). The cycle-axis
  ribbon adds "cf in" (each chain's operand stream, darker while every PE is on it) and "north"
  (its results leaving). "Next three-tile moment" jumps to an action that starts while a compute
  tile streams one chain in and drains the one before. The timing note names a stream call's
  chain and its bounds. On a deployment without a fabric, the panel says every call was one leg.
  A program that stores each C right after its chain shows no overlap: its stores serialize the
  chains (the plan's step-2 finding).

**A steady layout:** every area that changes from step to step has a fixed size, so nothing
jumps while the run plays. A value too wide for its box is cut with an ellipsis and shown in full
in its tooltip; a list that does not fit scrolls inside its box; an inspector row that does not
apply to a step shows a dash instead of disappearing; and stepping scrolls only the program
listing, never the window. `test_viewer.py` checks this in a headless Chrome when one is
available (`$CHROME`, one on `PATH`, or Playwright's): it steps every bundle at three widths and
fails if any panel moves or changes size.

**Honesty rules (plan §3.7):** the axis is the step, not time, because L-B has no time; L-T1
timing is a later overlay (step 5). Every residency, address, occupancy and loop index is read from
the record; the page computes none of them. A cut bundle (`kpu-run --dflow-limit`) shows its steps
up to the cut, while the ribbon and the slot histories still cover the whole run. A run L-B
refused shows the refusal. The page does not check invariants: that is `dflow_check.py`'s job.

`test_viewer.py` packs bundles, and runs the page's data layer (`<script id="dflow-core">`, which
has no DOM) in Node, comparing what it reads at sample steps (step fields, position, slots, and the
tile values the inspector shows, the overlay's legs, proximity, slot waits and time-to-action
lookup, and L2's and the fabric's contents and calls against an independent replay) with the
bundle read in Python. Every recorded L2 or fabric snapshot must find its tile where the page puts
it. A strict scan requires every interpolation that writes HTML to be escaped or reviewed. It reads versions 1 to 4.
