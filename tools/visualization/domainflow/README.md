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
  unbounded at L-B, since a level-1 program declares only L3.
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

**Navigation:** step, back, play (Space, the arrow keys, Home, End), the slider, the next
iteration at any depth, the next action of a kind or on the same tile, the release of the current
residency, the next slot-bound Load (with an overlay), and jump to a step.

**Honesty rules (plan §3.7):** the axis is the step, not time, because L-B has no time; L-T1
timing is a later overlay (step 5). Every residency, address, occupancy and loop index is read from
the record; the page computes none of them. A cut bundle (`kpu-run --dflow-limit`) shows its steps
up to the cut, while the ribbon and the slot histories still cover the whole run. A run L-B
refused shows the refusal. The page does not check invariants: that is `dflow_check.py`'s job.

`test_viewer.py` packs bundles, and runs the page's data layer (`<script id="dflow-core">`, which
has no DOM) in Node, comparing what it reads at sample steps (step fields, position, slots, and the
tile values the inspector shows, and the overlay's legs, proximity, slot waits and time-to-action
lookup) with the bundle read in Python. It reads versions 1 to 3.
