#!/usr/bin/env python3
"""
Derived spatial binding for a tile-flow record (#286; docs/plans/tile-flow-debugger.md §3.6).

At L-T1 the executor models L3 as one pooled capacity and binds moves to LANES of a mover pool,
not to places: it does not say which L3 tile a tile lives in, which DMA engine served a
transfer, or which NoC hubs a burst crossed. The viewer's per-resource swimlanes need all
three, so this tool DERIVES them from the record and the deployment's floorplan, by fixed,
stated policies. It changes no timing: every interval keeps the cycle the executor gave it.
What it adds is a place.

    python3 bind.py <bundle.tflow> --floorplan <fp.json>     # writes <bundle>/binding.json

THE POLICIES (also written into binding.json, so a viewer can show them):

  dram    Operands laid out in program order from address 0, each aligned to 4 KiB, each
          tile-major (tile (ti, tj) at base + (ti * tile_cols + tj) * tile_bytes). The top of
          memory is the next power of two above the last byte: the deployment declares no
          DRAM size.
  dma     A transfer goes to the memory controller that owns its DRAM address -- addresses
          interleaved across the controllers one tile at a time, as a multi-channel DRAM is --
          and to the first engine of that controller free when it starts. A controller asked
          for more transfers at once than it has engines is counted (`dma.overloaded`), never
          hidden: it is the schedule asking for more DRAM parallelism than that controller has.
          The executor itself models one pooled lane set and picks any free lane.
  l3      A tile is homed, when its residency begins, in the L3 tile that abuts the compute
          tile that next computes with it and holds the fewest tiles; per-tile capacity is
          the L3 capacity divided evenly, and a full neighbourhood falls back to the nearest
          L3 tile with room. Over-capacity homes are counted, never hidden.
  bm      A BlockMover move runs on the home L3 tile's mover facing the destination compute
          tile; if the home does not abut it, the tile first crosses the NoC to the abutting
          L3 tile nearest the home (a derived relay) and moves from there.
  noc     A DMA burst enters at the port its engine attaches to (the floorplan's first-pass
          attachment) and follows a shortest path over the torus's ring links to the home hub,
          deterministic on ties. Every hub and port on the path is busy for the transfer's
          whole interval: the executor models no per-hop NoC timing.
  ports   A port is two buses, injection (a DMA engine pushes a block in) and ejection (a block
          is pushed out to a DMA engine's buffer), each carrying ONE block at a time
          (docs/plans/noc-port-arbitration.md). L-T1 does not model ports: its DMA lanes run
          concurrently, so a bus that carries more than one block at once is the derived path
          over-subscribing it. That is reported per bus (peak, and transfers that started while
          the bus was held), never normalized away.

When the executor binds L3 slots itself (plan step 6), the l3 and bm policies are replaced by
the modelled binding and the viewer's rows stop saying "derived".

Exit codes: 0 written; 2 the bundle or floorplan could not be read, or do not match.

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trace"))
import tflow_check  # noqa: E402  (the one bundle reader)

# enum Hop, version 3: every hop is a push. Writeback is an ejection (a BlockMover pushes the
# block from L3 into a DMA engine buffer, across the NoC to the port) and then a DMA write of
# that buffer to DRAM, which does not touch the NoC.
HOP_DMA_IN, HOP_BM_DOWN, HOP_BM_UP, HOP_EJECT, HOP_DMA_WRITE = 0, 1, 4, 5, 6
EDGE_N, EDGE_E, EDGE_S, EDGE_W = 0, 1, 2, 3
ALIGN = 4096


class BindError(Exception):
    pass


def index_of(name, word):
    m = re.search(word + r"\[(\d+)\]", name)
    return int(m.group(1)) if m else None


def flat(blocks):
    out = []
    for b in blocks:
        out.append(b)
        out.extend(flat(b.get("children", [])))
    return out


def center(r):
    return (r[0] + r[2] / 2, r[1] + r[3] / 2)


def bind(rec, fp):
    m = rec["manifest"]
    dev = m["device"]
    if fp.get("device") != dev:
        raise BindError(f"the floorplan lays out device {fp.get('device')!r}, the record ran on {dev!r}")
    blocks = flat(fp["blocks"])
    l3 = sorted((index_of(b["name"], "l3"), b) for b in blocks if b["kind"] == "l3_tile")
    cf = sorted((index_of(b["name"], "cf"), b) for b in blocks if b["kind"] == "compute_tile")
    l3_blk = {i: b for i, b in l3}
    cf_blk = {i: b for i, b in cf}
    if not l3:
        raise BindError("the floorplan places no L3 tiles")

    # ---- abutment: which L3 tile touches which compute tile, and on which edge ----------
    pitch = l3[0][1]["rect"][2]
    abut = {}                                   # cf -> [(l3, edge)]
    for t, lb in l3:
        lx, ly = center(lb["rect"])
        for c, cb in cf:
            cx, cy = center(cb["rect"])
            dx, dy = cx - lx, cy - ly
            if abs(abs(dx) - pitch) < pitch * 0.3 and abs(dy) < pitch * 0.3:
                abut.setdefault(c, []).append((t, EDGE_E if dx > 0 else EDGE_W))
            elif abs(abs(dy) - pitch) < pitch * 0.3 and abs(dx) < pitch * 0.3:
                abut.setdefault(c, []).append((t, EDGE_S if dy > 0 else EDGE_N))
    names = {b["name"] for b in blocks}

    # ---- the NoC graph: hubs by ring links, ports to their fold hubs, engines to ports ----
    adj = {}
    attach = {}
    for link in fp.get("noc", []):
        a, b = link["a"], link["b"]
        if link["kind"] in ("ring", "port"):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        elif link["kind"] == "attach":
            attach[a] = b
    hub = lambda t: f"{dev}/l3[{t}]/noc"

    def path(src, dst):
        """Shortest path src -> dst over the NoC graph; ports are entered, not crossed."""
        if src == dst:
            return [src]
        prev = {src: None}
        q = deque([src])
        while q:
            n = q.popleft()
            for nb in sorted(adj.get(n, ())):
                if nb in prev or (nb != dst and "/port[" in nb):
                    continue
                prev[nb] = n
                if nb == dst:
                    out = [nb]
                    while prev[out[-1]] is not None:
                        out.append(prev[out[-1]])
                    return out[::-1]
                q.append(nb)
        return None

    # ---- DRAM layout ---------------------------------------------------------------------
    eb = m.get("element_bytes") or 4
    tensors, addr = [], 0
    for o in m.get("operands", []):
        ntr = -(-o["rows"] // o["tile_rows"]) if o["tile_rows"] else 0
        ntc = -(-o["cols"] // o["tile_cols"]) if o["tile_cols"] else 0
        tile_bytes = o["tile_rows"] * o["tile_cols"] * eb
        addr = -(-addr // ALIGN) * ALIGN
        tensors.append({"name": o["name"], "base": addr, "bytes": ntr * ntc * tile_bytes,
                        "tile_bytes": tile_bytes, "tile_rows": ntr, "tile_cols": ntc,
                        "shape": [o["rows"], o["cols"]], "tile_shape": [o["tile_rows"], o["tile_cols"]]})
        addr += ntr * ntc * tile_bytes
    top = 1
    while top < max(addr, 1):
        top *= 2

    # ---- DMA engines ---------------------------------------------------------------------
    engines = sorted((b["name"] for b in blocks if b["kind"] == "dma_engine"),
                     key=lambda n: (index_of(n, "mc"), index_of(n, "dma")))
    lanes = next((p["lanes"] for p in m.get("movers", []) if p["name"] == "dma"), 0)
    notes = []
    if engines and lanes != len(engines):
        notes.append(f"the run had {lanes} DMA lane(s) and the floorplan {len(engines)} DMA engine(s)")
    mc_engines = {}
    for k, n in enumerate(engines):
        mc_engines.setdefault(index_of(n, "mc"), []).append(k)
    mc_list = sorted(mc_engines)

    # ---- consumers: which compute tile computes with a tile, and when ----------------------
    res, tr, co, ot = rec["residency"], rec["transit"], rec["compute"], rec["op_tiles"]
    st = m["stations"]
    cf_of_station = {i: index_of(s["name"], "cf") for i, s in enumerate(st) if s["kind"] == "cf"}
    uses = {}                                    # tile -> sorted [(t0, t1, cf)]
    off, opt = ot["offset"], ot["tile"]
    for i in range(rec["compute_rows"]):
        op = co["op"][i]
        c = cf_of_station.get(co["station"][i])
        for j in range(off[op], off[op + 1]):
            uses.setdefault(opt[j], []).append((co["t0"][i], co["t1"][i], c))
    for v in uses.values():
        v.sort()

    def consumer(tile, t, after=True):
        v = uses.get(tile, [])
        if after:
            for t0, t1, c in v:
                if t0 >= t:
                    return c
        best = None
        for t0, t1, c in v:
            if t0 <= t:
                best = c
        return best if best is not None else (v[0][2] if v else None)

    # ---- L3 homes --------------------------------------------------------------------------
    l3_station = next((i for i, s in enumerate(st) if s["kind"] == "l3"), None)
    cap_total = st[l3_station]["capacity"] if l3_station is not None else 0
    per_tile = cap_total // len(l3) if cap_total else 0
    order = sorted(range(rec["residency_rows"]), key=lambda i: (res["t0"][i], i))
    home = [-1] * rec["residency_rows"]
    live = {t: [] for t in l3_blk}               # l3 -> [end times]
    overflow = 0

    def dist(a, b):
        (ax, ay), (bx, by) = center(l3_blk[a]["rect"]), center(l3_blk[b]["rect"])
        return abs(ax - bx) + abs(ay - by)

    for i in order:
        t0 = res["t0"][i]
        for t in live:
            live[t] = [e for e in live[t] if e > t0]
        room = lambda t: per_tile == 0 or len(live[t]) < per_tile
        c = consumer(res["tile"][i], t0)
        cands = [t for t, _ in abut.get(c, [])] if c is not None else []
        pick = min((t for t in cands if room(t)), key=lambda t: (len(live[t]), t), default=None)
        if pick is None:
            anchor = cands[0] if cands else min(live)
            pick = min((t for t in live if room(t)), key=lambda t: (dist(anchor, t), t), default=None)
        if pick is None:
            pick = min(live, key=lambda t: (len(live[t]), t))
            overflow += 1
        home[i] = pick
        live[pick].append(res["t1"][i])

    by_tile = {}
    for i in range(rec["residency_rows"]):
        by_tile.setdefault(res["tile"][i], []).append((res["t0"][i], res["t1"][i], home[i]))

    def home_at(tile, t):
        for t0, t1, h in by_tile.get(tile, ()):
            if t0 <= t <= t1:
                return h
        v = by_tile.get(tile)
        return v[-1][2] if v else None

    # ---- transits: engine, BlockMover, NoC path ------------------------------------------
    nodes, node_ix = [], {}

    def node(n):
        if n not in node_ix:
            node_ix[n] = len(nodes)
            nodes.append(n)
        return node_ix[n]

    # Engine assignment, in start order: the controller owning the tile's address, then its
    # first engine free at the transfer's start.
    tensor_of = {t["name"]: t for t in tensors}

    def addr_of(tile):
        name, ti, tj = m["tiles"][tile]
        t = tensor_of.get(name)
        return None if t is None else t["base"] + (ti * t["tile_cols"] + tj) * t["tile_bytes"]

    # Each ejection fills one engine's buffer, and that engine writes it to DRAM: pair every
    # ejection with the DMA write that follows it on the same (op, tile) chain.
    write_after = {}
    by_chain = {}
    for i in range(rec["transit_rows"]):
        if tr["hop"][i] in (HOP_EJECT, HOP_DMA_WRITE):
            by_chain.setdefault((tr["op"][i], tr["tile"][i]), []).append(i)
    for rows in by_chain.values():
        rows.sort(key=lambda i: (tr["t0"][i], i))
        for a, b in zip(rows, rows[1:]):
            if tr["hop"][a] == HOP_EJECT and tr["hop"][b] == HOP_DMA_WRITE:
                write_after[a] = b

    t_engine = [-1] * rec["transit_rows"]
    engine_free = [0.0] * len(engines)
    overloaded = 0
    dma_rows = sorted((i for i in range(rec["transit_rows"]) if tr["hop"][i] in (HOP_DMA_IN, HOP_EJECT)),
                      key=lambda i: (tr["t0"][i], i))
    for i in dma_rows:
        if not mc_list:
            break
        a = addr_of(tr["tile"][i])
        tile_bytes = tensor_of[m["tiles"][tr["tile"][i]][0]]["tile_bytes"] if a is not None else 1
        mc = mc_list[(a // tile_bytes) % len(mc_list)] if a is not None else mc_list[0]
        pool = mc_engines[mc]
        free = [k for k in pool if engine_free[k] <= tr["t0"][i]]
        if free:
            k = free[0]
        else:
            k = min(pool, key=lambda q: (engine_free[q], q))
            overloaded += 1
        t_engine[i] = k
        # An ejection holds its engine until that engine has written the buffer to DRAM.
        w = write_after.get(i)
        end = tr["t1"][w] if w is not None else tr["t1"][i]
        if w is not None:
            t_engine[w] = k
        engine_free[k] = max(engine_free[k], end)
    t_bm = [""] * rec["transit_rows"]
    t_path = [[] for _ in range(rec["transit_rows"])]
    unrouted = 0
    for i in range(rec["transit_rows"]):
        hop, tile, t0, t1 = tr["hop"][i], tr["tile"][i], tr["t0"][i], tr["t1"][i]
        h = home_at(tile, t0 if hop != HOP_DMA_IN else t1)
        if hop in (HOP_DMA_IN, HOP_EJECT) and engines:
            eng = engines[t_engine[i]]
            port = attach.get(eng)
            if port and h is not None:
                p = path(port, hub(h))
                if p:
                    t_path[i] = [node(n) for n in (p if hop == HOP_DMA_IN else p[::-1])]
                else:
                    unrouted += 1
            else:
                unrouted += 1
        elif hop in (HOP_BM_DOWN, HOP_BM_UP) and h is not None:
            c = consumer(tile, t1 if hop == HOP_BM_DOWN else t0, after=hop == HOP_BM_DOWN)
            sides = dict(abut.get(c, []))
            if h in sides:
                t_bm[i] = f"{dev}/l3[{h}]/bm[{sides[h]}]"
            elif sides:
                relay = min(sides, key=lambda t: (len(path(hub(h), hub(t)) or [0] * 99), t))
                t_bm[i] = f"{dev}/l3[{relay}]/bm[{sides[relay]}]"
                p = path(hub(h), hub(relay))
                if p:
                    t_path[i] = [node(n) for n in (p if hop == HOP_BM_DOWN else p[::-1])]
    # ---- ports: one block per bus ------------------------------------------------------
    port_buses = {}
    for i in range(rec["transit_rows"]):
        hop = tr["hop"][i]
        if hop not in (HOP_DMA_IN, HOP_EJECT) or not t_path[i]:
            continue
        end = nodes[t_path[i][0] if hop == HOP_DMA_IN else t_path[i][-1]]
        if "/port[" in end:
            bus = "inject" if hop == HOP_DMA_IN else "eject"
            port_buses.setdefault(end, {"inject": [], "eject": []})[bus].append((tr["t0"][i], tr["t1"][i]))
    ports = {name: {bus: bus_load(ivs) for bus, ivs in buses.items()}
             for name, buses in sorted(port_buses.items())}

    bms = sorted({b for b in t_bm if b})
    bm_ix = {b: k for k, b in enumerate(bms)}
    missing = [b for b in bms if b not in names]
    if missing:
        raise BindError(f"derived BlockMover {missing[0]} is not on the floorplan")

    return {
        "format": "kpu-tflow-binding", "version": 2, "derived": True,
        "device": dev,
        # Which run this binding indexes. Its arrays are per row of THAT run's tables, and
        # re-recording into the same folder leaves the old binding.json behind, so a reader
        # compares this with the manifest and ignores a binding that does not match.
        "bundle": bundle_identity(rec["manifest"]),
        "policies": {
            "dram": "operands in program order from 0, 4 KiB aligned, tile-major; top of memory = next power of two (no DRAM size is declared)",
            "dma": "the controller owning the tile's DRAM address (tile-interleaved across controllers), then its first free engine; overload counted",
            "l3": "homed at residency start in the least-loaded L3 tile abutting the next consuming compute tile; per-tile capacity = L3 capacity / L3 tiles; nearest L3 with room otherwise",
            "bm": "home L3's mover facing the destination compute tile, else a NoC relay to the nearest abutting L3",
            "noc": "shortest path over ring links from the engine's attached port to the home hub; every node busy for the whole transfer (no per-hop NoC timing at L-T1)",
            "ports": "two buses per port (injection, ejection), one block each; NOT modelled at L-T1, so a bus over one block is over-subscription by the derived path, counted per bus",
        },
        "notes": notes,
        "dram": {"top": top, "end": addr, "align": ALIGN, "layout": "tile-major", "tensors": tensors},
        "l3": {"tiles": [l3_blk[t]["name"] for t in sorted(l3_blk)], "per_tile_capacity": per_tile,
               "overflow": overflow, "residency_home": home},
        "engines": engines,
        "dma": {"interleave": "tile", "controllers": len(mc_list), "overloaded": overloaded},
        "transit_engine": t_engine,
        "bms": bms,
        "transit_bm": [bm_ix[b] if b else -1 for b in t_bm],
        "nodes": nodes,
        "transit_path": t_path,
        "unrouted": unrouted,
        "ports": {"capacity_per_bus": 1, "modelled": False, "buses": ports},
    }


def bus_load(intervals):
    """A bus's transfers, its peak simultaneous blocks, and how many transfers started while
    it already held one (ends before starts at the same cycle, as everywhere in the record)."""
    ev = sorted([(a, 1) for a, b in intervals if b > a] + [(b, -1) for a, b in intervals if b > a])
    cur = peak = over = 0
    for _, d in ev:
        if d > 0 and cur >= 1:
            over += 1
        cur += d
        peak = max(peak, cur)
    return {"transfers": len(intervals), "peak": peak, "oversubscribed": over}


def bundle_identity(manifest):
    """What a binding must match to be the binding of this bundle (index.html computes the same)."""
    t = manifest["tables"]
    return {"deployment_digest": manifest["deployment_digest"], "level": manifest["level"],
            "makespan": manifest["makespan"], "residency": t["residency"]["rows"],
            "transit": t["transit"]["rows"], "compute": t["compute"]["rows"]}


def main():
    ap = argparse.ArgumentParser(description="Derive a spatial binding for a .tflow bundle.")
    ap.add_argument("bundle")
    ap.add_argument("--floorplan", required=True)
    a = ap.parse_args()
    try:
        rec = tflow_check.load(a.bundle)
        fp = json.loads(Path(a.floorplan).read_text(encoding="utf-8"))
        if fp.get("format") != "kpu-floorplan":
            raise BindError("--floorplan is not a kpu-floorplan file")
        out = bind(rec, fp)
    except (tflow_check.Unreadable, BindError, OSError, ValueError, KeyError) as e:
        print(f"bind: {e}", file=sys.stderr)
        sys.exit(2)
    (Path(a.bundle) / "binding.json").write_text(json.dumps(out, separators=(",", ":")),
                                                 encoding="utf-8")
    print(f"bind: {a.bundle}/binding.json  {len(out['engines'])} DMA engines, "
          f"{len(out['l3']['tiles'])} L3 tiles ({out['l3']['per_tile_capacity']} slots each, "
          f"{out['l3']['overflow']} over), {out['dma']['overloaded']} DMA transfers past a "
          f"controller's engines, {len(out['bms'])} BlockMovers used, "
          f"{len(out['nodes'])} NoC nodes on paths, {out['unrouted']} unrouted"
          + "".join(f"\n  port {name}: {bus} peak {v['peak']} blocks on a 1-block bus, "
                    f"{v['oversubscribed']} of {v['transfers']} transfers over-subscribed "
                    f"(ports are not modelled at L-T1)"
                    for name, buses in out["ports"]["buses"].items()
                    for bus, v in buses.items() if v["peak"] > 1)
          + "".join(f"\n  note: {n}" for n in out["notes"]))


if __name__ == "__main__":
    main()
