// ============================================================================
// include/sw/kpu/timing/noc_topology.hpp
// The folded torus as the CSP NoC processes see it: directed channels between hubs, which
// ones are fold links owned by a port, and the next hop from any hub toward any hub
// (docs/plans/noc-port-arbitration.md step 4).
//
// CHANNELS. Every loop of the torus is a bidirectional ring, so each hub has a channel to the
// next hub and one to the previous hub on each of its two loops: 4 out, 4 in (§1.2). A channel
// is (loop, ring position, direction); the forward channels of a loop form one unidirectional
// ring and the backward channels the other. At the board's corners a row loop and a column
// loop fold over the same wire (ArrayLayout::links()); here they stay two logical channels, one
// per loop, so every hub keeps its four inputs. On a two-hub loop (the T4's) every channel is a
// fold link.
//
// FOLD LINKS belong to a port. The port sits on the link between hub_a and hub_b, so the two
// directed channels across it (a->b and b->a) are the port's to arbitrate: ring-through
// traffic, the blocks it injects, and the blocks it ejects all cross them.
//
// ROUTING IS DIMENSION-ORDERED (§3.5): a block travels its row loop to a hub on the
// destination's column loop, turns, and travels that column loop to the destination. It never
// turns back. That ordering is what, with the bubble rule, makes the fabric deadlock-free: no
// column ring ever waits on a row ring. Within that order the path is the shortest:
//   - of the two hubs where the source's row loop meets the destination's column loop, the
//     turn is at the one with the fewer total hops, the lower l3 index on a tie;
//   - along a loop, the shorter way round, forward on a tie.
// A block on a column loop is past its turn, so the next hop depends on that phase too.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/array_layout.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::timing {

using NocDim = sw::kpu::program::Dim;

struct NocChannel {
    NocDim id = 0;
    NocDim loop = 0;                        // index into ArrayLayout::loops()
    bool column = false;                    // a column loop's channel (the second dimension)
    NocDim from_hub = 0, to_hub = 0;        // l3 indices
    bool forward = true;                    // in the loop's ring order
    std::optional<NocDim> port;             // the port that owns it, if it is a fold link

    // The same unidirectional ring: a block continuing on it needs no bubble.
    bool same_ring(const NocChannel& o) const { return loop == o.loop && forward == o.forward; }
};

class NocTopology {
public:
    explicit NocTopology(const sw::kpu::program::platform::ArrayLayout& layout) {
        using Axis = sw::kpu::program::platform::NocLoop::Axis;
        if (!layout.has_noc())
            throw std::invalid_argument("NocTopology: " + layout.noc_reason());
        hubs_ = layout.l3_count();
        ports_ = layout.ports();
        loops_ = layout.loops();
        place_.assign(hubs_, {});
        fwd_.resize(loops_.size());
        bwd_.resize(loops_.size());
        for (NocDim li = 0; li < loops_.size(); ++li) {
            const auto& ring = loops_[li].hubs;
            const NocDim n = static_cast<NocDim>(ring.size());
            const bool column = loops_[li].axis == Axis::Column;
            for (NocDim i = 0; i < n; ++i) {
                (column ? place_[ring[i]].col : place_[ring[i]].row) = {li, i};
                fwd_[li].push_back(add(li, column, ring[i], ring[(i + 1) % n], true));
                bwd_[li].push_back(add(li, column, ring[i], ring[(i + n - 1) % n], false));
            }
        }
        // A port's fold link: the forward channel hub_a -> hub_b and the backward one back.
        fold_ab_.resize(ports_.size());
        fold_ba_.resize(ports_.size());
        for (const auto& p : ports_) {
            fold_ab_[p.index] = fwd_[p.loop][place(p.hub_a, p.loop)];
            fold_ba_[p.index] = bwd_[p.loop][place(p.hub_b, p.loop)];
            if (channels_[fold_ab_[p.index]].to_hub != p.hub_b ||
                channels_[fold_ba_[p.index]].to_hub != p.hub_a)
                throw std::logic_error("NocTopology: port " + std::to_string(p.index) +
                                       "'s fold link is not a pair of ring channels");
            channels_[fold_ab_[p.index]].port = p.index;
            channels_[fold_ba_[p.index]].port = p.index;
        }
        out_.assign(hubs_, {});
        for (const auto& c : channels_) out_[c.from_hub].push_back(c.id);
        in_.assign(hubs_, {});
        for (const auto& c : channels_) in_[c.to_hub].push_back(c.id);
    }

    NocDim hub_count() const { return hubs_; }
    NocDim port_count() const { return static_cast<NocDim>(ports_.size()); }
    const std::vector<NocChannel>& channels() const { return channels_; }
    const NocChannel& channel(NocDim c) const { return channels_.at(c); }
    const sw::kpu::program::platform::NocPort& port(NocDim k) const { return ports_.at(k); }
    const std::vector<NocDim>& out_channels(NocDim hub) const { return out_.at(hub); }
    const std::vector<NocDim>& in_channels(NocDim hub) const { return in_.at(hub); }

    // The fold channel of port k that leaves `from` (one of its two hubs).
    NocDim fold_from(NocDim k, NocDim from) const {
        const auto& p = ports_.at(k);
        if (from == p.hub_a) return fold_ab_[k];
        if (from == p.hub_b) return fold_ba_[k];
        throw std::invalid_argument("NocTopology::fold_from: hub " + std::to_string(from) +
                                    " is not on port " + std::to_string(k) + "'s fold link");
    }

    // Hops from `from` to `to` on the dimension-ordered route, starting before the turn.
    NocDim distance(NocDim from, NocDim to) const {
        const NocDim t = turn(from, to);
        return ring_distance(place_[from].row.loop, from, t) +
               ring_distance(place_[t].col.loop, t, to);
    }

    // The next channel from `hub` toward `target` (hub != target). `on_column` = the block
    // arrived on a column loop, so it has turned and stays on that loop.
    NocDim next_hop(NocDim hub, NocDim target, bool on_column) const {
        if (hub == target)
            throw std::invalid_argument("NocTopology::next_hop: hub " + std::to_string(hub) +
                                        " is the target");
        if (on_column) {
            if (place_[hub].col.loop != place_[target].col.loop)
                throw std::logic_error("NocTopology::next_hop: a block on column loop " +
                                       std::to_string(place_[hub].col.loop) +
                                       " is bound for hub " + std::to_string(target) +
                                       ", which is not on it");
            return step(place_[hub].col.loop, hub, target);
        }
        const NocDim t = turn(hub, target);
        return t == hub ? step(place_[hub].col.loop, hub, target)
                        : step(place_[hub].row.loop, hub, t);
    }

    // Which fold hub of port k a block bound for hub `dst` enters through (Q4): the one with
    // the shorter route, hub_a on a tie.
    NocDim injection_hub(NocDim k, NocDim dst) const {
        const auto& p = ports_.at(k);
        return distance(p.hub_b, dst) < distance(p.hub_a, dst) ? p.hub_b : p.hub_a;
    }

    // Which fold hub a block leaving `src` for port k exits from: the nearer, hub_a on a tie.
    NocDim exit_hub(NocDim k, NocDim src) const {
        const auto& p = ports_.at(k);
        return distance(src, p.hub_b) < distance(src, p.hub_a) ? p.hub_b : p.hub_a;
    }

private:
    struct At { NocDim loop = 0, pos = 0; };
    struct Place { At row, col; };

    NocDim hubs_ = 0;
    std::vector<sw::kpu::program::platform::NocPort> ports_;
    std::vector<sw::kpu::program::platform::NocLoop> loops_;
    std::vector<Place> place_;                          // each hub's position on its two loops
    std::vector<std::vector<NocDim>> fwd_, bwd_;        // [loop][pos] -> channel leaving pos
    std::vector<NocChannel> channels_;
    std::vector<NocDim> fold_ab_, fold_ba_;
    std::vector<std::vector<NocDim>> out_, in_;

    NocDim add(NocDim loop, bool column, NocDim from, NocDim to, bool forward) {
        NocChannel c;
        c.id = static_cast<NocDim>(channels_.size());
        c.loop = loop;
        c.column = column;
        c.from_hub = from;
        c.to_hub = to;
        c.forward = forward;
        channels_.push_back(c);
        return c.id;
    }

    NocDim place(NocDim hub, NocDim loop) const {
        const Place& p = place_[hub];
        return p.row.loop == loop ? p.row.pos : p.col.pos;
    }

    NocDim ring_distance(NocDim loop, NocDim from, NocDim to) const {
        const NocDim n = static_cast<NocDim>(loops_[loop].hubs.size());
        const NocDim f = (place(to, loop) + n - place(from, loop)) % n;
        return std::min(f, (n - f) % n);
    }

    // One hop along `loop` from `from` toward `to`: the shorter way, forward on a tie.
    NocDim step(NocDim loop, NocDim from, NocDim to) const {
        const NocDim n = static_cast<NocDim>(loops_[loop].hubs.size());
        const NocDim pos = place(from, loop);
        const NocDim f = (place(to, loop) + n - pos) % n;
        return f <= n - f ? fwd_[loop][pos] : bwd_[loop][pos];
    }

    // Where the route from `from` to `to` turns: a hub on both from's row loop and to's
    // column loop, the one with the fewer total hops, the lower index on a tie.
    NocDim turn(NocDim from, NocDim to) const {
        const NocDim row = place_[from].row.loop, col = place_[to].col.loop;
        std::optional<NocDim> best;
        NocDim best_d = 0;
        for (NocDim h : loops_[row].hubs) {
            if (place_[h].col.loop != col) continue;
            const NocDim d = ring_distance(row, from, h) + ring_distance(col, h, to);
            if (!best || d < best_d || (d == best_d && h < *best)) { best = h; best_d = d; }
        }
        if (!best)
            throw std::logic_error("NocTopology: row loop " + std::to_string(row) +
                                   " never meets column loop " + std::to_string(col));
        return *best;
    }
};

} // namespace sw::kpu::timing
