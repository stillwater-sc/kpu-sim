// ============================================================================
// include/sw/kpu/program/platform/array_layout.hpp
// Where the tiles ARE: the logical grid of L3 and compute tiles, the BlockMovers on the
// edges where they abut, and the folded-torus NoC over the L3 router hubs (#286 step 1;
// docs/01-architecture/kpu-architecture.md §5.1-5.2; docs/plans/tile-flow-debugger.md §3.1).
//
// This is the ONE place the layout is derived. The naming map reads it to name the
// BlockMovers and NoC ports, and the floorplan generator reads it to place them, so the two
// cannot disagree about which L3 edge has a mover or which hubs a loop visits.
//
// ---------------------------------------------------------------------------
// THE CHECKERBOARD (§5.1)
// ---------------------------------------------------------------------------
// An alternating grid: cell (r, c) is an L3 tile when r + c is even, a compute tile when it
// is odd. L3 tiles are numbered row-major over L3 cells, compute tiles row-major over CF
// cells. The KPU-T64 is the 8×8 board: 32 L3 tiles and 32 compute tiles.
//
// A layout exists only when the spec describes one. It needs `l3.tiles` declared and equal
// to `compute_tiles`, which an alternating board forces, and an even rows × cols grid of
// 2 × compute_tiles cells. A spec that declares something else (the 16-CF / 8-L3 fixture,
// or a CLI machine with no L3 modules) is a valid DEPLOYMENT with no LAYOUT: its L3 tiles
// and compute tiles are still named, its BlockMovers and NoC ports are not, and a floorplan
// for it is refused with the reason. That keeps every existing spec valid. A layout is
// derived from the spec; it is not a new rule the spec must satisfy.
//
// ---------------------------------------------------------------------------
// BLOCKMOVERS (§5.1, first pass)
// ---------------------------------------------------------------------------
// They belong to the L3 tile, one per edge that ABUTS a compute tile, and the edge index is
// the direction: 0 N, 1 E, 2 S, 3 W. An interior L3 tile has four; a boundary tile fewer.
// So `l3[t]/bm[e]` exists exactly when edge e of tile t touches a compute tile, and the
// indices of a corner tile are sparse ({1, 2} for the top-left one), not renumbered:
// renumbering would make bm[0] mean a different direction on different tiles.
//
// ---------------------------------------------------------------------------
// THE NoC: A FOLDED 2D TORUS (§5.2.1)
// ---------------------------------------------------------------------------
// Two rows form one loop around X, two columns one loop around Y. For rows 2p and 2p+1 the
// loop visits row 2p's L3 tiles left to right, then row 2p+1's right to left:
//
//      (0,0) (0,2) (0,4) (0,6) (1,7) (1,5) (1,3) (1,1) -> back to (0,0)
//
// so every link but the two fold links is two cells long, and the fold links join the rows'
// ends. An R × C board therefore has R/2 row loops of C hubs and C/2 column loops of R hubs.
// The 8×8 board is a 4×4 torus of loops, 8 hubs each.
//
// THE FOLD-END LINKS ARE PORTS, where traffic enters or leaves a loop and where the DMA
// channels attach. Port numbering is fixed: row loop p has port 2p (its W end) and 2p+1
// (its E end); column loop q has port R + 2q (N end) and R + 2q + 1 (S end).
//
// The torus needs both R and C even. A board without one has a layout (cells and
// BlockMovers) but no NoC, and says so.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::platform {

enum class CellKind : std::uint8_t { Empty, L3, Compute };

enum class Edge : std::uint8_t { N = 0, E = 1, S = 2, W = 3 };
inline const char* to_string(Edge e) {
    switch (e) {
        case Edge::N: return "N";
        case Edge::E: return "E";
        case Edge::S: return "S";
        case Edge::W: return "W";
    }
    return "?";
}

struct GridPos {
    Dim r = 0, c = 0;
    bool operator==(const GridPos& o) const { return r == o.r && c == o.c; }
};

struct Cell {
    CellKind kind = CellKind::Empty;
    Dim index = 0;                  // l3[index] or cf[index]
};

// One BlockMover: the L3 tile it lives on, the edge it sits on, the compute tile it feeds.
struct BlockMoverSite {
    Dim l3 = 0;
    Edge edge = Edge::N;
    Dim cf = 0;
};

// One loop of the torus: the L3 tiles its hubs belong to, in ring order.
struct NocLoop {
    enum class Axis : std::uint8_t { Row, Column } axis = Axis::Row;
    Dim index = 0;                  // row loop p, or column loop q
    std::vector<Dim> hubs;          // l3 indices, in ring order
};

// One fold-end port: the loop it belongs to, which end, and the two hubs of its fold link.
struct NocPort {
    Dim index = 0;                  // noc/port[index]
    Dim loop = 0;                   // index into ArrayLayout::loops()
    Dim axis_index = 0;             // which row loop or column loop (NocLoop::index)
    Edge side = Edge::W;            // the array edge it faces
    Dim hub_a = 0, hub_b = 0;       // the fold link it sits on (l3 indices)
    std::string label() const;      // "row1.E", "col0.N"
};

class ArrayLayout {
public:
    // The layout of device `d`, or nullopt with the reason in `why`.
    static std::optional<ArrayLayout> of(const DeviceSpecification& d, std::string* why = nullptr);

    Dim rows() const { return rows_; }
    Dim cols() const { return cols_; }
    const Cell& cell(Dim r, Dim c) const { return cells_.at(static_cast<std::size_t>(r) * cols_ + c); }
    GridPos l3_pos(Dim t) const { return l3_pos_.at(t); }
    GridPos cf_pos(Dim c) const { return cf_pos_.at(c); }
    Dim l3_count() const { return static_cast<Dim>(l3_pos_.size()); }
    Dim cf_count() const { return static_cast<Dim>(cf_pos_.size()); }

    // The compute tile across edge e of L3 tile t, if one abuts it.
    std::optional<Dim> abutting_cf(Dim t, Edge e) const;
    const std::vector<BlockMoverSite>& block_movers() const { return movers_; }

    // The NoC. Empty when the board has no torus; `noc_reason()` then says why.
    bool has_noc() const { return !loops_.empty(); }
    const std::string& noc_reason() const { return noc_reason_; }
    const std::vector<NocLoop>& loops() const { return loops_; }
    const std::vector<NocPort>& ports() const { return ports_; }
    // Every distinct hub-to-hub link, each once, as (lower l3 index, higher l3 index).
    // Fewer than 2 × hubs, because at each of the board's four corners a row loop and a
    // column loop fold over the same link: the 8×8 board has 64 loop links and 60 wires.
    std::vector<std::pair<Dim, Dim>> links() const;

private:
    Dim rows_ = 0, cols_ = 0;
    std::vector<Cell> cells_;
    std::vector<GridPos> l3_pos_, cf_pos_;
    std::vector<BlockMoverSite> movers_;
    std::vector<NocLoop> loops_;
    std::vector<NocPort> ports_;
    std::string noc_reason_;

    void place(Dim r, Dim c, CellKind k);
    void derive_movers();
    void derive_torus();
};

// The board a checkerboard spec implies: the declared array, or the squarest even grid of
// 2 × compute_tiles cells. nullopt with the reason when there is none.
std::optional<std::pair<Dim, Dim>> checkerboard_shape(const DeviceSpecification& d,
                                                      std::string* why = nullptr);

// A declared field the layout contradicts (plan Q8): `movers.block_movers` is a pool size the
// executors schedule today, while the layout has one mover per abutting edge. Reported, not
// refused: the executors still use the declared pool until a level models movers in place.
std::vector<std::string> layout_notes(const DeviceSpecification& d);

// ============================================================================
// Implementation (header-only; pure and deterministic)
// ============================================================================
inline std::string NocPort::label() const {
    const bool row = side == Edge::W || side == Edge::E;
    return std::string(row ? "row" : "col") + std::to_string(axis_index) + "." + to_string(side);
}

inline std::optional<std::pair<Dim, Dim>> checkerboard_shape(const DeviceSpecification& d,
                                                             std::string* why) {
    auto fail = [&](std::string s) -> std::optional<std::pair<Dim, Dim>> {
        if (why) *why = std::move(s);
        return std::nullopt;
    };
    const std::uint64_t cells = 2ull * d.compute_tiles;
    if (d.array.rows && d.array.cols) {
        if (static_cast<std::uint64_t>(*d.array.rows) * *d.array.cols != cells)
            return fail("array " + std::to_string(*d.array.rows) + "x" +
                        std::to_string(*d.array.cols) + " does not hold " +
                        std::to_string(cells) + " cells (2 x compute_tiles)");
        return std::make_pair(*d.array.rows, *d.array.cols);
    }
    // The squarest grid with both sides even (the torus folds by pairs), rows <= cols.
    std::optional<std::pair<Dim, Dim>> best;
    for (std::uint64_t r = 2; r * r <= cells; r += 2)
        if (cells % r == 0 && (cells / r) % 2 == 0)
            best = std::make_pair(static_cast<Dim>(r), static_cast<Dim>(cells / r));
    if (!best)
        return fail("no even rows x cols grid holds " + std::to_string(cells) +
                    " cells; declare array.rows and array.cols");
    return best;
}

inline void ArrayLayout::place(Dim r, Dim c, CellKind k) {
    Cell& cell = cells_[static_cast<std::size_t>(r) * cols_ + c];
    cell.kind = k;
    if (k == CellKind::L3) {
        cell.index = static_cast<Dim>(l3_pos_.size());
        l3_pos_.push_back({r, c});
    } else if (k == CellKind::Compute) {
        cell.index = static_cast<Dim>(cf_pos_.size());
        cf_pos_.push_back({r, c});
    }
}

inline std::optional<Dim> ArrayLayout::abutting_cf(Dim t, Edge e) const {
    const GridPos p = l3_pos(t);
    long r = p.r, c = p.c;
    switch (e) {
        case Edge::N: --r; break;
        case Edge::E: ++c; break;
        case Edge::S: ++r; break;
        case Edge::W: --c; break;
    }
    if (r < 0 || c < 0 || r >= static_cast<long>(rows_) || c >= static_cast<long>(cols_))
        return std::nullopt;
    const Cell& n = cell(static_cast<Dim>(r), static_cast<Dim>(c));
    if (n.kind != CellKind::Compute) return std::nullopt;
    return n.index;
}

inline void ArrayLayout::derive_movers() {
    for (Dim t = 0; t < l3_count(); ++t)
        for (Edge e : {Edge::N, Edge::E, Edge::S, Edge::W})
            if (const auto cf = abutting_cf(t, e)) movers_.push_back({t, e, *cf});
}

inline void ArrayLayout::derive_torus() {
    if (rows_ % 2 != 0 || cols_ % 2 != 0 || rows_ < 2 || cols_ < 2) {
        noc_reason_ = "the folded torus pairs rows and columns, so it needs an even number of "
                      "each; this board is " + std::to_string(rows_) + "x" + std::to_string(cols_);
        return;
    }
    auto l3_at = [&](Dim r, Dim c) { return cell(r, c).index; };
    // Row loops: row 2p left to right, then row 2p+1 right to left.
    for (Dim p = 0; p < rows_ / 2; ++p) {
        NocLoop loop{NocLoop::Axis::Row, p, {}};
        for (Dim c = 0; c < cols_; c += 2) loop.hubs.push_back(l3_at(2 * p, c));
        for (long c = static_cast<long>(cols_) - 1; c > 0; c -= 2)
            loop.hubs.push_back(l3_at(2 * p + 1, static_cast<Dim>(c)));
        loops_.push_back(std::move(loop));
    }
    // Column loops: column 2q top to bottom, then column 2q+1 bottom to top.
    for (Dim q = 0; q < cols_ / 2; ++q) {
        NocLoop loop{NocLoop::Axis::Column, q, {}};
        for (Dim r = 0; r < rows_; r += 2) loop.hubs.push_back(l3_at(r, 2 * q));
        for (long r = static_cast<long>(rows_) - 1; r > 0; r -= 2)
            loop.hubs.push_back(l3_at(static_cast<Dim>(r), 2 * q + 1));
        loops_.push_back(std::move(loop));
    }
    // Fold-end ports. In ring order the fold links are (half-1, half) at the far end and
    // (last, 0) at the near end, where half is the number of hubs on one row/column.
    for (Dim li = 0; li < loops_.size(); ++li) {
        const NocLoop& loop = loops_[li];
        const Dim n = static_cast<Dim>(loop.hubs.size()), half = n / 2;
        const bool row = loop.axis == NocLoop::Axis::Row;
        NocPort near{0, li, loop.index, row ? Edge::W : Edge::N, loop.hubs[n - 1], loop.hubs[0]};
        NocPort far{0, li, loop.index, row ? Edge::E : Edge::S, loop.hubs[half - 1], loop.hubs[half]};
        near.index = static_cast<Dim>(ports_.size());
        ports_.push_back(near);
        far.index = static_cast<Dim>(ports_.size());
        ports_.push_back(far);
    }
}

inline std::vector<std::pair<Dim, Dim>> ArrayLayout::links() const {
    std::vector<std::pair<Dim, Dim>> out;
    for (const NocLoop& loop : loops_)
        for (std::size_t i = 0; i < loop.hubs.size(); ++i) {
            Dim a = loop.hubs[i], b = loop.hubs[(i + 1) % loop.hubs.size()];
            if (a > b) std::swap(a, b);
            bool seen = false;
            for (const auto& l : out) seen = seen || (l.first == a && l.second == b);
            if (!seen) out.emplace_back(a, b);
        }
    return out;
}

inline std::optional<ArrayLayout> ArrayLayout::of(const DeviceSpecification& d, std::string* why) {
    auto fail = [&](std::string s) -> std::optional<ArrayLayout> {
        if (why) *why = std::move(s);
        return std::nullopt;
    };
    ArrayLayout L;
    if (d.topology == "checkerboard") {
        if (!d.l3.tiles)
            return fail("l3.tiles is not declared, so the checkerboard has no L3 tiles to place");
        if (*d.l3.tiles != d.compute_tiles)
            return fail("an alternating checkerboard has as many L3 tiles as compute tiles; "
                        "this device declares " + std::to_string(*d.l3.tiles) + " L3 and " +
                        std::to_string(d.compute_tiles) + " compute tiles");
        std::string shape_why;
        const auto shape = checkerboard_shape(d, &shape_why);
        if (!shape) return fail(shape_why);
        L.rows_ = shape->first;
        L.cols_ = shape->second;
        L.cells_.assign(static_cast<std::size_t>(L.rows_) * L.cols_, Cell{});
        for (Dim r = 0; r < L.rows_; ++r)
            for (Dim c = 0; c < L.cols_; ++c)
                L.place(r, c, (r + c) % 2 == 0 ? CellKind::L3 : CellKind::Compute);
        L.derive_movers();
        L.derive_torus();
        return L;
    }
    if (d.topology == "news") {
        // One compute tile fed by the four L3 tiles around it: a 3×3 cross.
        if (d.compute_tiles != 1)
            return fail("a news layout is one compute tile with four L3 tiles around it; "
                        "this device declares " + std::to_string(d.compute_tiles) + " compute tiles");
        if (!d.l3.tiles || *d.l3.tiles != 4)
            return fail("a news layout has exactly four L3 tiles; declare l3.tiles = 4");
        L.rows_ = L.cols_ = 3;
        L.cells_.assign(9, Cell{});
        L.place(0, 1, CellKind::L3);
        L.place(1, 0, CellKind::L3);
        L.place(1, 1, CellKind::Compute);
        L.place(1, 2, CellKind::L3);
        L.place(2, 1, CellKind::L3);
        L.derive_movers();
        L.noc_reason_ = "a news layout has no NoC: its four L3 tiles do not form a loop";
        return L;
    }
    if (d.topology == "single") {
        if (d.compute_tiles != 1) return fail("a single layout has one compute tile");
        if (!d.l3.tiles || *d.l3.tiles != 1)
            return fail("a single layout has one L3 tile beside its compute tile; declare "
                        "l3.tiles = 1");
        L.rows_ = 1;
        L.cols_ = 2;
        L.cells_.assign(2, Cell{});
        L.place(0, 0, CellKind::L3);
        L.place(0, 1, CellKind::Compute);
        L.derive_movers();
        L.noc_reason_ = "a single layout has one L3 tile and no NoC";
        return L;
    }
    return fail("unknown topology '" + d.topology + "'");
}

inline std::vector<std::string> layout_notes(const DeviceSpecification& d) {
    std::vector<std::string> out;
    const auto L = ArrayLayout::of(d);
    if (!L) return out;
    const auto n = L->block_movers().size();
    if (d.movers.block_movers != n)
        out.push_back("movers.block_movers declares " + std::to_string(d.movers.block_movers) +
                      " but the layout has " + std::to_string(n) +
                      " (one per L3 edge that abuts a compute tile); the executors still "
                      "schedule the declared pool");
    return out;
}

} // namespace sw::kpu::program::platform
