// ============================================================================
// src/program/mmio.cpp
// The call ABI as MMIO (#305 increment 3). See mmio.hpp for the address map and for why the
// bus, not the type system, is where "no payload" is checked at run time.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/orchestration/mmio.hpp>

#include <sw/kpu/program/platform/digest.hpp>
#include <sw/kpu/program/platform/resource_map.hpp>

#include <cstring>
#include <sstream>

namespace sw::kpu::orchestration {

namespace {
std::string hex(std::uint64_t v) {
    std::ostringstream s;
    s << "0x" << std::hex << v;
    return s.str();
}
} // namespace

// ---- the bus ----------------------------------------------------------------
void Bus::add(Region r) {
    for (const Region& o : regions_)
        if (r.base < o.base + o.size && o.base < r.base + r.size)
            throw std::invalid_argument("bus: " + r.label + " at " + hex(r.base) +
                                        " overlaps " + o.label + " at " + hex(o.base));
    regions_.push_back(std::move(r));
}

void Bus::map_ram(std::uint64_t base, std::vector<std::uint8_t>& mem, std::string label) {
    Region r;
    r.base = base;
    r.size = mem.size();
    r.kind = Kind::Ram;
    r.mem = &mem;
    r.label = std::move(label);
    add(std::move(r));
}

void Bus::map_mmio(std::uint64_t base, std::uint64_t size, ReadFn rd, WriteFn wr,
                   std::string label) {
    Region r;
    r.base = base;
    r.size = size;
    r.kind = Kind::Mmio;
    r.rd = std::move(rd);
    r.wr = std::move(wr);
    r.label = std::move(label);
    add(std::move(r));
}

void Bus::map_fault(std::uint64_t base, std::uint64_t size, std::string label) {
    if (size == 0) return;
    Region r;
    r.base = base;
    r.size = size;
    r.kind = Kind::Fault;
    r.label = std::move(label);
    add(std::move(r));
}

Bus::Region& Bus::find(std::uint64_t addr, bool write) {
    if (addr % 8 != 0)
        throw BusFault("bus: unaligned " + std::string(write ? "write" : "read") + " at " +
                       hex(addr));
    for (Region& r : regions_)
        if (addr >= r.base && addr + 8 <= r.base + r.size) {
            // THE ISOLATION PROPERTY. Tensor DRAM is mapped, and mapped as a fault: an
            // orchestrator that reaches for a tensor element finds a trap, not a value.
            if (r.kind == Kind::Fault)
                throw BusFault("bus: " + std::string(write ? "write" : "read") + " at " +
                               hex(addr) + " faults: " + r.label +
                               " is not the orchestrator's to touch");
            return r;
        }
    throw BusFault("bus: " + std::string(write ? "write" : "read") + " at " + hex(addr) +
                   " hits nothing mapped");
}

std::uint64_t Bus::read64(std::uint64_t addr) {
    Region& r = find(addr, false);
    std::uint64_t v = 0;
    if (r.kind == Kind::Ram) {
        const std::uint8_t* p = r.mem->data() + (addr - r.base);
        for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    } else {
        v = r.rd(addr - r.base);
    }
    log_.push_back(BusAccess{addr, false, v});
    return v;
}

void Bus::write64(std::uint64_t addr, std::uint64_t value) {
    Region& r = find(addr, true);
    // Logged BEFORE the side effect: a doorbell write services the ring, and the log should
    // read in the order the orchestrator acted, not the order the device responded.
    log_.push_back(BusAccess{addr, true, value});
    if (r.kind == Kind::Ram) {
        std::uint8_t* p = r.mem->data() + (addr - r.base);
        for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(value >> (8 * i));
    } else {
        r.wr(addr - r.base, value);
    }
}

std::string Bus::log_digest() const {
    std::string bytes;
    bytes.reserve(log_.size() * 24);
    for (const BusAccess& a : log_) {
        bytes += a.write ? 'W' : 'R';
        bytes += hex(a.addr);
        bytes += '=';
        bytes += hex(a.value);
        bytes += '\n';
    }
    return program::platform::digest_of(bytes);
}

// ---- the KPU register window ------------------------------------------------
KpuMmioDevice::KpuMmioDevice(KpuDevice& dev, std::vector<std::uint8_t>& ctrl,
                             std::uint64_t ctrl_base, const abi::NameTable& names)
    : dev_(dev), ctrl_(ctrl), ctrl_base_(ctrl_base), names_(names) {
    inventory_ = program::platform::ResourceMap(dev_.deployment()).enumerate();
}

std::uint8_t* KpuMmioDevice::ctrl_at(std::uint64_t addr, std::size_t bytes) {
    // The device reaches control memory directly -- it is the machine -- but only control
    // memory. A ring programmed to point anywhere else is refused here, not followed.
    if (addr < ctrl_base_ || addr + bytes > ctrl_base_ + ctrl_.size())
        throw BusFault("kpu: ring or diagnosis area at " + hex(addr) +
                       " is outside control memory");
    return ctrl_.data() + (addr - ctrl_base_);
}

std::pair<std::uint32_t, std::uint32_t> KpuMmioDevice::write_diag(const std::string& text) {
    if (text.empty() || diag_size_ == 0) return {0, 0};
    const std::uint64_t len = std::min<std::uint64_t>(text.size(), diag_size_);
    // Never split a text across the wrap: start over at zero instead. A reader takes each
    // text as soon as it sees the completion naming it, so overwriting old text is safe.
    if (diag_cursor_ + len > diag_size_) diag_cursor_ = 0;
    const std::uint64_t off = diag_cursor_;
    std::memcpy(ctrl_at(diag_base_ + off, len), text.data(), len);
    diag_cursor_ = (off + len + 7) & ~std::uint64_t{7};         // keep word-aligned starts
    return {static_cast<std::uint32_t>(off), static_cast<std::uint32_t>(len)};
}

void KpuMmioDevice::service_descriptors() {
    // THE DOORBELL: consume everything between head and the new tail, in order.
    while (dring_head_ != dring_tail_) {
        abi::DescriptorRecord rec{};
        std::memcpy(rec.data(), ctrl_at(dring_base_ + dring_head_ * abi::kDescriptorBytes,
                                        abi::kDescriptorBytes),
                    abi::kDescriptorBytes);
        dev_.submit(abi::decode(rec, names_));
        dring_head_ = (dring_head_ + 1) % dring_size_;
    }
    Completion c;
    while (dev_.pop_completion(c)) {
        const auto [off, len] = write_diag(c.diagnosis);
        for (const abi::CompletionRecord& r : abi::encode(c, names_, off, len))
            backlog_.push_back(r);
    }
    flush_completions();
}

void KpuMmioDevice::flush_completions() {
    // A completion that spans records goes in whole or waits: a reader must never see the
    // head of one before its continuation exists.
    while (!backlog_.empty()) {
        std::size_t span = 1;
        while (span <= backlog_.size() && backlog_[span - 1][19] != 0) ++span;
        std::uint64_t used = (cring_head_ + cring_size_ - cring_tail_) % cring_size_;
        if (used + span > cring_size_ - 1) return;            // full: wait for an acknowledge
        for (std::size_t i = 0; i < span; ++i) {
            std::memcpy(ctrl_at(cring_base_ + cring_head_ * abi::kCompletionBytes,
                                abi::kCompletionBytes),
                        backlog_.front().data(), abi::kCompletionBytes);
            backlog_.pop_front();
            cring_head_ = (cring_head_ + 1) % cring_size_;
        }
    }
}

std::uint64_t KpuMmioDevice::read_reg(std::uint64_t off) {
    using namespace abi::reg;
    const StatusSnapshot st = dev_.status();
    switch (off) {
        case ID:              return abi::kMagic;
        case ABI_VERSION:     return (abi::kAbiMajor << 16) | abi::kAbiMinor;
        case CAP_L3_TILES:    return st.l3_capacity;
        case CAP_COMPUTE:     return dev_.deployment().device_view().compute_tiles;
        case CAP_INVENTORY:   return inventory_.size();
        case INV_ENTRY:
            return inv_index_ < inventory_.size()
                       ? abi::pack_resource(inventory_[inv_index_], names_)
                       : ~std::uint64_t{0};
        case OP_COUNT:        return dev_.operator_count();
        case DRING_BASE:      return dring_base_;
        case DRING_SIZE:      return dring_size_;
        case DRING_HEAD:      return dring_head_;
        case CRING_BASE:      return cring_base_;
        case CRING_SIZE:      return cring_size_;
        case CRING_HEAD:      return cring_head_;
        case ST_CREDITS_FREE: return st.credits_free();
        case ST_RESERVED:     return st.reserved;
        case ST_HELD:         return st.held;
        case ST_COMPLETED:    return st.completed;
        case RES_RESULT: {
            const TileRef t{names_.tensor_name(static_cast<std::uint32_t>(res_tensor_)),
                            static_cast<program::Dim>(res_ti_),
                            static_cast<program::Dim>(res_tj_)};
            return dev_.is_resident(t) ? 1 : 0;
        }
        case MAN_VALID:       return dev_.manifest(static_cast<std::uint32_t>(man_op_)).valid;
        case MAN_PEAK_LIVE:
            return dev_.manifest(static_cast<std::uint32_t>(man_op_)).peak_live_tiles;
        case MAN_N_READS:
            return dev_.manifest(static_cast<std::uint32_t>(man_op_)).reads.size();
        case MAN_READ_TENSOR:
        case MAN_READ_TI:
        case MAN_READ_TJ: {
            const auto& reads = dev_.manifest(static_cast<std::uint32_t>(man_op_)).reads;
            if (man_read_ >= reads.size()) return ~std::uint64_t{0};
            const TileRef& t = reads[man_read_];
            if (off == MAN_READ_TENSOR) return names_.tensor_index(t.tensor);
            return off == MAN_READ_TI ? t.ti : t.tj;
        }
        case MAN_ERR_OFF:     return man_err_off_;
        case MAN_ERR_LEN:     return man_err_len_;
        case DIAG_BASE:       return diag_base_;
        case DIAG_SIZE:       return diag_size_;
        case IRQ_STATUS:      return cring_head_ != cring_tail_ ? 1 : 0;
        default:
            throw BusFault("kpu: read of register " + hex(off) + ", which does not exist");
    }
}

void KpuMmioDevice::write_reg(std::uint64_t off, std::uint64_t v) {
    using namespace abi::reg;
    switch (off) {
        case INV_INDEX:    inv_index_ = v; return;
        case DRING_BASE:   dring_base_ = v; dring_head_ = dring_tail_ = 0; return;
        case DRING_SIZE:   dring_size_ = v; dring_head_ = dring_tail_ = 0; return;
        case DRING_TAIL:
            if (dring_size_ == 0 || v >= dring_size_)
                throw BusFault("kpu: doorbell with tail " + std::to_string(v) +
                               " on a ring of " + std::to_string(dring_size_));
            dring_tail_ = v;
            service_descriptors();
            return;
        case CRING_BASE:   cring_base_ = v; cring_head_ = cring_tail_ = 0; return;
        case CRING_SIZE:   cring_size_ = v; cring_head_ = cring_tail_ = 0; return;
        case CRING_TAIL:
            if (cring_size_ == 0 || v >= cring_size_)
                throw BusFault("kpu: completion acknowledge " + std::to_string(v) +
                               " on a ring of " + std::to_string(cring_size_));
            cring_tail_ = v;
            flush_completions();
            return;
        case RES_Q_TENSOR: res_tensor_ = v; return;
        case RES_Q_TI:     res_ti_ = v; return;
        case RES_Q_TJ:     res_tj_ = v; return;
        case MAN_OP: {
            man_op_ = v;
            man_read_ = 0;
            man_err_off_ = man_err_len_ = 0;
            if (v >= dev_.operator_count())
                throw BusFault("kpu: manifest of operator " + std::to_string(v) + " of " +
                               std::to_string(dev_.operator_count()));
            const OperatorManifest& m = dev_.manifest(static_cast<std::uint32_t>(v));
            if (!m.valid) {
                const auto [o, n] = write_diag(m.error);
                man_err_off_ = o;
                man_err_len_ = n;
            }
            return;
        }
        case MAN_READ_IDX: man_read_ = v; return;
        case DIAG_BASE:    diag_base_ = v; diag_cursor_ = 0; return;
        case DIAG_SIZE:    diag_size_ = v; diag_cursor_ = 0; return;
        case IRQ_ACK:      return;   // the notifier is level-triggered on head != tail
        default:
            throw BusFault("kpu: write of register " + hex(off) + ", which does not exist");
    }
}

// ---- the port ---------------------------------------------------------------
namespace {
constexpr std::uint64_t kRingEntries = 64;
constexpr std::uint64_t kDiagBytes = 0x2000;
} // namespace

MmioPort::MmioPort(Bus& bus, std::uint64_t mmio_base, std::uint64_t ctrl_base,
                   const abi::NameTable& names)
    : bus_(bus), mmio_(mmio_base), dring_(ctrl_base),
      cring_(ctrl_base + kRingEntries * abi::kDescriptorBytes),
      diag_(ctrl_base + kRingEntries * (abi::kDescriptorBytes + abi::kCompletionBytes)),
      names_(names) {
    using namespace abi::reg;
    if (rd(ID) != abi::kMagic)
        throw BusFault("mmio port: no KPU at " + hex(mmio_base));
    if ((rd(ABI_VERSION) >> 16) != abi::kAbiMajor)
        throw BusFault("mmio port: the KPU speaks ABI major " +
                       std::to_string(rd(ABI_VERSION) >> 16));
    wr(DRING_BASE, dring_);
    wr(DRING_SIZE, kRingEntries);
    wr(CRING_BASE, cring_);
    wr(CRING_SIZE, kRingEntries);
    wr(DIAG_BASE, diag_);
    wr(DIAG_SIZE, kDiagBytes);
}

void MmioPort::submit(const Descriptor& d) {
    using namespace abi::reg;
    const std::uint64_t next = (dtail_ + 1) % kRingEntries;
    // The host device services the ring on every doorbell, so it is never full here; a full
    // ring would mean the device stopped consuming, and waiting on it would be a hang.
    if (next == rd(DRING_HEAD))
        throw BusFault("mmio port: descriptor ring full; the device is not consuming");
    const abi::DescriptorRecord rec = abi::encode(d, names_);
    const std::uint64_t at = dring_ + dtail_ * abi::kDescriptorBytes;
    for (std::size_t w = 0; w < abi::kDescriptorBytes / 8; ++w) {
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(rec[w * 8 + i]) << (8 * i);
        bus_.write64(at + w * 8, v);
    }
    dtail_ = next;
    wr(DRING_TAIL, dtail_);                                  // the doorbell
}

abi::CompletionRecord MmioPort::read_completion_record(std::uint64_t index) {
    abi::CompletionRecord rec{};
    const std::uint64_t at = cring_ + index * abi::kCompletionBytes;
    for (std::size_t w = 0; w < abi::kCompletionBytes / 8; ++w) {
        const std::uint64_t v = bus_.read64(at + w * 8);
        for (int i = 0; i < 8; ++i) rec[w * 8 + i] = static_cast<std::uint8_t>(v >> (8 * i));
    }
    return rec;
}

std::string MmioPort::read_text(std::uint32_t off, std::uint32_t len) {
    std::string out;
    out.reserve(len);
    for (std::uint32_t w = 0; w < len; w += 8) {
        const std::uint64_t v = bus_.read64(diag_ + off + w);
        for (std::uint32_t i = 0; i < 8 && w + i < len; ++i)
            out += static_cast<char>((v >> (8 * i)) & 0xFF);
    }
    return out;
}

bool MmioPort::poll_completion(Completion& out) {
    using namespace abi::reg;
    if (rd(IRQ_STATUS) == 0) return false;                   // nothing pending
    bool more = false;
    std::uint32_t off = 0, len = 0;
    out = abi::decode(read_completion_record(ctail_), names_, more, off, len);
    ctail_ = (ctail_ + 1) % kRingEntries;
    while (more) {
        abi::decode_continuation(read_completion_record(ctail_), names_, out, more);
        ctail_ = (ctail_ + 1) % kRingEntries;
    }
    // Read the text BEFORE acknowledging: the acknowledge frees ring space, and the device may
    // reuse the diagnosis area for the next completion it posts.
    if (len != 0) out.diagnosis = read_text(off, len);
    wr(CRING_TAIL, ctail_);
    wr(IRQ_ACK, 1);
    return true;
}

StatusSnapshot MmioPort::read_status() {
    using namespace abi::reg;
    StatusSnapshot s;
    s.l3_capacity = static_cast<std::uint32_t>(rd(CAP_L3_TILES));
    s.held = static_cast<std::uint32_t>(rd(ST_HELD));
    s.reserved = static_cast<std::uint32_t>(rd(ST_RESERVED));
    s.operators = static_cast<std::uint32_t>(rd(OP_COUNT));
    s.completed = static_cast<std::uint32_t>(rd(ST_COMPLETED));
    return s;
}

bool MmioPort::is_resident(const TileRef& t) {
    using namespace abi::reg;
    wr(RES_Q_TENSOR, names_.tensor_index(t.tensor));
    wr(RES_Q_TI, t.ti);
    wr(RES_Q_TJ, t.tj);
    return rd(RES_RESULT) != 0;
}

std::uint32_t MmioPort::operator_count() {
    return static_cast<std::uint32_t>(rd(abi::reg::OP_COUNT));
}

OperatorManifest MmioPort::manifest(std::uint32_t op) {
    using namespace abi::reg;
    OperatorManifest m;
    m.index = op;
    wr(MAN_OP, op);
    m.valid = rd(MAN_VALID) != 0;
    if (!m.valid) {
        m.error = read_text(static_cast<std::uint32_t>(rd(MAN_ERR_OFF)),
                            static_cast<std::uint32_t>(rd(MAN_ERR_LEN)));
        return m;
    }
    m.peak_live_tiles = static_cast<std::uint32_t>(rd(MAN_PEAK_LIVE));
    const std::uint64_t n = rd(MAN_N_READS);
    for (std::uint64_t i = 0; i < n; ++i) {
        wr(MAN_READ_IDX, i);
        TileRef t;
        t.tensor = names_.tensor_name(static_cast<std::uint32_t>(rd(MAN_READ_TENSOR)));
        t.ti = static_cast<program::Dim>(rd(MAN_READ_TI));
        t.tj = static_cast<program::Dim>(rd(MAN_READ_TJ));
        m.reads.push_back(t);
    }
    return m;
}

std::vector<program::platform::ResourceName> MmioPort::inventory() {
    using namespace abi::reg;
    std::vector<program::platform::ResourceName> out;
    const std::uint64_t n = rd(CAP_INVENTORY);
    for (std::uint64_t i = 0; i < n; ++i) {
        wr(INV_INDEX, i);
        out.push_back(abi::unpack_resource(rd(INV_ENTRY), names_));
    }
    return out;
}

// ---- the system -------------------------------------------------------------
MmioSystem::MmioSystem(KpuDevice& dev)
    : names_(dev.loadable(), dev.deployment()), ctrl_(kCtrlBytes, 0),
      regs_(dev, ctrl_, kCtrlBase, names_), mapped_(map_all(dev.loadable())),
      port_(bus_, kMmioBase, kCtrlBase, names_) {}

bool MmioSystem::map_all(const loadable::Loadable& l) {
    bus_.map_mmio(
        kMmioBase, abi::reg::kWindow, [this](std::uint64_t off) { return regs_.read_reg(off); },
        [this](std::uint64_t off, std::uint64_t v) { regs_.write_reg(off, v); }, "KPU registers");
    bus_.map_ram(kCtrlBase, ctrl_, "control memory");
    // TENSOR DRAM, mapped so that touching it is a fault rather than a miss. Overlap with the
    // control window is refused by the bus itself.
    for (const loadable::TensorRef& t : l.tensors)
        bus_.map_fault(t.device_address, t.size_bytes, "tensor DRAM \"" + t.name + "\"");
    return true;
}

} // namespace sw::kpu::orchestration
