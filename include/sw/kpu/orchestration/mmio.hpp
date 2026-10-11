// ============================================================================
// include/sw/kpu/orchestration/mmio.hpp
// The call ABI as MMIO (#305 increment 3): a bus, the control memory the rings live in, the
// KPU's register window, and the port an orchestrator drives them through.
//
//   orchestrator ── MmioPort ──► Bus ──► [ control memory ]   rings + diagnosis text
//                                    ──► [ KPU registers   ] ─► KpuMmioDevice ─► KpuDevice
//                                    ──► [ tensor DRAM     ]   FAULTS: never the orchestrator's
//
// THE BUS IS WHERE "NO PAYLOAD" BECOMES CHECKABLE AT RUN TIME. A type can show that a
// descriptor has no payload field; it cannot show that a register does not leak a buffer.
// So every orchestrator-side access goes through `Bus`, which logs it, and which maps the
// loadable's tensor DRAM as a FAULT. Two consequences a test can assert:
//
//   isolation         a read or write of tensor DRAM throws BusFault
//   non-interference  two runs whose tensor VALUES differ produce byte-identical bus logs --
//                     if anything the orchestrator observed depended on contents, they would
//                     not. This covers registers that do not exist yet.
//
// The device side reads and writes control memory directly: it is the machine, and its
// accesses are not the orchestrator's observations.
//
// TIMING OF THE HOST BUILD: the device services the descriptor ring synchronously, on the
// doorbell write, so completions are visible before the write returns. Deterministic, and no
// clock. Increment 4 replaces the trigger with Renode's virtual time; the ring protocol stays.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/orchestration/abi.hpp>
#include <sw/kpu/orchestration/kpu_device.hpp>
#include <sw/kpu/orchestration/port.hpp>

#include <cstdint>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::orchestration {

class BusFault : public std::runtime_error {
public:
    explicit BusFault(const std::string& what) : std::runtime_error(what) {}
};

struct BusAccess {
    std::uint64_t addr = 0;
    bool write = false;
    std::uint64_t value = 0;
};

// Aligned 8-byte accesses only: every register is 64 bits wide and every record a multiple
// of 8 bytes, and a narrower access path would be one more thing to keep consistent.
class Bus {
public:
    using ReadFn = std::function<std::uint64_t(std::uint64_t offset)>;
    using WriteFn = std::function<void(std::uint64_t offset, std::uint64_t value)>;

    void map_ram(std::uint64_t base, std::vector<std::uint8_t>& mem, std::string label);
    void map_mmio(std::uint64_t base, std::uint64_t size, ReadFn rd, WriteFn wr,
                  std::string label);
    void map_fault(std::uint64_t base, std::uint64_t size, std::string label);

    std::uint64_t read64(std::uint64_t addr);
    void write64(std::uint64_t addr, std::uint64_t value);

    const std::vector<BusAccess>& log() const { return log_; }
    std::string log_digest() const;

private:
    enum class Kind { Ram, Mmio, Fault };
    struct Region {
        std::uint64_t base = 0, size = 0;
        Kind kind = Kind::Fault;
        std::vector<std::uint8_t>* mem = nullptr;
        ReadFn rd;
        WriteFn wr;
        std::string label;
    };
    void add(Region r);
    Region& find(std::uint64_t addr, bool write);

    std::vector<Region> regions_;
    std::vector<BusAccess> log_;
};

// ----------------------------------------------------------------------------
// The KPU's register window: decodes MMIO into KpuDevice calls
// ----------------------------------------------------------------------------
class KpuMmioDevice {
public:
    KpuMmioDevice(KpuDevice& dev, std::vector<std::uint8_t>& ctrl, std::uint64_t ctrl_base,
                  const abi::NameTable& names);

    std::uint64_t read_reg(std::uint64_t offset);
    void write_reg(std::uint64_t offset, std::uint64_t value);

private:
    // The manifest list MAN_LIST selects: the operator's reads, inherits or retains.
    const std::vector<TileRef>& manifest_list() const {
        const OperatorManifest& m = dev_.manifest(static_cast<std::uint32_t>(man_op_));
        return man_list_ == 1 ? m.inherits : man_list_ == 2 ? m.retains : m.reads;
    }
    std::uint8_t* ctrl_at(std::uint64_t addr, std::size_t bytes);
    void service_descriptors();     // the doorbell
    void flush_completions();       // into the completion ring, as space allows
    std::pair<std::uint32_t, std::uint32_t> write_diag(const std::string& text);

    KpuDevice& dev_;
    std::vector<std::uint8_t>& ctrl_;
    std::uint64_t ctrl_base_;
    const abi::NameTable& names_;
    std::vector<program::platform::ResourceName> inventory_;

    std::uint64_t dring_base_ = 0, dring_size_ = 0, dring_head_ = 0, dring_tail_ = 0;
    std::uint64_t cring_base_ = 0, cring_size_ = 0, cring_head_ = 0, cring_tail_ = 0;
    std::uint64_t diag_base_ = 0, diag_size_ = 0, diag_cursor_ = 0;
    std::uint64_t inv_index_ = 0, res_tensor_ = 0, res_ti_ = 0, res_tj_ = 0;
    std::uint64_t man_op_ = 0, man_read_ = 0, man_list_ = 0, man_err_off_ = 0, man_err_len_ = 0;
    // Completions waiting to be posted, kept UNENCODED: encoding writes the diagnosis text,
    // so it waits until the completion ring and (for a refusal) the DIAG area exist.
    std::deque<Completion> backlog_;
};

// ----------------------------------------------------------------------------
// The orchestrator's side
// ----------------------------------------------------------------------------
class MmioPort final : public KpuPort {
public:
    // Lays out the rings and the diagnosis area in control memory and programs the device's
    // base/size registers -- the driver's job at bring-up, and the orchestrator's here.
    MmioPort(Bus& bus, std::uint64_t mmio_base, std::uint64_t ctrl_base,
             const abi::NameTable& names);

    void submit(const Descriptor& d) override;
    bool poll_completion(Completion& out) override;
    StatusSnapshot read_status() override;
    bool is_resident(const TileRef& t) override;
    std::uint32_t operator_count() override;
    OperatorManifest manifest(std::uint32_t op) override;
    std::vector<program::platform::ResourceName> inventory() override;

private:
    std::uint64_t rd(std::uint64_t reg) { return bus_.read64(mmio_ + reg); }
    void wr(std::uint64_t reg, std::uint64_t v) { bus_.write64(mmio_ + reg, v); }
    std::string read_text(std::uint32_t off, std::uint32_t len);
    abi::CompletionRecord read_completion_record(std::uint64_t index);

    Bus& bus_;
    std::uint64_t mmio_, dring_, cring_, diag_;
    const abi::NameTable& names_;
    std::uint64_t dtail_ = 0, ctail_ = 0;
};

// ----------------------------------------------------------------------------
// Everything wired together, for one run
// ----------------------------------------------------------------------------
// Fixed addresses, far above anything a loadable in this repo places tensors at; the
// constructor refuses a loadable whose tensor DRAM overlaps them rather than mapping one over
// the other.
class MmioSystem {
public:
    static constexpr std::uint64_t kMmioBase = 0x0000'7000'0000'0000ull;
    static constexpr std::uint64_t kCtrlBase = 0x0000'7001'0000'0000ull;
    static constexpr std::uint64_t kCtrlBytes = 0x4000;

    explicit MmioSystem(KpuDevice& dev);

    Bus& bus() { return bus_; }
    KpuPort& port() { return port_; }

private:
    bool map_all(const loadable::Loadable& l);

    // DECLARATION ORDER IS INITIALISATION ORDER, and it matters here: the port programs the
    // device's registers in its constructor, so the bus must be mapped before it exists.
    // `mapped_` is that step, placed between the two.
    abi::NameTable names_;
    std::vector<std::uint8_t> ctrl_;
    Bus bus_;
    KpuMmioDevice regs_;
    bool mapped_;
    MmioPort port_;
};

} // namespace sw::kpu::orchestration
