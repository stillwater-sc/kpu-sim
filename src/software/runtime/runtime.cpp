// KPU Runtime Implementation
// Host-side orchestration for kernel execution

#include <sw/runtime/runtime.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace sw::runtime {

using namespace sw::kpu;

// ============================================================================
// Constructor / Destructor
// ============================================================================

KPURuntime::KPURuntime(KPUSimulator* simulator, const Config& config)
    : simulator_(simulator)
    , config_(config) {

    if (!simulator_) {
        throw std::invalid_argument("KPURuntime: simulator cannot be null");
    }

    // Create resource manager for memory operations
    resource_manager_ = simulator_->create_resource_manager();

    // Create executor for kernel execution
    executor_ = std::make_unique<isa::ConcurrentExecutor>(config_.executor_config);

    // Initialize default stream (stream 0)
    streams_[0] = StreamState{};
}

KPURuntime::~KPURuntime() = default;

KPURuntime::KPURuntime(KPURuntime&&) noexcept = default;
KPURuntime& KPURuntime::operator=(KPURuntime&&) noexcept = default;

// ============================================================================
// Memory Management
// ============================================================================

Address KPURuntime::malloc(Size size, Size alignment) {
    return malloc_pool(size, config_.default_memory_pool, alignment);
}

Address KPURuntime::malloc_pool(Size size, ResourceType pool, Size alignment) {
    auto result = resource_manager_->allocate(pool, size, alignment);
    if (!result) {
        if (config_.verbose) {
            std::cerr << "KPURuntime::malloc: failed to allocate " << size
                      << " bytes from pool " << static_cast<int>(pool) << "\n";
        }
        return 0;
    }
    return *result;
}

void KPURuntime::free(Address ptr) {
    if (ptr == 0) return;

    bool success = resource_manager_->deallocate(ptr);
    if (!success && config_.verbose) {
        std::cerr << "KPURuntime::free: failed to deallocate address 0x"
                  << std::hex << ptr << std::dec << "\n";
    }
}

void KPURuntime::memcpy_h2d(Address dst, const void* src, Size size) {
    if (size == 0) return;
    if (!src) {
        throw std::invalid_argument("KPURuntime::memcpy_h2d: source is null");
    }

    // Write host data to device memory via resource manager
    resource_manager_->write(dst, src, size);
}

void KPURuntime::memcpy_d2h(void* dst, Address src, Size size) {
    if (size == 0) return;
    if (!dst) {
        throw std::invalid_argument("KPURuntime::memcpy_d2h: destination is null");
    }

    // Read device memory to host via resource manager
    resource_manager_->read(src, dst, size);
}

void KPURuntime::memcpy_d2d(Address dst, Address src, Size size) {
    if (size == 0) return;
    if (dst == src) return;

    // Copy within device memory via resource manager
    resource_manager_->copy(src, dst, size);
}

void KPURuntime::memcpy(void* dst, const void* src, Size size, MemcpyKind kind) {
    switch (kind) {
        case MemcpyKind::HostToDevice:
            memcpy_h2d(reinterpret_cast<Address>(dst), src, size);
            break;
        case MemcpyKind::DeviceToHost:
            memcpy_d2h(dst, reinterpret_cast<Address>(const_cast<void*>(src)), size);
            break;
        case MemcpyKind::DeviceToDevice:
            memcpy_d2d(reinterpret_cast<Address>(dst),
                       reinterpret_cast<Address>(const_cast<void*>(src)), size);
            break;
    }
}

void KPURuntime::memset(Address ptr, uint8_t value, Size size) {
    if (size == 0) return;

    // Create a buffer filled with the value and write it
    // For efficiency, write in chunks
    constexpr Size chunk_size = 4096;
    std::vector<uint8_t> buffer(std::min(size, chunk_size), value);

    Size remaining = size;
    Address current = ptr;
    while (remaining > 0) {
        Size to_write = std::min(remaining, chunk_size);
        resource_manager_->write(current, buffer.data(), to_write);
        current += to_write;
        remaining -= to_write;
    }
}

// ============================================================================
// Kernel Execution
// ============================================================================

LaunchResult KPURuntime::launch(const Kernel& kernel,
                                 const std::vector<Address>& args) {
    // Validate kernel
    if (!kernel.is_valid()) {
        return LaunchResult(false, 0, "Invalid kernel");
    }

    // Validate argument count
    const auto& kernel_args = kernel.arguments();
    if (args.size() != kernel_args.size()) {
        std::ostringstream ss;
        ss << "Argument count mismatch: expected " << kernel_args.size()
           << ", got " << args.size();
        return LaunchResult(false, 0, ss.str());
    }

    // Validate addresses (non-zero for non-optional args)
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == 0) {
            std::ostringstream ss;
            ss << "Null address for argument '" << kernel_args[i].name << "'";
            return LaunchResult(false, 0, ss.str());
        }
    }

    // Execute the kernel program
    Cycle cycles = executor_->execute(kernel.program());

    // Update statistics
    total_cycles_ += cycles;
    launch_count_++;

    if (config_.verbose) {
        std::cout << "KPURuntime::launch: executed " << kernel.program().name
                  << " in " << cycles << " cycles\n";
    }

    return LaunchResult(true, cycles);
}

void KPURuntime::launch_async(const Kernel& kernel,
                               const std::vector<Address>& args,
                               Stream stream) {
    if (!stream.valid) {
        throw std::invalid_argument("KPURuntime::launch_async: invalid stream");
    }

    auto it = streams_.find(stream.id);
    if (it == streams_.end() || !it->second.valid) {
        throw std::invalid_argument("KPURuntime::launch_async: stream not found");
    }

    // Queue the launch on the stream
    // Make copies of kernel and args for the lambda
    auto kernel_copy = kernel;
    auto args_copy = args;

    it->second.pending_ops.push([this, kernel_copy, args_copy]() {
        launch(kernel_copy, args_copy);
    });
}

// ============================================================================
// Synchronization
// ============================================================================

void KPURuntime::synchronize() {
    // Execute all pending operations on all streams
    for (auto& [id, state] : streams_) {
        execute_stream_operations(id);
    }
}

void KPURuntime::stream_synchronize(Stream stream) {
    if (!stream.valid) return;

    auto it = streams_.find(stream.id);
    if (it != streams_.end() && it->second.valid) {
        execute_stream_operations(stream.id);
    }
}

void KPURuntime::execute_stream_operations(size_t stream_id) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) return;

    auto& state = it->second;
    while (!state.pending_ops.empty()) {
        auto op = std::move(state.pending_ops.front());
        state.pending_ops.pop();
        op();
    }
}

// ============================================================================
// Streams
// ============================================================================

Stream KPURuntime::create_stream() {
    size_t id = next_stream_id_++;
    streams_[id] = StreamState{};
    return Stream(id);
}

void KPURuntime::destroy_stream(Stream stream) {
    if (!stream.valid || stream.id == 0) return;  // Don't destroy default stream

    auto it = streams_.find(stream.id);
    if (it != streams_.end()) {
        // Execute any pending operations first
        execute_stream_operations(stream.id);
        streams_.erase(it);
    }
}

// ============================================================================
// Events
// ============================================================================

Event KPURuntime::create_event() {
    size_t id = next_event_id_++;
    events_[id] = EventState{};
    return Event(id);
}

void KPURuntime::destroy_event(Event event) {
    if (!event.valid) return;

    auto it = events_.find(event.id);
    if (it != events_.end()) {
        events_.erase(it);
    }
}

void KPURuntime::record_event(Event event, Stream stream) {
    if (!event.valid) {
        throw std::invalid_argument("KPURuntime::record_event: invalid event");
    }

    auto it = events_.find(event.id);
    if (it == events_.end()) {
        throw std::invalid_argument("KPURuntime::record_event: event not found");
    }

    // Synchronize the stream first
    stream_synchronize(stream);

    // Record current cycle count
    it->second.recorded_cycle = total_cycles_;
    it->second.recorded = true;
}

void KPURuntime::wait_event(Event event) {
    if (!event.valid) return;

    auto it = events_.find(event.id);
    if (it == events_.end() || !it->second.recorded) {
        return;  // Event not found or not recorded - nothing to wait for
    }

    // In our synchronous model, events are already complete when recorded
    // In a real async implementation, this would block
}

float KPURuntime::elapsed_time(Event start, Event end) const {
    auto start_it = events_.find(start.id);
    auto end_it = events_.find(end.id);

    if (start_it == events_.end() || end_it == events_.end()) {
        return 0.0f;
    }

    if (!start_it->second.recorded || !end_it->second.recorded) {
        return 0.0f;
    }

    Cycle delta = end_it->second.recorded_cycle - start_it->second.recorded_cycle;

    // Convert cycles to milliseconds based on clock frequency
    // cycles / (clock_ghz * 1e9 Hz) * 1e3 ms/s = cycles / (clock_ghz * 1e6)
    return static_cast<float>(delta) / (config_.clock_ghz * 1e6f);
}

// ============================================================================
// Device Information
// ============================================================================

Size KPURuntime::get_total_memory() const {
    Size total = 0;

    // Sum up capacity of all external memory resources
    size_t count = resource_manager_->get_resource_count(ResourceType::EXTERNAL_MEMORY);
    for (size_t i = 0; i < count; ++i) {
        auto handle = resource_manager_->get_resource(ResourceType::EXTERNAL_MEMORY, i);
        total += handle.capacity;
    }

    return total;
}

Size KPURuntime::get_free_memory() const {
    Size free = 0;

    // Sum up available bytes in all external memory resources
    size_t count = resource_manager_->get_resource_count(ResourceType::EXTERNAL_MEMORY);
    for (size_t i = 0; i < count; ++i) {
        auto handle = resource_manager_->get_resource(ResourceType::EXTERNAL_MEMORY, i);
        free += resource_manager_->get_available_bytes(handle);
    }

    return free;
}

void KPURuntime::print_stats() const {
    std::cout << "KPU Runtime Statistics:\n";
    std::cout << "  Total cycles:    " << total_cycles_ << "\n";
    std::cout << "  Kernel launches: " << launch_count_ << "\n";
    std::cout << "  Active streams:  " << streams_.size() << "\n";
    std::cout << "  Active events:   " << events_.size() << "\n";
    std::cout << "  Total memory:    " << (get_total_memory() / (1024 * 1024)) << " MB\n";
    std::cout << "  Free memory:     " << (get_free_memory() / (1024 * 1024)) << " MB\n";

    if (launch_count_ > 0) {
        double avg_cycles = static_cast<double>(total_cycles_) / launch_count_;
        double avg_time_ms = avg_cycles / (config_.clock_ghz * 1e6);
        std::cout << "  Avg cycles/launch: " << std::fixed << std::setprecision(1) << avg_cycles << "\n";
        std::cout << "  Avg time/launch:   " << std::fixed << std::setprecision(3) << avg_time_ms << " ms\n";
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

void KPURuntime::validate_address(Address addr, const std::string& context) const {
    if (addr == 0) {
        throw std::invalid_argument(context + ": null address");
    }

    auto info = resource_manager_->get_allocation_info(addr);
    if (!info) {
        throw std::invalid_argument(context + ": address not allocated");
    }
}

} // namespace sw::runtime
