#pragma once
// KPU Runtime Library
// Host-side orchestration for kernel execution
//
// Provides a CUDA-like API for:
// - Memory management (malloc, memcpy)
// - Kernel launching
// - Synchronization
// - Streams and events for async execution

#include <sw/kpu/kpu_simulator.hpp>
#include <sw/kpu/resource_api.hpp>
#include <sw/kpu/kernel.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>

#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <string>

namespace sw::runtime {

// Forward declarations
class KPURuntime;

/**
 * @brief Handle for a stream (async execution queue)
 */
struct Stream {
    size_t id = 0;
    bool valid = false;

    Stream() = default;
    explicit Stream(size_t stream_id) : id(stream_id), valid(true) {}

    bool operator==(const Stream& other) const { return id == other.id && valid == other.valid; }
    bool operator!=(const Stream& other) const { return !(*this == other); }
};

/**
 * @brief Handle for an event (synchronization point)
 */
struct Event {
    size_t id = 0;
    bool valid = false;

    Event() = default;
    explicit Event(size_t event_id) : id(event_id), valid(true) {}

    bool operator==(const Event& other) const { return id == other.id && valid == other.valid; }
    bool operator!=(const Event& other) const { return !(*this == other); }
};

/**
 * @brief Result of a kernel launch
 */
struct LaunchResult {
    bool success = false;
    kpu::Cycle cycles = 0;
    std::string error;

    LaunchResult() = default;
    LaunchResult(bool ok, kpu::Cycle c, const std::string& err = "")
        : success(ok), cycles(c), error(err) {}
};

/**
 * @brief Memory copy direction
 */
enum class MemcpyKind {
    HostToDevice,    // memcpy_h2d
    DeviceToHost,    // memcpy_d2h
    DeviceToDevice   // memcpy_d2d
};

/**
 * @brief KPU Runtime - Host-side orchestration for kernel execution
 *
 * Provides a high-level API similar to CUDA for:
 * - Allocating and managing device memory
 * - Copying data between host and device
 * - Launching kernels
 * - Synchronizing execution
 *
 * Usage:
 * @code
 * KPUSimulator sim(config);
 * KPURuntime runtime(&sim);
 *
 * // Allocate device memory
 * Address A = runtime.malloc(M * K * sizeof(float));
 * Address B = runtime.malloc(K * N * sizeof(float));
 * Address C = runtime.malloc(M * N * sizeof(float));
 *
 * // Copy input data to device
 * runtime.memcpy_h2d(A, host_A.data(), M * K * sizeof(float));
 * runtime.memcpy_h2d(B, host_B.data(), K * N * sizeof(float));
 *
 * // Launch kernel
 * Kernel kernel = Kernel::create_matmul(M, N, K);
 * runtime.launch(kernel, {A, B, C});
 *
 * // Copy result back
 * runtime.memcpy_d2h(host_C.data(), C, M * N * sizeof(float));
 *
 * // Cleanup
 * runtime.free(A);
 * runtime.free(B);
 * runtime.free(C);
 * @endcode
 */
class KPURuntime {
public:
    /**
     * @brief Configuration for the runtime
     */
    struct Config {
        // Default memory pool for allocations
        kpu::ResourceType default_memory_pool = kpu::ResourceType::EXTERNAL_MEMORY;

        // Executor configuration
        kpu::isa::ResourceConfig executor_config;

        // Clock frequency for timing calculations (GHz)
        double clock_ghz = 1.0;

        // Enable verbose logging
        bool verbose = false;

        Config() {
            executor_config.num_memory_channels = 4;
            executor_config.num_block_movers = 8;
            executor_config.num_streamers = 16;
        }
    };

    // =========================================
    // Constructors
    // =========================================

    /**
     * @brief Construct runtime attached to a simulator
     * @param simulator Pointer to KPUSimulator (must outlive runtime)
     * @param config Runtime configuration
     */
    explicit KPURuntime(kpu::KPUSimulator* simulator,
                        const Config& config = Config{});

    ~KPURuntime();

    // Non-copyable
    KPURuntime(const KPURuntime&) = delete;
    KPURuntime& operator=(const KPURuntime&) = delete;

    // Movable
    KPURuntime(KPURuntime&&) noexcept;
    KPURuntime& operator=(KPURuntime&&) noexcept;

    // =========================================
    // Memory Management
    // =========================================

    /**
     * @brief Allocate device memory
     * @param size Size in bytes
     * @param alignment Alignment requirement (default 64 bytes)
     * @return Device address, or 0 if allocation failed
     */
    kpu::Address malloc(kpu::Size size, kpu::Size alignment = 64);

    /**
     * @brief Allocate device memory in a specific memory pool
     * @param size Size in bytes
     * @param pool Memory pool type
     * @param alignment Alignment requirement
     * @return Device address, or 0 if allocation failed
     */
    kpu::Address malloc_pool(kpu::Size size, kpu::ResourceType pool,
                              kpu::Size alignment = 64);

    /**
     * @brief Free device memory
     * @param ptr Device address to free
     */
    void free(kpu::Address ptr);

    /**
     * @brief Copy data from host to device
     * @param dst Device destination address
     * @param src Host source pointer
     * @param size Number of bytes to copy
     */
    void memcpy_h2d(kpu::Address dst, const void* src, kpu::Size size);

    /**
     * @brief Copy data from device to host
     * @param dst Host destination pointer
     * @param src Device source address
     * @param size Number of bytes to copy
     */
    void memcpy_d2h(void* dst, kpu::Address src, kpu::Size size);

    /**
     * @brief Copy data within device memory
     * @param dst Device destination address
     * @param src Device source address
     * @param size Number of bytes to copy
     */
    void memcpy_d2d(kpu::Address dst, kpu::Address src, kpu::Size size);

    /**
     * @brief Generic memory copy
     * @param dst Destination (host pointer or device address)
     * @param src Source (host pointer or device address)
     * @param size Number of bytes to copy
     * @param kind Copy direction
     */
    void memcpy(void* dst, const void* src, kpu::Size size, MemcpyKind kind);

    /**
     * @brief Set device memory to a value
     * @param ptr Device address
     * @param value Byte value to set
     * @param size Number of bytes
     */
    void memset(kpu::Address ptr, uint8_t value, kpu::Size size);

    // =========================================
    // Kernel Execution
    // =========================================

    /**
     * @brief Launch a kernel synchronously
     * @param kernel The kernel to execute
     * @param args Device addresses for kernel arguments
     * @return Launch result with cycle count
     *
     * Arguments must be provided in the order specified by kernel.arguments():
     * - For matmul: {A, B, C}
     * - For MLP with bias: {A, B, bias, C}
     */
    LaunchResult launch(const kpu::Kernel& kernel,
                        const std::vector<kpu::Address>& args);

    /**
     * @brief Launch a kernel asynchronously on a stream
     * @param kernel The kernel to execute
     * @param args Device addresses for kernel arguments
     * @param stream Stream for async execution
     *
     * The kernel will be queued on the stream and executed when
     * previous operations on that stream complete.
     */
    void launch_async(const kpu::Kernel& kernel,
                      const std::vector<kpu::Address>& args,
                      Stream stream);

    // =========================================
    // Synchronization
    // =========================================

    /**
     * @brief Wait for all operations to complete
     *
     * Blocks until all kernels and memory operations finish.
     */
    void synchronize();

    /**
     * @brief Wait for all operations on a stream to complete
     * @param stream The stream to synchronize
     */
    void stream_synchronize(Stream stream);

    // =========================================
    // Streams
    // =========================================

    /**
     * @brief Create a new execution stream
     * @return Stream handle
     */
    Stream create_stream();

    /**
     * @brief Destroy an execution stream
     * @param stream Stream to destroy
     */
    void destroy_stream(Stream stream);

    /**
     * @brief Get the default stream (stream 0)
     * @return Default stream
     */
    Stream default_stream() const { return Stream(0); }

    // =========================================
    // Events
    // =========================================

    /**
     * @brief Create a new event
     * @return Event handle
     */
    Event create_event();

    /**
     * @brief Destroy an event
     * @param event Event to destroy
     */
    void destroy_event(Event event);

    /**
     * @brief Record an event on a stream
     * @param event Event to record
     * @param stream Stream where event is recorded
     *
     * The event captures the completion of all prior operations on the stream.
     */
    void record_event(Event event, Stream stream);

    /**
     * @brief Wait for an event to complete
     * @param event Event to wait for
     */
    void wait_event(Event event);

    /**
     * @brief Calculate elapsed time between two events
     * @param start Start event
     * @param end End event
     * @return Elapsed time in milliseconds
     */
    float elapsed_time(Event start, Event end) const;

    // =========================================
    // Device Information
    // =========================================

    /**
     * @brief Get total device memory
     * @return Total memory in bytes
     */
    kpu::Size get_total_memory() const;

    /**
     * @brief Get available device memory
     * @return Free memory in bytes
     */
    kpu::Size get_free_memory() const;

    /**
     * @brief Get total cycles executed
     * @return Total simulated cycles
     */
    kpu::Cycle get_total_cycles() const { return total_cycles_; }

    /**
     * @brief Get total kernels launched
     * @return Number of kernel launches
     */
    size_t get_launch_count() const { return launch_count_; }

    /**
     * @brief Get the underlying simulator
     */
    kpu::KPUSimulator* simulator() const { return simulator_; }

    /**
     * @brief Get the resource manager
     */
    kpu::ResourceManager* resource_manager() const { return resource_manager_.get(); }

    /**
     * @brief Get runtime configuration
     */
    const Config& config() const { return config_; }

    /**
     * @brief Print runtime statistics
     */
    void print_stats() const;

private:
    kpu::KPUSimulator* simulator_;
    std::unique_ptr<kpu::ResourceManager> resource_manager_;
    std::unique_ptr<kpu::isa::ConcurrentExecutor> executor_;
    Config config_;

    // Stream management
    struct StreamState {
        std::queue<std::function<void()>> pending_ops;
        kpu::Cycle last_cycle = 0;
        bool valid = true;
    };
    std::unordered_map<size_t, StreamState> streams_;
    size_t next_stream_id_ = 1;  // 0 is default stream

    // Event management
    struct EventState {
        kpu::Cycle recorded_cycle = 0;
        bool recorded = false;
        bool valid = true;
    };
    std::unordered_map<size_t, EventState> events_;
    size_t next_event_id_ = 1;

    // Statistics
    kpu::Cycle total_cycles_ = 0;
    size_t launch_count_ = 0;

    // Helper methods
    void validate_address(kpu::Address addr, const std::string& context) const;
    void execute_stream_operations(size_t stream_id);
};

} // namespace sw::runtime
