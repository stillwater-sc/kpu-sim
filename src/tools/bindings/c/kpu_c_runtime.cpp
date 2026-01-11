/**
 * @file kpu_c_runtime.cpp
 * @brief Implementation of C API runtime bindings
 */

#include <sw/kpu/kpu_c_runtime.h>
#include <sw/runtime/runtime.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include <sw/kpu/kernel.hpp>

#include <cstring>
#include <exception>

namespace {

// Cast helpers
inline sw::runtime::KPURuntime* to_runtime(KPURuntimeHandle h) {
    return reinterpret_cast<sw::runtime::KPURuntime*>(h);
}

inline KPURuntimeHandle from_runtime(sw::runtime::KPURuntime* r) {
    return reinterpret_cast<KPURuntimeHandle>(r);
}

inline sw::kpu::KPUSimulator* to_simulator(KPUHandle h) {
    return reinterpret_cast<sw::kpu::KPUSimulator*>(h);
}

inline sw::kpu::Kernel* to_kernel(KPUKernelHandle h) {
    return reinterpret_cast<sw::kpu::Kernel*>(h);
}

// Stream/Event handle wrappers
// We use the id directly as the handle value
inline sw::runtime::Stream to_stream(KPUStreamHandle h) {
    if (!h) return sw::runtime::Stream();
    return sw::runtime::Stream(reinterpret_cast<size_t>(h));
}

inline KPUStreamHandle from_stream(const sw::runtime::Stream& s) {
    if (!s.valid) return nullptr;
    return reinterpret_cast<KPUStreamHandle>(s.id);
}

inline sw::runtime::Event to_event(KPUEventHandle h) {
    if (!h) return sw::runtime::Event();
    return sw::runtime::Event(reinterpret_cast<size_t>(h));
}

inline KPUEventHandle from_event(const sw::runtime::Event& e) {
    if (!e.valid) return nullptr;
    return reinterpret_cast<KPUEventHandle>(e.id);
}

// Exception to error code translation
KPUError translate_exception() {
    try {
        throw;
    } catch (const std::out_of_range&) {
        return KPU_ERROR_OUT_OF_BOUNDS;
    } catch (const std::invalid_argument&) {
        return KPU_ERROR_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        return KPU_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return KPU_ERROR_UNKNOWN;
    }
}

} // anonymous namespace

extern "C" {

/* ============================================================================
 * Runtime Lifecycle
 * ============================================================================ */

KPURuntimeHandle kpu_runtime_create(KPUHandle simulator,
                                     const KPURuntimeConfig* config) {
    if (!simulator) return nullptr;

    try {
        auto* sim = to_simulator(simulator);

        sw::runtime::KPURuntime::Config rt_config;
        if (config) {
            rt_config.clock_ghz = config->clock_ghz;
            rt_config.verbose = config->verbose;
        }

        auto* runtime = new sw::runtime::KPURuntime(sim, rt_config);
        return from_runtime(runtime);
    } catch (...) {
        return nullptr;
    }
}

void kpu_runtime_destroy(KPURuntimeHandle runtime) {
    if (runtime) {
        delete to_runtime(runtime);
    }
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

KPUAddress kpu_runtime_malloc(KPURuntimeHandle runtime,
                               KPUSize size,
                               KPUSize alignment) {
    if (!runtime) return 0;

    try {
        return to_runtime(runtime)->malloc(size, alignment ? alignment : 64);
    } catch (...) {
        return 0;
    }
}

void kpu_runtime_free(KPURuntimeHandle runtime, KPUAddress ptr) {
    if (!runtime || ptr == 0) return;

    try {
        to_runtime(runtime)->free(ptr);
    } catch (...) {
        // Silently ignore errors on free
    }
}

KPUError kpu_runtime_memcpy_h2d(KPURuntimeHandle runtime,
                                 KPUAddress dst,
                                 const void* src,
                                 KPUSize size) {
    if (!runtime) return KPU_ERROR_INVALID_HANDLE;
    if (!src) return KPU_ERROR_NULL_POINTER;

    try {
        to_runtime(runtime)->memcpy_h2d(dst, src, size);
        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

KPUError kpu_runtime_memcpy_d2h(KPURuntimeHandle runtime,
                                 void* dst,
                                 KPUAddress src,
                                 KPUSize size) {
    if (!runtime) return KPU_ERROR_INVALID_HANDLE;
    if (!dst) return KPU_ERROR_NULL_POINTER;

    try {
        to_runtime(runtime)->memcpy_d2h(dst, src, size);
        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

KPUError kpu_runtime_memcpy_d2d(KPURuntimeHandle runtime,
                                 KPUAddress dst,
                                 KPUAddress src,
                                 KPUSize size) {
    if (!runtime) return KPU_ERROR_INVALID_HANDLE;

    try {
        to_runtime(runtime)->memcpy_d2d(dst, src, size);
        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

KPUError kpu_runtime_memset(KPURuntimeHandle runtime,
                             KPUAddress ptr,
                             uint8_t value,
                             KPUSize size) {
    if (!runtime) return KPU_ERROR_INVALID_HANDLE;

    try {
        to_runtime(runtime)->memset(ptr, value, size);
        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

/* ============================================================================
 * Kernel Execution
 * ============================================================================ */

KPUError kpu_runtime_launch(KPURuntimeHandle runtime,
                             KPUKernelHandle kernel,
                             const KPUAddress* args,
                             uint32_t num_args,
                             KPULaunchResult* result) {
    if (!runtime) return KPU_ERROR_INVALID_HANDLE;
    if (!kernel) return KPU_ERROR_KERNEL_INVALID;
    if (!args && num_args > 0) return KPU_ERROR_NULL_POINTER;

    try {
        // Convert args to vector
        std::vector<sw::kpu::Address> arg_vec(args, args + num_args);

        // Launch kernel
        sw::runtime::LaunchResult cpp_result =
            to_runtime(runtime)->launch(*to_kernel(kernel), arg_vec);

        // Fill result if provided
        if (result) {
            result->success = cpp_result.success;
            result->cycles = cpp_result.cycles;
            if (!cpp_result.error.empty()) {
                std::strncpy(result->error, cpp_result.error.c_str(),
                            sizeof(result->error) - 1);
                result->error[sizeof(result->error) - 1] = '\0';
            } else {
                result->error[0] = '\0';
            }
        }

        return cpp_result.success ? KPU_SUCCESS : KPU_ERROR_LAUNCH_FAILED;
    } catch (...) {
        return translate_exception();
    }
}

void kpu_runtime_synchronize(KPURuntimeHandle runtime) {
    if (!runtime) return;

    try {
        to_runtime(runtime)->synchronize();
    } catch (...) {
        // Silently ignore errors
    }
}

/* ============================================================================
 * Stream Management
 * ============================================================================ */

KPUStreamHandle kpu_runtime_create_stream(KPURuntimeHandle runtime) {
    if (!runtime) return nullptr;

    try {
        sw::runtime::Stream stream = to_runtime(runtime)->create_stream();
        return from_stream(stream);
    } catch (...) {
        return nullptr;
    }
}

void kpu_runtime_destroy_stream(KPURuntimeHandle runtime, KPUStreamHandle stream) {
    if (!runtime || !stream) return;

    try {
        to_runtime(runtime)->destroy_stream(to_stream(stream));
    } catch (...) {
        // Silently ignore errors
    }
}

void kpu_runtime_stream_synchronize(KPURuntimeHandle runtime, KPUStreamHandle stream) {
    if (!runtime) return;

    try {
        to_runtime(runtime)->stream_synchronize(to_stream(stream));
    } catch (...) {
        // Silently ignore errors
    }
}

KPUStreamHandle kpu_runtime_default_stream(KPURuntimeHandle runtime) {
    if (!runtime) return nullptr;

    try {
        sw::runtime::Stream stream = to_runtime(runtime)->default_stream();
        return from_stream(stream);
    } catch (...) {
        return nullptr;
    }
}

KPUError kpu_runtime_launch_async(KPURuntimeHandle runtime,
                                   KPUKernelHandle kernel,
                                   const KPUAddress* args,
                                   uint32_t num_args,
                                   KPUStreamHandle stream) {
    if (!runtime) return KPU_ERROR_INVALID_HANDLE;
    if (!kernel) return KPU_ERROR_KERNEL_INVALID;
    if (!args && num_args > 0) return KPU_ERROR_NULL_POINTER;

    try {
        std::vector<sw::kpu::Address> arg_vec(args, args + num_args);
        to_runtime(runtime)->launch_async(*to_kernel(kernel), arg_vec, to_stream(stream));
        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

/* ============================================================================
 * Event Management
 * ============================================================================ */

KPUEventHandle kpu_runtime_create_event(KPURuntimeHandle runtime) {
    if (!runtime) return nullptr;

    try {
        sw::runtime::Event event = to_runtime(runtime)->create_event();
        return from_event(event);
    } catch (...) {
        return nullptr;
    }
}

void kpu_runtime_destroy_event(KPURuntimeHandle runtime, KPUEventHandle event) {
    if (!runtime || !event) return;

    try {
        to_runtime(runtime)->destroy_event(to_event(event));
    } catch (...) {
        // Silently ignore errors
    }
}

KPUError kpu_runtime_record_event(KPURuntimeHandle runtime,
                                   KPUEventHandle event,
                                   KPUStreamHandle stream) {
    if (!runtime) return KPU_ERROR_INVALID_HANDLE;
    if (!event) return KPU_ERROR_EVENT_NOT_FOUND;

    try {
        to_runtime(runtime)->record_event(to_event(event), to_stream(stream));
        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

void kpu_runtime_wait_event(KPURuntimeHandle runtime, KPUEventHandle event) {
    if (!runtime || !event) return;

    try {
        to_runtime(runtime)->wait_event(to_event(event));
    } catch (...) {
        // Silently ignore errors
    }
}

float kpu_runtime_elapsed_time(KPURuntimeHandle runtime,
                                KPUEventHandle start,
                                KPUEventHandle end) {
    if (!runtime || !start || !end) return 0.0f;

    try {
        return to_runtime(runtime)->elapsed_time(to_event(start), to_event(end));
    } catch (...) {
        return 0.0f;
    }
}

/* ============================================================================
 * Device Information
 * ============================================================================ */

KPUSize kpu_runtime_get_total_memory(KPURuntimeHandle runtime) {
    if (!runtime) return 0;

    try {
        return to_runtime(runtime)->get_total_memory();
    } catch (...) {
        return 0;
    }
}

KPUSize kpu_runtime_get_free_memory(KPURuntimeHandle runtime) {
    if (!runtime) return 0;

    try {
        return to_runtime(runtime)->get_free_memory();
    } catch (...) {
        return 0;
    }
}

KPUCycle kpu_runtime_get_total_cycles(KPURuntimeHandle runtime) {
    if (!runtime) return 0;

    try {
        return to_runtime(runtime)->get_total_cycles();
    } catch (...) {
        return 0;
    }
}

uint64_t kpu_runtime_get_launch_count(KPURuntimeHandle runtime) {
    if (!runtime) return 0;

    try {
        return to_runtime(runtime)->get_launch_count();
    } catch (...) {
        return 0;
    }
}

void kpu_runtime_print_stats(KPURuntimeHandle runtime) {
    if (!runtime) return;

    try {
        to_runtime(runtime)->print_stats();
    } catch (...) {
        // Silently ignore errors
    }
}

} // extern "C"
