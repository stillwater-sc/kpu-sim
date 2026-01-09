/**
 * @file runtime_demo.c
 * @brief KPU Runtime Library Demo (C API)
 *
 * This example mirrors examples/runtime/runtime_demo.cpp and demonstrates:
 * 1. Runtime creation with simulator
 * 2. Memory allocation (malloc/free)
 * 3. Data transfer (memcpy_h2d, memcpy_d2h)
 * 4. Kernel launch (low-level API)
 * 5. GraphExecutor (high-level API)
 * 6. Streams and events for timing
 *
 * Build and run:
 *   cmake --build --preset release
 *   ./build/examples/c/runtime_demo_c
 */

#include <sw/kpu/kpu_c_types.h>
#include <sw/kpu/kpu_c_api.h>
#include <sw/kpu/kpu_c_runtime.h>
#include <sw/kpu/kpu_c_kernel.h>
#include <sw/kpu/kpu_c_executor.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

/* Helper: print separator */
static void separator(const char* title) {
    if (title && title[0]) {
        printf("\n=== %s ", title);
        int len = 65 - (int)strlen(title);
        for (int i = 0; i < len; i++) putchar('=');
        printf("\n\n");
    } else {
        for (int i = 0; i < 70; i++) putchar('-');
        printf("\n");
    }
}

/* Helper: format bytes */
static const char* format_bytes(KPUSize bytes, char* buf, size_t buf_size) {
    if (bytes >= 1024 * 1024) {
        snprintf(buf, buf_size, "%" PRIu64 " MB", bytes / (1024 * 1024));
    } else if (bytes >= 1024) {
        snprintf(buf, buf_size, "%" PRIu64 " KB", bytes / 1024);
    } else {
        snprintf(buf, buf_size, "%" PRIu64 " B", bytes);
    }
    return buf;
}

/* Helper: fill array with random values */
static void fill_random(float* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

int main(void) {
    char buf[64];
    srand(42);  /* Fixed seed for reproducibility */

    printf("KPU Simulator - Runtime Library Demo (C API)\n");
    separator(NULL);

    /* =========================================================
     * 1. Runtime Creation
     * ========================================================= */
    separator("1. Runtime Creation");

    printf("Creating KPU simulator and runtime...\n\n");

    /* Create simulator with default sizes */
    KPUHandle sim = kpu_create(0, 0);
    if (!sim) {
        fprintf(stderr, "Failed to create simulator!\n");
        return 1;
    }

    /* Create runtime */
    KPURuntimeConfig rt_config;
    kpu_runtime_config_default(&rt_config);
    rt_config.clock_ghz = 1.0;

    KPURuntimeHandle runtime = kpu_runtime_create(sim, &rt_config);
    if (!runtime) {
        fprintf(stderr, "Failed to create runtime!\n");
        kpu_destroy(sim);
        return 1;
    }

    printf("  Simulator: created successfully\n");
    printf("  Runtime:   Clock = %.1f GHz\n", rt_config.clock_ghz);
    printf("  Memory:    Total = %s, Free = %s\n",
           format_bytes(kpu_runtime_get_total_memory(runtime), buf, sizeof(buf)),
           format_bytes(kpu_runtime_get_free_memory(runtime), buf, sizeof(buf)));

    /* =========================================================
     * 2. Memory Allocation
     * ========================================================= */
    separator("2. Memory Allocation");

    const KPUSize M = 64, N = 64, K = 64;
    const KPUSize elem_size = sizeof(float);
    const KPUSize A_bytes = M * K * elem_size;
    const KPUSize B_bytes = K * N * elem_size;
    const KPUSize C_bytes = M * N * elem_size;

    printf("Allocating device memory for %" PRIu64 "x%" PRIu64 " x %" PRIu64 "x%" PRIu64 " matmul...\n\n", M, K, K, N);

    KPUAddress A_dev = kpu_runtime_malloc(runtime, A_bytes, 0);
    KPUAddress B_dev = kpu_runtime_malloc(runtime, B_bytes, 0);
    KPUAddress C_dev = kpu_runtime_malloc(runtime, C_bytes, 0);

    printf("  A: %s @ 0x%" PRIx64 "\n", format_bytes(A_bytes, buf, sizeof(buf)), A_dev);
    printf("  B: %s @ 0x%" PRIx64 "\n", format_bytes(B_bytes, buf, sizeof(buf)), B_dev);
    printf("  C: %s @ 0x%" PRIx64 "\n", format_bytes(C_bytes, buf, sizeof(buf)), C_dev);
    printf("\n  Memory after allocation:\n");
    printf("    Free: %s\n", format_bytes(kpu_runtime_get_free_memory(runtime), buf, sizeof(buf)));

    /* =========================================================
     * 3. Data Transfer
     * ========================================================= */
    separator("3. Data Transfer");

    printf("Initializing host data and transferring to device...\n\n");

    float* A_host = (float*)malloc(A_bytes);
    float* B_host = (float*)malloc(B_bytes);
    float* C_host = (float*)malloc(C_bytes);

    fill_random(A_host, M * K);
    fill_random(B_host, K * N);
    memset(C_host, 0, C_bytes);

    kpu_runtime_memcpy_h2d(runtime, A_dev, A_host, A_bytes);
    kpu_runtime_memcpy_h2d(runtime, B_dev, B_host, B_bytes);
    kpu_runtime_memset(runtime, C_dev, 0, C_bytes);

    printf("  H2D: Copied A (%s)\n", format_bytes(A_bytes, buf, sizeof(buf)));
    printf("  H2D: Copied B (%s)\n", format_bytes(B_bytes, buf, sizeof(buf)));
    printf("  Memset: Cleared C\n");

    /* Test D2D copy */
    KPUAddress C_copy_dev = kpu_runtime_malloc(runtime, C_bytes, 0);
    kpu_runtime_memcpy_d2d(runtime, C_copy_dev, C_dev, C_bytes);
    printf("  D2D: Copied C to C_copy\n");
    kpu_runtime_free(runtime, C_copy_dev);

    /* =========================================================
     * 4. Kernel Launch (Low-Level API)
     * ========================================================= */
    separator("4. Kernel Launch (Low-Level API)");

    printf("Creating and launching matmul kernel...\n\n");

    KPUKernelHandle kernel = kpu_kernel_create_matmul(M, N, K, KPU_DTYPE_FLOAT32);
    if (!kernel) {
        fprintf(stderr, "Failed to create kernel!\n");
        goto cleanup;
    }

    printf("  Kernel: %s (%" PRIu64 "x%" PRIu64 "x%" PRIu64 ")\n",
           kpu_op_type_name(kpu_kernel_get_op_type(kernel)), M, N, K);
    printf("  Tiles:  Ti=%" PRIu64 ", Tj=%" PRIu64 ", Tk=%" PRIu64 "\n",
           kpu_kernel_get_Ti(kernel),
           kpu_kernel_get_Tj(kernel),
           kpu_kernel_get_Tk(kernel));

    KPUAddress args[3] = {A_dev, B_dev, C_dev};
    KPULaunchResult result;

    KPUError err = kpu_runtime_launch(runtime, kernel, args, 3, &result);
    if (err == KPU_SUCCESS && result.success) {
        printf("\n  Launch successful!\n");
        printf("  Cycles:     %" PRIu64 "\n", result.cycles);
        double time_ms = (double)result.cycles / (rt_config.clock_ghz * 1e6);
        printf("  Time (ms):  %.4f\n", time_ms);
    } else {
        printf("  Launch failed: %s\n", result.error);
    }

    /* Copy result back */
    kpu_runtime_memcpy_d2h(runtime, C_host, C_dev, C_bytes);

    printf("\n  Runtime Statistics:\n");
    printf("  Total Launches: %" PRIu64 "\n", kpu_runtime_get_launch_count(runtime));
    printf("  Total Cycles:   %" PRIu64 "\n", kpu_runtime_get_total_cycles(runtime));

    /* =========================================================
     * 5. GraphExecutor (High-Level API)
     * ========================================================= */
    separator("5. GraphExecutor (High-Level API)");

    printf("Using GraphExecutor for automatic memory management...\n\n");

    KPUExecutorHandle executor = kpu_executor_create(runtime);
    if (!executor) {
        fprintf(stderr, "Failed to create executor!\n");
        goto cleanup;
    }

    /* Create a larger matmul */
    const KPUSize M2 = 128, N2 = 128, K2 = 128;
    err = kpu_executor_create_matmul(executor, M2, N2, K2, KPU_DTYPE_FLOAT32);
    if (err != KPU_SUCCESS) {
        fprintf(stderr, "Failed to create matmul: %s\n", kpu_error_string(err));
        kpu_executor_destroy(executor);
        goto cleanup;
    }

    printf("  Created matmul: %" PRIu64 "x%" PRIu64 "x%" PRIu64 "\n", M2, N2, K2);

    /* Check tensor bindings */
    printf("\n  Tensor Bindings:\n");
    const char* binding_names[] = {"A", "B", "C"};
    for (int i = 0; i < 3; i++) {
        KPUTensorInfo info;
        if (kpu_executor_get_binding_info(executor, binding_names[i], &info) == KPU_SUCCESS) {
            printf("    %s: ", info.name);
            for (uint32_t d = 0; d < info.ndims; d++) {
                printf("%" PRIu64, info.shape[d]);
                if (d < info.ndims - 1) printf("x");
            }
            printf(" @ 0x%" PRIx64 "\n", info.device_address);
        }
    }

    /* Prepare input data */
    float* A2 = (float*)malloc(M2 * K2 * sizeof(float));
    float* B2 = (float*)malloc(K2 * N2 * sizeof(float));
    float* C2 = (float*)malloc(M2 * N2 * sizeof(float));

    fill_random(A2, M2 * K2);
    fill_random(B2, K2 * N2);

    /* Set inputs */
    KPUSize shape_A[2] = {M2, K2};
    KPUSize shape_B[2] = {K2, N2};

    kpu_executor_set_input(executor, "A", A2, shape_A, 2);
    kpu_executor_set_input(executor, "B", B2, shape_B, 2);
    printf("\n  Inputs set via set_input()\n");

    /* Execute */
    KPUExecutionResult exec_result;
    err = kpu_executor_execute(executor, &exec_result);

    if (exec_result.success) {
        printf("\n  Execution successful!\n");
        printf("  Cycles:    %" PRIu64 "\n", exec_result.cycles);
        printf("  Time (ms): %.4f\n", exec_result.time_ms);
    } else {
        printf("  Execution failed: %s\n", exec_result.error);
    }

    /* Get output */
    kpu_executor_get_output(executor, "C", C2);
    printf("  Output retrieved via get_output()\n");

    /* =========================================================
     * 6. Streams and Events
     * ========================================================= */
    separator("6. Streams and Events");

    printf("Demonstrating streams and events for async execution...\n\n");

    KPUStreamHandle stream = kpu_runtime_create_stream(runtime);
    KPUEventHandle start = kpu_runtime_create_event(runtime);
    KPUEventHandle end = kpu_runtime_create_event(runtime);

    printf("  Created stream and events\n");

    /* Record start, launch, record end */
    kpu_runtime_record_event(runtime, start, stream);
    kpu_runtime_launch_async(runtime, kernel, args, 3, stream);
    kpu_runtime_record_event(runtime, end, stream);

    /* Wait for completion */
    kpu_runtime_stream_synchronize(runtime, stream);
    printf("\n  Stream synchronized.\n");

    /* Get elapsed time */
    float elapsed = kpu_runtime_elapsed_time(runtime, start, end);
    printf("  Elapsed time (events): %.4f ms\n", elapsed);

    /* Cleanup events and stream */
    kpu_runtime_destroy_event(runtime, start);
    kpu_runtime_destroy_event(runtime, end);
    kpu_runtime_destroy_stream(runtime, stream);

    /* =========================================================
     * Final Statistics
     * ========================================================= */
    separator("Final Statistics");

    printf("Runtime Summary:\n");
    printf("  Total Kernel Launches: %" PRIu64 "\n", kpu_runtime_get_launch_count(runtime));
    printf("  Total Simulated Cycles: %" PRIu64 "\n", kpu_runtime_get_total_cycles(runtime));

    /* =========================================================
     * Cleanup
     * ========================================================= */
cleanup:
    separator("Cleanup");

    printf("Freeing resources...\n\n");

    if (executor) {
        kpu_executor_release(executor);
        kpu_executor_destroy(executor);
        printf("  Executor released and destroyed\n");
    }

    free(A2);
    free(B2);
    free(C2);

    kpu_runtime_free(runtime, A_dev);
    kpu_runtime_free(runtime, B_dev);
    kpu_runtime_free(runtime, C_dev);
    printf("  Device memory freed\n");

    if (kernel) {
        kpu_kernel_destroy(kernel);
        printf("  Kernel destroyed\n");
    }

    free(A_host);
    free(B_host);
    free(C_host);
    printf("  Host memory freed\n");

    printf("\n  Memory after free: %s\n",
           format_bytes(kpu_runtime_get_free_memory(runtime), buf, sizeof(buf)));

    kpu_runtime_destroy(runtime);
    kpu_destroy(sim);
    printf("  Runtime and simulator destroyed\n");

    separator(NULL);
    printf("\nRuntime library demo complete!\n");

    return 0;
}
