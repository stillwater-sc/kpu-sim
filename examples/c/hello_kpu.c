/**
 * @file hello_kpu.c
 * @brief Hello KPU - First KPU Program in C
 *
 * This example demonstrates the basics of the KPU C API:
 * - Creating a simulator with configuration
 * - Creating a runtime
 * - Allocating device memory
 * - Copying data between host and device
 * - Creating and launching a simple kernel
 * - Cleaning up resources
 *
 * The runtime is timing only today: a launch reports the kernel's cycles but
 * computes no values, so C is never written (#257 makes the host API return
 * simulator-computed values). This example checks what the runtime does do,
 * a host -> device -> host round trip, and exits non-zero if any step fails.
 *
 * Build and run:
 *   cmake --build --preset release
 *   ./build/examples/c/hello_kpu
 */

#include <sw/kpu/kpu_c_types.h>
#include <sw/kpu/kpu_c_api.h>
#include <sw/kpu/kpu_c_runtime.h>
#include <sw/kpu/kpu_c_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

int main(void) {
    printf("===========================================\n");
    printf(" Hello KPU - First KPU Program (C API)\n");
    printf("===========================================\n\n");

    /* =========================================================
     * 1. Create KPU Simulator
     * ========================================================= */
    printf("1. Creating KPU Simulator\n");
    printf("   -----------------------\n");

    /* Use default memory sizes */
    KPUHandle kpu = kpu_create(0, 0);  /* 0 = use defaults */
    if (!kpu) {
        fprintf(stderr, "Failed to create KPU simulator!\n");
        return 1;
    }

    printf("   Main memory size: %" PRIu64 " bytes (%.1f MB)\n",
           kpu_main_memory_size(kpu),
           (double)kpu_main_memory_size(kpu) / (1024.0 * 1024.0));
    printf("   Scratchpad size:  %" PRIu64 " bytes (%.1f KB)\n",
           kpu_scratchpad_size(kpu),
           (double)kpu_scratchpad_size(kpu) / 1024.0);

    /* =========================================================
     * 2. Create Runtime
     * ========================================================= */
    printf("\n2. Creating Runtime\n");
    printf("   -----------------\n");

    KPURuntimeConfig rt_config;
    kpu_runtime_config_default(&rt_config);
    rt_config.clock_ghz = 1.0;  /* 1 GHz for easy cycle-to-time conversion */

    KPURuntimeHandle runtime = kpu_runtime_create(kpu, &rt_config);
    if (!runtime) {
        fprintf(stderr, "Failed to create runtime!\n");
        kpu_destroy(kpu);
        return 1;
    }

    printf("   Clock frequency: %.1f GHz\n", rt_config.clock_ghz);
    printf("   Total memory:    %zu bytes\n", kpu_runtime_get_total_memory(runtime));
    printf("   Free memory:     %zu bytes\n", kpu_runtime_get_free_memory(runtime));

    /* =========================================================
     * 3. Allocate Device Memory
     * ========================================================= */
    printf("\n3. Allocating Device Memory\n");
    printf("   -------------------------\n");

    const KPUSize M = 32, N = 32, K = 32;
    const KPUSize elem_size = sizeof(float);
    const KPUSize A_bytes = M * K * elem_size;
    const KPUSize B_bytes = K * N * elem_size;
    const KPUSize C_bytes = M * N * elem_size;

    printf("   Matrix sizes: A[%zu x %zu], B[%zu x %zu], C[%zu x %zu]\n",
           M, K, K, N, M, N);

    KPUAddress A_dev = kpu_runtime_malloc(runtime, A_bytes, 0);
    KPUAddress B_dev = kpu_runtime_malloc(runtime, B_bytes, 0);
    KPUAddress C_dev = kpu_runtime_malloc(runtime, C_bytes, 0);

    if (A_dev == 0 || B_dev == 0 || C_dev == 0) {
        fprintf(stderr, "Failed to allocate device memory!\n");
        kpu_runtime_destroy(runtime);
        kpu_destroy(kpu);
        return 1;
    }

    printf("   A: %zu bytes @ 0x%" PRIx64 "\n", A_bytes, A_dev);
    printf("   B: %zu bytes @ 0x%" PRIx64 "\n", B_bytes, B_dev);
    printf("   C: %zu bytes @ 0x%" PRIx64 "\n", C_bytes, C_dev);
    printf("   Free after alloc: %zu bytes\n", kpu_runtime_get_free_memory(runtime));

    /* =========================================================
     * 4. Initialize and Copy Data
     * ========================================================= */
    printf("\n4. Initializing Data\n");
    printf("   ------------------\n");

    /* Allocate and initialize host arrays */
    float* A_host = (float*)malloc(A_bytes);
    float* B_host = (float*)malloc(B_bytes);
    float* C_host = (float*)malloc(C_bytes);

    if (!A_host || !B_host || !C_host) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        kpu_runtime_free(runtime, A_dev);
        kpu_runtime_free(runtime, B_dev);
        kpu_runtime_free(runtime, C_dev);
        kpu_runtime_destroy(runtime);
        kpu_destroy(kpu);
        return 1;
    }

    /* Fill A and B with ones, C with zeros */
    for (KPUSize i = 0; i < M * K; i++) A_host[i] = 1.0f;
    for (KPUSize i = 0; i < K * N; i++) B_host[i] = 1.0f;
    memset(C_host, 0, C_bytes);

    printf("   A initialized to all 1.0\n");
    printf("   B initialized to all 1.0\n");
    printf("   C initialized to all 0.0\n");

    /* Copy to device. Any failed step makes the example exit non-zero. */
    int status = 0;
    KPUError err = kpu_runtime_memcpy_h2d(runtime, A_dev, A_host, A_bytes);
    if (err != KPU_SUCCESS) {
        fprintf(stderr, "Failed to copy A: %s\n", kpu_error_string(err));
        status = 1;
    }

    err = kpu_runtime_memcpy_h2d(runtime, B_dev, B_host, B_bytes);
    if (err != KPU_SUCCESS) {
        fprintf(stderr, "Failed to copy B: %s\n", kpu_error_string(err));
        status = 1;
    }

    printf("   Data copied to device\n");

    /* =========================================================
     * 5. Create and Launch Kernel
     * ========================================================= */
    printf("\n5. Creating and Launching Kernel\n");
    printf("   ------------------------------\n");

    KPUKernelHandle kernel = kpu_kernel_create_matmul(M, N, K, KPU_DTYPE_FLOAT32);
    if (!kernel) {
        fprintf(stderr, "Failed to create kernel!\n");
        free(A_host); free(B_host); free(C_host);
        kpu_runtime_free(runtime, A_dev);
        kpu_runtime_free(runtime, B_dev);
        kpu_runtime_free(runtime, C_dev);
        kpu_runtime_destroy(runtime);
        kpu_destroy(kpu);
        return 1;
    }

    printf("   Kernel: %s\n", kpu_op_type_name(kpu_kernel_get_op_type(kernel)));
    printf("   Dimensions: M=%zu, N=%zu, K=%zu\n",
           kpu_kernel_get_M(kernel),
           kpu_kernel_get_N(kernel),
           kpu_kernel_get_K(kernel));
    printf("   Tile sizes: Ti=%zu, Tj=%zu, Tk=%zu\n",
           kpu_kernel_get_Ti(kernel),
           kpu_kernel_get_Tj(kernel),
           kpu_kernel_get_Tk(kernel));
    printf("   FLOPs: %zu\n", kpu_kernel_get_total_flops(kernel));
    printf("   Instructions: %zu\n", kpu_kernel_get_instruction_count(kernel));

    /* Launch kernel */
    KPUAddress args[3] = {A_dev, B_dev, C_dev};
    KPULaunchResult result;

    err = kpu_runtime_launch(runtime, kernel, args, 3, &result);
    if (err == KPU_SUCCESS && result.success) {
        printf("\n   Launch successful!\n");
        printf("   Cycles: %" PRIu64 "\n", result.cycles);
        printf("   Time:   %.4f ms (at %.1f GHz)\n",
               (double)result.cycles / (rt_config.clock_ghz * 1e6),
               rt_config.clock_ghz);
    } else {
        fprintf(stderr, "   Launch failed: %s\n", result.error);
        status = 1;
    }

    /* =========================================================
     * 6. Round Trip
     * ========================================================= */
    printf("\n6. Checking the Host <-> Device Round Trip\n");
    printf("   ----------------------------------------\n");
    printf("   The runtime is timing only: the launch computed no values,\n");
    printf("   so C is not checked (#257).\n");

    /* Read A back and compare it with what was written */
    float* A_back = (float*)malloc(A_bytes);
    err = A_back ? kpu_runtime_memcpy_d2h(runtime, A_back, A_dev, A_bytes) : KPU_SUCCESS;
    if (!A_back) {
        fprintf(stderr, "Failed to allocate host memory for the read-back\n");
        status = 1;
    } else if (err != KPU_SUCCESS) {
        fprintf(stderr, "Failed to copy A back: %s\n", kpu_error_string(err));
        status = 1;
    } else {
        int errors = 0;
        for (KPUSize i = 0; i < M * K; i++) {
            if (A_back[i] != A_host[i]) {
                if (errors < 5) {
                    printf("   Mismatch at A[%zu]: wrote %.1f, read %.1f\n",
                           i, A_host[i], A_back[i]);
                }
                errors++;
            }
        }
        if (errors == 0) {
            printf("   A read back unchanged: all %zu elements\n", M * K);
        } else {
            printf("   A read back with %d mismatches\n", errors);
            status = 1;
        }
    }
    free(A_back);

    /* =========================================================
     * 7. Statistics and Cleanup
     * ========================================================= */
    printf("\n7. Statistics\n");
    printf("   ----------\n");
    printf("   Total launches: %" PRIu64 "\n", kpu_runtime_get_launch_count(runtime));
    printf("   Total cycles:   %" PRIu64 "\n", kpu_runtime_get_total_cycles(runtime));

    printf("\n8. Cleanup\n");
    printf("   -------\n");

    free(A_host);
    free(B_host);
    free(C_host);
    printf("   Host memory freed\n");

    kpu_runtime_free(runtime, A_dev);
    kpu_runtime_free(runtime, B_dev);
    kpu_runtime_free(runtime, C_dev);
    printf("   Device memory freed\n");

    kpu_kernel_destroy(kernel);
    printf("   Kernel destroyed\n");

    kpu_runtime_destroy(runtime);
    printf("   Runtime destroyed\n");

    kpu_destroy(kpu);
    printf("   Simulator destroyed\n");

    printf("\n===========================================\n");
    printf(" Hello KPU %s\n", status == 0 ? "complete!" : "FAILED");
    printf("===========================================\n");

    return status;
}
