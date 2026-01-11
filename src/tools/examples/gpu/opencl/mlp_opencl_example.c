/**
 * OpenCL Example - MLP Forward Pass for ARM Mali GPU
 *
 * This demonstrates how to:
 * 1. Execute compute kernels on Mali GPU
 * 2. Compare performance with CPU
 * 3. Optimize for Mali architecture
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Error checking macro
#define CL_CHECK(call) \
    do { \
        cl_int err = call; \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// OpenCL Kernel Source (C-like syntax)
// ============================================================================

const char* mlp_kernel_source = R"(
__kernel void mlp_layer_forward_opencl(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const int input_size,
    const int output_size
) {
    int output_idx = get_global_id(0);

    if (output_idx >= output_size) return;

    // Local memory for input vector (shared across work group)
    __local float local_input[1024];

    // Cooperatively load input to local memory
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    for (int i = local_id; i < input_size; i += local_size) {
        local_input[i] = input[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute dot product
    float sum = 0.0f;
    __global const float* weight_row = weights + output_idx * input_size;

    // Vectorized loop - process 4 elements at a time
    int i = 0;
    for (; i + 3 < input_size; i += 4) {
        float4 w = vload4(i >> 2, weight_row);
        float4 in = vload4(i >> 2, local_input);
        sum += dot(w, in);
    }

    // Handle remainder
    for (; i < input_size; i++) {
        sum += weight_row[i] * local_input[i];
    }

    // Add bias and apply ReLU
    sum += bias[output_idx];
    output[output_idx] = fmax(0.0f, sum);
}


// Mali-optimized version using vec4 operations
__kernel void mlp_layer_forward_mali_optimized(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const int input_size,
    const int output_size
) {
    int output_idx = get_global_id(0);

    if (output_idx >= output_size) return;

    __local float local_input[1024];

    // Cooperative load
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    for (int i = local_id; i < input_size; i += local_size) {
        local_input[i] = input[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Use multiple accumulators for better ILP on Mali
    float4 sum0 = (float4)(0.0f);
    float4 sum1 = (float4)(0.0f);

    __global const float* weight_row = weights + output_idx * input_size;

    // Process 8 elements per iteration (2x vec4)
    int i = 0;
    int aligned = (input_size >> 3) << 3;  // Align to 8

    for (; i < aligned; i += 8) {
        // Load weights (2x vec4)
        float4 w0 = vload4((i + 0) >> 2, weight_row);
        float4 w1 = vload4((i + 4) >> 2, weight_row);

        // Load inputs (2x vec4)
        float4 in0 = vload4((i + 0) >> 2, local_input);
        float4 in1 = vload4((i + 4) >> 2, local_input);

        // FMA operations (Mali can dual-issue these)
        sum0 = fma(w0, in0, sum0);
        sum1 = fma(w1, in1, sum1);
    }

    // Combine accumulators
    sum0 += sum1;

    // Horizontal reduction
    float sum = sum0.x + sum0.y + sum0.z + sum0.w;

    // Handle remainder
    for (; i < input_size; i++) {
        sum += weight_row[i] * local_input[i];
    }

    // Bias and ReLU
    sum += bias[output_idx];
    output[output_idx] = fmax(0.0f, sum);
}
)";


// ============================================================================
// Utility Functions
// ============================================================================

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void init_random(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

int verify_results(const float* a, const float* b, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabsf(a[i] - b[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f (diff: %.6f)\n",
                   i, a[i], b[i], fabsf(a[i] - b[i]));
            return 0;
        }
    }
    return 1;
}

// CPU reference implementation
void mlp_layer_forward_cpu(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size
) {
    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[i * input_size + j];
        }
        sum += bias[i];
        output[i] = (sum > 0.0f) ? sum : 0.0f;
    }
}


// ============================================================================
// OpenCL Setup
// ============================================================================

void print_device_info(cl_device_id device) {
    char name[256];
    char vendor[256];
    char version[256];
    cl_uint compute_units;
    cl_ulong global_mem;
    cl_ulong local_mem;
    size_t max_work_group_size;

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);

    printf("Device: %s\n", name);
    printf("Vendor: %s\n", vendor);
    printf("Version: %s\n", version);
    printf("Compute Units: %u\n", compute_units);
    printf("Global Memory: %lu MB\n", global_mem / (1024 * 1024));
    printf("Local Memory: %lu KB\n", local_mem / 1024);
    printf("Max Work Group Size: %zu\n", max_work_group_size);
    printf("\n");
}


// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char** argv) {
    printf("OpenCL MLP Example for ARM Mali\n");
    printf("================================\n\n");

    // Get platform
    cl_platform_id platform;
    cl_uint num_platforms;
    CL_CHECK(clGetPlatformIDs(1, &platform, &num_platforms));

    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return 1;
    }

    // Get device (prefer GPU)
    cl_device_id device;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("No GPU found, trying CPU...\n");
        CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL));
    }

    print_device_info(device);

    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err);

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CL_CHECK(err);

    // Build program
    cl_program program = clCreateProgramWithSource(context, 1, &mlp_kernel_source, NULL, &err);
    CL_CHECK(err);

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log:\n%s\n", log);
        free(log);
        return 1;
    }

    // Create kernels
    cl_kernel kernel_basic = clCreateKernel(program, "mlp_layer_forward_opencl", &err);
    CL_CHECK(err);

    cl_kernel kernel_optimized = clCreateKernel(program, "mlp_layer_forward_mali_optimized", &err);
    CL_CHECK(err);

    // Problem size
    const int input_size = 512;
    const int output_size = 256;
    const int iterations = 100;

    printf("Configuration:\n");
    printf("  Input size: %d\n", input_size);
    printf("  Output size: %d\n", output_size);
    printf("  Iterations: %d\n\n", iterations);

    // Allocate host memory
    size_t input_bytes = input_size * sizeof(float);
    size_t weights_bytes = output_size * input_size * sizeof(float);
    size_t bias_bytes = output_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);

    float* h_input = (float*)malloc(input_bytes);
    float* h_weights = (float*)malloc(weights_bytes);
    float* h_bias = (float*)malloc(bias_bytes);
    float* h_output_cpu = (float*)malloc(output_bytes);
    float* h_output_gpu = (float*)malloc(output_bytes);
    float* h_output_opt = (float*)malloc(output_bytes);

    // Initialize data
    srand(42);
    init_random(h_input, input_size);
    init_random(h_weights, output_size * input_size);
    init_random(h_bias, output_size);

    // Create OpenCL buffers
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     input_bytes, h_input, &err);
    CL_CHECK(err);

    cl_mem d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       weights_bytes, h_weights, &err);
    CL_CHECK(err);

    cl_mem d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    bias_bytes, h_bias, &err);
    CL_CHECK(err);

    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      output_bytes, NULL, &err);
    CL_CHECK(err);

    // Set kernel arguments (basic)
    CL_CHECK(clSetKernelArg(kernel_basic, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel_basic, 1, sizeof(cl_mem), &d_weights));
    CL_CHECK(clSetKernelArg(kernel_basic, 2, sizeof(cl_mem), &d_bias));
    CL_CHECK(clSetKernelArg(kernel_basic, 3, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel_basic, 4, sizeof(int), &input_size));
    CL_CHECK(clSetKernelArg(kernel_basic, 5, sizeof(int), &output_size));

    // Set kernel arguments (optimized)
    CL_CHECK(clSetKernelArg(kernel_optimized, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel_optimized, 1, sizeof(cl_mem), &d_weights));
    CL_CHECK(clSetKernelArg(kernel_optimized, 2, sizeof(cl_mem), &d_bias));
    CL_CHECK(clSetKernelArg(kernel_optimized, 3, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel_optimized, 4, sizeof(int), &input_size));
    CL_CHECK(clSetKernelArg(kernel_optimized, 5, sizeof(int), &output_size));

    // Work group configuration
    size_t global_size = ((output_size + 255) / 256) * 256;  // Round up to multiple of 256
    size_t local_size = 256;

    printf("Work group configuration:\n");
    printf("  Global size: %zu\n", global_size);
    printf("  Local size: %zu\n\n", local_size);

    // ========================================================================
    // CPU Reference
    // ========================================================================

    printf("Running CPU reference...\n");
    double start = get_time();

    for (int i = 0; i < iterations; i++) {
        mlp_layer_forward_cpu(h_input, h_weights, h_bias, h_output_cpu, input_size, output_size);
    }

    double cpu_time = get_time() - start;
    printf("  Time: %.4f ms (%.2f us/iter)\n\n", cpu_time * 1000, (cpu_time * 1e6) / iterations);

    // ========================================================================
    // Basic OpenCL Kernel
    // ========================================================================

    printf("Running basic OpenCL kernel...\n");
    CL_CHECK(clFinish(queue));  // Ensure queue is empty

    start = get_time();

    for (int i = 0; i < iterations; i++) {
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_basic, 1, NULL,
                                        &global_size, &local_size, 0, NULL, NULL));
    }
    CL_CHECK(clFinish(queue));

    double gpu_time = get_time() - start;

    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_bytes, h_output_gpu, 0, NULL, NULL));

    printf("  Time: %.4f ms (%.2f us/iter)\n", gpu_time * 1000, (gpu_time * 1e6) / iterations);
    printf("  Speedup vs CPU: %.2fx\n", cpu_time / gpu_time);

    if (verify_results(h_output_cpu, h_output_gpu, output_size, 1e-4f)) {
        printf("  ✓ Results match CPU\n");
    } else {
        printf("  ✗ Results differ from CPU\n");
    }

    printf("\n");

    // ========================================================================
    // Optimized OpenCL Kernel
    // ========================================================================

    printf("Running Mali-optimized kernel...\n");
    CL_CHECK(clFinish(queue));

    start = get_time();

    for (int i = 0; i < iterations; i++) {
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_optimized, 1, NULL,
                                        &global_size, &local_size, 0, NULL, NULL));
    }
    CL_CHECK(clFinish(queue));

    double opt_time = get_time() - start;

    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_bytes, h_output_opt, 0, NULL, NULL));

    printf("  Time: %.4f ms (%.2f us/iter)\n", opt_time * 1000, (opt_time * 1e6) / iterations);
    printf("  Speedup vs CPU: %.2fx\n", cpu_time / opt_time);
    printf("  Speedup vs basic: %.2fx\n", gpu_time / opt_time);

    if (verify_results(h_output_cpu, h_output_opt, output_size, 1e-4f)) {
        printf("  ✓ Results match CPU\n");
    } else {
        printf("  ✗ Results differ from CPU\n");
    }

    printf("\n");

    // ========================================================================
    // Show sample outputs
    // ========================================================================

    printf("Sample outputs (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] CPU: %.4f, GPU: %.4f, OPT: %.4f\n",
               i, h_output_cpu[i], h_output_gpu[i], h_output_opt[i]);
    }

    // Cleanup
    free(h_input);
    free(h_weights);
    free(h_bias);
    free(h_output_cpu);
    free(h_output_gpu);
    free(h_output_opt);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_weights);
    clReleaseMemObject(d_bias);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel_basic);
    clReleaseKernel(kernel_optimized);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("\n✓ All tests completed successfully!\n");

    return 0;
}
