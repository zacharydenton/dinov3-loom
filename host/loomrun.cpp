// loomrun — launch one Loom-compiled kernel on the GPU and dump its buffers.
//
// The Loom AMDGPU ABI packs kernarg in declaration order: scalars by value at
// their natural alignment, then one 8-byte device pointer per `buffer` operand.
// Arguments are given on the command line in that same order.
//
//   loomrun --hsaco k.hsaco --kernel name --grid 201 --block 256 \
//           --i32 201 --in x.bin --in gamma.bin --in beta.bin --out y.bin:308736
//
// Built against real ROCm HIP on purpose: it keeps "is the kernel correct"
// independent of whether the HRX runtime is behaving.
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)

#define MAX_ARGS 32

enum arg_kind { ARG_I32, ARG_F32, ARG_BUFFER };

typedef struct {
    arg_kind kind;
    union { int i32; float f32; } scalar;
    char path[512];       // buffer: input file, or output file
    size_t bytes;         // buffer size
    int is_output;
    void *device_ptr;
} arg_t;

static void *read_file(const char *path, size_t *out_bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)n ? (size_t)n : 1);
    if (n && fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    fclose(f); *out_bytes = (size_t)n; return p;
}

// Append one value to the kernarg buffer at its natural alignment.
static size_t kernarg_append(unsigned char *buf, size_t offset, const void *src,
                             size_t size, size_t align) {
    offset = (offset + align - 1) & ~(align - 1);
    memcpy(buf + offset, src, size);
    return offset + size;
}

int main(int argc, char **argv) {
    const char *hsaco = NULL, *kernel = NULL;
    unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1};
    arg_t args[MAX_ARGS]; int arg_count = 0;
    int repeat = 1, verbose = 0;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        #define NEXT() (i + 1 < argc ? argv[++i] : (fprintf(stderr, "%s needs a value\n", a), exit(64), (char*)""))
        if      (!strcmp(a, "--hsaco"))  hsaco = NEXT();
        else if (!strcmp(a, "--kernel")) kernel = NEXT();
        else if (!strcmp(a, "--grid"))   sscanf(NEXT(), "%u,%u,%u", &grid[0], &grid[1], &grid[2]);
        else if (!strcmp(a, "--block"))  sscanf(NEXT(), "%u,%u,%u", &block[0], &block[1], &block[2]);
        else if (!strcmp(a, "--repeat")) repeat = atoi(NEXT());
        else if (!strcmp(a, "--verbose")) verbose = 1;
        else if (!strcmp(a, "--i32")) {
            args[arg_count].kind = ARG_I32;
            args[arg_count].scalar.i32 = atoi(NEXT());
            arg_count++;
        } else if (!strcmp(a, "--f32")) {
            args[arg_count].kind = ARG_F32;
            args[arg_count].scalar.f32 = (float)atof(NEXT());
            arg_count++;
        } else if (!strcmp(a, "--in")) {
            args[arg_count].kind = ARG_BUFFER; args[arg_count].is_output = 0;
            snprintf(args[arg_count].path, sizeof(args[arg_count].path), "%s", NEXT());
            arg_count++;
        } else if (!strcmp(a, "--out")) {
            // --out path:bytes
            const char *spec = NEXT();
            const char *colon = strrchr(spec, ':');
            if (!colon) { fprintf(stderr, "--out wants path:bytes\n"); return 64; }
            args[arg_count].kind = ARG_BUFFER; args[arg_count].is_output = 1;
            snprintf(args[arg_count].path, (size_t)(colon - spec) + 1, "%s", spec);
            args[arg_count].bytes = strtoull(colon + 1, NULL, 10);
            arg_count++;
        } else { fprintf(stderr, "unknown option %s\n", a); return 64; }
        if (arg_count > MAX_ARGS) { fprintf(stderr, "too many args\n"); return 64; }
    }
    if (!hsaco || !kernel) { fprintf(stderr, "need --hsaco and --kernel\n"); return 64; }

    HIP_CHECK(hipInit(0));
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, hsaco));
    hipFunction_t function;
    HIP_CHECK(hipModuleGetFunction(&function, module, kernel));

    unsigned char kernarg[512] = {0};
    size_t kernarg_size = 0;
    for (int i = 0; i < arg_count; ++i) {
        arg_t *arg = &args[i];
        if (arg->kind == ARG_I32) {
            kernarg_size = kernarg_append(kernarg, kernarg_size, &arg->scalar.i32, 4, 4);
        } else if (arg->kind == ARG_F32) {
            kernarg_size = kernarg_append(kernarg, kernarg_size, &arg->scalar.f32, 4, 4);
        } else {
            void *host = NULL;
            if (!arg->is_output) {
                host = read_file(arg->path, &arg->bytes);
                if (!host) return 1;
            }
            HIP_CHECK(hipMalloc(&arg->device_ptr, arg->bytes ? arg->bytes : 4));
            HIP_CHECK(hipMemset(arg->device_ptr, 0, arg->bytes ? arg->bytes : 4));
            if (host) {
                HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)arg->device_ptr, host, arg->bytes));
                free(host);
            }
            kernarg_size = kernarg_append(kernarg, kernarg_size, &arg->device_ptr, 8, 8);
        }
    }
    if (verbose) fprintf(stderr, "kernarg_size=%zu grid=%u,%u,%u block=%u,%u,%u\n",
                         kernarg_size, grid[0], grid[1], grid[2], block[0], block[1], block[2]);

    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernarg_size,
                       HIP_LAUNCH_PARAM_END };

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipModuleLaunchKernel(function, grid[0], grid[1], grid[2],
                                    block[0], block[1], block[2], 0, NULL, NULL, config));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(start, NULL));
    for (int r = 0; r < repeat; ++r) {
        HIP_CHECK(hipModuleLaunchKernel(function, grid[0], grid[1], grid[2],
                                        block[0], block[1], block[2], 0, NULL, NULL, config));
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipDeviceSynchronize());
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    printf("{\"launches\": %d, \"total_ms\": %.6f, \"per_launch_us\": %.3f}\n",
           repeat, elapsed_ms, 1000.0 * elapsed_ms / repeat);

    for (int i = 0; i < arg_count; ++i) {
        arg_t *arg = &args[i];
        if (arg->kind != ARG_BUFFER || !arg->is_output) continue;
        void *host = malloc(arg->bytes);
        HIP_CHECK(hipMemcpyDtoH(host, (hipDeviceptr_t)arg->device_ptr, arg->bytes));
        FILE *f = fopen(arg->path, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", arg->path); return 1; }
        fwrite(host, 1, arg->bytes, f);
        fclose(f); free(host);
    }
    for (int i = 0; i < arg_count; ++i)
        if (args[i].kind == ARG_BUFFER) (void)hipFree(args[i].device_ptr);
    return 0;
}
