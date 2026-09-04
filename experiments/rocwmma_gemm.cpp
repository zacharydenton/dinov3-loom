// What WMMA is worth on gfx1151: a rocWMMA fp16 GEMM at the shapes DINOv3
// ViT-S+ uses, next to the f32 number the Loom kernel gets on the same shape.
//
// C[m, n] = sum_k A[m, k] * W[n, k], A and W fp16, accumulation f32 — the same
// arithmetic torch's fp16 path does through hipBLASLt.
//
// This is deliberately a plain rocWMMA kernel, not a tuned one: it is here to
// bound the opportunity, not to be the final kernel.
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("%s: %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)

constexpr int WMMA = 16;
constexpr int WAVES_M = 4;   // 8 wave32s per block, arranged 4x2
constexpr int WAVES_N = 2;
constexpr int TILE_M = WAVES_M * WMMA;   // 64
constexpr int TILE_N = WAVES_N * WMMA;   // 32

__global__ __launch_bounds__(256) void wmma_gemm(const float16_t *__restrict__ a,
                                                 const float16_t *__restrict__ w,
                                                 float *__restrict__ c,
                                                 int m, int k, int n) {
    const int wave = threadIdx.x / 32;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;
    const int row = blockIdx.y * TILE_M + wave_m * WMMA;
    const int col = blockIdx.x * TILE_N + wave_n * WMMA;
    if (row >= m || col >= n) return;

    rocwmma::fragment<accumulator, WMMA, WMMA, WMMA, float32_t> acc;
    rocwmma::fill_fragment(acc, 0.0f);
    rocwmma::fragment<matrix_a, WMMA, WMMA, WMMA, float16_t, row_major> frag_a;
    rocwmma::fragment<matrix_b, WMMA, WMMA, WMMA, float16_t, col_major> frag_b;

    for (int step = 0; step < k; step += WMMA) {
        // W is [n, k] row-major, so W^T is [k, n] column-major with ld = k.
        rocwmma::load_matrix_sync(frag_a, a + size_t(row) * k + step, k);
        rocwmma::load_matrix_sync(frag_b, w + size_t(col) * k + step, k);
        rocwmma::mma_sync(acc, frag_a, frag_b, acc);
    }
    rocwmma::store_matrix_sync(c + size_t(row) * n + col, acc, n, rocwmma::mem_row_major);
}

static bool verbose = false;

static int run(const char *name, int m, int k, int n, int repeat) {
    std::vector<float16_t> a(size_t(m) * k), w(size_t(n) * k);
    unsigned state = 12345u;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return (float(state >> 8) / float(1 << 24) - 0.5f) * 0.4f;
    };
    for (size_t i = 0; i < a.size(); ++i) a[i] = float16_t(next());
    for (size_t i = 0; i < w.size(); ++i) w[i] = float16_t(next());

    float16_t *da, *dw; float *dc;
    HIP_CHECK(hipMalloc(&da, a.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&dw, w.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&dc, size_t(m) * n * sizeof(float)));
    HIP_CHECK(hipMemcpy(da, a.data(), a.size() * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dw, w.data(), w.size() * sizeof(float16_t), hipMemcpyHostToDevice));

    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
    dim3 block(256);
    hipLaunchKernelGGL(wmma_gemm, grid, block, 0, nullptr, da, dw, dc, m, k, n);
    HIP_CHECK(hipDeviceSynchronize());

    // Spot-check one output against a host dot product.
    std::vector<float> host(size_t(m) * n);
    HIP_CHECK(hipMemcpy(host.data(), dc, host.size() * sizeof(float), hipMemcpyDeviceToHost));
    // Check a handful of scattered outputs against host dot products.
    double worst = 0.0;
    int checked = 0;
    for (int row = 0; row < m && checked < 8; row += (m / 8) + 1) {
        for (int col = 0; col < n && checked < 8; col += (n / 4) + 1) {
            double expected = 0.0;
            for (int i = 0; i < k; ++i)
                expected += double(static_cast<float>(a[size_t(row) * k + i])) *
                            double(static_cast<float>(w[size_t(col) * k + i]));
            double got = host[size_t(row) * n + col];
            double denom = std::abs(expected) > 1e-3 ? std::abs(expected) : 1.0;
            worst = std::max(worst, std::abs(got - expected) / denom);
            ++checked;
        }
    }
    double error = worst;
    if (verbose)
        printf("    sample c[0,0]=%.6f  c[%d,%d]=%.6f\n", host[0], m / 2, n / 2,
               host[size_t(m / 2) * n + n / 2]);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, nullptr));
    for (int r = 0; r < repeat; ++r)
        hipLaunchKernelGGL(wmma_gemm, grid, block, 0, nullptr, da, dw, dc, m, k, n);
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double us = 1000.0 * ms / repeat;
    printf("%-14s M=%5d K=%5d N=%5d  %9.2f us  %8.1f GFLOP/s  rel_err=%.1e\n",
           name, m, k, n, us, 2.0 * m * k * n / (us * 1e-6) / 1e9, error);
    HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(dc));
    return 0;
}

int main(int argc, char **argv) {
    const int batch = argc > 1 ? atoi(argv[1]) : 32;
    verbose = argc > 2;
    const int rows = batch * 201;
    printf("rocWMMA fp16 (f32 accumulate), gfx1151, batch %d -> M=%d\n", batch, rows);
    run("qkv       ", rows, 384, 1152, 50);
    run("o         ", rows, 384, 384, 50);
    run("gate/up   ", rows, 384, 3072, 50);
    run("down      ", rows, 1536, 384, 50);
    return 0;
}
