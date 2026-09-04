// DINOv3 ViT-S+/16 forward pass on gfx1151, entirely in Loom-compiled kernels.
//
//   dinov3 --weights build/weights --kernels build/kernels \
//          --input patchified.bin --output out.bin [--repeat 50]
//
// The input is a pre-patchified [196 x 768] f32 image; patch extraction is a
// pure reshape (16x16 patches, stride 16) and stays on the host.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
    exit(1); } } while (0)

namespace {

// --profile: synchronise after every launch and attribute the time to a stage.
bool g_profile = false;
std::map<std::string, double> g_stage_us;
std::map<std::string, int> g_stage_calls;

struct Stage {
    const char *name;
    std::chrono::steady_clock::time_point start;
    explicit Stage(const char *n) : name(n) {
        if (g_profile) start = std::chrono::steady_clock::now();
    }
    ~Stage() {
        if (!g_profile) return;
        (void)hipDeviceSynchronize();
        auto us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        g_stage_us[name] += us;
        g_stage_calls[name] += 1;
    }
};

constexpr int HIDDEN = 384;
constexpr int HEADS = 6;
constexpr int HEAD_DIM = HIDDEN / HEADS;
constexpr int LAYERS = 12;
constexpr int INTERMEDIATE = 1536;
constexpr int PREFIX = 5;          // CLS + 4 registers
constexpr int PATCHES = 196;
constexpr int TOKENS = PREFIX + PATCHES;   // per image
constexpr int PATCH_K = 768;       // 3 * 16 * 16
constexpr int THREADS = 256;
constexpr int TILE = 64;           // wide matmul workgroup tile
constexpr int TILE_NARROW = 32;    // N tile for the narrow projections
constexpr int QKV = 3 * HIDDEN;          // fused q|k|v projection width
constexpr int GATEUP = 2 * INTERMEDIATE; // fused gate|up projection width

struct Span { size_t offset; size_t count; };

std::map<std::string, Span> read_manifest(const std::string &path) {
    std::map<std::string, Span> spans;
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    std::string name; size_t offset, count;
    while (in >> name >> offset >> count) spans[name] = {offset, count};
    return spans;
}

std::vector<char> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    std::vector<char> buffer(in.tellg());
    in.seekg(0);
    in.read(buffer.data(), buffer.size());
    return buffer;
}

// One compiled kernel plus the module that owns it.
struct Kernel {
    hipFunction_t function = nullptr;
    void load(const std::string &dir, const std::string &stem, const char *symbol) {
        hipModule_t module;
        HIP_CHECK(hipModuleLoad(&module, (dir + "/" + stem + ".hsaco").c_str()));
        HIP_CHECK(hipModuleGetFunction(&function, module, symbol));
    }
};

// Loom's AMDGPU kernarg ABI: scalars at natural alignment, then 8-byte pointers.
struct KernArgs {
    unsigned char bytes[128] = {0};
    size_t size = 0;
    void scalar_i32(int value) {
        size = (size + 3) & ~size_t(3);
        memcpy(bytes + size, &value, 4);
        size += 4;
    }
    void pointer(const void *value) {
        size = (size + 7) & ~size_t(7);
        memcpy(bytes + size, &value, 8);
        size += 8;
    }
};

void launch(hipFunction_t function, unsigned gx, unsigned gy, KernArgs &args) {
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size,
                       HIP_LAUNCH_PARAM_END };
    HIP_CHECK(hipModuleLaunchKernel(function, gx, gy, 1, THREADS, 1, 1, 0, nullptr,
                                    nullptr, config));
}

}  // namespace

int main(int argc, char **argv) {
    std::string weights_dir = "build/weights", kernels_dir = "build/kernels";
    std::string input_path, output_path;
    // Above this many rows the 64x64 tile is used for the N=384 projections
    // instead of 64x32. Measured best-of-3 at batch 32 gave 243 vs 222 img/s in
    // its favour and batch 8 gave 211 vs 230 against it -- both differences sit
    // inside this machine's run-to-run variance, so the default keeps the
    // simpler single-tile path and the flag is here to re-measure on an idle box.
    int repeat = 1, batch = 1, wide_threshold = 1 << 30;
    bool wmma = true, flash_attn = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--weights") weights_dir = next();
        else if (a == "--kernels") kernels_dir = next();
        else if (a == "--input") input_path = next();
        else if (a == "--output") output_path = next();
        else if (a == "--repeat") repeat = std::stoi(next());
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--wide-threshold") wide_threshold = std::stoi(next());
        else if (a == "--profile") g_profile = true;
        else if (a == "--f32") wmma = false;
        else if (a == "--no-flash") flash_attn = false;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 64; }
    }

    auto spans = read_manifest(weights_dir + "/manifest.txt");
    auto blob = read_file(weights_dir + "/weights.bin");
    auto spans16 = read_manifest(weights_dir + "/manifest_f16.txt");
    auto blob16 = read_file(weights_dir + "/weights_f16.bin");

    HIP_CHECK(hipInit(0));
    float *weights = nullptr;
    HIP_CHECK(hipMalloc(&weights, blob.size()));
    HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights, blob.data(), blob.size()));
    auto W = [&](const std::string &name) -> float * {
        auto it = spans.find(name);
        if (it == spans.end()) { fprintf(stderr, "missing weight %s\n", name.c_str()); exit(1); }
        return weights + it->second.offset;
    };
    unsigned short *weights16 = nullptr;
    HIP_CHECK(hipMalloc(&weights16, blob16.size()));
    HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights16, blob16.data(), blob16.size()));
    auto W16 = [&](const std::string &name) -> unsigned short * {
        auto it = spans16.find(name);
        if (it == spans16.end()) { fprintf(stderr, "missing f16 weight %s\n", name.c_str()); exit(1); }
        return weights16 + it->second.offset;
    };
    // Projection weights come from the f16 blob under WMMA, the f32 one otherwise.
    auto PW = [&](const std::string &name) -> const void * {
        return wmma ? (const void *)W16(name) : (const void *)W(name);
    };

    Kernel layernorm, rope, attention, swiglu, residual;
    Kernel matmul_patch, matmul_qkvo, matmul_qkv, matmul_gateup, matmul_down, scatter;
    Kernel matmul_qkvo_wide, matmul_down_wide;
    Kernel wmma_patch, wmma_qkv, wmma_o, wmma_gateup, wmma_down, flash;
    layernorm.load(kernels_dir, "layernorm", "dinov3_layernorm_f32");
    rope.load(kernels_dir, "rope", "dinov3_rope_2d_f32");
    attention.load(kernels_dir, "attention", "dinov3_attention_f32");
    swiglu.load(kernels_dir, "swiglu", "dinov3_swiglu_f32");
    residual.load(kernels_dir, "residual", "dinov3_residual_scale_f32");
    matmul_patch.load(kernels_dir, "matmul_k768_n384", "dinov3_matmul_bias_f32");
    scatter.load(kernels_dir, "embed_scatter", "dinov3_embed_scatter_f32");
    matmul_qkvo.load(kernels_dir, "matmul_narrow_k384_n384", "dinov3_matmul_bias_f32_narrow");
    matmul_qkv.load(kernels_dir, "matmul_k384_n1152", "dinov3_matmul_bias_f32");
    matmul_gateup.load(kernels_dir, "matmul_k384_n3072", "dinov3_matmul_bias_f32");
    matmul_down.load(kernels_dir, "matmul_narrow_k1536_n384", "dinov3_matmul_bias_f32_narrow");
    matmul_qkvo_wide.load(kernels_dir, "matmul_k384_n384", "dinov3_matmul_bias_f32");
    matmul_down_wide.load(kernels_dir, "matmul_k1536_n384", "dinov3_matmul_bias_f32");
    wmma_patch.load(kernels_dir, "wmma_k768_n384", "dinov3_matmul_bias_f16_wmma");
    wmma_qkv.load(kernels_dir, "wmma_k384_n1152", "dinov3_matmul_bias_f16_wmma");
    wmma_o.load(kernels_dir, "wmma_k384_n384", "dinov3_matmul_bias_f16_wmma");
    wmma_gateup.load(kernels_dir, "wmma_k384_n3072", "dinov3_matmul_bias_f16_wmma");
    wmma_down.load(kernels_dir, "wmma_k1536_n384", "dinov3_matmul_bias_f16_wmma");
    flash.load(kernels_dir, "flash_attention", "dinov3_flash_attention_f16_wmma");

    const int rows = batch * TOKENS;          // residual-stream rows
    const int patch_rows = batch * PATCHES;
    // The 64x32 tile exists to get enough workgroups out of N=384 at one image.
    // Once the batch supplies the workgroups, the 64x64 tile's better
    // compute-to-LDS ratio wins again.
    const bool wide_narrow_n = rows >= wide_threshold;
    Kernel &proj_kernel = wide_narrow_n ? matmul_qkvo_wide : matmul_qkvo;
    Kernel &down_kernel = wide_narrow_n ? matmul_down_wide : matmul_down;
    const int narrow_tile = wide_narrow_n ? TILE : TILE_NARROW;

    float *x, *h, *q, *k, *v, *attn, *proj, *gate, *up, *act, *mlp, *out, *image, *patched;
    auto alloc = [](float **p, size_t floats) { HIP_CHECK(hipMalloc(p, floats * sizeof(float))); };
    alloc(&x, size_t(rows) * HIDDEN);      alloc(&h, size_t(rows) * HIDDEN);
    alloc(&q, size_t(rows) * QKV);         alloc(&attn, size_t(rows) * HIDDEN);
    alloc(&proj, size_t(rows) * HIDDEN);   alloc(&mlp, size_t(rows) * HIDDEN);
    alloc(&gate, size_t(rows) * GATEUP);
    alloc(&act, size_t(rows) * INTERMEDIATE);  alloc(&out, size_t(rows) * HIDDEN);
    alloc(&patched, size_t(patch_rows) * HIDDEN);
    k = q + HIDDEN; v = q + 2 * HIDDEN; up = gate + INTERMEDIATE;
    alloc(&image, size_t(patch_rows) * PATCH_K);

    if (!input_path.empty()) {
        // One patchified image, replicated to fill the batch.
        auto pixels = read_file(input_path);
        for (int b = 0; b < batch; ++b)
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)(image + size_t(b) * PATCHES * PATCH_K),
                                    pixels.data(), size_t(PATCHES) * PATCH_K * sizeof(float)));
    }

    auto matmul = [&](Kernel &kernel, int m, int n, const float *a, const void *w,
                      const float *bias, float *c, int tile_n = TILE) {
        KernArgs args;
        args.scalar_i32(m);
        args.pointer(a); args.pointer(w); args.pointer(bias); args.pointer(c);
        launch(kernel.function, n / tile_n, (m + TILE - 1) / TILE, args);
    };
    auto norm = [&](const float *in, const float *gamma, const float *beta, float *dst) {
        KernArgs args;
        args.scalar_i32(rows);
        args.pointer(in); args.pointer(gamma); args.pointer(beta); args.pointer(dst);
        launch(layernorm.function, rows, 1, args);
    };

    auto forward = [&]() {
        // Patch embedding writes straight into the residual stream behind the
        // CLS and register tokens, so no copy is needed afterwards.
        { Stage stage("patch-embed matmul");
          matmul(wmma ? wmma_patch : matmul_patch, patch_rows, HIDDEN, image,
                 PW("patch_w"), W("patch_b"), patched); }
        { Stage stage("embed scatter");
          KernArgs args;
          args.scalar_i32(rows);
          args.pointer(patched); args.pointer(W("prefix")); args.pointer(x);
          launch(scatter.function, rows, 1, args); }

        for (int layer = 0; layer < LAYERS; ++layer) {
            std::string p = "l" + std::to_string(layer) + "_";
            { Stage stage("layernorm"); norm(x, W(p + "norm1_w"), W(p + "norm1_b"), h); }
            { Stage stage("qkv matmul");
              matmul(wmma ? wmma_qkv : matmul_qkv, rows, QKV, h,
                     PW(p + "qkv_w"), W(p + "qkv_b"), q); }

            { Stage stage("rope");
            for (float *target : {q, k}) {
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(target); args.pointer(W("rope_cos")); args.pointer(W("rope_sin"));
                launch(rope.function, rows, 1, args);
            } }
            { Stage stage("attention");
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(q); args.pointer(k); args.pointer(v); args.pointer(attn);
                if (wmma && flash_attn) {
                    // One workgroup per (16 query rows within an image, head).
                    launch(flash.function, batch * ((TOKENS + 15) / 16), HEADS, args);
                } else {
                    // One workgroup per (block of 8 queries within an image, head).
                    launch(attention.function, batch * ((TOKENS + 7) / 8), HEADS, args);
                }
            }
            { Stage stage("o matmul");
              matmul(wmma ? wmma_o : proj_kernel, rows, HIDDEN, attn,
                     PW(p + "o_w"), W(p + "o_b"), proj, wmma ? TILE : narrow_tile); }
            { Stage stage("residual+ls");
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(x); args.pointer(proj); args.pointer(W(p + "ls1"));
                launch(residual.function, rows, 1, args);
            }

            { Stage stage("layernorm"); norm(x, W(p + "norm2_w"), W(p + "norm2_b"), h); }
            { Stage stage("gate/up matmul");
              matmul(wmma ? wmma_gateup : matmul_gateup, rows, GATEUP, h,
                     PW(p + "gateup_w"), W(p + "gateup_b"), gate); }
            { Stage stage("swiglu");
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(gate); args.pointer(up); args.pointer(act);
                launch(swiglu.function, rows, 1, args);
            }
            { Stage stage("down matmul");
              matmul(wmma ? wmma_down : down_kernel, rows, HIDDEN, act,
                     PW(p + "down_w"), W(p + "down_b"), mlp, wmma ? TILE : narrow_tile); }
            { Stage stage("residual+ls");
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(x); args.pointer(mlp); args.pointer(W(p + "ls2"));
                launch(residual.function, rows, 1, args);
            }
        }
        { Stage stage("layernorm"); norm(x, W("norm_w"), W("norm_b"), out); }
    };

    forward();
    HIP_CHECK(hipDeviceSynchronize());

    if (repeat > 1) {
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start, nullptr));
        for (int r = 0; r < repeat; ++r) forward();
        HIP_CHECK(hipEventRecord(stop, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        int images = repeat * batch;
        printf("{\"batch\": %d, \"images\": %d, \"total_ms\": %.3f, "
               "\"ms_per_image\": %.4f, \"img_per_s\": %.2f}\n",
               batch, images, elapsed, elapsed / images, 1000.0 * images / elapsed);
    }

    if (g_profile) {
        double total = 0.0;
        for (auto &kv : g_stage_us) total += kv.second;
        std::vector<std::pair<double, std::string>> rows;
        for (auto &kv : g_stage_us) rows.push_back({kv.second, kv.first});
        std::sort(rows.rbegin(), rows.rend());
        printf("stage breakdown over %d image(s):\n", repeat > 1 ? repeat + 1 : 1);
        for (auto &r : rows)
            printf("  %-20s %9.3f ms  %5.1f%%  (%d launches)\n", r.second.c_str(),
                   r.first / 1000.0, 100.0 * r.first / total, g_stage_calls[r.second]);
        printf("  %-20s %9.3f ms\n", "total", total / 1000.0);
    }

    if (!output_path.empty()) {
        std::vector<float> host(size_t(rows) * HIDDEN);
        HIP_CHECK(hipMemcpyDtoH(host.data(), (hipDeviceptr_t)out, host.size() * sizeof(float)));
        std::ofstream(output_path, std::ios::binary)
            .write(reinterpret_cast<char *>(host.data()), host.size() * sizeof(float));
    }
    return 0;
}
