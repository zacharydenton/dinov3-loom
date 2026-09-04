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
// HEAD_DIM is implied by HIDDEN / HEADS; the kernels carry it as a config.
constexpr int LAYERS = 12;
constexpr int INTERMEDIATE = 1536;
constexpr int PREFIX = 5;          // CLS + 4 registers
constexpr int PATCHES = 196;
constexpr int TOKENS = PREFIX + PATCHES;   // per image
constexpr int PATCH_K = 768;       // 3 * 16 * 16
constexpr int THREADS = 256;
constexpr int TILE = 64;           // wide matmul workgroup tile
constexpr int QKV = 3 * HIDDEN;          // fused q|k|v projection width

struct Span { size_t offset; size_t count; };

std::map<std::string, Span> read_manifest(const std::string &path) {
    std::map<std::string, Span> spans;
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    std::string name; size_t offset, count;
    while (in >> name >> offset >> count) spans[name] = {offset, count};
    return spans;
}

// A manifest is generated data, but it is still input. Two things are checked:
// every span lies inside the blob, with the offset and count multiplications
// both overflow-checked; and every tensor the forward pass will read is present
// with exactly the element count the kernels are compiled for. The kernels take
// bare pointers and never see Span.count, so a manifest that declared patch_w
// as one element at the end of the blob would otherwise pass the bounds check
// and have the kernel read 294,911 elements past the allocation.
void check_spans(const std::map<std::string, Span> &spans, size_t blob_bytes,
                 size_t element_bytes, const char *what) {
    for (const auto &entry : spans) {
        size_t begin, extent, end;
        if (__builtin_mul_overflow(entry.second.offset, element_bytes, &begin) ||
            __builtin_mul_overflow(entry.second.count, element_bytes, &extent) ||
            __builtin_add_overflow(begin, extent, &end) || end > blob_bytes) {
            fprintf(stderr, "%s: span '%s' (offset %zu, count %zu) runs past the "
                            "%zu-byte blob\n", what, entry.first.c_str(),
                    entry.second.offset, entry.second.count, blob_bytes);
            exit(1);
        }
    }
}

void require(const std::map<std::string, Span> &spans, const std::string &name,
             size_t count, const char *what) {
    auto it = spans.find(name);
    if (it == spans.end()) {
        fprintf(stderr, "%s: missing tensor '%s'\n", what, name.c_str());
        exit(1);
    }
    if (it->second.count != count) {
        fprintf(stderr, "%s: tensor '%s' has %zu elements, the kernels expect %zu\n",
                what, name.c_str(), it->second.count, count);
        exit(1);
    }
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

void launchBlock(hipFunction_t function, unsigned gx, unsigned gy, unsigned block,
                 KernArgs &args) {
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size,
                       HIP_LAUNCH_PARAM_END };
    HIP_CHECK(hipModuleLaunchKernel(function, gx, gy, 1, block, 1, 1, 0, nullptr,
                                    nullptr, config));
}

void launch(hipFunction_t function, unsigned gx, unsigned gy, KernArgs &args) {
    launchBlock(function, gx, gy, THREADS, args);
}


// ---------------------------------------------------------------------------
// Resident session. These were locals in main(); they are file scope now so the
// launch helpers and the forward pass below keep the bodies they always had,
// and so a caller can init once and run many batches against the same context.
// ---------------------------------------------------------------------------
std::map<std::string, Span> spans, spans16;
float *weights = nullptr;
unsigned short *weights16 = nullptr;
Kernel scatter, wmma_patch, qkv_rope, attn_online, resid_o, resid_down;
Kernel wmma_swiglu, splitk_down, splitk_reduce, ln_rowwave, ln_rowwave_f32;
float *x, *h, *q, *k, *v, *attn, *act, *out, *image, *patched, *partials;
// The online attention kernel bounds its image index with a compiled
// max_images so the fragment loads prove in range; beyond it the modulo
// would silently fold later images onto earlier ones.
constexpr int MAX_IMAGES = 64;
// Split the down projection's k range when the batch is too small to fill
// the machine: at batch 1 it otherwise launches 24 workgroups against ~120
// slots. Measured 1.54x there and a loss from batch 2 up.
constexpr int SPLITS = 4;
int session_max_batch = 0;
int batch = 0, rows = 0, patch_rows = 0;
bool use_splitk = false;

float *W(const std::string &name) {
    auto it = spans.find(name);
    if (it == spans.end()) { fprintf(stderr, "missing weight %s\n", name.c_str()); exit(1); }
    return weights + it->second.offset;
}
unsigned short *W16(const std::string &name) {
    auto it = spans16.find(name);
    if (it == spans16.end()) { fprintf(stderr, "missing f16 weight %s\n", name.c_str()); exit(1); }
    return weights16 + it->second.offset;
}
// Projection weights come from the f16 blob under WMMA, the f32 one otherwise.
const void *PW(const std::string &name) { return (const void *)W16(name); }

// Sizing for one call. Buffers are allocated for the session maximum, so this
// only moves the launch bounds; nothing is reallocated between batches.
bool set_batch(int b) {
    if (b < 1 || b > session_max_batch) {
        fprintf(stderr, "batch %d out of range 1..%d\n", b, session_max_batch);
        return false;
    }
    batch = b;
    rows = batch * TOKENS;
    patch_rows = batch * PATCHES;
    use_splitk = rows <= TOKENS;
    return true;
}

// Bring up the device, upload the weights and load the kernels. Buffers are
// sized for `max_batch` and reused, so every later run is launches and two
// copies -- which is the whole point of holding the session open: the work
// below is ~100ms and used to be paid once per call.
int session_init(const char *weights_dir_c, const char *kernels_dir_c, int max_batch) {
    std::string weights_dir(weights_dir_c), kernels_dir(kernels_dir_c);
    if (max_batch < 1 || max_batch > MAX_IMAGES) {
        fprintf(stderr, "max_batch must be 1..%d (the kernels' compiled max_images)\n", MAX_IMAGES);
        return 64;
    }
    session_max_batch = max_batch;
    const int max_rows = max_batch * TOKENS;
    const int max_patch_rows = max_batch * PATCHES;

    spans = read_manifest(weights_dir + "/manifest.txt");
    auto blob = read_file(weights_dir + "/weights.bin");
    spans16 = read_manifest(weights_dir + "/manifest_f16.txt");
    auto blob16 = read_file(weights_dir + "/weights_f16.bin");
    check_spans(spans, blob.size(), sizeof(float), "manifest.txt");
    check_spans(spans16, blob16.size(), sizeof(uint16_t), "manifest_f16.txt");
    {
        const char *f32 = "manifest.txt", *f16 = "manifest_f16.txt";
        const size_t H = HIDDEN, I = INTERMEDIATE;
        require(spans16, "patch_w", H * PATCH_K, f16);
        require(spans, "patch_b", H, f32);
        require(spans, "prefix", size_t(PREFIX) * H, f32);
        require(spans, "rope_cos", size_t(PATCHES) * (H / HEADS), f32);
        require(spans, "rope_sin", size_t(PATCHES) * (H / HEADS), f32);
        for (int layer = 0; layer < LAYERS; ++layer) {
            std::string p = "l" + std::to_string(layer) + "_";
            require(spans16, p + "qkv_w", size_t(QKV) * H, f16);
            require(spans16, p + "o_w", H * H, f16);
            require(spans16, p + "gateup_w", 2 * I * H, f16);
            require(spans16, p + "down_w", H * I, f16);
            require(spans, p + "qkv_b", QKV, f32);
            require(spans, p + "o_b", H, f32);
            require(spans, p + "gateup_b", 2 * I, f32);
            require(spans, p + "down_b", H, f32);
            for (const char *v : {"ls1", "ls2", "norm1_w", "norm1_b", "norm2_w", "norm2_b"})
                require(spans, p + v, H, f32);
        }
        require(spans, "norm_w", H, f32);
        require(spans, "norm_b", H, f32);
    }

    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipMalloc(&weights, blob.size()));
    HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights, blob.data(), blob.size()));
    HIP_CHECK(hipMalloc(&weights16, blob16.size()));
    HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights16, blob16.data(), blob16.size()));

    // One configuration, so one kernel set. Earlier revisions carried f32,
    // non-fused and non-WMMA fallbacks selectable by flag; they multiplied the
    // dtype/layout combinations far past what was tested and several were
    // silently wrong. The measured-best path is the only path.
    scatter.load(kernels_dir, "embed_scatter", "dinov3_embed_scatter_f32");
    wmma_patch.load(kernels_dir, "wmma_k768_n384", "dinov3_matmul_bias_f16_wmma");
    qkv_rope.load(kernels_dir, "qkv_rope_k384_n1152", "dinov3_matmul_qkv_rope_f16_wmma");
    attn_online.load(kernels_dir, "attention_online_cf16", "dinov3_attention_online_f16_wmma_cf16");
    resid_o.load(kernels_dir, "resid_k384_n384", "dinov3_matmul_resid_f16_wmma");
    resid_down.load(kernels_dir, "resid_k1536_n384", "dinov3_matmul_resid_f16_wmma");
    wmma_swiglu.load(kernels_dir, "wmma_swiglu_k384_n1536", "dinov3_matmul_swiglu_f16_wmma");
    splitk_down.load(kernels_dir, "splitk_k1536_n384", "dinov3_matmul_splitk_f16_wmma");
    splitk_reduce.load(kernels_dir, "splitk_reduce_n384", "dinov3_splitk_reduce_f16");
    ln_rowwave.load(kernels_dir, "layernorm_rowwave", "dinov3_layernorm_rowwave_f16");
    ln_rowwave_f32.load(kernels_dir, "layernorm_rowwave_f32out", "dinov3_layernorm_rowwave_f32out");


    auto alloc = [](float **p, size_t floats) { HIP_CHECK(hipMalloc(p, floats * sizeof(float))); };
    // Every intermediate is f16, so the float-typed pointers are half-width
    // views of these buffers; the allocations below are generous by exactly
    // that factor rather than being tightened, which keeps the arithmetic here
    // obvious.
    alloc(&x, size_t(max_rows) * HIDDEN);      alloc(&h, size_t(max_rows) * HIDDEN);
    // A query tile may overhang the last image by up to 15 rows; the online
    // attention kernel masks those lanes but still forms the address.
    alloc(&q, size_t(max_rows + 16) * QKV);    alloc(&attn, size_t(max_rows) * HIDDEN);
    alloc(&act, size_t(max_rows) * INTERMEDIATE);  alloc(&out, size_t(max_rows) * HIDDEN);
    alloc(&patched, size_t(max_patch_rows) * HIDDEN);
    alloc(&image, size_t(max_patch_rows) * PATCH_K);
    // split-K runs only at batch 1, where rows == TOKENS
    alloc(&partials, size_t(SPLITS) * TOKENS * HIDDEN);
    // q/k/v are f16, so the head offsets are half as many float-sized steps.
    const int qkv_step = HIDDEN / 2;
    k = q + qkv_step; v = q + 2 * qkv_step;

    return 0;
}

void matmul(Kernel &kernel, int m, int n, const float *a, const void *w,
            const float *bias, float *c, int tile_n = TILE) {
        KernArgs args;
        args.scalar_i32(m);
        args.pointer(a); args.pointer(w); args.pointer(bias); args.pointer(c);
        launch(kernel.function, n / tile_n, (m + TILE - 1) / TILE, args);
}
    // A@W + bias scaled by lambda and accumulated into the residual stream.
void matmul_resid(Kernel &kernel, int m, int n, const float *a, const void *w,
                  const float *bias, float *stream, const float *lambda) {
        KernArgs args;
        args.scalar_i32(m);
        args.pointer(a); args.pointer(w); args.pointer(bias);
        args.pointer(stream); args.pointer(lambda);
        launch(kernel.function, n / TILE, (m + TILE - 1) / TILE, args);
}
    // `dst` is f16, so callers treat it as opaque.
void norm(const float *in, const float *gamma, const float *beta, void *dst,
          bool wide_out) {
        KernArgs args;
        args.scalar_i32(rows);
        args.pointer(in); args.pointer(gamma); args.pointer(beta); args.pointer(dst);
        // One row per wave, so eight rows per workgroup.
        launch(wide_out ? ln_rowwave_f32.function : ln_rowwave.function,
               (rows + 7) / 8, 1, args);
}


void forward() {
        // Patch embedding writes straight into the residual stream behind the
        // CLS and register tokens, so no copy is needed afterwards.
        { Stage stage("patch-embed matmul");
          matmul(wmma_patch, patch_rows, HIDDEN, image, PW("patch_w"), W("patch_b"), patched); }
        { Stage stage("embed scatter");
          KernArgs args;
          args.scalar_i32(rows);
          args.pointer(patched); args.pointer(W("prefix")); args.pointer(x);
          launch(scatter.function, rows, 1, args); }

        // Only the first norm1 stands alone; every later LayerNorm is fused
        // into the residual that precedes it, including across the layer
        // boundary.
        { Stage stage("layernorm");
          norm(x, W("l0_norm1_w"), W("l0_norm1_b"), h, false); }

        for (int layer = 0; layer < LAYERS; ++layer) {
            std::string p = "l" + std::to_string(layer) + "_";
            std::string next = "l" + std::to_string(layer + 1) + "_";

            { Stage stage("qkv matmul+rope");
              KernArgs args;
              args.scalar_i32(rows);
              args.pointer(h); args.pointer(PW(p + "qkv_w")); args.pointer(W(p + "qkv_b"));
              args.pointer(q); args.pointer(W("rope_cos")); args.pointer(W("rope_sin"));
              launch(qkv_rope.function, QKV / TILE, (rows + TILE - 1) / TILE, args); }

            { Stage stage("attention");
              KernArgs args;
              args.scalar_i32(rows);
              args.pointer(q); args.pointer(k); args.pointer(v); args.pointer(attn);
              // One wave32 per (16 query rows within an image, head).
              launchBlock(attn_online.function, batch * ((TOKENS + 15) / 16), HEADS, 32, args); }

            { Stage stage("o matmul+resid");
              matmul_resid(resid_o, rows, HIDDEN, attn, PW(p + "o_w"), W(p + "o_b"),
                           x, W(p + "ls1")); }
            { Stage stage("layernorm");
              norm(x, W(p + "norm2_w"), W(p + "norm2_b"), h, false); }

            // One workgroup accumulates both projections for the same output
            // columns, so the 2*INTERMEDIATE intermediate is never written.
            { Stage stage("gate/up+swiglu");
              KernArgs args;
              args.scalar_i32(rows);
              args.pointer(h); args.pointer(PW(p + "gateup_w")); args.pointer(W(p + "gateup_b"));
              args.pointer(act);
              launch(wmma_swiglu.function, INTERMEDIATE / TILE, (rows + TILE - 1) / TILE, args); }

            if (use_splitk) {
                Stage stage("down matmul+resid");
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(act); args.pointer(PW(p + "down_w"));
                args.pointer(W(p + "down_b")); args.pointer(partials);
                void *cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes,
                                HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size,
                                HIP_LAUNCH_PARAM_END };
                HIP_CHECK(hipModuleLaunchKernel(splitk_down.function, HIDDEN / TILE,
                                                (rows + TILE - 1) / TILE, SPLITS,
                                                THREADS, 1, 1, 0, nullptr, nullptr, cfg));
                KernArgs red;
                red.scalar_i32(rows);
                red.pointer(partials); red.pointer(W(p + "down_b"));
                red.pointer(x); red.pointer(W(p + "ls2"));
                launch(splitk_reduce.function, rows, 1, red);
            } else {
                Stage stage("down matmul+resid");
                matmul_resid(resid_down, rows, HIDDEN, act, PW(p + "down_w"),
                             W(p + "down_b"), x, W(p + "ls2"));
            }
            // Pairs with the *next* layer's norm1; the last layer's residual is
            // followed by the final norm below instead.
            if (layer + 1 < LAYERS) {
                Stage stage("layernorm");
                norm(x, W(next + "norm1_w"), W(next + "norm1_b"), h, false);
            }
        }
        // The final norm writes f32.
        { Stage stage("layernorm"); norm(x, W("norm_w"), W("norm_b"), out, true); }
}


// One batch: upload, run, download. `input` is batch x PATCHES x PATCH_K f32,
// `output` is batch x TOKENS x HIDDEN f32.
int session_run(const float *input, float *output, int b) {
    if (!set_batch(b)) return 64;
    HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)image, (void *)input,
                            size_t(patch_rows) * PATCH_K * sizeof(float)));
    forward();
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpyDtoH(output, (hipDeviceptr_t)out,
                            size_t(rows) * HIDDEN * sizeof(float)));
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// C API, for callers that hold the session open rather than paying init per
// batch. The same file still builds the CLI; main() below just drives these.
//
//   dinov3_init(weights_dir, kernels_dir, max_batch)  -> 0 on success
//   dinov3_run(input, output, batch)                  -> 0 on success
//
// `input`  is batch x 196 x 768 f32, contiguous, patchified.
// `output` is batch x 201 x 384 f32, written by the caller's allocation.
// Not thread-safe: one session, one device context, one caller at a time.
// ---------------------------------------------------------------------------
extern "C" int dinov3_init(const char *weights_dir, const char *kernels_dir, int max_batch) {
    return session_init(weights_dir, kernels_dir, max_batch);
}
extern "C" int dinov3_run(const float *input, float *output, int batch) {
    return session_run(input, output, batch);
}
extern "C" int dinov3_max_batch(void) { return session_max_batch; }

int main(int argc, char **argv) {
    std::string weights_dir = "build/weights", kernels_dir = "build/kernels";
    std::string input_path, output_path;
    int repeat = 1, batch = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(64); }
            return std::string(argv[++i]);
        };
        if (a == "--weights") weights_dir = next();
        else if (a == "--kernels") kernels_dir = next();
        else if (a == "--input") input_path = next();
        else if (a == "--output") output_path = next();
        else if (a == "--repeat") repeat = std::stoi(next());
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--profile") g_profile = true;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 64; }
    }

    if (int rc = session_init(weights_dir.c_str(), kernels_dir.c_str(), batch)) return rc;
    if (!set_batch(batch)) return 64;

    {
        // The input holds either one patchified image, which is replicated to
        // fill the batch (how the benchmarks are run), or exactly `batch` of
        // them, which is how inference is run.
        if (input_path.empty()) {
            fprintf(stderr, "--input is required: %d x %d f32 per image, either one "
                            "image or exactly --batch of them (tools/dinov3_loom.py "
                            "writes this)\n", PATCHES, PATCH_K);
            return 64;
        }
        const size_t per_image = size_t(PATCHES) * PATCH_K * sizeof(float);
        auto pixels = read_file(input_path);
        if (pixels.size() % per_image != 0) {
            fprintf(stderr, "%s is %zu bytes, not a multiple of %zu (%d patches x %d f32)\n",
                    input_path.c_str(), pixels.size(), per_image, PATCHES, PATCH_K);
            return 64;
        }
        const size_t supplied = pixels.size() / per_image;
        if (supplied != 1 && supplied != size_t(batch)) {
            fprintf(stderr, "%s holds %zu images; expected 1 (replicated) or %d (--batch)\n",
                    input_path.c_str(), supplied, batch);
            return 64;
        }
        for (int b = 0; b < batch; ++b) {
            const char *src = pixels.data() + (supplied == 1 ? 0 : size_t(b) * per_image);
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)(image + size_t(b) * PATCHES * PATCH_K),
                                    (void *)src, per_image));
        }
    }

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
        printf("stage breakdown over %d forward pass(es), %d image(s):\n",
               repeat + 1, (repeat + 1) * batch);
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
