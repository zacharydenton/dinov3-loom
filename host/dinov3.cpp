// DINOv3 ViT-S+/16 forward pass on gfx1151, entirely in Loom-compiled kernels.
//
//   dinov3 --weights build/weights --kernels build/kernels \
//          --input patchified.bin --output out.bin [--repeat 50]
//
// The input is a pre-patchified [196 x 768] f32 image; patch extraction is a
// pure reshape (16x16 patches, stride 16) and stays on the host.
#include <hip/hip_runtime.h>

#include "dinov3.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void hip_failure(hipError_t error, const char *expression,
                              const char *file, int line) {
    std::ostringstream message;
    message << file << ':' << line << ' ' << expression << ": "
            << hipGetErrorString(error);
    throw std::runtime_error(message.str());
}

#define HIP_CHECK(x)                                                           \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess)                                                  \
            hip_failure(e_, #x, __FILE__, __LINE__);                           \
    } while (0)

// --profile: synchronise after every launch and attribute the time to a stage.
struct Profiler {
    bool enabled = false;
    std::map<std::string, double> stage_us;
    std::map<std::string, int> stage_calls;
};

struct Stage {
    Profiler &profiler;
    const char *name;
    std::chrono::steady_clock::time_point start;
    Stage(Profiler &p, const char *n) : profiler(p), name(n) {
        if (profiler.enabled)
            start = std::chrono::steady_clock::now();
    }
    ~Stage() noexcept {
        if (!profiler.enabled)
            return;
        (void)hipDeviceSynchronize();
        auto us = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        profiler.stage_us[name] += us;
        profiler.stage_calls[name] += 1;
    }
};

constexpr int HIDDEN = 384;
constexpr int HEADS = 6;
// HEAD_DIM is implied by HIDDEN / HEADS; the kernels carry it as a config.
constexpr int LAYERS = 12;
constexpr int INTERMEDIATE = 1536;
constexpr int PREFIX = 5; // CLS + 4 registers
constexpr int PATCHES = 196;
constexpr int TOKENS = PREFIX + PATCHES; // per image
constexpr int PATCH_K = 768;             // 3 * 16 * 16
constexpr int THREADS = 256;
constexpr int TILE = 64;        // wide matmul workgroup tile
constexpr int QKV = 3 * HIDDEN; // fused q|k|v projection width
// The online attention kernel bounds its image index with a compiled
// max_images; beyond it the modulo would fold later images onto earlier ones.
constexpr int MAX_IMAGES = 64;
// Split the down projection at batch 1 to fill the machine's execution slots.
constexpr int SPLITS = 4;

struct Span {
    size_t offset;
    size_t count;
};

std::map<std::string, Span> read_manifest(const std::string &path) {
    std::map<std::string, Span> spans;
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot read " + path);
    std::string name;
    size_t offset, count;
    while (in >> name) {
        if (!(in >> offset >> count))
            throw std::runtime_error(path + ": malformed entry for '" + name +
                                     "'");
        if (!spans.emplace(name, Span{offset, count}).second)
            throw std::runtime_error(path + ": duplicate tensor '" + name +
                                     "'");
    }
    if (!in.eof())
        throw std::runtime_error(path + ": malformed manifest");
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
        if (__builtin_mul_overflow(entry.second.offset, element_bytes,
                                   &begin) ||
            __builtin_mul_overflow(entry.second.count, element_bytes,
                                   &extent) ||
            __builtin_add_overflow(begin, extent, &end) || end > blob_bytes) {
            std::ostringstream message;
            message << what << ": span '" << entry.first << "' (offset "
                    << entry.second.offset << ", count " << entry.second.count
                    << ") runs past the " << blob_bytes << "-byte blob";
            throw std::runtime_error(message.str());
        }
    }
}

void require(const std::map<std::string, Span> &spans, const std::string &name,
             size_t count, const char *what) {
    auto it = spans.find(name);
    if (it == spans.end())
        throw std::runtime_error(std::string(what) + ": missing tensor '" +
                                 name + "'");
    if (it->second.count != count) {
        std::ostringstream message;
        message << what << ": tensor '" << name << "' has " << it->second.count
                << " elements, the kernels expect " << count;
        throw std::runtime_error(message.str());
    }
}

std::vector<char> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("cannot read " + path);
    std::streampos end = in.tellg();
    if (end < 0)
        throw std::runtime_error("cannot determine size of " + path);
    std::vector<char> buffer(static_cast<size_t>(end));
    in.seekg(0);
    if (!buffer.empty() &&
        !in.read(buffer.data(), static_cast<std::streamsize>(buffer.size())))
        throw std::runtime_error("cannot read all of " + path);
    return buffer;
}

// One compiled kernel plus the module that owns it.
struct Kernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    Kernel() = default;
    Kernel(const Kernel &) = delete;
    Kernel &operator=(const Kernel &) = delete;
    ~Kernel() noexcept {
        if (module)
            (void)hipModuleUnload(module);
    }
    void load(const std::string &dir, const std::string &stem,
              const char *symbol) {
        HIP_CHECK(
            hipModuleLoad(&module, (dir + "/" + stem + ".hsaco").c_str()));
        HIP_CHECK(hipModuleGetFunction(&function, module, symbol));
    }
};

template <typename T> struct DeviceBuffer {
    T *data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() noexcept {
        if (data)
            (void)hipFree(data);
    }
    void allocate(size_t elements) {
        HIP_CHECK(
            hipMalloc(reinterpret_cast<void **>(&data), elements * sizeof(T)));
    }
    void allocate_bytes(size_t bytes) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&data), bytes));
    }
};

struct HipEvent {
    hipEvent_t value = nullptr;
    HipEvent() { HIP_CHECK(hipEventCreate(&value)); }
    HipEvent(const HipEvent &) = delete;
    HipEvent &operator=(const HipEvent &) = delete;
    ~HipEvent() noexcept {
        if (value)
            (void)hipEventDestroy(value);
    }
};

// Loom's AMDGPU kernarg ABI: scalars at natural alignment, then 8-byte
// pointers.
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

void launchBlock(hipFunction_t function, unsigned gx, unsigned gy,
                 unsigned block, KernArgs &args) {
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size,
                      HIP_LAUNCH_PARAM_END};
    HIP_CHECK(hipModuleLaunchKernel(function, gx, gy, 1, block, 1, 1, 0,
                                    nullptr, nullptr, config));
}

void launch(hipFunction_t function, unsigned gx, unsigned gy, KernArgs &args) {
    launchBlock(function, gx, gy, THREADS, args);
}

// One independently owned resident model. All mutable launch state, GPU
// allocations and loaded modules belong to this object; the C ABI below only
// exposes opaque pointers to it.
class Session {
    Profiler profiler;
    std::map<std::string, Span> spans, spans16;
    Kernel scatter, wmma_patch, qkv_rope, attn_online, resid_o, resid_down;
    Kernel wmma_swiglu, splitk_down, splitk_reduce, ln_rowwave, ln_rowwave_f32;
    DeviceBuffer<float> weights_storage;
    DeviceBuffer<unsigned short> weights16_storage;
    DeviceBuffer<float> x_storage, h_storage, q_storage, attn_storage;
    DeviceBuffer<float> act_storage, out_storage, image_storage,
        patched_storage;
    DeviceBuffer<float> partials_storage;
    float *weights = nullptr;
    unsigned short *weights16 = nullptr;
    float *x = nullptr, *h = nullptr, *q = nullptr, *k = nullptr, *v = nullptr;
    float *attn = nullptr, *act = nullptr, *out = nullptr, *image = nullptr;
    float *patched = nullptr, *partials = nullptr;
    int session_max_batch = 0;
    int batch = 0, rows = 0, patch_rows = 0;
    bool use_splitk = false;
    std::mutex run_mutex;

    float *W(const std::string &name) {
        auto it = spans.find(name);
        if (it == spans.end())
            throw std::runtime_error("missing weight " + name);
        return weights + it->second.offset;
    }
    unsigned short *W16(const std::string &name) {
        auto it = spans16.find(name);
        if (it == spans16.end())
            throw std::runtime_error("missing f16 weight " + name);
        return weights16 + it->second.offset;
    }
    // Projection weights come from the f16 blob under WMMA, the f32 one
    // otherwise.
    const void *PW(const std::string &name) { return (const void *)W16(name); }

  public:
    explicit Session(const char *weights_dir, const char *kernels_dir,
                     int max_batch, bool profile = false) {
        profiler.enabled = profile;
        session_init(weights_dir, kernels_dir, max_batch);
    }

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;

    int max_batch() const noexcept { return session_max_batch; }

    // Sizing for one call. Buffers are allocated for the session maximum, so
    // this only moves the launch bounds; nothing is reallocated between
    // batches.
    void set_batch(int b) {
        if (b < 1 || b > session_max_batch) {
            std::ostringstream message;
            message << "batch " << b << " out of range 1.."
                    << session_max_batch;
            throw std::invalid_argument(message.str());
        }
        batch = b;
        rows = batch * TOKENS;
        patch_rows = batch * PATCHES;
        use_splitk = rows <= TOKENS;
    }

    // Bring up the device, upload the weights and load the kernels. Buffers are
    // sized for `max_batch` and reused, so every later run is launches and two
    // copies -- which is the whole point of holding the session open: the work
    // below is ~100ms and used to be paid once per call.
    void session_init(const char *weights_dir_c, const char *kernels_dir_c,
                      int max_batch) {
        if (!weights_dir_c || !kernels_dir_c)
            throw std::invalid_argument(
                "weights_dir and kernels_dir must not be null");
        std::string weights_dir(weights_dir_c), kernels_dir(kernels_dir_c);
        if (max_batch < 1 || max_batch > MAX_IMAGES) {
            std::ostringstream message;
            message << "max_batch must be 1.." << MAX_IMAGES
                    << " (the kernels' compiled max_images)";
            throw std::invalid_argument(message.str());
        }
        session_max_batch = max_batch;
        const int max_rows = max_batch * TOKENS;
        const int max_patch_rows = max_batch * PATCHES;

        spans = read_manifest(weights_dir + "/manifest.txt");
        auto blob = read_file(weights_dir + "/weights.bin");
        spans16 = read_manifest(weights_dir + "/manifest_f16.txt");
        auto blob16 = read_file(weights_dir + "/weights_f16.bin");
        check_spans(spans, blob.size(), sizeof(float), "manifest.txt");
        check_spans(spans16, blob16.size(), sizeof(uint16_t),
                    "manifest_f16.txt");
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
                for (const char *v :
                     {"ls1", "ls2", "norm1_w", "norm1_b", "norm2_w", "norm2_b"})
                    require(spans, p + v, H, f32);
            }
            require(spans, "norm_w", H, f32);
            require(spans, "norm_b", H, f32);
        }

        HIP_CHECK(hipInit(0));
        weights_storage.allocate_bytes(blob.size());
        weights = weights_storage.data;
        HIP_CHECK(
            hipMemcpyHtoD((hipDeviceptr_t)weights, blob.data(), blob.size()));
        weights16_storage.allocate_bytes(blob16.size());
        weights16 = weights16_storage.data;
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights16, blob16.data(),
                                blob16.size()));

        // One configuration, so one kernel set. Earlier revisions carried f32,
        // non-fused and non-WMMA fallbacks selectable by flag; they multiplied
        // the dtype/layout combinations far past what was tested and several
        // were silently wrong. The measured-best path is the only path.
        scatter.load(kernels_dir, "embed_scatter", "dinov3_embed_scatter_f32");
        wmma_patch.load(kernels_dir, "wmma_k768_n384",
                        "dinov3_matmul_bias_f16_wmma");
        qkv_rope.load(kernels_dir, "qkv_rope_k384_n1152",
                      "dinov3_matmul_qkv_rope_f16_wmma");
        attn_online.load(kernels_dir, "attention_online_cf16",
                         "dinov3_attention_online_f16_wmma_cf16");
        resid_o.load(kernels_dir, "resid_k384_n384",
                     "dinov3_matmul_resid_f16_wmma");
        resid_down.load(kernels_dir, "resid_k1536_n384",
                        "dinov3_matmul_resid_f16_wmma");
        wmma_swiglu.load(kernels_dir, "wmma_swiglu_k384_n1536",
                         "dinov3_matmul_swiglu_f16_wmma");
        splitk_down.load(kernels_dir, "splitk_k1536_n384",
                         "dinov3_matmul_splitk_f16_wmma");
        splitk_reduce.load(kernels_dir, "splitk_reduce_n384",
                           "dinov3_splitk_reduce_f16");
        ln_rowwave.load(kernels_dir, "layernorm_rowwave",
                        "dinov3_layernorm_rowwave_f16");
        ln_rowwave_f32.load(kernels_dir, "layernorm_rowwave_f32out",
                            "dinov3_layernorm_rowwave_f32out");

        auto alloc = [](DeviceBuffer<float> &storage, float **pointer,
                        size_t floats) {
            storage.allocate(floats);
            *pointer = storage.data;
        };
        // Every intermediate is f16, so the float-typed pointers are half-width
        // views of these buffers; the allocations below are generous by exactly
        // that factor rather than being tightened, which keeps the arithmetic
        // here obvious.
        alloc(x_storage, &x, size_t(max_rows) * HIDDEN);
        alloc(h_storage, &h, size_t(max_rows) * HIDDEN);
        // A query tile may overhang the last image by up to 15 rows; the online
        // attention kernel masks those lanes but still forms the address.
        alloc(q_storage, &q, size_t(max_rows + 16) * QKV);
        alloc(attn_storage, &attn, size_t(max_rows) * HIDDEN);
        alloc(act_storage, &act, size_t(max_rows) * INTERMEDIATE);
        alloc(out_storage, &out, size_t(max_rows) * HIDDEN);
        alloc(patched_storage, &patched, size_t(max_patch_rows) * HIDDEN);
        alloc(image_storage, &image, size_t(max_patch_rows) * PATCH_K);
        // split-K runs only at batch 1, where rows == TOKENS
        alloc(partials_storage, &partials, size_t(SPLITS) * TOKENS * HIDDEN);
        // q/k/v are f16, so the head offsets are half as many float-sized
        // steps.
        const int qkv_step = HIDDEN / 2;
        k = q + qkv_step;
        v = q + 2 * qkv_step;
    }

    void matmul(Kernel &kernel, int m, int n, const float *a, const void *w,
                const float *bias, float *c, int tile_n = TILE) {
        KernArgs args;
        args.scalar_i32(m);
        args.pointer(a);
        args.pointer(w);
        args.pointer(bias);
        args.pointer(c);
        launch(kernel.function, n / tile_n, (m + TILE - 1) / TILE, args);
    }
    // A@W + bias scaled by lambda and accumulated into the residual stream.
    void matmul_resid(Kernel &kernel, int m, int n, const float *a,
                      const void *w, const float *bias, float *stream,
                      const float *lambda) {
        KernArgs args;
        args.scalar_i32(m);
        args.pointer(a);
        args.pointer(w);
        args.pointer(bias);
        args.pointer(stream);
        args.pointer(lambda);
        launch(kernel.function, n / TILE, (m + TILE - 1) / TILE, args);
    }
    // `dst` is f16, so callers treat it as opaque.
    void norm(const float *in, const float *gamma, const float *beta, void *dst,
              bool wide_out) {
        KernArgs args;
        args.scalar_i32(rows);
        args.pointer(in);
        args.pointer(gamma);
        args.pointer(beta);
        args.pointer(dst);
        // One row per wave, so eight rows per workgroup.
        launch(wide_out ? ln_rowwave_f32.function : ln_rowwave.function,
               (rows + 7) / 8, 1, args);
    }

    void forward() {
        // Patch embedding writes straight into the residual stream behind the
        // CLS and register tokens, so no copy is needed afterwards.
        {
            Stage stage(profiler, "patch-embed matmul");
            matmul(wmma_patch, patch_rows, HIDDEN, image, PW("patch_w"),
                   W("patch_b"), patched);
        }
        {
            Stage stage(profiler, "embed scatter");
            KernArgs args;
            args.scalar_i32(rows);
            args.pointer(patched);
            args.pointer(W("prefix"));
            args.pointer(x);
            launch(scatter.function, rows, 1, args);
        }

        // Only the first norm1 stands alone; every later LayerNorm is fused
        // into the residual that precedes it, including across the layer
        // boundary.
        {
            Stage stage(profiler, "layernorm");
            norm(x, W("l0_norm1_w"), W("l0_norm1_b"), h, false);
        }

        for (int layer = 0; layer < LAYERS; ++layer) {
            std::string p = "l" + std::to_string(layer) + "_";
            std::string next = "l" + std::to_string(layer + 1) + "_";

            {
                Stage stage(profiler, "qkv matmul+rope");
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(h);
                args.pointer(PW(p + "qkv_w"));
                args.pointer(W(p + "qkv_b"));
                args.pointer(q);
                args.pointer(W("rope_cos"));
                args.pointer(W("rope_sin"));
                launch(qkv_rope.function, QKV / TILE, (rows + TILE - 1) / TILE,
                       args);
            }

            {
                Stage stage(profiler, "attention");
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(q);
                args.pointer(k);
                args.pointer(v);
                args.pointer(attn);
                // One wave32 per (16 query rows within an image, head).
                launchBlock(attn_online.function, batch * ((TOKENS + 15) / 16),
                            HEADS, 32, args);
            }

            {
                Stage stage(profiler, "o matmul+resid");
                matmul_resid(resid_o, rows, HIDDEN, attn, PW(p + "o_w"),
                             W(p + "o_b"), x, W(p + "ls1"));
            }
            {
                Stage stage(profiler, "layernorm");
                norm(x, W(p + "norm2_w"), W(p + "norm2_b"), h, false);
            }

            // One workgroup accumulates both projections for the same output
            // columns, so the 2*INTERMEDIATE intermediate is never written.
            {
                Stage stage(profiler, "gate/up+swiglu");
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(h);
                args.pointer(PW(p + "gateup_w"));
                args.pointer(W(p + "gateup_b"));
                args.pointer(act);
                launch(wmma_swiglu.function, INTERMEDIATE / TILE,
                       (rows + TILE - 1) / TILE, args);
            }

            if (use_splitk) {
                Stage stage(profiler, "down matmul+resid");
                KernArgs args;
                args.scalar_i32(rows);
                args.pointer(act);
                args.pointer(PW(p + "down_w"));
                args.pointer(W(p + "down_b"));
                args.pointer(partials);
                void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size,
                               HIP_LAUNCH_PARAM_END};
                HIP_CHECK(hipModuleLaunchKernel(
                    splitk_down.function, HIDDEN / TILE,
                    (rows + TILE - 1) / TILE, SPLITS, THREADS, 1, 1, 0, nullptr,
                    nullptr, cfg));
                KernArgs red;
                red.scalar_i32(rows);
                red.pointer(partials);
                red.pointer(W(p + "down_b"));
                red.pointer(x);
                red.pointer(W(p + "ls2"));
                launch(splitk_reduce.function, rows, 1, red);
            } else {
                Stage stage(profiler, "down matmul+resid");
                matmul_resid(resid_down, rows, HIDDEN, act, PW(p + "down_w"),
                             W(p + "down_b"), x, W(p + "ls2"));
            }
            // Pairs with the *next* layer's norm1; the last layer's residual is
            // followed by the final norm below instead.
            if (layer + 1 < LAYERS) {
                Stage stage(profiler, "layernorm");
                norm(x, W(next + "norm1_w"), W(next + "norm1_b"), h, false);
            }
        }
        // The final norm writes f32.
        {
            Stage stage(profiler, "layernorm");
            norm(x, W("norm_w"), W("norm_b"), out, true);
        }
    }

    // One batch: upload, run, download. `input` is batch x PATCHES x PATCH_K
    // f32, `output` is batch x TOKENS x HIDDEN f32.
    void upload(const float *input, size_t input_elements, int b) {
        if (!input)
            throw std::invalid_argument("input must not be null");
        set_batch(b);
        const size_t expected = size_t(patch_rows) * PATCH_K;
        if (input_elements != expected) {
            std::ostringstream message;
            message << "input has " << input_elements << " f32 elements; batch "
                    << b << " requires exactly " << expected;
            throw std::invalid_argument(message.str());
        }
        // Attention operates in 16-query tiles. Clear the tail of the packed
        // f16 QKV buffer so a smaller call cannot observe padding left by a
        // preceding larger call in the same resident session.
        const size_t qkv_bytes = size_t(rows) * QKV * sizeof(uint16_t);
        HIP_CHECK(hipMemset(reinterpret_cast<char *>(q) + qkv_bytes, 0,
                            size_t(16) * QKV * sizeof(uint16_t)));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)image, (void *)input,
                                expected * sizeof(float)));
    }

    void synchronize() { HIP_CHECK(hipDeviceSynchronize()); }

    void download(float *output, size_t output_elements) {
        if (!output)
            throw std::invalid_argument("output must not be null");
        const size_t expected = size_t(rows) * HIDDEN;
        if (output_elements != expected) {
            std::ostringstream message;
            message << "output has " << output_elements
                    << " f32 elements; batch " << batch << " requires exactly "
                    << expected;
            throw std::invalid_argument(message.str());
        }
        HIP_CHECK(hipMemcpyDtoH(output, (hipDeviceptr_t)out,
                                expected * sizeof(float)));
    }

    void session_run(const float *input, size_t input_elements, float *output,
                     size_t output_elements, int b) {
        std::lock_guard<std::mutex> lock(run_mutex);
        if (!output)
            throw std::invalid_argument("output must not be null");
        set_batch(b);
        const size_t expected_output = size_t(rows) * HIDDEN;
        if (output_elements != expected_output) {
            std::ostringstream message;
            message << "output has " << output_elements
                    << " f32 elements; batch " << batch << " requires exactly "
                    << expected_output;
            throw std::invalid_argument(message.str());
        }
        upload(input, input_elements, b);
        forward();
        synchronize();
        download(output, output_elements);
    }

    void print_profile(int forwards) const {
        double total = 0.0;
        for (const auto &entry : profiler.stage_us)
            total += entry.second;
        std::vector<std::pair<double, std::string>> ordered;
        for (const auto &entry : profiler.stage_us)
            ordered.push_back({entry.second, entry.first});
        std::sort(ordered.rbegin(), ordered.rend());
        printf("stage breakdown over %d forward pass(es), %d image(s):\n",
               forwards, forwards * batch);
        for (const auto &entry : ordered)
            printf("  %-20s %9.3f ms  %5.1f%%  (%d launches)\n",
                   entry.second.c_str(), entry.first / 1000.0,
                   total ? 100.0 * entry.first / total : 0.0,
                   profiler.stage_calls.at(entry.second));
        printf("  %-20s %9.3f ms\n", "total", total / 1000.0);
    }
};

} // namespace

struct dinov3_session {
    Session value;
    dinov3_session(const char *weights_dir, const char *kernels_dir,
                   int max_batch)
        : value(weights_dir, kernels_dir, max_batch) {}
};

namespace {

void write_error(char *error, size_t capacity, const char *message) noexcept {
    if (!error || capacity == 0)
        return;
    std::snprintf(error, capacity, "%s", message ? message : "unknown error");
}

void clear_error(char *error, size_t capacity) noexcept {
    if (error && capacity)
        error[0] = '\0';
}

#ifndef DINOV3_LIBRARY
int parse_integer(const std::string &option, const std::string &text) {
    errno = 0;
    char *end = nullptr;
    long value = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        value < INT_MIN || value > INT_MAX)
        throw std::invalid_argument(option + " must be an integer, got '" +
                                    text + "'");
    return static_cast<int>(value);
}

int cli_main(int argc, char **argv) {
    std::string weights_dir = "build/weights", kernels_dir = "build/kernels";
    std::string input_path, output_path;
    int repeat = 1, batch = 1;
    bool profile = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc)
                throw std::invalid_argument(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--weights")
            weights_dir = next();
        else if (a == "--kernels")
            kernels_dir = next();
        else if (a == "--input")
            input_path = next();
        else if (a == "--output")
            output_path = next();
        else if (a == "--repeat")
            repeat = parse_integer(a, next());
        else if (a == "--batch")
            batch = parse_integer(a, next());
        else if (a == "--profile")
            profile = true;
        else
            throw std::invalid_argument("unknown option " + a);
    }

    if (batch < 1 || batch > MAX_IMAGES)
        throw std::invalid_argument("--batch must be 1..64");
    if (repeat < 1)
        throw std::invalid_argument("--repeat must be at least 1");
    if (input_path.empty()) {
        std::ostringstream message;
        message
            << "--input is required: " << PATCHES << " x " << PATCH_K
            << " f32 per image, either one image or exactly --batch of them";
        throw std::invalid_argument(message.str());
    }

    // The input holds either one patchified image, replicated for benchmarks,
    // or exactly `batch` images for inference.
    const size_t per_image = size_t(PATCHES) * PATCH_K * sizeof(float);
    auto pixels = read_file(input_path);
    if (pixels.size() % per_image != 0) {
        std::ostringstream message;
        message << input_path << " is " << pixels.size()
                << " bytes, not a multiple of " << per_image << " (" << PATCHES
                << " patches x " << PATCH_K << " f32)";
        throw std::invalid_argument(message.str());
    }
    const size_t supplied = pixels.size() / per_image;
    if (supplied != 1 && supplied != size_t(batch)) {
        std::ostringstream message;
        message << input_path << " holds " << supplied
                << " images; expected 1 (replicated) or " << batch
                << " (--batch)";
        throw std::invalid_argument(message.str());
    }
    std::vector<float> input(size_t(batch) * PATCHES * PATCH_K);
    for (int b = 0; b < batch; ++b) {
        const char *source =
            pixels.data() + (supplied == 1 ? 0 : size_t(b) * per_image);
        std::memcpy(input.data() + size_t(b) * PATCHES * PATCH_K, source,
                    per_image);
    }

    Session session(weights_dir.c_str(), kernels_dir.c_str(), batch, profile);
    session.upload(input.data(), input.size(), batch);
    session.forward();
    session.synchronize();

    if (repeat > 1) {
        HipEvent start, stop;
        HIP_CHECK(hipEventRecord(start.value, nullptr));
        for (int r = 0; r < repeat; ++r)
            session.forward();
        HIP_CHECK(hipEventRecord(stop.value, nullptr));
        session.synchronize();
        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start.value, stop.value));
        long long images = static_cast<long long>(repeat) * batch;
        printf("{\"batch\": %d, \"images\": %lld, \"total_ms\": %.3f, "
               "\"ms_per_image\": %.4f, \"img_per_s\": %.2f}\n",
               batch, images, elapsed, elapsed / images,
               1000.0 * images / elapsed);
    }

    if (profile)
        session.print_profile(1 + (repeat > 1 ? repeat : 0));

    if (!output_path.empty()) {
        std::vector<float> output(size_t(batch) * TOKENS * HIDDEN);
        session.download(output.data(), output.size());
        std::ofstream file(output_path, std::ios::binary);
        if (!file)
            throw std::runtime_error("cannot write " + output_path);
        file.write(reinterpret_cast<const char *>(output.data()),
                   static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!file)
            throw std::runtime_error("cannot write all of " + output_path);
    }
    return DINOV3_OK;
}
#endif

} // namespace

extern "C" uint32_t dinov3_abi_version(void) { return DINOV3_ABI_VERSION; }

extern "C" int dinov3_create(const char *weights_dir, const char *kernels_dir,
                             int max_batch, dinov3_session **out_session,
                             char *error, size_t error_capacity) {
    clear_error(error, error_capacity);
    if (!out_session) {
        write_error(error, error_capacity, "out_session must not be null");
        return DINOV3_INVALID_ARGUMENT;
    }
    *out_session = nullptr;
    try {
        *out_session = new dinov3_session(weights_dir, kernels_dir, max_batch);
        return DINOV3_OK;
    } catch (const std::invalid_argument &failure) {
        write_error(error, error_capacity, failure.what());
        return DINOV3_INVALID_ARGUMENT;
    } catch (const std::exception &failure) {
        write_error(error, error_capacity, failure.what());
        return DINOV3_ERROR;
    } catch (...) {
        write_error(error, error_capacity, "unknown C++ exception");
        return DINOV3_ERROR;
    }
}

extern "C" int dinov3_run(dinov3_session *session, const float *input,
                          size_t input_elements, float *output,
                          size_t output_elements, int batch, char *error,
                          size_t error_capacity) {
    clear_error(error, error_capacity);
    try {
        if (!session)
            throw std::invalid_argument("session must not be null");
        session->value.session_run(input, input_elements, output,
                                   output_elements, batch);
        return DINOV3_OK;
    } catch (const std::invalid_argument &failure) {
        write_error(error, error_capacity, failure.what());
        return DINOV3_INVALID_ARGUMENT;
    } catch (const std::exception &failure) {
        write_error(error, error_capacity, failure.what());
        return DINOV3_ERROR;
    } catch (...) {
        write_error(error, error_capacity, "unknown C++ exception");
        return DINOV3_ERROR;
    }
}

extern "C" int dinov3_max_batch(const dinov3_session *session) {
    return session ? session->value.max_batch() : 0;
}

extern "C" void dinov3_destroy(dinov3_session *session) { delete session; }

#ifndef DINOV3_LIBRARY
int main(int argc, char **argv) {
    try {
        return cli_main(argc, argv);
    } catch (const std::invalid_argument &failure) {
        fprintf(stderr, "%s\n", failure.what());
        return DINOV3_INVALID_ARGUMENT;
    } catch (const std::exception &failure) {
        fprintf(stderr, "%s\n", failure.what());
        return DINOV3_ERROR;
    } catch (...) {
        fprintf(stderr, "unknown C++ exception\n");
        return DINOV3_ERROR;
    }
}
#endif
