# dinov3-loom

DINOv3 ViT-S+/16 kernels written in **Loom**, AMD's new compiler substrate from
[ROCm/hrx-system](https://github.com/ROCm/hrx-system), targeting the Radeon 8060S
(gfx1151) in a Strix Halo APU.

## Why

`torch.compile` is the fastest way to run DINOv3 ViT-S+/16 on this iGPU, and
`max-autotune` -- Triton searching its whole tile space -- is the fastest
`torch.compile` gets. That is the bar. It is also a bar set by a general-purpose
compiler that has never seen gfx1151's 64 KB of LDS, its 32-wave cap, or its
appetite for exactly three workgroups per CU.

The question this repo exists to answer: **is that bar the silicon, or the
toolchain?** Loom lets you write every kernel directly, so the way to find out is
to write them.

It was the toolchain. The hand-written forward pass is ahead of `max-autotune` at
both ends measured -- **1.12x at batch 32, 1.08x at batch 1** -- at cosine 0.99998
against transformers. Numbers below.

## Model

`facebook/dinov3-vits16plus-pretrain-lvd1689m` — hidden 384, 12 layers, 6 heads
(head dim 64), MLP 1536 **gated (SiLU/SwiGLU)**, patch 16, 4 register tokens, RoPE
(theta 100), LayerScale, LayerNorm eps 1e-5. At 224×224 that is 196 + 1 + 4 = **201
tokens**.

Note the "+" in ViT-S+ is the gated MLP: ViT-S/16 and ViT-S+/16 have identical
dimensions, but S+ uses `use_gated_mlp: true` with SiLU where S uses plain GELU.

## Status

Complete forward pass, validated, benchmarked. One inference path, eleven
kernels, every elementwise op except LayerNorm fused into the matmul that feeds
it:

| kernel | what |
| --- | --- |
| `matmul_bias_f16_wmma` | patch embedding: `A @ W^T + bias`, f32 patches in, f16 weights, WMMA, f32 accumulate |
| `embed_scatter_f32` | places CLS, the 4 register tokens and the patches into the residual stream |
| `layernorm_rowwave_f16` / `_f32out` | LayerNorm, one row per wave, no LDS, no barriers; f16 out for the model, f32 for the final norm |
| `matmul_qkv_rope_f16_wmma` | the QKV projection with DINOv3's axial RoPE in the epilogue |
| `attention_online_f16_wmma_cf16` | online-softmax attention, one wave32 per (16 queries, head), K and V fragment-loaded from global |
| `matmul_resid_f16_wmma` | o and down projections with `x += lambda * (A@W + b)` -- LayerScale and the residual add -- in the epilogue |
| `matmul_swiglu_f16_wmma` | gate and up accumulated by the same workgroup, `silu(gate) * up` in the epilogue; the "+" in ViT-S+ |
| `matmul_splitk_f16_wmma` + `splitk_reduce_f16` | the down projection at batch 1, where 24 workgroups cannot fill 40 CUs |

`host/dinov3` chains them into the 12-layer model: 87 launches per image,
weights uploaded once, patch extraction on the host (16x16 stride 16 is a pure
reshape). The residual stream, every activation and every weight is f16;
accumulation is f32 throughout. All eleven configurations compile in about
120 ms total -- Loom emits an HSACO in ~2 ms with no LLVM in the loop.

## Correctness

`tools/validate.py` runs the model against HF transformers on three images:

```
  PASS image 0: cosine full=0.9999821888 cls=0.9999852319 mean-patch=0.9999833401 max_abs=2.41e-02
  PASS image 1: cosine full=0.9999807011 cls=0.9999844174 mean-patch=0.9999821151 max_abs=2.87e-02
  PASS image 2: cosine full=0.9999860516 cls=0.9999889179 mean-patch=0.9999875116 max_abs=2.29e-02
```

For scale: xdna-vision gates DINOv3 at cosine > 0.997. `tools/test_batch.py`
repeats this for four distinct images in one batched call.

`tools/reference.py` is an independent float64 NumPy implementation of the whole
architecture, agreeing with transformers to 9.3e-06. Every kernel that ships
has a unit test graded against it, not against torch, so a kernel bug cannot
hide behind a matching bug in the harness. `scripts/test.sh` runs them all.

## Benchmark

Radeon 8060S (gfx1151), torch 2.13.0 + ROCm, best of 3 interleaved rounds
(`tools/benchmark.py`; raw output in `docs/benchmark-2026-09-04-release.txt`).
Interleaving means both sides see the same conditions -- here an idle GPU and a
10-core CPU job resident throughout, which depresses both by a similar amount.
Torch's `max-autotune` figure has reproduced to within 0.5% across three runs on
three different days, so the comparison is stable even if the absolutes drift.

| configuration | img/s | vs best Loom |
| --- | ---: | ---: |
| **loom fp16, batch 32** | **1431.0** | **1.00x** |
| **loom fp16, batch 64** | **1324.9** | 0.93x |
| torch max-autotune fp16, batch 64 | 1280.7 | 0.89x |
| torch compile fp16, batch 64 | 1081.1 | 0.76x |
| **loom fp16, batch 8** | **1061.9** | 0.74x |
| torch eager fp16, batch 64 | 736.6 | 0.51x |
| **loom fp16, batch 1** | **645.9** | 0.45x |
| torch max-autotune fp16, batch 1 | 599.4 | 0.42x |
| torch compile fp16, batch 1 | 380.6 | 0.27x |
| torch eager fp16, batch 1 | 276.3 | 0.19x |
| torch max-autotune fp32, batch 64 | 241.9 | 0.17x |
| torch eager fp32, batch 1 | 114.6 | 0.08x |

Read it honestly:

- **Batch 32 is 1.12x over torch's best configuration** at any batch size. The
  margin is real -- it has held across three independent runs -- but it is not
  large, and torch's number is a compiler's output with nobody having touched a
  kernel.
- **Batch 1 is 1.08x over `max-autotune` at batch 1.** That took split-K on the
  down projection; before it the runner launched 24 workgroups against ~120
  slots and lost. Batch 1 is still throughput-limited by the subprocess-per-call
  design of the Python API, not by the kernels.
- **Batch 64 is slower than batch 32** on this repo and it is not the cache: the
  gate/up intermediate spills the 32 MB L3 from batch 28 up, and a sweep showed
  no cliff there. It has not been chased.
- Loom has no f32 path any more. The f32 rows are torch's, kept because they
  show what precision costs on this part: `max-autotune` fp32 at batch 64 is
  slower than Loom fp16 at batch 1.

Accuracy: cosine **0.99998** against transformers on the full fp16 path (f16
weights, f16 activations, f16 attention probabilities, f16 residual stream, f32
accumulation throughout). xdna-vision gates DINOv3 at 0.997.

There is **one inference path**. Earlier revisions carried f32, non-fused and
non-WMMA fallbacks behind flags; they multiplied the dtype and layout
combinations far past what was tested, and several combinations were silently
wrong. The runner now takes only `--weights`, `--kernels`, `--input`,
`--output`, `--batch`, `--repeat` and `--profile`.

Where the time goes at batch 32:

```
gate/up matmul + swiglu  35.4%
down matmul + residual   24.1%
qkv matmul + rope        16.3%
attention                12.9%
o matmul + residual       6.3%
layernorm                 3.0%
patch embed               1.7%
embed scatter             0.3%
```

Every elementwise stage except LayerNorm has been fused into the matmul that
produces its input: RoPE into the qkv epilogue, SwiGLU into gate/up, the
residual add and LayerScale into o and down.

The run was 22 -> 46 -> 80 -> 94 -> 121 -> 226 (f32) -> 468 (WMMA) -> 667
(flash attention) -> 758 (f16 activations) -> 915 (kernel fusion) -> 1303
(online-softmax attention) -> 1431 (f16 residual stream, RoPE and residual
folded into epilogues, one-row-per-wave LayerNorm) img/s; `docs/notes.md` has what each step was worth.

## Layout

```
kernels/    .loom sources for the eleven kernels the model launches
host/       loomrun.cpp — launches one compiled kernel and dumps its buffers
tools/      Python drivers: compile, run, compare against NumPy f64
scripts/    env.sh (toolchain + runtime paths), test.sh
docs/       notes.md — the compiler and ABI landmines found so far
```

## Running it

Needs the Loom toolchain from [ROCm/hrx-system](https://github.com/ROCm/hrx-system)
(`scripts/env.sh` points at the build) and ROCm for `hipcc`. Python dependencies
are in `requirements.txt`; the model is fetched from the Hub on first use, or set
`DINOV3_SNAPSHOT` to an existing snapshot directory.

```console
$ pip install -r requirements.txt
$ source scripts/env.sh
$ python3 tools/export_weights.py          # 115 MB blob + manifest
$ ./scripts/build_kernels.sh               # eleven HSACOs
$ /opt/rocm/bin/hipcc -O2 -o host/dinov3 host/dinov3.cpp
$ /opt/rocm/bin/hipcc -O2 -o host/loomrun host/loomrun.cpp

$ ./scripts/test.sh                        # kernels, error paths, end-to-end
$ ./host/dinov3 --input build/patchified.bin --batch 32 --repeat 40
{"batch": 32, "images": 1280, "total_ms": 886.043, "ms_per_image": 0.6922, "img_per_s": 1444.62}
$ python3 tools/benchmark.py               # vs torch, interleaved
```

`scripts/test.sh` is the one test command: it builds the kernels, checks the
generated kernel still matches its generator, runs every unit test, exercises
the runners' error paths, and finishes with the end-to-end comparison against
transformers. `--quick` skips the last step, which is the only one needing
torch.

Per-kernel tests go through `host/loomrun`, which launches one compiled HSACO
and dumps its buffers — `tools/test_layernorm.py`, `tools/test_matmul_wmma.py`,
`tools/test_fused_matmuls.py`, `tools/test_attention_online.py`. Each compares
against float64 NumPy from `tools/reference.py`, which is written independently
of the kernels and agrees with transformers to 9.3e-06, so a kernel bug cannot
hide behind a matching bug in the harness.

On Arch, see `docs/notes.md` — the distro's `hsa-rocr` aborts under HRX and
`scripts/env.sh` shims around it.

## Replacing a PyTorch DINOv3

`dinov3_loom.py` takes the same `pixel_values` an HF image processor produces
and returns the same array as `last_hidden_state`. It is a single module at the
repository root: `pip install -e .` makes it importable from anywhere, or run
from the root and it is on the path already. The swap is two lines:

```python
# before
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dinov3-vits16plus-pretrain-lvd1689m").eval().cuda()
with torch.no_grad():
    tokens = model(pixel_values=pixel_values.cuda()).last_hidden_state.cpu().numpy()

# after
from dinov3_loom import DINOv3Loom
model = DINOv3Loom()
tokens = model(pixel_values)          # numpy in, numpy out; no torch needed
```

Both give `(B, 201, 384)`: token 0 is CLS, 1..4 are the register tokens, 5.. are
the 196 patches. A complete example, with the preprocessing an image processor
would normally do:

```python
import numpy as np
from PIL import Image
from dinov3_loom import DINOv3Loom

MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

def preprocess(paths):
    out = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
        pixels = np.asarray(image, np.float32).transpose(2, 0, 1) / 255.0
        out.append((pixels - MEAN) / STD)
    return np.stack(out)

model = DINOv3Loom()
tokens = model(preprocess(["a.jpg", "b.jpg"]))   # (2, 201, 384)

cls = tokens[:, 0]                               # (2, 384) image embedding
patches = tokens[:, 5:].reshape(-1, 14, 14, 384) # (2, 14, 14, 384) dense features
```

`model.cls(...)` and `model.patch_mean(...)` are shorthands for the two things
people usually take off DINOv3.

### What to know before swapping

- **Accuracy is 0.99998 cosine against transformers**, not bitwise equality.
  That is well inside the noise for retrieval, clustering and linear probes; if
  you are comparing embeddings against a store built with the torch model,
  rebuild the store rather than mixing the two.
- **The shape is fixed at 224x224, ViT-S+/16.** The kernels are compiled for
  201 tokens, 384 hidden, 6 heads. A different resolution or variant needs
  different `--config` values in `scripts/build_kernels.sh`, and the RoPE and
  attention kernels assume head_dim 64.
- **Batches above 64 are split automatically**, because the kernels bound their
  image index at a compiled `max_images` of 64.
- **Each call is a subprocess**, costing a few milliseconds plus two file
  copies. That is fine amortised over a batch and wasteful for single images in
  a tight loop; batch 32 is where the throughput number comes from.
- **gfx1151 only.** The HSACOs are compiled for that target; `LOOM_TARGET`
  changes it, but nothing else here has been measured on another chip.

## Two things that will bite you

1. **Never put `where [range(...)]` on a launch argument** if you plan to launch the
   kernel from an ordinary HIP or CUDA host. It compiles, then miscompiles into an
   out-of-bounds store. Bisected and written up in `docs/notes.md`.
2. **`config.get` values are specialization constants**, bound with
   `loom-compile --config=k=v`, not runtime arguments. One HSACO per shape — which is
   fine, since compiling one takes about 2 ms.

## Licence

Apache-2.0, matching hrx-system. Full text in [LICENSE](LICENSE).
