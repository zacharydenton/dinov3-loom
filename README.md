# dinov3-loom

DINOv3 ViT-S+/16 kernels written in **Loom**, AMD's new compiler substrate from
[ROCm/hrx-system](https://github.com/ROCm/hrx-system), targeting the Radeon 8060S
(gfx1151) in a Strix Halo APU.

## Why

`torch.compile` gets DINOv3-ViT-L to ~88 img/s on this iGPU and then stops: the
plateau is batch-insensitive, and `max-autotune` **fails outright** because Triton's
attention BMM wants 96 KB of LDS where gfx1151 has 64. That is a Triton tiling limit,
not a hardware one. Loom lets you write the kernel directly — and the same compiler
already runs a WMMA flash-attention kernel on this chip fast enough to beat RADV
Vulkan by 45% on prefill in llama.cpp.

The question this repo exists to answer: **is 88 img/s the silicon, or the toolchain?**

It was the toolchain. A hand-written Loom forward pass reaches **1303 img/s** on
ViT-S+/16 at batch 32, ahead of `max-autotune` (1280.7), at cosine 0.99998
against transformers.

## Model

`facebook/dinov3-vits16plus-pretrain-lvd1689m` — hidden 384, 12 layers, 6 heads
(head dim 64), MLP 1536 **gated (SiLU/SwiGLU)**, patch 16, 4 register tokens, RoPE
(theta 100), LayerScale, LayerNorm eps 1e-5. At 224×224 that is 196 + 1 + 4 = **201
tokens**.

Note the "+" in ViT-S+ is the gated MLP: ViT-S/16 and ViT-S+/16 have identical
dimensions, but S+ uses `use_gated_mlp: true` with SiLU where S uses plain GELU.

## Status

Complete forward pass, validated, benchmarked. Six kernels:

| kernel | what |
| --- | --- |
| `layernorm_f32` | LayerNorm over the last dim; the one DINOv3 op ggml-hrx's corpus lacks |
| `matmul_bias_f32` | `C = A @ W^T + bias`, W row-major `[N, K]` — a torch `nn.Linear` weight verbatim. 64x64 tile, 4x4 register micro-tile |
| `matmul_bias_f32_narrow` | same, 64x32 tile, for the N=384 projections that cannot fill 40 CUs at 64 wide |
| `rope_2d_f32` | DINOv3 axial RoPE, patch tokens only, race-free in place |
| `attention_f32` | one workgroup per (token, head); two-stage LDS softmax |
| `swiglu_f32` | `silu(gate) * up` — the "+" in ViT-S+ |
| `residual_scale_f32` | LayerScale and residual fused, in place on the stream |

`host/dinov3` chains them into the 12-layer model: 87 launches per image,
weights uploaded once, patch extraction on the host (16x16 stride 16 is a pure
reshape). All nine kernel configurations compile in **106 ms total**.

## Correctness

`tools/validate.py` runs the model against HF transformers on three images:

```
  PASS image 0: cosine full=1.0000000000 cls=1.0000000000 mean-patch=1.0000000000 max_abs=2.81e-05
  PASS image 1: cosine full=1.0000000000 cls=1.0000000000 mean-patch=1.0000000000 max_abs=2.81e-05
  PASS image 2: cosine full=1.0000000000 cls=1.0000000000 mean-patch=1.0000000000 max_abs=4.26e-05
```

For scale: xdna-vision gates DINOv3 at cosine > 0.997.

`tools/reference.py` is an independent float64 NumPy implementation of the whole
architecture, agreeing with transformers to 9.3e-06. Every individual kernel is
graded against it, not against torch, so a kernel bug cannot hide behind a
matching bug in the harness.

## Benchmark

Radeon 8060S (gfx1151), torch 2.13.0 + ROCm, best of 3 interleaved rounds
(`tools/benchmark.py`) on an idle GPU. Interleaving means both sides see the
same conditions.

> **This table predates two changes and understates the current code.** f16
> attention/MLP branches measured **1.133x** at batch 32, and split-K on the down
> projection measured **1.179x** at batch 1 (which puts batch 1 at ~626 img/s,
> past torch max-autotune's 594.3). Both are A/B ratios measured interleaved; the
> absolute figures below have not been re-measured on an idle box since, so they
> are left as they were rather than scaled. Re-run `tools/benchmark.py` on a
> quiet machine to refresh them.

| configuration | img/s | vs best Loom |
| --- | ---: | ---: |
| **loom fp16, batch 32** | **1303.3** | **1.00x** |
| torch max-autotune fp16, batch 64 | 1280.7 | 0.98x |
| **loom fp16, batch 64** | **1220.4** | 0.94x |
| **loom fp16, batch 8** | **1215.2** | 0.93x |
| torch compile fp16, batch 64 | 1131.2 | 0.87x |
| torch eager fp16, batch 64 | 836.6 | 0.64x |
| torch max-autotune fp16, batch 1 | 594.3 | 0.46x |
| **loom fp16, batch 1** | **543.5** | 0.42x |
| torch compile fp16, batch 1 | 385.8 | 0.30x |
| torch eager fp16, batch 1 | 288.5 | 0.22x |
| torch max-autotune fp32, batch 64 | 255.9 | 0.20x |
| **loom fp32, batch 32** | **230.8** | 0.18x |
| torch compile fp32, batch 64 | 174.2 | 0.13x |
| torch eager fp32, batch 64 | 163.0 | 0.13x |
| **loom fp32, batch 1** | **130.1** | 0.10x |
| torch eager fp32, batch 1 | 115.2 | 0.09x |

Read it honestly:

- **It is faster than every torch configuration measured, at batch 32.** The
  margin over `max-autotune` is 1.8% -- a win, but a narrow one, and closer to
  run-to-run variance than the table's ordering suggests.
- **Batch 1 used to lose to `max-autotune`** (543.5 vs 594.3) and no longer
  does: split-K on the down projection took it to ~626 img/s. An earlier version
  of this table claimed batch 1 beat every torch configuration when it did not;
  that is now true, but it was not then.
- **The MLP is now the bottleneck**: gate/up+swiglu 30.1% of the forward pass,
  the down projection 22.3%. Attention, which used to be 36.4%, is 10.2%.

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
(online-softmax attention) img/s; `docs/notes.md` has what each step was worth.

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
{"batch": 32, "images": 1280, "total_ms": 907.125, "ms_per_image": 0.7087, "img_per_s": 1411.05}
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

`tools/dinov3_loom.py` takes the same `pixel_values` an HF image processor
produces and returns the same array as `last_hidden_state`. The swap is two
lines:

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
