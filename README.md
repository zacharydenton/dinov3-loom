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

`host/dinov3` chains them into the 12-layer model: 170 launches per image,
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
weights, f16 matmul-facing activations, f16 attention probabilities, f32
accumulation throughout). The pure f32 path is exact to ten digits. xdna-vision
gates DINOv3 at 0.997. `--f32`, `--no-flash`, `--f32-act`, `--f32-qkv` and
`--no-online-attn` select the slower, more accurate paths independently.

Where the time goes at batch 32, everything on:

```
gate/up matmul + swiglu  30.1%
down matmul              22.3%
residual+norm            13.2%
qkv matmul               12.9%
attention                10.2%
o matmul                  5.4%
rope                      2.9%
patch embed               1.6%
```

The run was 22 -> 46 -> 80 -> 94 -> 121 -> 226 (f32) -> 468 (WMMA) -> 667
(flash attention) -> 758 (f16 activations) -> 915 (kernel fusion) -> 1303
(online-softmax attention) img/s; `docs/notes.md` has what each step was worth.

## Layout

```
kernels/    .loom sources, one op per file, each with check.case blocks
host/       loomrun.cpp — launches one compiled kernel and dumps its buffers
tools/      Python drivers: compile, run, compare against NumPy f64
scripts/    env.sh (toolchain + runtime paths), test.sh
docs/       notes.md — the compiler and ABI landmines found so far
```

## Running it

```console
$ source scripts/env.sh
$ python3 tools/export_weights.py          # 115 MB blob + manifest
$ ./scripts/build_kernels.sh               # nine HSACOs, ~106 ms
$ /opt/rocm/bin/hipcc -O2 -o host/dinov3 host/dinov3.cpp
$ /opt/rocm/bin/hipcc -O2 -o host/loomrun host/loomrun.cpp

$ python3 tools/validate.py                # vs HF transformers
$ python3 tools/benchmark.py               # vs torch, interleaved
$ ./host/dinov3 --input build/patchified.bin --repeat 30
{"images": 30, "total_ms": 319.304, "ms_per_image": 10.643, "img_per_s": 93.95}
$ ./host/dinov3 --input build/patchified.bin --repeat 15 --profile
```

Per-kernel tests go through `host/loomrun`, which launches one compiled HSACO
and dumps its buffers: `python3 tools/test_matmul.py`, `tools/test_layernorm.py`,
`tools/test_kernels.py`. Each compares against float64 NumPy.

Prerequisites: a built `hrx-system` (`$HRX_BUILD`, default `~/code/hrx-system/build-cuda`)
and a working ROCm. On Arch, see `docs/notes.md` — the distro's `hsa-rocr` aborts
under HRX and `scripts/env.sh` shims around it.

## Two things that will bite you

1. **Never put `where [range(...)]` on a launch argument** if you plan to launch the
   kernel from an ordinary HIP or CUDA host. It compiles, then miscompiles into an
   out-of-bounds store. Bisected and written up in `docs/notes.md`.
2. **`config.get` values are specialization constants**, bound with
   `loom-compile --config=k=v`, not runtime arguments. One HSACO per shape — which is
   fine, since compiling one takes about 2 ms.

## Licence

Apache-2.0, matching hrx-system.
