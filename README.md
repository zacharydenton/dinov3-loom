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

Radeon 8060S (gfx1151), torch 2.13.0 + ROCm, best of 5 interleaved rounds
(`tools/benchmark.py`). The machine had other jobs on it throughout; interleaving
means both sides saw the same contention.

| configuration | img/s | vs best Loom |
| --- | ---: | ---: |
| torch max-autotune fp16, batch 64 | 1115.9 | 4.94x |
| torch compile fp16, batch 64 | 1091.5 | 4.83x |
| torch eager fp16, batch 64 | 838.5 | 3.71x |
| torch max-autotune fp16, batch 1 | 580.3 | 2.57x |
| torch compile fp16, batch 1 | 343.8 | 1.52x |
| torch eager fp16, batch 1 | 271.5 | 1.20x |
| torch max-autotune fp32, batch 64 | 231.4 | 1.02x |
| **loom fp32, batch 64** | **226.0** | **1.00x** |
| **loom fp32, batch 8** | **224.2** | 0.99x |
| **loom fp32, batch 32** | **221.8** | 0.98x |
| torch compile fp32, batch 64 | 165.1 | 0.73x |
| torch eager fp32, batch 64 | 146.5 | 0.65x |
| torch compile fp32, batch 1 | 125.5 | 0.56x |
| torch max-autotune fp32, batch 1 | 125.2 | 0.55x |
| **loom fp32, batch 1** | **117.7** | 0.52x |
| torch eager fp32, batch 1 | 115.8 | 0.51x |

Read it honestly:

- **At fp32 these kernels match the best torch can do.** 226.0 against
  max-autotune's 231.4 is a 2% gap, well inside this machine's run-to-run
  variance, and it is 37% ahead of `torch.compile` and 54% ahead of eager at the
  same batch. Batching is what closed it: 117.7 -> 226.0.
- **The batch curve is flat from 8 upward** (224 / 222 / 226 at 8 / 32 / 64), so
  batch 8 already buys nearly everything. Batch 1 costs about half the
  throughput, and there the Loom path is roughly level with torch eager.
- **fp16 torch is still 1.2-4.9x ahead**, and that gap is the whole remaining
  story. Those paths reach WMMA through hipBLASLt and Triton; these kernels are
  scalar f32 FMA and cannot. Closing it means f16 inputs with f32 accumulation
  through Loom's `vector.mma`, which is what ggml-hrx's own corpus does. That is
  the next kernel, not a tuning pass.
- **max-autotune did not fail on ViT-S+**, unlike the ViT-L result recorded
  earlier: it compiled and ran in every configuration and was torch's best fp32
  result. The 96 KB LDS request that breaks it at ViT-L does not arise at
  head_dim 64 with 201 tokens.

Getting here was 22 -> 46 -> 80 -> 94 -> 121 -> 226 img/s; `docs/notes.md` has
what each step was worth. The two largest were both LDS bank conflicts, found in
different kernels a day apart.

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
