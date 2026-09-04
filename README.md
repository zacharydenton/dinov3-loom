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

| kernel | state | notes |
| --- | --- | --- |
| `layernorm_f32` | **working, validated** | max_abs 1.6e-6, cosine 1.0000000000 vs f64 NumPy; 14.3 µs at 201×384 |
| matmul + bias | not started | ggml-hrx's `mul_mat_*_wmma` corpus is the starting point |
| flash attention | not started | ditto `flash_attention_f32_f16_wmma` |
| RoPE (2D axial) | not started | ggml's `rope_f32` is 1D; DINOv3 needs axial |
| SwiGLU MLP | not started | `mul_mat_swiglu_f32_f32_wmma` exists in the corpus |
| LayerScale + residual | not started | trivial once binary/unary are wired |
| patch embed | not started | 16×16 stride 16 is a reshape + matmul, no conv needed |
| end-to-end runner | not started | |

LayerNorm came first because it is the one op in a DINOv3 forward pass that
ggml-hrx's kernel corpus does **not** cover — it ships RMSNorm only.

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
$ /opt/rocm/bin/hipcc -O2 -o host/loomrun host/loomrun.cpp
$ python3 tools/test_layernorm.py
compiled layernorm_f32 for hidden=384
  PASS tokens=1     ( 130.83 us): max_abs=5.311e-07 max_rel=2.349e-05 cosine=1.00000000
  PASS tokens=3     ( 161.44 us): max_abs=7.566e-07 max_rel=1.654e-05 cosine=1.00000000
  PASS tokens=201   (  14.34 us): max_abs=1.632e-06 max_rel=8.170e-04 cosine=1.00000000
  PASS tokens=1024  (  41.96 us): max_abs=1.520e-06 max_rel=1.243e-02 cosine=1.00000000
```

(`max_rel` blows up only where the expected value passes through zero; cosine
similarity is 1.0 to ten digits in every case.)

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
