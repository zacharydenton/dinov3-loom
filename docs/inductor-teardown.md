# What torch's max-autotune kernel is doing

Dumped with `TORCH_LOGS=output_code TORCHINDUCTOR_MAX_AUTOTUNE=1` on ViT-S+ fp16
at batch 64, the configuration that runs 1.64x this repo. The generated code is
in `/tmp/inductor_cache`; what follows is what it actually emits.

## It is not winning on fusion

Steady-state, torch runs ~15 launches per layer:

```
triton_per_fused_add_mul_native_layer_norm_view    residual + layerscale + layernorm
triton_tem_fused_addmm_native_layer_norm_t_view    q projection
extern_kernels.mm                                  k projection (no bias -> plain mm)
triton_poi_fused_..._cos_..._sin_..._view    x3     RoPE, recomputed from arange
triton_tem_fused_addmm_native_layer_norm_t_view    v projection
triton_poi_fused_clone_transpose                   layout shuffle for SDPA
torch.ops.aten._scaled_dot_product_flash_attention attention
triton_tem_fused_addmm_clone_t_transpose_view      o projection
triton_per_fused_add_mul_native_layer_norm_view    residual + layerscale + layernorm
triton_tem_fused_addmm_t_view                x2    gate, up
triton_poi_fused_mul_silu_view                     swiglu
extern_kernels.addmm                               down projection
```

This repo runs **12**. Torch issues q, k and v as three separate matmuls and
gate/up as two, where both are fused here. And torch **recomputes the RoPE cos/sin
tables from `arange` in three kernels every layer**, where this repo precomputes
them once at export. On launch count and redundant work, this repo is ahead.

It does fuse two things worth having, both of which are separate passes here:

* `triton_per_fused_add_mul_native_layer_norm_view` folds the residual add, the
  LayerScale multiply and the next LayerNorm into one kernel. Here those are
  `residual_scale_f32` then `layernorm` -- 3.7% + 5.3% of the time, and one extra
  round trip through the residual stream.
* `triton_tem_fused_addmm_mul_silu_t_view` folds `silu(gate) * up` into the *up*
  projection's epilogue, so SwiGLU is free. (It only picks this for some layers;
  others get a standalone `triton_poi_fused_mul_silu_view`.)

## It is winning on the matmul inner loop

The winning autotune configs:

| kernel | BLOCK_M | BLOCK_N | BLOCK_K | GROUP_M | stages | warps | waves_per_eu |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q/k/v, o | 128 | 128 | 32 | 16 | 2 | 8 | 0 |
| gate/up (+silu) | 128 | 64 | 32 | 8 | 2 | 4 | 2 |
| patch embed | 128 | 128 | 16 | - | - | - | 0 |

against this repo's fixed **64 x 64 x 32, no pipelining, plain 2D grid order**.
Three differences, in the order they probably matter:

1. **`num_stages=2`.** The Triton k-loop is a plain `tl.load` -> `tl.dot` and the
   compiler software-pipelines it: the global load for block k+1 issues while
   block k is still in the MMA. The kernel here stages explicitly into LDS with
   **two barriers per k-step and no overlap at all** -- every global load is
   exposed. This is the single biggest structural gap.
2. **`BLOCK_M=128` against 64.** A workgroup reads `BLOCK_M x K` of A and
   `BLOCK_N x K` of W, so total W traffic scales as `1/BLOCK_M`. Doubling the M
   tile halves the weight reads. At batch 8 and up there are plenty of tiles to
   go round: torch reports `multi_processor_count=20` (WGPs, not the 40 CUs
   `rocminfo` shows), and 12864 rows over BLOCK_M=128 is 100 tiles.
3. **`GROUP_M` tile ordering** (8 or 16) rather than a raw 2D grid, so
   consecutive workgroups share A or W rows in cache.

`ACC_TYPE=tl.float32` and `matrix_instr_nonkdim=16` match what this repo already
does -- f32 accumulation over a 16x16x16 MMA.

## Attention is a library kernel, not generated

`torch.ops.aten._scaled_dot_product_flash_attention` resolves to **AOTriton**
(`torch.backends.cuda.preferred_rocm_fa_library() -> _ROCmFABackend.AOTriton`),
a hand-tuned prebuilt kernel. Inductor never sees inside it. That is the other
half of the gap: attention is 28.9% of the time here.

## What to take from it

In rough order of expected value:

1. **Double-buffer the matmul k-loop.** Stage block k+1 into a second LDS slab
   while the WMMA fragments for block k are still issuing, and drop from two
   barriers per step to one. This is `num_stages=2` done by hand.
2. **BLOCK_M=128.** Halves weight traffic; costs 16 accumulator fragments per
   wave instead of 8, which should still fit.
3. **Fuse residual + LayerScale + LayerNorm into one kernel.** 9% of the time
   plus a round trip, and it is a small kernel to write.
4. **Fuse SwiGLU into the up projection's epilogue.** Only 4.1% now that the
   activations are f16, so this is last.

Nothing here suggests the tiles should be autotuned rather than chosen: the two
winning shapes differ only in BLOCK_N, and both are what you would pick by hand
from the operand shapes.
