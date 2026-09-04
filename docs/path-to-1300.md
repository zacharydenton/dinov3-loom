# The path past torch max-autotune (1287.7 img/s)

Current: **915.5 img/s**, or 1.092 ms per image at batch 32.
Target: 1287.7 img/s, or 0.777 ms. The gap is **0.315 ms per image**.

## Where the time and the FLOPs are

DINOv3 ViT-S+ at 201 tokens costs 12.24 GFLOP per image:

| | GFLOP/img | ms/img | share of time | TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| attention | 0.745 | 0.397 | 36.4% | **1.88** |
| gate/up + swiglu | 5.69 | 0.236 | 21.6% | 24.1 |
| down | 2.85 | 0.166 | 15.2% | 17.2 |
| qkv | 2.13 | 0.112 | 10.3% | 19.0 |
| residual + norm | - | 0.081 | 7.4% | - |
| o | 0.712 | 0.044 | 4.0% | 16.3 |
| rope | - | 0.033 | 3.0% | - |
| patch embed, scatter, final norm | 0.116 | 0.023 | 2.1% | - |

Measured under load, the GPU boosts to **2.4-2.7 GHz** (`pp_dpm_sclk` at 97-100%
busy; the 620 MHz readings elsewhere in these notes are idle). At 2.6 GHz and the
usual 512 fp16 FLOP/clock/CU for RDNA3 WMMA, 40 CUs give a peak near **53
TFLOP/s**.

So the matmuls run at **30-45% of peak**, which is respectable and at or above
the untuned rocWMMA reference in `experiments/` (20.7 TFLOP/s on the gate/up
shape). **Attention runs at 3.5% of peak** -- 36% of the time for 6% of the FLOPs.
Confirmed standalone, not just derived from the profile share: 1.97 TFLOP/s at
batch 8 and 1.78 at batch 32.

Nothing here is bandwidth-bound. Total traffic is roughly 34 MB per image, or
31 GB/s at the current rate, against ~256 GB/s of LPDDR5X.

## The arithmetic

Fixing attention alone clears the target:

| attention at | attention ms | total ms | img/s |
| --- | ---: | ---: | ---: |
| 1.88 TFLOP/s (today) | 0.397 | 1.092 | 915 |
| 8 TFLOP/s | 0.093 | 0.788 | 1269 |
| 10 TFLOP/s | 0.075 | 0.770 | 1299 |
| 12 TFLOP/s | 0.062 | 0.757 | **1321** |
| 20 TFLOP/s | 0.037 | 0.732 | 1366 |

12 TFLOP/s is 23% of peak -- *less* than what the matmuls in this repo already
achieve. Everything else is secondary: even making the whole elementwise budget
free (10.4%, 0.114 ms) only reaches 978 img/s on its own.

## Why attention is slow, and what to do

The kernel is a two-pass softmax with one workgroup per (16 query rows, head).
Four things cost it:

1. **K and V are re-staged once per query tile.** A 201-token image has 13 query
   tiles, so each head's K and V are read 13 times, and the ~12 barriers per
   workgroup are amortised over only 16 queries.
2. **P*V uses four of the eight waves.** The output is 16x64, which is four 16x16
   fragments.
3. **Scores make two full round trips through LDS** -- accumulator fragment out,
   softmax reads back, f16 probabilities written, `lhs` fragment reads those.
4. **The softmax is three passes** over a 201-wide row (max, exp and sum,
   normalise and narrow).

The fix for (1) is to **invert the loop nest**: one workgroup per (image, head),
staging a K/V block once and streaming all 13 query tiles against it. K/V loads
per (image, head) drop from 52 to 4, and the staging barriers amortise over 13x
more work. That requires real FlashAttention bookkeeping -- a running max and sum
per query row, rescaling the accumulator whenever the max moves -- which is
exactly what the current kernel skips because 201 tokens fit in LDS. Skipping it
was right for correctness-first; it is now the thing in the way.

(2) follows from (1): with K/V staged once the LDS budget is smaller, so the
32-row query tile that lost on occupancy earlier may fit.

(3) is worth checking independently: if a wave32 16x16 f32 `result` fragment can
be reinterpreted as a 16x16 f16 `lhs` fragment, or converted with lane shuffles,
both round trips disappear. The ggml-hrx corpus stores and reloads, so Loom may
not expose it directly.

(4) falls out of (1) too -- online softmax works on one 64-key block at a time,
in registers, instead of three passes over 201 elements in LDS.

## After attention

In order, worth roughly 10% together:

* **RoPE into the QKV matmul epilogue** (3.0%). It is a per-element rotation on q
  and k, and the epilogue already holds those values in registers. Removes two
  launches and two read-modify-write passes over the qkv buffer per layer.
* **The down projection is the weakest matmul** at 17.2 TFLOP/s, against 24.1 for
  gate/up. K=1536 means 48 k-steps; BLOCK_K=64 would halve the loop overhead.
* **Fuse the down projection's epilogue into residual+norm.** Its output feeds
  straight there, and that pair is another 7.4%.

## The honest caveat

Torch's 1287.7 comes from AOTriton -- a prebuilt, hand-tuned flash attention that
inductor calls rather than generates. Matching it means writing a genuinely good
flash attention, which is the hardest kernel in this model. The matmul side of
this repo is already competitive; the attention side is a real project, not a
tuning pass.

## What `~/code/hrx-demos` changes about all of this

`ROCm/hrx-demos` is a complete Ideogram 4 text-to-image implementation on HRX
with **255 Loom kernels** -- unlike `hrx-loom-kernels`, which has eight. It is the
reference for how AMD actually writes these, and it contradicts the central
assumption of this repo.

### Their WMMA kernels use no LDS at all

`kernels/ideogram4/linear_bf16_bf16_wmma_m128n64_2wave.loom` has **zero**
`buffer.alloca`. Both fragments load straight from global views:

```
%lhs = vector.fragment.load<lhs> %input_view[...]
%rhs = vector.fragment.load<rhs> %weight_view[...]
```

The `rhs` transpose is a strided *view* over the global weight buffer
(`encoding.layout.strided [1, %hidden_size]`), not a staged copy. They also
declare `buffer.assume.alignment {minimum_alignment = 16}` so the compiler can
issue wide loads.

Every kernel here stages A and W through LDS with barriers. That machinery -- and
the double-buffering, tile-widening and prefetch experiments that all failed
around it -- exists to solve a problem their kernels do not have.

**But the naive version of this is worse, and it is worth knowing why.** A no-LDS
16x16 tile with one wave, measured on the model's shapes:

| shape (M=6432) | no-LDS 16x16 | this repo, LDS-staged 64x64 |
| --- | ---: | ---: |
| qkv (K=384, N=1152) | 8.57 TFLOP/s | **18.79** |
| gate/up (K=384, N=3072) | 6.95 | **16.92** |
| down (K=1536, N=384) | 5.35 | **13.26** |

At a 16x16 tile, A is re-read N/16 times and W is re-read M/16 times, and nothing
recovers that. AMD's shape is **128x64 across two waves** -- 8192 outputs over 64
lanes is 128 accumulators per lane, sixteen fragments. The reuse comes from *the
register tile*, not from LDS. That is the technique: make the register block big
enough that operands are read once and held, and skip the staging entirely.

### They ship the attention kernel this repo needs

`kernels/ideogram4/attention_online_bf16_wmma.loom` is online-softmax
FlashAttention in 301 lines:

* **One wave32 per workgroup**, `workgroup_size(32)`, one 16-query tile per head.
* **512 bytes of LDS**, used only to exchange probabilities. No K/V staging.
* K and V fragment-load from global; K's transpose is a strided view.
* Twenty loop-carried `vector<8xf32>` values -- running max and sum split across
  even/odd lane groups, plus sixteen accumulator fragments -- held in registers
  across the whole key loop.
* Query tokens padded to a multiple of 16 so every tile is whole and no bounds
  guard is needed in the inner loop.

That is the 5x. It is not a tuning pass, but it is also not research: the design
is sitting in `~/code/hrx-demos` and it targets gfx1100, one revision of the same
architecture.

### Revised plan

1. **Port the online attention shape.** One wave per (16 queries, head), K/V from
   global, online softmax in registers, LDS only for the probability exchange.
   head_size is 64 here rather than 256, so the QK inner loop is 4 head tiles and
   the accumulator is 4 fragments -- smaller than theirs. Attention is 36% of the
   time at 1.88 TFLOP/s; anything above 12 clears the target on its own.
2. **Re-do the matmul as a large register tile with no LDS**, m128n64 over two
   waves. This repo is at 13-19 TFLOP/s with LDS staging; the question is whether
   their shape beats it. Given that all three of the failed experiments here were
   LDS-and-occupancy problems, it probably does.
3. **fp8.** Ideogram 4 runs fp8 weights (`linear_fp8_bf16_wmma*`), which would
   halve weight traffic again. DINOv3 weights would very likely tolerate it at
   the 0.997 gate this model is judged by.
