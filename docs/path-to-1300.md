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
