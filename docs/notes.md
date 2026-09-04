# Working notes

## `where [range(...)]` on a launch argument is part of HRX's dispatch ABI

A kernel written like the ggml-hrx corpus does:

```
} launch(%token_count: index, %input: buffer, %output: buffer) where [range(%token_count, 1, 4096)] {
```

compiles fine, and `loom-compile --backend=amdgpu-hal` emits a clean HSACO with the
expected kernarg layout. Launched through `hipModuleLaunchKernel` it dies with:

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory
beyond the largest legal address.
```

Bisected to exactly that clause. The same kernel with the clause removed runs
correctly; a kernel that declares the `index` argument but never *uses* it also runs
(the argument is dead-code-eliminated before the constraint matters). The
disassembly of the failing kernel shows the guard lowered to a branch that clobbers
`s[0:1]` — the kernarg segment pointer — via a wave64 `s_and_saveexec_b64` in a
wave32 kernel.

Read: those constraints are a contract HRX's own dispatch layer upholds when it
builds the dispatch packet, not a free-standing hint. Kernels meant to be launched
by an ordinary HIP/CUDA host must not carry them. Everything here is written without
`where` clauses on launch arguments; `index.assume` inside the body is fine.

Worth reporting upstream — silently miscompiling rather than refusing to compile is
the bad failure mode.

## Config values must be bound at compile time

`config.get` results carry a `no_ordinary_uses` constraint. Compiling without
`--config=<key>=<value>` fails with:

```
target 'amdgpu-rdna3-5' ... rejected 'config.get' value 'result' ... constraint
'no_ordinary_uses' is not satisfied
```

So `hidden_size` and `epsilon` are specialization constants, not runtime arguments.
That is the point of the design — one HSACO per shape, JIT-compiled in ~2 ms — but it
means the host has to compile per configuration rather than pass a struct.

## `iree-test-loom` segfaults at teardown

`iree-test-loom <kernel> --device=amdgpu` runs the `check.case` samples, then
segfaults inside `hsa_executable_destroy` in the ROCm 7.14 loader, sometimes before
the JSON result reaches stdout. That is why validation here goes through
`host/loomrun` and NumPy instead of the `check` dialect. The `check.case` blocks are
kept in the kernels because they document intent and `loom-format --check` still
validates them.

## Arch's ROCm aborts under HRX

`hsa-rocr` on Arch is built with `-D_GLIBCXX_ASSERTIONS` (from `/etc/makepkg.conf`),
and HSA's completion thread trips a `std::vector` bounds assert as soon as HRX opens
a queue. `scripts/env.sh` puts `~/.local/rocm-hrx` (ROCm 7.14, extracted from the
kyuz0 Strix Halo toolbox) ahead of it. `host/loomrun` links real ROCm HIP and is
unaffected.

## Matmul: what actually moved the needle

Measured on gfx1151 with the four shapes DINOv3 ViT-S+ uses, in order of payoff:

1. **Pad the LDS row stride to KC+1.** The 64x64 tile stages A and W slabs as
   `[TILE][KC]` with KC = 32 floats. A workitem's four A reads per k-step are then
   exactly 32 floats apart, which is the LDS bank count — every one of them hits
   the same bank. Changing the row stride to 33 spread them across four banks and
   took `gate/up` from 161 to 2007 GFLOP/s. **One constant, 10x.** This was by far
   the largest single change in the project.
2. **Fuse q/k/v into one [1152, 384] matmul, and gate/up into [3072, 384].**
   At 201 tokens a 64x64 tile over N=384 is only 24 workgroups on a 40-CU part.
   Fusing turns three launches of 24 into one of 72, and two of 96 into one of
   192. 46 -> 80 img/s. The weights are concatenated at export time, and rope,
   attention and swiglu grew a `row_stride` config so they can read their slice
   of a fused buffer in place rather than needing a de-interleave pass.
3. **A 64x32 tile for the narrow projections.** `o_proj` and `down_proj` have
   N=384 and cannot be fused with anything. Halving the N tile doubles their
   workgroup count while keeping the 4-deep micro-tile in M. 80 -> 94 img/s.
4. **Spread attention's V accumulation over all 256 lanes.** The first version
   gave one output channel to each of the first `head_dim` workitems and left the
   other 192 idle. Splitting the keys into `256 / head_dim` strided groups and
   combining their partials through LDS cut attention from 19% to well under it.

Tile shapes that lost: 32x32 (four times the workgroups, but 2x2 micro-tiles
halve the compute-to-LDS ratio) and 48x48. 64x64 for wide N, 64x32 for narrow N
is the configuration in the repo.

## Benchmarking on a busy machine

`/sys/class/drm/card1/device/pp_dpm_sclk` showed the GPU pinned at 953 MHz out of
2900 with `gpu_busy_percent` at 54 while two unrelated jobs were running. Isolated
kernel timings varied by 5x run to run, which made a tile-shape sweep almost
useless. `tools/benchmark.py` interleaves the Loom and torch measurements and
keeps the best of N rounds per configuration, so both sides see the same
contention and the reported number is the least-disturbed sample.

## Batching, and the second bank conflict

Batching packs several images into one residual stream, which turns every matmul
from M=201 into M=batch*201 and fixes the occupancy problem at the source. Three
kernels had to learn about image boundaries first:

* `rope_2d_f32` takes a `tokens_per_image` config and derives the RoPE position
  from the offset inside the image, not inside the batch.
* `attention_f32` takes the same config and restricts keys and values to the
  query's own image.
* `embed_scatter_f32` is new: the patch matmul emits one contiguous
  `[batch*196 x 384]` block, but the residual stream interleaves each image's CLS
  and register tokens in front of its patches, so a kernel places them.

That alone took batch 4 to 178 img/s. Then attention became the top cost at 34%,
for a specific reason: with one query per workgroup, every query re-read the
entire K and V for its head from global memory — about 740 MB per image per
forward pass at 201 tokens. Giving each workgroup a block of 8 queries and
staging K/V through LDS in 32-key chunks cuts that eightfold, with one wave32
subgroup per query so every softmax reduction is subgroup-local and needs no
barrier.

The rewrite was initially **4x slower** than the kernel it replaced. Same cause as
the matmul: the K chunk was stored as `[32][64]`, so consecutive lanes read rows
64 floats apart and collided in the same LDS bank on every score. Padding the row
stride to 65 took it from 512 us back to 131 us — and now with an eighth of the
memory traffic. Two independent kernels, the same mistake, a day apart.

Final: **226 img/s at batch 64**, up from 117.7 at batch 1. The curve is flat from
batch 8 (224) onward.

## A tile choice that could not be settled

Once the batch supplies workgroups, the 64x64 tile ought to beat 64x32 on the
N=384 projections again. Best-of-3 said 243 vs 222 img/s for wide at batch 32,
and 211 vs 230 against it at batch 8 — with individual samples for one
configuration ranging from 115 to 243. The difference is inside the noise floor
of this machine, so the default keeps the single narrow-tile path and
`--wide-threshold` is there to re-measure on an idle box.

## WMMA, and what the corpus had to teach

`vector.mma` plus `vector.fragment.load<lhs>/<rhs>` and
`vector.fragment.store<result>` are Loom's WMMA surface -- the compiler owns the
per-lane fragment layout, so it is about as much work as writing the same kernel
with rocWMMA. Two things were not obvious:

1. **The rhs fragment wants logical `[k][n]`.** Weights are stored `[n][k]`, so a
   naive staging gives the transpose of what the fragment load expects. The fix
   is a strided view over the same LDS bytes:
   `encoding.layout.strided [1, %c40]` over a physically `[n][40]` tile reads as
   logical `[k][n]`. The corpus does exactly this for its activation fragment.
2. **The corpus WMMA kernels are wave64**, which is why their f32 accumulator is
   `vector<4xf32>`. At `subgroup_size = 32` a 16x16 f32 fragment is
   `vector<8xf32>`, and using the corpus's width gives
   `matrix constraint 'wave_size' is not satisfied (source_bits=0, target_bits=256)`
   -- 256 bits per lane being the tell.

Activations stay f32 and are narrowed with `vector.fptrunc` while being staged
into LDS, so only the weights changed dtype and no other kernel was touched.
Accumulation is f32, matching what hipBLASLt does for torch's fp16 path.

Result: 10-13 TFLOP/s on the model's shapes against ~2 TFLOP/s for the scalar f32
kernel, and it beats the untuned rocWMMA reference in `experiments/` on the
`down` shape (11.1 vs 6.5 TFLOP/s). End to end 1.8-2.3x, and 467.9 img/s at
batch 32.

Where the time goes now, at batch 32:

```
attention              36.7%   <- still scalar f32
gate/up matmul         14.4%
swiglu                 10.1%
down matmul             9.4%
residual+ls             7.4%
layernorm               7.3%
qkv matmul              5.9%
o matmul                4.9%
rope                    3.1%
```

The matmuls are no longer the problem. Attention is, and the elementwise kernels
together are another 28% -- swiglu alone moves ~44 MB per image in f32, which an
f16 handoff to the down projection would halve.

## Flash attention

201 tokens is small enough that the whole score row fits in LDS, so this is a
two-pass softmax rather than a real FlashAttention: compute every score, take the
row max and sum, then do P*V. No online rescaling, no running statistics carried
across key blocks -- strictly less arithmetic than the streaming form, and much
less code.

One workgroup per (16 query rows, head), 8 wave32s. QK^T gives each wave one
16x16 score tile per 128-key block; P*V gives four of the eight waves one 16x16
output tile each. The two layout tricks:

* QK^T's rhs wants logical `[dim][key]`; K is staged `[key][dim]`, so a
  `encoding.layout.strided [1, 68]` view reads it transposed.
* P*V's rhs wants logical `[key][dim]`, which is the staged order already.

Row strides are padded to odd dword counts (68 f16 = 34 dwords, 212 f32) so the
16 rows a fragment load touches spread across banks. Third kernel in this repo
where that mattered.

3.5x on the kernel, 1.35x end to end. Accuracy drops from cosine 0.999999 to
0.99998 because the probabilities are narrowed to f16 before the P*V fragments,
on top of the f16 weights.

Attention is now 23.6% of the time and **swiglu has grown to 14.5%** -- it is pure
f32 bandwidth, reading two [rows, 1536] tensors and writing a third. Having it
emit f16 straight into the down projection's staging would halve that, and is
probably the next thing worth doing.

## f16 activations, and why the read side dominates

swiglu had grown to 14.5% of the time, which looked like a write-bandwidth
problem: two `[rows, 1536]` f32 reads and one f32 write. Producing it as f16
halves that, and the same argument applies to layernorm and the attention output
-- all three feed nothing but a matmul, which narrows them to f16 on the way into
LDS regardless.

The bigger half of the win is on the read side, and it is not obvious from the
profile. A matmul workgroup stages a 64-row slab of A per k-step, so A is read
once per *column tile*: gate/up has 3072/64 = **48** of them, qkv has 18, down
has 6. Halving the dtype halves all of that.

Measured 1.17x at batch 8 and 1.25x at batch 32, with swiglu falling from 14.5%
to 4.1% of the time and accuracy unmoved at cosine 0.99998.

Both dtype paths are generated rather than copied -- `tools/gen_matmul_wmma.py`
emits the (A f16) and (C f16) matmul variants from the base kernel, and
`tools/gen_f16_variants.py` does the same for layernorm, swiglu and flash
attention -- so they cannot drift apart. The residual stream, the norm statistics
and the final output stay f32.

## f16 q/k/v: a negative result

The same argument that made layernorm and swiglu emit f16 should apply to q/k/v:
have the QKV projection write f16, teach rope and flash attention to read it, and
the attention staging becomes a straight copy instead of a load-and-narrow. The
kernels are all generated (`rope_2d_f16.loom`,
`flash_attention_f16_wmma_af16_cf16.loom`, an f16-out QKV matmul) and compile
clean, and accuracy holds at cosine 0.9999993.

**It is slower.** Best of 4, alternating:

| | f16 q/k/v | f32 q/k/v |
| --- | ---: | ---: |
| batch 8 | 662.4 | **769.8** |
| batch 32 | 529.2 | **703.0** |

The f16 flash-attention variant compiles to 13328 bytes against 9192 for the f32
one, so something in the f16 path is generating substantially more code rather
than less work -- but the measurement is consistent across batch sizes and
repeats, so it was not worth chasing further. `--f16-qkv` keeps the path
reachable for anyone who wants to dig; the default is off.

Worth recording because the reasoning was sound and the result was not: the win
from f16 activations came from the *read amplification* on the matmul A operand
(48 column tiles for gate/up), and q/k/v have no such amplification -- attention
reads them once per query tile, and rope reads them once. There was never much
bandwidth there to save.

## Lever 1: hand double-buffering the k-loop -- rejected

Inductor's matmuls run with `num_stages=2`, so Triton software-pipelines the
k-loop: the global load for block k+1 issues while block k is still in the MMA.
The kernel here staged explicitly into LDS with two barriers per k-step and no
overlap, which looked like the obvious gap.

Implemented it: two LDS slabs each for A and W, a prologue staging block 0, and a
loop body that issues block k+1's loads, does block k's MMA, then writes the
prefetched values into the other slab -- one barrier per step instead of two.
Correct (cosine unchanged) and kept in `experiments/` for reference.

**It is slower.** Best of 3, alternating, idle GPU:

| | double-buffered | single |
| --- | ---: | ---: |
| batch 8 | 766.6 | **795.6** |
| batch 32 | 687.0 | **731.3** |

The cost shows up in the resources: VGPRs go 64 -> 72 and LDS 20480 -> 28672
bytes per workgroup, which drops the workgroups resident per WGP from 6 to 4.
At this tile size there are already enough waves in flight to hide the global
load latency, so paying occupancy to prefetch it explicitly is a straight loss.

That is also why it *does* pay for inductor: its BLOCK_M=128 tile uses far more
registers per workgroup and runs at much lower occupancy, so it has latency left
to hide. The technique is not wrong, it is coupled to the tile size.

## Lever 2: fusing residual + LayerScale + LayerNorm -- kept

`triton_per_fused_add_mul_native_layer_norm_view` is the one inductor fusion
worth copying. In a transformer block the residual add, the LayerScale multiply
and the next LayerNorm always occur together, and splitting them costs a launch
plus a full re-read of the residual stream.

`residual_layernorm_f32.loom` does all three. The trick that makes it cheap:
hidden_size <= 512 with 256 workitems means every workitem owns at most **two**
channels, so the updated values stay in registers between the accumulate pass and
the normalize pass and the re-read disappears entirely rather than merely being
cached.

Wiring it up needs the loop restructured -- only the very first `norm1` stands
alone, every later LayerNorm is fused into the residual before it, *including
across the layer boundary* (layer n's `ls2` residual pairs with layer n+1's
`norm1`). The final norm writes f32 so it stays separate. That takes 49 launches
per forward down to 26.

**Kept.** Best of 3, alternating:

| | fused | split |
| --- | ---: | ---: |
| batch 8 | **817.3** | 763.0 |
| batch 32 | **800.3** | 764.8 |

+7.1% and +4.6%, accuracy unchanged.

## Lever 3: SwiGLU as a matmul epilogue -- kept

Inductor folds `silu(gate) * up` into the *up* projection's epilogue, reading the
already-computed gate result from memory. That shape does not fit here, because
gate and up are one fused `[rows, 2*width]` matmul and the two halves are
produced by different workgroups.

The version that does fit is a **dual-N** matmul: one workgroup accumulates both
projections for the same output columns, holding four accumulator fragments per
wave instead of two, and applies SwiGLU in the epilogue. The
`[rows, 2*width]` intermediate is then never written at all.

Traffic, in units of `[rows, width]` f16 per layer:

| | before | after |
| --- | ---: | ---: |
| gate/up matmul writes | 2 | 1 |
| swiglu reads + writes | 3 | 0 |
| down matmul reads A (6 column tiles) | 6 | 6 |
| **total** | **11** | **7** |

**Kept**, and it is the largest single win since the WMMA matmul itself. Best of
3, alternating:

| | fused | split |
| --- | ---: | ---: |
| batch 8 | **863.4** | 811.4 |
| batch 32 | **917.2** | 776.6 |

Accuracy improves slightly too (0.9999996 against 0.9999995): the intermediate no
longer makes a round trip through f16.

Note this cost 88 VGPRs and 31744 bytes of LDS against 64 and 18432 -- the same
resource increase that made hand double-buffering a loss. The difference is that
this one removes a whole pass rather than hiding latency the hardware was already
hiding.

## A trap worth recording

`tools/validate.py` used to write its synthetic image to `build/patchified.bin`,
which is also the benchmark's input. Running it silently repointed every
subsequent hand comparison at a different picture, and the next accuracy check
read cosine 0.548 and looked like a catastrophic regression in a kernel that was
in fact correct. It now writes to `/tmp/validate_patchified.bin`.

## Lever 4: a 32-row attention query tile -- rejected

K and V are re-staged once per query tile, so at 16 rows a 201-token image
stages them 13 times. Widening to 32 rows halves that to 7, and it also gives the
P*V phase 2x4 = 8 output fragments so all eight waves work instead of four. Both
arguments are correct.

It still loses, because the same change costs 53760 bytes of LDS against 40064
and halves the workgroup count:

| | 32-row | 16-row |
| --- | ---: | ---: |
| batch 8 | 801.9 | **902.4** |
| batch 32 | **923.6** | 908.2 |

The batch-32 gain is 1.7%, inside the noise floor. The batch-8 loss is 11% and is
not: 8 images x 7 tiles x 6 heads is 336 workgroups where 16 rows gives 624, and
the larger LDS footprint reduces residency on top of that. Kept in
`experiments/flash_attention_f16_wmma_32row.loom`.

Third negative in a row from the same cause, which is worth stating plainly:
**on this part, at these sizes, anything that trades occupancy for locality
loses.** Double-buffering, f16 q/k/v and the wider attention tile all failed that
way. The changes that won -- LDS padding, operand fusion, f16 activations,
SwiGLU as an epilogue -- all removed work or traffic without spending registers
or LDS. The one apparent exception, the dual-N SwiGLU matmul, spends both but
deletes an entire pass in exchange.

## Lever 5: the online-softmax attention -- landed, 4.4x

`kernels/attention_online_f16_wmma.loom` is a port of
`hrx-demos/kernels/ideogram4/attention_online_bf16_wmma.loom` to DINOv3's
head_dim of 64: one wave32 per (16 query rows, head), 512 bytes of LDS used only
to reinterpret an f32 score fragment as an f16 `lhs`, K and V fragment-loaded
straight from global with K transposed by a strided view, and eight loop-carried
`vector<8xf32>` values -- running max and sum split across the two lane groups,
plus four accumulators for the 64 output channels.

It took three findings to make it lower. Every global `vector.fragment.load` was
rejected with `source-to-low constraint 'fragment_memory.dynamic_stride' is not
satisfied`, including the plainest dense one, which is what finally pointed away
from the transposed view and the loop.

### 1. `index.assume` on a fragment-load index term is fatal

This is the whole thing. Reduced to a single load with the preamble kept
verbatim, swapping the row index tells the story immediately:

| load | result |
| --- | --- |
| `%v_view[%query_origin, %channel0]` | rejected |
| `%v_view[%c0, %channel0]` | compiles |
| `%v_view[%query_origin, %c0]` | rejected |
| `%v_view[%query_origin0, %channel0]` (same value, no `index.assume`) | compiles |

The bound's magnitude is irrelevant -- `range(%x, 0, 255)` is rejected exactly
like `le(%x, %tile_origin_limit)`. `index.assume` replaces the value with an
opaque one, so the fragment-memory planner can no longer compute its
`byte_facts` and rejects the term.

The constraint is misnamed, which is what made this take so long.
`loom/src/loom/target/arch/amdgpu/lower/matrix_fragment_memory_plan.c:1352`
checks that a dynamic address term's byte facts fit in unsigned 32 bits, not
that the stride is dynamic. Twelve earlier hypotheses all asked *which facts to
state*; the answer was to state none. `hrx-demos` uses no `index.assume`
anywhere, and that is not a stylistic choice.

### 2. `index.rem` is how the subrange analysis gets a bound

Removing the assumes exposed the obligation they were there to discharge:
`SUBRANGE/010`, the fragment footprint's upper bound. `index.rem` is the
primitive the analysis follows through a workgroup id -- `%tile_in_image =
index.rem %tile_id, %tiles_per_image` proved on its own. So `%head` and `%image`
each get a no-op `rem` against a ceiling, which supplies the bound structurally
instead of by assertion. Neither changes any address at runtime.

### 3. The assumes also blocked constant folding

`index.assume` on `%hidden_size` and `%tokens_per_image` stopped those configs
folding to literals, which left the `rem` divisors opaque and the *lower* bound
(`0 <= origin`) unprovable. Dropping them closed it. Raw `config.get` values
everywhere, exactly as hrx-demos does it.

### A trap on the way

Adding `where [range(%token_count, 16, 65536)]` to the launch made it compile
before finding (2) and (3) -- and reintroduced the known launch-argument
miscompile: a wild address under plain `hipModuleLaunchKernel`, faulting
identically at every buffer size from 217 to 8192 rows. The `rem` bounds make it
unnecessary. The rule from the earlier `where` bug stands: **never put `where` on
a launch argument.**

### What it is worth

Against the flash kernel it replaces, standalone, batch 4, stride 384:

| | us | GFLOP/s | max_abs |
| --- | ---: | ---: | ---: |
| `flash_attention_f16_wmma` | 131.87 | 1882 | 2.2e-04 |
| `attention_online_f16_wmma` | 30.01 | 8272 | 4.9e-05 |

**4.4x, and more accurate besides.** Attention falls from 36.4% of the forward
pass to 10.2%. The prediction in `path-to-1300.md` was that 12 TFLOP/s on
attention would reach 1321 img/s; 8.3 TFLOP/s predicted ~1269 and the model
measured 1312.9 at batch 32.

The kernel requires f16 q/k/v, which flips `--f16-qkv` from a measured loss into
the default. That earlier negative result was real but narrow: narrowing bought
the *old* attention nothing, so it only cost bandwidth. The lesson is that a
negative result about a supporting change is only valid against the consumer it
was measured with.

The bottleneck is now the MLP: gate/up+swiglu at 30.1% and the down projection at
22.3%.

## Lever 6: the no-LDS m128n64 matmul -- measured, rejected

`experiments/matmul_bias_f16_wmma_m128n64.loom` is a port of
`hrx-demos/kernels/ideogram4/linear_bf16_bf16_wmma_m128n64_2wave.loom`: two
wave32s, each owning a 64x64 output tile as sixteen 16x16 accumulators, so one
k-step is four `lhs` + four `rhs` global fragment loads feeding sixteen
`vector.mma`. A 4:1 compute-to-load ratio against the staged kernel's 1:1, zero
LDS in the loop, zero barriers. With the `index.assume` rule from Lever 5 it
compiled first try; only the LDS staging view needed fixing (fragment memory ops
want a 2-D view, so each wave takes its own by `index.scale` offset).

Measured on an idle GPU, batch 32 shapes, against `matmul_bias_f16_wmma_af16`:

| shape | m128n64 | TFLOP/s | af16 | TFLOP/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| o (k384 n384) | 98.73 us | 19.2 | 116.92 us | 16.2 | **1.18x** |
| qkv (k384 n1152) | 331.41 us | 17.2 | 307.98 us | 18.5 | 0.93x |
| gate/up (k384 n3072) | 1238.43 us | 12.3 | 1026.88 us | 14.8 | 0.83x |
| down (k1536 n384) | 1222.24 us | 6.2 | 420.47 us | 18.0 | **0.34x** |
| down, batch 8 | 191.96 us | 9.9 | 147.05 us | 12.9 | 0.77x |

Correct everywhere (cosine 1.00000000), and slower everywhere that matters.

The reason is in the kernel descriptor:

```
m128n64:  vgpr_count 200, group_segment 8704
af16:     vgpr_count  64, group_segment 18432
```

Sixteen `vector<8xf32>` accumulators plus eight live fragments is 200 VGPRs.
gfx1151 has 1536 VGPRs per SIMD in wave32, so that is **7 waves per SIMD against
the 16-wave cap** -- 44% occupancy -- and because every operand now comes from
global there is no LDS prefetch left to hide the latency with. The staged kernel
spends more LDS (18432 B) precisely to stay at 64 VGPRs and full occupancy.

The k=1536 case is worst because it has the most k-steps to stall on, and the
fewest n-tiles (6) to spread the stalls across.

**This is the third time the same law has decided a measurement in this repo**,
after hand double-buffering and the 32-row attention tile: *on gfx1151 at these
sizes, anything that trades occupancy for locality loses.* hrx-demos targets
gfx1100 -- a discrete part with real GDDR bandwidth and different balance. Their
kernel shapes are not portable to an APU on LPDDR5X, even though their *idioms*
(global fragment loads, no `index.assume`) very much are: those were worth 4.4x
on attention in Lever 5.

The one shape that wins, `o`, is 5.4% of the forward pass, so 1.18x there is
~0.8% overall -- inside run-to-run variance. Not wired in. The kernel is kept in
`experiments/` because the *next* shape someone tries may have the arithmetic
intensity to pay for the registers.

## Lever 7: tuning occupancy for gfx1151 -- the kernels were already on the peak

The Radeon 8060S (gfx1151) gives a CU **64 KB of LDS**, **1536 VGPRs per SIMD**
(two SIMD32 per CU) and caps it at **32 waves**. So full occupancy needs
<= 96 VGPRs per wave and <= 16 KB of LDS per 256-thread workgroup. An audit of
every kernel against both budgets:

| kernel | VGPR | LDS | waves/CU | limited by |
| --- | ---: | ---: | ---: | --- |
| wmma_swiglu_k384_n1536 | 88 | 31744 | 16 (50%) | LDS |
| wmma matmuls (all) | 64 | 18432 | 24 (75%) | LDS |
| attention_online_cf16 | 168 | 4608 | 14 (43%) | LDS/slots |
| layernorm, rope, residual, scatter | <=16 | <=2048 | 32 (100%) | - |

**Nothing is VGPR-limited.** Every kernel has register budget to spare while LDS
holds occupancy down, which looks like an obvious imbalance to fix.

It is not. The three LDS allocations in each matmul -- A stage, W stage, result
stage -- have disjoint lifetimes: A and W are dead once the k-loop's last
workgroup barrier retires, which is exactly when the result tiles come alive.
Laying them over one arena takes the matmuls 18432 -> 10240 B and the SwiGLU
kernel 31744 -> 16384 B, both to a full 32 waves/CU. Correctness holds (cosine
1.00000000, model 0.99998).

**Every matmul stage got slower.** At batch 32, profiled: down 176.5 -> 254.1 ms,
o 44.0 -> 61.1, qkv 105.3 -> 133.9, gate/up 244.5 -> 278.9.

Sweeping the LDS arena to control occupancy directly shows why -- the optimum is
interior, and the shipped configuration is sitting on it:

| LDS/wg | workgroups/CU | waves/CU | down | qkv | o |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10240 | 4 (capped) | 32 | 237.8 | 123.1 | 57.8 |
| 16384 | 4 | 32 | 249.3 | 129.7 | 58.5 |
| **21504** | **3** | **24** | **183.5** | **105.1** | **42.6** |
| 32768 | 2 | 16 | 299.9 | 151.1 | 65.3 |
| 65536 | 1 | 8 | 385.6 | 210.4 | 82.5 |

Both directions cost roughly 2x at the extremes. 24 waves/CU is enough to hide
latency; past that, the extra concurrent workgroups are extra concurrent global
streams competing for 32 KB of L1 and 2 MB of L2, and the streaming footprint per
CU is what actually binds. The 18432 B the kernels already used lands on 3
workgroups/CU.

Moving the SwiGLU kernel from its 2 workgroups/CU to 3 (arena padded to 21504)
looked promising on a single reading (227.1 vs 244.5 ms) but was 0.98x on the
stage and 0.92x overall over best-of-5 interleaved. That first number was an
outlier; the machine had a 10-core CPU job on it throughout, which widens the
spread on end-to-end timing enough to fool a single sample.

**Reverted, nothing shipped.** The useful output is the sweep: for this repo's
matmul shape on gfx1151 the target is *3 workgroups per CU*, not maximum
occupancy, and there is ~21.5 KB of LDS available at that occupancy against the
18.4 KB currently used. Spending that 3 KB on a deeper k-block, rather than on
more workgroups, is the version of this idea that has not been tried.

## Lever 8: a 64x128 tile for the latency-bound down projection -- rejected

Classifying stages by *what bounds them* rather than by their share of the
profile (see Lever 9 below for the table) put the down projection at 32% of peak
compute and 22% of peak bandwidth -- bound by neither, i.e. stalling. The
standard answer is more work in flight per wave, and the register budget said
there was room: at the measured-optimal 3 workgroups/CU (24 waves, 12 per SIMD)
each wave may use 1536/12 = **128 VGPRs**, and the 64x64 kernel uses **64**.

`experiments/matmul_bias_f16_wmma_n128.loom` doubles the tile to 64x128: eight
waves with four accumulator fragments each instead of two, W staged 128 rows
wide, four `vector.mma` per `lhs` fragment load instead of two. It lands at 88
VGPRs, comfortably inside the budget.

It is slower at every occupancy point, and the occupancy sweep reproduces:

| LDS/wg | wg/CU | waves | n128 down | n64 down |
| ---: | ---: | ---: | ---: | ---: |
| 15360 | 4 | 32 | 12.6 TFLOP/s | 15.4 |
| **21504** | **3** | **24** | **16.4** | **17.5** |
| 23552 | 2 | 16 | 15.0 | 16.4 |

0.94x at the best occupancy point, 0.87-0.97x across the other shapes (o, qkv).
Correct throughout (cosine 0.99999998).

So the wider tile does not help even with occupancy held at its optimum, and the
VGPR headroom is not the binding resource. Together with Lever 6 (a 128x64
no-LDS tile, 0.34-1.18x) and Lever 7 (aliasing LDS for full occupancy, uniformly
slower), **three separate attempts to restructure this matmul have now lost.**
The 64x64 staged tile at 3 workgroups/CU is a local optimum in tile shape,
occupancy and LDS budget simultaneously, and the ~32-46% of peak these kernels
reach is not reachable-past by tiling changes. Whatever is left is instruction
scheduling and software pipelining inside the k-loop, which this Loom revision
does not obviously expose.

## Lever 9: classify stages by what bounds them, not by their share

The profile says where time goes; it does not say what to do about it. Dividing
each stage's FLOPs and bytes by its measured time, against the 8060S's ~53
TFLOP/s and ~256 GB/s, says which lever can even work. At batch 32, before the
f16-branch change:

| stage | ms | TFLOP/s | %peak | GB/s | %peak | bound by |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gate/up | 7.49 | 24.3 | 46% | 43 | 17% | compute |
| down | 5.60 | 16.3 | 31% | 66 | 26% | **neither -- latency** |
| resid+norm | 2.95 | - | - | 141 | 55% | **bandwidth** |
| attention | 2.60 | 9.2 | 17% | 91 | 36% | neither |
| o | 1.30 | 17.5 | 33% | 139 | 54% | **bandwidth** |
| qkv | 3.22 | 21.2 | 40% | 77 | 30% | compute |
| rope | 0.71 | - | - | 333 | 130% | cache-resident |

Two things fall out that the share-of-time view hides.

`rope` reads and writes at an apparent 333 GB/s, which is above what the memory
can do -- so it is being served from L2/L3, having just been written by the qkv
matmul. It is already fast; fusing it would save traffic that is not actually
going to memory. Deprioritised.

`o` and `resid+norm` were the two bandwidth-bound stages, and both sat on the
same intermediate: the o/down projections wrote f32 into `proj`/`mlp` and
resid+norm read it straight back. That branch is produced by a matmul and
consumed exactly once, so f32 doubled its traffic for precision that does not
survive the residual add. Narrowing it to f16 was worth **1.13x end to end**
(1176.6 -> 1333.4 img/s, best-of-5 interleaved) at unchanged accuracy -- and it
is the only one of four attempted optimisations in this round that worked.

The other three all tried to make a compute- or latency-bound kernel faster by
restructuring it, and all three lost. The one that worked removed bytes from a
stage that the numbers said was bandwidth-bound. Worth remembering as the order
to try things in.

## Lever 10: split-K for the down projection at batch 1 -- shipped

At batch 1 the matmuls do not fill the machine. With a 64x64 tile and m=201:

| stage | workgroups | of ~120 slots | TFLOP/s | % of peak |
| --- | ---: | ---: | ---: | ---: |
| qkv | 72 | 60% | 7.7 | 14.6% |
| o | 24 | 20% | 3.3 | 6.2% |
| down | 24 | 20% | 4.6 | 8.7% |

Weights are 97% of the traffic at batch 1 (4.72 MB per layer against 0.154 MB of
activations), which suggests fp8 weights -- but the achieved bandwidth is only
~24 GB/s, **9% of peak**, so batch 1 is not bandwidth-bound and halving the
weights would not have helped. It is starved of workgroups.

Two ways to add workgroups. A narrower tile
(`experiments/matmul_bias_f16_wmma_narrow.loom`, 64x32, one accumulator per
wave) doubles them and **loses**: 0.70x on down at batch 1, 0.68-0.93x
elsewhere, because halving the output per workgroup halves arithmetic intensity
and that costs more than the parallelism gains.

Splitting the k range keeps the 64x64 tile's intensity and multiplies the
workgroup count instead. `experiments/matmul_splitk_f16_wmma.loom` puts the k
chunk on a third grid axis, each workgroup accumulating into its own slab of an
f32 partials buffer; `splitk_reduce_f16` sums the slabs, adds the bias and
narrows to f16. Down at batch 1, splits=4: **18.1 us + 2.9 us against 32.4 us**,
96 workgroups instead of 24, stable across runs where the baseline drifts
32-49 us on a contended box.

It is a batch-1 optimisation only -- 0.80x at batch 4, 0.48x at batch 32 -- so
it is gated at `rows <= splitk_rows` (default 201, i.e. batch 1; `--splitk-rows`
overrides). Batch 2 measured 0.986x end to end, inside the noise but not a win,
so the gate excludes it.

End to end at batch 1, repeat=800, best-of-7 interleaved: **531.0 -> 625.9
img/s, 1.179x** (an earlier best-of-5 at repeat=300 gave 552.7 -> 632.3,
1.144x). That clears torch max-autotune's 594.3 at batch 1, the one
configuration this repo had never won. Accuracy unchanged at cosine 0.99998.

A note on measurement: in one round batch 32 read 1.099x between two configs
that differ only by a flag that is *gated off* at that batch -- i.e. identical
work. The noise floor on this box, with a 10-core CPU job resident, is about
+/-10% on end-to-end throughput at low repeat counts. Anything under ~1.15x
needs repeat=800 and best-of-7 before it means anything.

## Lever 11: k-block 64 -- rejected, and what it says about the ceiling

The k-loop pays two workgroup barriers per k-block. At k=384 with a 32-deep
block that is 12 trips and 24 barriers for 24 MMAs per wave. Doubling the block
to 64 halves both, and the LDS arithmetic works out exactly: A and W staging at
64x72 f16 is 9216 B each, and with the result tiles laid over them the arena is
18432 B -- **3 workgroups/CU, the measured optimum**, unchanged. It compiles at
96 VGPRs, inside the 128 available at that occupancy.

It loses. Two runs, batch 32: qkv 0.92x/0.93x, gate/up 0.96x/0.92x, o 0.78x/1.07x,
down 1.01x/1.04x. So the barriers were not the bottleneck -- doubling the block
doubles the LDS staging per trip as well, and the loop is bound by the
LDS->register fragment loads and the MMA issue rate rather than by
synchronisation.

**That is the fifth consecutive failed attempt to make this matmul faster** --
128x64 no-LDS (Lever 6), LDS aliasing for occupancy (7), 64x128 register
blocking (8), 64x32 narrow (10), and now k-depth. The structure resists tile
shape, occupancy, LDS budget, register blocking and k-block depth alike. The
46% of peak that gate/up reaches is what this kernel design gives on this part.

### So: is there a path to 2000 img/s?

Not in fp16. From 1333 img/s (0.750 ms/img, 16.3 TFLOP/s effective):

| | ms/img | img/s |
| --- | ---: | ---: |
| today | 0.750 | 1333 |
| fuse *all* elementwise away | 0.625 | 1599 |
| + attention 9.5 -> 20 TFLOP/s | 0.585 | 1711 |
| + every matmul at gate/up's 46% | 0.511 | 1959 |

The last row is the fp16 ceiling and it is still under 2000 -- and it assumes
zero elementwise cost, attention more than doubled, and the down/qkv/o matmuls
all lifted to the best matmul's efficiency after five attempts to lift any of
them have failed. Realistically fp16 tops out around **1600-1700**.

**int8 WMMA is the only thing that moves the ceiling.** RDNA3.5 runs
`v_wmma_i32_16x16x16_iu8` at 1024 ops/clk/CU against f16's 512, so peak roughly
doubles to ~106 TOPS, and Loom exposes the op (`amdgpu.v_wmma_i32_16x16x16_iu8`,
alongside iu4 and a 16x16x64 iu8 form). Halving matmul time at today's
utilisation:

| int8 utilisation | ms/img | img/s |
| --- | ---: | ---: |
| same as today's f16 | 0.449 | **2228** |
| 75% of today's | 0.540 | 1851 |
| 60% of today's | 0.631 | 1584 |

So 2000 is reachable through int8 and essentially nowhere else. The work is
per-channel weight quantisation, activation scales, an int32 accumulator and a
requantising epilogue, with the residual stream, norms and softmax staying
floating point. The open question is accuracy: this repo currently sits at
cosine 0.99998 against a 0.997 gate, which is a lot of headroom, but int8 PTQ on
a 12-layer ViT with LayerScale needs measuring, not assuming.

Note also that fp8 is *not* an option here: gfx1151 is RDNA3.5 and its WMMA has
no fp8 form (the fp8 opcodes in the Loom tables are for later targets). fp8
weights would also not have helped batch 1, which is workgroup-starved rather
than bandwidth-starved -- see Lever 10.

## Lever 12: attention LDS aliasing -- rejected, same law again

The online attention kernel is one wave per workgroup, so LDS per workgroup sets
occupancy directly: 4608 B (512 B score scratch + 4096 B result stage) allowed
14 waves/CU against the 18 its 168 VGPRs permit. The two have disjoint lifetimes
-- the scratch is dead once the key loop's last barrier retires -- so aliasing
them into one 4096 B arena costs nothing and lifts the ceiling to 16 waves.
Publishing the result in two 16x32 halves instead gets 2048 B and the full 18.

Standalone it looked promising (4096 B: 1.04x, 1.05x, 0.96x at batch 32 across
runs; a consistent ~1.2x at batch 8). End to end it is not there:

| batch | 4608 B / 14 waves | 4096 B / 16 waves | |
| ---: | ---: | ---: | ---: |
| 4 | 1191.7 | 1191.8 | 1.000x |
| 8 | 1207.2 | 1226.0 | 1.016x |
| 16 | 1300.1 | 1349.4 | 1.038x |
| 32 | 1363.7 | 1323.1 | 0.970x |

Batch 16 and batch 32 disagree in sign, so this is inside the noise. Reverted.

That is the second time raising occupancy by aliasing dead LDS has failed to pay
(Lever 7 was the matmul), and it is the same conclusion: **on this part the
occupancy that matters is already being reached, and spare LDS is not a
resource that converts into throughput.** The standalone-vs-end-to-end gap is
also worth noting -- a kernel measured alone at 1.2x contributed nothing in the
model, because the stage it belongs to is not what the pass is waiting on.

## Lever 13: the residual stream in f16 -- shipped

`residual+norm` is a third of the model's memory traffic: at batch 32 it runs
twice per layer and each launch moves 29.6 MB (read x, write x, read branch,
write h). Two thirds of that is the f32 residual stream itself, 39.5 MB per
layer of ~182 MB.

Narrowing `x` to f16 touches four base kernels -- `residual_layernorm_f32`,
`residual_scale_f32`, `layernorm_f32`, `embed_scatter_f32` -- and the three
variants generated from them. Nothing else reads `x`: the matmuls consume `h`,
the normed output, which was already f16.

**Accuracy is unchanged: cosine 0.9999829 against 0.9999808 before.** The worry
was that 24 roundings of an accumulating residual stream would compound, and it
does not, because LayerNorm renormalises the stream every layer -- the absolute
scale never drifts far enough for f16's ~3 decimal digits to matter.

The stage is **14.7% faster** (107.6 -> 91.8 ms at batch 32, best of 3 profiled
runs), which matches the prediction: a 33% traffic cut on a stage measured at
54% of peak bandwidth.

End to end that is ~1.6% and **not measurable on this box**. Best-of-5 at
repeat=20 read 1.039x at batch 32 and 1.114x at batch 8, which looked like a
win; best-of-7 at repeat=50 read 1.003x and the batch-1 path 0.995x. The first
set was noise. Shipped anyway on the strength of the stage measurement and the
11% cut in total memory traffic -- which is worth more than the wall clock
suggests on an APU whose bandwidth is shared with the CPU.

The measurement lesson is the same one from Lever 12, sharper: when a change is
worth a couple of percent end to end, **profile the stage it touches** rather
than trying to resolve it in throughput. The stage measurement was clean and
repeatable at 14.7%; the end-to-end measurement of the same change swung between
0.995x and 1.114x depending on the protocol.

## Lever 14: RoPE folded into the QKV epilogue -- shipped

RoPE was two full read-modify-write passes over q and k per layer, 19.8 MB of
traffic at batch 32 for no FLOPs. Lever 9 measured it at an apparent 333 GB/s --
above what the memory can do, so it was being served from cache, having just
been written by the QKV matmul. That made it look like a poor target. It is not:
the traffic may be cached, but the pass still costs a launch and a round trip
through the cache hierarchy, and folding it into the producer removes both.

The rotation pairs channel `c` with `c+32` inside a 64-wide head. The n-tile is
64 wide and head-aligned, so both partners are inside the workgroup -- but in
*different waves*: `wave_n` 0 owns tile columns 0-31 and `wave_n` 1 owns 32-63,
as subgroups 2m and 2m+1. They swap through the staging slots the epilogue
already writes, which costs widening the two epilogue barriers from subgroup to
workgroup scope. Occupancy is unchanged (64 VGPRs, 18432 B).

Two things fall out of the layout. Each wave reads `cos`/`sin` at *its own*
column: for a head-aligned 64-wide tile the channel inside the head is exactly
`out_col_local`, so the low wave computes `low*cos - high*sin` and the high wave
`high*cos + low*sin` from the same table entry. And the partner's bias sits 32
columns away and has to be applied before the rotation mixes them.

On the combined qkv+rope stage, best-of-N profiled:

| batch | separate | fused | |
| ---: | ---: | ---: | ---: |
| 1 | 163.0 ms | 115.4 | **1.41x** |
| 2 | 165.8 | 126.5 | 1.31x |
| 4 | 131.1 | 96.2 | 1.36x |
| 8 | 101.1 | 83.2 | 1.22x |
| 32 | 165.9 | 152.3 | 1.09x |

End to end **1.027x at batch 1 (599.7 -> 615.6) and 1.027x at batch 32
(1392.2 -> 1429.9)**. Both batches agreeing on the same figure is what makes it
credible at this size; a single batch reading 1.03x would not have been.
Accuracy unchanged at cosine 0.99998.

Two Loom notes. The partner wave index must be rebuilt as `2*wave_m + (1 -
wave_n)` rather than `subgroup +/- 1`: the latter is the same value but its range
does not follow from `wave_m`'s, so the staging read cannot prove in bounds. And
a `buffer.view` with a *dynamic* base offset does not lower for `vector.load`
(`source_memory.view_base`), so the partner slot is reached through one flat
`view<128x16xf32>` over the whole staging area instead.

A measurement warning: a standalone harness put this at **0.64x** at batch 1,
against the in-model 1.41x. The standalone compared one fused launch against a
matmul plus two rope launches timed separately at repeat=50, which amortises
launch overhead differently from the real sequence. Where the two disagree, the
in-model number is the one that counts.
