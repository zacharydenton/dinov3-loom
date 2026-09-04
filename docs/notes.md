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
