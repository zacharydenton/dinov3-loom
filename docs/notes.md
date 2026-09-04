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
