# experiments

## `rocwmma_gemm.cpp` — what WMMA is worth on gfx1151

rocWMMA is a HIP C++ header library, so it cannot be called from a Loom kernel:
Loom emits AMDGPU machine code directly with no LLVM and no HIP in the loop.
They are alternative toolchains, not composable layers. Loom's own equivalent is
the `vector.mma` op, which has exactly the RDNA3 WMMA shapes:

```
vector.mma : vector<16xf16>, vector<16xf16>, vector<4xf32>    // f32 accumulate
vector.mma : vector<16xf16>, vector<16xf16>, vector<8xf16>    // f16 accumulate
```

paired with `vector.fragment.load<lhs>` / `<rhs>`, which handle the per-lane
fragment layout the way `load_matrix_sync` does. ggml-hrx's own kernel corpus is
built on those two ops.

What rocWMMA *is* good for here is measuring the prize. This benchmark runs a
plain (deliberately untuned) rocWMMA fp16 GEMM with f32 accumulation over the
four shapes DINOv3 ViT-S+ uses, verified against host dot products on random
data:

```
rocWMMA fp16 (f32 accumulate), gfx1151, batch 4 -> M=804
qkv        M=804 K= 384 N=1152     49.95 us   14242.2 GFLOP/s  rel_err=2.9e-05
o          M=804 K= 384 N= 384     27.47 us    8631.4 GFLOP/s  rel_err=4.1e-05
gate/up    M=804 K= 384 N=3072     91.49 us   20733.0 GFLOP/s  rel_err=5.2e-06
down       M=804 K=1536 N= 384    145.51 us    6518.1 GFLOP/s  rel_err=2.7e-05
```

The f32 scalar-FMA matmul in `kernels/` peaks around 2000 GFLOP/s on the same
shapes, and the whole model runs at roughly 2.8 TFLOP/s effective. So WMMA is
worth **7-10x on the matmuls**, which is the entire remaining gap to torch's fp16
path and rather more than the 1.2-4.9x that shows up end to end (the model is
only ~70% matmul, and torch carries its own overheads).

```console
$ /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 -I/opt/rocm/include \
    -o experiments/rocwmma_gemm experiments/rocwmma_gemm.cpp
$ ./experiments/rocwmma_gemm 4 -v     # batch, verbose
```

Note the corpus stages its fragments through padded LDS views (`view<64x40xf16>`,
not 64x32) — the same bank-conflict avoidance that was worth 10x in the f32
matmul and 4x in attention.
