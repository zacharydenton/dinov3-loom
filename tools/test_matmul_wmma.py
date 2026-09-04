"""matmul_bias_f16_wmma vs a float64 NumPy reference, over the model's shapes.

A is f32 in memory and narrowed on the way into LDS; W is f16; accumulation is
f32. The reference rounds both operands to f16 the same way, so the comparison
isolates the kernel from the dtype choice.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

SHAPES = [
    ("patch-embed", 196, 768, 384),
    ("o          ", 201, 384, 384),
    ("qkv        ", 201, 384, 1152),
    ("gate/up    ", 201, 384, 3072),
    ("down       ", 201, 1536, 384),
    ("batch32 qkv", 6432, 384, 1152),
    ("batch32 g/u", 6432, 384, 3072),
    ("batch32 down", 6432, 1536, 384),
]


def main() -> int:
    ok = True
    rng = np.random.default_rng(1)
    with workdir() as tmp:
        tmp = Path(tmp)
        for name, m, k, n in SHAPES:
            hsaco = tmp / f"wmma_{k}_{n}.hsaco"
            compile_kernel(ROOT / "kernels/matmul_bias_f16_wmma.loom",
                           "dinov3_matmul_bias_f16_wmma",
                           {"dinov3.matmul_bias_f16_wmma.k_size": k,
                            "dinov3.matmul_bias_f16_wmma.n_size": n}, hsaco)
            a = (rng.standard_normal((m, k)) * 0.1).astype(np.float32)
            w = (rng.standard_normal((n, k)) * 0.1).astype(np.float16)
            bias = rng.standard_normal(n).astype(np.float32)
            grid = (n // 64, (m + 63) // 64, 1)
            (c,), timing = launch(hsaco, "dinov3_matmul_bias_f16_wmma", grid, (256, 1, 1),
                                  [("i32", m), ("in", a), ("in_f16", w), ("in", bias),
                                   ("out", ((m, n), np.float32))], tmp, repeat=50)
            expected = (a.astype(np.float16).astype(np.float64)
                        @ w.astype(np.float64).T + bias)
            gflops = 2.0 * m * k * n / (timing["per_launch_us"] * 1e-6) / 1e9
            ok &= report(f"{name} M={m:5d} K={k:5d} N={n:5d} "
                         f"({timing['per_launch_us']:8.2f} us, {gflops:7.1f} GFLOP/s)",
                         c, expected, atol=2e-3, rtol=2e-3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
