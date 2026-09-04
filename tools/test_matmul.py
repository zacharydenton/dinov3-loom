"""matmul_bias_f32 vs a float64 NumPy reference, over every shape DINOv3 ViT-S+ uses."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

# (name, M, K, N)
SHAPES = [
    ("patch-embed", 196, 768, 384),
    ("qkvo       ", 201, 384, 384),
    ("mlp-gate/up", 201, 384, 1536),
    ("mlp-down   ", 201, 1536, 384),
    ("odd-M      ", 37, 384, 384),
]


def main() -> int:
    ok = True
    rng = np.random.default_rng(1)
    with workdir() as tmp:
        tmp = Path(tmp)
        for name, m, k, n in SHAPES:
            hsaco = tmp / f"mm_{k}_{n}.hsaco"
            compile_kernel(ROOT / "kernels/matmul_bias_f32.loom", "dinov3_matmul_bias_f32",
                           {"dinov3.matmul_bias_f32.k_size": k,
                            "dinov3.matmul_bias_f32.n_size": n}, hsaco)
            a = rng.standard_normal((m, k), dtype=np.float32) * 0.1
            w = rng.standard_normal((n, k), dtype=np.float32) * 0.1
            bias = rng.standard_normal(n, dtype=np.float32)
            grid = (n // 64, (m + 63) // 64, 1)
            (c,), timing = launch(hsaco, "dinov3_matmul_bias_f32", grid, (256, 1, 1),
                                  [("i32", m), ("in", a), ("in", w), ("in", bias),
                                   ("out", ((m, n), np.float32))], tmp, repeat=50)
            expected = a.astype(np.float64) @ w.astype(np.float64).T + bias
            flops = 2.0 * m * k * n
            gflops = flops / (timing["per_launch_us"] * 1e-6) / 1e9
            ok &= report(f"{name} M={m:4d} K={k:5d} N={n:5d} "
                         f"({timing['per_launch_us']:7.2f} us, {gflops:6.1f} GFLOP/s)",
                         c, expected, atol=1e-4, rtol=1e-4)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
