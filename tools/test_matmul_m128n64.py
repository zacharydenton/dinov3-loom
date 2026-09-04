"""matmul_bias_f16_wmma_m128n64 (no LDS in the k-loop) vs the staged af16 kernel."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

SHAPES = [
    ("o           ", 6432, 384, 384),
    ("qkv         ", 6432, 384, 1152),
    ("gate/up     ", 6432, 384, 3072),
    ("down        ", 6432, 1536, 384),
    ("down b8     ", 1608, 1536, 384),
]
NEW = "dinov3.matmul_bias_f16_wmma_m128n64"
OLD = "dinov3.matmul_bias_f16_wmma_af16"


def main() -> int:
    ok = True
    rng = np.random.default_rng(1)
    print(f"{'shape':<14s} {'m128n64 us':>11s} {'TF/s':>7s} {'af16 us':>9s} {'TF/s':>7s} {'speedup':>8s}")
    with workdir() as tmp:
        tmp = Path(tmp)
        for name, m, k, n in SHAPES:
            flops = 2.0 * m * k * n
            # A is padded to a multiple of 128 rows: a tile may overhang m.
            padded = (m + 127) // 128 * 128
            a = (rng.standard_normal((padded, k)) * 0.1).astype(np.float16)
            w = (rng.standard_normal((n, k)) * 0.1).astype(np.float16)
            bias = rng.standard_normal(n).astype(np.float32)
            expected = (a[:m].astype(np.float64) @ w.astype(np.float64).T
                        + bias.astype(np.float64))

            new_hsaco = tmp / f"new_{k}_{n}.hsaco"
            compile_kernel(ROOT / "experiments/matmul_bias_f16_wmma_m128n64.loom",
                           "dinov3_matmul_bias_f16_wmma_m128n64",
                           {f"{NEW}.k_size": k, f"{NEW}.n_size": n,
                            f"{NEW}.token_capacity": 131072,
                            f"{NEW}.max_m_tiles": 512}, new_hsaco)
            args = [("i32", m), ("in_f16", a), ("in_f16", w), ("in", bias),
                    ("out", ((m, n), np.float32))]
            (c_new,), t_new = launch(new_hsaco, "dinov3_matmul_bias_f16_wmma_m128n64",
                                     (n // 64, (m + 127) // 128, 1), (64, 1, 1),
                                     args, tmp, repeat=50)

            old_hsaco = tmp / f"old_{k}_{n}.hsaco"
            compile_kernel(ROOT / "kernels/matmul_bias_f16_wmma_af16.loom",
                           "dinov3_matmul_bias_f16_wmma_af16",
                           {f"{OLD}.k_size": k, f"{OLD}.n_size": n}, old_hsaco)
            (c_old,), t_old = launch(old_hsaco, "dinov3_matmul_bias_f16_wmma_af16",
                                     (n // 64, (m + 63) // 64, 1), (256, 1, 1),
                                     args, tmp, repeat=50)

            un, uo = t_new["per_launch_us"], t_old["per_launch_us"]
            print(f"{name:<14s} {un:11.2f} {flops/(un*1e-6)/1e12:7.1f} "
                  f"{uo:9.2f} {flops/(uo*1e-6)/1e12:7.1f} {uo/un:7.2f}x")
            ok &= report(f"  {name} correctness", c_new, expected, atol=3e-2, rtol=3e-2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
