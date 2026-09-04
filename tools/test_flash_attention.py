"""flash_attention_f16_wmma vs a float64 NumPy reference."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import reference as R

T, H, D, HID = R.TOKENS, R.HEADS, R.HEAD_DIM, R.HIDDEN


def main() -> int:
    ok = True
    rng = np.random.default_rng(5)
    with workdir() as tmp:
        tmp = Path(tmp)
        for batch, stride in ((1, HID), (4, HID)):
            hsaco = tmp / f"fa_{stride}.hsaco"
            compile_kernel(ROOT / "kernels/flash_attention_f16_wmma.loom",
                           "dinov3_flash_attention_f16_wmma",
                           {"dinov3.flash_attention_f16_wmma.hidden_size": HID,
                            "dinov3.flash_attention_f16_wmma.qkv_stride": stride,
                            "dinov3.flash_attention_f16_wmma.tokens_per_image": T,
                            "dinov3.flash_attention_f16_wmma.scale": D ** -0.5}, hsaco)
            rows = batch * T
            # q, k, v live in one [rows, stride] buffer when stride > hidden.
            q = rng.standard_normal((rows, HID), dtype=np.float32)
            k = rng.standard_normal((rows, HID), dtype=np.float32)
            v = rng.standard_normal((rows, HID), dtype=np.float32)
            args = [("i32", rows), ("in", q), ("in", k), ("in", v),
                    ("out", ((rows, HID), np.float32))]
            grid = (batch * ((T + 15) // 16), H, 1)
            (o,), timing = launch(hsaco, "dinov3_flash_attention_f16_wmma", grid,
                                  (256, 1, 1), args, tmp, repeat=50)
            # reference, per image
            expected = np.empty((rows, HID), dtype=np.float64)
            for b in range(batch):
                sl = slice(b * T, (b + 1) * T)
                qh = q[sl].astype(np.float16).astype(np.float64).reshape(T, H, D).transpose(1, 0, 2)
                kh = k[sl].astype(np.float16).astype(np.float64).reshape(T, H, D).transpose(1, 0, 2)
                vh = v[sl].astype(np.float16).astype(np.float64).reshape(T, H, D).transpose(1, 0, 2)
                ctx = R.softmax((qh * D ** -0.5) @ kh.transpose(0, 2, 1)) @ vh
                expected[sl] = ctx.transpose(1, 0, 2).reshape(T, HID)
            flops = batch * H * (2.0 * T * T * D * 2)
            ok &= report(f"batch={batch} stride={stride:5d} "
                         f"({timing['per_launch_us']:8.2f} us, "
                         f"{flops / (timing['per_launch_us'] * 1e-6) / 1e9:7.1f} GFLOP/s)",
                         o, expected, atol=3e-3, rtol=3e-3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
