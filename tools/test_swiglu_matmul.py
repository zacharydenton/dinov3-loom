"""matmul_swiglu_f16_wmma vs float64 NumPy.

  C = silu(A @ Wgate^T + bgate) * (A @ Wup^T + bup)

W is [2n, k] with the gate rows first and the up rows second, and the bias is
laid out the same way, so this is exactly what tools/export_weights.py emits as
gateup_w / gateup_b. n_size is the per-projection width (INTERMEDIATE), not the
fused 2n.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import reference as R

HID, INTER = R.HIDDEN, 4 * R.HIDDEN
P = "dinov3.matmul_swiglu_f16_wmma"


def main() -> int:
    ok = True
    rng = np.random.default_rng(13)
    with workdir() as tmp:
        tmp = Path(tmp)
        hsaco = tmp / "swiglu.hsaco"
        compile_kernel(ROOT / "kernels/matmul_swiglu_f16_wmma.loom", "dinov3_matmul_swiglu_f16_wmma",
                       {f"{P}.k_size": HID, f"{P}.n_size": INTER}, hsaco)
        for rows in (R.TOKENS, 8 * R.TOKENS):
            a = (rng.standard_normal((rows, HID)) * 0.1).astype(np.float16)
            w = (rng.standard_normal((2 * INTER, HID)) * 0.1).astype(np.float16)
            bias = rng.standard_normal(2 * INTER).astype(np.float32)
            (c,), timing = launch(hsaco, "dinov3_matmul_swiglu_f16_wmma",
                                  (INTER // 64, (rows + 63) // 64, 1), (256, 1, 1),
                                  [("i32", rows), ("in_f16", a), ("in_f16", w), ("in", bias),
                                   ("out_f16", ((rows, INTER), np.float16))], tmp, repeat=20)
            pre = a.astype(np.float64) @ w.astype(np.float64).T + bias.astype(np.float64)
            gate, up = pre[:, :INTER], pre[:, INTER:]
            want = R.silu(gate) * up
            flops = 2.0 * rows * HID * 2 * INTER
            ok &= report(f"rows={rows:<5d} ({timing['per_launch_us']:8.2f} us, "
                         f"{flops / (timing['per_launch_us'] * 1e-6) / 1e12:5.1f} TFLOP/s)",
                         c, want, atol=6e-2, rtol=6e-2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
