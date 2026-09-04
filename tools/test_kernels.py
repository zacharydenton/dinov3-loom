"""rope_2d, attention, swiglu and residual_scale vs float64 NumPy."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import reference as R

T, H, D, HID, INT, PREFIX = R.TOKENS, R.HEADS, R.HEAD_DIM, R.HIDDEN, R.INTERMEDIATE, R.PREFIX


def main() -> int:
    ok = True
    rng = np.random.default_rng(3)
    with workdir() as tmp:
        tmp = Path(tmp)

        # --- RoPE -----------------------------------------------------------
        hsaco = tmp / "rope.hsaco"
        compile_kernel(ROOT / "kernels/rope_2d_f32.loom", "dinov3_rope_2d_f32",
                       {"dinov3.rope_2d_f32.hidden_size": HID,
                        "dinov3.rope_2d_f32.head_dim": D,
                        "dinov3.rope_2d_f32.prefix": PREFIX,
                        "dinov3.rope_2d_f32.row_stride": HID,
                        "dinov3.rope_2d_f32.tokens_per_image": T}, hsaco)
        x = rng.standard_normal((T, HID), dtype=np.float32)
        cos, sin = R.rope_tables()
        cos32, sin32 = cos.astype(np.float32), sin.astype(np.float32)
        # In-place kernels must run exactly once for the correctness check.
        (y,), timing = launch(hsaco, "dinov3_rope_2d_f32", (T, 1, 1), (256, 1, 1),
                              [("i32", T), ("inout", (x, (T, HID))), ("in", cos32), ("in", sin32)],
                              tmp, repeat=1)
        xh = x.astype(np.float64).reshape(T, H, D).transpose(1, 0, 2)
        p = xh[:, PREFIX:, :]
        expected = xh.copy()
        expected[:, PREFIX:, :] = p * cos + R.rotate_half(p) * sin
        expected = expected.transpose(1, 0, 2).reshape(T, HID)
        ok &= report(f"rope_2d       ({timing['per_launch_us']:7.2f} us)", y, expected)

        # --- attention ------------------------------------------------------
        hsaco = tmp / "attn.hsaco"
        compile_kernel(ROOT / "kernels/attention_f32.loom", "dinov3_attention_f32",
                       {"dinov3.attention_f32.hidden_size": HID,
                        "dinov3.attention_f32.head_dim": D,
                        "dinov3.attention_f32.scale": D ** -0.5,
                        "dinov3.attention_f32.qkv_stride": HID,
                        "dinov3.attention_f32.tokens_per_image": T}, hsaco)
        q = rng.standard_normal((T, HID), dtype=np.float32)
        k = rng.standard_normal((T, HID), dtype=np.float32)
        v = rng.standard_normal((T, HID), dtype=np.float32)
        (o,), timing = launch(hsaco, "dinov3_attention_f32", ((T + 7) // 8, H, 1), (256, 1, 1),
                              [("i32", T), ("in", q), ("in", k), ("in", v),
                               ("out", ((T, HID), np.float32))], tmp, repeat=100)
        qh = q.astype(np.float64).reshape(T, H, D).transpose(1, 0, 2)
        kh = k.astype(np.float64).reshape(T, H, D).transpose(1, 0, 2)
        vh = v.astype(np.float64).reshape(T, H, D).transpose(1, 0, 2)
        ctx = R.softmax((qh @ kh.transpose(0, 2, 1)) * D ** -0.5) @ vh
        expected = ctx.transpose(1, 0, 2).reshape(T, HID)
        ok &= report(f"attention     ({timing['per_launch_us']:7.2f} us)", o, expected, atol=1e-4, rtol=1e-4)

        # --- swiglu ---------------------------------------------------------
        hsaco = tmp / "swiglu.hsaco"
        compile_kernel(ROOT / "kernels/swiglu_f32.loom", "dinov3_swiglu_f32",
                       {"dinov3.swiglu_f32.width": INT,
                        "dinov3.swiglu_f32.row_stride": INT}, hsaco)
        gate = rng.standard_normal((T, INT), dtype=np.float32) * 2.0
        up = rng.standard_normal((T, INT), dtype=np.float32)
        (s,), timing = launch(hsaco, "dinov3_swiglu_f32", (T, 1, 1), (256, 1, 1),
                              [("i32", T), ("in", gate), ("in", up),
                               ("out", ((T, INT), np.float32))], tmp, repeat=100)
        expected = R.silu(gate.astype(np.float64)) * up.astype(np.float64)
        ok &= report(f"swiglu        ({timing['per_launch_us']:7.2f} us)", s, expected, atol=1e-5, rtol=1e-5)

        # --- residual + layerscale -----------------------------------------
        hsaco = tmp / "residual.hsaco"
        compile_kernel(ROOT / "kernels/residual_scale_f32.loom", "dinov3_residual_scale_f32",
                       {"dinov3.residual_scale_f32.hidden_size": HID}, hsaco)
        base = rng.standard_normal((T, HID), dtype=np.float32)
        branch = rng.standard_normal((T, HID), dtype=np.float32)
        lam = rng.standard_normal(HID, dtype=np.float32)
        (r,), timing = launch(hsaco, "dinov3_residual_scale_f32", (T, 1, 1), (256, 1, 1),
                              [("i32", T), ("inout", (base, (T, HID))), ("in", branch), ("in", lam)],
                              tmp, repeat=1)
        expected = base.astype(np.float64) + branch.astype(np.float64) * lam
        ok &= report(f"residual+ls   ({timing['per_launch_us']:7.2f} us)", r, expected)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
