"""The three matmuls with fused epilogues, against float64 NumPy references.

  qkv+rope   A@W + bias, then DINOv3's axial RoPE on the q and k thirds
  resid      x += lambda * (A@W + bias)
  splitk     the same, with the k range cut four ways and reduced
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import reference as R

T, H, D, HID = R.TOKENS, R.HEADS, R.HEAD_DIM, R.HIDDEN
QKV, PREFIX, INTER = 3 * HID, 5, 4 * HID


def rope_tables(rng):
    pos = np.arange(T - PREFIX)
    freq = np.arange(D) * 0.013
    return (np.cos(np.outer(pos, freq)).astype(np.float32),
            np.sin(np.outer(pos, freq)).astype(np.float32))


def check_qkv_rope(tmp, rng, rows):
    P = "dinov3.matmul_qkv_rope_f16_wmma"
    hsaco = tmp / "qkv_rope.hsaco"
    compile_kernel(ROOT / "kernels/matmul_qkv_rope_f16_wmma.loom",
                   "dinov3_matmul_qkv_rope_f16_wmma",
                   {f"{P}.k_size": HID, f"{P}.n_size": QKV, f"{P}.head_dim": D,
                    f"{P}.prefix": PREFIX, f"{P}.tokens_per_image": T,
                    f"{P}.rope_channels": 2 * HID}, hsaco)
    a = (rng.standard_normal((rows, HID)) * 0.1).astype(np.float16)
    w = (rng.standard_normal((QKV, HID)) * 0.1).astype(np.float16)
    bias = rng.standard_normal(QKV).astype(np.float32)
    cos, sin = rope_tables(rng)
    cosb = np.zeros((rows, D), np.float32); sinb = np.zeros((rows, D), np.float32)
    cosb[:T - PREFIX] = cos; sinb[:T - PREFIX] = sin
    (got,), timing = launch(hsaco, "dinov3_matmul_qkv_rope_f16_wmma",
                            (QKV // 64, (rows + 63) // 64, 1), (256, 1, 1),
                            [("i32", rows), ("in_f16", a), ("in_f16", w), ("in", bias),
                             ("out_f16", ((rows, QKV), np.float16)),
                             ("in", cosb), ("in", sinb)], tmp, repeat=20)
    want = a.astype(np.float64) @ w.astype(np.float64).T + bias.astype(np.float64)
    for third in range(2):                       # q and k rotate, v does not
        sl = slice(third * HID, (third + 1) * HID)
        block = want[:, sl].reshape(rows, H, D)
        for t in range(rows):
            in_image = t % T
            if in_image < PREFIX:
                continue
            p = in_image - PREFIX
            low = block[t, :, :D // 2].copy(); high = block[t, :, D // 2:].copy()
            block[t, :, :D // 2] = low * cos[p, :D // 2] - high * sin[p, :D // 2]
            block[t, :, D // 2:] = high * cos[p, D // 2:] + low * sin[p, D // 2:]
        want[:, sl] = block.reshape(rows, HID)
    return report(f"qkv+rope   rows={rows:<5d} ({timing['per_launch_us']:7.2f} us)",
                  got, want, atol=6e-2, rtol=6e-2)


def check_resid(tmp, rng, rows, k):
    P = "dinov3.matmul_resid_f16_wmma"
    hsaco = tmp / f"resid_{k}.hsaco"
    compile_kernel(ROOT / "kernels/matmul_resid_f16_wmma.loom",
                   "dinov3_matmul_resid_f16_wmma",
                   {f"{P}.k_size": k, f"{P}.n_size": HID}, hsaco)
    a = (rng.standard_normal((rows, k)) * 0.1).astype(np.float16)
    w = (rng.standard_normal((HID, k)) * 0.1).astype(np.float16)
    bias = rng.standard_normal(HID).astype(np.float32)
    lam = (rng.standard_normal(HID) * 0.1).astype(np.float32)
    stream = (rng.standard_normal((rows, HID))).astype(np.float16)
    (got,), timing = launch(hsaco, "dinov3_matmul_resid_f16_wmma",
                            (HID // 64, (rows + 63) // 64, 1), (256, 1, 1),
                            [("i32", rows), ("in_f16", a), ("in_f16", w), ("in", bias),
                             ("inout_f16", (stream, (rows, HID))), ("in", lam)],
                            # repeat=1: this kernel accumulates into the stream,
                            # so a second launch would add the branch twice.
                            tmp, repeat=1)
    want = (stream.astype(np.float64)
            + lam.astype(np.float64)
            * (a.astype(np.float64) @ w.astype(np.float64).T + bias.astype(np.float64)))
    return report(f"resid k={k:<5d} rows={rows:<5d} ({timing['per_launch_us']:7.2f} us)",
                  got, want, atol=6e-2, rtol=6e-2)


def check_splitk(tmp, rng, rows):
    S, R2 = "dinov3.matmul_splitk_f16_wmma", "dinov3.splitk_reduce_f16"
    k, splits = 4 * HID, 4
    hm = tmp / "splitk.hsaco"; hr = tmp / "splitk_reduce.hsaco"
    compile_kernel(ROOT / "kernels/matmul_splitk_f16_wmma.loom", "dinov3_matmul_splitk_f16_wmma",
                   {f"{S}.k_size": k, f"{S}.n_size": HID, f"{S}.splits": splits}, hm)
    compile_kernel(ROOT / "kernels/splitk_reduce_f16.loom", "dinov3_splitk_reduce_f16",
                   {f"{R2}.n_size": HID, f"{R2}.splits": splits}, hr)
    a = (rng.standard_normal((rows, k)) * 0.1).astype(np.float16)
    w = (rng.standard_normal((HID, k)) * 0.1).astype(np.float16)
    bias = rng.standard_normal(HID).astype(np.float32)
    lam = (rng.standard_normal(HID) * 0.1).astype(np.float32)
    stream = (rng.standard_normal((rows, HID))).astype(np.float16)
    (partials,), _ = launch(hm, "dinov3_matmul_splitk_f16_wmma",
                            (HID // 64, (rows + 63) // 64, splits), (256, 1, 1),
                            [("i32", rows), ("in_f16", a), ("in_f16", w), ("in", bias),
                             ("out", ((splits * rows, HID), np.float32))], tmp, repeat=10)
    (got,), timing = launch(hr, "dinov3_splitk_reduce_f16", (rows, 1, 1), (256, 1, 1),
                            [("i32", rows), ("in", partials), ("in", bias),
                             ("inout_f16", (stream, (rows, HID))), ("in", lam)],
                            tmp, repeat=1)   # accumulates into the stream
    want = (stream.astype(np.float64)
            + lam.astype(np.float64)
            * (a.astype(np.float64) @ w.astype(np.float64).T + bias.astype(np.float64)))
    return report(f"splitk     rows={rows:<5d} ({timing['per_launch_us']:7.2f} us)",
                  got, want, atol=6e-2, rtol=6e-2)


def main() -> int:
    ok = True
    rng = np.random.default_rng(7)
    with workdir() as tmp:
        tmp = Path(tmp)
        for rows in (T, 8 * T):
            ok &= check_qkv_rope(tmp, rng, rows)
        for k in (HID, INTER):
            ok &= check_resid(tmp, rng, 8 * T, k)
        ok &= check_splitk(tmp, rng, T)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
