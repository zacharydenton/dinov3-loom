"""attention_online_f16_wmma (the hrx-demos shape) vs a float64 NumPy reference."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import reference as R

T, H, D, HID = R.TOKENS, R.HEADS, R.HEAD_DIM, R.HIDDEN
CAPACITY = 262144
MAX_IMAGES = 64
PREFIX = "dinov3.attention_online_f16_wmma"


def main() -> int:
    ok = True
    rng = np.random.default_rng(5)
    with workdir() as tmp:
        tmp = Path(tmp)
        # The f32-output source, and the generated f16-output variant the
        # model actually launches; every case runs against both.
        VARIANTS = (("", "out", np.float32), ("_cf16", "out_f16", np.float16))
        CASES = ((1, HID), (4, HID), (4, 3 * HID))
        for (suffix, out_kind, out_dtype), (batch, stride) in itertools.product(VARIANTS, CASES):
            symbol = "dinov3_attention_online_f16_wmma" + suffix
            ns = PREFIX + suffix
            hsaco = tmp / f"ao{suffix}_{batch}_{stride}.hsaco"
            compile_kernel(ROOT / f"kernels/attention_online_f16_wmma{suffix}.loom", symbol,
                           {f"{ns}.hidden_size": HID,
                            f"{ns}.qkv_stride": stride,
                            f"{ns}.tokens_per_image": T,
                            f"{ns}.scale": D ** -0.5,
                            f"{ns}.max_images": MAX_IMAGES,
                            f"{ns}.token_capacity": CAPACITY}, hsaco)
            rows = batch * T
            # A query tile may overhang the last image by up to 15 rows; the
            # kernel masks those lanes but still forms the address, so the
            # buffers carry 16 rows of headroom.
            padded = rows + 16
            # Distinct q, k and v: identical buffers would hide a swapped or
            # mis-strided pointer, which is exactly what this test should catch.
            q, k, v = (rng.standard_normal((padded, stride), dtype=np.float32).astype(np.float16)
                       for _ in range(3))
            args = [("i32", rows), ("in_f16", q), ("in_f16", k), ("in_f16", v),
                    (out_kind, ((rows, HID), out_dtype))]
            grid = (batch * ((T + 15) // 16), H, 1)
            (o,), timing = launch(hsaco, symbol, grid,
                                  (32, 1, 1), args, tmp, repeat=50)
            expected = np.empty((rows, HID), dtype=np.float64)
            for b in range(batch):
                sl = slice(b * T, (b + 1) * T)
                def heads(buf):
                    return buf[sl, :HID].astype(np.float64).reshape(T, H, D).transpose(1, 0, 2)
                qh, kh, vh = heads(q), heads(k), heads(v)
                ctx = R.softmax((qh * D ** -0.5) @ kh.transpose(0, 2, 1)) @ vh
                expected[sl] = ctx.transpose(1, 0, 2).reshape(T, HID)
            flops = batch * H * (2.0 * T * T * D * 2)
            ok &= report(f"{suffix or 'f32':<5s} batch={batch} stride={stride:5d} "
                         f"({timing['per_launch_us']:8.2f} us, "
                         f"{flops / (timing['per_launch_us'] * 1e-6) / 1e9:7.1f} GFLOP/s)",
                         o, expected, atol=3e-3, rtol=3e-3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
