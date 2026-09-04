"""embed_scatter_f32 vs NumPy: CLS, the register tokens and the patches land in
the right rows of the residual stream, for a batch, and nothing is read from
outside either source tensor.

This is the kernel that carried an out-of-bounds read through validation: it
loaded the 5-row prefix tensor at every patch row and discarded the value, so
the model was right and the kernel was not. The prefix and patch buffers here
are allocated at exactly their declared size, so an over-read is a fault, not a
silently discarded value.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import reference as R

T, HID, PREFIX, PATCHES = R.TOKENS, R.HIDDEN, R.PREFIX, R.PATCHES
P = "dinov3.embed_scatter_f32"


def main() -> int:
    ok = True
    rng = np.random.default_rng(11)
    with workdir() as tmp:
        tmp = Path(tmp)
        hsaco = tmp / "scatter.hsaco"
        compile_kernel(ROOT / "kernels/embed_scatter_f32.loom", "dinov3_embed_scatter_f32",
                       {f"{P}.hidden_size": HID, f"{P}.tokens_per_image": T, f"{P}.prefix": PREFIX},
                       hsaco)
        for batch in (1, 3):
            rows = batch * T
            patches = rng.standard_normal((batch * PATCHES, HID)).astype(np.float32)
            prefix = rng.standard_normal((PREFIX, HID)).astype(np.float32)   # exactly 5 rows
            (x,), timing = launch(hsaco, "dinov3_embed_scatter_f32", (rows, 1, 1), (256, 1, 1),
                                  [("i32", rows), ("in", patches), ("in", prefix),
                                   ("out_f16", ((rows, HID), np.float16))], tmp, repeat=1)
            want = np.empty((rows, HID), np.float64)
            for b in range(batch):
                want[b * T:b * T + PREFIX] = prefix
                want[b * T + PREFIX:(b + 1) * T] = patches[b * PATCHES:(b + 1) * PATCHES]
            ok &= report(f"batch={batch} ({timing['per_launch_us']:6.2f} us)", x, want,
                         atol=2e-3, rtol=2e-3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
