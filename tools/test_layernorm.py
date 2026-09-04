"""LayerNorm kernel vs a float64 NumPy reference."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

HIDDEN = 384          # DINOv3 ViT-S+
EPS = 1e-5
CASES = [1, 3, 201, 1024]   # 201 = 196 patches + CLS + 4 registers


def main() -> int:
    ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        hsaco = tmp / "layernorm.hsaco"
        compile_kernel(ROOT / "kernels/layernorm_f32.loom", "dinov3_layernorm_f32",
                       {"dinov3.layernorm_f32.hidden_size": HIDDEN,
                        "dinov3.layernorm_f32.epsilon": EPS}, hsaco)
        print(f"compiled layernorm_f32 for hidden={HIDDEN}")
        rng = np.random.default_rng(0)
        for tokens in CASES:
            # layernorm reads the residual stream, which is f16.
            x = (rng.standard_normal((tokens, HIDDEN), dtype=np.float32) * 3.0 + 1.5).astype(np.float16)
            gamma = rng.standard_normal(HIDDEN, dtype=np.float32)
            beta = rng.standard_normal(HIDDEN, dtype=np.float32)
            (y,), timing = launch(
                hsaco, "dinov3_layernorm_f32", (tokens, 1, 1), (256, 1, 1),
                [("i32", tokens), ("in_f16", x), ("in", gamma), ("in", beta),
                 ("out", ((tokens, HIDDEN), np.float32))],
                tmp, repeat=50)
            xd = x.astype(np.float64)
            mean = xd.mean(axis=1, keepdims=True)
            var = xd.var(axis=1, keepdims=True)
            expected = ((xd - mean) / np.sqrt(var + EPS) * gamma + beta)
            ok &= report(f"tokens={tokens:<5d} ({timing['per_launch_us']:7.2f} us)", y, expected,
                         atol=2e-3, rtol=2e-3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
