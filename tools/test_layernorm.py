"""layernorm_rowwave_f16 and _f32out vs a float64 NumPy reference.

The f16-out kernel feeds every matmul; the f32-out one is the final norm."""
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
        rng = np.random.default_rng(0)
        for variant, out_kind, out_dtype in (("f16", "out_f16", np.float16),
                                             ("f32out", "out", np.float32)):
          name = f"dinov3_layernorm_rowwave_{variant}"
          hsaco = tmp / f"{variant}.hsaco"
          compile_kernel(ROOT / f"kernels/layernorm_rowwave_{variant}.loom", name,
                         {f"dinov3.layernorm_rowwave_{variant}.hidden_size": HIDDEN,
                          f"dinov3.layernorm_rowwave_{variant}.epsilon": EPS}, hsaco)
          print(f"compiled {name} for hidden={HIDDEN}")
          for tokens in CASES:
            # layernorm reads the residual stream, which is f16.
            x = (rng.standard_normal((tokens, HIDDEN), dtype=np.float32) * 3.0 + 1.5).astype(np.float16)
            gamma = rng.standard_normal(HIDDEN, dtype=np.float32)
            beta = rng.standard_normal(HIDDEN, dtype=np.float32)
            (y,), timing = launch(
                hsaco, name, ((tokens + 7) // 8, 1, 1), (256, 1, 1),
                [("i32", tokens), ("in_f16", x), ("in", gamma), ("in", beta),
                 (out_kind, ((tokens, HIDDEN), out_dtype))],
                tmp, repeat=50)
            xd = x.astype(np.float64)
            mean = xd.mean(axis=1, keepdims=True)
            var = xd.var(axis=1, keepdims=True)
            expected = ((xd - mean) / np.sqrt(var + EPS) * gamma + beta)
            ok &= report(f"{variant:<6s} tokens={tokens:<5d} ({timing['per_launch_us']:7.2f} us)", y, expected,
                         atol=2e-3, rtol=2e-3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
