"""Benchmark HF DINOv3 ViT-S+ on the same GPU, for comparison with host/dinov3."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as R
from transformers import AutoModel


def bench(model, batch, dtype, steps=30, warmup=10):
    x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=dtype)
    with torch.no_grad():
        for _ in range(warmup):
            model(pixel_values=x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            model(pixel_values=x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return batch * steps / elapsed


def main() -> None:
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    rows = []
    for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        model = AutoModel.from_pretrained(str(R.SNAPSHOT), dtype=dtype).eval().cuda()
        for batch in (1, 8, 64):
            rows.append((f"eager {dtype_name}", batch, bench(model, batch, dtype)))
            print(f"  {rows[-1][0]:<16s} batch={batch:<3d} {rows[-1][2]:8.2f} img/s", flush=True)
        try:
            compiled = torch.compile(model)
            for batch in (1, 8, 64):
                rows.append((f"compile {dtype_name}", batch, bench(compiled, batch, dtype)))
                print(f"  {rows[-1][0]:<16s} batch={batch:<3d} {rows[-1][2]:8.2f} img/s", flush=True)
        except Exception as exc:  # torch.compile is fragile on ROCm
            print(f"  compile {dtype_name}: FAILED ({type(exc).__name__}: {str(exc)[:120]})")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
