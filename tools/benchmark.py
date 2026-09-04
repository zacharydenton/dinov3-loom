"""Interleaved benchmark: Loom runner vs torch DINOv3 ViT-S+ on the same GPU.

The machine this was written on runs other jobs, so the GPU spends much of its
time at a low DPM state. Runs are interleaved and the best of N rounds is kept
for every configuration, which is the honest way to compare under contention:
the best sample is the one least disturbed by whatever else was resident.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as R

ROOT = Path(__file__).resolve().parent.parent
ROUNDS = 5


def loom_throughput(images: int = 30) -> float:
    result = subprocess.run(
        [str(ROOT / "host/dinov3"), "--input", str(ROOT / "build/patchified.bin"),
         "--repeat", str(images)], check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(result.stdout.strip().splitlines()[-1])["img_per_s"]


def torch_throughput(model, batch: int, dtype, steps: int = 20) -> float:
    x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=dtype)
    with torch.no_grad():
        for _ in range(5):
            model(pixel_values=x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            model(pixel_values=x)
        torch.cuda.synchronize()
    return batch * steps / (time.perf_counter() - start)


def main() -> None:
    image = np.load("/tmp/px.npy") if Path("/tmp/px.npy").exists() else \
        np.random.default_rng(0).standard_normal((3, 224, 224))
    R.patchify(image).astype(np.float32).tofile(ROOT / "build/patchified.bin")

    print(f"DINOv3 ViT-S+/16, 224px, 201 tokens — {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__}; best of {ROUNDS} interleaved rounds\n")

    models = {}
    for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        base = AutoModel.from_pretrained(str(R.SNAPSHOT), dtype=dtype).eval().cuda()
        models[f"torch eager {dtype_name}"] = (base, dtype)
        for mode, label in (("default", "compile"), ("max-autotune", "max-autotune")):
            try:
                candidate = torch.compile(base, mode=mode)
                # Force the compile now so a failure is attributed to the mode
                # rather than showing up as a mystery zero later.
                torch_throughput(candidate, 1, dtype, steps=1)
                models[f"torch {label} {dtype_name}"] = (candidate, dtype)
            except Exception as exc:
                detail = str(exc).strip().splitlines()
                print(f"  torch.compile {mode} {dtype_name}: FAILED "
                      f"({type(exc).__name__}: {detail[0][:150] if detail else ''})")

    best: dict[str, float] = {}
    for round_index in range(ROUNDS):
        best["loom fp32 (batch 1)"] = max(best.get("loom fp32 (batch 1)", 0.0),
                                          loom_throughput())
        for name, (model, dtype) in models.items():
            for batch in (1, 64):
                key = f"{name} (batch {batch})"
                best[key] = max(best.get(key, 0.0), torch_throughput(model, batch, dtype))
        print(f"  round {round_index + 1}/{ROUNDS} done", flush=True)

    print(f"\n{'configuration':<32s} {'img/s':>9s}   vs Loom")
    baseline = best["loom fp32 (batch 1)"]
    for name, value in sorted(best.items(), key=lambda kv: -kv[1]):
        marker = "  <-- this repo" if name.startswith("loom") else ""
        print(f"{name:<32s} {value:9.1f}   {value / baseline:5.2f}x{marker}")


if __name__ == "__main__":
    main()
