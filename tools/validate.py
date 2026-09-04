"""End-to-end validation: the Loom runner vs HF transformers DINOv3 ViT-S+.

Compares the full 201x384 output, then the two embeddings anyone actually uses
(the CLS token and the mean of the patch tokens).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as R

ROOT = Path(__file__).resolve().parent.parent


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_image(seed: int) -> np.ndarray:
    """A deterministic, image-like input: smooth low-frequency content, not noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:224, 0:224] / 224.0
    base = np.stack([
        np.sin(6 * np.pi * x) * np.cos(4 * np.pi * y),
        np.exp(-((x - 0.3) ** 2 + (y - 0.6) ** 2) * 8),
        (x + y) / 2,
    ])
    return (base + 0.15 * rng.standard_normal((3, 224, 224))).astype(np.float64)


def main() -> int:
    # Keep torch on the CPU so the reference is exact f32 and does not fight the
    # runner for the GPU; the runner gets its own environment back below.
    gpu_env = dict(os.environ)
    gpu_env.pop("HIP_VISIBLE_DEVICES", None)
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(R.snapshot()), dtype=torch.float32).eval()
    ok = True
    for seed in (0, 1, 2):
        image = make_image(seed)
        # Deliberately not build/patchified.bin: that file is the benchmark's
        # input, and clobbering it here silently invalidates every hand
        # comparison against a saved reference afterwards.
        R.patchify(image).astype(np.float32).tofile("/tmp/validate_patchified.bin")
        subprocess.run([str(ROOT / "host/dinov3"),
                        "--weights", str(ROOT / "build/weights"),
                        "--kernels", str(ROOT / "build/kernels"),
                        "--input", "/tmp/validate_patchified.bin",
                        "--output", "/tmp/loom_validate.bin"] + sys.argv[1:],
                       check=True, cwd=ROOT,
                       capture_output=True, env=gpu_env)
        loom = np.fromfile("/tmp/loom_validate.bin", dtype=np.float32).reshape(R.TOKENS, R.HIDDEN)

        with torch.no_grad():
            hf = model(pixel_values=torch.from_numpy(image[None]).float()) \
                .last_hidden_state[0].numpy()

        full = cosine(loom, hf)
        cls = cosine(loom[0], hf[0])
        mean = cosine(loom[R.PREFIX:].mean(0), hf[R.PREFIX:].mean(0))
        max_abs = np.abs(loom.astype(np.float64) - hf.astype(np.float64)).max()
        # Gate on the full output too: CLS and the patch mean can both look
        # healthy while individual patch tokens are wrong.
        good = full > 0.9999 and cls > 0.9999 and mean > 0.9999
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} image {seed}: "
              f"cosine full={full:.10f} cls={cls:.10f} mean-patch={mean:.10f} "
              f"max_abs={max_abs:.2e}")
    print("\nreference: xdna-vision gates DINOv3 at cosine > 0.997")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
