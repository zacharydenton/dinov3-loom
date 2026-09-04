"""The batched Python API vs transformers, on distinct images.

validate.py runs one image at a time, so it cannot catch a batch that folds
images together or returns them in the wrong order. This does: four different
images through DINOv3Loom in one call, each compared against its own HF output,
plus a cross-image check that would fail if the batch were being replicated.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))          # reference
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # dinov3_loom
import reference as R
from dinov3_loom import DINOv3Loom


def image(seed: int) -> np.ndarray:
    """Deterministic, image-like: smooth low-frequency content, not noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:R.IMAGE, 0:R.IMAGE] / R.IMAGE
    base = np.stack([
        np.sin((3 + seed) * np.pi * x) * np.cos(4 * np.pi * y),
        np.exp(-((x - 0.2 * seed) ** 2 + (y - 0.6) ** 2) * 8),
        (x + y) / 2,
    ])
    return (base + 0.15 * rng.standard_normal((3, R.IMAGE, R.IMAGE))).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    import torch
    from transformers import AutoModel

    batch = np.stack([image(s) for s in range(4)])
    hf = AutoModel.from_pretrained(str(R.snapshot())).eval()
    with torch.no_grad():
        want = hf(pixel_values=torch.from_numpy(batch)).last_hidden_state.numpy()
    with DINOv3Loom() as loom:
        # Exercise the drop-in path from a torch tensor as well as coexistence
        # between torch's ROCm libraries and the resident C ABI.
        got = loom(torch.from_numpy(batch))

    if got.shape != want.shape:
        print(f"  FAIL shape {got.shape} != {want.shape}")
        return 1

    ok = True
    for i in range(len(batch)):
        c = cosine(got[i], want[i])
        good = c > 0.9999
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} image {i}: cosine {c:.10f}")

    # If the runner replicated one image across the batch, or returned them out
    # of order, every row would match every other. These inputs are similar by
    # construction, so the bar is loose -- it only has to catch identity.
    cross = max(cosine(got[i], want[j])
                for i in range(len(batch)) for j in range(len(batch)) if i != j)
    distinct = cross < 0.999
    ok &= distinct
    print(f"  {'PASS' if distinct else 'FAIL'} images are distinct: "
          f"worst cross-image cosine {cross:.6f} (must be < 0.999)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
