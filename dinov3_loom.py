"""Drop-in replacement for a PyTorch DINOv3 ViT-S+/16 forward pass.

    from dinov3_loom import DINOv3Loom
    model = DINOv3Loom()
    tokens = model(pixel_values)          # (B, 201, 384) float32

`pixel_values` is what an HF image processor produces: (B, 3, 224, 224),
already resized and normalised. The return matches
`AutoModel(...).last_hidden_state` -- token 0 is CLS, tokens 1..4 are the
register tokens, 5.. are the 196 patches.

The GPU work runs in `host/dinov3`, one subprocess per call. That costs a few
milliseconds of process start and two file copies, so this is worth it for
batches, not for single images in a tight loop. Everything here is numpy; torch
is only needed if the caller happens to be using it.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

GRID, PATCH = 14, 16
PATCHES = GRID * GRID          # 196
PATCH_K = 3 * PATCH * PATCH    # 768
TOKENS, HIDDEN = 201, 384
MAX_BATCH = 64                 # the kernels' compiled max_images


def patchify(pixel_values: np.ndarray) -> np.ndarray:
    """(B, 3, 224, 224) -> (B, 196, 768), in Conv2d weight order [c, kh, kw]."""
    b, c, h, w = pixel_values.shape
    if (c, h, w) != (3, GRID * PATCH, GRID * PATCH):
        raise ValueError(f"expected (B, 3, {GRID * PATCH}, {GRID * PATCH}), got {pixel_values.shape}")
    x = pixel_values.reshape(b, c, GRID, PATCH, GRID, PATCH)
    x = x.transpose(0, 2, 4, 1, 3, 5)          # b, gh, gw, c, kh, kw
    return x.reshape(b, PATCHES, PATCH_K)


class DINOv3Loom:
    """Runs the Loom kernels through host/dinov3."""

    def __init__(self, weights: str | Path | None = None,
                 kernels: str | Path | None = None,
                 runner: str | Path | None = None):
        # Resolved against the caller's cwd *now*, because the subprocess runs
        # with cwd=ROOT: a relative path that passed exists() here would
        # otherwise fail there.
        self.weights = Path(weights or ROOT / "build/weights").resolve()
        self.kernels = Path(kernels or ROOT / "build/kernels").resolve()
        self.runner = Path(runner or ROOT / "host/dinov3").resolve()
        for path, hint in ((self.runner, "hipcc -O2 -o host/dinov3 host/dinov3.cpp"),
                           (self.kernels, "./scripts/build_kernels.sh"),
                           (self.weights, "python3 tools/export_weights.py")):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; run: {hint}")

    def __call__(self, pixel_values) -> np.ndarray:
        """(B, 3, 224, 224) -> (B, 201, 384) float32, batches of at most 64."""
        array = np.asarray(getattr(pixel_values, "cpu", lambda: pixel_values)(),
                           dtype=np.float32)
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4 or len(array) == 0:
            raise ValueError(f"expected (B, 3, 224, 224) with B >= 1, got {array.shape}")
        out = [self._run(array[i:i + MAX_BATCH])
               for i in range(0, len(array), MAX_BATCH)]
        return np.concatenate(out, axis=0)

    def _run(self, batch: np.ndarray) -> np.ndarray:
        n = len(batch)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            np.ascontiguousarray(patchify(batch), dtype=np.float32).tofile(src)
            # The runner links its own ROCm; a caller that has sourced
            # scripts/env.sh would otherwise hand it the HRX loader.
            env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
            subprocess.run(
                [str(self.runner), "--weights", str(self.weights),
                 "--kernels", str(self.kernels), "--input", str(src),
                 "--output", str(dst), "--batch", str(n)],
                check=True, capture_output=True, cwd=ROOT, env=env)
            return np.fromfile(dst, dtype=np.float32).reshape(n, TOKENS, HIDDEN)

    # Convenience mirrors of what people actually take off DINOv3.
    def cls(self, pixel_values) -> np.ndarray:
        return self(pixel_values)[:, 0]

    def patch_mean(self, pixel_values) -> np.ndarray:
        return self(pixel_values)[:, TOKENS - PATCHES:].mean(axis=1)
