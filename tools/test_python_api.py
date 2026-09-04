"""Lifecycle, error-boundary and correctness tests for the resident Python API."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dinov3_loom import DINOv3Error, DINOv3Loom, HIDDEN, TOKENS, patchify


def image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((3, 224, 224), dtype=np.float32)


def require_raises(kind, phrase: str, call) -> None:
    try:
        call()
    except kind as failure:
        assert phrase in str(failure), str(failure)
    else:
        raise AssertionError(f"expected {kind.__name__} containing {phrase!r}")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.ravel().astype(np.float64)
    right = right.ravel().astype(np.float64)
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def cli(batch: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.bin"
        output = Path(directory) / "output.bin"
        patchify(batch).tofile(source)
        environment = dict(os.environ)
        environment.pop("LD_LIBRARY_PATH", None)
        subprocess.run(
            [str(ROOT / "host/dinov3"), "--weights", str(ROOT / "build/weights"),
             "--kernels", str(ROOT / "build/kernels"), "--input", str(source),
             "--output", str(output), "--batch", str(len(batch))],
            check=True, capture_output=True, cwd=ROOT, env=environment,
        )
        return np.fromfile(output, np.float32).reshape(len(batch), TOKENS, HIDDEN)


def main() -> None:
    require_raises(ValueError, "B >= 1", lambda: patchify(np.empty((0, 3, 224, 224))))
    require_raises(ValueError, "expected", lambda: patchify(np.empty((1, 3, 223, 224))))
    require_raises(ValueError, "1..64", lambda: DINOv3Loom(max_batch=65))

    # Native initialization failures must return through the ABI, not call
    # exit() and take this Python interpreter with them.
    with tempfile.TemporaryDirectory() as empty_weights:
        require_raises(
            DINOv3Error, "manifest.txt",
            lambda: DINOv3Loom(weights=empty_weights, max_batch=1),
        )
    with tempfile.TemporaryDirectory() as empty_kernels:
        require_raises(
            DINOv3Error, "hipModuleLoad",
            lambda: DINOv3Loom(kernels=empty_kernels, max_batch=1),
        )

    batch = np.stack([image(seed) for seed in range(4)])
    expected_four = cli(batch)
    expected_chunks = np.concatenate((cli(batch[:2]), cli(batch[2:])))
    expected_one = cli(batch[:1])

    with DINOv3Loom(max_batch=2) as model:
        assert model._native.dinov3_max_batch(model._handle) == 2

        # Four images take two resident calls; each must bit-match the same
        # launch shape through the CLI. Different total batch sizes can round
        # differently, but remain well above the project's accuracy threshold.
        got = model(batch)
        assert got.shape == (4, TOKENS, HIDDEN)
        assert got.dtype == np.float32
        assert np.array_equal(got, expected_chunks)
        assert all(cosine(got[i], expected_four[i]) > 0.99999 for i in range(4))

        # Varying batch sizes exercises both the normal and batch-1 split-K
        # paths without reallocating or leaking state between calls.
        first = model(batch[0])
        saved = first.copy()
        model(batch[1:3])
        assert np.array_equal(first, saved), "a later call mutated an earlier result"
        assert np.array_equal(first, expected_one)

        # The ABI itself rejects undersized buffers before launching anything.
        patches = patchify(batch[:1])
        output = np.empty((1, TOKENS, HIDDEN), np.float32)
        error = ctypes.create_string_buffer(1024)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        status = model._native.dinov3_run(
            model._handle,
            patches.ctypes.data_as(float_pointer), patches.size - 1,
            output.ctypes.data_as(float_pointer), output.size, 1,
            error, len(error),
        )
        assert status == 64
        assert b"input has" in error.value
        error.value = b""
        status = model._native.dinov3_run(
            model._handle,
            patches.ctypes.data_as(float_pointer), patches.size,
            output.ctypes.data_as(float_pointer), output.size - 1, 1,
            error, len(error),
        )
        assert status == 64
        assert b"output has" in error.value

        # The Python lock protects the shared device/output buffers while
        # ctypes releases the GIL around native calls.
        want = [model(batch[i]) for i in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            threaded = list(pool.map(model, [batch[0], batch[1]]))
        assert all(np.array_equal(a, b) for a, b in zip(threaded, want))

        # A second object owns a second native session and can coexist with the
        # first. Closing either must not invalidate the other.
        with DINOv3Loom(max_batch=1) as other:
            assert np.array_equal(other(batch[0]), expected_one)
        assert np.array_equal(model(batch[0]), expected_one)

    assert model.closed
    model.close()
    require_raises(DINOv3Error, "closed", lambda: model(batch[0]))
    print("  PASS resident Python API: ABI errors, batching, threads, sessions, close")


if __name__ == "__main__":
    main()
