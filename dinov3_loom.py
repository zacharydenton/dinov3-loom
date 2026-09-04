"""DINOv3 ViT-S+/16 inference through resident Loom kernels.

    import dinov3_loom

    model = dinov3_loom.DINOv3Loom()
    tokens = model(pixel_values)          # (B, 201, 384) float32
    model.close()

``pixel_values`` is the array an HF image processor produces: float-compatible
``(B, 3, 224, 224)`` data, already resized and normalized. The return matches
``AutoModel(...).last_hidden_state``: token 0 is CLS, tokens 1..4 are registers,
and tokens 5..200 are the 196 patches.

Importing this module does not initialize HIP. Each model owns an independent
native session; construction uploads the weights and subsequent calls reuse its
GPU allocations and loaded kernels.
"""
from __future__ import annotations

import ctypes
import operator
import os
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

GRID, PATCH = 14, 16
PATCHES = GRID * GRID
PATCH_K = 3 * PATCH * PATCH
TOKENS, HIDDEN = 201, 384
MAX_BATCH = 64
DEFAULT_MAX_BATCH = 32

_ABI_VERSION = 1
_ERROR_CAPACITY = 4096
_FloatPointer = ctypes.POINTER(ctypes.c_float)


class DINOv3Error(RuntimeError):
    """The native DINOv3 runtime could not initialize or complete a call."""


def patchify(pixel_values: np.ndarray) -> np.ndarray:
    """Convert ``(B, 3, 224, 224)`` to contiguous patch-major f32 data."""
    array = np.asarray(pixel_values, dtype=np.float32)
    expected = (3, GRID * PATCH, GRID * PATCH)
    if array.ndim != 4 or array.shape[0] < 1 or tuple(array.shape[1:]) != expected:
        raise ValueError(
            f"expected (B, 3, {GRID * PATCH}, {GRID * PATCH}) with B >= 1, "
            f"got {array.shape}"
        )
    b, c, _, _ = array.shape
    array = np.ascontiguousarray(array)
    array = array.reshape(b, c, GRID, PATCH, GRID, PATCH)
    array = array.transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(array.reshape(b, PATCHES, PATCH_K))


def _path(value: str | os.PathLike[str] | None, environment: str,
          default: Path) -> Path:
    if value is None:
        value = os.environ.get(environment, default)
    return Path(value).expanduser().resolve()


def _message(error: ctypes.Array[ctypes.c_char], fallback: str) -> str:
    return error.value.decode("utf-8", errors="replace") or fallback


class DINOv3Loom:
    """A resident, reusable Loom inference session.

    Calls on one object are serialized and may safely come from multiple Python
    threads. Use separate objects for independent sessions. A session inherited
    across ``fork()`` is deliberately rejected because HIP state is not
    fork-safe.
    """

    def __init__(
        self,
        weights: str | os.PathLike[str] | None = None,
        kernels: str | os.PathLike[str] | None = None,
        library: str | os.PathLike[str] | None = None,
        max_batch: int = DEFAULT_MAX_BATCH,
    ) -> None:
        try:
            max_batch = operator.index(max_batch)
        except TypeError as failure:
            raise TypeError("max_batch must be an integer") from failure
        if not 1 <= max_batch <= MAX_BATCH:
            raise ValueError(f"max_batch must be in 1..{MAX_BATCH}, got {max_batch}")

        self.weights = _path(weights, "DINOV3_LOOM_WEIGHTS", ROOT / "build/weights")
        self.kernels = _path(kernels, "DINOV3_LOOM_KERNELS", ROOT / "build/kernels")
        self.library = _path(library, "DINOV3_LOOM_LIBRARY", ROOT / "build/libdinov3.so")
        for path, hint in (
            (self.library, "./scripts/build_host.sh"),
            (self.kernels, "./scripts/build_kernels.sh"),
            (self.weights, "python3 tools/export_weights.py"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; run: {hint}")

        try:
            native = ctypes.CDLL(self.library)
        except OSError as failure:
            raise DINOv3Error(f"cannot load {self.library}: {failure}") from failure
        try:
            self._configure(native)
        except AttributeError as failure:
            raise DINOv3Error(
                f"{self.library} does not expose the DINOv3 ABI; rebuild it with "
                "./scripts/build_host.sh"
            ) from failure
        abi = native.dinov3_abi_version()
        if abi != _ABI_VERSION:
            raise DINOv3Error(
                f"{self.library} uses ABI {abi}; Python expects ABI {_ABI_VERSION}; "
                "rebuild it with ./scripts/build_host.sh"
            )

        # Allocate the reusable host output before the native session so a rare
        # NumPy allocation failure cannot strand an already-created GPU session.
        self._output = np.empty((max_batch, TOKENS, HIDDEN), dtype=np.float32)
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._native = native
        self._handle: ctypes.c_void_p | None = None

        handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = native.dinov3_create(
            os.fsencode(self.weights), os.fsencode(self.kernels), max_batch,
            ctypes.byref(handle), error, len(error),
        )
        if status:
            raise DINOv3Error(_message(error, f"dinov3_create failed with status {status}"))
        if not handle:
            raise DINOv3Error("dinov3_create succeeded without returning a session")
        self._handle = handle
        self.max_batch = max_batch

    @staticmethod
    def _configure(native: ctypes.CDLL) -> None:
        native.dinov3_abi_version.argtypes = []
        native.dinov3_abi_version.restype = ctypes.c_uint32
        native.dinov3_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        native.dinov3_create.restype = ctypes.c_int
        native.dinov3_run.argtypes = [
            ctypes.c_void_p,
            _FloatPointer,
            ctypes.c_size_t,
            _FloatPointer,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        native.dinov3_run.restype = ctypes.c_int
        native.dinov3_max_batch.argtypes = [ctypes.c_void_p]
        native.dinov3_max_batch.restype = ctypes.c_int
        native.dinov3_destroy.argtypes = [ctypes.c_void_p]
        native.dinov3_destroy.restype = None

    @property
    def closed(self) -> bool:
        """Whether this model has released its native session."""
        return self._handle is None

    def _ensure_usable(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise DINOv3Error("this DINOv3Loom session is closed")
        if os.getpid() != self._pid:
            raise DINOv3Error(
                "this DINOv3Loom session was inherited across fork; create a new "
                "model in the child process"
            )
        return self._handle

    def close(self) -> None:
        """Release all GPU allocations and loaded modules; safe to call twice."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            handle = getattr(self, "_handle", None)
            self._handle = None
            # Destroying inherited HIP state in a forked child is itself unsafe.
            if handle is not None and os.getpid() == self._pid:
                self._native.dinov3_destroy(handle)

    def __enter__(self) -> DINOv3Loom:
        self._ensure_usable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors run during interpreter teardown, when module globals
            # and the dynamic loader may already be partly dismantled.
            pass

    def __call__(self, pixel_values) -> np.ndarray:
        """Return ``(B, 201, 384)`` f32 tokens, splitting large batches."""
        value = pixel_values
        detach = getattr(value, "detach", None)
        if detach is not None:
            value = detach()
        cpu = getattr(value, "cpu", None)
        if cpu is not None:
            value = cpu()
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 3:
            array = array[None]
        patches = patchify(array)

        with self._lock:
            self._ensure_usable()
            pieces = [
                self._run(patches[start:start + self.max_batch])
                for start in range(0, len(patches), self.max_batch)
            ]
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)

    def _run(self, patches: np.ndarray) -> np.ndarray:
        handle = self._ensure_usable()
        batch = len(patches)
        output = self._output[:batch]
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = self._native.dinov3_run(
            handle,
            patches.ctypes.data_as(_FloatPointer), patches.size,
            output.ctypes.data_as(_FloatPointer), output.size, batch,
            error, len(error),
        )
        if status:
            raise DINOv3Error(_message(error, f"dinov3_run failed with status {status}"))
        # The native output buffer is reused, so every public result must own
        # its data and remain unchanged by later calls.
        return output.copy()

    def cls(self, pixel_values) -> np.ndarray:
        """Return the CLS token for each image."""
        return self(pixel_values)[:, 0]

    def patch_mean(self, pixel_values) -> np.ndarray:
        """Return the mean of the 196 patch tokens for each image."""
        return self(pixel_values)[:, TOKENS - PATCHES:].mean(axis=1)


__all__ = [
    "DEFAULT_MAX_BATCH",
    "DINOv3Error",
    "DINOv3Loom",
    "HIDDEN",
    "MAX_BATCH",
    "PATCHES",
    "TOKENS",
    "patchify",
]
