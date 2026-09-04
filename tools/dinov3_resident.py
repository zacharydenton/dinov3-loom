"""DINOv3 ViT-S+/16 on Loom kernels, with the device context held open.

`dinov3_loom.DINOv3Loom` spawns `host/dinov3` per call, which costs ~100ms of
HIP init, module loading and weight upload before any arithmetic happens -- at
batch 32 that is five times the compute it wraps, and it made the fast kernels
lose to torch by 2.3x. This binds the same code as a shared library instead:
init once, then every call is two copies and 87 launches.

    model = DINOv3Resident(max_batch=32)
    tokens = model(pixel_values)      # (B, 3, 224, 224) f32 -> (B, 201, 384) f32

Not thread-safe: one session, one device context. Callers running it beside a
torch model on the same GPU are fine -- the contexts coexist -- but the calls
into this object must be serialised.
"""
import ctypes, os
import numpy as np

HIDDEN, TOKENS, PATCHES, PATCH_K, PATCH = 384, 201, 196, 768, 16
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def patchify(pixel_values):
    """(B, 3, 224, 224) -> (B, 196, 768), the reshape the kernels expect.

    16x16 patches at stride 16, channel-major within a patch, which is what
    `nn.Conv2d(3, 384, 16, stride=16)` sees when its weights are flattened.
    """
    b, c, h, w = pixel_values.shape
    if (c, h, w) != (3, 224, 224):
        raise ValueError(f"expected (B, 3, 224, 224), got {pixel_values.shape}")
    x = np.ascontiguousarray(pixel_values, dtype=np.float32)
    x = x.reshape(b, c, h // PATCH, PATCH, w // PATCH, PATCH)
    return np.ascontiguousarray(x.transpose(0, 2, 4, 1, 3, 5).reshape(b, PATCHES, PATCH_K))


class DINOv3Resident:
    def __init__(self, max_batch=32, lib=None, weights=None, kernels=None):
        lib = lib or os.path.join(ROOT, "build", "libdinov3.so")
        self._lib = ctypes.CDLL(lib)
        self._lib.dinov3_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.dinov3_init.restype = ctypes.c_int
        self._lib.dinov3_run.argtypes = [
            np.ctypeslib.ndpointer(np.float32, ndim=3, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(np.float32, ndim=3, flags="C_CONTIGUOUS"),
            ctypes.c_int]
        self._lib.dinov3_run.restype = ctypes.c_int

        rc = self._lib.dinov3_init(
            (weights or os.path.join(ROOT, "build", "weights")).encode(),
            (kernels or os.path.join(ROOT, "build", "kernels")).encode(), max_batch)
        if rc:
            raise RuntimeError(f"dinov3_init failed ({rc})")
        self.max_batch = max_batch
        # Reused across calls; the C side writes into it.
        self._out = np.empty((max_batch, TOKENS, HIDDEN), dtype=np.float32)

    def __call__(self, pixel_values):
        """(B, 3, 224, 224) f32 -> (B, 201, 384) f32. Splits above max_batch."""
        patches = patchify(pixel_values)
        n = len(patches)
        if n <= self.max_batch:
            return self._run(patches)
        return np.concatenate([self._run(patches[i:i + self.max_batch])
                               for i in range(0, n, self.max_batch)])

    def _run(self, patches):
        n = len(patches)
        rc = self._lib.dinov3_run(patches, self._out, n)
        if rc:
            raise RuntimeError(f"dinov3_run failed ({rc})")
        return self._out[:n].copy()

    def cls(self, pixel_values):
        return self(pixel_values)[:, 0]

    def patch_mean(self, pixel_values):
        return self(pixel_values)[:, 5:].mean(axis=1)
