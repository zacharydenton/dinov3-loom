"""Float64 NumPy reference for DINOv3 ViT-S+/16, matching transformers exactly.

Kept deliberately independent of torch so it can be the ground truth the Loom
kernels are graded against.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

MODEL_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"


def _snapshot() -> Path:
    """Locate the model, downloading it on first use.

    DINOV3_SNAPSHOT overrides, for offline or vendored copies. Otherwise this
    resolves through huggingface_hub, which caches under HF_HOME.
    """
    override = os.environ.get("DINOV3_SNAPSHOT")
    if override:
        return Path(override)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - dependency guidance
        raise SystemExit(
            "huggingface_hub is required to resolve the model. Install the "
            "dependencies (pip install -r requirements.txt), or point "
            "DINOV3_SNAPSHOT at an existing snapshot directory."
        )
    return Path(snapshot_download(MODEL_ID))


SNAPSHOT = _snapshot()

HIDDEN = 384
HEADS = 6
HEAD_DIM = HIDDEN // HEADS
LAYERS = 12
INTERMEDIATE = 1536
PATCH = 16
IMAGE = 224
REGISTERS = 4
PREFIX = 1 + REGISTERS
GRID = IMAGE // PATCH
PATCHES = GRID * GRID
TOKENS = PREFIX + PATCHES
EPS = 1e-5
ROPE_THETA = 100.0


def load_weights(path: Path = SNAPSHOT / "model.safetensors") -> dict[str, np.ndarray]:
    return {k: v.astype(np.float64) for k, v in load_file(str(path)).items()}


def rope_tables() -> tuple[np.ndarray, np.ndarray]:
    """cos/sin of shape (PATCHES, HEAD_DIM), float64."""
    coords_h = (np.arange(0.5, GRID) / GRID)
    coords_w = (np.arange(0.5, GRID) / GRID)
    grid_h, grid_w = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack([grid_h, grid_w], axis=-1).reshape(-1, 2)      # (196, 2)
    coords = 2.0 * coords - 1.0
    inv_freq = 1.0 / ROPE_THETA ** np.arange(0.0, 1.0, 4.0 / HEAD_DIM)   # (16,)
    angles = 2.0 * math.pi * coords[:, :, None] * inv_freq[None, None, :]  # (196, 2, 16)
    angles = angles.reshape(PATCHES, -1)                                   # (196, 32)
    angles = np.tile(angles, 2)                                            # (196, 64)
    return np.cos(angles), np.sin(angles)


def patchify(pixel_values: np.ndarray) -> np.ndarray:
    """(3, 224, 224) -> (196, 768) in Conv2d weight order [c, kh, kw]."""
    c, h, w = pixel_values.shape
    x = pixel_values.reshape(c, GRID, PATCH, GRID, PATCH)      # c, gh, kh, gw, kw
    x = x.transpose(1, 3, 0, 2, 4)                             # gh, gw, c, kh, kw
    return x.reshape(PATCHES, c * PATCH * PATCH)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * weight + bias


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def forward(pixel_values: np.ndarray, weights: dict[str, np.ndarray],
            capture: dict | None = None) -> np.ndarray:
    """pixel_values: (3, 224, 224) float64. Returns (TOKENS, HIDDEN) after the final norm."""
    patch_weight = weights["embeddings.patch_embeddings.weight"].reshape(HIDDEN, -1)
    patch_bias = weights["embeddings.patch_embeddings.bias"]
    patches = patchify(pixel_values) @ patch_weight.T + patch_bias      # (196, 384)

    x = np.concatenate([
        weights["embeddings.cls_token"].reshape(1, HIDDEN),
        weights["embeddings.register_tokens"].reshape(REGISTERS, HIDDEN),
        patches,
    ], axis=0)                                                          # (201, 384)
    if capture is not None:
        capture["embeddings"] = x.copy()

    cos, sin = rope_tables()

    for layer in range(LAYERS):
        p = f"layer.{layer}."
        residual = x
        h = layer_norm(x, weights[p + "norm1.weight"], weights[p + "norm1.bias"])

        q = h @ weights[p + "attention.q_proj.weight"].T + weights[p + "attention.q_proj.bias"]
        k = h @ weights[p + "attention.k_proj.weight"].T
        v = h @ weights[p + "attention.v_proj.weight"].T + weights[p + "attention.v_proj.bias"]

        q = q.reshape(TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)       # (H, T, D)
        k = k.reshape(TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = v.reshape(TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)

        qp, kp = q[:, PREFIX:, :], k[:, PREFIX:, :]
        q = np.concatenate([q[:, :PREFIX, :], qp * cos + rotate_half(qp) * sin], axis=1)
        k = np.concatenate([k[:, :PREFIX, :], kp * cos + rotate_half(kp) * sin], axis=1)
        if capture is not None and layer == 0:
            capture["q0"], capture["k0"], capture["v0"] = q.copy(), k.copy(), v.copy()

        scores = (q @ k.transpose(0, 2, 1)) * (HEAD_DIM ** -0.5)        # (H, T, T)
        context = softmax(scores) @ v                                    # (H, T, D)
        context = context.transpose(1, 0, 2).reshape(TOKENS, HIDDEN)
        attn = context @ weights[p + "attention.o_proj.weight"].T + weights[p + "attention.o_proj.bias"]
        x = residual + attn * weights[p + "layer_scale1.lambda1"]

        residual = x
        h = layer_norm(x, weights[p + "norm2.weight"], weights[p + "norm2.bias"])
        gate = h @ weights[p + "mlp.gate_proj.weight"].T + weights[p + "mlp.gate_proj.bias"]
        up = h @ weights[p + "mlp.up_proj.weight"].T + weights[p + "mlp.up_proj.bias"]
        mlp = (silu(gate) * up) @ weights[p + "mlp.down_proj.weight"].T + weights[p + "mlp.down_proj.bias"]
        x = residual + mlp * weights[p + "layer_scale2.lambda1"]
        if capture is not None:
            capture[f"layer{layer}"] = x.copy()

    return layer_norm(x, weights["norm.weight"], weights["norm.bias"])


def pooled(output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cls, mean-of-patches) — the two embeddings xdna-vision exposes."""
    return output[0], output[PREFIX:].mean(axis=0)
