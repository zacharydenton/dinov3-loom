"""Flatten the DINOv3 ViT-S+ safetensors into one f32 blob plus a manifest.

Weights land in exactly the layout the kernels want:
  * nn.Linear weights stay [N, K] (matmul_bias_f32 reads them row-major)
  * the patch-embed Conv2d weight is reshaped to [384, 768], which is the same
    matmul once the image is patchified with a 16x16 stride-16 grid
  * k_proj has no bias in this config, so a zero vector is emitted for it
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as R

OUT = Path(__file__).resolve().parent.parent / "build/weights"


def main() -> None:
    weights = R.load_weights()
    OUT.mkdir(parents=True, exist_ok=True)
    blob, manifest, offset = [], [], 0
    # Projection weights also go out as f16 for the WMMA matmul, in a separate
    # blob so the f32 manifest keeps float-indexed offsets.
    blob16, manifest16, offset16 = [], [], 0

    def emit(name: str, array: np.ndarray) -> None:
        nonlocal offset
        flat = np.ascontiguousarray(array, dtype=np.float32).ravel()
        blob.append(flat)
        manifest.append(f"{name} {offset} {flat.size}")
        offset += flat.size

    def emit16(name: str, array: np.ndarray) -> None:
        nonlocal offset16
        flat = np.ascontiguousarray(array, dtype=np.float16).ravel()
        blob16.append(flat)
        manifest16.append(f"{name} {offset16} {flat.size}")
        offset16 += flat.size

    def emit_both(name: str, array: np.ndarray) -> None:
        emit(name, array)
        emit16(name, array)

    emit_both("patch_w", weights["embeddings.patch_embeddings.weight"].reshape(R.HIDDEN, -1))
    emit("patch_b", weights["embeddings.patch_embeddings.bias"])
    emit("prefix", np.concatenate([
        weights["embeddings.cls_token"].reshape(1, R.HIDDEN),
        weights["embeddings.register_tokens"].reshape(R.REGISTERS, R.HIDDEN)]))

    for layer in range(R.LAYERS):
        p = f"layer.{layer}."
        emit(f"l{layer}_norm1_w", weights[p + "norm1.weight"])
        emit(f"l{layer}_norm1_b", weights[p + "norm1.bias"])
        # q, k and v share one [3*hidden, hidden] matmul: three launches of 24
        # workgroups become one of 72, which matters a lot on a 40-CU part.
        emit_both(f"l{layer}_qkv_w", np.concatenate([
            weights[p + "attention.q_proj.weight"],
            weights[p + "attention.k_proj.weight"],
            weights[p + "attention.v_proj.weight"]]))
        emit(f"l{layer}_qkv_b", np.concatenate([
            weights[p + "attention.q_proj.bias"],
            np.zeros(R.HIDDEN),                            # key_bias is false
            weights[p + "attention.v_proj.bias"]]))
        emit_both(f"l{layer}_o_w", weights[p + "attention.o_proj.weight"])
        emit(f"l{layer}_o_b", weights[p + "attention.o_proj.bias"])
        emit(f"l{layer}_ls1", weights[p + "layer_scale1.lambda1"])
        emit(f"l{layer}_norm2_w", weights[p + "norm2.weight"])
        emit(f"l{layer}_norm2_b", weights[p + "norm2.bias"])
        emit_both(f"l{layer}_gateup_w", np.concatenate([
            weights[p + "mlp.gate_proj.weight"], weights[p + "mlp.up_proj.weight"]]))
        emit(f"l{layer}_gateup_b", np.concatenate([
            weights[p + "mlp.gate_proj.bias"], weights[p + "mlp.up_proj.bias"]]))
        emit_both(f"l{layer}_down_w", weights[p + "mlp.down_proj.weight"])
        emit(f"l{layer}_down_b", weights[p + "mlp.down_proj.bias"])
        emit(f"l{layer}_ls2", weights[p + "layer_scale2.lambda1"])

    emit("norm_w", weights["norm.weight"])
    emit("norm_b", weights["norm.bias"])
    cos, sin = R.rope_tables()
    emit("rope_cos", cos)
    emit("rope_sin", sin)

    np.concatenate(blob).tofile(OUT / "weights.bin")
    (OUT / "manifest.txt").write_text("\n".join(manifest) + "\n")
    np.concatenate(blob16).tofile(OUT / "weights_f16.bin")
    (OUT / "manifest_f16.txt").write_text("\n".join(manifest16) + "\n")
    print(f"wrote {OUT}/weights.bin ({offset * 4 / 1e6:.1f} MB, {len(manifest)} tensors)")
    print(f"wrote {OUT}/weights_f16.bin ({offset16 * 2 / 1e6:.1f} MB, {len(manifest16)} tensors)")


if __name__ == "__main__":
    main()
