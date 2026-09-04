#!/usr/bin/env bash
# Compile every kernel at every configuration the model needs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
out=build/kernels
mkdir -p "$out"

compile() { # name root output-stem config...
  local src="kernels/$1.loom" root="$2" stem="$3"; shift 3
  local args=()
  for c in "$@"; do args+=("--config=$c"); done
  "$LOOM_COMPILE" "$src" --backend=amdgpu-hal --target="$LOOM_TARGET" \
    --root="@$root" "${args[@]}" --output="$out/$stem.hsaco"
  printf '  %-22s %s\n' "$stem" "$(stat -c%s "$out/$stem.hsaco") bytes"
}

echo "compiling for $LOOM_TARGET"
compile layernorm_f32 dinov3_layernorm_f32 layernorm \
  dinov3.layernorm_f32.hidden_size=384 dinov3.layernorm_f32.epsilon=1e-5
compile rope_2d_f32 dinov3_rope_2d_f32 rope \
  dinov3.rope_2d_f32.hidden_size=384 dinov3.rope_2d_f32.head_dim=64 dinov3.rope_2d_f32.prefix=5 \
  dinov3.rope_2d_f32.row_stride=1152 dinov3.rope_2d_f32.tokens_per_image=201
compile attention_f32 dinov3_attention_f32 attention \
  dinov3.attention_f32.hidden_size=384 dinov3.attention_f32.head_dim=64 \
  dinov3.attention_f32.scale=0.125 dinov3.attention_f32.qkv_stride=1152 \
  dinov3.attention_f32.tokens_per_image=201
compile embed_scatter_f32 dinov3_embed_scatter_f32 embed_scatter \
  dinov3.embed_scatter_f32.hidden_size=384 dinov3.embed_scatter_f32.tokens_per_image=201 \
  dinov3.embed_scatter_f32.prefix=5
compile swiglu_f32 dinov3_swiglu_f32 swiglu dinov3.swiglu_f32.width=1536 \
  dinov3.swiglu_f32.row_stride=3072
compile residual_scale_f32 dinov3_residual_scale_f32 residual \
  dinov3.residual_scale_f32.hidden_size=384
# Wide N (>= 1024) has enough 64-wide tiles to fill the GPU; narrow N does not,
# so those get the 64x32 tile, which doubles the workgroup count.
for kn in 768:384 384:1152 384:3072 384:384 1536:384; do
  k="${kn%%:*}"; n="${kn##*:}"
  compile matmul_bias_f32 dinov3_matmul_bias_f32 "matmul_k${k}_n${n}" \
    "dinov3.matmul_bias_f32.k_size=$k" "dinov3.matmul_bias_f32.n_size=$n"
done
for kn in 384:384 1536:384; do
  k="${kn%%:*}"; n="${kn##*:}"
  compile matmul_bias_f32_narrow dinov3_matmul_bias_f32_narrow "matmul_narrow_k${k}_n${n}" \
    "dinov3.matmul_bias_f32_narrow.k_size=$k" "dinov3.matmul_bias_f32_narrow.n_size=$n"
done
