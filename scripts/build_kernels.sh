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
compile matmul_swiglu_f16_wmma dinov3_matmul_swiglu_f16_wmma wmma_swiglu_k384_n1536 \
  dinov3.matmul_swiglu_f16_wmma.k_size=384 dinov3.matmul_swiglu_f16_wmma.n_size=1536
compile residual_layernorm_f32 dinov3_residual_layernorm_f32 residual_layernorm \
  dinov3.residual_layernorm_f32.hidden_size=384 dinov3.residual_layernorm_f32.epsilon=1e-5
compile layernorm_f32_to_f16 dinov3_layernorm_f32_f16 layernorm_f16 \
  dinov3.layernorm_f32_f16.hidden_size=384 dinov3.layernorm_f32_f16.epsilon=1e-5
compile swiglu_f16 dinov3_swiglu_f32_f16 swiglu_f16 \
  dinov3.swiglu_f32_f16.width=1536 dinov3.swiglu_f32_f16.row_stride=3072
compile flash_attention_f16_wmma_cf16 dinov3_flash_attention_f16_wmma_cf16 flash_attention_cf16 \
  dinov3.flash_attention_f16_wmma_cf16.hidden_size=384 \
  dinov3.flash_attention_f16_wmma_cf16.qkv_stride=1152 \
  dinov3.flash_attention_f16_wmma_cf16.tokens_per_image=201 \
  dinov3.flash_attention_f16_wmma_cf16.scale=0.125
compile flash_attention_f16_wmma dinov3_flash_attention_f16_wmma flash_attention \
  dinov3.flash_attention_f16_wmma.hidden_size=384 \
  dinov3.flash_attention_f16_wmma.qkv_stride=1152 \
  dinov3.flash_attention_f16_wmma.tokens_per_image=201 \
  dinov3.flash_attention_f16_wmma.scale=0.125
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
# WMMA variants: f16 weights, f32 activations narrowed on the way into LDS,
# f32 accumulation.
for kn in 768:384 384:384 384:1152 384:3072 1536:384; do
  k="${kn%%:*}"; n="${kn##*:}"
  compile matmul_bias_f16_wmma dinov3_matmul_bias_f16_wmma "wmma_k${k}_n${n}" \
    "dinov3.matmul_bias_f16_wmma.k_size=$k" "dinov3.matmul_bias_f16_wmma.n_size=$n"
done
# f16-activation variants: A already f16 for the projections fed by an
# f16-producing kernel, and f16 out where the only consumer is another matmul.
for kn in 384:1152 384:384 1536:384; do
  k="${kn%%:*}"; n="${kn##*:}"
  compile matmul_bias_f16_wmma_af16 dinov3_matmul_bias_f16_wmma_af16 "wmma_af16_k${k}_n${n}" \
    "dinov3.matmul_bias_f16_wmma_af16.k_size=$k" "dinov3.matmul_bias_f16_wmma_af16.n_size=$n"
done
for kn in 384:3072 384:1152; do
  k="${kn%%:*}"; n="${kn##*:}"
  compile matmul_bias_f16_wmma_af16_cf16 dinov3_matmul_bias_f16_wmma_af16_cf16 "wmma_af16_cf16_k${k}_n${n}" \
    "dinov3.matmul_bias_f16_wmma_af16_cf16.k_size=$k" "dinov3.matmul_bias_f16_wmma_af16_cf16.n_size=$n"
done
compile rope_2d_f16 dinov3_rope_2d_f32_f16 rope_f16 \
  dinov3.rope_2d_f32_f16.hidden_size=384 dinov3.rope_2d_f32_f16.head_dim=64 \
  dinov3.rope_2d_f32_f16.prefix=5 dinov3.rope_2d_f32_f16.row_stride=1152 \
  dinov3.rope_2d_f32_f16.tokens_per_image=201
compile flash_attention_f16_wmma_af16_cf16 dinov3_flash_attention_f16_wmma_cf16_af16 flash_attention_af16 \
  dinov3.flash_attention_f16_wmma_cf16_af16.hidden_size=384 \
  dinov3.flash_attention_f16_wmma_cf16_af16.qkv_stride=1152 \
  dinov3.flash_attention_f16_wmma_cf16_af16.tokens_per_image=201 \
  dinov3.flash_attention_f16_wmma_cf16_af16.scale=0.125
# The hrx-demos online-softmax attention: one wave32 per (16 queries, head), K
# and V read from global as fragments. token_capacity only bounds the views;
# max_images bounds the image index so the fragment loads prove in range.
compile attention_online_f16_wmma_cf16 dinov3_attention_online_f16_wmma_cf16 attention_online_cf16 \
  dinov3.attention_online_f16_wmma_cf16.hidden_size=384 \
  dinov3.attention_online_f16_wmma_cf16.qkv_stride=1152 \
  dinov3.attention_online_f16_wmma_cf16.tokens_per_image=201 \
  dinov3.attention_online_f16_wmma_cf16.scale=0.125 \
  dinov3.attention_online_f16_wmma_cf16.max_images=64 \
  dinov3.attention_online_f16_wmma_cf16.token_capacity=262144
