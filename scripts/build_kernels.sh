#!/usr/bin/env bash
# Compile every kernel the model launches, at the configuration it launches it.
# There is one inference path, so this is the complete set: eleven HSACOs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
out=build/kernels
mkdir -p "$out"

compile() { # source-stem root-symbol output-stem config...
  local src="kernels/$1.loom" root="$2" stem="$3"; shift 3
  local args=()
  for c in "$@"; do args+=("--config=$c"); done
  "$LOOM_COMPILE" "$src" --backend=amdgpu-hal --target="$LOOM_TARGET" \
    --root="@$root" "${args[@]}" --output="$out/$stem.hsaco"
  printf '  %-26s %s\n' "$stem" "$(stat -c%s "$out/$stem.hsaco") bytes"
}

echo "compiling for $LOOM_TARGET"

# Patch embedding: the only matmul whose activations arrive f32, straight from
# the patchified image.
compile matmul_bias_f16_wmma dinov3_matmul_bias_f16_wmma wmma_k768_n384 \
  dinov3.matmul_bias_f16_wmma.k_size=768 dinov3.matmul_bias_f16_wmma.n_size=384

compile embed_scatter_f32 dinov3_embed_scatter_f32 embed_scatter \
  dinov3.embed_scatter_f32.hidden_size=384 dinov3.embed_scatter_f32.tokens_per_image=201 \
  dinov3.embed_scatter_f32.prefix=5

# LayerNorm, one row per wave. f16 out feeds a matmul; f32 out is the final norm.
compile layernorm_rowwave_f16 dinov3_layernorm_rowwave_f16 layernorm_rowwave \
  dinov3.layernorm_rowwave_f16.hidden_size=384 dinov3.layernorm_rowwave_f16.epsilon=1e-5
compile layernorm_rowwave_f32out dinov3_layernorm_rowwave_f32out layernorm_rowwave_f32out \
  dinov3.layernorm_rowwave_f32out.hidden_size=384 dinov3.layernorm_rowwave_f32out.epsilon=1e-5

# QKV projection with RoPE folded into the epilogue.
compile matmul_qkv_rope_f16_wmma dinov3_matmul_qkv_rope_f16_wmma qkv_rope_k384_n1152 \
  dinov3.matmul_qkv_rope_f16_wmma.k_size=384 dinov3.matmul_qkv_rope_f16_wmma.n_size=1152 \
  dinov3.matmul_qkv_rope_f16_wmma.head_dim=64 dinov3.matmul_qkv_rope_f16_wmma.prefix=5 \
  dinov3.matmul_qkv_rope_f16_wmma.tokens_per_image=201 \
  dinov3.matmul_qkv_rope_f16_wmma.rope_channels=768

# Online-softmax attention: one wave32 per (16 query rows, head).
compile attention_online_f16_wmma_cf16 dinov3_attention_online_f16_wmma_cf16 attention_online_cf16 \
  dinov3.attention_online_f16_wmma_cf16.hidden_size=384 \
  dinov3.attention_online_f16_wmma_cf16.qkv_stride=1152 \
  dinov3.attention_online_f16_wmma_cf16.tokens_per_image=201 \
  dinov3.attention_online_f16_wmma_cf16.scale=0.125 \
  dinov3.attention_online_f16_wmma_cf16.max_images=64 \
  dinov3.attention_online_f16_wmma_cf16.token_capacity=262144

# o and down projections, with the residual add and LayerScale in the epilogue.
for kn in 384:384 1536:384; do
  k="${kn%%:*}"; n="${kn##*:}"
  compile matmul_resid_f16_wmma dinov3_matmul_resid_f16_wmma "resid_k${k}_n${n}" \
    "dinov3.matmul_resid_f16_wmma.k_size=$k" "dinov3.matmul_resid_f16_wmma.n_size=$n"
done

# gate/up with SwiGLU as its epilogue: one workgroup accumulates both halves.
compile matmul_swiglu_f16_wmma dinov3_matmul_swiglu_f16_wmma wmma_swiglu_k384_n1536 \
  dinov3.matmul_swiglu_f16_wmma.k_size=384 dinov3.matmul_swiglu_f16_wmma.n_size=1536

# Split-K down projection plus its reduction, for batch 1.
compile matmul_splitk_f16_wmma dinov3_matmul_splitk_f16_wmma splitk_k1536_n384 \
  dinov3.matmul_splitk_f16_wmma.k_size=1536 dinov3.matmul_splitk_f16_wmma.n_size=384 \
  dinov3.matmul_splitk_f16_wmma.splits=4
compile splitk_reduce_f16 dinov3_splitk_reduce_f16 splitk_reduce_n384 \
  dinov3.splitk_reduce_f16.n_size=384 dinov3.splitk_reduce_f16.splits=4
