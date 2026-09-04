#!/usr/bin/env bash
# The one test command. Builds the kernels, runs every unit test against a
# float64 NumPy reference on the real GPU, checks the runner's error paths, then
# validates the whole model against transformers.
#
#   scripts/test.sh          everything
#   scripts/test.sh --quick  skip the transformers comparison (no torch needed)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

quick=0
[ "${1:-}" = "--quick" ] && quick=1

status=0
step() {
  local name="$1"; shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then
    printf '  ok\n'
  else
    printf '  FAILED: %s\n' "$name"
    status=1
  fi
}

# Anything the tests compile has to build first.
step "loom sources are canonically formatted" bash -c '
  "$LOOM_FORMAT" --check kernels/*.loom experiments/*.loom experiments/superseded/*.loom'
step "build kernels" ./scripts/build_kernels.sh
step "generated kernels match their generator" bash -c '
  before=$(sha256sum kernels/attention_online_f16_wmma_cf16.loom)
  python3 tools/gen_f16_variants.py >/dev/null
  [ "$before" = "$(sha256sum kernels/attention_online_f16_wmma_cf16.loom)" ]'

[ -x host/loomrun ] || /opt/rocm/bin/hipcc -O2 -o host/loomrun host/loomrun.cpp
[ -x host/dinov3 ] || /opt/rocm/bin/hipcc -O2 -o host/dinov3 host/dinov3.cpp

step "layernorm"          python3 tools/test_layernorm.py
step "matmul (WMMA)"      python3 tools/test_matmul_wmma.py
step "fused-epilogue matmuls" python3 tools/test_fused_matmuls.py
step "online attention"   python3 tools/test_attention_online.py

# Error paths: each of these must exit non-zero with a diagnostic rather than
# reading uninitialised memory or overflowing a buffer.
step "runner rejects bad input" bash -c '
  set -e
  ! ./host/dinov3 --batch                       2>&1 | grep -q "needs a value"    && exit 1
  ! ./host/dinov3 --input build/patchified.bin --batch 0   2>&1 | grep -q "must be 1"       && exit 1
  ! ./host/dinov3 --input build/patchified.bin --batch 999 2>&1 | grep -q "must be 1"       && exit 1
  ! ./host/dinov3                               2>&1 | grep -q "input is required" && exit 1
  printf short > /tmp/dinov3-short.bin
  ! ./host/dinov3 --input /tmp/dinov3-short.bin 2>&1 | grep -q "not a multiple"     && exit 1
  # a whole number of images, but not the number asked for
  cat build/patchified.bin build/patchified.bin > /tmp/dinov3-two.bin
  ! ./host/dinov3 --input /tmp/dinov3-two.bin --batch 3 2>&1 | grep -q "holds 2 images" && exit 1
  ! ./host/loomrun --hsaco x --kernel k --out "$(python3 -c "print(\"A\"*900)"):16" 2>&1 \
      | grep -q "path too long"                                                    && exit 1
  args=(--hsaco x --kernel k); for i in $(seq 40); do args+=(--i32 1); done
  ! ./host/loomrun "${args[@]}" 2>&1 | grep -q "too many args"                     && exit 1
  rm -f /tmp/dinov3-short.bin /tmp/dinov3-two.bin'

if [ "$quick" = 0 ]; then
  # torch links its own ROCm; env.sh points LD_LIBRARY_PATH at the HRX
  # runtime, which makes `import torch` segfault. Only the Loom toolchain
  # needs that path, and validate.py shells out to host/dinov3 for the GPU
  # work, so drop it for this step.
  step "end-to-end vs transformers" env -u LD_LIBRARY_PATH python3 tools/validate.py
  step "batched python API vs transformers" env -u LD_LIBRARY_PATH python3 tools/test_batch.py
fi

printf '\n'
[ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
