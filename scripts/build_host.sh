#!/usr/bin/env bash
# Build the command-line tools and the resident C ABI used by dinov3_loom.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

hipcc="${HIPCC:-/opt/rocm/bin/hipcc}"
mkdir -p build

"$hipcc" -O2 -Wall -Werror -fPIC -shared -DDINOV3_LIBRARY \
  -o build/libdinov3.so host/dinov3.cpp
"$hipcc" -O2 -Wall -Werror -o host/dinov3 host/dinov3.cpp
"$hipcc" -O2 -Wall -Werror -o host/loomrun host/loomrun.cpp

printf 'built %s\n' build/libdinov3.so host/dinov3 host/loomrun
