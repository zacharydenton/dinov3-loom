#!/usr/bin/env bash
# Run every check.case in the given kernels on the real GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
status=0
for f in "${@:-kernels/*.loom}"; do
  for k in $f; do
    printf '=== %s\n' "$k"
    "$LOOM_FORMAT" --check "$k" >/dev/null || { echo "  format: FAIL"; status=1; }
    out=$("$IREE_TEST_LOOM" "$k" --device=amdgpu 2>&1) || { echo "  $out"; status=1; continue; }
    python3 - "$out" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
bad = d["failed_sample_count"]
print(f"  {d['sample_count'] - bad}/{d['sample_count']} samples pass"
      f"{' — FAILURES:' if bad else ''}")
for s in d["samples"]:
    if not s["passed"]:
        for fl in s["expectations"]["failures"]:
            print("   ", json.dumps(fl)[:400])
sys.exit(1 if bad else 0)
PY
  done
done
exit $status
