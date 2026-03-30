#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LEGACY=("C_mandelbrot" "cfiles" "cuda_mandelbrot" "Mandelbrot_set")

TMP_BEFORE="${ROOT}/fractal_studio/runtime/legacy_before.sha"
TMP_AFTER="${ROOT}/fractal_studio/runtime/legacy_after.sha"

hash_tree() {
  local out="$1"
  : > "$out"
  for d in "${LEGACY[@]}"; do
    if [ -d "${ROOT}/${d}" ]; then
      find "${ROOT}/${d}" -type f -print0 | sort -z | xargs -0 sha256sum >> "$out"
    fi
  done
}

hash_tree "$TMP_BEFORE"
hash_tree "$TMP_AFTER"

if ! diff -q "$TMP_BEFORE" "$TMP_AFTER" >/dev/null; then
  echo "Legacy directories changed"
  exit 1
fi

echo "Legacy directories unchanged"
