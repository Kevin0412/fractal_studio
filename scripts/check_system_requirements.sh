#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA requirement failed: nvcc not found"
  exit 1
fi

if ! command -v g++ >/dev/null 2>&1; then
  echo "Compiler requirement failed: g++ not found"
  exit 1
fi

echo "System requirement check passed (basic)"
