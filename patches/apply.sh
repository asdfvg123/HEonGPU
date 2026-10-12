#!/usr/bin/env bash
# Apply the local submodule patches (idempotent).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
apply() {
    local sub="$1" patch="$2"
    if (cd "$ROOT/$sub" && git apply -R --check "$ROOT/patches/$patch") >/dev/null 2>&1; then
        echo "[ok] $sub: $patch already applied"
    else
        (cd "$ROOT/$sub" && git apply "$ROOT/patches/$patch")
        echo "[ok] $sub: $patch applied"
    fi
}
apply thirdparty/GPU-NTT gpu-ntt.patch
apply thirdparty/RNGonGPU rngongpu-gaussian-sign.patch
