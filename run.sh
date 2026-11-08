#!/usr/bin/env bash
# Usage: ~/voice-models/run.sh <command...>   e.g. run.sh python bench/tts_bench.py
# Sets caches + library paths so torch / ctranslate2 wheels work on NixOS (via nix-ld),
# without any system change. Library paths point only inside ~/voice-models.
set -euo pipefail
source "$HOME/voice-models/env.sh"
SP="$VM_ROOT/.venv/lib/python3.11/site-packages/nvidia"
LIBS="$SP/cublas/lib:$SP/cudnn/lib:$SP/cuda_runtime/lib:$SP/cuda_nvrtc/lib"
LIBS="$LIBS:/run/opengl-driver/lib"  # libcuda; libstdc++/zlib already come from nix-ld
export LD_LIBRARY_PATH="$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$VM_ROOT/.venv/bin:$PATH"
cd "$VM_ROOT"
exec "$@"
