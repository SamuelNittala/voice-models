# Source this file: everything (caches, models, venv) stays inside ~/voice-models.
export VM_ROOT="$HOME/voice-models"
export PATH="$VM_ROOT/nix/uv/bin:$VM_ROOT/nix/ffmpeg-bin/bin:$PATH"
export UV_CACHE_DIR="$VM_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$VM_ROOT/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_PROJECT_ENVIRONMENT="$VM_ROOT/.venv"
export HF_HOME="$VM_ROOT/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME="$VM_ROOT/cache/torch"
export PIP_CACHE_DIR="$VM_ROOT/cache/pip"
export XDG_CACHE_HOME="$VM_ROOT/cache/xdg"
export TRITON_CACHE_DIR="$VM_ROOT/cache/triton"
export NUMBA_CACHE_DIR="$VM_ROOT/cache/numba"
export MPLCONFIGDIR="$VM_ROOT/cache/matplotlib"
export TMPDIR="$VM_ROOT/tmp"
export DO_NOT_TRACK=1
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"
export UV_PYTHON_BIN_DIR="$VM_ROOT/python/bin"
export UV_PYTHON_INSTALL_BIN=0
export UV_TOOL_DIR="$VM_ROOT/uv-tools"
export UV_TOOL_BIN_DIR="$VM_ROOT/uv-tools/bin"
export CUDA_CACHE_PATH="$VM_ROOT/cache/nv"
