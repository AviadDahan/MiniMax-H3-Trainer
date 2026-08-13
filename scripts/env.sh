#!/usr/bin/env bash
# h3-trainer environment.
#
# Everything this project touches -- HuggingFace weights, torch/triton kernel caches,
# pip/uv wheels, wandb spool -- is redirected onto /data so $HOME never gets bloated.
#
# Source this before running anything:
#     source /data/aviad/github/h3-trainer/scripts/env.sh
#
# PYTORCH_CUDA_ALLOC_CONF must be set *before* torch is imported (FIX8 in the
# reference repo's FIXES.md); that is the main reason this is a shell file and
# not python-side configuration.

H3_ROOT="${H3_ROOT:-/data/aviad}"

# ---------------------------------------------------------------- HuggingFace
export HF_HOME="$H3_ROOT/hf-cache"
export HF_HUB_CACHE="$H3_ROOT/hf-cache/hub"
export HF_DATASETS_CACHE="$H3_ROOT/hf-cache/datasets"
export HF_ASSETS_CACHE="$H3_ROOT/hf-cache/assets"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---------------------------------------------------------------------- torch
export TORCH_HOME="$H3_ROOT/caches/torch"
export TORCHINDUCTOR_CACHE_DIR="$H3_ROOT/caches/inductor"
export TRITON_CACHE_DIR="$H3_ROOT/caches/triton"
export CUDA_CACHE_PATH="$H3_ROOT/caches/nv"

# ------------------------------------------------------------ package managers
export PIP_CACHE_DIR="$H3_ROOT/caches/pip"
export UV_CACHE_DIR="$H3_ROOT/caches/uv"
export UV_PYTHON_INSTALL_DIR="$H3_ROOT/caches/uv-python"
export XDG_CACHE_HOME="$H3_ROOT/caches/xdg"

# ---------------------------------------------------------------------- wandb
export WANDB_DIR="$H3_ROOT/caches/wandb"
export WANDB_CACHE_DIR="$H3_ROOT/caches/wandb/cache"
export WANDB_CONFIG_DIR="$H3_ROOT/caches/wandb/config"
export WANDB_ARTIFACT_DIR="$H3_ROOT/caches/wandb/artifacts"

# With no credentials, run W&B offline rather than letting it block on an
# interactive login (which, on a detached training job, means a run that hangs
# forever at startup). Offline runs are complete on disk and can be uploaded
# afterwards with `wandb sync $WANDB_DIR/wandb/offline-run-*`.
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

# ------------------------------------------------------------------- training
# Long packed sequences fragment the allocator badly; expandable segments is the
# difference between a run that fits and one that OOMs at step ~40.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Tokenizers forks in dataloader workers otherwise emit a warning storm.
export TOKENIZERS_PARALLELISM=false
# No NVLink on this box (PCIe only); P2P/IB disabled avoids NCCL hangs on
# multi-NUMA PCIe topologies. Override if your machine has NVLink.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# ----------------------------------------------------------------- convenience
export H3_MODELS="$H3_ROOT/models"
export H3_DATASETS="$H3_ROOT/datasets"
export H3_RUNS="$H3_ROOT/runs"
export H3_MODEL_PATH="${H3_MODEL_PATH:-$H3_ROOT/models/MiniMax-H3}"

export PATH="$H3_ROOT/bin:$PATH"

# Activate the project venv when it exists (skip if already inside one).
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$H3_ROOT/envs/h3/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$H3_ROOT/envs/h3/bin/activate"
fi
