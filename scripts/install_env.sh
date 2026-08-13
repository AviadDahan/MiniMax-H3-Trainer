#!/usr/bin/env bash
# Create the MiniMax H3 Trainer python environment at $H3_ROOT/envs/h3.
#
# MiniMax-H3's diffusers classes (MiniMaxH3Transformer3DModel, AutoencoderKLMiniMaxH3,
# the minimax_h3 modular pipeline + packing helpers) are not in any released diffusers
# wheel, so diffusers is installed from the pinned commit the H3 integration landed on.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

DIFFUSERS_COMMIT="abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
VENV="${H3_ROOT:-/data/aviad}/envs/h3"

uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# torch first, from the cu128 index, so nothing else drags in a CPU build.
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.8" torchvision torchaudio

# transformers >= 5.x is required, not the 4.57.3 the public H3 reference trainer
# pins: H3's conditioner needs Qwen3-VL's `mm_token_type_ids` (per-token modality
# ids driving its 3D rotary layout), and both the processor helper that builds
# them and the model argument that consumes them are absent from 4.57.
uv pip install \
    "transformers>=5.15,<6" \
    "accelerate>=1.10" \
    "deepspeed>=0.18" \
    "peft>=0.18" \
    "safetensors>=0.4" \
    "av>=12" \
    "huggingface_hub[cli,hf_transfer]>=0.26" \
    "pydantic>=2.7" \
    "pyyaml" \
    "wandb" \
    "optimum-quanto" \
    "bitsandbytes" \
    "numpy" \
    "pillow" \
    "tqdm" \
    "rich" \
    "pandas" \
    "matplotlib" \
    "pytest"

uv pip install --no-deps "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_COMMIT}"
# diffusers' own runtime deps, minus the pieces already pinned above.
uv pip install "importlib_metadata" "filelock" "regex" "requests" "Jinja2"

python - <<'PY'
import torch, diffusers, transformers
print(f"torch        {torch.__version__}  cuda={torch.version.cuda}  gpus={torch.cuda.device_count()}")
print(f"diffusers    {diffusers.__version__}")
print(f"transformers {transformers.__version__}")
from diffusers import MiniMaxH3Transformer3DModel, AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.modular_pipelines.minimax_h3 import packing, packing_ref2va
print("MiniMax-H3 classes + packing helpers import cleanly")
PY
