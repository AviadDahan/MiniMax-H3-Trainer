# Scripts

Everything in `scripts/`. All Python entry points accept `--help`; all of them expect
`source scripts/env.sh` to have run first.

## Environment

| script | purpose |
|---|---|
| `install_env.sh` | uv venv at `$H3_ROOT/envs/h3`: torch ≥ 2.8, `transformers >= 5.15`, diffusers at the pinned commit |
| `env.sh` | redirects HF/torch/triton/pip/W&B caches onto `/data` and sets `PYTORCH_CUDA_ALLOC_CONF` before torch is imported |
| `download_model.sh` | weights, ~210GB (skips the duplicated `FL2VA/` and `Ref2VA/` trees) |

## Data

| script | purpose |
|---|---|
| `normalize_clips.py` | raw footage → 24.000 fps, bucket resolution, 32kHz stereo, and a manifest |
| `process_dataset.py` | latent cache: VAE pass (shardable with `torchrun`) + conditioner pass (loaded once) |
| `extract_pose.py` | footage → frame-aligned skeleton videos, for structural IC-LoRA |
| `generate_character_dataset.py` | build a character dataset with H3 itself |
| `split_manifest.py` | split a manifest into train/held-out shards |

## Training

| script | purpose |
|---|---|
| `train.py` | training; `--set key=value` overrides any config key, `--print-config` resolves without loading the model |
| `plot_metrics.py` | read a training curve honestly - sigma-controlled trend, per-bin means |

## Inference and export

| script | purpose |
|---|---|
| `generate.py` | inference; `--ab` for a same-seed adapter comparison. [Full reference](inference.md) |
| `evaluate_lora.py` | A/B several prompts on one pipeline load, with contact sheets |
| `export_lora.py` | adapter → ComfyUI fused-QKV layout, bit-exact |
