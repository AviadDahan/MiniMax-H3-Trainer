# Quick start

From nothing to a trained adapter. Assumes Linux + CUDA and at least one 48GB GPU.

## 1. Environment

```bash
bash scripts/install_env.sh
source scripts/env.sh
```

`install_env.sh` builds a Python 3.11 venv at `$H3_ROOT/envs/h3` with torch ≥ 2.8, `transformers >= 5.15`
(4.57 lacks the `mm_token_type_ids` support H3's conditioner needs) and diffusers at the pinned commit
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` — the H3 classes are in no released wheel.

`env.sh` points every cache at `/data` and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
before torch is imported. Source it in every shell; the scripts also source it themselves.

You need `ffmpeg` and `ffprobe` on `PATH` (a static build in `$H3_ROOT/bin` is fine).

## 2. Weights

```bash
bash scripts/download_model.sh          # ~210GB into $H3_MODELS/MiniMax-H3
```

The HF repo ships the same weights twice: a diffusers-native flat tree at the root and the original
MiniMax packaging under `FL2VA/` and `Ref2VA/` (144GB each, custom modelling code). Only the flat tree
is usable here, so the script skips the duplicates.

Verify:

```bash
python scripts/generate.py --prompt "a red balloon drifts over a field, wind in the grass" \
    --resolution-bucket 512x512x39 --steps 8 --out /tmp/hello.mp4
```

## 3. A dataset

One row per clip, in `.json`, `.jsonl` or `.csv`:

```json
[{"id": "clip001", "video": "clips/clip001.mp4",
  "caption": "A woman in a red coat walks through rain, speaking: \"not again\"."}]
```

Clips should be 5–15s, at exactly 24.000 fps, with their real audio track kept — H3 trains video and
audio jointly, so a silent track teaches silence. `scripts/normalize_clips.py` does the fps,
resolution and audio normalization for a directory of raw footage:

```bash
python scripts/normalize_clips.py raw/ clips/ --resolution-bucket 704x704x124 --manifest dataset.json
```

## 4. Cache the latents

```bash
python scripts/process_dataset.py dataset.json \
    --model-path $H3_MODELS/MiniMax-H3 \
    --resolution-bucket 704x704x107 \
    --decode 2
```

Two passes run: the VAEs (shardable — add `torchrun --nproc_per_node 8`), then the 32B conditioner
once, spread across the visible GPUs.

**Look at the `--decode` output.** `.precomputed/decoded_videos/*.mp4` are your clips round-tripped
through both VAEs. If they look and sound right, the encoding recipe is right; if they do not, no
amount of training will fix it.

Bucket geometry is validated up front: width/height divisible by 32, frames of the form `17n + 5`
(22, 39, 56, 73, 90, 107, 124…), duration ≤ 15s.

## 5. Train

```bash
# 8x48GB, quantized base, full replicas (recommended without NVLink)
torchrun --nproc_per_node 8 scripts/train.py configs/t2va_lora_low_vram.yaml \
    --set data.preprocessed_data_root=$(pwd)/.precomputed output_dir=$H3_RUNS/my_lora

# 8x80GB, bf16 weights sharded
deepspeed --num_gpus 8 scripts/train.py configs/t2va_lora.yaml
```

`--set key.path=value` overrides any config key from the command line.

Watch `loss_video` and `loss_audio` separately. A total loss that looks healthy while the audio term
is flat means the audio branch is not learning — the most common way an H3 fine-tune goes quietly
wrong.

## 6. Check the result

```bash
python scripts/generate.py --prompt "<a held-out prompt>" \
    --lora $H3_RUNS/my_lora/checkpoint-0002000 --ab --out result.mp4
```

`--ab` generates the same seed twice, once with the adapter disabled, and writes `result.base.mp4`
alongside `result.mp4`. Comparing those two is the only honest way to see what the adapter did.

Each checkpoint also contains `lora_comfyui.safetensors`, converted to the community fused-QKV layout
that ComfyUI and published H3 adapters use.

## Where things go

```
$H3_ROOT/models/MiniMax-H3      weights
$H3_ROOT/datasets/<name>/       clips + .precomputed/
$H3_ROOT/runs/<name>/           checkpoints, validation media, train.log, metrics.jsonl
```
