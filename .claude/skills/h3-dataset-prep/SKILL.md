---
name: h3-dataset-prep
description: Turn raw footage into a MiniMax-H3 latent cache. Use when building a manifest, choosing a resolution bucket, normalizing clips to H3's geometry, writing captions that describe sound as well as picture, encoding with process_dataset.py, or diagnosing a cache that trains smoothly and generates garbage.
---

# Preparing a dataset for MiniMax H3

H3 rejects most footage on geometry, and accepts the rest while silently learning whatever is wrong
with it. Get this stage right and training is uneventful.

## The constraints, before anything else

| constraint | value | what goes wrong otherwise |
|---|---|---|
| frame rate | exactly **24.000 fps** | a 25 fps clip is 4% slow motion, and systematic slow motion is one of the first things a LoRA learns |
| frames | `17n + 5` (22, 39, 56, ... 124, 141) | the video VAE cannot encode anything else |
| duration | **5-15s** to match what H3 generates | shorter trains fine but is out of distribution |
| width, height | divisible by 32 | VAE 16x, transformer patch 2x |
| audio | keep the real soundtrack | video and audio train jointly; a silent track teaches silence |

`scripts/normalize_clips.py` enforces all of it over a directory. Retime genuine slow motion rather
than shipping it.

## Manifest

`.json`, `.jsonl` or `.csv`, one row per clip. `video` and `caption` are required; `audio`,
`first_frame`, `last_frame`, `reference_image`, `reference_video`, `reference_audio` and `id` are
optional. Full alias table in [docs/dataset-preparation.md](../../../docs/dataset-preparation.md).

## Captions describe sound too

H3 conditions on audio, so a caption that only describes the picture is half a caption. For a
character adapter the tagged form keeps the split explicit:

```
[VISUAL] TRIGGER, a medium close-up of a woman at a kitchen table in morning light.
[SPEECH] TRIGGER speaks in a warm, slightly husky voice: "I keep meaning to write this down."
[SOUNDS] cutlery clinking faintly and a kettle in the background.
```

Put the trigger in the captions **or** pass `--lora-trigger`, never both.

For prompt *structure* beyond this - the `integrated_multimodal_description` / `overall_soundscape` /
`non_diegetic_music` fields H3 was trained on - MiniMax publishes the authoritative guide as a skill
in their own repo: https://github.com/MiniMax-AI/MiniMax-H3 under `.claude/skills/h3-prompt-writing`.
Prefer it over guessing; it is first-party and it is not reproduced here.

## Encoding

```bash
source scripts/env.sh
torchrun --nproc_per_node=8 scripts/process_dataset.py dataset.json \
    --model-path $H3_MODEL_PATH --resolution-bucket 704x704x124 --decode 2
```

Two passes: the VAEs (small, shards across GPUs) and the 32B conditioner (loaded once). Run them
apart with `--only media` / `--only text` when scheduling matters.

**Use `torchrun`.** Plain `python` makes the job rank 0 of 1 - one GPU busy, the rest idle, no
warning.

## Then actually look at the round-trips

`.precomputed/decoded_videos/*.mp4` are the clips back through both VAEs. Watch them **with sound**.
This is the single highest-value check in the pipeline: normalization, channel order, framing and
planar-vs-interleaved audio errors are obvious here and invisible in the loss.

## Reference media (IC-LoRA)

Add `--references` and a reference column. One decision matters: `--reference-canvas`.

* `native` (default) reproduces inference - the reference goes on **its own** 768-short-edge canvas
  whatever the target bucket. Faithful, portable, and expensive: a 124-frame reference is tens of
  thousands of rows by itself.
* `target` reuses the target bucket. Much cheaper, but conditions the model on a geometry inference
  never produces.

The choice is recorded in `index.json` because it is baked into the cached rows, and generation must
use the same one.

## Sanity numbers

Row count drives everything downstream and is quadratic in cost:

| bucket | rows | note |
|---|---|---|
| 512x512x124 | ~10k | ~19 s/step on 4 A6000s |
| 448x768x124 + matching reference | ~27k | ~82 s/step |

Work out the row count before committing to a bucket, not after.
