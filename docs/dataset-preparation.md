# Dataset preparation

## The manifest

`.json` (a list), `.jsonl` (one object per line) or `.csv`. One row per clip:

```json
{"id": "clip001",
 "video": "clips/clip001.mp4",
 "caption": "TRIGGER, a woman in a red coat walks through rain. She says: \"not again\".",
 "audio": "audio/clip001.wav",
 "first_frame": "frames/clip001.png",
 "reference_image": "refs/face.png",
 "reference_video": "refs/motion.mp4",
 "reference_audio": "refs/voice.wav"}
```

Only `video` and `caption` are required. Column aliases: `video` ← `target_video`, `media_path`,
`file`; `caption` ← `prompt`, `text`; `reference_video` ← `ref_media_path`. `id` defaults to the
video's filename stem.

## Clip requirements

| | requirement | why |
|---|---|---|
| frame rate | exactly **24.000 fps** | H3's rotary clock counts latent frames; a 25 fps clip used as-is is 4% slow motion, and systematic slow motion is one of the first things a LoRA learns |
| duration | 5–15s for anything you also want to generate | H3 only generates in that window; shorter clips train but are out of distribution |
| frames | `17n + 5` — 22, 39, 56, 73, 90, 107, 124, 141, … | what the video VAE encodes |
| resolution | divisible by 32 | VAE 16x, transformer patch 2x |
| audio | keep the real track | video and audio train jointly; silent tracks teach silence |

Normalize with ffmpeg, and retime slow-motion footage rather than shipping it:

```bash
ffmpeg -i in.mp4 -vf "fps=24,scale=704:704:force_original_aspect_ratio=increase,crop=704:704" \
       -c:a aac -ar 32000 -ac 2 out.mp4

# genuine slow motion: retime video and audio together
ffmpeg -i slowmo.mp4 -filter_complex "[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]" -map "[v]" -map "[a]" out.mp4
```

Aim for 50–200 clips for a style or character adapter. Vary everything except the thing you are
teaching: if the subject is a character, the backgrounds, framing, lighting and action all have to
move, or the adapter learns the room.

## Captions

One flowing paragraph per clip describing subject, action, setting, lighting, camera **and sound**.
H3 conditions on audio too, so describing it is not optional decoration.

For a character adapter, the tagged form makes the split explicit:

```
[VISUAL] TRIGGER, a medium close-up of a woman seated at a kitchen table in morning light.
[SPEECH] TRIGGER speaks in a warm, slightly husky mid-range voice: "I keep meaning to write this down."
[SOUNDS] cutlery clinking faintly and a kettle in the background.
```

Pick **one** place for the trigger: either bake it into the captions or pass `--lora-trigger`, never
both — duplicating it degrades prompt adherence.

## Preprocessing

```bash
python scripts/process_dataset.py dataset.json \
    --model-path /data/aviad/models/MiniMax-H3 \
    --resolution-bucket 704x704x107 \
    --decode 2
```

Two passes run automatically:

1. **media** — both VAEs. Small models; shard it with `torchrun --nproc_per_node 8`.
2. **text** — the Qwen3-VL-32B conditioner, loaded once and spread across the visible GPUs. It is far
   too large to replicate per worker, so it runs on rank 0 for the whole dataset.

Run them separately with `--only media` / `--only text` when you want to schedule them differently.

Useful flags: `--keyframes first_frame,last_frame` (encode keyframe conditioning *and* put its vision
block in the prompt presentation), `--references` (IC-LoRA reference media), `--lora-trigger`,
`--skip-audio`, `--limit`, `--overwrite`, `--decode N`.

### Always look at `--decode`

`.precomputed/decoded_videos/*.mp4` are your clips round-tripped through both VAEs. Watch them with
sound. If they look and sound right, the encoding recipe is right. If they do not, nothing downstream
can fix it — and the failure modes (channel order, normalization, framing, planar-vs-interleaved
audio) all produce a training run that looks perfectly healthy.

## Output

```
<dataset>/.precomputed/
    index.json                    geometry, row counts, has_audio, caption per sample
    latents/<id>.safetensors      target video rows
    audio_latents/<id>...         target audio rows (absent = silent clip)
    conditions/<id>...            prompt_embeds + text_token_tags
    first_frame_latents/<id>...   keyframe rows
    reference_latents/<id>...     reference rows + latent geometry
    decoded_videos/<id>.mp4       --decode round-trips
```

Point `data.preprocessed_data_root` at the `.precomputed` directory.

Re-running skips samples that are already cached; `--overwrite` forces re-encoding. Caption
embeddings are model-specific — if you change the conditioner, re-run the text pass.

## Generating a character dataset

When you have no footage, `scripts/generate_character_dataset.py` builds one with H3 itself:

```bash
python scripts/generate_character_dataset.py --stage anchor --out-dir data/character
python scripts/generate_character_dataset.py --stage clips  --out-dir data/character --count 36
python scripts/generate_character_dataset.py --stage review --out-dir data/character
```

The anchor stage generates one clip and keeps its sharpest frame as the identity reference and its
soundtrack as the voice reference. The clips stage conditions every generation on both via Ref2VA, so
appearance and voice stay fixed while scene, framing and dialogue vary. The review stage drops clips
that are too dark, near-static or silent, and writes what it rejected to `rejected.json`.
