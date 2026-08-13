---
base_model: MiniMaxAI/MiniMax-H3
tags:
  - minimax-h3
  - lora
  - text-to-video
  - audio-video
license: other
license_name: minimax-h3-community-license
---

# {{ADAPTER_NAME}}

A LoRA for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), trained with
[h3-trainer](https://github.com/{{REPO}}).

{{ONE_PARAGRAPH_DESCRIPTION}}

**Trigger:** `{{TRIGGER}}` — {{HOW_TO_USE_THE_TRIGGER}}

## What it changes

{{WHAT_THE_ADAPTER_DOES}}

H3 generates video and audio jointly, so state plainly whether this adapter was trained on the audio
branch as well. An adapter trained video-only will not change what a scene sounds like.

| | |
|---|---|
| modalities trained | {{video / audio / both}} |
| variant | {{fl2va \| ref2va}} |
| conditioning | {{none \| first_frame \| reference}} |

## Usage

ComfyUI: drop `{{FILENAME}}.safetensors` into `models/loras/` and load it between the model loader and
the sampler. The file is in the community fused-QKV layout.

With h3-trainer:

```bash
python scripts/generate.py \
    --prompt "{{EXAMPLE_PROMPT}}" \
    --lora {{FILENAME}}.safetensors --lora-scale 1.0 \
    --resolution-bucket 704x704x124 --out out.mp4
```

H3 generates 5–15s only (frame counts 124, 141, 158 … 345) and is guidance-distilled, so there is no
negative prompt and no CFG scale.

## Training

| | |
|---|---|
| dataset | {{N}} clips, {{WHERE_FROM}} |
| resolution | {{WxHxF}} ({{DURATION}}s at 24 fps) |
| rank / alpha | {{RANK}} / {{ALPHA}} |
| target modules | {{TARGETS}} |
| learning rate | {{LR}}, {{SCHEDULER}} |
| steps | {{STEPS}} |
| precision | {{bf16}} |
| hardware | {{HARDWARE}} |

Captions follow:

```
{{CAPTION_EXAMPLE}}
```

## Evaluation

Same seed, adapter off versus on, on held-out prompts:

{{AB_RESULTS}}

## Limitations

{{LIMITATIONS}}

## License

The base weights are governed by the MiniMax H3 Community License Agreement, which this adapter
inherits. {{DATA_PROVENANCE}}
