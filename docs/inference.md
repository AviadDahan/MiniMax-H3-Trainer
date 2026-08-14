# Inference

`scripts/generate.py` loads the pipeline, optionally applies an adapter, and writes an `.mp4` with
picture and 32kHz stereo muxed together.

```bash
python scripts/generate.py --prompt "..." --out out.mp4                      # text → video+audio
python scripts/generate.py --prompt "..." --image first.png --out i2v.mp4    # image → video
python scripts/generate.py --prompt "..." --variant ref2va \
    --reference-image face.png --reference-audio voice.wav --out ref.mp4     # omni-reference
python scripts/generate.py --prompt "..." --lora ckpt/ --ab --out out.mp4    # A/B, same seed
```

## Arguments

| argument | default | notes |
|---|---|---|
| `--prompt` | required | |
| `--model-path` | `/data/aviad/models/MiniMax-H3` | |
| `--variant` | `fl2va` | `ref2va` is required for in-context references |
| `--out` | `generation.mp4` | |
| `--resolution-bucket` | `704x704x107` | `WxHxF`; W/H divisible by 32, `F = 17n + 5`. **Generation is limited to 5-15s** (124…345 frames) |
| `--steps` | 30 | guidance-distilled: one forward per step, no CFG, no negative prompt |
| `--seed` | 42 | |
| `--placement` | `shard` | `shard` (bf16 across GPUs) · `quantize` · `bf16` ⚠️ · `offload` ⚠️ |
| `--quantization` | `nf4-bnb` | with `--placement quantize`; 4-bit degrades output badly |
| `--device` | `cuda:0` | the card index-consuming modules are pinned to |
| `--lora` / `--lora-scale` | - / `1.0` | checkpoint directory or `.safetensors` |
| `--ab` | off | also generate the same seed with the adapter disabled, as `<out>.base.mp4` |
| `--image` / `--last-image` | - | first/last keyframe (`fl2va`) |
| `--reference-image` / `--reference-video` / `--reference-audio` | - | repeatable; `ref2va` only |
| `--reference-canvas` | `native` | must match what the adapter trained with (`process_dataset.py --reference-canvas`) |

⚠️ marks a code path that has never been run - see the [untested list](hardware.md#untested).

## Placement

`shard` spreads `transformer_blocks.*` across the visible GPUs in bf16 and pins every
index-consuming module to one device, which is what makes multi-GPU generation work at all - a plain
`device_map="auto"` lands the output projection on a different card from the packed layout's index
vectors and raises a device mismatch. It needs ~66GB of VRAM in total and is the only placement worth
evaluating an adapter on.

`quantize` fits the model on one card. See the [4-bit warning](hardware.md#4-bit-is-for-training-not-for-looking-at)
before you judge anything by its output.

## Judging an adapter

Same-seed A/B is the only comparison worth making. `--ab` does one prompt; `scripts/evaluate_lora.py`
does several on a single pipeline load and writes contact sheets. Then ask:

1. Does the trigger produce the learned concept?
2. Does it hold in scenes the dataset never contained, or has the adapter memorized the training
   backgrounds and wardrobe?
3. Does an **untriggered** prompt still behave? A character adapter that quietly rewrites every other
   prompt is broken, and this is the check people skip.

## ComfyUI

Every checkpoint contains `lora_comfyui.safetensors`, converted to the community fused-QKV layout -
Q/K/V re-fused from the diffusers split, bit-exact against the PEFT weights.
`scripts/export_lora.py` does the same conversion for an arbitrary adapter.
