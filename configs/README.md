# Configs

Every config is the same `flexible` strategy with different flags. See
[docs/training-modes.md](../docs/training-modes.md) for how each mode maps onto it, and
[docs/configuration-reference.md](../docs/configuration-reference.md) for every key.

| config | mode | acceleration | notes |
|---|---|---|---|
| [`t2va_lora.yaml`](./t2va_lora.yaml) | text → video+audio | `deepspeed_zero3` | the baseline; needs 80GB-class cards |
| [`character_av_lora.yaml`](./character_av_lora.yaml) | text → video+audio | `model_parallel` | full bf16 on 48GB cards; the recipe used for the character adapter |
| [`t2va_lora_low_vram.yaml`](./t2va_lora_low_vram.yaml) | text → video+audio | `ddp` + NF4 | full replicas per GPU; read the quantization warning inside |
| [`i2v_lora.yaml`](./i2v_lora.yaml) | image → video+audio | `deepspeed_zero3` | `first_frame` conditioning |
| [`v2a_lora.yaml`](./v2a_lora.yaml) | video → audio | `deepspeed_zero3` | video frozen, audio generated |
| [`ref2va_ic_lora.yaml`](./ref2va_ic_lora.yaml) | IC-LoRA | `deepspeed_zero3` | `reference` conditioning; requires `variant: ref2va` |

Pick the one closest to your task, change `model.model_path`, `data.preprocessed_data_root` and
`output_dir`, and override anything else from the command line:

```bash
python scripts/train.py configs/character_av_lora.yaml \
    --set optimization.steps=2000 lora.rank=32 wandb.enabled=true
```

**Choosing an acceleration strategy** is mostly a memory question — the table in the
[README](../README.md#hardware) has the measured numbers. On 48GB cards, `model_parallel` is the only
way to train against a full-precision base.
