# h3-trainer

LoRA and IC-LoRA training for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — the
open-weight 33B omni-modal model that generates video **and** synchronized stereo audio from a single
packed sequence.

The ergonomics follow [LTX-2's `ltx-trainer`](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer):
declarative YAML configs, one flexible conditioning strategy that covers every training mode, offline
latent caching, W&B logging, validation sampling, and several acceleration backends. The numerics
follow MiniMax-H3, which differs from the usual flow-matching conventions in ways that are invisible
at runtime and corrupt weights silently.

---

## Why this exists

MiniMax released H3's weights "to support further development, including fine-tuning" but shipped no
trainer, and the diffusers integration is inference-only. The one public trainer,
[MiniMax-H3-FineTuning](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning), is a research script:
argparse only, one sample per step, no shuffling, no W&B, no validation sampling, no optimizer state
in checkpoints, and no reference conditioning at all. Its real contribution is `FIXES.md` — nine
model-specific corrections learned the hard way. Those are preserved here, with tests, plus:

| | reference trainer | h3-trainer |
|---|---|---|
| configuration | CLI flags | YAML, validated (`extra="forbid"`) |
| training modes | t2va / fl2va | t2va, i2v, fl2va, v2a, a2v, **IC-LoRA (ref2va)** |
| **reference conditioning** | not implemented | packed, masked out of the loss, trained |
| **prompt vision blocks** | text only | keyframe / reference blocks in the presentation |
| LoRA targets | `to_qkv` (matches nothing — see below) | verified against the model at startup |
| batching | 1 sample/step, no shuffle | bucketed batching, shuffling, grad accumulation |
| acceleration | DDP, ZeRO-3 | DDP, ZeRO-2/3, **NF4/INT8/FP8 quantized base** |
| resume | weights only | weights + optimizer + scheduler + step |
| logging | text file | text + JSONL + W&B, per-modality losses, media |
| validation | held-out loss | held-out loss **and** real generation |
| export | raw tensors | ComfyUI-compatible fused-QKV adapters |
| inference | — | `scripts/generate.py`, adapter A/B on one seed |

### Two upstream bugs this trainer avoids

**LoRA that never touches Q/K/V.** The reference trainer targets `["to_qkv", "to_out.0", "linear_1",
"linear_2"]`. Those are names from the *original MiniMax packaging*; the diffusers conversion splits
attention into `to_q`/`to_k`/`to_v` and names the feed-forward `ff.net.0.proj`/`ff.net.2`. PEFT only
raises when *no* target matches, so `to_qkv` is silently dropped while `linear_1`/`linear_2` match the
time embedder — producing a plausible parameter count from an adapter that never adapts attention.
`h3_trainer.lora.verify_target_modules` turns that into a startup error, and
`scripts/export_lora.py` re-fuses Q/K/V on export so adapters still load in ComfyUI.

**A stale transformers pin.** H3's conditioner needs Qwen3-VL's `mm_token_type_ids`; neither the
processor helper that builds them nor the model argument that consumes them exists in the pinned
4.57.3. `transformers >= 5.15` is required.

---

## Install

```bash
git clone <this repo> && cd h3-trainer
bash scripts/install_env.sh          # python 3.11 venv + torch 2.8 + pinned diffusers commit
source scripts/env.sh                # caches -> /data, allocator config, venv
bash scripts/download_model.sh       # ~210GB (skips the duplicated FL2VA/ and Ref2VA/ trees)
```

`scripts/env.sh` redirects every cache (HF hub, torch, triton, inductor, pip, uv, W&B) onto `/data`
and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch is imported — long packed
sequences fragment the allocator badly enough to OOM with tens of GB nominally free.

## Train

```bash
# 1. cache latents (VAE pass shards across GPUs; the 32B conditioner runs once)
python scripts/process_dataset.py dataset.json \
    --model-path /data/aviad/models/MiniMax-H3 \
    --resolution-bucket 704x704x107 --decode 2

# 2. train
torchrun --nproc_per_node 8 scripts/train.py configs/t2va_lora_low_vram.yaml   # quantized replicas
deepspeed --num_gpus 8 scripts/train.py configs/t2va_lora.yaml                 # bf16, ZeRO-3

# 3. generate, adapter off vs on, same seed
python scripts/generate.py --prompt "..." --lora /data/aviad/runs/.../checkpoint-0002000 --ab --out out.mp4
```

`--decode N` VAE round-trips N samples back to mp4. Run it once on any new dataset: it catches
normalization, channel-order and framing mistakes in seconds, which otherwise show up only as a
fine-tune that trains smoothly and generates garbage.

### Dataset format

`.json`, `.jsonl` or `.csv`, one row per clip:

```json
{"id": "clip001", "video": "clips/clip001.mp4",
 "caption": "TRIGGER, a woman in a red coat walks through rain. She says: \"not again\".",
 "reference_image": "refs/face.png", "reference_audio": "refs/voice.wav"}
```

Columns: `video` (aliases `target_video`, `media_path`), `caption`/`prompt`, `audio`, `first_frame`,
`last_frame`, `reference_image`, `reference_video`, `reference_audio`, `id`.

### Training modes

Every mode is the same strategy with different flags — see [docs/training-modes.md](docs/training-modes.md).

| mode | config | what changes |
|---|---|---|
| text → video+audio | [`t2va_lora.yaml`](configs/t2va_lora.yaml) | both modalities generated |
| image → video | [`i2v_lora.yaml`](configs/i2v_lora.yaml) | `first_frame` condition |
| video → audio | [`v2a_lora.yaml`](configs/v2a_lora.yaml) | `video.is_generated: false` |
| IC-LoRA | [`ref2va_ic_lora.yaml`](configs/ref2va_ic_lora.yaml) | `reference` condition + `variant: ref2va` |
| low VRAM | [`t2va_lora_low_vram.yaml`](configs/t2va_lora_low_vram.yaml) | NF4 base, DDP replicas |

## Hardware

H3 is 33B — 66GB in bf16 — and the conditioner is another 63GB. On 80GB cards, ZeRO-3 with bf16
weights is the straightforward choice. On 48GB cards (this repo was built on 8×A6000, no NVLink):

* **quantized base + DDP** (`nf4-bnb`, ~17GB/GPU) keeps full replicas and moves only LoRA gradients.
  On a PCIe-only machine this is much faster than ZeRO-3, which all-gathers every layer's parameters
  on every forward.
* **ZeRO-3** shards the bf16 weights 8 ways (~8.3GB/GPU) and is exact; use it when fidelity matters
  more than throughput.

Preprocessing and inference place the conditioner with `device_map="auto"` across whatever GPUs are
visible; `scripts/generate.py --placement offload` falls back to host RAM for a single card.

## The H3 quirks that matter

Encoded in [`flow_matching.py`](src/h3_trainer/flow_matching.py) and
[`packing.py`](src/h3_trainer/packing.py), each with a test in [`tests/`](tests/):

* **`t = 1 - sigma`.** t=1 is clean, t=0 is pure noise — the opposite of the SD3/Wan convention.
* **Data-ward velocity `v = x0 - eps`.** The scheduler reconstructs `x0 = x_t + (1-t)v`; regressing
  the usual `eps - x0` trains the model to walk away from the data.
* **Two noise schedules.** Video (shift 12.0) and audio (shift 3.0) are noised at *different* sigmas
  drawn from one shared uniform, mirroring how the two schedulers advance in lockstep at inference.
* **Zero-audio placeholders** are weighted to 0 rather than dropped, so gradients still flow through
  the audio head and DDP/ZeRO stay in sync.
* **Conditioning rows** are inputs, never targets: visual conditioning is noise-augmented to t=0.999,
  reference audio is passed clean at t=1.0, and both are masked out of the loss.
* **Conditioning latents** are encoded the way inference encodes them — posterior *sampled* under a
  generator seeded with 42, then rounded through float16 — while targets take the posterior mode.
* **Geometry:** 24.000 fps, `frames % 17 == 5`, height/width divisible by 32, 32kHz stereo audio on a
  40Hz latent grid. All validated before anything is encoded.
* **No CFG.** The checkpoints are guidance-distilled: one forward per step, no negative prompt.

## Documentation

* [docs/quick-start.md](docs/quick-start.md) — first run, end to end
* [docs/dataset-preparation.md](docs/dataset-preparation.md) — manifests, buckets, captions
* [docs/configuration-reference.md](docs/configuration-reference.md) — every config key
* [docs/training-modes.md](docs/training-modes.md) — how each mode maps onto the flexible strategy
* [docs/h3-quirks.md](docs/h3-quirks.md) — the model contract, in detail
* [docs/troubleshooting.md](docs/troubleshooting.md) — OOM, hangs, silent failures

## License

Apache-2.0. The MiniMax-H3 weights are governed by the MiniMax H3 Community License Agreement.
