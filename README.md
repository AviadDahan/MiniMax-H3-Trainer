<div align="center">

# h3-trainer

**LoRA and IC-LoRA training for MiniMax-H3 — the open-weight 33B model that generates video *and* synchronized stereo audio.**

[![Base model](https://img.shields.io/badge/base%20model-MiniMax--H3-blue)](https://huggingface.co/MiniMaxAI/MiniMax-H3)
[![Design reference](https://img.shields.io/badge/design%20reference-ltx--trainer-green)](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer)
[![diffusers](https://img.shields.io/badge/diffusers-pinned%20abc5e9b-orange)](https://github.com/huggingface/diffusers)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

</div>

---

## TL;DR

- Train **LoRA** adapters on H3 — character, style, voice, image-to-video, video-to-audio.
- Train **IC-LoRA** adapters, where a reference image/video/audio is packed *in-context* into the
  sequence. **Nothing else public trains this on H3.**
- Configs, ergonomics and CLI shaped after LTX-2's `ltx-trainer`; numerics shaped after H3, which
  differs from ordinary flow matching in ways that silently corrupt weights.
- Runs on **48GB cards** — the 66GB bf16 transformer is split across GPUs in one process, no
  quantization required.
- Every checkpoint exports a **ComfyUI-loadable** adapter, bit-exact.

```bash
bash scripts/install_env.sh && source scripts/env.sh    # environment, caches on /data
bash scripts/download_model.sh                          # weights (~210GB)

python scripts/process_dataset.py dataset.json --model-path $H3_MODEL_PATH \
    --resolution-bucket 704x704x124 --decode 2          # cache latents, eyeball the round-trip
python scripts/train.py configs/character_av_lora.yaml  # train
python scripts/generate.py --prompt "..." --lora runs/.../checkpoint-0000600 --ab --out out.mp4
```

---

## Demo

<!-- DEMO:START -->
*Filled in from the finished character run. Live results meanwhile: `artifacts/character-run/eval_*/`.*
<!-- DEMO:END -->

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
| training modes | t2va / fl2va | t2va, i2v, fl2va, v2a, a2v, **IC-LoRA** |
| **reference conditioning** | not implemented | packed, masked from loss, trained |
| **prompt vision blocks** | text only | keyframe / reference blocks in the presentation |
| LoRA targets | `to_qkv` — matches nothing (below) | verified against the model at startup |
| batching | 1 sample/step, no shuffle | bucketed batching, shuffling, grad accumulation |
| acceleration | DDP, ZeRO-3 | + **model-parallel**, + NF4/int8/fp8 quantization |
| resume | weights only | weights + optimizer + scheduler + step |
| logging | text file | text + JSONL + W&B, per-modality losses, media |
| validation | held-out loss | held-out loss **and** real generation |
| export | raw tensors | ComfyUI-compatible fused-QKV adapters |
| inference | — | `generate.py`, adapter A/B on one seed |

### Two upstream bugs this avoids

**A LoRA that never touches Q/K/V.** The reference trainer targets `["to_qkv", "to_out.0", "linear_1",
"linear_2"]` — names from the *original* MiniMax packaging. The diffusers conversion splits attention
into `to_q`/`to_k`/`to_v` and names the feed-forward `ff.net.0.proj`/`ff.net.2`. PEFT only errors when
*no* target matches, so `to_qkv` is silently dropped while `linear_1`/`linear_2` match the time
embedder — a healthy parameter count from an adapter that never adapts attention. Targets are now
verified against the loaded model at startup, and `export_lora.py` re-fuses Q/K/V on the way out so
adapters still load in ComfyUI.

**A stale transformers pin.** H3's conditioner needs Qwen3-VL's `mm_token_type_ids`; neither the
processor helper that builds them nor the model argument that consumes them exists in the pinned
4.57.3. **`transformers >= 5.15` is required.**

---

## Installation

**Prerequisites:** Linux, CUDA, Python 3.11+, and either ≥2 GPUs of 48GB or one 80GB card.
`ffmpeg`/`ffprobe` on `PATH`.

```bash
git clone <this repo> && cd h3-trainer
bash scripts/install_env.sh     # uv venv, torch 2.8, pinned diffusers, transformers 5.15
source scripts/env.sh           # redirects every cache onto /data; sets the CUDA allocator config
bash scripts/download_model.sh  # ~210GB (skips the duplicated FL2VA/ and Ref2VA/ trees)
```

`env.sh` matters more than it looks: it sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
*before* torch is imported, and points HF/torch/triton/pip/W&B caches at `/data` so `$HOME` stays small.

---

## Prompt format

H3 conditions on audio as well as video, so captions that ignore sound train a model that ignores
sound. For character work the tagged form makes the split explicit:

| section | what goes in it | example |
|---|---|---|
| `[VISUAL]` | subject, framing, setting, lighting, camera | `OHWXMIRA, a medium close-up of a woman at a kitchen table, morning light behind her.` |
| `[SPEECH]` | who speaks, how, and the literal line | `OHWXMIRA speaks in a warm, slightly husky mid-range voice: "Honestly? I would do the whole thing again."` |
| `[SOUNDS]` | ambience and diegetic sound | `cutlery clinking faintly and a kettle in the background.` |

Put the trigger token in the captions **or** pass `--lora-trigger`, never both — duplicating it
degrades prompt adherence.

---

## Training

### 1. Prepare a dataset

`.json`, `.jsonl` or `.csv`, one row per clip. Columns: `video`, `caption`, and optionally `audio`,
`first_frame`, `last_frame`, `reference_image`, `reference_video`, `reference_audio`, `id`.

Clips must be **exactly 24.000 fps**, `frames % 17 == 5`, dimensions divisible by 32, with their real
audio kept. `scripts/normalize_clips.py` does all of that for a directory of raw footage.

### 2. Cache the latents

```bash
python scripts/process_dataset.py dataset.json \
    --model-path $H3_MODEL_PATH --resolution-bucket 704x704x124 --decode 2
```

Two passes run: the VAEs (small, shardable with `torchrun`) and the 32B conditioner (loaded once,
spread across the visible GPUs). **Watch the `--decode` output** — those are your clips round-tripped
through both VAEs, and they catch normalization, channel-order and framing mistakes that otherwise
appear only as a fine-tune that trains smoothly and generates garbage.

### 3. Train

```bash
python scripts/train.py configs/character_av_lora.yaml          # 48GB cards, model-parallel bf16
deepspeed --num_gpus 8 scripts/train.py configs/t2va_lora.yaml  # 80GB cards, ZeRO-3 data-parallel
python scripts/train.py configs/t2va_lora.yaml --set optimization.steps=2000 lora.rank=32
```

| mode | config | what changes |
|---|---|---|
| text → video+audio | [`t2va_lora.yaml`](configs/t2va_lora.yaml) | both modalities generated |
| character AV (48GB) | [`character_av_lora.yaml`](configs/character_av_lora.yaml) | bf16 `model_parallel` |
| image → video | [`i2v_lora.yaml`](configs/i2v_lora.yaml) | `first_frame` condition |
| video → audio | [`v2a_lora.yaml`](configs/v2a_lora.yaml) | `video.is_generated: false` |
| **IC-LoRA** | [`ref2va_ic_lora.yaml`](configs/ref2va_ic_lora.yaml) | `reference` condition + `variant: ref2va` |
| low VRAM | [`t2va_lora_low_vram.yaml`](configs/t2va_lora_low_vram.yaml) | NF4 base, DDP replicas |

Watch `loss_video` and `loss_audio` **separately**. A healthy total hiding a flat audio term is the
most common way an H3 fine-tune goes quietly wrong. `python scripts/plot_metrics.py <run>` reads the
curve with sigma controlled, which is the only way it's readable at all.

---

## Inference

```bash
python scripts/generate.py --prompt "..." --out out.mp4                      # text → video+audio
python scripts/generate.py --prompt "..." --image first.png --out i2v.mp4    # image → video
python scripts/generate.py --prompt "..." --variant ref2va \
    --reference-image face.png --reference-audio voice.wav --out ref.mp4     # omni-reference
python scripts/generate.py --prompt "..." --lora ckpt/ --ab --out out.mp4    # A/B, same seed
```

| argument | default | notes |
|---|---|---|
| `--resolution-bucket` | `704x704x124` | `WxHxF`; **generation is limited to 5–15s** (124…345 frames) |
| `--steps` | 30 | guidance-distilled: one forward per step, no CFG, no negative prompt |
| `--placement` | `shard` | `shard` (bf16 across GPUs) · `quantize` · `bf16` · `offload` |
| `--quantization` | `nf4-bnb` | with `--placement quantize`; 4-bit degrades output badly |
| `--lora` / `--lora-scale` | — | checkpoint directory or `.safetensors` |
| `--ab` | off | also generate the same seed with the adapter disabled |
| `--seed` | 42 | |

---

## Method

H3 denoises **one packed sequence** holding text, conditioning and both target modalities at once:

```
[ text | conditioning blocks | target audio | target video ]
```

Training has to reproduce that layout exactly, which is where the interesting parts are.

1. **Reference conditioning (IC-LoRA).** Pre-encoded reference latents are concatenated ahead of the
   targets. Reference rows attend bidirectionally with everything else, are pinned at their
   conditioning timestep, and are **excluded from the loss** — they are inputs, not targets. A video
   reference's soundtrack rows are packed immediately before its video rows and share their rotary
   origin, exactly as generated audio and video do.
2. **Two noise schedules.** Video (shift 12.0) and audio (shift 3.0) are noised at *different* sigmas
   drawn from a single uniform, mirroring how H3's two schedulers advance in lockstep at inference.
   Independent draws train pairings the model never sees.
3. **H3's time convention.** `t = 1 − σ` (t=1 clean), and the model predicts a **data-ward** velocity
   `v = x₀ − ε`, reconstructed as `x₀ = x_t + (1−t)·v`. Both are inverted relative to the usual
   SD3/Wan convention, and both fail silently.
4. **Vision blocks in the prompt.** Keyframes and references don't only contribute latent rows — each
   prepends a label and a vision block to the conditioner's presentation, and those rows are tagged
   *video*, not text, which is what the transformer's AdaLN modulation keys off.

The LoRA targets `to_q`, `to_k`, `to_v`, `to_out.0` by default (add `ff.net.0.proj`, `ff.net.2` for
capacity) — the real module names in the diffusers checkpoint.

Full detail: [docs/h3-quirks.md](docs/h3-quirks.md).

---

## Hardware

H3 is 33B — 66GB in bf16 — and the conditioner is another 63GB. On 48GB cards the right answer isn't
the obvious one. Measured on 8×A6000 (no NVLink):

| strategy | weights/GPU | 48GB? | notes |
|---|---|---|---|
| `model_parallel` | ~8GB (8 GPUs), ~22GB (3) | **yes** | one process, blocks split across GPUs, full bf16. Floor is 2 GPUs — 66GB cannot fit on one. |
| `ddp` + `nf4-bnb` | ~18GB | yes | full replicas; see the 4-bit warning |
| `ddp` + `int8-quanto` | ~33GB | tight | little room for activations at real resolutions |
| `deepspeed_zero3` | ~8GB after partition | **no** | each rank holds all 66GB *before* partitioning. Fine on 80GB cards. |

Model-parallel runs blocks **sequentially**, so extra GPUs buy memory, not speed (~19 s/step either
way). The reason to use fewer GPUs per run is *concurrency* — four 2-GPU runs instead of one 4-GPU run.

> **4-bit is for training, not for looking at.** NF4 fits the model on one card, but quantizing H3's
> AdaLN branches destroys generation — an NF4 sample here decoded to noise indistinguishable from
> decoding random latents. Train against it if you must; evaluate against bf16.

---

## Results

What has actually been measured on this hardware, not claims:

| check | result |
|---|---|
| VAE round-trip (video) | 22.8 dB PSNR on a hard synthetic pattern; colour, geometry and motion intact |
| VAE round-trip (audio) | dominant frequency preserved (439.5 Hz), 0.82 waveform correlation |
| Overfit test (numerics proof) | video loss −25% to −77% within matched sigma bins over 150 steps; sigma-controlled trend −0.673 |
| IC-LoRA packing | a reference image contributes 4,096 rows to an 8,741-row sequence; loss over the 448 target rows only |
| ComfyUI export | bit-exact against the PEFT weights (max abs difference 0.000e+00) |
| Character LoRA | trigger flips output to the learned concept while a control prompt is unchanged; see [Demo](#demo) |
| Unit tests | 31 passing |

---

## Roadmap

**Shipped**

- [x] LoRA training — t2va, i2v, fl2va, v2a, a2v from one `flexible` strategy
- [x] IC-LoRA — reference rows packed, masked from loss, trained
- [x] Model-parallel bf16 training on 48GB cards
- [x] Two-pass latent caching with VAE round-trip verification
- [x] Exact resume — weights + optimizer + scheduler + step
- [x] ComfyUI adapter export
- [x] Inference with multi-GPU sharding and same-seed A/B

**Next**

- [ ] **Control adapters** — skeleton → video via IC-LoRA, then depth and edge by the same route.
      H3 has no native structural conditioning; IC-LoRA is how you add it.
- [ ] **Real-footage pipeline** — automatic AV captioning, scene splitting, slow-motion detection.
      Everything trained so far is H3-generated, which is self-distillation and easier than the real thing.
- [ ] **Identity and voice metrics** — face and speaker embedding distance, so "did it learn the
      character" is a number rather than a contact sheet
- [ ] **Audio-branch per-sigma validation** — video has it, audio only lands in the total

**Later**

- [ ] ZeRO-3 via `deepspeed.zero.Init`, so sharded data-parallel works on <80GB cards
- [ ] Real batching — needs fixed-length captions, and H3 exposes no attention mask over the sequence
- [ ] Sequence parallelism for 15s clips at 704p (~40k rows)
- [ ] Video extension, inpainting and outpainting modes
- [ ] Multi-resolution bucket training in a single run
- [ ] `torch.compile` on the block stack; caching the precomputable AdaLN branches
- [ ] Hub publishing with generated model cards, and a ComfyUI node

---

## Ethical considerations

This trains models that reproduce a specific person's **appearance and voice** together. That is
straightforwardly dual-use.

- Train on people who have **explicitly consented**, or on synthetic identities. The character example
  here is synthetic on purpose — invented from a text description, not a real person.
- **Label generated media** as synthetic when you publish it.
- Non-consensual likeness or voice cloning, impersonation and fraud are out of scope for this project
  and prohibited by the base model's licence.

---

## Documentation

* [docs/quick-start.md](docs/quick-start.md) — first run, end to end
* [docs/dataset-preparation.md](docs/dataset-preparation.md) — manifests, buckets, captions
* [docs/configuration-reference.md](docs/configuration-reference.md) — every config key
* [docs/training-modes.md](docs/training-modes.md) — how each mode maps onto the strategy
* [docs/h3-quirks.md](docs/h3-quirks.md) — the model contract, in detail
* [docs/troubleshooting.md](docs/troubleshooting.md) — OOM, hangs, silent failures
* [artifacts/README.md](artifacts/README.md) — every run output and what it demonstrates

## Scripts

| script | purpose |
|---|---|
| `install_env.sh` / `env.sh` / `download_model.sh` | environment, caches on `/data`, weights |
| `process_dataset.py` | latent cache: VAE pass (shardable) + conditioner pass (once) |
| `train.py` | training; `--set key=value` overrides any config key |
| `generate.py` | inference; `--ab` for a same-seed adapter comparison |
| `evaluate_lora.py` | A/B several prompts on one pipeline load, with contact sheets |
| `plot_metrics.py` | read a training curve honestly — sigma-controlled trend, per-bin means |
| `export_lora.py` | adapter → ComfyUI fused-QKV layout |
| `normalize_clips.py` | raw footage → 24.000 fps, bucket resolution, 32kHz stereo |
| `generate_character_dataset.py` | build a character dataset with H3 itself |
| `collect_artifacts.sh` | surface all run outputs under `artifacts/` |

---

## Acknowledgements

- **MiniMax** for [H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and for releasing the weights.
- **Lightricks** for [LTX-2's `ltx-trainer`](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer),
  the design reference for the config schema, the flexible strategy and the overall shape of this tool.
- **[MiniMax-H3-FineTuning](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning)** for `FIXES.md`,
  which documents H3's numeric conventions and the ZeRO-3 landmines — the hard part, done first.
- **HuggingFace** for the diffusers H3 integration this builds directly on.

## License

Apache-2.0. The MiniMax-H3 weights are governed by the MiniMax H3 Community License Agreement.
