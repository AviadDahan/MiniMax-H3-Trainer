<div align="center">

# h3-trainer

**LoRA and IC-LoRA training for MiniMax-H3 — the open-weight 33B model that generates video *and* synchronized stereo audio.**

[![Base model](https://img.shields.io/badge/base%20model-MiniMax--H3-blue)](https://huggingface.co/MiniMaxAI/MiniMax-H3)
[![Design reference](https://img.shields.io/badge/design%20reference-ltx--trainer-green)](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer)
[![diffusers](https://img.shields.io/badge/diffusers-pinned%20abc5e9b-orange)](https://github.com/huggingface/diffusers)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

</div>

---

## 📚 Documentation

* [docs/quick-start.md](docs/quick-start.md) — first run, end to end
* [docs/dataset-preparation.md](docs/dataset-preparation.md) — manifests, buckets, captions
* [docs/configuration-reference.md](docs/configuration-reference.md) — every config key
* [docs/training-modes.md](docs/training-modes.md) — how each mode maps onto the strategy
* [docs/h3-quirks.md](docs/h3-quirks.md) — the model contract, in detail
* [docs/troubleshooting.md](docs/troubleshooting.md) — OOM, hangs, silent failures
* [artifacts/README.md](artifacts/README.md) — every run output and what it demonstrates

---

## ⚡ TL;DR

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

## 🎬 Demo

<!-- DEMO:START -->
A character AV LoRA trained with this repo: **36 clips, rank 16, 1200 steps, ~6 h on 4×A6000**. The
character is synthetic — invented from a text description, generated with H3 itself, so no real
person's likeness or voice is involved ([how](docs/dataset-preparation.md#generating-a-character-dataset)).

**Identity holds across scenes.** Anchor on the left; two generations from the finished adapter in
unrelated settings, same seed family, neither scene in the training set:

![anchor beside two generations](docs/demo/character_identity.png)

**The trigger does the work.** Same prompt, same seed — **base on the left, adapter on the right**.
`OHWXMIRA` turns a canal landscape into the character, speaking:

<p align="center">
  <img src="docs/demo/character_ab_canal.webp" alt="base versus adapter, canal prompt" width="720">
</p>

The same adapter in a scene the dataset never contained — the face carries over, the workshop does not
come from the training clips:

<p align="center">
  <img src="docs/demo/character_workshop.webp" alt="the same character in a workshop" width="480">
</p>

**And it stays contained.** No trigger, base left, adapter right. A character LoRA that quietly
rewrites every *other* prompt is a broken one:

<p align="center">
  <img src="docs/demo/character_ab_control.webp" alt="base versus adapter, untriggered control prompt" width="720">
</p>

**Turn the sound on.** The previews above are silent, and H3's whole point is that video and audio come
out of one pass. The mp4s carry the generated 32kHz stereo:
[canal A/B](docs/demo/character_ab_canal.mp4) ·
[workshop](docs/demo/character_workshop_lora.mp4) ·
[untriggered control A/B](docs/demo/character_ab_control.mp4)

Held-out video loss: 0.454 → 0.2675 (150) → 0.2415 (450) → 0.2430 (800) → **0.2356 (1200)**. Note the
run looked converged from 450 to 800 and then improved again — one reason `plot_metrics.py` reports a
sigma-controlled trend rather than a raw curve.

**What this does not show.** Voice identity is unverified: the clips carry speech and the audio branch
trains, but "does it sound like the same person" was never measured, only that speech is present and
in the expected band. The adapter also learns the anchor's wardrobe along with the face, which is what
36 clips of one outfit buys you. And nothing here is evidence about *real* footage — the whole dataset
came out of H3, which is a clean test of the trainer and an easier problem than the real thing.
<!-- DEMO:END -->

---

## 🔍 What it does

MiniMax released H3's weights "to support further development, including fine-tuning" but shipped no
trainer, and the diffusers integration is inference-only. This is the trainer.

| | |
|---|---|
| **training modes** | t2va, i2v, fl2va, v2a, a2v and IC-LoRA, all from one `flexible` strategy |
| **reference conditioning** | reference image/video/audio packed in-context, masked from the loss, trained |
| **prompt vision blocks** | keyframes and references appear in the conditioner's presentation, tagged as video |
| **configuration** | YAML, validated — unknown keys fail at load, not six hours in |
| **data** | offline two-pass latent cache; VAE round-trip verification built in |
| **batching** | layout-bucketed sampler, per-epoch shuffling, gradient accumulation |
| **acceleration** | model-parallel bf16, DDP, ZeRO-2/3, NF4/int8/fp8 quantization of the frozen base |
| **checkpoints** | trainable tensors only; exact resume of weights, optimizer, scheduler and step |
| **logging** | text + JSONL + W&B, with video and audio losses kept separate |
| **validation** | held-out loss on a seeded sigma grid, and real generation |
| **export** | ComfyUI-loadable adapters, Q/K/V re-fused, bit-exact |
| **inference** | multi-GPU sharded generation, same-seed A/B with the adapter on and off |

### Two things that will bite you

**LoRA targets that match nothing.** The diffusers conversion of H3 splits attention into
`to_q`/`to_k`/`to_v` and names the feed-forward `ff.net.0.proj`/`ff.net.2`. The *original* MiniMax
packaging uses a fused `to_qkv` and `linear_1`/`linear_2`, and those names appear in circulation. PEFT
only errors when **no** target matches, so a list mixing the two silently drops the ones that don't
exist — `to_qkv` matches nothing while `linear_1`/`linear_2` match the time embedder, giving a healthy
parameter count from an adapter that never adapts attention. Every target is verified against the
loaded model at startup here, and `export_lora.py` re-fuses Q/K/V on the way out so adapters still load
in ComfyUI.

**`transformers` must be ≥ 5.15.** H3's conditioner needs Qwen3-VL's `mm_token_type_ids`; neither the
processor helper that builds them nor the model argument that consumes them exists in 4.57.x.

---

## 🛠️ Installation

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

## 🏋️ Training

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

## 🎥 Inference

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

## 🧮 Method

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

## 💻 Hardware

H3 is 33B — 66GB in bf16 — and the conditioner is another 63GB. On 48GB cards the right answer isn't
the obvious one. Measured on 8×A6000 (no NVLink):

| strategy | weights/GPU | 48GB? | notes |
|---|---|---|---|
| `model_parallel` | ~8GB (8 GPUs), ~22GB (3) | **yes** | one process, blocks split across GPUs, full bf16. Floor is 2 GPUs — 66GB cannot fit on one. |
| `ddp` + `nf4-bnb` | ~18GB | yes | full replicas; see the 4-bit warning |
| `ddp` + `int8-quanto` | ~33GB | tight | little room for activations at real resolutions |
| `deepspeed_zero3` | ~8GB after partition | **no** | each rank holds all 66GB *before* partitioning. Fine on 80GB cards. |

Model-parallel runs blocks **sequentially**, so extra GPUs buy memory, not speed. The reason to use
fewer GPUs per run is *concurrency* — four 2-GPU runs instead of one 4-GPU run.

**Sequence length is the cost driver, and it is quadratic.** Attention has no mask over the packed
sequence, so everything attends to everything. Measured on 4 GPUs:

| what | rows | VRAM/GPU | step |
|---|---|---|---|
| 512×512×124, no reference | 9,970 | 23 GB | ~19 s |
| 448×768×124 **+ a reference video** | 27,364 | 31 GB | ~82 s |

A reference costs as many rows as the target it conditions, so an IC-LoRA sequence is roughly twice a
plain one before resolution is even considered — and 2.7× the rows came out at 4.3× the time. Plan
control-adapter runs around a bucket you can afford, not the largest one that fits in memory.

> **4-bit is for training, not for looking at.** NF4 fits the model on one card, but quantizing H3's
> AdaLN branches destroys generation — an NF4 sample here decoded to noise indistinguishable from
> decoding random latents. Train against it if you must; evaluate against bf16.

---

## 📊 Results

What has actually been measured on this hardware, not claims:

| check | result |
|---|---|
| VAE round-trip (video) | 22.8 dB PSNR on a hard synthetic pattern; colour, geometry and motion intact |
| VAE round-trip (audio) | dominant frequency preserved (439.5 Hz), 0.82 waveform correlation |
| Overfit test (numerics proof) | video loss −25% to −77% within matched sigma bins over 150 steps; sigma-controlled trend −0.673 |
| IC-LoRA packing | a reference image contributes 4,096 rows to an 8,741-row sequence; loss over the 448 target rows only |
| ComfyUI export | bit-exact against the PEFT weights (max abs difference 0.000e+00) |
| Character LoRA | 36 clips, rank 16, 1200 steps: identity holds across unseen scenes, control prompts unchanged; see [Demo](#-demo) |
| Unit tests | 46 passing |

---

## 🗺️ Roadmap

**Shipped**

- [x] LoRA training — t2va, i2v, fl2va, v2a, a2v from one `flexible` strategy
- [x] IC-LoRA — reference rows packed, masked from loss, trained
- [x] Model-parallel bf16 training on 48GB cards
- [x] Two-pass latent caching with VAE round-trip verification
- [x] Exact resume — weights + optimizer + scheduler + step
- [x] ComfyUI adapter export
- [x] Inference with multi-GPU sharding and same-seed A/B

**Next — more of the model reachable**

- [ ] **Structural control via IC-LoRA** — skeleton → video, then depth and edge by the same route.
      H3 has no native structural conditioning; in-context reference video is how you add it.
      Needs a conditioning-extraction step per control type, nothing new in the trainer.
- [ ] **Video extension, inpainting and outpainting** — LTX-2 has them, H3's packed layout can express
      them, and they are what people ask for after character adapters.
- [ ] **Config files for `a2v` and first+last-frame** — both expressible today, neither shipped.
- [ ] **Multi-resolution bucket training in a single run**, so a dataset isn't forced to one geometry.

**Next — scale**

- [ ] **ZeRO-3 via `deepspeed.zero.Init`** — sharded data-parallel currently cannot start on <80GB
      cards, because each rank materializes all 66GB before partitioning. The largest single
      throughput win available on 48GB hardware.
- [ ] **Real batching** — H3's batch axis replicates one shared layout, so batching needs identical
      geometry *and* caption length. Padding rows exist (tag `-1`, kept as a separate attention
      document) but do not solve it on their own: the per-row tags and position ids are shared across
      the batch, so items must agree on the layout, not merely on the length. Fixed-length captions
      plus a shared bucket geometry would unlock it.
- [ ] **Sequence parallelism** for 15s clips at 704p (~40k rows), which no single-GPU activation
      budget covers.
- [ ] **`torch.compile`** on the block stack, and caching the precomputable AdaLN branches (about a
      third of the parameters) during training.

**Later — surface**

- [ ] Hub publishing with generated model cards
- [ ] A ComfyUI node, so exported adapters have a first-class path to a UI
- [ ] Characterise the quantization/quality trade-off: is int8 good enough to *evaluate* on, or only
      to train against? (4-bit is documented as unusable for generation, but not measured.)

---

## ⚖️ Ethical considerations

This trains models that reproduce a specific person's **appearance and voice** together. That is
straightforwardly dual-use.

- Train on people who have **explicitly consented**, or on synthetic identities. The character example
  here is synthetic on purpose — invented from a text description, not a real person.
- **Label generated media** as synthetic when you publish it.
- Non-consensual likeness or voice cloning, impersonation and fraud are out of scope for this project
  and prohibited by the base model's licence.

---

## 📦 Scripts

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
| `extract_pose.py` | footage → frame-aligned skeleton videos, for structural IC-LoRA |
| `generate_character_dataset.py` | build a character dataset with H3 itself |
| `collect_artifacts.sh` | surface all run outputs under `artifacts/` |

---

## 🙏 Acknowledgements

- **MiniMax** for [H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and for releasing the weights.
- **Lightricks** for [LTX-2's `ltx-trainer`](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer),
  the design reference for the config schema, the flexible strategy and the overall shape of this tool.
- **[MiniMax-H3-FineTuning](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning)** for `FIXES.md` —
  H3's numeric conventions, the ZeRO-3 landmines, and the measured ~70k-token sequence ceiling,
  summarised in [docs/h3-quirks.md](docs/h3-quirks.md#measurements-from-prior-work).
- **HuggingFace** for the diffusers H3 integration this builds directly on.

## 📄 License

Apache-2.0. The MiniMax-H3 weights are governed by the MiniMax H3 Community License Agreement.
