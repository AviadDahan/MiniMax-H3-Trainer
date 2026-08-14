# Artifacts

**In a fresh clone this directory is empty apart from this file.** Everything below describes runs
made on the development machine (8×A6000); the media and weights are gitignored, and the symlinks are
local. This page is kept in the repo as the *index* - what was produced, and what each item
demonstrates - so the claims in the README can be traced to something specific even when you cannot
see the bytes. The demo media that ships with the repo is in [`../docs/demo/`](../docs/demo).

Rebuild on a machine that has the runs:

```bash
bash scripts/collect_artifacts.sh
```

**Live work is symlinked, not copied**, so results appear here the moment they land:

| link | points at |
|---|---|
| `character-run/` | the live character run: `checkpoint-*`, `eval_*`, `train.log`, `metrics.jsonl`, `wandb/` |
| `character-dataset/` | anchor, 36 clips, manifest, `.precomputed` latent cache |
| `pose-run/` | the pose IC-LoRA run: `checkpoint-*`, `train.log`, `metrics.jsonl`, `wandb/` |
| `pose-dataset/` | 46 skeleton/footage pairs from synthetic dancers, the split, and the `.precomputed448` cache |
| `pose-run-champ-withheld/` | the earlier pose run, kept only so the published figures can be traced. **Its weights are not for release** - see below |

There are deliberately no blanket `runs/` or `datasets/` links. `artifacts/` reads as curated output,
and pointing it at every directory on the machine surfaced training data that must not ship.

Finished results (`inference/`, `verification/`, `smoke-runs/`) are copied, so they survive if
the run directories are cleaned up. Media and weights are gitignored; this file is the committed index.

To watch the newest evaluation as it appears:

```bash
ls -t artifacts/character-run/eval_*/          # newest first
grep 'val/' artifacts/character-run/train.log  # held-out loss per checkpoint
```

## `inference/`

| file | what it shows |
|---|---|
| `cat_shard.mp4` | bf16 transformer sharded across 8 GPUs, 512×512×124, 20 steps. Coherent video with a synchronized meow (transient at ~0.9s, energy at ~1069 Hz). **This is what working inference looks like.** |
| `cat.mp4` | the same prompt and seed with an NF4 base. Coloured static - kept to show that 4-bit quantization of H3's AdaLN branches destroys generation. |

## `character-run/` and `character-dataset/` (symlinks)

The synthetic character AV LoRA experiment. Paths below are relative to `character-dataset/`
(anchor, clips, manifest) or `character-run/` (checkpoints, evaluations, logs).

| path | contents |
|---|---|
| `character-dataset/anchor/` | `identity.png` (the identity reference), `voice.wav` (the voice reference), `anchor.mp4` (the generation both were taken from), `anchor.json` (character description, trigger, prompt) |
| `clips/` | 36 training clips, generated with Ref2VA conditioned on the anchor image + voice. 512×512×124 at 24fps with real audio. |
| `dataset.json` / `rejected.json` | the manifest after automatic QC (36 kept, 0 rejected) |
| `precomputed_index.json` | the latent cache index: row counts, geometry, `has_audio` per sample |
| `vae_decode_check/` | clips round-tripped through both VAEs - the check that the encoding recipe is right |
| `adapters/` | `stepNNNNNNN_peft.safetensors` (training layout) and `stepNNNNNNN_comfyui.safetensors` (community fused-QKV layout, loadable in ComfyUI) per checkpoint |
| `logs/` | `train.log`, `metrics.jsonl`, `metrics.png`, and `wandb/` (offline runs, syncable with `wandb sync`) |
| `evaluations/eval_*` | same-seed A/B per checkpoint: `sampleN_base.mp4`, `sampleN_lora.mp4`, `sampleN_compare.png`, `report.json` |

**Reading the A/B sheets:** top row is the base model, bottom row is the same seed with the adapter.
Samples 0 and 1 use the `OHWXMIRA` trigger; sample 2 (a retriever) is a control for prompt-adherence
collapse.

## `pose-run/` and `pose-dataset/` (symlinks)

The skeleton-conditioned IC-LoRA - the first structural control adapter on H3, and the releasable
version of it.

**Every frame behind this run is generated.** Eight synthetic people from H3 itself, set dancing by
Wan-Dancer (Apache-2.0) at 480x832, cut to 448x768x124. Nobody real appears in the training data, and
the driving music never reaches the weights: clips are muxed silent, so the audio branch trains at
weight 0.

| path | contents |
|---|---|
| `pose-dataset/targets/` | 50 clips: 61 cut from 6 generated dances at a 1.7 s stride, 3 rejected for framing, 8 held out with `ref02` |
| `pose-dataset/poses/` | the matching MediaPipe skeleton renders, frame-aligned with their targets |
| `pose-dataset/rejected.json` | the 3 drops and why (limbs leaving frame mid-move) |
| `pose-dataset/heldout_poses/` `heldout_targets/` | **an entire person** (`ref02`), never cut into the manifest and never encoded. Following one of these skeletons cannot be memorization |
| `pose-dataset/dataset_train.json` / `dataset_heldout.json` | 46 / 4 split on top of that |
| `pose-dataset/.precomputed448/` | the latent cache at 448x768x124, with `reference_canvas` recorded in `index.json` |
| `pose-run/` | `train.log`, `metrics.jsonl`, `wandb/`, and `checkpoint-*` every 150 steps |

Read the curve with `python scripts/plot_metrics.py artifacts/pose-run`; the raw one is unreadable
because the sigma draw dominates. Sequence length is 27,360 rows: 12,432 target video, 12,432 skeleton
reference, 414 audio, and the caption.

### `pose-run-champ-withheld/`

The run that first demonstrated the mechanism, at 320x576. It is kept here because the figures in the
top-level README were produced from it and a claim should be traceable to the thing that produced it.

**Its weights are not published and should not be.** It was fitted to scraped dance footage whose
subjects never consented, one of them apparently a child. The pipeline and the findings survive that;
the weights do not. The dataset it used is not linked from `artifacts/` at all.

## `verification/`

| file | what it shows |
|---|---|
| `vae_roundtrip.png` | source vs VAE round-trip, 3 frames. Colour, geometry and motion preserved; only fine text degrades. |
| `decoded_noise.png` | random latents through the video VAE. Identical in character to `inference/cat.mp4`, which is how the NF4 failure was diagnosed as "the transformer produced noise" rather than "the VAE mis-decoded". |
| `gen_frames.png` | frames from the NF4 generation |
| `character_check.png` | anchor vs two Ref2VA clips - identity holding across scenes |
| `smoke_*.mp4` | synthetic clips round-tripped through both VAEs, used to verify audio interleaving and the 32kHz stereo path |

## `smoke-runs/`

`train.log` and `metrics.jsonl` for each correctness run. Read any of them with
`python scripts/plot_metrics.py artifacts/smoke-runs/<name>`.

| run | what it established |
|---|---|
| `overfit_test` | video loss drops 25-77% within matched sigma bins over 150 steps on 2 clips (sigma-controlled fit −0.673) - the check that catches an inverted timestep or velocity sign |
| `iclora_smoke` | IC-LoRA trains: a reference image contributes 4096 rows to an 8741-row packed sequence, with loss over the 448 target rows only |
| `character_feasibility` | full-resolution training fits (41.7M trainable of 33.2B, ~19s/step, 23GB/GPU) |
| `smoke_lowvram` | NF4 + DDP path runs |
| `smoke_zero3` | ZeRO-3 **fails** on 48GB cards: each rank materializes 66GB before partitioning |

## Not collected

Optimizer state (`optimizer.pt`, ~160MB per checkpoint) stays in `/data/aviad/runs/character_av_lora`.
It is resume state, not a deliverable. Base model weights live in `/data/aviad/models/MiniMax-H3`.
