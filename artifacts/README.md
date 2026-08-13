# Artifacts

Everything this repo produced on 8×A6000, in one place. Rebuild with:

```bash
bash scripts/collect_artifacts.sh
```

**Live work is symlinked, not copied**, so results appear here the moment they land:

| link | points at |
|---|---|
| `character-run/` | the live character run: `checkpoint-*`, `eval_*`, `train.log`, `metrics.jsonl`, `wandb/` |
| `character-dataset/` | anchor, 36 clips, manifest, `.precomputed` latent cache |
| `runs/` | every run directory |

Finished, static evidence (`inference/`, `verification/`, `smoke-runs/`) is copied, so it survives if
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
| `cat.mp4` | the same prompt and seed with an NF4 base. Coloured static — kept as the evidence that 4-bit quantization of H3's AdaLN branches destroys generation. |

## `character-run/` and `character-dataset/` (symlinks)

The synthetic character AV LoRA experiment. Paths below are relative to `character-dataset/`
(anchor, clips, manifest) or `character-run/` (checkpoints, evaluations, logs).

| path | contents |
|---|---|
| `character-dataset/anchor/` | `identity.png` (the identity reference), `voice.wav` (the voice reference), `anchor.mp4` (the generation both were taken from), `anchor.json` (character description, trigger, prompt) |
| `clips/` | 36 training clips, generated with Ref2VA conditioned on the anchor image + voice. 512×512×124 at 24fps with real audio. |
| `dataset.json` / `rejected.json` | the manifest after automatic QC (36 kept, 0 rejected) |
| `precomputed_index.json` | the latent cache index: row counts, geometry, `has_audio` per sample |
| `vae_decode_check/` | clips round-tripped through both VAEs — the check that the encoding recipe is right |
| `adapters/` | `stepNNNNNNN_peft.safetensors` (training layout) and `stepNNNNNNN_comfyui.safetensors` (community fused-QKV layout, loadable in ComfyUI) per checkpoint |
| `logs/` | `train.log`, `metrics.jsonl`, `metrics.png`, and `wandb/` (offline runs, syncable with `wandb sync`) |
| `evaluations/eval_*` | same-seed A/B per checkpoint: `sampleN_base.mp4`, `sampleN_lora.mp4`, `sampleN_compare.png`, `report.json` |

**Reading the A/B sheets:** top row is the base model, bottom row is the same seed with the adapter.
Samples 0 and 1 use the `OHWXMIRA` trigger; sample 2 (a retriever) is a control for prompt-adherence
collapse.

## `verification/`

| file | what it shows |
|---|---|
| `vae_roundtrip.png` | source vs VAE round-trip, 3 frames. Colour, geometry and motion preserved; only fine text degrades. |
| `decoded_noise.png` | random latents through the video VAE. Identical in character to `inference/cat.mp4`, which is how the NF4 failure was diagnosed as "the transformer produced noise" rather than "the VAE mis-decoded". |
| `gen_frames.png` | frames from the NF4 generation |
| `character_check.png` | anchor vs two Ref2VA clips — identity holding across scenes |
| `smoke_*.mp4` | synthetic clips round-tripped through both VAEs, used to verify audio interleaving and the 32kHz stereo path |

## `smoke-runs/`

`train.log` and `metrics.jsonl` for each correctness run. Read any of them with
`python scripts/plot_metrics.py artifacts/smoke-runs/<name>`.

| run | what it established |
|---|---|
| `overfit_test` | video loss drops 25–77% within matched sigma bins over 150 steps on 2 clips (sigma-controlled fit −0.673) — the check that catches an inverted timestep or velocity sign |
| `iclora_smoke` | IC-LoRA trains: a reference image contributes 4096 rows to an 8741-row packed sequence, with loss over the 448 target rows only |
| `character_feasibility` | full-resolution training fits (41.7M trainable of 33.2B, ~19s/step, 23GB/GPU) |
| `smoke_lowvram` | NF4 + DDP path runs |
| `smoke_zero3` | ZeRO-3 **fails** on 48GB cards: each rank materializes 66GB before partitioning |

## Not collected

Optimizer state (`optimizer.pt`, ~160MB per checkpoint) stays in `/data/aviad/runs/character_av_lora`.
It is resume state, not a deliverable. Base model weights live in `/data/aviad/models/MiniMax-H3`.
