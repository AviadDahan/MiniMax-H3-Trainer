# Configuration reference

> ⚠️ marks an option that is implemented and reachable but has **never been run**. See the
> [untested list](../README.md#untested). Everything unmarked has at least one real run behind it.

Every key of `H3TrainerConfig` ([`src/h3_trainer/config.py`](../src/h3_trainer/config.py)). Unknown
keys are rejected at load - a typo fails immediately rather than being ignored for six hours.

Any key can be overridden from the command line:

```bash
python scripts/train.py configs/t2va_lora.yaml --set optimization.steps=500 lora.rank=32
```

## `model`

| key | default | notes |
|---|---|---|
| `model_path` | - | local diffusers-layout MiniMax-H3 directory; must exist |
| `variant` | `fl2va` | `fl2va` → subfolder `transformer`; `ref2va` → `transformer_ref` |
| `training_mode` | `lora` | `lora`, `full` ⚠️ (needs `deepspeed_zero3`), `heads` ⚠️ (proj_out only - a pipeline smoke test) |
| `load_checkpoint` | `null` | checkpoint directory or file; a directory resolves to its latest |

## `lora`

| key | default | notes |
|---|---|---|
| `rank` | 16 | 16 is a strong default; higher overfits faster on small sets |
| `alpha` | 16 | effective scale is `alpha / rank` |
| `dropout` | 0.0 | must stay 0 with gradient checkpointing (recompute must be deterministic) |
| `target_modules` | `[to_q, to_k, to_v, to_out.0]` | real names in the diffusers checkpoint; add `ff.net.0.proj`, `ff.net.2` for capacity |
| `init_lora_weights` | `gaussian` | |

Names from the original MiniMax packaging (`to_qkv`, `qkv_proj`) or from LTX (`attn1.to_q`) are
rejected with an explanation. PEFT would silently ignore them, and an adapter that never touches
attention trains happily while doing very little.

Every target is verified against the loaded model at startup; a target matching nothing is fatal.

## `training_strategy`

```yaml
training_strategy:
  name: flexible
  video: { is_generated: true, latents_dir: latents, conditions: [] }
  audio: { is_generated: true, latents_dir: audio_latents, conditions: [] }
```

`is_generated: false` packs the modality clean and excludes it from the loss. At least one modality
must be generated.

Condition types (video only, except `reference`):

| type | keys | meaning |
|---|---|---|
| `first_frame` | `latents_dir`, `probability` | keyframe anchored at the start |
| `last_frame` | `latents_dir`, `probability` | keyframe anchored at the end |
| `reference` | `modality`, `latents_dir`, `probability` | in-context reference (IC-LoRA); requires `variant: ref2va` |

`probability` below 1.0 drops the condition on that fraction of steps, which keeps the unconditioned
path alive.

## `optimization`

| key | default | notes |
|---|---|---|
| `learning_rate` | 1e-4 | 1e-4 slow-cooked or 2e-4 fast are both reasonable for LoRA |
| `steps` | 2000 | scale with dataset size |
| `batch_size` | 1 | only samples with an identical packed layout can share a batch |
| `gradient_accumulation_steps` | 1 | the practical way to raise the effective batch |
| `max_grad_norm` | 1.0 | |
| `optimizer_type` | `adamw` | or `adamw8bit` ⚠️ (bitsandbytes) |
| `adam_betas` | `[0.9, 0.95]` | |
| `weight_decay` | 0.01 | |
| `scheduler_type` | `linear` | `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial` |
| `scheduler_params` | `{}` | e.g. `{num_cycles: 2}`, `{power: 2.0}` |
| `warmup_steps` | 0 | |
| `enable_gradient_checkpointing` | `true` | |
| `max_seq_tokens` | 70000 | pre-flight gate; oversized samples are skipped before the forward pass |

## `acceleration`

| key | default | notes |
|---|---|---|
| `strategy` | `ddp` | `ddp`, `model_parallel`, `deepspeed_zero2` ⚠️, `deepspeed_zero3` ⚠️ |
| `deepspeed_config` | `null` | explicit JSON; otherwise generated from this section |
| `mixed_precision_mode` | `bf16` | `no`, `fp16`, `bf16` |
| `quantization` | `none` | `nf4-bnb`, `int8-quanto` ⚠️, `fp8-quanto` ⚠️, `int8-bnb` ⚠️ - frozen base only |
| `offload_optimizer_during_validation` | `false` | |

**Choosing a strategy** is a memory question first:

* `ddp` - one full replica per GPU. Needs quantization on anything under 80GB, and then only LoRA
  gradients cross the bus. The fastest option when the model fits.
* `model_parallel` - a single process holding one copy of the bf16 weights, split by transformer
  block across the GPUs (~8GB each on 8 cards), with the modules that consume the packed layout's
  index vectors pinned to one device. Full precision on small cards; no data parallelism, so raise
  `gradient_accumulation_steps` instead. Launch with plain `python`, not `torchrun`/`deepspeed`.
* `deepspeed_zero3` - shards weights, gradients and optimizer state. Each rank still constructs the
  whole model *before* partitioning, so it needs cards that can hold 66GB.

Quantization and DeepSpeed ZeRO are mutually exclusive and the config says so: ZeRO partitions and
all-gathers raw parameter tensors, which quantized modules no longer expose.

Rough transformer footprints: bf16 66GB · int8 33GB · **nf4 ~18GB**. Note that 4-bit degrades this
model's *generation* badly (it quantizes the AdaLN branches); train against it if you must, but
evaluate against bf16.

## `data`

| key | default | notes |
|---|---|---|
| `preprocessed_data_root` | - | the `.precomputed` directory |
| `num_dataloader_workers` | 2 | |
| `val_split_every` | 20 | `md5(id) % N == 0` → validation; 0 disables |
| `shuffle` | `true` | |

## `validation`

| key | default | notes |
|---|---|---|
| `samples` | `[]` | each with `prompt`, optional `conditions`, `video_dims`, `seed` |
| `video_dims` | `[704, 704, 107]` | `(width, height, frames)`, validated |
| `frame_rate` | 24.0 | |
| `seed` | 42 | |
| `inference_steps` | 30 | |
| `interval` | 250 | steps between validations; `null` disables |
| `sample_media` | `true` | `false` keeps the cheap held-out loss only |
| `loss_sigmas` | `[0.3, 0.6, 0.9]` | fixed `u` grid, seeded noise |
| `max_loss_samples` | 8 | |
| `skip_initial_validation` | `false` | |

**No `negative_prompt`, no CFG scales, no `generate_audio`.** H3 is guidance-distilled - one forward
per step, no unconditional branch - and video and audio are always generated jointly.

## `checkpoints`

| key | default | notes |
|---|---|---|
| `interval` | 250 | |
| `keep_last_n` | -1 | -1 keeps everything |
| `precision` | `bfloat16` | or `float32` |
| `save_training_state` | `full` | `full` (optimizer + scheduler), `minimal`, `off` |

Only trainable tensors are saved. Each checkpoint also gets `lora_comfyui.safetensors` in the
community fused-QKV layout.

## `flow_matching`

| key | default | notes |
|---|---|---|
| `timestep_sampling_mode` | `uniform` | `uniform`, `logit_normal`, `shifted_logit_normal` |
| `timestep_sampling_params` | `{}` | e.g. `{mean: 0.0, std: 1.0, shift: 3.0}` |
| `video_shift` | 12.0 | **shipped value - changing it desynchronizes training from inference** |
| `audio_shift` | 3.0 | ditto |

## `hub` / `wandb` / global

| key | default |
|---|---|
| `hub.push_to_hub` ⚠️ / `hub.hub_model_id` | `false` / `null` |
| `wandb.enabled` / `project` / `entity` / `name` / `tags` / `log_validation_videos` | `false` / `minimax-h3-agentic-trainer` / `null` / `null` / `[]` / `true` |
| `seed` | 42 |
| `output_dir` | `outputs/h3_lora` |

With no W&B credentials on the machine, `scripts/env.sh` sets `WANDB_MODE=offline` so a detached
training job never blocks on an interactive login. Offline runs are complete on disk; upload them
later with `wandb sync $WANDB_DIR/wandb/offline-run-*`. Set `WANDB_API_KEY` (or run `wandb login`) to
stream live instead.

W&B logs `loss`, `loss_video`, `loss_audio`, `audio_weight`, `lr`, `grad_norm`, `sigma_video`,
`sigma_audio`, `u`, `seq_len`, `steps_per_sec`, `vram_gb`, `skipped_long`, plus validation losses per
sigma and validation media. The same metrics go to `train.log` and `metrics.jsonl` regardless - a
curve that only exists in the cloud is a curve you can lose.
