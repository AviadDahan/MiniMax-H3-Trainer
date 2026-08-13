---
name: h3-lora-run
description: Run and judge a MiniMax-H3 LoRA or IC-LoRA training job. Use when picking a config, running pre-flight checks before committing hours of GPU time, choosing an acceleration strategy for the available cards, reading a loss curve whose raw form is unreadable, or deciding whether a finished adapter actually works.
---

# Running a training job on MiniMax H3

A run here is hours. Everything below is minutes, and each item catches a failure that is otherwise
invisible until the end.

## Pre-flight

1. **`source scripts/env.sh`.** Sets the CUDA allocator config *before* torch imports and points every
   cache at `$H3_ROOT`. Skipping it produces `model_path refers to $H3_MODEL_PATH, which is not set`.
2. **`--print-config`.** Resolves the config including `--set` overrides and `${VAR}` paths. A typo'd
   key fails here rather than after the model loads; unknown keys are rejected by design.
3. **Row count against the clock.** Cost is quadratic in sequence length: ~10k rows is ~19 s/step,
   ~27k rows ~82 s/step on 4 A6000s. Multiply by the step count *before* launching.
4. **Strategy fits the cards.** On 48GB use `model_parallel` - one process, blocks split across GPUs,
   full bf16. `deepspeed_zero3` cannot start: every rank materializes all 66GB before partitioning.
   Model-parallel is sequential, so extra GPUs buy memory, not speed.

## Launching

```bash
source scripts/env.sh
python scripts/train.py configs/character_av_lora.yaml
python scripts/train.py configs/t2va_lora.yaml --set optimization.steps=2000 lora.rank=32
```

Anything marked ⚠️ in the docs is implemented but has never been run. If a task depends on one, say so
before starting rather than discovering it at step 0.

Do **not** gate automation on `pgrep -f <pattern>` - the check matches its own shell and returns
immediately. Gate on the artefact the job writes.

## Reading the run

* **`loss_video` and `loss_audio` separately.** A healthy total hiding a flat audio term is the most
  common way an H3 fine-tune is quietly broken. On silent footage the audio term is *weighted to zero*
  by design, so `loss == loss_video` exactly is correct there, not a bug.
* **Never judge the raw curve.** The sigma draw dominates step-to-step variance, so it looks like
  noise whatever is happening. `python scripts/plot_metrics.py <run>` reports a sigma-controlled
  trend, which is the only readable form.
* **Held-out loss over checkpoints is not monotonic.** One run here was flat from step 450 to 800 and
  then improved through 1200. Do not call convergence off a plateau.
* `metrics.jsonl` is one JSON object per step for programmatic reading; the console logs every Nth.

## Judging the adapter

Same-seed A/B is the only comparison worth making - `scripts/generate.py --ab`, or
`scripts/evaluate_lora.py` for several prompts on one pipeline load. Ask three questions:

1. Does the trigger produce the learned concept?
2. Does it hold in scenes the dataset never contained, or has it memorized backgrounds and wardrobe?
3. Does an **untriggered** prompt still behave? An adapter that rewrites every other prompt is broken.
   This is the check people skip.

Evaluate in **bf16**. NF4 fits the model on one card and its output decodes to noise, so 4-bit is for
training against, never for looking at.

## When it goes wrong

[docs/troubleshooting.md](../../../docs/troubleshooting.md) covers OOM, hangs and silent failures.
The distributed landmines - activation checkpointing under ZeRO-3, skip decisions that must be
all-reduced, checkpointing before validating - are in
[docs/h3-quirks.md](../../../docs/h3-quirks.md).

If a numerics change is involved, the overfit test is the real proof: a couple of clips, ~150 steps,
loss must fall within matched sigma bins. `pytest` passes without touching media encoding at all, so
it cannot catch an encoder that fails on a real clip.
