# Driving h3-trainer

How to *use* this repo. [AGENTS.md](AGENTS.md) is the companion for *changing* it — invariants,
layout, and the traps that have cost time here. Read that one before editing code.

## Which document answers which question

| you want to | read |
|---|---|
| run something end to end for the first time | [docs/quick-start.md](docs/quick-start.md) |
| know what a config key does | [docs/configuration-reference.md](docs/configuration-reference.md) |
| pick a training mode, or add one | [docs/training-modes.md](docs/training-modes.md) |
| turn footage into a dataset | [docs/dataset-preparation.md](docs/dataset-preparation.md) |
| understand *why* the numerics look unusual | [docs/h3-quirks.md](docs/h3-quirks.md) |
| diagnose an OOM, a hang, or output that is subtly wrong | [docs/troubleshooting.md](docs/troubleshooting.md) |
| know whether a capability has actually been run | the [untested list](README.md#untested) |
| see what a finished run looks like | [artifacts/README.md](artifacts/README.md) |

**`⚠️` in the docs means implemented but never run.** It is not decoration. If a task depends on a
`⚠️` option, say so up front rather than discovering it at step 0.

## The loop

Three stages, always in this order. Each one is separately checkable, which is the point.

```bash
source scripts/env.sh                    # required first: cache locations + allocator config

python scripts/process_dataset.py dataset.json \
    --model-path $H3_MODEL_PATH --resolution-bucket 704x704x124 --decode 2
python scripts/train.py configs/t2va_lora.yaml
python scripts/generate.py --prompt "..." --lora <run>/checkpoint-0000600 --ab --out out.mp4
```

Multi-GPU preprocessing shards by rank — use `torchrun --nproc_per_node=N`, or you silently get one
GPU doing all the work.

## Before you start a long run

A training run here is hours. These checks are minutes, and each one catches a class of failure that
is otherwise invisible until the end:

1. **`--print-config`.** Resolves the whole config, including `--set` overrides and `${VAR}` paths.
   If a key is wrong, it fails here rather than after the model loads.
2. **Look at `--decode` output.** Those are clips round-tripped through both VAEs. Normalization,
   channel-order and framing mistakes are obvious in the video and invisible in the loss.
3. **Check the row count against your budget.** Sequence cost is quadratic: ~10k rows is ~19 s/step,
   ~27k rows is ~82 s/step on 4 GPUs. Multiply by your step count *before* launching.
4. **Confirm the strategy fits.** On 48GB cards use `model_parallel`. `deepspeed_zero3` cannot start —
   each rank builds the whole 66GB model before partitioning.

## Reading a run

* **`loss_video` and `loss_audio` separately.** A healthy-looking total hiding a flat audio term is
  the most common way an H3 fine-tune is quietly broken.
* **Never eyeball the raw curve.** The sigma draw dominates step-to-step variance, so the curve looks
  like noise whatever is happening. `python scripts/plot_metrics.py <run>` reports a sigma-controlled
  trend, which is the only readable form.
* **Held-out loss over checkpoints is the real signal**, and it is not monotonic — one run here was
  flat from step 450 to 800 and then improved to 1200. Do not call convergence off a plateau.
* `metrics.jsonl` is one JSON object per step, for programmatic reading.

## Judging an adapter

Same-seed A/B is the only comparison worth making: `scripts/generate.py --ab`, or
`scripts/evaluate_lora.py` for several prompts on one pipeline load. Then ask three questions:

1. Does the trigger produce the learned concept?
2. Does it hold across scenes the dataset never contained — or has it only memorized the training
   backgrounds and wardrobe?
3. Does an **untriggered** prompt still behave? A character adapter that rewrites every other prompt
   is broken, and this is the check people skip.

## Conventions worth knowing before you are surprised

* H3 generates **5–15 s** only, at exactly 24.000 fps, with `frames % 17 == 5` and dimensions
  divisible by 32. The config validates all of this at load.
* There is **no CFG** — the checkpoints are guidance-distilled. No negative prompt, no guidance scale;
  those knobs do not exist here because they would do nothing.
* **4-bit destroys generation.** NF4 fits the model on one card and its output decodes to noise. Train
  against it if you must; evaluate in bf16.
* `batch_size > 1` needs samples with *identical* layouts including caption length. Use
  `gradient_accumulation_steps` instead.

## Reporting honestly

This repo distinguishes what was measured from what is expected, and that distinction is worth
preserving. If a run is inconclusive, say so and show the number. If a capability is on the untested
list, do not describe it as working because the code looks right — the two most costly bugs found here
both passed code review and the full test suite before failing on real data.
