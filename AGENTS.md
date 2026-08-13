# Working on MiniMax H3 Agentic Trainer

Notes for coding agents. The short version: this repo trains LoRA and IC-LoRA adapters on
MiniMax-H3, and almost everything that can go wrong here goes wrong **silently** — the loss curve
looks fine, training completes, and the adapter is subtly worthless. Bias toward verifying against
the pinned diffusers integration rather than reasoning from what flow matching usually does.

## Environment

```bash
source scripts/env.sh                       # required before anything
/data/aviad/envs/h3/bin/python -m pytest -q  # 48 tests, no GPU, ~3s
/data/aviad/envs/h3/bin/python -m ruff check .
```

`scripts/env.sh` is not optional. It sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
**before torch is imported** and redirects every cache (HF, torch, triton, pip, W&B) onto `/data` so
`$HOME` does not fill up. There is no `python` on `PATH` — use the venv interpreter above.

Hardware here is 8×A6000 (48GB, no NVLink). H3 is 66GB in bf16, so it never fits on one card:
`acceleration.strategy: model_parallel` splits the blocks across GPUs in one process. ZeRO-3 does
**not** start on 48GB cards — each rank materializes the whole model before partitioning.

## The invariants

These are load-bearing. Breaking one produces a run that trains smoothly and generates garbage.
Full detail in [docs/h3-quirks.md](docs/h3-quirks.md); tests pin them in
`tests/test_flow_matching.py`.

| invariant | where |
|---|---|
| `t = 1 - sigma`, and `t = 1` is **clean** data | `flow_matching.py` |
| target is the data-ward velocity `clean - noise`, never `noise - clean` | `flow_matching.py` |
| one uniform `u` mapped through **both** shift curves (video 12.0, audio 3.0) | `flow_matching.py` |
| silent clips weight the audio loss to **0** rather than dropping the term (dropping desynchronizes ranks) | `flow_matching.py` |
| conditioning rows are inputs, never targets, and are masked out of the loss | `packing.py` |
| model constants are **re-exported** from diffusers, never restated | `constants.py` |

That last one is a rule, not a preference: if you need a number the pipeline also uses, import it
through `constants.py`. A local `SEED = 42` that silently disagrees with upstream is exactly the
failure this repo exists to prevent — and it has already happened once.

## Layout

```
src/h3_trainer/
  flow_matching.py      sigma sampling, timestep conversion, velocity target, loss masking
  packing.py            wrappers over the diffusers packing helpers
  training_strategies/  flexible.py -- every training mode is this one strategy with flags
  preprocessing/        builder.py (the two passes), encoders.py (VAEs + conditioner), media.py
  trainer.py            the loop; model_parallel / DDP / ZeRO setup
configs/                one YAML per mode; all of them are loaded by tests/test_configs.py
scripts/                process_dataset.py, train.py, generate.py, evaluate_lora.py, extract_pose.py
docs/                   the model contract, config reference, troubleshooting
```

Adding a training mode should mean **writing a YAML file**, not a code path. If a change needs new
branching in `flexible.py`, that is a signal to reconsider.

## Traps that have actually cost time here

* **`process_dataset.py` shards by `RANK`/`WORLD_SIZE`.** Run with plain `python` and it silently
  becomes rank 0 of 1 — one GPU busy, the rest idle, four times slower, no warning. Use
  `torchrun --nproc_per_node=N`.
* **The cache-skip keys off `index.json`, not the latent files.** If a run dies before writing the
  index, the `.safetensors` on disk are invisible and everything re-encodes.
* **Reference videos have a canvas.** At inference H3 puts a reference on *its own* 768-short-edge
  canvas, not the target's, which changes its row count and rotary grid. `--reference-canvas` selects
  this and the choice is recorded in `index.json`. Training and generation must agree.
* **The tests do not touch media encoding.** They need no GPU, which is deliberate, but it means an
  encoder change can pass CI and still `NameError` on the first real clip. Exercise a real
  preprocessing run before trusting an `encoders.py` change.
* **Never gate a background job on `pgrep -f <pattern>`** — the check matches its own shell and
  returns immediately. Gate on the artefact the job produces.
* **Sequence length is quadratic and dominates everything.** 9,970 rows ≈ 19 s/step; 27,364 rows ≈
  82 s/step. Before proposing a bucket, work out the row count.

## Verifying a change

* `pytest` is necessary and not sufficient — see the encoding trap above.
* `process_dataset.py --decode 2` writes clips round-tripped through both VAEs. **Look at them.**
  This catches normalization, channel-order and framing errors that are otherwise invisible until
  generation.
* A raw H3 loss curve is unreadable — the sigma draw dominates the step-to-step variance. Use
  `scripts/plot_metrics.py`, which reports a sigma-controlled trend.
* For numerics changes, the overfit test is the real proof: a couple of clips, ~150 steps, loss must
  fall within matched sigma bins.

## Conventions

* Comments explain **why**, not what. Most comments here record a constraint of the model or a
  failure that was actually hit; keep that bar.
* Commit messages: a short imperative subject, then prose explaining the reasoning. State what was
  wrong and what the consequence was, not just what changed.
* Do not commit weights, datasets, or run outputs. `docs/demo/` is the one exception, for README
  media.
* Be accurate about what has been *measured* versus what is expected. The README distinguishes them
  deliberately, and claims like "matches inference bit-for-bit" have already turned out to be false
  in one place.
