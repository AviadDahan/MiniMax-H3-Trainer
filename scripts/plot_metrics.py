#!/usr/bin/env python
"""Read an H3 training curve honestly.

    python scripts/plot_metrics.py runs/character_av_lora

A raw H3 loss curve is close to unreadable: every step samples a different noise
level, and loss depends far more strongly on sigma than on how well the model is
doing. Two consecutive steps can differ 10x for that reason alone.

So this reports the loss **with sigma controlled**:

* a least-squares fit of ``loss ~ 1 + u + step``, whose step coefficient is the
  trend that is not explained by the noise level -- negative means learning;
* per-sigma-bin means for the first and last portions of the run, which is the
  same thing without assuming linearity;
* video and audio separately, because a healthy total loss hiding a flat audio
  term is the most common way an H3 fine-tune goes quietly wrong.

With matplotlib installed it also writes a PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl" if run_dir.is_dir() else run_dir
    if not path.exists():
        raise SystemExit(f"No metrics at {path}")
    rows = []
    with path.open() as handle:
        for line in handle:
            entry = json.loads(line)
            if "train/loss" in entry:
                rows.append(entry)
    if not rows:
        raise SystemExit(f"{path} contains no training rows")
    return rows


def report(rows: list[dict]) -> None:
    u = np.array([r["train/u"] for r in rows])
    step = np.array([r["step"] for r in rows], dtype=float)
    head, tail = step <= np.quantile(step, 0.25), step >= np.quantile(step, 0.75)

    print(f"{len(rows)} steps, {int(step.max())} total\n")
    for key, label in (("train/loss_video", "video"), ("train/loss_audio", "audio")):
        if key not in rows[0]:
            continue
        loss = np.array([r[key] for r in rows])
        design = np.stack([np.ones_like(u), u, step / max(step.max(), 1)], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, loss, rcond=None)
        verdict = "learning" if coefficients[2] < 0 else "NOT learning"
        print(f"{label}: sigma-controlled trend {coefficients[2]:+.4f} -> {verdict}")
        for low, high in BINS:
            mask = (u >= low) & (u < high)
            early, late = loss[mask & head], loss[mask & tail]
            if len(early) > 1 and len(late) > 1:
                change = 100 * (late.mean() / max(early.mean(), 1e-9) - 1)
                print(f"    u {low:.2f}-{high:.2f}:  {early.mean():.4f} -> {late.mean():.4f}  ({change:+.1f}%)")
        print()

    weights = np.array([r.get("train/audio_weight", 1.0) for r in rows])
    if weights.mean() < 1.0:
        print(f"NOTE: {100 * (1 - weights.mean()):.0f}% of steps had no real audio track "
              f"(audio loss weighted to 0) -- those clips are not training the audio branch.\n")
    skipped = rows[-1].get("train/skipped_long", 0)
    if skipped:
        print(f"NOTE: {int(skipped)} samples were skipped for exceeding max_seq_tokens.\n")


def plot(rows: list[dict], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping the plot)")
        return

    step = [r["step"] for r in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, label in (("train/loss_video", "video"), ("train/loss_audio", "audio")):
        if key in rows[0]:
            axes[0].plot(step, [r[key] for r in rows], linewidth=0.7, alpha=0.75, label=label)
    axes[0].set(xlabel="step", ylabel="loss", title="raw loss (dominated by sigma)")
    axes[0].legend()

    u = np.array([r["train/u"] for r in rows])
    video = np.array([r["train/loss_video"] for r in rows])
    steps = np.array(step, dtype=float)
    for low, high in BINS:
        mask = (u >= low) & (u < high)
        if mask.sum() < 5:
            continue
        window = max(3, int(mask.sum() // 8))
        smoothed = np.convolve(video[mask], np.ones(window) / window, mode="valid")
        axes[1].plot(steps[mask][window - 1 :], smoothed, label=f"u {low:.2f}-{high:.2f}")
    axes[1].set(xlabel="step", ylabel="video loss", title="video loss per sigma bin (the readable one)")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output, dpi=120)
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory or metrics.jsonl")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load(args.run)
    report(rows)
    plot(rows, args.out or (args.run if args.run.is_dir() else args.run.parent) / "metrics.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
