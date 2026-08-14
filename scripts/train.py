#!/usr/bin/env python
"""Train a MiniMax-H3 LoRA / IC-LoRA from a YAML config.

    # single GPU (quantized base recommended on <80GB cards)
    python scripts/train.py configs/t2va_lora_low_vram.yaml

    # 8 GPUs, full replicas (quantized base), no cross-GPU parameter traffic
    torchrun --nproc_per_node 8 scripts/train.py configs/character_av_lora.yaml

    # 8 GPUs, bf16 weights sharded with ZeRO-3
    deepspeed --num_gpus 8 scripts/train.py configs/t2va_lora.yaml

Command-line overrides use dotted paths, e.g.:

    python scripts/train.py configs/t2va_lora.yaml \
        --set optimization.steps=50 output_dir=/data/aviad/runs/smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from h3_trainer import logger  # noqa: E402
from h3_trainer.config import H3TrainerConfig  # noqa: E402
from h3_trainer.trainer import H3Trainer  # noqa: E402


def apply_overrides(payload: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` overrides, parsing values as YAML scalars."""
    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"--set expects key=value, got {override!r}")
        path, raw = override.split("=", 1)
        cursor = payload
        parts = path.split(".")
        for key in parts[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[parts[-1]] = yaml.safe_load(raw)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    parser.add_argument("--print-config", action="store_true", help="validate, print, and exit")
    parser.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)  # deepspeed launcher
    args = parser.parse_args()

    with args.config.open() as handle:
        payload = yaml.safe_load(handle)
    payload = apply_overrides(payload, args.set)
    config = H3TrainerConfig.model_validate(payload)

    if args.print_config:
        print(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, width=100))
        return 0

    if int(os.environ.get("RANK", "0")) == 0:
        logger.info(
            "%s -> %s | %s / %s | %d steps @ lr %g",
            args.config,
            config.output_dir,
            config.model.variant,
            config.model.training_mode,
            config.optimization.steps,
            config.optimization.learning_rate,
        )

    trainer = H3Trainer(config)
    if trainer.context.is_main:
        # config.output_dir is the root; this launch has its own directory under it.
        logger.info("run directory: %s (also reachable as %s/latest)", trainer.output_dir, config.output_dir)
    trainer.setup()
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
