#!/usr/bin/env python
"""Convert a trained checkpoint into a portable LoRA file.

    python scripts/export_lora.py runs/character/checkpoint-0001200 --out mira.safetensors

The default ``comfyui`` layout re-fuses Q/K/V into the single ``qkv_proj`` the
original MiniMax packaging uses, which is what ComfyUI's loader and the published
H3 adapters expect. Three rank-r updates concatenated along the output axis are
exactly one rank-3r update with a block-diagonal B, so nothing is approximated.

Use ``--format peft`` to keep the raw training layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from h3_trainer import logger  # noqa: E402
from h3_trainer.checkpointing import find_latest_checkpoint, load_checkpoint_weights  # noqa: E402
from h3_trainer.lora import export_lora  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="checkpoint directory, run directory, or adapter file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format", choices=("comfyui", "peft"), default="comfyui")
    args = parser.parse_args()

    checkpoint = find_latest_checkpoint(args.checkpoint)
    if checkpoint is None:
        raise SystemExit(f"No checkpoint found under {args.checkpoint}")
    weights, state = load_checkpoint_weights(checkpoint)

    metadata = {"h3_trainer_step": str(state.step), "source_checkpoint": str(checkpoint)}
    config_path = (checkpoint if checkpoint.is_dir() else checkpoint.parent) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        metadata.update(
            {
                "variant": str(config.get("model", {}).get("variant", "")),
                "rank": str(config.get("lora", {}).get("rank", "")),
                "alpha": str(config.get("lora", {}).get("alpha", "")),
                "target_modules": ",".join(config.get("lora", {}).get("target_modules", [])),
            }
        )

    export_lora(weights, args.out, metadata=metadata, fmt=args.format)
    logger.info("Exported step %d from %s", state.step, checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
