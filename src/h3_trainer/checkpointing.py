"""Saving and restoring training state.

Three hard-won rules are encoded here:

* **Save only what trains.** A full H3 state dict is ~66GB; writing it from rank 0
  stalls the process long enough for the NCCL watchdog to kill the run. LoRA and
  head checkpoints are a few megabytes.
* **Gather ZeRO-3 shards carefully.** ``GatheredParameters`` can race the
  prefetcher and hand rank 0 empty tensors. A checkpoint of empty tensors looks
  fine until it breaks every future resume, so an empty gather is detected and
  the save is *skipped* rather than written.
* **Restore before sharding.** Weights must be loaded while parameters are still
  full tensors -- after ZeRO-3 partitions them, ``load_state_dict`` sees shape-[0]
  shards and silently loads nothing.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from h3_trainer import logger

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


@dataclass
class TrainingState:
    step: int = 0
    epoch: int = 0
    best_val_loss: float | None = None
    samples_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "samples_seen": self.samples_seen,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingState:
        return cls(
            step=int(payload.get("step", 0)),
            epoch=int(payload.get("epoch", 0)),
            best_val_loss=payload.get("best_val_loss"),
            samples_seen=int(payload.get("samples_seen", 0)),
        )


def checkpoint_dir(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / f"checkpoint-{step:07d}"


def find_latest_checkpoint(path: str | Path) -> Path | None:
    """Newest ``checkpoint-*`` under a directory, or the path itself if it is one."""
    path = Path(path)
    if path.is_file():
        return path
    if (path / "adapter.safetensors").exists():
        return path
    candidates = [
        entry for entry in path.glob("checkpoint-*") if CHECKPOINT_PATTERN.search(entry.name)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: int(CHECKPOINT_PATTERN.search(entry.name).group(1)))


def gather_trainable_state(
    model: torch.nn.Module, engine: Any | None, dtype: torch.dtype
) -> dict[str, torch.Tensor] | None:
    """Trainable tensors as a plain state dict, gathering ZeRO-3 shards if needed.

    Returns ``None`` when the gather produced empty tensors -- see the module
    docstring; a skipped save beats a corrupt one.
    """
    module = engine.module if engine is not None else model
    named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    if not named:
        raise RuntimeError("Nothing is trainable -- refusing to write an empty checkpoint")

    if engine is not None and _is_zero3(engine):
        import deepspeed

        torch.cuda.synchronize()
        with deepspeed.zero.GatheredParameters([parameter for _, parameter in named]):
            state = {name: parameter.detach().to(dtype).cpu().clone() for name, parameter in named}
    else:
        state = {name: parameter.detach().to(dtype).cpu().clone() for name, parameter in named}

    empty = [name for name, tensor in state.items() if tensor.numel() == 0]
    if empty:
        logger.error(
            "Checkpoint gather produced %d empty tensors (e.g. %s); skipping this save",
            len(empty),
            empty[:2],
        )
        return None
    return state


def _is_zero3(engine: Any) -> bool:
    try:
        return int(engine.zero_optimization_stage()) == 3
    except Exception:
        return False


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    state: TrainingState,
    engine: Any | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    dtype: torch.dtype = torch.bfloat16,
    save_training_state: str = "full",
    config_snapshot: dict[str, Any] | None = None,
    is_main_process: bool = True,
) -> Path | None:
    """Write ``checkpoint-<step>/`` with the adapter weights and optional optimizer state."""
    weights = gather_trainable_state(model, engine, dtype)
    if not is_main_process:
        return None
    if weights is None:
        return None

    import shutil

    target = checkpoint_dir(output_dir, step)
    staging = target.with_suffix(".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    save_file(weights, str(staging / "adapter.safetensors"), metadata={"step": str(step)})
    with (staging / "training_state.json").open("w") as handle:
        json.dump(state.to_dict(), handle, indent=1)
    if config_snapshot is not None:
        with (staging / "config.json").open("w") as handle:
            json.dump(config_snapshot, handle, indent=1, default=str)

    if save_training_state == "full" and optimizer is not None and engine is None:
        # Under DeepSpeed the optimizer owns fp32 flat partitions that a plain
        # state_dict cannot round-trip; use engine.save_checkpoint for that path.
        payload: dict[str, Any] = {"optimizer": optimizer.state_dict()}
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        torch.save(payload, staging / "optimizer.pt")

    # os.replace refuses to overwrite a non-empty directory, and the same step can
    # legitimately be saved twice (interval save, then the final save).
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)
    logger.info("Saved checkpoint %s (%d tensors)", target, len(weights))
    return target


def prune_checkpoints(output_dir: str | Path, keep_last_n: int) -> None:
    if keep_last_n < 0:
        return
    import shutil

    checkpoints = sorted(
        (entry for entry in Path(output_dir).glob("checkpoint-*") if entry.is_dir()),
        key=lambda entry: int(CHECKPOINT_PATTERN.search(entry.name).group(1)),
    )
    for stale in checkpoints[: max(0, len(checkpoints) - keep_last_n)]:
        shutil.rmtree(stale, ignore_errors=True)
        logger.info("Pruned old checkpoint %s", stale)


def load_checkpoint_weights(path: str | Path) -> tuple[dict[str, torch.Tensor], TrainingState]:
    """Read adapter weights + training state from a checkpoint directory or file."""
    path = Path(path)
    if path.is_dir():
        weights_path = path / "adapter.safetensors"
        state_path = path / "training_state.json"
    else:
        weights_path, state_path = path, path.parent / "training_state.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"No adapter weights at {weights_path}")
    weights = load_file(str(weights_path))
    state = TrainingState()
    if state_path.exists():
        with state_path.open() as handle:
            state = TrainingState.from_dict(json.load(handle))
    return weights, state


def apply_checkpoint(model: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Load trainable weights into an *unsharded* model, strictly enough to catch mismatches.

    ``strict=False`` is required (the frozen base is absent from the checkpoint),
    which also means a checkpoint from a different adapter configuration would
    load *nothing* and still look successful. Unexpected keys are therefore fatal.
    """
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected:
        raise ValueError(
            f"Checkpoint does not match this model: {len(unexpected)} unexpected keys "
            f"(e.g. {unexpected[:3]}). It was probably saved with different lora.target_modules "
            f"or a different training_mode."
        )
    loaded = len(weights)
    logger.info("Loaded %d trainable tensors (%d base tensors left untouched)", loaded, len(missing))
