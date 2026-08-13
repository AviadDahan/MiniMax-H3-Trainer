"""Run logging: a plain text log that survives anything, plus optional W&B."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from h3_trainer import logger


class RunLogger:
    """Writes metrics to ``train.log`` / ``metrics.jsonl`` and, optionally, W&B.

    The file logs are not a fallback -- they are the primary record. A long run
    on a shared box loses its W&B connection often enough that a training curve
    which only exists in the cloud is a training curve you can lose.
    """

    def __init__(
        self,
        output_dir: str | Path,
        wandb_config: Any | None = None,
        config_snapshot: dict[str, Any] | None = None,
        is_main_process: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.is_main_process = is_main_process
        self._run = None
        self._text = None
        self._jsonl = None

        if not is_main_process:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._text = (self.output_dir / "train.log").open("a", buffering=1)
        self._jsonl = (self.output_dir / "metrics.jsonl").open("a", buffering=1)
        self._write_text(f"=== run started {datetime.now(UTC).isoformat()} ===")

        if wandb_config is not None and wandb_config.enabled:
            try:
                import wandb

                self._run = wandb.init(
                    project=wandb_config.project,
                    entity=wandb_config.entity,
                    name=wandb_config.name,
                    tags=wandb_config.tags,
                    dir=str(self.output_dir),
                    config=config_snapshot,
                )
                logger.info("W&B run: %s", self._run.url)
            except Exception as exc:  # never let telemetry take down a training run
                logger.warning("W&B init failed (%s); continuing with file logging only", exc)

    def _write_text(self, message: str) -> None:
        if self._text is not None:
            self._text.write(f"{datetime.now(UTC).isoformat()} {message}\n")

    def log(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "",
        console: bool = False,
    ) -> None:
        if not self.is_main_process:
            return
        payload = {f"{prefix}{key}": value for key, value in metrics.items()}
        payload["step"] = step
        if self._jsonl is not None:
            self._jsonl.write(json.dumps(payload) + "\n")
        self._write_text(" ".join(f"{key}={_format(value)}" for key, value in payload.items()))
        if console:
            logger.info(_console_line(metrics, step, prefix))
        if self._run is not None:
            self._run.log(payload, step=step)

    def log_media(self, name: str, path: str | Path, step: int, caption: str | None = None) -> None:
        if not self.is_main_process:
            return
        self._write_text(f"media {name}={path}")
        if self._run is None:
            return
        try:
            import wandb

            self._run.log({name: wandb.Video(str(path), caption=caption, format="mp4")}, step=step)
        except Exception as exc:
            logger.warning("Could not log media to W&B: %s", exc)

    def message(self, text: str) -> None:
        if not self.is_main_process:
            return
        logger.info(text)
        self._write_text(text)

    def summary(self, values: dict[str, Any]) -> None:
        if not self.is_main_process:
            return
        self._write_text("summary " + json.dumps(values, default=str))
        if self._run is not None:
            self._run.summary.update(values)

    def close(self) -> None:
        if not self.is_main_process:
            return
        self._write_text("=== run finished ===")
        for handle in (self._text, self._jsonl):
            if handle is not None:
                handle.close()
        if self._run is not None:
            self._run.finish()


def _console_line(metrics: dict[str, float], step: int, prefix: str) -> str:
    """A compact one-liner: the numbers you actually watch while a run is going.

    The per-modality losses are always shown -- a joint audio-video run where the
    total looks healthy and the audio term is flat is the failure this line exists
    to make visible.
    """
    label = "val " if prefix.startswith("val") else "step"
    parts = [f"{label} {step}"]
    for key, short in (
        ("loss", "loss"),
        ("loss_video", "video"),
        ("loss_audio", "audio"),
        ("lr", "lr"),
        ("grad_norm", "gnorm"),
        ("sigma_video", "sv"),
        ("seq_len", "seq"),
        ("steps_per_sec", "it/s"),
        ("vram_gb", "vram"),
    ):
        if key in metrics:
            value = metrics[key]
            if key == "seq_len":
                parts.append(f"{short}={int(value)}")
            elif key == "lr":
                parts.append(f"{short}={value:.2e}")
            elif key == "vram_gb":
                parts.append(f"{short}={value:.1f}G")
            else:
                parts.append(f"{short}={value:.4f}")
    return "  ".join(parts)


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}" if abs(value) < 1e4 else f"{value:.3e}"
    return str(value)
