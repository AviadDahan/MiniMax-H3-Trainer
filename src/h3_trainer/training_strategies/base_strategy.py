"""The strategy interface: batch of latents in, packed model inputs and loss out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from h3_trainer.flow_matching import SigmaPair
from h3_trainer.packing import PackedBatch


@dataclass
class LossBreakdown:
    """Per-modality losses, kept apart because they behave very differently.

    A run where the total loss looks fine but the audio term is flat is a run
    where the audio branch is not learning -- that is invisible in a single
    scalar, and it is the most common way an H3 fine-tune goes quietly wrong.
    """

    total: torch.Tensor
    video: torch.Tensor
    audio: torch.Tensor
    audio_weight: float
    extras: dict[str, float] = field(default_factory=dict)

    def as_log_dict(self, prefix: str = "") -> dict[str, float]:
        payload = {
            f"{prefix}loss": float(self.total.detach()),
            f"{prefix}loss_video": float(self.video.detach()),
            f"{prefix}loss_audio": float(self.audio.detach()),
            f"{prefix}audio_weight": self.audio_weight,
        }
        payload.update({f"{prefix}{key}": value for key, value in self.extras.items()})
        return payload


class TrainingStrategy(ABC):
    """Turns a cached sample into a packed sequence, and model outputs into a loss."""

    @abstractmethod
    def prepare(
        self,
        batch: dict[str, Any],
        sigmas: SigmaPair,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        generator: torch.Generator | None = None,
        noise_seed: int | None = None,
    ) -> PackedBatch: ...

    @abstractmethod
    def compute_loss(
        self,
        prediction_video: torch.Tensor,
        prediction_audio: torch.Tensor,
        packed: PackedBatch,
    ) -> LossBreakdown: ...
