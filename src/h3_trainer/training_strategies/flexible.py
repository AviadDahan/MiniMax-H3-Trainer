"""The one training strategy: everything is a combination of config flags.

t2va, i2v/fl2va, a2v, v2a and IC-LoRA are not separate code paths. They are
answers to two questions asked per modality:

* is this modality **generated** (noised, predicted, and in the loss), or is it
  **frozen conditioning** (packed clean and excluded from the loss)?
* what **conditioning blocks** ride in front of the targets -- keyframes, or
  in-context references?

That is the whole design. Adding a new mode should mean writing a YAML file,
not a class.
"""

from __future__ import annotations

from typing import Any

import torch

from h3_trainer import logger
from h3_trainer.config import FlexibleStrategyConfig
from h3_trainer.constants import Geometry
from h3_trainer.flow_matching import (
    SigmaPair,
    audio_loss_weight,
    flow_matching_loss,
    seeded_noise_like,
    timestep_from_sigma,
)
from h3_trainer.packing import (
    PackedBatch,
    assemble,
    build_layout,
    check_reference_rows,
    prepared_references_from_cache,
)
from h3_trainer.training_strategies.base_strategy import LossBreakdown, TrainingStrategy


class FlexibleStrategy(TrainingStrategy):
    def __init__(self, config: FlexibleStrategyConfig) -> None:
        self.config = config
        self._warned_missing: set[str] = set()

    # ------------------------------------------------------------------ prepare

    def prepare(
        self,
        batch: dict[str, Any],
        sigmas: SigmaPair,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        generator: torch.Generator | None = None,
        noise_seed: int | None = None,
    ) -> PackedBatch:
        record = batch["record"]
        geometry: Geometry = record.geometry
        video_latents = batch["video"]
        audio_latents = batch["audio"]
        text_embeds = batch["text"]

        keyframe_anchors, condition_video = self._collect_keyframes(batch, generator)
        references, reference_video, reference_audio = self._collect_references(batch, generator)
        if reference_video is not None:
            condition_video = reference_video

        layout = build_layout(
            num_text_rows=text_embeds.shape[-2],
            latent_frames=geometry.latent_frames,
            latent_height=geometry.latent_height,
            latent_width=geometry.latent_width,
            audio_latents=geometry.audio_latents,
            keyframe_anchors=keyframe_anchors,
            references=references,
            tags=batch.get("text_token_tags"),
        )

        # A modality that is not generated is packed clean at t=1 and never
        # enters the loss -- that is what makes audio-to-video and
        # video-to-audio fall out of the same code path.
        sigma_video = sigmas.video if self.config.video.is_generated else 0.0
        sigma_audio = sigmas.audio if self.config.audio.is_generated else 0.0

        noise_video = noise_audio = None
        if noise_seed is not None:
            # Validation: noise must be a fixed function of (sample, sigma) so the
            # curve reflects the weights and nothing else.
            noise_video = seeded_noise_like(video_latents, noise_seed, device)
            noise_audio = seeded_noise_like(audio_latents, noise_seed + 1, device)

        return assemble(
            layout=layout,
            target_video=video_latents,
            target_audio=audio_latents,
            text_embeds=text_embeds,
            sigma_video=sigma_video,
            sigma_audio=sigma_audio,
            video_timestep=float(timestep_from_sigma(sigma_video)),
            audio_timestep=float(timestep_from_sigma(sigma_audio)),
            condition_video=condition_video,
            condition_audio=reference_audio,
            generator=generator,
            noise_video=noise_video,
            noise_audio=noise_audio,
            device=device,
            dtype=dtype,
        )

    def _collect_keyframes(
        self, batch: dict[str, Any], generator: torch.Generator | None
    ) -> tuple[tuple[str, ...], torch.Tensor | None]:
        """Assemble the keyframe conditioning block, honouring per-condition probability."""
        anchors: list[str] = []
        rows: list[torch.Tensor] = []
        for condition in self.config.video.conditions:
            if condition.type not in ("first_frame", "last_frame"):
                continue
            if not self._draw(condition.probability, generator):
                continue
            cached = batch.get(condition.type)
            if cached is None:
                self._warn_once(
                    condition.type,
                    f"'{condition.type}' conditioning is configured but no cached latents were found; "
                    f"re-run process_dataset.py with the matching column.",
                )
                continue
            anchors.append("first" if condition.type == "first_frame" else "last")
            rows.append(cached)
        if not rows:
            return (), None
        return tuple(anchors), torch.cat([r if r.dim() == 2 else r.squeeze(0) for r in rows], dim=0)

    def _collect_references(
        self, batch: dict[str, Any], generator: torch.Generator | None
    ) -> tuple[list | None, torch.Tensor | None, torch.Tensor | None]:
        """Build the in-context reference blocks (and their rows) for the ref2va layout.

        Dropping the reference with probability ``1 - p`` is deliberate: an
        IC-LoRA that has only ever seen a reference forgets how to generate
        without one, and the unconditional path is what validation prompts and
        plain text-to-video use.
        """
        wanted = [
            condition
            for modality in (self.config.video, self.config.audio)
            for condition in modality.conditions
            if condition.type == "reference"
        ]
        if not wanted or not any(self._draw(condition.probability, generator) for condition in wanted):
            return None, None, None

        cached = batch.get("reference")
        if cached is None:
            self._warn_once(
                "reference",
                "Reference conditioning is configured but no cached reference latents were found; "
                "re-run process_dataset.py with a reference_video / reference_image / reference_audio column.",
            )
            return None, None, None

        references = prepared_references_from_cache(
            [(condition.modality, cached, geom) for condition, geom in zip(wanted, batch["reference_geometry"])]
        )
        video_rows = batch.get("reference_video_rows")
        audio_rows = batch.get("reference_audio_rows")
        check_reference_rows(references, video_rows, audio_rows)
        return references, video_rows, audio_rows

    @staticmethod
    def _draw(probability: float, generator: torch.Generator | None) -> bool:
        if probability >= 1.0:
            return True
        if probability <= 0.0:
            return False
        return bool(torch.rand((), generator=generator) < probability)

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned_missing:
            self._warned_missing.add(key)
            logger.warning(message)

    # ------------------------------------------------------------------- loss

    def compute_loss(
        self,
        prediction_video: torch.Tensor,
        prediction_audio: torch.Tensor,
        packed: PackedBatch,
    ) -> LossBreakdown:
        device = prediction_video.device
        zero = torch.zeros((), device=device, dtype=torch.float32)

        if self.config.video.is_generated:
            video_loss = flow_matching_loss(
                prediction_video,
                packed.clean_video,
                packed.noise_video,
                row_mask=packed.video_loss_mask,
            )
        else:
            video_loss = zero

        if self.config.audio.is_generated:
            weight = audio_loss_weight(packed.clean_audio)
            audio_loss = flow_matching_loss(
                prediction_audio,
                packed.clean_audio,
                packed.noise_audio,
                row_mask=packed.audio_loss_mask,
            )
        else:
            weight = 0.0
            audio_loss = zero

        # The audio term stays in the graph even at weight 0 so that every rank
        # produces gradients for the same parameter set; dropping it makes DDP
        # and ZeRO-3 disagree about which buckets to reduce and hang.
        total = video_loss + weight * audio_loss
        return LossBreakdown(
            total=total,
            video=video_loss.detach(),
            audio=audio_loss.detach(),
            audio_weight=weight,
            extras={"seq_len": float(packed.sequence_length)},
        )


def get_training_strategy(config: FlexibleStrategyConfig) -> TrainingStrategy:
    if config.name != "flexible":
        raise ValueError(f"Unknown training strategy {config.name!r}")
    return FlexibleStrategy(config)
