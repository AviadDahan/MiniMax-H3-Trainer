"""Building the packed sequence the H3 transformer consumes.

H3 does not take separate video / audio / text tensors with cross-attention
between them. It takes one flat sequence of rows:

    [ text | conditioning blocks | target audio | target video ]

plus per-row metadata (rotary position ids, modality tags, and index vectors
saying which rows are which). ``diffusers`` builds that layout for inference;
this module reuses those same builders for training, which is the only way to
guarantee the layout a LoRA is trained under is the layout it is used under.

Everything the trainer needs beyond the raw layout is derived here:

* the row order the model expects for ``hidden_states`` / ``audio_hidden_states``
  (conditioning rows first, then targets),
* the loss masks that exclude conditioning rows -- they are inputs, not targets,
* the ``(timestep, timestep_indices)`` pair, with target rows at their sampled
  noise level and conditioning rows pinned near t=1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from h3_trainer import logger
from h3_trainer.constants import CONDITION_TIMESTEP, PATCH_SIZE
from h3_trainer.flow_matching import add_noise

try:
    from diffusers.modular_pipelines.minimax_h3.packing import (
        MiniMaxH3PackedSequence,
        build_packed_sequence,
        build_row_timesteps,
    )
    from diffusers.modular_pipelines.minimax_h3.packing_ref2va import build_ref2va_packed_sequence
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the pinned diffusers commit (scripts/install_env.sh)") from exc

#: Row structural fields the transformer takes as keyword arguments.
LAYOUT_FIELDS = ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")


@dataclass
class PackedBatch:
    """One packed sequence, ready to hand to the transformer.

    ``video`` and ``audio`` are already in the order the layout indexes them:
    conditioning rows first, then target rows. ``video_loss_mask`` /
    ``audio_loss_mask`` select the target rows back out of the model output.
    """

    video: torch.Tensor  # (batch, video_rows, channels)
    audio: torch.Tensor  # (batch, audio_rows, channels)
    text: torch.Tensor  # (batch, text_rows, text_dim)
    timestep: torch.Tensor
    timestep_indices: torch.Tensor
    layout_kwargs: dict[str, torch.Tensor]
    video_loss_mask: torch.Tensor
    audio_loss_mask: torch.Tensor
    clean_video: torch.Tensor  # target rows only
    clean_audio: torch.Tensor  # target rows only
    noise_video: torch.Tensor
    noise_audio: torch.Tensor
    sequence_length: int
    metadata: dict[str, object] = field(default_factory=dict)

    def to_model_kwargs(self) -> dict[str, object]:
        return {
            "hidden_states": self.video,
            "audio_hidden_states": self.audio,
            "encoder_hidden_states": self.text,
            "timestep": self.timestep,
            "timestep_indices": self.timestep_indices,
            "attention_kwargs": None,
            "return_dict": False,
            **self.layout_kwargs,
        }


def text_token_tags(num_text_rows: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Modality tags for a plain text prompt (no embedded vision blocks)."""
    from h3_trainer.constants import MINIMAX_H3_TEXT_TAG

    return torch.full((num_text_rows,), MINIMAX_H3_TEXT_TAG, dtype=torch.long, device=device)


def build_layout(
    num_text_rows: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    audio_latents: int,
    keyframe_anchors: tuple[str, ...] = (),
    references: list | None = None,
    tags: torch.Tensor | None = None,
) -> MiniMaxH3PackedSequence:
    """Build the packed layout for one sample.

    With ``references`` the ref2va layout is used (``[text | reference blocks |
    target audio | target video]``); otherwise the t2va/fl2va layout, whose
    conditioning blocks are keyframes anchored ``"first"`` and/or ``"last"``.

    ``tags`` are the per-text-row modality tags produced when the prompt was
    encoded. They matter whenever the prompt embeds vision blocks (keyframe
    ``<Picture N>`` labels, reference media): those rows are tagged *video*, not
    text, and the transformer's AdaLN modulation keys off the tag. Passing
    ``None`` assumes a pure-text prompt.
    """
    if tags is None:
        tags = text_token_tags(num_text_rows)
    else:
        tags = tags.to(torch.long).flatten()
        if tags.numel() != num_text_rows:
            raise ValueError(
                f"text_token_tags has {tags.numel()} entries but the prompt embedding has "
                f"{num_text_rows} rows -- the cached conditions are inconsistent."
            )
    if references:
        if keyframe_anchors:
            raise ValueError(
                "ref2va packs reference blocks, not keyframe anchors; drop the first_frame/last_frame "
                "conditions or switch to the fl2va variant."
            )
        return build_ref2va_packed_sequence(
            text_token_tags=tags,
            references=references,
            num_latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=audio_latents,
            patch_size=PATCH_SIZE,
        )
    return build_packed_sequence(
        text_token_tags=tags,
        num_latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=audio_latents,
        patch_size=PATCH_SIZE,
        keyframe_anchors=keyframe_anchors,
    )


#: Order of the integer fields in a cached reference's ``geometry`` tensor.
REFERENCE_GEOMETRY_FIELDS = ("num_latent_frames", "latent_height", "latent_width", "num_audio_latents")
REFERENCE_KINDS = ("image", "video", "audio")


def encode_reference_geometry(
    kind: str,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
) -> torch.Tensor:
    """Pack a reference's latent geometry into the tensor stored beside its rows."""
    if kind not in REFERENCE_KINDS:
        raise ValueError(f"Reference kind must be one of {REFERENCE_KINDS}, got {kind!r}")
    return torch.tensor(
        [REFERENCE_KINDS.index(kind), num_latent_frames, latent_height, latent_width, num_audio_latents],
        dtype=torch.int64,
    )


def prepared_references_from_cache(entries: list[tuple[str, torch.Tensor, torch.Tensor | None]], geometry=None):
    """Rebuild the reference descriptors the ref2va layout builder needs.

    At inference a reference is prepared from raw media and its latent geometry
    falls out of the VAE encode. In training the encode already happened offline,
    so the geometry travels with the cached rows and is replayed here. Only the
    geometry fields matter to the packer -- ``image``/``frames``/``waveform`` are
    what the *encoder* blocks consume, and those already ran.

    Args:
        entries: ``(configured_modality, rows, geometry_tensor)`` per reference.
            The kind stored with the cache wins over the configured modality: the
            cache knows what was actually encoded.
    """
    from diffusers.modular_pipelines.minimax_h3.packing_ref2va import MiniMaxH3PreparedReference

    references = []
    for configured_kind, _rows, geometry_tensor in entries:
        if geometry_tensor is None:
            raise ValueError(
                "Cached reference latents carry no geometry tensor; re-run process_dataset.py so the "
                "reference rows are written together with their latent geometry."
            )
        values = [int(v) for v in geometry_tensor.flatten().tolist()]
        kind = REFERENCE_KINDS[values[0]]
        if kind != configured_kind:
            logger.debug("Reference cached as %r while config says %r; using the cache", kind, configured_kind)
        fields = dict(zip(REFERENCE_GEOMETRY_FIELDS, values[1:], strict=True))
        references.append(
            MiniMaxH3PreparedReference(
                kind=kind,
                has_audio=fields["num_audio_latents"] > 0,
                num_latent_frames=max(1, fields["num_latent_frames"]),
                latent_height=fields["latent_height"],
                latent_width=fields["latent_width"],
                num_audio_latents=fields["num_audio_latents"],
            )
        )
    return references


def check_reference_rows(
    references: list,
    video_rows: torch.Tensor | None,
    audio_rows: torch.Tensor | None,
) -> None:
    """Verify cached reference rows against the geometry the layout will assume."""
    expected_video = sum(r.num_video_rows for r in references if r.kind != "audio")
    expected_audio = sum(r.num_audio_rows for r in references if r.has_audio)
    actual_video = 0 if video_rows is None else video_rows.shape[-2]
    actual_audio = 0 if audio_rows is None else audio_rows.shape[-2]
    if actual_video != expected_video or actual_audio != expected_audio:
        raise ValueError(
            f"Cached reference rows disagree with their geometry: video {actual_video} vs "
            f"{expected_video}, audio {actual_audio} vs {expected_audio}."
        )


def layout_kwargs(layout: MiniMaxH3PackedSequence, device: torch.device | str) -> dict[str, torch.Tensor]:
    return {name: getattr(layout, name).to(device) for name in LAYOUT_FIELDS}


def loss_masks(layout: MiniMaxH3PackedSequence, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks over the video / audio row axes selecting the *target* rows.

    Conditioning rows sit at the front of each modality's index vector. They are
    given to the model as near-clean inputs and must never contribute loss --
    regressing them teaches the model to denoise something it was handed.
    """
    num_video = int(layout.video_indices.numel())
    num_audio = int(layout.audio_indices.numel())
    video_mask = torch.ones(num_video, dtype=torch.bool, device=device)
    audio_mask = torch.ones(num_audio, dtype=torch.bool, device=device)
    video_mask[: layout.num_condition_video_rows] = False
    audio_mask[: layout.num_condition_audio_rows] = False
    return video_mask, audio_mask


#: Reference audio rows are handed to the model completely clean -- unlike visual
#: conditioning rows, the inference path applies no noise augmentation to them.
CONDITION_AUDIO_TIMESTEP = 1.0


def row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(timestep, timestep_indices)``, matching the inference pinning exactly.

    Visual conditioning rows sit at ``max(t, 0.999)`` -- pinned just shy of clean,
    but never *behind* the generated rows -- while reference audio rows sit at
    ``1.0``, fully clean.
    """
    timestep, indices = build_row_timesteps(
        layout,
        float(video_timestep),
        float(audio_timestep),
        max(float(video_timestep), float(CONDITION_TIMESTEP)),
        float(CONDITION_AUDIO_TIMESTEP),
    )
    return timestep.to(device), indices.to(device)


def noise_condition_rows(clean_rows: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Apply the noise augmentation visual conditioning rows carry at inference.

    Keyframe and visual reference rows are mixed at t = ``CONDITION_TIMESTEP``
    (0.999), i.e. sigma = 0.001: a whisper of noise, not clean data. Feeding
    perfectly clean conditioning trains a mismatch against every generation.
    """
    return add_noise(clean_rows, noise, sigma=1.0 - float(CONDITION_TIMESTEP))


def assemble(
    *,
    layout: MiniMaxH3PackedSequence,
    target_video: torch.Tensor,
    target_audio: torch.Tensor,
    text_embeds: torch.Tensor,
    sigma_video: float,
    sigma_audio: float,
    video_timestep: float,
    audio_timestep: float,
    condition_video: torch.Tensor | None = None,
    condition_audio: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    noise_video: torch.Tensor | None = None,
    noise_audio: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> PackedBatch:
    """Noise the targets, prepend the conditioning rows, and assemble model inputs.

    ``target_video`` / ``target_audio`` are the clean cached latent rows, unbatched
    ``(rows, channels)`` or batched ``(batch, rows, channels)``.
    """
    target_video = _ensure_batched(target_video).to(device)
    target_audio = _ensure_batched(target_audio).to(device)
    text_embeds = _ensure_batched(text_embeds).to(device, dtype)

    if noise_video is None:
        noise_video = torch.randn(target_video.shape, generator=generator, device=device, dtype=torch.float32)
    if noise_audio is None:
        noise_audio = torch.randn(target_audio.shape, generator=generator, device=device, dtype=torch.float32)
    noise_video = noise_video.to(device, target_video.dtype)
    noise_audio = noise_audio.to(device, target_audio.dtype)

    noisy_video = add_noise(target_video, noise_video, sigma_video)
    noisy_audio = add_noise(target_audio, noise_audio, sigma_audio)

    video_rows = _prepend_conditions(noisy_video, condition_video, device, generator, noise_augment=True)
    # Reference audio rows go in clean: the inference path noise-augments visual
    # conditioning only, and pins the audio references at t = 1.0.
    audio_rows = _prepend_conditions(noisy_audio, condition_audio, device, generator, noise_augment=False)

    expected_video = int(layout.video_indices.numel())
    expected_audio = int(layout.audio_indices.numel())
    if video_rows.shape[1] != expected_video or audio_rows.shape[1] != expected_audio:
        raise ValueError(
            f"Row count does not match the packed layout: video {video_rows.shape[1]} vs {expected_video}, "
            f"audio {audio_rows.shape[1]} vs {expected_audio}. The cached latents were probably encoded "
            f"at a different geometry than the layout was built for."
        )
    if text_embeds.shape[1] != int(layout.text_indices.numel()):
        raise ValueError(
            f"Text rows {text_embeds.shape[1]} do not match the layout's {int(layout.text_indices.numel())}"
        )

    timestep, timestep_indices = row_timesteps(layout, video_timestep, audio_timestep, device)
    video_mask, audio_mask = loss_masks(layout, device)

    return PackedBatch(
        video=video_rows.to(dtype),
        audio=audio_rows.to(dtype),
        text=text_embeds,
        timestep=timestep,
        timestep_indices=timestep_indices,
        layout_kwargs=layout_kwargs(layout, device),
        video_loss_mask=video_mask,
        audio_loss_mask=audio_mask,
        clean_video=target_video,
        clean_audio=target_audio,
        noise_video=noise_video,
        noise_audio=noise_audio,
        sequence_length=int(layout.sequence_length),
    )


def _ensure_batched(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dim() == 3 else tensor.unsqueeze(0)


def _prepend_conditions(
    target_rows: torch.Tensor,
    condition_rows: torch.Tensor | None,
    device: torch.device | str,
    generator: torch.Generator | None,
    noise_augment: bool,
) -> torch.Tensor:
    if condition_rows is None:
        return target_rows
    condition_rows = _ensure_batched(condition_rows).to(device, target_rows.dtype)
    if noise_augment:
        noise = torch.randn(condition_rows.shape, generator=generator, device=device, dtype=torch.float32)
        condition_rows = noise_condition_rows(condition_rows, noise.to(condition_rows.dtype))
    return torch.cat([condition_rows, target_rows], dim=1)
