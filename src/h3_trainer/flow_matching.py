"""The rectified-flow objective, in MiniMax-H3's conventions.

H3 differs from the SD3/Wan-style flow-matching convention that most training code
assumes, in two ways that are invisible at runtime and corrupt weights silently:

1. **Time runs backwards relative to sigma.** The transformer's time input is
   ``t = 1 - sigma``: ``t = 1`` is clean data, ``t = 0`` is pure noise. Feed it
   sigma directly and every sample is labelled with the opposite noise level.

2. **The prediction is a data-ward velocity** ``v = x0 - eps``. The scheduler
   reconstructs ``x0 = x_t + (1 - t) * v`` -- note the plus. Regressing the usual
   ``eps - x0`` trains the model to move away from the data.

On top of that, H3 noises video and audio at *different* sigmas: both come from a
single uniform draw ``u`` mapped through two shifted schedules (shift 12.0 for
video, 3.0 for audio), which is how the two schedulers advance in lockstep at
inference. Sampling two independent sigmas trains pairings the model never sees.

Every one of these is covered by a unit test in ``tests/test_flow_matching.py``;
they are the cheapest possible insurance against a silently wrong run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from h3_trainer.constants import DEFAULT_AUDIO_SHIFT, DEFAULT_VIDEO_SHIFT


def shift_sigma(u: torch.Tensor | float, shift: float) -> torch.Tensor | float:
    """Map a uniform ``u`` in [0, 1] onto a shifted rectified-flow schedule.

    ``sigma = shift * u / (1 + (shift - 1) * u)``. Larger shift pushes mass toward
    high noise, which is what video needs (shift 12.0) and audio does not (3.0).
    """
    return shift * u / (1.0 + (shift - 1.0) * u)


def timestep_from_sigma(sigma: torch.Tensor | float) -> torch.Tensor | float:
    """H3's model contract: ``t = 1 - sigma`` (t=1 clean, t=0 pure noise)."""
    return 1.0 - sigma


@dataclass(frozen=True)
class SigmaPair:
    """The two noise levels for one training step, drawn from a shared ``u``."""

    u: float
    video: float
    audio: float

    @property
    def video_timestep(self) -> float:
        return float(timestep_from_sigma(self.video))

    @property
    def audio_timestep(self) -> float:
        return float(timestep_from_sigma(self.audio))

    @classmethod
    def from_u(
        cls,
        u: float,
        video_shift: float = DEFAULT_VIDEO_SHIFT,
        audio_shift: float = DEFAULT_AUDIO_SHIFT,
    ) -> SigmaPair:
        return cls(
            u=float(u),
            video=float(shift_sigma(u, video_shift)),
            audio=float(shift_sigma(u, audio_shift)),
        )


def add_noise(clean: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    """``x_t = (1 - sigma) * x0 + sigma * eps`` -- the rectified-flow interpolant."""
    return (1.0 - sigma) * clean + sigma * noise


def velocity_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """H3's regression target: the **data-ward** velocity ``v = x0 - eps``."""
    return clean - noise


def flow_matching_loss(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    weight: float | torch.Tensor = 1.0,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE against the data-ward velocity, in fp32.

    Args:
        prediction: model output for these rows, ``(batch, rows, channels)``.
        clean: the clean latents for these rows, same shape (or unbatched).
        noise: the noise that was mixed in, same shape as ``clean``.
        weight: scalar multiplier. Used to zero out placeholder audio while
            keeping the term in the graph, so gradients (of zero) still flow
            through the audio head and DDP/ZeRO stay in sync across ranks. See
            ``audio_loss_weight``.
        row_mask: optional ``(rows,)`` boolean mask selecting the rows that carry
            loss. Conditioning rows (keyframes, IC-LoRA references) are excluded
            this way -- they are inputs, not targets.
    """
    target = velocity_target(clean, noise)
    if target.dim() == prediction.dim() - 1:
        target = target.unsqueeze(0)
    pred = prediction.float()
    target = target.float()
    if row_mask is not None:
        mask = row_mask.to(pred.device)
        pred = pred[..., mask, :]
        target = target[..., mask, :]
    if pred.numel() == 0:
        return pred.sum() * 0.0
    return torch.nn.functional.mse_loss(pred, target) * weight


def audio_loss_weight(clean_audio: torch.Tensor) -> float:
    """1.0 for a real audio track, 0.0 for the all-zero placeholder.

    H3 trains video and audio jointly, so a clip with no soundtrack is cached as
    zero audio rows. Regressing those teaches the audio head to predict silence
    (in latent space, to predict *noise*), which is worse than not training it.
    Zeroing the weight -- rather than dropping the term -- keeps the audio head in
    the autograd graph so every rank produces the same set of gradients.
    """
    return 1.0 if bool(clean_audio.abs().sum() > 0) else 0.0


def seeded_noise_like(
    reference: torch.Tensor,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Deterministic noise for validation, generated on CPU so it is device-agnostic.

    A validation loss is only comparable across steps if the noise is a fixed
    function of (sample, sigma) -- otherwise the curve is dominated by which noise
    happened to be drawn.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
    noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    return noise.to(device if device is not None else reference.device, reference.dtype)
