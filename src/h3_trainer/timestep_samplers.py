"""Where along the flow trajectory each training step lands.

A sampler produces a single uniform-ish ``u`` in (0, 1); ``flow_matching.SigmaPair``
then maps that one draw through H3's two shifted schedules. Keeping the samplers
in ``u`` space rather than sigma space is what guarantees the video and audio
sigmas stay paired the way inference pairs them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch

TimestepSamplingMode = Literal["uniform", "logit_normal", "shifted_logit_normal"]


class TimestepSampler(ABC):
    """Samples ``u`` in (0, 1)."""

    @abstractmethod
    def sample(self, generator: torch.Generator | None = None) -> float: ...

    @staticmethod
    def _rand(generator: torch.Generator | None) -> torch.Tensor:
        return torch.rand((), generator=generator)


class UniformTimestepSampler(TimestepSampler):
    """Uniform in ``[min_u, max_u]``.

    The default clamp mirrors the reference H3 trainer: the endpoints are both
    degenerate (u=0 is clean data, u=1 is pure noise) and contribute gradients
    that are either trivial or enormous.
    """

    def __init__(self, min_u: float = 0.02, max_u: float = 0.98) -> None:
        if not 0.0 <= min_u < max_u <= 1.0:
            raise ValueError(f"Require 0 <= min_u < max_u <= 1, got ({min_u}, {max_u})")
        self.min_u = min_u
        self.max_u = max_u

    def sample(self, generator: torch.Generator | None = None) -> float:
        return float(self.min_u + (self.max_u - self.min_u) * self._rand(generator))


class LogitNormalTimestepSampler(TimestepSampler):
    """``u = sigmoid(N(mean, std))`` -- concentrates steps in the middle of the trajectory.

    The middle is where the model actually has to decide what the video is; the
    extremes are close to identity maps. This is the SD3 recipe.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, min_u: float = 0.02, max_u: float = 0.98) -> None:
        self.mean = mean
        self.std = std
        self.min_u = min_u
        self.max_u = max_u

    def sample(self, generator: torch.Generator | None = None) -> float:
        normal = torch.randn((), generator=generator) * self.std + self.mean
        return float(torch.sigmoid(normal).clamp(self.min_u, self.max_u))


class ShiftedLogitNormalTimestepSampler(LogitNormalTimestepSampler):
    """Logit-normal biased toward high noise, with a uniform escape hatch.

    Long sequences need proportionally more of the high-noise regime -- that is
    where global structure is decided, and a 10k-token video has a lot of global
    structure. ``uniform_fraction`` of draws fall back to uniform so the tails
    never go completely unvisited (a purely logit-normal schedule can leave the
    endpoints untrained enough to show up as artifacts at inference).
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        shift: float = 1.0,
        uniform_fraction: float = 0.1,
        min_u: float = 0.02,
        max_u: float = 0.98,
    ) -> None:
        super().__init__(mean=mean, std=std, min_u=min_u, max_u=max_u)
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self.shift = shift
        self.uniform_fraction = uniform_fraction
        self._uniform = UniformTimestepSampler(min_u, max_u)

    def sample(self, generator: torch.Generator | None = None) -> float:
        if self.uniform_fraction > 0 and float(self._rand(generator)) < self.uniform_fraction:
            return self._uniform.sample(generator)
        u = super().sample(generator)
        shifted = self.shift * u / (1.0 + (self.shift - 1.0) * u)
        return float(min(max(shifted, self.min_u), self.max_u))


def build_timestep_sampler(mode: TimestepSamplingMode, params: dict[str, Any] | None = None) -> TimestepSampler:
    params = dict(params or {})
    samplers = {
        "uniform": UniformTimestepSampler,
        "logit_normal": LogitNormalTimestepSampler,
        "shifted_logit_normal": ShiftedLogitNormalTimestepSampler,
    }
    if mode not in samplers:
        raise ValueError(f"Unknown timestep_sampling_mode {mode!r}; choose from {sorted(samplers)}")
    try:
        return samplers[mode](**params)
    except TypeError as exc:
        raise ValueError(f"Bad timestep_sampling_params for mode {mode!r}: {exc}") from exc
