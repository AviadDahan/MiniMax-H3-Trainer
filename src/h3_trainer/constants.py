"""MiniMax-H3 geometry constants and the derived quantities the trainer needs.

Everything here is re-exported from the diffusers H3 integration rather than
re-derived, so a change upstream cannot silently desynchronize training from
inference. The one thing this module adds is *validation*: H3 rejects geometry
in ways that are cheap to check up front and expensive to discover after an
hour of VAE encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

from h3_trainer import logger

try:
    from diffusers.modular_pipelines.minimax_h3.packing import (
        MINIMAX_H3_AUDIO_CHANNELS,
        MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
        MINIMAX_H3_AUDIO_TAG,
        MINIMAX_H3_CANVAS_MULTIPLE,
        MINIMAX_H3_FPS,
        MINIMAX_H3_FRAMES_PER_CHUNK,
        MINIMAX_H3_KEYFRAME_ENCODE_SEED,
        MINIMAX_H3_KEYFRAME_NOISE_AUG,
        MINIMAX_H3_LATENTS_PER_CHUNK,
        MINIMAX_H3_MAX_DURATION,
        MINIMAX_H3_MAX_PIXELS,
        MINIMAX_H3_MIN_DURATION,
        MINIMAX_H3_PIXEL_MEAN,
        MINIMAX_H3_PIXEL_STD,
        MINIMAX_H3_SHORT_EDGE,
        MINIMAX_H3_TEXT_ENCODER_LAYER,
        MINIMAX_H3_TEXT_TAG,
        MINIMAX_H3_VIDEO_TAG,
        align_num_frames,
        audio_latent_num_frames,
        resolve_canvas_size,
        video_latent_num_frames,
    )
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "The MiniMax-H3 classes are not in any released diffusers wheel. Install the pinned "
        "integration commit:\n"
        "    pip install --no-deps "
        "'git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc'\n"
        "or just run scripts/install_env.sh."
    ) from exc

#: Video VAE spatial compression (16x) times the transformer patch (2x) = 32x.
VAE_SPATIAL_COMPRESSION = 16
#: Transformer patch over the video latent grid: (t, h, w).
PATCH_SIZE = (1, 2, 2)
#: Effective pixel-space multiple that height/width must satisfy.
SPATIAL_MULTIPLE = MINIMAX_H3_CANVAS_MULTIPLE  # 32
#: Waveform samples per audio latent at 32 kHz on the 40 Hz latent grid.
AUDIO_SAMPLES_PER_LATENT = 800
#: Audio VAE sample rate.
AUDIO_SAMPLE_RATE = AUDIO_SAMPLES_PER_LATENT * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND  # 32000

#: Timestep pinned onto conditioning rows (keyframes / references). H3 works in
#: t = 1 - sigma, so 0.999 means "almost clean, with a whisper of noise aug".
CONDITION_TIMESTEP = MINIMAX_H3_KEYFRAME_NOISE_AUG

#: Seed for the posterior sample taken when encoding conditioning media. Fixed
#: upstream, not a request seed -- re-exported rather than restated so a change
#: there cannot leave training conditioning on latents inference never sees.
CONDITION_ENCODE_SEED = MINIMAX_H3_KEYFRAME_ENCODE_SEED

#: Default shifts of the two rectified-flow schedules. Video and audio are noised
#: at *different* sigmas drawn from the same uniform u -- see flow_matching.py.
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0

__all__ = [
    "AUDIO_SAMPLES_PER_LATENT",
    "AUDIO_SAMPLE_RATE",
    "CONDITION_ENCODE_SEED",
    "CONDITION_TIMESTEP",
    "DEFAULT_AUDIO_SHIFT",
    "DEFAULT_VIDEO_SHIFT",
    "MINIMAX_H3_AUDIO_CHANNELS",
    "MINIMAX_H3_AUDIO_LATENTS_PER_SECOND",
    "MINIMAX_H3_AUDIO_TAG",
    "MINIMAX_H3_FPS",
    "MINIMAX_H3_FRAMES_PER_CHUNK",
    "MINIMAX_H3_LATENTS_PER_CHUNK",
    "MINIMAX_H3_MAX_DURATION",
    "MINIMAX_H3_MAX_PIXELS",
    "MINIMAX_H3_MIN_DURATION",
    "MINIMAX_H3_PIXEL_MEAN",
    "MINIMAX_H3_PIXEL_STD",
    "MINIMAX_H3_SHORT_EDGE",
    "MINIMAX_H3_TEXT_ENCODER_LAYER",
    "MINIMAX_H3_TEXT_TAG",
    "MINIMAX_H3_VIDEO_TAG",
    "PATCH_SIZE",
    "SPATIAL_MULTIPLE",
    "VAE_SPATIAL_COMPRESSION",
    "Geometry",
    "align_num_frames",
    "audio_latent_num_frames",
    "resolve_canvas_size",
    "generation_frame_counts",
    "valid_frame_counts",
    "video_latent_num_frames",
]


def generation_frame_counts() -> list[int]:
    """The frame counts H3 will actually generate: 17n+5 inside the 5-15s window."""
    low = MINIMAX_H3_MIN_DURATION * MINIMAX_H3_FPS
    high = MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS
    return [f for f in valid_frame_counts(int(high)) if low <= f <= high]


def valid_frame_counts(max_frames: int = 400) -> list[int]:
    """The frame counts H3's video VAE accepts: 17*n + 5 (22, 39, 56, 73, ...)."""
    counts, n = [], 1
    while True:
        frames = MINIMAX_H3_FRAMES_PER_CHUNK * n + MINIMAX_H3_LATENTS_PER_CHUNK
        if frames > max_frames:
            return counts
        counts.append(frames)
        n += 1


@dataclass(frozen=True)
class Geometry:
    """One resolution bucket, with every derived row count H3 needs.

    Construct with :meth:`create` so the H3 constraints are checked once, loudly,
    instead of surfacing as a shape error deep inside the transformer.
    """

    width: int
    height: int
    num_frames: int

    @classmethod
    def create(cls, width: int, height: int, num_frames: int) -> Geometry:
        errors = []
        if width % SPATIAL_MULTIPLE or height % SPATIAL_MULTIPLE:
            errors.append(
                f"width and height must be divisible by {SPATIAL_MULTIPLE} (got {width}x{height}); "
                f"the VAE compresses 16x and the transformer patches another 2x"
            )
        if num_frames != align_num_frames(num_frames):
            nearby = [f for f in valid_frame_counts(num_frames + 40) if abs(f - num_frames) <= 40]
            errors.append(
                f"num_frames must be of the form 17*n + 5 (got {num_frames}); nearest valid: {nearby}"
            )
        duration = num_frames / MINIMAX_H3_FPS
        if duration > MINIMAX_H3_MAX_DURATION:
            errors.append(
                f"{num_frames} frames is {duration:.1f}s at {MINIMAX_H3_FPS} fps, beyond H3's "
                f"{MINIMAX_H3_MAX_DURATION}s maximum"
            )
        if errors:
            raise ValueError("Invalid MiniMax-H3 geometry:\n  - " + "\n  - ".join(errors))
        geometry = cls(width=width, height=height, num_frames=num_frames)
        if duration < MINIMAX_H3_MIN_DURATION:
            # Training packs any 17n+5 length happily, and short clips are useful
            # for smoke tests -- but H3 only ever generates 5-15s, so a fine-tune
            # on sub-5s clips is training out of the model's own distribution.
            logger.warning(
                "%s is %.2fs; H3 generates %.0f-%.0fs, so training on clips this short is "
                "out of distribution (fine for smoke tests). Nearest in-range frame counts: %s",
                geometry,
                duration,
                MINIMAX_H3_MIN_DURATION,
                MINIMAX_H3_MAX_DURATION,
                generation_frame_counts()[:3],
            )
        return geometry

    def require_generatable(self) -> Geometry:
        """Raise unless this geometry is one H3 can actually generate.

        The inference pipeline rejects anything outside 5-15 seconds. Checking it
        here means a validation config fails at load rather than 40 minutes into a
        training run.
        """
        if not MINIMAX_H3_MIN_DURATION <= self.duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"{self} is {self.duration:.2f}s; H3 generates only "
                f"{MINIMAX_H3_MIN_DURATION:.0f}-{MINIMAX_H3_MAX_DURATION:.0f}s. "
                f"Valid frame counts: {generation_frame_counts()}"
            )
        return self

    @property
    def duration(self) -> float:
        return self.num_frames / MINIMAX_H3_FPS

    @property
    def latent_height(self) -> int:
        return self.height // VAE_SPATIAL_COMPRESSION

    @property
    def latent_width(self) -> int:
        return self.width // VAE_SPATIAL_COMPRESSION

    @property
    def latent_frames(self) -> int:
        return video_latent_num_frames(self.num_frames)

    @property
    def audio_latents(self) -> int:
        """Audio latents per channel on the 40 Hz grid."""
        return audio_latent_num_frames(self.num_frames)

    @property
    def video_rows(self) -> int:
        _, ph, pw = PATCH_SIZE
        return self.latent_frames * (self.latent_height // ph) * (self.latent_width // pw)

    @property
    def audio_rows(self) -> int:
        """Audio rows are channel-major: [ch0 x N, ch1 x N]."""
        return self.audio_latents * MINIMAX_H3_AUDIO_CHANNELS

    def sequence_length(self, text_rows: int, condition_rows: int = 0) -> int:
        return text_rows + condition_rows + self.audio_rows + self.video_rows

    def __str__(self) -> str:
        return f"{self.width}x{self.height}x{self.num_frames}"

    @classmethod
    def parse(cls, spec: str) -> Geometry:
        """Parse a ``WIDTHxHEIGHTxFRAMES`` bucket string, e.g. ``704x704x107``."""
        parts = spec.lower().replace(" ", "").split("x")
        if len(parts) != 3:
            raise ValueError(f"Resolution bucket must look like 'WIDTHxHEIGHTxFRAMES', got {spec!r}")
        try:
            width, height, frames = (int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Resolution bucket must be three integers, got {spec!r}") from exc
        return cls.create(width=width, height=height, num_frames=frames)
