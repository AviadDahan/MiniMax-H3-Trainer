"""Decoding video and audio into exactly the form H3's VAEs expect.

Three constraints drive everything here:

* **24.000 fps, exactly.** H3 is a 24 fps model and its rotary clock counts latent
  frames, not seconds. A 25 fps clip fed in unchanged is a clip in 4% slow motion,
  and that is precisely the kind of systematic bias a LoRA learns first.
* **Cover-fit framing, never stretch.** Aspect distortion is learnable, so we
  scale to cover and centre-crop instead of squashing.
* **32 kHz stereo audio on an 800-sample-per-latent grid.** Mono is duplicated,
  short tracks are padded, long ones truncated to the video's duration.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageOps

from h3_trainer import logger
from h3_trainer.constants import (
    AUDIO_SAMPLES_PER_LATENT,
    AUDIO_SAMPLE_RATE,
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
)


@dataclass
class DecodedClip:
    """One clip, decoded and framed. ``frames`` is ``(num_frames, H, W, 3)`` uint8."""

    frames: np.ndarray
    source_fps: float
    source_frames: int

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])


def probe_fps(path: str | Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        return float(rate) if rate else float(MINIMAX_H3_FPS)


def cover_fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale to cover the canvas, then centre-crop. Never distorts the aspect."""
    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    resized = image.resize(
        (math.ceil(source_width * scale), math.ceil(source_height * scale)), Image.Resampling.BICUBIC
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def resample_frame_indices(source_count: int, source_fps: float, target_count: int) -> list[int]:
    """Nearest-neighbour resample onto H3's 24 fps grid.

    Whole frames are dropped or duplicated -- the same thing ffmpeg's ``fps``
    filter does, and the same thing the reference inference path does for video
    references. Interpolating instead would invent motion the source never had.
    """
    if source_count <= 0:
        raise ValueError("Cannot resample an empty clip")
    step = source_fps / float(MINIMAX_H3_FPS)
    indices = [min(source_count - 1, int(round(index * step))) for index in range(target_count)]
    return indices


def decode_video(
    path: str | Path,
    num_frames: int,
    width: int,
    height: int,
    start_frame: int = 0,
) -> DecodedClip:
    """Decode ``num_frames`` frames at 24 fps, cover-fit to ``width`` x ``height``."""
    path = str(path)
    source_fps = probe_fps(path)
    # Read enough source frames to cover num_frames at 24 fps, plus a little slack
    # for containers whose reported rate is slightly off.
    needed_source = int(math.ceil(num_frames * source_fps / MINIMAX_H3_FPS)) + 2

    decoded: list[Image.Image] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index < start_frame:
                continue
            decoded.append(frame.to_image().convert("RGB"))
            if len(decoded) >= needed_source:
                break
    if not decoded:
        raise RuntimeError(f"No video frames decoded from {path}")

    indices = resample_frame_indices(len(decoded), source_fps, num_frames)
    frames = np.stack([np.asarray(cover_fit(decoded[i], width, height), dtype=np.uint8) for i in indices])
    return DecodedClip(frames=frames, source_fps=source_fps, source_frames=len(decoded))


def load_image(path: str | Path, width: int, height: int) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(str(path))).convert("RGB")
    return cover_fit(image, width, height)


def frames_to_pixel_tensor(frames: np.ndarray) -> torch.Tensor:
    """``(F, H, W, 3)`` uint8 -> ``(1, 3, F, H, W)`` float in [0, 1]."""
    tensor = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2)[None]
    return tensor.to(torch.float32).div(255.0)


def extract_audio(
    source: str | Path,
    num_frames: int,
    audio_path: str | Path | None = None,
) -> torch.Tensor | None:
    """Extract a clip's soundtrack as 32 kHz stereo, aligned to the video length.

    Returns ``None`` when there is no usable track -- the caller writes zero audio
    rows and the trainer zeroes the audio loss weight for that sample.
    """
    media = str(audio_path or source)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        # ffmpeg does the decode, the channel-fold and the resample in one pass,
        # and 16-bit PCM is pinned so the wav can be read with the standard
        # library -- torchaudio's loader now delegates to an optional codec
        # backend, which is a dependency this does not need.
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", media,
                "-vn", "-ac", str(MINIMAX_H3_AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE),
                "-c:a", "pcm_s16le", str(wav_path),
            ],
            capture_output=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 1024:
            logger.debug("No audio track in %s (%s)", media, result.stderr.decode()[:200])
            return None
        waveform = read_wav(wav_path)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(MINIMAX_H3_AUDIO_CHANNELS, 1)
    waveform = waveform[:MINIMAX_H3_AUDIO_CHANNELS]

    if waveform.abs().sum() == 0:
        logger.debug("Silent audio track in %s", media)
        return None
    return align_waveform(waveform, num_frames)


def read_wav(path: str | Path) -> torch.Tensor:
    """Read a 16-bit PCM wav into a ``(channels, samples)`` float tensor in [-1, 1]."""
    import wave

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM wav, got {sample_width * 8}-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(samples.reshape(-1, channels).T.copy())


def align_waveform(waveform: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Pad or truncate a stereo waveform onto the audio VAE's latent grid."""
    from h3_trainer.constants import audio_latent_num_frames

    target_samples = audio_latent_num_frames(num_frames) * AUDIO_SAMPLES_PER_LATENT
    if waveform.shape[1] < target_samples:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[1]))
    return waveform[:, :target_samples].contiguous()


def write_video_with_audio(
    frames: np.ndarray,
    path: str | Path,
    waveform: torch.Tensor | None = None,
    fps: float = float(MINIMAX_H3_FPS),
) -> Path:
    """Mux frames (and optionally stereo audio) into an mp4.

    Used by ``--decode`` verification and by validation logging: a video without
    its audio is only half of what H3 produces, and silently dropping the audio
    track is how audio regressions go unnoticed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width, stream.height = int(frames.shape[2]), int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}

        audio_stream = None
        if waveform is not None:
            audio_stream = container.add_stream("aac", rate=AUDIO_SAMPLE_RATE)
            audio_stream.layout = "stereo"

        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

        if audio_stream is not None:
            # Packed s16 wants the channels interleaved (L,R,L,R...). Flattening
            # the planar (2, N) tensor row-major instead concatenates the two
            # channels, which plays back at double speed and an octave high.
            samples = waveform.clamp(-1, 1).mul(32767).to(torch.int16).numpy()
            interleaved = samples.T.reshape(1, -1)
            audio_frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(interleaved), format="s16", layout="stereo"
            )
            audio_frame.sample_rate = AUDIO_SAMPLE_RATE
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
    return path
