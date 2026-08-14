#!/usr/bin/env python
"""Normalize raw footage into clips H3 can train on.

    python scripts/normalize_clips.py raw/ out/ --resolution-bucket 704x704x124

Every clip comes out at exactly 24.000 fps, cover-fitted to the bucket resolution
(scaled to cover, then centre-cropped -- never stretched), trimmed to the bucket's
frame count, with 32kHz stereo audio preserved.

Why each of those matters is in docs/dataset-preparation.md; the short version is
that a 25 fps clip left alone is a clip in 4% slow motion, aspect distortion is
learnable, and a silent track teaches silence.

Footage longer than the bucket yields one clip by default, from ``--start``. Pass
``--stride`` to walk the whole thing instead, which is what long source material
wants -- a 25 s dance is five clips at 124 frames, not one:

    python scripts/normalize_clips.py dances/ out/ \\
        --resolution-bucket 448x768x124 --stride 2.6

A stride shorter than the clip overlaps them. That is usually what you want from a
small pool of source videos: successive windows share frames but start on different
poses, so they are near-duplicates in appearance and distinct in motion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from h3_trainer import logger  # noqa: E402
from h3_trainer.constants import AUDIO_SAMPLE_RATE, MINIMAX_H3_FPS, Geometry  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr[:200]}")
    return json.loads(result.stdout)


def clip_offsets(duration: float, span: float, start: float, stride: float, max_clips: int = 0) -> list[float]:
    """Start times of every clip that fits in ``duration`` seconds of source.

    ``span`` is how much *source* one clip consumes, which differs from the clip's
    own duration whenever ``--retime`` is in play. A clip is only emitted if it
    fits whole -- a short tail is dropped rather than padded, since ffmpeg would
    otherwise write a clip with fewer frames than the bucket and preprocessing
    would reject it much later.
    """
    if stride <= 0:
        return [start]
    offsets, offset = [], start
    while offset + span <= duration:
        offsets.append(offset)
        offset += stride
        if max_clips and len(offsets) >= max_clips:
            break
    return offsets


def normalize(
    source: Path,
    destination: Path,
    geometry: Geometry,
    start: float,
    retime: float,
    drop_audio: bool = False,
) -> bool:
    duration = geometry.num_frames / MINIMAX_H3_FPS
    video_filter = (
        f"fps={MINIMAX_H3_FPS},"
        f"scale={geometry.width}:{geometry.height}:force_original_aspect_ratio=increase,"
        f"crop={geometry.width}:{geometry.height}"
    )
    command = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-i", str(source)]
    if retime != 1.0:
        # Retime video and audio together -- speeding up slow-motion footage
        # without also speeding up its soundtrack desynchronizes the pair, and
        # H3 trains them jointly.
        chains = [f"[0:v]setpts={1 / retime}*PTS,{video_filter}[v]"]
        maps = ["-map", "[v]"]
        if not drop_audio:
            chains.append(f"[0:a]atempo={retime}[a]")
            maps += ["-map", "[a]?"]
        command += ["-filter_complex", ";".join(chains), *maps]
    else:
        command += ["-vf", video_filter]
    command += [
        "-t", f"{duration}",
        "-frames:v", str(geometry.num_frames),
        "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
    ]
    if drop_audio:
        # Silent targets are a deliberate choice, not a shortcut: zero audio latents
        # are weighted to 0 rather than dropped (docs/h3-quirks.md #4), so the audio
        # branch stays in the graph and learns nothing. Use this when the soundtrack
        # is not yours to redistribute, or is unrelated to what the video shows.
        command += ["-an"]
    else:
        command += ["-c:a", "aac", "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "2"]
    command.append(str(destination))
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        logger.error("ffmpeg failed on %s: %s", source.name, result.stderr.decode()[:200])
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--resolution-bucket", default="704x704x124")
    parser.add_argument("--start", type=float, default=0.0, help="seconds to skip (trims fade-ins)")
    parser.add_argument(
        "--retime",
        type=float,
        default=1.0,
        help="speed factor for slow-motion footage, e.g. 2.0 for 50%% speed source. "
        "Audio is retimed with it.",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=0.0,
        help="seconds between clip starts; 0 (default) takes a single clip per source. "
        "Below the bucket duration the clips overlap.",
    )
    parser.add_argument("--max-clips", type=int, default=0, help="cap clips per source (0 = as many as fit)")
    parser.add_argument(
        "--drop-audio",
        action="store_true",
        help="write silent clips -- the audio loss is then weighted to 0 (docs/h3-quirks.md #4)",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="also write a dataset manifest stub")
    args = parser.parse_args()

    geometry = Geometry.parse(args.resolution_bucket)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in args.input_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not sources:
        raise SystemExit(f"No video files under {args.input_dir}")

    rows, skipped = [], []
    for index, source in enumerate(sources, 1):
        info = probe(source)
        duration = float(info.get("format", {}).get("duration", 0))
        needed = geometry.num_frames / MINIMAX_H3_FPS / max(args.retime, 1e-6) + args.start
        if duration < needed:
            skipped.append(f"{source.name}: {duration:.1f}s < {needed:.1f}s needed")
            continue
        has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        if not has_audio and not args.drop_audio:
            logger.warning("%s has no audio track; it will train with the audio loss weighted to 0", source.name)

        # How much of the *source* one clip consumes: retiming 2x slow-motion footage
        # means 124 output frames come from half as many seconds of input.
        span = geometry.num_frames / MINIMAX_H3_FPS / max(args.retime, 1e-6)
        offsets = clip_offsets(duration, span, args.start, args.stride, args.max_clips)

        for clip_index, offset in enumerate(offsets):
            # Single-clip runs keep the bare stem, so ids and manifests from before
            # the stride option still round-trip unchanged.
            stem = source.stem if args.stride <= 0 else f"{source.stem}_{clip_index:02d}"
            destination = args.output_dir / f"{stem}.mp4"
            if normalize(source, destination, geometry, offset, args.retime, args.drop_audio):
                rows.append({"id": stem, "video": str(destination), "caption": ""})
                logger.info(
                    "[%d/%d] %s @%.1fs -> %s", index, len(sources), source.name, offset, destination.name
                )

    if args.manifest:
        args.manifest.write_text(json.dumps(rows, indent=1))
        logger.info("Wrote %s with %d rows -- fill in the captions before preprocessing", args.manifest, len(rows))

    logger.info("Normalized %d clips to %s at %s", len(rows), args.output_dir, geometry)
    for entry in skipped:
        logger.warning("skipped %s", entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
