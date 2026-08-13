#!/usr/bin/env python
"""Normalize raw footage into clips H3 can train on.

    python scripts/normalize_clips.py raw/ out/ --resolution-bucket 704x704x124

Every clip comes out at exactly 24.000 fps, cover-fitted to the bucket resolution
(scaled to cover, then centre-cropped -- never stretched), trimmed to the bucket's
frame count, with 32kHz stereo audio preserved.

Why each of those matters is in docs/dataset-preparation.md; the short version is
that a 25 fps clip left alone is a clip in 4% slow motion, aspect distortion is
learnable, and a silent track teaches silence.
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


def normalize(source: Path, destination: Path, geometry: Geometry, start: float, retime: float) -> bool:
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
        command += [
            "-filter_complex",
            f"[0:v]setpts={1 / retime}*PTS,{video_filter}[v];[0:a]atempo={retime}[a]",
            "-map", "[v]", "-map", "[a]?",
        ]
    else:
        command += ["-vf", video_filter]
    command += [
        "-t", f"{duration}",
        "-frames:v", str(geometry.num_frames),
        "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "2",
        str(destination),
    ]
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
        if not has_audio:
            logger.warning("%s has no audio track; it will train with the audio loss weighted to 0", source.name)

        destination = args.output_dir / f"{source.stem}.mp4"
        if normalize(source, destination, geometry, args.start, args.retime):
            rows.append({"id": source.stem, "video": str(destination), "caption": ""})
            logger.info("[%d/%d] %s -> %s", index, len(sources), source.name, destination.name)

    if args.manifest:
        args.manifest.write_text(json.dumps(rows, indent=1))
        logger.info("Wrote %s with %d rows -- fill in the captions before preprocessing", args.manifest, len(rows))

    logger.info("Normalized %d clips to %s at %s", len(rows), args.output_dir, geometry)
    for entry in skipped:
        logger.warning("skipped %s", entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
