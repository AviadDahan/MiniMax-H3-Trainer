#!/usr/bin/env python
"""Render skeleton videos from human footage, for pose-conditioned IC-LoRA.

    python scripts/extract_pose.py raw/ out/ --resolution-bucket 512x512x124 \
        --manifest pose_dataset.json

H3 has no native structural conditioning — no pose, depth or edge input anywhere in the packed
sequence. IC-LoRA is how you add it: the skeleton video becomes an in-context *reference*, its latents
are packed ahead of the targets, and the adapter learns to follow it. This script produces the pairs.

For every input clip it writes:

* ``targets/<id>.mp4``    — the footage, normalized to the bucket (24.000 fps, cropped, audio kept)
* ``poses/<id>.mp4``      — a skeleton rendering, frame-aligned with the target
* a manifest row pairing them as ``video`` / ``reference_video``

Clips are **rejected** unless a full body is visible for most of their length. That is partly a
quality filter — a pose adapter trained on torso-only crops learns very little about bodies — and
partly a content one, since it drops framing that is about the body rather than the movement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from h3_trainer import logger  # noqa: E402
from h3_trainer.constants import Geometry  # noqa: E402
from h3_trainer.preprocessing.media import decode_video, write_video_with_audio  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

#: MediaPipe Pose landmark indices, and the bones drawn between them.
SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24),                      # torso
    (11, 13), (13, 15), (12, 14), (14, 16),                      # arms
    (23, 25), (25, 27), (24, 26), (26, 28),                      # legs
    (27, 31), (28, 32),                                          # feet
    (0, 11), (0, 12),                                            # neck
]
#: Distinct colour per bone, so the adapter can tell limbs apart -- the same reason
#: OpenPose renderings are colour-coded rather than monochrome.
BONE_COLOURS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
]
#: Landmarks that must be visible for a frame to count as full-body.
FULL_BODY = (11, 12, 23, 24, 27, 28)


def draw_line(canvas: np.ndarray, p0, p1, colour, thickness: int = 4) -> None:
    """Anti-aliasing-free line, drawn by sampling — avoids a hard OpenCV dependency."""
    (x0, y0), (x1, y1) = p0, p1
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    height, width = canvas.shape[:2]
    for t in np.linspace(0, 1, steps):
        x, y = int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))
        lo, hi = -(thickness // 2), thickness // 2 + 1
        canvas[max(0, y + lo) : min(height, y + hi), max(0, x + lo) : min(width, x + hi)] = colour


DEFAULT_POSE_MODEL = Path("/data/aviad/models/mediapipe/pose_landmarker_full.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def ensure_pose_model(path: Path = DEFAULT_POSE_MODEL) -> Path:
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Fetching the pose model to %s", path)
        urllib.request.urlretrieve(POSE_MODEL_URL, path)
    return path


def render_skeletons(frames: np.ndarray, min_visibility: float = 0.5, model_path: Path | None = None):
    """Frames -> (skeleton frames, fraction of frames with a full body visible).

    Uses MediaPipe's tasks API in VIDEO mode, which tracks across frames rather
    than detecting each one independently — the skeleton is much steadier, and a
    jittery reference teaches the adapter to generate jitter.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    height, width = frames.shape[1:3]
    out = np.zeros_like(frames)
    full_body = 0

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_pose_model(model_path or DEFAULT_POSE_MODEL))),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        for index, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect_for_video(image, int(index * 1000 / 24))
            if not result.pose_landmarks:
                continue
            landmarks = result.pose_landmarks[0]
            points = [(lm.x * width, lm.y * height, getattr(lm, "visibility", 1.0)) for lm in landmarks]

            if all(points[i][2] >= min_visibility for i in FULL_BODY):
                full_body += 1

            for bone, (a, b) in enumerate(SKELETON):
                if points[a][2] < min_visibility or points[b][2] < min_visibility:
                    continue
                draw_line(out[index], points[a][:2], points[b][:2], BONE_COLOURS[bone % len(BONE_COLOURS)])
            for a, b in SKELETON:
                for i in (a, b):
                    if points[i][2] >= min_visibility:
                        x, y = int(points[i][0]), int(points[i][1])
                        out[index][max(0, y - 3) : y + 4, max(0, x - 3) : x + 4] = (255, 255, 255)
    finally:
        # mediapipe 1.0's destructor can raise during interpreter teardown.
        try:
            landmarker.close()
        except Exception:
            pass

    return out, full_body / max(len(frames), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--resolution-bucket", default="512x512x124")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--min-full-body",
        type=float,
        default=0.7,
        help="reject a clip unless this fraction of frames shows a full body "
        "(hips, shoulders and ankles visible)",
    )
    parser.add_argument(
        "--caption",
        default="a person dancing, full body in frame, following the motion of the reference skeleton",
        help="caption for every pair; the reference carries the structure, the caption the content",
    )
    args = parser.parse_args()

    geometry = Geometry.parse(args.resolution_bucket)
    targets = args.output_dir / "targets"
    poses = args.output_dir / "poses"
    targets.mkdir(parents=True, exist_ok=True)
    poses.mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in args.input_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit(f"No videos under {args.input_dir}")

    rows, rejected = [], []
    for index, source in enumerate(sources, 1):
        sample_id = source.stem[:16]
        try:
            clip = decode_video(source, geometry.num_frames, geometry.width, geometry.height)
        except Exception as exc:
            rejected.append(f"{sample_id}: decode failed ({exc})")
            continue

        skeletons, full_body_fraction = render_skeletons(clip.frames)
        if full_body_fraction < args.min_full_body:
            rejected.append(f"{sample_id}: full body visible in only {full_body_fraction:.0%} of frames")
            logger.info("[%d/%d] %s rejected (%.0f%% full body)", index, len(sources), sample_id,
                        100 * full_body_fraction)
            continue

        write_video_with_audio(clip.frames, targets / f"{sample_id}.mp4")
        write_video_with_audio(skeletons, poses / f"{sample_id}.mp4")
        rows.append(
            {
                "id": sample_id,
                "video": str(targets / f"{sample_id}.mp4"),
                "reference_video": str(poses / f"{sample_id}.mp4"),
                "caption": args.caption,
            }
        )
        logger.info("[%d/%d] %s ok (%.0f%% full body)", index, len(sources), sample_id,
                    100 * full_body_fraction)

    manifest = args.manifest or args.output_dir / "dataset.json"
    manifest.write_text(json.dumps(rows, indent=1))
    (args.output_dir / "rejected.json").write_text(json.dumps(rejected, indent=1))
    logger.info("Kept %d pairs, rejected %d -> %s", len(rows), len(rejected), manifest)
    for entry in rejected[:10]:
        logger.info("  rejected %s", entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
