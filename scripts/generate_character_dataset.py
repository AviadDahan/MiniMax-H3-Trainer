#!/usr/bin/env python
"""Build a character AV dataset by generating it with H3 itself.

The recipe (adapted from the LTX-2.3 talking-head AV-LoRA methodology, run
entirely locally):

1. **Anchor.** One text-to-video+audio generation fixes the character: a frame
   becomes the identity reference, and the clip's own soundtrack becomes the
   voice reference. Everything downstream is conditioned on those two files, so
   appearance and voice come from the same source.
2. **Clips.** N generations conditioned on the anchor -- Ref2VA with the image
   and voice as in-context references, or FL2VA with the image as a first frame
   -- varying scene, framing, action and spoken line.
3. **Captions.** Written in the tagged AV form with a trigger token, so the
   adapter learns to attach both the look and the sound to that token.

Two things this deliberately does *not* do: it does not silently reuse one clip
many times (an adapter trained on near-duplicates memorises rather than
generalises), and it does not drop the audio track (H3 trains video and audio
jointly, so silent training clips teach silence).

    python scripts/generate_character_dataset.py --stage anchor --out-dir /data/aviad/datasets/character
    python scripts/generate_character_dataset.py --stage clips  --out-dir /data/aviad/datasets/character --count 40
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from h3_trainer import logger  # noqa: E402
from h3_trainer.constants import Geometry  # noqa: E402
from h3_trainer.inference import GenerationRequest, H3Pipeline  # noqa: E402
from h3_trainer.preprocessing.media import decode_video, extract_audio, read_wav  # noqa: E402

TRIGGER = "OHWXMIRA"

#: The character, described once. Every prompt reuses this clause so the
#: generator has no room to drift on the things that define the identity.
CHARACTER = (
    "a woman in her early thirties with shoulder-length dark curly hair, warm olive skin, "
    "thick eyebrows, small silver hoop earrings and a rust-orange ribbed knit sweater"
)

VOICE = "a warm, slightly husky mid-range voice with a calm, unhurried delivery"

ANCHOR_PROMPT = (
    f"A close-up portrait of {CHARACTER}, seated indoors facing the camera in soft window light. "
    f"She looks directly at the lens and speaks a single sentence in {VOICE}, saying: "
    f'"I think the light in here is perfect this time of day." Her expression is friendly and '
    f"relaxed, with small natural head movements. The audio is her clear speech in a quiet room "
    f"with faint ambience."
)

#: Scene variations. Framing, background, lighting and action all move; the
#: character clause does not. That contrast is what a character LoRA needs --
#: everything except the identity has to vary, or the adapter learns the room.
SCENES = [
    ("seated at a kitchen table with morning light from a window behind her", "a medium close-up", "cutlery clinking faintly and a kettle in the background"),
    ("standing in a bookshop aisle, shelves of paperbacks behind her", "a waist-up shot", "pages turning and quiet footsteps"),
    ("walking slowly along a tree-lined street on an overcast afternoon", "a tracking medium shot", "footsteps on pavement and distant traffic"),
    ("sitting on a park bench with green foliage behind her", "a close-up", "birdsong and a light breeze in the leaves"),
    ("in a small home studio with a microphone just out of frame", "a tight close-up", "a very quiet, treated room"),
    ("leaning against a brick wall in evening light, warm orange tones", "a medium shot", "distant city hum"),
    ("at a cafe table by a rain-streaked window", "a close-up over the shoulder", "rain on glass and a milk steamer"),
    ("in a bright hallway with white walls, turning to face the camera", "a medium close-up", "a quiet interior with faint echo"),
    ("seated in a car in the passenger seat, daylight through the windscreen", "a close-up", "muted road noise"),
    ("standing in a kitchen chopping herbs, glancing up at the camera", "a waist-up shot", "a knife on a board and a extractor fan"),
    ("on a balcony at dusk with string lights behind her", "a medium close-up", "distant conversation and traffic"),
    ("in an art gallery with a large painting behind her", "a medium shot", "a hushed room with soft footsteps"),
]

LINES = [
    "I keep meaning to write this down before I forget it.",
    "It took me a while, but I think I finally understand what you meant.",
    "Give me a second, I want to get this exactly right.",
    "Honestly? I would do the whole thing again.",
    "There is a version of this that works, I just have not found it yet.",
    "You were right about the first part, and wrong about everything after.",
    "I like it here in the mornings, before anyone else turns up.",
    "Let me try to explain it a different way.",
    "That is the part nobody tells you about.",
    "I am not worried. Not about this, anyway.",
    "We can start over from the beginning if you want.",
    "It sounded better in my head, I will admit that.",
]


def build_prompt(scene: tuple[str, str, str], line: str) -> tuple[str, str]:
    """Return (generation prompt, training caption).

    The generation prompt is plain description -- what H3 responds to. The
    caption is the tagged form with the trigger, which is what the adapter is
    trained against.
    """
    place, framing, ambience = scene
    prompt = (
        f"{framing.capitalize()} of {CHARACTER}, {place}. She speaks to the camera in {VOICE}, "
        f'saying: "{line}" Natural expression and small head movements. The audio is her clear '
        f"speech over {ambience}."
    )
    caption = (
        f"[VISUAL] {TRIGGER}, {framing} of a woman {place}. "
        f"[SPEECH] {TRIGGER} speaks in {VOICE}: \"{line}\" "
        f"[SOUNDS] {ambience}."
    )
    return prompt, caption


def pick_anchor_frame(frames: np.ndarray) -> int:
    """Choose the sharpest frame as the identity reference.

    Sharpness via a Laplacian-style gradient energy: a motion-blurred anchor
    propagates that blur into every clip conditioned on it.
    """
    scores = []
    for frame in frames:
        grey = frame.astype(np.float32).mean(axis=2)
        gx = np.diff(grey, axis=1)
        gy = np.diff(grey, axis=0)
        scores.append(float((gx**2).mean() + (gy**2).mean()))
    return int(np.argmax(scores))


def write_wav(waveform, path: Path) -> Path:
    import wave

    samples = (waveform.clamp(-1, 1) * 32767).to(dtype=__import__("torch").int16).numpy()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(32000)
        handle.writeframes(samples.T.tobytes())
    return path


def stage_anchor(args, pipeline: H3Pipeline, geometry: Geometry) -> None:
    out = Path(args.out_dir)
    (out / "anchor").mkdir(parents=True, exist_ok=True)
    clip_path = out / "anchor" / "anchor.mp4"

    request = GenerationRequest(
        prompt=ANCHOR_PROMPT,
        geometry=geometry,
        seed=args.seed,
        num_inference_steps=args.steps,
    )
    pipeline.generate(request, clip_path)

    frames = decode_video(clip_path, geometry.num_frames, geometry.width, geometry.height).frames
    index = pick_anchor_frame(frames)
    image_path = out / "anchor" / "identity.png"
    Image.fromarray(frames[index]).save(image_path)

    waveform = extract_audio(clip_path, geometry.num_frames)
    voice_path = None
    if waveform is not None:
        voice_path = write_wav(waveform, out / "anchor" / "voice.wav")

    logger.info("Anchor frame %d -> %s", index, image_path)
    logger.info("Voice reference -> %s", voice_path)
    (out / "anchor" / "anchor.json").write_text(
        json.dumps(
            {
                "trigger": TRIGGER,
                "character": CHARACTER,
                "voice": VOICE,
                "prompt": ANCHOR_PROMPT,
                "identity_image": str(image_path),
                "voice_audio": str(voice_path) if voice_path else None,
                "frame_index": index,
            },
            indent=1,
        )
    )


def stage_clips(args, pipeline: H3Pipeline, geometry: Geometry) -> None:
    out = Path(args.out_dir)
    anchor = json.loads((out / "anchor" / "anchor.json").read_text())
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = []
    manifest_path = out / "dataset.json"
    if manifest_path.exists():
        rows = json.loads(manifest_path.read_text())

    done = {row["id"] for row in rows}
    plans = []
    for index in range(args.count):
        sample_id = f"{TRIGGER.lower()}_{index:03d}"
        if sample_id in done:
            continue
        scene = SCENES[index % len(SCENES)]
        line = LINES[(index * 7 + 3) % len(LINES)]
        prompt, caption = build_prompt(scene, line)

        references, image = [], None
        if args.variant == "ref2va":
            references = [{"image": anchor["identity_image"]}]
            if anchor.get("voice_audio"):
                references.append({"audio": anchor["voice_audio"]})
        else:
            image = anchor["identity_image"]

        plans.append(
            (
                sample_id,
                caption,
                GenerationRequest(
                    prompt=prompt,
                    geometry=geometry,
                    seed=rng.randrange(1, 2**31),
                    num_inference_steps=args.steps,
                    image=image,
                    references=references,
                ),
            )
        )

    # Condition everything on one load of the 32B conditioner, then load the
    # denoiser once and sweep. Per-clip conditioning would spend far more time
    # loading weights than generating.
    logger.info("Conditioning %d prompts", len(plans))
    conditionings = pipeline.encode_conditioning_batch([plan[2] for plan in plans])

    for (sample_id, caption, request), conditioning in zip(plans, conditionings):
        clip_path = clips_dir / f"{sample_id}.mp4"
        try:
            pipeline.generate_prepared(request, conditioning, clip_path)
        except Exception as exc:
            logger.error("Clip %s failed: %s", sample_id, exc)
            continue
        rows.append({"id": sample_id, "video": str(clip_path), "caption": caption, "prompt": request.prompt})
        manifest_path.write_text(json.dumps(rows, indent=1))
        logger.info("[%d/%d] %s", len(rows), args.count, clip_path)

    logger.info("Wrote %s with %d clips", manifest_path, len(rows))


def stage_review(args) -> None:
    """Automatic QC: drop clips that are too dark, static or silent.

    Cheap filters only. They catch the failures that are objectively wrong; a
    human still has to look at the rest, which is what ``--review`` prints.
    """
    out = Path(args.out_dir)
    rows = json.loads((out / "dataset.json").read_text())
    geometry = Geometry.parse(args.resolution_bucket)
    kept, dropped = [], []

    for row in rows:
        frames = decode_video(row["video"], geometry.num_frames, geometry.width, geometry.height).frames
        brightness = float(frames.mean())
        motion = float(np.abs(np.diff(frames[::8].astype(np.float32), axis=0)).mean())
        waveform = extract_audio(row["video"], geometry.num_frames)
        loudness = 0.0 if waveform is None else float(waveform.abs().mean())

        reasons = []
        if brightness < 25:
            reasons.append(f"too dark ({brightness:.0f})")
        if motion < 1.0:
            reasons.append(f"near-static ({motion:.2f})")
        if loudness < 1e-4:
            reasons.append("silent")
        (dropped if reasons else kept).append({**row, "reasons": reasons})

    (out / "dataset.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "reasons"} for r in kept], indent=1))
    (out / "rejected.json").write_text(json.dumps(dropped, indent=1))
    logger.info("QC kept %d, dropped %d", len(kept), len(dropped))
    for row in dropped:
        logger.info("  dropped %s: %s", row["id"], ", ".join(row["reasons"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("anchor", "clips", "review"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/data/aviad/models/MiniMax-H3"))
    parser.add_argument("--variant", choices=("fl2va", "ref2va"), default="ref2va")
    parser.add_argument("--resolution-bucket", default="512x512x124")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--placement", choices=("shard", "quantize", "bf16", "offload"), default="shard")
    args = parser.parse_args()

    geometry = Geometry.parse(args.resolution_bucket).require_generatable()
    if args.stage == "review":
        stage_review(args)
        return 0

    pipeline = H3Pipeline(
        args.model_path,
        variant="fl2va" if args.stage == "anchor" else args.variant,
        placement=args.placement,
    )
    if args.stage == "anchor":
        stage_anchor(args, pipeline, geometry)
    else:
        stage_clips(args, pipeline, geometry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
