#!/usr/bin/env python
"""A/B an adapter against its own base, on one pipeline load.

    python scripts/evaluate_lora.py --lora runs/character/checkpoint-0000600 \
        --prompts prompts.txt --out-dir runs/character/eval

For every prompt it generates twice from the same seed -- once with the adapter
disabled, once enabled -- and writes a contact sheet putting the two side by
side. That comparison is the only honest read on what an adapter did: judging a
generation on its own conflates the adapter with the prompt and the seed.

Both generations reuse one loaded transformer, so the ~5 minute load is paid once
rather than twice per prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from h3_trainer import logger  # noqa: E402
from h3_trainer.constants import Geometry  # noqa: E402
from h3_trainer.inference import GenerationRequest, H3Pipeline  # noqa: E402
from h3_trainer.preprocessing.media import decode_video, extract_audio  # noqa: E402


def contact_sheet(base: Path, adapted: Path, geometry: Geometry, out: Path, columns: int = 4) -> Path:
    """One row per variant, sampled evenly across the clip."""
    tile = 224
    rows = []
    for path in (base, adapted):
        frames = decode_video(path, geometry.num_frames, geometry.width, geometry.height).frames
        picks = np.linspace(0, len(frames) - 1, columns).astype(int)
        rows.append(
            np.concatenate(
                [np.asarray(Image.fromarray(frames[i]).resize((tile, tile))) for i in picks], axis=1
            )
        )
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    return out


def audio_summary(path: Path, num_frames: int) -> dict:
    waveform = extract_audio(path, num_frames)
    if waveform is None:
        return {"present": False}
    mono = waveform.mean(0).numpy()
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    frequencies = np.fft.rfftfreq(len(mono), 1 / 32000)
    voiced = spectrum[(frequencies > 80) & (frequencies < 400)].sum() / max(spectrum.sum(), 1e-9)
    return {
        "present": True,
        "rms": round(float(mono.std()), 5),
        "peak": round(float(np.abs(mono).max()), 4),
        # Energy in the fundamental range of human speech: a crude but useful
        # signal for "did it actually produce a voice".
        "voice_band_share": round(float(voiced), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/data/aviad/models/MiniMax-H3"))
    parser.add_argument("--variant", choices=("fl2va", "ref2va"), default="fl2va")
    parser.add_argument("--prompts", type=Path, default=None, help="one prompt per line")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--resolution-bucket", default="512x512x124")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--placement", choices=("shard", "quantize", "bf16", "offload"), default="shard")
    args = parser.parse_args()

    prompts = list(args.prompt)
    if args.prompts:
        prompts += [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    if not prompts:
        raise SystemExit("Give at least one --prompt or a --prompts file")

    geometry = Geometry.parse(args.resolution_bucket).require_generatable()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requests = [
        GenerationRequest(
            prompt=prompt, geometry=geometry, seed=args.seed + index, num_inference_steps=args.steps
        )
        for index, prompt in enumerate(prompts)
    ]

    pipeline = H3Pipeline(args.model_path, variant=args.variant, placement=args.placement)
    conditionings = pipeline.encode_conditioning_batch(requests)

    # Baseline first, while the transformer is still untouched.
    base_paths = []
    for index, (request, conditioning) in enumerate(zip(requests, conditionings)):
        base_paths.append(
            pipeline.generate_prepared(request, conditioning, args.out_dir / f"sample{index}_base.mp4")
        )

    pipeline.load_lora(args.lora, scale=args.lora_scale)
    report = []
    for index, (request, conditioning) in enumerate(zip(requests, conditionings)):
        adapted = pipeline.generate_prepared(
            request, conditioning, args.out_dir / f"sample{index}_lora.mp4"
        )
        sheet = contact_sheet(
            base_paths[index], adapted, geometry, args.out_dir / f"sample{index}_compare.png"
        )
        report.append(
            {
                "prompt": request.prompt,
                "seed": request.seed,
                "base": str(base_paths[index]),
                "lora": str(adapted),
                "compare": str(sheet),
                "audio_base": audio_summary(base_paths[index], geometry.num_frames),
                "audio_lora": audio_summary(adapted, geometry.num_frames),
            }
        )
        logger.info("Compared sample %d -> %s", index, sheet)

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1))
    logger.info("Wrote %s", args.out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
