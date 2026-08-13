#!/usr/bin/env python
"""Generate video + synchronized stereo audio with MiniMax-H3.

    # text to video+audio
    python scripts/generate.py --prompt "a cat knocks a mug off a table" \
        --model-path /data/aviad/models/MiniMax-H3 --out cat.mp4

    # with a trained adapter, and the same seed without it, for an A/B
    python scripts/generate.py --prompt "..." --lora runs/character/checkpoint-0002000 \
        --ab --out character.mp4

    # image-to-video
    python scripts/generate.py --prompt "..." --image first_frame.png --out i2v.mp4

    # omni-reference (Ref2VA)
    python scripts/generate.py --prompt "..." --variant ref2va \
        --reference-image face.png --reference-audio voice.wav --out ref.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from h3_trainer import logger  # noqa: E402
from h3_trainer.constants import Geometry  # noqa: E402
from h3_trainer.inference import GenerationRequest, H3Pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/data/aviad/models/MiniMax-H3"))
    parser.add_argument("--variant", choices=("fl2va", "ref2va"), default="fl2va")
    parser.add_argument("--out", type=Path, default=Path("generation.mp4"))
    parser.add_argument(
        "--resolution-bucket",
        default="704x704x107",
        help="WIDTHxHEIGHTxFRAMES; width/height divisible by 32, frames of the form 17*n+5",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--placement",
        choices=("shard", "quantize", "bf16", "offload"),
        default="shard",
        help="shard: bf16, transformer blocks spread over the GPUs with the index-consuming heads "
        "pinned to one card (full quality; needs ~66GB total VRAM). quantize: single GPU, ~18GB NF4 "
        "or ~33GB int8 -- 4-bit noticeably degrades this model. bf16: one 80GB card. offload: host RAM.",
    )
    parser.add_argument(
        "--quantization",
        choices=("nf4-bnb", "int8-bnb", "int8-quanto", "fp8-quanto"),
        default="nf4-bnb",
        help="Used with --placement quantize. nf4-bnb is smallest and loads fastest; "
        "int8 variants are closer to bf16 output but need ~33GB.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora", type=Path, default=None, help="checkpoint directory or adapter.safetensors")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--ab",
        action="store_true",
        help="also generate the same seed with the adapter disabled, as <out>.base.mp4",
    )
    parser.add_argument("--image", type=Path, default=None, help="first frame (fl2va)")
    parser.add_argument("--last-image", type=Path, default=None, help="last frame (fl2va)")
    parser.add_argument("--reference-image", type=Path, action="append", default=[])
    parser.add_argument("--reference-video", type=Path, action="append", default=[])
    parser.add_argument(
        "--reference-canvas",
        choices=("native", "target"),
        default="native",
        help="must match what the adapter was trained with (process_dataset.py --reference-canvas)",
    )
    parser.add_argument("--reference-audio", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geometry = Geometry.parse(args.resolution_bucket).require_generatable()

    references = (
        [{"image": str(path)} for path in args.reference_image]
        + [{"video": str(path)} for path in args.reference_video]
        + [{"audio": str(path)} for path in args.reference_audio]
    )
    if references and args.variant != "ref2va":
        raise SystemExit("References require --variant ref2va (the FL2VA transformer has no reference rows)")

    request = GenerationRequest(
        prompt=args.prompt,
        geometry=geometry,
        seed=args.seed,
        num_inference_steps=args.steps,
        image=args.image,
        last_image=args.last_image,
        references=references,
        reference_canvas=args.reference_canvas,
    )

    pipeline = H3Pipeline(
        args.model_path,
        variant=args.variant,
        placement=args.placement,
        quantization=args.quantization,
        device=args.device,
    )

    if args.ab:
        # Baseline first, while the transformer is still untouched by any adapter.
        base_path = args.out.with_suffix(".base.mp4")
        pipeline.generate(request, base_path)
        logger.info("Baseline (no adapter): %s", base_path)

    if args.lora is not None:
        pipeline.load_lora(args.lora, scale=args.lora_scale)

    pipeline.generate(request, args.out)
    logger.info("Done: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
