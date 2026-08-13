#!/usr/bin/env python
"""Encode a dataset into MiniMax-H3 latents, once, before training.

    python scripts/process_dataset.py dataset.json \
        --resolution-bucket 704x704x107 \
        --model-path /data/aviad/models/MiniMax-H3

Multi-GPU:

    accelerate launch --num_processes 8 scripts/process_dataset.py dataset.json ...

Manifest columns (aliases in brackets): ``video`` [target_video, media_path],
``caption`` [prompt], ``audio``, ``first_frame``, ``last_frame``,
``reference_video`` [ref_media_path], ``reference_image``, ``reference_audio``,
``id``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must precede the torch import (see FIXES.md #8 / scripts/env.sh).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from h3_trainer import logger  # noqa: E402
from h3_trainer.constants import Geometry  # noqa: E402
from h3_trainer.preprocessing.builder import (  # noqa: E402
    MediaPass,
    ProcessOptions,
    TextPass,
    read_manifest,
    verify_cache,
    write_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="dataset .json / .jsonl / .csv")
    parser.add_argument("--model-path", type=Path, required=True, help="MiniMax-H3 diffusers directory")
    parser.add_argument(
        "--resolution-bucket",
        default="704x704x107",
        help="WIDTHxHEIGHTxFRAMES. Width/height divisible by 32, frames of the form 17*n+5.",
    )
    parser.add_argument("--output", type=Path, default=None, help="default: <manifest dir>/.precomputed")
    parser.add_argument("--skip-audio", action="store_true", help="do not encode soundtracks")
    parser.add_argument("--lora-trigger", default=None, help="prepend a trigger phrase to every caption")
    parser.add_argument(
        "--keyframes",
        default="",
        help="comma-separated keyframe conditioning to encode: first_frame,last_frame",
    )
    parser.add_argument("--references", action="store_true", help="encode IC-LoRA reference media")
    parser.add_argument("--decode", type=int, default=0, metavar="N", help="VAE round-trip N samples to mp4")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--only",
        choices=("media", "text"),
        default=None,
        help="run a single pass. media = both VAEs (shardable across GPUs); "
        "text = the 32B conditioner (loaded once, spread over the visible GPUs).",
    )
    parser.add_argument(
        "--text-device-map",
        default="auto",
        help="device_map for the conditioner: 'auto' spreads it over the visible GPUs, "
        "'none' forces a single device (needs an 80GB+ card).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geometry = Geometry.parse(args.resolution_bucket)
    output = args.output or args.manifest.parent / ".precomputed"
    output.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = args.device if args.device == "cpu" else f"cuda:{local_rank}"

    manifest = read_manifest(args.manifest)
    logger.info(
        "%s -> %s | %s (%.2fs, %d video rows, %d audio rows per sample)",
        args.manifest,
        output,
        geometry,
        geometry.duration,
        geometry.video_rows,
        geometry.audio_rows,
    )

    if args.limit:
        manifest = manifest[: args.limit]

    options = ProcessOptions(
        model_path=args.model_path,
        output=output,
        geometry=geometry,
        encode_audio=not args.skip_audio,
        lora_trigger=args.lora_trigger,
        keyframes=tuple(k.strip() for k in args.keyframes.split(",") if k.strip()),
        references=args.references,
        decode_check=args.decode,
        overwrite=args.overwrite,
        device=device,
        text_device_map=None if args.text_device_map == "none" else args.text_device_map,
        limit=args.limit,
        rank=rank,
        world_size=world_size,
    )

    records: list = []
    if args.only in (None, "media"):
        media = MediaPass(options)
        records = media.run(manifest)
        media.unload()
        records = _gather(records, rank, world_size)
        if rank == 0:
            write_index(output, records, geometry)

    if args.only in (None, "text"):
        if not records:
            records = _load_records(output)
        # The conditioner is far too large to replicate per rank; rank 0 runs it
        # for the whole dataset while the others wait.
        if rank == 0:
            text = TextPass(options)
            records = text.run(manifest, records)
            text.unload()
        _barrier(world_size)

    if rank == 0:
        index_path = write_index(output, records, geometry)
        report = verify_cache(output)
        logger.info("Wrote %s with %d samples", index_path, report["num_samples"])
        for problem in report["problems"]:
            logger.error("cache problem: %s", problem)
        if options.dropped:
            logger.warning("%d samples were dropped:", len(options.dropped))
            for entry in options.dropped[:20]:
                logger.warning("  %s", entry)
    return 0


def _gather(records: list, rank: int, world_size: int) -> list:
    if world_size <= 1:
        return records
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, records)
    dist.barrier()
    return [record for shard in gathered for record in shard]


def _barrier(world_size: int) -> None:
    if world_size <= 1:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def _load_records(output: Path) -> list:
    import json

    index_path = output / "index.json"
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found -- run the media pass first (--only media)")
    with index_path.open() as handle:
        return json.load(handle)["samples"]


if __name__ == "__main__":
    raise SystemExit(main())
