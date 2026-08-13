"""Turning a dataset manifest into the precomputed latent cache.

The cache is written once and read every step, so everything expensive -- video
decode, both VAEs, the 32B conditioner -- happens here and never during training.
That is what leaves the 33B transformer as the only large thing on the GPU.

**Two passes, on purpose.** H3's conditioner is Qwen3-VL-32B: ~64GB in bf16, more
than a single 48GB card holds, and far too much to replicate once per data-parallel
worker. So the work splits:

* the **media** pass runs the two VAEs (small, ~11GB together) and shards cleanly
  across GPUs, one process per rank;
* the **text** pass loads the conditioner once, spread across whatever GPUs are
  available, and encodes every caption.

Layout::

    <output>/
        index.json                    one record per sample
        latents/<id>.safetensors      target video rows
        audio_latents/<id>...         target audio rows
        conditions/<id>...            prompt embeddings + per-row modality tags
        first_frame_latents/<id>...   keyframe rows        (when configured)
        reference_latents/<id>...     in-context reference rows + geometry
"""

from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from h3_trainer import logger
from h3_trainer.constants import Geometry
from h3_trainer.preprocessing.encoders import H3Encoders
from h3_trainer.preprocessing.media import (
    decode_video,
    extract_audio,
    load_image,
    write_video_with_audio,
)

#: Manifest column aliases, canonical name first.
COLUMN_ALIASES = {
    "video": ("video", "target_video", "media_path", "file"),
    "caption": ("caption", "prompt", "text"),
    "audio": ("audio", "target_audio"),
    "first_frame": ("first_frame", "first_frame_image"),
    "last_frame": ("last_frame", "last_frame_image"),
    "reference_video": ("reference_video", "reference_videos", "ref_media_path"),
    "reference_image": ("reference_image", "reference_images"),
    "reference_audio": ("reference_audio", "reference_audios"),
    "id": ("id", "sample_id"),
}

REFERENCE_COLUMNS = (("image", "reference_image"), ("video", "reference_video"), ("audio", "reference_audio"))


def _pick(row: dict[str, Any], field_name: str) -> Any:
    for alias in COLUMN_ALIASES[field_name]:
        if alias in row and row[alias] not in (None, "", []):
            return row[alias]
    return None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def sample_id_for(row: dict[str, Any], fallback: str) -> str:
    explicit = _pick(row, "id")
    if explicit:
        return str(explicit)
    video = _pick(row, "video")
    return Path(str(video)).stem if video else fallback


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a .json (list), .jsonl (one object per line) or .csv manifest."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("samples", payload.get("data"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a list of samples")
        return payload
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported manifest format {path.suffix!r} (use .json, .jsonl or .csv)")


@dataclass
class ProcessOptions:
    model_path: Path
    output: Path
    geometry: Geometry
    encode_audio: bool = True
    lora_trigger: str | None = None
    keyframes: tuple[str, ...] = ()
    references: bool = False
    decode_check: int = 0
    overwrite: bool = False
    device: str = "cuda"
    text_device_map: str | None = "auto"
    limit: int = 0
    rank: int = 0
    world_size: int = 1
    dropped: list[str] = field(default_factory=list)

    def caption_for(self, row: dict[str, Any]) -> str:
        caption = str(_pick(row, "caption") or "")
        if self.lora_trigger:
            caption = f"{self.lora_trigger} {caption}".strip()
        return caption


class MediaPass:
    """VAE encoding: target video, target audio, keyframes, reference media."""

    def __init__(self, options: ProcessOptions) -> None:
        self.options = options
        self.encoders = H3Encoders(
            options.model_path,
            device=options.device,
            need_video=True,
            need_audio=True,
            need_text=False,
        )

    def _dir(self, name: str) -> Path:
        path = self.options.output / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(self, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
        options = self.options
        shard = manifest[options.rank :: options.world_size]
        logger.info(
            "Media pass: %d of %d samples on rank %d/%d at %s",
            len(shard),
            len(manifest),
            options.rank,
            options.world_size,
            options.geometry,
        )
        records = []
        for index, row in enumerate(shard):
            sample_id = sample_id_for(row, f"sample{index:06d}")
            target = self.options.output / "latents" / f"{sample_id}.safetensors"
            if target.exists() and not options.overwrite:
                logger.info("[%d/%d] %s cached", index + 1, len(shard), sample_id)
                record = _read_indexed_record(options.output, sample_id)
                if record is not None:
                    records.append(record)
                    continue
            try:
                records.append(self.process(sample_id, row))
                logger.info("[%d/%d] %s encoded", index + 1, len(shard), sample_id)
            except Exception as exc:
                logger.error("[%d/%d] %s failed: %s", index + 1, len(shard), sample_id, exc)
                logger.debug(traceback.format_exc())
                options.dropped.append(f"{sample_id}: {exc}")
        return records

    def process(self, sample_id: str, row: dict[str, Any]) -> dict[str, Any]:
        options = self.options
        geometry = options.geometry
        video_path = _pick(row, "video")
        if video_path is None:
            raise ValueError("manifest row has no video column")

        clip = decode_video(video_path, geometry.num_frames, geometry.width, geometry.height)
        video_rows = self.encoders.encode_video_target(clip.frames)
        save_file(
            {"latents": video_rows},
            str(self._dir("latents") / f"{sample_id}.safetensors"),
            metadata={"geometry": str(geometry), "source_fps": f"{clip.source_fps:.3f}"},
        )

        audio_rows = None
        if options.encode_audio:
            waveform = extract_audio(video_path, geometry.num_frames, _pick(row, "audio"))
            if waveform is not None:
                audio_rows = self.encoders.encode_audio(waveform)
                save_file(
                    {"latents": audio_rows}, str(self._dir("audio_latents") / f"{sample_id}.safetensors")
                )
            else:
                logger.warning(
                    "%s has no usable audio track; it trains with the audio loss weighted to 0", sample_id
                )

        self._encode_keyframes(sample_id, row, clip.frames)
        reference_geometry = self._encode_references(sample_id, row)

        if options.decode_check > 0:
            self._decode_check(sample_id, video_rows, audio_rows, geometry)
            options.decode_check -= 1

        return {
            "id": sample_id,
            "width": geometry.width,
            "height": geometry.height,
            "num_frames": geometry.num_frames,
            "video_rows": int(video_rows.shape[0]),
            "audio_rows": int(audio_rows.shape[0]) if audio_rows is not None else geometry.audio_rows,
            "text_rows": 0,  # filled in by the text pass
            "has_audio": audio_rows is not None,
            "caption": options.caption_for(row),
            "source": str(video_path),
            "source_fps": round(clip.source_fps, 3),
            "reference_geometry": reference_geometry,
        }

    def _encode_keyframes(self, sample_id: str, row: dict[str, Any], frames: np.ndarray) -> None:
        geometry = self.options.geometry
        for name, frame_index in (("first_frame", 0), ("last_frame", -1)):
            if name not in self.options.keyframes:
                continue
            explicit = _pick(row, name)
            if explicit:
                array = np.asarray(load_image(explicit, geometry.width, geometry.height), dtype=np.uint8)[None]
            else:
                array = frames[frame_index][None]
            rows, _ = self.encoders.encode_visual_condition(array, is_image=True)
            save_file({"latents": rows}, str(self._dir(f"{name}_latents") / f"{sample_id}.safetensors"))

    def _encode_references(self, sample_id: str, row: dict[str, Any]) -> list[list[int]]:
        """Encode in-context references; returns their geometry for the text pass."""
        if not self.options.references:
            return []
        from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (
            prepare_reference_image,
            resolve_reference_image_size,
        )

        geometry = self.options.geometry
        video_chunks, audio_chunks, geometries = [], [], []

        for kind, column in REFERENCE_COLUMNS:
            for media in _as_list(_pick(row, column)):
                if kind == "image":
                    original = Image.open(str(media)).convert("RGB")
                    width, height = resolve_reference_image_size(original.width, original.height)
                    encoded = self.encoders.encode_reference(
                        "image", image=prepare_reference_image(original, height, width)
                    )
                elif kind == "video":
                    clip = decode_video(media, geometry.num_frames, geometry.width, geometry.height)
                    encoded = self.encoders.encode_reference(
                        "video",
                        frames=clip.frames,
                        waveform=extract_audio(media, geometry.num_frames),
                    )
                else:
                    waveform = extract_audio(media, geometry.num_frames)
                    if waveform is None:
                        raise ValueError(f"Reference audio {media} has no usable track")
                    encoded = self.encoders.encode_reference("audio", waveform=waveform)

                if encoded.video_rows is not None:
                    video_chunks.append(encoded.video_rows)
                if encoded.audio_rows is not None:
                    audio_chunks.append(encoded.audio_rows)
                geometries.append(encoded.geometry)

        if not geometries:
            return []
        payload: dict[str, torch.Tensor] = {"geometry": torch.stack(geometries)}
        if video_chunks:
            payload["latents"] = torch.cat(video_chunks)
        if audio_chunks:
            payload["audio_latents"] = torch.cat(audio_chunks)
        save_file(payload, str(self._dir("reference_latents") / f"{sample_id}.safetensors"))
        return [[int(v) for v in tensor.tolist()] for tensor in geometries]

    def _decode_check(
        self, sample_id: str, video_rows: torch.Tensor, audio_rows: torch.Tensor | None, geometry: Geometry
    ) -> None:
        """VAE round-trip one sample back to an mp4.

        Catches normalization, channel-order and framing mistakes in seconds --
        mistakes that otherwise surface only as a fine-tune that trains smoothly
        and generates garbage.
        """
        frames = self.encoders.decode_video(
            video_rows, geometry.latent_frames, geometry.latent_height, geometry.latent_width
        )
        waveform = self.encoders.decode_audio(audio_rows, geometry.audio_latents) if audio_rows is not None else None
        path = write_video_with_audio(frames, self._dir("decoded_videos") / f"{sample_id}.mp4", waveform=waveform)
        logger.info("Wrote decode check to %s", path)

    def unload(self) -> None:
        self.encoders.unload()


class TextPass:
    """Caption encoding with the Qwen3-VL conditioner, loaded exactly once."""

    def __init__(self, options: ProcessOptions) -> None:
        self.options = options
        self.encoders = H3Encoders(
            options.model_path,
            device=options.device,
            need_video=False,
            need_audio=False,
            need_text=True,
            text_device_map=options.text_device_map,
        )

    def run(self, manifest: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        options = self.options
        by_id = {record["id"]: record for record in records}
        rows_by_id = {sample_id_for(row, f"sample{i:06d}"): row for i, row in enumerate(manifest)}
        output_dir = options.output / "conditions"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Text pass: encoding %d captions", len(by_id))
        for index, (sample_id, record) in enumerate(sorted(by_id.items())):
            path = output_dir / f"{sample_id}.safetensors"
            if path.exists() and not options.overwrite and record.get("text_rows"):
                continue
            row = rows_by_id.get(sample_id, {})
            try:
                embeds, tags = self._encode(record, row)
            except Exception as exc:
                logger.error("[%d/%d] %s caption failed: %s", index + 1, len(by_id), sample_id, exc)
                logger.debug(traceback.format_exc())
                options.dropped.append(f"{sample_id} (caption): {exc}")
                continue
            save_file(
                {"prompt_embeds": embeds, "text_token_tags": tags},
                str(path),
                metadata={"caption": record.get("caption", "")},
            )
            record["text_rows"] = int(embeds.shape[0])
            logger.info("[%d/%d] %s -> %d text rows", index + 1, len(by_id), sample_id, record["text_rows"])
        return list(by_id.values())

    def _encode(self, record: dict[str, Any], row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        caption = record.get("caption", "")
        if self.options.references and record.get("reference_geometry"):
            return self.encoders.encode_ref2va_prompt(
                caption, self._rebuild_reference_presentation(record, row)
            )
        keyframes = self._rebuild_keyframes(record, row)
        return self.encoders.encode_prompt(caption, keyframes or None)

    def _rebuild_keyframes(self, record: dict[str, Any], row: dict[str, Any]) -> list[Image.Image]:
        """Recover the keyframe images the conditioner sees.

        The prompt presentation must match the one the packed layout was built
        for: each keyframe contributes a ``"<Picture i>: "`` label and a vision
        block whose rows are tagged as video.
        """
        if not self.options.keyframes:
            return []
        geometry = self.options.geometry
        images: list[Image.Image] = []
        clip = None
        for name, frame_index in (("first_frame", 0), ("last_frame", -1)):
            if name not in self.options.keyframes:
                continue
            explicit = _pick(row, name)
            if explicit:
                images.append(load_image(explicit, geometry.width, geometry.height))
                continue
            if clip is None:
                clip = decode_video(record["source"], geometry.num_frames, geometry.width, geometry.height)
            images.append(Image.fromarray(clip.frames[frame_index]))
        return images

    def _rebuild_reference_presentation(self, record: dict[str, Any], row: dict[str, Any]) -> list:
        """Rebuild the reference descriptors the conditioner needs.

        The latents were encoded in the media pass, and their geometry travels in
        the index. What the *conditioner* needs is the pixels -- an image, or a
        video's frames -- which are cheap to recover without touching a VAE. The
        media has to be attached to each descriptor because images and videos take
        different preprocessing paths.
        """
        from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (
            prepare_reference_image,
            resolve_reference_image_size,
        )

        from h3_trainer.packing import prepared_references_from_cache

        geometry = self.options.geometry
        geometries = [torch.tensor(entry, dtype=torch.int64) for entry in record["reference_geometry"]]
        references = prepared_references_from_cache([("image", None, tensor) for tensor in geometries])

        cursor = 0
        for kind, column in REFERENCE_COLUMNS:
            for media in _as_list(_pick(row, column)):
                reference = references[cursor]
                cursor += 1
                if kind == "image":
                    original = Image.open(str(media)).convert("RGB")
                    width, height = resolve_reference_image_size(original.width, original.height)
                    reference.image = prepare_reference_image(original, height, width)
                elif kind == "video":
                    clip = decode_video(media, geometry.num_frames, geometry.width, geometry.height)
                    reference.frames = np.asarray(clip.frames, dtype=np.uint8)
        return references

    def unload(self) -> None:
        self.encoders.unload()


def _read_indexed_record(output: Path, sample_id: str) -> dict[str, Any] | None:
    index_path = output / "index.json"
    if not index_path.exists():
        return None
    with index_path.open() as handle:
        for entry in json.load(handle).get("samples", []):
            if entry.get("id") == sample_id:
                return entry
    return None


def write_index(output: Path, records: list[dict[str, Any]], geometry: Geometry) -> Path:
    """Merge records into ``index.json`` (idempotent across reruns and ranks)."""
    index_path = output / "index.json"
    existing: dict[str, dict[str, Any]] = {}
    if index_path.exists():
        with index_path.open() as handle:
            for entry in json.load(handle).get("samples", []):
                existing[entry["id"]] = entry
    for record in records:
        merged = dict(existing.get(record["id"], {}))
        for key, value in record.items():
            # The two passes fill in different fields; a media-pass record carries
            # text_rows=0 and must not clobber a caption the text pass wrote.
            if key == "text_rows" and not value and merged.get("text_rows"):
                continue
            if value is not None:
                merged[key] = value
        existing[record["id"]] = merged

    samples = sorted(existing.values(), key=lambda entry: entry["id"])
    incomplete = [entry["id"] for entry in samples if not entry.get("text_rows")]
    if incomplete:
        logger.warning(
            "%d samples have no caption embedding yet (run the text pass); training will skip them",
            len(incomplete),
        )
    payload = {
        "version": 1,
        "geometry": str(geometry),
        "num_samples": len(samples),
        "num_trainable": len(samples) - len(incomplete),
        "samples": samples,
    }
    with index_path.open("w") as handle:
        json.dump(payload, handle, indent=1)
    return index_path


def verify_cache(output: Path) -> dict[str, Any]:
    """Sanity-check a finished cache: row counts agree with the stored geometry."""
    with (output / "index.json").open() as handle:
        payload = json.load(handle)
    problems = []
    for record in payload["samples"]:
        geometry = Geometry.create(record["width"], record["height"], record["num_frames"])
        latents = load_file(str(output / "latents" / f"{record['id']}.safetensors"))["latents"]
        if latents.shape[0] != geometry.video_rows:
            problems.append(f"{record['id']}: {latents.shape[0]} video rows, expected {geometry.video_rows}")
    return {"num_samples": len(payload["samples"]), "problems": problems}
