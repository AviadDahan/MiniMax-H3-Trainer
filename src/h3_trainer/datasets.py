"""Loading the precomputed latent cache.

``scripts/process_dataset.py`` writes one ``.safetensors`` per sample per tensor
kind into an LTX-style tree::

    <root>/.precomputed/
        index.json                  one record per sample: geometry, row counts, flags
        latents/<id>.safetensors    patchified video latent rows
        audio_latents/<id>...       channel-major audio latent rows
        conditions/<id>...          Qwen3-VL layer-50 prompt embeddings
        reference_latents/<id>...   IC-LoRA reference rows (optional)
        first_frame_latents/<id>... keyframe rows (optional)

Nothing here touches a VAE or a text encoder: by training time everything is
already latents, which is what keeps the 33B transformer the only large thing on
the GPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from h3_trainer import logger
from h3_trainer.constants import Geometry

INDEX_FILENAME = "index.json"

#: Audio VAE latent channels (``audio_in_channels`` in the transformer config).
AUDIO_LATENT_CHANNELS = 32


def is_validation_id(sample_id: str, every: int) -> bool:
    """Deterministic held-out split: ``md5(id) % every == 0``.

    Hash-based rather than index-based so the split survives adding, removing or
    reordering samples -- a validation curve is only comparable across runs if
    the split is stable.
    """
    if every <= 0:
        return False
    digest = hashlib.md5(sample_id.encode()).hexdigest()[:8]
    return int(digest, 16) % every == 0


@dataclass
class SampleRecord:
    """One row of ``index.json``."""

    id: str
    width: int
    height: int
    num_frames: int
    video_rows: int
    audio_rows: int
    text_rows: int
    has_audio: bool
    caption: str = ""
    extras: dict[str, Any] = None  # type: ignore[assignment]

    @property
    def geometry(self) -> Geometry:
        return Geometry.create(self.width, self.height, self.num_frames)

    @property
    def sequence_length(self) -> int:
        return self.text_rows + self.audio_rows + self.video_rows

    @property
    def bucket_key(self) -> tuple[int, int, int, int]:
        """Samples sharing this key have identical packed layouts.

        H3's batch axis is a pure replication axis over one shared layout, so
        only samples with the same row counts (including caption length) can ride
        in the same micro-batch.
        """
        return (self.video_rows, self.audio_rows, self.text_rows, int(self.has_audio))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SampleRecord:
        known = {field for field in cls.__dataclass_fields__ if field != "extras"}
        extras = {key: value for key, value in payload.items() if key not in known}
        return cls(**{key: payload[key] for key in known if key in payload}, extras=extras)


class PrecomputedDataset(Dataset):
    """Latent rows + prompt embeddings for one split."""

    def __init__(
        self,
        root: str | Path,
        data_sources: dict[str, str],
        split: str = "train",
        val_split_every: int = 20,
        max_seq_tokens: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Preprocessed data root does not exist: {self.root}")
        self.data_sources = dict(data_sources)
        self.split = split

        index_path = self.root / INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} is missing. Run scripts/process_dataset.py to build the latent cache."
            )
        with index_path.open() as handle:
            payload = json.load(handle)
        records = [SampleRecord.from_dict(entry) for entry in payload["samples"]]
        # The media pass writes records before the text pass fills in captions;
        # a sample with no prompt embedding cannot be packed.
        unconditioned = [record.id for record in records if record.text_rows <= 0]
        if unconditioned:
            logger.warning(
                "%d samples have no cached caption embedding and are skipped (run "
                "process_dataset.py --only text)",
                len(unconditioned),
            )
            records = [record for record in records if record.text_rows > 0]

        if split == "train":
            records = [r for r in records if not is_validation_id(r.id, val_split_every)]
        elif split == "val":
            records = [r for r in records if is_validation_id(r.id, val_split_every)]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        if max_seq_tokens is not None:
            kept = [r for r in records if r.sequence_length <= max_seq_tokens]
            if len(kept) != len(records):
                logger.warning(
                    "%d/%d %s samples exceed max_seq_tokens=%d and were dropped from the dataset",
                    len(records) - len(kept),
                    len(records),
                    split,
                    max_seq_tokens,
                )
            records = kept

        self.records = records
        if not self.records and split == "train":
            raise RuntimeError(
                f"No training samples under {self.root}. Either the cache is empty or every sample "
                f"landed in the validation split (data.val_split_every={val_split_every})."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _load_tensor(self, source: str, sample_id: str, required: bool) -> dict[str, torch.Tensor] | None:
        directory = self.data_sources.get(source)
        if directory is None:
            if required:
                raise KeyError(f"No data source configured for {source!r}")
            return None
        path = self.root / directory / f"{sample_id}.safetensors"
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing {source} tensor for sample {sample_id}: {path}")
            return None
        return load_file(str(path))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        video = self._load_tensor("video", record.id, required=True)["latents"]
        audio_payload = self._load_tensor("audio", record.id, required=False)
        text_payload = self._load_tensor("text", record.id, required=True)
        text = text_payload["prompt_embeds"]

        if audio_payload is not None:
            audio = audio_payload["latents"]
        else:
            # A silent clip still occupies audio rows in the packed sequence; the
            # loss weight (not the rows) is what gets zeroed. See
            # flow_matching.audio_loss_weight.
            audio = torch.zeros(record.audio_rows, AUDIO_LATENT_CHANNELS, dtype=video.dtype)

        item: dict[str, Any] = {
            "id": record.id,
            "record": record,
            "video": video,
            "audio": audio,
            "text": text,
            "caption": record.caption,
        }
        # Vision blocks embedded in the prompt (keyframe "<Picture N>" labels,
        # reference media) are tagged as video rather than text, and the
        # transformer's AdaLN modulation keys off that tag.
        if "text_token_tags" in text_payload:
            item["text_token_tags"] = text_payload["text_token_tags"]

        for source in ("first_frame", "last_frame"):
            payload = self._load_tensor(source, record.id, required=False)
            if payload is not None:
                item[source] = payload["latents"]

        reference = self._load_tensor("reference", record.id, required=False)
        if reference is not None:
            item["reference"] = True
            item["reference_geometry"] = list(reference["geometry"])
            item["reference_video_rows"] = reference.get("latents")
            item["reference_audio_rows"] = reference.get("audio_latents")
        return item


def identity_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate for micro-batch 1 (the common case): pass the sample straight through."""
    if len(batch) != 1:
        raise ValueError(
            f"identity_collate received {len(batch)} samples; use bucket_collate for real batching"
        )
    return batch[0]


def bucket_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack samples that share an identical packed layout.

    Raises rather than padding. Padding itself is supported -- rows tagged `-1`
    are kept as a separate attention document, exactly as the reference
    implementation does when it pads to a multiple of 64 -- but it does not help
    here. H3's batch axis is a pure *replication* axis: `token_tags`,
    `position_ids` and the index tensors describe one layout that every item in
    the batch shares. Two samples of different shape cannot be made to share it
    by padding, because they would need different tags. Bucketing is the way in,
    and a padless sequence also keeps the unmasked attention backends available.
    """
    if len(batch) == 1:
        return batch[0]
    keys = {item["record"].bucket_key for item in batch}
    if len(keys) != 1:
        raise ValueError(
            f"Cannot batch samples with different packed layouts: {sorted(keys)}. Use a "
            f"BucketBatchSampler, or set optimization.batch_size: 1."
        )
    merged = dict(batch[0])
    merged["id"] = [item["id"] for item in batch]
    merged["caption"] = [item["caption"] for item in batch]
    for key in ("video", "audio", "text", "reference", "first_frame", "last_frame"):
        if key in batch[0]:
            merged[key] = torch.stack([item[key] for item in batch], dim=0)
    return merged


class BucketBatchSampler(torch.utils.data.Sampler):
    """Yields batches of indices that share a packed layout.

    Shuffling happens inside each bucket and across buckets every epoch, so a
    dataset with a handful of geometries still trains in a well-mixed order --
    unlike the reference trainer, which walks a fixed file list forever.
    """

    def __init__(
        self,
        dataset: PrecomputedDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.buckets: dict[tuple, list[int]] = {}
        for index, record in enumerate(dataset.records):
            self.buckets.setdefault(record.bucket_key, []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.buckets.values():
            order = list(indices)
            if self.shuffle:
                permutation = torch.randperm(len(order), generator=generator).tolist()
                order = [order[i] for i in permutation]
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in permutation]
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total
