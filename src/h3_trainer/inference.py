"""Running MiniMax-H3 generation, with or without a trained adapter.

Inference matters here beyond producing videos: it is how validation samples are
made, how a synthetic character dataset is built, and how a finished adapter is
judged (same seed, adapter off vs on).

**The memory problem, and the way around it.** A full H3 generation involves the
33B transformer (66GB bf16), the Qwen3-VL-32B conditioner (63GB) and two VAEs
(11GB) -- about 140GB of weights. Loading all of it at once is what makes H3
awkward on anything smaller than a DGX.

It is also unnecessary. The conditioner runs *once*, before denoising starts, and
its output is a small tensor. So generation here runs in two stages:

1. **Condition.** The conditioner encodes the prompt (plus any keyframe or
   reference vision blocks) into layer-50 hidden states and per-row modality
   tags. It is spread across the GPUs, used, and unloaded.
2. **Denoise.** A pipeline assembled *without* the text-encoder block takes those
   embeddings directly, so only the transformer, the VAEs and the schedulers are
   resident -- ~77GB in bf16, or ~28GB with an NF4 transformer.

Peak memory halves, and stage 1 reuses exactly the same encoding path the
training cache is built with, so what the adapter was trained against and what it
is sampled with cannot drift apart.

**Sharding needs care.** H3's forward takes the packed layout's index vectors as
arguments and uses them for ``index_select`` against its own inputs and outputs,
outside any accelerate hook. A plain ``device_map="auto"`` therefore breaks: it
puts the output projection on the last GPU while the index tensors are still on
the first. The ``shard`` placement distributes only ``transformer_blocks.*`` and
pins every index-consuming module to one device, which gives full-precision
inference across small cards. See ``model_loader.build_transformer_device_map``.

**A note on 4-bit.** NF4 fits the model on a single 48GB card, but quantizing
H3's AdaLN modulation branches to 4 bits degrades generation badly -- verified
here, where an NF4 run decoded to noise indistinguishable from decoding random
latents. Keep 4-bit for memory-bound *training* (where the LoRA adapts around it)
and use ``shard`` for anything whose output you intend to look at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from h3_trainer import logger
from h3_trainer.constants import Geometry
from h3_trainer.preprocessing.media import write_video_with_audio

#: How to place the denoising transformer. It must end up on ONE device (see the
#: module docstring), so the choice is really "how do we make 66GB fit there".
PLACEMENT_MODES = ("shard", "quantize", "bf16", "offload")


@dataclass
class GenerationRequest:
    """One generation. ``references`` is Ref2VA only; ``image``/``last_image`` FL2VA only."""

    prompt: str
    geometry: Geometry
    seed: int = 42
    num_inference_steps: int = 30
    image: str | Path | None = None
    last_image: str | Path | None = None
    references: list[dict[str, Any]] = field(default_factory=list)
    #: Canvas a reference *video* is encoded on -- ``native`` (the inference
    #: recipe: the reference's own aspect at a 768 short edge) or ``target``
    #: (the generated bucket). This must match what the adapter was trained
    #: against; a mismatch changes the reference's row count and rotary grid.
    reference_canvas: str = "native"

    @property
    def uses_references(self) -> bool:
        return bool(self.references)

    @property
    def keyframe_paths(self) -> list[str]:
        return [str(p) for p in (self.image, self.last_image) if p is not None]


@dataclass
class Conditioning:
    """Everything stage 2 needs from the conditioner and the VAEs."""

    prompt_embeds: torch.Tensor
    text_token_tags: torch.Tensor
    keyframes: list = field(default_factory=list)
    prepared_references: list = field(default_factory=list)
    condition_latents: torch.Tensor | None = None
    audio_condition_latents: torch.Tensor | None = None


class H3Pipeline:
    def __init__(
        self,
        model_path: str | Path,
        variant: str = "fl2va",
        placement: str = "shard",
        quantization: str = "nf4-bnb",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if placement not in PLACEMENT_MODES:
            raise ValueError(f"placement must be one of {PLACEMENT_MODES}, got {placement!r}")
        self.model_path = Path(model_path)
        self.variant = variant
        self.placement = placement
        self.quantization = quantization
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        self._adapter: tuple[Path, float] | None = None

    # ------------------------------------------------------- stage 1: condition

    def encode_conditioning(self, request: GenerationRequest) -> Conditioning:
        """Condition a single request (loads and unloads the conditioner)."""
        return self.encode_conditioning_batch([request])[0]

    def encode_conditioning_batch(self, requests: list[GenerationRequest]) -> list[Conditioning]:
        """Condition many requests on one load of the conditioner.

        Loading Qwen3-VL-32B takes minutes, so doing it per clip turns a dataset
        generation run into hours of loading. Everything needing the conditioner
        therefore happens in one pass, before any denoising starts.
        """
        from h3_trainer.preprocessing.encoders import H3Encoders

        needs_vae = any(request.uses_references for request in requests)
        encoders = H3Encoders(
            self.model_path,
            device=self.device,
            need_video=needs_vae,
            need_audio=needs_vae,
            need_text=True,
            text_device_map=self._conditioner_device_map(),
        )
        results: list[Conditioning] = []
        try:
            for index, request in enumerate(requests):
                if request.uses_references:
                    conditioning = self._encode_reference_conditioning(encoders, request)
                else:
                    keyframes = self._load_keyframes(request)
                    embeds, tags = encoders.encode_prompt(request.prompt, keyframes or None)
                    conditioning = Conditioning(
                        prompt_embeds=embeds, text_token_tags=tags, keyframes=keyframes
                    )
                logger.info(
                    "[%d/%d] conditioned: %d text rows (%d tagged as vision)",
                    index + 1,
                    len(requests),
                    conditioning.prompt_embeds.shape[0],
                    int((conditioning.text_token_tags == 0).sum()),
                )
                results.append(conditioning)
        finally:
            encoders.unload()
            torch.cuda.empty_cache()
        return results

    def _conditioner_device_map(self) -> str | dict:
        """Spread the conditioner, but keep it off the denoising GPU when possible.

        Freeing a device_map'd model is reliable now (see ``H3Encoders.unload``),
        but not having to free it at all is better: with more than one GPU the
        conditioner simply never touches the card the transformer will occupy.
        """
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count <= 1:
            return "auto"
        index = torch.device(self.device).index or 0
        return {i: ("0GiB" if i == index else "40GiB") for i in range(count)}

    def _load_keyframes(self, request: GenerationRequest) -> list:
        from diffusers.modular_pipelines.minimax_h3.packing import prepare_keyframe_image
        from diffusers.utils import load_image

        images = []
        for index, path in enumerate(request.keyframe_paths):
            # The first keyframe is the geometry anchor and is stretched onto the
            # canvas; later ones are cover-cropped. That asymmetry is the
            # inference contract, not a detail.
            images.append(
                prepare_keyframe_image(
                    load_image(path),
                    request.geometry.height,
                    request.geometry.width,
                    stretch=index == 0,
                )
            )
        return images

    def _encode_reference_conditioning(self, encoders, request: GenerationRequest) -> Conditioning:
        """Prepare Ref2VA references: VAE rows, latent geometry, and the prompt presentation."""
        from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (
            MiniMaxH3PreparedReference,
            prepare_reference_image,
            resolve_reference_image_size,
        )
        from PIL import Image

        from h3_trainer.packing import noise_condition_rows
        from h3_trainer.preprocessing.media import decode_video, extract_audio, reference_video_canvas

        prepared: list[MiniMaxH3PreparedReference] = []
        video_rows, audio_rows = [], []
        media: list[tuple[str, Any]] = []

        for entry in request.references:
            if entry.get("image"):
                original = Image.open(entry["image"]).convert("RGB")
                width, height = resolve_reference_image_size(original.width, original.height)
                image = prepare_reference_image(original, height, width)
                encoded = encoders.encode_reference("image", image=image)
                media.append(("image", image))
            elif entry.get("video"):
                if request.reference_canvas == "native":
                    width, height = reference_video_canvas(entry["video"])
                else:
                    width, height = request.geometry.width, request.geometry.height
                clip = decode_video(entry["video"], request.geometry.num_frames, width, height)
                waveform = extract_audio(entry["video"], request.geometry.num_frames)
                encoded = encoders.encode_reference("video", frames=clip.frames, waveform=waveform)
                media.append(("frames", np.asarray(clip.frames, dtype=np.uint8)))
            else:
                waveform = extract_audio(entry["audio"], request.geometry.num_frames)
                if waveform is None:
                    raise ValueError(f"Reference audio {entry['audio']} has no usable track")
                encoded = encoders.encode_reference("audio", waveform=waveform)
                media.append(("audio", None))

            values = [int(v) for v in encoded.geometry.tolist()]
            prepared.append(
                MiniMaxH3PreparedReference(
                    kind=encoded.kind,
                    has_audio=values[4] > 0,
                    num_latent_frames=max(1, values[1]),
                    latent_height=values[2],
                    latent_width=values[3],
                    num_audio_latents=values[4],
                )
            )
            if encoded.video_rows is not None:
                video_rows.append(encoded.video_rows)
            if encoded.audio_rows is not None:
                audio_rows.append(encoded.audio_rows)

        # The conditioner takes different preprocessing paths for images and
        # videos, so each descriptor carries its own media.
        for reference, (kind, value) in zip(prepared, media):
            if kind == "image":
                reference.image = value
            elif kind == "frames":
                reference.frames = value
        embeds, tags = encoders.encode_ref2va_prompt(request.prompt, prepared)

        condition_latents = torch.cat(video_rows) if video_rows else None
        if condition_latents is not None:
            # Visual conditioning rows are handed to the transformer at their
            # noise-augmentation level, not clean.
            generator = torch.Generator(device="cpu").manual_seed(request.seed)
            noise = torch.randn(condition_latents.shape, generator=generator, dtype=condition_latents.dtype)
            condition_latents = noise_condition_rows(condition_latents, noise)
        return Conditioning(
            prompt_embeds=embeds,
            text_token_tags=tags,
            prepared_references=prepared,
            condition_latents=condition_latents,
            audio_condition_latents=torch.cat(audio_rows) if audio_rows else None,
        )

    # --------------------------------------------------------- stage 2: denoise

    def load(self):
        """Load the denoising half: transformer, VAEs, schedulers. No conditioner."""
        from diffusers.modular_pipelines import SequentialPipelineBlocks
        from diffusers.modular_pipelines.minimax_h3 import modular_blocks_minimax_h3 as blocks_module

        full = (
            blocks_module.MiniMaxH3Ref2VABlocks()
            if self.variant == "ref2va"
            else blocks_module.MiniMaxH3Blocks()
        )
        # Drop only the text encoder: that is the 63GB component stage 1 replaced,
        # and its outputs (prompt_embeds, text_token_tags) are passed in instead.
        # The reference encoder stays -- it needs the VAEs, which are resident
        # here anyway, and it is what fills in each reference's latent geometry
        # before the layout is built.
        skip = {"text_encoder"}
        trimmed = SequentialPipelineBlocks.from_blocks_dict(
            {name: block for name, block in full.sub_blocks.items() if name not in skip}
        )
        logger.info("Denoising pipeline blocks: %s", list(trimmed.sub_blocks))
        pipeline = trimmed.init_pipeline(str(self.model_path))

        pipeline.load_components(names=["scheduler", "audio_scheduler", "vae", "audio_vae"], dtype=torch.float32)
        for name in ("vae", "audio_vae"):
            setattr(pipeline, name, getattr(pipeline, name).to(self.device))

        if self.placement == "quantize":
            from h3_trainer.model_loader import TRANSFORMER_SUBFOLDERS
            from h3_trainer.quantization import load_quantized_transformer

            transformer = load_quantized_transformer(
                self.model_path, TRANSFORMER_SUBFOLDERS[self.variant], self.quantization, self.dtype
            )
            if self.quantization.endswith("-quanto"):
                transformer = transformer.to(self.device)
            pipeline.update_components(**{self.transformer_component: transformer})
        elif self.placement == "shard":
            from h3_trainer.model_loader import load_sharded_transformer

            primary = torch.device(self.device).index or 0
            transformer = load_sharded_transformer(
                self.model_path, variant=self.variant, dtype=self.dtype, primary=primary
            )
            pipeline.update_components(**{self.transformer_component: transformer})
        else:
            from h3_trainer.model_loader import load_transformer

            device = "cpu" if self.placement == "offload" else self.device
            transformer = load_transformer(
                self.model_path, variant=self.variant, dtype=self.dtype, device=device
            )
            pipeline.update_components(**{self.transformer_component: transformer})

        self.pipeline = pipeline
        if self.transformer is None:
            raise RuntimeError(
                f"The transformer was not registered as {self.transformer_component!r}; the blocks "
                f"expect one of {pipeline.component_names}."
            )
        self._report_placement()
        if self._adapter is not None:
            self._apply_adapter(*self._adapter)
        return self

    @property
    def transformer_component(self) -> str:
        """What the blocks call the transformer.

        The Ref2VA blocks register it as ``transformer_ref``, not ``transformer``.
        Registering under the wrong name is silently accepted and leaves the
        component unset -- which surfaces much later as "NoneType is not
        callable" inside the denoise loop.
        """
        return "transformer_ref" if self.variant == "ref2va" else "transformer"

    @property
    def transformer(self):
        return getattr(self.pipeline, self.transformer_component, None)

    @property
    def transformer_device(self) -> torch.device:
        """Where the model's *inputs* belong.

        With a sharded model that is the primary device -- the one holding the
        input/output projections and therefore the one the index vectors are
        selected against -- not wherever ``next(parameters())`` happens to live.
        """
        transformer = self.transformer
        if transformer is None:
            return torch.device(self.device)
        if self.placement == "shard":
            return next(transformer.proj_in.parameters()).device
        return next(transformer.parameters()).device

    def _report_placement(self) -> None:
        for name in (self.transformer_component, "vae", "audio_vae"):
            component = getattr(self.pipeline, name, None)
            if component is None or not hasattr(component, "parameters"):
                continue
            devices = sorted({str(p.device) for p in component.parameters()})
            logger.info("  %-11s on %s", name, ", ".join(devices))
        if torch.cuda.is_available():
            free = [round(torch.cuda.mem_get_info(i)[0] / 1e9, 1) for i in range(torch.cuda.device_count())]
            logger.info("  free VRAM per GPU (GB): %s", free)

    # ----------------------------------------------------------------- adapters

    def load_lora(self, path: str | Path, scale: float = 1.0) -> None:
        """Apply a trained adapter, reconstructing its shape from the checkpoint."""
        self._adapter = (Path(path), scale)
        if self.pipeline is not None:
            self._apply_adapter(Path(path), scale)

    def _apply_adapter(self, path: Path, scale: float) -> None:
        from peft import LoraConfig
        from safetensors.torch import load_file

        from h3_trainer.checkpointing import find_latest_checkpoint

        checkpoint = find_latest_checkpoint(path)
        if checkpoint is None:
            raise FileNotFoundError(f"No adapter found at {path}")
        weights_path = checkpoint / "adapter.safetensors" if checkpoint.is_dir() else checkpoint
        weights = load_file(str(weights_path))
        lora_keys = [key for key in weights if ".lora_A" in key]
        if not lora_keys:
            raise ValueError(f"{weights_path} contains no LoRA tensors")

        rank = max(weights[key].shape[0] for key in lora_keys)
        targets = sorted({_target_suffix(key) for key in lora_keys})
        transformer = self.transformer
        transformer.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=int(round(rank * scale)),
                target_modules=targets,
                lora_dropout=0.0,
            )
        )
        missing, unexpected = transformer.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(f"Adapter does not match this transformer: {unexpected[:3]}")
        logger.info(
            "Applied adapter %s (rank %d, targets %s, scale %.2f)", weights_path, rank, targets, scale
        )

    # --------------------------------------------------------------- generating

    @torch.no_grad()
    def generate(self, request: GenerationRequest, output_path: str | Path) -> Path:
        return self.generate_prepared(request, self.encode_conditioning(request), output_path)

    @torch.no_grad()
    def generate_prepared(
        self, request: GenerationRequest, conditioning: Conditioning, output_path: str | Path
    ) -> Path:
        """Denoise a request whose conditioning was computed earlier."""
        if self.pipeline is None:
            self.load()

        # Stage 1 leaves its tensors on the CPU (it may have been spread across
        # other GPUs). The pipeline passes user-supplied embeddings straight to
        # the transformer without moving them, so place them here.
        device = self.transformer_device
        kwargs: dict[str, Any] = {
            "prompt_embeds": conditioning.prompt_embeds[None].to(device, self.dtype),
            # The layout is built on the CPU, so the tags stay there -- only the
            # embeddings themselves reach the transformer.
            "text_token_tags": conditioning.text_token_tags.cpu(),
        }
        if conditioning.keyframes:
            kwargs["keyframes"] = conditioning.keyframes
        if request.uses_references:
            # The pipeline's own reference encoder re-derives the latents and,
            # crucially, the latent geometry the packed layout is built from.
            # Stage 1 only needed the references to build the *prompt*
            # presentation, so the raw descriptors are what to pass here.
            from diffusers.modular_pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference

            kwargs["references"] = [MiniMaxH3Reference(**entry) for entry in request.references]

        logger.info(
            "Denoising %s, %d steps, seed %d: %s",
            request.geometry,
            request.num_inference_steps,
            request.seed,
            request.prompt[:80],
        )
        result = self.pipeline(
            height=request.geometry.height,
            width=request.geometry.width,
            num_frames=request.geometry.num_frames,
            num_inference_steps=request.num_inference_steps,
            generator=torch.Generator(device="cpu").manual_seed(request.seed),
            output_type="np",
            **kwargs,
        )
        return self.write(result, output_path)

    @staticmethod
    def write(result, output_path: str | Path) -> Path:
        """Write frames + audio to an mp4.

        H3 generates the audio jointly with the video, so a silent mp4 is a broken
        result rather than a smaller one -- hence the explicit warning.
        """
        frames = _first(result, ("videos", "frames", "video"))
        audio = _first(result, ("audios", "audio"))
        if frames is None:
            raise RuntimeError(f"Pipeline returned no video (fields: {_fields(result)})")

        frames = _to_numpy(frames)
        if frames.ndim == 5:
            frames = frames[0]
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)

        waveform = None
        if audio is not None:
            waveform = torch.as_tensor(_to_numpy(audio)).float()
            while waveform.ndim > 2:
                waveform = waveform[0]
            if waveform.ndim == 1:
                waveform = waveform[None].repeat(2, 1)
        else:
            logger.warning("Pipeline returned no audio track")

        path = write_video_with_audio(frames, output_path, waveform=waveform)
        logger.info("Wrote %s (%d frames, audio=%s)", path, frames.shape[0], waveform is not None)
        return path


def _target_suffix(key: str) -> str:
    """``...attn.to_out.0.lora_A.weight`` -> ``to_out.0``; ``...attn.to_q...`` -> ``to_q``."""
    path = key.split(".lora_A")[0]
    parts = path.split(".")
    if parts[-1].isdigit() and len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]


def _to_numpy(value) -> np.ndarray:
    """Pipeline outputs can be numpy or CUDA tensors depending on ``output_type``."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fields(result) -> list[str]:
    if isinstance(result, dict):
        return sorted(result)
    return [name for name in dir(result) if not name.startswith("_")]


def _first(result, names: tuple[str, ...]):
    for name in names:
        value = getattr(result, name, None)
        if value is None and isinstance(result, dict):
            value = result.get(name)
        if value is not None:
            if isinstance(value, (list, tuple)):
                return value[0] if len(value) else None
            return value
    return None
