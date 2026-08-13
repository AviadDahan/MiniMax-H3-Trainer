"""Encoding media into H3's latent rows, with the released model's exact recipes.

Two recipes, and they are *different* on purpose -- this is the single easiest
place to introduce a silent train/inference mismatch:

**Targets** (what the model learns to produce) take the posterior *mode*: the
deterministic latent, no sampling noise in the regression target.

**Conditioning** (keyframes and in-context references) reproduces the inference
path exactly: the posterior is *sampled* under a generator seeded with 42
independently of any request seed, and the sampled latent is rounded through
float16 before normalization -- which throws away ~11 bits of every conditioning
latent. The released model was conditioned that way; conditioning it any other
way at training time teaches it against inputs it will never see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from h3_trainer import logger
from h3_trainer.constants import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    PATCH_SIZE,
)
from h3_trainer.model_loader import load_audio_vae, load_text_encoder, load_video_vae
from h3_trainer.packing import encode_reference_geometry
from h3_trainer.preprocessing.media import frames_to_pixel_tensor

#: The seed H3 encodes conditioning latents under, fixed in the released pipeline.
KEYFRAME_ENCODE_SEED = 42


@dataclass
class EncodedReference:
    """Cached rows plus the latent geometry the ref2va packer replays."""

    kind: str
    video_rows: torch.Tensor | None
    audio_rows: torch.Tensor | None
    geometry: torch.Tensor
    #: Vision-token count of the reference's block(s), needed to rebuild the prompt presentation.
    block_token_counts: list[int]


class H3Encoders:
    """The video VAE, audio VAE and Qwen3-VL conditioner, loaded on demand."""

    def __init__(
        self,
        model_path: str | Path,
        device: torch.device | str = "cuda",
        need_video: bool = True,
        need_audio: bool = True,
        need_text: bool = True,
        text_device_map: str | dict | None = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.vae = load_video_vae(model_path, device) if need_video else None
        self.audio_vae = load_audio_vae(model_path, device) if need_audio else None
        # The conditioner is 32B and generally has to be spread over several GPUs;
        # its input device is therefore not necessarily this object's device.
        self.text = load_text_encoder(model_path, device, device_map=text_device_map) if need_text else None

        if self.vae is not None:
            config = self.vae.config
            self._latents_mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1)
            self._latents_std = torch.tensor(config.latents_std).view(1, -1, 1, 1, 1)
        if self.audio_vae is not None:
            config = self.audio_vae.config
            self._audio_mean = torch.tensor(config.latents_mean).view(1, 1, -1)
            self._audio_std = torch.tensor(config.latents_std).view(1, 1, -1)
            self.audio_latent_channels = int(config.latent_channels)

        self._pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        self._pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)

    # --------------------------------------------------------------- video

    def _normalize_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        return (pixels.to(self.device, torch.float32) - self._pixel_mean) / self._pixel_std

    @torch.no_grad()
    def encode_video_target(self, frames: np.ndarray) -> torch.Tensor:
        """Target video -> patchified latent rows ``(rows, channels * 4)``."""
        from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents

        pixels = self._normalize_pixels(frames_to_pixel_tensor(frames))
        latents = self.vae.encode(pixels).latent_dist.mode()
        normalized = (latents.float().cpu() - self._latents_mean) / self._latents_std
        return patchify_video_latents(normalized, PATCH_SIZE).contiguous()

    @torch.no_grad()
    def encode_visual_condition(self, frames: np.ndarray, is_image: bool) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Keyframe / visual reference -> rows, reproducing the inference recipe.

        An image goes through the spatial encoder alone; a video goes through the
        17-frames-per-5-latents temporal chunking.
        """
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents

        pixels = self._normalize_pixels(frames_to_pixel_tensor(frames))
        moments = self.vae._encode_clip(pixels) if is_image else self.vae._encode(pixels)
        posterior = DiagonalGaussianDistribution(moments)
        latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
        # The float16 round-trip is part of the contract, not an optimization.
        latents = latents.to(torch.float16).float().cpu()
        geometry = (int(latents.shape[2]), int(latents.shape[3]), int(latents.shape[4]))
        normalized = (latents - self._latents_mean) / self._latents_std
        return patchify_video_latents(normalized, PATCH_SIZE).contiguous(), geometry

    @torch.no_grad()
    def decode_video(self, rows: torch.Tensor, latent_frames: int, latent_height: int, latent_width: int) -> np.ndarray:
        """Inverse of :meth:`encode_video_target`, for ``--decode`` verification."""
        from diffusers.modular_pipelines.minimax_h3.packing import unpatchify_video_tokens

        latents = unpatchify_video_tokens(
            rows,
            num_latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            channels=int(self.vae.config.latent_channels),
            patch_size=PATCH_SIZE,
        )
        latents = latents * self._latents_std + self._latents_mean
        pixels = self.vae.decode(latents.to(self.device, torch.float32)).sample
        pixels = pixels * self._pixel_std + self._pixel_mean
        pixels = pixels.clamp(0, 1).mul(255).to(torch.uint8).cpu()[0]
        return pixels.permute(1, 2, 3, 0).numpy()

    # --------------------------------------------------------------- audio

    @torch.no_grad()
    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Stereo waveform -> channel-major latent rows ``[ch0 x N, ch1 x N]``.

        The mono audio VAE sees the two channels as two batch items, which is why
        the rows come out channel-major rather than interleaved.
        """
        posterior = self.audio_vae.encode(waveform.to(self.device)[:, None], return_dict=False)[0]
        latents = posterior.mode().float().cpu().transpose(1, 2)  # (2, T, C)
        normalized = (latents - self._audio_mean) / self._audio_std
        return normalized.reshape(-1, self.audio_latent_channels).contiguous()

    @torch.no_grad()
    def decode_audio(self, rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
        from diffusers.modular_pipelines.minimax_h3.packing import unpack_audio_tokens

        latents = unpack_audio_tokens(rows, num_audio_latents)
        latents = latents * self._audio_std.transpose(1, 2) + self._audio_mean.transpose(1, 2)
        waveform = self.audio_vae.decode(latents.to(self.device, torch.float32)).sample
        return waveform.float().cpu().reshape(2, -1)

    # ---------------------------------------------------------------- text

    @torch.no_grad()
    def encode_prompt(
        self, prompt: str, images: list[Image.Image] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompt (plus keyframe vision blocks) -> layer-50 states and row tags.

        This mirrors the inference presentation exactly: each keyframe prepends a
        ``"<Picture i>: "`` label and a vision block, with no chat template and no
        special tokens. Vision-block rows are tagged *video*, and the transformer's
        AdaLN modulation keys off that tag -- getting it wrong changes how every
        text row is modulated.
        """
        tokenizer, processor, model = self.text.tokenizer, self.text.processor, self.text.model
        pixel_values = image_grid_thw = None
        token_ids: list[int] = []
        token_tags: list[int] = []

        if images:
            vision = processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            merge_size = processor.image_processor.merge_size**2
            for index in range(len(images)):
                num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
                label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = (
                    [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

        return self._run_conditioner(token_ids, token_tags, pixel_values, image_grid_thw)

    @torch.no_grad()
    def encode_ref2va_prompt(self, prompt: str, references: list) -> tuple[torch.Tensor, torch.Tensor]:
        """The ref2va presentation: labelled reference blocks, then the prompt.

        Each reference must carry its media: ``.image`` for an image reference,
        ``.frames`` for a video one. Audio references are labelled but never reach
        the conditioner -- a waveform has no vision block.

        Images and videos go through *different* preprocessors. A video reference
        is seen as a 2 fps block view and must be fed as ``pixel_values_videos``
        with its own grid; running those blocks through the image processor
        produces token counts that do not match the presentation, and the model
        rejects the batch with "Image features and image tokens do not match".
        """
        from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (
            build_ref2va_presentation,
            sample_reference_video_frames,
        )

        processor = self.text.processor
        merge_size = processor.image_processor.merge_size**2

        pixel_values = image_grid_thw = None
        image_token_counts: list[int] = []
        images = [reference.image for reference in references if reference.kind == "image"]
        if images:
            vision = processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            image_token_counts = [int(grid.prod()) // merge_size for grid in image_grid_thw]

        pixel_values_videos = video_grid_thw = None
        video_block_token_counts: list[int] = []
        videos = [reference for reference in references if reference.kind == "video"]
        if videos:
            sampled = [sample_reference_video_frames(reference.frames) for reference in videos]
            for reference, (_, timestamps) in zip(videos, sampled):
                reference.block_timestamps = timestamps
            vision = processor.video_processor(
                videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
            )
            pixel_values_videos, video_grid_thw = vision["pixel_values_videos"], vision["video_grid_thw"]
            video_block_token_counts = [int(grid[1]) * int(grid[2]) // merge_size for grid in video_grid_thw]

        token_ids, token_tags = build_ref2va_presentation(
            self.text.tokenizer, prompt, references, image_token_counts, video_block_token_counts
        )
        return self._run_conditioner(
            token_ids, token_tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw
        )

    def _run_conditioner(
        self,
        token_ids: list[int],
        token_tags: list[int],
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model, processor = self.text.model, self.text.processor
        num_layers = model.config.text_config.num_hidden_layers
        if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError(
                f"H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] but this text encoder "
                f"has only {num_layers} layers."
            )
        # Inputs go on the conditioner's own input device, which is not this
        # object's device when the 32B model is sharded across GPUs.
        text_device = self.text.device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=text_device)
        mm_token_type_ids = torch.tensor(
            processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=text_device
        )
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=None if pixel_values is None else pixel_values.to(text_device, model.dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(text_device),
            pixel_values_videos=(
                None if pixel_values_videos is None else pixel_values_videos.to(text_device, model.dtype)
            ),
            video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(text_device),
            use_cache=False,
            output_hidden_states=True,
        )
        embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER][0].to(torch.bfloat16).cpu()
        return embeds.contiguous(), torch.tensor(token_tags, dtype=torch.long)

    # ----------------------------------------------------------- references

    @torch.no_grad()
    def encode_reference(
        self,
        kind: str,
        frames: np.ndarray | None = None,
        image: Image.Image | None = None,
        waveform: torch.Tensor | None = None,
    ) -> EncodedReference:
        """Encode one in-context reference into cached rows plus its geometry."""
        video_rows = audio_rows = None
        latent_frames = latent_height = latent_width = 0
        num_audio_latents = 0

        if kind == "image":
            if image is None:
                raise ValueError("An image reference needs an image")
            array = np.asarray(image, dtype=np.uint8)[None]
            video_rows, (latent_frames, latent_height, latent_width) = self.encode_visual_condition(
                array, is_image=True
            )
        elif kind == "video":
            if frames is None:
                raise ValueError("A video reference needs frames")
            video_rows, (latent_frames, latent_height, latent_width) = self.encode_visual_condition(
                frames, is_image=False
            )
        elif kind != "audio":
            raise ValueError(f"Unknown reference kind {kind!r}")

        if waveform is not None:
            audio_rows = self.encode_audio(waveform)
            num_audio_latents = audio_rows.shape[0] // 2

        geometry = encode_reference_geometry(
            kind, latent_frames, latent_height, latent_width, num_audio_latents
        )
        logger.debug(
            "Encoded %s reference: %s video rows, %s audio rows",
            kind,
            0 if video_rows is None else video_rows.shape[0],
            0 if audio_rows is None else audio_rows.shape[0],
        )
        return EncodedReference(
            kind=kind, video_rows=video_rows, audio_rows=audio_rows, geometry=geometry, block_token_counts=[]
        )

    def unload(self) -> None:
        """Release every model and actually give the VRAM back.

        Dropping the references is not enough when a model was placed with
        ``device_map``: accelerate attaches dispatch hooks that keep the modules
        (and their weights) reachable, so the memory stays allocated until the
        hooks are removed and a collection runs. Inference depends on this --
        stage 2 needs the card the conditioner was just using.
        """
        import gc

        try:
            from accelerate.hooks import remove_hook_from_module
        except ImportError:  # pragma: no cover
            remove_hook_from_module = None

        for name in ("vae", "audio_vae"):
            model = getattr(self, name, None)
            if model is not None and remove_hook_from_module is not None:
                remove_hook_from_module(model, recurse=True)
            setattr(self, name, None)

        if self.text is not None:
            if remove_hook_from_module is not None:
                remove_hook_from_module(self.text.model, recurse=True)
            self.text.model = None
            self.text = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free = [round(torch.cuda.mem_get_info(i)[0] / 1e9, 1) for i in range(torch.cuda.device_count())]
            logger.info("Encoders unloaded; free VRAM per GPU (GB): %s", free)
