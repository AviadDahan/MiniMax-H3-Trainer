"""Generating validation clips with the adapter applied.

The point of this pass is to see and *hear* what the adapter is doing, on fixed
prompts and a fixed seed, so successive checkpoints are comparable. It is
deliberately separate from the held-out loss: the loss says whether the model is
fitting, the samples say whether it is fitting the thing you wanted.

Memory: the transformer already on the GPU is reused rather than reloaded (66GB
is not a thing to load twice), while the VAEs and the conditioner are brought in
around the sampling call and released after it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from h3_trainer import logger
from h3_trainer.config import H3TrainerConfig
from h3_trainer.constants import Geometry
from h3_trainer.preprocessing.media import write_video_with_audio


class ValidationRunner:
    def __init__(self, config: H3TrainerConfig, transformer: torch.nn.Module, device: torch.device) -> None:
        self.config = config
        self.transformer = transformer
        self.device = device
        self.output_dir = Path(config.output_dir) / "validation"
        self._pipeline = None

    def _build_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from diffusers import ModularPipeline

        blocks = "MiniMaxH3Ref2VABlocks" if self.config.model.variant == "ref2va" else "MiniMaxH3Blocks"
        pipeline = ModularPipeline.from_pretrained(
            str(self.config.model.model_path), blocks_name=blocks
        )
        # Everything except the transformer: that one is already resident and
        # carries the weights we are validating.
        pipeline.load_components(
            names=["text_encoder", "tokenizer", "processor", "vae", "audio_vae", "scheduler", "audio_scheduler"],
            dtype=torch.bfloat16,
        )
        pipeline.update_components(transformer=self.transformer)
        self._pipeline = pipeline
        return pipeline

    @torch.no_grad()
    def run(self, step: int) -> list[Path]:
        config = self.config.validation
        if not config.samples:
            return []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            pipeline = self._build_pipeline()
        except Exception as exc:
            logger.warning("Could not build the validation pipeline (%s); skipping media sampling", exc)
            return []

        was_training = self.transformer.training
        self.transformer.eval()
        outputs: list[Path] = []
        for index, sample in enumerate(config.samples):
            width, height, frames = sample.video_dims or config.video_dims
            geometry = Geometry.create(width=width, height=height, num_frames=frames)
            seed = sample.seed if sample.seed is not None else config.seed
            try:
                result = pipeline(
                    prompt=sample.prompt,
                    height=geometry.height,
                    width=geometry.width,
                    num_frames=geometry.num_frames,
                    num_inference_steps=config.inference_steps,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    output_type="np",
                    **self._conditioning_kwargs(sample),
                )
            except Exception as exc:
                logger.warning("Validation sample %d failed: %s", index, exc)
                continue
            outputs.append(self._write(result, step, index))

        if was_training:
            self.transformer.train()
        return outputs

    def _conditioning_kwargs(self, sample) -> dict:
        """Map validation conditions onto the pipeline's conditioning inputs."""
        from PIL import Image

        kwargs: dict = {}
        references = []
        for condition in sample.conditions:
            if condition.type == "first_frame":
                kwargs["image"] = Image.open(condition.image).convert("RGB")
            elif condition.type == "last_frame":
                kwargs["last_image"] = Image.open(condition.image).convert("RGB")
            elif condition.type == "reference":
                from diffusers.modular_pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference

                references.append(
                    MiniMaxH3Reference(
                        image=condition.image, video=condition.video, audio=condition.audio
                    )
                )
        if references:
            kwargs["references"] = references
        return kwargs

    def _write(self, result, step: int, index: int) -> Path:
        """Write the generated clip as an mp4 with its audio muxed in.

        A silent mp4 hides exactly the failure mode joint AV training is prone
        to, so the audio track always goes in when the pipeline produced one.
        """
        frames = _extract(result, ("videos", "frames", "video"))
        audio = _extract(result, ("audios", "audio"))
        if frames is None:
            raise RuntimeError("Validation pipeline returned no video")

        frames = np.asarray(frames)
        if frames.ndim == 5:
            frames = frames[0]
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)

        waveform = None
        if audio is not None:
            waveform = torch.as_tensor(np.asarray(audio)).float()
            waveform = waveform.reshape(-1) if waveform.ndim == 1 else waveform.reshape(waveform.shape[-2], -1)

        path = self.output_dir / f"step{step:07d}_sample{index}.mp4"
        return write_video_with_audio(frames, path, waveform=waveform, fps=self.config.validation.frame_rate)


def _extract(result, names: tuple[str, ...]):
    for name in names:
        value = getattr(result, name, None)
        if value is None and isinstance(result, dict):
            value = result.get(name)
        if value is not None:
            return value[0] if isinstance(value, (list, tuple)) and len(value) else value
    return None
