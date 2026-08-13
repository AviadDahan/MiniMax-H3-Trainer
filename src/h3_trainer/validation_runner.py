"""Generating validation clips with the adapter applied.

The point of this pass is to see and *hear* what the adapter is doing, on fixed
prompts and a fixed seed, so successive checkpoints are comparable. It is
deliberately separate from the held-out loss: the loss says whether the model is
fitting, the samples say whether it is fitting the thing you wanted.

Memory is what makes this hard, and it is why the pass is split in two. The
transformer already on the GPU is reused rather than reloaded (66GB is not a
thing to load twice), but the *conditioner* is another 63GB and will not fit
beside it on 48GB cards.

So prompts are encoded **once, before training starts**: the conditioner is
loaded, every validation sample (including the vision blocks of any reference
media) is turned into embeddings, and it is released again. The in-loop sampling
pass then runs a pipeline with the text encoder removed, needing only the
resident transformer and the two VAEs -- about 11GB rather than 63.

This is what makes `validation.sample_media: true` usable on small cards, and it
is the same two-stage split `inference.py` uses.
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
        #: prompt embeddings + per-row tags per validation sample, filled by prepare()
        self._conditioning: list[dict] | None = None

    def prepare(self) -> None:
        """Encode every validation prompt once, then release the conditioner.

        Called before the transformer is placed, so the 63GB conditioner has the
        machine to itself. Failing here is not fatal: sampling falls back to
        letting the pipeline encode prompts itself, which needs the conditioner
        resident and therefore only works with memory to spare.
        """
        samples = self.config.validation.samples
        if not samples or not self.config.validation.sample_media:
            return
        from h3_trainer.preprocessing.encoders import H3Encoders

        needs_media = any(condition.type == "reference" for sample in samples for condition in sample.conditions)
        logger.info("Pre-encoding %d validation prompt(s) before the conditioner is released", len(samples))
        encoders = None
        try:
            encoders = H3Encoders(
                self.config.model.model_path,
                device=self.device,
                need_video=needs_media,
                need_audio=needs_media,
                need_text=True,
                text_device_map="auto",
            )
            prepared = []
            for sample in samples:
                references = [
                    {"image": c.image, "video": c.video, "audio": c.audio}
                    for c in sample.conditions
                    if c.type == "reference"
                ]
                if references:
                    embeds, tags = encoders.encode_ref2va_prompt(sample.prompt, references)
                else:
                    embeds, tags = encoders.encode_prompt(sample.prompt, None)
                prepared.append({"prompt_embeds": embeds, "text_token_tags": tags, "references": references})
            self._conditioning = prepared
            logger.info("Validation prompts encoded; the conditioner is not needed again this run")
        except Exception as exc:
            logger.warning(
                "Could not pre-encode validation prompts (%s). Sampling will fall back to encoding "
                "in-loop, which needs the 63GB conditioner alongside the transformer.",
                exc,
            )
            self._conditioning = None
        finally:
            if encoders is not None:
                encoders.unload()

    def _build_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from diffusers.modular_pipelines import SequentialPipelineBlocks
        from diffusers.modular_pipelines.minimax_h3 import modular_blocks_minimax_h3 as blocks_module

        full = (
            blocks_module.MiniMaxH3Ref2VABlocks()
            if self.config.model.variant == "ref2va"
            else blocks_module.MiniMaxH3Blocks()
        )
        names = ["scheduler", "audio_scheduler", "vae", "audio_vae"]
        if self._conditioning is not None:
            # Drop the text encoder: prepare() already produced its outputs. The
            # reference encoder stays -- it needs the VAEs, which are resident
            # here anyway, and it fills in each reference's latent geometry
            # before the packed layout is built.
            blocks = SequentialPipelineBlocks.from_blocks_dict(
                {name: block for name, block in full.sub_blocks.items() if name != "text_encoder"}
            )
        else:
            blocks = full
            names += ["text_encoder", "tokenizer", "processor"]

        pipeline = blocks.init_pipeline(str(self.config.model.model_path))
        pipeline.load_components(names=names, dtype=torch.bfloat16)
        for name in ("vae", "audio_vae"):
            component = getattr(pipeline, name, None)
            if component is not None:
                setattr(pipeline, name, component.to(self.device))
        # The transformer is already resident and carries the weights under test.
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
                    **self._prompt_kwargs(index, sample),
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

    def _prompt_kwargs(self, index: int, sample) -> dict:
        """Cached embeddings when prepare() ran, otherwise the raw prompt."""
        if self._conditioning is None:
            return {"prompt": sample.prompt}
        cached = self._conditioning[index]
        return {
            "prompt_embeds": cached["prompt_embeds"][None].to(self.device, torch.bfloat16),
            # Built on the CPU; only the embeddings themselves reach the transformer.
            "text_token_tags": cached["text_token_tags"].cpu(),
        }

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
