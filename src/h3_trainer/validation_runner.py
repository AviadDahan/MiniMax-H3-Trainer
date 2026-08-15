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

import signal
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from h3_trainer import logger
from h3_trainer.config import H3TrainerConfig
from h3_trainer.constants import Geometry
from h3_trainer.preprocessing.media import write_video_with_audio


@contextmanager
def time_budget(seconds: int):
    """Abandon the wrapped call if it outlives its budget.

    SIGALRM fires in the main thread between Python-level operations, which is
    where a diffusers denoising loop spends its time, so a sampler that has
    wandered onto the CPU can still be interrupted. Outside the main thread (or on
    a platform without SIGALRM) this is a no-op rather than an error: a missing
    watchdog should not stop a run from training.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    try:
        previous = signal.signal(signal.SIGALRM, _raise_timeout)
    except (ValueError, AttributeError):  # no SIGALRM here
        yield
        return
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _raise_timeout(signum, frame):
    raise TimeoutError("validation sample exceeded its time budget")


class ValidationRunner:
    def __init__(
        self,
        config: H3TrainerConfig,
        transformer: torch.nn.Module,
        device: torch.device,
        output_dir: Path | str | None = None,
    ) -> None:
        self.config = config
        self.transformer = transformer
        self.device = device
        # The run's own directory, not the configured root. Runs are timestamped
        # underneath that root, so resolving against it put every run's clips in
        # one shared folder where each overwrote the last at matching steps -- and
        # left them somewhere `<run>/validation` did not point at.
        self.output_dir = Path(output_dir or config.output_dir) / "validation"
        self._pipeline = None
        #: prompt embeddings + per-row tags per validation sample, filled by prepare()
        self._conditioning: list[dict] | None = None
        #: set when a sample blows its budget; keeps later checkpoints from paying it again
        self._sampling_disabled = False

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
                references = self._prepared_references(sample)
                if references:
                    embeds, tags = encoders.encode_ref2va_prompt(sample.prompt, references)
                else:
                    embeds, tags = encoders.encode_prompt(sample.prompt, None)
                prepared.append({"prompt_embeds": embeds, "text_token_tags": tags, "references": references})
            self._conditioning = prepared
            logger.info("Validation prompts encoded; the conditioner is not needed again this run")
        except Exception as exc:
            # Loud, and with the traceback. The fallback is not a mild degradation:
            # it needs the 63GB conditioner resident alongside a transformer that
            # already fills the cards, so in practice it OOMs or lands the encoder
            # on the CPU, where a single sample runs past any useful budget.
            logger.error(
                "Could not pre-encode validation prompts (%s). Sampling falls back to encoding in-loop, "
                "which needs the 63GB conditioner alongside the transformer and usually cannot fit.",
                exc,
                exc_info=True,
            )
            self._conditioning = None
        finally:
            if encoders is not None:
                encoders.unload()

    def _reference_requests(self, sample) -> list:
        """A sample's reference conditions as ref2va *request* objects.

        These carry paths. Decoding happens inside the dataclass, so building one
        is also what validates that the media exists and is readable.
        """
        from diffusers.modular_pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference

        return [
            MiniMaxH3Reference(image=condition.image, video=condition.video, audio=condition.audio)
            for condition in sample.conditions
            if condition.type == "reference"
        ]

    def _prepared_references(self, sample) -> list:
        """Resolve a sample's references the way the ref2va blocks resolve theirs.

        ``encode_ref2va_prompt`` needs *prepared* references: a ``kind``, decoded
        pixels, and the ``block_timestamps`` it fills in as it samples the vision
        blocks. Request objects -- and, before this, bare dicts -- carry none of
        that, and the presentation cannot be built without it. Handing dicts over
        raised ``'dict' object has no attribute 'kind'``, which was swallowed into
        the in-loop fallback: the pipeline then loaded the 63GB conditioner on the
        CPU and one sample ran past 1800 s.

        The resolve is the blocks' own static method rather than a copy of it. The
        rules it encodes -- 24 fps resample, the 768 canvas of the reference's own
        aspect ratio, truncation to the generated frame count -- have to match what
        the pipeline will do at sampling time, and a second implementation here
        would only match until diffusers changed one of them.
        """
        requests = self._reference_requests(sample)
        if not requests:
            return []
        from types import SimpleNamespace

        from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep

        from h3_trainer.constants import AUDIO_SAMPLE_RATE

        _, _, num_frames = sample.video_dims or self.config.validation.video_dims
        # Only consulted for a reference that carries its own soundtrack.
        components = SimpleNamespace(audio_sampling_rate=AUDIO_SAMPLE_RATE)
        prepared, _ = MiniMaxH3Ref2VASetupStep.prepare_references(components, requests, num_frames)
        return prepared

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
        # `text_encoder` is only in `names` on the fallback path, and load_components
        # leaves what it loads on the CPU. Left there the conditioner still *works*,
        # which is the trap: it produces correct embeddings roughly a hundred times
        # too slowly, so the run looks hung rather than misconfigured.
        for name in ("vae", "audio_vae", "text_encoder"):
            component = getattr(pipeline, name, None)
            if component is not None and hasattr(component, "to"):
                setattr(pipeline, name, component.to(self.device))
        # The transformer is already resident and carries the weights under test.
        #
        # The component is named per variant -- `transformer` for fl2va, but
        # `transformer_ref` for ref2va -- and `update_components` DROPS any keyword
        # it does not recognize, with only a warning. Passing the wrong name leaves
        # the pipeline holding its own transformer: untrained, and never placed on
        # a device, so sampling runs on CPU and a 448x768x124 clip never finishes.
        # That looks exactly like a hang, and it means every media sample would
        # have been of the base model rather than the adapter. Resolve the name
        # from the pipeline itself and refuse to continue if it is absent.
        name = self._transformer_component_name(pipeline)
        pipeline.update_components(**{name: self.transformer})
        if getattr(pipeline, name, None) is not self.transformer:
            raise RuntimeError(
                f"validation pipeline kept its own {name!r} after update_components; "
                "the adapter under test would not be the thing sampled"
            )
        self._pipeline = pipeline
        return pipeline

    def _transformer_component_name(self, pipeline) -> str:
        """Which component holds the denoiser, as this pipeline names it.

        Asks the pipeline rather than trusting the variant, then cross-checks
        against the name the config expects. The two disagreeing means the blocks
        changed shape underneath us, which is worth failing on -- this bug existed
        because the mapping was written out by hand here while
        ``ModelConfig.transformer_subfolder`` and ``inference.py`` already knew it.
        """
        specs = getattr(pipeline, "_component_specs", {})
        expected = self.config.model.transformer_subfolder
        if expected in specs:
            return expected
        for candidate in ("transformer_ref", "transformer"):
            if candidate in specs:
                logger.warning(
                    "validation pipeline names its denoiser %r, not %r as the %s variant implies",
                    candidate,
                    expected,
                    self.config.model.variant,
                )
                return candidate
        raise RuntimeError(
            f"no transformer component in the validation pipeline; it declares {sorted(specs)}. "
            "Sampling cannot test the adapter without one."
        )

    @torch.no_grad()
    def run(self, step: int) -> list[Path]:
        config = self.config.validation
        if not config.samples or self._sampling_disabled:
            return []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            pipeline = self._build_pipeline()
        except Exception as exc:
            # Not fatal -- a broken sampler should not end a run that is otherwise
            # training correctly -- but it is an error, not a warning. Logged at
            # warning level this scrolled past unread while the run produced no
            # media at all, which is indistinguishable from a run that was never
            # asked to.
            logger.error("Validation media sampling is BROKEN and will be skipped: %s", exc, exc_info=True)
            return []

        was_training = self.transformer.training
        self.transformer.eval()
        outputs: list[Path] = []
        for index, sample in enumerate(config.samples):
            width, height, frames = sample.video_dims or config.video_dims
            geometry = Geometry.create(width=width, height=height, num_frames=frames)
            seed = sample.seed if sample.seed is not None else config.seed
            try:
                with time_budget(config.sample_timeout_seconds):
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
                # Inside the guard on purpose. Writing the clip is the cheapest
                # step here and the likeliest to raise on a type mismatch, and a
                # sample that generated correctly and then failed on the way to
                # disk used to take thirteen hours of training down with it.
                outputs.append(self._write(result, step, index))
            except TimeoutError:
                # Sampling this slowly means it is not running where it should be.
                # Twice now a misplaced component sent denoising to the CPU, where a
                # 448x768x124 clip does not finish in any useful time, and training
                # sat blocked behind it -- once for 1h50m before anyone looked.
                # Give up on media for the rest of the run rather than pay this at
                # every remaining checkpoint; the loss validation still runs.
                self._sampling_disabled = True
                logger.error(
                    "Validation sample %d exceeded %ds and was abandoned; media sampling is now OFF "
                    "for the rest of this run. Training continues. A sample this slow usually means "
                    "the denoiser is not on the GPU.",
                    index,
                    config.sample_timeout_seconds,
                )
                break
            except Exception as exc:
                logger.error("Validation sample %d failed: %s", index, exc, exc_info=True)
                continue

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
        for condition in sample.conditions:
            if condition.type == "first_frame":
                kwargs["image"] = Image.open(condition.image).convert("RGB")
            elif condition.type == "last_frame":
                kwargs["last_image"] = Image.open(condition.image).convert("RGB")
        references = self._reference_requests(sample)
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

        frames = _to_numpy(frames)
        if frames.ndim == 5:
            frames = frames[0]
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)

        waveform = None
        if audio is not None:
            waveform = torch.as_tensor(_to_numpy(audio)).float()
            waveform = waveform.reshape(-1) if waveform.ndim == 1 else waveform.reshape(waveform.shape[-2], -1)

        path = self.output_dir / f"step{step:07d}_sample{index}.mp4"
        return write_video_with_audio(frames, path, waveform=waveform, fps=self.config.validation.frame_rate)


def _to_numpy(value) -> np.ndarray:
    """Whatever the pipeline returned, as a host-side array.

    ``output_type="np"`` converts the video but not the audio, which comes back
    as a CUDA tensor -- and ``np.asarray`` on one of those raises rather than
    copying, so a sample that generated perfectly well died on the way to disk.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", torch.float32).numpy()
    return np.asarray(value)


def _extract(result, names: tuple[str, ...]):
    for name in names:
        value = getattr(result, name, None)
        if value is None and isinstance(result, dict):
            value = result.get(name)
        if value is not None:
            return value[0] if isinstance(value, (list, tuple)) and len(value) else value
    return None
